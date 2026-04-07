from __future__ import annotations

import argparse
import csv
import json
import os
import time
import random
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data_ablation import get_shift_dataloaders
from utils.engine import evaluate, train_one_epoch
from models.composite_vit import get_composite_vit
from models.vpt import get_vpt_vit


@dataclass
class RunConfig:
    experiment: str
    variant: str
    dataset: str
    model_name: str
    stage_dir: str
    data_dir: str
    output_dir: str
    epochs: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    seed: int
    train_interpolation: str
    test_interpolation: str
    eval_interpolations: List[str]
    pretrained: bool
    freeze_backbone: bool
    rank: int
    init_scale: float
    dropout: float
    reduction: int
    use_amp: bool
    input_size: int
    use_imagenet_stats: bool
    enable_train_aug: bool
    enable_train_affine: bool
    grad_checkpointing: bool
    checkpoint_path: str
    lora_r: int
    use_dora: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_csv_row(path: str, row: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def make_run_name(cfg: RunConfig) -> str:
    if cfg.experiment.lower() == "ood_eval":
        ckpt_name = os.path.splitext(os.path.basename(cfg.checkpoint_path))[0]
        return f"ood_eval__{ckpt_name}"
    return (
        f"{cfg.experiment}"
        f"__{cfg.dataset}"
        f"__{cfg.variant}"
        f"__train-{cfg.train_interpolation}"
        f"__test-{cfg.test_interpolation}"
        f"__seed-{cfg.seed}"
    )


def build_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_optimizer(model: nn.Module, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=lr, weight_decay=weight_decay)


def get_composite_model_stats(model: nn.Module) -> Dict[str, float]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = 100.0 * trainable_params / max(total_params, 1)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": ratio,
    }


def evaluate_across_interpolations(
        model: nn.Module,
        cfg: RunConfig,
        device: torch.device,
        criterion: nn.Module,
        num_classes: int,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    iid_acc = None

    for interp in cfg.eval_interpolations:
        _, eval_loader, _ = get_shift_dataloaders(
            dataset_name=cfg.dataset,
            data_dir=cfg.data_dir,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            train_interpolation=cfg.train_interpolation,
            test_interpolation=interp,
            input_size=cfg.input_size,
            use_imagenet_stats=cfg.use_imagenet_stats,
            enable_train_aug=cfg.enable_train_aug,
            enable_train_affine=cfg.enable_train_affine,
        )
        metrics = evaluate(
            model=model,
            dataloader=eval_loader,
            criterion=criterion,
            device=device,
            epoch=0,
            use_amp=cfg.use_amp,
        )
        results[interp] = {
            "loss": float(metrics["loss"]),
            "acc": float(metrics["acc"]),
            "lfer": float(metrics.get("lfer", 0.0)),
        }

        if interp == cfg.train_interpolation:
            iid_acc = float(metrics["acc"])

    if iid_acc is None and cfg.train_interpolation in results:
        iid_acc = float(results[cfg.train_interpolation]["acc"])

    if iid_acc is not None:
        for interp in results:
            results[interp]["drop_from_iid"] = float(iid_acc - results[interp]["acc"])
    else:
        for interp in results:
            results[interp]["drop_from_iid"] = 0.0

    return results


def run_train(cfg: RunConfig) -> None:
    set_seed(cfg.seed)
    device = build_device()

    train_loader, val_loader, num_classes = get_shift_dataloaders(
        dataset_name=cfg.dataset,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        train_interpolation=cfg.train_interpolation,
        test_interpolation=cfg.test_interpolation,
        input_size=cfg.input_size,
        use_imagenet_stats=cfg.use_imagenet_stats,
        enable_train_aug=cfg.enable_train_aug,
        enable_train_affine=cfg.enable_train_affine,
    )

    assert cfg.pretrained is True, "Pretrained weights must be used (pretrained=True)!"

    if cfg.variant == "vpt":
        model = get_vpt_vit(
            model_name=cfg.model_name,
            num_classes=num_classes,
            num_prompts=cfg.rank,
            enable_grad_checkpointing=cfg.grad_checkpointing,
        ).to(device)
    else:
        model = get_composite_vit(
            model_name=cfg.model_name,
            num_classes=num_classes,
            lora_r=cfg.lora_r,
            use_dora=cfg.use_dora,
            spi_rank=cfg.rank,
            init_scale=cfg.init_scale,
            dropout=cfg.dropout,
            enable_grad_checkpointing=cfg.grad_checkpointing,
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    run_name = make_run_name(cfg)
    run_dir = os.path.join(cfg.output_dir, run_name)
    ensure_dir(run_dir)

    save_json(asdict(cfg), os.path.join(run_dir, "config.json"))

    param_stats = get_composite_model_stats(model)
    save_json(param_stats, os.path.join(run_dir, "trainable_params.json"))

    best_acc = -1.0
    best_path = os.path.join(run_dir, "best.pt")
    history_csv = os.path.join(run_dir, "history.csv")
    summary_csv = os.path.join(cfg.output_dir, "summary.csv")

    peak_memory_list = []
    epoch_time_list = []

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            use_amp=cfg.use_amp,
            scheduler=scheduler,
            scheduler_on_step=False,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            use_amp=cfg.use_amp,
        )

        epoch_time_sec = time.perf_counter() - epoch_start
        peak_memory_mb = 0.0
        if device.type == "cuda":
            peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / 1024 / 1024)

        peak_memory_list.append(peak_memory_mb)
        epoch_time_list.append(float(epoch_time_sec))

        row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["acc"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_sec": float(epoch_time_sec),
            "peak_memory_mb": float(peak_memory_mb),
        }
        append_csv_row(history_csv, row)

        if val_metrics["acc"] > best_acc:
            best_acc = float(val_metrics["acc"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "num_classes": num_classes,
                    "best_acc": best_acc,
                    "epoch": epoch,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    ood_results = evaluate_across_interpolations(
        model=model,
        cfg=cfg,
        device=device,
        criterion=criterion,
        num_classes=num_classes,
    )
    save_json(ood_results, os.path.join(run_dir, "ood_results.json"))

    iid_acc = ood_results.get(cfg.train_interpolation, {}).get("acc", best_acc)
    row = {
        "experiment": cfg.experiment,
        "dataset": cfg.dataset,
        "variant": cfg.variant,
        "train_interpolation": cfg.train_interpolation,
        "test_interpolation": cfg.test_interpolation,
        "seed": cfg.seed,
        "best_val_acc": best_acc,
        "iid_acc": iid_acc,
        "trainable_params": param_stats.get("trainable_params", ""),
        "total_params": param_stats.get("total_params", ""),
        "trainable_ratio": param_stats.get("trainable_ratio", ""),
        "peak_memory_mb": max(peak_memory_list) if peak_memory_list else 0.0,
        "mean_epoch_time_sec": sum(epoch_time_list) / len(epoch_time_list) if epoch_time_list else 0.0,
        "best_checkpoint": best_path,
    }
    for interp in cfg.eval_interpolations:
        metrics = ood_results.get(interp, {})
        row[f"{interp}_acc"] = metrics.get("acc", "")
        row[f"{interp}_drop"] = metrics.get("drop_from_iid", "")
    append_csv_row(summary_csv, row)

    print(f"[Done] Best checkpoint saved to: {best_path}")
    print(f"[Summary] peak_memory_mb={row['peak_memory_mb']:.2f}, mean_epoch_time_sec={row['mean_epoch_time_sec']:.2f}")


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Phase 2 CIFAR-100 Benchmark")

    parser.add_argument("--experiment", type=str, default="A")
    parser.add_argument("--variant", type=str, default="composite")
    parser.add_argument("--dataset", type=str, default="CIFAR100")
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--stage-dir", type=str, default="two")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-interpolation", type=str, default="bicubic")
    parser.add_argument("--test-interpolation", type=str, default="bicubic")
    parser.add_argument("--eval-interpolations", type=str, default="bicubic,bilinear,nearest")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--init-scale", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--reduction", type=int, default=4)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--use-imagenet-stats", action="store_true", default=True)
    parser.add_argument("--enable-train-aug", action="store_true", default=True)
    parser.add_argument("--enable-train-affine", action="store_true")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--use-dora", action="store_true")

    args = parser.parse_args()
    eval_interpolations = [x.strip() for x in args.eval_interpolations.split(",") if x.strip()]

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = args.output_dir if args.output_dir else os.path.join(project_root, args.stage_dir, "outputs")

    return RunConfig(
        experiment=args.experiment,
        variant=args.variant,
        dataset=args.dataset,
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=output_dir,
        stage_dir=args.stage_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        train_interpolation=args.train_interpolation,
        test_interpolation=args.test_interpolation,
        eval_interpolations=eval_interpolations,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        rank=args.rank,
        init_scale=args.init_scale,
        dropout=args.dropout,
        reduction=args.reduction,
        use_amp=args.use_amp,
        input_size=args.input_size,
        use_imagenet_stats=args.use_imagenet_stats,
        enable_train_aug=args.enable_train_aug,
        enable_train_affine=args.enable_train_affine,
        grad_checkpointing=args.grad_checkpointing,
        checkpoint_path=args.checkpoint_path,
        lora_r=args.lora_r,
        use_dora=args.use_dora,
    )


def main():
    cfg = parse_args()
    run_train(cfg)


if __name__ == "__main__":
    main()