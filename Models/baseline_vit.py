from typing import Dict
import timm
import torch.nn as nn

def get_baseline_vit(
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    num_classes: int = 100,
    freeze_backbone: bool = True,
    enable_grad_checkpointing: bool = False,
):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "head"):
            for param in model.head.parameters():
                param.requires_grad = True

    if enable_grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    return model

def get_trainable_parameter_stats(model: nn.Module) -> Dict[str, float]:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    ratio = 100.0 * trainable_params / max(all_params, 1)
    return {
        "total_params": all_params,
        "trainable_params": trainable_params,
        "trainable_ratio": ratio,
    }

def print_trainable_parameters(model: nn.Module) -> None:
    stats = get_trainable_parameter_stats(model)
    print(
        f"Total params: {stats['total_params']:,} | "
        f"Trainable params: {stats['trainable_params']:,} | "
        f"Trainable ratio: {stats['trainable_ratio']:.4f}%"
    )