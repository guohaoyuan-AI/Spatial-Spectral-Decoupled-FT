from __future__ import annotations
import os
from typing import Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import GaussianBlur, InterpolationMode
from utils.dataset import CIFARCDataset, TinyImageNet, _canonical_dataset_name, _resolve_stats

INTERP_MAP = {
    "bicubic": InterpolationMode.BICUBIC,
    "bilinear": InterpolationMode.BILINEAR,
    "nearest": InterpolationMode.NEAREST,
}

def _build_train_transform(
    dataset_name: str,
    input_size: int,
    interpolation: str,
    use_imagenet_stats: bool,
    enable_train_aug: bool,
    enable_train_affine: bool,
):
    interp_mode = INTERP_MAP.get(interpolation.lower())
    if interp_mode is None:
        raise ValueError(f"Unsupported train interpolation: {interpolation}")

    mean, std = _resolve_stats(dataset_name, use_imagenet_stats)
    crop_size = 32 if dataset_name.startswith("CIFAR") else 64
    padding = 4 if dataset_name.startswith("CIFAR") else 8

    ops = []
    if enable_train_aug:
        ops.extend(
            [
                transforms.RandomCrop(crop_size, padding=padding),
                transforms.RandomHorizontalFlip(),
            ]
        )
        if enable_train_affine:
            ops.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.02, 0.02),
                    interpolation=interp_mode,
                )
            )

    ops.extend(
        [
            transforms.Resize((input_size, input_size), interpolation=interp_mode),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transforms.Compose(ops)

def _build_test_transform(
    dataset_name: str,
    input_size: int,
    interpolation: str,
    use_imagenet_stats: bool,
    apply_dynamic_blur: bool,
):
    interp_mode = INTERP_MAP.get(interpolation.lower())
    if interp_mode is None:
        raise ValueError(f"Unsupported test interpolation: {interpolation}")

    mean, std = _resolve_stats(dataset_name, use_imagenet_stats)
    ops = []
    if apply_dynamic_blur:
        ops.append(GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
    ops.extend(
        [
            transforms.Resize((input_size, input_size), interpolation=interp_mode),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transforms.Compose(ops)

def get_shift_dataloaders(
    dataset_name: str = "CIFAR10",
    data_dir: str = "./data",
    batch_size: int = 96,
    num_workers: int = 4,
    train_interpolation: str = "bicubic",
    test_interpolation: str = "bicubic",
    corruption_type: Optional[str] = None,
    severity: int = 1,
    apply_dynamic_blur: bool = False,
    input_size: int = 224,
    use_imagenet_stats: bool = True,
    enable_train_aug: bool = True,
    enable_train_affine: bool = False,
    persistent_workers: Optional[bool] = None,
):
    dataset_name = _canonical_dataset_name(dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    train_transform = _build_train_transform(
        dataset_name=dataset_name,
        input_size=input_size,
        interpolation=train_interpolation,
        use_imagenet_stats=use_imagenet_stats,
        enable_train_aug=enable_train_aug,
        enable_train_affine=enable_train_affine,
    )
    test_transform = _build_test_transform(
        dataset_name=dataset_name,
        input_size=input_size,
        interpolation=test_interpolation,
        use_imagenet_stats=use_imagenet_stats,
        apply_dynamic_blur=apply_dynamic_blur,
    )

    num_classes = 100 if dataset_name == "CIFAR100" else 200 if dataset_name == "TINY-IMAGENET" else 10

    if corruption_type is not None:
        corruption_root = os.path.join(data_dir, f"{dataset_name}-C")
        train_dataset = None
        test_dataset = CIFARCDataset(
            root_dir=corruption_root,
            corruption_type=corruption_type,
            severity=severity,
            transform=test_transform,
        )
    elif dataset_name == "TINY-IMAGENET":
        tiny_root = os.path.join(data_dir, "tiny-imagenet-200")
        train_dataset = TinyImageNet(root=tiny_root, split="train", transform=train_transform)
        test_dataset = TinyImageNet(root=tiny_root, split="val", transform=test_transform)
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
    else:
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    common_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }

    train_loader = None
    if train_dataset is not None:
        train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)
    return train_loader, test_loader, num_classes