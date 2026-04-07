from __future__ import annotations
import math
from typing import Dict
import torch
import torch.nn as nn

from models.afm import DynamicSpectralModulation
from models.baseline_vit import get_baseline_vit, get_trainable_parameter_stats
from models.spi import SpectralPhaseSpatialAdapter

class SPIBlockWrapper(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        channel_dim: int,
        num_patches: int,
        rank: int = 16,
        init_scale: float = 1e-4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_block = original_block
        self.spi = SpectralPhaseSpatialAdapter(
            channel_dim=channel_dim,
            num_patches=num_patches,
            rank=rank,
            init_scale=init_scale,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.original_block(x)
        x = self.spi(x)
        return x

class AFMBlockWrapper(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        channel_dim: int,
        num_patches: int,
        init_scale: float = 1e-4,
        reduction: int = 4,
    ):
        super().__init__()
        self.original_block = original_block
        self.afm = DynamicSpectralModulation(
            channel_dim=channel_dim,
            num_patches=num_patches,
            reduction=reduction,
            init_scale=init_scale,
        )

    def forward(self, x):
        x = self.original_block(x)
        x = self.afm(x)
        return x

class SPIAFMBlockWrapper(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        channel_dim: int,
        num_patches: int,
        rank: int = 16,
        init_scale: float = 1e-4,
        dropout: float = 0.0,
        reduction: int = 4,
    ):
        super().__init__()
        self.original_block = original_block
        self.spi = SpectralPhaseSpatialAdapter(
            channel_dim=channel_dim,
            num_patches=num_patches,
            rank=rank,
            init_scale=init_scale,
            dropout=dropout,
        )
        self.afm = DynamicSpectralModulation(
            channel_dim=channel_dim,
            num_patches=num_patches,
            reduction=reduction,
            init_scale=init_scale,
        )

    def forward(self, x):
        x = self.original_block(x)
        x = self.spi(x)
        x = self.afm(x)
        return x

class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base.in_features
        out_features = base.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out

class DoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base.in_features
        out_features = base.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dora_scale = nn.Parameter(torch.ones(out_features))

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base_out = self.base(x)
        update = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        update = update * self.dora_scale
        return base_out + update

class LoRABlockWrapper(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        rank: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_block = original_block

        if hasattr(self.original_block, "attn"):
            attn = self.original_block.attn
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                attn.qkv = LoRALinear(attn.qkv, rank=rank, alpha=rank, dropout=dropout)
            if hasattr(attn, "proj") and isinstance(attn.proj, nn.Linear):
                attn.proj = LoRALinear(attn.proj, rank=rank, alpha=rank, dropout=dropout)

    def forward(self, x):
        return self.original_block(x)

class DoRABlockWrapper(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        rank: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_block = original_block

        if hasattr(self.original_block, "attn"):
            attn = self.original_block.attn
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                attn.qkv = DoRALinear(attn.qkv, rank=rank, alpha=rank, dropout=dropout)
            if hasattr(attn, "proj") and isinstance(attn.proj, nn.Linear):
                attn.proj = DoRALinear(attn.proj, rank=rank, alpha=rank, dropout=dropout)

    def forward(self, x):
        return self.original_block(x)

def _freeze_for_peft(model: nn.Module, variant: str) -> None:
    variant = variant.lower()

    if variant in {"fullft", "full_ft", "full-finetune"}:
        for _, param in model.named_parameters():
            param.requires_grad = True
        return

    for _, param in model.named_parameters():
        param.requires_grad = False

    if hasattr(model, "head"):
        for param in model.head.parameters():
            param.requires_grad = True

    for name, param in model.named_parameters():
        if variant in {"baseline", "head_only", "head-only"}:
            continue
        if variant == "spi" and ".spi." in name:
            param.requires_grad = True
        elif variant == "afm" and ".afm." in name:
            param.requires_grad = True
        elif variant in {"spi_afm", "spiafm", "spi+afm"} and (".spi." in name or ".afm." in name):
            param.requires_grad = True
        elif variant == "lora" and (".lora_A." in name or ".lora_B." in name):
            param.requires_grad = True
        elif variant == "dora" and (".lora_A." in name or ".lora_B." in name or ".dora_scale" in name):
            param.requires_grad = True

def _replace_blocks(
    model: nn.Module,
    variant: str,
    rank: int,
    init_scale: float,
    dropout: float,
    reduction: int,
) -> nn.Module:
    variant = variant.lower()
    if variant in {"baseline", "fullft", "head_only", "head-only"}:
        return model

    channel_dim = model.embed_dim
    num_patches = model.patch_embed.num_patches

    wrapped_blocks = []
    for block in model.blocks:
        if variant == "spi":
            wrapped_blocks.append(
                SPIBlockWrapper(
                    original_block=block,
                    channel_dim=channel_dim,
                    num_patches=num_patches,
                    rank=rank,
                    init_scale=init_scale,
                    dropout=dropout,
                )
            )
        elif variant == "afm":
            wrapped_blocks.append(
                AFMBlockWrapper(
                    original_block=block,
                    channel_dim=channel_dim,
                    num_patches=num_patches,
                    init_scale=init_scale,
                    reduction=reduction,
                )
            )
        elif variant in {"spi_afm", "spiafm", "spi+afm"}:
            wrapped_blocks.append(
                SPIAFMBlockWrapper(
                    original_block=block,
                    channel_dim=channel_dim,
                    num_patches=num_patches,
                    rank=rank,
                    init_scale=init_scale,
                    dropout=dropout,
                    reduction=reduction,
                )
            )
        elif variant == "lora":
            wrapped_blocks.append(
                LoRABlockWrapper(
                    original_block=block,
                    rank=rank,
                    dropout=dropout,
                )
            )
        elif variant == "dora":
            wrapped_blocks.append(
                DoRABlockWrapper(
                    original_block=block,
                    rank=rank,
                    dropout=dropout,
                )
            )
        else:
            raise ValueError(f"Unsupported variant: {variant}")

    model.blocks = nn.Sequential(*wrapped_blocks)
    return model

def build_model(
    variant: str = "baseline",
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    num_classes: int = 10,
    freeze_backbone: bool = True,
    rank: int = 16,
    init_scale: float = 1e-4,
    dropout: float = 0.0,
    reduction: int = 4,
    enable_grad_checkpointing: bool = False,
):
    variant = variant.lower()
    model = get_baseline_vit(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        enable_grad_checkpointing=enable_grad_checkpointing,
    )
    model = _replace_blocks(
        model=model,
        variant=variant,
        rank=rank,
        init_scale=init_scale,
        dropout=dropout,
        reduction=reduction,
    )
    _freeze_for_peft(model, variant=variant)
    return model

def summarize_trainable_params(model: nn.Module) -> Dict[str, float]:
    return get_trainable_parameter_stats(model)