import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig

from .baseline_vit import get_baseline_vit
from .afm import DynamicSpectralModulation
from .spi import SpectralPhaseSpatialAdapter


class PluginBlockWrapper(nn.Module):
    def __init__(
            self,
            original_block: nn.Module,
            channel_dim: int,
            num_patches: int,
            spi_rank: int = 16,
            init_scale: float = 1e-4,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.original_block = original_block

        self.spi = SpectralPhaseSpatialAdapter(
            channel_dim=channel_dim,
            num_patches=num_patches,
            rank=spi_rank,
            init_scale=init_scale,
            dropout=dropout,
        )

        self.afm = DynamicSpectralModulation(
            channel_dim=channel_dim,
            num_patches=num_patches,
            init_scale=init_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.original_block(x)
        x = self.spi(x)
        x = self.afm(x)
        return x


def get_composite_vit(
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 10,
        lora_r: int = 16,
        use_dora: bool = False,
        spi_rank: int = 16,
        init_scale: float = 1e-4,
        dropout: float = 0.0,
        enable_grad_checkpointing: bool = False,
):
    model = get_baseline_vit(
        model_name=model_name,
        pretrained=True,
        num_classes=num_classes,
        freeze_backbone=True,
        enable_grad_checkpointing=enable_grad_checkpointing,
    )

    channel_dim = model.embed_dim
    num_patches = model.patch_embed.num_patches

    wrapped_blocks = []
    for block in model.blocks:
        wrapped_blocks.append(
            PluginBlockWrapper(
                original_block=block,
                channel_dim=channel_dim,
                num_patches=num_patches,
                spi_rank=spi_rank,
                init_scale=init_scale,
                dropout=dropout,
            )
        )
    model.blocks = nn.Sequential(*wrapped_blocks)

    if lora_r > 0:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none",
            use_dora=use_dora,
        )
        composite_model = get_peft_model(model, peft_config)
    else:
        composite_model = model

    for name, param in composite_model.named_parameters():
        if any(tag in name for tag in ["spi", "afm", "head"]):
            param.requires_grad = True

    return composite_model


def print_composite_model_stats(model: nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lora_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora" in n)
    spi_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "spi" in n)
    afm_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "afm" in n)
    head_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "head" in n)

    print("=== Composite Model Parameter Statistics ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.4f}%)")
    print(f"  - Channel Bypass (LoRA/DoRA): {lora_params:,}")
    print(f"  - Spatial Plugin (SPI): {spi_params:,}")
    print(f"  - Frequency Plugin (AFM): {afm_params:,}")
    print(f"  - Classification Head: {head_params:,}")
    print("============================")