import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralPhaseSpatialAdapter(nn.Module):


    def __init__(
        self,
        channel_dim: int,
        num_patches: int = 196,
        rank: int = 16,
        init_scale: float = 1e-4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.h = int(num_patches ** 0.5)
        self.w = self.h
        self.rank = rank
        if rank % 2 != 0:
            raise ValueError("rank 必须为偶数")
        if self.h * self.w != num_patches:
            raise ValueError(f"num_patches 必须是完全平方数，当前为 {num_patches}")

        self.pre_norm = nn.LayerNorm(channel_dim)
        self.spatial_down = nn.Linear(self.num_patches, rank, bias=False)
        self.spatial_up = nn.Linear(rank, self.num_patches, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(init_scale))

        self._init_spectral_phase_weights()

    def _select_low_frequencies(self, num_pairs: int) -> List[Tuple[int, int]]:
        candidates = []
        max_freq = max(self.h, self.w)
        for u in range(max_freq):
            for v in range(max_freq):
                if u >= self.h or v >= self.w:
                candidates.append((u, v, u * u + v * v, u + v))
        candidates.sort(key=lambda item: (item[2], item[3], item[0], item[1]))

        selected = []
        for u, v, _, _ in candidates:
            if u == 0 and v == 0:
                continue
            selected.append((u, v))
            if len(selected) == num_pairs:
                break
        if len(selected) < num_pairs:
            raise RuntimeError("可用低频基底数量不足")
        return selected

    def _init_spectral_phase_weights(self) -> None:
        num_pairs = self.rank // 2
        x_grid = torch.arange(self.h, dtype=torch.float32).view(self.h, 1).expand(self.h, self.w).flatten()
        y_grid = torch.arange(self.w, dtype=torch.float32).view(1, self.w).expand(self.h, self.w).flatten()

        freqs = self._select_low_frequencies(num_pairs)
        weight_init = torch.zeros((self.num_patches, self.rank), dtype=torch.float32)
        for i, (u, v) in enumerate(freqs):
            phase = 2 * math.pi * ((u * x_grid / self.h) + (v * y_grid / self.w))
            weight_init[:, 2 * i] = torch.cos(phase)
            weight_init[:, 2 * i + 1] = torch.sin(phase)

        q, _ = torch.linalg.qr(weight_init, mode="reduced")
        q = q[:, : self.rank]
        with torch.no_grad():
            self.spatial_down.weight.copy_(q.T)
            self.spatial_up.weight.copy_(q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"期望输入形状为 (B, N+1, C)，当前为 {tuple(x.shape)}")
        if x.size(1) != self.num_patches + 1:
            raise ValueError(f"token 数量不匹配，期望 {self.num_patches + 1}，当前为 {x.size(1)}")

        cls_token = x[:, :1, :]
        x_spatial = x[:, 1:, :]
        residual = x_spatial

        x_spatial = self.pre_norm(x_spatial).transpose(1, 2)
        x_low_rank = self.spatial_down(x_spatial)
        x_low_rank = F.gelu(x_low_rank)
        x_recon = self.spatial_up(x_low_rank).transpose(1, 2)
        x_recon = self.dropout(x_recon)

        x_spatial_out = residual + self.residual_scale * x_recon
        return torch.cat([cls_token, x_spatial_out], dim=1)
