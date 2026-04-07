import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicSpectralModulation(nn.Module):
    def __init__(self, channel_dim: int, num_patches: int = 196, reduction: int = 4, init_scale: float = 1e-4):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_patches = num_patches
        self.h = int(num_patches ** 0.5)
        self.w = self.h
        if self.h * self.w != num_patches:
            raise ValueError(f"num_patches must be a perfect square, got {num_patches}")

        hidden_dim = max(channel_dim // reduction, 8)
        self.pre_norm = nn.LayerNorm(channel_dim)
        self.mlp = nn.Sequential(
            nn.Linear(channel_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

        fy = torch.fft.fftfreq(self.h) * self.h
        fx = torch.fft.rfftfreq(self.w) * self.w
        distance_sq = fy[:, None] ** 2 + fx[None, :] ** 2
        self.register_buffer("distance_sq", distance_sq.view(1, 1, self.h, self.w // 2 + 1), persistent=False)

        self.mu_max = math.sqrt((self.h / 2) ** 2 + (self.w / 2) ** 2)
        self.eps = 1e-6
        self.residual_scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, N+1, C), got {tuple(x.shape)}")
        if x.size(1) != self.num_patches + 1:
            raise ValueError(f"Token count mismatch, expected {self.num_patches + 1}, got {x.size(1)}")

        cls_token = x[:, :1, :]
        x_spatial = x[:, 1:, :]
        b, _, c = x_spatial.shape

        x_spatial = self.pre_norm(x_spatial)
        x_grid = x_spatial.transpose(1, 2).reshape(b, c, self.h, self.w)
        x_mean = x_grid.mean(dim=(-2, -1), keepdim=True)
        x_centered = x_grid - x_mean

        spectrum = torch.fft.rfft2(x_centered, norm="ortho")
        amplitude = torch.log1p(torch.abs(spectrum))
        energy = amplitude.mean(dim=(-2, -1))

        mu_raw, sigma_raw = self.mlp(energy).unbind(dim=-1)
        mu = (self.mu_max * torch.sigmoid(mu_raw)).clamp_min(1.0).view(b, 1, 1, 1)
        sigma = F.softplus(sigma_raw).clamp_min(0.1).view(b, 1, 1, 1)

        mask = torch.exp(-self.distance_sq / (2.0 * (mu ** 2) * sigma + self.eps))
        filtered = torch.fft.irfft2(spectrum * mask, s=(self.h, self.w), norm="ortho") + x_mean

        filtered_tokens = filtered.reshape(b, c, self.num_patches).transpose(1, 2)
        delta = filtered_tokens - x_spatial
        x_spatial_out = x[:, 1:, :] + self.residual_scale * delta
        return torch.cat([cls_token, x_spatial_out], dim=1)