"""
shadecraft/models/unet.py

Baseline U-Net for ShadeCraft.

- Default input channels: 5 (RGB + OSM building mask + Canny edges)
- Output channels: 1 (shade probability / mask)

Optional time conditioning:
If you pass a time vector to forward(..., t=...), it will FiLM-condition the
bottleneck latent. For baseline training, just call forward(x) and ignore t.

This model is lightweight and Colab-friendly.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Building blocks
# ----------------------------

class DoubleConv(nn.Module):
    """(Conv -> GN -> ReLU) * 2"""
    def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 8):
        super().__init__()
        # Ensure groups divides channels
        g1 = min(gn_groups, out_ch)
        while out_ch % g1 != 0 and g1 > 1:
            g1 -= 1

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g1, out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g1, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, gn_groups=gn_groups)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True, gn_groups: int = 8):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = DoubleConv(in_ch, out_ch, gn_groups=gn_groups)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, gn_groups=gn_groups)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Handle odd input sizes by padding x1 to match x2
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        if diff_x != 0 or diff_y != 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2]
            )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 conv"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation for optional time conditioning.
    Produces per-channel scale and shift applied to bottleneck features.
    """
    def __init__(self, time_dim: int, channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z: (B, C, H, W)
        t: (B, time_dim)
        """
        gamma_beta = self.mlp(t)  # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return z * (1 + gamma) + beta


# ----------------------------
# UNet Model
# ----------------------------

@dataclass
class UNetConfig:
    in_channels: int = 5
    out_channels: int = 1
    base_channels: int = 64
    bilinear: bool = True
    gn_groups: int = 8

    # Optional time conditioning
    use_time: bool = False
    time_dim: int = 4  # e.g., [time_of_day, azimuth, altitude, day_of_year]


class UNet(nn.Module):
    """
    Baseline U-Net for shade prediction.

    Forward:
      y = model(x)                 # baseline
      y = model(x, t=time_vector)  # if use_time=True
    """
    def __init__(self, cfg: UNetConfig = UNetConfig()):
        super().__init__()
        self.cfg = cfg
        c = cfg.base_channels
        factor = 2 if cfg.bilinear else 1

        self.inc   = DoubleConv(cfg.in_channels, c, gn_groups=cfg.gn_groups)
        self.down1 = Down(c, c*2, gn_groups=cfg.gn_groups)
        self.down2 = Down(c*2, c*4, gn_groups=cfg.gn_groups)
        self.down3 = Down(c*4, c*8, gn_groups=cfg.gn_groups)
        self.down4 = Down(c*8, c*16 // factor, gn_groups=cfg.gn_groups)

        self.use_time = cfg.use_time
        if self.use_time:
            self.film = FiLM(cfg.time_dim, c*16 // factor)
        else:
            self.film = None

        self.up1  = Up(c*16, c*8 // factor, bilinear=cfg.bilinear, gn_groups=cfg.gn_groups)
        self.up2  = Up(c*8,  c*4 // factor, bilinear=cfg.bilinear, gn_groups=cfg.gn_groups)
        self.up3  = Up(c*4,  c*2 // factor, bilinear=cfg.bilinear, gn_groups=cfg.gn_groups)
        self.up4  = Up(c*2,  c, bilinear=cfg.bilinear, gn_groups=cfg.gn_groups)
        self.outc = OutConv(c, cfg.out_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, in_channels, H, W)
        t: (B, time_dim) optional

        returns:
          logits: (B, out_channels, H, W)
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.use_time:
            if t is None:
                raise ValueError("UNetConfig.use_time=True, but no time vector t was provided.")
            x5 = self.film(x5, t)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        logits = self.outc(x)
        return logits


# ----------------------------
# Quick sanity check (optional)
# ----------------------------
if __name__ == "__main__":
    cfg = UNetConfig(in_channels=5, out_channels=1, use_time=False)
    model = UNet(cfg)
    x = torch.randn(2, 5, 512, 512)
    y = model(x)
    print("Output shape:", y.shape)  # (2, 1, 512, 512)
