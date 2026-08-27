"""Gradient-Aware Refinement Block (GARB).

Fuses Sobel gradients of the input edge map with bottleneck features so
shade boundaries stay sharp. Used as a lightweight plug-in at the U-Net
bottleneck; the paper evaluates a multi-scale variant on ControlNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelGrad(nn.Module):
    """Compute Sobel gradients (dx, dy) for a single-channel input."""
    def __init__(self):
        super().__init__()
        # Sobel kernels
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, H, W)
        returns: (B, 2, H, W) concatenated [dx, dy]
        """
        dx = F.conv2d(x, self.kx, padding=1)
        dy = F.conv2d(x, self.ky, padding=1)
        return torch.cat([dx, dy], dim=1)


class BasicGARB(nn.Module):
    """Gradient-aware refinement applied at a single feature scale."""
    def __init__(self, latent_channels: int, edge_weight: float = 1.0):
        super().__init__()
        self.edge_weight = edge_weight
        self.sobel = SobelGrad()

        # We first reduce latent to 1 channel to compute its gradients
        self.latent_reduce = nn.Conv2d(latent_channels, 1, kernel_size=1)

        # Fuse: [latent, latent_grad(2ch), edge_grad(2ch)] -> latent_channels
        in_fuse_ch = latent_channels + 2 + 2

        self.fuse = nn.Sequential(
            nn.Conv2d(in_fuse_ch, latent_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=self._best_gn_groups(latent_channels), num_channels=latent_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=self._best_gn_groups(latent_channels), num_channels=latent_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _best_gn_groups(channels: int, max_groups: int = 8) -> int:
        """Pick a GroupNorm group count that divides channels."""
        g = min(max_groups, channels)
        while channels % g != 0 and g > 1:
            g -= 1
        return g

    def forward(self, latent: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        latent: (B, C, H, W)
        edges:  (B, 1, H_in, W_in)  -> resized to match latent spatial size

        returns refined latent: (B, C, H, W)
        """
        B, C, H, W = latent.shape

        # Resize edge map to bottleneck size
        edges_rs = F.interpolate(edges, size=(H, W), mode="bilinear", align_corners=False)

        # Edge gradients (structure)
        edge_grads = self.sobel(edges_rs) * self.edge_weight  # (B, 2, H, W)

        # Latent gradients (where UNet is blurry)
        latent_1 = self.latent_reduce(latent)                 # (B, 1, H, W)
        latent_grads = self.sobel(latent_1)                   # (B, 2, H, W)

        # Fuse everything
        x = torch.cat([latent, latent_grads, edge_grads], dim=1)
        refined = self.fuse(x)

        # Residual add keeps it stable + easy to train
        return latent + refined


# Optional quick check
if __name__ == "__main__":
    garb = BasicGARB(latent_channels=128)
    latent = torch.randn(2, 128, 32, 32)
    edges = torch.randn(2, 1, 256, 256)
    out = garb(latent, edges)
    print(out.shape)  # (2, 128, 32, 32)
