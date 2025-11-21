"""
shadecraft/data/dataset.py

Dataset + DataLoader utilities for ShadeCraft.

Expected folder structure:
data/
 ├── raw/           # input NAIP RGB images (.tif or .png)
 ├── masks/         # building masks (optional)
 ├── edges/         # precomputed edge maps (optional)
 └── shadows/       # ground truth shadow masks (binary)

If masks/ or edges/ are missing, dataset will auto-generate:
 - building mask = zeros
 - edges = Canny edge detection (OpenCV)

All outputs are PyTorch tensors ready for UNet + GARB:
   x    -> (5, H, W)   [RGB + mask + edges]
   edges-> (1, H, W)
   y    -> (1, H, W)   [shadow target]
"""

from __future__ import annotations
import os
from typing import Callable, Optional, Tuple

import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------
# Utility functions
# -----------------------------------------------------------

def load_grayscale(path: str):
    """Load grayscale image as float32 [0–1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return img.astype(np.float32) / 255.0


def load_rgb(path: str):
    """Load RGB data as float32 [0–1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read RGB image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def canny_edges(img_gray: np.ndarray):
    """Compute Canny edges (fallback if none provided)."""
    edges = cv2.Canny((img_gray * 255).astype(np.uint8), 100, 200)
    return edges.astype(np.float32) / 255.0


# -----------------------------------------------------------
# Main Dataset
# -----------------------------------------------------------

class ShadeCraftDataset(Dataset):
    def __init__(
        self,
        root: str,
        patch_size: Optional[int] = None,
        transform: Optional[Callable] = None,
        return_paths: bool = False
    ):
        """
        root: path to the data/ directory.
        Expected subfolders: raw/, masks/, shadows/, (optional) edges/
        """
        self.root = root
        self.patch_size = patch_size
        self.transform = transform
        self.return_paths = return_paths

        self.raw_dir = os.path.join(root, "raw")
        self.mask_dir = os.path.join(root, "masks")
        self.edge_dir = os.path.join(root, "edges")
        self.shad_dir = os.path.join(root, "shadows")

        self.fnames = sorted(os.listdir(self.raw_dir))

        print(f"[ShadeCraftDataset] Loaded {len(self.fnames)} samples.")

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        name = os.path.splitext(fname)[0]

        # ---------------------------------------------------
        # Load input RGB
        # ---------------------------------------------------
        rgb_path = os.path.join(self.raw_dir, fname)
        rgb = load_rgb(rgb_path)  # (H, W, 3)

        # ---------------------------------------------------
        # Load building mask (optional)
        # ---------------------------------------------------
        mask_path = os.path.join(self.mask_dir, f"{name}.png")
        if os.path.exists(mask_path):
            mask = load_grayscale(mask_path)
        else:
            mask = np.zeros(rgb.shape[:2], dtype=np.float32)

        # ---------------------------------------------------
        # Load edges (optional)
        # ---------------------------------------------------
        edge_path = os.path.join(self.edge_dir, f"{name}.png")
        if os.path.exists(edge_path):
            edges = load_grayscale(edge_path)
        else:
            # fallback: compute Canny on grayscale RGB
            gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = canny_edges(gray.astype(np.float32) / 255.0)

        # ---------------------------------------------------
        # Load shadow ground truth (required)
        # ---------------------------------------------------
        shadow_path = os.path.join(self.shad_dir, f"{name}.png")
        shadow = load_grayscale(shadow_path)  # (H, W)

        # ---------------------------------------------------
        # Optional random cropping (for training)
        # ---------------------------------------------------
        if self.patch_size is not None:
            H, W = rgb.shape[:2]
            ph = self.patch_size
            pw = self.patch_size

            top = np.random.randint(0, H - ph + 1)
            left = np.random.randint(0, W - pw + 1)

            rgb = rgb[top:top+ph, left:left+pw]
            mask = mask[top:top+ph, left:left+pw]
            edges = edges[top:top+ph, left:left+pw]
            shadow = shadow[top:top+ph, left:left+pw]

        # ---------------------------------------------------
        # Build input tensor for UNet
        # ---------------------------------------------------
        # (RGB:3), (mask:1), (edges:1) -> 5 channels
        x = np.concatenate([
            rgb,                       # (H, W, 3)
            mask[..., None],           # (H, W, 1)
            edges[..., None]           # (H, W, 1)
        ], axis=-1)

        x = torch.from_numpy(x).permute(2, 0, 1).float()  # (C, H, W)
        edges_t = torch.from_numpy(edges).unsqueeze(0).float()
        y = torch.from_numpy(shadow).unsqueeze(0).float()  # target mask

        if self.transform:
            x, edges_t, y = self.transform(x, edges_t, y)

        if self.return_paths:
            return x, edges_t, y, name

        return x, edges_t, y


# -----------------------------------------------------------
# Convenience function to build a DataLoader
# -----------------------------------------------------------

def make_dataloader(
    root: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 2,
    patch_size: Optional[int] = None,
):
    ds = ShadeCraftDataset(root=root, patch_size=patch_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return ds, dl
