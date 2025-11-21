import os
import sys
import time
import random
import numpy as np
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

# -------------------------------------------------------------------
# Colab / repo path setup
# -------------------------------------------------------------------
# Make sure Python can see /content/shadecraft as repo root
if "/content/shadecraft" not in sys.path:
    sys.path.append("/content/shadecraft")

from shadecraft.data.dataloader import ShadeCraftPatchDataset
from shadecraft.models.unet import UNet, UNetConfig
# from shadecraft.models.garb import BasicGARB   # optional later


# -------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_coeff(pred, target, eps=1e-6):
    """
    pred, target: (B,1,H,W) in {0,1} or [0,1]
    """
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dice = 1 - dice_coeff(probs, target)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


def save_checkpoint(path, model, optimizer, epoch, best_val):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "best_val": best_val,
    }, path)


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def train(
    tiles_root="/content/shadecraft/data/processed/2013/tiles/256",
    out_dir="/content/shadecraft/checkpoints",
    run_name="unet_2013",
    epochs=15,
    batch_size=8,
    lr=1e-3,
    val_split=0.15,
    num_workers=2,
    use_edges=True,
    use_buildings=True,
    seed=42,
    amp=True,
):
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -------------------------
    # 1) Dataset + split
    # -------------------------
    full_ds = ShadeCraftPatchDataset(
        root=tiles_root,
        use_edges=use_edges,
        use_buildings=use_buildings
    )

    n_total = len(full_ds)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    print(f"Train patches: {n_train}, Val patches: {n_val}")

    # -------------------------
    # 2) Model
    # -------------------------
    # Your dataloader produces 6 channels:
    # 4 RGBNIR + 1 building mask + 1 edges = 6
    in_ch = 4 + (1 if use_buildings else 0) + (1 if use_edges else 0)

    cfg = UNetConfig(
        in_channels=in_ch,
        out_channels=1,
        use_time=False,
    )
    model = UNet(cfg).to(device)

    # Optional later:
    # garb = BasicGARB(latent_channels=cfg.base_channels * 16).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = BCEDiceLoss(bce_weight=0.6)

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_val = float("inf")

    # -------------------------
    # 3) Training loop
    # -------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_loss /= max(len(train_loader), 1)

        # -------------------------
        # 4) Validation
        # -------------------------
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=amp):
                    logits = model(x)
                    loss = criterion(logits, y)

                val_loss += loss.item()
                probs = torch.sigmoid(logits)
                val_dice += dice_coeff(probs, y).item()

        val_loss /= max(len(val_loader), 1)
        val_dice /= max(len(val_loader), 1)

        dt = time.time() - t0
        print(f"\nEpoch {epoch}: "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_dice={val_dice:.4f} | "
              f"time={dt:.1f}s")

        # -------------------------
        # 5) Checkpointing
        # -------------------------
        ckpt_path = os.path.join(out_dir, f"{run_name}_last.pt")
        save_checkpoint(ckpt_path, model, optimizer, epoch, best_val)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(out_dir, f"{run_name}_best.pt")
            save_checkpoint(best_path, model, optimizer, epoch, best_val)
            print(f"✅ New best checkpoint saved: {best_path}")

    print("\nTraining complete.")
    print("Best val loss:", best_val)


if __name__ == "__main__":
    train(
        tiles_root="/content/shadecraft/data/processed/2013/tiles/256",
        out_dir="/content/shadecraft/checkpoints",
        run_name="unet_2013",
        epochs=20,
        batch_size=8,
        lr=1e-3,
        val_split=0.15,
        num_workers=2,
        use_edges=True,
        use_buildings=True,
        seed=42,
        amp=True,
    )
