import os
import sys
import time
import random
import numpy as np
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---------------------------------------------------------
# Setup for Colab environment
# ---------------------------------------------------------
if "/content/shadecraft" not in sys.path:
    sys.path.append("/content/shadecraft")

from shadecraft.data.dataloader import ShadeCraftPatchDataset
from shadecraft.models.unet import UNet, UNetConfig
# from shadecraft.models.garb import BasicGARB   # (optional later)


# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_coeff(pred, target, eps=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + eps) / (union + eps)

    return dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = bce_weight

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dice = 1 - dice_coeff(probs, target)
        return self.alpha * bce + (1 - self.alpha) * dice


def save_checkpoint(path, model, optimizer, epoch, best_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss
    }, path)


# ---------------------------------------------------------
# Visualization Utilities for TensorBoard
# ---------------------------------------------------------
def log_sample(writer, x, y, logits, step, max_items=3):
    x = x[:max_items].detach().cpu()
    y = y[:max_items].detach().cpu()
    pred = torch.sigmoid(logits[:max_items]).detach().cpu()

    for i in range(min(max_items, x.shape[0])):
        rgb = x[i, :3].permute(1,2,0)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

        writer.add_image(f"sample/{i}/input_rgb", rgb, step, dataformats="HWC")
        writer.add_image(f"sample/{i}/mask_gt", y[i], step)
        writer.add_image(f"sample/{i}/mask_pred", pred[i], step)


# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
def train(
    tiles_root="/content/shadecraft/data/processed/2013/tiles/256",
    out_dir="/content/shadecraft/checkpoints",
    run_name="unet_2013",
    epochs=25,
    batch_size=8,
    lr=1e-3,
    val_split=0.15,
    num_workers=2,
    use_edges=True,
    use_buildings=True,
    seed=42,
    amp=True,
    patience=5,              # Early stopping patience
    log_images_every=2       # Log sample masks every N epochs
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    writer = SummaryWriter(log_dir=f"/content/logs/{run_name}")

    # -------------------------------------------------
    # 1. Dataset + Split
    # -------------------------------------------------
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

    print(f"Train: {n_train}  |  Val: {n_val}")

    # -------------------------------------------------
    # 2. Model, Loss, Optimizer, Scheduler
    # -------------------------------------------------
    in_ch = 4 + (1 if use_buildings else 0) + (1 if use_edges else 0)

    cfg = UNetConfig(in_channels=in_ch, out_channels=1)
    model = UNet(cfg).to(device)

    criterion = BCEDiceLoss(bce_weight=0.6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # -------------------------------------------------
    # 3. Train Loop
    # -------------------------------------------------
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        # -----------------------------
        # Training
        # -----------------------------
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

        train_loss /= len(train_loader)
        writer.add_scalar("loss/train", train_loss, epoch)

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        val_loss = 0
        val_dice = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                x = x.to(device)
                y = y.to(device)

                with torch.cuda.amp.autocast(enabled=amp):
                    logits = model(x)
                    loss = criterion(logits, y)

                val_loss += loss.item()
                val_dice += dice_coeff(torch.sigmoid(logits), y).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metric/dice", val_dice, epoch)

        # Log example predictions
        if epoch % log_images_every == 0:
            log_sample(writer, x, y, logits, epoch, max_items=3)

        scheduler.step()

        dt = time.time() - t0
        print(f"Epoch {epoch}: train={train_loss:.4f}  val={val_loss:.4f}  dice={val_dice:.4f}  time={dt:.1f}s")

        # -----------------------------
        # Checkpointing
        # -----------------------------
        save_checkpoint(
            os.path.join(out_dir, f"{run_name}_last.pt"),
            model, optimizer, epoch, best_val
        )

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            save_checkpoint(
                os.path.join(out_dir, f"{run_name}_best.pt"),
                model, optimizer, epoch, best_val
            )
            print("🔥 New BEST model saved.")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("⛔ Early stopping triggered.")
            break

    writer.close()
    print("Training complete. Best val loss:", best_val)


# ---------------------------------------------------------
# Entrypoint for Colab
# ---------------------------------------------------------
if __name__ == "__main__":
    train()
