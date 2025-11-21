import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ShadeCraftPatchDataset(Dataset):
    """
    Loads 256×256 patches from:
      - images/
      - masks/
      - edges/
      - shadows/
    """
    def __init__(self, root, use_edges=True, use_buildings=False, transform=None):
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.edge_dir = os.path.join(root, "edges")
        self.shad_dir = os.path.join(root, "shadows")

        self.use_edges = use_edges
        self.use_buildings = use_buildings
        self.transform = transform

        # All files follow: img_00000.npy, mask_00000.npy, etc.
        self.names = sorted([f.split(".")[0].split("_")[1] 
                             for f in os.listdir(self.img_dir) 
                             if f.endswith(".npy")])

        print(f"[ShadeCraftPatchDataset] Loaded {len(self.names)} patches")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        nid = self.names[idx]

        # ---------- Load image patch ----------
        img = np.load(os.path.join(self.img_dir, f"img_{nid}.npy")).astype(np.float32)

        # normalize to [0,1]
        img = img / 255.0 

        # shape: (H, W, C) → convert to (C, H, W)
        img = np.transpose(img, (2, 0, 1))

        inputs = [img]  # main input

        # ---------- Building mask (optional) ----------
        if self.use_buildings:
            bmask = np.load(os.path.join(self.mask_dir, f"mask_{nid}.npy")).astype(np.float32)
            bmask = np.expand_dims(bmask, 0)
            inputs.append(bmask)

        # ---------- Edge map (optional) ----------
        if self.use_edges:
            edges = np.load(os.path.join(self.edge_dir, f"edge_{nid}.npy")).astype(np.float32)
            edges = edges / 255.0
            edges = np.expand_dims(edges, 0)
            inputs.append(edges)

        # Concatenate all channels
        x = np.concatenate(inputs, axis=0)

        # ---------- Shadow mask is the target ----------
        y = np.load(os.path.join(self.shad_dir, f"shadow_{nid}.npy")).astype(np.float32)
        y = np.expand_dims(y, 0)

        # Optional transforms
        if self.transform:
            x, y = self.transform(x, y)

        return torch.tensor(x), torch.tensor(y)


def get_dataloader(root, batch_size=8, shuffle=True, num_workers=2,
                   use_edges=True, use_buildings=False):
    dataset = ShadeCraftPatchDataset(
        root=root,
        use_edges=use_edges,
        use_buildings=use_buildings
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataset, loader
