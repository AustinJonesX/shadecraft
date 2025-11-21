import os
import numpy as np
from tqdm import tqdm

def extract_patches(arr, patch_size=256):
    """
    Extract non-overlapping patches from a HxWxC array.
    Returns: list of patches
    """
    H, W = arr.shape[:2]
    patches = []

    for y in range(0, H, patch_size):
        if y + patch_size > H:
            continue
        for x in range(0, W, patch_size):
            if x + patch_size > W:
                continue
            patch = arr[y:y+patch_size, x:x+patch_size]
            patches.append((patch, x, y))

    return patches

def tile_dataset(
    processed_dir,
    patch_size=256,
    min_building_pixels=50,
    year="2013"
):
    """
    processed_dir: directory with image.npy, mask.npy, edges.npy, shadows.npy
    patch_size: size of square tile
    min_building_pixels: threshold to skip empty regions
    """
    print(f"=== Tiling dataset for {year} ===")

    image = np.load(os.path.join(processed_dir, "image.npy"))
    mask = np.load(os.path.join(processed_dir, "mask.npy"))
    edges = np.load(os.path.join(processed_dir, "edges.npy"))
    shadows = np.load(os.path.join(processed_dir, "shadows.npy"))

    out_dir = os.path.join(processed_dir, f"tiles/{patch_size}")
    img_dir = os.path.join(out_dir, "images")
    mask_dir = os.path.join(out_dir, "masks")
    edge_dir = os.path.join(out_dir, "edges")
    shad_dir = os.path.join(out_dir, "shadows")

    # make dirs
    for d in [img_dir, mask_dir, edge_dir, shad_dir]:
        os.makedirs(d, exist_ok=True)

    # extract patches
    print("Extracting image patches...")
    img_patches = extract_patches(image, patch_size)
    mask_patches = extract_patches(mask, patch_size)
    edge_patches = extract_patches(edges, patch_size)
    shad_patches = extract_patches(shadows, patch_size)

    assert len(img_patches) == len(mask_patches)
    N = len(img_patches)

    print(f"Total patches: {N}")

    saved_count = 0

    print("Saving filtered patches...")
    for i in tqdm(range(N)):
        img_patch, x, y = img_patches[i]
        mask_patch, _, _ = mask_patches[i]
        edge_patch, _, _ = edge_patches[i]
        shad_patch, _, _ = shad_patches[i]

        # Skip empty regions (no buildings)
        if mask_patch.sum() < min_building_pixels:
            continue

        # save patches
        np.save(os.path.join(img_dir, f"img_{saved_count:05d}.npy"), img_patch)
        np.save(os.path.join(mask_dir, f"mask_{saved_count:05d}.npy"), mask_patch)
        np.save(os.path.join(edge_dir, f"edge_{saved_count:05d}.npy"), edge_patch)
        np.save(os.path.join(shad_dir, f"shadow_{saved_count:05d}.npy"), shad_patch)

        saved_count += 1

    print(f"✔ Saved {saved_count} patches for year {year}.")


if __name__ == "__main__":
    # Example use (2013 dataset)
    tile_dataset(
        processed_dir="/content/shadecraft/data/processed/2013",
        patch_size=256,
        min_building_pixels=50,
        year="2013"
    )
