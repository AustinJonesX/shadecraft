# ShadeCraft

**Learning Time-Dynamic Urban Shadow Patterns with Deep Generative Models**

Replicating and improving [DeepShade](https://arxiv.org/abs/2507.12103) for high-resolution shade simulation in hot-arid cities.

[Paper (PDF)](papers/ShadeCraft.pdf) · [DeepShade (arXiv:2507.12103)](https://arxiv.org/abs/2507.12103) · [DeepShade code](https://github.com/LongchaoDa/DeepShade_repo) · [Dataset](https://huggingface.co/datasets/DARL-ASU/DeepShade)

Vraj Ashokbhai Chaudhari · Austin Jones · Atharva Shirode · Tejas Rajaram Uttare
<br>
Ira A. Fulton Schools of Engineering, Arizona State University

---

## Overview

Accurate urban shade maps matter for heat-aware routing, cooling-effect estimates, and public-health planning. [DeepShade](https://arxiv.org/abs/2507.12103) showed that a text-conditioned ControlNet can synthesize time-of-day shade from satellite imagery, building outlines, and solar prompts — but generated shadow edges are often soft and slightly misaligned.

ShadeCraft keeps that pipeline and adds a **Gradient-Aware Refinement Block (GARB)** between ControlNet and the diffusion decoder. GARB reinjects multi-scale gradient structure from Canny edges and intermediate latents so building-cast shadows stay sharp.

On the DeepShade **Tempe** split, GARB improves the metrics that matter for downstream shade analysis:

| Metric | Baseline (DeepShade) | ShadeCraft (GARB) | Relative change |
| :--- | ---: | ---: | ---: |
| **B-IoU** ↑ | 0.0774 | **0.0983** | **+27.11%** |
| **mIoU** ↑ | 0.2889 | **0.3059** | **+5.86%** |
| MSE ↓ | 18.17 | **17.74** | **−2.34%** |
| SSIM ↑ | **0.9697** | 0.9688 | −0.09% |
| LPIPS ↓ | **0.2692** | 0.3321 | +23.34% |

B-IoU and mIoU measure shade **location and boundary alignment**. The LPIPS increase comes from texture noise in non-shadow regions; it does not move predicted shade. See the [paper](papers/ShadeCraft.pdf) for qualitative examples and discussion of that tradeoff.

## Method

```
RGB skeleton + Canny edges + text prompt (time / solar angle)
        │
        ▼
   ControlNet U-Net          ← DeepShade backbone
        │  features at 320 / 640 / 1280 channels
        ▼
   Gradient-Aware Refinement Block (GARB)
        │  edge projection → concat → conv refine → gated residual
        ▼
   Diffusion decoder
        │
        ▼
   Time-aware shade map
```

GARB is applied at each ControlNet scale (320, 640, 1280 channels):

1. **Edge projection** — normalize the hint image and map it into the feature space with two SiLU convolutions.
2. **Refinement** — concatenate projected edges with ControlNet features and pass them through a three-layer conv block. The last layer is **zero-initialized**, so GARB starts as the identity and does not destroy a pretrained backbone.
3. **Gated residual** — a learned sigmoid gate (initialized at ~12% mix) controls how much refinement is added back.

This repository also includes a compact `BasicGARB` module and a U-Net training path for lighter-weight shade-mask experiments (see [Repository structure](#repository-structure)).

## Dataset

Experiments in the paper use the **Tempe, Arizona** subset of [DARL-ASU/DeepShade](https://huggingface.co/datasets/DARL-ASU/DeepShade) (2,048 train / 877 test). Each sample pairs:

- satellite RGB
- a building-skeleton snapshot
- a Canny edge map
- a text prompt such as `"Right now, it is 6:00 PM in a day."` or `"Solar declination: -20.7°"`
- a ground-truth shade map from Blender-based 3D illumination

The U-Net utilities below can also ingest local NAIP rasters and OSM building footprints (see `shadecraft/data/preprocess.py`).

## Repository structure

```
shadecraft/
├── models/
│   ├── garb.py              # Gradient-Aware Refinement Block
│   └── unet.py              # U-Net baseline (+ optional GARB / time FiLM)
├── data/
│   ├── preprocess.py        # NAIP + OSM → image, mask, edges, shadows
│   ├── tile_dataset.py      # Non-overlapping 256×256 patches
│   ├── dataset.py           # Image-folder dataset (PNG/JPG)
│   └── dataloader.py        # Pre-tiled .npy patch dataset
└── training/
    └── training_unet.py     # Train loop, AMP, TensorBoard, early stopping
papers/
├── ShadeCraft.pdf           # This project
└── DeepShade.pdf            # Original DeepShade paper (Da et al., 2025)
```

## Setup

Python 3.9+ and a CUDA GPU are recommended.

```bash
git clone https://github.com/AustinJonesX/shadecraft.git
cd shadecraft
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Preprocess a local raster (optional)

Run the following from the repository root.

```bash
python -m shadecraft.data.preprocess \
  --raster-path data/raw/tempe_2013_naip.tif \
  --footprints-path data/osm/tempe_buildings.geojson \
  --save-dir data/processed/2013
```

```bash
python -m shadecraft.data.tile_dataset \
  --processed-dir data/processed/2013 \
  --patch-size 256
```

### Train

```bash
python -m shadecraft.training.training_unet \
  --tiles-root data/processed/2013/tiles/256 \
  --out-dir checkpoints \
  --run-name unet_garb \
  --use-edges --use-garb \
  --epochs 25 --batch-size 8
```

Omit `--use-garb` for the U-Net-only baseline. Logs go to `logs/<run-name>` for TensorBoard:

```bash
tensorboard --logdir logs
```

### DeepShade ControlNet baseline

The paper’s ControlNet training and single-image inference follow the original authors’ scripts:

```bash
# from https://github.com/LongchaoDa/DeepShade_repo
python ControlNet/run_vanillaControlnet_train_dlc.py
python ControlNet/a_inference/infer_single.py
```

Paper runs used a learning rate of `1e-4`, AdamW, 50 epochs, and DDIM sampling (50 steps, guidance scale 9.0). The GARB model was trained on an NVIDIA RTX 5090 (batch size 6, gradient accumulation 4).

## Citation

If you use this work, please cite ShadeCraft and the original DeepShade paper:

```bibtex
@article{chaudhari2025shadecraft,
  title     = {ShadeCraft: Learning Time-Dynamic Urban Shadow Patterns with Deep Generative Models},
  author    = {Chaudhari, Vraj Ashokbhai and Jones, Austin and Shirode, Atharva and Uttare, Tejas Rajaram},
  year      = {2025},
  note      = {Arizona State University}
}

@article{da2025deepshade,
  title   = {DeepShade: Enable Shade Simulation by Text-conditioned Image Generation},
  author  = {Da, Longchao and Liu, Xiangrui and Shivakoti, Mithun and Kutralingam, Thirulogasankar Pranav and Yang, Yezhou and Wei, Hua},
  journal = {arXiv preprint arXiv:2507.12103},
  year    = {2025}
}
```

## Acknowledgements

ShadeCraft builds on [DeepShade](https://github.com/LongchaoDa/DeepShade_repo) (Da et al., 2025) and the [DARL-ASU/DeepShade](https://huggingface.co/datasets/DARL-ASU/DeepShade) dataset. GARB is informed by structural refinement ideas in ShadowGAN, ARShadowGAN, and Deep Umbra. See the [paper](papers/ShadeCraft.pdf) for full references.

## License

Code in this repository is released under the [MIT License](LICENSE). The bundled PDFs remain under their original authors’ copyright.
