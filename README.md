# Synthetic Data Generation → Object Detection Training

Train an object detector on **synthetic data generated with NVIDIA Omniverse Replicator**, running
natively on this machine's GPU with **PyTorch** (Faster R-CNN, ResNet-18 + FPN backbone).

The entire workflow lives in one notebook: **[`notebooks/local_train.ipynb`](notebooks/local_train.ipynb)**.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate

# PyTorch first — from NVIDIA's index, not PyPI (no aarch64+CUDA wheels there)
pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
pip install -r requirements.txt

cp .env.example .env        # optional — every default is sane
jupyter lab notebooks/local_train.ipynb
```

Run the cells top to bottom. Step 0 checks the GPU before anything expensive happens.

## What the notebook does

| Step | Stage |
|---|---|
| 0 | Prerequisites check — GPU, CUDA, compute capability, real kernel launch |
| 1 | Configuration — paths and hyperparameters |
| 2 | Training environment — TF32, cuDNN autotune, measured throughput |
| 3 | Pre-trained ResNet-18 + FPN backbone |
| 4 | Dataset: Replicator → KITTI → `torch.utils.data.Dataset` |
| 5 | Training parameters — SGD, warmup + cosine schedule, AMP |
| 6 | Train |
| 7 | Evaluate — COCO-style AP\@0.5, AP\@0.75, mAP\@[.5:.95] |
| 8 | Visualize — ground truth vs. predictions |
| 9 | *Optional* — export to TorchScript / ONNX |

## Why not TAO Toolkit?

The reference workflow uses TAO Toolkit's `detectnet_v2`, which ships **only** in a linux/amd64
container. This host is **aarch64** (GB10 / DGX Spark), and no amount of emulation fixes that:
`nvidia-container-toolkit` injects the host's aarch64 `libcuda.so` into the container, and an
emulated x86 process cannot load an aarch64 shared object. QEMU would give you an amd64 userspace
with **zero CUDA**, and `detectnet_v2` is GPU-only.

So this notebook trains the same thing — a ResNet-18-backboned detector, from an ImageNet-pretrained
backbone, on Replicator-generated synthetic data — on a native arm64 PyTorch stack. The GB10 is a
Blackwell part and comfortably outpaces the RTX A6000 that the original "~1 hour" figure assumes.

If you specifically need TAO (e.g. a DeepStream `.etlt`), run it on an x86_64 host. Steps 4 and 8
here are framework-agnostic and still apply.

## Input data

Point `RAW_DATA_DIR` at Omniverse Replicator `BasicWriter` output:

```
data/raw/<any-subdir>/
├── rgb_0000.png
├── bounding_box_2d_tight_0000.npy
├── bounding_box_2d_tight_labels_0000.json
└── ...
```

Step 4 converts it to KITTI — scaling, clamping, and filtering out occluded, degenerate and
undersized boxes — then wraps it in a `Dataset`. KITTI is kept as the on-disk intermediate because
it is human-inspectable and portable to other toolchains. If you already have a KITTI dataset, drop
it into `data/kitti/{images,labels}` and skip to step 4.3.

## Layout

```
├── notebooks/local_train.ipynb   # the workflow — everything is here
├── data/raw/                     # Replicator output (you provide)
├── data/kitti/                   # converted dataset
└── results/                      # checkpoints, model_best.pt, model_last.pt
```

`data/`, `results/`, and `.venv/` are gitignored.

## Configuration

All knobs live in `.env` (see `.env.example`) or the Step 1 cell:

| Variable | Default | Effect |
|---|---|---|
| `TARGET_CLASS` | `palletjack` | Must match the semantic label in your Replicator scene |
| `BATCH_SIZE` | `4` | Raise until VRAM is full; halve on OOM |
| `NUM_EPOCHS` | `20` | Main wall-clock lever |
| `LEARNING_RATE` | `5e-3` | Scale roughly linearly with batch size |
| `BACKBONE` | `resnet18` | `resnet34` / `resnet50` for more capacity |
| `USE_AMP` | `1` | Mixed precision |
| `VAL_SPLIT` | `0.2` | Validation fraction |

## Verification

The pipeline was exercised end-to-end on this host against a generated toy dataset (24 images):
all 17 code cells execute, loss decreases, and the model reaches **AP\@0.5 = 0.976 /
mAP\@[.5:.95] = 0.686** after 30 epochs. The AP implementation is unit-tested against known cases
(perfect predictions, no predictions, no ground truth, duplicate detections, IoU-threshold
boundaries, score ordering), and the horizontal-flip augmentation is checked for width
preservation, centre mirroring and double-flip identity.

Troubleshooting for the common failures — OOM, `NaN` loss, mAP stuck at 0, sim-to-real gap — is in
the last section of the notebook.
