# Synthetic Data Generation → Object Detection Training

Train an object detector on **synthetic data generated with NVIDIA Omniverse Replicator**, running
natively on an **NVIDIA DGX Spark (GB10)** with PyTorch.

The entire workflow lives in one notebook: **[`notebooks/local_train.ipynb`](notebooks/local_train.ipynb)**.

```bash
python3 -m venv .venv && source .venv/bin/activate

# PyTorch first — from NVIDIA's index, not PyPI (no aarch64+CUDA wheels there)
pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
pip install -r requirements.txt

cp .env.example .env        # optional — every default is tuned for this machine
jupyter lab notebooks/local_train.ipynb
```

Run the cells top to bottom. Step 0 verifies the GPU before anything expensive happens.

---

## Platform

This project is built and tuned for the machine it runs on. Everything below was measured here,
not quoted from a spec sheet:

| | |
|---|---|
| Device | NVIDIA GB10 (Grace-Blackwell, DGX Spark) |
| Architecture | **aarch64** |
| Compute capability | **sm_121** |
| SMs | 48 |
| Unified memory | 130.7 GB |
| Driver / CUDA | 580.173.02 / CUDA 13.0 |
| PyTorch | 2.13.0+cu130, torchvision 0.28.0+cu130 |
| matmul TF32 | 19.4 TFLOP/s |
| matmul BF16 | 45.6 TFLOP/s |

**On sm_121:** the installed PyTorch build ships `sm_80/90/100/110/120` binaries and no exact
`sm_121`. It works anyway — `sm_120` kernels are compatible within the same major architecture.
This is worth knowing because the failure mode when it *doesn't* work is not a clean error at
import; it is `CUDA error: no kernel image is available` at the first kernel launch, long after
`torch.cuda.is_available()` has cheerfully returned `True`. Step 0 of the notebook therefore runs
an actual conv2d forward+backward rather than trusting the availability flag.

---

## Why this diverges from the reference workflow

The NVIDIA reference workflow trains with **TAO Toolkit's `detectnet_v2`**. That cannot run on this
machine, and no amount of configuration fixes it.

TAO's TF1 container is published **linux/amd64 only**. The instinct is to reach for emulation —
`docker run --platform linux/amd64` — but that fails for a structural reason rather than a
performance one:

```
$ docker run --rm --gpus all ubuntu:22.04 ls /usr/lib/*/libcuda.so.*
/usr/lib/aarch64-linux-gnu/libcuda.so.580.173.02     ← the only libcuda that exists
```

`nvidia-container-toolkit` does not put a CUDA driver *inside* the image; it bind-mounts the
**host's** driver libraries in. Those are aarch64 ELF objects, and an emulated x86 process cannot
load them. There is no amd64 driver build that talks to an aarch64 kernel module, so the
substitution has nothing to substitute. QEMU would hand you an amd64 userspace with **zero CUDA**,
and `detectnet_v2` is GPU-only — it would not fall back to CPU, it would fail.

(For completeness: `qemu-user-static` is not installed here either, so `--platform linux/amd64`
currently dies with `exec format error` before it gets as far as CUDA.)

So the workflow was retargeted to a native arm64 PyTorch stack that trains the **same kind of
model** on the **same data**.

### What changed

| Reference workflow (TAO) | This project | Why |
|---|---|---|
| `nvcr.io/nvidia/tao/tao-toolkit` container | Native PyTorch, no container | TAO's TF1 image is amd64-only |
| `detectnet_v2` (GridBox, TF1) | Faster R-CNN (torchvision) | Equivalent two-stage detector, available natively on arm64 |
| ResNet-18 backbone from NGC | ResNet-18 + FPN, ImageNet weights from torchvision | Same backbone, no NGC API key needed |
| KITTI → **TFRecords** via `dataset_convert` | KITTI → `torch.utils.data.Dataset` | TFRecords exists to make TF1 I/O fast; a `DataLoader` with workers already is |
| Protobuf spec files (`specs/*.txt`) | One Python config cell | The spec files were TAO's interface, not the model's |
| `tao detectnet_v2 evaluate` | Self-contained COCO-style AP | Avoids `pycocotools`, which needs compiling on aarch64 |
| `.etlt` export for DeepStream | TorchScript / ONNX | `.etlt` is a TAO-specific format |
| DBSCAN box clustering | NMS (built into torchvision) | Framework-native equivalent |

**What did not change:** the Replicator → KITTI conversion, KITTI as the on-disk format, the
train/val split, the augmentation strategy, and the metrics being reported. Steps 4 and 8 are
framework-agnostic — if you later get access to an x86_64 machine and want TAO's `.etlt` output,
the converted dataset in `data/kitti/` feeds straight into `dataset_convert` unmodified.

### DGX Spark-specific tuning

Two defaults are set from measurement on this hardware rather than convention:

**Batch size.** Measured at 1280×720 with this exact model, AMP on:

| batch | peak GPU memory | throughput |
|---|---|---|
| 4  | 2.2 GB  |  9.7 img/s |
| 8  | 4.0 GB  | 10.5 img/s |
| 16 | 7.5 GB  | 10.7 img/s |
| 32 | 14.6 GB | 10.7 img/s |

Throughput plateaus at 16, and even batch 32 uses barely a tenth of the 130 GB unified memory. The
usual "raise it until you OOM" advice is actively misleading here — **you stop gaining speed long
before you run out of memory**. The default is 16, with `LEARNING_RATE` scaled to the standard
`lr = 0.02 @ batch 16` Faster R-CNN recipe. If you change one, scale the other linearly.

**Precision.** TF32 matmuls, cuDNN autotuning and AMP are enabled in step 2. BF16 is ~2.3× TF32 on
this part, which is most of where the speed comes from.

---

## Architecture

### Pipeline

```
Omniverse Replicator (BasicWriter)
    rgb_*.png  ·  bounding_box_2d_tight_*.npy  ·  *_labels_*.json
        │
        │  4.1  convert_replicator_to_kitti()
        │       scale → clamp to frame → drop occluded / degenerate / undersized / wrong-class
        ▼
    data/kitti/{images,labels}          ← portable, inspectable, TAO-compatible
        │
        │  4.4  KittiDetectionDataset + deterministic seeded split
        │       train: ColorJitter + horizontal flip     ·     val: no augmentation
        ▼
    DataLoader  (collate_fn → list[Tensor], list[dict])
        │
        ▼
    FasterRCNN(resnet18-FPN)  ──  6. train  ──▶  results/model_{last,best}.pt
        │
        ├── 7. evaluate()  →  AP@0.5 · AP@0.75 · mAP@[.5:.95]
        └── 8. predict()   →  ground-truth vs prediction overlays
```

### Model

```
input  3 × 720 × 1280
  └─ ResNet-18            ImageNet-pretrained; stem + stage 1 frozen, stages 2-4 trainable
      └─ FPN              P2 P3 P4 P5 + pool, 256 channels each
          └─ RPN          anchors 32/64/128/256/512 px × aspect {0.5, 1.0, 2.0}
              └─ RoIAlign → box head → 2 classes (__background__, palletjack)
```

Freezing the earliest stages both speeds up training and reduces overfitting, which matters more
than usual on synthetic data — the low-level features are the ones least in need of adaptation and
most prone to latching onto renderer artifacts.

### Training

SGD (momentum 0.9, weight decay 5e-4), linear warmup then cosine decay, AMP with gradient scaling,
gradient-norm clipping at 10.0, and a guard that skips any batch producing a non-finite loss. The
best checkpoint by validation mAP is kept separately from the last, because detection models
overfit synthetic data readily and the final epoch is often not the best one.

---

## Notebook walkthrough

| Step | Stage |
|---|---|
| 0 | Prerequisites — GPU, CUDA, compute capability, real kernel launch |
| 1 | Configuration — paths and hyperparameters |
| 2 | Training environment — TF32, cuDNN autotune, measured throughput |
| 3 | Pre-trained ResNet-18 + FPN backbone |
| 4 | Dataset — Replicator → KITTI → `Dataset`, with sanity checks and GT visualization |
| 5 | Training parameters — optimizer, schedule, AMP |
| 6 | Train |
| 7 | Evaluate — AP@0.5, AP@0.75, mAP@[.5:.95] |
| 8 | Visualize — ground truth vs predictions |
| 9 | *Optional* — export to TorchScript / ONNX |

## Input data

Point `RAW_DATA_DIR` at Omniverse Replicator `BasicWriter` output:

```
data/raw/<any-subdir>/
├── rgb_0000.png
├── bounding_box_2d_tight_0000.npy
├── bounding_box_2d_tight_labels_0000.json
└── ...
```

The converter handles both structured and plain `.npy` bbox arrays, and falls back to
`TARGET_CLASS` when the labels JSON is missing. If you already have a KITTI dataset, drop it into
`data/kitti/{images,labels}` and skip to step 4.3.

One filtering subtlety worth knowing: `min_box_px` is measured in **output** pixels, after
resizing. When upscaling 640→1280 a 4 px source box becomes 8 px and survives despite carrying no
real signal, so set it relative to your output resolution.

## Configuration

All knobs live in `.env` (see `.env.example`) or the step 1 cell:

| Variable | Default | Effect |
|---|---|---|
| `TARGET_CLASS` | `palletjack` | Must match the semantic label in your Replicator scene, lowercase |
| `BATCH_SIZE` | `16` | Measured optimum on this hardware — see the table above |
| `LEARNING_RATE` | `2e-2` | Scale linearly with `BATCH_SIZE` |
| `NUM_EPOCHS` | `20` | Main wall-clock lever |
| `BACKBONE` | `resnet18` | `resnet34` / `resnet50` for more capacity |
| `IMAGE_WIDTH/HEIGHT` | `1280`/`720` | Images are resized to this on conversion |
| `USE_AMP` | `1` | Mixed precision |
| `VAL_SPLIT` | `0.2` | Validation fraction |
| `NUM_WORKERS` | `8` | Lower if workers get killed for lack of shared memory |

## Layout

```
├── notebooks/local_train.ipynb   # the workflow — everything is here
├── data/raw/                     # Replicator output (you provide)
├── data/kitti/                   # converted dataset
└── results/                      # model_last.pt, model_best.pt
```

`data/`, `results/` and `.venv/` are gitignored.

## Verification

Exercised end-to-end on this machine against a generated toy dataset (96 images, a bright box on
noise), at the shipped defaults:

- All **17 code cells execute** top to bottom
- Loss converges 1.00 → 0.021 over 30 epochs
- **AP@0.5 = 1.000 · AP@0.75 = 1.000 · mAP@[.5:.95] = 0.936**

The AP implementation is unit-tested against known cases — perfect predictions → 1.0, no
predictions → 0.0, no ground truth → NaN (excluded from the mean rather than dragging it to zero),
duplicate detections counted as false positives, IoU-threshold boundaries, and score-ordering
effects. The horizontal-flip augmentation is checked for width preservation, centre mirroring,
double-flip identity, and agreement with an actual image flip.

That unit-testing is not ceremony: an untrained model and a broken evaluator both report
`mAP = 0.0000`, and the two are indistinguishable without it.

The batch-size and throughput figures come from direct measurement; the convergence figures come
from the toy dataset. Neither is a claim about accuracy on real Replicator data, which depends
entirely on your scene.

## Next steps

- Re-generate synthetic data with wider domain randomization and compare mAP — the highest-leverage
  loop in the whole workflow.
- Mix in a small set of labeled **real** images (even 5–10%) and re-train; this usually closes most
  of the sim-to-real gap.
- Try `resnet50` once the data pipeline is settled.

Troubleshooting for the common failures — OOM, `NaN` loss, mAP stuck at 0, sim-to-real gap — is in
the final section of the notebook.
