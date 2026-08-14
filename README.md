# Synthetic Data Generation → Object Detection Training

Train an object detector on **synthetic data generated with NVIDIA Omniverse Replicator**, using
**TAO Toolkit** (`detectnet_v2`, ResNet-18 backbone).

The entire workflow lives in one notebook: **[`notebooks/local_train.ipynb`](notebooks/local_train.ipynb)**.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then set NGC_API_KEY and LOCAL_PROJECT_DIR
jupyter lab notebooks/local_train.ipynb
```

Run the cells top to bottom. Step 0 checks your host before anything expensive happens.

## What the notebook does

| Step | Stage | Runs on |
|---|---|---|
| 0 | Prerequisites check (GPU, Docker, disk, architecture) | host |
| 1 | Configuration — paths and hyperparameters | host |
| 2 | Set up TAO Toolkit via a Docker container | host → container |
| 3 | Download a pre-trained object detection model | host (NGC) |
| 4 | Convert the dataset: Replicator → KITTI → **TFRecords** | host + container |
| 5 | Specify training parameters (batch size, learning rate, …) | host |
| 6 | Train with TAO Toolkit | container (GPU) |
| 7 | Evaluate on held-out test data | container (GPU) |
| 8 | Visualize results — GT vs. predictions | host |
| 9 | *Optional* — prune and export for deployment | container (GPU) |

Training with the default parameters takes roughly **one hour on an NVIDIA RTX A6000**.

## Input data

Point `RAW_DATA_DIR` at Omniverse Replicator `BasicWriter` output:

```
data/raw/<any-subdir>/
├── rgb_0000.png
├── bounding_box_2d_tight_0000.npy
├── bounding_box_2d_tight_labels_0000.json
└── ...
```

Step 4 converts it to KITTI (scaling, clamping, and filtering out occluded, degenerate, and
undersized boxes), then to sharded TFRecords. If you already have a KITTI dataset, drop it into
`data/kitti/{images,labels}` and skip straight to step 4.5.

## Layout

```
├── notebooks/local_train.ipynb   # the workflow — everything is here
├── specs/                        # TAO spec files, written by the notebook
├── data/raw/                     # Replicator output (you provide)
├── data/kitti/                   # converted dataset
├── data/tfrecords/               # sharded TFRecords
├── pretrained_models/            # backbone downloaded from NGC
└── results/                      # checkpoints, logs, inference overlays
```

Everything under `data/`, `results/`, and `pretrained_models/` is gitignored.

## Running on aarch64

> This project directory sits on an **aarch64** host (GB10 / DGX Spark).

The TAO Toolkit TF1 container that provides `detectnet_v2` is published for **linux/amd64 only**,
so steps 2–7 cannot run here. Step 0 detects this and prints your options:

- **Run steps 2–7 on an x86_64 machine or cloud instance**, then copy the trained `.hdf5` back to
  `results/` and run step 8 locally. This is the recommended path.
- **Skip training** and use the pre-trained model shipped with the module — steps 0, 1, 3, 4.1–4.4
  and 8 are pure Python and run fine on any architecture.

## Configuration

All knobs are in `.env` (see `.env.example`) or the Step 1 cell. The ones worth touching first:

| Variable | Default | Effect |
|---|---|---|
| `NGC_API_KEY` | — | Required to pull the container and backbone |
| `TARGET_CLASS` | `palletjack` | Must match the semantic label in your Replicator scene |
| `BATCH_SIZE` | `4` | Raise until GPU memory is full; halve on OOM |
| `NUM_EPOCHS` | `80` | Main wall-clock lever |
| `LEARNING_RATE` | `5e-4` | Scale roughly with batch size |
| `VAL_SPLIT` | `0.2` | Validation fraction |

Troubleshooting for the common failures — arm64 manifest errors, OOM, `mAP` stuck at 0, sim-to-real
gap — is in the last section of the notebook.
