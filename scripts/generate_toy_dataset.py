#!/usr/bin/env python3
"""Generate a toy Omniverse Replicator-shaped dataset for pipeline verification.

There is no real Replicator scene wired up yet, so this stands in for it: it writes
the same BasicWriter layout the converter in the notebook expects (rgb_*.png +
bounding_box_2d_tight_*.npy + *_labels_*.json), a single bright rectangle on
noise standing in for the target object. Enough to exercise every stage of the
pipeline — conversion, augmentation, training, evaluation — without Omniverse.

Usage:
    python scripts/generate_toy_dataset.py [--out data/raw] [--n 96]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def generate_toy_dataset(out_dir, n=96, width=960, height=540, seed=42, target_class="palletjack"):
    rng = np.random.default_rng(seed)
    scene_dir = Path(out_dir) / "toy_scene_0"
    scene_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        img = rng.normal(90, 25, size=(height, width, 3)).clip(0, 255).astype(np.uint8)

        bw, bh = int(rng.integers(40, 220)), int(rng.integers(40, 220))
        x1, y1 = int(rng.integers(0, width - bw)), int(rng.integers(0, height - bh))
        x2, y2 = x1 + bw, y1 + bh
        img[y1:y2, x1:x2] = rng.integers(200, 256, size=(bh, bw, 3))
        Image.fromarray(img, "RGB").save(scene_dir / f"rgb_{i:04d}.png")

        occlusion = float(rng.uniform(0.0, 0.3))
        boxes = np.array(
            [(1, float(x1), float(y1), float(x2), float(y2), occlusion)],
            dtype=[
                ("semanticId", "i4"),
                ("x_min", "f4"), ("y_min", "f4"),
                ("x_max", "f4"), ("y_max", "f4"),
                ("occlusionRatio", "f4"),
            ],
        )
        np.save(scene_dir / f"bounding_box_2d_tight_{i:04d}.npy", boxes)
        (scene_dir / f"bounding_box_2d_tight_labels_{i:04d}.json").write_text(
            json.dumps({"1": {"class": target_class}})
        )

    return scene_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-class", default="palletjack")
    args = parser.parse_args()

    scene = generate_toy_dataset(
        args.out, n=args.n, width=args.width, height=args.height,
        seed=args.seed, target_class=args.target_class,
    )
    print(f"wrote {args.n} toy frames -> {scene}")
