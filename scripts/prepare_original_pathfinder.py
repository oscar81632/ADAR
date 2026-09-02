#!/usr/bin/env python3
"""Convert downloaded Original Pathfinder images into VSTD-compatible format."""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np


CLASSES = [
    {
        "id": 0,
        "name": "separate endpoints",
        "prompt": "two dots are on separate paths",
    },
    {
        "id": 1,
        "name": "connected endpoints",
        "prompt": "two dots are connected by the same path",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Prepare Original Pathfinder as a CLIP-style classification dataset.")
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--train_size", type=int, default=700)
    parser.add_argument("--val_size", type=int, default=150)
    parser.add_argument("--test_size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--path_length", type=int, default=14)
    args = parser.parse_args()

    metadata = np.load(args.metadata, allow_pickle=True)
    rows = []
    for row in metadata:
        image_name = str(row[1])
        label = int(row[3])
        src = args.source_dir / image_name
        if not src.exists():
            raise FileNotFoundError(src)
        rows.append(
            {
                "source_image": image_name,
                "label": label,
                "class_name": CLASSES[label]["name"],
                "prompt": CLASSES[label]["prompt"],
                "path_length": int(float(row[5])),
                "image_size": 128,
                "grid_size": 0,
                "distractor_ratio": 0.0,
                "decoy_arrow_count": 0,
                "start_end_manhattan": 0,
            }
        )

    rng = random.Random(args.seed)
    by_class = {0: [], 1: []}
    for row in rows:
        by_class[row["label"]].append(row)
    for values in by_class.values():
        rng.shuffle(values)

    total_requested = args.train_size + args.val_size + args.test_size
    if total_requested > len(rows):
        raise ValueError(f"Requested {total_requested} samples but only found {len(rows)}.")
    if args.train_size % 2 or args.val_size % 2 or args.test_size % 2:
        raise ValueError("train/val/test sizes must be even for balanced splits.")
    per_class_needed = total_requested // 2
    for label, values in by_class.items():
        if len(values) < per_class_needed:
            raise ValueError(
                f"Class {label} has {len(values)} samples, but balanced splits require {per_class_needed}."
            )

    splits = (
        [("train", args.train_size), ("val", args.val_size), ("test", args.test_size)]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "classes.json").open("w", encoding="utf-8") as f:
        json.dump(CLASSES, f, indent=2)

    out_rows = []
    cursors = {0: 0, 1: 0}
    for split, size in splits:
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_rows = []
        for label in [0, 1]:
            start = cursors[label]
            end = start + size // 2
            split_rows.extend(by_class[label][start:end])
            cursors[label] = end
        rng.shuffle(split_rows)
        for idx, row in enumerate(split_rows):
            dst_rel = f"{split}/{idx:06d}.png"
            shutil.copy2(args.source_dir / row["source_image"], args.output_dir / dst_rel)
            out = dict(row)
            out["split"] = split
            out["image"] = dst_rel
            out.pop("source_image", None)
            out_rows.append(out)

    with (args.output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "dataset_name": args.output_dir.name,
        "source": "Original Pathfinder, pathfinder128/curv_contour_length_14",
        "path_length": args.path_length,
        "sizes": {"train": args.train_size, "val": args.val_size, "test": args.test_size},
        "label_counts": {
            split: {
                str(label): sum(1 for row in out_rows if row["split"] == split and row["label"] == label)
                for label in [0, 1]
            }
            for split, _ in splits
        },
    }
    with (args.output_dir / "dataset_config.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
