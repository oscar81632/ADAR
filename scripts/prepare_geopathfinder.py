#!/usr/bin/env python3
"""Prepare GeoPathfinder as a VSTD-compatible CLIP-style dataset."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


CLASSES = [
    {
        "id": 0,
        "name": "disconnected land points",
        "prompt": "two red dots are separated by water",
    },
    {
        "id": 1,
        "name": "connected land points",
        "prompt": "two red dots are connected by land",
    },
]


def relative_image_path(output_dir: Path, image_path: Path) -> str:
    return os.path.relpath(image_path.resolve(), output_dir.resolve())


def main():
    parser = argparse.ArgumentParser(description="Prepare GeoPathfinder for CLIP-style training.")
    parser.add_argument("--source_dir", type=Path, required=True, help="Extracted geopathfinder directory.")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    metadata_path = args.source_dir / "metadata.parquet"
    metadata = pd.read_parquet(metadata_path)
    metadata = metadata[metadata["labeled"]].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "classes.json").open("w", encoding="utf-8") as f:
        json.dump(CLASSES, f, indent=2)

    rows = []
    missing = []
    for item in metadata.itertuples(index=False):
        sample_id = str(item.sample_id)
        split = str(item.split)
        for label, folder, distance_key in [
            (0, "disconnect", "disconnect_distance_px"),
            (1, "connect", "connect_distance_px"),
        ]:
            image_path = args.source_dir / "pathfinder" / folder / f"{sample_id}.jpg"
            if not image_path.exists():
                missing.append(str(image_path))
                continue
            rows.append(
                {
                    "split": split,
                    "image": relative_image_path(args.output_dir, image_path),
                    "label": label,
                    "class_name": CLASSES[label]["name"],
                    "prompt": CLASSES[label]["prompt"],
                    "sample_id": sample_id,
                    "region": str(item.region),
                    "image_size": int(item.patch_pixels),
                    "resolution_m": float(item.resolution_m),
                    "water_fraction": float(item.water_fraction),
                    "path_length": int(round(float(getattr(item, distance_key)))),
                    "point_distance_px": float(getattr(item, distance_key)),
                    "grid_size": 0,
                    "distractor_ratio": 0.0,
                    "decoy_arrow_count": 0,
                    "start_end_manhattan": 0,
                }
            )

    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first missing: {missing[0]}")

    with (args.output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "dataset_name": args.output_dir.name,
        "source": str(args.source_dir),
        "classes": CLASSES,
        "sizes": {
            split: sum(1 for row in rows if row["split"] == split)
            for split in ["train", "val", "test"]
        },
        "label_counts": {
            split: {
                str(label): sum(1 for row in rows if row["split"] == split and row["label"] == label)
                for label in [0, 1]
            }
            for split in ["train", "val", "test"]
        },
        "note": "Each labeled patch contributes one connect image and one disconnect image.",
    }
    with (args.output_dir / "dataset_config.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
