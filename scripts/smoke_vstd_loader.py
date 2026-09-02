#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader

from src.vstd.dataset import SimpleImageTransform, VSTDDataset, collate_vstd


def main():
    parser = argparse.ArgumentParser(description="Smoke test the VSTD dataset loader.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    ds = VSTDDataset(
        args.dataset_dir,
        split=args.split,
        transform=SimpleImageTransform(image_size=None),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_vstd)
    batch = next(iter(loader))

    print(f"dataset_dir: {args.dataset_dir}")
    print(f"split: {args.split}")
    print(f"samples: {len(ds)}")
    print(f"classes: {len(ds.classes)}")
    print(f"prompts: {ds.prompts}")
    print(f"image batch shape: {tuple(batch['image'].shape)}")
    print(f"labels: {batch['label'].tolist()}")
    print("first metadata:")
    first = batch["metadata"][0]
    for key in [
        "image",
        "class_name",
        "path_length",
        "start_end_manhattan",
        "num_turns",
        "path_bbox_area",
        "long_dependency_score",
        "distractor_ratio",
    ]:
        print(f"  {key}: {first[key]}")


if __name__ == "__main__":
    main()
