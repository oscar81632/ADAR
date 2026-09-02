#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw


def load_metadata(dataset_dir: Path, split: str):
    rows = []
    with (dataset_dir / "metadata.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] == split:
                rows.append(row)
    return rows


def make_preview(dataset_dir: Path, split: str, output: Path, count: int, seed: int, cols: int):
    rng = random.Random(seed)
    rows = load_metadata(dataset_dir, split)
    if not rows:
        raise ValueError(f"No rows found for split={split!r}")
    samples = rng.sample(rows, min(count, len(rows)))

    thumbs = []
    label_h = 32
    for sample in samples:
        img = Image.open(dataset_dir / sample["image"]).convert("RGB")
        canvas = Image.new("RGB", (img.width, img.height + label_h), (255, 255, 255))
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        text = f"{sample['class_name']} | L={sample['path_length']} D={sample['distractor_ratio']}"
        draw.text((6, img.height + 8), text, fill=(20, 20, 20))
        thumbs.append(canvas)

    rows_n = (len(thumbs) + cols - 1) // cols
    w, h = thumbs[0].size
    grid = Image.new("RGB", (cols * w, rows_n * h), (238, 238, 238))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * w
        y = (idx // cols) * h
        grid.paste(thumb, (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)
    print(f"Saved preview: {output}")


def main():
    parser = argparse.ArgumentParser(description="Create a preview grid for a VSTD dataset.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = args.dataset_dir / "previews" / f"{args.split}_preview.png"
    make_preview(args.dataset_dir, args.split, output, args.count, args.seed, args.cols)


if __name__ == "__main__":
    main()
