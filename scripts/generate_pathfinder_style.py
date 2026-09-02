#!/usr/bin/env python3
"""Generate a Pathfinder-style visual long-range dependency dataset.

This is an exploratory dataset inspired by the LRA Pathfinder task, not the
official LRA release. Samples ask whether two marked endpoints are connected by
the same long path in a noisy image.
"""

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw


CLASSES = [
    {
        "id": 0,
        "name": "disconnected endpoints",
        "prompt": "two marked endpoints are not connected by one path",
    },
    {
        "id": 1,
        "name": "connected endpoints",
        "prompt": "two marked endpoints are connected by one path",
    },
]


def neighbors(cell, grid_size):
    r, c = cell
    out = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            out.append((nr, nc))
    return out


def random_walk(rng, grid_size, length, occupied=None):
    occupied = set(occupied or [])
    for _ in range(200):
        start = (rng.randrange(grid_size), rng.randrange(grid_size))
        if start in occupied:
            continue
        path = [start]
        used = set(occupied)
        used.add(start)
        for _step in range(length):
            candidates = [cell for cell in neighbors(path[-1], grid_size) if cell not in used]
            if not candidates:
                break
            # Prefer continuing away from the start a little; this reduces tiny loops.
            sr, sc = start
            candidates.sort(key=lambda x: abs(x[0] - sr) + abs(x[1] - sc), reverse=True)
            top_k = candidates[: min(3, len(candidates))]
            nxt = rng.choice(top_k)
            path.append(nxt)
            used.add(nxt)
        if len(path) >= length * 0.75:
            return path
    return path


def cell_center(cell, grid_size, image_size, margin):
    r, c = cell
    usable = image_size - 2 * margin
    step = usable / (grid_size - 1)
    return (margin + c * step, margin + r * step)


def jitter_point(rng, point, amount):
    return (point[0] + rng.uniform(-amount, amount), point[1] + rng.uniform(-amount, amount))


def draw_polyline(draw, points, color, width):
    if len(points) < 2:
        return
    draw.line(points, fill=color, width=width, joint="curve")
    radius = max(1, width // 2)
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def draw_endpoint(draw, point, fill, outline, radius):
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=max(2, radius // 3))


def draw_sample(rng, label, args):
    scale = args.supersample
    image_size = args.image_size * scale
    margin = args.margin * scale
    grid_size = args.grid_size
    line_width = args.line_width * scale
    endpoint_radius = args.endpoint_radius * scale
    jitter = args.jitter * scale

    image = Image.new("RGB", (image_size, image_size), (246, 247, 248))
    draw = ImageDraw.Draw(image)

    # Subtle background noise grid, visually similar in spirit to Pathfinder clutter.
    if args.draw_grid:
        step = (image_size - 2 * margin) / (grid_size - 1)
        grid_color = (232, 235, 238)
        for i in range(grid_size):
            x = margin + i * step
            y = margin + i * step
            draw.line([(x, margin), (x, image_size - margin)], fill=grid_color, width=max(1, scale))
            draw.line([(margin, y), (image_size - margin, y)], fill=grid_color, width=max(1, scale))

    path_length = rng.randint(args.min_path_length, args.max_path_length)
    occupied = set()
    main_path = random_walk(rng, grid_size, path_length, occupied)
    occupied.update(main_path)
    paths = [main_path]

    if label == 0:
        second = random_walk(rng, grid_size, path_length, occupied)
        occupied.update(second)
        paths.append(second)
        endpoint_cells = [main_path[0], second[-1]]
    else:
        endpoint_cells = [main_path[0], main_path[-1]]

    decoy_count = rng.randint(args.min_decoys, args.max_decoys)
    for _ in range(decoy_count):
        decoy_len = rng.randint(args.min_decoy_length, args.max_decoy_length)
        decoy = random_walk(rng, grid_size, decoy_len, occupied if args.avoid_overlap else None)
        if args.avoid_overlap:
            occupied.update(decoy)
        paths.append(decoy)

    for idx, path in enumerate(paths):
        if len(path) < 2:
            continue
        color_value = rng.randint(30, 90) if idx < (2 if label == 0 else 1) else rng.randint(80, 180)
        color = (color_value, color_value, color_value)
        points = [jitter_point(rng, cell_center(cell, grid_size, image_size, margin), jitter) for cell in path]
        draw_polyline(draw, points, color=color, width=line_width)

    endpoint_points = [cell_center(cell, grid_size, image_size, margin) for cell in endpoint_cells]
    draw_endpoint(draw, endpoint_points[0], fill=(242, 73, 73), outline=(95, 18, 18), radius=endpoint_radius)
    draw_endpoint(draw, endpoint_points[1], fill=(73, 118, 242), outline=(18, 36, 95), radius=endpoint_radius)

    if scale > 1:
        image = image.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)

    metadata = {
        "label": label,
        "class_name": CLASSES[label]["name"],
        "prompt": CLASSES[label]["prompt"],
        "path_length": len(main_path),
        "grid_size": grid_size,
        "image_size": args.image_size,
        "endpoint_cells": [list(cell) for cell in endpoint_cells],
        "main_path": [list(cell) for cell in main_path],
        "num_decoys": decoy_count,
        "distractor_ratio": decoy_count / max(1, args.max_decoys),
        "decoy_arrow_count": decoy_count,
        "start_end_manhattan": abs(endpoint_cells[0][0] - endpoint_cells[1][0])
        + abs(endpoint_cells[0][1] - endpoint_cells[1][1]),
    }
    return image, metadata


def generate_split(rng, split, count, out_dir, args, rows):
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        label = idx % 2
        if rng.random() < 0.5:
            label = 1 - label
        image, metadata = draw_sample(rng, label, args)
        rel_path = f"{split}/{idx:06d}.png"
        image.save(out_dir / rel_path)
        metadata.update({"image": rel_path, "split": split})
        rows.append(metadata)


def main():
    parser = argparse.ArgumentParser(description="Generate Pathfinder-style long-range visual reasoning data.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_size", type=int, default=6000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=2000)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--grid_size", type=int, default=18)
    parser.add_argument("--min_path_length", type=int, default=42)
    parser.add_argument("--max_path_length", type=int, default=72)
    parser.add_argument("--min_decoys", type=int, default=8)
    parser.add_argument("--max_decoys", type=int, default=16)
    parser.add_argument("--min_decoy_length", type=int, default=4)
    parser.add_argument("--max_decoy_length", type=int, default=12)
    parser.add_argument("--margin", type=int, default=16)
    parser.add_argument("--line_width", type=int, default=3)
    parser.add_argument("--endpoint_radius", type=int, default=7)
    parser.add_argument("--jitter", type=float, default=2.5)
    parser.add_argument("--supersample", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--draw_grid", action="store_true")
    parser.add_argument("--avoid_overlap", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "classes.json").open("w", encoding="utf-8") as f:
        json.dump(CLASSES, f, indent=2)

    rows = []
    generate_split(rng, "train", args.train_size, args.output_dir, args, rows)
    generate_split(rng, "val", args.val_size, args.output_dir, args, rows)
    generate_split(rng, "test", args.test_size, args.output_dir, args, rows)
    with (args.output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = {
        "dataset_name": "pathfinder_style",
        "official_lra": False,
        "note": "Exploratory Pathfinder-style data inspired by LRA Pathfinder; not the official LRA release.",
        "sizes": {"train": args.train_size, "val": args.val_size, "test": args.test_size},
        "config": vars(args),
    }
    with (args.output_dir / "dataset_config.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
