#!/usr/bin/env python3
"""Analyze ADAR alpha and patch-gate statistics against VSTD outcomes."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.vstd.dataset import VSTDDataset, collate_vstd, load_json
from train_vstd_clip_alignment import build_text_features, load_clip
from visualize_vit_readout_gates import gate_values


META_FIELDS = [
    "path_length",
    "distractor_ratio",
    "decoy_arrow_count",
    "start_end_manhattan",
    "num_turns",
    "path_bbox_area",
    "path_coverage",
    "long_dependency_score",
]


STAT_FIELDS = [
    "alpha",
    "gate_mean",
    "gate_std",
    "path_gate_mean",
    "nonpath_gate_mean",
    "path_enrichment",
    "endpoint_gate",
    "decoy_gate_mean",
    "black_decoy_gate_mean",
    "gray_decoy_gate_mean",
]


def load_args(checkpoint_args, dataset_dir, device):
    args = dict(checkpoint_args)
    args["dataset_dir"] = Path(dataset_dir)
    args["output_dir"] = Path(args.get("output_dir", "runs/unknown"))
    args["device"] = device
    return SimpleNamespace(**args)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / max(1, len(values))


def std(values):
    values = [v for v in values if v is not None]
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) <= 1:
        return None
    xs, ys = zip(*pairs)
    xm, ym = mean(xs), mean(ys)
    xv = sum((x - xm) ** 2 for x in xs)
    yv = sum((y - ym) ** 2 for y in ys)
    if xv <= 0 or yv <= 0:
        return None
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / math.sqrt(xv * yv)


def mask_mean(values, cells):
    if not cells:
        return None
    grid = values.shape[0]
    selected = []
    for cell in cells:
        r, c = int(cell[0]), int(cell[1])
        if 0 <= r < grid and 0 <= c < grid:
            selected.append(values[r, c].item())
    return mean(selected) if selected else None


def gate_stats(gates, row):
    grid = int(gates.numel() ** 0.5)
    values = gates.reshape(grid, grid)
    path_cells = [tuple(cell) for cell in row.get("path", [])]
    endpoint = row.get("end")
    nonpath_cells = [(r, c) for r in range(grid) for c in range(grid) if (r, c) not in set(path_cells)]
    decoy_cells = []
    black_cells = []
    gray_cells = []
    for item in row.get("distractors", []):
        if item.get("type") != "arrow":
            continue
        cell = tuple(item["cell"])
        decoy_cells.append(cell)
        if item.get("color") == "black" or item.get("decoy_path"):
            black_cells.append(cell)
        else:
            gray_cells.append(cell)

    path_mean = mask_mean(values, path_cells)
    nonpath_mean = mask_mean(values, nonpath_cells)
    endpoint_gate = mask_mean(values, [endpoint]) if endpoint else None
    return {
        "gate_mean": float(values.mean().item()),
        "gate_std": float(values.std(unbiased=True).item()),
        "path_gate_mean": path_mean,
        "nonpath_gate_mean": nonpath_mean,
        "path_enrichment": path_mean - nonpath_mean if path_mean is not None and nonpath_mean is not None else None,
        "endpoint_gate": endpoint_gate,
        "decoy_gate_mean": mask_mean(values, decoy_cells),
        "black_decoy_gate_mean": mask_mean(values, black_cells),
        "gray_decoy_gate_mean": mask_mean(values, gray_cells),
    }


@torch.no_grad()
def collect(args):
    checkpoint_path = args.run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = args.run_dir / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_args = load_args(checkpoint["args"], args.dataset_dir, args.device)
    device = torch.device(args.device)
    model, preprocess, clip_module = load_clip(model_args, device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    classes = load_json(args.dataset_dir / "classes.json")
    prompts = [item["prompt"] for item in classes]
    text_features = build_text_features(model, clip_module, prompts, device)

    dataset = VSTDDataset(args.dataset_dir, args.split, transform=preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_vstd)

    records = []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        image_features = F.normalize(model.encode_image(images).float(), dim=-1)
        logits = model.logit_scale.exp().detach().float() * image_features @ text_features.t()
        preds = logits.argmax(dim=1).cpu()
        gates, alphas = gate_values(model, images)
        for idx, row in enumerate(batch["metadata"]):
            label = int(labels[idx].item())
            pred = int(preds[idx].item())
            record = {
                "image": row["image"],
                "label": label,
                "pred": pred,
                "correct": int(label == pred),
                "alpha": float(alphas[idx].item()),
                **gate_stats(gates[idx], row),
            }
            for field in META_FIELDS:
                record[field] = row.get(field)
            records.append(record)
    return records


def grouped(records, group_key, value_keys):
    buckets = defaultdict(list)
    for record in records:
        buckets[record[group_key]].append(record)
    rows = []
    for bucket in sorted(buckets):
        chunk = buckets[bucket]
        row = {"bucket": bucket, "n": len(chunk), "acc": mean([r["correct"] for r in chunk])}
        for key in value_keys:
            row[f"{key}_mean"] = mean([r.get(key) for r in chunk])
            row[f"{key}_std"] = std([r.get(key) for r in chunk])
        rows.append(row)
    return rows


def write_md(path, summary):
    lines = [
        "# ADAR Gate Statistics",
        "",
        f"- Run: `{summary['run_dir']}`",
        f"- Dataset: `{summary['dataset_dir']}`",
        f"- Split: `{summary['split']}`",
        f"- Samples: {summary['n']}",
        f"- Accuracy: {summary['accuracy']:.4f}",
        "",
        "## Correlation With Correctness",
        "",
        "| Signal | r with correct |",
        "|---|---:|",
    ]
    for key, value in summary["corr_with_correct"].items():
        lines.append(f"| {key} | {'n/a' if value is None else f'{value:.4f}'} |")
    lines += ["", "## Correlation With Metadata", ""]
    for stat, vals in summary["corr_with_metadata"].items():
        lines += [f"### {stat}", "", "| Metadata | r |", "|---|---:|"]
        for key, value in vals.items():
            lines.append(f"| {key} | {'n/a' if value is None else f'{value:.4f}'} |")
        lines.append("")
    lines += ["## Correct vs Wrong Means", "", "| Group | n | acc | alpha | path enrich. | path gate | endpoint gate | decoy gate |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["by_correct"]:
        name = "correct" if row["bucket"] == 1 else "wrong"
        lines.append(
            f"| {name} | {row['n']} | {row['acc']:.4f} | {row['alpha_mean']:.4f} | "
            f"{row['path_enrichment_mean']:.4f} | {row['path_gate_mean_mean']:.4f} | "
            f"{row['endpoint_gate_mean']:.4f} | {row['decoy_gate_mean_mean']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = collect(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_dir": str(args.run_dir),
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "n": len(records),
        "accuracy": mean([r["correct"] for r in records]),
        "corr_with_correct": {
            key: pearson([r.get(key) for r in records], [r["correct"] for r in records])
            for key in STAT_FIELDS
        },
        "corr_with_metadata": {
            stat: {meta: pearson([r.get(stat) for r in records], [r.get(meta) for r in records]) for meta in META_FIELDS}
            for stat in STAT_FIELDS
        },
        "by_correct": grouped(records, "correct", ["alpha", "path_enrichment", "path_gate_mean", "endpoint_gate", "decoy_gate_mean"]),
        "by_path_length": grouped(records, "path_length", ["alpha", "path_enrichment", "path_gate_mean", "endpoint_gate", "decoy_gate_mean"]),
    }

    with (args.output_dir / "gate_records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / "gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_md(args.output_dir / "GATE_STATS.md", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
