#!/usr/bin/env python3
"""Analyze image-level ADAR routing coefficients on VSTD.

This script scans a split, records the per-image alpha value from the ADAR
router, and summarizes how alpha relates to correctness and dataset metadata.
"""

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
from visualize_vit_readout_gates import vit_tokens


NUMERIC_FIELDS = [
    "path_length",
    "distractor_ratio",
    "decoy_arrow_count",
    "start_end_manhattan",
    "num_turns",
    "path_bbox_area",
    "path_coverage",
    "long_dependency_score",
]


def load_args(checkpoint_args, dataset_dir, device):
    args = dict(checkpoint_args)
    args["dataset_dir"] = Path(dataset_dir)
    args["output_dir"] = Path(args.get("output_dir", "runs/unknown"))
    args["device"] = device
    return SimpleNamespace(**args)


def mean(values):
    return sum(values) / max(1, len(values))


def std(values):
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) <= 1:
        return None
    xs, ys = zip(*pairs)
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 0 or y_var <= 0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return cov / math.sqrt(x_var * y_var)


def summarize_group(records, key):
    buckets = defaultdict(list)
    for record in records:
        buckets[record[key]].append(record["alpha"])
    rows = []
    for value in sorted(buckets):
        alphas = buckets[value]
        rows.append(
            {
                "bucket": value,
                "n": len(alphas),
                "alpha_mean": mean(alphas),
                "alpha_std": std(alphas),
            }
        )
    return rows


def summarize_by_quantile(records, key, num_bins=4):
    values = sorted({record[key] for record in records if record.get(key) is not None})
    if not values:
        return []
    if len(values) <= num_bins:
        return summarize_group(records, key)

    sorted_records = sorted([record for record in records if record.get(key) is not None], key=lambda item: item[key])
    rows = []
    for bin_idx in range(num_bins):
        start = round(bin_idx * len(sorted_records) / num_bins)
        end = round((bin_idx + 1) * len(sorted_records) / num_bins)
        chunk = sorted_records[start:end]
        if not chunk:
            continue
        alphas = [record["alpha"] for record in chunk]
        rows.append(
            {
                "bucket": f"{chunk[0][key]}..{chunk[-1][key]}",
                "n": len(alphas),
                "alpha_mean": mean(alphas),
                "alpha_std": std(alphas),
            }
        )
    return rows


@torch.no_grad()
def collect_records(args):
    checkpoint_path = args.run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = args.run_dir / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_args = load_args(checkpoint["args"], args.dataset_dir, args.device)

    device = torch.device(args.device)
    model, preprocess, clip_module = load_clip(model_args, device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    relevant_missing = [
        key
        for key in missing
        if key.startswith("visual.path_readout.router")
        or key.startswith("visual.path_readout.token_gate")
        or key.startswith("visual.path_readout.fuse.")
        or key in {"visual.ln_post.weight", "visual.ln_post.bias", "visual.proj"}
    ]
    if relevant_missing or unexpected:
        raise RuntimeError(
            "Checkpoint did not cleanly load the ADAR components needed for alpha analysis. "
            f"Relevant missing keys: {relevant_missing}; unexpected keys: {unexpected}"
        )
    model.eval()

    classes = load_json(args.dataset_dir / "classes.json")
    prompts = [item["prompt"] for item in classes]
    text_features = build_text_features(model, clip_module, prompts, device)

    dataset = VSTDDataset(args.dataset_dir, args.split, transform=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_vstd,
    )

    readout = getattr(model.visual, "path_readout", None)
    if readout is None or not hasattr(readout, "router"):
        raise ValueError("This checkpoint does not expose visual.path_readout.router.")

    records = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        image_features = model.encode_image(images)
        image_features = F.normalize(image_features.float(), dim=-1)
        logits = model.logit_scale.exp().detach().float() * image_features @ text_features.t()
        preds = logits.argmax(dim=1)

        tokens = vit_tokens(model.visual, images)
        cls = tokens[:, 0, :]
        alpha = torch.sigmoid(readout.router(cls)).squeeze(-1).float().cpu()

        for idx, row in enumerate(batch["metadata"]):
            label = int(labels[idx].item())
            pred = int(preds[idx].item())
            record = {
                "image": row["image"],
                "alpha": float(alpha[idx].item()),
                "label": label,
                "pred": pred,
                "correct": int(pred == label),
            }
            for field in NUMERIC_FIELDS:
                record[field] = row.get(field)
            records.append(record)

    return records


def write_markdown(output_path, summary):
    lines = [
        "# ADAR Alpha Analysis",
        "",
        f"- Run: `{summary['run_dir']}`",
        f"- Dataset: `{summary['dataset_dir']}`",
        f"- Split: `{summary['split']}`",
        f"- Samples: {summary['n']}",
        f"- Accuracy: {summary['accuracy']:.4f}",
        f"- Alpha mean/std/min/max: {summary['alpha_mean']:.4f} / {summary['alpha_std']:.4f} / {summary['alpha_min']:.4f} / {summary['alpha_max']:.4f}",
        "",
        "## Correlations With Alpha",
        "",
        "| Variable | Pearson r |",
        "|---|---:|",
    ]
    for key, value in summary["correlations"].items():
        formatted = "n/a" if value is None else f"{value:.4f}"
        lines.append(f"| {key} | {formatted} |")

    lines.extend(["", "## Correctness Groups", "", "| Group | n | Alpha mean | Alpha std |", "|---|---:|---:|---:|"])
    for row in summary["correctness_groups"]:
        group = "correct" if row["bucket"] == 1 else "wrong"
        lines.append(f"| {group} | {row['n']} | {row['alpha_mean']:.4f} | {row['alpha_std']:.4f} |")

    for section_key, title in [
        ("path_length_groups", "Path Length"),
        ("distractor_ratio_groups", "Distractor Ratio"),
        ("decoy_arrow_count_groups", "Decoy Arrow Count"),
        ("long_dependency_score_groups", "Long-Dependency Score"),
    ]:
        lines.extend(["", f"## {title}", "", "| Bucket | n | Alpha mean | Alpha std |", "|---|---:|---:|---:|"])
        for row in summary[section_key]:
            lines.append(f"| {row['bucket']} | {row['n']} | {row['alpha_mean']:.4f} | {row['alpha_std']:.4f} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze ADAR alpha values against VSTD metadata.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = collect_records(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    alphas = [record["alpha"] for record in records]
    correct = [record["correct"] for record in records]
    summary = {
        "run_dir": str(args.run_dir),
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "n": len(records),
        "accuracy": mean(correct),
        "alpha_mean": mean(alphas),
        "alpha_std": std(alphas),
        "alpha_min": min(alphas),
        "alpha_max": max(alphas),
        "correlations": {
            "correct": pearson(alphas, correct),
            **{field: pearson(alphas, [record.get(field) for record in records]) for field in NUMERIC_FIELDS},
        },
        "correctness_groups": summarize_group(records, "correct"),
        "path_length_groups": summarize_group(records, "path_length"),
        "distractor_ratio_groups": summarize_group(records, "distractor_ratio"),
        "decoy_arrow_count_groups": summarize_by_quantile(records, "decoy_arrow_count"),
        "long_dependency_score_groups": summarize_by_quantile(records, "long_dependency_score"),
    }

    with (args.output_dir / "alpha_records.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image", "alpha", "label", "pred", "correct", *NUMERIC_FIELDS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / "alpha_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(args.output_dir / "ALPHA_ANALYSIS.md", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
