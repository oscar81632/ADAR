#!/usr/bin/env python3
"""Evaluate trained VSTD runs on one fixed internal diagnostic subset."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.find_diagnostic_splits import load_run_predictions


def match_condition(row, condition):
    path_length = int(row["path_length"])
    turns = int(row["num_turns"])
    manhattan = int(row["start_end_manhattan"])
    longdep = float(row.get("long_dependency_score", 0.0))
    decoys = int(row.get("decoy_arrow_count", 0))
    counts = {"up": 0, "down": 0, "left": 0, "right": 0}
    for a, b in zip(row["path"], row["path"][1:]):
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        if dr == -1:
            counts["up"] += 1
        elif dr == 1:
            counts["down"] += 1
        elif dc == -1:
            counts["left"] += 1
        elif dc == 1:
            counts["right"] += 1
    total_steps = max(1, sum(counts.values()))
    reverse_ratio = (counts["left"] + counts["up"]) / total_steps

    if condition == "long_turn_manhattan_high":
        return (
            turns >= 25
            and manhattan >= 16
            and longdep >= 0.40
        )
    if condition == "long_turn_reverse":
        return turns >= 25 and longdep >= 0.40 and reverse_ratio >= 0.60
    if condition == "long_turn_high_decoy":
        return turns >= 25 and longdep >= 0.40 and decoys >= 24
    if condition == "large_manhattan_high_decoy":
        return manhattan >= 16 and decoys >= 24 and path_length >= 40
    if condition == "high_turn_only":
        return turns >= 25
    raise ValueError(f"Unsupported condition: {condition}")


def accuracy(preds, labels, mask):
    return float((preds[mask] == labels[mask]).float().mean().item())


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fixed VSTD diagnostic subset.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--condition", type=str, default="long_turn_manhattan_high")
    parser.add_argument("--run", action="append", nargs=2, metavar=("NAME", "RUN_DIR"), required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    labels = None
    mask = None
    count = None
    for name, run_dir in args.run:
        preds, run_labels, metadata = load_run_predictions(
            Path(run_dir), args.dataset_dir, args.split, args.batch_size, args.device
        )
        if labels is None:
            labels = run_labels
            mask = torch.tensor([match_condition(row, args.condition) for row in metadata], dtype=torch.bool)
            count = int(mask.sum().item())
        elif not torch.equal(labels, run_labels):
            raise RuntimeError(f"Label order mismatch for run {name}")
        all_mask = torch.ones_like(mask, dtype=torch.bool)
        rows.append(
            {
                "method": name,
                "run_dir": run_dir,
                "subset_acc": accuracy(preds, labels, mask),
                "overall_acc": accuracy(preds, labels, all_mask),
                "subset_count": count,
            }
        )

    rows.sort(key=lambda row: row["subset_acc"], reverse=True)
    best_baseline = max((row["subset_acc"] for row in rows if row["method"] in {"vit", "vim"}), default=None)
    result = {
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "condition": args.condition,
        "subset_count": count,
        "results": rows,
        "best_base_vit_vim_acc": best_baseline,
    }

    output = args.output or Path("runs/internal_fixed_diagnostic_split.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
