#!/usr/bin/env python3
"""Analyze a trained VSTD CLIP-alignment run by path geometry buckets."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.vstd.dataset import VSTDDataset, collate_vstd, load_json
from train_vstd_clip_alignment import build_text_features, evaluate, load_clip


def direction_counts(path):
    counts = {"up": 0, "down": 0, "left": 0, "right": 0}
    for a, b in zip(path, path[1:]):
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
        else:
            raise ValueError(f"Non-adjacent path step: {a} -> {b}")
    return counts


def net_direction(row):
    start_r, start_c = row["start"]
    end_r, end_c = row["end"]
    dr = end_r - start_r
    dc = end_c - start_c
    if abs(dc) >= abs(dr):
        if dc > 0:
            return "net_left_to_right"
        if dc < 0:
            return "net_right_to_left"
    else:
        if dr > 0:
            return "net_top_to_bottom"
        if dr < 0:
            return "net_bottom_to_top"
    return "net_tie"


def scan_step_direction(row):
    counts = direction_counts(row["path"])
    aligned = counts["right"] + counts["down"]
    reverse = counts["left"] + counts["up"]
    total = max(1, aligned + reverse)
    aligned_ratio = aligned / total
    reverse_ratio = reverse / total
    if aligned_ratio >= 0.60:
        return "mostly_scan_aligned_right_down"
    if reverse_ratio >= 0.60:
        return "mostly_reverse_left_up"
    return "mixed_scan_direction"


def turn_bin(row):
    turns = int(row["num_turns"])
    if turns <= 10:
        return "turns_00_10"
    if turns <= 16:
        return "turns_11_16"
    if turns <= 24:
        return "turns_17_24"
    return "turns_25_plus"


def accuracy_by_fn(preds, labels, metadata, fn):
    buckets = {}
    for idx, row in enumerate(metadata):
        key = fn(row)
        bucket = buckets.setdefault(key, [0, 0])
        bucket[0] += int(preds[idx].item() == labels[idx].item())
        bucket[1] += 1
    return {
        str(key): {"acc": correct / max(1, total), "count": total}
        for key, (correct, total) in sorted(buckets.items())
    }


def load_args(checkpoint_args, dataset_dir, device):
    args = dict(checkpoint_args)
    args["dataset_dir"] = Path(dataset_dir)
    args["output_dir"] = Path(args.get("output_dir", "runs/unknown"))
    args["device"] = device
    return SimpleNamespace(**args)


def main():
    parser = argparse.ArgumentParser(description="Analyze trained VSTD run by path geometry.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.run_dir / "model.pt", map_location="cpu")
    model_args = load_args(checkpoint["args"], args.dataset_dir, args.device)
    if model_args.encoder == "vim" and (args.device == "cpu" or not torch.cuda.is_available()):
        raise RuntimeError("Vim/Mamba analysis requires CUDA in this environment.")

    device = torch.device(args.device)
    model, preprocess, clip_module = load_clip(model_args, device)
    model.load_state_dict(checkpoint["model"], strict=True)
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
    result = evaluate(model, loader, text_features, device)
    analysis = {
        "run_dir": str(args.run_dir),
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "loss": result["loss"],
        "acc": result["acc"],
        "accuracy_by_net_direction": accuracy_by_fn(
            result["preds"], result["labels"], result["metadata"], net_direction
        ),
        "accuracy_by_scan_step_direction": accuracy_by_fn(
            result["preds"], result["labels"], result["metadata"], scan_step_direction
        ),
        "accuracy_by_turn_bin": accuracy_by_fn(
            result["preds"], result["labels"], result["metadata"], turn_bin
        ),
    }

    output = args.output or (args.run_dir / f"{args.split}_geometry_analysis.json")
    with output.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(json.dumps(analysis, indent=2), flush=True)


if __name__ == "__main__":
    main()
