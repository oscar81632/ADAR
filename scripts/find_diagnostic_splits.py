#!/usr/bin/env python3
"""Find internal diagnostic VSTD subsets where methods differ strongly.

This is an analysis helper. It does not change training, data, or loss.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.vstd.dataset import VSTDDataset, collate_vstd, load_json
from train_vstd_clip_alignment import build_text_features, load_clip, logits_from_images


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
    return counts


def scan_step_direction(row):
    counts = direction_counts(row["path"])
    aligned = counts["right"] + counts["down"]
    reverse = counts["left"] + counts["up"]
    total = max(1, aligned + reverse)
    if aligned / total >= 0.60:
        return "scan_right_down"
    if reverse / total >= 0.60:
        return "reverse_left_up"
    return "mixed_scan"


def net_direction(row):
    start_r, start_c = row["start"]
    end_r, end_c = row["end"]
    dr = end_r - start_r
    dc = end_c - start_c
    if abs(dc) >= abs(dr):
        if dc > 0:
            return "net_right"
        if dc < 0:
            return "net_left"
    else:
        if dr > 0:
            return "net_down"
        if dr < 0:
            return "net_up"
    return "net_tie"


def turn_bin(row):
    turns = int(row["num_turns"])
    if turns <= 16:
        return "turns_le16"
    if turns <= 24:
        return "turns_17_24"
    return "turns_ge25"


def path_length_bin(row):
    length = int(row["path_length"])
    if length <= 24:
        return "length_24"
    if length <= 32:
        return "length_32"
    if length <= 40:
        return "length_40"
    return "length_48"


def decoy_bin(row):
    decoys = int(row.get("decoy_arrow_count", 0))
    if decoys <= 19:
        return "decoy_le19"
    if decoys <= 23:
        return "decoy_20_23"
    return "decoy_ge24"


def manhattan_bin(row):
    distance = int(row["start_end_manhattan"])
    if distance <= 8:
        return "manhattan_le8"
    if distance <= 14:
        return "manhattan_10_14"
    return "manhattan_ge16"


def longdep_bin(row):
    score = float(row.get("long_dependency_score", 0.0))
    if score < 0.30:
        return "longdep_low"
    if score < 0.40:
        return "longdep_mid"
    return "longdep_high"


FEATURES = {
    "scan": scan_step_direction,
    "net": net_direction,
    "turn": turn_bin,
    "length": path_length_bin,
    "decoy": decoy_bin,
    "manhattan": manhattan_bin,
    "longdep": longdep_bin,
}


def load_run_predictions(run_dir, dataset_dir, split, batch_size, device):
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu")
    model_args = dict(checkpoint["args"])
    model_args.setdefault("path_adapter_layers", 0)
    model_args.setdefault("path_adapter_bottleneck", 128)
    model_args.setdefault("path_adapter_dropout", 0.1)
    model_args.setdefault("path_adapter_pooling", "forward")
    model_args["dataset_dir"] = Path(dataset_dir)
    model_args["output_dir"] = Path(model_args.get("output_dir", run_dir))
    model_args["device"] = device
    model_args = SimpleNamespace(**model_args)

    model, preprocess, clip_module = load_clip(model_args, torch.device(device))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    classes = load_json(Path(dataset_dir) / "classes.json")
    prompts = [item["prompt"] for item in classes]
    text_features = build_text_features(model, clip_module, prompts, torch.device(device))
    dataset = VSTDDataset(dataset_dir, split, transform=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        collate_fn=collate_vstd,
    )

    preds = []
    labels = []
    metadata = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = logits_from_images(model, images, text_features)
            preds.append(logits.argmax(dim=1).cpu())
            labels.append(batch["label"].cpu())
            metadata.extend(batch["metadata"])
    return torch.cat(preds), torch.cat(labels), metadata


def make_candidate_masks(metadata, max_combo):
    feature_values = {
        name: [fn(row) for row in metadata]
        for name, fn in FEATURES.items()
    }
    candidates = []
    for combo_size in range(1, max_combo + 1):
        for names in itertools.combinations(FEATURES.keys(), combo_size):
            value_sets = [sorted(set(feature_values[name])) for name in names]
            for values in itertools.product(*value_sets):
                mask = [
                    all(feature_values[name][idx] == value for name, value in zip(names, values))
                    for idx in range(len(metadata))
                ]
                label = " & ".join(f"{name}={value}" for name, value in zip(names, values))
                candidates.append((label, torch.tensor(mask, dtype=torch.bool)))
    return candidates


def accuracy(preds, labels, mask):
    total = int(mask.sum().item())
    if total == 0:
        return None
    return float((preds[mask] == labels[mask]).float().mean().item())


def main():
    parser = argparse.ArgumentParser(description="Find internal diagnostic VSTD subsets.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--run", action="append", nargs=2, metavar=("NAME", "RUN_DIR"), required=True)
    parser.add_argument("--target", action="append", required=True)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min_count", type=int, default=120)
    parser.add_argument("--max_combo", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    predictions = {}
    labels = None
    metadata = None
    for name, run_dir in args.run:
        preds, run_labels, run_metadata = load_run_predictions(
            Path(run_dir), args.dataset_dir, args.split, args.batch_size, args.device
        )
        predictions[name] = preds
        if labels is None:
            labels = run_labels
            metadata = run_metadata
        elif not torch.equal(labels, run_labels):
            raise RuntimeError(f"Label order mismatch for run {name}")

    rows = []
    for label, mask in make_candidate_masks(metadata, args.max_combo):
        count = int(mask.sum().item())
        if count < args.min_count:
            continue
        accs = {name: accuracy(preds, labels, mask) for name, preds in predictions.items()}
        target_acc = max(accs[name] for name in args.target)
        best_target = max(args.target, key=lambda name: accs[name])
        baseline_acc = max(accs[name] for name in args.baseline)
        baseline_mean = sum(accs[name] for name in args.baseline) / len(args.baseline)
        overall_acc = {name: accuracy(preds, labels, torch.ones_like(mask, dtype=torch.bool)) for name, preds in predictions.items()}
        rows.append(
            {
                "split": label,
                "count": count,
                "best_target": best_target,
                "target_acc": target_acc,
                "best_baseline_acc": baseline_acc,
                "baseline_mean_acc": baseline_mean,
                "gain_vs_best_baseline": target_acc - baseline_acc,
                "gain_vs_baseline_mean": target_acc - baseline_mean,
                "acc": accs,
                "overall_acc": overall_acc,
            }
        )

    rows.sort(key=lambda row: (row["gain_vs_best_baseline"], row["target_acc"], row["count"]), reverse=True)
    result = {
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "targets": args.target,
        "baselines": args.baseline,
        "min_count": args.min_count,
        "max_combo": args.max_combo,
        "top": rows[: args.top_k],
    }
    output = args.output or Path("runs/internal_diagnostic_splits.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
