#!/usr/bin/env python3
"""Select VSTD samples whose ADAR gates are visually/quantitatively informative."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.vstd.dataset import VSTDDataset, collate_vstd, load_json
from train_vstd_clip_alignment import build_text_features, load_clip
from visualize_vit_readout_gates import gate_values, overlay_gate


def load_args(checkpoint_args, dataset_dir, output_dir, device):
    args = dict(checkpoint_args)
    args["dataset_dir"] = Path(dataset_dir)
    args["output_dir"] = Path(output_dir)
    args["device"] = device
    return SimpleNamespace(**args)


def mask_mean(values, cells):
    if not cells:
        return float("nan")
    picks = [float(values[row, col].item()) for row, col in cells]
    return sum(picks) / len(picks)


def score_row(gates, row):
    grid = int(gates.numel() ** 0.5)
    values = gates.reshape(grid, grid)
    values = (values - values.min()) / (values.max() - values.min()).clamp_min(1e-6)

    path_cells = [tuple(cell) for cell in row.get("path", [])]
    end_cell = [tuple(row["end"])] if "end" in row else []
    decoy_cells = [
        tuple(item["cell"])
        for item in row.get("distractors", [])
        if item.get("type") == "arrow" or item.get("decoy_path")
    ]
    occupied = set(path_cells) | set(end_cell) | set(decoy_cells)
    bg_cells = [(r, c) for r in range(grid) for c in range(grid) if (r, c) not in occupied]

    path_mean = mask_mean(values, path_cells)
    end_mean = mask_mean(values, end_cell)
    decoy_mean = mask_mean(values, decoy_cells)
    bg_mean = mask_mean(values, bg_cells)

    score = (path_mean - bg_mean) + 0.5 * (end_mean - bg_mean) + 0.25 * (path_mean - decoy_mean)
    return {
        "score": score,
        "path_gate": path_mean,
        "endpoint_gate": end_mean,
        "decoy_gate": decoy_mean,
        "background_gate": bg_mean,
    }


def main():
    parser = argparse.ArgumentParser(description="Select informative ADAR gate heatmaps.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--scan_limit", type=int, default=3000)
    parser.add_argument("--min_path_length", type=int, default=48)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else args.run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = args.run_dir / "model.pt"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_args = load_args(checkpoint["args"], args.dataset_dir, args.output_dir, args.device)
    device = torch.device(args.device)
    model, preprocess, clip_module = load_clip(model_args, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    classes = load_json(args.dataset_dir / "classes.json")
    prompts = [item["prompt"] for item in classes]
    text_features = build_text_features(model, clip_module, prompts, device)

    dataset = VSTDDataset(args.dataset_dir, args.split, transform=preprocess)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_vstd)
    candidates = []
    seen = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        metadata = batch["metadata"]
        image_features = F.normalize(model.encode_image(images).float(), dim=-1)
        logits = model.logit_scale.exp().detach().float() * image_features @ text_features.t()
        preds = logits.argmax(dim=1).cpu()
        gates, alphas = gate_values(model, images)
        labels_cpu = labels.cpu()

        for idx, row in enumerate(metadata):
            seen += 1
            if seen > args.scan_limit:
                break
            if int(row["path_length"]) < args.min_path_length:
                continue
            if int(preds[idx].item()) != int(labels_cpu[idx].item()):
                continue
            metrics = score_row(gates[idx], row)
            candidates.append(
                {
                    **metrics,
                    "image": row["image"],
                    "label": int(labels_cpu[idx].item()),
                    "pred": int(preds[idx].item()),
                    "class_name": row.get("class_name"),
                    "path_length": int(row["path_length"]),
                    "decoy_arrow_count": int(row["decoy_arrow_count"]),
                    "alpha": float(alphas[idx].item()),
                    "gates_index": len(candidates),
                    "row": row,
                    "gates": gates[idx],
                }
            )
        if seen > args.scan_limit:
            break

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = candidates[: args.top_k]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for out_idx, item in enumerate(selected):
        row = item["row"]
        output_path = args.output_dir / (
            f"{out_idx:03d}_score{item['score']:.3f}_L{item['path_length']}"
            f"_decoy{item['decoy_arrow_count']}.png"
        )
        title = (
            f"label={item['label']} pred={item['pred']} L={item['path_length']} "
            f"decoy={item['decoy_arrow_count']} score={item['score']:.3f}"
        )
        overlay_gate(args.dataset_dir / row["image"], item["gates"], output_path, item["alpha"], title)
        clean = {key: value for key, value in item.items() if key not in {"row", "gates"}}
        clean["output"] = str(output_path)
        summary.append(clean)

    with (args.output_dir / "selected_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
