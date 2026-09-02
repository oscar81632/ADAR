#!/usr/bin/env python3
"""Visualize ViT decoy-aware readout gates on VSTD samples."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.vstd.dataset import VSTDDataset, collate_vstd, load_json
from train_vstd_clip_alignment import build_text_features, load_clip


def load_args(checkpoint_args, dataset_dir, device):
    args = dict(checkpoint_args)
    args["dataset_dir"] = Path(dataset_dir)
    args["output_dir"] = Path(args.get("output_dir", "runs/unknown"))
    args["device"] = device
    return SimpleNamespace(**args)


def vit_tokens(visual, images):
    x = visual.conv1(images.type(visual.conv1.weight.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype)
    cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    return x.permute(1, 0, 2)


@torch.no_grad()
def gate_values(model, images):
    tokens = vit_tokens(model.visual, images)
    cls = tokens[:, 0, :]
    patches = tokens[:, 1:, :]
    readout = model.visual.path_readout
    gates = torch.sigmoid(readout.token_gate(patches)).squeeze(-1)
    alpha = torch.sigmoid(readout.router(cls)).squeeze(-1) if hasattr(readout, "router") else torch.ones_like(gates[:, 0])
    return gates.float().cpu(), alpha.float().cpu()


def overlay_gate(image_path, gates, output_path, alpha_value, title):
    image = Image.open(image_path).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    grid = int(gates.numel() ** 0.5)
    cell_w = image.width / grid
    cell_h = image.height / grid
    values = gates.reshape(grid, grid)
    values = (values - values.min()) / (values.max() - values.min()).clamp_min(1e-6)
    for row in range(grid):
        for col in range(grid):
            value = float(values[row, col].item())
            strength = value**1.8
            color = (255, int(230 * (1.0 - strength)), 0, int(185 * strength))
            x0 = int(col * cell_w)
            y0 = int(row * cell_h)
            x1 = int((col + 1) * cell_w)
            y1 = int((row + 1) * cell_h)
            draw.rectangle([x0, y0, x1, y1], fill=color)
    merged = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 32), "white")
    canvas.paste(merged, (0, 32))
    text_draw = ImageDraw.Draw(canvas)
    text_draw.text((6, 8), f"alpha={alpha_value:.3f} | {title}", fill=(0, 0, 0))
    canvas.save(output_path)


def path_gate_score(gates, row):
    grid = int(gates.numel() ** 0.5)
    values = gates.reshape(grid, grid)
    mask = torch.zeros_like(values, dtype=torch.bool)
    for cell in row.get("path", []):
        r, c = int(cell[0]), int(cell[1])
        if 0 <= r < grid and 0 <= c < grid:
            mask[r, c] = True
    if not mask.any() or (~mask).sum() == 0:
        return None
    path_mean = values[mask].mean()
    nonpath_mean = values[~mask].mean()
    endpoint = row.get("end")
    endpoint_gate = None
    if endpoint is not None:
        r, c = int(endpoint[0]), int(endpoint[1])
        if 0 <= r < grid and 0 <= c < grid:
            endpoint_gate = values[r, c]
    return {
        "path_mean": float(path_mean.item()),
        "nonpath_mean": float(nonpath_mean.item()),
        "path_enrichment": float((path_mean - nonpath_mean).item()),
        "endpoint_gate": float(endpoint_gate.item()) if endpoint_gate is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize ADAR/decoy-aware patch gates.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--min_path_length", type=int, default=48)
    parser.add_argument("--scan_limit", type=int, default=512)
    parser.add_argument("--select_by", choices=["first", "path_enrichment", "endpoint_gate"], default="first")
    parser.add_argument("--correct_only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_path = args.run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = args.run_dir / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"Loaded checkpoint: {checkpoint_path}", flush=True)
    model_args = load_args(checkpoint["args"], args.dataset_dir, args.device)
    device = torch.device(args.device)
    model, preprocess, clip_module = load_clip(model_args, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    classes = load_json(args.dataset_dir / "classes.json")
    prompts = [item["prompt"] for item in classes]
    text_features = build_text_features(model, clip_module, prompts, device)

    dataset = VSTDDataset(args.dataset_dir, args.split, transform=preprocess)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_vstd)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    seen = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        metadata = batch["metadata"]
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features.float(), dim=-1)
        logits = model.logit_scale.exp().detach().float() * image_features @ text_features.t()
        preds = logits.argmax(dim=1).cpu()
        gates, alphas = gate_values(model, images)
        labels_cpu = labels.cpu()
        for idx, row in enumerate(metadata):
            if int(row["path_length"]) < args.min_path_length:
                continue
            if args.correct_only and int(preds[idx].item()) != int(labels_cpu[idx].item()):
                continue
            image_path = args.dataset_dir / row["image"]
            title = f"label={labels_cpu[idx].item()} pred={preds[idx].item()} L={row['path_length']} decoy={row['decoy_arrow_count']}"
            scores = path_gate_score(gates[idx], row) or {}
            candidates.append(
                {
                    "image_path": image_path,
                    "gates": gates[idx],
                    "alpha": float(alphas[idx].item()),
                    "title": title,
                    "image": row["image"],
                    "label": int(labels_cpu[idx].item()),
                    "pred": int(preds[idx].item()),
                    "path_length": int(row["path_length"]),
                    "decoy_arrow_count": int(row["decoy_arrow_count"]),
                    **scores,
                }
            )
            seen += 1
            if args.select_by == "first" and len(candidates) >= args.num_samples:
                break
            if args.select_by != "first" and seen >= args.scan_limit:
                break
        if args.select_by == "first" and len(candidates) >= args.num_samples:
            break
        if args.select_by != "first" and seen >= args.scan_limit:
            break

    if args.select_by != "first":
        candidates.sort(key=lambda item: item.get(args.select_by) if item.get(args.select_by) is not None else -1e9, reverse=True)

    summary = []
    for saved, item in enumerate(candidates[: args.num_samples]):
        output_path = args.output_dir / f"{saved:03d}_L{item['path_length']}_decoy{item['decoy_arrow_count']}.png"
        overlay_gate(item["image_path"], item["gates"], output_path, item["alpha"], item["title"])
        record = {key: value for key, value in item.items() if key not in {"image_path", "gates", "title"}}
        record["output"] = str(output_path)
        summary.append(record)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
