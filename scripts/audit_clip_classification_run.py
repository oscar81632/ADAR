#!/usr/bin/env python3
"""Audit prediction distribution for a CLIP-style VSTD-compatible run."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_vstd_clip_alignment import build_text_features, load_clip, logits_from_images
from src.vstd.dataset import VSTDDataset, collate_vstd, load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--vit_path_adapter", default="none")
    parser.add_argument("--clip_model", default="ViT-B/16")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    load_args = SimpleNamespace(
        encoder="vit",
        clip_model=args.clip_model,
        vim_model_name="vim_base",
        vim_ckpt=None,
        path_adapter_layers=0,
        train_scope="full",
        path_adapter_bottleneck=128,
        path_adapter_dropout=0.1,
        path_adapter_pooling="forward",
        vit_path_adapter=args.vit_path_adapter,
        vit_path_adapter_bottleneck=128,
        vit_path_adapter_dropout=0.1,
        vit_path_adapter_hops=4,
        vit_path_adapter_residual_scale=1.0,
    )
    device = torch.device(args.device)
    model, preprocess, clip_module = load_clip(load_args, device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)

    classes = load_json(args.dataset_dir / "classes.json")
    prompts = [item["prompt"] for item in classes]
    text_features = build_text_features(model, clip_module, prompts, device)
    ds = VSTDDataset(args.dataset_dir, args.split, transform=preprocess)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_vstd)

    num_classes = len(classes)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    pred_counts = torch.zeros(num_classes, dtype=torch.long)
    label_counts = torch.zeros(num_classes, dtype=torch.long)
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = logits_from_images(model, images, text_features)
            losses.append(torch.nn.functional.cross_entropy(logits, labels, reduction="sum").cpu())
            preds = logits.argmax(dim=1)
            for label, pred in zip(labels.cpu(), preds.cpu()):
                confusion[int(label), int(pred)] += 1
                pred_counts[int(pred)] += 1
                label_counts[int(label)] += 1
    total = int(label_counts.sum())
    correct = int(confusion.diag().sum())
    result = {
        "dataset_dir": str(args.dataset_dir),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "split": args.split,
        "vit_path_adapter": args.vit_path_adapter,
        "acc": correct / max(1, total),
        "loss": float(torch.stack(losses).sum().item() / max(1, total)),
        "label_counts": label_counts.tolist(),
        "pred_counts": pred_counts.tolist(),
        "confusion_rows_label_cols_pred": confusion.tolist(),
        "prompts": prompts,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
