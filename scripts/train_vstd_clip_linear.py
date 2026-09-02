#!/usr/bin/env python3
"""Frozen CLIP image-feature linear baseline for VSTD.

This script uses the repo-local modified CLIP package in `external/clip_vstd`.
It still requires the Python runtime dependencies, including torchvision.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adar_clip import import_clip
from src.vstd.dataset import VSTDDataset, collate_vstd, load_json


def load_clip(args, device):
    try:
        clip = import_clip()
    except Exception as exc:
        raise RuntimeError(
            "Unable to import the repo-local modified CLIP package. Expected it at "
            "external/clip_vstd inside this research repo."
        ) from exc

    kwargs = {}
    if args.encoder == "vim":
        # Match the archived Vim baseline: Vim keeps the Vim visual backbone while
        # using the ViT-B/16 CLIP text side and projection dimensionality.
        args.clip_model = "ViT-B/16"
        kwargs.update(
            {
                "vision_backbone": "vim",
                "vision_model_name": args.vim_model_name,
                "vision_ckpt_path": args.vim_ckpt,
            }
        )
    model, preprocess = clip.load(args.clip_model, device=device, jit=False, **kwargs)
    model = model.float().eval()
    return model, preprocess


@torch.no_grad()
def extract_features(model, loader, device):
    features = []
    labels = []
    metadata = []
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        feats = model.encode_image(images)
        feats = F.normalize(feats.float(), dim=-1)
        features.append(feats.cpu())
        labels.append(batch["label"].cpu())
        metadata.extend(batch["metadata"])
        if step % 10 == 0:
            seen = sum(chunk.shape[0] for chunk in features)
            print(f"  extracted {seen} samples", flush=True)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0), metadata


def maybe_subset(dataset, max_samples):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def feature_cache_path(output_dir, split):
    return output_dir / "features" / f"{split}.pt"


def load_or_extract_features(model, loader, device, output_dir, split, use_cache):
    path = feature_cache_path(output_dir, split)
    if use_cache and path.exists():
        payload = torch.load(path, map_location="cpu")
        print(f"Loaded cached {split} features: {path}", flush=True)
        return payload["features"], payload["labels"], payload["metadata"]

    print(f"Extracting frozen features for split={split}...", flush=True)
    features, labels, metadata = extract_features(model, loader, device)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "labels": labels, "metadata": metadata}, path)
        print(f"Saved cached {split} features: {path}", flush=True)
    return features, labels, metadata


def train_linear(train_x, train_y, val_x, val_y, num_classes, epochs, lr, batch_size, device, output_dir):
    head = nn.Linear(train_x.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    log_path = output_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(train_x.shape[0], device=device)
            total = 0
            correct = 0
            loss_sum = 0.0
            head.train()
            for start in range(0, train_x.shape[0], batch_size):
                idx = perm[start:start + batch_size]
                x = train_x[idx]
                y = train_y[idx]
                optimizer.zero_grad(set_to_none=True)
                logits = head(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * y.size(0)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.size(0)

            head.eval()
            with torch.no_grad():
                val_logits = head(val_x)
                val_loss = F.cross_entropy(val_logits, val_y).item()
                val_acc = (val_logits.argmax(dim=1) == val_y).float().mean().item()
            train_loss = loss_sum / max(1, total)
            train_acc = correct / max(1, total)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            writer.writerow(row)
            f.flush()
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
    return head


def accuracy_by_bucket(logits, labels, metadata, key):
    preds = logits.argmax(dim=1).cpu()
    labels = labels.cpu()
    buckets = {}
    for idx, row in enumerate(metadata):
        value = row[key]
        bucket = buckets.setdefault(value, [0, 0])
        bucket[0] += int(preds[idx].item() == labels[idx].item())
        bucket[1] += 1
    return {str(k): v[0] / max(1, v[1]) for k, v in sorted(buckets.items())}


def main():
    parser = argparse.ArgumentParser(description="Train a frozen CLIP linear baseline on VSTD.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("runs/clip_linear_vstd"))
    parser.add_argument("--encoder", choices=["vit", "vim"], default="vit")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16")
    parser.add_argument("--vim_model_name", type=str, default="vim_base")
    parser.add_argument("--vim_ckpt", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--no_feature_cache", action="store_true")
    args = parser.parse_args()

    if args.encoder == "vim" and not args.vim_ckpt:
        raise ValueError("--vim_ckpt is required for --encoder vim")
    if args.encoder == "vim" and (args.device == "cpu" or not torch.cuda.is_available()):
        raise RuntimeError(
            "Vim/Mamba baseline requires CUDA in this environment because "
            "causal_conv1d and Triton kernels are CUDA-only. Run on a GPU node "
            "or use --encoder vit for CPU baselines."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, preprocess = load_clip(args, device)

    datasets = {
        split: VSTDDataset(args.dataset_dir, split, transform=preprocess)
        for split in ["train", "val", "test"]
    }
    datasets = {split: maybe_subset(ds, args.max_samples_per_split) for split, ds in datasets.items()}
    loaders = {
        split: DataLoader(
            ds,
            batch_size=args.feature_batch_size,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_vstd,
        )
        for split, ds in datasets.items()
    }

    print("Runtime:", flush=True)
    print(f"  device: {device}", flush=True)
    print(f"  cuda_available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  cuda_device_count: {torch.cuda.device_count()}", flush=True)
        print(f"  cuda_device_name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  cuda_version: {torch.version.cuda}", flush=True)
    print(f"  encoder: {args.encoder}", flush=True)
    print(f"  clip_model: {args.clip_model}", flush=True)
    print("Dataset sizes:", {split: len(ds) for split, ds in datasets.items()}, flush=True)
    use_cache = not args.no_feature_cache
    train_x, train_y, _ = load_or_extract_features(model, loaders["train"], device, args.output_dir, "train", use_cache)
    val_x, val_y, _ = load_or_extract_features(model, loaders["val"], device, args.output_dir, "val", use_cache)
    test_x, test_y, test_meta = load_or_extract_features(model, loaders["test"], device, args.output_dir, "test", use_cache)

    head = train_linear(
        train_x,
        train_y,
        val_x,
        val_y,
        num_classes=len(load_json(args.dataset_dir / "classes.json")),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        output_dir=args.output_dir,
    )

    head.eval()
    with torch.no_grad():
        test_logits = head(test_x.to(device)).cpu()
        test_loss = F.cross_entropy(test_logits, test_y).item()
        test_acc = (test_logits.argmax(dim=1) == test_y).float().mean().item()

    result = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "accuracy_by_path_length": accuracy_by_bucket(test_logits, test_y, test_meta, "path_length"),
        "accuracy_by_distractor_ratio": accuracy_by_bucket(test_logits, test_y, test_meta, "distractor_ratio"),
        "accuracy_by_start_end_manhattan": accuracy_by_bucket(test_logits, test_y, test_meta, "start_end_manhattan"),
    }
    with (args.output_dir / "test_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    torch.save({"head": head.state_dict(), "args": vars(args)}, args.output_dir / "linear_head.pt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
