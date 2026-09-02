#!/usr/bin/env python3
"""Exploratory logits-level ADAR fusion for OpenAI CLIP ViT-B/16 on VSTD.

This script is intentionally separate from the main thesis training script. It
tests whether preserving the vanilla CLS logits and adding a gated path-logit
branch helps VSTD generalization.
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adar_clip import import_clip
from src.vstd.dataset import VSTDDataset, collate_vstd, load_json


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class LateFusionReadout(nn.Module):
    def __init__(self, width: int, bottleneck: int = 128, dropout: float = 0.1):
        super().__init__()
        self.token_gate = nn.Sequential(OrderedDict([
            ("ln", nn.LayerNorm(width)),
            ("down", nn.Linear(width, bottleneck)),
            ("act", QuickGELU()),
            ("up", nn.Linear(bottleneck, 1)),
        ]))
        self.router = nn.Sequential(OrderedDict([
            ("ln", nn.LayerNorm(width)),
            ("down", nn.Linear(width, bottleneck)),
            ("act", QuickGELU()),
            ("up", nn.Linear(bottleneck, 1)),
        ]))
        self.fuse = nn.Sequential(OrderedDict([
            ("ln", nn.LayerNorm(width * 3)),
            ("down", nn.Linear(width * 3, width)),
            ("act", QuickGELU()),
            ("drop", nn.Dropout(dropout)),
            ("up", nn.Linear(width, width)),
        ]))

    def forward(self, cls_tokens, patch_tokens):
        gates = torch.sigmoid(self.token_gate(patch_tokens)).squeeze(-1).clamp(1e-4, 1.0 - 1e-4)
        keep_weights = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
        decoy_scores = 1.0 - gates
        decoy_weights = decoy_scores / decoy_scores.sum(dim=1, keepdim=True).clamp_min(1e-6)
        kept = torch.einsum("bn,bnd->bd", keep_weights, patch_tokens)
        decoy = torch.einsum("bn,bnd->bd", decoy_weights, patch_tokens)
        alpha = torch.sigmoid(self.router(cls_tokens)).to(dtype=cls_tokens.dtype)
        path_state = self.fuse(torch.cat([cls_tokens, alpha * kept, alpha * (kept - decoy)], dim=-1))
        return path_state, alpha


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_requires_grad(module_or_param, value: bool):
    if isinstance(module_or_param, torch.nn.Parameter):
        module_or_param.requires_grad = value
    elif hasattr(module_or_param, "parameters"):
        for param in module_or_param.parameters():
            param.requires_grad = value


def encode_vit_tokens(visual, images):
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


def project_visual_tokens(visual, tokens):
    tokens = visual.ln_post(tokens)
    if visual.proj is not None:
        tokens = tokens @ visual.proj
    return tokens


@torch.no_grad()
def build_text_features(model, clip_module, prompts, device):
    tokens = clip_module.tokenize(prompts).to(device)
    was_training = model.training
    model.eval()
    text_features = model.encode_text(tokens)
    text_features = F.normalize(text_features.float(), dim=-1)
    if was_training:
        model.train()
    return text_features


def logits_from_images(model, readout, images, text_features):
    visual = model.visual
    tokens = encode_vit_tokens(visual, images)
    cls_tokens = tokens[:, 0, :]
    patch_tokens = tokens[:, 1:, :]
    path_tokens, alpha = readout(cls_tokens, patch_tokens)
    cls_features = F.normalize(project_visual_tokens(visual, cls_tokens).float(), dim=-1)
    path_features = F.normalize(project_visual_tokens(visual, path_tokens).float(), dim=-1)
    logit_scale = model.logit_scale.exp().detach().float()
    cls_logits = logit_scale * cls_features @ text_features.t()
    path_logits = logit_scale * path_features @ text_features.t()
    return cls_logits + alpha.float() * path_logits


@torch.no_grad()
def evaluate(model, readout, loader, text_features, device):
    model.eval()
    readout.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    metadata = []
    all_preds = []
    all_labels = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = logits_from_images(model, readout, images, text_features)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)
        loss_sum += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        metadata.extend(batch["metadata"])
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    return {
        "loss": loss_sum / max(1, total),
        "acc": correct / max(1, total),
        "preds": torch.cat(all_preds, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "metadata": metadata,
    }


def accuracy_by_bucket(preds, labels, metadata, key):
    buckets = {}
    for idx, row in enumerate(metadata):
        value = row.get(key, 0)
        bucket = buckets.setdefault(value, [0, 0])
        bucket[0] += int(preds[idx].item() == labels[idx].item())
        bucket[1] += 1
    return {str(k): v[0] / max(1, v[1]) for k, v in sorted(buckets.items())}


def main():
    parser = argparse.ArgumentParser(description="Run exploratory logits-level ADAR late fusion on VSTD.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--clip_model", type=str, default="ViT-B/16")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    clip_module = import_clip()
    model, preprocess = clip_module.load(args.clip_model, device=device, jit=False, vit_path_adapter="none")
    model = model.float()
    width = model.visual.class_embedding.shape[0]
    readout = LateFusionReadout(width=width, bottleneck=args.bottleneck, dropout=args.dropout).to(device)

    for param in model.parameters():
        param.requires_grad = False
    set_requires_grad(model.visual, True)
    for param in model.parameters():
        if param is model.logit_scale:
            param.requires_grad = False

    trainable_params = [p for p in list(model.visual.parameters()) + list(readout.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    classes = load_json(args.dataset_dir / "classes.json")
    prompts = [item["prompt"] for item in classes]
    text_features = build_text_features(model, clip_module, prompts, device)
    datasets = {
        split: VSTDDataset(args.dataset_dir, split, transform=preprocess)
        for split in ["train", "val", "test"]
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=collate_vstd),
        "val": DataLoader(datasets["val"], batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=collate_vstd),
        "test": DataLoader(datasets["test"], batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=collate_vstd),
    }

    print("Runtime:", flush=True)
    print(f"  device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"  cuda_device_name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  clip_model: {args.clip_model}", flush=True)
    print(f"  seed: {args.seed}", flush=True)
    print(f"  trainable_params: {sum(p.numel() for p in trainable_params):,}", flush=True)
    print(f"  prompts: {prompts}", flush=True)

    best_val_acc = -1.0
    best_epoch = 0
    best_path = args.output_dir / "best_model.pt"
    with (args.output_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "train_time_sec"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            readout.train()
            total = 0
            correct = 0
            loss_sum = 0.0
            start_time = time.perf_counter()
            for batch in loaders["train"]:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = logits_from_images(model, readout, images, text_features)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                preds = logits.argmax(dim=1)
                loss_sum += loss.item() * labels.size(0)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_time = time.perf_counter() - start_time
            val = evaluate(model, readout, loaders["val"], text_features, device)
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / max(1, total),
                "train_acc": correct / max(1, total),
                "val_loss": val["loss"],
                "val_acc": val["acc"],
                "train_time_sec": train_time,
            }
            writer.writerow(row)
            f.flush()
            if val["acc"] > best_val_acc:
                best_val_acc = val["acc"]
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "readout": readout.state_dict(), "args": vars(args), "best_epoch": best_epoch}, best_path)
            print(
                f"epoch={epoch} train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} "
                f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} time={train_time:.2f}s",
                flush=True,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    readout.load_state_dict(checkpoint["readout"], strict=True)
    text_features = build_text_features(model, clip_module, prompts, device)
    test = evaluate(model, readout, loaders["test"], text_features, device)
    result = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test["loss"],
        "test_acc": test["acc"],
        "accuracy_by_path_length": accuracy_by_bucket(test["preds"], test["labels"], test["metadata"], "path_length"),
        "accuracy_by_distractor_ratio": accuracy_by_bucket(test["preds"], test["labels"], test["metadata"], "distractor_ratio"),
        "accuracy_by_decoy_arrow_count": accuracy_by_bucket(test["preds"], test["labels"], test["metadata"], "decoy_arrow_count"),
        "accuracy_by_start_end_manhattan": accuracy_by_bucket(test["preds"], test["labels"], test["metadata"], "start_end_manhattan"),
        "prompts": prompts,
        "method": "experimental_late_fusion_adar",
    }
    with (args.output_dir / "test_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    torch.save({"model": model.state_dict(), "readout": readout.state_dict(), "args": vars(args)}, args.output_dir / "model.pt")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
