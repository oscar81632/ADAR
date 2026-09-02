#!/usr/bin/env python3
"""Exploratory HuggingFace VLM baselines for VSTD.

This script intentionally lives outside the main ADAR result path. It trains
vanilla image-text endpoint classifiers for modern CLIP-style models such as
SigLIP and EVA-CLIP, while keeping the text tower frozen.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vstd.dataset import VSTDDataset, load_json


class ADARReadout(nn.Module):
    """Adaptive decoy-aware readout over final patch tokens."""

    def __init__(self, width: int, bottleneck: int = 128, dropout: float = 0.1, residual_scale: float = 1.0):
        super().__init__()
        self.gate = nn.Linear(width, 1)
        self.route = nn.Linear(width, 1)
        self.fusion = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, width),
        )
        self.residual_scale = residual_scale

    def forward(self, patch_tokens, global_token):
        gates = torch.sigmoid(self.gate(patch_tokens))
        weights = gates / (gates.sum(dim=1, keepdim=True) + 1e-6)
        z_keep = (weights * patch_tokens).sum(dim=1)
        z_supp = patch_tokens.mean(dim=1) - z_keep
        alpha = torch.sigmoid(self.route(global_token))
        fusion_input = torch.cat([global_token, alpha * z_keep, alpha * z_supp], dim=-1)
        return global_token + self.residual_scale * self.fusion(fusion_input)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pil_collate(batch):
    return {
        "image": [item["image"] for item in batch],
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "metadata": [item["metadata"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
        "class_name": [item["class_name"] for item in batch],
    }


def move_to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def load_hf_model(model_id: str, device, trust_remote_code: bool = False):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = model.float().to(device)
    return model, processor


def add_adar_readout(model, bottleneck: int, dropout: float, residual_scale: float):
    width = getattr(model.config.vision_config, "hidden_size", None)
    if width is None:
        raise RuntimeError("Could not infer vision hidden size from model.config.vision_config.hidden_size.")
    model.vstd_adar_readout = ADARReadout(width, bottleneck, dropout, residual_scale)


def processor_text_kwargs(processor, prompts):
    try:
        return processor(text=prompts, padding="max_length", return_tensors="pt")
    except Exception:
        return processor(text=prompts, padding=True, return_tensors="pt")


@torch.no_grad()
def build_text_features(model, processor, prompts, device):
    model.eval()
    text_inputs = move_to_device(processor_text_kwargs(processor, prompts), device)
    if hasattr(model, "get_text_features"):
        text_features = model.get_text_features(**text_inputs)
    else:
        outputs = model(**text_inputs)
        text_features = getattr(outputs, "text_embeds", None)
        if text_features is None:
            raise RuntimeError("Model does not expose get_text_features or text_embeds.")
    return F.normalize(text_features.float(), dim=-1)


def encode_images(model, processor, images, device):
    image_inputs = processor(images=images, return_tensors="pt")
    image_inputs = move_to_device(image_inputs, device)
    if hasattr(model, "vstd_adar_readout"):
        vision_outputs = model.vision_model(
            pixel_values=image_inputs["pixel_values"],
            output_hidden_states=False,
            return_dict=True,
        )
        image_features = model.vstd_adar_readout(
            vision_outputs.last_hidden_state.float(),
            vision_outputs.pooler_output.float(),
        )
        return F.normalize(image_features.float(), dim=-1)
    if hasattr(model, "get_image_features"):
        image_features = model.get_image_features(**image_inputs)
    else:
        outputs = model(**image_inputs)
        image_features = getattr(outputs, "image_embeds", None)
        if image_features is None:
            raise RuntimeError("Model does not expose get_image_features or image_embeds.")
    return F.normalize(image_features.float(), dim=-1)


def logits_from_features(model, image_features, text_features):
    logits = image_features @ text_features.t()
    logit_scale = getattr(model, "logit_scale", None)
    if logit_scale is not None:
        logits = logits * logit_scale.exp().detach().float()
    logit_bias = getattr(model, "logit_bias", None)
    if logit_bias is not None:
        logits = logits + logit_bias.detach().float()
    return logits


def configure_trainability(model):
    for param in model.parameters():
        param.requires_grad = False

    trainable_roots = [
        "vision_model",
        "vision_tower",
        "visual",
        "image_encoder",
        "visual_projection",
        "vision_projection",
        "vstd_adar_readout",
    ]
    for name, param in model.named_parameters():
        if any(name == root or name.startswith(root + ".") for root in trainable_roots):
            param.requires_grad = True

    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError(
            "No trainable vision-side parameters found. Inspect model.named_parameters() "
            "and extend trainable_roots for this architecture."
        )
    return params


@torch.no_grad()
def evaluate(model, processor, loader, text_features, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    all_preds = []
    all_labels = []
    metadata = []
    for batch in loader:
        labels = batch["label"].to(device, non_blocking=True)
        image_features = encode_images(model, processor, batch["image"], device)
        logits = logits_from_features(model, image_features, text_features)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)
        loss_sum += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        metadata.extend(batch["metadata"])
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


def count_params(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def save_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def jsonable_args(args):
    payload = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def main():
    parser = argparse.ArgumentParser(description="Train exploratory HF VLM baselines on VSTD.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--readout", choices=["none", "adar"], default="none")
    parser.add_argument("--adar_bottleneck", type=int, default=128)
    parser.add_argument("--adar_dropout", type=float, default=0.1)
    parser.add_argument("--adar_residual_scale", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    eval_batch_size = args.eval_batch_size or args.batch_size

    model, processor = load_hf_model(args.model_id, device, args.trust_remote_code)
    if args.readout == "adar":
        add_adar_readout(model, args.adar_bottleneck, args.adar_dropout, args.adar_residual_scale)
        model.to(device)
    trainable_params = configure_trainability(model)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    classes = load_json(args.dataset_dir / "classes.json")
    prompts = [item["prompt"] for item in classes]
    text_features = build_text_features(model, processor, prompts, device)

    datasets = {
        split: VSTDDataset(args.dataset_dir, split, return_pil=True)
        for split in ["train", "val", "test"]
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=pil_collate,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=pil_collate,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=pil_collate,
        ),
    }

    total_params, trainable_count = count_params(model)
    runtime = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_id": args.model_id,
        "readout": args.readout,
        "total_params": total_params,
        "trainable_params": trainable_count,
        "prompts": prompts,
    }
    save_json(args.output_dir / "args.json", {**jsonable_args(args), **runtime})
    print(json.dumps(runtime, indent=2), flush=True)

    best_val_acc = -1.0
    best_epoch = 0
    best_model_path = args.output_dir / "best_model.pt"
    log_path = args.output_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "train_time_sec"],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0
            correct = 0
            loss_sum = 0.0
            start_time = time.perf_counter()
            for batch in loaders["train"]:
                labels = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                image_features = encode_images(model, processor, batch["image"], device)
                logits = logits_from_features(model, image_features, text_features)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                preds = logits.argmax(dim=1)
                loss_sum += loss.item() * labels.size(0)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_time = time.perf_counter() - start_time
            val_result = evaluate(model, processor, loaders["val"], text_features, device)
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / max(1, total),
                "train_acc": correct / max(1, total),
                "val_loss": val_result["loss"],
                "val_acc": val_result["acc"],
                "train_time_sec": train_time,
            }
            writer.writerow(row)
            f.flush()
            print(row, flush=True)
            if val_result["acc"] > best_val_acc:
                best_val_acc = val_result["acc"]
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "args": jsonable_args(args)}, best_model_path)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_result = evaluate(model, processor, loaders["test"], text_features, device)
    result_payload = {
        "model_id": args.model_id,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_result["loss"],
        "test_acc": test_result["acc"],
        "path_length_acc": accuracy_by_bucket(
            test_result["preds"], test_result["labels"], test_result["metadata"], "path_length"
        ),
        "decoy_arrow_count_acc": accuracy_by_bucket(
            test_result["preds"], test_result["labels"], test_result["metadata"], "decoy_arrow_count"
        ),
        "total_params": total_params,
        "trainable_params": trainable_count,
    }
    save_json(args.output_dir / "test_result.json", result_payload)
    print(json.dumps(result_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
