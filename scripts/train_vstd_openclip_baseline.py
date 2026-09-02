#!/usr/bin/env python3
"""Exploratory OpenCLIP/EVA-CLIP baselines for VSTD."""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vstd.dataset import VSTDDataset, collate_vstd, load_json


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_requires_grad(module, value: bool):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = value


def configure_trainability(model):
    for param in model.parameters():
        param.requires_grad = False
    set_requires_grad(getattr(model, "visual", None), True)
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable visual parameters found.")
    return params


@torch.no_grad()
def build_text_features(model, tokenizer, prompts, device):
    was_training = model.training
    model.eval()
    tokens = tokenizer(prompts).to(device)
    text_features = model.encode_text(tokens)
    text_features = F.normalize(text_features.float(), dim=-1)
    if was_training:
        model.train()
    return text_features


def image_features_from_images(model, images):
    image_features = model.encode_image(images)
    return F.normalize(image_features.float(), dim=-1)


def logits_from_image_features(model, image_features, text_features):
    logit_scale = model.logit_scale.exp().detach().float()
    return logit_scale * image_features @ text_features.t()


@torch.no_grad()
def evaluate(model, loader, text_features, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    all_preds = []
    all_labels = []
    metadata = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        image_features = image_features_from_images(model, images)
        logits = logits_from_image_features(model, image_features, text_features)
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


def jsonable_args(args):
    payload = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def save_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train exploratory OpenCLIP baselines on VSTD.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="EVA02-B-16")
    parser.add_argument("--pretrained", type=str, default="merged2b_s8b_b131k")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    eval_batch_size = args.eval_batch_size or args.batch_size

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device=device,
    )
    model = model.float()
    tokenizer = open_clip.get_tokenizer(args.model_name)
    trainable_params = configure_trainability(model)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    classes = load_json(args.dataset_dir / "classes.json")
    prompts = [item["prompt"] for item in classes]
    text_features = build_text_features(model, tokenizer, prompts, device)

    datasets = {
        split: VSTDDataset(args.dataset_dir, split, transform=preprocess)
        for split in ["train", "val", "test"]
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_vstd,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_vstd,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_vstd,
        ),
    }

    total_params = sum(param.numel() for param in model.parameters())
    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    runtime = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
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
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                image_features = image_features_from_images(model, images)
                logits = logits_from_image_features(model, image_features, text_features)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                preds = logits.argmax(dim=1)
                loss_sum += loss.item() * labels.size(0)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_time = time.perf_counter() - start_time
            val_result = evaluate(model, loaders["val"], text_features, device)
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
    test_result = evaluate(model, loaders["test"], text_features, device)
    result_payload = {
        "model_name": args.model_name,
        "pretrained": args.pretrained,
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
