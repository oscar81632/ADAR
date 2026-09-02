#!/usr/bin/env python3
"""CLIP-style image-classification fine-tuning for exploratory ADAR comparisons.

This script intentionally lives outside the main VSTD pipeline. It reuses the
repo-local modified CLIP implementation, freezes the text tower, and fine-tunes
only the visual side on CIFAR-10 or STL-10 class prompts.
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
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, STL10

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adar_clip import import_clip


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

STL10_CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


DATASET_CLASSES = {
    "cifar10": CIFAR10_CLASSES,
    "stl10": STL10_CLASSES,
}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_requires_grad(module_or_param, value: bool):
    if module_or_param is None:
        return
    if isinstance(module_or_param, torch.nn.Parameter):
        module_or_param.requires_grad = value
    elif hasattr(module_or_param, "parameters"):
        for param in module_or_param.parameters():
            param.requires_grad = value


def load_clip(args, device):
    clip = import_clip()
    model, preprocess = clip.load(
        args.clip_model,
        device=device,
        jit=False,
        vit_path_adapter=args.vit_path_adapter,
        vit_path_adapter_bottleneck=args.vit_path_adapter_bottleneck,
        vit_path_adapter_dropout=args.vit_path_adapter_dropout,
        vit_path_adapter_hops=args.vit_path_adapter_hops,
        vit_path_adapter_residual_scale=args.vit_path_adapter_residual_scale,
    )
    return model.float(), preprocess, clip


def configure_full_visual_ft(model):
    for param in model.parameters():
        param.requires_grad = False
    set_requires_grad(model.visual, True)
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable visual parameters found.")
    return params


def configure_trainability(model, train_scope: str):
    for param in model.parameters():
        param.requires_grad = False

    visual = model.visual
    if train_scope == "full":
        set_requires_grad(visual, True)
    elif train_scope == "projection":
        set_requires_grad(getattr(visual, "ln_post", None), True)
        set_requires_grad(getattr(visual, "proj", None), True)
    elif train_scope == "readout_projection":
        set_requires_grad(getattr(visual, "path_readout", None), True)
        set_requires_grad(getattr(visual, "ln_post", None), True)
        set_requires_grad(getattr(visual, "proj", None), True)
        if getattr(visual, "path_readout", None) is None:
            raise ValueError("--train_scope readout_projection requires --vit_path_adapter != none")
    else:
        raise ValueError(f"Unsupported train_scope: {train_scope}")

    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable visual parameters found.")
    return params


def collate_cifar(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return {"image": images, "label": labels}


def dataset_targets(dataset):
    if hasattr(dataset, "targets"):
        return [int(label) for label in dataset.targets]
    if hasattr(dataset, "labels"):
        return [int(label) for label in dataset.labels]
    raise ValueError(f"Cannot find targets/labels on dataset type {type(dataset).__name__}.")


def balanced_indices_by_class(dataset, class_names, per_class: int, rng: random.Random):
    by_class = {idx: [] for idx in range(len(class_names))}
    for idx, label in enumerate(dataset_targets(dataset)):
        by_class[int(label)].append(idx)
    selected = []
    for label, indices in by_class.items():
        if len(indices) < per_class:
            raise ValueError(f"Class {label} has only {len(indices)} samples, cannot select {per_class}.")
        label_indices = indices[:]
        rng.shuffle(label_indices)
        selected.extend(label_indices[:per_class])
    rng.shuffle(selected)
    return selected


def build_datasets(args, preprocess):
    if args.dataset == "cifar10":
        trainval = CIFAR10(root=str(args.data_root), train=True, download=args.download, transform=preprocess)
        test = CIFAR10(root=str(args.data_root), train=False, download=args.download, transform=preprocess)
    elif args.dataset == "stl10":
        trainval = STL10(root=str(args.data_root), split="train", download=args.download, transform=preprocess)
        test = STL10(root=str(args.data_root), split="test", download=args.download, transform=preprocess)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    rng = random.Random(args.split_seed)
    if args.train_per_class > 0:
        train_indices = balanced_indices_by_class(trainval, DATASET_CLASSES[args.dataset], args.train_per_class, rng)
        train_index_set = set(train_indices)
        remaining = [idx for idx in range(len(trainval)) if idx not in train_index_set]
        rng.shuffle(remaining)
        val_indices = remaining[: min(args.val_size, len(remaining))]
    else:
        indices = list(range(len(trainval)))
        rng.shuffle(indices)
        val_size = min(args.val_size, max(1, len(indices) // 10))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
    return {
        "train": Subset(trainval, train_indices),
        "val": Subset(trainval, val_indices),
        "test": test,
    }


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


def image_features_from_images(model, images):
    image_features = model.encode_image(images)
    return F.normalize(image_features.float(), dim=-1)


def logits_from_image_features(model, image_features, text_features):
    logit_scale = model.logit_scale.exp().detach().float()
    return logit_scale * image_features @ text_features.t()


@torch.no_grad()
def evaluate(model, loader, text_features, class_names, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    num_classes = text_features.size(0)
    class_total = [0 for _ in range(num_classes)]
    class_correct = [0 for _ in range(num_classes)]
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
        for label, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
            class_total[label] += 1
            class_correct[label] += int(label == pred)
    return {
        "loss": loss_sum / max(1, total),
        "acc": correct / max(1, total),
        "per_class_acc": {
            class_names[idx]: class_correct[idx] / max(1, class_total[idx])
            for idx in range(num_classes)
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP ViT-B/16 or ADAR readout on CIFAR-10/STL-10.")
    parser.add_argument("--dataset", choices=["cifar10", "stl10"], default="cifar10")
    parser.add_argument("--data_root", type=Path, default=Path("<LOCAL_ROOT>/CLIP/data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--clip_model", type=str, default="ViT-B/16")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_size", type=int, default=5000)
    parser.add_argument(
        "--train_per_class",
        type=int,
        default=0,
        help="If positive, use a balanced CIFAR-10 training subset with this many images per class.",
    )
    parser.add_argument("--split_seed", type=int, default=123)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="a photo of a {}",
        help="Prompt template with one {} placeholder for the CIFAR-10 class name.",
    )
    parser.add_argument(
        "--train_scope",
        choices=["full", "projection", "readout_projection"],
        default="full",
        help="Train the full visual tower, only CLIP visual projection, or ADAR readout plus projection.",
    )
    parser.add_argument(
        "--vit_path_adapter",
        choices=[
            "none",
            "adaptive_decoy_aware",
            "adaptive_decoy_aware_mean_supp",
            "adaptive_decoy_aware_no_outer_cls",
            "adaptive_decoy_aware_no_inner_cls",
            "adaptive_decoy_aware_static_alpha",
            "adaptive_keep_only",
            "adaptive_supp_only",
            "adaptive_supp_only_clean",
            "adaptive_keep_only_clean",
            "adaptive_keep_only_2branch",
        ],
        default="none",
    )
    parser.add_argument("--vit_path_adapter_bottleneck", type=int, default=128)
    parser.add_argument("--vit_path_adapter_dropout", type=float, default=0.1)
    parser.add_argument("--vit_path_adapter_hops", type=int, default=4)
    parser.add_argument("--vit_path_adapter_residual_scale", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, preprocess, clip_module = load_clip(args, device)
    trainable_params = configure_trainability(model, args.train_scope)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    class_names = DATASET_CLASSES[args.dataset]
    prompts = [args.prompt_template.format(name) for name in class_names]
    text_features = build_text_features(model, clip_module, prompts, device)
    datasets = build_datasets(args, preprocess)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_cifar,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_cifar,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_cifar,
        ),
    }

    config = vars(args).copy()
    config.update(
        {
            "dataset": args.dataset,
            "classes": class_names,
            "prompts": prompts,
            "train_size": len(datasets["train"]),
            "val_size_actual": len(datasets["val"]),
            "test_size": len(datasets["test"]),
            "trainable_params": sum(param.numel() for param in trainable_params),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    )
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    print("Runtime:", flush=True)
    print(f"  device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"  cuda_device_name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  clip_model: {args.clip_model}", flush=True)
    print(f"  dataset: {args.dataset}", flush=True)
    print(f"  vit_path_adapter: {args.vit_path_adapter}", flush=True)
    print(f"  train_scope: {args.train_scope}", flush=True)
    print(f"  train_per_class: {args.train_per_class}", flush=True)
    print(f"  train/val/test: {len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])}", flush=True)
    print(f"  trainable_params: {config['trainable_params']:,}", flush=True)
    print(f"  prompts: {prompts}", flush=True)

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
            train_loss = loss_sum / max(1, total)
            train_acc = correct / max(1, total)
            val = evaluate(model, loaders["val"], text_features, class_names, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val["loss"],
                "val_acc": val["acc"],
                "train_time_sec": train_time,
            }
            writer.writerow(row)
            f.flush()
            if val["acc"] > best_val_acc:
                best_val_acc = val["acc"]
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "args": config, "best_epoch": best_epoch}, best_model_path)
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} time={train_time:.2f}s",
                flush=True,
            )

    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        text_features = build_text_features(model, clip_module, prompts, device)

    test = evaluate(model, loaders["test"], text_features, class_names, device)
    result = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test["loss"],
        "test_acc": test["acc"],
        "per_class_acc": test["per_class_acc"],
        "prompts": prompts,
    }
    with (args.output_dir / "test_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    torch.save({"model": model.state_dict(), "args": config}, args.output_dir / "model.pt")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
