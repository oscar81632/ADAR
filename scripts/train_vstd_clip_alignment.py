#!/usr/bin/env python3
"""CLIP-style VSTD training for visual-side fine-tuning and readout ablations.

The text encoder is kept frozen and each VSTD endpoint class is represented by
one or more text prompts. The image side is trained with the standard CLIP-style
classification objective:

    logits = logit_scale * normalize(image_features) @ normalize(text_features).T
    loss = CE(logits, labels)

The --train_scope option controls whether the visual projection, selected visual
blocks, or the full visual side is fine-tuned. ViT readout variants such as ADAR
are selected with --vit_path_adapter.
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adar_clip import import_clip
from src.vstd.dataset import VSTDDataset, collate_vstd, load_json


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
    try:
        clip = import_clip()
    except Exception as exc:
        raise RuntimeError(
            "Unable to import the repo-local modified CLIP package. Expected it at "
            "external/clip_vstd inside this research repo."
        ) from exc

    kwargs = {}
    if args.encoder == "vim":
        args.clip_model = "ViT-B/16"
        kwargs.update(
            {
                "vision_backbone": "vim",
                "vision_model_name": args.vim_model_name,
                "vision_ckpt_path": args.vim_ckpt,
                "path_adapter_layers": args.path_adapter_layers if args.train_scope.startswith("path_adapter") else 0,
                "path_adapter_bottleneck": args.path_adapter_bottleneck,
                "path_adapter_dropout": args.path_adapter_dropout,
                "path_adapter_pooling": args.path_adapter_pooling,
            }
        )
    elif args.encoder == "vit":
        kwargs.update(
            {
                "vit_path_adapter": args.vit_path_adapter,
                "vit_path_adapter_bottleneck": args.vit_path_adapter_bottleneck,
                "vit_path_adapter_dropout": args.vit_path_adapter_dropout,
                "vit_path_adapter_hops": args.vit_path_adapter_hops,
                "vit_path_adapter_residual_scale": args.vit_path_adapter_residual_scale,
            }
        )
    model, preprocess = clip.load(args.clip_model, device=device, jit=False, **kwargs)
    model = model.float()
    return model, preprocess, clip


def set_last_k_trainable(blocks, k: int):
    total = len(blocks)
    if k <= 0 or k > total:
        raise ValueError(f"--last_k_blocks must be in [1, {total}], got {k}")
    for block in blocks[-k:]:
        set_requires_grad(block, True)


def configure_trainability(model, train_scope: str, last_k_blocks: int = 2):
    for param in model.parameters():
        param.requires_grad = False

    visual = model.visual
    if train_scope == "projection":
        set_requires_grad(getattr(visual, "ln_post", None), True)
        set_requires_grad(getattr(visual, "proj", None), True)
    elif train_scope == "backbone":
        set_requires_grad(getattr(visual, "backbone", None), True)
        if not hasattr(visual, "backbone"):
            raise ValueError("--train_scope backbone is only supported for Vim-style wrappers with .backbone")
    elif train_scope == "full":
        set_requires_grad(visual, True)
    elif train_scope == "last_k":
        set_requires_grad(getattr(visual, "ln_post", None), True)
        set_requires_grad(getattr(visual, "proj", None), True)
        if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
            set_last_k_trainable(visual.transformer.resblocks, last_k_blocks)
        elif hasattr(visual, "backbone") and hasattr(visual.backbone, "layers"):
            set_last_k_trainable(visual.backbone.layers, last_k_blocks)
            set_requires_grad(getattr(visual.backbone, "norm_f", None), True)
        else:
            raise ValueError("--train_scope last_k could not find visual transformer/resblocks or Vim backbone/layers")
    elif train_scope == "path_adapter":
        if not hasattr(visual, "path_adapters") or len(visual.path_adapters) == 0:
            raise ValueError("--train_scope path_adapter requires Vim path adapters to be enabled")
        set_requires_grad(visual.path_adapters, True)
        set_requires_grad(getattr(visual, "path_pooler", None), True)
        set_requires_grad(getattr(visual, "ln_post", None), True)
        set_requires_grad(getattr(visual, "proj", None), True)
    elif train_scope == "path_adapter_last_k":
        if not hasattr(visual, "path_adapters") or len(visual.path_adapters) == 0:
            raise ValueError("--train_scope path_adapter_last_k requires Vim path adapters to be enabled")
        if not hasattr(visual, "backbone") or not hasattr(visual.backbone, "layers"):
            raise ValueError("--train_scope path_adapter_last_k is only supported for Vim backbones")
        set_requires_grad(visual.path_adapters, True)
        set_requires_grad(getattr(visual, "path_pooler", None), True)
        set_last_k_trainable(visual.backbone.layers, last_k_blocks)
        set_requires_grad(getattr(visual.backbone, "norm_f", None), True)
        set_requires_grad(getattr(visual, "ln_post", None), True)
        set_requires_grad(getattr(visual, "proj", None), True)
    elif train_scope == "path_adapter_full":
        if not hasattr(visual, "path_adapters") or len(visual.path_adapters) == 0:
            raise ValueError("--train_scope path_adapter_full requires Vim path adapters to be enabled")
        set_requires_grad(visual, True)
    else:
        raise ValueError(f"Unsupported train_scope: {train_scope}")

    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable visual parameters found.")
    return params


def configure_vit_readout_warmup(model):
    for param in model.parameters():
        param.requires_grad = False
    visual = model.visual
    if not hasattr(visual, "path_readout") or visual.path_readout is None:
        raise ValueError("--vit_readout_warmup_epochs requires --vit_path_adapter != none")
    set_requires_grad(visual.path_readout, True)
    set_requires_grad(getattr(visual, "ln_post", None), True)
    set_requires_grad(getattr(visual, "proj", None), True)
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable ViT readout parameters found.")
    return params


def build_optimizer(model, args):
    if args.train_scope != "path_adapter_last_k":
        params = [param for param in model.parameters() if param.requires_grad]
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay), {
            "default": {"lr": args.lr, "params": sum(param.numel() for param in params)}
        }

    groups = {
        "adapter": {"params": [], "lr": args.adapter_lr or args.lr},
        "head": {"params": [], "lr": args.head_lr or args.lr},
        "backbone": {"params": [], "lr": args.backbone_lr or args.lr},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("visual.path_adapters") or name.startswith("visual.path_pooler"):
            groups["adapter"]["params"].append(param)
        elif name.startswith("visual.proj") or name.startswith("visual.ln_post"):
            groups["head"]["params"].append(param)
        else:
            groups["backbone"]["params"].append(param)

    param_groups = [group for group in groups.values() if group["params"]]
    summary = {
        name: {
            "lr": group["lr"],
            "params": sum(param.numel() for param in group["params"]),
        }
        for name, group in groups.items()
        if group["params"]
    }
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay), summary


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


def build_hard_negative_prompts(classes, mode: str):
    if mode == "none":
        return []

    templates = {
        "decoy": [
            "a false decoy path ending at a {}",
            "a distractor arrow path ending at a {}",
        ],
        "state_tracking": [
            "a path that briefly passes near a {} but does not end there",
            "a distracting path ending at a {}",
            "a wrong state-tracking trace ending at a {}",
        ],
    }[mode]
    return [template.format(item["name"].replace("_", " ")) for item in classes for template in templates]


def image_features_from_images(model, images):
    image_features = model.encode_image(images)
    return F.normalize(image_features.float(), dim=-1)


def logits_from_image_features(model, image_features, text_features):
    logit_scale = model.logit_scale.exp().detach().float()
    return logit_scale * image_features @ text_features.t()


def logits_from_images(model, images, text_features):
    image_features = image_features_from_images(model, images)
    return logits_from_image_features(model, image_features, text_features)


def make_consistency_view(images, noise_std: float, brightness: float):
    view = images
    if brightness > 0:
        scale = 1.0 + (torch.rand(images.size(0), 1, 1, 1, device=images.device) * 2.0 - 1.0) * brightness
        view = view * scale
    if noise_std > 0:
        view = view + torch.randn_like(view) * noise_std
    return view


@torch.no_grad()
def evaluate(model, loader, text_features, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    metadata = []
    all_preds = []
    all_labels = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = logits_from_images(model, images, text_features)
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
    parser = argparse.ArgumentParser(description="Train CLIP-style visual fine-tuning and readout variants on VSTD.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("runs/clip_alignment_vstd"))
    parser.add_argument("--encoder", choices=["vit", "vim"], default="vim")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16")
    parser.add_argument("--vim_model_name", type=str, default="vim_base")
    parser.add_argument("--vim_ckpt", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--train_scope",
        type=str,
        default="projection",
        choices=["projection", "backbone", "last_k", "path_adapter", "path_adapter_last_k", "path_adapter_full", "full"],
        help="Train visual projection head only, visual backbone only, last-k visual blocks plus projection, path adapters plus projection, path adapters plus last-k blocks, path-adapter-augmented full visual side, or full visual side. Text encoder is always frozen.",
    )
    parser.add_argument("--last_k_blocks", type=int, default=2)
    parser.add_argument("--path_adapter_layers", type=int, default=4)
    parser.add_argument("--path_adapter_bottleneck", type=int, default=128)
    parser.add_argument("--path_adapter_dropout", type=float, default=0.1)
    parser.add_argument(
        "--path_adapter_pooling",
        choices=[
            "forward",
            "bidirectional",
            "start_bidirectional",
            "geometry_bidirectional",
            "path_state",
            "hybrid_path_state",
            "gated_hybrid_path_state",
            "omni_path_state",
            "memory_omni_path_state",
            "endpoint_omni_path_state",
            "layerwise_omni_path_state",
            "decoy_suppression",
            "soft_decoy_suppression",
        ],
        default="forward",
    )
    parser.add_argument("--adapter_lr", type=float, default=None)
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--backbone_lr", type=float, default=None)
    parser.add_argument(
        "--vit_path_adapter",
        choices=[
            "none",
            "start_pool",
            "path_state",
            "decoy_aware",
            "multihead_decoy_aware",
            "directional_decoy_aware",
            "adaptive_decoy_aware",
            "adaptive_decoy_aware_mean_supp",
            "adaptive_decoy_aware_no_outer_cls",
            "adaptive_decoy_aware_no_inner_cls",
            "adaptive_decoy_aware_static_alpha",
            "adaptive_decoy_aware_relu_supp",
            "adaptive_keep_only",
            "adaptive_supp_only",
            "adaptive_supp_only_clean",
            "adaptive_keep_only_clean",
            "adaptive_keep_only_2branch",
            "adaptive_residual_decoy_aware",
            "geo_complement_adar",
            "geo_twoquery_adar",
            "geo_band_adar",
            "geo_layer_mixed_adar",
            "experimental_attention_pool_adar",
            "experimental_decoy_contrast_adar",
            "experimental_multiquery_adar",
            "slot_decoy_aware",
            "adaptive_slot_decoy_aware",
            "layerwise_decoy_aware",
            "start_path_state",
            "multihop",
            "multihop_zero",
            "multiquery_multihop",
            "decoy_multihop",
            "decoy_fusion_multihop",
        ],
        default="none",
        help="Optional path-aware ViT readout head.",
    )
    parser.add_argument("--vit_path_adapter_bottleneck", type=int, default=128)
    parser.add_argument("--vit_path_adapter_dropout", type=float, default=0.1)
    parser.add_argument("--vit_path_adapter_hops", type=int, default=4)
    parser.add_argument("--vit_path_adapter_residual_scale", type=float, default=1.0)
    parser.add_argument(
        "--vit_readout_warmup_epochs",
        type=int,
        default=0,
        help="For ViT path adapters, train readout+projection only for N epochs before full visual fine-tuning.",
    )
    parser.add_argument(
        "--hard_negative_prompts",
        choices=["none", "decoy", "state_tracking"],
        default="none",
        help="Append wrong-path text prompts during training only. Evaluation still uses the original class prompts.",
    )
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=0.0,
        help="Weight for image-embedding consistency between the original image and a weak tensor-level augmented view.",
    )
    parser.add_argument("--consistency_noise_std", type=float, default=0.015)
    parser.add_argument("--consistency_brightness", type=float, default=0.04)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.encoder == "vim" and not args.vim_ckpt:
        raise ValueError("--vim_ckpt is required for --encoder vim")
    if args.encoder == "vim" and (args.device == "cpu" or not torch.cuda.is_available()):
        raise RuntimeError("Vim/Mamba requires CUDA for projection-head probing in this environment.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    eval_batch_size = args.eval_batch_size or args.batch_size

    model, preprocess, clip_module = load_clip(args, device)
    if args.encoder == "vit" and args.vit_path_adapter != "none" and args.vit_readout_warmup_epochs > 0:
        trainable_params = configure_vit_readout_warmup(model)
    else:
        trainable_params = configure_trainability(model, args.train_scope, args.last_k_blocks)
    optimizer, optimizer_summary = build_optimizer(model, args)

    classes = load_json(args.dataset_dir / "classes.json")
    prompts = [item["prompt"] for item in classes]
    hard_negative_prompts = build_hard_negative_prompts(classes, args.hard_negative_prompts)
    train_prompts = prompts + hard_negative_prompts
    eval_text_features = build_text_features(model, clip_module, prompts, device)
    train_text_features = build_text_features(model, clip_module, train_prompts, device)

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

    print("Runtime:", flush=True)
    print(f"  device: {device}", flush=True)
    print(f"  cuda_available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  cuda_device_name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  cuda_version: {torch.version.cuda}", flush=True)
    print(f"  encoder: {args.encoder}", flush=True)
    print(f"  clip_model: {args.clip_model}", flush=True)
    print(f"  seed: {args.seed}", flush=True)
    print(f"  train_scope: {args.train_scope}", flush=True)
    print(f"  hard_negative_prompts: {args.hard_negative_prompts}", flush=True)
    print(f"  train_prompt_count: {len(train_prompts)}", flush=True)
    print(f"  consistency_weight: {args.consistency_weight}", flush=True)
    if args.train_scope in {"last_k", "path_adapter_last_k"}:
        print(f"  last_k_blocks: {args.last_k_blocks}", flush=True)
    if args.train_scope.startswith("path_adapter"):
        print(f"  path_adapter_layers: {args.path_adapter_layers}", flush=True)
        print(f"  path_adapter_bottleneck: {args.path_adapter_bottleneck}", flush=True)
        print(f"  path_adapter_pooling: {args.path_adapter_pooling}", flush=True)
    if args.encoder == "vit":
        print(f"  vit_path_adapter: {args.vit_path_adapter}", flush=True)
        if args.vit_path_adapter != "none":
            print(f"  vit_path_adapter_bottleneck: {args.vit_path_adapter_bottleneck}", flush=True)
            print(f"  vit_path_adapter_hops: {args.vit_path_adapter_hops}", flush=True)
            print(f"  vit_path_adapter_residual_scale: {args.vit_path_adapter_residual_scale}", flush=True)
            print(f"  vit_readout_warmup_epochs: {args.vit_readout_warmup_epochs}", flush=True)
    print(f"  trainable_params: {sum(p.numel() for p in trainable_params):,}", flush=True)
    print(f"  optimizer_groups: {optimizer_summary}", flush=True)
    print(f"  prompts: {prompts}", flush=True)
    if hard_negative_prompts:
        print(f"  hard_negative_prompt_examples: {hard_negative_prompts[:4]}", flush=True)

    log_path = args.output_dir / "train_log.csv"
    best_val_acc = -1.0
    best_epoch = 0
    best_model_path = args.output_dir / "best_model.pt"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "train_time_sec"],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            if (
                epoch == args.vit_readout_warmup_epochs + 1
                and args.encoder == "vit"
                and args.vit_path_adapter != "none"
                and args.vit_readout_warmup_epochs > 0
            ):
                trainable_params = configure_trainability(model, args.train_scope, args.last_k_blocks)
                optimizer, optimizer_summary = build_optimizer(model, args)
                print(
                    f"Finished ViT readout warmup; switched optimizer_groups: {optimizer_summary}",
                    flush=True,
                )
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
                logits = logits_from_image_features(model, image_features, train_text_features)
                loss = F.cross_entropy(logits, labels)
                if args.consistency_weight > 0:
                    aug_images = make_consistency_view(
                        images,
                        noise_std=args.consistency_noise_std,
                        brightness=args.consistency_brightness,
                    )
                    aug_features = image_features_from_images(model, aug_images)
                    aug_logits = logits_from_image_features(model, aug_features, train_text_features)
                    aug_loss = F.cross_entropy(aug_logits, labels)
                    consistency_loss = 1.0 - F.cosine_similarity(image_features, aug_features, dim=-1).mean()
                    loss = 0.5 * (loss + aug_loss) + args.consistency_weight * consistency_loss
                loss.backward()
                optimizer.step()
                preds = logits[:, : len(prompts)].argmax(dim=1)
                loss_sum += loss.item() * labels.size(0)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = loss_sum / max(1, total)
            train_acc = correct / max(1, total)
            train_time = time.perf_counter() - start_time
            val = evaluate(model, loaders["val"], eval_text_features, device)
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
                torch.save({"model": model.state_dict(), "args": vars(args), "best_epoch": best_epoch}, best_model_path)
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} "
                f"time={train_time:.2f}s",
                flush=True,
            )

    if best_model_path.exists():
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint["model"], strict=True)
        eval_text_features = build_text_features(model, clip_module, prompts, device)

    test = evaluate(model, loaders["test"], eval_text_features, device)
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
        "hard_negative_prompts": hard_negative_prompts,
    }
    with (args.output_dir / "test_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.output_dir / "model.pt")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
