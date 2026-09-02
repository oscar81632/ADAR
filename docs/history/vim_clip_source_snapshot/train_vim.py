import os
import argparse
import csv
import json
import subprocess
import sys
import time

import pandas as pd
import matplotlib.pyplot as plt

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, Caltech101, OxfordIIITPet, STL10
from PIL import Image
import clip

from lora_vim import (
    apply_lora_to_vim,
    mark_only_lora_and_head_trainable,
    split_lora_and_head_parameters,
    summarize_lora,
)
from adapter_vim import (
    apply_adapters_to_vim,
    mark_adapter_trainable,
    split_adapter_head_backbone_parameters,
    summarize_adapters,
)



def sync_if_cuda(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def timer_now(device):
    sync_if_cuda(device)
    return time.perf_counter()


def save_run_args(args, args_path, extra=None):
    payload = {
        "argv": sys.argv,
        "args": vars(args),
        "extra": extra or {},
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    os.makedirs(os.path.dirname(args_path), exist_ok=True)
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Saved args to: {args_path}")


class PreprocessedImageDataset(Dataset):
    def __init__(self, dataset, preprocess, class_names):
        self.dataset = dataset
        self.preprocess = preprocess
        self.class_names = list(class_names)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        img = self.preprocess(img)
        return img, int(label)


def collate_fn(batch):
    images = torch.stack([x[0] for x in batch], dim=0)
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    return images, labels


def normalize_dataset_name(name):
    aliases = {
        "cifar": "cifar10",
        "cifar10": "cifar10",
        "caltech": "caltech101",
        "caltech101": "caltech101",
        "oxfordpet": "oxfordpets",
        "oxfordpets": "oxfordpets",
        "oxfordpest": "oxfordpets",
        "oxfordpests": "oxfordpets",
        "pets": "oxfordpets",
        "stl": "stl10",
        "stl10": "stl10",
    }
    key = name.lower().replace("-", "").replace("_", "")
    if key not in aliases:
        raise ValueError(f"Unsupported dataset: {name}")
    return aliases[key]


def get_class_names(dataset, dataset_name):
    if hasattr(dataset, "classes"):
        return list(dataset.classes)
    if hasattr(dataset, "categories"):
        return list(dataset.categories)
    if dataset_name == "oxfordpets" and hasattr(OxfordIIITPet, "_CLASSES"):
        return list(OxfordIIITPet._CLASSES)
    raise ValueError(f"Cannot infer class names for dataset: {dataset_name}")


def parse_int_list(value):
    if value is None:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_class_count_list(value):
    if value is None:
        return []
    counts = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"all", "full"}:
            counts.append(None)
        else:
            counts.append(int(item))
    return counts


def parse_float_list(value):
    if value is None:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def format_hparam(value):
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def format_run_dataset_stem(run_name, dataset_name):
    return run_name if run_name.endswith(f"_{dataset_name}") else f"{run_name}_{dataset_name}"


def parse_float_range(value, default):
    if value is None:
        return default
    items = parse_float_list(value)
    if len(items) != 2:
        raise ValueError("Range arguments must be two comma-separated values: min,max")
    low, high = items
    if low < 0 or high <= 0 or low > high:
        raise ValueError("Range must satisfy 0 <= min <= max and max > 0")
    return low, high


def sample_log_uniform(rng, low, high):
    if low <= 0:
        raise ValueError("log-uniform lower bound must be > 0")
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_uniform(rng, low, high):
    return float(rng.uniform(low, high))


def next_numbered_prefix(parent_dir):
    os.makedirs(parent_dir, exist_ok=True)
    max_prefix = 0
    for name in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, name)
        if not os.path.isdir(full_path):
            continue
        prefix = name.split("_", 1)[0]
        if prefix.isdigit():
            max_prefix = max(max_prefix, int(prefix))
    return max_prefix + 1


VALUELESS_CLI_ARGS = {
    "--hparam_search",
    "--hparam_random_search",
    "--finetune_vit",
    "--save_final_weights",
    "--save_epoch_checkpoints",
    "--lora_freeze_head",
    "--adapter_freeze_head",
}


def remove_cli_args(argv, option_names, valueless_options=None):
    valueless_options = set(valueless_options or VALUELESS_CLI_ARGS)
    cleaned = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue

        matched = False
        for option in option_names:
            if item == option:
                if option not in valueless_options:
                    skip_next = True
                matched = True
                break
            if item.startswith(option + "="):
                matched = True
                break

        if not matched:
            cleaned.append(item)
    return cleaned


def get_dataset_train_labels(dataset_name, data_root, seed, caltech_test_ratio):
    dataset_name = normalize_dataset_name(dataset_name)

    if dataset_name == "cifar10":
        ds = CIFAR10(root=data_root, train=True, download=True)
        labels = list(ds.targets)
        class_names = get_class_names(ds, dataset_name)

    elif dataset_name == "stl10":
        ds = STL10(root=data_root, split="train", download=True)
        labels = list(ds.labels)
        class_names = get_class_names(ds, dataset_name)

    elif dataset_name == "oxfordpets":
        ds = OxfordIIITPet(root=data_root, split="trainval", target_types="category", download=True)
        labels = [ds[idx][1] for idx in range(len(ds))]
        class_names = get_class_names(ds, dataset_name)

    elif dataset_name == "caltech101":
        ds = Caltech101(root=data_root, target_type="category", download=True)
        all_labels = list(ds.y) if hasattr(ds, "y") else [ds[idx][1] for idx in range(len(ds))]
        train_indices, _ = split_indices_by_label(all_labels, caltech_test_ratio, seed)
        labels = [all_labels[idx] for idx in train_indices]
        class_names = get_class_names(ds, dataset_name)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    counts_by_class = {int(label): labels.count(label) for label in sorted(set(labels))}
    return labels, counts_by_class, class_names, dataset_name


def run_class_count_sweep(args):
    counts = parse_class_count_list(args.class_counts)
    if not counts:
        raise ValueError("--class_counts is empty")

    _, counts_by_class, _, dataset_name = get_dataset_train_labels(
        args.dataset,
        args.data_root,
        args.seed,
        args.caltech_test_ratio,
    )
    max_per_class = min(counts_by_class.values())
    print(f"Dataset: {dataset_name}")
    print(f"Minimum available train samples per class: {max_per_class}")

    base_argv = remove_cli_args(
        sys.argv[1:],
        {
            "--class_counts",
            "--train_per_class",
            "--indices_path",
            "--output_dir",
            "--run_name",
            "--ckpt_dir",
            "--log_file",
            "--save_path",
        },
    )

    run_index = next_numbered_prefix(args.output_dir)
    for count in counts:
        if count is not None:
            if count <= 0:
                raise ValueError(f"Invalid class count: {count}")
            if count > max_per_class:
                print(f"[Skip] class{count}: requested {count}/class, but {dataset_name} only has {max_per_class}/class in train split.")
                continue
            run_name = f"class{count}"
            sample_label = str(count)
            count_args = ["--train_per_class", str(count)]
        else:
            run_name = "all"
            sample_label = "all"
            count_args = []

        run_output_dir = os.path.join(args.output_dir, f"{run_index}_{run_name}")
        run_index += 1
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            *base_argv,
            *count_args,
            "--output_dir", run_output_dir,
            "--run_name", run_name,
        ]
        print("\n=== Class-count sweep ===")
        print("Dataset:", dataset_name)
        print("Samples per class:", sample_label)
        print("Output dir:", run_output_dir)
        print("Command:", " ".join(cmd))
        subprocess.run(cmd, check=True)

def summarize_hparam_run(run_output_dir, run_name, dataset_name, lr_proj, lr_backbone, weight_decay, lr_adapter=None, adapter_dropout=None):
    log_path = os.path.join(run_output_dir, "log", f"{run_name}_train_log.csv")
    final_test_path = os.path.join(run_output_dir, "args", f"{run_name}_{dataset_name}_final_test_result.json")

    row = {
        "run_dir": run_output_dir,
        "run_name": run_name,
        "lr_proj": lr_proj,
        "lr_backbone": lr_backbone,
        "lr_adapter": lr_adapter,
        "adapter_dropout": adapter_dropout,
        "weight_decay": weight_decay,
        "best_epoch": None,
        "best_val_acc": None,
        "best_val_mean_gap": None,
        "final_test_acc": None,
        "final_test_mean_gap": None,
        "final_test_inference_time_sec": None,
        "final_test_inference_samples_per_sec": None,
        "status": "ok",
    }

    try:
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            if not df.empty and "acc" in df.columns:
                best_idx = df["acc"].idxmax()
                best_row = df.loc[best_idx]
                row["best_epoch"] = int(best_row["epoch"])
                row["best_val_acc"] = float(best_row["acc"])
                row["best_val_mean_gap"] = float(best_row["mean_gap"]) if "mean_gap" in df.columns else None
        else:
            row["status"] = "missing_log"

        if os.path.exists(final_test_path):
            with open(final_test_path, "r", encoding="utf-8") as f:
                final_result = json.load(f)
            row["final_test_acc"] = final_result.get("acc")
            row["final_test_mean_gap"] = final_result.get("mean_gap")
            row["final_test_inference_time_sec"] = final_result.get("inference_time_sec")
            row["final_test_inference_samples_per_sec"] = final_result.get("inference_samples_per_sec")
        else:
            row["status"] = "missing_final_test" if row["status"] == "ok" else row["status"]
    except Exception as exc:
        row["status"] = f"summary_error: {exc}"

    return row


def build_hparam_trials(args):
    if args.hparam_random_search:
        rng = np.random.default_rng(args.seed)
        lr_proj_range = parse_float_range(args.lr_proj_range, (1e-5, 1e-3))
        lr_backbone_range = parse_float_range(args.lr_backbone_range, (1e-7, 3e-5))
        lr_adapter_range = parse_float_range(args.lr_adapter_range, (args.lr_adapter, args.lr_adapter))
        adapter_dropout_range = parse_float_range(args.adapter_dropout_range, (args.adapter_dropout, args.adapter_dropout))
        weight_decay_range = parse_float_range(args.weight_decay_range, (1e-6, 1e-3))
        trials = []
        for _ in range(args.hparam_trials):
            if args.weight_decay_zero_prob > 0 and rng.random() < args.weight_decay_zero_prob:
                weight_decay = 0.0
            else:
                weight_decay = sample_log_uniform(rng, *weight_decay_range)
            trials.append(
                {
                    "lr_proj": sample_log_uniform(rng, *lr_proj_range),
                    "lr_backbone": sample_log_uniform(rng, *lr_backbone_range),
                    "lr_adapter": sample_log_uniform(rng, *lr_adapter_range),
                    "adapter_dropout": sample_uniform(rng, *adapter_dropout_range),
                    "weight_decay": weight_decay,
                }
            )
        return trials

    lr_proj_values = parse_float_list(args.lr_proj_values) or [args.lr_proj]
    lr_backbone_values = parse_float_list(args.lr_backbone_values) or [args.lr_backbone]
    lr_adapter_values = parse_float_list(args.lr_adapter_values) or [args.lr_adapter]
    adapter_dropout_values = parse_float_list(args.adapter_dropout_values) or [args.adapter_dropout]
    weight_decay_values = parse_float_list(args.weight_decay_values) or [args.weight_decay]
    return [
        {
            "lr_proj": lr_proj,
            "lr_backbone": lr_backbone,
            "lr_adapter": lr_adapter,
            "adapter_dropout": adapter_dropout,
            "weight_decay": weight_decay,
        }
        for lr_proj in lr_proj_values
        for lr_backbone in lr_backbone_values
        for lr_adapter in lr_adapter_values
        for adapter_dropout in adapter_dropout_values
        for weight_decay in weight_decay_values
    ]


def run_hparam_sweep(args):
    trials = build_hparam_trials(args)
    if not trials:
        raise ValueError("No hyperparameter trials were generated")

    dataset_name = normalize_dataset_name(args.dataset)
    base_argv = remove_cli_args(
        sys.argv[1:],
        {
            "--hparam_search",
            "--hparam_random_search",
            "--hparam_trials",
            "--lr_proj_values",
            "--lr_backbone_values",
            "--lr_adapter_values",
            "--adapter_dropout_values",
            "--weight_decay_values",
            "--lr_proj_range",
            "--lr_backbone_range",
            "--lr_adapter_range",
            "--adapter_dropout_range",
            "--weight_decay_range",
            "--weight_decay_zero_prob",
            "--lr_proj",
            "--lr_backbone",
            "--lr_adapter",
            "--adapter_dropout",
            "--weight_decay",
            "--class_counts",
            "--output_dir",
            "--run_name",
            "--ckpt_dir",
            "--log_file",
            "--save_path",
        },
    )

    os.makedirs(args.output_dir, exist_ok=True)
    summary_rows = []
    run_index = next_numbered_prefix(args.output_dir)

    for trial_idx, trial in enumerate(trials, start=1):
        lr_proj = trial["lr_proj"]
        lr_backbone = trial["lr_backbone"]
        lr_adapter = trial.get("lr_adapter", args.lr_adapter)
        adapter_dropout = trial.get("adapter_dropout", args.adapter_dropout)
        weight_decay = trial["weight_decay"]
        prefix = f"t{trial_idx:03d}_" if args.hparam_random_search else ""
        run_name = (
            f"{prefix}"
            f"lp{format_hparam(lr_proj)}_"
            f"lb{format_hparam(lr_backbone)}_"
            f"la{format_hparam(lr_adapter)}_"
            f"ad{format_hparam(adapter_dropout)}_"
            f"wd{format_hparam(weight_decay)}"
        )
        run_output_dir = os.path.join(args.output_dir, f"{run_index}_{run_name}")
        run_index += 1
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            *base_argv,
            "--lr_proj", str(lr_proj),
            "--lr_backbone", str(lr_backbone),
            "--lr_adapter", str(lr_adapter),
            "--adapter_dropout", str(adapter_dropout),
            "--weight_decay", str(weight_decay),
            "--output_dir", run_output_dir,
            "--run_name", run_name,
        ]

        print("\n=== Hyperparameter random search ===" if args.hparam_random_search else "\n=== Hyperparameter sweep ===")
        print("Trial:", f"{trial_idx}/{len(trials)}")
        print("Dataset:", dataset_name)
        print("lr_proj:", lr_proj)
        print("lr_backbone:", lr_backbone)
        print("lr_adapter:", lr_adapter)
        print("adapter_dropout:", adapter_dropout)
        print("weight_decay:", weight_decay)
        print("Output dir:", run_output_dir)
        print("Command:", " ".join(cmd))

        status = "ok"
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            status = f"failed_returncode_{exc.returncode}"
            print(f"[Failed] {run_name}: {status}")

        row = summarize_hparam_run(
            run_output_dir=run_output_dir,
            run_name=run_name,
            dataset_name=dataset_name,
            lr_proj=lr_proj,
            lr_backbone=lr_backbone,
            weight_decay=weight_decay,
            lr_adapter=lr_adapter,
            adapter_dropout=adapter_dropout,
        )
        row["trial"] = trial_idx
        row["search_type"] = "random" if args.hparam_random_search else "grid"
        if status != "ok":
            row["status"] = status
        summary_rows.append(row)

        summary_path = os.path.join(args.output_dir, "hparam_summary.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"Updated summary: {summary_path}")

    summary_path = os.path.join(args.output_dir, "hparam_summary.csv")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    valid_rows = summary_df[summary_df["best_val_acc"].notna()] if "best_val_acc" in summary_df else pd.DataFrame()
    if not valid_rows.empty:
        best = valid_rows.sort_values("best_val_acc", ascending=False).iloc[0]
        print("\n=== Best hyperparameters by validation accuracy ===")
        print(f"run_dir: {best['run_dir']}")
        print(f"lr_proj: {best['lr_proj']}")
        print(f"lr_backbone: {best['lr_backbone']}")
        if 'lr_adapter' in best:
            print(f"lr_adapter: {best['lr_adapter']}")
        if 'adapter_dropout' in best:
            print(f"adapter_dropout: {best['adapter_dropout']}")
        print(f"weight_decay: {best['weight_decay']}")
        print(f"best_val_acc: {best['best_val_acc']}")
        print(f"final_test_acc: {best['final_test_acc']}")
    print(f"Saved hparam summary to: {summary_path}")


def load_index_map(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {int(label): list(indices) for label, indices in payload["indices_by_class"].items()}


def save_index_map(path, dataset_name, split_name, seed, samples_per_class, indices_by_class, class_names):
    payload = {
        "dataset": dataset_name,
        "split": split_name,
        "seed": seed,
        "samples_per_class": samples_per_class,
        "class_names": list(class_names),
        "indices_by_class": {str(label): list(indices) for label, indices in indices_by_class.items()},
        "selected_indices": [idx for label in sorted(indices_by_class) for idx in indices_by_class[label]],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Saved split indices to: {path}")


def subset_dataset_per_class(dataset, samples_per_class, seed, indices_path=None, save_indices_path=None, class_names=None, dataset_name="dataset"):
    if samples_per_class is None and indices_path is None:
        return dataset, None
    if samples_per_class is not None and samples_per_class <= 0:
        raise ValueError("--train_per_class must be positive")

    labels = list(dataset.targets) if hasattr(dataset, "targets") else [dataset[idx][1] for idx in range(len(dataset))]

    if indices_path is not None:
        indices_by_class = load_index_map(indices_path)
        selected_indices = [idx for label in sorted(indices_by_class) for idx in indices_by_class[label]]
        if samples_per_class is not None:
            expected = samples_per_class * len(indices_by_class)
            if len(selected_indices) != expected:
                raise ValueError(
                    f"Loaded {len(selected_indices)} indices from {indices_path}, "
                    f"but --train_per_class expects {expected}"
                )
        if save_indices_path is not None:
            save_index_map(
                save_indices_path,
                dataset_name=dataset_name,
                split_name="train",
                seed=seed,
                samples_per_class=samples_per_class,
                indices_by_class=indices_by_class,
                class_names=class_names or [],
            )
        return Subset(dataset, selected_indices), indices_by_class

    rng = np.random.default_rng(seed)
    indices_by_class = {}

    for label in sorted(set(labels)):
        label_indices = [idx for idx, item_label in enumerate(labels) if item_label == label]
        if samples_per_class > len(label_indices):
            raise ValueError(
                f"Requested {samples_per_class} samples for class {label}, "
                f"but only {len(label_indices)} are available"
            )
        rng.shuffle(label_indices)
        indices_by_class[int(label)] = label_indices[:samples_per_class]

    selected_indices = [idx for label in sorted(indices_by_class) for idx in indices_by_class[label]]
    if save_indices_path is not None:
        save_index_map(
            save_indices_path,
            dataset_name=dataset_name,
            split_name="train",
            seed=seed,
            samples_per_class=samples_per_class,
            indices_by_class=indices_by_class,
            class_names=class_names or [],
        )
    return Subset(dataset, selected_indices), indices_by_class

def split_indices_by_label(labels, test_ratio, seed):
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for label in sorted(set(labels)):
        label_indices = [idx for idx, item_label in enumerate(labels) if item_label == label]
        rng.shuffle(label_indices)

        if len(label_indices) < 2:
            train_indices.extend(label_indices)
            continue

        test_size = int(round(len(label_indices) * test_ratio))
        test_size = max(1, min(test_size, len(label_indices) - 1))
        test_indices.extend(label_indices[:test_size])
        train_indices.extend(label_indices[test_size:])

    if not train_indices or not test_indices:
        raise ValueError("split ratio produced an empty train or test split")

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices




def get_labels(dataset):
    if isinstance(dataset, Subset):
        return [int(dataset.dataset[idx][1]) for idx in dataset.indices]
    if hasattr(dataset, "targets"):
        return [int(label) for label in dataset.targets]
    if hasattr(dataset, "labels"):
        return [int(label) for label in dataset.labels]
    if hasattr(dataset, "y"):
        return [int(label) for label in dataset.y]
    return [int(dataset[idx][1]) for idx in range(len(dataset))]


def indices_by_class_from_indices(labels, indices):
    indices_by_class = {}
    for idx in indices:
        label = int(labels[idx])
        indices_by_class.setdefault(label, []).append(int(idx))
    return {label: indices_by_class[label] for label in sorted(indices_by_class)}


def save_eval_result(path, result, split_name, epoch=None):
    if result is None:
        return
    payload = {
        "split": split_name,
        "epoch": epoch,
        "mean_gap": result["mean_gap"],
        "acc": result["acc"],
        "eval_time_sec": result["eval_time_sec"],
        "text_time_sec": result["text_time_sec"],
        "inference_time_sec": result["inference_time_sec"],
        "inference_samples_per_sec": result["inference_samples_per_sec"],
        "inference_ms_per_sample": result["inference_ms_per_sample"],
        "eval_samples_per_sec": result["eval_samples_per_sec"],
        "gaps": result["gaps"],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Saved {split_name} result to: {path}")

def get_preprocess_image_size(preprocess, default=224):
    for transform in getattr(preprocess, "transforms", []):
        if isinstance(transform, transforms.CenterCrop):
            size = transform.size
            if isinstance(size, (tuple, list)):
                return int(size[0])
            return int(size)
    return int(default)


def get_preprocess_normalize(preprocess):
    for transform in getattr(preprocess, "transforms", []):
        if isinstance(transform, transforms.Normalize):
            return transforms.Normalize(mean=transform.mean, std=transform.std)
    return transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )


def build_train_preprocess(eval_preprocess, args):
    mode = args.train_aug
    if mode == "none":
        return eval_preprocess

    image_size = get_preprocess_image_size(eval_preprocess)
    normalize = get_preprocess_normalize(eval_preprocess)
    transform_list = [
        transforms.RandomResizedCrop(
            image_size,
            scale=(args.aug_crop_scale_min, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=args.aug_hflip_prob),
    ]

    if mode in {"rrc_flip_jitter", "randaugment"}:
        transform_list.append(
            transforms.ColorJitter(
                brightness=args.color_jitter_brightness,
                contrast=args.color_jitter_contrast,
                saturation=args.color_jitter_saturation,
                hue=args.color_jitter_hue,
            )
        )

    if mode == "randaugment":
        transform_list.append(
            transforms.RandAugment(
                num_ops=args.randaugment_num_ops,
                magnitude=args.randaugment_magnitude,
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )

    transform_list.extend(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transforms.Compose(transform_list)


def build_datasets(dataset_name, data_root, preprocess, seed, caltech_test_ratio, val_ratio=0.5, train_per_class=None, indices_path=None, save_indices_path=None, save_val_indices_path=None, save_test_indices_path=None, train_preprocess=None):
    dataset_name = normalize_dataset_name(dataset_name)

    if dataset_name == "cifar10":
        train_base = CIFAR10(root=data_root, train=True, download=True)
        test_base = CIFAR10(root=data_root, train=False, download=True)
        class_names = get_class_names(train_base, dataset_name)
    elif dataset_name == "stl10":
        train_base = STL10(root=data_root, split="train", download=True)
        test_base = STL10(root=data_root, split="test", download=True)
        class_names = get_class_names(train_base, dataset_name)

    elif dataset_name == "oxfordpets":
        train_base = OxfordIIITPet(
            root=data_root,
            split="trainval",
            target_types="category",
            download=True,
        )
        test_base = OxfordIIITPet(
            root=data_root,
            split="test",
            target_types="category",
            download=True,
        )
        class_names = get_class_names(train_base, dataset_name)

    elif dataset_name == "caltech101":
        base = Caltech101(
            root=data_root,
            target_type="category",
            download=True,
        )
        class_names = get_class_names(base, dataset_name)
        labels = list(base.y) if hasattr(base, "y") else [base[idx][1] for idx in range(len(base))]
        train_indices, test_indices = split_indices_by_label(labels, caltech_test_ratio, seed)
        train_base = Subset(base, train_indices)
        test_base = Subset(base, test_indices)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_base, _ = subset_dataset_per_class(
        train_base,
        train_per_class,
        seed,
        indices_path=indices_path,
        save_indices_path=save_indices_path,
        class_names=class_names,
        dataset_name=dataset_name,
    )

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val_ratio must be between 0 and 1")

    test_labels = get_labels(test_base)
    val_indices, final_test_indices = split_indices_by_label(test_labels, 1.0 - val_ratio, seed)

    if save_val_indices_path is not None:
        save_index_map(
            save_val_indices_path,
            dataset_name=dataset_name,
            split_name="valid_from_test",
            seed=seed,
            samples_per_class=None,
            indices_by_class=indices_by_class_from_indices(test_labels, val_indices),
            class_names=class_names,
        )
    if save_test_indices_path is not None:
        save_index_map(
            save_test_indices_path,
            dataset_name=dataset_name,
            split_name="test_remaining",
            seed=seed,
            samples_per_class=None,
            indices_by_class=indices_by_class_from_indices(test_labels, final_test_indices),
            class_names=class_names,
        )

    train_ds = PreprocessedImageDataset(train_base, train_preprocess or preprocess, class_names)
    val_ds = PreprocessedImageDataset(Subset(test_base, val_indices), preprocess, class_names)
    test_ds = PreprocessedImageDataset(Subset(test_base, final_test_indices), preprocess, class_names)
    return train_ds, val_ds, test_ds, class_names, dataset_name


def build_class_prompts(class_names):
    return [f"a photo of a {name}" for name in class_names]


def clip_classification_loss(image_features, class_text_features, labels, logit_scale, label_smoothing=0.0):
    image_features = F.normalize(image_features, dim=-1)
    class_text_features = F.normalize(class_text_features, dim=-1)
    logits = logit_scale * image_features @ class_text_features.t()
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


def clip_contrastive_loss(image_features, text_features, logit_scale, label_smoothing=0.0):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    targets = torch.arange(image_features.size(0), device=image_features.device)

    loss_i = F.cross_entropy(logits_per_image, targets, label_smoothing=label_smoothing)
    loss_t = F.cross_entropy(logits_per_text, targets, label_smoothing=label_smoothing)
    return (loss_i + loss_t) / 2


def evaluate_simple(model, preprocess, device, class_names=None, test_image_path="Dog.jpg"):
    model.eval()

    if class_names is None:
        class_names = ["dog", "cat", "automobile"]

    prompts = build_class_prompts(class_names)
    text = clip.tokenize(prompts).to(device)

    if not os.path.exists(test_image_path):
        print(f"[Warn] {test_image_path} not found, skip simple eval.")
        return None

    img = preprocess(Image.open(test_image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        image_feat = model.encode_image(img)
        text_feat = model.encode_text(text)
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        sim = image_feat @ text_feat.t()

    scores = sim.squeeze(0).detach().cpu().tolist()
    for p, s in zip(prompts, scores):
        print(f"{p}: {s:.4f}")

    return scores


def evaluate_dataset(model, test_loader, class_names, device):
    model.eval()
    class_order = list(class_names)
    eval_start = timer_now(device)

    if len(class_order) < 2:
        print("[Warn] mean_gap needs at least 2 classes; skip dataset eval.")
        return None

    prompts = build_class_prompts(class_order)
    text = clip.tokenize(prompts).to(device)

    text_start = timer_now(device)
    with torch.no_grad():
        text_feat = model.encode_text(text)
        text_feat = F.normalize(text_feat, dim=-1)
    text_time_sec = timer_now(device) - text_start

    num_classes = len(class_order)
    score_sums = torch.zeros(num_classes, num_classes)
    counts = torch.zeros(num_classes, dtype=torch.long)
    correct = 0
    total = 0

    inference_start = timer_now(device)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            image_feat = model.encode_image(images)
            image_feat = F.normalize(image_feat, dim=-1)
            sims = image_feat @ text_feat.t()

            preds = sims.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

            sims_cpu = sims.detach().cpu()
            labels_cpu = labels.detach().cpu()
            for label_idx in range(num_classes):
                mask = labels_cpu == label_idx
                if mask.any():
                    score_sums[label_idx] += sims_cpu[mask].sum(dim=0)
                    counts[label_idx] += mask.sum().item()
    inference_time_sec = timer_now(device) - inference_start
    eval_time_sec = timer_now(device) - eval_start

    print("\n=== Test Dataset Evaluation ===")

    if total == 0:
        print("[Warn] no test samples found.")
        return None

    avg_scores = {}
    gaps = {}

    for label, cls in enumerate(class_order):
        if counts[label].item() == 0:
            print(f"[Warn] no valid test samples for class: {cls}")
            return None

        avg = score_sums[label] / counts[label].item()
        avg_scores[cls] = avg.tolist()

        wrong_indices = [i for i in range(num_classes) if i != label]
        correct_score = avg[label]
        best_wrong_score = avg[wrong_indices].max()
        gap = (correct_score - best_wrong_score).item()
        gaps[cls] = gap


    acc = correct / total
    mean_gap = sum(gaps.values()) / len(gaps)
    inference_samples_per_sec = total / inference_time_sec if inference_time_sec > 0 else float("inf")
    inference_ms_per_sample = 1000.0 * inference_time_sec / total
    eval_samples_per_sec = total / eval_time_sec if eval_time_sec > 0 else float("inf")

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Mean gap: {mean_gap:.4f}")
    print(f"Eval time: {eval_time_sec:.4f} sec")
    print(f"Inference time: {inference_time_sec:.4f} sec")
    print(f"Inference throughput: {inference_samples_per_sec:.2f} samples/sec")
    print(f"Inference latency: {inference_ms_per_sample:.4f} ms/sample")

    return {
        "class_order": class_order,
        "avg_scores": avg_scores,
        "gaps": gaps,
        "mean_gap": mean_gap,
        "acc": acc,
        "eval_time_sec": eval_time_sec,
        "text_time_sec": text_time_sec,
        "inference_time_sec": inference_time_sec,
        "inference_samples_per_sec": inference_samples_per_sec,
        "inference_ms_per_sample": inference_ms_per_sample,
        "eval_samples_per_sec": eval_samples_per_sec,
    }


def build_log_header(class_names):
    return [
        "epoch",
        "loss",
        "mean_gap",
        "acc",
        "train_time_sec",
        "eval_time_sec",
        "text_time_sec",
        "inference_time_sec",
        "inference_samples_per_sec",
        "inference_ms_per_sample",
        "eval_samples_per_sec",
    ]


def build_log_row(epoch, loss, train_time_sec, dataset_result):
    return [
        epoch,
        loss,
        dataset_result["mean_gap"],
        dataset_result["acc"],
        train_time_sec,
        dataset_result["eval_time_sec"],
        dataset_result["text_time_sec"],
        dataset_result["inference_time_sec"],
        dataset_result["inference_samples_per_sec"],
        dataset_result["inference_ms_per_sample"],
        dataset_result["eval_samples_per_sec"],
    ]


def write_log_row(log_file, row):
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ModelEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, value in model.state_dict().items():
            if torch.is_floating_point(value):
                self.shadow[name] = value.detach().clone()

    @torch.no_grad()
    def update(self, model):
        state = model.state_dict()
        for name, value in state.items():
            if name not in self.shadow or not torch.is_floating_point(value):
                continue
            self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, model):
        self.backup = {}
        state = model.state_dict()
        for name in self.shadow:
            self.backup[name] = state[name].detach().clone()

    @torch.no_grad()
    def copy_to(self, model):
        state = model.state_dict()
        for name, value in self.shadow.items():
            state[name].copy_(value)

    @torch.no_grad()
    def restore(self, model):
        if not self.backup:
            return
        state = model.state_dict()
        for name, value in self.backup.items():
            state[name].copy_(value)
        self.backup = {}


def iter_params(obj):
    if obj is None:
        return []
    if isinstance(obj, torch.nn.Parameter):
        return [obj]
    if hasattr(obj, "parameters"):
        return list(obj.parameters())
    return []


def set_trainable(obj, requires_grad=True):
    for p in iter_params(obj):
        p.requires_grad = requires_grad


def get_vim_backbone_layers(model):
    backbone = getattr(model.visual, "backbone", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        return None
    return list(layers)


def set_vim_layer_range_trainable(model, start_idx, end_idx):
    layers = get_vim_backbone_layers(model)
    if layers is None:
        raise ValueError("Vim backbone does not expose backbone.layers; cannot use layer-wise two-stage training.")
    total_layers = len(layers)
    if total_layers < end_idx:
        raise ValueError(f"Vim two-stage expects at least {end_idx} layers, but found {total_layers}.")
    for layer in layers[start_idx:end_idx]:
        set_trainable(layer, True)


def configure_visual_trainability(model, args, stage_name="joint"):
    for p in model.parameters():
        p.requires_grad = False
    model.logit_scale.requires_grad = False

    if args.encoder == "vit":
        set_trainable(model.visual, True)
        return

    if args.vim_train_mode == "adapter":
        adapter_backbone_mode = {
            "adapter_only": "none",
            "adapter_back12": "back12",
            "adapter_full": "full",
        }[args.adapter_train_mode]
        mark_adapter_trainable(
            model,
            train_head=not args.adapter_freeze_head,
            train_backbone=adapter_backbone_mode,
        )
        return

    if args.vim_train_mode == "lora":
        mark_only_lora_and_head_trainable(model, train_head=not args.lora_freeze_head)
        return

    # The projection head stays trainable for all Vim training modes/stages.
    set_trainable(getattr(model.visual, "ln_post", None), True)
    set_trainable(getattr(model.visual, "proj", None), True)

    if stage_name == "head_only":
        pass
    elif stage_name == "front12":
        set_vim_layer_range_trainable(model, 0, 12)
    elif stage_name == "back12":
        set_vim_layer_range_trainable(model, 12, 24)
    else:
        set_trainable(getattr(model.visual, "backbone", None), True)


def build_lora_optimizer(model, args):
    lora_params, head_params = split_lora_and_head_parameters(model)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr_proj})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.lr_lora})
    if not param_groups:
        raise ValueError("No trainable LoRA/head parameters found.")

    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

def build_adapter_optimizer(model, args):
    adapter_params, head_params, backbone_params = split_adapter_head_backbone_parameters(model)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr_proj})
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": args.lr_adapter})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr_backbone})
    if not param_groups:
        raise ValueError("No trainable adapter/head/backbone parameters found.")

    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

def build_visual_optimizer(model, args):
    if args.encoder == "vim" and args.vim_train_mode == "adapter":
        return build_adapter_optimizer(model, args)
    if args.encoder == "vim" and args.vim_train_mode == "lora":
        return build_lora_optimizer(model, args)

    proj_params = []
    proj_params.extend(iter_params(getattr(model.visual, "ln_post", None)))
    proj_params.extend(iter_params(getattr(model.visual, "proj", None)))
    proj_params = [param for param in proj_params if param.requires_grad]
    proj_param_ids = {id(param) for param in proj_params}

    encoder_params = [
        param for param in model.visual.parameters()
        if param.requires_grad and id(param) not in proj_param_ids
    ]

    param_groups = []
    if proj_params:
        param_groups.append({"params": proj_params, "lr": args.lr_proj})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": args.lr_backbone})
    if not param_groups:
        raise ValueError("No trainable image encoder parameters found.")

    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

def count_trainable_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def plot_training_log(log_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(log_path)

    if "epoch" not in df.columns or df.empty:
        print(f"[Warn] no plottable rows in: {log_path}")
        return

    if "loss" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["loss"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss vs Epoch")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"))
        plt.close()

    if "acc" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["acc"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Test Accuracy vs Epoch")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "acc_curve.png"))
        plt.close()

    time_cols = [col for col in ["train_time_sec", "eval_time_sec", "inference_time_sec"] if col in df.columns]
    if time_cols:
        plt.figure(figsize=(8, 5))
        for col in time_cols:
            plt.plot(df["epoch"], df[col], marker="o", label=col)
        plt.xlabel("Epoch")
        plt.ylabel("Seconds")
        plt.title("Runtime vs Epoch")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "runtime_curve.png"))
        plt.close()

    if "inference_samples_per_sec" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["inference_samples_per_sec"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Samples/sec")
        plt.title("Inference Throughput vs Epoch")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "inference_throughput_curve.png"))
        plt.close()

    if "inference_ms_per_sample" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["inference_ms_per_sample"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("ms/sample")
        plt.title("Inference Latency vs Epoch")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "inference_latency_curve.png"))
        plt.close()

    if "mean_gap" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["mean_gap"], marker="o", linestyle="--", label="mean gap")
        plt.xlabel("Epoch")
        plt.ylabel("Similarity Gap")
        plt.title("Mean Similarity Gap vs Epoch")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "similarity_gap_curve.png"))
        plt.close()



    print(f"Saved plots to: {output_dir}")


def save_checkpoint(model, optimizer, epoch, acc, loss, save_path, mean_gap=None, dataset_name=None):
    payload = {
        "epoch": epoch,
        "acc": acc,
        "mean_gap": mean_gap,
        "loss": loss,
        "dataset": dataset_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    tmp_path = f"{save_path}.tmp"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, save_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    print(f"Saved checkpoint: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset to train on and evaluate on: cifar10, caltech101, oxfordpets, or stl10.",
    )
    parser.add_argument("--caltech_test_ratio", type=float, default=0.2)
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.5,
        help="Fraction of the official test split used as validation for early stopping. The rest is kept as final test.",
    )
    parser.add_argument("--train_per_class", type=int, default=None)
    parser.add_argument("--indices_path", type=str, default=None)
    parser.add_argument(
        "--train_aug",
        type=str,
        default="none",
        choices=["none", "rrc_flip", "rrc_flip_jitter", "randaugment"],
        help="Train-time image augmentation. Validation/test always use the original CLIP preprocess.",
    )
    parser.add_argument("--aug_crop_scale_min", type=float, default=0.6)
    parser.add_argument("--aug_hflip_prob", type=float, default=0.5)
    parser.add_argument("--color_jitter_brightness", type=float, default=0.2)
    parser.add_argument("--color_jitter_contrast", type=float, default=0.2)
    parser.add_argument("--color_jitter_saturation", type=float, default=0.2)
    parser.add_argument("--color_jitter_hue", type=float, default=0.05)
    parser.add_argument("--randaugment_num_ops", type=int, default=2)
    parser.add_argument("--randaugment_magnitude", type=int, default=7)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--use_ema", action="store_true", help="Evaluate and checkpoint an exponential moving average of model weights.")
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_epoch", type=int, default=1)
    parser.add_argument(
        "--class_counts",
        type=str,
        default=None,
        help="Comma-separated per-class training sizes to run sequentially, or all/full for the full train split, e.g. all,1000,500,50.",
    )
    # CIFAR10:      5000/class
    # STL10:        500/class
    # OxfordPets:   蝝?100/class trainval
    # Caltech101:   瘥?銝摰?銝???code ????caltech_test_ratio ??train/test

    parser.add_argument("--encoder", type=str, default="vim", choices=["vit", "vim"])
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/16",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"],
        help="CLIP model to use when --encoder vit. Vim fine-tuning keeps the Vim visual backbone.",
    )
    parser.add_argument("--vim_ckpt", type=str, default=None)
    parser.add_argument(
        "--finetune_vit",
        action="store_true",
        help="When --encoder vit, fine-tune the CLIP ViT image encoder instead of only running zero-shot evaluation.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr_proj", type=float, default=5e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_lora", type=float, default=1e-4, help="Learning rate for Vim LoRA parameters when --vim_train_mode lora.")
    parser.add_argument("--lr_adapter", type=float, default=1e-4, help="Learning rate for Vim adapter parameters when --vim_train_mode adapter.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--loss_mode",
        type=str,
        default="auto",
        choices=["auto", "class", "contrastive"],
        help="Training loss: auto uses contrastive for ViT fine-tuning and class loss for Vim; class uses class-prompt CE; contrastive uses CLIP-style batch image/text loss.",
    )
    parser.add_argument(
        "--vim_train_mode",
        type=str,
        default="joint",
        choices=["joint", "two_stage", "head_backbone", "lora", "adapter"],
        help="Vim training mode. joint trains backbone/head together; two_stage trains layer groups; head_backbone trains ln_post/proj head first then full backbone+head; lora trains LoRA adapters plus ln_post/proj head; adapter trains Vim adapters with configurable backbone scope.",
    )
    parser.add_argument("--lora_targets", type=str, default="out_proj", help="Comma-separated Vim mixer projections to wrap with LoRA: in_proj,out_proj,x_proj,dt_proj.")
    parser.add_argument("--lora_layers", type=str, default="last12", help="Vim layers for LoRA: last12/back12, front12/first12, all, start:end, or comma indices.")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_freeze_head", action="store_true", help="When --vim_train_mode lora, train only LoRA params and keep Vim ln_post/proj head frozen.")
    parser.add_argument("--adapter_layers", type=str, default="last12", help="Vim layers for adapters: last12/back12, front12/first12, all, start:end, or comma indices.")
    parser.add_argument("--adapter_dim", type=int, default=64, help="Bottleneck dimension for Vim adapters.")
    parser.add_argument("--adapter_feature_dim", type=int, default=768, help="Hidden feature dimension passed through Vim adapters.")
    parser.add_argument("--adapter_dropout", type=float, default=0.1)
    parser.add_argument("--adapter_scale", type=float, default=1.0)
    parser.add_argument("--adapter_train_mode", type=str, default="adapter_only", choices=["adapter_only", "adapter_back12", "adapter_full"], help="Train adapters+head only, adapters+back12+head, or adapters+full backbone+head.")
    parser.add_argument("--adapter_freeze_head", action="store_true", help="When --vim_train_mode adapter, train only adapter/backbone params and keep Vim ln_post/proj head frozen.")
    parser.add_argument("--stage1_epochs", type=int, default=None, help="Stage-1 epochs for staged Vim training modes. For two_stage this is front12; for head_backbone this is head-only. Defaults to --epochs.")
    parser.add_argument("--stage2_epochs", type=int, default=None, help="Stage-2 epochs for staged Vim training modes. For two_stage this is back12; for head_backbone this is full backbone+head. Defaults to --epochs.")
    parser.add_argument(
        "--hparam_search",
        action="store_true",
        help="Run a grid search over lr_proj/lr_backbone/weight_decay values and write hparam_summary.csv.",
    )
    parser.add_argument(
        "--hparam_random_search",
        action="store_true",
        help="Run reproducible random search over lr_proj/lr_backbone/weight_decay. Implies hyperparameter search mode.",
    )
    parser.add_argument("--hparam_trials", type=int, default=20, help="Number of random-search trials.")
    parser.add_argument("--lr_proj_values", type=str, default=None, help="Comma-separated lr_proj values for --hparam_search.")
    parser.add_argument("--lr_backbone_values", type=str, default=None, help="Comma-separated lr_backbone values for --hparam_search.")
    parser.add_argument("--lr_adapter_values", type=str, default=None, help="Comma-separated lr_adapter values for --hparam_search.")
    parser.add_argument("--adapter_dropout_values", type=str, default=None, help="Comma-separated adapter_dropout values for --hparam_search.")
    parser.add_argument("--weight_decay_values", type=str, default=None, help="Comma-separated weight_decay values for --hparam_search.")
    parser.add_argument("--lr_proj_range", type=str, default="1e-5,1e-3", help="min,max log-uniform range for random lr_proj.")
    parser.add_argument("--lr_backbone_range", type=str, default="1e-7,3e-5", help="min,max log-uniform range for random lr_backbone.")
    parser.add_argument("--lr_adapter_range", type=str, default=None, help="min,max log-uniform range for random lr_adapter. Defaults to fixed --lr_adapter.")
    parser.add_argument("--adapter_dropout_range", type=str, default=None, help="min,max uniform range for random adapter_dropout. Defaults to fixed --adapter_dropout.")
    parser.add_argument("--weight_decay_range", type=str, default="1e-6,1e-3", help="min,max log-uniform range for random weight_decay when nonzero.")
    parser.add_argument("--weight_decay_zero_prob", type=float, default=0.2, help="Probability of sampling weight_decay=0 in random search.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument(
        "--save_final_weights",
        action="store_true",
        help="Save final partial weights to final_weights.pth. Off by default because best.pth already contains full model_state_dict.",
    )
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument(
        "--save_epoch_checkpoints",
        action="store_true",
        help="Save full epoch_XXX.pth checkpoints. Off by default to avoid filling disk during sweeps.",
    )
    parser.add_argument("--output_dir", type=str, default="<LOCAL_ROOT>/CLIP/ft_vim_outputs")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    if args.hparam_search or args.hparam_random_search:
        if args.hparam_trials <= 0:
            raise ValueError("--hparam_trials must be positive")
        if not 0.0 <= args.weight_decay_zero_prob <= 1.0:
            raise ValueError("--weight_decay_zero_prob must be between 0 and 1")
        run_hparam_sweep(args)
        return

    if args.class_counts is not None:
        run_class_count_sweep(args)
        return

    set_seed(args.seed)

    if not 0.0 < args.caltech_test_ratio < 1.0:
        raise ValueError("--caltech_test_ratio must be between 0 and 1")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val_ratio must be between 0 and 1")
    if not 0.0 < args.aug_crop_scale_min <= 1.0:
        raise ValueError("--aug_crop_scale_min must be in (0, 1]")
    if not 0.0 <= args.aug_hflip_prob <= 1.0:
        raise ValueError("--aug_hflip_prob must be between 0 and 1")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label_smoothing must be in [0, 1)")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema_decay must be between 0 and 1")
    if args.ema_start_epoch < 1:
        raise ValueError("--ema_start_epoch must be >= 1")
    if args.vim_train_mode in {"two_stage", "head_backbone", "lora", "adapter"} and args.encoder != "vim":
        raise ValueError("--vim_train_mode two_stage/head_backbone/lora/adapter is only supported with --encoder vim")
    if args.vim_train_mode == "adapter":
        if args.adapter_dim <= 0:
            raise ValueError("--adapter_dim must be positive")
        if args.adapter_feature_dim <= 0:
            raise ValueError("--adapter_feature_dim must be positive")
        if not 0.0 <= args.adapter_dropout < 1.0:
            raise ValueError("--adapter_dropout must be in [0, 1)")
        if args.lr_adapter <= 0:
            raise ValueError("--lr_adapter must be positive")
    if args.vim_train_mode == "lora":
        if args.lora_rank <= 0:
            raise ValueError("--lora_rank must be positive")
        if args.lora_alpha <= 0:
            raise ValueError("--lora_alpha must be positive")
        if not 0.0 <= args.lora_dropout < 1.0:
            raise ValueError("--lora_dropout must be in [0, 1)")
        if args.lr_lora <= 0:
            raise ValueError("--lr_lora must be positive")
    if args.stage1_epochs is not None and args.stage1_epochs <= 0:
        raise ValueError("--stage1_epochs must be positive")
    if args.stage2_epochs is not None and args.stage2_epochs <= 0:
        raise ValueError("--stage2_epochs must be positive")

    train_loss_mode = args.loss_mode
    if train_loss_mode == "auto":
        train_loss_mode = "contrastive" if args.encoder == "vit" and args.finetune_vit else "class"

    dataset_name = normalize_dataset_name(args.dataset)
    eval_batch_size = args.eval_batch_size or args.batch_size
    if args.run_name:
        run_name = args.run_name
    elif args.encoder == "vit":
        safe_clip_model = args.clip_model.replace("/", "").replace("@", "_").replace("-", "").lower()
        run_name = f"vit_{safe_clip_model}_{dataset_name}"
    else:
        run_name = f"{args.encoder}_{dataset_name}"
    ckpt_dir = args.ckpt_dir or os.path.join(args.output_dir, "checkpoints")
    final_weights_path = args.save_path or os.path.join(ckpt_dir, "final_weights.pth")
    log_dir = os.path.join(args.output_dir, "log")
    args_dir = os.path.join(args.output_dir, "args")
    run_log_file = args.log_file or os.path.join(log_dir, f"{run_name}_train_log.csv")
    run_args_file = os.path.join(args_dir, f"{run_name}_args.json")
    run_dataset_stem = format_run_dataset_stem(run_name, dataset_name)
    save_indices_path = None
    if args.train_per_class is not None or args.indices_path is not None:
        save_indices_path = os.path.join(args_dir, f"{run_dataset_stem}_train_indices.json")
    save_val_indices_path = os.path.join(args_dir, f"{run_dataset_stem}_valid_indices.json")
    save_test_indices_path = os.path.join(args_dir, f"{run_dataset_stem}_test_indices.json")
    final_test_result_path = os.path.join(args_dir, f"{run_dataset_stem}_final_test_result.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Dataset:", dataset_name)
    print("CLIP model:", args.clip_model)
    print("Train loss mode:", train_loss_mode)
    print("Train augmentation:", args.train_aug)
    print("Label smoothing:", args.label_smoothing)
    print("EMA:", args.use_ema)
    if args.use_ema:
        print("EMA decay:", args.ema_decay)
        print("EMA start epoch:", args.ema_start_epoch)
    print("Vim train mode:", args.vim_train_mode)
    if args.encoder == "vim" and args.vim_train_mode == "adapter":
        print("Adapter layers:", args.adapter_layers)
        print("Adapter dim:", args.adapter_dim)
        print("Adapter feature dim:", args.adapter_feature_dim)
        print("Adapter dropout:", args.adapter_dropout)
        print("Adapter scale:", args.adapter_scale)
        print("Adapter train mode:", args.adapter_train_mode)
        print("Adapter lr:", args.lr_adapter)
        print("Adapter freeze head:", args.adapter_freeze_head)
    if args.encoder == "vim" and args.vim_train_mode == "lora":
        print("LoRA targets:", args.lora_targets)
        print("LoRA layers:", args.lora_layers)
        print("LoRA rank:", args.lora_rank)
        print("LoRA alpha:", args.lora_alpha)
        print("LoRA dropout:", args.lora_dropout)
        print("LoRA lr:", args.lr_lora)
        print("LoRA freeze head:", args.lora_freeze_head)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(run_log_file), exist_ok=True)
    os.makedirs(args_dir, exist_ok=True)
    final_weights_dir = os.path.dirname(final_weights_path)
    if final_weights_dir:
        os.makedirs(final_weights_dir, exist_ok=True)
    print("Run name:", run_name)
    print("Checkpoint dir:", ckpt_dir)
    print("Final weights path:", final_weights_path)
    print("Log file:", run_log_file)
    print("Args file:", run_args_file)
    save_run_args(
        args,
        run_args_file,
        extra={
            "dataset_name": dataset_name,
            "eval_batch_size": eval_batch_size,
            "run_name": run_name,
            "ckpt_dir": ckpt_dir,
            "final_weights_path": final_weights_path,
            "log_file": run_log_file,
            "args_file": run_args_file,
            "valid_indices_file": save_val_indices_path,
            "test_indices_file": save_test_indices_path,
            "final_test_result_file": final_test_result_path,
            "train_loss_mode": train_loss_mode,
            "train_aug": args.train_aug,
            "label_smoothing": args.label_smoothing,
            "use_ema": args.use_ema,
            "ema_decay": args.ema_decay,
            "ema_start_epoch": args.ema_start_epoch,
            "vim_train_mode": args.vim_train_mode,
            "stage1_epochs": args.stage1_epochs,
            "stage2_epochs": args.stage2_epochs,
            "lora_targets": args.lora_targets,
            "lora_layers": args.lora_layers,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lr_lora": args.lr_lora,
            "lora_freeze_head": args.lora_freeze_head,
            "adapter_layers": args.adapter_layers,
            "adapter_dim": args.adapter_dim,
            "adapter_feature_dim": args.adapter_feature_dim,
            "adapter_dropout": args.adapter_dropout,
            "adapter_scale": args.adapter_scale,
            "adapter_train_mode": args.adapter_train_mode,
            "lr_adapter": args.lr_adapter,
            "adapter_freeze_head": args.adapter_freeze_head,
        },
    )

    if args.encoder == "vit":
        model, preprocess = clip.load(args.clip_model, device=device)
        model = model.float()
    else:
        if args.vim_ckpt is None:
            raise ValueError("--vim_ckpt is required when --encoder vim")

        model, preprocess = clip.load(
            "ViT-B/16",
            device=device,
            vision_backbone="vim",
            vision_model_name="vim_base",
            vision_ckpt_path=args.vim_ckpt,
        )
        model = model.float()
        if args.vim_train_mode == "adapter":
            adapter_replacements = apply_adapters_to_vim(
                model.visual,
                layers=args.adapter_layers,
                feature_dim=args.adapter_feature_dim,
                adapter_dim=args.adapter_dim,
                dropout=args.adapter_dropout,
                scale=args.adapter_scale,
            )
            mark_adapter_trainable(
                model,
                train_head=not args.adapter_freeze_head,
                train_backbone={
                    "adapter_only": "none",
                    "adapter_back12": "back12",
                    "adapter_full": "full",
                }[args.adapter_train_mode],
            )
            adapter_summary = summarize_adapters(model, adapter_replacements)
            print(
                f"Applied Vim adapters: replacements={adapter_summary.num_replacements}, "
                f"adapter_params={adapter_summary.adapter_params:,}, "
                f"trainable_adapter_params={adapter_summary.trainable_adapter_params:,}"
            )
        if args.vim_train_mode == "lora":
            lora_replacements = apply_lora_to_vim(
                model.visual,
                targets=args.lora_targets,
                layers=args.lora_layers,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
            )
            mark_only_lora_and_head_trainable(model, train_head=not args.lora_freeze_head)
            lora_summary = summarize_lora(model, lora_replacements)
            print(
                f"Applied Vim LoRA: replacements={lora_summary.num_replacements}, "
                f"lora_params={lora_summary.lora_params:,}, "
                f"trainable_lora_params={lora_summary.trainable_lora_params:,}"
            )

    train_preprocess = build_train_preprocess(preprocess, args)

    train_ds, val_ds, test_ds, class_names, dataset_name = build_datasets(
        dataset_name=args.dataset,
        data_root=args.data_root,
        preprocess=preprocess,
        train_preprocess=train_preprocess,
        seed=args.seed,
        caltech_test_ratio=args.caltech_test_ratio,
        val_ratio=args.val_ratio,
        train_per_class=args.train_per_class,
        indices_path=args.indices_path,
        save_indices_path=save_indices_path,
        save_val_indices_path=save_val_indices_path,
        save_test_indices_path=save_test_indices_path,
    )
    print(f"Train samples: {len(train_ds)}")
    print(f"Valid samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Classes: {len(class_names)}")

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        generator=g,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
    )

    log_header = build_log_header(class_names)
    with open(run_log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(log_header)

    if args.encoder == "vit" and not args.finetune_vit:
        print("\n=== ViT Zero-shot Valid Evaluation ===")
        val_result = evaluate_dataset(model, val_loader, class_names, device)
        if val_result is not None:
            write_log_row(
                run_log_file,
                build_log_row(
                    epoch=0,
                    loss="",
                    train_time_sec=0.0,
                    dataset_result=val_result,
                ),
            )
            plot_training_log(run_log_file, args.output_dir)
        print("\n=== ViT Zero-shot Final Test Evaluation ===")
        test_result = evaluate_dataset(model, test_loader, class_names, device)
        save_eval_result(final_test_result_path, test_result, split_name="test", epoch=0)
        return

    if args.encoder == "vit":
        print("\n=== ViT Image Encoder Fine-tuning ===")
    elif args.vim_train_mode == "two_stage":
        print("\n=== Vim Two-stage Fine-tuning ===")
    elif args.vim_train_mode == "head_backbone":
        print("\n=== Vim Head-then-Backbone Fine-tuning ===")
    elif args.vim_train_mode == "adapter":
        print("\n=== Vim Adapter Fine-tuning ===")
    elif args.vim_train_mode == "lora":
        print("\n=== Vim LoRA Fine-tuning ===")
    else:
        print("\n=== Vim Joint Fine-tuning ===")

    # print("\nTrainable parameters:")
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(f"  {name}: {tuple(p.shape)}")

    class_prompts = build_class_prompts(class_names)
    class_tokens = clip.tokenize(class_prompts).to(device)

    class_text_features = None
    if train_loss_mode == "class":
        with torch.no_grad():
            class_text_features = model.encode_text(class_tokens)
            class_text_features = F.normalize(class_text_features, dim=-1)

    if args.encoder == "vim" and args.vim_train_mode == "two_stage":
        stage_plan = [
            {"name": "front12", "label": "front12_layers_plus_proj_head", "epochs": args.stage1_epochs or args.epochs},
            {"name": "back12", "label": "back12_layers_plus_proj_head", "epochs": args.stage2_epochs or args.epochs},
        ]
    elif args.encoder == "vim" and args.vim_train_mode == "head_backbone":
        stage_plan = [
            {"name": "head_only", "label": "ln_post_plus_proj_head", "epochs": args.stage1_epochs or args.epochs},
            {"name": "joint", "label": "full_backbone_plus_proj_head", "epochs": args.stage2_epochs or args.epochs},
        ]
    elif args.encoder == "vim" and args.vim_train_mode == "adapter":
        stage_plan = [{"name": "adapter", "label": f"adapter_{args.adapter_train_mode}", "epochs": args.epochs}]
    elif args.encoder == "vim" and args.vim_train_mode == "lora":
        stage_plan = [{"name": "lora", "label": "lora_plus_proj_head", "epochs": args.epochs}]
    else:
        stage_plan = [{"name": "joint", "label": "joint", "epochs": args.epochs}]

    best_acc = float("-inf")
    best_mean_gap = float("-inf")
    best_epoch = -1
    epochs_no_improve = 0
    stop_training = False

    global_epoch = 0
    best_ckpt_path = os.path.join(ckpt_dir, "best.pth")
    ema = None

    for stage_idx, stage in enumerate(stage_plan):
        stage_name = stage["name"]
        stage_label = stage["label"]
        stage_epochs = stage["epochs"]

        configure_visual_trainability(model, args, stage_name=stage_name)
        optimizer = build_visual_optimizer(model, args)
        print(f"\n=== Stage {stage_idx + 1}/{len(stage_plan)}: {stage_label} ({stage_epochs} epochs) ===")
        print(f"Trainable parameters: {count_trainable_params(model):,}")

        epochs_no_improve = 0
        for stage_epoch in range(stage_epochs):
            global_epoch += 1
            running_loss = 0.0
            model.train()

            train_start = timer_now(device)
            for step, (images, labels) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                image_features = model.encode_image(images)
                logit_scale = model.logit_scale.exp().detach()
                if train_loss_mode == "contrastive":
                    batch_text_tokens = class_tokens[labels]
                    with torch.no_grad():
                        batch_text_features = model.encode_text(batch_text_tokens)
                    loss = clip_contrastive_loss(
                        image_features=image_features,
                        text_features=batch_text_features,
                        logit_scale=logit_scale,
                        label_smoothing=args.label_smoothing,
                    )
                else:
                    loss = clip_classification_loss(
                        image_features=image_features,
                        class_text_features=class_text_features,
                        labels=labels,
                        logit_scale=logit_scale,
                        label_smoothing=args.label_smoothing,
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if args.use_ema and global_epoch >= args.ema_start_epoch:
                    if ema is None:
                        ema = ModelEMA(model, args.ema_decay)
                    else:
                        ema.update(model)

                running_loss += loss.item()

            train_time_sec = timer_now(device) - train_start
            train_samples_per_sec = len(train_ds) / train_time_sec if train_time_sec > 0 else float("inf")
            avg_loss = running_loss / len(train_loader)
            print(
                f"Stage {stage_label} Epoch [{stage_epoch+1}/{stage_epochs}] "
                f"Global Epoch [{global_epoch}] Step [{step+1}/{len(train_loader)}] Loss: {avg_loss:.4f}"
            )
            print(f"Training time: {train_time_sec:.4f} sec ({train_samples_per_sec:.2f} samples/sec)")
            print(f"\n=== Global epoch {global_epoch} done ===")

            use_ema_for_eval = ema is not None and global_epoch >= args.ema_start_epoch
            if use_ema_for_eval:
                ema.store(model)
                ema.copy_to(model)
            dataset_result = evaluate_dataset(model, val_loader, class_names, device)
            should_stop_stage = False

            if dataset_result is not None:
                current_acc = dataset_result["acc"]
                current_mean_gap = dataset_result["mean_gap"]

                write_log_row(
                    run_log_file,
                    build_log_row(
                        epoch=global_epoch,
                        loss=avg_loss,
                        train_time_sec=train_time_sec,
                        dataset_result=dataset_result,
                    ),
                )

                if args.save_epoch_checkpoints:
                    epoch_ckpt_path = os.path.join(ckpt_dir, f"epoch_{global_epoch:03d}.pth")
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=global_epoch,
                        acc=current_acc,
                        loss=avg_loss,
                        mean_gap=current_mean_gap,
                        dataset_name=dataset_name,
                        save_path=epoch_ckpt_path,
                    )

                if current_acc > best_acc + args.min_delta:
                    best_acc = current_acc
                    best_mean_gap = current_mean_gap
                    best_epoch = global_epoch
                    epochs_no_improve = 0

                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=global_epoch,
                        acc=current_acc,
                        loss=avg_loss,
                        mean_gap=current_mean_gap,
                        dataset_name=dataset_name,
                        save_path=best_ckpt_path,
                    )

                    print(f"[Best] epoch={best_epoch}, val_acc={best_acc:.4f}, val_mean_gap={best_mean_gap:.4f}")
                else:
                    epochs_no_improve += 1
                    print(f"[No Improve] {epochs_no_improve}/{args.patience} (val_acc={current_acc:.4f}, best={best_acc:.4f})")

                if epochs_no_improve >= args.patience:
                    print(
                        f"\n[Early Stop] Stop stage {stage_label} at stage epoch {stage_epoch+1}. "
                        f"Best global epoch = {best_epoch}, best val_acc = {best_acc:.4f}"
                    )
                    print()
                    should_stop_stage = True

                print()

            if use_ema_for_eval:
                ema.restore(model)
            model.train()
            if should_stop_stage:
                break

        if args.encoder == "vim" and args.vim_train_mode in {"two_stage", "head_backbone"} and stage_idx < len(stage_plan) - 1:
            if best_epoch > 0 and os.path.exists(best_ckpt_path):
                checkpoint = torch.load(best_ckpt_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded best checkpoint before next stage: epoch={best_epoch}, val_acc={best_acc:.4f}")

    best_ckpt_path = os.path.join(ckpt_dir, "best.pth")
    if best_epoch > 0 and os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best checkpoint for final test: epoch={best_epoch}, val_acc={best_acc:.4f}")

    print("\n=== Final Test Evaluation ===")
    final_test_result = evaluate_dataset(model, test_loader, class_names, device)
    save_eval_result(final_test_result_path, final_test_result, split_name="test", epoch=best_epoch if best_epoch > 0 else None)

    if args.save_final_weights:
        tmp_final_weights_path = f"{final_weights_path}.tmp"
        try:
            if args.encoder == "vit":
                final_payload = {
                    "visual": model.visual.state_dict(),
                    "encoder": args.encoder,
                    "clip_model": args.clip_model,
                    "dataset": dataset_name,
                    "class_names": class_names,
                }
            else:
                final_payload = {
                    "ln_post": model.visual.ln_post.state_dict(),
                    "proj": model.visual.proj.state_dict(),
                    "backbone": model.visual.backbone.state_dict(),
                    "encoder": args.encoder,
                    "clip_model": args.clip_model,
                    "dataset": dataset_name,
                    "class_names": class_names,
                }
            torch.save(final_payload, tmp_final_weights_path)
            os.replace(tmp_final_weights_path, final_weights_path)
        except Exception:
            if os.path.exists(tmp_final_weights_path):
                try:
                    os.remove(tmp_final_weights_path)
                except OSError:
                    pass
            raise
        print(f"Saved fine-tuned weights to: {final_weights_path}")
    else:
        print("Skipped final_weights.pth. Use --save_final_weights to save it.")

    plot_training_log(run_log_file, args.output_dir)


if __name__ == "__main__":
    main()


# Load fine-tuned checkpoints example:
# ckpt = torch.load("/checkpoints/final_weights.pth", map_location="cpu")
# model.visual.ln_post.load_state_dict(ckpt["ln_post"])
# model.visual.proj.load_state_dict(ckpt["proj"])
