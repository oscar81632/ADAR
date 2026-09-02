import os
import argparse
import csv

import pandas as pd
import matplotlib.pyplot as plt

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from PIL import Image
import clip


log_file = "<LOCAL_ROOT>/CLIP/ft_vim_outputs/train_log.csv"


class CIFAR10TextDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool, preprocess):
        self.ds = CIFAR10(root=root, train=train, download=True)
        self.preprocess = preprocess
        self.class_names = self.ds.classes

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        img = self.preprocess(img)
        return img, label


def collate_fn(batch):
    images = torch.stack([x[0] for x in batch], dim=0)
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    return images, labels


def build_class_prompts(class_names):
    return [f"a photo of a {name}" for name in class_names]


def clip_classification_loss(image_features, class_text_features, labels, logit_scale):
    image_features = F.normalize(image_features, dim=-1)
    class_text_features = F.normalize(class_text_features, dim=-1)
    logits = logit_scale * image_features @ class_text_features.t()
    return F.cross_entropy(logits, labels)


def clip_contrastive_loss(image_features, text_features, logit_scale):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    targets = torch.arange(image_features.size(0), device=image_features.device)

    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
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


def evaluate_dataset(model, preprocess, device, root="test_data", class_names=None):
    model.eval()

    if not os.path.isdir(root):
        print(f"[Warn] missing eval root: {root}")
        return None

    if class_names is None:
        class_order = sorted(
            name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        )
    else:
        class_order = [
            name for name in class_names
            if os.path.isdir(os.path.join(root, name))
        ]

    if len(class_order) == 0:
        print(f"[Warn] no class folders found under: {root}")
        return None

    if len(class_order) < 2:
        print("[Warn] mean_gap needs at least 2 classes; skip dataset eval.")
        return None

    prompts = build_class_prompts(class_order)
    text = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_feat = model.encode_text(text)
        text_feat = F.normalize(text_feat, dim=-1)

    results = {cls: [] for cls in class_order}

    for cls in class_order:
        cls_path = os.path.join(root, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            try:
                img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            except Exception as e:
                print(f"[Warn] failed to read {img_path}: {e}")
                continue

            with torch.no_grad():
                image_feat = model.encode_image(img)
                image_feat = F.normalize(image_feat, dim=-1)
                sim = image_feat @ text_feat.t()
                sim = sim.squeeze(0).detach().cpu().tolist()

            results[cls].append(sim)

    print("\n=== Dataset Evaluation ===")

    correct = 0
    total = 0
    avg_scores = {}
    gaps = {}

    for label, cls in enumerate(class_order):
        if len(results[cls]) == 0:
            print(f"[Warn] no valid images for class: {cls}")
            return None

        sims = torch.tensor(results[cls])
        avg = sims.mean(dim=0)
        avg_scores[cls] = avg.tolist()

        wrong_indices = [i for i in range(len(class_order)) if i != label]
        correct_score = avg[label]
        best_wrong_score = avg[wrong_indices].max()
        gap = (correct_score - best_wrong_score).item()
        gaps[cls] = gap

        print(f"\nClass: {cls}")
        for prompt_cls, score in zip(class_order, avg):
            print(f"  {prompt_cls}: {score:.4f}")
        print(f"  gap: {gap:.4f}")

        preds = sims.argmax(dim=1)
        correct += (preds == label).sum().item()
        total += len(preds)

    acc = correct / total
    mean_gap = sum(gaps.values()) / len(gaps)
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Mean gap: {mean_gap:.4f}")

    return {
        "class_order": class_order,
        "avg_scores": avg_scores,
        "gaps": gaps,
        "mean_gap": mean_gap,
        "acc": acc,
    }


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        plt.title("Dataset Accuracy vs Epoch")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "acc_curve.png"))
        plt.close()

    gap_cols = [col for col in df.columns if col.endswith("_gap") and col != "mean_gap"]
    if gap_cols or "mean_gap" in df.columns:
        plt.figure(figsize=(8, 5))
        for col in gap_cols:
            plt.plot(df["epoch"], df[col], marker="o", label=col[:-4])
        if "mean_gap" in df.columns:
            plt.plot(df["epoch"], df["mean_gap"], marker="o", linestyle="--", label="mean gap")
        plt.xlabel("Epoch")
        plt.ylabel("Similarity Gap")
        plt.title("Correct-vs-Wrong Similarity Gap vs Epoch")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "similarity_gap_curve.png"))
        plt.close()

    positive_cols = []
    for col in df.columns:
        if "_as_" not in col:
            continue
        lhs, rhs = col.split("_as_", 1)
        if lhs == rhs:
            positive_cols.append(col)

    if positive_cols:
        plt.figure(figsize=(8, 5))
        for col in positive_cols:
            plt.plot(df["epoch"], df[col], marker="o", label=col.replace("_as_", " as "))
        plt.xlabel("Epoch")
        plt.ylabel("Similarity")
        plt.title("Positive Similarity vs Epoch")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "positive_similarity_curve.png"))
        plt.close()

    source_classes = []
    for col in df.columns:
        if "_as_" not in col:
            continue
        source_cls = col.split("_as_", 1)[0]
        if source_cls not in source_classes:
            source_classes.append(source_cls)

    for source_cls in source_classes:
        row_cols = [col for col in df.columns if col.startswith(f"{source_cls}_as_")]
        if not row_cols:
            continue

        plt.figure(figsize=(8, 5))
        for col in row_cols:
            prompt_cls = col.split("_as_", 1)[1]
            plt.plot(df["epoch"], df[col], marker="o", label=f"{prompt_cls} prompt")
        plt.xlabel("Epoch")
        plt.ylabel("Similarity")
        plt.title(f"{source_cls} Images: Similarity vs Epoch")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        safe_name = source_cls.replace(os.sep, "_").replace(" ", "_")
        plt.savefig(os.path.join(output_dir, f"{safe_name}_row_curve.png"))
        plt.close()

    print(f"Saved plots to: {output_dir}")


def save_checkpoint(model, optimizer, epoch, acc, loss, save_path, mean_gap=None):
    torch.save(
        {
            "epoch": epoch,
            "acc": acc,
            "mean_gap": mean_gap,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )
    print(f"Saved checkpoint: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--encoder", type=str, default="vim", choices=["vit", "vim"])
    parser.add_argument("--vim_ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr_proj", type=float, default=5e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="vim_proj_head_ft.pth")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--ckpt_dir", type=str, default="<LOCAL_ROOT>/CLIP/checkpoints")
    args = parser.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.encoder == "vit":
        model, preprocess = clip.load("ViT-B/32", device=device)
        model = model.float()

        print("\n=== ViT Zero-shot Dataset Evaluation ===")
        evaluate_dataset(model, preprocess, device, root="test_data")
        return

    if args.encoder == "vim":
        if args.vim_ckpt is None:
            raise ValueError("--vim_ckpt is required when --encoder vim")

        model, preprocess = clip.load(
            "ViT-B/32",
            device=device,
            vision_backbone="vim",
            vision_model_name="vim_base",
            vision_ckpt_path=args.vim_ckpt,
        )

    for p in model.parameters():
        p.requires_grad = False

    for p in model.visual.ln_post.parameters():
        p.requires_grad = True

    for p in model.visual.proj.parameters():
        p.requires_grad = True

    model.logit_scale.requires_grad = False

    for p in model.visual.backbone.parameters():
        p.requires_grad = True

    print("\nTrainable parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {name}: {tuple(p.shape)}")

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_ds = CIFAR10TextDataset(
        root=args.data_root,
        train=True,
        preprocess=preprocess,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        generator=g,
    )

    class_names = train_ds.class_names

    eval_root = "test_data"
    if os.path.isdir(eval_root):
        eval_class_names = sorted(
            name for name in os.listdir(eval_root)
            if os.path.isdir(os.path.join(eval_root, name))
        )
    else:
        eval_class_names = list(class_names)

    log_header = ["epoch", "loss"]
    log_header.extend(
        f"{source_cls}_as_{prompt_cls}"
        for source_cls in eval_class_names
        for prompt_cls in eval_class_names
    )
    log_header.extend(f"{cls}_gap" for cls in eval_class_names)
    log_header.extend(["mean_gap", "acc"])

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(log_header)

    class_prompts = build_class_prompts(class_names)
    class_tokens = clip.tokenize(class_prompts).to(device)

    with torch.no_grad():
        class_text_features = model.encode_text(class_tokens)
        class_text_features = F.normalize(class_text_features, dim=-1)

    proj_params = list(model.visual.ln_post.parameters()) + list(model.visual.proj.parameters())

    backbone_params = []
    for name, p in model.visual.backbone.named_parameters():
        if p.requires_grad:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": proj_params, "lr": args.lr_proj},
            {"params": backbone_params, "lr": args.lr_backbone},
        ],
        weight_decay=args.weight_decay,
    )

    best_mean_gap = float("-inf")
    best_epoch = -1
    epochs_no_improve = 0
    stop_training = False

    for epoch in range(args.epochs):
        running_loss = 0.0
        model.train()

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            image_features = model.encode_image(images)
            logit_scale = model.logit_scale.exp().detach()
            loss = clip_classification_loss(
                image_features=image_features,
                class_text_features=class_text_features,
                labels=labels,
                logit_scale=logit_scale,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Step [{step+1}/{len(train_loader)}] Loss: {avg_loss:.4f}")
        print(f"\n=== Epoch {epoch+1} done ===")

        dataset_result = evaluate_dataset(
            model,
            preprocess,
            device,
            root=eval_root,
            class_names=eval_class_names,
        )
        model.train()

        if dataset_result is not None:
            current_acc = dataset_result["acc"]
            current_mean_gap = dataset_result["mean_gap"]
            result_class_order = dataset_result["class_order"]

            log_row = [epoch + 1, avg_loss]
            log_row.extend(
                dataset_result["avg_scores"][source_cls][prompt_idx]
                for source_cls in result_class_order
                for prompt_idx, _ in enumerate(result_class_order)
            )
            log_row.extend(dataset_result["gaps"][cls] for cls in result_class_order)
            log_row.extend([current_mean_gap, current_acc])

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(log_row)

            epoch_ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch+1:03d}.pth")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                acc=current_acc,
                loss=avg_loss,
                mean_gap=current_mean_gap,
                save_path=epoch_ckpt_path,
            )

            if current_mean_gap > best_mean_gap + args.min_delta:
                best_mean_gap = current_mean_gap
                best_epoch = epoch + 1
                epochs_no_improve = 0

                best_ckpt_path = os.path.join(args.ckpt_dir, "best.pth")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    acc=current_acc,
                    loss=avg_loss,
                    mean_gap=current_mean_gap,
                    save_path=best_ckpt_path,
                )

                print(f"[Best] epoch={best_epoch}, mean_gap={best_mean_gap:.4f}, acc={current_acc:.4f}")
            else:
                epochs_no_improve += 1
                print(f"[No Improve] {epochs_no_improve}/{args.patience} (mean_gap={current_mean_gap:.4f}, best={best_mean_gap:.4f})")

            if epochs_no_improve >= args.patience:
                print(f"\n[Early Stop] Stop at epoch {epoch+1}. Best epoch = {best_epoch}, best mean_gap = {best_mean_gap:.4f}")
                stop_training = True

            print()

        if stop_training:
            break

    torch.save(
        {
            "ln_post": model.visual.ln_post.state_dict(),
            "proj": model.visual.proj.state_dict(),
            "backbone": model.visual.backbone.state_dict(),
        },
        args.save_path,
    )
    print(f"Saved fine-tuned projection head to: {args.save_path}")

    output_dir = "<LOCAL_ROOT>/CLIP/ft_vim_outputs"
    plot_training_log(log_file, output_dir)


if __name__ == "__main__":
    main()


# Load fine-tuned checkpoints example:
# ckpt = torch.load("/checkpoints/vim_proj_head_ft.pth", map_location="cpu")
# model.visual.ln_post.load_state_dict(ckpt["ln_post"])
# model.visual.proj.load_state_dict(ckpt["proj"])