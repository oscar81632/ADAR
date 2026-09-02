# Baseline Notes

## Implemented

| Script | Purpose |
|---|---|
| `scripts/smoke_vstd_loader.py` | Confirms the VSTD loader reads images, labels, prompts, and metadata. |
| `scripts/train_vstd_toy_cnn.py` | Minimal CNN training-loop sanity check without torchvision or CLIP. |
| `scripts/train_vstd_clip_linear.py` | Frozen CLIP image-feature linear classifier for ViT or Vim. |
| `scripts/train_vstd_clip_alignment.py` | CLIP-style projection-head probing using fixed text prompt prototypes. |

## What the Toy CNN Means

The toy CNN is not a research baseline. It is only a pipeline sanity check. Low
accuracy is expected because the dataset requires explicit path following and
the tiny CNN has weak global reasoning ability.

## First Real Baseline

The first meaningful baseline should be:

```bash
python3 scripts/train_vstd_clip_linear.py \
  --dataset_dir datasets/vstd_easy \
  --output_dir runs/vit_b16_clip_linear_vstd_easy \
  --encoder vit \
  --clip_model ViT-B/16 \
  --epochs 20
```

This freezes CLIP image features and trains only a linear classifier. If this is
near chance, then fixed CLIP features do not encode enough path-following
information and we should move to fine-tuning.

## Key Result Fields

`test_result.json` contains:

- `test_acc`
- `accuracy_by_path_length`
- `accuracy_by_distractor_ratio`
- `accuracy_by_start_end_manhattan`

These are more important than a single aggregate accuracy because the thesis
claim depends on degradation under longer dependency paths.

## Current ViT-B/16 Frozen Baseline

Run:

```bash
python scripts/train_vstd_clip_linear.py \
  --dataset_dir datasets/vstd_easy \
  --output_dir runs/vit_b16_clip_linear_vstd_easy \
  --encoder vit \
  --clip_model ViT-B/16 \
  --epochs 20 \
  --batch_size 128 \
  --feature_batch_size 64 \
  --device cpu
```

Result:

```text
test_acc = 0.4875
```

Accuracy by distractor ratio:

```text
D=0.0  -> 0.9984
D=0.15 -> 0.2632
D=0.25 -> 0.2442
```

This is a useful sanity result: frozen ViT-CLIP features can almost perfectly
solve images without distractors, but degrade sharply when distractor arrows and
shapes are introduced. This supports the benchmark design because the task is
not just object/color recognition; distractors force path tracking.

Updated 100-epoch frozen-feature results are summarized in
`docs/BASELINE_RESULTS.md`.

## Vim Runtime Note

The current Vim/Mamba implementation cannot run on CPU in this environment.
`causal_conv1d` and Triton layer norm kernels require CUDA. Therefore:

- ViT baselines can run on CPU.
- Vim baselines must run on a GPU node.

Known Vim checkpoint:

```text
checkpoints/vim-base/vim_b_midclstok_81p9acc.pth
```

GPU command:

```bash
python scripts/train_vstd_clip_linear.py \
  --dataset_dir datasets/vstd_easy \
  --output_dir runs/vim_clip_linear_vstd_easy \
  --encoder vim \
  --vim_ckpt checkpoints/vim-base/vim_b_midclstok_81p9acc.pth \
  --epochs 20 \
  --batch_size 128 \
  --feature_batch_size 32 \
  --device cuda
```

The script forces `ViT-B/16` for the CLIP text side when `--encoder vim`,
matching the archived Vim baseline protocol.
