# Next Steps for the Visual Path Following Direction

## Phase 1: Validate the Dataset

1. Inspect `datasets/vstd_tiny/previews/train_preview.png`.
2. Decide whether the visual style is readable enough:
   - arrow size
   - start marker visibility
   - final object visibility
   - distractor density
3. Generate a slightly larger pilot dataset, for example:

```bash
python3 scripts/generate_vstd.py \
  --config configs/vstd_tiny.yaml \
  --output_root datasets \
  --overwrite
```

## Phase 2: Add Difficulty Splits

Create configs for:

- `vstd_easy`: short paths, few distractors.
- `vstd_medium`: medium paths, moderate distractors.
- `vstd_hard`: long paths, many distractors.

The important paper figure should be a curve, not just one accuracy number:

```text
path length / distractor density -> accuracy
```

## Phase 3: Build a Clean Dataset Loader

Add a PyTorch dataset that reads:

- `metadata.jsonl`
- PNG image paths
- labels
- prompts from `classes.json`

This loader should support both ViT-CLIP and Vim-CLIP experiments.

Status: implemented in `src/vstd/dataset.py`.

Smoke test:

```bash
python3 scripts/smoke_vstd_loader.py \
  --dataset_dir datasets/vstd_tiny \
  --split train \
  --batch_size 4
```

## Phase 4: Run First Baselines

Start with:

1. ViT-CLIP frozen image encoder + linear/projection classifier.
2. ViT-CLIP full fine-tuning.
3. Vim-CLIP projection-head-only fine-tuning.
4. Vim-CLIP adapter fine-tuning.

Do not start with all hyperparameter sweeps. First confirm the task is learnable.

Current scripts:

```bash
# Environment-independent data/training smoke test.
python3 scripts/train_vstd_toy_cnn.py \
  --dataset_dir datasets/vstd_tiny \
  --output_dir runs/toy_cnn_vstd_tiny_smoke \
  --epochs 2 \
  --batch_size 32 \
  --device cpu
```

```bash
# Frozen CLIP image-feature linear baseline.
# Uses the vendored modified CLIP package in external/clip_vstd.
python3 scripts/train_vstd_clip_linear.py \
  --dataset_dir datasets/vstd_easy \
  --output_dir runs/vit_b16_clip_linear_vstd_easy \
  --encoder vit \
  --clip_model ViT-B/16 \
  --epochs 20
```

For Vim:

```bash
python3 scripts/train_vstd_clip_linear.py \
  --dataset_dir datasets/vstd_easy \
  --output_dir runs/vim_clip_linear_vstd_easy \
  --encoder vim \
  --clip_model ViT-B/32 \
  --vim_ckpt /path/to/vim_checkpoint.pth \
  --epochs 20
```

The scripts use the vendored CLIP package in this repository, but the active
Python environment still needs the CLIP runtime dependencies such as
`torchvision`.

## Phase 5: Make the Research Claim

The target claim should be:

> Vim is not universally better than ViT, but it becomes more competitive when
> classification depends on long-range visual state tracking under increasing
> path length and distractor density.

Required evidence:

- Accuracy by path length.
- Accuracy by distractor density.
- Inference latency/throughput.
- Similarity gap between correct prompt and strongest incorrect prompt.
- Representation geometry before/after fine-tuning.

## Immediate TODO

- Generate `vstd_medium` and `vstd_hard` once the easy split is visually approved.
- Train first baselines on `vstd_easy` before running large sweeps.
- Run Vim frozen-feature baseline on a GPU node.
- Generate `vstd_medium` only after Vim/ViT easy results are available.
- Add full fine-tuning scripts only after frozen-feature results are inspected.
- Next priority: implement VSTD fine-tuning so ViT and Vim can adapt to the path-following task.
- `vstd_easy` is solved by Vim full visual fine-tuning, so use it mainly as a sanity check.
- Generate and evaluate `vstd_medium` next; keep `vstd_hard` for the main stress test.
