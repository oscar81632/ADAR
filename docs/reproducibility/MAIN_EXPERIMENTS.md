# Main VSTD Experiments

This file records the paper-facing experiments after the project pivoted from
Vim replacement to ViT readout design.

## Environment

Run commands from the repository root. The training scripts load the vendored
CLIP-VSTD package automatically, so no external CLIP source path is required.

```bash
cd ADAR_Research
```

Generate the main VSTD split before training:

```bash
python scripts/generate_vstd.py \
  --config configs/vstd_decoy_trainhard.yaml \
  --output_root datasets \
  --overwrite
```

All main experiments use:

```text
dataset: datasets/vstd_decoy_trainhard
epochs: 6
batch_size: 16
eval_batch_size: 64
lr: 1e-5
weight_decay: 1e-4
checkpoint selection: best validation accuracy
device: cuda
```

## Baseline: CLIP ViT-B/16

```bash
python scripts/train_vstd_clip_alignment.py \
  --dataset_dir datasets/vstd_decoy_trainhard \
  --output_dir runs/vit_b16_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111 \
  --encoder vit --clip_model ViT-B/16 --train_scope full \
  --epochs 6 --batch_size 16 --eval_batch_size 64 \
  --lr 1e-5 --weight_decay 1e-4 --seed 111 --device cuda

python scripts/train_vstd_clip_alignment.py \
  --dataset_dir datasets/vstd_decoy_trainhard \
  --output_dir runs/vit_b16_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5 \
  --encoder vit --clip_model ViT-B/16 --train_scope full \
  --epochs 6 --batch_size 16 --eval_batch_size 64 \
  --lr 1e-5 --weight_decay 1e-4 --seed 123 --device cuda

python scripts/train_vstd_clip_alignment.py \
  --dataset_dir datasets/vstd_decoy_trainhard \
  --output_dir runs/vit_b16_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed321 \
  --encoder vit --clip_model ViT-B/16 --train_scope full \
  --epochs 6 --batch_size 16 --eval_batch_size 64 \
  --lr 1e-5 --weight_decay 1e-4 --seed 321 --device cuda
```

## CLIP Visual Encoder Baselines

RN50 and RN101 are included as CLIP ResNet baselines. RN50x4 is the preferred
scaled-ResNet comparison because its visual-side parameter count is close to
ViT-B/16. RN50x4 uses a smaller batch size because the larger visual tower has
higher GPU memory usage.

```bash
python scripts/train_vstd_clip_alignment.py \
  --dataset_dir datasets/vstd_decoy_trainhard \
  --output_dir runs/rn50_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5 \
  --encoder vit --clip_model RN50 --train_scope full \
  --epochs 6 --batch_size 16 --eval_batch_size 64 \
  --lr 1e-5 --weight_decay 1e-4 --seed 123 --device cuda

python scripts/train_vstd_clip_alignment.py \
  --dataset_dir datasets/vstd_decoy_trainhard \
  --output_dir runs/rn101_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5 \
  --encoder vit --clip_model RN101 --train_scope full \
  --epochs 6 --batch_size 16 --eval_batch_size 64 \
  --lr 1e-5 --weight_decay 1e-4 --seed 123 --device cuda

python scripts/train_vstd_clip_alignment.py \
  --dataset_dir datasets/vstd_decoy_trainhard \
  --output_dir runs/rn50x4_clip_fullft_vstd_decoy_trainhard_6e_b8_lr1e-5_run2 \
  --encoder vit --clip_model RN50x4 --train_scope full \
  --epochs 6 --batch_size 8 --eval_batch_size 32 \
  --lr 1e-5 --weight_decay 1e-4 --seed 123 --device cuda
```

## Main Method: ADAR

ADAR uses CLIP ViT-B/16 with adaptive decoy-aware readout:

```bash
python scripts/train_vstd_clip_alignment.py \
  --dataset_dir datasets/vstd_decoy_trainhard \
  --output_dir runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111 \
  --encoder vit --clip_model ViT-B/16 --train_scope full \
  --epochs 6 --batch_size 16 --eval_batch_size 64 \
  --lr 1e-5 --weight_decay 1e-4 \
  --vit_path_adapter adaptive_decoy_aware \
  --seed 111 --device cuda

python scripts/train_vstd_clip_alignment.py \
  --dataset_dir datasets/vstd_decoy_trainhard \
  --output_dir runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123 \
  --encoder vit --clip_model ViT-B/16 --train_scope full \
  --epochs 6 --batch_size 16 --eval_batch_size 64 \
  --lr 1e-5 --weight_decay 1e-4 \
  --vit_path_adapter adaptive_decoy_aware \
  --seed 123 --device cuda

python scripts/train_vstd_clip_alignment.py \
  --dataset_dir datasets/vstd_decoy_trainhard \
  --output_dir runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed321 \
  --encoder vit --clip_model ViT-B/16 --train_scope full \
  --epochs 6 --batch_size 16 --eval_batch_size 64 \
  --lr 1e-5 --weight_decay 1e-4 \
  --vit_path_adapter adaptive_decoy_aware \
  --seed 321 --device cuda
```

## Main Ablations

Use the same command as ADAR and replace `--vit_path_adapter` with:

```text
decoy_aware
slot_decoy_aware
adaptive_slot_decoy_aware
directional_decoy_aware
multihead_decoy_aware
layerwise_decoy_aware
multihop
multihop_zero
```

The paper-facing result summary is in:

```text
docs/BASELINE_RESULTS.md
```

Qualitative ADAR gate visualizations are generated with:

```bash
python scripts/visualize_vit_readout_gates.py \
  --run_dir runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111 \
  --dataset_dir datasets/vstd_decoy_trainhard \
  --output_dir docs/figures/adar_gate_examples \
  --num_samples 12 --min_path_length 48 \
  --select_by path_enrichment --correct_only --device cuda
```
