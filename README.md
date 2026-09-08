# ADAR: Adaptive Decoy-Aware Readout for CLIP

This repository contains the research code for **ADAR** and the **Visual State
Tracking Dataset (VSTD)**. The project studies whether CLIP-style
vision-language models can follow long visual paths under heavy visual decoys,
then improves the ViT readout stage without changing the training loss or adding
path-level supervision.

## Highlights

- **Task:** VSTD asks a model to follow a rendered path and classify the endpoint
  from text prompts.
- **Method:** ADAR adds a decoy-aware readout on top of CLIP ViT-B/16. It keeps
  the CLIP-style image-text objective unchanged.
- **Main result:** Under the same 6-epoch visual fine-tuning protocol, ADAR
  improves CLIP ViT-B/16 from **78.60%** to **88.46%** mean test accuracy over
  three seeds on the OOD VSTD split.
- **Long-path result:** On the longest path group, ADAR improves mean accuracy
  from **72.42%** to **84.78%**.
- **Reproducibility:** The repo vendors the modified CLIP/Vim code needed by the
  experiments, so an external CLIP source tree is not required.

## Repository Layout

```text
configs/        VSTD generation configs and experiment settings.
docs/           Main results, reproducibility commands, and research notes.
external/       Repo-local CLIP and Vim code used by the experiments.
scripts/        Dataset generation, training, evaluation, and visualization CLIs.
src/            VSTD implementation and repo-local CLIP loader.
```

Generated datasets, run outputs, checkpoints, and external downloaded datasets
are intentionally excluded from Git. They can be regenerated with the commands
below.

Some metadata/config fields retain the earlier internal key name
`distractor_ratio`; in the paper and README, these are referred to as visual
decoys.

## Install

Python 3.10+ is recommended. For CUDA training, install a PyTorch build that
matches your driver and CUDA runtime.

```bash
git clone https://github.com/oscar81632/ADAR.git
cd ADAR
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first CLIP run downloads OpenAI CLIP weights through the standard CLIP cache
mechanism.

## Quick Reproduction

All commands should be run from the repository root.

### 1. Generate the Main VSTD Split

```bash
python scripts/generate_vstd.py \
  --config configs/vstd_decoy_trainhard.yaml \
  --output_root datasets \
  --overwrite
```

This creates `datasets/vstd_decoy_trainhard/` with train, validation, and test
images plus metadata.

### 2. Run the ViT-B/16 Baseline

```bash
python scripts/train_vstd_clip_alignment.py \
  --dataset_dir datasets/vstd_decoy_trainhard \
  --output_dir runs/vit_b16_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123 \
  --encoder vit --clip_model ViT-B/16 --train_scope full \
  --epochs 6 --batch_size 16 --eval_batch_size 64 \
  --lr 1e-5 --weight_decay 1e-4 \
  --seed 123 --device cuda
```

### 3. Run ADAR

```bash
python scripts/train_vstd_clip_alignment.py \
  --dataset_dir datasets/vstd_decoy_trainhard \
  --output_dir runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123 \
  --encoder vit --clip_model ViT-B/16 --train_scope full \
  --epochs 6 --batch_size 16 --eval_batch_size 64 \
  --lr 1e-5 --weight_decay 1e-4 \
  --vit_path_adapter adaptive_decoy_aware \
  --seed 123 --device cuda
```

### 4. Analyze Long-Path Behavior

```bash
python scripts/analyze_vstd_run.py \
  --run_dir runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123 \
  --dataset_dir datasets/vstd_decoy_trainhard \
  --split test \
  --output runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123/test_geometry_analysis.json \
  --device cuda
```

### 5. Generate ADAR Gate Heatmaps

```bash
python scripts/visualize_vit_readout_gates.py \
  --run_dir runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123 \
  --dataset_dir datasets/vstd_decoy_trainhard \
  --output_dir docs/figures/adar_gate_examples \
  --num_samples 12 --min_path_length 48 \
  --select_by path_enrichment --correct_only \
  --device cuda
```

## Main Protocol

The paper-facing protocol uses:

```text
dataset: datasets/vstd_decoy_trainhard
epochs: 6
batch_size: 16
eval_batch_size: 64
learning rate: 1e-5
weight decay: 1e-4
checkpoint selection: best validation accuracy
seeds: 111, 123, 321
```

For the full three-seed protocol, CLIP ResNet baselines, Vim baseline,
ablations, and exact command list, see
[`docs/reproducibility/MAIN_EXPERIMENTS.md`](docs/reproducibility/MAIN_EXPERIMENTS.md).

Paper-ready tables are summarized in
[`docs/paper_tables/MAIN_RESULTS.md`](docs/paper_tables/MAIN_RESULTS.md).

## Vim Baseline Checkpoint

The Vim baseline uses the official Vim-base ImageNet-1K checkpoint. It is not
tracked because it is a large model file.

Expected path:

```text
checkpoints/vim-base/vim_b_midclstok_81p9acc.pth
```

Download:

```bash
mkdir -p checkpoints/vim-base
curl -L \
  -o checkpoints/vim-base/vim_b_midclstok_81p9acc.pth \
  https://huggingface.co/hustvl/Vim-base-midclstok/resolve/main/vim_b_midclstok_81p9acc.pth
```

## Notes for Reviewers

- The main contribution is not a new dataset alone. VSTD is used to expose a
  long-path, decoy-heavy failure mode in CLIP-style visual readout.
- ADAR changes the readout architecture while keeping the CLIP-style image-text
  alignment loss unchanged.
- The repo includes exploratory notes for transparency, but the recommended
  entry points are the README, `docs/reproducibility/MAIN_EXPERIMENTS.md`, and
  `docs/paper_tables/MAIN_RESULTS.md`.
