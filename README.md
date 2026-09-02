# ADAR Research Workspace

This workspace contains the paper-facing code, datasets, run logs, and analysis
for **ADAR: Adaptive Decoy-Aware Readout** on the Visual State Tracking Dataset
(VSTD). It is intended to be self-contained for reproducing the thesis
experiments.

## Directory Layout

```text
ADAR_Research/
  configs/        Experiment configs and dataset generation configs.
  datasets/       Generated datasets such as VSTD. Do not commit large images.
  checkpoints/    Local model checkpoints needed by non-CLIP baselines.
  docs/           Notes, repo inventory, paper planning, and experiment logs.
  external/       Vendored CLIP/Vim code needed by the experiments.
  figures/        Paper-ready figures copied or generated from runs.
  notebooks/      Exploratory notebooks.
  runs/           New training/evaluation outputs.
  scripts/        CLI entrypoints for generation, training, and analysis.
  src/
    vstd/         Visual State Tracking Dataset code.
    training/     Training adapters/wrappers for the clean experiments.
    analysis/     Metrics, plots, and representation geometry analysis.
```

## Self-Contained Runtime

The paper-facing modified CLIP package is vendored inside this repo at
`external/clip_vstd/`. The Vim model definitions needed by the Vim baseline are
vendored at `external/vim_vstd/`.

Training scripts import these vendored packages automatically through
`src/adar_clip.py`, so reproducing the experiments does not require an external
CLIP source tree.

## Recommended Workflow

1. Generate datasets into `datasets/vstd_*`.
2. Train/evaluate into `runs/<experiment_name>`.
3. Save final plots into `figures/<experiment_name>`.
4. Document conclusions in `docs/`.

Run commands from the repo root with a Python environment that has PyTorch,
Pillow, PyYAML, NumPy, and the CLIP runtime dependencies installed:

```bash
python scripts/<script>.py
```

## Reproducing the Main Experiments

The thesis mainline uses the `vstd_decoy_trainhard` split: training samples are
shorter and less cluttered, while validation and test samples contain longer
paths and heavier decoys. From a fresh checkout, run the following commands from
the repository root.

### 1. Generate VSTD

```bash
python scripts/generate_vstd.py \
  --config configs/vstd_decoy_trainhard.yaml \
  --output_root datasets \
  --overwrite
```

This creates:

```text
datasets/vstd_decoy_trainhard/
  train/
  val/
  test/
  classes.json
  metadata.jsonl
  dataset_config.yaml
```

### 2. Run the Main ViT Baseline

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

### 4. Analyze by Path Geometry

```bash
python scripts/analyze_vstd_run.py \
  --run_dir runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123 \
  --dataset_dir datasets/vstd_decoy_trainhard \
  --split test \
  --output runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123/test_geometry_analysis.json \
  --device cuda
```

### 5. Generate Gate Heatmaps

```bash
python scripts/visualize_vit_readout_gates.py \
  --run_dir runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123 \
  --dataset_dir datasets/vstd_decoy_trainhard \
  --output_dir docs/figures/adar_gate_examples \
  --num_samples 12 --min_path_length 48 \
  --select_by path_enrichment --correct_only \
  --device cuda
```

For the full three-seed protocol, CLIP ResNet baselines, Vim baseline, and
ablation variants, see `docs/reproducibility/MAIN_EXPERIMENTS.md`. The
paper-ready numbers are summarized in `docs/paper_tables/MAIN_RESULTS.md`.

## Local Checkpoints

The Vim baseline checkpoint used in the thesis is expected at:

```text
checkpoints/vim-base/vim_b_midclstok_81p9acc.pth
```

It is not tracked in this repository because the file is large. Download the
official Vim-base checkpoint from the upstream Hugging Face model card:

- Vim-base checkpoint page: https://huggingface.co/hustvl/Vim-base-midclstok
- Original Vim repository: https://github.com/hustvl/Vim

Example download command:

```bash
mkdir -p checkpoints/vim-base
curl -L \
  -o checkpoints/vim-base/vim_b_midclstok_81p9acc.pth \
  https://huggingface.co/hustvl/Vim-base-midclstok/resolve/main/vim_b_midclstok_81p9acc.pth
```

CLIP model weights are still handled by the standard CLIP download/cache
mechanism.

## Paper Mainline

The current thesis mainline is **ADAR: Adaptive Decoy-Aware Readout** for CLIP
ViT-B/16 on VSTD.

Start from:

- `docs/reproducibility/MAIN_EXPERIMENTS.md` for commands.
- `docs/paper_tables/MAIN_RESULTS.md` for paper-ready result tables.
- `docs/BASELINE_RESULTS.md` for full experiment history.
- `docs/figures/adar_gate_examples/` for qualitative gate heatmaps.
