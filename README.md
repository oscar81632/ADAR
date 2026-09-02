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

## Local Checkpoints

The Vim baseline checkpoint used in the thesis is expected at:

```text
checkpoints/vim-base/vim_b_midclstok_81p9acc.pth
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
