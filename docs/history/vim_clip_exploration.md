# Vim+CLIP Exploration Archive

This folder preserves the compact research history from the earlier `CLIP/`
workspace. The original workspace contains datasets, checkpoints, generated run
directories, and many random-search outputs, so it should not be uploaded to
GitHub as-is.

## Why This Archive Exists

The initial thesis direction explored whether replacing CLIP's ViT visual
encoder with Vim could form a strong research contribution. The experiments
covered full fine-tuning, projection-head probing, MLP heads, adapters, LoRA,
and hyperparameter search on standard classification datasets such as CIFAR-10,
Oxford-IIIT Pets, STL-10, and Caltech-101.

These experiments are useful because they document the negative and transitional
evidence behind the final thesis direction. They showed that a simple
vision-encoder replacement story was not enough by itself, which motivated the
shift toward a task-driven benchmark and readout-level architectural design:
VSTD and ADAR.

## Archived Source Snapshot

Selected modified source files are preserved in:

```text
docs/history/vim_clip_source_snapshot/
```

Included files:

```text
vim_wrapper.py
train_vim.py
train_vim_proj.py
adapter_vim.py
lora_vim.py
```

These files capture the main local changes used to connect Vim-style visual
encoders, projection heads, adapters, and LoRA-style modules to the CLIP
training/evaluation workflow.

## Archived Results

The full result index is:

```text
docs/history/vim_clip_results/vim_clip_result_index.csv
```

It indexes all discovered `*_final_test_result.json` files from the old `CLIP/`
workspace and records:

```text
experiment_dir
dataset
encoder
method
train_per_class
epochs_config
epoch_result
acc
mean_gap
split
result_path
args_path
log_path
```

Representative results are copied into:

```text
docs/history/vim_clip_results/selected_final_results/
docs/history/vim_clip_results/selected_args/
docs/history/vim_clip_results/selected_logs/
```

Human-readable selected result table:

```text
docs/history/vim_clip_results/selected_result_summary.md
```

## What Was Intentionally Excluded

The archive intentionally excludes:

```text
checkpoints/
*.pth
datasets/
CLIP/data/
__pycache__/
full random-search folders
duplicated run directories without a written conclusion
```

The goal is to preserve the research trail, not to reproduce the old 73GB
workspace verbatim.

## How To Use This Archive

Use this archive when explaining the research transition:

1. The project started from a Vim-as-CLIP-encoder replacement direction.
2. Standard classification datasets showed that the replacement was feasible,
   but not a strong enough standalone contribution.
3. Adapter, LoRA, and projection-head variants were explored, but the story
   remained mostly engineering-oriented.
4. This motivated the final thesis direction: design a task that stresses visual
   state tracking and propose a readout mechanism, ADAR, for that setting.

In a public GitHub repo, this archive should be treated as historical context.
The main reproducible research contribution remains the cleaned ADAR/VSTD code,
configs, and paper-facing experiment summaries.
