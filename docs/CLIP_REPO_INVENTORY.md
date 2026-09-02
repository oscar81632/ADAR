# Archived CLIP Workspace Inventory

This note records the structure of an earlier CLIP working directory used during
project exploration. The final ADAR experiments use the vendored code in this
repository.

## Core Code

| Path | Purpose |
|---|---|
| `clip/` | Modified CLIP Python package. |
| `clip/vim_wrapper.py` | Vim visual backbone wrapper and projection head. |
| `train_vim.py` | Main fine-tuning script for ViT/Vim, class loss, contrastive loss, timing, sweeps. |
| `train_vim_proj.py` | Earlier projection-head training script. |
| `lora_vim.py` | LoRA injection utilities for Vim mixer projections. |
| `adapter_vim.py` | Residual adapter utilities for selected Vim layers. |
| `unified_baseline.py` | PCA, prototype similarity, and representation geometry analysis. |
| `unified_zeroshot.py` | Zero-shot evaluation over CIFAR-10, STL-10, Caltech101, OxfordPets. |

## Data and Tests

| Path | Purpose |
|---|---|
| `data/` | Downloaded datasets and archives. |
| `test_data/` | Small test data from the original CLIP repo. |
| `tests/` | Existing CLIP tests. |

## Major Result Families

| Prefix | Meaning |
|---|---|
| `baseline_vim_*` | Vim baseline fine-tuning runs. |
| `baseline_vit_*` | ViT baseline fine-tuning runs. |
| `clip_vim_*_baseline_outputs` | Representation analysis outputs for Vim. |
| `clip_vit_*_baseline_outputs` | Representation analysis outputs for ViT. |
| `ft_vim_B16_outputs_*` | Vim-B/16 full or standard fine-tuning outputs. |
| `ft_vim_B32_outputs_*` | Vim-B/32 related fine-tuning outputs. |
| `ft_vim_adapter_outputs_*` | Vim adapter fine-tuning outputs. |
| `ft_vim_lora_outputs_*` | Vim LoRA fine-tuning outputs. |
| `ft_vit_B16_outputs_*` | ViT-B/16 fine-tuning outputs. |
| `nft_vit_*` | Non-fine-tuned or baseline ViT evaluation outputs. |
| `random_*` | Random/grid hyperparameter search outputs. |
| `vim_zeroshot_outputs` | Vim zero-shot evaluation outputs. |
| `vit_zeroshot_outputs` | ViT zero-shot evaluation outputs. |

## Current Policy

- Do not add new experiment output folders to the archived workspace.
- Put new generated datasets in `datasets/`.
- Put new runs in `runs/`.
- Put new paper figures in `figures/`.
- Paper-facing scripts use vendored code in `external/`.

## Future Cleanup Option

After key results are backed up, we can optionally archive old runs into:

```text
archived_clip_outputs/
  baselines/
  finetune_vim/
  finetune_vit/
  lora_adapter/
  zeroshot/
  random_search/
```

For now, avoid moving them because it may break relative paths and make old logs
harder to trace.
