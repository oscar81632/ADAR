# CLIP-VSTD Vendored Package

This directory contains the modified CLIP package used by the VSTD thesis
experiments. It is vendored here so the main ADAR research repo can reproduce
the paper experiments without relying on an external CLIP source tree.

Training scripts load it automatically through `src/adar_clip.py`:

```bash
python scripts/train_vstd_clip_alignment.py ...
```

Important VSTD modifications:

- `clip/model.py`
  - Adds `ViTPathReadout`.
  - Adds ViT readout modes such as:
    - `decoy_aware`
    - `adaptive_decoy_aware`
    - `slot_decoy_aware`
    - `adaptive_slot_decoy_aware`
    - ablation variants including multihop and layerwise variants.
  - Extends `VisionTransformer` and `CLIP.build_model` to accept
    `vit_path_adapter` options.
- `clip/clip.py`
  - Extends `clip.load(...)` with `vit_path_adapter` arguments.
- `clip/vim_wrapper.py`
  - Preserved for Vim baseline comparisons.
  - Uses the minimal vendored Vim model definitions in `external/vim_vstd/`.
  - ADAR does not require Vim.

The current paper main method is:

```text
ADAR: Adaptive Decoy-Aware Readout
```

It is selected with:

```bash
--encoder vit --clip_model ViT-B/16 --vit_path_adapter adaptive_decoy_aware
```
