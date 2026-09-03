# Research Notes

This note summarizes the main research decisions and implementation state behind
the ADAR/VSTD project.

## Final Research Direction

The project started from an attempt to combine Vim with CLIP by replacing the
CLIP visual encoder. That direction produced useful baselines and implementation
experience, but the final contribution shifted toward a clearer research
question:

> Can CLIP-style vision-language models perform long-sequence visual state
> tracking under decoy-heavy distribution shifts, and can a better ViT readout
> improve this behavior?

The final project should therefore be framed around:

1. **VSTD:** a synthetic benchmark for visual state tracking.
2. **ADAR:** an adaptive decoy-aware readout for CLIP ViT-B/16.
3. **Evaluation:** controlled baselines, ablations, path-length breakdowns, and
   gate visualizations.

Vim is kept as a baseline and background motivation, not as the main proposed
method.

## VSTD

VSTD stands for Visual State Tracking Dataset. Each image contains a grid, a
start marker, arrows, endpoint candidates, and decoy paths. The model must infer
the correct endpoint by following the path from the start marker.

The main split is:

```text
configs/vstd_decoy_trainhard.yaml
```

Main protocol:

```text
train: shorter paths and fewer decoys
validation/test: longer paths and heavier decoys
image size: 224 x 224
grid size: 14 x 14
patch size for CLIP ViT-B/16: 16 x 16
```

The benchmark is intended to test:

- long-path state tracking
- robustness to decoy paths
- train-test difficulty shift
- whether image-text alignment can handle procedural visual reasoning

## ADAR

ADAR stands for Adaptive Decoy-Aware Readout. It modifies the visual readout of
CLIP ViT-B/16 while preserving the CLIP-style image-text objective.

The core idea is to avoid relying only on the class token. ADAR computes:

- a gate score for each patch
- a selected-path summary, usually denoted `Z_keep`
- a complementary residual/context summary, usually denoted `Z_supp`
- an image-level routing coefficient, usually denoted `alpha`
- a fused readout through a lightweight MLP `F`

The final feature keeps the original class token as a residual anchor and adds
the routed patch evidence before CLIP projection.

Important implementation location:

```text
external/clip_vstd/clip/model.py
scripts/train_vstd_clip_alignment.py
```

The most important conceptual clarification:

`Z_supp` should not be described as simply "throwing away decoys." It is better
described as complementary residual evidence. The model compares selected path
evidence against remaining visual context, which is useful when decoys make the
correct path ambiguous.

## Main Experimental Protocol

All main full fine-tuning results use:

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

The main commands are documented in:

```text
docs/reproducibility/MAIN_EXPERIMENTS.md
```

The main result tables are documented in:

```text
docs/paper_tables/MAIN_RESULTS.md
docs/BASELINE_RESULTS.md
```

## Main Result Summary

Under the shared 6-epoch protocol:

```text
CLIP ViT-B/16 full FT mean test acc: 0.7860
ADAR full FT mean test acc:          0.8846
Mean improvement:                    +0.0986
```

ADAR also improves long-path accuracy, especially for length-48 samples.

Baselines include:

- CLIP RN50
- CLIP RN101
- CLIP RN50x4
- CLIP ViT-B/32
- CLIP ViT-B/16
- Vim-CLIP full fine-tuning
- ADAR ablations

## Ablation Interpretation

The important ablation lesson is that both selected and complementary evidence
matter.

Use consistent notation:

```text
Z_keep: selected path-biased evidence
Z_supp: complementary residual evidence
alpha: image-level adaptive routing coefficient
F: fusion MLP
```

The story should emphasize:

- `Z_keep` helps focus on likely path evidence.
- `Z_supp` preserves context needed to compare against decoys.
- `alpha` adjusts how strongly the patch summaries should affect each image.
- ADAR improves most when paths are long and decoy-heavy.

Avoid claiming the gate heatmap perfectly identifies the correct path. The
qualitative examples are better used as evidence that the readout learns
structured patch preferences, not as proof of explicit symbolic path tracing.

## Vim-CLIP Exploration

The original Vim-CLIP work is archived in:

```text
docs/history/vim_clip_exploration.md
docs/history/vim_clip_source_snapshot/
docs/history/vim_clip_results/
```

This history explains why the thesis moved away from the simple story of
"replace ViT with Vim." The useful retained value is:

- Vim wrapper implementation
- projection/adapters/LoRA experiments
- CIFAR/OxfordPets/VSTD experiment records
- evidence that the stronger final contribution is ADAR/VSTD

## Checkpoints

The Vim checkpoint is intentionally not committed because it is large.

Download source:

```text
https://huggingface.co/hustvl/Vim-base-midclstok
```

Expected path:

```text
checkpoints/vim-base/vim_b_midclstok_81p9acc.pth
```

CLIP model weights are handled by the CLIP download/cache mechanism.

## External Dataset Explorations

The project also explored:

- Pathfinder-style synthetic data
- LRA Pathfinder
- GeoPathfinder
- CIFAR-10
- STL-10
- SigLIP / EVA-CLIP style baselines

These were exploratory and should not replace the main project narrative unless
new results are intentionally promoted into the paper.
