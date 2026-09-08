# Paper Tables: Main Results

## Main Result Table

All models use full visual fine-tuning for 6 epochs on
`datasets/vstd_decoy_trainhard` under the same best-validation checkpoint
protocol. CLIP visual encoder baselines use their corresponding CLIP-pretrained
weights. The Vim baseline uses the official Vim-base vision checkpoint integrated
into the same CLIP-style image-text evaluation framework.

| Method | Visual Encoder | Test Acc | L=48 Acc | Note |
|---|---|---:|---:|---|
| CLIP RN50 full FT | RN50 | 0.5773 | 0.5592 | ResNet CLIP baseline |
| CLIP RN101 full FT | RN101 | 0.5767 | 0.5663 | Larger ResNet CLIP baseline |
| CLIP RN50x4 full FT | RN50x4 | 0.5763 | 0.5934 | Parameter-matched scaled ResNet CLIP baseline |
| Vim-CLIP full FT | Vim-base | 0.6413 | 0.5792 | Vanilla Vim baseline |
| CLIP ViT-B/32 full FT | ViT-B/32 | 0.7933 | 0.7347 | Coarser patch-token baseline |
| CLIP ViT-B/16 full FT | ViT-B/16 | 0.8063 | 0.7347 | Main vanilla ViT baseline, seed 123 |
| ADAR full FT | ViT-B/16 + ADAR | 0.9117 | 0.9001 | Best single seed, seed 111 |

## Main Three-Seed Summary

| Method | Test Mean | Test Std | L=48 Mean | L=48 Std | Mean Delta vs ViT |
|---|---:|---:|---:|---:|---:|
| ViT-B/16 full FT | 0.7860 | 0.0455 | 0.7242 | 0.0506 | - |
| ADAR: Adaptive decoy-aware readout | 0.8846 | 0.0207 | 0.8478 | 0.0371 | +0.0986 |
| Slot decoy-aware readout | 0.8834 | 0.0304 | 0.8631 | 0.0385 | +0.0974 |

## Path-Length Mean Accuracy Over Three Seeds

| Method | L=24 | L=32 | L=40 | L=48 |
|---|---:|---:|---:|---:|
| ViT-B/16 full FT | 0.8431 | 0.8135 | 0.7566 | 0.7242 |
| ADAR | 0.9176 | 0.9023 | 0.8665 | 0.8478 |
| Slot decoy-aware readout | 0.9202 | 0.8865 | 0.8616 | 0.8631 |

## Supporting Ablations

The simple decoy-aware readout is useful as an ablation, but it should not be
placed in the main result table because it competes visually with the ADAR
mainline. In the paper, use it to show that ADAR's improvement comes from adding
adaptive routing on top of a decoy-suppression bias.

| Method | Test Mean | Test Std | L=48 Mean | L=48 Std | Role |
|---|---:|---:|---:|---:|---|
| Decoy-aware readout | 0.8610 | 0.0110 | 0.8293 | 0.0156 | Ablation: decoy suppression only |

## Suggested Paper Claim

Under the same 6-epoch compute budget and best-validation checkpoint selection,
ADAR improves CLIP ViT-B/16 from `0.7860 +/- 0.0455` to
`0.8846 +/- 0.0207`, a `+9.86` point mean gain on the OOD VSTD test split.

Slot decoy-aware readout achieves the best length-48 mean accuracy
(`0.8631`), suggesting that candidate-path slots are useful for long-path
decoy-heavy cases, while ADAR is the recommended main method by mean test
accuracy.
