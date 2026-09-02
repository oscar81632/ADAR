# Thesis Outline

Working title:

**Visual State Tracking with CLIP: A Synthetic Benchmark and Adaptive
Decoy-Aware Readout for Long-Path Reasoning**

## Chapter 1: Introduction

Motivate the problem from the current vision-language model setting: CLIP-style
models are strong at image-text alignment, but it is less clear whether their
visual representations preserve the step-by-step state needed for long visual
tracking. The thesis studies this question with a controlled synthetic task
instead of only natural-image classification.

Main research questions:

1. Can CLIP visual encoders track a start-conditioned path and align the final
   visual state to text?
2. How do ResNet, ViT, and Vim visual encoders behave under long-path and decoy
   shift?
3. Can a lightweight architecture change improve ViT without changing the loss
   or task data?

## Chapter 2: Related Work

Cover CLIP and vision-language pretraining, ViT and patch-token representations,
state-space visual backbones such as Vim, synthetic visual reasoning datasets,
and token readout or attention pooling mechanisms.

## Chapter 3: VSTD Dataset

Introduce VSTD as a start-conditioned visual state tracking dataset. Define the
image construction, start marker, true path, decoy paths, endpoint classes, and
text prompts. Emphasize the train-easy/test-hard setting: models train on
shorter, cleaner paths and are evaluated on longer paths with more decoys.

This chapter should explain what VSTD measures:

- long visual dependency tracking
- start-conditioned state propagation
- resistance to visually plausible decoys
- text alignment of the final tracked state

## Chapter 4: Method

This chapter should not contain the vanilla Vim baseline results. Keep it for
the proposed model and the training/evaluation protocol.

### 4.1 CLIP-Based VSTD Formulation

Describe how an image is encoded by the CLIP visual encoder and compared with
class text prompts using the CLIP image-text similarity objective. This section
defines the task interface, not the baseline results.

### 4.2 Adaptive Decoy-Aware Readout

Present ADAR as the main architectural contribution. The key idea is to keep the
CLIP ViT backbone but replace the single global readout with a lightweight
module that reweights final patch tokens according to path-relevant and
decoy-suppression evidence.

Explain the two components as one coherent mainline:

1. Decoy-aware token scoring suppresses visually plausible but path-irrelevant
   distractor regions.
2. Adaptive routing mixes the original CLIP readout and the decoy-aware path
   readout, allowing the model to preserve useful pretrained ViT behavior while
   specializing for VSTD.

### 4.3 Experimental Protocol

Move this material into Chapter 6 when writing the final thesis if it becomes
too result-heavy. The protocol details can also be split: a short method-side
description here, and the full settings in Chapter 6.

Use the same protocol for all main comparisons:

- CLIP-pretrained initialization
- full visual fine-tuning
- 6 epochs
- batch size 16
- learning rate `1e-5`
- weight decay `1e-4`
- best-validation checkpoint selection
- final evaluation on the OOD VSTD test split

## Chapter 5: Implementation

Describe the repository, dataset generator, CLIP-VSTD package, training script,
and evaluation utilities. This can be shorter if the thesis format does not
require a separate implementation chapter.

## Chapter 6: Experiments and Results

Put the vanilla Vim baseline, CLIP encoder baselines, ADAR mainline, and 4.3
protocol details together here so the reader can compare them directly.

### 6.1 Experimental Setup

Describe the dataset split, text prompts, training budget, checkpoint selection,
and metrics. This absorbs the result-facing part of the old 4.3 material.

### 6.2 Baseline Comparison

Report the CLIP visual encoder baselines and vanilla Vim in one table:

| Method | Visual Encoder | Test Acc | L=48 Acc |
|---|---|---:|---:|
| CLIP RN50 full FT | RN50 | 0.5773 | 0.5592 |
| CLIP RN101 full FT | RN101 | 0.5767 | 0.5663 |
| Vim-CLIP full FT | Vim-base | 0.6413 | 0.5792 |
| CLIP ViT-B/32 full FT | ViT-B/32 | 0.7933 | 0.7347 |
| CLIP ViT-B/16 full FT | ViT-B/16 | 0.8063 | 0.7347 |

The point of this section is to show that ViT-B/16 is a strong baseline and that
the thesis is not merely beating weak ResNet or Vim baselines.

### 6.3 Main Results: ADAR

Place ADAR next to the strongest vanilla ViT baseline:

| Method | Test Mean | Test Std | L=48 Mean | L=48 Std |
|---|---:|---:|---:|---:|
| CLIP ViT-B/16 full FT | 0.7860 | 0.0455 | 0.7242 | 0.0506 |
| ADAR | 0.8846 | 0.0207 | 0.8478 | 0.0371 |

Main claim: under the same 6-epoch budget and best-validation checkpoint
selection, ADAR improves the three-seed mean by `+9.86` percentage points.

### 6.4 Path-Length Analysis

Use the length breakdown to show that ADAR helps most in the long-path regime:

| Method | L=24 | L=32 | L=40 | L=48 |
|---|---:|---:|---:|---:|
| CLIP ViT-B/16 full FT | 0.8431 | 0.8135 | 0.7566 | 0.7242 |
| ADAR | 0.9176 | 0.9023 | 0.8665 | 0.8478 |

### 6.5 Ablation Study

Put decoy-aware here, not in the main result table. It supports the ADAR story
without stealing the main result.

| Method | Test Mean | Test Std | L=48 Mean | L=48 Std |
|---|---:|---:|---:|---:|
| Decoy-aware readout | 0.8610 | 0.0110 | 0.8293 | 0.0156 |
| ADAR | 0.8846 | 0.0207 | 0.8478 | 0.0371 |

Interpretation: decoy suppression is already useful, but adaptive routing gives
the best overall mean by combining the original CLIP readout with the
task-aware readout.

### 6.6 Qualitative Analysis

Use ADAR gate heatmaps from `docs/figures/adar_gate_examples`. Show examples
where the model follows the true path while suppressing decoy paths, plus one
failure case.

## Chapter 7: Discussion

Discuss why VSTD is difficult, why ViT performs better than ResNet and Vim in
this setting, and why a readout-level architecture change is enough to improve
tracking without changing the loss or training data.

Important framing:

- Vim is included as a baseline, but the main contribution is not "Vim beats
  ViT."
- The stronger claim is that CLIP ViT contains useful patch-level evidence, but
  the default global readout is not ideal for start-conditioned long-path
  tracking.
- ADAR improves the way the model reads out path-relevant patch tokens.

## Chapter 8: Conclusion

Summarize VSTD, the baseline findings, and ADAR. End with limitations and future
work such as applying the readout to natural visual navigation, diagrams,
robotic path following, UI trajectory following, or video state tracking.
