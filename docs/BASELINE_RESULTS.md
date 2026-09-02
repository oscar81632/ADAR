# VSTD Easy Frozen-Feature Baseline Results

Dataset: `datasets/vstd_easy`

Setup:

- Frozen visual encoder.
- Train only a linear classifier on image features.
- 100 epochs.
- RTX 4090.
- 8 classes.

## Aggregate Results

| Encoder | Output Dir | Test Accuracy | Test Loss |
|---|---|---:|---:|
| ViT-B/16 CLIP | `runs/vit_b16_clip_linear_vstd_easy_100e` | 0.5285 | 1.3779 |
| Vim-base CLIP wrapper | `runs/vim_clip_linear_vstd_easy_100e` | 0.4195 | 1.7093 |

## Accuracy by Path Length

| Encoder | L=4 | L=8 | L=12 |
|---|---:|---:|---:|
| ViT-B/16 CLIP | 0.5115 | 0.5498 | 0.5260 |
| Vim-base CLIP wrapper | 0.4063 | 0.4250 | 0.4279 |

## Accuracy by Distractor Ratio

| Encoder | D=0.0 | D=0.15 | D=0.25 |
|---|---:|---:|---:|
| ViT-B/16 CLIP | 1.0000 | 0.3191 | 0.3064 |
| Vim-base CLIP wrapper | 0.7834 | 0.2809 | 0.2254 |

## Initial Interpretation

Frozen ViT-CLIP features are stronger than frozen Vim features on `vstd_easy`.
However, both models show a sharp drop once distractors are present. This is a
useful benchmark signal: solving the dataset is not equivalent to recognizing
the final color/shape object. The model must identify the correct path and
ignore distractor arrows/shapes.

The next meaningful comparison should not stop at frozen features. It should test:

- Vim projection-head fine-tuning.
- Vim adapter fine-tuning.
- ViT fine-tuning under the same dataset.
- Accuracy by path length, distractor ratio, and long-dependency score.

The current result says:

> Frozen pretrained representations alone do not solve visual state tracking
> under distractors.

This supports moving to adaptation/fine-tuning experiments.

## Projection-Head CLIP Alignment Probing

Dataset: `datasets/vstd_easy`

Setup:

- Use class text prompts from `classes.json`.
- Freeze text encoder.
- Freeze visual backbone.
- Train only the visual projection side:
  - ViT: `visual.ln_post + visual.proj`
  - Vim: `visual.ln_post + visual.proj`
- Loss:

```text
CE(logit_scale * normalize(image_features) @ normalize(text_features).T, label)
```

Prompts:

```text
a path ending at a red circle
a path ending at a red square
a path ending at a blue circle
a path ending at a blue square
a path ending at a green circle
a path ending at a green square
a path ending at a yellow circle
a path ending at a yellow square
```

### Aggregate Results

| Encoder | Output Dir | Epochs | Trainable Params | Test Accuracy | Test Loss |
|---|---|---:|---:|---:|---:|
| ViT-B/16 CLIP | `runs/vit_b16_clip_alignment_vstd_easy_10e_b32` | 10 | 394,752 | 0.7755 | 0.6051 |
| Vim-base CLIP wrapper | `runs/vim_clip_alignment_vstd_easy_10e_b32` | 10 | 1,969,920 | 0.4810 | 1.3540 |

### Accuracy by Distractor Ratio

| Encoder | D=0.0 | D=0.15 | D=0.25 |
|---|---:|---:|---:|
| ViT-B/16 CLIP | 0.9952 | 0.7353 | 0.6156 |
| Vim-base CLIP wrapper | 0.9411 | 0.2897 | 0.2514 |

### Interpretation

Projection-head probing is a more CLIP-relevant baseline than frozen linear
probing because it uses the text encoder as fixed class prototypes.

Current observation:

- ViT benefits strongly from CLIP-style projection probing.
- Vim improves over its frozen-feature linear probe, but still struggles under
  distractors.
- This suggests the randomly initialized Vim projection head is not enough to
  recover strong path-following alignment by itself.

Next step:

- Train Vim adapters or LoRA modules, not only the projection head.
- Compare with ViT fine-tuning under the same CLIP-style class text loss.

## Vim Full Visual Fine-Tuning

Dataset: `datasets/vstd_easy`

Setup:

- Use fixed class text prompt prototypes.
- Freeze text encoder.
- Train the full Vim visual side, without adapter or LoRA.
- Loss is the same CLIP-style class text alignment objective.

Run:

```text
runs/vim_clip_fullft_vstd_easy_10e_b16_lr1e-5
```

| Encoder | Train Scope | Epochs | Trainable Params | Test Accuracy | Test Loss |
|---|---|---:|---:|---:|---:|
| Vim-base CLIP wrapper | full visual side | 10 | 98,799,360 | 1.0000 | 0.000001 |
| ViT-B/16 CLIP | full visual side | 10 | 86,192,640 | 1.0000 | 0.000002 |

Accuracy by distractor ratio:

| D=0.0 | D=0.15 | D=0.25 |
|---:|---:|---:|
| 1.0000 | 1.0000 | 1.0000 |

Interpretation:

`vstd_easy` is fully learnable when the entire visual side is fine-tuned. Both
Vim-base and ViT-B/16 reach perfect test accuracy. This is a useful sanity
baseline, but it also means the easy split is too simple for the main research
comparison. The next comparison should use:

- `vstd_medium` / `vstd_hard`
- parameter-efficient Vim adapters or LoRA
- matched ViT fine-tuning

The key research question should move from "can Vim learn the task?" to:

> How much adaptation is required, and does Vim remain robust as path length,
> spatial dependency, and distractor density increase?

## VSTD Medium Results

Dataset: `datasets/vstd_medium`

Compared with `vstd_easy`, this split increases path length and distractor
density:

```text
path_lengths = [12, 16, 20, 24]
distractor_ratios = [0.15, 0.25, 0.4]
train / val / test = 10000 / 2000 / 3000
```

### Projection-Head CLIP Alignment Probing

Setup:

- Use fixed class text prompt prototypes.
- Freeze text encoder.
- Freeze visual backbone.
- Train only `visual.ln_post + visual.proj`.
- 10 epochs, batch size 32.

| Encoder | Output Dir | Test Accuracy | Test Loss |
|---|---|---:|---:|
| ViT-B/16 CLIP | `runs/vit_b16_clip_alignment_vstd_medium_10e_b32` | 0.7313 | 0.7139 |
| Vim-base CLIP wrapper | `runs/vim_clip_alignment_vstd_medium_10e_b32` | 0.2370 | 1.9047 |

Accuracy by distractor ratio:

| Encoder | D=0.15 | D=0.25 | D=0.4 |
|---|---:|---:|---:|
| ViT-B/16 CLIP | 0.8260 | 0.7362 | 0.6284 |
| Vim-base CLIP wrapper | 0.2875 | 0.2294 | 0.1918 |

### Full Visual Fine-Tuning Sanity Check

Setup:

- Freeze text encoder.
- Train the full visual side.
- 3 epochs, batch size 16, learning rate `1e-5`.

| Encoder | Output Dir | Test Accuracy | Test Loss |
|---|---|---:|---:|
| Vim-base CLIP wrapper | `runs/vim_clip_fullft_vstd_medium_3e_b16_lr1e-5` | 1.0000 | 0.000002 |
| ViT-B/16 CLIP | `runs/vit_b16_clip_fullft_vstd_medium_3e_b16_lr1e-5` | 1.0000 | 0.000005 |

Accuracy by distractor ratio:

| Encoder | D=0.15 | D=0.25 | D=0.4 |
|---|---:|---:|---:|
| Vim-base CLIP wrapper | 1.0000 | 1.0000 | 1.0000 |
| ViT-B/16 CLIP | 1.0000 | 1.0000 | 1.0000 |

Interpretation:

- `vstd_medium` is difficult for Vim projection-head-only alignment.
- The same split is fully learnable when either full visual side is fine-tuned.
- This creates a useful gap for the thesis: the task is learnable, but the
  amount and location of adaptation matter.

Next comparison:

- Vim adapters / LoRA on the decoy-OOD setting below.

## Decoy-OOD Path Following

The first OOD attempt only increased path length and gray distractor density.
ViT full fine-tuning still reached 1.0000 test accuracy, which exposed an
important shortcut: the true endpoint was the only bright colored object.

The generator was updated so coherent decoy paths can be added. In
`vstd_decoy_ood`, the test split contains black decoy paths with equally bright
endpoint objects. This forces the model to identify the purple start cell and
follow the connected path, instead of selecting the brightest object or any
black-arrow endpoint.

Dataset: `datasets/vstd_decoy_ood`

```text
train:
  path_lengths = [8, 12, 16, 20]
  distractor_ratios = [0.15, 0.25]
  decoy paths = 0

val:
  path_lengths = [12, 16, 20, 24]
  distractor_ratios = [0.25, 0.4]
  decoy paths = 0

test:
  path_lengths = [24, 32, 40, 48]
  distractor_ratios = [0.25, 0.4]
  black decoy paths = 2
```

Setup:

- Freeze text encoder.
- Train the full visual side on the train split.
- Evaluate on decoy-OOD test.
- 3 epochs, batch size 16, learning rate `1e-5`.

| Encoder | Output Dir | Val Accuracy | Test Accuracy | Test Loss |
|---|---|---:|---:|---:|
| ViT-B/16 CLIP | `runs/vit_b16_clip_fullft_vstd_decoy_ood_3e_b16_lr1e-5` | 1.0000 | 0.3697 | 5.5535 |
| Vim-base CLIP wrapper | `runs/vim_clip_fullft_vstd_decoy_ood_3e_b16_lr1e-5` | 1.0000 | 0.4390 | 4.7690 |

Accuracy by path length:

| Encoder | L=24 | L=32 | L=40 | L=48 |
|---|---:|---:|---:|---:|
| ViT-B/16 CLIP | 0.3863 | 0.3669 | 0.3794 | 0.3453 |
| Vim-base CLIP wrapper | 0.4428 | 0.4457 | 0.4519 | 0.4144 |

Accuracy by distractor ratio:

| Encoder | D=0.25 | D=0.4 |
|---|---:|---:|
| ViT-B/16 CLIP | 0.3629 | 0.3766 |
| Vim-base CLIP wrapper | 0.4428 | 0.4351 |

Geometry breakdown:

| Encoder | mostly right/down | mostly left/up | mixed |
|---|---:|---:|---:|
| ViT-B/16 CLIP | 0.3963 | 0.3403 | 0.3712 |
| Vim-base CLIP wrapper | 0.4134 | 0.4333 | 0.4509 |

Turn-count breakdown:

| Encoder | turns 0-10 | turns 11-16 | turns 17-24 | turns 25+ |
|---|---:|---:|---:|---:|
| ViT-B/16 CLIP | 0.3235 | 0.3912 | 0.3710 | 0.3569 |
| Vim-base CLIP wrapper | 0.5588 | 0.4444 | 0.4398 | 0.4312 |

Interpretation:

- Both models fit train/val but fail on black-decoy OOD test.
- The failure mode is coherent false paths with equally salient endpoints, not
  ordinary gray distractors.
- Vim performs better than ViT under this stress setting, but both leave large
  headroom for a path-aware adapter.
- ViT shows a small scan-alignment effect: paths with mostly right/down steps
  are easier than mostly left/up steps.
- Vim's advantage is not limited to scan-aligned paths; it is strongest on mixed
  paths and remains more stable as turn count increases.

Next comparison:

- Train with a small amount of decoy exposure, then test on harder decoys.
- Add a Vim path-aware adapter aimed at start-conditioned path propagation and
  decoy suppression.

## Decoy-Trainhard Generalization

`vstd_decoy_trainhard` is the next benchmark for adapter development. Unlike
`vstd_decoy_ood`, the train split includes a small amount of decoy supervision,
while the test split contains longer paths and stronger decoys.

Dataset: `datasets/vstd_decoy_trainhard`

```text
train:
  path_lengths = [8, 12, 16, 20]
  distractor_ratios = [0.15, 0.25]
  black decoy paths = 1
  decoy length = 4-8

val:
  path_lengths = [12, 16, 20, 24]
  distractor_ratios = [0.25, 0.4]
  black decoy paths = 1
  decoy length = 6-10

test:
  path_lengths = [24, 32, 40, 48]
  distractor_ratios = [0.25, 0.4]
  black decoy paths = 2
  decoy length = 8-14
```

### Full and Partial Fine-Tuning

| Encoder | Train Scope | Output Dir | Epochs | Trainable Params | Val Accuracy | Test Accuracy | Test Loss |
|---|---|---|---:|---:|---:|---:|---:|
| ViT-B/16 CLIP | full visual side | `runs/vit_b16_clip_fullft_vstd_decoy_trainhard_3e_b16_lr1e-5` | 3 | 86,192,640 | 0.9010 | 0.6470 | 1.0232 |
| ViT-B/16 CLIP | full visual side | `runs/vit_b16_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 6 | 86,192,640 | 0.9760 | 0.8063 | 0.6963 |
| Vim-base CLIP wrapper | full visual side | `runs/vim_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 6 | 98,799,360 | 0.9045 | 0.6413 | 1.9311 |
| Vim-base CLIP wrapper | last 4 layers + projection | `runs/vim_clip_last4_vstd_decoy_trainhard_6e_b16_lr1e-4` | 6 | 17,985,024 | 0.6760 | 0.4747 | 1.6800 |

### CLIP Visual Encoder Baselines

These runs compare publicly available CLIP-pretrained visual encoders under the
same VSTD protocol: full visual fine-tuning, 6 epochs, batch size 16, learning
rate `1e-5`, weight decay `1e-4`, and best-validation checkpoint selection.

| CLIP Visual Encoder | Output Dir | Best Val Accuracy | Test Accuracy | L=48 Accuracy |
|---|---|---:|---:|---:|
| RN50 | `runs/rn50_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 0.8875 | 0.5773 | 0.5592 |
| RN101 | `runs/rn101_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 0.8910 | 0.5767 | 0.5663 |
| RN50x4 | `runs/rn50x4_clip_fullft_vstd_decoy_trainhard_6e_b8_lr1e-5_run2` | 0.8850 | 0.5763 | 0.5934 |
| ViT-B/32 | `runs/vit_b32_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 0.9585 | 0.7933 | 0.7347 |
| ViT-B/16 | `runs/vit_b16_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 0.9760 | 0.8063 | 0.7347 |
| ViT-B/16 + Adaptive Decoy-Aware Readout | `runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111` | 0.9895 | 0.9117 | 0.9001 |

Parameter scale for the CLIP visual encoders:

| Encoder | Visual Params | Total CLIP Params | Status |
|---|---:|---:|---|
| RN50 | 38.32M | 102.01M | Completed |
| RN101 | 56.26M | 119.69M | Completed |
| RN50x4 | 87.14M | 178.30M | Completed; batch size 8 due to activation memory |
| ViT-B/32 | 87.85M | 151.28M | Completed |
| ViT-B/16 | 86.19M | 149.62M | Completed |
| Vim-base CLIP wrapper | 98.80M | 162.23M | Completed |
| ViT-B/16 + ADAR | 96.16M | 159.59M | Completed |

RN50x4 is the most useful additional CLIP ResNet baseline because its visual
parameter count is nearly the same as ViT-B/16. This makes it a stronger test
of whether VSTD favors ViT-style patch tokens rather than simply larger visual
models.

Interpretation:

- CLIP ResNet encoders perform much worse than CLIP ViT encoders on the OOD
  VSTD test split. RN50, RN101, and RN50x4 all remain near 0.58 test
  accuracy. RN50x4 is especially informative because its visual parameter
  count is close to ViT-B/16, suggesting that the gap is not explained by
  parameter count alone.
- ViT-B/32 is much stronger than ResNets, and ViT-B/16 is slightly stronger
  than ViT-B/32. This supports using ViT-B/16 as the main CLIP baseline: finer
  patch-token representations appear important for path and decoy reasoning.
- Adaptive Decoy-Aware Readout improves the strong ViT-B/16 baseline rather
  than only beating weak CNN baselines.

The three-seed ViT-B/16 baseline is reported in the ViT architecture section
below; its mean test accuracy is `0.7860`, so ADAR improves the fair three-seed
ViT mean by `+0.0986`.

Accuracy by path length:

| Encoder / Scope | L=24 | L=32 | L=40 | L=48 |
|---|---:|---:|---:|---:|
| ViT full 3e | 0.7152 | 0.6799 | 0.5981 | 0.5877 |
| ViT full 6e | 0.8830 | 0.8271 | 0.7730 | 0.7347 |
| Vim full | 0.7009 | 0.6594 | 0.6195 | 0.5792 |
| Vim last4 | 0.5007 | 0.4904 | 0.4459 | 0.4593 |

Geometry breakdown:

| Encoder / Scope | mostly right/down | mostly left/up | mixed |
|---|---:|---:|---:|
| ViT full | 0.6033 | 0.6222 | 0.6738 |
| Vim full | 0.6900 | 0.5827 | 0.6453 |
| Vim last4 | 0.3754 | 0.4765 | 0.5128 |

Turn-count breakdown:

| Encoder / Scope | turns 0-10 | turns 11-16 | turns 17-24 | turns 25+ |
|---|---:|---:|---:|---:|
| ViT full | 0.6667 | 0.7047 | 0.6655 | 0.5922 |
| Vim full | 0.7619 | 0.6988 | 0.6510 | 0.5939 |
| Vim last4 | 0.5238 | 0.4737 | 0.4881 | 0.4604 |

Interpretation:

- A small amount of decoy supervision improves test accuracy substantially
  compared with zero-decoy training.
- Full ViT and full Vim are now close, around 0.64 test accuracy, but both still
  degrade on longer paths.
- Vim full fine-tuning needed more epochs to reach the same validation range as
  ViT; the 3-epoch Vim run was underfit and should not be used as the main
  comparison.
- Vim last-4 fine-tuning uses far fewer trainable parameters, but it remains far
  below full fine-tuning. This creates a clear target for a path-aware adapter:
  beat last-4 partial FT and move toward full FT.

### Path-Aware Vim Adapter V1/V2

Implementation:

- Added optional Path-Aware State Adapters to the Vim wrapper.
- Added a Path-Aware Token Pooler for adapter-enabled Vim runs.
- Text encoder remains frozen.
- Vim backbone remains frozen for `path_adapter`.
- Trainable modules are path adapters, path pooler, `ln_post`, and projection
  head.

The adapter is enabled only when requested, so previous Vim/ViT baselines are
not changed.

| Method | Output Dir | Trainable Params | Best Val Accuracy | Test Accuracy | Notes |
|---|---|---:|---:|---:|---|
| PASA-v1 state adapters | `runs/vim_pasa4_vstd_decoy_trainhard_10e_b16_lr5e-4` | 3,943,168 | 0.3640 | 0.3043 | State adapters only; too weak. |
| PASA-v2 state adapters + token pooler | `runs/vim_pasa_pool4_vstd_decoy_trainhard_10e_b16_lr5e-4_bestval` | 5,915,649 | 0.6385 | 0.4480 | Much better, but still below last4. |
| PASA-v2 + last1 Vim layer | `runs/vim_pasa_pool4_last1_vstd_decoy_trainhard_10e_b16_lr5e-4` | 9,920,001 | 0.5935 | 0.4040 | High LR unstable for last-layer tuning. |
| PASA-v3 + last1, grouped LR | `runs/vim_pasa_pool4_last1_vstd_decoy_trainhard_10e_group_lr` | 9,920,001 | 0.6445 | 0.4660 | Close to last4, but not above it. |
| PASA-v3 + last2, grouped LR | `runs/vim_pasa_pool4_last2_vstd_decoy_trainhard_10e_group_lr` | 13,923,585 | 0.6575 | 0.4620 | More unfrozen layers did not improve test accuracy. |
| PASA-v2, 8 adapter layers | `runs/vim_pasa_pool8_vstd_decoy_trainhard_10e_b16_lr5e-4` | 7,888,897 | 0.6500 | 0.4470 | Deeper adapter coverage did not improve test accuracy. |
| PASA-augmented full FT | `runs/vim_pasa_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 102,745,089 | 0.9190 | 0.6600 | Improves over vanilla Vim full FT. |
| BiPASA, bidirectional pooling | `runs/vim_bipasa_pool4_vstd_decoy_trainhard_10e_b16_lr5e-4` | 6,703,874 | 0.6300 | 0.4553 | Slightly improves adapter-only PASA. |
| BiPASA-augmented full FT | `runs/vim_bipasa_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 103,533,314 | 0.9150 | 0.6707 | Best full-FT result so far. |
| Start-Aware BiPASA | `runs/vim_start_bipasa_pool4_vstd_decoy_trainhard_10e_b16_lr5e-4` | 7,788,547 | 0.6470 | 0.4593 | Learned start query slightly improves adapter-only BiPASA, but remains below last4. |
| Start-Aware BiPASA full FT | `runs/vim_start_bipasa_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 104,617,987 | 0.8965 | 0.6397 | Worse than BiPASA-full; start query overfits the easier validation distribution. |
| Geometry-aware BiPASA | `runs/vim_geo_bipasa_pool4_vstd_decoy_trainhard_10e_b16_lr5e-4` | 9,066,756 | 0.6470 | 0.4623 | Adds row/column-aware pooling; improves adapter-only BiPASA but still below last4. |
| Geometry-aware BiPASA + last1 | `runs/vim_geo_bipasa_pool4_last1_vstd_decoy_trainhard_10e_group_lr` | 13,071,108 | 0.6545 | 0.4623 | Extra last-layer tuning did not improve OOD test accuracy. |
| Geometry-aware BiPASA full FT | `runs/vim_geo_bipasa_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 105,896,196 | 0.9230 | 0.6660 | Improves over vanilla Vim full FT, but remains slightly below BiPASA-full. |
| Path-State PASA | `runs/vim_path_state_pool4_vstd_decoy_trainhard_10e_b16_lr5e-4` | 7,202,434 | 0.6520 | 0.4493 | Learned local propagation fits validation but weakens OOD decoy generalization. |
| Path-State PASA full FT | `runs/vim_path_state_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 104,031,874 | 0.9150 | 0.6707 | Matches BiPASA-full; helps reverse/turn-heavy cases but hurts scan-aligned cases. |
| Hybrid Path-State BiPASA full FT | `runs/vim_hybrid_path_state_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 105,016,964 | 0.9205 | 0.6787 | Best result so far; combines bidirectional pooling with local propagation. |
| Gated Hybrid Path-State full FT | `runs/vim_gated_hybrid_path_state_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 107,188,233 | 0.8980 | 0.6437 | Dynamic expert gating is hard to optimize and hurts OOD generalization. |
| Omni Path-State full FT | `runs/vim_omni_path_state_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 106,593,414 | 0.9185 | 0.6787 | Matches Hybrid; improves long-path accuracy at length 48 compared with Hybrid. |
| Memory Omni Path-State full FT | `runs/vim_memory_omni_path_state_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 107,481,479 | 0.9205 | 0.6740 | Improves length-48 accuracy but does not improve the selected long-turn diagnostic subset. |
| Endpoint Omni Path-State full FT | `runs/vim_endpoint_omni_path_state_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 109,943,687 | 0.9150 | 0.6633 | Path-conditioned endpoint refinement did not improve overall or selected split accuracy. |
| Layer-wise Omni Path-State full FT | `runs/vim_layerwise_omni_path_state_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 108,570,503 | 0.9145 | 0.6720 | Multi-layer path summaries did not improve over original Omni. |
| Decoy-Suppression full FT | `runs/vim_decoy_suppression_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 104,829,067 | 0.9200 | 0.6690 | Multi-hypothesis keep/suppress pooling did not beat Hybrid/Omni. |
| Soft Decoy-Suppression full FT | `runs/vim_soft_decoy_suppression_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 106,007,436 | 0.8970 | 0.6277 | Token reweighting from decoy hypotheses is unstable without extra supervision. |

Interpretation:

- Pure state adapters are insufficient when the final representation still
  depends mainly on the Vim class token.
- Token-level path-aware pooling is important; it raises PASA from 0.3043 to
  0.4480.
- PASA-v2 uses about one third of the trainable parameters of Vim last4
  fine-tuning, but does not yet beat it.
- Grouped learning rates are important for PASA + last-k tuning. PASA-v3 + last1
  gets close to Vim last4 while using fewer trainable parameters, but it still
  does not exceed last4.
- PASA-augmented full fine-tuning improves Vim full FT from 0.6413 to 0.6600.
  This suggests the path-aware token pooling can improve the upper bound, not
  only parameter efficiency. However, the fair 6-epoch ViT baseline reaches
  0.8063, so these Vim-side improvements should not be framed as beating ViT
  full fine-tuning.
- Bidirectional path-aware pooling improves PASA-full further to 0.6707 and
  reduces the left/up weakness introduced by forward-only pooling.
- Start-Aware BiPASA adds a learned start query without changing the data or
  loss. It gives a small adapter-only gain over BiPASA, but hurts full-FT OOD
  test accuracy. For now it is best treated as an ablation showing that explicit
  start conditioning is not sufficient by itself; the stronger result is still
  bidirectional path-aware pooling.
- Geometry-aware BiPASA adds row/column-aware pooling without changing the data
  or loss. It modestly improves adapter-only BiPASA and improves over vanilla
  Vim full FT, but it does not beat BiPASA-full. This suggests the main useful
  inductive bias is bidirectional path pooling; explicit 2D row/column context
  is reasonable but not yet the decisive ingredient.
- Path-State PASA adds learned local propagation over the 2D patch grid. It is
  not effective as an adapter-only method, but under full fine-tuning it matches
  BiPASA-full. The breakdown suggests it helps reverse and high-turn cases while
  trading off scan-aligned right/down cases.
- Hybrid Path-State BiPASA is the strongest architecture so far. It combines
  BiPASA's bidirectional global path pooling with local state propagation, and
  improves the full-FT result from 0.6707 to 0.6787.
- Gated Hybrid underperforms, suggesting that per-image expert selection is too
  hard to learn from the current training signal. Fixed fusion is more stable.
- Omni Path-State keeps fixed fusion but adds column-major path summaries. It
  ties Hybrid overall and gives better length-48 accuracy, but it still does not
  reach 70% test accuracy or the fair 6-epoch ViT baseline.
- Memory Omni adds learned memory tokens over propagated path states. It improves
  the length-48 bucket, but does not improve the selected long-turn diagnostic
  subset, so the original Omni remains the clearest analysis model.
- Endpoint Omni adds path-conditioned endpoint re-attention. It does not improve
  the benchmark, suggesting that endpoint refinement is not the main bottleneck
  under the current image-level supervision.
- Layer-wise Omni fuses path summaries from multiple late Vim layers. It does
  not improve over original Omni, suggesting the final-layer multi-directional
  summary is already the more stable representation for this benchmark.
- Decoy-Suppression creates multiple path hypotheses and subtracts low-confidence
  summaries. The idea targets the right failure mode, but this hard subtractive
  version likely removes some true-path evidence together with decoy evidence.
- Soft Decoy-Suppression replaces hard subtraction with token reweighting masks.
  It performs worse, suggesting that decoy hypotheses are difficult to learn from
  image-level endpoint labels alone.

PASA-full geometry breakdown:

| Method | mostly right/down | mostly left/up | mixed |
|---|---:|---:|---:|
| Vim full | 0.6900 | 0.5827 | 0.6453 |
| PASA-full | 0.7416 | 0.5524 | 0.6702 |
| BiPASA-full | 0.6778 | 0.6343 | 0.6821 |
| Geometry-aware BiPASA-full | 0.6292 | 0.6434 | 0.6892 |
| Path-State PASA-full | 0.6231 | 0.6601 | 0.6934 |

PASA-full turn-count breakdown:

| Method | turns 0-10 | turns 11-16 | turns 17-24 | turns 25+ |
|---|---:|---:|---:|---:|
| Vim full | 0.7619 | 0.6988 | 0.6510 | 0.5939 |
| PASA-full | 0.8095 | 0.7237 | 0.6758 | 0.6020 |
| BiPASA-full | 0.7619 | 0.7325 | 0.6894 | 0.6118 |
| Geometry-aware BiPASA-full | 0.7619 | 0.7310 | 0.6741 | 0.6162 |
| Path-State PASA-full | 0.7619 | 0.7266 | 0.6758 | 0.6296 |

Current status against the three goals:

1. PASA has not yet beaten Vim last4 partial FT. Best so far is 0.4660 vs
   last4's 0.4747. Geometry-aware BiPASA adapter-only reaches 0.4623.
2. PASA is close to last4 with fewer parameters, but still needs one more method
   iteration to become a strong parameter-efficient result.
3. BiPASA-full improves vanilla full FT from 0.6413 to 0.6707. It is a useful
   positive signal. Hybrid/Omni Path-State further improve the best full-FT
   Vim-side result to 0.6787, but still remain below the 70% target and far
   below ViT-B/16 full FT at 6 epochs.

### Seed Repeat: Full FT Comparison

The initial comparison used the default seed 123. A second training seed was
added to check whether the BiPASA-full gain persists.

| Method | Seed | Output Dir | Best Val Accuracy | Test Accuracy |
|---|---:|---|---:|---:|
| ViT-B/16 full FT | 123 | `runs/vit_b16_clip_fullft_vstd_decoy_trainhard_3e_b16_lr1e-5` | 0.9010 | 0.6470 |
| ViT-B/16 full FT | 123 | `runs/vit_b16_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 0.9760 | 0.8063 |
| Vim full FT | 123 | `runs/vim_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 0.9045 | 0.6413 |
| BiPASA-full | 123 | `runs/vim_bipasa_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 0.9150 | 0.6707 |
| ViT-B/16 full FT | 111 | `runs/vit_b16_clip_fullft_vstd_decoy_trainhard_3e_b16_lr1e-5_seed111` | 0.9100 | 0.6307 |
| Vim full FT | 111 | `runs/vim_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111` | 0.9150 | 0.6333 |
| BiPASA-full | 111 | `runs/vim_bipasa_full_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111` | 0.9070 | 0.6487 |

Mean over the two seeds:

| Method | Mean Test Accuracy |
|---|---:|
| ViT-B/16 full FT | 0.6388 |
| Vim full FT | 0.6373 |
| BiPASA-full | 0.6597 |

Interpretation:

- The original seed-repeat table compared BiPASA-full against a 3-epoch ViT
  baseline. That comparison is not fair.
- After adding the 6-epoch ViT baseline, ViT-B/16 full FT is much stronger on
  `vstd_decoy_trainhard` than all current Vim-side variants.
- The valid Vim-side claim is narrower: BiPASA/Hybrid/Omni improve vanilla Vim
  full FT, but they do not beat a fairly trained ViT baseline.

Next method iteration:

- Keep bidirectional path-aware token pooling for full-FT experiments, since it
  improves the left/up weakness of forward-only PASA-full.
- For parameter-efficient experiments, bidirectional pooling alone is not enough;
  the next step is to add a stronger path-state objective or a more structured
  2D geometry-aware pooler.
- Consider an auxiliary endpoint-consistency or decoy-suppression loss only if
  the architecture-only adapter remains below last4.

## VSTD-FS: Fixed-Start Long-Sequence Pilot

Dataset: `datasets/vstd_fs`

VSTD-FS fixes the start cell at the top-left corner `(0, 0)` to remove start
localization as a confound and focus on long-sequence tracking under stronger
decoys. The current pilot uses train paths of length 24/32 and test paths of
length 48/56.

| Method | Output Dir | Best Val Accuracy | Test Accuracy | Notes |
|---|---|---:|---:|---|
| ViT-B/16 full FT | `runs/vit_b16_clip_fullft_vstd_fs_3e_b16_lr1e-5` | 0.7356 | 0.5113 | Fixed-start long-decoy test is hard for ViT. |
| Vim full FT | `runs/vim_clip_fullft_vstd_fs_6e_b16_lr1e-5` | 0.7762 | 0.5817 | Best pilot result; fixed top-left start favors Vim's sequential bias. |
| BiPASA-full | `runs/vim_bipasa_full_vstd_fs_6e_b16_lr1e-5` | 0.7625 | 0.5333 | Adapter hurts under this fixed-start setting. |
| Omni Path-State full FT | `runs/vim_omni_path_state_full_vstd_fs_6e_b16_lr1e-5` | 0.7156 | 0.5358 | Does not beat vanilla Vim on this pilot split. |

Interpretation:

- VSTD-FS is difficult and avoids the trivial full-FT-to-100% issue.
- However, the current fixed-top-left setting favors vanilla Vim more than Omni:
  Vim reaches 0.5817 while Omni reaches 0.5358.
- This suggests that fixing the start at `(0, 0)` changes the benchmark from
  multi-directional path reasoning toward a sequence-order-friendly task.
- If VSTD-FS is kept, the next version should force more early reverse/up-left
  loops or use multiple fixed corners so that a single raster-order bias is not
  sufficient.

## VSTD-LD: Long-Dependency Stress Test

Dataset: `datasets/vstd_ld`

VSTD-LD keeps the start position free, but increases path length, turn count,
start/end Manhattan distance, and decoy strength. The goal is to stress long
state tracking without simplifying the task into a fixed-start scan pattern.

Dataset summary:

| Split | Samples | Mean Long-Dependency Score |
|---|---:|---:|
| train | 10000 | 0.2370 |
| val | 2000 | 0.3351 |
| test | 3000 | 0.4727 |

Results:

| Method | Output Dir | Epochs | Best Val Accuracy | Test Accuracy |
|---|---|---:|---:|---:|
| ViT-B/16 full FT | `runs/vit_b16_clip_fullft_vstd_ld_3e_b16_lr1e-5` | 3 | 0.5875 | 0.5510 |
| Vim full FT | `runs/vim_clip_fullft_vstd_ld_3e_b16_lr1e-5` | 3 | 0.4990 | 0.4890 |
| Vim full FT | `runs/vim_clip_fullft_vstd_ld_6e_b16_lr1e-5` | 6 | 0.6325 | 0.5160 |
| Omni Path-State full FT | `runs/vim_omni_path_state_full_vstd_ld_6e_b16_lr1e-5` | 6 | 0.7080 | 0.5677 |
| ViT-B/16 full FT | `runs/vit_b16_clip_fullft_vstd_ld_6e_b16_lr1e-5` | 6 | 0.7495 | 0.6663 |

Interpretation:

- VSTD-LD is a useful stress test, but it is not currently an Omni-advantage
  benchmark.
- With the fair 6-epoch schedule, ViT-B/16 is clearly strongest on this split.
- Omni improves over vanilla Vim, but the gap is not enough for the main thesis
  claim. Treat VSTD-LD as a negative/control stress test rather than the main
  headline result.

## Training-Signal Ablation: Hard Negatives and Consistency

Dataset: `datasets/vstd_decoy_trainhard`

These runs use Omni Path-State full FT. Evaluation still uses only the original
eight class prompts; hard-negative prompts are appended during training only.

| Method | Output Dir | Seed | Best Val Accuracy | Test Accuracy |
|---|---|---:|---:|---:|
| ViT-B/16 full FT | `runs/vit_b16_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 123 | 0.9760 | 0.8063 |
| Omni Path-State full FT | `runs/vim_omni_path_state_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 123 | 0.9120 | 0.6787 |
| + hard-negative prompts | `runs/vim_omni_hardneg_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 123 | 0.9125 | 0.6717 |
| + consistency | `runs/vim_omni_consistency_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 123 | 0.9260 | 0.6907 |
| + hard-negative prompts + consistency | `runs/vim_omni_hardneg_consistency_full_vstd_decoy_trainhard_6e_b16_lr1e-5` | 123 | 0.9235 | 0.7153 |
| + hard-negative prompts + consistency | `runs/vim_omni_hardneg_consistency_full_vstd_decoy_trainhard_6e_b16_lr1e-5_seed321` | 321 | 0.9235 | 0.6657 |

Interpretation:

- Consistency regularization improves the default-seed Omni result from 0.6787
  to 0.6907.
- Hard-negative prompts alone do not improve the result, but combining hard
  negatives with consistency reaches 0.7153 on seed 123.
- The second seed drops to 0.6657, so the combined setting is promising but
  seed-sensitive. It should not be the only headline claim yet.
- Even the strongest Omni training-signal variant remains below the fair
  6-epoch ViT full-FT baseline of 0.8063.
- The stable takeaway is that VSTD benefits from state-consistent visual
  representations; the prompt hard negatives need more tuning before becoming a
  robust method contribution.

## ViT Path-Aware Architecture Sweep

Dataset: `datasets/vstd_decoy_trainhard`

After the fair 6-epoch ViT baseline proved much stronger than the Vim-side
variants, the main direction was shifted from Vim replacement to ViT
architecture improvement. These runs keep CLIP ViT-B/16 as the visual backbone
and add a lightweight path-aware readout module on top of the final patch
tokens.

All runs use the same setting: full visual fine-tuning, 6 epochs, batch size 16,
learning rate `1e-5`, weight decay `1e-4`, seed 123.

| Method | Output Dir | Best Val Accuracy | Test Accuracy | L=48 Accuracy | Delta vs ViT 6e |
|---|---|---:|---:|---:|---:|
| ViT-B/16 full FT | `runs/vit_b16_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 0.9760 | 0.8063 | 0.7347 | - |
| Start-conditioned pooling | `runs/vit_start_pool_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 0.9600 | 0.7750 | 0.6961 | -0.0313 |
| Local path-state propagation | `runs/vit_path_state_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 0.9600 | 0.7743 | 0.7118 | -0.0320 |
| Decoy-aware token reweighting | `runs/vit_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 0.9875 | 0.8510 | 0.8074 | +0.0447 |
| Start-conditioned path-state | `runs/vit_start_path_state_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 0.9695 | 0.7773 | 0.7332 | -0.0290 |
| Multi-hop path readout | `runs/vit_multihop_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 0.9920 | 0.9047 | 0.8559 | +0.0983 |

Multi-hop seed repeat:

| Method | Seed | Output Dir | Best Val Accuracy | Test Accuracy | L=48 Accuracy |
|---|---:|---|---:|---:|---:|
| ViT-B/16 full FT | 123 | `runs/vit_b16_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 0.9760 | 0.8063 | 0.7347 |
| ViT-B/16 full FT | 111 | `runs/vit_b16_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111` | 0.9420 | 0.7230 | 0.6576 |
| ViT-B/16 full FT | 321 | `runs/vit_b16_clip_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed321` | 0.9795 | 0.8287 | 0.7803 |
| Decoy-aware token reweighting | 123 | `runs/vit_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 0.9875 | 0.8510 | 0.8074 |
| Decoy-aware token reweighting | 111 | `runs/vit_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111` | 0.9770 | 0.8557 | 0.8374 |
| Decoy-aware token reweighting | 321 | `runs/vit_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed321` | 0.9845 | 0.8763 | 0.8431 |
| Multi-hop path readout | 123 | `runs/vit_multihop_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 0.9920 | 0.9047 | 0.8559 |
| Multi-hop path readout | 321 | `runs/vit_multihop_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed321` | 0.9910 | 0.8857 | 0.8502 |
| Multi-hop path readout | 111 | `runs/vit_multihop_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111` | 0.9700 | 0.8037 | 0.7418 |

Decoy-aware mean over three seeds: test accuracy `0.8610`, population std
`0.0110`; length-48 accuracy `0.8293`, population std `0.0156`.

ViT-B/16 mean over three seeds: test accuracy `0.7860`, population std
`0.0455`; length-48 accuracy `0.7242`, population std `0.0506`.

Multi-hop mean over three seeds: test accuracy `0.8647`, population std
`0.0438`; length-48 accuracy `0.8160`, population std `0.0525`.

Additional stability attempts:

| Method | Seed | Output Dir | Best Val Accuracy | Test Accuracy | L=48 Accuracy |
|---|---:|---|---:|---:|---:|
| Decoy-gated multi-hop, residual scale 0.2 | 123 | `runs/vit_decoy_multihop_rs020_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5` | 0.9600 | 0.7613 | 0.6776 |
| Multi-hop, residual scale 0.2 | 111 | `runs/vit_multihop_rs020_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111` | 0.9685 | 0.7713 | 0.7218 |
| Multi-hop, 1-epoch readout warmup | 111 | `runs/vit_multihop_warm1_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111` | 0.9605 | 0.8190 | 0.7575 |
| Multi-query multi-hop | 111 | `runs/vit_multiquery_multihop_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111` | 0.9710 | 0.7997 | 0.7489 |
| Decoy-fusion multi-hop | 111 | `runs/vit_decoy_fusion_multihop_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111` | 0.9715 | 0.8127 | 0.7447 |
| Zero-init residual multi-hop | 111 | `runs/vit_multihop_zero_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111` | 0.9790 | 0.8453 | 0.8088 |
| Multi-head decoy-aware | 111 | `runs/vit_multihead_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111` | 0.9785 | 0.8297 | 0.7632 |
| Layer-wise decoy-aware | 111 | `runs/vit_layerwise_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111` | 0.9630 | 0.7940 | 0.7290 |
| Adaptive decoy-aware | 111 | `runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111` | 0.9895 | 0.9117 | 0.9001 |
| Adaptive decoy-aware | 123 | `runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123` | 0.9845 | 0.8613 | 0.8188 |
| Adaptive decoy-aware | 321 | `runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed321` | 0.9910 | 0.8807 | 0.8245 |
| Directional decoy-aware | 111 | `runs/vit_directional_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111` | 0.9885 | 0.8947 | 0.8702 |
| Slot decoy-aware | 111 | `runs/vit_slot_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111` | 0.9920 | 0.9103 | 0.9116 |
| Slot decoy-aware | 123 | `runs/vit_slot_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123` | 0.9905 | 0.8990 | 0.8602 |
| Slot decoy-aware | 321 | `runs/vit_slot_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed321` | 0.9815 | 0.8410 | 0.8174 |

Adaptive decoy-aware mean over three seeds: test accuracy `0.8846`,
population std `0.0207`; length-48 accuracy `0.8478`, population std `0.0371`.

Slot decoy-aware mean over three seeds: test accuracy `0.8834`, population std
`0.0304`; length-48 accuracy `0.8631`, population std `0.0385`.

Main three-seed summary:

| Method | Test Mean | Test Std | L=48 Mean | L=48 Std | Mean Delta vs ViT |
|---|---:|---:|---:|---:|---:|
| ViT-B/16 full FT | 0.7860 | 0.0455 | 0.7242 | 0.0506 | - |
| Decoy-aware readout | 0.8610 | 0.0110 | 0.8293 | 0.0156 | +0.0750 |
| ADAR: Adaptive decoy-aware readout | 0.8846 | 0.0207 | 0.8478 | 0.0371 | +0.0986 |
| Slot decoy-aware readout | 0.8834 | 0.0304 | 0.8631 | 0.0385 | +0.0974 |

Path-length mean accuracy over three seeds:

| Method | L=24 | L=32 | L=40 | L=48 |
|---|---:|---:|---:|---:|
| ViT-B/16 full FT | 0.8431 | 0.8135 | 0.7566 | 0.7242 |
| Decoy-aware readout | 0.9007 | 0.8737 | 0.8367 | 0.8293 |
| ADAR | 0.9176 | 0.9023 | 0.8665 | 0.8478 |
| Slot decoy-aware readout | 0.9202 | 0.8865 | 0.8616 | 0.8631 |

Interpretation:

- Direct start-conditioned pooling and local path-state propagation hurt the
  strong ViT baseline. They likely disturb the already-useful CLS-token
  representation without providing enough extra supervision.
- Decoy-aware token reweighting is the first clearly positive ViT architecture:
  it improves overall accuracy by 4.47 points and length-48 accuracy by 7.27
  points.
- Multi-hop path readout is the strongest result so far. It improves the fair
  ViT baseline from 0.8063 to 0.9047, and improves the length-48 bucket from
  0.7347 to 0.8559.
- Multi-hop is promising but seed-sensitive. Two of three seeds clearly improve
  the ViT baseline, while seed 111 roughly matches it. The next step is to
  stabilize the module with safer initialization, residual scaling, or a
  staged schedule that warms up the readout before full fine-tuning.
- Decoy-aware token reweighting is currently the most stable method: all three
  seeds beat the fair ViT baseline, with low variance. It should be treated as
  the reliable architecture contribution.
- Multi-hop should be presented as a high-ceiling extension. Residual scaling
  and readout warmup did not solve its seed sensitivity, so more careful
  stabilization is still needed before claiming stable 90%+ performance.
- Among the stabilization attempts, zero-init residual multi-hop is the most
  promising: it lifts the weak seed 111 from 0.8037 to 0.8453. However, it still
  does not match the stable decoy-aware result on the same seed.
- Strengthening decoy-aware with multi-head gates or late-layer fusion did not
  improve over the simple single-gate decoy-aware readout. This suggests the
  stable gain comes from a simple final-token reweighting bias rather than
  extra readout capacity.
- Adaptive decoy-aware is the strongest mean method so far among the three-seed
  runs. It improves the three-seed mean from decoy-aware's 0.8610 to 0.8846 and
  reaches 0.9117 on seed 111, but its variance is higher than simple
  decoy-aware.
- Slot decoy-aware is a strong new candidate: on seed 111 it reaches 0.9103 and
  length-48 accuracy 0.9116, suggesting that maintaining multiple candidate
  path hypotheses is useful for long decoy-heavy cases. Its three-seed mean is
  close to adaptive decoy-aware, and it gives the best length-48 mean among the
  decoy-aware variants, but its seed-321 result is weaker.
- This suggests the new thesis direction should be framed as: VSTD exposes a
  need for structured path readout on top of strong ViT features; the useful
  architectural bias is not simply "start awareness", but iterative
  query-based path evidence aggregation and decoy-resistant token selection.

Qualitative visualization:

- ADAR gate heatmaps were generated with
  `scripts/visualize_vit_readout_gates.py`.
- Output directory: `docs/figures/adar_gate_examples`.
- The generated examples focus on length-48 test samples with strong decoy
  paths, which are the key OOD cases for the paper.

Paper-ready main claim:

- Under the same 6-epoch compute budget and best-validation checkpoint
  selection, CLIP ViT-B/16 reaches `0.7860 +/- 0.0455` over three seeds.
- Simple decoy-aware readout improves this to `0.8610 +/- 0.0110`.
- ADAR improves the mean further to `0.8846 +/- 0.0207`, a `+9.86` point mean
  improvement over vanilla ViT.
- Slot decoy-aware readout reaches a similar mean (`0.8834`) and gives the best
  length-48 mean (`0.8631`), making it a useful extension for long-path
  analysis.
