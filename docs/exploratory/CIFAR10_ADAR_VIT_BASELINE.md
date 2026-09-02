# CIFAR-10 Exploratory Full Fine-Tuning

This is an exploratory experiment separate from the main VSTD thesis results.
Both models use the repo-local CLIP ViT-B/16 image-text pipeline with a frozen
text encoder and full visual fine-tuning.

## Setup

| Item | Setting |
|---|---|
| Dataset | CIFAR-10 |
| Train / Val / Test | 45,000 / 5,000 / 10,000 |
| Split seed | 123 |
| Prompt template | `a photo of a {class}` |
| CLIP backbone | ViT-B/16 |
| Text tower | OpenAI CLIP ViT-B/16 text encoder, frozen |
| Train scope | Full visual fine-tuning |
| Epochs | 6 |
| Batch size | 16 |
| Eval batch size | 128 |
| Learning rate | 1e-5 |
| Weight decay | 1e-4 |
| GPU | NVIDIA GeForce RTX 4090 |

## Results

| Model | Run directory | Trainable params | Best epoch | Best val acc | Test acc |
|---|---:|---:|---:|---:|---:|
| Vanilla ViT-B/16 CLIP | `runs/exploratory_cifar_vit_b16_fullft_6e_b16_lr1e-5_seed123` | 86.19M | 6 | 96.52 | 96.25 |
| ADAR ViT-B/16 CLIP | `runs/exploratory_cifar_adar_fullft_6e_b16_lr1e-5_seed123` | 99.81M | 2 | 96.42 | 96.66 |

## Limited-Data Non-Full Fine-Tuning

This follow-up uses 1,000 training images per CIFAR-10 class, for 10,000 total
training images. The validation set still uses 5,000 held-out images from the
original CIFAR-10 training split, and the official 10,000-image test split is
unchanged.

| Model | Train scope | Run directory | Trainable params | Best epoch | Best val acc | Test acc |
|---|---|---:|---:|---:|---:|---:|
| Vanilla ViT-B/16 CLIP | projection only | `runs/exploratory_cifar1000_vit_b16_projection_6e_b16_lr1e-5_seed123` | 0.39M | 4 | 96.04 | 95.15 |
| ADAR ViT-B/16 CLIP | readout + projection | `runs/exploratory_cifar1000_adar_readout_projection_6e_b16_lr1e-5_seed123` | 14.01M | 4 | 95.98 | 95.21 |

The limited-data result is nearly tied. ADAR remains slightly higher on test
accuracy in this single seed, but the improvement is only 0.06 percentage
points. This supports a conservative interpretation: CIFAR-10 does not strongly
exercise ADAR's intended long-path/decoy-aware inductive bias.

## STL-10 Full Fine-Tuning

This follow-up uses STL-10 as another natural-image classification sanity check.
The official STL-10 training split contains only 5,000 labeled images, so the
experiment uses a 4,500 / 500 train-validation split and evaluates on the
official 8,000-image test split.

| Model | Run directory | Trainable params | Best epoch | Best val acc | Test acc |
|---|---:|---:|---:|---:|---:|
| Vanilla ViT-B/16 CLIP | `runs/exploratory_stl10_vit_b16_fullft_6e_b16_lr1e-5_seed123` | 86.19M | 1 | 98.00 | 97.44 |
| ADAR ViT-B/16 CLIP | `runs/exploratory_stl10_adar_fullft_6e_b16_lr1e-5_seed123` | 99.81M | 1 | 98.00 | 96.76 |

Unlike VSTD, STL-10 is a natural object-recognition dataset without explicit
long-path tracking or dense decoy structure. In this single-seed full fine-tuning
run, ADAR is lower than the vanilla CLIP ViT-B/16 baseline. This supports a
task-specific interpretation of ADAR: the readout bias is useful when the task
requires comparing selected path evidence against distracting context, but it is
not expected to universally improve standard object classification.

## Notes

- CIFAR-10 is a coarse object-category recognition task, not a long-path or
  decoy-heavy reasoning task.
- ADAR does not harm this setting and obtains a small test improvement in this
  single-seed run, but the margin is small enough that it should be treated as
  exploratory rather than a thesis claim.
- The result is useful as a sanity check: ADAR's readout modification can still
  operate under ordinary image classification prompts, but VSTD remains the
  task where the architectural motivation is clearest.

## Per-Class Test Accuracy

| Class | Vanilla ViT-B/16 | ADAR |
|---|---:|---:|
| airplane | 97.20 | 98.10 |
| automobile | 96.00 | 95.70 |
| bird | 95.60 | 96.30 |
| cat | 89.90 | 94.70 |
| deer | 97.80 | 98.60 |
| dog | 92.90 | 91.20 |
| frog | 98.10 | 98.90 |
| horse | 97.90 | 96.90 |
| ship | 98.70 | 98.30 |
| truck | 98.40 | 97.90 |
