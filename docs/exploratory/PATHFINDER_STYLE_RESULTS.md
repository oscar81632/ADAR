# Pathfinder-Style Exploratory Results

This experiment follows the recommendation to test a public-benchmark-adjacent
long-range visual reasoning task. The official LRA Pathfinder release was not
directly accessible in this environment, so this dataset is a small
Pathfinder-style exploratory benchmark generated locally.

Important: these are not official LRA Pathfinder numbers.

## Dataset

| Item | Setting |
|---|---|
| Dataset name | `pathfinder_style` |
| Official LRA release | No |
| Task | Binary image-text classification |
| Classes | disconnected endpoints / connected endpoints |
| Train / Val / Test | 6,000 / 1,000 / 2,000 |
| Image size | 224 x 224 |
| Grid size | 18 x 18 |
| Main path length | 42-72 steps |
| Decoy paths | 8-16 per image |
| Prompts | `two marked endpoints are not connected by one path`; `two marked endpoints are connected by one path` |

The task asks whether two marked endpoints belong to the same long path under
visual clutter. It is intended to probe long-range spatial dependency and
decoy robustness, similar in spirit to LRA Pathfinder.

## Results

Both models use OpenAI CLIP ViT-B/16 with a frozen text tower and full visual
fine-tuning for 6 epochs.

| Model | Run directory | Trainable params | Best epoch | Best val acc | Test acc |
|---|---:|---:|---:|---:|---:|
| Vanilla ViT-B/16 CLIP | `runs/exploratory_pathfinder_style_vit_b16_fullft_6e_b16_lr1e-5_seed123` | 86.19M | 6 | 97.00 | 97.25 |
| ADAR ViT-B/16 CLIP | `runs/exploratory_pathfinder_style_adar_fullft_6e_b16_lr1e-5_seed123` | 99.81M | 6 | 97.40 | 98.00 |

## Interpretation

ADAR improves the single-seed Pathfinder-style test accuracy by 0.75 percentage
points. This is a smaller margin than the main VSTD result, but the direction is
consistent with the thesis story: ADAR is more useful on tasks where the readout
must compare selected path evidence against visually distracting context.

The result should be treated as supporting exploratory evidence rather than a
formal public benchmark claim, because the official LRA Pathfinder data was not
used.
