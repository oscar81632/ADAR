# Original Pathfinder Exploratory Results

This experiment tests whether the Original Pathfinder classification data can
be adapted to the existing VSTD/CLIP-style pipeline.

## Source

| Item | Setting |
|---|---|
| Source URL | `https://connectomics.clps.brown.edu/tf_records/Pathfinder/` |
| Dataset variant | `pathfinder128/curv_contour_length_14` |
| Local raw data | `external_datasets/OriginalPathfinder/pathfinder128/curv_contour_length_14` |
| Converted dataset | `datasets/original_pathfinder14_950_balanced` |
| Task | Binary classification |
| Prompts | `two dots are on separate paths`; `two dots are connected by the same path` |

The public directory exposes `imgs/` and `metadata/`. The metadata file has
1,000 samples. Column index 3 is used as the binary connectivity label.

## Converted Split

The available class distribution is slightly imbalanced, so the converted split
uses 950 balanced samples.

| Split | Samples | Class 0 | Class 1 |
|---|---:|---:|---:|
| Train | 650 | 325 | 325 |
| Val | 150 | 75 | 75 |
| Test | 150 | 75 | 75 |

## Results

Both models use the same CLIP-style classification objective as VSTD, with a
frozen text tower and full visual fine-tuning for 6 epochs.

| Model | Run directory | Best val acc | Test acc |
|---|---:|---:|---:|
| Vanilla ViT-B/16 CLIP | `runs/exploratory_original_pathfinder14_balanced_vit_b16_fullft_6e_b16_lr1e-5_seed123` | 50.00 | 50.00 |
| ADAR ViT-B/16 CLIP | `runs/exploratory_original_pathfinder14_balanced_adar_fullft_6e_b16_lr1e-5_seed123` | 50.00 | 50.00 |

## Interpretation

The dataset can be downloaded and converted into the VSTD-compatible
classification format, but this small public subset does not work well with the
current CLIP-style fine-tuning setup. Both vanilla ViT-B/16 and ADAR remain at
chance accuracy.

## Sanity Audit

An additional audit checked whether the 50% result came from a broken accuracy
calculation or from degenerate predictions.

| Model | Split | Prediction counts | Confusion matrix rows=label cols=pred |
|---|---|---:|---:|
| Vanilla zero-shot | test | `[0, 150]` | `[[0, 75], [0, 75]]` |
| Vanilla fine-tuned | train | `[650, 0]` | `[[325, 0], [325, 0]]` |
| Vanilla fine-tuned | test | `[150, 0]` | `[[75, 0], [75, 0]]` |
| ADAR zero-shot | test | `[0, 150]` | `[[0, 75], [0, 75]]` |
| ADAR fine-tuned | train | `[650, 0]` | `[[325, 0], [325, 0]]` |
| ADAR fine-tuned | test | `[150, 0]` | `[[75, 0], [75, 0]]` |

The result is therefore a real training failure under this pipeline: zero-shot
CLIP predicts only the connected class, while fine-tuning flips to predicting
only the separate class. The balanced train/validation/test splits confirm that
the 50% accuracy is not caused by class imbalance.

Likely reasons:

- The available public subset for this variant contains only 1,000 images.
- Original Pathfinder images are 128 x 128 with very fine dashed contours; CLIP
  ViT-B/16 uses 16 x 16 patches after resizing to 224 x 224, so the relevant
  contour signal may be too fine for this setup.
- The task is not naturally image-text semantic classification; the prompts
  provide weak supervision for a low-level path-connectivity problem.

This result should not be used as a positive thesis result. It is useful mainly
as evidence that Original Pathfinder is accessible but requires a more
specialized training/evaluation pipeline than the current VSTD CLIP-style setup.
