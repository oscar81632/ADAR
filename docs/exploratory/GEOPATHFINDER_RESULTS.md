# GeoPathfinder Exploratory Results

This note records an external long-range spatial reasoning experiment on
GeoPathfinder. The experiment is kept separate from the thesis-facing VSTD
results.

## Dataset

- Source: `external_datasets/GeoPathfinder/geopathfinder`
- Prepared dataset: `datasets/geopathfinder_clip`
- Task: binary CLIP-style image-text classification.
- Image size: 256 x 256 satellite patches, resized by the CLIP preprocess.
- Classes and prompts:
  - `0`: `two red dots are separated by water`
  - `1`: `two red dots are connected by land`

Official split, converted at the image level while preserving patch-level split:

| Split | Images | Class 0 | Class 1 |
|---|---:|---:|---:|
| Train | 8,436 | 4,218 | 4,218 |
| Val | 1,056 | 528 | 528 |
| Test | 1,054 | 527 | 527 |

Each labeled patch contributes one connected image and one disconnected image.

## Protocol

The runs follow the main ADAR fine-tuning protocol:

- CLIP model: `ViT-B/16`
- Text encoder: frozen
- Visual side: full fine-tuning
- Epochs: 6
- Batch size: 16
- Eval batch size: 64
- Learning rate: `1e-5`
- Weight decay: `1e-4`
- Checkpoint selection: best validation accuracy
- Seed: 123

Additional exploratory runs vary the fine-tuning scope and regularization. These
settings are explicitly marked in the results table.

## Results

| Model | Best Epoch | Best Val Acc. | Test Acc. | Run |
|---|---:|---:|---:|---|
| Vanilla CLIP ViT-B/16 | 3 | 0.7140 | 0.7438 | `runs/exploratory_geopathfinder_vit_b16_fullft_6e_b16_lr1e-5_seed123` |
| ADAR | 2 | 0.7140 | 0.7163 | `runs/exploratory_geopathfinder_adar_fullft_6e_b16_lr1e-5_seed123` |
| Geo-Complement ADAR | 5 | 0.7055 | 0.7239 | `runs/exploratory_geopathfinder_geo_complement_adar_fullft_6e_b16_lr1e-5_seed123` |
| Geo-TwoQuery ADAR | 2 | 0.7074 | 0.7211 | `runs/exploratory_geopathfinder_geo_twoquery_adar_fullft_6e_b16_lr1e-5_seed123` |
| Geo-Band ADAR | 2 | 0.7169 | 0.7372 | `runs/exploratory_geopathfinder_geo_band_adar_fullft_6e_b16_lr1e-5_seed123` |
| Geo-LayerMixed ADAR | 4 | 0.7131 | 0.7296 | `runs/exploratory_geopathfinder_geo_layer_mixed_adar_fullft_6e_b16_lr1e-5_seed123` |
| Vanilla ViT-B/16, stronger WD | 5 | 0.7188 | 0.7315 | `runs/exploratory_geopathfinder_vit_b16_fullft_6e_b16_lr1e-5_wd1e-3_seed123` |
| Vanilla ViT-B/16, last-4 FT + stronger WD | 4 | 0.6496 | 0.6528 | `runs/exploratory_geopathfinder_vit_b16_last4_6e_b16_lr3e-5_wd1e-3_seed123` |
| Geo-Band ADAR, last-4 FT + stronger reg. | 5 | 0.6562 | 0.6727 | `runs/exploratory_geopathfinder_geo_band_adar_last4_6e_b16_lr3e-5_wd1e-3_do020_seed123` |
| Geo-Band ADAR, warmup1 + full FT + stronger reg. | 5 | 0.7131 | 0.7457 | `runs/exploratory_geopathfinder_geo_band_adar_warmup1_fullft_6e_b16_lr1e-5_wd1e-3_do020_seed123` |

## Readout Variants

- Geo-Complement ADAR replaces the original `mean - Z_keep` suppressed branch
  with complement-weighted pooling, so the second branch represents low-gate
  context rather than a feature subtraction.
- Geo-TwoQuery ADAR uses two learnable query pools to read two complementary
  summaries, intended to separate connected evidence from barrier evidence.
- Geo-Band ADAR adds a learnable 14 x 14 spatial prior to the gated pooling
  weights. This keeps the ADAR readout form but may exploit placement
  regularities, so it should be treated as an exploratory variant.
- Geo-LayerMixed ADAR learns a weighted mixture of the final four ViT token
  layers before applying complement-weighted ADAR pooling.
- The last-4 FT setting updates only the final four visual transformer blocks
  and the readout head. It is included as a parameter-efficient control.
- The warmup setting trains only the readout module for the first epoch, then
  switches to full visual fine-tuning.

## Interpretation

Under the current CLIP-style prompt fine-tuning setup, the original ADAR readout
does not improve over vanilla ViT-B/16 on GeoPathfinder. This is a useful
external result: it suggests that the VSTD gains do not automatically transfer
to satellite connectivity reasoning without adapting the readout, prompts, or
training recipe to the geospatial domain.

The validation curves also show rapid overfitting after the best checkpoint.
GeoPathfinder may require stronger regularization, resampled placements, or
evaluation on the provided easy/medium/hard placement variants to understand
whether the model learns connectivity or placement-specific shortcuts.

Among the tested readout variants, Geo-Band ADAR with one epoch of readout
warmup, full visual fine-tuning, stronger weight decay, and adapter dropout
performs best. It reaches 0.7457 test accuracy, which is slightly above the
original vanilla ViT-B/16 run and about 2.9 points above the original ADAR run.
The stronger-weight-decay vanilla control drops to 0.7315, so the gain is not
explained by weight decay alone.

The last-4 FT controls underfit strongly for both vanilla ViT-B/16 and Geo-Band
ADAR. This suggests that GeoPathfinder needs broader visual adaptation than the
parameter-efficient setting used here. For a paper-facing claim, these
exploratory results should be repeated with multiple seeds and evaluated on the
provided placement variants before being included in the main thesis narrative.
