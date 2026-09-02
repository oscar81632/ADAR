# LRA Pathfinder128 Hard Shard 1 Results

This note records an exploratory external-dataset check using a small publicly
downloadable Pathfinder128 hard shard.

## Dataset

- Source: `external_datasets/LRA_Pathfinder128_Hard_Shard1`
- Prepared dataset: `datasets/lra_pathfinder128_hard_shard1_split`
- Task: binary image-text classification.
- Prompts:
  - `two dots are on separate paths`
  - `two dots are connected by the same path`
- Split:
  - Train: 700 images, 350 per class.
  - Val: 100 images, 50 per class.
  - Test: 150 images, 75 per class.

The available public shard contains only 1,000 images, so this is a smoke-level
external check rather than a full LRA benchmark.

## Protocol

The runs use the same fine-tuning budget as the main ADAR experiments:

- CLIP model: `ViT-B/16`
- Text encoder: frozen
- Image side: full fine-tuning
- Epochs: 6
- Batch size: 16
- Eval batch size: 64
- Learning rate: `1e-5`
- Weight decay: `1e-4`
- Checkpoint selection: best validation accuracy
- Seed: 123

## Results

| Model | Best Val Acc. | Test Acc. | Run |
|---|---:|---:|---|
| Vanilla CLIP ViT-B/16 | 0.5500 | 0.5200 | `runs/exploratory_lra_pathfinder128_hard_shard1_vit_b16_fullft_6e_b16_lr1e-5_seed123` |
| ADAR | 0.5000 | 0.5000 | `runs/exploratory_lra_pathfinder128_hard_shard1_adar_fullft_6e_b16_lr1e-5_seed123` |

## Interpretation

Both models remain near chance level on this small shard. This suggests that the
current CLIP-style prompt fine-tuning setup does not yet provide a meaningful
external validation result on the downloaded Pathfinder shard. The most likely
issues are data scale, the small validation/test sizes, and the mismatch between
Pathfinder's binary connectivity labels and natural-language CLIP prompts.

For a paper-facing external benchmark, the next step should be to obtain the
full processed LRA Pathfinder train/dev/test splits, or to construct a larger
Pathfinder-compatible subset with reliable labels and enough samples for stable
validation.
