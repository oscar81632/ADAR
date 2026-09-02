# ADAR Alpha Analysis

- Run: `runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed321`
- Dataset: `datasets/vstd_decoy_trainhard`
- Split: `test`
- Samples: 3000
- Accuracy: 0.8807
- Alpha mean/std/min/max: 0.4349 / 0.0333 / 0.3466 / 0.5279

## Correlations With Alpha

| Variable | Pearson r |
|---|---:|
| correct | 0.0517 |
| path_length | -0.0221 |
| distractor_ratio | 0.0439 |
| decoy_arrow_count | 0.0012 |
| start_end_manhattan | 0.0475 |
| num_turns | -0.0200 |
| path_bbox_area | -0.0167 |
| path_coverage | -0.0221 |
| long_dependency_score | 0.0146 |

## Correctness Groups

| Group | n | Alpha mean | Alpha std |
|---|---:|---:|---:|
| wrong | 358 | 0.4302 | 0.0321 |
| correct | 2642 | 0.4355 | 0.0334 |

## Path Length

| Bucket | n | Alpha mean | Alpha std |
|---|---:|---:|---:|
| 24 | 769 | 0.4355 | 0.0344 |
| 32 | 781 | 0.4356 | 0.0327 |
| 40 | 749 | 0.4347 | 0.0323 |
| 48 | 701 | 0.4336 | 0.0337 |

## Distractor Ratio

| Bucket | n | Alpha mean | Alpha std |
|---|---:|---:|---:|
| 0.25 | 1489 | 0.4334 | 0.0336 |
| 0.4 | 1511 | 0.4364 | 0.0329 |

## Decoy Arrow Count

| Bucket | n | Alpha mean | Alpha std |
|---|---:|---:|---:|
| 16..20 | 750 | 0.4352 | 0.0332 |
| 20..22 | 750 | 0.4344 | 0.0324 |
| 22..24 | 750 | 0.4351 | 0.0337 |
| 24..28 | 750 | 0.4349 | 0.0338 |

## Long-Dependency Score

| Bucket | n | Alpha mean | Alpha std |
|---|---:|---:|---:|
| 0.11318681318681319..0.22218733647305078 | 750 | 0.4342 | 0.0345 |
| 0.22218733647305078..0.2795918367346939 | 750 | 0.4365 | 0.0327 |
| 0.28008895866038724..0.34296180010465727 | 750 | 0.4337 | 0.0335 |
| 0.34306645735217167..0.6000000000000001 | 750 | 0.4353 | 0.0322 |
