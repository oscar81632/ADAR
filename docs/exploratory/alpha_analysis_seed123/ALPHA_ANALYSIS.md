# ADAR Alpha Analysis

- Run: `runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123`
- Dataset: `datasets/vstd_decoy_trainhard`
- Split: `test`
- Samples: 3000
- Accuracy: 0.8613
- Alpha mean/std/min/max: 0.4822 / 0.0273 / 0.4217 / 0.5493

## Correlations With Alpha

| Variable | Pearson r |
|---|---:|
| correct | 0.0311 |
| path_length | 0.0819 |
| distractor_ratio | 0.0175 |
| decoy_arrow_count | -0.0030 |
| start_end_manhattan | 0.0540 |
| num_turns | 0.0818 |
| path_bbox_area | 0.0730 |
| path_coverage | 0.0819 |
| long_dependency_score | 0.0805 |

## Correctness Groups

| Group | n | Alpha mean | Alpha std |
|---|---:|---:|---:|
| wrong | 416 | 0.4801 | 0.0273 |
| correct | 2584 | 0.4826 | 0.0272 |

## Path Length

| Bucket | n | Alpha mean | Alpha std |
|---|---:|---:|---:|
| 24 | 769 | 0.4782 | 0.0268 |
| 32 | 781 | 0.4828 | 0.0279 |
| 40 | 749 | 0.4838 | 0.0266 |
| 48 | 701 | 0.4845 | 0.0274 |

## Distractor Ratio

| Bucket | n | Alpha mean | Alpha std |
|---|---:|---:|---:|
| 0.25 | 1489 | 0.4818 | 0.0272 |
| 0.4 | 1511 | 0.4827 | 0.0273 |

## Decoy Arrow Count

| Bucket | n | Alpha mean | Alpha std |
|---|---:|---:|---:|
| 16..20 | 750 | 0.4825 | 0.0272 |
| 20..22 | 750 | 0.4818 | 0.0270 |
| 22..24 | 750 | 0.4824 | 0.0277 |
| 24..28 | 750 | 0.4823 | 0.0272 |

## Long-Dependency Score

| Bucket | n | Alpha mean | Alpha std |
|---|---:|---:|---:|
| 0.11318681318681319..0.22218733647305078 | 750 | 0.4790 | 0.0271 |
| 0.22218733647305078..0.2795918367346939 | 750 | 0.4832 | 0.0275 |
| 0.28008895866038724..0.34296180010465727 | 750 | 0.4816 | 0.0269 |
| 0.34306645735217167..0.6000000000000001 | 750 | 0.4852 | 0.0272 |
