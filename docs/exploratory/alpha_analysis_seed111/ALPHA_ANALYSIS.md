# ADAR Alpha Analysis

- Run: `runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111`
- Dataset: `datasets/vstd_decoy_trainhard`
- Split: `test`
- Samples: 3000
- Accuracy: 0.9117
- Alpha mean/std/min/max: 0.3117 / 0.0433 / 0.2214 / 0.4404

## Correlations With Alpha

| Variable | Pearson r |
|---|---:|
| correct | 0.0623 |
| path_length | -0.0781 |
| distractor_ratio | 0.0155 |
| decoy_arrow_count | -0.0906 |
| start_end_manhattan | -0.0111 |
| num_turns | -0.0750 |
| path_bbox_area | -0.0809 |
| path_coverage | -0.0781 |
| long_dependency_score | -0.0582 |

## Correctness Groups

| Group | n | Alpha mean | Alpha std |
|---|---:|---:|---:|
| wrong | 265 | 0.3030 | 0.0421 |
| correct | 2735 | 0.3125 | 0.0433 |

## Path Length

| Bucket | n | Alpha mean | Alpha std |
|---|---:|---:|---:|
| 24 | 769 | 0.3158 | 0.0428 |
| 32 | 781 | 0.3145 | 0.0446 |
| 40 | 749 | 0.3082 | 0.0437 |
| 48 | 701 | 0.3078 | 0.0412 |

## Distractor Ratio

| Bucket | n | Alpha mean | Alpha std |
|---|---:|---:|---:|
| 0.25 | 1489 | 0.3110 | 0.0433 |
| 0.4 | 1511 | 0.3124 | 0.0432 |

## Decoy Arrow Count

| Bucket | n | Alpha mean | Alpha std |
|---|---:|---:|---:|
| 16..20 | 750 | 0.3184 | 0.0429 |
| 20..22 | 750 | 0.3100 | 0.0421 |
| 22..24 | 750 | 0.3117 | 0.0425 |
| 24..28 | 750 | 0.3067 | 0.0447 |

## Long-Dependency Score

| Bucket | n | Alpha mean | Alpha std |
|---|---:|---:|---:|
| 0.11318681318681319..0.22218733647305078 | 750 | 0.3150 | 0.0427 |
| 0.22218733647305078..0.2795918367346939 | 750 | 0.3137 | 0.0437 |
| 0.28008895866038724..0.34296180010465727 | 750 | 0.3099 | 0.0441 |
| 0.34306645735217167..0.6000000000000001 | 750 | 0.3082 | 0.0423 |
