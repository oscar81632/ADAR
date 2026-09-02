# ADAR Gate Statistics

- Run: `runs/vit_adaptive_decoy_aware_fullft_vstd_decoy_fullgrid_test_6e_b16_lr1e-5_seed123`
- Dataset: `datasets/vstd_decoy_fullgrid_test`
- Split: `test`
- Samples: 3000
- Accuracy: 0.6523

## Correlation With Correctness

| Signal | r with correct |
|---|---:|
| alpha | 0.0245 |
| gate_mean | 0.0006 |
| gate_std | 0.0313 |
| path_gate_mean | -0.0326 |
| nonpath_gate_mean | 0.0078 |
| path_enrichment | -0.0518 |
| endpoint_gate | 0.1380 |
| decoy_gate_mean | 0.0121 |
| black_decoy_gate_mean | 0.0059 |
| gray_decoy_gate_mean | 0.0205 |

## Correlation With Metadata

### alpha

| Metadata | r |
|---|---:|
| path_length | 0.0189 |
| distractor_ratio | n/a |
| decoy_arrow_count | -0.0063 |
| start_end_manhattan | -0.0088 |
| num_turns | 0.0131 |
| path_bbox_area | -0.0101 |
| path_coverage | 0.0189 |
| long_dependency_score | -0.0038 |

### gate_mean

| Metadata | r |
|---|---:|
| path_length | -0.1313 |
| distractor_ratio | n/a |
| decoy_arrow_count | -0.0403 |
| start_end_manhattan | -0.0381 |
| num_turns | -0.1061 |
| path_bbox_area | -0.0905 |
| path_coverage | -0.1313 |
| long_dependency_score | -0.0917 |

### gate_std

| Metadata | r |
|---|---:|
| path_length | 0.0919 |
| distractor_ratio | n/a |
| decoy_arrow_count | -0.0022 |
| start_end_manhattan | 0.0314 |
| num_turns | 0.0750 |
| path_bbox_area | 0.0554 |
| path_coverage | 0.0919 |
| long_dependency_score | 0.0638 |

### path_gate_mean

| Metadata | r |
|---|---:|
| path_length | -0.0347 |
| distractor_ratio | n/a |
| decoy_arrow_count | 0.0145 |
| start_end_manhattan | 0.0061 |
| num_turns | -0.0345 |
| path_bbox_area | 0.0232 |
| path_coverage | -0.0347 |
| long_dependency_score | 0.0031 |

### nonpath_gate_mean

| Metadata | r |
|---|---:|
| path_length | -0.0872 |
| distractor_ratio | n/a |
| decoy_arrow_count | -0.0560 |
| start_end_manhattan | -0.0368 |
| num_turns | -0.0627 |
| path_bbox_area | -0.0775 |
| path_coverage | -0.0872 |
| long_dependency_score | -0.0741 |

### path_enrichment

| Metadata | r |
|---|---:|
| path_length | 0.0371 |
| distractor_ratio | n/a |
| decoy_arrow_count | 0.0738 |
| start_end_manhattan | 0.0438 |
| num_turns | 0.0138 |
| path_bbox_area | 0.1064 |
| path_coverage | 0.0371 |
| long_dependency_score | 0.0758 |

### endpoint_gate

| Metadata | r |
|---|---:|
| path_length | 0.0049 |
| distractor_ratio | n/a |
| decoy_arrow_count | -0.0081 |
| start_end_manhattan | 0.0224 |
| num_turns | 0.0032 |
| path_bbox_area | 0.0128 |
| path_coverage | 0.0049 |
| long_dependency_score | 0.0189 |

### decoy_gate_mean

| Metadata | r |
|---|---:|
| path_length | -0.0880 |
| distractor_ratio | n/a |
| decoy_arrow_count | -0.0558 |
| start_end_manhattan | -0.0373 |
| num_turns | -0.0633 |
| path_bbox_area | -0.0785 |
| path_coverage | -0.0880 |
| long_dependency_score | -0.0750 |

### black_decoy_gate_mean

| Metadata | r |
|---|---:|
| path_length | -0.1048 |
| distractor_ratio | n/a |
| decoy_arrow_count | -0.0420 |
| start_end_manhattan | -0.0531 |
| num_turns | -0.0745 |
| path_bbox_area | -0.0976 |
| path_coverage | -0.1048 |
| long_dependency_score | -0.0958 |

### gray_decoy_gate_mean

| Metadata | r |
|---|---:|
| path_length | -0.0061 |
| distractor_ratio | n/a |
| decoy_arrow_count | -0.0336 |
| start_end_manhattan | 0.0146 |
| num_turns | -0.0051 |
| path_bbox_area | 0.0001 |
| path_coverage | -0.0061 |
| long_dependency_score | 0.0067 |

## Correct vs Wrong Means

| Group | n | acc | alpha | path enrich. | path gate | endpoint gate | decoy gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| wrong | 1043 | 0.0000 | 0.3169 | -0.0068 | 0.5005 | 0.5129 | 0.5071 |
| correct | 1957 | 1.0000 | 0.3196 | -0.0075 | 0.4999 | 0.5207 | 0.5073 |
