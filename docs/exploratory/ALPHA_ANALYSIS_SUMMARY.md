# ADAR Alpha Analysis Summary

This exploratory analysis measures the image-level routing coefficient `alpha`
from the ADAR router on the VSTD test split.

## Runs

| Seed | Test acc. | Alpha mean | Alpha std. | corr(alpha, correct) | corr(alpha, path length) | corr(alpha, decoy count) | corr(alpha, long-dep. score) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 111 | 0.9117 | 0.3117 | 0.0433 | 0.0623 | -0.0781 | -0.0906 | -0.0582 |
| 123 | 0.8613 | 0.4822 | 0.0273 | 0.0311 | 0.0819 | -0.0030 | 0.0805 |
| 321 | 0.8807 | 0.4349 | 0.0333 | 0.0517 | -0.0221 | 0.0012 | 0.0146 |

## Interpretation

The absolute value of `alpha` is not directly comparable across random seeds:
different runs learn different calibration ranges. Within each run, however,
correct predictions have slightly higher `alpha` than wrong predictions.

| Seed | Wrong alpha mean | Correct alpha mean | Difference |
|---:|---:|---:|---:|
| 111 | 0.3030 | 0.3125 | +0.0095 |
| 123 | 0.4801 | 0.4826 | +0.0025 |
| 321 | 0.4302 | 0.4355 | +0.0053 |

The relationship between `alpha` and explicit difficulty metadata is weak and
not stable across seeds. In particular, path length, decoy count, and the
long-dependency score do not show a consistent correlation sign. This suggests
that the router should not be described as a calibrated difficulty estimator.
It is better described as an image-conditioned coefficient that controls how
strongly the ADAR branch modifies the original CLIP image representation.

Full per-image records are saved under:

- `docs/exploratory/alpha_analysis_seed111/alpha_records.csv`
- `docs/exploratory/alpha_analysis_seed123/alpha_records.csv`
- `docs/exploratory/alpha_analysis_seed321/alpha_records.csv`
