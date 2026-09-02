# Exploratory Modern VLM Baselines

These runs are intentionally separated from the main ADAR thesis results. They
test whether stronger CLIP-style pretrained models can solve VSTD with the same
vanilla image-text endpoint objective.

## Protocol

- Dataset: `datasets/vstd_decoy_trainhard`
- Train/validation/test split: same as the thesis main setting
- Epochs: 6
- Batch size: 16
- Evaluation batch size: 64
- Optimizer: AdamW
- Learning rate: `1e-5`
- Weight decay: `1e-4`
- Seed: 123
- Text side: frozen
- Trainable side: visual encoder and visual projection only
- Loss: cross-entropy over the original endpoint prompts
- Unless explicitly marked as `+ ADAR`, no ADAR, no path adapter, no
  hard-negative prompts, and no data changes are used.
- GPU: NVIDIA GeForce RTX 4090

## Results

| Model | Params | Trainable Params | Best Epoch | Best Val Acc | Test Acc | L=48 Acc | Train Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| SigLIP Base B/16 (`google/siglip-base-patch16-224`) | 203.16M | 92.88M | 4 | 0.9880 | 0.9160 | 0.8830 | 305.8s |
| SigLIP Base B/16 + ADAR (`residual_scale=1.0`) | 203.56M | 93.28M | 3 | 0.9860 | 0.8627 | 0.8388 | 320.6s |
| SigLIP Base B/16 + ADAR (`residual_scale=0.25`) | 203.56M | 93.28M | 6 | 0.9875 | 0.8960 | 0.8845 | 309.1s |
| EVA02-CLIP-B/16 (`EVA02-B-16`, `merged2b_s8b_b131k`) | 149.69M | 86.26M | 6 | 0.9970 | 0.8923 | 0.8916 | 271.3s |

For reference only, the main thesis ADAR three-seed mean is `0.8846` test
accuracy and `0.8478` on the L=48 bucket. These exploratory runs should not be
merged into the main thesis result tables unless we decide to reframe the
baseline story.

## Path-Length Breakdown

| Model | L=24 | L=32 | L=40 | L=48 |
|---|---:|---:|---:|---:|
| SigLIP Base B/16 | 0.9532 | 0.9270 | 0.8972 | 0.8830 |
| SigLIP Base B/16 + ADAR (`residual_scale=1.0`) | 0.8960 | 0.8809 | 0.8318 | 0.8388 |
| SigLIP Base B/16 + ADAR (`residual_scale=0.25`) | 0.9324 | 0.9117 | 0.8531 | 0.8845 |
| EVA02-CLIP-B/16 | 0.8882 | 0.8937 | 0.8959 | 0.8916 |

## OpenAI CLIP ADAR Variants Inspired by SigLIP

SigLIP suggested a representation-preserving design: keep the pretrained global
image representation as the main pathway and let an image-dependent coefficient
control how much patch evidence is injected. To test this idea without changing
the thesis mainline, an exploratory readout mode named
`adaptive_residual_decoy_aware` was added to the repo-local CLIP implementation.

The new variant differs from the main ADAR readout as follows:

```text
Main ADAR:
z = z_cls + lambda * F([z_cls; alpha*z_keep; alpha*z_supp])

Adaptive-residual ADAR:
z = z_cls + lambda * alpha * F([z_cls; alpha*z_keep; alpha*z_supp])
```

The second form makes the residual itself image-adaptive. Both runs use the
same OpenAI CLIP ViT-B/16 backbone, same 6-epoch full fine-tuning protocol, and
same seed 123 as the thesis baseline.

| Method | Residual Scale | Best Epoch | Best Val Acc | Test Acc | L=48 Acc |
|---|---:|---:|---:|---:|---:|
| ViT-B/16 vanilla, seed 123 | - | 6 | 0.9760 | 0.8063 | 0.7347 |
| Main ADAR, seed 123 | 1.0 | 3 | 0.9845 | 0.8613 | 0.8188 |
| Adaptive-residual ADAR | 1.0 | 4 | 0.9775 | 0.8527 | 0.8117 |
| Adaptive-residual ADAR | 0.25 | 5 | 0.9825 | 0.8397 | 0.7917 |
| Adaptive keep-only ADAR, no `z_supp` | 1.0 | 6 | 0.9645 | 0.8107 | 0.7589 |
| Adaptive keep-only ADAR, 2-branch | 1.0 | 6 | 0.9550 | 0.7537 | 0.6748 |
| Adaptive supp-only ADAR, no `z_keep`, seed 123 | 1.0 | 6 | 0.9775 | 0.8757 | 0.8231 |
| ADAR with ReLU `z_supp`, seed 123 | 1.0 | 5 | 0.9870 | 0.8620 | 0.8046 |
| ADAR with mean `z_supp`, seed 123 | 1.0 | 5 | 0.9890 | 0.8663 | 0.8274 |
| ADAR without outer `z_cls`, seed 123 | 1.0 | 5 | 0.9555 | 0.7303 | 0.6305 |
| ADAR without inner `z_cls`, seed 123 | 1.0 | 5 | 0.9905 | 0.8753 | 0.8245 |
| ADAR with static learnable `alpha`, seed 123 | 1.0 | 5 | 0.9895 | 0.8653 | 0.8188 |
| Attention-pooling ADAR | 1.0 | 6 | 0.9850 | 0.8303 | 0.7789 |
| Decoy-contrast ADAR | 1.0 | 5 | 0.9895 | 0.8753 | 0.8160 |
| Multi-query ADAR | 1.0 | 5 | 0.9845 | 0.8463 | 0.8074 |
| Late-fusion ADAR | logits-level | 5 | 0.9870 | 0.8487 | 0.8088 |

The adaptive-residual version improves over vanilla ViT-B/16 but does not beat
the best ADAR mainline. This suggests that the original ADAR branch-level
routing is better suited to OpenAI CLIP ViT-B/16 than adding an additional
image-level residual gate.

Among the additional exploratory variants, decoy-contrast ADAR is the strongest
single run (`0.8753` test accuracy). However, it still does not surpass the
main ADAR three-seed mean (`0.8846`) or the best main ADAR seed (`0.9117`).
The keep-only ablations remove the suppressed-evidence summary. The zero-fill
version feeds `F([z_cls; alpha*z_keep; 0])`, while the stricter 2-branch version
uses a separate `F_2([z_cls; alpha*z_keep])` MLP. Both variants degrade, and the
2-branch version performs even worse than vanilla ViT-B/16. Conversely, the
supp-only ablation removes the selected-evidence summary and performs strongly
on this seed. This suggests that `z_supp` is not merely an auxiliary feature:
for VSTD, complementary evidence can itself be highly informative, likely
because decoy-heavy samples are defined by the contrast between selected and
non-selected visual evidence.
Attention-pooling, multi-query pooling, and logits-level late fusion all improve
over vanilla ViT-B/16 but do not provide a cleaner thesis mainline than the
current ADAR design.

The ReLU `z_supp` variant clamps the complementary residual summary to
non-negative components, using `relu(mean(patches) - z_keep)`. Its seed-123
overall accuracy is close to the main ADAR seed-123 result, but its longest-path
accuracy is lower. This suggests that the signed residual representation may
carry useful directional information for long-path cases; positive residual
components alone should not be interpreted as a clean decoy summary.

The mean-`z_supp` variant replaces the residual branch with the global patch
mean, using `z_supp = mean(patches)` rather than `mean(patches) - z_keep`. The
single-seed result was slightly above the main ADAR seed-123 run, and the
three-seed check below remains competitive with the main ADAR mean. This
suggests that the third branch may be useful as a global patch-context branch
in addition to, or instead of, a strict complementary residual branch.

### Mean-Context Branch Three-Seed Check

| Seed | Best Epoch | Best Val Acc | Test Acc | L=48 Acc |
|---:|---:|---:|---:|---:|
| 111 | 6 | 0.9835 | 0.8627 | 0.8417 |
| 123 | 5 | 0.9890 | 0.8663 | 0.8274 |
| 321 | 6 | 0.9910 | 0.8957 | 0.8645 |
| Mean | - | - | 0.8749 | 0.8445 |
| Std. Dev. | - | - | 0.0148 | 0.0153 |

Compared with the main ADAR three-seed result (`0.8846` test accuracy and
`0.8478` on L=48), this variant is slightly lower overall but very close on
long-path accuracy. This makes the mean-context interpretation a plausible
simplification, although the thesis mainline is kept unchanged because the
original residual formulation remains the reported best mean result.

Two additional checks isolate the class-token conditioning and routing design.
Removing the inner class-token input from the fusion MLP uses
`z_cls + lambda * F_2([alpha*z_keep; alpha*z_supp])`. Replacing the
image-conditioned router `alpha=sigmoid(r(z_cls))` with a single static
learnable scalar keeps the three-branch fusion but makes `alpha` shared by all
images. The single-seed results looked competitive, but the three-seed checks
below show that these simplifications are less reliable than the main ADAR
configuration.

### Inner-Class-Token and Static-Alpha Three-Seed Checks

| Method | Seed | Best Epoch | Best Val Acc | Test Acc | L=48 Acc |
|---|---:|---:|---:|---:|---:|
| ADAR w/o inner `z_cls` | 111 | 5 | 0.9860 | 0.8597 | 0.8288 |
| ADAR w/o inner `z_cls` | 123 | 5 | 0.9905 | 0.8753 | 0.8245 |
| ADAR w/o inner `z_cls` | 321 | 6 | 0.9500 | 0.7367 | 0.6748 |
| ADAR w/o inner `z_cls`, mean | - | - | - | 0.8239 | 0.7760 |
| ADAR w/o inner `z_cls`, std. dev. | - | - | - | 0.0620 | 0.0716 |
| Static-alpha ADAR | 111 | 6 | 0.9865 | 0.8660 | 0.8288 |
| Static-alpha ADAR | 123 | 5 | 0.9895 | 0.8653 | 0.8188 |
| Static-alpha ADAR | 321 | 5 | 0.9830 | 0.8430 | 0.7732 |
| Static-alpha ADAR, mean | - | - | - | 0.8581 | 0.8069 |
| Static-alpha ADAR, std. dev. | - | - | - | 0.0107 | 0.0242 |

The no-inner-`z_cls` variant is unstable across seeds and should not replace the
main formulation. Static alpha is more stable, but it remains below the main
ADAR three-seed mean (`0.8846` test accuracy and `0.8478` on L=48). This
supports keeping the image-conditioned router in the thesis mainline, while
also suggesting that the main performance gain is primarily driven by the
gated selected/complementary evidence construction.

An additional seed-123 check removes the outer residual class-token anchor,
using only `lambda * F([z_cls; alpha*z_keep; alpha*z_supp])` as the visual
representation. This causes a large drop (`0.7303` test accuracy and `0.6305`
on L=48), supporting the residual-adapter interpretation: ADAR should correct
the original CLIP class-token representation rather than replace it outright.

### Adaptive Supp-Only Three-Seed Check

The strong seed-123 supp-only result was rerun with the same three seeds used in
the thesis main comparison. This variant removes the selected-evidence summary
and uses the complementary suppressed-evidence branch with the original
class-token residual.

| Seed | Best Epoch | Best Val Acc | Test Acc | L=48 Acc |
|---:|---:|---:|---:|---:|
| 111 | 6 | 0.9915 | 0.9090 | 0.9001 |
| 123 | 6 | 0.9775 | 0.8757 | 0.8231 |
| 321 | 6 | 0.9915 | 0.9060 | 0.8759 |
| Mean | - | - | 0.8969 | 0.8664 |
| Std. Dev. | - | - | 0.0151 | 0.0322 |

This is a surprisingly strong exploratory result: the supp-only variant slightly
exceeds the current three-seed ADAR mean (`0.8846` test accuracy and `0.8478`
L=48 accuracy). It should be treated carefully because it changes the method
story: instead of balanced selected/complementary evidence, the result suggests
that the complementary branch alone captures much of the discriminative signal
on VSTD. Before promoting it to the thesis mainline, it would be worth checking
whether this behavior is stable under a cleanly named architecture and whether
the qualitative explanation remains defensible.

### Clean Supp-Only Architecture Check

To verify whether the strong supp-only result comes from the idea itself rather
than unused branches in the original three-branch fusion module, a clean
two-branch supp-only readout was added. This architecture only uses the class
token and the complementary suppressed-evidence summary:

```text
z = z_cls + lambda * F_2([z_cls; alpha*z_supp])
```

It removes the selected-evidence branch and avoids constructing an unused
`z_keep` input in the final fusion module.

| Seed | Best Epoch | Best Val Acc | Test Acc | L=48 Acc |
|---:|---:|---:|---:|---:|
| 111 | 4 | 0.9220 | 0.6823 | 0.6320 |
| 123 | 6 | 0.9760 | 0.8040 | 0.7233 |
| 321 | 6 | 0.9870 | 0.8870 | 0.8802 |
| Mean | - | - | 0.7911 | 0.7451 |
| Std. Dev. | - | - | 0.0841 | 0.1025 |

This clean check does not reproduce the strong three-branch supp-only result.
The result suggests that the original supp-only ablation should be interpreted
as evidence that `z_supp` is useful, not as a clean replacement for the main
ADAR design. The much larger variance also makes it a poor thesis mainline
candidate without additional architectural work.

### Clean Keep-Only Architecture Check

For symmetry, a clean two-branch keep-only readout was also run with three
seeds. This variant removes the complementary suppressed-evidence branch and
only uses the class token and selected kept-evidence summary:

```text
z = z_cls + lambda * F_2([z_cls; alpha*z_keep])
```

| Seed | Best Epoch | Best Val Acc | Test Acc | L=48 Acc |
|---:|---:|---:|---:|---:|
| 111 | 6 | 0.9640 | 0.7703 | 0.7147 |
| 123 | 6 | 0.9880 | 0.8817 | 0.8616 |
| 321 | 4 | 0.9805 | 0.8780 | 0.8445 |
| Mean | - | - | 0.8433 | 0.8069 |
| Std. Dev. | - | - | 0.0516 | 0.0656 |

The clean keep-only variant is stronger than the clean supp-only variant, but
it still trails the full ADAR three-seed result. This supports the thesis
interpretation that selected and complementary evidence are both useful: kept
evidence gives the main discriminative signal, while suppressed evidence
provides an additional contrastive summary that improves robustness on
decoy-heavy and long-path samples.

## Output Directories

- SigLIP:
  `runs/exploratory_vlm_baselines/siglip_base_b16_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5`
- SigLIP + ADAR:
  `runs/exploratory_vlm_baselines/siglip_base_b16_adar_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5`
- SigLIP + ADAR, conservative residual:
  `runs/exploratory_vlm_baselines/siglip_base_b16_adar025_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5`
- EVA02-CLIP:
  `runs/exploratory_vlm_baselines/eva02_clip_b16_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5`
- OpenAI CLIP adaptive-residual ADAR:
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_residual_adar_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123`
- OpenAI CLIP adaptive-residual ADAR, conservative residual:
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_residual_adar025_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123`
- OpenAI CLIP adaptive keep-only ADAR, no `z_supp`:
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_keep_only_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123`
- OpenAI CLIP adaptive keep-only ADAR, 2-branch:
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_keep_only_2branch_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123`
- OpenAI CLIP adaptive supp-only ADAR, no `z_keep`:
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_supp_only_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123`
- OpenAI CLIP adaptive supp-only ADAR, no `z_keep`, additional seeds:
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_supp_only_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111`
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_supp_only_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed321`
- OpenAI CLIP ADAR with ReLU `z_supp`:
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_relu_supp_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123`
- OpenAI CLIP ADAR with mean `z_supp`:
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_mean_supp_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123`
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_mean_supp_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111`
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_mean_supp_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed321`
- OpenAI CLIP ADAR without outer `z_cls`:
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_no_outer_cls_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123`
- OpenAI CLIP ADAR without inner `z_cls`:
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_no_inner_cls_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123`
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_no_inner_cls_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111`
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_no_inner_cls_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed321`
- OpenAI CLIP ADAR with static learnable `alpha`:
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_static_alpha_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123`
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_static_alpha_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111`
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_static_alpha_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed321`
- OpenAI CLIP clean adaptive supp-only ADAR, no unused branch:
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_supp_only_clean_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111`
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_supp_only_clean_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123`
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_supp_only_clean_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed321`
- OpenAI CLIP clean adaptive keep-only ADAR, no unused branch:
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_keep_only_clean_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed111`
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_keep_only_clean_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123`
  `runs/exploratory_vlm_baselines/vit_b16_adaptive_keep_only_clean_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed321`
- OpenAI CLIP attention-pooling ADAR:
  `runs/exploratory_vlm_baselines/vit_b16_experimental_attention_pool_adar_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123`
- OpenAI CLIP decoy-contrast ADAR:
  `runs/exploratory_vlm_baselines/vit_b16_experimental_decoy_contrast_adar_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123`
- OpenAI CLIP multi-query ADAR:
  `runs/exploratory_vlm_baselines/vit_b16_experimental_multiquery_adar_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123`
- OpenAI CLIP late-fusion ADAR:
  `runs/exploratory_vlm_baselines/vit_b16_experimental_late_fusion_adar_fullft_vstd_decoy_trainhard_6e_b16_lr1e-5_seed123`

The `runs/` directory is ignored by git, so the large checkpoints should not be
committed by default.

## Notes

- SigLIP Base slightly exceeds the ADAR single-seed result on overall test
  accuracy in this exploratory setting.
- EVA02-CLIP-B/16 is closer in total parameter count to OpenAI CLIP ViT-B/16
  and still outperforms the main three-seed ADAR mean.
- Directly attaching the OpenAI-CLIP-style ADAR readout to SigLIP does not
  improve the vanilla SigLIP baseline. A full residual (`residual_scale=1.0`)
  hurts both overall and L=48 accuracy, while a smaller residual
  (`residual_scale=0.25`) nearly recovers L=48 accuracy but still lowers overall
  test accuracy. This suggests that SigLIP's native attention pooling is already
  strong and that a SigLIP-specific readout design would be needed.
- Adding a second image-level residual gate to the OpenAI CLIP ADAR mainline is
  also not better than the original ADAR design. The useful part of ADAR appears
  to be branch-level path/decoy routing rather than globally suppressing the
  residual.
- Removing the suppressed-evidence summary causes a large drop from the main
  ADAR seed-123 result. The zero-fill no-`z_supp` version drops from `0.8613`
  to `0.8107` test accuracy, while the stricter 2-branch version drops to
  `0.7537`. On L=48, the corresponding scores are `0.8188`, `0.7589`, and
  `0.6748`. This supports keeping `z_supp` in the main formulation.
- Removing the selected-evidence summary produces a surprisingly strong
  result. Across three seeds, the supp-only variant reaches `0.8969` test
  accuracy and `0.8664` on L=48. This strengthens the interpretation that
  `z_supp` carries meaningful complementary information, but it also suggests
  the current ADAR story may need refinement if this variant is promoted.
- A clean two-branch supp-only architecture does not reproduce that strong
  result, reaching only `0.7911` mean test accuracy with high variance. This
  makes the original balanced ADAR design safer as the thesis mainline, while
  the supp-only ablation remains useful evidence for the importance of
  complementary evidence.
- A clean two-branch keep-only architecture reaches `0.8433` mean test accuracy
  and `0.8069` on L=48. This is better than clean supp-only but still below
  full ADAR, supporting the use of both `z_keep` and `z_supp` in the final
  formulation.
- The SigLIP-inspired OpenAI CLIP ablations do not produce a better replacement
  for the thesis mainline. Decoy-contrast pooling is a useful negative result:
  explicitly subtracting a decoy summary helps over vanilla ViT-B/16, but the
  original adaptive branch routing remains more stable.
- These results suggest that newer CLIP-family pretraining recipes can provide
  stronger vanilla image-text alignment on VSTD, but they do not invalidate the
  ADAR architecture story unless they are introduced into the main thesis
  comparison protocol.
