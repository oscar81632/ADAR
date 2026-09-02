# Selected Vim+CLIP Exploration Results

This file summarizes representative outputs archived from the older `CLIP/` workspace.
Large checkpoints, datasets, and full run folders are intentionally excluded.

| Experiment | Dataset | Encoder | Method | Train/class | Epoch | Acc | Result |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `baseline_vit_b16_ft_caltech/class25` | caltech | vit | vit fine-tuning baseline | 25 | 17 | 0.9734 | `baseline_vit_b16_ft_caltech/class25/args/vit_vitb16_caltech101_final_test_result.json` |
| `ft_vim_head_outputs_cifar/1_class50` | cifar10 | vim | projection/head probing | 50 | 19 | 0.9232 | `ft_vim_head_outputs_cifar/1_class50/args/class50_cifar10_final_test_result.json` |
| `ft_vim_adapter_outputs_cifar/6_class50_best` | cifar10 | vim | vim adapter | 50 | 18 | 0.9404 | `ft_vim_adapter_outputs_cifar/6_class50_best/args/class50_cifar10_final_test_result.json` |
| `random_adapter_full_dim192_wide_class50/41_best_t041_lp0p000116012_lb7p34592em05_la0p000406183_ad0p163584_wd7p32702em06` | cifar10 | vim | vim adapter | 50 | 20 | 0.9412 | `random_adapter_full_dim192_wide_class50/41_best_t041_lp0p000116012_lb7p34592em05_la0p000406183_ad0p163584_wd7p32702em06/args/t041_lp0p000116012_lb7p34592em05_la0p000406183_ad0p163584_wd7p32702em06_cifar10_final_test_result.json` |
| `baseline_vim_b16_ft_cifar_dim128/A3_randaugment_ls005_ema095_magn12_epoch30_best/seed0` | cifar10 | vim | vim fine-tuning baseline | 50 | 29 | 0.9506 | `baseline_vim_b16_ft_cifar_dim128/A3_randaugment_ls005_ema095_magn12_epoch30_best/seed0/args/vim_cifar10_final_test_result.json` |
| `baseline_vim_b16_ft_cifar_dim128/A3_randaugment_ls005_ema095_magn12_epoch30_best/seed1` | cifar10 | vim | vim fine-tuning baseline | 50 | 28 | 0.9500 | `baseline_vim_b16_ft_cifar_dim128/A3_randaugment_ls005_ema095_magn12_epoch30_best/seed1/args/vim_cifar10_final_test_result.json` |
| `baseline_vim_b16_ft_cifar_dim128/A3_randaugment_ls005_ema095_magn12_epoch30_best/seed2` | cifar10 | vim | vim fine-tuning baseline | 50 | 27 | 0.9574 | `baseline_vim_b16_ft_cifar_dim128/A3_randaugment_ls005_ema095_magn12_epoch30_best/seed2/args/vim_cifar10_final_test_result.json` |
| `baseline_vim_b16_ft_cifar_dim128/A3_randaugment_ls005_ema095_magn12_epoch30_best/seed3` | cifar10 | vim | vim fine-tuning baseline | 50 | 30 | 0.9578 | `baseline_vim_b16_ft_cifar_dim128/A3_randaugment_ls005_ema095_magn12_epoch30_best/seed3/args/vim_cifar10_final_test_result.json` |
| `baseline_vim_b16_ft_cifar_dim128/A3_randaugment_ls005_ema095_magn12_epoch30_best/seed42` | cifar10 | vim | vim fine-tuning baseline | 50 | 30 | 0.9524 | `baseline_vim_b16_ft_cifar_dim128/A3_randaugment_ls005_ema095_magn12_epoch30_best/seed42/args/vim_cifar10_final_test_result.json` |
| `baseline_vim_b16_ft_cifar_dim96_trial14/seed0` | cifar10 | vim | vim fine-tuning baseline | 50 | 25 | 0.9496 | `baseline_vim_b16_ft_cifar_dim96_trial14/seed0/args/vim_cifar10_final_test_result.json` |
| `baseline_vim_b16_ft_cifar_dim96_trial14/seed1` | cifar10 | vim | vim fine-tuning baseline | 50 | 24 | 0.9514 | `baseline_vim_b16_ft_cifar_dim96_trial14/seed1/args/vim_cifar10_final_test_result.json` |
| `baseline_vim_b16_ft_cifar_dim96_trial14/seed2` | cifar10 | vim | vim fine-tuning baseline | 50 | 23 | 0.9580 | `baseline_vim_b16_ft_cifar_dim96_trial14/seed2/args/vim_cifar10_final_test_result.json` |
| `baseline_vim_b16_ft_cifar_dim96_trial14/seed3` | cifar10 | vim | vim fine-tuning baseline | 50 | 29 | 0.9546 | `baseline_vim_b16_ft_cifar_dim96_trial14/seed3/args/vim_cifar10_final_test_result.json` |
| `ft_vim_B16_outputs_cifar/1_all` | cifar10 | vim | vim fine-tuning baseline | None | 7 | 0.9804 | `ft_vim_B16_outputs_cifar/1_all/args/t002_lp0p000116012_lb1p43622em05_wd0_cifar10_final_test_result.json` |
| `ft_vim_B16_outputs_cifar/2_class4000` | cifar10 | vim | vim fine-tuning baseline | 4000 | 13 | 0.9790 | `ft_vim_B16_outputs_cifar/2_class4000/args/class4000_cifar10_final_test_result.json` |
| `ft_vim_B16_outputs_cifar/3_class3000` | cifar10 | vim | vim fine-tuning baseline | 3000 | 3 | 0.9740 | `ft_vim_B16_outputs_cifar/3_class3000/args/class3000_cifar10_final_test_result.json` |
| `ft_vim_B16_outputs_cifar/4_class2000` | cifar10 | vim | vim fine-tuning baseline | 2000 | 5 | 0.9748 | `ft_vim_B16_outputs_cifar/4_class2000/args/class2000_cifar10_final_test_result.json` |
| `ft_vim_B16_outputs_cifar/5_class1000` | cifar10 | vim | vim fine-tuning baseline | 1000 | 8 | 0.9682 | `ft_vim_B16_outputs_cifar/5_class1000/args/class1000_cifar10_final_test_result.json` |
| `ft_vim_B16_outputs_cifar/6_class500` | cifar10 | vim | vim fine-tuning baseline | 500 | 10 | 0.9626 | `ft_vim_B16_outputs_cifar/6_class500/args/class500_cifar10_final_test_result.json` |
| `ft_vim_B16_outputs_cifar/7_class100` | cifar10 | vim | vim fine-tuning baseline | 100 | 20 | 0.9360 | `ft_vim_B16_outputs_cifar/7_class100/args/class100_cifar10_final_test_result.json` |
| `ft_vim_B16_outputs_cifar/8_class50` | cifar10 | vim | vim fine-tuning baseline | 50 | 19 | 0.9152 | `ft_vim_B16_outputs_cifar/8_class50/args/class50_cifar10_final_test_result.json` |
| `ft_vim_B16_outputs_cifar/9_class10` | cifar10 | vim | vim fine-tuning baseline | 10 | 20 | 0.8288 | `ft_vim_B16_outputs_cifar/9_class10/args/class10_cifar10_final_test_result.json` |
| `ft_vim_lora_outputs_cifar/10_class50` | cifar10 | vim | vim lora | 50 | 20 | 0.1764 | `ft_vim_lora_outputs_cifar/10_class50/args/class50_cifar10_final_test_result.json` |
| `ft_vim_lora_outputs_cifar/11_class50` | cifar10 | vim | vim lora | 50 | 20 | 0.2850 | `ft_vim_lora_outputs_cifar/11_class50/args/class50_cifar10_final_test_result.json` |
| `ft_vim_lora_outputs_cifar/12_class50` | cifar10 | vim | vim lora | 50 | 18 | 0.2894 | `ft_vim_lora_outputs_cifar/12_class50/args/class50_cifar10_final_test_result.json` |
| `ft_vim_lora_outputs_cifar/1_class50_all_outproj_rank8_alpha16_lr3e-4` | cifar10 | vim | vim lora | 50 | 28 | 0.3196 | `ft_vim_lora_outputs_cifar/1_class50_all_outproj_rank8_alpha16_lr3e-4/args/class50_cifar10_final_test_result.json` |
| `ft_vim_lora_outputs_cifar/2_class50_last12_outproj_rank4_alpha4_lr1e-5` | cifar10 | vim | vim lora | 50 | 20 | 0.8460 | `ft_vim_lora_outputs_cifar/2_class50_last12_outproj_rank4_alpha4_lr1e-5/args/class50_cifar10_final_test_result.json` |
| `ft_vim_lora_outputs_cifar/3_class50_last12_outproj_rank4_alpha1_lr1e-6_decay0` | cifar10 | vim | vim lora | 50 | 27 | 0.8482 | `ft_vim_lora_outputs_cifar/3_class50_last12_outproj_rank4_alpha1_lr1e-6_decay0/args/class50_cifar10_final_test_result.json` |
| `ft_vim_lora_outputs_cifar/4_class50` | cifar10 | vim | vim lora | 50 | 14 | 0.8440 | `ft_vim_lora_outputs_cifar/4_class50/args/class50_cifar10_final_test_result.json` |
| `ft_vim_lora_outputs_cifar/5_class50` | cifar10 | vim | vim lora | 50 | 20 | 0.8496 | `ft_vim_lora_outputs_cifar/5_class50/args/class50_cifar10_final_test_result.json` |
| `ft_vim_lora_outputs_cifar/6_class50` | cifar10 | vim | vim lora | 50 | 14 | 0.8440 | `ft_vim_lora_outputs_cifar/6_class50/args/class50_cifar10_final_test_result.json` |
| `ft_vim_lora_outputs_cifar/7_class50` | cifar10 | vim | vim lora | 50 | 20 | 0.8492 | `ft_vim_lora_outputs_cifar/7_class50/args/class50_cifar10_final_test_result.json` |
| `ft_vim_lora_outputs_cifar/8_class50` | cifar10 | vim | vim lora | 50 | 14 | 0.8440 | `ft_vim_lora_outputs_cifar/8_class50/args/class50_cifar10_final_test_result.json` |
| `ft_vim_lora_outputs_cifar/9_class50` | cifar10 | vim | vim lora | 50 | 1 | 0.1392 | `ft_vim_lora_outputs_cifar/9_class50/args/class50_cifar10_final_test_result.json` |
| `ft_vim_3MLP_outputs_cifar/9_class50_2xdrop_1024_768_drop0.35_0.55_best` | cifar10 | vim | vim mlp head | 50 | 20 | 0.9376 | `ft_vim_3MLP_outputs_cifar/9_class50_2xdrop_1024_768_drop0.35_0.55_best/args/class50_cifar10_final_test_result.json` |
| `baseline_vit_b16_ft_cifar/class50/1_bad_epoch20` | cifar10 | vit | vit fine-tuning baseline | 50 | 5 | 0.9268 | `baseline_vit_b16_ft_cifar/class50/1_bad_epoch20/args/vit_vitb16_cifar10_final_test_result.json` |
| `baseline_vit_b16_ft_cifar/class50/2_bad_epoch30` | cifar10 | vit | vit fine-tuning baseline | 50 | 27 | 0.9334 | `baseline_vit_b16_ft_cifar/class50/2_bad_epoch30/args/vit_vitb16_cifar10_final_test_result.json` |
| `baseline_vit_b16_ft_cifar/class50/3_good_epoch20` | cifar10 | vit | vit fine-tuning baseline | 50 | 9 | 0.9560 | `baseline_vit_b16_ft_cifar/class50/3_good_epoch20/args/vit_vitb16_cifar10_final_test_result.json` |
| `baseline_vit_b16_ft_cifar/class50/4_good_epoch30` | cifar10 | vit | vit fine-tuning baseline | 50 | 9 | 0.9560 | `baseline_vit_b16_ft_cifar/class50/4_good_epoch30/args/vit_vitb16_cifar10_final_test_result.json` |
| `oxfordpets_trial14_class50/drop020` | oxfordpets | vim | other | 50 | 25 | 0.9351 | `oxfordpets_trial14_class50/drop020/args/vim_oxfordpets_final_test_result.json` |
| `random_vim_dim96_oxfordpets_class50_contrastive_narrow/14_t014_lp0p000180303_lb2p4467em05_la0p00033377_ad0p188766_wd4p4429em07` | oxfordpets | vim | other | 50 | 25 | 0.9362 | `random_vim_dim96_oxfordpets_class50_contrastive_narrow/14_t014_lp0p000180303_lb2p4467em05_la0p00033377_ad0p188766_wd4p4429em07/args/t014_lp0p000180303_lb2p4467em05_la0p00033377_ad0p188766_wd4p4429em07_oxfordpets_final_test_result.json` |
| `ft_vim_B16_outputs_oxfordpets/1_all` | oxfordpets | vim | vim fine-tuning baseline | None | 15 | 0.9221 | `ft_vim_B16_outputs_oxfordpets/1_all/args/t006_lp5p11743em05_lb2p53826em05_wd0p000188046_oxfordpets_final_test_result.json` |
| `ft_vim_B16_outputs_oxfordpets/2_class50` | oxfordpets | vim | vim fine-tuning baseline | 50 | 8 | 0.9106 | `ft_vim_B16_outputs_oxfordpets/2_class50/args/class50_oxfordpets_final_test_result.json` |
| `ft_vim_B16_outputs_oxfordpets/3_class10` | oxfordpets | vim | vim fine-tuning baseline | 10 | 18 | 0.8272 | `ft_vim_B16_outputs_oxfordpets/3_class10/args/class10_oxfordpets_final_test_result.json` |
| `baseline_vit_b16_ft_oxfordpets/class50` | oxfordpets | vit | vit fine-tuning baseline | 50 | 10 | 0.9504 | `baseline_vit_b16_ft_oxfordpets/class50/args/vit_vitb16_oxfordpets_final_test_result.json` |
| `baseline_vit_b16_ft_stl10/class50` | stl10 | vit | vit fine-tuning baseline | 50 | 1 | 0.9785 | `baseline_vit_b16_ft_stl10/class50/args/vit_vitb16_stl10_final_test_result.json` |
