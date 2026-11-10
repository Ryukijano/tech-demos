# exp1_cpu_token_quant

Status: **FAIL**

Means are over seeds 0/1/2. ± is the sample standard deviation (ddof=1).

## Gates

| Gate | Result |
|---|---|
| prop1 | met |
| common_mode | met |
| token_blind | met |
| restore_special | not met |
| restore_metric | not met |
| still_degrades | met |
| row_selective | not met |
| speed | met |
| w8_order | not met |

## Per seed, W4A4

| seed | method | special_rel | patch_rel | metric_rel | special_mse | latency_s |
|---|---|---:|---:|---:|---:|---:|
| 0 | naive | 0.8951 | 0.6308 | 0.3519 | 0.027819 | 0.011834 |
| 0 | rotation | 1.0966 | 0.2687 | 0.2523 | 0.037813 | 0.026235 |
| 0 | uniform | 1.0891 | 0.2650 | 0.2537 | 0.036957 | 0.015412 |
| 0 | qmap | 1.0436 | 0.2652 | 0.2665 | 0.034167 | 0.025883 |
| 1 | naive | 1.1854 | 0.7211 | 0.8460 | 0.042999 | 0.013001 |
| 1 | rotation | 1.4064 | 0.4086 | 0.2816 | 0.047005 | 0.026121 |
| 1 | uniform | 1.3721 | 0.4045 | 0.2792 | 0.044824 | 0.015362 |
| 1 | qmap | 1.2818 | 0.4046 | 0.2579 | 0.040251 | 0.025908 |
| 2 | naive | 1.0964 | 0.7027 | 0.6552 | 0.034490 | 0.012114 |
| 2 | rotation | 1.1992 | 0.3716 | 0.2601 | 0.037202 | 0.026188 |
| 2 | uniform | 1.1830 | 0.3693 | 0.2553 | 0.035655 | 0.015475 |
| 2 | qmap | 1.1367 | 0.3691 | 0.2446 | 0.032921 | 0.026156 |

## Means

- fp_metric: 0.018636
- snr_ratio_rel_change: 9.26811e-08
- delta_shrink_qkv_fc1: 4.62762
- w4_special_naive: 1.05896
- w4_special_rot: 1.23407
- w4_special_qmap: 1.15402
- w4_patch_naive: 0.684848
- w4_patch_rot: 0.349615
- w4_metric_naive: 0.617675
- w4_metric_rot: 0.264643
- w4_metric_qmap: 0.256312
- w4_mse_rot: 0.0406734
- w4_mse_uniform: 0.0391452
- w4_mse_qmap: 0.0357796
- w8_metric_naive: 0.0282027
- w8_metric_rot: 0.0235501
- w8_metric_qmap: 0.0242475
- w8_latency_ratio: 1.02643
- inv_kappa_qkv: 6.02
- gamma: 280518
- eta_w4: 0.176857

Thresholds are the `thresholds` block of config.yaml. This file is generated.
