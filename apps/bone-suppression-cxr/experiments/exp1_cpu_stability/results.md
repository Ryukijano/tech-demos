# exp1_cpu_stability results

**Status: PARTIAL** (precondition met; SSIM criterion not met on all seeds; latency criterion met on all seeds)

Thresholds (fixed before running): mean ssim_bone >= 0.95, mean latency_s_pipeline <= 15.0 s, per seed.

## Per seed (mean over 4 images)

| seed | mean_ssim_bone | min_ssim_bone | mean_ssim_lung | mean_latency_s | max_latency_s | ssim_pass | latency_pass |
|---|---|---|---|---|---|---|---|
| 0 | 0.9453 | 0.9115 | 0.9358 | 10.37 | 12.62 | False | True |
| 1 | 0.9419 | 0.8970 | 0.9356 | 10.81 | 13.30 | False | True |
| 2 | 0.9445 | 0.9047 | 0.9326 | 9.99 | 19.48 | False | True |

## Per image

| seed | image | ssim_bone | psnr_bone_db | ssim_lung | t_bone_s | t_lung_s | latency_s_pipeline |
|---|---|---|---|---|---|---|---|
| 0 | JPCLN030 | 0.9115 | 33.6 | 0.9129 | 4.94 | 7.68 | 12.62 |
| 0 | JPCNN001 | 0.9617 | 35.5 | 0.8856 | 6.33 | 4.55 | 10.88 |
| 0 | cxr_train_1_a_1 | 0.9542 | 30.1 | 0.9688 | 4.32 | 4.44 | 8.76 |
| 0 | cxr_train_8_a_2 | 0.9539 | 31.8 | 0.9759 | 4.47 | 4.74 | 9.21 |
| 1 | JPCLN030 | 0.8970 | 30.1 | 0.9043 | 4.27 | 4.34 | 8.60 |
| 1 | JPCNN001 | 0.9604 | 35.0 | 0.8897 | 4.35 | 4.05 | 8.40 |
| 1 | cxr_train_1_a_1 | 0.9527 | 30.6 | 0.9685 | 4.49 | 8.44 | 12.93 |
| 1 | cxr_train_8_a_2 | 0.9573 | 32.0 | 0.9798 | 8.19 | 5.11 | 13.30 |
| 2 | JPCLN030 | 0.9047 | 31.9 | 0.8982 | 3.65 | 15.83 | 19.48 |
| 2 | JPCNN001 | 0.9618 | 35.2 | 0.8877 | 3.48 | 3.50 | 6.97 |
| 2 | cxr_train_1_a_1 | 0.9539 | 30.2 | 0.9645 | 3.27 | 3.67 | 6.93 |
| 2 | cxr_train_8_a_2 | 0.9577 | 31.7 | 0.9798 | 3.47 | 3.08 | 6.55 |

## Precondition: reproduce the authors' published bone PNGs

| image | mean abs diff (8-bit levels) |
|---|---|
| JPCLN030 | 0.00001 |
| JPCNN001 | 0.00001 |

## Runtime

- cold first call: bone 6.0 s, lung 5.6 s (excluded); load: bone 1.0 s, lung 0.7 s
- env: {'python': '3.13.5', 'torch': '2.14.1+cpu', 'threads': 8, 'cpu': 'x86_64', 'nproc': 8, 'loadavg_start': [2.17919921875, 11.4931640625, 72.75927734375], 'started': '2026-10-11T09:24:22+0100', 'maxrss_mb': 1507.2578125, 'loadavg_end': [34.74609375, 16.6611328125, 51.77490234375], 'finished': '2026-10-11T09:31:27+0100'}
- clean-input determinism (max |run1-run2|): JPCLN030 0.0e+00, JPCNN001 0.0e+00, cxr_train_1_a_1 0.0e+00, cxr_train_8_a_2 0.0e+00
