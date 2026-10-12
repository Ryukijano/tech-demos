# exp1_cpu_match results

**Status: FAIL** (latency+determinism not met; rotation LoMa-R>LoMa-B at ≥45° not met)

Thresholds (fixed before running): mean latency_s_match ≤ 30.0 s per seed; n_matches identical across seeds 0/1/2; LoMa-R mean inlier_ratio > LoMa-B for every angle ≥ 45°.

## Per seed (LoMa-B, mean over pairs)

| seed | mean_latency_s | max_latency_s | mean_n_matches | mean_inlier_ratio | latency_pass | determinism_pass |
|---|---|---|---|---|---|---|
| 0 | 8.65 | 8.73 | 541.00 | 0.1799 | True | False |
| 1 | 8.44 | 8.77 | 539.67 | 0.1887 | True | False |
| 2 | 8.26 | 8.70 | 537.67 | 0.1657 | True | False |

## Per pair × seed

| seed | pair | n_matches | n_inliers | inlier_ratio | latency_s_match |
|---|---|---|---|---|---|
| 0 | pair_0015 | 971 | 144 | 0.1483 | 8.60 |
| 0 | pair_0022 | 367 | 105 | 0.2861 | 8.62 |
| 0 | pair_toronto | 285 | 30 | 0.1053 | 8.73 |
| 1 | pair_0015 | 975 | 157 | 0.1610 | 8.28 |
| 1 | pair_0022 | 361 | 108 | 0.2992 | 8.77 |
| 1 | pair_toronto | 283 | 30 | 0.1060 | 8.27 |
| 2 | pair_0015 | 969 | 133 | 0.1373 | 7.97 |
| 2 | pair_0022 | 365 | 96 | 0.2630 | 8.70 |
| 2 | pair_toronto | 279 | 27 | 0.0968 | 8.11 |

## Determinism (n_matches across seeds)

| pair | n_matches_across_seeds | identical |
|---|---|---|
| pair_0015 | 971,975,969 | False |
| pair_0022 | 367,361,365 | False |
| pair_toronto | 285,283,279 | False |

## Rotation criterion (mean inlier_ratio over breakpoint images)

| angle | mean_inlier_B | mean_inlier_R | R_better |
|---|---|---|---|
| 45 | 0.1217 | 0.1356 | True |
| 60 | 0.0950 | 0.1140 | True |
| 90 | 0.1288 | 0.1379 | True |
| 135 | 0.1907 | 0.1155 | False |
| 180 | 0.1287 | 0.1356 | True |

## Runtime

- cold first match: LoMa-B 8.6s, LoMa-R 7.5s
- load: LoMa-B 6.7s, LoMa-R 8.1s
- env: {'python': '3.12.3', 'torch': '2.14.1+cpu', 'threads': 4, 'cpu': 'x86_64', 'nproc': 4, 'cuda': False, 'loadavg_start': [0.19580078125, 0.25048828125, 0.1728515625], 'started': '2026-10-12T08:41:04+0000', 'maxrss_mb': 4548.80078125, 'loadavg_end': [4.02001953125, 3.01318359375, 1.5146484375], 'finished': '2026-10-12T08:47:51+0000'}
- disclosure: Thresholds written before the official multi-seed run. One disclosed smoke (2026-10-12) ran LoMa-B once on pair_0015 after weights download: load 12.4 s, warm match 9.17 s, n_matches=968. The 30 s latency budget was therefore left unchanged (clearly not too tight). Determinism and rotation thresholds were not informed by that smoke.
