# Results — FAIL

Threshold commit recorded in `raw.json`: `bd3afc3f98151c85e3da838e1baf580fa30f1e81`.

| Metric | Bar | Value | |
|---|---|---:|---|
| online_gain | >= 0.15 | 0.0946 | miss |
| ml_end_gap | <= 0.08 | 0.0695 | met |
| ml_traj_mae | <= 0.1 | 0.0389 | met |
| naive_collapse | >= 0.15 | 0.0000 | miss |
| lastk_end_gap | <= 0.1 | 0.0565 | met |

| Seed | Base | Online | ML | Naive | Last-4 | Gain | ML traj MAE | Naive ‖Δθ‖∞ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.089 | 0.183 | 0.121 | 0.183 | 0.137 | 0.094 | 0.039 | 0.00e+00 |
| 1 | 0.089 | 0.178 | 0.114 | 0.178 | 0.115 | 0.089 | 0.036 | 0.00e+00 |
| 2 | 0.089 | 0.190 | 0.109 | 0.190 | 0.130 | 0.101 | 0.041 | 0.00e+00 |

Seed-mean final reward: online 0.184 ± 0.006, ML 0.114 ± 0.006, naive 0.184 ± 0.006, last-4 0.127 ± 0.011.

## Breakpoints (seed 0 unless noted)

| Run | Final | Base |
|---|---:|---:|
| shuffle | 0.104 | 0.089 |
| reverse | 0.097 | 0.089 |
| repeats-ml | 0.141 | 0.089 |
| repeats-naive | 0.000 | 0.089 |
| lastk-tiny | 0.073 | 0.089 |
| foreign-init | 0.043 | 0.000 |
| cross-seed | 0.114 | 0.089 |
| perturb-ml-0.001 | 0.121 | 0.089 |
| perturb-naive-0.001 | 0.068 | 0.089 |
| perturb-ml-0.01 | 0.123 | 0.088 |
| perturb-naive-0.01 | 0.049 | 0.088 |

## Specialists (report-only)

Addition specialist final 0.183. Subtraction specialist 0.003 → 0.026. Alternating ML student addition 0.112, subtraction 0.023.

