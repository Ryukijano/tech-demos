# exp1_cpu_addition results — **FAIL**

Primary metric: exact-match accuracy on 512 held-out 4-digit additions (looped K=4 vs baseline K=1, matched params).

| seed | looped exact | baseline exact | gap | digit_acc looped | digit_acc baseline |
|---|---:|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
| 1 | 0.990 | 1.000 | -0.010 | 0.998 | 1.000 |
| 2 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |

**Mean ± std:** looped **0.997 ± 0.006**, baseline **1.000 ± 0.000**, gap **-0.003**.

Status rule: PASS needs looped > baseline on every seed AND mean gap ≥ 0.10. Got every_seed=False, mean_gap=-0.003 → **FAIL**.

Params (matched): 105856.
