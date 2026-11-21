# Sudoku-Extreme subset, exact-board accuracy

Mean ± sample standard deviation over inference seeds 0/1/2. One released checkpoint. 96 highest-rated test puzzles, one trajectory each.

| Segment | Microsteps | Exact accuracy | Cell accuracy |
|---|---|---|---|
| 1 | 21 | 0.479 ± 0.065 | 0.809 ± 0.022 |
| 2 | 42 | 0.611 ± 0.052 | 0.855 ± 0.018 |
| 4 | 84 | 0.806 ± 0.024 | 0.923 ± 0.015 |
| 8 | 168 | 0.920 ± 0.037 | 0.968 ± 0.012 |
| 16 | 336 | 0.972 ± 0.006 | 0.989 ± 0.002 |

Status: **PASS** (4/4 bars).

Gain segment 16 − 4: 0.167 ± 0.028 (bar ≥ 0.02, met).

| Seed | Energy depth | AUROC(−E) | gtan incorrect / correct |
|---|---|---|---|
| 0 | 4 | 1.000 | 89.72 |
| 1 | 8 | 1.000 | 91.55 |
| 2 | 8 | 1.000 | 102.22 |

