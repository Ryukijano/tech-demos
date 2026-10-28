# exp1_cpu_sudoku results — **FAIL**

Primary metric: full-puzzle solve rate on 48 held-out puzzles (50 clues, NFE=32).

| seed | binary solve | masked solve | gap | cell_acc binary | cell_acc masked |
|---|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.729 | -0.729 | 0.816 | 0.958 |
| 1 | 0.000 | 0.688 | -0.688 | 0.789 | 0.960 |
| 2 | 0.000 | 0.646 | -0.646 | 0.769 | 0.945 |

**Mean ± std:** binary **0.000 ± 0.000**, masked **0.688 ± 0.042**, gap **-0.688**.

Status rule: PASS needs binary > masked on every seed AND mean gap ≥ 0.05. Got every_seed=False, mean_gap=-0.688 → **FAIL**.

Params (matched): 111369.
