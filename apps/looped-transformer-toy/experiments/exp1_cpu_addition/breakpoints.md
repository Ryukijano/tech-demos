# Breakpoints (seed 0)

## Test-time loops (trained looped model, K_train=4)

| n_loops | looped exact | digit_acc |
|---:|---:|---:|
| 1 | 0.627 | 0.897 |
| 2 | 0.998 | 1.000 |
| 4 | 1.000 | 1.000 |
| 8 | 1.000 | 1.000 |
| 16 | 0.934 | 0.979 |

## Digit length (ID=4; OOD shorter/longer)

| n_digits | looped exact | baseline exact | gap |
|---:|---:|---:|---:|
| 2 | 0.004 | 0.000 | +0.004 |
| 3 | 0.000 | 0.000 | +0.000 |
| 4 | 1.000 | 1.000 | +0.000 |
| 5 | 0.000 | 0.000 | +0.000 |
| 6 | 0.000 | 0.000 | +0.000 |

## Training-step ablation

| steps | looped exact | baseline exact | gap |
|---:|---:|---:|---:|
| 500 | 0.000 | 0.002 | -0.002 |
| 2500 | 1.000 | 1.000 | +0.000 |

Honest note: this toy uses a ~50–100k-param network on synthetic addition — far from the paper's up-to-1B LM experiments. Breakpoints show where the *toy* saturates or breaks, not where the paper's method fails.
