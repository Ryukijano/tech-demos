# Breakpoints (seed 0, not part of the status)

Init-std and Langevin noise use the 32 hardest boards and 8 segments. Rating bands use 32 boards and 16 segments. The unit-RMS projection removes the scale of the initial Gaussian, so init_std is a weak knob. noise_scale is added after that projection.

## Initial Gaussian std

| Value | Accuracy at segment 1 | Accuracy at segment 8 |
|---|---|---|
| 0.01 | 0.438 | 0.906 |
| 1.0 | 0.406 | 0.938 |
| 8.0 | 0.406 | 0.875 |

## Langevin noise scale

| Value | Accuracy at segment 1 | Accuracy at segment 8 |
|---|---|---|
| 0.0 | 0.438 | 0.938 |
| 0.05 | 0.531 | 0.906 |
| 0.4 | 0.406 | 0.969 |
| 1.0 | 0.031 | 0.031 |
| 2.0 | 0.000 | 0.000 |

## Rating band

| Band | Segment 1 | Segment 4 | Segment 16 |
|---|---|---|---|
| easiest | 0.906 | 1.000 | 1.000 |
| median | 0.375 | 0.656 | 0.969 |
| hardest | 0.562 | 0.844 | 0.969 |

