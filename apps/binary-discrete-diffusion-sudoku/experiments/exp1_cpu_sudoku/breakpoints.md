# Breakpoints (seed 0)

## Fewer clues (harder puzzles)

| n_clues | binary solve | masked solve | gap |
|---:|---:|---:|---:|
| 50 | 0.000 | 0.625 | -0.625 |
| 40 | 0.000 | 0.188 | -0.188 |
| 30 | 0.000 | 0.000 | +0.000 |
| 22 | 0.000 | 0.000 | +0.000 |

## Fewer denoising steps (NFE)

| NFE | binary solve | masked solve | gap |
|---:|---:|---:|---:|
| 32 | 0.021 | 0.812 | -0.792 |
| 16 | 0.000 | 0.625 | -0.625 |
| 8 | 0.000 | 0.354 | -0.354 |
| 4 | 0.000 | 0.271 | -0.271 |
| 2 | 0.000 | 0.042 | -0.042 |

## Training-step ablation

| steps | binary solve | masked solve | gap |
|---:|---:|---:|---:|
| 500 | 0.000 | 0.000 | +0.000 |
| 3000 | 0.000 | 0.729 | -0.729 |

Honest note: this toy uses a ~110k-param network and synthetic 50-clue boards — far from the paper's 43M DiT on Sudoku-Extreme. Breakpoints show where the *toy* comparison collapses, not where the paper's method fails.
