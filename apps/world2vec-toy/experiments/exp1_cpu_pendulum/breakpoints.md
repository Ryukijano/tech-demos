# Breakpoints

## Long horizon LRC

| kind | H | LRC | probe R² | horizon state R² |
|---|---:|---:|---:|---:|
| world2vec | 1 | 0.9528 | 0.8537 | 0.8521 |
| world2vec | 2 | 0.9348 | 0.8537 | 0.8435 |
| world2vec | 4 | 0.8545 | 0.8611 | 0.8307 |
| world2vec | 8 | 0.5085 | 0.8566 | 0.7279 |
| world2vec | 16 | -0.1900 | 0.8296 | 0.2581 |
| world2vec | 24 | -0.4052 | 0.8513 | 0.0157 |
| discrete | 1 | 0.9538 | 0.7103 | 0.7180 |
| discrete | 2 | 0.9169 | 0.7103 | 0.7143 |
| discrete | 4 | 0.7542 | 0.7251 | 0.6682 |
| discrete | 8 | 0.0968 | 0.7308 | 0.2890 |
| discrete | 16 | -0.6436 | 0.7260 | -0.3815 |
| discrete | 24 | -0.9135 | 0.7167 | -0.3529 |

## Observation noise

| kind | noise_std | LRC | probe R² |
|---|---:|---:|---:|
| world2vec | 0.0 | 0.0335 | 0.6613 |
| discrete | 0.0 | -0.2220 | 0.6674 |
| world2vec | 0.05 | -0.1731 | 0.6625 |
| discrete | 0.05 | -0.0856 | 0.6672 |
| world2vec | 0.15 | -0.0619 | 0.6639 |
| discrete | 0.15 | -0.1036 | 0.6643 |

## Irregular Δt (Euler step scale)

| kind | dt_scale | LRC | horizon state R² |
|---|---:|---:|---:|
| world2vec | 0.5 | -0.0987 | 0.5056 |
| world2vec | 1.0 | 0.5085 | 0.7279 |
| world2vec | 2.0 | -1.6134 | -0.2533 |
| discrete | 0.5 | 0.0968 | 0.2890 |
| discrete | 1.0 | 0.0968 | 0.2890 |
| discrete | 2.0 | 0.0968 | 0.2890 |

