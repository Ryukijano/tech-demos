# Breakpoints (seed 0, not in the PASS rule)

| case | n, d, k | train MSE ratio @ 2.5 | wall ratio |
|---|---|---:|---:|
| high_dim | 4000, 512, 64 | 1.0006 | 0.349 |
| imbalanced | 20000, 32, 128 | 0.9758 | 0.387 |
| large_k | 8000, 32, 2000 | 0.9479 | 1.122 |
| tiny_n | 200, 16, 20 | 1.0761 | 31.286 |

Library-default caps on the mixture, seed 0, same init, 25 Lloyd iterations.
Full-data train MSE 11.477895. Default max 256 / min 39 train MSE 11.477895.
Default / full = 1.0000.
