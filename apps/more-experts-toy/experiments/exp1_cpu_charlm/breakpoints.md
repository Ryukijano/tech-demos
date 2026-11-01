# Breakpoints (seed 0)

## Reuse factor extremes

| kind | R | steps | val_ce | bpc | reuse_rate | random |
|---|---:|---:|---:|---:|---:|---:|
| more | 2 | 800 | 1.8999 | 2.7409 | 0.184356689453125 | 0.25 |
| more | 4 | 400 | 2.3017 | 3.3206 | 0.11961873372395833 | 0.125 |
| more_nodepth | 4 | 400 | 2.3021 | 3.3212 | 0.13326009114583334 | 0.125 |

## Few experts (N=2)

| kind | N | steps | val_ce | bpc |
|---|---:|---:|---:|---:|
| moe | 2 | 400 | 2.2814 | 3.2913 |
| more | 2 | 400 | 2.3200 | 3.3471 |

## Top-1 routing

| kind | K | steps | val_ce | bpc |
|---|---:|---:|---:|---:|
| moe | 1 | 400 | 2.4319 | 3.5085 |
| more | 1 | 400 | 2.4309 | 3.5070 |

Honest note: this toy uses a ~100k-param network on a 50k-char public-domain excerpt — far from the paper's 114M–1.15B C4 runs. Breakpoints show where the *toy* saturates or breaks.
