# Breakpoints — `exp1_cpu_views` (seed 0)

## Bad views (corruptions of one observation)

- Good-view JEPA (primary, seed 0): CKA **0.9797**
- Bad-view JEPA: CKA **0.9551** (noise_std=0.5, dropout_p=0.3, steps=800)
- Δ (good − bad): **+0.0246**

## Embedding dimension

| emb_dim | steps | Procrustes CKA | kNN overlap |
|---:|---:|---:|---:|
| 2 | 400 | 0.8597 | 0.0747 |
| 4 | 400 | 0.7483 | 0.0939 |
| 8 | 400 | 0.9195 | 0.1236 |
| 16 | 400 | 0.9632 | 0.1333 |

## Training length

| steps | Procrustes CKA | kNN overlap |
|---:|---:|---:|
| 50 | 0.8416 | 0.1293 |
| 200 | 0.8472 | 0.1239 |
| 800 | 0.9847 | 0.1178 |

