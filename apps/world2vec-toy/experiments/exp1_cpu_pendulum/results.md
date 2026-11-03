# Results — `exp1_cpu_pendulum`

**Status: FAIL**

Primary metric: LRC at horizon H=8 (higher is better). Anti-collapse gate: linear-probe R² of `[sin θ, cos θ, ω]` from latents.

| seed | world2vec LRC | discrete LRC | recon LRC | w2v probe R² | discrete probe R² | Δ LRC |
|---|---:|---:|---:|---:|---:|---:|
| 0 | **0.5085** | 0.0968 | 0.1364 | 0.8566 | 0.7308 | +0.4117 |
| 1 | **0.1952** | 0.5209 | 0.2535 | 0.7526 | 0.7204 | -0.3257 |
| 2 | **0.3493** | 0.5900 | 0.2093 | 0.9019 | 0.7731 | -0.2407 |

**Mean ± std LRC:** world2vec **0.3510 ± 0.1566**, discrete **0.4026 ± 0.2671**, recon **0.1997 ± 0.0591** (gap vs discrete **-0.0516**; bar ≥ 0.05).

**Mean probe R² (world2vec):** **0.8371 ± 0.0766** (bar ≥ 0.5).

Every-seed LRC win vs discrete: **False**.

