# Results — `exp1_cpu_views`

**Status: PARTIAL**

Primary metric: Procrustes-aligned linear CKA (higher is better).

| seed | JEPA CKA | recon CKA | random CKA | JEPA−recon | JEPA−random | JEPA kNN |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.9797 | 0.7871 | 0.2370 | +0.1926 | +0.7427 | 0.1204 |
| 1 | 0.9282 | 0.8000 | 0.2437 | +0.1283 | +0.6846 | 0.1356 |
| 2 | 0.9662 | 0.7833 | 0.3285 | +0.1829 | +0.6377 | 0.1159 |

**Mean ± std:** JEPA **0.9580 ± 0.0267**, recon **0.7901 ± 0.0088**, random **0.2697 ± 0.0510**.

Gaps: vs recon **+0.1679** (need ≥ 0.2); vs random **+0.6883** (need ≥ 0.35); JEPA mean need ≥ 0.55.

Every-seed beats recon: **True**; beats random: **True**.

