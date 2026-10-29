# exp1_cpu_auditable results — **FAIL**

Primary metric: `auroc_gap = AUROC(tau=0.00) − AUROC(tau=0.85)` on 96-example balanced subsamples (dmitva/human_ai_generated_text).

- Mean AUROC τ=0.85: **0.8235 ± 0.0308**
- Mean AUROC τ=0.00: **0.9825 ± 0.0063**
- Mean gap: **0.1590 ± 0.0303** (threshold ≤ 0.05)
- Every-seed pass: **False**

| seed | AUROC τ=0.85 | AUROC τ=0.00 | gap | n_active 0.85 | pass |
|---|---:|---:|---:|---:|:---:|
| 0 | 0.7886 | 0.9826 | 0.1940 | 72.0 | no |
| 1 | 0.8351 | 0.9761 | 0.1411 | 72.1 | no |
| 2 | 0.8468 | 0.9887 | 0.1419 | 73.7 | no |

Status: **FAIL**

