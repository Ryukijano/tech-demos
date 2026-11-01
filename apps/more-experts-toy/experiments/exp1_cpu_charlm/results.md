# exp1_cpu_charlm results — **FAIL**

Primary metric: validation cross-entropy (nats) on held-out characters (seq_len=64, steps=800). Lower is better.

| seed | dense | MoE | MoRE | MoRE no-depth | MoE−MoRE |
|---|---:|---:|---:|---:|---:|
| 0 | 1.8973 | 1.8957 | 1.8999 | 1.8974 | -0.0042 |
| 1 | 1.8775 | 1.8306 | 1.8415 | 1.8445 | -0.0110 |
| 2 | 1.9081 | 1.8869 | 1.8915 | 1.8883 | -0.0046 |

**Mean ± std (val CE):**
- dense **1.8943 ± 0.0155**
- MoE **1.8710 ± 0.0353**
- MoRE **1.8776 ± 0.0316**
- MoRE no-depth **1.8767 ± 0.0283**
- gap MoE−MoRE **-0.0066** (need ≥ MoRE std 0.0316)

Status rule: PASS needs mean(MoRE) < mean(MoE) AND gap ≥ std(MoRE). Got more_beats=False, gap_ok=False → **FAIL**.

Params: {'dense': 126048, 'moe': 201888, 'more': 202848, 'more_nodepth': 202656}.

## Routing (MoRE adjacent shared-pool reuse)

Mean adjacent reuse rate **0.2217** vs random baseline **0.2500** (ratio **0.89×**).

