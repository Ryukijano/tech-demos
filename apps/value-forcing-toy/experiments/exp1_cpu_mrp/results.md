# exp1_cpu_mrp results

Status: **FAIL**

Pre-registered pass rule: mean W1(VF) ≤ 0.80 × mean W1(endpoint), mean W1(VF) < mean W1(QR), and VF std ratio inside [0.8, 1.2], averaged over seeds 0/1/2.

| Check | Result |
|---|---|
| `vf_w1_at_least_20pct_below_endpoint` | not met |
| `vf_w1_below_qr` | not met |
| `vf_std_ratio_in_0.8_1.2` | not met |
| `vf_w1_below_endpoint` | not met |
| `vf_std_ratio_in_0.5_1.5` | met |

## Mean over seeds

| Method | mean W1 | std ratio |
|---|---:|---:|
| Value Forcing | 0.4422 ± 0.0144 | 0.753 ± 0.009 |
| Endpoint flow | 0.4384 ± 0.0224 | 0.637 ± 0.050 |
| QR-DQN | 0.2222 ± 0.0054 | 0.498 ± 0.011 |

Relative W1 reduction vs endpoint: **-0.9%** (threshold 20%).

## Per seed

| Seed | VF W1 | Endpoint W1 | QR W1 | VF std ratio | Endpoint std ratio | QR std ratio |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.4550 | 0.4537 | 0.2174 | 0.761 | 0.598 | 0.507 |
| 1 | 0.4452 | 0.4490 | 0.2281 | 0.744 | 0.693 | 0.485 |
| 2 | 0.4266 | 0.4127 | 0.2213 | 0.753 | 0.620 | 0.502 |

## Per-state W1 (mean over seeds)

| State | true mean | true std | VF | Endpoint | QR |
|---|---:|---:|---:|---:|---:|
| MIX | 1.080 | 1.080 | 0.6038 | 0.6174 | 0.5493 |
| HIGH | 2.700 | 0.000 | 0.4116 | 0.3835 | 0.0683 |
| HIGH2 | 1.500 | 0.000 | 0.4001 | 0.3714 | 0.0449 |
| LOW | 0.000 | 0.000 | 0.4067 | 0.3670 | 0.0201 |
| LOW2 | 0.000 | 0.000 | 0.4013 | 0.3577 | 0.0179 |
| SINK | 0.000 | 0.000 | 0.4044 | 0.3718 | 0.0192 |
| WIDE | 0.982 | 1.104 | 0.4201 | 0.5133 | 0.4632 |
| CHAIN | 1.264 | 0.864 | 0.4899 | 0.5254 | 0.5951 |

