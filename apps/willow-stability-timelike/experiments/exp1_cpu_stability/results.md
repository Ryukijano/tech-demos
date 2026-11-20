# exp1 results

Status: **PASS**

Lambda_T at cycles = patch width, p = 1e-3, no leakage. eps = 1 - (1 - P_shot)^(1/cycles). A zero-failure denominator uses P < 3/N.

| seed | d | cycles | fails/shots | P_L | eps |
|---|---|---|---|---|---|
| 0 | 3 | 3 | 135/200000 | 6.750e-04 | 2.251e-04 |
| 0 | 5 | 5 | 47/600000 | 7.833e-05 | 1.567e-05 |
| 0 | 7 | 7 | 21/1500000 | 1.400e-05 | 2.000e-06 |
| 1 | 3 | 3 | 121/200000 | 6.050e-04 | 2.017e-04 |
| 1 | 5 | 5 | 55/600000 | 9.167e-05 | 1.833e-05 |
| 1 | 7 | 7 | 19/1500000 | 1.267e-05 | 1.810e-06 |
| 2 | 3 | 3 | 144/200000 | 7.200e-04 | 2.401e-04 |
| 2 | 5 | 5 | 51/600000 | 8.500e-05 | 1.700e-05 |
| 2 | 7 | 7 | 23/1500000 | 1.533e-05 | 2.190e-06 |

| seed | Lambda_T 3→5 | Lambda_T 5→7 | mean | lower bound? |
|---|---|---|---|---|
| 0 | 14.364 | 7.834 | 11.099 | False |
| 1 | 11.002 | 10.132 | 10.567 | False |
| 2 | 14.121 | 7.761 | 10.941 | False |

Seed mean of Lambda_T_mean: **10.869 ± 0.273** (bar > 1.2).

## Same-cycle ratio at c = 4 (reported, not a pass bar)

| seed | Lambda_fixed 3→5 | Lambda_fixed 5→7 | mean |
|---|---|---|---|
| 0 | 0.295 | 0.376 | 0.336 |
| 1 | 0.230 | 0.453 | 0.342 |
| 2 | 0.209 | 0.397 | 0.303 |

## Leakage knee and walking gain (pooled over seeds, d = 3)

- early drop P(2)/P(4) = 4.790 (bar ≥ 2)
- late floor P(32)/P(8) = 0.797 (bar ≥ 0.5)
- leakage floor / Pauli bound at 32 cycles = 196.7 (bar ≥ 10)
- walking gain at 32 cycles = 59.000 (lower bound, walking had 0 failures) (bar ≥ 3)

Per-cycle logical error, mean ± sample std over seeds:

| variant | cycles | P_L mean | P_L std | eps mean |
|---|---|---|---|---|
| leakage | 2 | 2.4083e-02 | 1.1577e-03 | 1.2115e-02 |
| leakage | 4 | 5.0278e-03 | 4.2763e-04 | 1.2593e-03 |
| leakage | 6 | 5.5000e-03 | 5.0690e-04 | 9.1879e-04 |
| leakage | 8 | 6.1667e-03 | 1.3411e-03 | 7.7299e-04 |
| leakage | 12 | 6.1667e-03 | 4.1667e-04 | 5.1535e-04 |
| leakage | 16 | 6.5833e-03 | 8.3333e-05 | 4.1273e-04 |
| leakage | 24 | 5.8333e-03 | 3.6324e-04 | 2.4374e-04 |
| leakage | 32 | 4.9167e-03 | 5.7735e-04 | 1.5402e-04 |
| pauli | 2 | 2.1000e-02 | 7.6281e-04 | 1.0556e-02 |
| pauli | 4 | 2.8333e-04 | 8.0364e-05 | 7.0841e-05 |
| pauli | 6 | 8.3333e-06 | 1.4434e-05 | 1.3889e-06 |
| pauli | 8 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| pauli | 12 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| pauli | 16 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| pauli | 24 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| pauli | 32 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| walking | 2 | 2.4083e-02 | 1.1577e-03 | 1.2115e-02 |
| walking | 4 | 2.2500e-03 | 3.0046e-04 | 5.6298e-04 |
| walking | 6 | 5.5556e-05 | 9.6225e-05 | 9.2599e-06 |
| walking | 8 | 2.7778e-05 | 4.8113e-05 | 3.4723e-06 |
| walking | 12 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| walking | 16 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| walking | 24 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| walking | 32 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |

## Adaptive stopping (report-only, not in the status)

Pre-registered gap threshold T = 4.0 nats.
Mean stopping cycle 4.01 versus best fixed cycle 4 (time change -0.2%).
Spacetime error 2.9759e-03 versus best fixed 3.1833e-03 (spacetime change +6.5%).

Toy spacetime error is 0.5 * (memory P_L(c) + stability failure). The hardware 38% / 34% figures are QUOTED and are not this run.

| T (nats) | mean cycles | P_stab | P_ST | best fixed c | best fixed P_ST | spacetime change | time change |
|---|---|---|---|---|---|---|---|
| 1.0 | 4.01 | 5.083e-03 | 3.017e-03 | 4 | 3.183e-03 | +5.2% | -0.1% |
| 2.0 | 4.01 | 5.000e-03 | 2.976e-03 | 4 | 3.183e-03 | +6.5% | -0.2% |
| 4.0 | 4.01 | 5.000e-03 | 2.976e-03 | 4 | 3.183e-03 | +6.5% | -0.2% |
| 8.0 | 4.01 | 5.000e-03 | 2.976e-03 | 4 | 3.183e-03 | +6.5% | -0.2% |
| 12.0 | 4.15 | 5.167e-03 | 3.079e-03 | 4 | 3.183e-03 | +3.3% | -3.7% |

## QUOTED hardware numbers (not measured here)

- mu_T = 1.47 ± 0.25 (Pauli model 1.97 ± 0.44), block suppression on Willow
- walking stability floor ≤ 3.20e-8 at 60 cycles
- paper: two orders of magnitude below the standard construction at 60 cycles
- bookmark post: four orders of magnitude below any experiment in the literature at 60 cycles
- adaptive stopping: about 38% lower spacetime logical error and 34% lower mean stopping time
