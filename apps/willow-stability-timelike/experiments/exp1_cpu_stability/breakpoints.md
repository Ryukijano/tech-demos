# Breakpoints (seed 0)

Lambda_T uses cycles = d. A value below 1 means the wider patch has the higher per-cycle error.

| p | Lambda_T 3→5 | Lambda_T 5→7 | mean |
|---|---|---|---|
| 0.001 | 12.335 | 7.000 | 9.668 |
| 0.003 | 2.913 | 3.809 | 3.361 |
| 0.006 | 1.647 | 1.843 | 1.745 |
| 0.008 | 1.460 | 1.286 | 1.373 |
| 0.01 | 1.193 | 1.114 | 1.154 |
| 0.012 | 0.985 | 1.036 | 1.011 |
| 0.015 | 0.905 | 0.967 | 0.936 |
| 0.02 | 0.841 | 0.955 | 0.898 |
| 0.03 | 0.907 | 1.130 | 1.019 |
| 0.05 | 1.271 | 1.355 | 1.313 |

Lambda_T mean first drops below 1 at p = 0.015 on this grid.
At higher p the ratio climbs back through 1 (p = 0.03, 0.05). All three distances are then in the high-error regime, so the ratio is no longer a clean threshold diagnostic.

## Walking gain versus burst rate (d=3, 32 cycles, tau=8)

| p_start | P_leak | P_walk | gain |
|---|---|---|---|
| 0.0003 | 5.167e-03 | 0.000e+00 | 10.333 |
| 0.003 | 4.867e-02 | 0.000e+00 | 97.333 |
| 0.01 | 1.770e-01 | 0.000e+00 | 354.0 |
| 0.03 | 3.853e-01 | 0.000e+00 | 770.7 |
| 0.08 | 4.795e-01 | 2.067e-02 | 23.202 |
| 0.12 | 4.885e-01 | 1.327e-01 | 3.682 |
| 0.2 | 4.382e-01 | 4.213e-01 | 1.040 |

Walking gain first falls below 3x at p_start = 0.2.

## Walking gain versus lifetime (p_start = 3e-4)

| tau | P_leak | P_walk | gain |
|---|---|---|---|
| 1 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 4 | 1.000e-03 | 0.000e+00 | 2.000 |
| 8 | 5.167e-03 | 0.000e+00 | 10.333 |
| 20 | 1.433e-02 | 0.000e+00 | 28.667 |

Walking gain is below 3x at tau=1 (both zero), tau=2 (both zero), tau=4 (2.000). Short lifetimes produce too few logical failures for the 2-cycle clear to matter; from tau = 8 upward the clear removes a visible floor.
