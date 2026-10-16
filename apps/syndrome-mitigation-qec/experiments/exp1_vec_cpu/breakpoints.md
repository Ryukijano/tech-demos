# Breakpoints (seed 0)

Where dual-decoder VEC breaks down, honestly. All runs CPU-only; `degen` = a contested sector had zero calibration shots, so the exact Eq. (11) weights are undefined (the paper sees the same: no S_Y counts for p ≤ 0.01 at d=7).

## 1. Physical error rate sweep (cal+eval at each p)

| d | p | p_L MWPM | p_L ML | p_res PS | p_res VEC | eta | note |
|---|---|---|---|---|---|---|---|
| 3 | 0.01 | 1.575e-03 | 1.350e-03 | 1.066e-03 | — | — | degenerate (empty Y sector) |
| 3 | 0.02 | 5.990e-03 | 5.035e-03 | 4.012e-03 | 1.439e-03 | 1.2e+02 | eta > 4 |
| 3 | 0.03 | 0.013 | 0.0112 | 8.682e-03 | 1.532e-03 | 7.1 | eta > 4 |
| 3 | 0.05 | 0.0336 | 0.0294 | 0.0225 | 5.744e-04 | 5.7 | eta > 4 |
| 3 | 0.08 | 0.0785 | 0.0689 | 0.0534 | 2.527e-03 | 6.8 | eta > 4 |
| 3 | 0.11 | 0.133 | 0.121 | 0.0995 | 6.220e-03 | 15 | eta > 4 |
| 3 | 0.15 | 0.216 | 0.198 | 0.162 | 6.150e-03 | 26 | eta > 4; near/above threshold |
| 3 | 0.19 | 0.3 | 0.28 | 0.238 | 0.016 | 36 | eta > 4; near/above threshold |
| 5 | 0.01 | 1.300e-04 | 1.100e-04 | 1.000e-04 | — | — | degenerate (empty Y sector) |
| 5 | 0.02 | 1.140e-03 | 6.400e-04 | 4.404e-04 | 1.936e-04 | 2.1 |  |
| 5 | 0.03 | 3.750e-03 | 2.060e-03 | 1.644e-03 | 4.613e-05 | 2.5 |  |
| 5 | 0.05 | 0.0163 | 0.0103 | 7.784e-03 | 7.098e-04 | 2.4 |  |
| 5 | 0.08 | 0.0544 | 0.0366 | 0.0277 | 2.091e-03 | 2.9 |  |
| 5 | 0.11 | 0.119 | 0.0854 | 0.0649 | 4.250e-03 | 3.5 |  |
| 5 | 0.15 | 0.227 | 0.176 | 0.138 | 3.864e-03 | 5.3 | eta > 4; near/above threshold |
| 5 | 0.19 | 0.344 | 0.287 | 0.236 | 5.835e-03 | 10 | eta > 4; near/above threshold |

Advantage collapses toward threshold (p≈0.19): at d=3 eta blows up past 15 above p=0.1, and at p=0.02 the near-empty contested sectors make eta≈123 (tiny p_L denominator, noisy weights). At p=0.01 the Y sector is empty on both distances — exact 4-sector VEC is undefined without more calibration shots.

## 2. Calibration-budget sweep (weights from first N_cal shots, full eval)

| d | N_cal | p_res VEC | C | eta | note |
|---|---|---|---|---|---|
| 3 | 1,000 | — | — | — | degenerate |
| 3 | 3,000 | 0.0181 | 2.2 | 35 |  |
| 3 | 10,000 | 2.016e-03 | 1.15 | 4.3 |  |
| 3 | 30,000 | 1.209e-03 | 1.16 | 4.7 |  |
| 3 | 100,000 | 1.957e-03 | 1.22 | 6.4 |  |
| 3 | 300,000 | 6.292e-04 | 1.2 | 5.8 |  |
| 3 | 1,000,000 | 5.862e-04 | 1.19 | 5.7 |  |
| 5 | 1,000 | — | — | — | degenerate |
| 5 | 3,000 | — | — | — | degenerate |
| 5 | 10,000 | 5.77 | 6.220e+04 | 3.71e+06 | weights blow up |
| 5 | 30,000 | 1.047e-03 | 1.04 | 2.3 |  |
| 5 | 100,000 | 4.276e-04 | 1.04 | 2.4 |  |
| 5 | 300,000 | 2.263e-04 | 1.04 | 2.4 |  |

Below ~1e4 (d=3) / ~3e4 (d=5) calibration shots the contested sectors hold only O(1) shots: weights are undefined or wild (d=5, N_cal=1e4: C≈6e4, p_res≈5.8 — worse than useless). p_res then improves roughly as 1/sqrt(N_cal) to the eval floor.

## 3. Noise drift: weights calibrated at p=0.05, applied at p'

| d | p' | p_L(p') | p_res PS(p') | p_res VEC drifted | still <PS? |
|---|---|---|---|---|---|
| 3 | 0.01 | 1.575e-03 | 1.066e-03 | 9.445e-06 | yes |
| 3 | 0.02 | 5.990e-03 | 4.012e-03 | 7.649e-04 | yes |
| 3 | 0.03 | 0.013 | 8.682e-03 | 2.208e-04 | yes |
| 3 | 0.05 | 0.0336 | 0.0225 | 7.909e-04 | yes |
| 3 | 0.08 | 0.0785 | 0.0534 | 1.510e-03 | yes |
| 3 | 0.11 | 0.133 | 0.0995 | 0.0219 | yes |
| 3 | 0.15 | 0.216 | 0.162 | 0.027 | yes |
| 3 | 0.19 | 0.3 | 0.238 | 0.0627 | yes |
| 5 | 0.01 | 1.300e-04 | 1.000e-04 | 7.462e-05 | yes |
| 5 | 0.02 | 1.140e-03 | 4.404e-04 | 1.893e-04 | yes |
| 5 | 0.03 | 3.750e-03 | 1.644e-03 | 2.632e-04 | yes |
| 5 | 0.05 | 0.0163 | 7.784e-03 | 3.570e-04 | yes |
| 5 | 0.08 | 0.0544 | 0.0277 | 3.100e-03 | yes |
| 5 | 0.11 | 0.119 | 0.0649 | 0.0104 | yes |
| 5 | 0.15 | 0.227 | 0.138 | 0.0296 | yes |
| 5 | 0.19 | 0.344 | 0.236 | 0.0593 | yes |

Mild drift (p 0.03–0.08 on weights from 0.05) keeps VEC below postselection; strong drift (p'≥0.11) degrades VEC to a few percent residual, though still below the unmitigated LER in this range — miscalibration bias scales with ||p_I|| (paper Eq. 14), not with p_L.

## 4. Weight sweep of one contested sector (eval split)

DISCLOSED post-hoc grid extension: the pre-registered absolute grid [−1, 1] does not bracket the d=3 optimum (w*_X ≈ −2.08; the paper's −0.42 is a d=7 value), so relative points around w* were added after the official run. No threshold depends on this grid.

| d | sector | w* | argmin over grid | p_res(min) | p_res(w=0, PS) | p_res(w=1, plain) |
|---|---|---|---|---|---|---|
| 3 | X | -2.083 | -2.083 | 5.862e-04 | 0.0114 | 0.0171 |
| 3 | Y | 7.762 | 9.703 | 3.789e-04 | 1.999e-03 | 1.817e-03 |
| 5 | X | -0.901 | -0.811 | 2.195e-04 | 3.775e-03 | 8.158e-03 |
| 5 | Y | 0.236 | 0.236 | 3.159e-04 | 3.506e-04 | 3.702e-04 |

## 5. Report-only stretch: syndrome-only noise learning (d=3, inhomogeneous p)

Per-qubit true rates U(np.float64(0.04), np.float64(0.078)) (seed 777); 1,000,000 unlabelled learning shots. Walsh–Hadamard fidelities with 14 characters (8 degree-1 + 6 adjacent same-sector degree-2), design-matrix rank 9/9, bias-corrected weighted NNLS (Eqs. S.2–S.9).

- learned rates: median relative error 0.31% (max 1.12%)
- p_L (MWPM) = 0.0432; postselection bias = 0.0277
- VEC with LEARNED weights (no true labels anywhere in calibration): p_res = 4.801e-04 (C = 1.2283)
- VEC with oracle weights on the same shots: p_res = 4.687e-04 (C = 1.2445)

Learned-weight VEC matches oracle-weight VEC within its statistical floor — the paper's claim that syndrome records alone suffice for calibration holds in this toy.
