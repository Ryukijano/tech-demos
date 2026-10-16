# exp1_vec_cpu results

Status (pre-registered rule): **PARTIAL**

Primary metric: `p_res = || sum_k q_k w_k p_k ||_1` on held-out eval shots, weights fitted on the separate calibration split (oracle-calibrated from true frame flips, as disclosed in config.yaml). Error bars: paired delete-1 jackknife, B=20.

| d | seed | p_L MWPM | p_L ML | p_res postselect | p_res VEC | 0.1·p_L | C | eta | ≤0.1·p_L? | <PS? | eta<4? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0 | 0.034±2.0e-04 | 0.0294 | 0.0226±1.5e-04 | **5.862e-04**±4.9e-04 | 3.397e-03 | 1.1934±0.0085 | 5.69 | yes | yes | **no** |
| 3 | 1 | 0.034±1.4e-04 | 0.0294 | 0.0226±1.2e-04 | **2.231e-04**±2.5e-04 | 3.399e-03 | 1.1925±0.0065 | 5.66 | yes | yes | **no** |
| 3 | 2 | 0.0339±1.3e-04 | 0.029 | 0.0223±1.1e-04 | **4.083e-04**±3.3e-04 | 3.388e-03 | 1.1835±0.0087 | 5.41 | yes | yes | **no** |
| 5 | 0 | 0.0167±1.8e-04 | 0.0102 | 7.734e-03±1.3e-04 | **3.159e-04**±2.3e-04 | 1.674e-03 | 1.0409±0.0008 | 2.44 | yes | yes | yes |
| 5 | 1 | 0.0165±1.6e-04 | 9.816e-03 | 7.347e-03±1.0e-04 | **4.815e-04**±2.4e-04 | 1.647e-03 | 1.0399±0.0006 | 2.42 | yes | yes | yes |
| 5 | 2 | 0.0162±1.5e-04 | 9.816e-03 | 7.367e-03±1.2e-04 | **5.044e-04**±2.6e-04 | 1.621e-03 | 1.0396±0.0006 | 2.44 | yes | yes | yes |

Report-only d=7 (seed 0, 200,000 cal / 200,000 eval shots): p_L=7.825e-03, p_L(ML)=3.075e-03, p_res_PS=2.199e-03, p_res_VEC=1.391e-04±1.2e-04, C=1.0135, eta=1.73.

## Direct observable check (ideal +1; report-only)

`<Z_L>` on `|0_L>` and `<X_L>` on `|+_L>`; jackknife 1σ. (Code-capacity + frame simulation: readout-flip statistics are state-independent, so these follow from the same joint counts.)

| d | seed | plain Z_L | postselect Z_L | VEC Z_L | plain X_L | VEC X_L |
|---|---|---|---|---|---|---|
| 3 | 0 | 0.96497±0.00032 | 0.97722±0.00031 | **1.00017±0.00053** | 0.96453±0.00034 | **1.00075±0.00052** |
| 3 | 1 | 0.96489±0.00028 | 0.97729±0.00025 | **1.00015±0.00031** | 0.96456±0.00022 | **0.99979±0.00039** |
| 3 | 2 | 0.96478±0.00022 | 0.97757±0.00024 | **1.00017±0.00048** | 0.96494±0.00019 | **1.00046±0.00051** |
| 5 | 0 | 0.98331±0.00025 | 0.99230±0.00018 | **1.00011±0.00021** | 0.98258±0.00028 | **0.99976±0.00024** |
| 5 | 1 | 0.98316±0.00030 | 0.99245±0.00026 | **0.99992±0.00032** | 0.98333±0.00021 | **1.00056±0.00026** |
| 5 | 2 | 0.98356±0.00032 | 0.99280±0.00024 | **1.00072±0.00025** | 0.98341±0.00020 | **0.99980±0.00027** |
| 7 | 0 | 0.99208±0.00027 | 0.99806±0.00021 | **1.00044±0.00024** | 0.99214±0.00034 | **1.00046±0.00024** |

## Report-only: sector shares, weights, conditional matrices

**d=3, seed 0** — shares q(I,X,Y,Z) = (0.982, 9.030e-03, 2.680e-04, 8.342e-03); weights w(I,X,Y,Z) = (1.0548, -2.0830, 7.7623, -2.3282); 254 distinct syndromes decoded exactly; cal 0.65s + eval 0.51s.

| sector \ error | I | X | Y | Z |
|---|---|---|---|---|
| I | 0.9774 | 0.0108 | 0.0006 | 0.0112 |
| X | 0.3484 | 0.5940 | 0.0310 | 0.0266 |
| Y | 0.2164 | 0.2425 | 0.2761 | 0.2649 |
| Z | 0.3400 | 0.0279 | 0.0333 | 0.5988 |

**d=5, seed 0** — shares q(I,X,Y,Z) = (0.989, 5.558e-03, 1.400e-04, 5.460e-03); weights w(I,X,Y,Z) = (1.0214, -0.9011, 0.2357, -0.9253); 60,030 distinct syndromes decoded exactly; cal 0.54s + eval 0.42s.

| sector \ error | I | X | Y | Z |
|---|---|---|---|---|
| I | 0.9921 | 0.0039 | 0.0001 | 0.0039 |
| X | 0.1976 | 0.7830 | 0.0130 | 0.0065 |
| Y | 0.0714 | 0.1143 | 0.7857 | 0.0286 |
| Z | 0.2161 | 0.0073 | 0.0081 | 0.7685 |

**d=7, seed 0** — shares q(I,X,Y,Z) = (0.993, 3.295e-03, 6.500e-05, 3.345e-03); weights w(I,X,Y,Z) = (1.0096, -0.4045, -0.0440, -0.4338); 173,024 distinct syndromes decoded exactly; cal 3.91s + eval 3.14s.

| sector \ error | I | X | Y | Z |
|---|---|---|---|---|
| I | 0.9976 | 0.0011 | 0.0000 | 0.0012 |
| X | 0.1563 | 0.8392 | 0.0030 | 0.0015 |
| Y | 0.0000 | 0.0000 | 0.8462 | 0.1538 |
| Z | 0.1525 | 0.0015 | 0.0060 | 0.8401 |

Paper reference point (QUOTED from arXiv:2610.12400 Supp. II.B, d=7, p=0.05; not claimed reproduced): P_XX≈P_ZZ≈0.86, P_YY≈0.78, w_I*≈1.01, w_X*=w_Z*≈−0.42, w_Y*≈−4e-2. Our report-only d=7 run gives w=(1.010, −0.405, −0.044, −0.434) and P_XX=0.839, P_ZZ=0.840 (P_YY=0.846 from only ~13 Y-sector calibration shots, so it carries a large statistical error).

## Disclosure

- Smoke (code-path check, 2e4 shots, seed 0, recorded in `smoke.json`): d=3 p_L=3.3e-2, p_res_VEC=3.4e-3 (stat floor); d=5 p_L=1.8e-2, p_res_VEC=1.7e-3. No threshold touched.
- Oracle calibration: sector centroids use true frame flips on the calibration split, exactly like the paper's Fig. 2 benchmark; the noise-learning breakpoint uses no true labels.
- Total official wall time: 13.13s on 4 vCPUs.
