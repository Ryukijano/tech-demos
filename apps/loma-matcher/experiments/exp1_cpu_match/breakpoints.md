# Breakpoints — in-plane rotation of image B (seed 0)

ok ≥ 0.5, degraded ≥ 0.2, broken < 0.2 (by MAGSAC inlier_ratio).

Story: LoMa-R should retain higher inlier_ratio than LoMa-B once rotation exceeds ~45°.

| image | model | level | n_matches | n_inliers | inlier_ratio | latency_s_match | verdict |
|---|---|---|---|---|---|---|---|
| pair_0015 | LoMa-B | 0 | 974 | 159 | 0.1632 | 7.86 | broken |
| pair_0015 | LoMa-B | 15 | 987 | 147 | 0.1489 | 8.23 | broken |
| pair_0015 | LoMa-B | 30 | 967 | 116 | 0.1200 | 8.92 | broken |
| pair_0015 | LoMa-B | 45 | 928 | 140 | 0.1509 | 9.65 | broken |
| pair_0015 | LoMa-B | 60 | 949 | 121 | 0.1275 | 9.89 | broken |
| pair_0015 | LoMa-B | 90 | 982 | 153 | 0.1558 | 9.38 | broken |
| pair_0015 | LoMa-B | 135 | 952 | 138 | 0.1450 | 9.62 | broken |
| pair_0015 | LoMa-B | 180 | 955 | 157 | 0.1644 | 8.82 | broken |
| pair_0015 | LoMa-R | 0 | 967 | 146 | 0.1510 | 7.85 | broken |
| pair_0015 | LoMa-R | 15 | 983 | 140 | 0.1424 | 7.30 | broken |
| pair_0015 | LoMa-R | 30 | 959 | 124 | 0.1293 | 7.70 | broken |
| pair_0015 | LoMa-R | 45 | 930 | 135 | 0.1452 | 8.12 | broken |
| pair_0015 | LoMa-R | 60 | 945 | 130 | 0.1376 | 7.61 | broken |
| pair_0015 | LoMa-R | 90 | 981 | 153 | 0.1560 | 7.49 | broken |
| pair_0015 | LoMa-R | 135 | 941 | 130 | 0.1382 | 7.94 | broken |
| pair_0015 | LoMa-R | 180 | 978 | 170 | 0.1738 | 7.69 | broken |
| pair_toronto | LoMa-B | 0 | 281 | 36 | 0.1281 | 8.84 | broken |
| pair_toronto | LoMa-B | 15 | 242 | 35 | 0.1446 | 9.38 | broken |
| pair_toronto | LoMa-B | 30 | 226 | 26 | 0.1150 | 9.76 | broken |
| pair_toronto | LoMa-B | 45 | 227 | 21 | 0.0925 | 9.71 | broken |
| pair_toronto | LoMa-B | 60 | 208 | 13 | 0.0625 | 9.27 | broken |
| pair_toronto | LoMa-B | 90 | 285 | 29 | 0.1018 | 8.97 | broken |
| pair_toronto | LoMa-B | 135 | 55 | 13 | 0.2364 | 9.89 | degraded |
| pair_toronto | LoMa-B | 180 | 258 | 24 | 0.0930 | 8.94 | broken |
| pair_toronto | LoMa-R | 0 | 283 | 26 | 0.0919 | 7.41 | broken |
| pair_toronto | LoMa-R | 15 | 252 | 30 | 0.1190 | 7.70 | broken |
| pair_toronto | LoMa-R | 30 | 233 | 28 | 0.1202 | 7.99 | broken |
| pair_toronto | LoMa-R | 45 | 246 | 31 | 0.1260 | 7.82 | broken |
| pair_toronto | LoMa-R | 60 | 232 | 21 | 0.0905 | 7.81 | broken |
| pair_toronto | LoMa-R | 90 | 292 | 35 | 0.1199 | 7.61 | broken |
| pair_toronto | LoMa-R | 135 | 226 | 21 | 0.0929 | 7.77 | broken |
| pair_toronto | LoMa-R | 180 | 277 | 27 | 0.0975 | 7.57 | broken |

## First angle that leaves 'ok' (per image × model)

- pair_0015 / LoMa-B: 0° → broken (inlier_ratio 0.163)
- pair_toronto / LoMa-B: 0° → broken (inlier_ratio 0.128)
- pair_0015 / LoMa-R: 0° → broken (inlier_ratio 0.151)
- pair_toronto / LoMa-R: 0° → broken (inlier_ratio 0.092)

## LoMa-R vs LoMa-B crossover

- pair_0015 @ 0°: B=0.163, R=0.151, Δ(R−B)=-0.012
- pair_0015 @ 15°: B=0.149, R=0.142, Δ(R−B)=-0.007
- pair_0015 @ 30°: B=0.120, R=0.129, Δ(R−B)=+0.009
- pair_0015 @ 45°: B=0.151, R=0.145, Δ(R−B)=-0.006
- pair_0015 @ 60°: B=0.128, R=0.138, Δ(R−B)=+0.010
- pair_0015 @ 90°: B=0.156, R=0.156, Δ(R−B)=+0.000
- pair_0015 @ 135°: B=0.145, R=0.138, Δ(R−B)=-0.007
- pair_0015 @ 180°: B=0.164, R=0.174, Δ(R−B)=+0.009
- pair_toronto @ 0°: B=0.128, R=0.092, Δ(R−B)=-0.036
- pair_toronto @ 15°: B=0.145, R=0.119, Δ(R−B)=-0.026
- pair_toronto @ 30°: B=0.115, R=0.120, Δ(R−B)=+0.005
- pair_toronto @ 45°: B=0.093, R=0.126, Δ(R−B)=+0.034
- pair_toronto @ 60°: B=0.062, R=0.091, Δ(R−B)=+0.028
- pair_toronto @ 90°: B=0.102, R=0.120, Δ(R−B)=+0.018
- pair_toronto @ 135°: B=0.236, R=0.093, Δ(R−B)=-0.143
- pair_toronto @ 180°: B=0.093, R=0.097, Δ(R−B)=+0.004
