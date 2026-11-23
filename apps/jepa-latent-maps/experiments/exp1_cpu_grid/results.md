# exp1_cpu_grid — FAIL

Thresholds were read from `config.yaml` (frozen before this run).

| Check | Bar | Result |
|---|---|---|
| every seed slow agent R² ≥ 0.6 | fixed | **not met** |
| every seed slow geodesic ρ ≥ 0.45 | fixed | **not met** |
| every seed slow free-object R² ≥ 0.45 | fixed | **not met** |
| every seed slow geodesic ρ > VICReg | fixed | **not met** |
| every seed SVReg geodesic ρ > VICReg | fixed | **not met** |
| every seed slow free-object R² > SVReg | fixed | **not met** |
| mean slow agent R² ≥ 0.7 | fixed | **not met** |
| mean SVReg−VICReg geodesic ≥ 0.05 (-0.0020) | fixed | **not met** |
| mean Slow−VICReg geodesic ≥ 0.05 (0.0036) | fixed | **not met** |
| mean Slow−SVReg free-object R² ≥ 0.1 (-0.0291) | fixed | **not met** |
| mean SVReg−Slow geodesic ≤ 0.05 (-0.0056) | fixed | **met** |

## Per seed

| seed | method | agent R² | geodesic ρ | free-object R² | jaggedness | PR |
|---|---|---:|---:|---:|---:|---:|
| 0 | VICReg | 0.0536 | 0.0780 | -0.2817 | 2.2114 | 2.7314 |
| 0 | SVReg | 0.0799 | 0.0794 | -0.2554 | 2.7780 | 2.6864 |
| 0 | Slow SVReg | 0.0914 | 0.0909 | -0.2546 | 2.7073 | 2.5763 |
| 1 | VICReg | 0.0388 | 0.0606 | -0.1962 | 3.2701 | 2.6673 |
| 1 | SVReg | 0.0493 | 0.0512 | -0.2102 | 2.8650 | 2.5452 |
| 1 | Slow SVReg | 0.0571 | 0.0567 | -0.2200 | 2.6396 | 2.5344 |
| 2 | VICReg | 0.0329 | 0.0580 | -0.1901 | 2.5065 | 2.7147 |
| 2 | SVReg | 0.0262 | 0.0599 | -0.2095 | 3.2531 | 2.6745 |
| 2 | Slow SVReg | -0.0069 | 0.0597 | -0.2878 | 3.0722 | 2.6625 |

## Mean ± sample std

- **VICReg:** Agent R² 0.0418 ± 0.0106; Geodesic ρ 0.0655 ± 0.0109; Free-object R² -0.2227 ± 0.0512
- **SVReg:** Agent R² 0.0518 ± 0.0269; Geodesic ρ 0.0635 ± 0.0144; Free-object R² -0.2251 ± 0.0263
- **Slow SVReg:** Agent R² 0.0472 ± 0.0499; Geodesic ρ 0.0691 ± 0.0190; Free-object R² -0.2541 ± 0.0339

## Observation ceiling (report only)

Linear ridge on the noisy pixels: agent R² 0.3382, free-object R² 0.1360.
