# Breakpoints

Seed 0 families from `config.yaml`. Not part of the PASS rule.

## Very light

| Mass (g) | Mean estimate (g) | Grasp |
|---:|---:|---:|
| 1.0 | 9.92 | 1.00 |
| 5.0 | 12.79 | 1.00 |
| 13.0 | 15.02 | 1.00 |

## Very heavy

| Mass (kg) | Mean estimate (kg) | Grasp |
|---:|---:|---:|
| 1.5 | -1.000 | 0.00 |
| 3.0 | -1.000 | 0.00 |
| 6.0 | -1.000 | 0.00 |

## Torque noise, 0.208 kg can

| Noise (N·m) | Mean estimate (kg) |
|---:|---:|
| 0.00 | 0.214 ± 0.000 |
| 0.03 | 0.216 ± 0.004 |
| 0.10 | 0.233 ± 0.006 |
| 0.30 | 0.214 ± 0.069 |

## Near-rigid springs

| k (N/m) | Majority levels |
|---:|---|
| 1680 | 5, 4, 4, 4 |
| 5000 | 5, 5, 5, 5 |
| 20000 | 0, 0, 0, 0 |
| 100000 | 5, 5, 5, 5 |

## Finger friction, 0.40 kg

| μ | Mean estimate (kg) | Grasp |
|---:|---:|---:|
| 0.05 | -1.000 | 0.00 |
| 0.15 | -1.000 | 0.00 |
| 0.40 | 0.408 | 1.00 |
| 1.00 | 0.409 | 1.00 |

## Jaw frictionloss, k = 1050 N/m (level 3)

| Frictionloss (N) | Levels | Median slope (N/m) |
|---:|---|---:|
| 0.0 | 3, 3, 3, 3 | 1055.3 |
| 1.0 | 3, 3, 3, 3 | 1048.1 |
| 4.0 | 3, 3, 3, 3 | 1046.2 |
| 12.0 | 3, 3, 3, 3 | 1053.1 |
