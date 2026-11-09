# exp1_cpu_tunnel results

Status: **PASS**

MuJoCo 4.0.0. `mjENBL_SPECULATIVE` = 128 (XML `speculative`, default `disable`).

Pre-registered PASS rule: speculative ON tunneling rate ≤ 0.05 at 2, 4, 8 and 16 ms; speculative OFF ≤ 0.10 at 2 ms, ≥ 0.35 at 8 ms, and ≥ 0.60 at 16 ms. Means over seeds 0/1/2.

| Check | Result |
|---|---|
| `on_rate_at_most_0.05_all_primary_dts` | met |
| `off_rate_at_most_0.10_at_1x` | met |
| `off_rate_at_least_0.35_at_4x` | met |
| `off_rate_at_least_0.60_at_8x` | met |
| `partial_on_rate_at_most_0.20` | met |
| `partial_gap_at_least_0.25_at_8x` | met |
| `partial_off_above_on_at_4x` | met |

## Mean over seeds

| dt (ms) | × default | OFF rate | ON rate | Kinematic skip | OFF penetration (mm) | ON penetration (mm) | OFF KE drift | ON KE drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1× | 0.005 ± 0.009 | 0.000 ± 0.000 | 0.000 ± 0.000 | 3.521 ± 0.270 | 0.313 ± 0.025 | -0.974 ± 0.010 | -1.000 ± 0.000 |
| 4 | 2× | 0.443 ± 0.050 | 0.000 ± 0.000 | 0.010 ± 0.018 | 4.550 ± 0.655 | 0.260 ± 0.067 | -0.553 ± 0.049 | -1.000 ± 0.000 |
| 8 | 4× | 0.792 ± 0.024 | 0.000 ± 0.000 | 0.448 ± 0.077 | 2.511 ± 0.349 | 0.312 ± 0.052 | -0.208 ± 0.024 | -1.000 ± 0.000 |
| 16 | 8× | 0.958 ± 0.024 | 0.000 ± 0.000 | 0.807 ± 0.055 | 0.880 ± 0.416 | 0.415 ± 0.093 | -0.042 ± 0.024 | -1.000 ± 0.000 |

Kinematic skip is the fraction of the same shots whose evenly spaced free-flight samples miss the wall. It is not a MuJoCo result. Contact penetration is the mean, over shots, of the deepest negative contact distance seen by `mj_step1` (0 if that mode never touched).

## Per seed

| Seed | dt (ms) | OFF | ON | Kinematic | OFF contact | ON contact |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 0.016 | 0.000 | 0.000 | 1.000 | 1.000 |
| 0 | 4 | 0.500 | 0.000 | 0.031 | 0.969 | 1.000 |
| 0 | 8 | 0.812 | 0.000 | 0.500 | 0.500 | 1.000 |
| 0 | 16 | 0.938 | 0.000 | 0.812 | 0.188 | 1.000 |
| 1 | 2 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| 1 | 4 | 0.406 | 0.000 | 0.000 | 1.000 | 1.000 |
| 1 | 8 | 0.766 | 0.000 | 0.484 | 0.516 | 1.000 |
| 1 | 16 | 0.984 | 0.000 | 0.859 | 0.141 | 1.000 |
| 2 | 2 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| 2 | 4 | 0.422 | 0.000 | 0.000 | 1.000 | 1.000 |
| 2 | 8 | 0.797 | 0.000 | 0.359 | 0.641 | 1.000 |
| 2 | 16 | 0.953 | 0.000 | 0.750 | 0.250 | 1.000 |

## Wall-clock (`mj_step`, after warmup)

| Spheres | dt (ms) | OFF µs/step | ON µs/step | ON / OFF |
|---|---:|---:|---:|---:|
| 1 | 2 | 1.72 | 1.83 | 1.07 |
| 1 | 16 | 1.73 | 1.81 | 1.05 |
| 8 | 2 | 3.54 | 4.09 | 1.15 |
| 8 | 16 | 3.54 | 4.24 | 1.20 |
