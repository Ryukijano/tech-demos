# exp1_cpu_minatar results — **FAIL**

Primary metric: mean episode return over 100 evaluation episodes vs uniform-random.

## breakout

| seed | agent mean | random mean | ratio |
|---|---:|---:|---:|
| 0 | 0.410 | 0.470 | 0.87× |
| 1 | 0.830 | 0.570 | 1.46× |
| 2 | 0.940 | 0.560 | 1.68× |

**Mean ± std:** agent **0.727 ± 0.280**, random **0.533 ± 0.055** (every_seed_win=False, mean≥2×random=False).

## space_invaders

| seed | agent mean | random mean | ratio |
|---|---:|---:|---:|
| 0 | 0.230 | 2.380 | 0.10× |
| 1 | 2.330 | 2.960 | 0.79× |
| 2 | 0.250 | 2.760 | 0.09× |

**Mean ± std:** agent **0.937 ± 1.207**, random **2.700 ± 0.295** (every_seed_win=False, mean≥2×random=False).

Status rule → **FAIL**. Params (example): 459805.
