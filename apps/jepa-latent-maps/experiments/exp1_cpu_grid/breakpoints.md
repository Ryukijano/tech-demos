# Breakpoints (seed 0)

## latent_dim

Section B: at latent size 3, below the 4-D agent+object state, SVReg drops the object on a large fraction of runs; with more capacity the object is kept. Dim 3 is the primary setting.

| setting | method | agent R² | geodesic ρ | free-object R² | jaggedness |
|---|---|---:|---:|---:|---:|
| dim=2 | Slow SVReg | -0.0300 | 0.0354 | -0.2383 | 2.5766 |
| dim=2 | SVReg | -0.0272 | 0.0357 | -0.2280 | 2.6977 |
| dim=2 | VICReg | 0.0136 | 0.0571 | -0.3034 | 2.5946 |
| dim=3 | Slow SVReg | 0.0914 | 0.0909 | -0.2546 | 2.7073 |
| dim=3 | SVReg | 0.0799 | 0.0794 | -0.2554 | 2.7780 |
| dim=3 | VICReg | 0.0536 | 0.0780 | -0.2817 | 2.2114 |
| dim=8 | Slow SVReg | 0.0874 | 0.0661 | -0.3169 | 2.7848 |
| dim=8 | SVReg | 0.0826 | 0.0638 | -0.3027 | 2.8952 |
| dim=8 | VICReg | 0.0917 | 0.0375 | -0.2800 | 2.6598 |
| dim=16 | Slow SVReg | 0.1622 | 0.0983 | -0.2509 | 2.6969 |
| dim=16 | SVReg | 0.1466 | 0.0714 | -0.2367 | 2.6055 |
| dim=16 | VICReg | 0.1218 | 0.0486 | -0.2347 | 2.5713 |

## reg_weight

SVReg weight λ, with slow weight kept at 1/λ and the VICReg variance hinge matched to the same collapse penalty. The report's grids run from 0.3 to 1000; these four values are inside that range.

| setting | method | agent R² | geodesic ρ | free-object R² | jaggedness |
|---|---|---:|---:|---:|---:|
| λ=0.3 | Slow SVReg | 0.0262 | 0.0321 | -0.2198 | 2.5261 |
| λ=0.3 | SVReg | 0.0170 | 0.0449 | -0.2073 | 2.7784 |
| λ=0.3 | VICReg | 0.0158 | 0.0233 | -0.2312 | 2.2785 |
| λ=1.0 | Slow SVReg | 0.0015 | 0.0493 | -0.2379 | 2.9655 |
| λ=1.0 | SVReg | 0.0628 | 0.0597 | -0.2362 | 2.8831 |
| λ=1.0 | VICReg | 0.0512 | 0.0598 | -0.2141 | 3.1520 |
| λ=10.0 | Slow SVReg | 0.0914 | 0.0909 | -0.2546 | 2.7073 |
| λ=10.0 | SVReg | 0.0799 | 0.0794 | -0.2554 | 2.7780 |
| λ=10.0 | VICReg | 0.0536 | 0.0780 | -0.2817 | 2.2114 |
| λ=100.0 | Slow SVReg | 0.0778 | 0.0774 | -0.2594 | 2.6389 |
| λ=100.0 | SVReg | 0.0709 | 0.0731 | -0.2599 | 2.6751 |
| λ=100.0 | VICReg | 0.0450 | 0.0656 | -0.2769 | 2.0612 |

## short_training

Under-trained versus the primary budget of 800 steps.

| setting | method | agent R² | geodesic ρ | free-object R² | jaggedness |
|---|---|---:|---:|---:|---:|
| steps=100 | Slow SVReg | -0.0014 | 0.0207 | -0.2714 | 2.8500 |
| steps=100 | SVReg | -0.0024 | 0.0139 | -0.2716 | 2.8552 |
| steps=100 | VICReg | -0.0013 | 0.0047 | -0.2736 | 2.6735 |
| steps=400 | Slow SVReg | 0.0490 | 0.0661 | -0.2782 | 2.5473 |
| steps=400 | SVReg | 0.0364 | 0.0599 | -0.2782 | 2.6887 |
| steps=400 | VICReg | 0.0125 | 0.0495 | -0.2769 | 2.2640 |
