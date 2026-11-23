# jepa-latent-maps

CPU-only clear-experiment toy of **The Content–Geometry Tradeoff in World Models: When JEPA Learns Latent Maps** (Ren, Wang, Hoang, Kuang & LeCun — NYU / AMI Labs, technical report **AGL-TR-2026-04**, 18 Nov 2026, [doi:10.5281/zenodo.24356797](https://doi.org/10.5281/zenodo.24356797)). Found via bookmarked posts by [@askalphaxiv](https://x.com/askalphaxiv/status/2124121398268559382) and [@TheTuringPost](https://x.com/TheTuringPost/status/2124303722671153346). Those posts were not fetched. The report was checked on [Zenodo](https://zenodo.org/records/24356797), the [project page](https://agenticlearning.ai/content-geometry-tradeoff/), and the [lab PDF](https://cdn.agenticlearning.ai/25c41dcbc4b7a11e/content-geometry-tradeoff-2026.pdf) on 2026-11-23. **There is no arXiv id** and **no official code**. This is a clean-room toy. See `NOTICE`.

## What is implemented (report idea, toy scale)

- **Grid:** 7×7 two-room maze, one door. A point agent takes uniform-random actions. A block stays put until the agent enters its Chebyshev radius, then jumps to a random free cell outside that radius (Section B: "the object jumps to a random position when the arm enters a contact radius").
- **Observation:** two noisy occupancy maps (agent, block), noise std 0.30, the report's linear-world noise level.
- **Model:** MLP encoder (98→64→3) and an action-conditioned MLP predictor. Latent size 3 is below the 4-D agent+object state, the capacity regime of Section B. No stop-gradient on the prediction target.
- **VICReg:** variance hinge + covariance on every frame in the batch (Bardes et al., 2022). No term pulls consecutive latents together. This is the distributional anti-collapse regularizer the report says was built for i.i.d. multi-view batches.
- **SVReg:** one-sided floor on the latent standard deviation along random directions (Equation 1).
- **Slow SVReg:** SVReg, plus a variance-normalized slowness term on the last latent channel (Equation 2; the report uses the last 1/8, at least one channel) and a gated residual predictor initialized at 0.1 (Equation 3).
- **Collapse scale, fixed in advance:** at a fully collapsed latent, SVReg's penalty is λ and VICReg's variance hinge is `vicreg_weight × 25`. Primary λ = 10, so `vicreg_weight = 0.4`. Slow-channel weight is `1/λ = 0.1`, the coupling in the report's Table 6. Not fit on these metrics.
- **Geometry:** held-out ridge R² of agent position, and Spearman correlation between latent L2 and maze geodesic (Section 3).
- **Content:** held-out ridge R² of the block on frames where it did not just jump and the agent is outside the contact neighborhood (the report's free-block readout).

Not claimed: PointMaze / Reacher / Cube / PushT planning, ResNet or ViT world models, SIGReg-BN, stop-gradient, city-walk maps, or any number in the report's Tables 1–11.

## Status (honest): **PENDING**

The official seeds have not been trained in this commit. The pass/fail bar below is frozen in `config.yaml` and in this section. A later commit fills the table. It does not edit the bar.

| Check (fixed before the official run) | Threshold |
|---|---|
| Every seed, Slow SVReg agent R² | ≥ 0.60 |
| Every seed, Slow SVReg geodesic ρ | ≥ 0.45 |
| Every seed, Slow SVReg free-object R² | ≥ 0.45 |
| Every seed, Slow geodesic ρ > VICReg | strict |
| Every seed, SVReg geodesic ρ > VICReg | strict |
| Every seed, Slow free-object R² > SVReg | strict |
| Mean Slow agent R² | ≥ 0.70 |
| Mean (SVReg − VICReg) geodesic ρ | ≥ 0.05 |
| Mean (Slow − VICReg) geodesic ρ | ≥ 0.05 |
| Mean (Slow − SVReg) free-object R² | ≥ 0.10 |
| Mean (SVReg − Slow) geodesic ρ | ≤ 0.05 |

Status rule: **PASS** if every row holds. **PARTIAL** if mean Slow geodesic beats VICReg and mean Slow free-object R² beats SVReg, but any row fails. **FAIL** otherwise.

The 0.05 / 0.10 gaps are toy margins chosen before training. They are not copied from a table. The report's U-maze geodesic gap (SVReg − VICReg) is about 0.22 and its Medium gap is about 0.03. The report calls an object "kept" at R² ≥ 0.70 in the linear experiment; the absolute floors here are lower because the encoder is a 3-D bottleneck on noisy pixels.

## Experiment (`experiments/exp1_cpu_grid/`)

- **Hypothesis:** Slow SVReg keeps a latent map of the agent and a decodable resting block; VICReg's map is worse; SVReg's map is better than VICReg's and its block readout is worse than Slow SVReg's.
- **Seeds:** 0, 1, 2. They change initialization and minibatch order. The three trajectories (train / ridge-fit / ridge-score) are fixed.
- **Breakpoints (seed 0):** latent dim {2, 3, 8, 16}; regularizer weight λ ∈ {0.3, 1, 10, 100} with the same 1/λ and VICReg coupling; training length {100, 400} against the primary 800 steps.
- **Machine:** CPU only.

## Run (CPU)

```bash
cd apps/jepa-latent-maps
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
./run_all.sh
# python src/run_experiment.py --smoke   # disclosed code-path check; does not set thresholds
```

## Caveats

- **Not the authors' code or scale.** No official repository is linked from the project page or the Zenodo record (checked 2026-11-23). A third-party toy, `aicumene/BOID-JEPA`, is not vendored.
- **Not the visual protocol.** The report trains ResNet encoders and block-causal ViT predictors on 5k–470k frames at 128–224 px and plans with CEM. That setup is not run. A Colab of it would need code the authors have not released.
- **Jaggedness** uses a trailing window of 3 because the scripted route is 9 cells. The report uses 9 on long routes. Jaggedness is report-only.
- The block jump is the Section B contact rule on a grid, not a physical push, and not PushT or Cube.
- Trained weights stay out of git (`work/` is git-ignored).

## Licences

- App harness / this folder: **MIT** (`LICENSE`). Clean-room; no upstream code vendored.
- The report is an open Zenodo record (AGL-TR-2026-04). It is not a software licence, and no code was copied from it.
