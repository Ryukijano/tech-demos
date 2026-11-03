# world2vec — short explainer

## What it is

Schmidt, Gonzalez Laiz, Szuppinger & Schneider (Helmholtz Munich / MCML), *world2vec: Mechanistic world models through latent dynamics learning*, [arXiv:2610.40637](https://arxiv.org/abs/2610.40637) (Oct 2026). Learns an encoder and continuous-time action-conditioned latent dynamics from observations and actions **without rewards**, by multi-step latent dynamics matching plus anti-collapse (LeJEPA / DCL). Aims for faithful long-horizon rollouts, pixel control, and interpretable system ID — not just reward.

Found via [@shumpeiMaxwell](https://x.com/shumpeiMaxwell/status/2116343144732377139). Project: [worldmodels.science](https://worldmodels.science/). Official repo: [dynamical-inference/world2vec](https://github.com/dynamical-inference/world2vec) (code not released yet). **Not vendored.**

## What ran here

Clean-room CPU toy under `apps/world2vec-toy/`:

- Tiny rendered damped pendulum (24×24) + torque actions; CNN encoder + Euler latent field vs discrete next-latent and recon baselines (~85k params).
- Primary: LRC R² at horizon H=8; gate: linear-probe R² of true state. Seeds 0/1/2.
- Pre-registered: every-seed LRC win + mean gap ≥ 0.05 + mean probe ≥ 0.50.

## Results

**FAIL.** world2vec LRC **0.351 ± 0.157** vs discrete **0.403 ± 0.267** (gap **−0.052**); only seed 0 wins. Probe R² **0.837 ± 0.077** clears the bar (no collapse). Seed 0 long-horizon curves favor continuous matching; noise and irregular-Δt breakpoints are brittle.

## Caveats

Toy ≠ paper. Not a reproduction of MuJoCo control, DreamerV3/TD-MPC2/LeWM, or 71-attractor ID. High seed variance; discrete absolute predictors can win at this scale.

## Links

- Paper: https://arxiv.org/abs/2610.40637
- Project: https://worldmodels.science/
- X: https://x.com/shumpeiMaxwell/status/2116343144732377139
- Demo README: `apps/world2vec-toy/README.md`
- Pages card: https://ryukijano.github.io/tech-demos/
