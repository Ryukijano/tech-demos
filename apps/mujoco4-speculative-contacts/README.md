# mujoco4-speculative-contacts

CPU-only clear experiment on **speculative contacts** in MuJoCo 4.0.0 (Google DeepMind). Found via a bookmarked X post by [@_bbelousov](https://x.com/_bbelousov/status/2119200725008024035). That post was **not** fetched. The feature was checked on the [4.0.0 release](https://github.com/google-deepmind/mujoco/releases/tag/4.0.0), the [changelog](https://mujoco.readthedocs.io/en/stable/changelog.html), [discussion #3773](https://github.com/google-deepmind/mujoco/discussions/3773), the [speculative-contacts chapter](https://mujoco.readthedocs.io/en/latest/computation/speculative.html), the [XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html), and an installed PyPI wheel `mujoco==4.0.0`. MuJoCo is **Apache-2.0** and is **not vendored**. See `NOTICE`.

![panel](artifacts/mujoco4-speculative-contacts.png)

*One registered shot: a 7 mm sphere at 4 m/s, 16 ms timestep (8× the default), into a 4 mm wall. Left, speculative contacts off. Right, on. Video: [`artifacts/mujoco4-speculative-contacts.mp4`](artifacts/mujoco4-speculative-contacts.mp4).*

## The flag

MuJoCo 4.0.0 adds an optional enable flag. It is off unless you turn it on.

```xml
<option timestep="0.016"><flag speculative="enable"/></option>
```

In the Python bindings that bit is `mujoco.mjtEnableBit.mjENBL_SPECULATIVE` = **128** (`1<<7`) inside `mjModel.opt.enableflags`. Compiling the XML above sets that bit; `speculative="disable"` (the default) leaves it clear. The installed wheel reports `__version__ == "4.0.0"`. The engine still only does this for **convex primitives and meshes on CPU**. Heightfields, SDFs, flexes, MJX, and MuJoCo Warp are out of scope, and this experiment does not touch them.

Speculative contacts build a contact before the geometries overlap: a swept broadphase, conservative advancement for a time of impact, a line-of-sight test, and a synthetic distance so the solver arrives on the surface at the end of the step. The sweep uses pre-step `qvel` and ignores acceleration. A body that is still at rest, and is then accelerated through a wall inside one step, can still tunnel. That case is a breakpoint below, not the headline test.

## Hypothesis

With the flag **on**, a free sphere fired at a thin fixed box does not end strictly on the far side of the wall, at timesteps 1×, 2×, 4×, and 8× the default 0.002 s. With the flag **off**, the same shots tunnel at 4× and 8×, and the wall still holds at 1×. Seeds 0, 1, and 2 redraw the radius and the speed. Within a seed, both flags and every timestep see the same shots.

The shot box is fixed so that, at 0.002 s, `speed × dt` is shorter than `thickness + radius` for every draw (a discrete sample cannot jump from “separated” to “center already past the far face”). At 0.008 s and 0.016 s the free-flight step is usually longer than the whole overlap window. A kinematic prior, from 20,000 uniform draws and **no** MuJoCo call, puts the discrete-skip rate at 0, 0.011, 0.455, and 0.828 for 2, 4, 8, and 16 ms.

**Tunneled** means the sphere center’s maximum x satisfies `center_x − radius > half_thickness + 0.1 mm`. That counts a missed collision and a contact that does not keep the sphere on the near side. Secondary, not gated: contact rate, peak contact penetration, and `(KE_final − KE_init) / KE_init`.

## Status (honest): **PENDING**

Official seeds 0/1/2 are recorded by `./run_all.sh` into `experiments/exp1_cpu_tunnel/results.md` and `status.json`. This headline is updated from that run. Thresholds below were written before that run.

| Check (fixed in `config.yaml` before the run) | Threshold |
|---|---|
| ON tunneling rate at 2, 4, 8, and 16 ms | ≤ 0.05 |
| OFF tunneling rate at 2 ms (1×) | ≤ 0.10 |
| OFF tunneling rate at 8 ms (4×) | ≥ 0.35 |
| OFF tunneling rate at 16 ms (8×) | ≥ 0.60 |

Status rule: **PASS** needs all four. **PARTIAL** if ON stays ≤ 0.20 at every primary timestep, OFF − ON ≥ 0.25 at 16 ms, and OFF > ON at 8 ms, but a strict bar fails. **FAIL** otherwise. The 0.35 and 0.60 floors sit below the kinematic skip prior (0.455 at 4×, 0.828 at 8×).

## Experiment (`experiments/exp1_cpu_tunnel/`)

- **Scene:** fixed box, half-thickness 2 mm (4 mm wall), 10 cm in y and z. Sphere density 1000 kg/m³, radius uniform on [6 mm, 8 mm], speed uniform on [2.5, 4.5] m/s, launched from x = −0.08 m. Gravity off. Integrator Euler, solver Newton, 100 iterations (MuJoCo 4.0.0 defaults). Contact `condim=1`, `solref="0.002 1"`, `solimp="0.95 0.99 0.001"`.
- **Shots:** 64 per seed. Horizon is a 0.30 m free flight plus 8 settle steps.
- **Pre-registration:** `config.yaml` holds the hypothesis, the metric, and the thresholds. A disclosed design probe (one deterministic shot, then 40 shots on seed 0) checked the flag bit and that a near-face overlap stops a density-1000 sphere. It did not move a threshold. `smoke.json`, when present, is a 4-shot code-path check only.
- **Seeds:** 0, 1, 2.
- **Machine:** CPU only. `./run_all.sh` runs the sweep, the breakpoints, the figure, and the video.

![plot](experiments/exp1_cpu_tunnel/plot.png)

Tables: `results.md` / `results.csv`. Verdict: `status.json`. Raw rollouts: `raw.json`.

## Breakpoints (`breakpoints.md`)

Seed 0, one controlled projectile per cell. Not part of the PASS rule.

- **Huge speed.** 20, 80, 300, and 1000 m/s at 2 ms and at 16 ms. The sweep uses pre-step velocity, so a fast constant speed is the case the flag is for, until time-of-impact or the solver gives up.
- **Huge timestep.** 32 ms through 0.5 s at 4 m/s. No applied acceleration, so a miss here is numerical. The docs say a timestep still cannot be arbitrarily large when acceleration during the step is ignored; that is the next cell.
- **Thinner wall.** Half-thickness 0.5 mm, 0.1 mm, 0.02 mm, and 0.005 mm, at 16 ms.
- **Acceleration from rest.** `qvel = 0`, gravity 50, 200, and 800 m/s² along +x, 16 ms, 1 mm wall. The first Euler step moves `a dt²`. Once that step clears the wall, both flags are expected to tunnel.
- **Wall-clock.** Microseconds per `mj_step`, flag off vs on, 1 and 8 fast spheres, 2 ms and 16 ms, after a warmup.

## Run (CPU)

```bash
cd apps/mujoco4-speculative-contacts
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
# python src/run_experiment.py --smoke   # disclosed tiny code-path check
```

The still and the MP4 are rendered offscreen with MuJoCo’s OSMesa software renderer (`MUJOCO_GL=osmesa`, set inside `src/make_media.py` before MuJoCo is imported). If OSMesa cannot start, that script writes a matplotlib trajectory animation instead and says so in `artifacts/media_backend.txt`.

## Caveats

- **Not DeepMind’s demo.** Discussion #3773 describes a robot arm catching a 0.7 mm card, a Newton’s cradle of thin plates, and a chain. This experiment is a sphere against one box. It does not measure those scenes, and it does not measure elasto-plasticity or Filament.
- **Convex primitives on CPU only.** A sphere and a box are in the documented coverage. Nothing here speaks for meshes at scale, flexes, SDFs, heightfields, MJX, or Warp.
- **Acceleration is invisible to the sweep.** The rest-to-wall breakpoint is the failure mode the computation chapter describes. A large constant `qvel` is a different question from a large force applied to a body that is still at rest.
- **Contact is unilateral.** A discrete sample whose center has already crossed the far face can report a contact that does not push the sphere back to the near side. The tunneling rate counts that as a pass-through. It can sit above the pure kinematic skip rate.
- **Energy drift is not a pass gate.** The contact is the default damped `solref`, so a caught sphere loses most of its kinetic energy. A tunneled sphere keeps it. That is reported, not thresholded.
- Upstream Apache-2.0 code was **not** copied into this directory.

## Licences

- This directory: **Apache-2.0** (`LICENSE`), chosen to match MuJoCo.
- MuJoCo (DeepMind Technologies Limited): **Apache-2.0**. Install `mujoco==4.0.0` from PyPI. Not vendored.
