# MuJoCo 4 speculative contacts — short explainer

MuJoCo 4.0.0 (Google DeepMind, 6 November 2026) is the first major release since 3.0. One of its new switches is speculative contacts. Discrete collision only looks at the pose at each timestep, so a fast or thin shape can jump from one side of an obstacle to the other without ever overlapping in a sampled pose. The new flag sweeps convex geoms forward by their pre-step velocity and adds a contact early enough for the solver to stop them. It is off by default.

Turn it on in MJCF with `<flag speculative="enable"/>`, or in Python by setting bit `mjENBL_SPECULATIVE` (128) on `mjModel.opt.enableflags`. The 4.0.0 wheel, the changelog, discussion #3773, and the XML reference agree on that name. Coverage in this release is convex primitives and meshes on CPU.

This CPU experiment fires a small dense sphere at a 4 mm wall. Radius and speed are redrawn for seeds 0, 1, and 2. The timestep runs from the default 2 ms out to 16 ms. The pre-registered bar is: with the flag on, the tunneling rate stays at or below 0.05 at every one of those timesteps; with the flag off, the wall holds at 2 ms (rate ≤ 0.10) and most shots tunnel at 8 ms (≥ 0.35) and at 16 ms (≥ 0.60). Penetration and kinetic-energy drift are reported and are not part of that bar. Separate breakpoints push speed, timestep, wall thickness, and acceleration-from-rest, which the docs say the sweep cannot see because it ignores acceleration.

The official numbers are in `experiments/exp1_cpu_tunnel/results.md` after `./run_all.sh`.

Links: [release 4.0.0](https://github.com/google-deepmind/mujoco/releases/tag/4.0.0), [discussion #3773](https://github.com/google-deepmind/mujoco/discussions/3773), [speculative contacts](https://mujoco.readthedocs.io/en/latest/computation/speculative.html), [bookmark](https://x.com/_bbelousov/status/2119200725008024035) (not re-fetched).
