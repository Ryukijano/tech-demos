# Breakpoints

Seed 0. One controlled projectile per cell, not the random primary box. These cells do not change the PASS rule. `tunneled` means the sphere finished strictly on the far side of the wall (or the state diverged).

## Huge initial speed

4 mm wall, radius 7 mm, the media-shot geometry. Speculative contacts sweep pre-step velocity, so a large constant speed is the case they are built for.

| Speed (m/s) | dt (ms) | OFF tunneled | ON tunneled | OFF contact | ON contact |
|---:|---:|---|---|---|---|
| 20 | 2 | no | no | yes | yes |
| 20 | 16 | yes | no | no | yes |
| 80 | 2 | yes | no | no | yes |
| 80 | 16 | yes | no | no | yes |
| 300 | 2 | yes | no | no | yes |
| 300 | 16 | yes | no | no | yes |
| 1000 | 2 | yes | no | no | yes |
| 1000 | 16 | yes | no | no | yes |

## Huge timestep

Speed 4 m/s, radius 7 mm, 4 mm wall. No applied acceleration, so a miss here is numerical rather than the documented acceleration gap.

| dt (ms) | × default | OFF tunneled | ON tunneled | OFF contact | ON contact | diverged OFF/ON |
|---:|---:|---|---|---|---|---|
| 32 | 16× | yes | no | no | yes | no/no |
| 64 | 32× | yes | no | no | yes | no/no |
| 128 | 64× | yes | no | no | yes | no/no |
| 256 | 128× | yes | no | no | yes | no/no |
| 500 | 250× | yes | no | no | yes | no/no |

## Thinner wall

Speed 4 m/s, radius 7 mm, timestep 16 ms. Half-thickness is the parameter.

| Half-thickness (mm) | OFF tunneled | ON tunneled | OFF contact | ON contact |
|---:|---|---|---|---|
| 0.5000 | yes | no | no | yes |
| 0.1000 | yes | no | no | yes |
| 0.0200 | yes | no | no | yes |
| 0.0050 | yes | no | no | yes |

## Acceleration from rest

Documented limit: the sweep uses pre-step `qvel` and ignores acceleration. The sphere starts at rest. The first-step Euler displacement is `a dt²` with dt = 16 ms. A 1 mm wall (half-thickness 0.5 mm) sits in front of a 6 mm sphere launched from x = −0.03 m.

| Gravity (m/s²) | First-step x (m) | OFF tunneled | ON tunneled | OFF contact | ON contact |
|---:|---:|---|---|---|---|
| 50 | -0.0172 | yes | no | no | yes |
| 200 | 0.0212 | yes | yes | no | no |
| 800 | 0.1748 | yes | yes | no | no |

## Wall-clock

See the table in `results.md`. Cost is microseconds per `mj_step` on this CPU, flag off versus on.
