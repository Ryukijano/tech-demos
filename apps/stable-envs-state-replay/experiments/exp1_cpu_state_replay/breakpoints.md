# Breakpoints (report-only)

## Non-native resolution

- TwoRoom size 112: refused. `TwoRoom renders at 224px only (got 112); every length in the reference is in output pixels. Resize the frames afterwards instead.`
- TwoRoom size 448: refused. `TwoRoom renders at 224px only (got 448); every length in the reference is in output pixels. Resize the frames afterwards instead.`
- CartPole size 300: refused. `CartPole renders only at its reference's native 400x600 (got size=300); the reference draws every primitive in fixed pixels. Resample the output instead.`
- CartPole size 1200: refused. `CartPole renders only at its reference's native 400x600 (got size=1200); the reference draws every primitive in fixed pixels. Resample the output instead.`

Nearest 2× upsample of TwoRoom seed 0 is [128, 448, 448, 3] and 77070602 bytes versus native npz 19267850 bytes. Nearest-neighbour upsample of native frames, not a stable-envs render.

## long_episode

- TwoRoom B=1 T=512 frames=512: PNG saving 99.43%, npz saving 99.99%, lossless 1.000, replay 27398.8 fps, load 16530.9 fps, max abs 0.
- Fraction of steps whose agent position did not change: 0.000.

## many_envs

- TwoRoom B=64 T=8 frames=512: PNG saving 99.44%, npz saving 99.99%, lossless 1.000, replay 31474.8 fps, load 15162.3 fps, max abs 0.

## third_family

- Acrobot B=2 T=16 frames=32: PNG saving 91.40%, npz saving 99.96%, lossless 1.000, replay 10555.8 fps, load 4192.7 fps, max abs 0.

## Variation omitted

- TwoRoom collected with agent.color [0, 80, 255]. Replay at the default appearance matches 0.000 of frames (max abs 255). Passing the stored variation row back to render matches 1.000 (max abs 0).
- State file 2269 bytes; state plus variation row 2966 bytes.

## Fresh process

- Parent collection hash `cfe0dbfb0f6fd9bf5cee6e8b8c1a2dff433cf00bc3227428eb732dddd5315e30` vs a new process re-render `cfe0dbfb0f6fd9bf5cee6e8b8c1a2dff433cf00bc3227428eb732dddd5315e30`: match.

## PNG compress level 9

- TwoRoom seed 0 at compress_level 9: 413926 bytes, saving vs states 99.02%.
