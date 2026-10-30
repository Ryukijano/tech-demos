# NanoDreamer on MinAtar — short explainer

## What it is

Benjamin Bai, *NanoDreamer: Learning to Act Through Imagination* (October 2026). A ~750k-parameter Dreamer-style world model + actor–critic that trains the actor entirely in imagined rollouts and reports beating random on all five MinAtar games after 500k policy steps per game (~3h on an M5 Pro). Write-up: https://www.benjamin-bai.com/projects/nanodreamer — code: https://github.com/benyebai/NanoDreamer.

Found via [@benbye](https://x.com/benbye/status/2115488344050192584).

## Licence

Verified 2026-10-30: GitHub `license=null`, no LICENSE file in the repo → **all rights reserved**. This demo is a **clean-room PyTorch reimplementation** of the write-up; no upstream source is vendored or installed.

## What ran here

Independent CPU toy under `apps/nanodreamer-minatar/`:

- ~460k-param CNN + causal transformer world model + MLP actor/critic.
- Imagination-only actor updates; MinAtar Breakout + Space Invaders; 16k policy steps; seeds 0/1/2.
- Pre-registered hypothesis: beat random on both games (every seed, and mean ≥ 2× random).

## Results

**FAIL.** Breakout mean 0.727 vs random 0.533 (misses every-seed and 2× bars). Space Invaders mean 0.937 vs random 2.700 (below random on every seed). Breakpoints: under 4k steps or a ~74k-param stack the Breakout toy collapses; horizon 5 helps vs horizon 1.

## Caveats

Toy ≠ upstream 500k-step table. No custom autograd; simplified scalar rewards. Honest negative at this CPU budget.

## Links

- Write-up: https://www.benjamin-bai.com/projects/nanodreamer
- Upstream (unlicensed): https://github.com/benyebai/NanoDreamer
- X: https://x.com/benbye/status/2115488344050192584
- Demo README: `apps/nanodreamer-minatar/README.md`
