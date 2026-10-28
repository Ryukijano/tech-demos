# Binary Discrete Diffusion on Sudoku — short explainer

## What it is

Casado Noguerales et al. (ETH / MPI-IS / ELLIS), *Binary Discrete Diffusion: Candidate-Set Diffusion for Logic and Planning Tasks*, [arXiv:2610.28543](https://arxiv.org/abs/2610.28543) (20 Oct 2026). Instead of masking or uniformly replacing tokens, the noisy state at each position is a **set of candidate tokens** encoded as a binary membership vector. Denoising adds/removes candidates. On Sudoku-Extreme (and Boxoban / graph coloring) their 43M DiT with the **binary symmetric** kernel beats matched masked and uniform baselines, especially on hard instances.

Found via [@rodcasnog](https://x.com/rodcasnog/status/2113287405142675759). **No official code** located (arXiv, [author page](https://rodcasnog.github.io/research/), GitHub search, 2026-10-28).

## What ran here

Independent CPU toy under `apps/binary-discrete-diffusion-sudoku/`:

- Matched ~111k-param transformers, binary-symmetric training vs masked training.
- 50-clue synthetic 9×9 Sudoku, 48 held-out puzzles, NFE=32, seeds 0/1/2.
- Pre-registered hypothesis: binary solve rate > masked on every seed with mean gap ≥ 5 pp.

## Results

**FAIL.** Masked solve rate **0.688 ± 0.042**; binary **0.000 ± 0.000**. Binary still reaches ~0.79 cell accuracy but almost never a full valid board under the toy prune sampler. Breakpoints: masked collapses as clues drop to ~30 or NFE→2; binary stays near zero.

## Caveats

Toy ≠ paper. Not a reproduction of Table 1. Simplified samplers; no Sudoku-Extreme; no 43M model. A GPU Colab notebook is prepared (unexecuted) for a larger run — needs owner GPU budget.

## Links

- Paper: https://arxiv.org/abs/2610.28543
- X: https://x.com/rodcasnog/status/2113287405142675759
- Demo README: `apps/binary-discrete-diffusion-sudoku/README.md`
- Pages card: https://ryukijano.github.io/tech-demos/
