# MoRE: Mixture of Reused Experts — short explainer

## What it is

Qiu, Acikalin et al. (Cornell), *MoRE: Mixture of Reused Experts*, [arXiv:2609.18176](https://arxiv.org/abs/2609.18176) (COLM 2026). In a Mixture-of-Experts transformer, groups of **R** adjacent layers share one expert pool of **R·N** experts; each layer keeps its own router; a small learnable **depth embedding** is added to each layer's input before routing. Claim: lower perplexity than a standard MoE at matched parameters and compute (paper scales 114M–1.15B on C4).

Found via [@askalphaxiv](https://x.com/askalphaxiv/status/2116032693280571682). Official full-scale code: [EricQiu6/MoRE](https://github.com/EricQiu6/MoRE) (Apache-2.0) — **not vendored** here.

## What ran here

Clean-room CPU toy under `apps/more-experts-toy/`:

- MoRE R=2 vs matched MoE (same total unique experts L·N, same top-K) vs dense (FLOPs-matched FFN) vs MoRE-without-depth-embeddings.
- Character LM on a trimmed Project Gutenberg *Alice* excerpt (~52k chars), ~200k params, 800 steps, seeds 0/1/2.
- Pre-registered hypothesis: mean(val CE MoRE) < mean(MoE) with MoE−MoRE gap ≥ std(MoRE).

## Results

**FAIL.** MoRE **1.878 ± 0.032**; MoE **1.871 ± 0.035**; gap **−0.007**. Dense **1.894 ± 0.016**. MoRE reuse rate 0.22 vs random 0.25. Breakpoints: R=4 and N=2 still do not flip the ordering; top-1 ties.

## Caveats

Toy ≠ paper. Not a reproduction of the 114M–1.15B C4 / FineWeb / downstream tables. No DeepSeek-v3 shared-expert / bias load-balancing stack — Switch-style top-K + aux loss only.

## Links

- Paper: https://arxiv.org/abs/2609.18176
- X: https://x.com/askalphaxiv/status/2116032693280571682
- Official code: https://github.com/EricQiu6/MoRE
- Demo README: `apps/more-experts-toy/README.md`
- Pages card: https://ryukijano.github.io/tech-demos/
