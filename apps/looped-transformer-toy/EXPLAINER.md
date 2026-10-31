# How to Loop Your Transformer — short explainer

## What it is

Ekbote et al. (FAIR at Meta / MIT), *How to Loop Your Transformer: Loop Scaling, Specialization, and Parallel Inference*, [arXiv:2610.40325](https://arxiv.org/abs/2610.40325) (Oct 2026). Looping reuses transformer weights by repeatedly applying the same stack (or parts of it), trading compute for depth at a fixed parameter budget. The paper compares full-stack / middle-stack / per-layer looping, adds loop-specific LoRA adapters when gains saturate, and studies Parallel Loop Transformers for decoding latency — on dense decoder-only LMs up to 1B params.

Found via [@iScienceLuvr](https://x.com/iScienceLuvr/status/2116286474090107140). **No official code** located (arXiv; HF Papers 404 for this id, 2026-10-31).

## What ran here

Clean-room CPU toy under `apps/looped-transformer-toy/`:

- Full-stack looping only: 2 shared blocks × K=4 vs parameter-matched K=1 baseline (~106k params).
- LSB-first 4-digit addition, 512 eval examples, seeds 0/1/2.
- Pre-registered hypothesis: looped exact-match > baseline on every seed with mean gap ≥ 10 pp.

## Results

**FAIL.** Looped **0.997 ± 0.006**; baseline **1.000 ± 0.000** (ceiling). Gap **−0.003**. Breakpoints: the trained looped model needs K≥2 at test time (K=1 → 0.627); K=16 drops to 0.934; neither model length-generalizes off 4 digits.

## Caveats

Toy ≠ paper. Not a reproduction of the 1B LM / DCLM tables. No LoRA specialization or Parallel Loop Transformers. Ceiling on a too-easy algorithmic task falsifies the pre-registered gain bar.

## Links

- Paper: https://arxiv.org/abs/2610.40325
- X: https://x.com/iScienceLuvr/status/2116286474090107140
- Demo README: `apps/looped-transformer-toy/README.md`
- Pages card: https://ryukijano.github.io/tech-demos/
