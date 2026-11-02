# Platonic JEPA — short explainer

## What it is

Balestriero & LeCun (Brown / NYU / AMI Labs), *Proving and Improving the Platonic Representation Hypothesis with JEPAs*, [arXiv:2610.39302](https://arxiv.org/abs/2610.39302) (Oct 2026). Separately trained JEPAs with whitened second-order anti-collapse learn the spectral embedding of their positive pairs; when positives are **good views** (two observations of the same world state), solutions coincide **up to a rotation** (Platonic Representation Theorem). **Bad views** (corruptions of one observation) spend dimensions on modality nuisance that no rotation can align. The paper also frames next-token prediction as a bad-view JEPA and shows good-view text JEPAs aligning with vision far more sample-efficiently than large LMs.

Found via [@HuggingPapers](https://x.com/HuggingPapers/status/2116804737207021735). Official code: [galilai-group/prh-proof](https://github.com/galilai-group/prh-proof); demo Space: [galilai-group/prh_demo](https://huggingface.co/spaces/galilai-group/prh_demo). **Not vendored.**

## What ran here

Clean-room CPU toy under `apps/jepa-platonic-toy/`:

- Synthetic 32-state world, two linear modalities; VICReg-style whitened JEPA (~7k params).
- Procrustes-aligned linear CKA vs reconstruction and random baselines, seeds 0/1/2.
- Pre-registered hypothesis: every-seed win + mean CKA ≥ 0.55 + gaps ≥ 0.20 (vs recon) / ≥ 0.35 (vs random).

## Results

**PARTIAL.** JEPA **0.958 ± 0.027**; recon **0.790 ± 0.009**; random **0.270 ± 0.051**. Every-seed wins hold; gap vs recon **0.168** misses the ≥0.20 bar. Breakpoints: bad views barely drop CKA (−0.025); emb=4 soft-breaks; 50 steps already ~0.84.

## Caveats

Toy ≠ paper. Not a reproduction of Flickr/COCO/Recap-DataComp or 70B LM tables. Reconstruction already aligns strongly on this linear world.

## Links

- Paper: https://arxiv.org/abs/2610.39302
- X: https://x.com/HuggingPapers/status/2116804737207021735
- Demo README: `apps/jepa-platonic-toy/README.md`
- Pages card: https://ryukijano.github.io/tech-demos/
