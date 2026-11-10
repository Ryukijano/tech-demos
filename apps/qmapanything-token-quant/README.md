# qmapanything-token-quant

CPU-only clear experiment for the structural claim in He, Zhang, Zhang, Sui, See, Boukhers, and Yang, [arXiv:2611.07955](https://arxiv.org/abs/2611.07955), *Rotation Is Token-Blind*. Found via a bookmarked X post by [@vincieye](https://x.com/vincieye/status/2119415188013138070). That post was **not** fetched. The abstract, Sections 3–4, and the appendices were read from the arXiv HTML.

**QMapAnything code was not available** on 2026-11-10. The reproducibility statement says code and configs will be released upon acceptance. MapAnything ([facebookresearch/map-anything](https://github.com/facebookresearch/map-anything), Apache-2.0) is not run and not vendored. This directory is a clean-room toy. See `NOTICE`.

## Hypothesis

On a small transformer whose patch tokens and one special token share a per-tensor activation grid, a randomized Hadamard rotation reduces typical-token quantization error but leaves the special token's SNR ratio unchanged, and that special token's error still moves the metric scale. A row-selective affine correction, in the spirit of QMapAnything's \(b_s\), restores special-token fidelity and metric-scale error without a material latency increase over rotation alone.

## Substitution

MapAnything is a DINOv2 feed-forward geometry model. The measurements below are from a 4-block pre-LN transformer, width 64, trained on synthetic scenes (about 136k parameters). Patch linear inputs get two Laplace outlier channels after LayerNorm (S3). The scale token skips LayerNorm, is written only by the global blocks (S4), and is the only input to the metric-scale head (S2). Each global qkv and fc1 input rescales that row onto \(\max\|h_{\text{patch}}\|_2 / 6.02\), the MapAnything \(1/\kappa_s\) in Table 5. A probe without the pin measured qkv \(1/\kappa \approx 95\). Heads stay in FP. Quantization is symmetric absmax fake-quant, per-output-channel weights, per-tensor activations, \(R = DH\) with signs first. No INT kernels, no TensorRT, no GPU.

## Status

The official seeds have not been scored in this commit. Thresholds in `experiments/exp1_cpu_token_quant/config.yaml` are frozen. `./run_all.sh` writes `status.json`.

| Check (fixed before the official run) | Bar |
|---|---|
| Prop. 1 SNR-ratio change, and qkv/fc1 step shrink | ≤ 0.02 and ≥ 1.25 |
| W4A4 patch relative error, rotation / naive | ≤ 0.80 |
| W4A4 special / patch relative error after rotation | ≥ 2 |
| W4A4 special-token MSE, qmap / rotation | ≤ 0.80 |
| W4A4 metric-scale error, qmap / rotation | ≤ 0.80 |
| W4A4 rotation metric error vs FP, and absolute floor | ≥ 1.5× and ≥ 0.05 |
| W4A4 special-token MSE, qmap / uniform \(\beta\) | ≤ 0.90 |
| W8A8 latency, qmap / rotation | ≤ 1.15 |
| W8A8 metric order naive ≥ rotation ≥ qmap | 2% slack |

**PASS** needs all nine. **PARTIAL** if the Prop. 1, common-mode, token-blind, and latency checks hold and at least one of the two restoration checks holds, but something else misses. **FAIL** otherwise. W4A8 and the breakpoint sweeps are report-only.

Paper anchors, not our targets to hit exactly: Table 2 W4A4 scale rel is 0.48 (RTN), 0.20 (QuaRot), 0.12 (QMapAnything), 0.05 (FP). Appendix D's held-out scale-token MSE drop is 34.2%. Table 4's W8A8 latency is \(0.32\times\) FP32 for both naive INT8 and QMapAnything on one H20. This toy does not claim that speedup.

## Run (CPU)

```bash
cd apps/qmapanything-token-quant
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
./run_all.sh
PYTHONPATH=src python -m unittest test_quant -v
```

No downloads. No GPU.

## Licences

- This directory: **MIT** (`LICENSE`).
- MapAnything: Apache-2.0, not copied.
- QMapAnything: no public code as of 2026-11-10.
