# flexglue-doppelganger

CPU-only clear experiment for **FlexGlue** ([cvg/FlexGlue](https://github.com/cvg/FlexGlue)): sparse-to-dense matching with a pair covisibility head that is meant to reject doppelgangers (look-alike, non-overlapping pairs). Paper: Lindenberger, Sarlin, Miksik, and Pollefeys, [arXiv:2610.32976](https://arxiv.org/abs/2610.32976). Found via a bookmarked X post by [@PhilippCSE](https://x.com/PhilippCSE/status/2117655733911044147). Claims below were checked against the paper and the repository, not against the post.

> Upstream inference code is **MIT** (`LICENSE-UPSTREAM`). The ONNX descriptor redistributes a DINOv3 ConvNeXt-B backbone under the **DINOv3 License** (`DINOv3_LICENSE.md`). That backbone is gated on Hugging Face; this demo does **not** download `facebook/dinov3-convnext-base-pretrain-lvd1689m` and does not use a token. It uses the public FlexGlue v0.1 ONNX files instead. Weights are not committed.

The measured table, plot, and PASS/PARTIAL/FAIL line are written by `./run_all.sh` into `experiments/exp1_cpu_covisibility/`. This README is updated with those numbers after the official run.

## What the paper actually reports

Table 3 (detection of doppelgangers), FlexGlue AUROC / AP:

| Training labels | Doppelgangers AUROC | ViSym-F AUROC | LaMAR AUROC | GPU time |
|---|---|---|---|---|
| matching only | 0.970 | 0.953 | 0.885 | 28.7 ms |
| DG + matching | 0.984 | 0.959 | 0.888 | 28.7 ms |

Inference in the paper uses pair threshold \(t = 0.9\). The README example uses `min_covisibility=0.8`. The covisibility score is the mean of per-token votes. Early exit stops on confidently non-covisible pairs; this demo runs the full fp32 matcher so the score is the final one, and it leaves `min_covisibility` at 0 so match counts are not gated by the head.

The 28.7 ms figure is the paper's GPU matching time, not a CPU number.

## Hypothesis

FlexGlue's covisibility head separates true covisible pairs from doppelganger pairs on CPU. On a tiny balanced Doppelgangers sample, AUROC of `covisibility_score` is at least **0.90**, and that AUROC beats a plain FlexGlue match-count score by at least **0.05**. SIFT Lowe-ratio match count on the same keypoints is a second, report-only baseline.

The 0.90 bar is below both Table 3 Doppelgangers numbers (0.984 and 0.970) and above LaMAR (0.888 / 0.885). The gap is the tiny-set discount: 6 scenes out of 8 (12 pairs), SIFT keypoints instead of the paper's detector, and CPU ONNX numerics. One rank swap in a 6×6 set moves AUROC by about 0.028. The partial floor is **0.80**, just under LaMAR.

PASS needs both the 0.90 bar and the 0.05 margin. PARTIAL needs AUROC ≥ 0.80 and exactly one of those two. FAIL otherwise. Thresholds live in `experiments/exp1_cpu_covisibility/config.yaml` and were fixed before any matcher call.

## Data

Doppelgangers test pairs (Cai et al., ICCV 2023). Code repo is MIT. Images are Wikimedia Commons. The harness keeps only CC0, public domain, and CC BY (no NC, no ND, no SA), in alphabetical scene order, first positive and first negative that download, 8 scenes, then each seed draws 6 of those 8. Provenance is written to `provenance.md`. Images and ONNX files go to `work/` and are not committed.

The upstream `berlin1` / `berlin2` pair is a qualitative anchor only. It is not in the AUROC.

Seeds 0, 1, and 2 control which scenes are drawn and which SIFT keypoints are kept (768 drawn from the top 2048). The descriptor is the exported 512×512 ONNX graph.

## Run (CPU)

```bash
cd apps/flexglue-doppelganger
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
# python3 src/run_experiment.py --smoke   # one pair, disclosed, does not move the AUROC bar
```

ONNX downloads are about 550 MB. The image sample is a few dozen Commons files. No GPU.

Colab (CPU runtime, not a paid GPU): [`notebooks/flexglue_doppelganger_colab.ipynb`](notebooks/flexglue_doppelganger_colab.ipynb).

## Caveats

- This is a 12-pair sample, not the full Doppelgangers / ViSym / LaMAR tables.
- The exported descriptor always squashes to 512×512, so a larger input does not grow the ConvNeXt grid.
- SIFT is the detector in the upstream ONNX example. The paper's tables use other keypoints.
- DINOv3 weights inside the ONNX file are under the DINOv3 License, not MIT.
