# FlexGlue — short explainer

Lindenberger, Sarlin, Miksik, and Pollefeys (ETH Zurich / Google / Microsoft Spatial AI Lab), *FlexGlue: Sparse-to-Dense Feature Matching and Visual Disambiguation*, [arXiv:2610.32976](https://arxiv.org/abs/2610.32976). A DINOv3-ConvNeXt encodes each image on a stride-8 grid, a transformer mixes the two grids, and each query keypoint is read out by a windowed soft-argmax plus a small refinement head. A linear head turns every token into a covisibility vote; the pair score is the average of those votes. Pairs below a threshold are rejected, and inference can stop on an early layer when the pair is already confidently non-overlapping.

Found via [@PhilippCSE](https://x.com/PhilippCSE/status/2117655733911044147). Code: [cvg/FlexGlue](https://github.com/cvg/FlexGlue) (MIT). The demo uses the public v0.1 ONNX models on CPU. It does not download the gated Hugging Face DINOv3 checkpoint.

**PARTIAL.** On a 12-pair Doppelgangers sample, covisibility AUROC is **0.824 ± 0.058** (bar 0.90, partial floor 0.80) and beats FlexGlue match-count AUROC by **0.181 ± 0.146** (bar 0.05). SIFT ratio AUROC is 0.593 ± 0.112. Sofia's labelled-positive pair scores 0.019; a Berlin Cathedral doppelganger scores 0.898.

- Paper: `https://arxiv.org/abs/2610.32976`
- Demo: `apps/flexglue-doppelganger/README.md`
