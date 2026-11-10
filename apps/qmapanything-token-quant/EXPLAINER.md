# What this toy is checking

He et al. (arXiv:2611.07955) argue that feed-forward geometry transformers break the usual post-training quantization story. Patch tokens and a handful of special tokens (MapAnything's scale token, a camera token elsewhere) share one activation step. The special token can be small, so it occupies few of the levels the outliers set, and it can multiply the whole metric output. A Hadamard rotation is the same matrix on every token. It can shrink the shared step, which helps the common case, and it leaves the ratio of token SNRs where it was. QMapAnything keeps that rotation and adds a bias that only the special row sees, and only in the blocks that actually write that row.

The official code was not released. MapAnything itself is too heavy for this CPU box and is not what a quantization claim has to be tested on. The toy trains a width-64 transformer to read a metric scale off one token and local 2D points off the others, then fake-quantizes the four linear maps in each block at W4A4, W8A8, and W4A8.

Three recipes:

- **naive** — round weights per output channel and activations with one absmax
- **rotation** — randomized Hadamard, folded into the weights, same shared step
- **qmap** — that rotation, plus the token-average bias, plus a special-row bias fit on the calibration set

`uniform` (rotation + token-average bias only) is logged so a gain that is really just ordinary bias correction is not called row-selective.

The pass rule was written from the paper's tables before the three-seed run. It asks rotation to help patch tokens, to leave the special token at least twice as wrong as a patch token, and to leave the metric scale degraded, and it asks the special-row bias to cut special-token MSE and metric error further, at no more than 15% extra fake-quant time. Missing that is a failed reproduction of the claim on this toy, not a quiet edit of the bar.
