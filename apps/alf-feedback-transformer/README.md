# alf-feedback-transformer

CPU toy of **Attention Latent Feedback** (Xiong, Chen, Chen, Su, Yang, Luo, and Liu, [arXiv:2611.31154](https://arxiv.org/abs/2611.31154)). During decoding, deeper states of the previous token already exist, but a standard transformer only lets a sublayer read that same depth. ALF lets every sublayer attend to the previous position from its own depth upward, and also adds a projection of the previous top state to the token embedding. The Full-bandwidth Transformer (FBT) feeds back only the final state.

Found via a bookmarked X post by [@rosinality](https://x.com/rosinality/status/2125165189570167284). The post was **not fetched**. The bookmark text (“Feedback transformer with flexible connection (not only the last layer).”) was checked against the arXiv abstract and HTML. Official code is MIT at [xjxyys/ALF](https://github.com/xjxyys/ALF) and was **not vendored**. This directory reimplements equations (3.1)–(3.8) and the reported FBT gate product. See `NOTICE`.

## Hypothesis (pre-registered)

On held-out **answer tokens** of a 3-register trace in \(\mathbb{Z}/5\mathbb{Z}\) (12 rounds, checksum digit after each op), seeds 0/1/2, parameter counts matched within 2%:

1. ALF answer-token NLL is strictly below FBT and strictly below the vanilla transformer on every seed.
2. Seed-mean gap versus vanilla is at least **0.10 nats**.
3. Seed-mean gap versus FBT is at least **0.04 nats**.

PASS is all three. PARTIAL is a seed-mean win for ALF with a seed or a gap missed. FAIL is anything else. Breakpoints are not in the status.

Vanilla is scored with one pass. FBT and ALF are scored with three Jacobi passes, last pass only, matching the paper’s evaluation protocol at toy width. The primary metric is `answer_nats` (natural-log cross-entropy on the checksum tokens). Chance is \(\ln 5 \approx 1.609\).

Thresholds live in `experiments/exp1_cpu_register/config.yaml` and in the list above. They were committed before seeds 0/1/2.

## What the paper reports (quoted, not re-run)

Standard-residual held-out loss in nats, Table 2, three evaluation passes for FBT and ALF:

| Model | MoE-1.6B | MoE-2.6B | MoE-4.0B |
|---|---:|---:|---:|
| Standard residual | 2.0577 | 1.9400 | 1.8456 |
| + FBT | 2.0447 | 1.9261 | 1.8322 |
| + ALF | 2.0333 | 1.9170 | 1.8238 |

The abstract’s 0.022–0.024 nat reduction is that table (0.0244, 0.0230, 0.0218). ALF stays at least 0.008 nats under FBT at every scale. MoE-4.0B domain-balanced accuracy on seventeen tasks moves from 45.1% to 47.0%. On MoE-1.6B, ALF matches a 1.38× larger baseline token budget. A fixed nearest-four window keeps 89%–94% of ALF’s loss reduction. Figure 2: feeding back one intermediate block instead of the final block lowers loss by \(4.4 \times 10^{-3}\) and \(4.1 \times 10^{-3}\) nats on MoE-1.6B and MoE-2.6B. Decoding overhead is about 1.1% FLOPs. None of those runs are repeated here.

## Run (CPU)

```bash
cd apps/alf-feedback-transformer
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
./run_all.sh
```

`./run_all.sh` runs the unit checks, a 20-step smoke on seed 99 (finite loss only), seeds 0/1/2, the breakpoints, the figure, and the MP4. No GPU.

The official MoE ladder is sketched and **not run** in `notebooks/alf_official_moe_colab.ipynb`. That job needs NVIDIA GPUs. A budget for it needs Gyanateet’s approval.

After the run, open the viewer from this directory:

```bash
python3 -m http.server 43133
```

Then open `http://127.0.0.1:43133/viewer/`.

## Status

Pending seeds 0/1/2. The bars above are fixed. This section is replaced with the measured table after the run. The bars are not edited in that replacement.

## Breakpoints (not in the status)

Seed 0, after the measured run: sequence length, Jacobi pass count, an inference-only nearest-4 and last-sublayer read, retrained depth 1 / 2 / 8, and retrained ALF rank 1 / 16. Written to `experiments/exp1_cpu_register/breakpoints.md`.

## Licences

App harness: **MIT** (`LICENSE`). Official ALF code is MIT and is not included. Marin is not included. No paper figures are copied.
