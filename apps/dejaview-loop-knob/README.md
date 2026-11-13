# dejaview-loop-knob

CPU-only, pre-registered experiment on the central knob of **Déjà View (DVLT)**,
*Looping Transformers for Multi-View 3D Reconstruction* (NVIDIA et al.,
[arXiv:2605.30215](https://arxiv.org/abs/2605.30215), NeurIPS 2026 oral):
the loop count **K** — one shared transformer block applied K times — as an
**inference-time compute/quality dial** on a single 117M-parameter checkpoint.

Found via a bookmarked X post by
[@NVIDIAAI](https://x.com/NVIDIAAI/status/2120617173286883692) (related quote
bookmark by [@kwangmoo_yi](https://x.com/kwangmoo_yi/status/2121148907464101903)).
**Neither post was fetched** — no X API, no X MCP, no x.com scraping. Every
claim here was verified from the [arXiv abstract](https://arxiv.org/abs/2605.30215),
the [official repository](https://github.com/nv-tlabs/dvlt) (pinned commit
`f59bb18`), and the [Hugging Face model card](https://huggingface.co/nvidia/dvlt)
on 2026-11-13.

**The code under test is the official `dvlt` package** (not a reimplementation),
installed from the pinned commit at setup time. Weights download from
`nvidia/dvlt` at run time and are **not committed**. The K dial is upstream's
own `inference_steps` attribute, read by the looped block at every `predict`.

## Hypothesis (pre-registered)

> On small synthetic multi-view scenes with known cameras, raising K of the
> released checkpoint from 8 to 16 monotonically improves camera-pose accuracy
> (pose AUC@30 over all view pairs, upstream metric) and does not worsen
> scale-aligned depth error (AbsRel, median alignment), while CPU latency per
> scene grows roughly linearly in K.

Metrics, protocol, and numeric bars were frozen in
[`experiments/exp1_cpu_loop_knob/config.yaml`](experiments/exp1_cpu_loop_knob/config.yaml)
**before** the measured run (see the pre-registration commit; `raw.json` records
the config SHA-256). The bars:

| Check | Bar |
|---|---|
| Floor: mean pose AUC@30 at K=16 | ≥ 0.40 |
| H1a: mean AUC@30 gain, K=8 → K=16 | ≥ +0.01 |
| H1b: mean AUC@30 monotone over K ∈ {8,10,12,14,16} | each step ≥ −0.005 |
| H1c: mean depth AbsRel at K=16 vs K=8 | ≤ |
| H2: latency vs K linear fit over full grid {1…32} | R² ≥ 0.98, slope > 0 |

PASS = all five. FAIL = floor broken, or no quality improvement at all 8→16
(neither H1a nor H1c). PARTIAL = anything else. Breakpoints are report-only.

## Status (honest): **pending — run not yet executed at this commit**

## Data

Four procedural Kubric-style tabletop scenes (textured ground + 5–7 textured
primitives), rendered by a pure-numpy perspective-correct z-buffer rasterizer
(`src/make_scenes.py`) at 504² → 252². Ground-truth cameras are exact by
construction (OpenCV c2w), ground-truth depth comes from the z-buffer, sky is
masked out. No third-party assets, so the data is licence-clean and CPU-cheap.
20 candidate views per scene; **seed s ∈ {0,1,2} draws which 6 views each
scene contributes**. Synthetic data is an easy, in-distribution-adjacent domain
for DVLT (its training mix includes synthetic datasets such as Kubric) — this
experiment probes the K dial, not in-the-wild robustness.

## Run it

```bash
cd apps/dejaview-loop-knob
bash setup.sh                                  # venv + pinned dvlt clone + CPU torch
.venv/bin/python src/make_scenes.py --out data/scenes
.venv/bin/python src/run_experiment.py         # measured run -> raw.json
.venv/bin/python src/analyze.py                # tables, plots, status.json
python3 -m http.server 43130 -d viewer         # open http://127.0.0.1:43130
```

First run downloads ~450 MB of DVLT weights from Hugging Face plus the DINOv2
backbone from Meta's CDN. No credentials required. `HF_TOKEN` is optional (rate
limits only); never commit it.

## Licences

- DVLT **code**: Apache-2.0, portions under VGGT License v1 —
  [`LICENSES/DVLT-APACHE-2.0.txt`](LICENSES/DVLT-APACHE-2.0.txt),
  [`LICENSES/VGGT-LICENSE.txt`](LICENSES/VGGT-LICENSE.txt). Not vendored.
- DVLT **weights** (`nvidia/dvlt`): **NVIDIA License — non-commercial research
  and evaluation use only** —
  [`LICENSES/NVIDIA-MODEL-LICENSE.txt`](LICENSES/NVIDIA-MODEL-LICENSE.txt).
  Downloaded at run time, never committed. This app uses them strictly for
  research evaluation. The model card states the released checkpoint is valid
  for K ∈ [8, 16]; this experiment deliberately also probes outside that range
  and labels those rows as breakpoints.
- This app's own code (`src/`, `viewer/`): Apache-2.0 ([`LICENSE`](LICENSE)).

See [`NOTICE`](NOTICE) for the full attribution map.
