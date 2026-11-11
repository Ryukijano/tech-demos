# emeans-vs-kmeans

CPU comparison of FAISS 1.16.0 e-means against FAISS Lloyd k-means. The algorithm is Bessa et al., [arXiv:2611.09501](https://arxiv.org/abs/2611.09501). Found via a bookmarked X post by [@swann_bessa](https://x.com/swann_bessa/status/2120200714630594911). That post was **not** fetched. The abstract, Section 3, and Appendix D were read from the arXiv HTML. The FAISS release was checked in the 1.16.0 changelog and by importing `faiss.contrib.e_means.NumpyEmeans` from `faiss-cpu==1.16.0`.

**Status: pending the pre-registered run.** Thresholds live in `experiments/exp1_cpu_emeans/config.yaml` and are not to be edited after scores exist. This file's result table is written once `raw.json` exists.

## Hypothesis

On CPU, released e-means at 2.5 epochs (10× fewer than 25) reaches FAISS Lloyd's training mean squared error within 1%, and the training call takes at most 1/10 of Lloyd's wall-clock.

## What is fixed before the run

- **Primary metric:** `train_mse_ratio_at_gate_epochs`. Per seed, fresh nearest-centroid train MSE of `NumpyEmeans(nepochs=2.5)` divided by the same MSE for `faiss.Kmeans(niter=25)`.
- **Co-primary:** `wall_time_ratio_at_gate_epochs`, `perf_counter` around `train()` only.
- **Bar:** mean ratio ≤ 1.01 and mean wall ratio ≤ 0.10 on every gated dataset. Sample std uses ddof=1 over seeds 0, 1, 2.
- **PASS** needs both bars on mixture, stretched Gaussian, and MNIST. **PARTIAL** if the loss bar holds and the clock bar does not. **FAIL** if any dataset misses the loss bar.
- Lloyd sees the full matrix (`max_points_per_centroid=1e9`, `min_points_per_centroid=1`). Those two overrides are the paper's FAISS baseline. The library defaults (256 and 39) are a report-only control.
- e-means hyperparameters stay on the library's automatic law. Nothing is swept.

## Run (CPU)

```bash
cd apps/emeans-vs-kmeans
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```

MNIST is downloaded into `work/mnist/` (gitignored) from the CVDF or OSSCI mirror. No GPU. `python src/run_experiment.py --smoke` rewrites `smoke.json` only.

## Licences

- This directory: **MIT** (`LICENSE`).
- FAISS (Meta): **MIT**. Not vendored. See `NOTICE`.
