# ebr-test-time-descent

CPU subset of the official **Energy-Based Reasoners** Sudoku-Extreme checkpoint (Chen, Gladstone, Li, Ji, Du, [arXiv:2611.08977](https://arxiv.org/abs/2611.08977)). The model learns a scalar energy over a latent state and reasons by gradient descent on it. Found via a bookmarked X post by [@askalphaxiv](https://x.com/askalphaxiv/status/2123301495617565124); the post was **not fetched**. The bookmark text was checked against the paper, the [project page](https://energy-based-reasoners.github.io/), and the [GitHub README](https://github.com/wwillchen/ebr).

Upstream code is MIT (see `NOTICE` and `LICENSES/EBR-UPSTREAM-MIT.txt`). It is cloned at a pinned commit into `work/ebr` and is not vendored. The released weights are one EMA checkpoint (1.8M parameters, training step 60 000). ARC checkpoints are not in the release. The upstream README asks for CUDA 12.x and an H200 for training; Sudoku and Maze training use fused Triton kernels. Their evaluator falls back to CPU eager float32, and that is the path used here. The full test (422 786 boards × 8 samples × 64 segments) was not run. [`notebooks/ebr_official_eval_colab.ipynb`](notebooks/ebr_official_eval_colab.ipynb) only records that a GPU budget decision would be required for that protocol. It was not executed.

## Hypothesis (pre-registered)

Section 4.1 says that, despite training with only four iterations, exact accuracy keeps improving as descent continues. Table A.2 sets Sudoku training segments to 4 and evaluation segments to 64, with 21 microsteps inside each segment. Section 4.2 and figure 6 say solved trajectories end at lower energy and with a smaller tangent gradient than unsolved ones.

On the **96 highest-rated** Sudoku-Extreme test puzzles, one trajectory each, inference seeds 0/1/2:

1. Seed-mean exact accuracy at segment 16 exceeds segment 4 by at least 0.02.
2. That curve does not fall by more than one board from 4→8 or 8→16.
3. Every seed has AUROC(−energy) ≥ 0.70 at the deepest segment in {16, 8, 4, 2, 1} that still has at least 5 correct and 5 incorrect boards.
4. At that depth, the mean tangent gradient on incorrect boards is at least twice the mean on correct boards.

PASS is all four. PARTIAL is at least two. FAIL is fewer than two. Breakpoints are not in the status. The ± in the paper is over three training seeds. Only one checkpoint is released, so the ± here is trajectory noise.

Thresholds live in `experiments/exp1_cpu_sudoku_extreme/config.yaml` and were committed before the measured seeds.

## Run (CPU)

```bash
cd apps/ebr-test-time-descent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```

`./run_all.sh` clones the pinned upstream commit, checks the checkpoint sha256, builds the rating subsets from the public test CSV, runs a 2-board smoke (seed 99, not a result), then seeds 0/1/2 and the breakpoints. No GPU. Batches of 16 boards share the CPU threads.

After the run, open the viewer:

```bash
python3 -m http.server 43131 -d viewer
```

## What this does not claim

Paper numbers below are **quoted**, not reproduced: 99.4% single-sample and 99.9% with 5 samples on the full Sudoku-Extreme test, 87.4% / 88.3% on Maze-Hard, near-100% N-Queens and zero graph-coloring conflicts, and 345× fewer inference FLOPs than EqR at matched accuracy. Table 3’s learned-transition ablation (98.96% → 60.05%) was not re-trained. This subset is the hard tail of the test set, so accuracy will sit below the full-test headline.
