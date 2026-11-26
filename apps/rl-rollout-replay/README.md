# rl-rollout-replay

CPU toy of **Reinforcement Learning Can Be Recovered From Its Own Rollouts** (Sheng, Gu, Zeng, Zhou, Viel, Wang, Tajwar, Zanette, [arXiv:2611.33102](https://arxiv.org/abs/2611.33102)). A small GRU learns two-digit addition by on-policy REINFORCE. The same base checkpoint is then trained again on the archived rollouts.

Found via a bookmarked X post by [@rosinality](https://x.com/rosinality/status/2125531243681038427). The post was **not fetched**. The claims below were checked on the [arXiv HTML (v2)](https://arxiv.org/html/2611.33102v2).

The HTML does not link official code. The paper's language-model runs use verl, which is not vendored. This app is a clean-room reimplementation. See `NOTICE`. No paper figures are copied.

## Hypothesis (locked)

On held-out two-digit addition, answer digits written least-significant first, seeds 0/1/2 share one partially trained base checkpoint.

1. Online REINFORCE raises sampled exact-match reward by at least 0.15.
2. One maximum-likelihood update per archived batch, in generation order, with no correct/incorrect split, finishes within 0.08 of the online seed-mean and has trajectory MAE at most 0.10.
3. One REINFORCE update per archived batch (the paper's uncorrected replay) finishes at least 0.15 below online.
4. Cycling maximum likelihood on only the last 4 batches, for the same number of updates, finishes within 0.10 of online.

PASS is all five bars. PARTIAL is bars 1 and the ML endpoint gap, with at least one other bar missed. FAIL is a missed gain or a missed ML endpoint gap. Breakpoints are not in the status.

The bars live in `experiments/exp1_cpu_addition/config.yaml` and are committed before seeds 0/1/2. An online-only probe chose the step budget (SFT 120 steps at lr 0.02, then 280 REINFORCE steps at lr 0.003). It did not set the bars and did not run replay.

## Run (CPU)

```bash
cd apps/rl-rollout-replay
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
./run_all.sh
```

No GPU and no network. `./run_all.sh --smoke` checks the code path only (`smoke.json`, seed 99). It is not a result.

After the run:

```bash
python3 -m http.server 43132 -d viewer
```

## What the paper reports (QUOTED, not re-run)

Checked on the v2 HTML:

- Uncorrected MaxRL replay on MATH-500 improves, then falls to nearly zero by step 100. Truncated importance sampling tracks until about step 600, then drops.
- One maximum-likelihood update per batch tracks online MaxRL. Table 5(a) trajectory errors are 0.002 to 0.011, and endpoint errors are 0.003 to 0.013, depending on the benchmark.
- On Knights and Knaves, the pass@k endpoint gap is 71.9 points for the base model, 0.9 after replay of the last batch, and 0.3 after the full archive.
- For SmolLM2-360M on GSM8K, last-1 / last-5 / last-50 endpoint errors are 5.2 / 3.0 / 0.9 points. A single late batch is not enough in every setting.
- Shuffled order levels off below online. Reverse order rises, then falls back toward the base.
- Maze, all 1000 steps, 8 repeats each: pass@1 0.238 versus 0.252 online. The last step alone reaches 0.157. The last 10 steps reach 0.247.
- One student can track a math specialist and a Knights-and-Knaves specialist by alternating one likelihood update on each archive. Replay of the two final batches keeps most of both skills.

## What this toy actually runs

A 20,077-parameter GRU. The prompt is two 2-digit numbers and an operator. The answer is three digits, least-significant first. The reward used by REINFORCE is exact match of those digits. The reported metric is the same event under 8 fresh samples on 256 held-out sums (temperature 1). The advantage is `r - group mean` (Table 1, weight 1), not MaxRL's `1/p` weight. One AdamW step per batch. Maximum-likelihood replay uses the token NLL of every archived answer. Naive replay uses the stored rewards and the replay policy's log-probabilities.

Paper numbers above are quoted. Numbers produced by `./run_all.sh` are this toy only.
