# sae-text-detector — short explainer

## What it is
CPU clear-experiment kit around **DetectorScope** (Dugan et al., arXiv:2610.30006): an AI-text detector whose score is a sum of sparse-autoencoder features with validated natural-language descriptions. Uses the official [`google-deepmind/detectorscope`](https://github.com/google-deepmind/detectorscope) package and Hub weights [`liamdugan/detector-scope`](https://huggingface.co/liamdugan/detector-scope). Bookmark source: [@_akhaliq](https://x.com/_akhaliq/status/2114730846426890689).

## What ran
- Pre-registered hypothesis in `experiments/exp1_cpu_auditable/config.yaml` (committed before the measured run).
- Dataset: `dmitva/human_ai_generated_text` (CC-BY-4.0), 96-example balanced subsamples, seeds 0/1/2.
- Compared logreg heads at τ=0.85 (auditable default) vs τ=0.00 (all SAE features).
- Primary metric: AUROC gap (0.00 − 0.85); pass if gap ≤ 0.05 and AUROC_0.85 ≥ 0.70 on every seed.
- No GPU, no paid APIs. Gemma-2-2b extractor downloaded from the Hub repo (~10 GB) on first run.

## Results
| seed | AUROC τ=0.85 | AUROC τ=0.00 | gap |
|---|---:|---:|---:|
| 0 | 0.789 | 0.983 | 0.194 |
| 1 | 0.835 | 0.976 | 0.141 |
| 2 | 0.847 | 0.989 | 0.142 |

Mean gap **0.159 ± 0.030** → status **FAIL** (AUROC bar met; gap bar missed on every seed).

## Breakpoints
- Truncation to 40 chars: both heads weak; τ=0.95 → 0 active features → chance.
- Top-k≤4 features: AUROC ~0.71; recovers near full head by k=32–64.

## Caveats
- Small CPU slice, not COLING/RAID. Domain shift vs the paper’s training mix likely widens the τ=0.85 vs 0.00 gap.
- context-vault write: not done (private repo not accessible from this agent).

## Links
- App: `apps/sae-text-detector/`
- Paper: https://arxiv.org/abs/2610.30006
- Code: https://github.com/google-deepmind/detectorscope
- Artifacts: `artifacts/sae-text-detector.png`, `.mp4`
