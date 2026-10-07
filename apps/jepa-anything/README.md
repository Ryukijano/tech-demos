# apps/jepa-anything

Kit for [JEPA-Anything](https://github.com/Gen-Verse/JEPA-Anything) (paper [arXiv:2609.20800](https://arxiv.org/abs/2609.20800), HF collection [Gen-Verse/jepa-anything](https://huggingface.co/collections/Gen-Verse/jepa-anything)).

Domain-agnostic predictive world models via Orthogonal Predictive Factorization (OPF). This app ships a **Colab-ready notebook** and local scripts — not a full training stack.

## What actually runs without weights

| Path | Needs GPU? | Needs HF download? | What it does |
| --- | --- | --- | --- |
| `python3 scripts/dry_run.py` | No | No | Validates planned clone/install/infer commands; optional stdlib checks |
| Colab notebook (dry-run cells) | No | No | Same honesty path in Colab |
| `recipes/synthetic-linear-dynamics` (upstream) | No | Clone only | Structural recipe — **no trained performance claims** |
| `scripts/run_light_inference.py --download` | Optional | Yes (~1.9 MB) | Downloads **Synthetic Causal Dynamics** checkpoint from HF dataset and runs smoke infer if `infer_dynamics.py` is available |

**Honest limitation:** Full paper metrics need domain datasets + matching checkpoints (often hundreds of MB–GB). CI / cloud agents should treat dry-run as the guaranteed path.

## Lightest domain checkpoint (chosen)

**Synthetic Causal Dynamics** — ~1.9 MB checkpoint in the HF dataset:

```
hf://datasets/Gen-Verse/jepa-anything/05_Ten_Task_Dynamics/Synthetic_Causal_Dynamics/checkpoints/Synthetic_Causal_Dynamics_final_checkpoint.pt
```

Upstream smoke (from that experiment dir, after cloning their dataset layout or using our download helper):

```bash
python ../../test/infer_dynamics.py \
  --checkpoint checkpoints/Synthetic_Causal_Dynamics_final_checkpoint.pt \
  --output /tmp/jepa_inference_smoke/predictions.npz
```

## Prerequisites

- Python 3.10+
- `git`, `pip`
- Optional: PyTorch + CUDA for real inference
- Optional: `huggingface_hub` for checkpoint download

## How to run

```bash
cd apps/jepa-anything

# Always-safe dry run
python3 scripts/dry_run.py

# Clone upstream + install core (network)
python3 scripts/dry_run.py --clone

# Download lightest checkpoint + attempt smoke infer
python3 scripts/run_light_inference.py --download
```

Colab: upload / open `notebooks/jepa_anything_demo.ipynb` and run top-to-bottom. Set `DRY_RUN = True` to skip weight downloads.

## GPU / RAM needs

| Workload | Rough need |
| --- | --- |
| Upstream structural recipe | CPU, &lt;2 GB RAM |
| Synthetic Causal Dynamics smoke | CPU OK; GPU optional; ~2 MB checkpoint + PyTorch |
| Vision / Weather / Molecular domain ckpts | Often GPU + multi-GB VRAM/disk — see each HF model page |

Checkpoints live under `Gen-Verse/jepa-anything` (dataset) organized by domain folders, paired with `test/infer_dynamics.py` / `test/infer_vision.py`.

## Known limits

- Upstream GitHub core ≠ full research weights; weights are released domain-by-domain on HF.
- This kit does not bundle third-party weights in git.
- Dry-run does not claim paper benchmark reproduction.
