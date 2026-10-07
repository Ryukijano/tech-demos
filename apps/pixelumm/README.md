# apps/pixelumm

Kit for [PixelUMM](https://nv-tlabs.github.io/PixelUMM/) — encoder-free unified image/video understand+generate in pixel space (Qwen3-8B backbone).

Upstream:

- Code: [nv-tlabs/PixelUMM](https://github.com/nv-tlabs/PixelUMM) (**Apache-2.0**)
- Weights: [nvidia/PixelUMM](https://huggingface.co/nvidia/PixelUMM) (**NVIDIA One-Way Noncommercial License** — research/eval only)
- Guardrails: [GUARDRAILS.md](https://github.com/nv-tlabs/PixelUMM/blob/main/GUARDRAILS.md) (Cosmos guardrail is **gated**; default ON for T2V)

## Licenses (read this)

| Artifact | License |
| --- | --- |
| Inference / training code | Apache-2.0 (file-specific notices may apply) |
| PixelUMM checkpoint | NVIDIA One-Way Noncommercial — **no commercial use** |
| Qwen3-8B config/tokenizer | Apache-2.0 |
| Cosmos guardrail weights | Separate gated model terms |

This kit does **not** disable safety/guardrails by default. `--no-guardrails` is documented as **opt-in only** (mainly relevant for T2V). Our default path is **image understanding** (`--task image-vlm`), which does not route through the T2V guardrail stack.

## What this ships

- `scripts/smoke_test.py` — fails gracefully without GPU/weights
- `scripts/setup_env.sh` — mirrors upstream ENVIRONMENT.md steps
- `scripts/run_image_vlm.sh` — minimal **image understanding** command
- `samples/prompt.txt` — sample prompt
- Honest README about VRAM / HF auth

## Prerequisites

- Linux x86-64, Python 3.12
- NVIDIA GPU + CUDA 13 toolchain (upstream pins FlashAttention CUDA builds)
- FFmpeg shared libs (for video paths)
- Disk: multi-tens of GB for checkpoint + deps
- HF account: PixelUMM weights are downloadable under the noncommercial license; **Cosmos guardrail** for T2V is gated — request access before T2V

VRAM: plan for a high-end GPU (community notes often cite **24GB+** for comfortable use). Exact footprint depends on resolution/task.

## How to run

### Smoke test (no GPU required)

```bash
cd apps/pixelumm
python3 scripts/smoke_test.py
```

Exits 0 with a clear message when CUDA/weights are missing; exits non-zero only on unexpected script errors.

### Full image understanding (GPU + weights)

```bash
# 1) Clone + env (see upstream ENVIRONMENT.md — long CUDA build)
bash scripts/setup_env.sh

# 2) Download checkpoint per upstream CHECKPOINT.md
#    export PIXELUMM_CKPT=/abs/path/to/S8-F22-R05
#    export PIXELUMM_QWEN_DIR=/abs/path/to/qwen3-8b-config-tokenizer
#    export PIXELUMM_OUTPUT=/abs/path/to/out

# 3) CPU-side checkpoint check (upstream)
CUDA_VISIBLE_DEVICES="" python check_checkpoint.py \
  --checkpoint "$PIXELUMM_CKPT" --llm-path "$PIXELUMM_QWEN_DIR"

# 4) Image VLM
bash scripts/run_image_vlm.sh /absolute/path/to/photo.jpg
```

Equivalent upstream command:

```bash
python inference.py \
  --checkpoint "$PIXELUMM_CKPT" --llm-path "$PIXELUMM_QWEN_DIR" \
  --task image-vlm --image /absolute/path/to/photo.jpg \
  --prompt "Describe the image." \
  --output "$PIXELUMM_OUTPUT/image-answer.txt"
```

### Guardrails note (T2V only)

Default T2V runs enable Cosmos + related checks. To skip (not recommended):

```bash
python inference.py ... --task t2v ... --no-guardrails   # OPT-IN only
```

You remain responsible for license compliance and output review.

## HF auth

```bash
huggingface-cli login   # or: hf auth login
```

- PixelUMM checkpoint: accept noncommercial terms on the model card.
- T2V guardrails: separately request access to `nvidia/Cosmos-1.0-Guardrail` — login alone is not enough.

## Known limits

- Not runnable in this cloud agent environment end-to-end (no suitable GPU / FlashAttn CUDA build assumed).
- Full T2V is heavy; this kit targets **image-vlm** only.
- Do not commit weights or HF tokens.
