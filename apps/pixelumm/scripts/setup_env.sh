#!/usr/bin/env bash
# Clone PixelUMM and print upstream environment install steps.
# Full CUDA / FlashAttention build follows ENVIRONMENT.md — not automated here
# (needs matching CUDA toolkit + GPU arch).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/work"
REPO="$WORK/PixelUMM"

mkdir -p "$WORK"
if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/nv-tlabs/PixelUMM.git "$REPO"
else
  echo "Reuse clone at $REPO"
fi

cat <<EOF

Cloned: $REPO

Next (from upstream ENVIRONMENT.md) — run on a CUDA 13 Linux box:

  cd $REPO
  python3.12 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements-torch-cu130.txt
  python -m pip install -r requirements.txt
  python -m pip install -r requirements-flash-build.txt
  # Build FlashAttention for your GPU arch (80/90/100):
  FLASH_ATTN_CUDA_ARCHS=90 MAX_JOBS=4 NVCC_THREADS=1 \\
    FLASH_ATTENTION_FORCE_BUILD=TRUE \\
    python -m pip install --no-build-isolation --no-deps -r requirements-flash-attn.txt

Download weights per CHECKPOINT.md, then:

  export PIXELUMM_CKPT=/absolute/path/to/S8-F22-R05
  export PIXELUMM_QWEN_DIR=/absolute/path/to/qwen3-8b-config-tokenizer
  export PIXELUMM_OUTPUT=/absolute/path/to/out

  bash $ROOT/scripts/run_image_vlm.sh /absolute/path/to/photo.jpg

Guardrails: leave defaults ON. Do not pass --no-guardrails unless you explicitly opt in.
T2V also needs gated access to nvidia/Cosmos-1.0-Guardrail (HF auth after access grant).

EOF
