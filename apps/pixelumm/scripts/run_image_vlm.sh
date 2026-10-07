#!/usr/bin/env bash
# Minimal IMAGE UNDERSTANDING path (not T2V).
# Does NOT pass --no-guardrails. Guardrail flags are T2V-oriented; we keep defaults.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${PIXELUMM_REPO:-$ROOT/work/PixelUMM}"
IMAGE="${1:-}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT/samples/prompt.txt}"

if [[ -z "${PIXELUMM_CKPT:-}" || -z "${PIXELUMM_QWEN_DIR:-}" ]]; then
  echo "Set PIXELUMM_CKPT and PIXELUMM_QWEN_DIR (see README / upstream CHECKPOINT.md)." >&2
  exit 2
fi
if [[ -z "$IMAGE" ]]; then
  echo "Usage: $0 /absolute/path/to/image.jpg" >&2
  exit 2
fi
if [[ ! -f "$REPO/inference.py" ]]; then
  echo "Missing $REPO/inference.py — run scripts/setup_env.sh first." >&2
  exit 2
fi

PROMPT="$(cat "$PROMPT_FILE")"
OUT_DIR="${PIXELUMM_OUTPUT:-$ROOT/work/out}"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/image-answer.txt"

echo "Running image-vlm (guardrails defaults untouched; no --no-guardrails)."
cd "$REPO"
python inference.py \
  --checkpoint "$PIXELUMM_CKPT" \
  --llm-path "$PIXELUMM_QWEN_DIR" \
  --task image-vlm \
  --image "$IMAGE" \
  --prompt "$PROMPT" \
  --output "$OUT"

echo "Wrote $OUT"
