#!/usr/bin/env bash
# CPU-only end-to-end: pre-registered exp1 (good-view JEPA cross-modal CKA),
# breakpoints, tables/plot, media (PNG + MP4; needs ffmpeg). Install deps first:
#   uv venv .venv && source .venv/bin/activate
#   uv pip install --index-url https://download.pytorch.org/whl/cpu torch
#   uv pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"
python3 src/run_experiment.py "$@"
python3 src/analyze.py
python3 src/make_media.py
# Copy into docs/assets when present (Pages card)
if [[ -d ../../docs/assets ]]; then
  cp -f artifacts/jepa-platonic-toy.png ../../docs/assets/jepa-platonic-toy.png
  cp -f artifacts/jepa-platonic-toy.mp4 ../../docs/assets/jepa-platonic-toy.mp4
fi
echo "done"
