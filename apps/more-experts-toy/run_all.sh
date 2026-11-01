#!/usr/bin/env bash
# CPU-only end-to-end: pre-registered exp1 (MoRE vs MoE vs dense),
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
  cp -f artifacts/more-experts-toy.png ../../docs/assets/more-experts-toy.png
  cp -f artifacts/more-experts-toy.mp4 ../../docs/assets/more-experts-toy.mp4
fi
echo "done"
