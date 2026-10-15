#!/usr/bin/env bash
# CPU-only end-to-end: trains the 2D denoiser+corrector (seeds 0/1/2), runs the
# pre-registered exp1 comparison + breakpoints, writes tables/plot, then media
# (PNG grid + MP4; needs ffmpeg). Install deps first:
#   uv venv .venv && source .venv/bin/activate
#   uv pip install --index-url https://download.pytorch.org/whl/cpu torch
#   uv pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"
python3 src/run_experiment.py "$@"
python3 src/analyze.py
python3 src/make_media.py
# Copy into docs/assets when present (Pages card)
if [[ -d ../../docs/assets ]]; then
  cp -f artifacts/correction-training-2d-grid.png ../../docs/assets/correction-training-2d.png
  cp -f artifacts/correction-training-2d.mp4 ../../docs/assets/correction-training-2d.mp4
fi
echo "done"
