#!/usr/bin/env bash
# CPU-only end-to-end: official DetectorScope clear-experiment (seeds 0/1/2),
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
  cp -f artifacts/sae-text-detector.png ../../docs/assets/sae-text-detector.png
  cp -f artifacts/sae-text-detector.mp4 ../../docs/assets/sae-text-detector.mp4
fi
echo "done"
