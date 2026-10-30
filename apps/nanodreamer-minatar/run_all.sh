#!/usr/bin/env bash
# CPU-only end-to-end: pre-registered exp1 (Dreamer toy vs random on MinAtar),
# breakpoints, tables/plot, media (PNG + MP4; needs ffmpeg). Install deps first:
#   python3 -m venv .venv && source .venv/bin/activate
#   python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
#   python -m pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"
python3 src/run_experiment.py "$@"
python3 src/analyze.py
python3 src/make_media.py
# Copy into docs/assets when present (Pages card)
if [[ -d ../../docs/assets ]]; then
  cp -f artifacts/nanodreamer-minatar.png ../../docs/assets/nanodreamer-minatar.png
  cp -f artifacts/nanodreamer-minatar.mp4 ../../docs/assets/nanodreamer-minatar.mp4
fi
echo "done"
