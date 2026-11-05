#!/usr/bin/env bash
# CPU-only end-to-end: pre-registered exp1 (Value Forcing on a tiny MRP),
# breakpoints, tables/plot, media (PNG + MP4; needs ffmpeg). Install deps first:
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install --index-url https://download.pytorch.org/whl/cpu torch
#   pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
python3 src/run_experiment.py "$@"
python3 src/analyze.py
python3 src/make_media.py
if [[ -d ../../docs/assets ]]; then
  cp -f artifacts/value-forcing-toy.png ../../docs/assets/value-forcing-toy.png
  cp -f artifacts/value-forcing-toy.mp4 ../../docs/assets/value-forcing-toy.mp4
fi
echo "done"
