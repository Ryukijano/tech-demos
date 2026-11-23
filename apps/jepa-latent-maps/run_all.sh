#!/usr/bin/env bash
# CPU-only end-to-end: pre-registered exp1 (VICReg vs SVReg vs Slow SVReg),
# breakpoints, tables/plot, media (PNG + MP4; needs ffmpeg). Install deps first:
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install --index-url https://download.pytorch.org/whl/cpu torch
#   pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python3 src/run_experiment.py "$@"
python3 src/analyze.py
python3 src/make_media.py
if [[ -d ../../docs/assets ]]; then
  cp -f artifacts/jepa-latent-maps.png ../../docs/assets/jepa-latent-maps.png
  cp -f artifacts/jepa-latent-maps.mp4 ../../docs/assets/jepa-latent-maps.mp4
fi
echo "done"
