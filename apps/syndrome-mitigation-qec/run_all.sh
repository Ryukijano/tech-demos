#!/usr/bin/env bash
# CPU-only end-to-end: decoder verification, pre-registered exp1 run (d=3,5
# primary + d=7 report-only), breakpoint sweeps, tables/plot, media (PNG grid
# + MP4; needs ffmpeg). ~2.5 min on 4 vCPUs. Install deps first:
#   uv venv .venv && source .venv/bin/activate
#   uv pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"
python3 src/verify.py                       # exact-ML decoder vs brute force
# python3 src/run_experiment.py --smoke     # disclosed tiny code-path check
python3 src/run_experiment.py               # official run (seeds fixed in config)
python3 src/run_experiment.py --breakpoints
python3 src/analyze.py
python3 src/make_media.py
# Copy into docs/assets when present (Pages card)
if [[ -d ../../docs/assets ]]; then
  cp -f artifacts/syndrome-mitigation-qec.png ../../docs/assets/syndrome-mitigation-qec.png
  cp -f artifacts/syndrome-mitigation-qec.mp4 ../../docs/assets/syndrome-mitigation-qec.mp4
fi
echo "done"
