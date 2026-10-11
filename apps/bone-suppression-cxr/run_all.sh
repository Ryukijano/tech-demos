#!/usr/bin/env bash
# CPU-only end-to-end: downloads weights + examples from Hugging Face (no token needed),
# runs exp1 (seeds 0/1/2 + breakpoints), writes tables/plot, then artifacts (PNG grid + MP4; needs ffmpeg).
set -euo pipefail
cd "$(dirname "$0")"
python3 src/run_experiment.py "$@"
python3 src/analyze.py
python3 src/make_media.py
