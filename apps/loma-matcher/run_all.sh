#!/usr/bin/env bash
# CPU-only end-to-end: install tip — use run_all after deps are installed.
# Downloads LoMa-B / LoMa-R weights + upstream sample assets, runs exp1
# (seeds 0/1/2 + rotation breakpoints), writes tables/plot, then artifacts
# (PNG grid + MP4; needs ffmpeg).
set -euo pipefail
cd "$(dirname "$0")"
python3 src/run_experiment.py "$@"
python3 src/analyze.py
python3 src/make_media.py
# Copy into docs/assets when present (Pages card)
if [[ -d ../../docs/assets ]]; then
  cp -f artifacts/loma-matcher-grid.png ../../docs/assets/loma-matcher.png
  cp -f artifacts/loma-matcher-rotation.mp4 ../../docs/assets/loma-matcher.mp4
fi
echo "done"
