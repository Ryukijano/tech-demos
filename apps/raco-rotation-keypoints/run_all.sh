#!/usr/bin/env bash
# CPU-only end-to-end. Install first (see README).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
python3 -m unittest test_geometry -v
python3 src/run_experiment.py "$@"
if [[ "${1:-}" == "--smoke" ]]; then
  echo "smoke only; tables and media come from a full run"
  exit 0
fi
python3 src/analyze.py
python3 src/make_media.py
if [[ -d ../../docs/assets ]]; then
  cp -f artifacts/raco-rotation-keypoints.png ../../docs/assets/raco-rotation-keypoints.png
  cp -f artifacts/raco-rotation-keypoints.mp4 ../../docs/assets/raco-rotation-keypoints.mp4
fi
echo "done"
