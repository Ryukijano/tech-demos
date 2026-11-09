#!/usr/bin/env bash
# CPU-only end-to-end: pre-registered exp1 (MuJoCo 4.0.0 speculative contacts),
# breakpoints, tables/plot, EGL still + MP4 (matplotlib fallback if EGL fails).
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
#   ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python3 src/run_experiment.py "$@"
if [[ "${1:-}" == "--smoke" ]]; then
  echo "smoke only"
  exit 0
fi
python3 src/analyze.py
python3 src/make_media.py
if [[ -d ../../docs/assets ]]; then
  cp -f artifacts/mujoco4-speculative-contacts.png ../../docs/assets/mujoco4-speculative-contacts.png
  cp -f artifacts/mujoco4-speculative-contacts.mp4 ../../docs/assets/mujoco4-speculative-contacts.mp4
fi
echo "done"
