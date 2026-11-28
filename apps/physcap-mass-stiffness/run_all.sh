#!/usr/bin/env bash
# CPU-only: pre-registered PhysCaP-style mass and stiffness probes,
# breakpoints, tables, plot, and an offscreen still + MP4.
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
#   ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
python3 src/run_experiment.py "$@"
if [[ "${1:-}" == "--smoke" ]]; then
  echo "smoke only"
  exit 0
fi
python3 src/analyze.py
python3 src/make_media.py
if [[ -d ../../docs/assets ]]; then
  cp -f artifacts/physcap-mass-stiffness.png ../../docs/assets/physcap-mass-stiffness.png
  cp -f artifacts/physcap-mass-stiffness.mp4 ../../docs/assets/physcap-mass-stiffness.mp4
fi
echo "done"
