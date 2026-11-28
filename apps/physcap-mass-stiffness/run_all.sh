#!/usr/bin/env bash
# CPU-only: pre-registered PhysCaP-style mass and stiffness probes,
# breakpoints, tables, plot, and an offscreen still + MP4.
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
#   ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
# mkdir is the lock. It does not follow a symlink planted in /tmp, and it
# works where flock is missing. A killed run can leave this directory behind.
LOCK_DIR="$(pwd)/.run-lock"
if [[ -L "$LOCK_DIR" ]]; then
  echo "refusing lock through a symlink: $LOCK_DIR" >&2
  exit 1
fi
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "another run holds $LOCK_DIR ; remove it only if that run has died" >&2
  exit 1
fi
cleanup() { rmdir "$LOCK_DIR" 2>/dev/null || true; }
trap cleanup EXIT
# src goes on sys.path inside Python so Windows does not need ':' in PYTHONPATH.
PY='import sys; sys.path.insert(0, "src"); '
python3 -c "${PY}from run_experiment import main; main(sys.argv[1:])" "$@"
if [[ "${1:-}" == "--smoke" ]]; then
  echo "smoke only"
  exit 0
fi
python3 -c "${PY}from analyze import main; main()"
python3 -c "${PY}from make_media import main; main()"
if [[ -d ../../docs/assets ]]; then
  cp -f artifacts/physcap-mass-stiffness.png ../../docs/assets/physcap-mass-stiffness.png
  # Copy the MP4 only when this run encoded it. A skipped encode deletes the
  # clip, including one that was already in the checkout, so the pages folder
  # does not keep the previous video next to the new still.
  if grep -q 'mp4 skipped' artifacts/media_backend.txt; then
    rm -f ../../docs/assets/physcap-mass-stiffness.mp4
  elif [[ -f artifacts/physcap-mass-stiffness.mp4 ]]; then
    cp -f artifacts/physcap-mass-stiffness.mp4 ../../docs/assets/physcap-mass-stiffness.mp4
  fi
fi
echo "done"
