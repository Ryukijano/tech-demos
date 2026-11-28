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
# One run at a time. Workers import this package, so spawn on Windows can
# unpickle them. A direct `python src/run_experiment.py` on Windows falls
# back to in-process execution.
LOCK="${TMPDIR:-/tmp}/physcap-mass-stiffness.lock"
exec 9>"$LOCK"
if command -v flock >/dev/null 2>&1; then
  flock 9
fi
python3 -c 'from run_experiment import main; import sys; main(sys.argv[1:])' "$@"
if [[ "${1:-}" == "--smoke" ]]; then
  echo "smoke only"
  exit 0
fi
python3 -c 'from analyze import main; main()'
python3 -c 'from make_media import main; main()'
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
