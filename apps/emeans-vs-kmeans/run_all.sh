#!/usr/bin/env bash
# CPU-only end-to-end. Install first:
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
python3 -m unittest test_cluster -v
python3 src/run_experiment.py "$@"
if [[ "${1:-}" == "--smoke" ]]; then
  echo "smoke only; tables and media come from a full run"
  exit 0
fi
python3 src/analyze.py
python3 src/make_media.py
if [[ -d ../../docs/assets ]]; then
  cp -f artifacts/emeans-vs-kmeans.png ../../docs/assets/emeans-vs-kmeans.png
  cp -f artifacts/emeans-vs-kmeans.mp4 ../../docs/assets/emeans-vs-kmeans.mp4
fi
echo "done"
