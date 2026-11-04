#!/usr/bin/env bash
# CPU-only FlexGlue covisibility experiment.
# Downloads ONNX weights + a small Doppelgangers image sample into work/ (not committed).
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
python3 src/run_experiment.py "$@"
python3 src/analyze.py
python3 src/make_media.py
echo "done"
