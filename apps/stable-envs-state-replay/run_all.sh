#!/usr/bin/env bash
# CPU-only reproduction of the pre-registered state-replay experiment.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMBA_NUM_THREADS=4
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -U pip
  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install -r requirements.txt
fi
.venv/bin/python src/run_experiment.py
.venv/bin/python src/analyze.py
.venv/bin/python src/make_media.py
