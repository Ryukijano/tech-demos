#!/usr/bin/env bash
# CPU-only: verify the teleport circuit, smoke, official seeds 0/1/2,
# breakpoints, tables, PNG and MP4.
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"
python3 src/verify.py
python3 src/run_experiment.py --smoke --workers 4
python3 src/run_experiment.py --workers 4
python3 src/run_experiment.py --breakpoints --workers 4
python3 src/analyze.py
python3 src/make_media.py
echo "done"
