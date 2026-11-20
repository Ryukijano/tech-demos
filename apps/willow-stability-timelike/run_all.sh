#!/usr/bin/env bash
# CPU-only: verify the stability circuit, run the pre-registered experiment
# (seeds 0/1/2), breakpoints, tables, and the PNG/MP4. A few minutes on 4 vCPUs.
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"
python src/verify.py
python src/run_experiment.py --workers 4
python src/run_experiment.py --breakpoints --workers 4
python src/analyze.py
python src/make_media.py
echo "done"
