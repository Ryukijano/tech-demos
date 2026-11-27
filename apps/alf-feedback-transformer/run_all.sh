#!/usr/bin/env bash
# CPU-only: unit checks, pre-registered register-trace comparison, plot, MP4.
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install --index-url https://download.pytorch.org/whl/cpu torch
#   pip install -r requirements.txt
#   ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
python3 src/test_model.py
python3 src/run_experiment.py
python3 src/analyze.py
python3 src/make_media.py
mkdir -p viewer
cp -f artifacts/alf-feedback-transformer.png viewer/plot.png
cp -f artifacts/alf-feedback-transformer.mp4 viewer/alf-feedback-transformer.mp4
cp -f experiments/exp1_cpu_register/status.json viewer/summary.json
if [[ -d ../../docs/assets ]]; then
  cp -f artifacts/alf-feedback-transformer.png ../../docs/assets/alf-feedback-transformer.png
  cp -f artifacts/alf-feedback-transformer.mp4 ../../docs/assets/alf-feedback-transformer.mp4
fi
echo "done"
