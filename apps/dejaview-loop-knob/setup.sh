#!/usr/bin/env bash
# Reproducible CPU-only environment for dejaview-loop-knob.
set -euo pipefail
cd "$(dirname "$0")"

DVLT_COMMIT=f59bb18259ab446a7585c4f47731a77cfdefa426

if [ ! -d vendor/dvlt ]; then
  git clone https://github.com/nv-tlabs/dvlt vendor/dvlt
fi
git -C vendor/dvlt checkout "$DVLT_COMMIT"

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e vendor/dvlt --no-deps

echo "OK. Weights download from https://huggingface.co/nvidia/dvlt on first run."
