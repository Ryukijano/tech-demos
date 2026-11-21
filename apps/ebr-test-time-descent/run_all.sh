#!/usr/bin/env bash
# CPU-only. Clones the pinned official EBR repo, downloads the Sudoku-Extreme
# checkpoint and test CSV, then runs the pre-registered seeds and breakpoints.
# A few minutes on 4 vCPUs. No GPU.
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=5036bda850a3db12f0db381935d1a4cbbfda9c30
if [[ ! -d work/ebr/models ]]; then
  mkdir -p work
  git clone --filter=blob:none --no-checkout https://github.com/wwillchen/ebr.git work/ebr
  git -C work/ebr fetch --depth 1 origin "$COMMIT"
  git -C work/ebr checkout --detach FETCH_HEAD
fi
HEAD=$(git -C work/ebr rev-parse HEAD)
if [[ "$HEAD" != "$COMMIT" ]]; then
  echo "work/ebr is $HEAD, expected $COMMIT" >&2
  exit 1
fi
export HF_HOME="${HF_HOME:-$PWD/work/hf}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export EBR_FUSED_ATTN_IMPL=reference
python src/prepare.py
python src/verify.py
python src/run_experiment.py --smoke
python src/run_experiment.py
python src/run_experiment.py --breakpoints
python src/analyze.py
python src/make_media.py
echo "done"
