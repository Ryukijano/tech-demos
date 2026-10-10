#!/usr/bin/env bash
# Re-run exp1 + exp2 on CPU and rebuild tables/plots/UI data. exp3 is Colab-only (exp3_colab.ipynb).
set -euo pipefail
export OMP_WAIT_POLICY=PASSIVE  # avoid OpenMP spin under CPU contention
cd "$(dirname "$0")"; H=$PWD; PY=$H/.venv/bin/python
E1=experiments/exp1_bald_vs_random; E2=experiments/exp2_regio_inference
for s in 0 1 2; do nice $PY src/al_bald.py --config $E1/config.yaml --data_dir upstream/data/training_data --out $E1/raw_seed$s.csv --seeds $s; done
$PY src/analyze_exp1.py $E1
cd upstream; for s in 0 1 2; do for f in labelled unlabelled; do
  PYTHONPATH=$H/shims $PY $H/src/regio_json.py $H/$E2/$f.csv $H/$E2/raw_${f}_seed$s.json --seed $s --smiles_col SMILES --id_col ID; done; done
cd $H; $PY src/summarize_exp2.py $E2 ui
echo "UI: (cd ui && python3 -m http.server 8765) -> http://localhost:8765"
