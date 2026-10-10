#!/usr/bin/env bash
# CPU setup: venv + upstream clone (pinned) + released EquiformerV2 weights. ~1.5 GB disk.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install -q rdkit xgboost scikit-learn pandas matplotlib pyyaml tabulate
.venv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
V=$(.venv/bin/python -c "import torch;print(torch.__version__.split('+')[0].rsplit('.',1)[0]+'.0')")
.venv/bin/pip install -q torch_geometric torchmetrics openpyxl e3nn timm hyperopt mendeleev
.venv/bin/pip install -q pyg-lib -f "https://data.pyg.org/whl/torch-${V}+cpu.html" || echo "pyg-lib wheel missing: radius_graph will fail"
[ -d upstream ] || git clone -q https://github.com/minotm/active-drug-discovery upstream
git -C upstream checkout -q c687cb1c52704a249f929c81bff5ca0ba21020f6
[ -f upstream/src/model_weights/equiformer_regio.pth ] || curl -L -o upstream/src/model_weights/equiformer_regio.pth \
  https://github.com/minotm/active-drug-discovery/releases/download/v1.0.1/equiformer_regio.pth
echo OK
