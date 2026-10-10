# SPDX-License-Identifier: GPL-3.0-or-later
# Thin wrapper around upstream predict_regio.py (minotm/active-drug-discovery, GPLv3).
# Runs upstream main() unchanged, but intercepts process_predictions() to export
# per-atom borylation probabilities (mean/std over 5 conformers) as JSON + RDKit SVG.
# Usage (cwd = upstream checkout):  python regio_json.py IN.csv OUT.json --seed 0 [--smiles_col SMILES --id_col ID]
import sys, json, os, time, random, resource
import numpy as np
out_json = sys.argv[2]; seed = 0
if '--seed' in sys.argv:
    i = sys.argv.index('--seed'); seed = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
sys.argv[2] = 'output/_regio_tmp.xlsx'
import torch
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.set_num_threads(int(os.environ.get('AB_THREADS', '4')))
sys.path.insert(0, os.getcwd())
import predict_regio as pr
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
RECORDS = []
_orig = pr.process_predictions
def hook(smi, preds_mean, preds_std, data, device, args):
    g = pr.process_conformer(data, smi, "a", device, args)
    truth = g.rxn_trg.cpu().numpy().astype(float).ravel()
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    heavy, atoms = 0, []
    for a in mol.GetAtoms():
        if a.GetSymbol() == 'H': continue
        i = a.GetIdx()
        cand = a.GetSymbol() == 'C' and any(n.GetSymbol() == 'H' for n in a.GetNeighbors())
        atoms.append(dict(idx=i, symbol=a.GetSymbol(), candidate=bool(cand),
                          p=float(preds_mean[i]) if cand else 0.0, std=float(preds_std[i]) if cand else 0.0,
                          label=float(truth[i]) if i < len(truth) else 0.0))
    m2 = Chem.MolFromSmiles(smi)
    d = rdMolDraw2D.MolDraw2DSVG(420, 320); o = d.drawOptions(); o.addAtomIndices = False
    hl, cols, rad = [], {}, {}
    for at in atoms:
        if at['candidate'] and at['p'] >= 0.05:
            hl.append(at['idx']); cols[at['idx']] = (1.0, 1.0 - min(at['p'], 1.0), 1.0 - min(at['p'], 1.0)); rad[at['idx']] = 0.35 + 0.25 * at['p']
            m2.GetAtomWithIdx(at['idx']).SetProp('atomNote', f"{at['p']:.2f}")
    d.DrawMolecule(m2, highlightAtoms=hl, highlightAtomColors=cols, highlightAtomRadii=rad, highlightBonds=[]); d.FinishDrawing()
    c = [a for a in atoms if a['candidate']]
    top = max(c, key=lambda a: a['p']) if c else None
    labelled = [a['idx'] for a in atoms if a['label'] > 0]
    RECORDS.append(dict(smiles=smi, atoms=atoms, svg=d.GetDrawingText(), top1_idx=top['idx'] if top else None,
                        top1_p=top['p'] if top else None, labelled_sites=labelled,
                        top1_correct=(top['idx'] in labelled) if (top and labelled) else None))
    return _orig(smi, preds_mean, preds_std, data, device, args)
pr.process_predictions = hook
t0 = time.time(); pr.main()
json.dump(dict(seed=seed, device='cpu', wall_s=time.time() - t0, torch=torch.__version__,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, molecules=RECORDS),
          open(out_json, 'w'), indent=1)
print('wrote', out_json)
