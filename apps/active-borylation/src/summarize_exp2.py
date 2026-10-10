# SPDX-License-Identifier: GPL-3.0-or-later
import sys, glob, json, pandas as pd
E, UI = sys.argv[1], sys.argv[2]
rows, ui = [], {}
for f in sorted(glob.glob(f'{E}/raw_*_seed*.json')):
    d = json.load(open(f)); kind = 'labelled' if 'raw_labelled' in f else 'unlabelled'
    for m in d['molecules']:
        rows.append(dict(seed=d['seed'], set=kind, smiles=m['smiles'], top1_idx=m['top1_idx'], top1_p=round(m['top1_p'], 4),
                         labelled_sites=';'.join(map(str, m['labelled_sites'])), top1_correct=m['top1_correct'],
                         run_wall_s=round(d['wall_s'], 1), peak_rss_mb=int(d['peak_rss_mb'])))
        if d['seed'] == 0: ui[m['smiles']] = dict(m, set=kind)
df = pd.DataFrame(rows); df.to_csv(f'{E}/results.csv', index=False)
lab = df[df.set == 'labelled']; acc = lab.groupby('seed').top1_correct.mean()
per_mol = df.groupby(['seed', 'set']).run_wall_s.first() / 3
passed = bool((acc == 1.0).all() and (per_mol < 60).all())
with open(f'{E}/results.md', 'w') as f:
    f.write('# exp2_regio_inference\n\n' + df.to_markdown(index=False) + '\n\n')
    f.write(f"top1_site_accuracy per seed: {acc.to_dict()}\n\nmax wall_s per molecule (incl. model load): {per_mol.max():.1f}\n\n**Result: {'PASS' if passed else 'FAIL'}**\n\n")
    f.write('Caveat: all 3 labelled example molecules also appear in the upstream regioselectivity training set, so this is a load-and-run sanity check, not a generalisation test.\n')
json.dump(list(ui.values()), open(f'{UI}/predictions.json', 'w'), indent=1)
open(f'{UI}/predictions.js', 'w').write('window.PREDICTIONS = ' + json.dumps(list(ui.values())) + ';\n')
print(acc.to_dict(), per_mol.max(), 'PASS' if passed else 'FAIL')
