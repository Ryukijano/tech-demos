# SPDX-License-Identifier: GPL-3.0-or-later
import sys, glob, os, json, yaml, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
E = sys.argv[1]; cfg = yaml.safe_load(open(f'{E}/config.yaml'))
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{E}/raw_seed*.csv'))], ignore_index=True)
df.to_csv(f'{E}/results.csv', index=False)
rows = []
for seed, g in df.groupby('seed'):
    target = 0.9 * g.full_pool_mcc.iloc[0]
    r = dict(seed=seed, full_pool_mcc=round(g.full_pool_mcc.iloc[0], 4), target_mcc=round(target, 4))
    for st in ['bald', 'random']:
        h = g[g.strategy == st].sort_values('molecules')
        hit = h[h.test_mcc >= target]
        r[f'{st}_molecules_to_target'] = int(hit.molecules.iloc[0]) if len(hit) else np.nan
        r[f'{st}_final_mcc'] = round(h.test_mcc.iloc[-1], 4)
        r[f'{st}_auc_mcc'] = round(np.trapezoid(h.test_mcc, h.molecules) / (h.molecules.max() - h.molecules.min()), 4)
    r['bald_better'] = bool(r['bald_molecules_to_target'] < r['random_molecules_to_target']) if not np.isnan(r['random_molecules_to_target']) else (not np.isnan(r['bald_molecules_to_target']))
    rows.append(r)
s = pd.DataFrame(rows)
mb, mr = s.bald_molecules_to_target.mean(), s.random_molecules_to_target.mean()
passed = bool(s.bald_better.sum() >= 2 and mb < mr)
s.to_csv(f'{E}/summary.csv', index=False)
with open(f'{E}/results.md', 'w') as f:
    f.write(f"# {cfg['exp_id']}\n\n**Hypothesis:** {cfg['hypothesis'].strip()}\n\n**Metric:** {cfg['metric']}; target = {cfg['target']['definition']}\n\n")
    f.write(f"**Pass rule (fixed before running):** {cfg['pass']}\n\n")
    f.write(s.to_markdown(index=False) + '\n\n')
    f.write(f"Mean molecules to target: BALD {mb:.1f}, random {mr:.1f}. BALD better on {int(s.bald_better.sum())}/3 seeds.\n\n**Result: {'PASS' if passed else 'FAIL'}**\n")
fig, ax = plt.subplots(figsize=(7, 4.5))
for st, c in [('bald', '#e4572e'), ('random', '#4c78a8')]:
    p = df[df.strategy == st].pivot_table(index='molecules', columns='seed', values='test_mcc')
    m, sd = p.mean(1), p.std(1)
    ax.plot(p.index, m, color=c, label=f'{st.upper() if st=="bald" else "Random"} (mean of {p.shape[1]} seeds)')
    ax.fill_between(p.index, m - sd, m + sd, color=c, alpha=0.2)
ax.axhline(s.target_mcc.mean(), ls='--', color='grey', label=f'target MCC (mean) {s.target_mcc.mean():.3f}')
ax.set_xlabel('substrate molecules sampled'); ax.set_ylabel('test MCC (held-out substrates)')
ax.set_title('Retrospective active learning, C–H borylation (XGBoost ensemble)'); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f'{E}/plot.png', dpi=130)
print(s.to_string()); print('PASS' if passed else 'FAIL', mb, mr)
