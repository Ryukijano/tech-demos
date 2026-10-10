# SPDX-License-Identifier: GPL-3.0-or-later
import sys, glob, os, pandas as pd
D = sys.argv[1]; LIMIT_S = 1800; MEM_MB = 3072
st = {}
if os.path.exists(f'{D}/status.txt'):
    for l in open(f'{D}/status.txt'): n, _, c = l.split(); st[n] = int(c)
rows = []
for log in sorted(glob.glob(f'{D}/*.log')):
    n = os.path.basename(log)[:-4]; ens, b = n.split('_'); csv = f'{D}/{n}.csv'
    if os.path.exists(csv):
        df = pd.read_csv(csv); last = df.iloc[-1]
        timeout = bool(df.get('timeout', pd.Series([False])).fillna(False).astype(bool).any())
        rss = float(last.peak_rss_mb)
        ft = 'TIMEOUT' if timeout else ('OOM' if rss > MEM_MB else 'OK')
        rows.append(dict(config=n, ensemble_size=int(ens[3:]), batch_molecules=int(b[1:]), iterations=len(df),
                         molecules_reached=int(last.molecules), final_mcc=round(last.test_mcc, 4),
                         wall_s=round(last.wall_s, 1), peak_rss_mb=int(rss), outcome=ft))
    else:
        rows.append(dict(config=n, ensemble_size=int(ens[3:]), batch_molecules=int(b[1:]),
                         outcome='RUNNING' if n not in st else f'CRASH(exit {st[n]})'))
df = pd.DataFrame(rows).sort_values(['batch_molecules', 'ensemble_size']); df.to_csv(f'{D}/../breakpoints.csv', index=False)
open(f'{D}/../breakpoints.md', 'w').write('# exp1 extremes (seed 0, BALD only, 30-min wall limit per config)\n\n' + df.to_markdown(index=False) + '\n')
print(df.to_string())
