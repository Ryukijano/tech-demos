# SPDX-License-Identifier: GPL-3.0-or-later
# Derived from minotm/active-drug-discovery (GPLv3): acquisition functions in
# src/active_learning_utils/acq_fns.py and ECFP featurisation in
# src/binary_utils/binary_data_utils_xgb.py. Simplified, retrospective re-implementation.
import argparse, json, os, sys, time, resource
import numpy as np, pandas as pd, yaml
from scipy.special import logsumexp
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')

# ---- from acq_fns.py (Kirsch et al. 2019 BALD) ----
def get_entropy(stacked_logits):
    probs = np.exp(stacked_logits)
    return -np.sum(probs * stacked_logits, axis=-1, keepdims=False)
def get_mean_entropy(stacked_logits):
    return np.mean(get_entropy(stacked_logits), axis=1)
def logit_mean(stacked_logits):
    return logsumexp(stacked_logits, axis=1) - np.log(np.shape(stacked_logits)[1])
def mutual_information(stacked_logits):  # [N x ensemble x classes] log-probs
    # NOTE upstream names are swapped (entropy_mean/mean_entropy) but the result is
    # E[H] - H[E]; BALD is H[E] - E[H] = -(upstream). We return the standard BALD score.
    return get_entropy(logit_mean(stacked_logits)) - get_mean_entropy(stacked_logits)

FILES = [('pre_AL_binary_and_yield.csv', ';'), ('AL_1_binary_and_yield.csv', '\t'),
         ('AL_2_binary_and_yield.csv', '\t'), ('AL_3_binary_and_yield.csv', '\t')]

def ecfp(smi, bits=512):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    return np.array(rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits).GetFingerprint(mol), dtype=np.uint8)

def load(data_dir, cfg):
    d = pd.concat([pd.read_csv(os.path.join(data_dir, f), sep=s) for f, s in FILES], ignore_index=True)
    f = cfg['features']
    fps = {s: ecfp(s, f['ecfp_bits']) for s in d.startingmat_1_smiles.unique()}
    X_fp = np.stack([fps[s] for s in d.startingmat_1_smiles])
    X_oh = pd.get_dummies(d[f['one_hot']].fillna('none').astype(str)).values.astype(np.float32)
    X_num = d[f['numeric']].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    X = np.hstack([X_fp, X_oh, X_num]).astype(np.float32)
    return X, d.binary.values.astype(int), d.startingmat_1_smiles.values

def fit_ens(X, y, m, n, seed):
    spw = max((y == 0).sum(), 1) / max((y == 1).sum(), 1)
    models = []
    for k in range(n):
        rng = np.random.RandomState(seed * 1000 + k)
        idx = rng.choice(len(X), len(X), replace=True)  # bootstrap for ensemble diversity
        if len(np.unique(y[idx])) < 2: idx = np.arange(len(X))
        c = xgb.XGBClassifier(n_estimators=m['n_estimators'], max_depth=m['max_depth'], learning_rate=m['learning_rate'],
                              subsample=m['subsample'], colsample_bytree=m['colsample_bytree'], n_jobs=m['n_jobs'],
                              scale_pos_weight=spw, random_state=seed * 1000 + k, tree_method='hist', verbosity=0)
        c.fit(X[idx], y[idx]); models.append(c)
    return models

def ens_logp(models, X):
    p = np.stack([c.predict_proba(X) for c in models], axis=1)  # N x E x 2
    return np.log(np.clip(p, 1e-7, 1))

def mcc(models, X, y):
    return matthews_corrcoef(y, np.exp(ens_logp(models, X)).mean(1).argmax(1))

def run(cfg, X, y, subs, seed, strategy, ens_size=None, batch=None, time_limit=None):
    m = dict(cfg['model']); a = cfg['al']
    n_ens = ens_size or m['ensemble_size']; batch = batch or a['batch_molecules']
    us = np.unique(subs)
    tr_s, te_s = train_test_split(us, test_size=cfg['data']['test_frac_substrates'], random_state=seed)
    te = np.isin(subs, te_s); Xte, yte = X[te], y[te]
    rng = np.random.RandomState(seed)
    while True:  # redraw the initial set until it contains both classes (XGBoost needs 2 classes)
        pool = list(rng.permutation(tr_s)); lab = pool[:a['init_molecules']]; pool = pool[a['init_molecules']:]
        if len(np.unique(y[np.isin(subs, lab)])) == 2: break
    rows = []; t0 = time.time()
    while True:
        mask = np.isin(subs, lab)
        models = fit_ens(X[mask], y[mask], m, n_ens, seed)
        rows.append(dict(seed=seed, strategy=strategy, ensemble_size=n_ens, batch=batch, molecules=len(lab),
                         reactions=int(mask.sum()), test_mcc=mcc(models, Xte, yte), wall_s=time.time() - t0))
        print(json.dumps(rows[-1]), flush=True)
        if len(lab) >= a['max_molecules'] or not pool: break
        if time_limit and time.time() - t0 > time_limit: rows[-1]['timeout'] = True; break
        if strategy == 'random':
            pick = pool[:batch]
        else:
            pm = np.isin(subs, pool)
            sc = mutual_information(ens_logp(models, X[pm]))
            s = pd.Series(sc).groupby(subs[pm]).mean().sort_values(ascending=False)
            pick = list(s.index[:batch])
        lab += pick; pool = [p for p in pool if p not in set(pick)]
    full = fit_ens(X[np.isin(subs, tr_s)], y[np.isin(subs, tr_s)], m, n_ens, seed)
    return rows, mcc(full, Xte, yte)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--config'); ap.add_argument('--data_dir'); ap.add_argument('--out')
    ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--ens', type=int); ap.add_argument('--batch', type=int)
    ap.add_argument('--strategies', default='bald,random'); ap.add_argument('--time_limit', type=float)
    ap.add_argument('--max_molecules', type=int)
    A = ap.parse_args(); cfg = yaml.safe_load(open(A.config))
    if A.max_molecules: cfg['al']['max_molecules'] = A.max_molecules
    X, y, subs = load(A.data_dir, cfg)
    allrows, fulls = [], []
    for seed in map(int, A.seeds.split(',')):
        for st in A.strategies.split(','):
            r, f = run(cfg, X, y, subs, seed, st, A.ens, A.batch, A.time_limit)
            for x in r: x['full_pool_mcc'] = f
            allrows += r
    df = pd.DataFrame(allrows); df['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    df.to_csv(A.out, index=False)
