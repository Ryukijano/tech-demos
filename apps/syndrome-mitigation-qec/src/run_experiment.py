"""exp1_vec_cpu driver.

Usage:
  python src/run_experiment.py --smoke        # disclosed tiny code-path check
  python src/run_experiment.py                # official pre-registered run
  python src/run_experiment.py --breakpoints  # breakpoint sweeps (after official)

Writes into experiments/exp1_vec_cpu/:
  smoke.json | raw.json, results.csv | breakpoints_raw.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import vecqec as V  # noqa: E402

EXP = Path(__file__).resolve().parent.parent / "experiments" / "exp1_vec_cpu"


def run_one(d, p, seed, n_cal, n_eval, cal_shards, eval_shards,
            cal_seed, eval_seed, ml_cache=None):
    """One (d, p, seed) pipeline. Returns dict with counts + summary stats."""
    code = V.RotatedSurfaceCode(d)
    probs = V.depolarizing_probs(np.full(d * d, p))
    mwpm = V.MWPMDecoder(code, p)
    ml = ml_cache if ml_cache is not None else V.ExactMLDecoder(code, probs)
    t0 = time.time()
    cal = V.sample_shots(code, np.full(d * d, p), n_cal, cal_seed, mwpm, ml)
    t_cal = time.time() - t0
    t0 = time.time()
    ev = V.sample_shots(code, np.full(d * d, p), n_eval, eval_seed, mwpm, ml)
    t_eval = time.time() - t0
    cal_counts = V.joint_counts(cal, cal_shards)
    eval_counts = V.joint_counts(ev, eval_shards)
    out = summarize(cal_counts, eval_counts)
    out.update(d=d, p=p, seed=seed, n_cal=n_cal, n_eval=n_eval,
               t_cal_s=round(t_cal, 2), t_eval_s=round(t_eval, 2),
               n_distinct_syndromes=len(ml.cache),
               cal_counts=cal_counts.tolist(), eval_counts=eval_counts.tolist())
    return out, ml


def summarize(cal_counts, eval_counts):
    """All exp1 metrics from sharded joint (sector, resid) counts."""
    cal_tot = cal_counts.sum(axis=0)
    eval_tot = eval_counts.sum(axis=0)
    q, pc = V.sector_stats(cal_tot)
    w, C, degen = V.vec_weights(q, pc)
    res = dict(degenerate=bool(degen))
    pl, pl_se = V.jackknife_scalar(eval_counts, V.plain_pl)
    ps, ps_se = V.jackknife_scalar(eval_counts, V.postselect_bias)
    res.update(p_l_mwpm=pl, p_l_mwpm_se=pl_se,
               p_res_postselect=ps, p_res_postselect_se=ps_se,
               p_l_ml=V.ml_pl(eval_tot))
    # report-only sector shares and conditional matrix (report order I,X,Y,Z)
    ro = V.REPORT_ORDER
    cond = np.zeros((4, 4))
    for i, s in enumerate(ro):
        ns = cal_tot[s].sum()
        if ns > 0:
            cond[i] = cal_tot[s][ro] / ns
    res["sector_shares_IXYZ"] = [float(q[s]) for s in ro]
    res["conditional_matrix_IXYZ"] = cond.tolist()
    if not degen:
        jk = V.jackknife_pres(cal_counts, eval_counts)
        eta = (jk["C"] - 1.0) / pl if pl > 0 else float("nan")
        res.update(weights_IXYZ=[float(w[s]) for s in ro], C=jk["C"],
                   C_se=jk["C_se"], eta=eta,
                   p_res_vec=jk["p_res_vec"], p_res_vec_se=jk["p_res_vec_se"])
        # direct observable check (ideal +1 for both)
        w_plain = np.ones(4)
        w_ps = V.postselect_weights(q)
        obs = {}
        for name, ww in [("plain", w_plain), ("postselect", w_ps), ("vec", w)]:
            z, x = V.observable_estimates(eval_tot, ww)
            # jackknife over eval shards with fixed weights
            grp = V._regroup(eval_counts, min(20, eval_counts.shape[0]))
            B = grp.shape[0]
            reps = np.array([V.observable_estimates(grp.sum(axis=0) - grp[b], ww)
                             for b in range(B)])
            se = np.sqrt((B - 1) / B * ((reps - reps.mean(axis=0)) ** 2).sum(axis=0))
            obs[name] = dict(ZL=z, ZL_se=float(se[0]), XL=x, XL_se=float(se[1]))
        res["observables"] = obs
    return res


def official(cfg):
    results = []
    t_start = time.time()
    for dcfg in cfg["runs"]:
        d, p = dcfg["d"], cfg["noise"]["p"]
        for seed in dcfg["seeds"]:
            cal_seed = cfg["seeding"]["cal_base"] + 100 * d + seed
            eval_seed = cfg["seeding"]["eval_base"] + 100 * d + seed
            t0 = time.time()
            out, _ = run_one(d, p, seed, dcfg["n_cal"], dcfg["n_eval"],
                             dcfg["cal_shards"], dcfg["eval_shards"],
                             cal_seed, eval_seed)
            out["role"] = dcfg["role"]
            out["wall_s"] = round(time.time() - t0, 2)
            results.append(out)
            print(f"d={d} seed={seed} [{dcfg['role']}]: "
                  f"p_L={out['p_l_mwpm']:.3e}  "
                  f"p_res_PS={out['p_res_postselect']:.3e}  "
                  f"p_res_VEC={out.get('p_res_vec', float('nan')):.3e}  "
                  f"eta={out.get('eta', float('nan')):.2f}  "
                  f"({out['wall_s']}s)", flush=True)
    raw = dict(config=cfg, results=results,
               total_wall_s=round(time.time() - t_start, 2))
    (EXP / "raw.json").write_text(json.dumps(raw, indent=1))
    # flat csv
    import csv
    with open(EXP / "results.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["d", "seed", "role", "p_l_mwpm", "p_l_mwpm_se", "p_l_ml",
                     "p_res_postselect", "p_res_postselect_se",
                     "p_res_vec", "p_res_vec_se", "C", "eta",
                     "w_I", "w_X", "w_Y", "w_Z",
                     "q_I", "q_X", "q_Y", "q_Z", "wall_s"])
        for r in results:
            ws = r.get("weights_IXYZ", [float("nan")] * 4)
            qs = r["sector_shares_IXYZ"]
            cw.writerow([r["d"], r["seed"], r["role"],
                         r["p_l_mwpm"], r["p_l_mwpm_se"], r["p_l_ml"],
                         r["p_res_postselect"], r["p_res_postselect_se"],
                         r.get("p_res_vec"), r.get("p_res_vec_se"),
                         r.get("C"), r.get("eta"),
                         *ws, *qs, r["wall_s"]])
    print(f"official run done in {raw['total_wall_s']}s -> raw.json, results.csv")


def breakpoints(cfg):
    bp = cfg["breakpoints"]
    raw = json.loads((EXP / "raw.json").read_text())
    out = dict(p_sweep=[], cal_budget=[], drift=[], weight_sweep=[],
               noise_learning=None)
    t_start = time.time()

    # -- (a) physical error rate sweep (seed 0), reused for drift ------------
    sweep_eval_counts = {}   # (d, p) -> eval 4x4 counts
    for d in bp["p_sweep"]["d_values"]:
        n = bp["p_sweep"]["n_shots"][str(d)]
        for p in bp["p_sweep"]["p_values"]:
            t0 = time.time()
            r, _ = run_one(d, p, 0, n, n, 20, 20,
                           cfg["seeding"]["cal_base"] + 100 * d + 50,
                           cfg["seeding"]["eval_base"] + 100 * d + 50)
            sweep_eval_counts[(d, p)] = np.array(r["eval_counts"]).sum(axis=0)
            rec = {k: r.get(k) for k in
                   ["d", "p", "degenerate", "p_l_mwpm", "p_l_ml",
                    "p_res_postselect", "p_res_vec", "p_res_vec_se", "C", "eta",
                    "weights_IXYZ", "sector_shares_IXYZ"]}
            rec["wall_s"] = round(time.time() - t0, 2)
            out["p_sweep"].append(rec)
            print(f"p-sweep d={d} p={p}: p_L={r['p_l_mwpm']:.3e} "
                  f"p_res_VEC={r.get('p_res_vec', float('nan')):.3e} "
                  f"eta={r.get('eta', float('nan')):.2f} degen={r['degenerate']}",
                  flush=True)

    # -- (b) calibration-budget sweep from official shard prefixes -----------
    for rr in raw["results"]:
        if rr["role"] != "primary" or rr["seed"] != 0:
            continue
        d = rr["d"]
        cal_counts = np.array(rr["cal_counts"])
        eval_tot = np.array(rr["eval_counts"]).sum(axis=0)
        n_cal, n_shards = rr["n_cal"], cal_counts.shape[0]
        per_shard = n_cal // n_shards
        for budget in bp["cal_budget"]["budgets"]:
            k = max(1, int(round(budget / per_shard)))
            if k > n_shards:
                continue
            pref = cal_counts[:k].sum(axis=0)
            q, pc = V.sector_stats(pref)
            w, C, degen = V.vec_weights(q, pc)
            rec = dict(d=d, n_cal_budget=int(k * per_shard), degenerate=bool(degen))
            if not degen:
                rec.update(p_res_vec=V.residual_bias(eval_tot, w), C=float(C),
                           eta=float((C - 1) / V.plain_pl(eval_tot)))
            out["cal_budget"].append(rec)

    # -- (c) drift: weights calibrated at p, applied at p' -------------------
    for rr in raw["results"]:
        if rr["role"] != "primary" or rr["seed"] != 0:
            continue
        d = rr["d"]
        if d not in bp["p_sweep"]["d_values"]:
            continue
        q, pc = V.sector_stats(np.array(rr["cal_counts"]).sum(axis=0))
        w, _, degen = V.vec_weights(q, pc)
        if degen:
            continue
        for p2 in bp["p_sweep"]["p_values"]:
            ec = sweep_eval_counts[(d, p2)]
            out["drift"].append(dict(
                d=d, p_cal=rr["p"], p_apply=p2,
                p_res_vec_drift=V.residual_bias(ec, w),
                p_l_mwpm=V.plain_pl(ec),
                p_res_postselect=V.postselect_bias(ec)))

    # -- (d) weight sweep of one contested sector ----------------------------
    for rr in raw["results"]:
        if rr["role"] != "primary" or rr["seed"] != 0:
            continue
        d = rr["d"]
        cal_tot = np.array(rr["cal_counts"]).sum(axis=0)
        eval_tot = np.array(rr["eval_counts"]).sum(axis=0)
        q, pc = V.sector_stats(cal_tot)
        wstar, _, degen = V.vec_weights(q, pc)
        if degen:
            continue
        for sec_name, sec in [("X", 1), ("Y", 3)]:
            for wq in bp["weight_sweep"]["grid"]:
                w = wstar.copy()
                # adjust consensus weight to keep sum_k q_k w_k = 1 (paper's
                # Fig. 2d convention), other contested weights at w*
                w[sec] = wq
                w[0] = wstar[0] - (q[sec] / q[0]) * (wq - wstar[sec])
                out["weight_sweep"].append(dict(
                    d=d, sector=sec_name, w_q=float(wq),
                    w_q_star=float(wstar[sec]),
                    p_res=V.residual_bias(eval_tot, w),
                    C=float(np.sum(q * w ** 2))))

    # -- (e) report-only stretch: syndrome-only noise learning at d=3 --------
    if bp.get("noise_learning", {}).get("enabled", False):
        out["noise_learning"] = noise_learning_stretch(cfg, bp["noise_learning"])

    out["total_wall_s"] = round(time.time() - t_start, 2)
    (EXP / "breakpoints_raw.json").write_text(json.dumps(out, indent=1))
    print(f"breakpoints done in {out['total_wall_s']}s -> breakpoints_raw.json")


def noise_learning_stretch(cfg, nl):
    """d=3, inhomogeneous per-qubit p: learn p_q from unlabelled calibration
    syndromes (Eqs. 16-17 / S.2-S.9), rebuild D_off and the VEC weights from
    the LEARNED model only (no true labels), evaluate on held-out shots."""
    d = 3
    rng = np.random.default_rng(nl["noise_seed"])
    p_true = rng.uniform(*nl["p_range"], size=d * d)
    code = V.RotatedSurfaceCode(d)
    probs_true = V.depolarizing_probs(p_true)
    p_nom = float(np.mean(p_true))
    mwpm = V.MWPMDecoder(code, p_nom)   # fixed real-time device
    ml_true = V.ExactMLDecoder(code, probs_true)
    t0 = time.time()
    cal = V.sample_shots(code, p_true, nl["n_learn"], nl["cal_seed"], mwpm, ml_true)
    ev = V.sample_shots(code, p_true, nl["n_eval"], nl["eval_seed"], mwpm, ml_true)
    # learn rates from unlabelled syndromes only
    bits = ((cal.key[:, None] >> np.arange(code.nd)[None, :]) & 1)
    p_hat, info = V.learn_qubit_rates(code, bits)
    # learned-model calibration: analytic sectors/centroids, learned D_off
    q_l, pc_l, cls_off_l = V.model_calibration_d3(
        code, V.depolarizing_probs(p_hat), mwpm)
    w_l, C_l, degen_l = V.vec_weights(q_l, pc_l)
    # evaluate learned-weight VEC on eval shots, with sectors from learned D_off
    sz = ev.key & ((1 << code.n_z) - 1)
    sx = ev.key >> code.n_z
    cls_rt = mwpm.decode(sz, sx)
    sector_l = cls_rt ^ cls_off_l[ev.key]
    counts_l = np.zeros((4, 4), dtype=np.int64)
    np.add.at(counts_l, (sector_l, ev.resid), 1)
    # oracle counterpart on the same shots (true-model D_off + oracle centroids)
    cal_counts = V.joint_counts(cal, 20).sum(axis=0)
    eval_counts = V.joint_counts(ev, 20).sum(axis=0)
    q_o, pc_o = V.sector_stats(cal_counts)
    w_o, C_o, _ = V.vec_weights(q_o, pc_o)
    res = dict(
        p_true=p_true.tolist(), p_hat=p_hat.tolist(),
        median_rel_err=float(np.median(np.abs(p_hat - p_true) / p_true)),
        max_rel_err=float(np.max(np.abs(p_hat - p_true) / p_true)),
        learner_info=info, n_learn=nl["n_learn"], n_eval=nl["n_eval"],
        p_l_mwpm=V.plain_pl(eval_counts),
        p_res_postselect=V.postselect_bias(eval_counts),
        degenerate_learned=bool(degen_l),
        wall_s=round(time.time() - t0, 2))
    if not degen_l:
        res.update(p_res_vec_learned=V.residual_bias(counts_l, w_l),
                   C_learned=float(C_l),
                   weights_learned_IXYZ=[float(w_l[s]) for s in V.REPORT_ORDER])
    if w_o is not None:
        res.update(p_res_vec_oracle=V.residual_bias(eval_counts, w_o),
                   C_oracle=float(C_o))
    return res


def smoke():
    """Disclosed tiny code-path check. Not an official run; no thresholds
    were tuned on it (thresholds are fixed in config.yaml)."""
    t0 = time.time()
    rec = {"note": "smoke = code-path check only, 2e4 shots, seed 0",
           "runs": []}
    for d in (3, 5):
        out, _ = run_one(d, 0.05, 0, 20_000, 20_000, 20, 20, 999, 1999)
        out.pop("cal_counts"); out.pop("eval_counts")
        rec["runs"].append(out)
        print(f"smoke d={d}: p_L={out['p_l_mwpm']:.3e} "
              f"p_res_PS={out['p_res_postselect']:.3e} "
              f"p_res_VEC={out.get('p_res_vec', float('nan')):.3e} "
              f"eta={out.get('eta', float('nan')):.2f}")
    rec["wall_s"] = round(time.time() - t0, 2)
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / "smoke.json").write_text(json.dumps(rec, indent=1))
    print(f"smoke done in {rec['wall_s']}s -> smoke.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--breakpoints", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        cfg = yaml.safe_load((EXP / "config.yaml").read_text())
        if args.breakpoints:
            breakpoints(cfg)
        else:
            official(cfg)
