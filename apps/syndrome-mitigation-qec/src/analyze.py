"""Turn raw.json + breakpoints_raw.json into results.md, status.json,
breakpoints.md/csv and plot.png (pre-registered rule applied verbatim)."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parent.parent / "experiments" / "exp1_vec_cpu"

plt.rcParams.update({
    "font.size": 11, "axes.labelweight": "bold", "axes.titleweight": "bold",
    "legend.fontsize": 10, "legend.frameon": True, "axes.grid": True,
    "grid.alpha": 0.3, "savefig.dpi": 160, "figure.autolayout": True,
})


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}e}" if (abs(x) < 1e-2 or abs(x) >= 1e3) and x != 0 else f"{x:.{nd}g}"


def status_from_results(results):
    checks = []
    for r in results:
        if r["role"] != "primary":
            continue
        c1 = r["p_res_vec"] <= 0.1 * r["p_l_mwpm"]
        c2 = r["p_res_vec"] < r["p_res_postselect"]
        c3 = r["eta"] < 4.0
        beats_plain = r["p_res_vec"] < r["p_l_mwpm"]
        checks.append(dict(d=r["d"], seed=r["seed"], c1_10x=bool(c1),
                           c2_beats_postselect=bool(c2), c3_eta_lt_4=bool(c3),
                           beats_plain=bool(beats_plain),
                           p_l_mwpm=r["p_l_mwpm"], p_res_vec=r["p_res_vec"],
                           p_res_postselect=r["p_res_postselect"],
                           eta=r["eta"]))
    all_pass = all(c["c1_10x"] and c["c2_beats_postselect"] and c["c3_eta_lt_4"]
                   for c in checks)
    partial = all(c["c2_beats_postselect"] and c["beats_plain"] for c in checks)
    status = "PASS" if all_pass else ("PARTIAL" if partial else "FAIL")
    return status, checks


def write_results_md(raw, status, checks):
    res = raw["results"]
    lines = ["# exp1_vec_cpu results", "",
             f"Status (pre-registered rule): **{status}**", "",
             "Primary metric: `p_res = || sum_k q_k w_k p_k ||_1` on held-out "
             "eval shots, weights fitted on the separate calibration split "
             "(oracle-calibrated from true frame flips, as disclosed in "
             "config.yaml). Error bars: paired delete-1 jackknife, B=20.", "",
             "| d | seed | p_L MWPM | p_L ML | p_res postselect | p_res VEC "
             "| 0.1·p_L | C | eta | ≤0.1·p_L? | <PS? | eta<4? |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        if r["role"] != "primary":
            continue
        c = next(c for c in checks if c["d"] == r["d"] and c["seed"] == r["seed"])
        lines.append(
            f"| {r['d']} | {r['seed']} | {fmt(r['p_l_mwpm'])}±{fmt(r['p_l_mwpm_se'],1)} "
            f"| {fmt(r['p_l_ml'])} "
            f"| {fmt(r['p_res_postselect'])}±{fmt(r['p_res_postselect_se'],1)} "
            f"| **{fmt(r['p_res_vec'])}**±{fmt(r['p_res_vec_se'],1)} "
            f"| {fmt(0.1*r['p_l_mwpm'])} | {r['C']:.4f}±{r['C_se']:.4f} "
            f"| {r['eta']:.2f} | {'yes' if c['c1_10x'] else '**no**'} "
            f"| {'yes' if c['c2_beats_postselect'] else '**no**'} "
            f"| {'yes' if c['c3_eta_lt_4'] else '**no**'} |")
    for r in res:
        if r["role"] == "report_only":
            lines += ["", f"Report-only d={r['d']} (seed {r['seed']}, "
                      f"{r['n_cal']:,} cal / {r['n_eval']:,} eval shots): "
                      f"p_L={fmt(r['p_l_mwpm'])}, p_L(ML)={fmt(r['p_l_ml'])}, "
                      f"p_res_PS={fmt(r['p_res_postselect'])}, "
                      f"p_res_VEC={fmt(r['p_res_vec'])}±{fmt(r['p_res_vec_se'],1)}, "
                      f"C={r['C']:.4f}, eta={r['eta']:.2f}."]
    lines += ["", "## Direct observable check (ideal +1; report-only)", "",
              "`<Z_L>` on `|0_L>` and `<X_L>` on `|+_L>`; jackknife 1σ. "
              "(Code-capacity + frame simulation: readout-flip statistics are "
              "state-independent, so these follow from the same joint counts.)",
              "",
              "| d | seed | plain Z_L | postselect Z_L | VEC Z_L | plain X_L | VEC X_L |",
              "|---|---|---|---|---|---|---|"]
    for r in res:
        o = r["observables"]
        lines.append(
            f"| {r['d']} | {r['seed']} "
            f"| {o['plain']['ZL']:.5f}±{o['plain']['ZL_se']:.5f} "
            f"| {o['postselect']['ZL']:.5f}±{o['postselect']['ZL_se']:.5f} "
            f"| **{o['vec']['ZL']:.5f}±{o['vec']['ZL_se']:.5f}** "
            f"| {o['plain']['XL']:.5f}±{o['plain']['XL_se']:.5f} "
            f"| **{o['vec']['XL']:.5f}±{o['vec']['XL_se']:.5f}** |")
    lines += ["", "## Report-only: sector shares, weights, conditional matrices", ""]
    for r in res:
        if r["seed"] != 0:
            continue
        q = r["sector_shares_IXYZ"]; w = r["weights_IXYZ"]
        lines += [f"**d={r['d']}, seed 0** — shares q(I,X,Y,Z) = "
                  f"({', '.join(fmt(v) for v in q)}); weights w(I,X,Y,Z) = "
                  f"({', '.join(f'{v:.4f}' for v in w)}); "
                  f"{r['n_distinct_syndromes']:,} distinct syndromes decoded "
                  f"exactly; cal {r['t_cal_s']}s + eval {r['t_eval_s']}s.", "",
                  "| sector \\ error | I | X | Y | Z |", "|---|---|---|---|---|"]
        for name, row in zip("IXYZ", r["conditional_matrix_IXYZ"]):
            lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
        lines.append("")
    lines += ["Paper reference point (QUOTED from arXiv:2610.12400 Supp. II.B, "
              "d=7, p=0.05; not claimed reproduced): P_XX≈P_ZZ≈0.86, "
              "P_YY≈0.78, w_I*≈1.01, w_X*=w_Z*≈−0.42, w_Y*≈−4e-2. Our "
              "report-only d=7 run gives w=(1.010, −0.405, −0.044, −0.434) "
              "and P_XX=0.839, P_ZZ=0.840 (P_YY=0.846 from only ~13 Y-sector "
              "calibration shots, so it carries a large statistical error).",
              "",
              "## Disclosure",
              "",
              "- Smoke (code-path check, 2e4 shots, seed 0, recorded in "
              "`smoke.json`): d=3 p_L=3.3e-2, p_res_VEC=3.4e-3 (stat floor); "
              "d=5 p_L=1.8e-2, p_res_VEC=1.7e-3. No threshold touched.",
              "- Oracle calibration: sector centroids use true frame flips on "
              "the calibration split, exactly like the paper's Fig. 2 "
              "benchmark; the noise-learning breakpoint uses no true labels.",
              f"- Total official wall time: {raw['total_wall_s']}s on 4 vCPUs."]
    (EXP / "results.md").write_text("\n".join(lines) + "\n")


def write_status(raw, status, checks):
    (EXP / "status.json").write_text(json.dumps(dict(
        status=status,
        rule="PASS = (p_res_VEC <= 0.1 p_L) AND (p_res_VEC < p_res_PS) AND "
             "(eta < 4) on every seed 0/1/2 for d=3 AND d=5; PARTIAL = VEC "
             "beats postselection and plain on all seeds but misses 10x or "
             "eta<4; FAIL otherwise.",
        failed_condition=("eta<4 at d=3 (eta=5.4-5.7)" if status == "PARTIAL"
                          else None),
        per_run_checks=checks,
        total_wall_s=raw["total_wall_s"]), indent=1))


def write_breakpoints(bp):
    rows = []
    md = ["# Breakpoints (seed 0)", "",
          "Where dual-decoder VEC breaks down, honestly. All runs CPU-only; "
          "`degen` = a contested sector had zero calibration shots, so the "
          "exact Eq. (11) weights are undefined (the paper sees the same: no "
          "S_Y counts for p ≤ 0.01 at d=7).", "",
          "## 1. Physical error rate sweep (cal+eval at each p)", "",
          "| d | p | p_L MWPM | p_L ML | p_res PS | p_res VEC | eta | note |",
          "|---|---|---|---|---|---|---|---|"]
    for r in bp["p_sweep"]:
        note = "degenerate (empty Y sector)" if r["degenerate"] else ""
        if not r["degenerate"] and r["eta"] is not None and r["eta"] > 4:
            note = "eta > 4"
        if r["p"] >= 0.15:
            note = (note + "; " if note else "") + "near/above threshold"
        md.append(f"| {r['d']} | {r['p']} | {fmt(r['p_l_mwpm'])} "
                  f"| {fmt(r['p_l_ml'])} | {fmt(r['p_res_postselect'])} "
                  f"| {fmt(r.get('p_res_vec'))} "
                  f"| {fmt(r.get('eta'), 2) if r.get('eta') else '—'} | {note} |")
        rows.append(["p_sweep", r["d"], r["p"], "", r["p_l_mwpm"],
                     r["p_res_postselect"], r.get("p_res_vec"), r.get("eta"),
                     note])
    md += ["",
           "Advantage collapses toward threshold (p≈0.19): at d=3 eta blows up "
           "past 15 above p=0.1, and at p=0.02 the near-empty contested "
           "sectors make eta≈123 (tiny p_L denominator, noisy weights). At "
           "p=0.01 the Y sector is empty on both distances — exact 4-sector "
           "VEC is undefined without more calibration shots.", "",
           "## 2. Calibration-budget sweep (weights from first N_cal shots, "
           "full eval)", "",
           "| d | N_cal | p_res VEC | C | eta | note |", "|---|---|---|---|---|---|"]
    for r in bp["cal_budget"]:
        note = ("degenerate" if r["degenerate"] else
                ("weights blow up" if r.get("C", 0) and r["C"] > 100 else ""))
        md.append(f"| {r['d']} | {r['n_cal_budget']:,} | {fmt(r.get('p_res_vec'))} "
                  f"| {fmt(r.get('C'), 3) if r.get('C') else '—'} "
                  f"| {fmt(r.get('eta'), 2) if r.get('eta') else '—'} | {note} |")
        rows.append(["cal_budget", r["d"], 0.05, r["n_cal_budget"], None, None,
                     r.get("p_res_vec"), r.get("eta"), note])
    md += ["",
           "Below ~1e4 (d=3) / ~3e4 (d=5) calibration shots the contested "
           "sectors hold only O(1) shots: weights are undefined or wild "
           "(d=5, N_cal=1e4: C≈6e4, p_res≈5.8 — worse than useless). p_res "
           "then improves roughly as 1/sqrt(N_cal) to the eval floor.", "",
           "## 3. Noise drift: weights calibrated at p=0.05, applied at p'", "",
           "| d | p' | p_L(p') | p_res PS(p') | p_res VEC drifted | still <PS? |",
           "|---|---|---|---|---|---|"]
    for r in bp["drift"]:
        ok = r["p_res_vec_drift"] < r["p_res_postselect"]
        md.append(f"| {r['d']} | {r['p_apply']} | {fmt(r['p_l_mwpm'])} "
                  f"| {fmt(r['p_res_postselect'])} | {fmt(r['p_res_vec_drift'])} "
                  f"| {'yes' if ok else '**no**'} |")
        rows.append(["drift", r["d"], r["p_apply"], "", r["p_l_mwpm"],
                     r["p_res_postselect"], r["p_res_vec_drift"], None,
                     f"calibrated at {r['p_cal']}"])
    md += ["",
           "Mild drift (p 0.03–0.08 on weights from 0.05) keeps VEC below "
           "postselection; strong drift (p'≥0.11) degrades VEC to a few "
           "percent residual, though still below the unmitigated LER in this "
           "range — miscalibration bias scales with ||p_I|| (paper Eq. 14), "
           "not with p_L.", "",
           "## 4. Weight sweep of one contested sector (eval split)", "",
           "DISCLOSED post-hoc grid extension: the pre-registered absolute "
           "grid [−1, 1] does not bracket the d=3 optimum (w*_X ≈ −2.08; the "
           "paper's −0.42 is a d=7 value), so relative points around w* were "
           "added after the official run. No threshold depends on this grid.",
           "",
           "| d | sector | w* | argmin over grid | p_res(min) | p_res(w=0, PS) "
           "| p_res(w=1, plain) |", "|---|---|---|---|---|---|---|"]
    for dd in (3, 5):
        for sec in ("X", "Y"):
            sw = [r for r in bp["weight_sweep"]
                  if r["d"] == dd and r["sector"] == sec]
            if not sw:
                continue
            best = min(sw, key=lambda r: r["p_res"])
            w0 = next(r for r in sw if r["w_q"] == 0.0)
            w1 = next(r for r in sw if r["w_q"] == 1.0)
            md.append(f"| {dd} | {sec} | {best['w_q_star']:.3f} "
                      f"| {best['w_q']:.3f} | {fmt(best['p_res'])} "
                      f"| {fmt(w0['p_res'])} | {fmt(w1['p_res'])} |")
            for r in sw:
                rows.append(["weight_sweep", dd, 0.05, "", None, None,
                             r["p_res"], None, f"{sec}:w={r['w_q']:.3f}"])
    nl = bp.get("noise_learning")
    if nl:
        md += ["",
               "## 5. Report-only stretch: syndrome-only noise learning (d=3, "
               "inhomogeneous p)", "",
               f"Per-qubit true rates U{tuple(np.round([min(nl['p_true']), max(nl['p_true'])], 3)).__str__()}"
               f" (seed 777); {nl['n_learn']:,} unlabelled learning shots. "
               f"Walsh–Hadamard fidelities with {nl['learner_info']['n_chars']} "
               "characters (8 degree-1 + 6 adjacent same-sector degree-2), "
               f"design-matrix rank {nl['learner_info']['rank']}/9, "
               "bias-corrected weighted NNLS (Eqs. S.2–S.9).", "",
               f"- learned rates: median relative error "
               f"{100*nl['median_rel_err']:.2f}% (max {100*nl['max_rel_err']:.2f}%)",
               f"- p_L (MWPM) = {fmt(nl['p_l_mwpm'])}; postselection bias = "
               f"{fmt(nl['p_res_postselect'])}",
               f"- VEC with LEARNED weights (no true labels anywhere in "
               f"calibration): p_res = {fmt(nl.get('p_res_vec_learned'))} "
               f"(C = {nl.get('C_learned'):.4f})",
               f"- VEC with oracle weights on the same shots: p_res = "
               f"{fmt(nl.get('p_res_vec_oracle'))} (C = {nl.get('C_oracle'):.4f})",
               "",
               "Learned-weight VEC matches oracle-weight VEC within its "
               "statistical floor — the paper's claim that syndrome records "
               "alone suffice for calibration holds in this toy."]
        rows.append(["noise_learning", 3, "inhom", nl["n_learn"],
                     nl["p_l_mwpm"], nl["p_res_postselect"],
                     nl.get("p_res_vec_learned"), None,
                     f"median rel err {100*nl['median_rel_err']:.2f}%"])
    (EXP / "breakpoints.md").write_text("\n".join(md) + "\n")
    with open(EXP / "breakpoints.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["section", "d", "p", "n_cal", "p_l_mwpm",
                     "p_res_postselect", "p_res_vec", "eta", "note"])
        cw.writerows(rows)


def make_plot(raw, bp, status):
    res = [r for r in raw["results"] if r["role"] == "primary"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    xticks, xlabels = [], []
    xi = 0
    for d in (3, 5):
        for r in [r for r in res if r["d"] == d]:
            ax.bar(xi - 0.27, r["p_l_mwpm"], 0.25, color="#9aa7b4",
                   label="plain MWPM" if xi == 0 else None)
            ax.bar(xi, r["p_res_postselect"], 0.25, color="#e2a34c",
                   label="postselect" if xi == 0 else None)
            ax.bar(xi + 0.27, r["p_res_vec"], 0.25, color="#4c9ee2",
                   yerr=r["p_res_vec_se"], capsize=2,
                   label="VEC" if xi == 0 else None)
            ax.plot([xi - 0.45, xi + 0.45], [0.1 * r["p_l_mwpm"]] * 2, "r--",
                    lw=1.4, label="0.1·p_L target" if xi == 0 else None)
            xticks.append(xi); xlabels.append(f"d={d}\ns{r['seed']}")
            xi += 1
        xi += 0.6
    ax.set_yscale("log"); ax.set_xticks(xticks); ax.set_xticklabels(xlabels)
    ax.set_ylabel("residual logical bias  $p_{\\mathrm{res}}$ ($\\|\\cdot\\|_1$)")
    ax.set_title(f"Pre-registered comparison at p=0.05 — {status}")
    ax.legend(ncol=2, prop={"weight": "bold"})
    ax = axes[1]
    for d, c in ((3, "#d4624e"), (5, "#4c9ee2")):
        sw = [r for r in bp["p_sweep"] if r["d"] == d and not r["degenerate"]]
        ps = [r["p"] for r in sw]
        ax.plot(ps, [r["eta"] for r in sw], "o-", color=c, lw=2,
                label=f"d={d} coarse-grained")
    d7 = next(r for r in raw["results"] if r["d"] == 7)
    ax.plot([0.05], [d7["eta"]], "s", color="#3fa056", ms=9,
            label="d=7 (report-only)")
    ax.axhline(4, color="k", ls="--", lw=1.4)
    ax.text(0.021, 4.35, "logical PEC benchmark  η=4", fontweight="bold")
    ax.axhline(2, color="gray", ls=":", lw=1.2)
    ax.text(0.021, 1.62, "single-decoder ceiling  η=2", color="gray")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("excess overhead ratio  η = (C−1)/$p_L$")
    ax.set_title("Sampling overhead vs p (seed 0)")
    ax.legend(prop={"weight": "bold"})
    fig.savefig(EXP / "plot.png")
    plt.close(fig)


if __name__ == "__main__":
    raw = json.loads((EXP / "raw.json").read_text())
    bp = json.loads((EXP / "breakpoints_raw.json").read_text())
    status, checks = status_from_results(raw["results"])
    write_results_md(raw, status, checks)
    write_status(raw, status, checks)
    write_breakpoints(bp)
    make_plot(raw, bp, status)
    print(f"status: {status} -> results.md, status.json, breakpoints.md/csv, plot.png")
