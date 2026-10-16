"""Media for the PR / Pages card (CPU + ffmpeg only):

  artifacts/syndrome-mitigation-qec.png  - 2x3 results grid
  artifacts/syndrome-mitigation-qec.mp4  - running <Z_L> estimate vs shots for
                                           plain / postselect / VEC (d=3 & d=5)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import vecqec as V  # noqa: E402

EXP = ROOT / "experiments" / "exp1_vec_cpu"
ART = ROOT / "artifacts"

plt.rcParams.update({
    "font.size": 10.5, "axes.labelweight": "bold", "axes.titleweight": "bold",
    "legend.fontsize": 9.5, "legend.frameon": True, "axes.grid": True,
    "grid.alpha": 0.3, "savefig.dpi": 150,
})
C_PLAIN, C_PS, C_VEC = "#9aa7b4", "#e2a34c", "#4c9ee2"


def grid_png(raw, bp):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6))
    fig.suptitle("syndrome-mitigation-qec — dual-decoder Virtual Error "
                 "Cancellation (arXiv:2610.12400, toy re-implementation, CPU)",
                 fontweight="bold", fontsize=13)
    res = [r for r in raw["results"] if r["role"] == "primary"]

    ax = axes[0, 0]
    xt, xl, xi = [], [], 0
    for d in (3, 5):
        for r in [r for r in res if r["d"] == d]:
            ax.bar(xi - 0.27, r["p_l_mwpm"], 0.25, color=C_PLAIN,
                   label="plain MWPM" if xi == 0 else None)
            ax.bar(xi, r["p_res_postselect"], 0.25, color=C_PS,
                   label="postselect" if xi == 0 else None)
            ax.bar(xi + 0.27, r["p_res_vec"], 0.25, color=C_VEC,
                   yerr=r["p_res_vec_se"], capsize=2,
                   label="VEC" if xi == 0 else None)
            ax.plot([xi - 0.45, xi + 0.45], [0.1 * r["p_l_mwpm"]] * 2, "r--",
                    lw=1.3, label="0.1·$p_L$ target" if xi == 0 else None)
            xt.append(xi); xl.append(f"d={d}\ns{r['seed']}"); xi += 1
        xi += 0.6
    ax.set_yscale("log"); ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel("residual logical bias $p_{\\mathrm{res}}$")
    ax.set_title("Residual bias at p=0.05 (held-out eval, seeds 0/1/2)")
    ax.legend(ncol=2, prop={"weight": "bold"})

    ax = axes[0, 1]
    for d, c in ((3, "#d4624e"), (5, C_VEC)):
        sw = [r for r in bp["p_sweep"] if r["d"] == d and not r["degenerate"]]
        ax.plot([r["p"] for r in sw], [r["eta"] for r in sw], "o-", color=c,
                lw=2, label=f"d={d}")
    d7 = next(r for r in raw["results"] if r["d"] == 7)
    ax.plot([0.05], [d7["eta"]], "s", color="#3fa056", ms=9,
            label="d=7 (report-only)")
    ax.axhline(4, color="k", ls="--", lw=1.3)
    ax.axhline(2, color="gray", ls=":", lw=1.1)
    ax.text(0.021, 4.4, "logical PEC  η=4", fontweight="bold")
    ax.text(0.021, 1.6, "single-decoder  η=2", color="gray")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("η = (C−1)/$p_L$")
    ax.set_title("Sampling overhead vs p (seed 0)")
    ax.legend(prop={"weight": "bold"})

    ax = axes[0, 2]
    for d, c in ((3, "#d4624e"), (5, C_VEC)):
        sw = sorted([r for r in bp["weight_sweep"]
                     if r["d"] == d and r["sector"] == "X"],
                    key=lambda r: r["w_q"])
        ws = [r["w_q"] for r in sw]
        ax.plot(ws, [r["p_res"] for r in sw], "o-", color=c, lw=2, ms=4,
                label=f"d={d}, X sector")
        ax.axvline(sw[0]["w_q_star"], color=c, ls="--", lw=1.2)
    ax.axvline(0, color=C_PS, lw=1.6)
    ax.axvline(1, color=C_PLAIN, lw=1.6)
    ax.text(-0.08, 0.82, "postselect\n$w_Q$=0", color="#a06c20", ha="right",
            fontweight="bold", transform=ax.get_xaxis_transform())
    ax.text(0.95, 0.62, "plain\n$w_Q$=1", color="#5c6b7a", ha="right",
            fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_yscale("log")
    ax.set_xlabel("contested sector weight $w_X$ (dashed: $w_X^*$)")
    ax.set_ylabel("$p_{\\mathrm{res}}$ on eval")
    ax.set_title("Weighting contested shots (paper Fig. 2d analogue)")
    ax.legend(prop={"weight": "bold"})

    ax = axes[1, 0]
    cond = np.array(d7["conditional_matrix_IXYZ"])
    im = ax.imshow(cond, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            v = cond[i, j]
            ax.text(j, i, f"{v:.2f}" if v >= 5e-3 else f"{v:.0e}",
                    ha="center", va="center", fontsize=9,
                    color="white" if v > 0.6 else "black",
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(4)); ax.set_xticklabels(["I", "X", "Y", "Z"])
    ax.set_yticks(range(4)); ax.set_yticklabels(["I", "X", "Y", "Z"])
    ax.set_xlabel("conditional logical error Q′ (real-time frame)")
    ax.set_ylabel("disagreement sector Q")
    ax.set_title("d=7 conditional matrix at p=0.05\n"
                 "(paper quotes P_XX≈P_ZZ≈0.86, P_YY≈0.78)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    for d, c in ((3, "#d4624e"), (5, C_VEC)):
        cb = [r for r in bp["cal_budget"]
              if r["d"] == d and not r["degenerate"] and r.get("p_res_vec", 10) < 1]
        ax.plot([r["n_cal_budget"] for r in cb], [r["p_res_vec"] for r in cb],
                "o-", color=c, lw=2, label=f"d={d} VEC")
    for r in [r for r in res if r["seed"] == 0]:
        c = "#d4624e" if r["d"] == 3 else C_VEC
        ax.axhline(r["p_res_postselect"], color=c, ls=":", lw=1.3)
    ax.text(1.1e3, 2.4e-2, "dotted: postselection bias", fontweight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("calibration shots $N_{\\mathrm{cal}}$")
    ax.set_ylabel("$p_{\\mathrm{res}}$ VEC (full eval)")
    ax.set_title("Calibration budget (empty/wild sectors cut below 1e4)")
    ax.legend(prop={"weight": "bold"})

    ax = axes[1, 2]
    nl = bp["noise_learning"]
    pt, ph = np.array(nl["p_true"]), np.array(nl["p_hat"])
    lim = [0.025, 0.09]
    ax.plot(lim, lim, "k--", lw=1.2, label="$\\hat p_q = p_q$")
    ax.plot(pt, ph, "o", color="#3fa056", ms=8, mec="k")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("true per-qubit rate $p_q$")
    ax.set_ylabel("learned $\\hat p_q$ (syndromes only)")
    ax.set_title(f"d=3 noise learning (Eqs. 16–17): median rel. err. "
                 f"{100*nl['median_rel_err']:.2f}%\nVEC learned "
                 f"$p_{{res}}$={nl['p_res_vec_learned']:.1e} vs oracle "
                 f"{nl['p_res_vec_oracle']:.1e}")
    ax.legend(prop={"weight": "bold"})

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ART / "syndrome-mitigation-qec.png")
    plt.close(fig)
    print("wrote artifacts/syndrome-mitigation-qec.png")


def running_estimates(raw, d, n_show):
    """Recreate official per-shot eval data (deterministic stim seeds) and
    return running <Z_L> estimates for plain / postselect / VEC."""
    cfg = raw["config"]
    dcfg = next(c for c in cfg["runs"] if c["d"] == d)
    p = cfg["noise"]["p"]
    rr = next(r for r in raw["results"] if r["d"] == d and r["seed"] == 0)
    q, pc = V.sector_stats(np.array(rr["cal_counts"]).sum(axis=0))
    w_vec, _, _ = V.vec_weights(q, pc)
    code = V.RotatedSurfaceCode(d)
    mwpm = V.MWPMDecoder(code, p)
    ml = V.ExactMLDecoder(code, V.depolarizing_probs(np.full(d * d, p)))
    ev = V.sample_shots(code, np.full(d * d, p), n_show,
                        cfg["seeding"]["eval_base"] + 100 * d + 0, mwpm, ml)
    m = np.where(ev.resid & 1, -1.0, 1.0)          # Z_L readout flip
    n = np.arange(1, n_show + 1)
    run_plain = np.cumsum(m) / n
    w_ps = np.where(ev.sector == 0, 1.0, 0.0)
    run_ps = np.cumsum(w_ps * m) / np.maximum(np.cumsum(w_ps), 1)
    wv = w_vec[ev.sector]
    run_vec = np.cumsum(wv * m) / n
    return n, run_plain, run_ps, run_vec


def mp4(raw):
    n_show = 300_000
    data = {d: running_estimates(raw, d, n_show) for d in (3, 5)}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    fig.suptitle("Measure now, mitigate later — running $\\langle Z_L\\rangle$ "
                 "estimate on $|0_L\\rangle$ (ideal +1), p=0.05, seed 0",
                 fontweight="bold", fontsize=13)
    arts = {}
    for ax, d in zip(axes, (3, 5)):
        n, rp, rs, rv = data[d]
        ax.set_xscale("log")
        ax.set_xlim(100, n_show)
        ax.set_ylim(0.955, 1.01)
        ax.axhline(1.0, color="k", lw=1.2, ls="--")
        ax.text(120, 1.0012, "ideal", fontweight="bold")
        lp, = ax.plot([], [], color=C_PLAIN, lw=2.4, label="plain MWPM (w≡1)")
        ls_, = ax.plot([], [], color=C_PS, lw=2.4, label="postselect ($w_Q$=0)")
        lv, = ax.plot([], [], color=C_VEC, lw=2.6, label="VEC (Eq. 11 weights)")
        ax.set_xlabel("shots N"); ax.set_ylabel("$\\langle Z_L\\rangle$ estimate")
        ax.set_title(f"d={d} rotated surface code")
        ax.legend(loc="lower right", prop={"weight": "bold"})
        arts[d] = (lp, ls_, lv)
    fig.subplots_adjust(bottom=0.2, top=0.86, wspace=0.25)
    txt = fig.text(0.5, 0.03, "", ha="center", fontsize=11, fontweight="bold")

    frames = 240
    idxs = np.unique(np.logspace(2, np.log10(n_show), frames).astype(int))

    def update(f):
        k = idxs[min(f, len(idxs) - 1)]
        for d in (3, 5):
            n, rp, rs, rv = data[d]
            lp, ls_, lv = arts[d]
            lp.set_data(n[:k], rp[:k])
            ls_.set_data(n[:k], rs[:k])
            lv.set_data(n[:k], rv[:k])
        txt.set_text(f"N = {k:,} shots — negative VEC weights on decoder-"
                     f"disagreement shots cancel the residual logical bias")
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(idxs), blit=False)
    ani.save(ART / "syndrome-mitigation-qec.mp4",
             writer=animation.FFMpegWriter(fps=24, bitrate=2200))
    plt.close(fig)
    print("wrote artifacts/syndrome-mitigation-qec.mp4")


if __name__ == "__main__":
    raw = json.loads((EXP / "raw.json").read_text())
    bp = json.loads((EXP / "breakpoints_raw.json").read_text())
    ART.mkdir(exist_ok=True)
    grid_png(raw, bp)
    mp4(raw)
