"""raw.json -> results.csv/.md, breakpoints.csv/.md, plot.png, status.json."""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))
import lmx

EXP = os.path.join(lmx.APP, "experiments/exp1_cpu_match")
cfg = yaml.safe_load(open(os.path.join(EXP, "config.yaml")))
raw = json.load(open(os.path.join(EXP, "raw.json")))
th = cfg["thresholds"]

rows = []
for s, d in raw["seeds"].items():
    for pid, r in d.items():
        rows.append(
            dict(
                seed=int(s),
                pair=pid,
                model=r["model"],
                n_matches=r["n_matches"],
                n_inliers=r["n_inliers"],
                inlier_ratio=r["inlier_ratio"],
                latency_s_match=r["latency_s_match"],
            )
        )
per = pd.DataFrame(rows).sort_values(["seed", "pair"])
per.to_csv(os.path.join(EXP, "results_per_pair.csv"), index=False)

g = (
    per.groupby("seed")
    .agg(
        mean_latency_s=("latency_s_match", "mean"),
        max_latency_s=("latency_s_match", "max"),
        mean_n_matches=("n_matches", "mean"),
        mean_inlier_ratio=("inlier_ratio", "mean"),
    )
    .reset_index()
)
g["latency_pass"] = g.mean_latency_s <= th["latency_s_match_max"]

# Determinism: n_matches identical across seeds for each pair
det_ok = True
det_rows = []
for pid, d in per.groupby("pair"):
    counts = sorted(d.n_matches.unique().tolist())
    ok = len(counts) == 1
    det_ok = det_ok and ok
    det_rows.append(dict(pair=pid, n_matches_across_seeds=",".join(str(c) for c in d.sort_values("seed").n_matches.tolist()), identical=ok))
det_df = pd.DataFrame(det_rows)
g["determinism_pass"] = det_ok
g.to_csv(os.path.join(EXP, "results.csv"), index=False)

latency_all = bool(g.latency_pass.all())
det_block = bool(det_ok and latency_all)

# Rotation block
bp = pd.DataFrame(raw["breakpoints"])
bp.to_csv(os.path.join(EXP, "breakpoints.csv"), index=False)
ok_lvl = float(cfg["breakpoints"]["degrade_levels"]["ok"])
deg_lvl = float(cfg["breakpoints"]["degrade_levels"]["degraded"])


def lab(v: float) -> str:
    if v >= ok_lvl:
        return "ok"
    if v >= deg_lvl:
        return "degraded"
    return "broken"


if not bp.empty:
    bp["verdict"] = bp.inlier_ratio.map(lab)

ang_ge = float(th["rotation_angles_ge_deg"])
rot_ok = False
rot_detail = []
if not bp.empty and set(bp.model.unique()) >= {"LoMa-B", "LoMa-R"}:
    angles = sorted(a for a in bp.level.unique() if float(a) >= ang_ge)
    checks = []
    for ang in angles:
        sub = bp[bp.level == ang]
        mean_b = float(sub[sub.model == "LoMa-B"].inlier_ratio.mean())
        mean_r = float(sub[sub.model == "LoMa-R"].inlier_ratio.mean())
        better = mean_r > mean_b
        checks.append(better)
        rot_detail.append(dict(angle=ang, mean_inlier_B=mean_b, mean_inlier_R=mean_r, R_better=better))
    rot_ok = bool(checks) and all(checks)
else:
    rot_detail = []

if det_block and rot_ok:
    status = "PASS"
elif det_block != rot_ok:
    status = "PARTIAL"
else:
    status = "FAIL"


def md(df, cols, fmt):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |")
    return "\n".join(out)


f4, f2 = "{:.4f}", "{:.2f}"
lines = [
    "# exp1_cpu_match results\n",
    f"**Status: {status}** (latency+determinism {'met' if det_block else 'not met'}; "
    f"rotation LoMa-R>LoMa-B at ≥{ang_ge:g}° {'met' if rot_ok else 'not met'})\n",
    f"Thresholds (fixed before running): mean latency_s_match ≤ {th['latency_s_match_max']} s per seed; "
    f"n_matches identical across seeds 0/1/2; LoMa-R mean inlier_ratio > LoMa-B for every angle ≥ {ang_ge:g}°.\n",
    "## Per seed (LoMa-B, mean over pairs)\n",
    md(
        g,
        ["seed", "mean_latency_s", "max_latency_s", "mean_n_matches", "mean_inlier_ratio", "latency_pass", "determinism_pass"],
        {"mean_latency_s": f2, "max_latency_s": f2, "mean_n_matches": f2, "mean_inlier_ratio": f4},
    ),
    "\n## Per pair × seed\n",
    md(
        per,
        ["seed", "pair", "n_matches", "n_inliers", "inlier_ratio", "latency_s_match"],
        {"inlier_ratio": f4, "latency_s_match": f2},
    ),
    "\n## Determinism (n_matches across seeds)\n",
    md(det_df, ["pair", "n_matches_across_seeds", "identical"], {}),
    "\n## Rotation criterion (mean inlier_ratio over breakpoint images)\n",
]
if rot_detail:
    rdf = pd.DataFrame(rot_detail)
    lines.append(md(rdf, ["angle", "mean_inlier_B", "mean_inlier_R", "R_better"], {"mean_inlier_B": f4, "mean_inlier_R": f4}))
else:
    lines.append("_No rotation breakpoint rows (skipped or incomplete)._\n")

lines += [
    "\n## Runtime\n",
    f"- cold first match: " + ", ".join(f"{k} {v:.1f}s" for k, v in raw.get("cold_s", {}).items()),
    f"- load: " + ", ".join(f"{k} {v:.1f}s" for k, v in raw.get("load_s", {}).items()),
    f"- env: {raw['env']}",
    f"- disclosure: {cfg.get('disclosure', '').strip()}",
]
open(os.path.join(EXP, "results.md"), "w").write("\n".join(lines) + "\n")

# breakpoints md
blines = [
    "# Breakpoints — in-plane rotation of image B (seed 0)\n",
    f"ok ≥ {ok_lvl}, degraded ≥ {deg_lvl}, broken < {deg_lvl} (by MAGSAC inlier_ratio).\n",
    "Story: LoMa-R should retain higher inlier_ratio than LoMa-B once rotation exceeds ~45°.\n",
]
if not bp.empty:
    blines.append(
        md(
            bp.sort_values(["image", "model", "level"]),
            ["image", "model", "level", "n_matches", "n_inliers", "inlier_ratio", "latency_s_match", "verdict"],
            {"inlier_ratio": f4, "latency_s_match": f2},
        )
    )
    blines.append("\n## First angle that leaves 'ok' (per image × model)\n")
    for (im, model), d in bp.groupby(["image", "model"], sort=False):
        d = d.sort_values("level")
        bad = d[d.verdict != "ok"]
        blines.append(
            f"- {im} / {model}: "
            + ("never (all levels ok)" if bad.empty else f"{bad.iloc[0]['level']}° → {bad.iloc[0]['verdict']} (inlier_ratio {bad.iloc[0]['inlier_ratio']:.3f})")
        )
    blines.append("\n## LoMa-R vs LoMa-B crossover\n")
    for im in bp.image.unique():
        for ang in sorted(bp.level.unique()):
            sub = bp[(bp.image == im) & (bp.level == ang)]
            if set(sub.model) >= {"LoMa-B", "LoMa-R"}:
                ib = float(sub[sub.model == "LoMa-B"].inlier_ratio.iloc[0])
                ir = float(sub[sub.model == "LoMa-R"].inlier_ratio.iloc[0])
                blines.append(f"- {im} @ {ang}°: B={ib:.3f}, R={ir:.3f}, Δ(R−B)={ir - ib:+.3f}")
else:
    blines.append("_empty_\n")
open(os.path.join(EXP, "breakpoints.md"), "w").write("\n".join(blines) + "\n")

# plot — publication-friendly
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

ax = axes[0]
x = np.arange(len(g))
ax.bar(x, g.mean_latency_s, color="#4C78A8", edgecolor="black", linewidth=0.6, label="mean / seed")
ax.scatter(per.seed.map({s: i for i, s in enumerate(g.seed)}), per.latency_s_match, color="black", s=18, zorder=3, label="per pair")
ax.axhline(th["latency_s_match_max"], color="#E45756", ls="--", lw=1.5, label="budget")
ax.set_xticks(x, [f"seed {s}" for s in g.seed])
ax.set_ylabel("Latency (s)")
ax.set_title("LoMa-B CPU match latency")
leg = ax.legend(frameon=True, fancybox=False, edgecolor="black")
for t in leg.get_texts():
    t.set_fontweight("bold")

ax = axes[1]
width = 0.35
pairs_u = list(det_df.pair)
for i, s in enumerate(g.seed):
    vals = [int(per[(per.seed == s) & (per.pair == p)].n_matches.iloc[0]) for p in pairs_u]
    ax.bar(np.arange(len(pairs_u)) + (i - 1) * width, vals, width=width, label=f"seed {s}", edgecolor="black", linewidth=0.5)
ax.set_xticks(np.arange(len(pairs_u)), [p.replace("pair_", "") for p in pairs_u], rotation=15)
ax.set_ylabel("Correspondence count")
ax.set_title("Determinism: n_matches by seed")
leg = ax.legend(frameon=True, fancybox=False, edgecolor="black")
for t in leg.get_texts():
    t.set_fontweight("bold")

ax = axes[2]
if not bp.empty:
    for model, color in [("LoMa-B", "#E45756"), ("LoMa-R", "#54A24B")]:
        sub = bp[bp.model == model].groupby("level", as_index=False).inlier_ratio.mean().sort_values("level")
        ax.plot(sub.level, sub.inlier_ratio, "o-", color=color, lw=2, markersize=6, label=model)
    ax.axvline(ang_ge, color="#F58518", ls=":", lw=1.5, label=f"criterion ≥ {ang_ge:g}°")
    ax.axhline(ok_lvl, color="#54A24B", ls="--", alpha=0.5)
    ax.axhline(deg_lvl, color="#E45756", ls="--", alpha=0.5)
ax.set_xlabel("Rotation of image B (deg)")
ax.set_ylabel("Mean MAGSAC inlier ratio")
ax.set_ylim(-0.02, 1.02)
ax.set_title("Breakpoint: rotation (B vs R)")
leg = ax.legend(frameon=True, fancybox=False, edgecolor="black")
for t in leg.get_texts():
    t.set_fontweight("bold")

fig.suptitle(f"loma-matcher exp1_cpu_match — status {status}", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(EXP, "plot.png"), dpi=140)
json.dump(
    dict(
        status=status,
        latency_all_seeds=latency_all,
        determinism=det_ok,
        latency_and_determinism=det_block,
        rotation_R_better_ge_threshold=rot_ok,
        rotation_detail=rot_detail,
    ),
    open(os.path.join(EXP, "status.json"), "w"),
    indent=1,
)
print(status)
print(g.to_string())
print("determinism:", det_ok, "rotation:", rot_ok)
