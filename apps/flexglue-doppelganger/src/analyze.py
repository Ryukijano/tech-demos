"""raw.json -> results table, breakpoints, plot, status. Thresholds come from config.yaml."""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
import fgx

EXP = os.path.join(fgx.APP, "experiments/exp1_cpu_covisibility")
cfg = yaml.safe_load(open(os.path.join(EXP, "config.yaml"), encoding="utf-8"))
raw = json.load(open(os.path.join(EXP, "raw.json"), encoding="utf-8"))
th = cfg["thresholds"]


def _auroc_triple(pairs: list[dict]) -> tuple[float, float, float]:
    labels = np.array([p["label"] for p in pairs], dtype=np.int64)
    cov = fgx.auroc(np.array([p["covisibility"] for p in pairs], dtype=np.float64), labels)
    fg = fgx.auroc(np.array([p["n_matches"] for p in pairs], dtype=np.float64), labels)
    sift = fgx.auroc(np.array([p["n_sift"] for p in pairs], dtype=np.float64), labels)
    return cov, fg, sift


def _mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(arr.mean())


seed_rows = []
per_lines = ["seed,pair,scene,label,covisibility,n_matches,n_inliers,n_sift,n_sift_inliers,match_s,uncached_s,n_kpts0,n_kpts1"]
for seed, payload in raw["seeds"].items():
    pairs = payload["pairs"]
    cov, fg, sift = _auroc_triple(pairs)
    pos = [p for p in pairs if p["label"] == 1]
    neg = [p for p in pairs if p["label"] == 0]
    seed_rows.append(
        dict(
            seed=int(seed),
            n_pos=len(pos),
            n_neg=len(neg),
            scenes=" ".join(payload["scenes"]),
            auroc_covis=cov,
            auroc_fg_matches=fg,
            auroc_sift=sift,
            margin=cov - fg,
            mean_covis_pos=_mean([p["covisibility"] for p in pos]),
            mean_covis_neg=_mean([p["covisibility"] for p in neg]),
            mean_matches_pos=_mean([p["n_matches"] for p in pos]),
            mean_matches_neg=_mean([p["n_matches"] for p in neg]),
            mean_inliers_pos=_mean([p["n_inliers"] for p in pos]),
            mean_inliers_neg=_mean([p["n_inliers"] for p in neg]),
            mean_sift_pos=_mean([p["n_sift"] for p in pos]),
            mean_sift_neg=_mean([p["n_sift"] for p in neg]),
            mean_match_s=_mean([p["match_s"] for p in pairs]),
            mean_uncached_s=_mean([p["uncached_s"] for p in pairs]),
        )
    )
    for p in pairs:
        per_lines.append(
            ",".join(
                str(x)
                for x in (
                    seed,
                    p["id"],
                    p["scene"].replace(",", " "),
                    p["label"],
                    f"{p['covisibility']:.6f}",
                    p["n_matches"],
                    p["n_inliers"],
                    p["n_sift"],
                    p["n_sift_inliers"],
                    f"{p['match_s']:.4f}",
                    f"{p['uncached_s']:.4f}",
                    p["n_kpts0"],
                    p["n_kpts1"],
                )
            )
        )

with open(os.path.join(EXP, "results_per_pair.csv"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(per_lines) + "\n")


def col(name: str) -> np.ndarray:
    return np.array([row[name] for row in seed_rows], dtype=np.float64)


def mean_std(name: str) -> tuple[float, float]:
    arr = col(name)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std


mean_c, std_c = mean_std("auroc_covis")
mean_f, std_f = mean_std("auroc_fg_matches")
mean_s, std_s = mean_std("auroc_sift")
mean_margin, std_margin = mean_std("margin")
mean_rt, std_rt = mean_std("mean_uncached_s")
mean_match_s, std_match_s = mean_std("mean_match_s")
mean_in_pos, _ = mean_std("mean_inliers_pos")
mean_in_neg, _ = mean_std("mean_inliers_neg")

clause_auroc = bool(np.isfinite(mean_c) and mean_c >= float(th["auroc_covis_min"]))
clause_margin = bool(np.isfinite(mean_margin) and mean_margin >= float(th["auroc_margin_vs_matchcount"]))
if clause_auroc and clause_margin:
    status = "PASS"
elif np.isfinite(mean_c) and mean_c >= float(th["auroc_covis_partial_min"]) and (clause_auroc != clause_margin):
    status = "PARTIAL"
else:
    status = "FAIL"

header = [
    "seed",
    "n_pos",
    "n_neg",
    "auroc_covis",
    "auroc_fg_matches",
    "auroc_sift",
    "margin",
    "mean_covis_pos",
    "mean_covis_neg",
    "mean_inliers_pos",
    "mean_inliers_neg",
    "mean_sift_pos",
    "mean_sift_neg",
    "mean_match_s",
    "mean_uncached_s",
]
with open(os.path.join(EXP, "results.csv"), "w", encoding="utf-8") as handle:
    handle.write(",".join(header) + "\n")
    for row in seed_rows:
        handle.write(",".join(str(row[k]) if not isinstance(row[k], float) else f"{row[k]:.6f}" for k in header) + "\n")


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


md = []
md.append("# Exp1 results — FlexGlue covisibility on CPU")
md.append("")
md.append(f"**Status: {status}**")
md.append("")
md.append("Mean ± sample std (ddof=1) across seeds 0/1/2.")
md.append("")
md.append("| metric | mean ± std | bar |")
md.append("|---|---|---|")
md.append(f"| AUROC covisibility | {fmt(mean_c, std_c)} | ≥ {th['auroc_covis_min']} ({'met' if clause_auroc else 'not met'}) |")
md.append(f"| AUROC FlexGlue match count | {fmt(mean_f, std_f)} | baseline |")
md.append(f"| AUROC SIFT ratio count | {fmt(mean_s, std_s)} | report-only |")
md.append(f"| margin (covis − match count) | {fmt(mean_margin, std_margin)} | ≥ {th['auroc_margin_vs_matchcount']} ({'met' if clause_margin else 'not met'}) |")
md.append(f"| uncached seconds / pair | {fmt(mean_rt, std_rt)} | report-only |")
md.append(f"| matcher seconds / pair | {fmt(mean_match_s, std_match_s)} | report-only |")
md.append(f"| MAGSAC inliers pos / neg | {mean_in_pos:.1f} / {mean_in_neg:.1f} | report-only |")
md.append("")
md.append("| seed | AUROC covis | AUROC FG matches | AUROC SIFT | margin | covis pos | covis neg | inliers pos | inliers neg | uncached s |")
md.append("|---|---|---|---|---|---|---|---|---|---|")
for row in seed_rows:
    md.append(
        f"| {row['seed']} | {row['auroc_covis']:.4f} | {row['auroc_fg_matches']:.4f} | {row['auroc_sift']:.4f} | "
        f"{row['margin']:.4f} | {row['mean_covis_pos']:.3f} | {row['mean_covis_neg']:.3f} | "
        f"{row['mean_inliers_pos']:.1f} | {row['mean_inliers_neg']:.1f} | {row['mean_uncached_s']:.2f} |"
    )
md.append("")
md.append("![plot](plot.png)")
md.append("")
with open(os.path.join(EXP, "results.md"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(md) + "\n")

# Breakpoints
bp = raw.get("breakpoints", [])
bp_header = "kind,level,id,label,covisibility,n_matches,n_inliers,n_sift,match_s,describe_s,uncached_s,rss_mb,hwm_mb"
with open(os.path.join(EXP, "breakpoints.csv"), "w", encoding="utf-8") as handle:
    handle.write(bp_header + "\n")
    for rec in bp:
        handle.write(
            ",".join(
                str(rec.get(k, ""))
                if not isinstance(rec.get(k), float)
                else f"{rec[k]:.4f}"
                for k in bp_header.split(",")
            )
            + "\n"
        )

ok_bar = float(cfg["breakpoints"]["positive_score_labels"]["ok"])
deg_bar = float(cfg["breakpoints"]["positive_score_labels"]["degraded"])


def score_label(value: float) -> str:
    if not np.isfinite(value):
        return "broken"
    if value >= ok_bar:
        return "ok"
    if value >= deg_bar:
        return "degraded"
    return "broken"


bp_md = ["# Breakpoints", "", "Positive-pair covisibility labels use the paper threshold t=0.9 (ok) and 0.5 (degraded). Negatives are reported raw.", ""]
bp_md.append("| kind | level | pair | label | covisibility | tag | FG matches | inliers | SIFT | match s | RSS MB |")
bp_md.append("|---|---|---|---|---|---|---|---|---|---|---|")
for rec in bp:
    tag = score_label(float(rec["covisibility"])) if int(rec["label"]) == 1 else ""
    bp_md.append(
        f"| {rec['kind']} | {rec['level']} | {rec['id']} | {rec['label']} | {float(rec['covisibility']):.3f} | {tag} | "
        f"{rec['n_matches']} | {rec['n_inliers']} | {rec['n_sift']} | {float(rec['match_s']):.2f} | {float(rec['rss_mb']):.0f} |"
    )
bp_md.append("")
with open(os.path.join(EXP, "breakpoints.md"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(bp_md) + "\n")

status_doc = dict(
    status=status,
    mean_auroc_covis=mean_c,
    std_auroc_covis=std_c,
    mean_auroc_fg_matches=mean_f,
    std_auroc_fg_matches=std_f,
    mean_auroc_sift=mean_s,
    std_auroc_sift=std_s,
    mean_margin_vs_matchcount=mean_margin,
    std_margin_vs_matchcount=std_margin,
    clause_auroc=clause_auroc,
    clause_margin=clause_margin,
    mean_uncached_s=mean_rt,
    std_uncached_s=std_rt,
    mean_match_s=mean_match_s,
    mean_inliers_pos=mean_in_pos,
    mean_inliers_neg=mean_in_neg,
    auroc_covis_min=float(th["auroc_covis_min"]),
    auroc_margin_vs_matchcount=float(th["auroc_margin_vs_matchcount"]),
    auroc_covis_partial_min=float(th["auroc_covis_partial_min"]),
)
with open(os.path.join(EXP, "status.json"), "w", encoding="utf-8") as handle:
    json.dump(status_doc, handle, indent=1)
print("status", status, fmt(mean_c, std_c), "margin", fmt(mean_margin, std_margin))

# Provenance from the manifest (written before matching).
manifest_path = os.path.join(EXP, "pool_manifest.json")
prov = ["# Image provenance", "", "Selected by the rule in `config.yaml` before any matcher call. Images are not committed.", ""]
if os.path.isfile(manifest_path):
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    prov.append("| pair | scene | label | file | author | license | commons |")
    prov.append("|---|---|---|---|---|---|---|")
    for item in manifest["pairs"]:
        for name, author, lic, url in zip(item["files"], item["authors"], item["licenses"], item["urls"]):
            prov.append(f"| {item['id']} | {item['scene']} | {item['label']} | {name} | {author} | {lic} | {url} |")
    prov.append("")
    prov.append(f"Skipped scenes / pairs during download: {len(manifest.get('skips', []))}.")
    prov.append("")
prov.append("Upstream anchor `berlin1.jpg` / `berlin2.jpg` is downloaded from the FlexGlue repo (MIT tree) and is excluded from the AUROC.")
prov.append("")
with open(os.path.join(EXP, "provenance.md"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(prov) + "\n")

# Plot
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Liberation Serif"],
        "axes.labelsize": 9,
        "axes.labelweight": "bold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.2), dpi=300)
methods = ["Covisibility", "FG matches", "SIFT ratio"]
means = [mean_c, mean_f, mean_s]
stds = [std_c, std_f, std_s]
colors = ["#0072B2", "#E69F00", "#009E73"]
x = np.arange(len(methods))
axes[0].bar(x, means, yerr=stds, color=colors, width=0.72, capsize=3, error_kw={"elinewidth": 0.8, "capthick": 0.8})
axes[0].axhline(float(th["auroc_covis_min"]), color="#D55E00", ls="--", lw=1.0, label=f"pass bar {th['auroc_covis_min']}")
axes[0].axhline(float(th["auroc_covis_partial_min"]), color="#999999", ls=":", lw=1.0, label=f"partial floor {th['auroc_covis_partial_min']}")
axes[0].set_xticks(x)
axes[0].set_xticklabels(methods)
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("AUROC")
axes[0].set_xlabel("Score")
axes[0].legend(prop={"weight": "bold", "size": 7}, frameon=True, loc="lower right")
axes[0].set_title("Pair separation", fontsize=10, fontweight="bold")

seed_colors = {0: "#0072B2", 1: "#E69F00", 2: "#009E73"}
for seed, payload in raw["seeds"].items():
    rng = np.random.default_rng(int(seed) + 17)
    for pair in payload["pairs"]:
        jitter = float(rng.uniform(-0.08, 0.08))
        axes[1].scatter(
            int(pair["label"]) + jitter,
            pair["covisibility"],
            s=18,
            color=seed_colors[int(seed)],
            zorder=3,
            label=f"seed {seed}" if pair is payload["pairs"][0] else None,
        )
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Doppelganger", "Covisible"])
axes[1].set_xlim(-0.4, 1.4)
axes[1].set_ylim(-0.02, 1.02)
axes[1].set_ylabel("Covisibility score")
axes[1].set_xlabel("Doppelgangers label")
axes[1].axhline(0.9, color="#D55E00", ls="--", lw=0.8)
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles, labels, prop={"weight": "bold", "size": 7}, frameon=True, loc="best")
axes[1].set_title("Scores by label", fontsize=10, fontweight="bold")
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(EXP, "plot.png"), bbox_inches="tight")
fig.savefig(os.path.join(EXP, "plot.pdf"), bbox_inches="tight")
print("wrote plot", flush=True)
