# SPDX-License-Identifier: Apache-2.0
"""Aggregate raw.json, judge the pre-registered bars, emit tables + plots + status."""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_loop_knob"
CFG = yaml.safe_load((EXP / "config.yaml").read_text())
RAW = json.loads((EXP / "raw.json").read_text())
TH = CFG["thresholds"]
K_MAIN = CFG["k_grid_main"]
SEEDS = CFG["seeds"]

rows = RAW["rows"]
all_k = sorted({r["k"] for r in rows})


def cells(k, metric):
    return [r[metric] for r in rows if r["k"] == k]


def seed_means(k, metric):
    return [float(np.mean([r[metric] for r in rows if r["k"] == k and r["seed"] == s])) for s in SEEDS]


agg = {}
for k in all_k:
    agg[k] = {}
    for m in ("Auc_30", "Auc_15", "R_error", "T_error", "AbsRel", "Delta1", "latency_s"):
        agg[k][m] = float(np.mean(cells(k, m)))
        agg[k][m + "_seedstd"] = float(np.std(seed_means(k, m), ddof=1))

# ---------------- pre-registered checks ----------------
floor_ok = agg[16]["Auc_30"] >= TH["floor_auc30_at_k16"]
gain = agg[16]["Auc_30"] - agg[8]["Auc_30"]
h1a = gain >= TH["h1a_auc30_gain_8_to_16"]
tol = TH["h1b_monotone_tolerance"]
h1b = all(agg[K_MAIN[i + 1]]["Auc_30"] >= agg[K_MAIN[i]]["Auc_30"] - tol for i in range(len(K_MAIN) - 1))
h1c = agg[16]["AbsRel"] <= agg[8]["AbsRel"]

lat_k = np.array(all_k, dtype=float)
lat_mean = np.array([agg[k]["latency_s"] for k in all_k])
slope, intercept = np.polyfit(lat_k, lat_mean, 1)
pred = slope * lat_k + intercept
r2 = 1.0 - np.sum((lat_mean - pred) ** 2) / np.sum((lat_mean - lat_mean.mean()) ** 2)
h2 = (r2 >= TH["h2_latency_r2_min"]) and slope > 0

if not floor_ok or (not h1a and not h1c):
    status = "FAIL"
elif floor_ok and h1a and h1b and h1c and h2:
    status = "PASS"
else:
    status = "PARTIAL"

status_obj = {
    "status": status,
    "config_sha256_at_run": RAW["config_sha256"],
    "config_sha256_now": hashlib.sha256((EXP / "config.yaml").read_bytes()).hexdigest(),
    "checks": {
        "floor_auc30_at_k16": {"bar": TH["floor_auc30_at_k16"], "value": round(agg[16]["Auc_30"], 4), "ok": bool(floor_ok)},
        "h1a_auc30_gain_8_to_16": {"bar": TH["h1a_auc30_gain_8_to_16"], "value": round(gain, 4), "ok": bool(h1a)},
        "h1b_monotone_main_grid": {"tolerance": tol, "ok": bool(h1b),
                                   "series": {str(k): round(agg[k]["Auc_30"], 4) for k in K_MAIN}},
        "h1c_absrel_no_worse": {"k8": round(agg[8]["AbsRel"], 4), "k16": round(agg[16]["AbsRel"], 4), "ok": bool(h1c)},
        "h2_latency_linear": {"bar_r2": TH["h2_latency_r2_min"], "r2": round(float(r2), 4),
                              "slope_s_per_step": round(float(slope), 4),
                              "intercept_s": round(float(intercept), 4), "ok": bool(h2)},
    },
}
(EXP / "status.json").write_text(json.dumps(status_obj, indent=2))
print(json.dumps(status_obj, indent=2))

# ---------------- tables ----------------
lines = ["| K | grid | pose AUC@30 | R err (deg) | T err (deg) | depth AbsRel | latency (s) |",
         "|---:|---|---:|---:|---:|---:|---:|"]
csv = ["k,grid,auc30_mean,auc30_seedstd,r_error,t_error,absrel,absrel_seedstd,latency_mean,latency_seedstd"]
for k in all_k:
    a = agg[k]
    grid = "main" if k in K_MAIN else "break"
    lines.append(
        f"| {k} | {grid} | {a['Auc_30']:.3f} ± {a['Auc_30_seedstd']:.3f} | {a['R_error']:.2f} | "
        f"{a['T_error']:.2f} | {a['AbsRel']:.4f} ± {a['AbsRel_seedstd']:.4f} | {a['latency_s']:.2f} |"
    )
    csv.append(f"{k},{grid},{a['Auc_30']:.4f},{a['Auc_30_seedstd']:.4f},{a['R_error']:.3f},{a['T_error']:.3f},"
               f"{a['AbsRel']:.5f},{a['AbsRel_seedstd']:.5f},{a['latency_s']:.3f},{a['latency_s_seedstd']:.3f}")
(EXP / "results.md").write_text("\n".join(lines) + "\n")
(EXP / "results.csv").write_text("\n".join(csv) + "\n")

bp_lines = ["| label | K | img | views | pose AUC@30 | R err | AbsRel | latency (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
for r in RAW["breakpoint_rows"]:
    if "error" in r:
        bp_lines.append(f"| {r['label']} | {r['k']} | {r['img_size']} | {r['n_views']} | CRASH: `{r['error'][:60]}` | | | |")
    else:
        bp_lines.append(f"| {r['label']} | {r['k']} | {r['img_size']} | {r['n_views']} | {r['Auc_30']:.3f} | "
                        f"{r['R_error']:.2f} | {r['AbsRel']:.4f} | {r['latency_s']:.2f} |")
(EXP / "breakpoints.md").write_text("\n".join(bp_lines) + "\n")

# ---------------- plots ----------------
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 13, "axes.labelweight": "bold",
    "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
ks = np.array(all_k)
auc = np.array([agg[k]["Auc_30"] for k in all_k])
auc_sd = np.array([agg[k]["Auc_30_seedstd"] for k in all_k])
absrel = np.array([agg[k]["AbsRel"] for k in all_k])
absrel_sd = np.array([agg[k]["AbsRel_seedstd"] for k in all_k])
lat_sd = np.array([agg[k]["latency_s_seedstd"] for k in all_k])

for ax in axes[:2]:
    ax.axvspan(8, 16, color="#2b8a3e", alpha=0.10, lw=0)

axes[0].errorbar(ks, auc, yerr=auc_sd, marker="o", lw=2.2, capsize=4, color="#1f72b1",
                 label="pose AUC@30 (mean ± seed std)")
axes[0].axhline(TH["floor_auc30_at_k16"], ls="--", lw=1.6, color="#c0392b",
                label=f"pre-registered floor at K=16 ({TH['floor_auc30_at_k16']})")
axes[0].set_xlabel("Loop count K")
axes[0].set_ylabel("Pose AUC@30 ↑")
axes[0].set_title("Camera-pose quality vs K")
axes[0].legend(frameon=False, prop={"weight": "bold"})

axes[1].errorbar(ks, absrel, yerr=absrel_sd, marker="s", lw=2.2, capsize=4, color="#8e44ad",
                 label="depth AbsRel (median-aligned)")
axes[1].set_xlabel("Loop count K")
axes[1].set_ylabel("Depth AbsRel ↓")
axes[1].set_title("Depth quality vs K")
axes[1].legend(frameon=False, prop={"weight": "bold"})

axes[2].errorbar(ks, lat_mean, yerr=lat_sd, marker="o", lw=0, elinewidth=1.5, capsize=4,
                 color="#1f72b1", label="measured (mean ± seed std)")
xfit = np.linspace(min(all_k), max(all_k), 100)
axes[2].plot(xfit, slope * xfit + intercept, lw=2.2, color="#e67e22",
             label=f"linear fit: {slope:.3f}·K + {intercept:.2f}  (R²={r2:.3f})")
axes[2].set_xlabel("Loop count K")
axes[2].set_ylabel("CPU latency per scene (s)")
axes[2].set_title("Latency vs K (6 views, 252², 4 threads)")
axes[2].legend(frameon=False, prop={"weight": "bold"})

green = "trained range K∈[8,16] shaded"
fig.suptitle(f"DVLT loop-count dial on CPU — status {status}   ({green})", fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(EXP / "plot.png", bbox_inches="tight")
fig.savefig(EXP / "plot.pdf", bbox_inches="tight")

# viewer summary
summary = {"status": status, "agg": {str(k): agg[k] for k in all_k}, "k_main": K_MAIN,
           "checks": status_obj["checks"], "breakpoints": RAW["breakpoint_rows"],
           "machine": RAW["machine"], "wall_time_s": RAW["wall_time_s"]}
(APP / "viewer" / "summary.json").write_text(json.dumps(summary, indent=1))


# ---------------- viewer camera alignment (display only, not a metric) ----------------
def umeyama(src, dst):
    """Similarity transform (s, R, t) minimising ||dst - (s R src + t)||^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (sc**2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


vd_path = APP / "viewer" / "viewer_data.json"
vd = json.loads(vd_path.read_text())
for k, entry in list(vd.items()):
    if not isinstance(entry, dict) or "pred_c2w" not in entry:
        continue
    pred = np.array(entry["pred_c2w"])  # (S, 3, 4)
    gt = np.array(entry["gt_c2w"])
    s, R, t = umeyama(pred[:, :, 3], gt[:, :, 3])
    centers = (s * (R @ pred[:, :, 3].T)).T + t
    fwd = (R @ pred[:, :, 2].T).T  # camera +z in world after alignment
    entry["pred_centers_aligned"] = centers.round(4).tolist()
    entry["pred_forward_aligned"] = fwd.round(4).tolist()
    entry["gt_centers"] = gt[:, :, 3].round(4).tolist()
    entry["gt_forward"] = gt[:, :, 2].round(4).tolist()
    del entry["pred_c2w"], entry["gt_c2w"]
vd_path.write_text(json.dumps(vd))
print("analysis written")
