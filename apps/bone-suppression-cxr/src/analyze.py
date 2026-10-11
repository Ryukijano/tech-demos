"""raw.json -> results.csv/.md, breakpoints.csv/.md, plot.png, status.json (applies config thresholds)."""
import json, os, sys, math
import yaml, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__)); import bsx
EXP = os.path.join(bsx.APP, "experiments/exp1_cpu_stability")
cfg = yaml.safe_load(open(os.path.join(EXP, "config.yaml"))); raw = json.load(open(os.path.join(EXP, "raw.json")))
th = cfg["thresholds"]

rows = []
for s, d in raw["seeds"].items():
    for k, r in d.items():
        rows.append(dict(seed=int(s), image=k, kind=raw["clean"][k]["kind"], ssim_bone=r["ssim_bone"], psnr_bone_db=r["psnr_bone"],
                         max_abs_diff_bone=r["max_abs_diff_bone"], ssim_lung=r["ssim_lung"], t_bone_s=r["t_bone"],
                         t_lung_s=r["t_lung"], latency_s_pipeline=r["latency_s_pipeline"]))
per = pd.DataFrame(rows).sort_values(["seed", "image"]); per.to_csv(os.path.join(EXP, "results_per_image.csv"), index=False)
g = per.groupby("seed").agg(mean_ssim_bone=("ssim_bone", "mean"), min_ssim_bone=("ssim_bone", "min"),
                            mean_ssim_lung=("ssim_lung", "mean"), mean_latency_s=("latency_s_pipeline", "mean"),
                            max_latency_s=("latency_s_pipeline", "max")).reset_index()
g["ssim_pass"] = g.mean_ssim_bone >= th["ssim_bone_min"]; g["latency_pass"] = g.mean_latency_s <= th["latency_s_pipeline_max"]
g.to_csv(os.path.join(EXP, "results.csv"), index=False)
ref = {k: raw["clean"][k]["ref_mae_8bit"] for k in raw["clean"] if "ref_mae_8bit" in raw["clean"][k]}
pre = all(v <= cfg["precondition"]["ref_mae_8bit_max"] for v in ref.values())
a, b = bool(g.ssim_pass.all()), bool(g.latency_pass.all())
status = "FAIL" if not pre else ("PASS" if a and b else ("PARTIAL" if a != b else "FAIL"))

bp = pd.DataFrame(raw["breakpoints"]); bp.to_csv(os.path.join(EXP, "breakpoints.csv"), index=False)
ok, deg = cfg["breakpoints"]["degrade_levels"]["ok"], cfg["breakpoints"]["degrade_levels"]["degraded"]
def lab(v): return "ok" if v >= ok else ("degraded" if v >= deg else "broken")
bp["verdict"] = bp.ssim_bone.map(lab)

def md(df, cols, fmt):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |")
    return "\n".join(out)
f4 = "{:.4f}"; f2 = "{:.2f}"
lines = [f"# exp1_cpu_stability results\n", f"**Status: {status}** (precondition {'met' if pre else 'NOT met'}; "
         f"SSIM criterion {'met' if a else 'not met'} on all seeds; latency criterion {'met' if b else 'not met'} on all seeds)\n",
         f"Thresholds (fixed before running): mean ssim_bone >= {th['ssim_bone_min']}, mean latency_s_pipeline <= {th['latency_s_pipeline_max']} s, per seed.\n",
         "## Per seed (mean over 4 images)\n",
         md(g, ["seed", "mean_ssim_bone", "min_ssim_bone", "mean_ssim_lung", "mean_latency_s", "max_latency_s", "ssim_pass", "latency_pass"],
            {"mean_ssim_bone": f4, "min_ssim_bone": f4, "mean_ssim_lung": f4, "mean_latency_s": f2, "max_latency_s": f2}),
         "\n## Per image\n",
         md(per, ["seed", "image", "ssim_bone", "psnr_bone_db", "ssim_lung", "t_bone_s", "t_lung_s", "latency_s_pipeline"],
            {"ssim_bone": f4, "psnr_bone_db": "{:.1f}", "ssim_lung": f4, "t_bone_s": f2, "t_lung_s": f2, "latency_s_pipeline": f2}),
         "\n## Precondition: reproduce the authors' published bone PNGs\n",
         "| image | mean abs diff (8-bit levels) |\n|---|---|"] + [f"| {k} | {v:.5f} |" for k, v in ref.items()] + [
         "\n## Runtime\n", f"- cold first call: bone {raw['cold_s_bone']:.1f} s, lung {raw['cold_s_lung']:.1f} s (excluded); load: bone {raw['load_s_bone']:.1f} s, lung {raw['load_s_lung']:.1f} s",
         f"- env: {raw['env']}", "- clean-input determinism (max |run1-run2|): " + ", ".join(f"{k} {v['deterministic_max_abs']:.1e}" for k, v in raw["clean"].items())]
open(os.path.join(EXP, "results.md"), "w").write("\n".join(lines) + "\n")

# breakpoints md: where does it first fall below ok / degraded, per family & image
blines = ["# Breakpoints (seed 0, bone model; SSIM vs clean bone output)\n",
          f"ok >= {ok}, degraded >= {deg}, broken < {deg}. Rotation compares with the clean output rotated by the same angle.\n",
          md(bp, ["image", "family", "level", "ssim_bone", "psnr_bone", "bone_mean", "looks_inverted", "verdict"],
             {"ssim_bone": f4, "psnr_bone": "{:.1f}", "bone_mean": f4}), "\n## First level that leaves 'ok'\n"]
for (im, fa), d in bp.groupby(["image", "family"], sort=False):
    bad = d[d.verdict != "ok"]
    blines.append(f"- {im} / {fa}: " + ("never (all levels ok)" if bad.empty else f"{bad.iloc[0]['level']} -> {bad.iloc[0]['verdict']} (SSIM {bad.iloc[0]['ssim_bone']:.3f})"))
open(os.path.join(EXP, "breakpoints.md"), "w").write("\n".join(blines) + "\n")

# plot
fams = [f for f in cfg["breakpoints"]["families"] if f != "inverted"]
fig, ax = plt.subplots(2, 4, figsize=(17, 8))
ax = ax.ravel()
x = np.arange(len(g)); ax[0].bar(x, g.mean_ssim_bone, color="#7ee0b0"); ax[0].axhline(th["ssim_bone_min"], color="r", ls="--", label="threshold")
ax[0].scatter(per.seed, per.ssim_bone, color="k", s=12, zorder=3, label="per image")
ax[0].set_xticks(x, [f"seed {s}" for s in g.seed]); ax[0].set_ylim(min(0.9, per.ssim_bone.min() - 0.01), 1.0); ax[0].set_title(f"bone SSIM, noise σ={cfg['perturbation']['sigma']}"); ax[0].legend(fontsize=8)
ax[1].bar(x, g.mean_latency_s, color="#8ab4ff"); ax[1].axhline(th["latency_s_pipeline_max"], color="r", ls="--", label="budget")
ax[1].scatter(per.seed, per.latency_s_pipeline, color="k", s=12, zorder=3); ax[1].set_xticks(x, [f"seed {s}" for s in g.seed]); ax[1].set_title("CPU latency / image (bone+lung), s"); ax[1].legend(fontsize=8)
for i, f in enumerate(fams):
    A = ax[i + 2]
    for im, d in bp[bp.family == f].groupby("image"):
        A.plot([str(v) for v in d.level], d.ssim_bone, "o-", label=im)
    A.axhline(ok, color="g", ls=":"); A.axhline(deg, color="r", ls=":"); A.set_title(f"breakpoint: {f}"); A.set_ylim(0, 1.02); A.legend(fontsize=7)
fig.suptitle(f"bone-suppression-cxr exp1 — status {status}  (inverted input SSIM: " + ", ".join(f"{r.image} {r.ssim_bone:.2f}" for r in bp[bp.family=='inverted'].itertuples()) + ")")
fig.tight_layout(); fig.savefig(os.path.join(EXP, "plot.png"), dpi=110)
json.dump(dict(status=status, precondition=pre, ssim_all_seeds=a, latency_all_seeds=b), open(os.path.join(EXP, "status.json"), "w"), indent=1)
print(status); print(g.to_string())
