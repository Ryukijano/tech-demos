"""Turn raw.json into results.csv / results.md / status.json / plot.png and
breakpoints.csv / breakpoints.md, applying the pre-registered pass rule only."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_2d"

plt.rcParams.update({
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "figure.dpi": 140,
})


def main() -> None:
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    raw = json.loads((EXP / "raw.json").read_text())

    seeds = sorted(raw["seeds"].keys(), key=int)
    k_values = list(cfg["sampling"]["K_values"])
    best_ks = [k for k in k_values if k > 0]

    # ---------------- per-seed table + pass rule ----------------
    rows = []
    per_seed_pass = {}
    for s in seeds:
        info = raw["seeds"][s]
        by_k = {r["K"]: r for r in info["rows"]}
        base = by_k[0]["mmd2"]
        best_k = min(best_ks, key=lambda k: by_k[k]["mmd2"])
        best = by_k[best_k]["mmd2"]
        per_seed_pass[s] = best < base
        for k in k_values:
            r = by_k[k]
            rows.append({
                "seed": s, "K": k, "steps": r["steps"], "nfe": r["nfe"],
                "mmd2": r["mmd2"], "sliced_w1": r["sliced_w1"],
                "nearest_mode_mse": r["nearest_mode_mse"],
                "mode_coverage": r["mode_coverage"],
                "sample_wall_s": round(r["sample_wall_s"], 2),
                "train_wall_s": round(info["train_wall_s"], 1),
                "is_best_K": k == best_k,
            })

    with (EXP / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_pass = sum(per_seed_pass.values())
    # Runaway-cost guard from the pre-registered pass rule (5x the report-only budgets).
    t_max = float(cfg["thresholds"]["train_wall_s_max"])
    s_max = float(cfg["thresholds"]["sample_wall_s_max"])
    runaway = any(raw["seeds"][s]["train_wall_s"] > 5 * t_max for s in seeds) or any(
        r["sample_wall_s"] > 5 * s_max
        for s in seeds for r in raw["seeds"][s]["rows"] if r["K"] == 0
    )
    if runaway:
        status = "FAIL"
    elif n_pass == len(seeds):
        status = "PASS"
    elif n_pass == len(seeds) - 1:
        status = "PARTIAL"
    else:
        status = "FAIL"

    (EXP / "status.json").write_text(json.dumps({
        "status": status,
        "per_seed_pass": per_seed_pass,
        "n_seeds_pass": n_pass,
        "runaway_cost_guard_tripped": runaway,
        "gamma": raw["gamma"],
    }, indent=2))

    # ---------------- results.md ----------------
    md = ["# exp1_cpu_2d results", "",
          f"Status (pre-registered rule): **{status}**", "",
          f"RBF kernel gamma (median heuristic on eval data): `{raw['gamma']:.4f}`", "",
          "| seed | K | steps | NFE | MMD² ×10³ | sliced W1 | nearest-mode MSE | coverage | sample s | best K? |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['seed']} | {r['K']} | {r['steps']} | {r['nfe']} | {r['mmd2']*1e3:.3f} "
            f"| {r['sliced_w1']:.4f} | {r['nearest_mode_mse']:.4f} | {r['mode_coverage']:.2f} "
            f"| {r['sample_wall_s']} | {'**yes**' if r['is_best_K'] else ''} |")
    md += ["", "| seed | train s | MMD²(K=0) ×10³ | best MMD²(K>0) ×10³ | K>0 wins? |",
           "|---|---|---|---|---|"]
    for s in seeds:
        by_k = {r["K"]: r for r in raw["seeds"][s]["rows"]}
        best_k = min(best_ks, key=lambda k: by_k[k]["mmd2"])
        md.append(f"| {s} | {raw['seeds'][s]['train_wall_s']:.1f} | {by_k[0]['mmd2']*1e3:.3f} "
                  f"| {by_k[best_k]['mmd2']*1e3:.3f} (K={best_k}) "
                  f"| {'yes' if per_seed_pass[s] else '**no**'} |")
    md += ["", f"Wall-time secondaries (report-only): train ≤ {t_max:.0f}s, "
           f"2k-sample K=0 pass ≤ {s_max:.0f}s. Runaway guard tripped: {runaway}.", ""]
    (EXP / "results.md").write_text("\n".join(md))

    # ---------------- plot.png ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for i, s in enumerate(seeds):
        by_k = {r["K"]: r for r in raw["seeds"][s]["rows"]}
        ys = [by_k[k]["mmd2"] * 1e3 for k in k_values]
        axes[0].plot(k_values, ys, "o-", color=colors[i], label=f"seed {s}", lw=2)
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_xticks(k_values)
    axes[0].set_xticklabels([str(k) for k in k_values])
    axes[0].set_xlabel("corrections per step K (matched NFE = 120)")
    axes[0].set_ylabel("unbiased MMD² ×10³ (RBF)")
    axes[0].set_title("Primary metric: MMD² vs K")
    axes[0].legend(prop={"weight": "bold"})
    axes[0].grid(alpha=0.3)

    for i, s in enumerate(seeds):
        by_k = {r["K"]: r for r in raw["seeds"][s]["rows"]}
        ys = [by_k[k]["nearest_mode_mse"] for k in k_values]
        axes[1].plot(k_values, ys, "s--", color=colors[i], label=f"seed {s}", lw=2)
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_xticks(k_values)
    axes[1].set_xticklabels([str(k) for k in k_values])
    axes[1].set_xlabel("corrections per step K (matched NFE = 120)")
    axes[1].set_ylabel("mean nearest-mode MSE")
    axes[1].set_title("Report-only: nearest-mode MSE vs K")
    axes[1].legend(prop={"weight": "bold"})
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"correction-training-2d · exp1_cpu_2d · status {status}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(EXP / "plot.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------- breakpoints ----------------
    if raw.get("breakpoints"):
        bp = raw["breakpoints"]
        bp_rows = []
        for block, entries in bp.items():
            for r in entries:
                bp_rows.append({
                    "block": block, "setting": r.get("setting", ""),
                    "K": r["K"], "steps": r["steps"], "nfe": r["nfe"],
                    "nfe_budget": r["nfe_budget"], "mmd2": r["mmd2"],
                    "sliced_w1": r["sliced_w1"],
                    "nearest_mode_mse": r["nearest_mode_mse"],
                    "mode_coverage": r["mode_coverage"],
                })
        with (EXP / "breakpoints.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bp_rows[0].keys()))
            w.writeheader()
            w.writerows(bp_rows)

        md = ["# Breakpoints (seed 0)", "",
              "Pushing the method to extremes. All numbers are MMD² ×10³ vs the fixed eval set",
              "for that data setting (lower is better).", "",
              "## K to the extreme (matched NFE 120)", "",
              "| K | steps | NFE | MMD² ×10³ | coverage |", "|---|---|---|---|---|"]
        for r in bp["k_extreme"]:
            md.append(f"| {r['K']} | {r['steps']} | {r['nfe']} | {r['mmd2']*1e3:.3f} "
                      f"| {r['mode_coverage']:.2f} |")
        md += ["", "## NFE budget sweep (K=0 vs K=4)", "",
               "| budget | K | steps | NFE | MMD² ×10³ |", "|---|---|---|---|---|"]
        for r in bp["nfe_sweep"]:
            md.append(f"| {r['nfe_budget']} | {r['K']} | {r['steps']} | {r['nfe']} "
                      f"| {r['mmd2']*1e3:.3f} |")
        md += ["", "## Mixture difficulty (retrained per setting, NFE 120)", "",
               "| setting | K | MMD² ×10³ | coverage |", "|---|---|---|---|"]
        for r in bp["difficulty"]:
            md.append(f"| {r['setting']} | {r['K']} | {r['mmd2']*1e3:.3f} "
                      f"| {r['mode_coverage']:.2f} |")
        md.append("")
        (EXP / "breakpoints.md").write_text("\n".join(md))

    print(f"status: {status} (seeds passing: {n_pass}/{len(seeds)})")


if __name__ == "__main__":
    main()
