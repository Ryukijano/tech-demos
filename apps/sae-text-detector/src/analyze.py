"""Turn raw.json into results / status / plot / breakpoints tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_auditable"

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
    thr = cfg["thresholds"]

    rows = []
    gaps = []
    a_s = []
    a_d = []
    per_seed_ok = {}
    for s in seeds:
        r = raw["seeds"][s]
        gap = r["auroc_gap"]
        ok = (
            gap <= float(thr["max_auroc_gap"])
            and r["auroc_tau_sparse"] >= float(thr["min_auroc_sparse"])
        )
        per_seed_ok[s] = bool(ok)
        gaps.append(gap)
        a_s.append(r["auroc_tau_sparse"])
        a_d.append(r["auroc_tau_dense"])
        rows.append({
            "seed": s,
            "auroc_tau_0.85": round(r["auroc_tau_sparse"], 4),
            "auroc_tau_0.00": round(r["auroc_tau_dense"], 4),
            "auroc_gap": round(gap, 4),
            "accuracy_tau_0.85": round(r["accuracy_tau_sparse"], 4),
            "mean_n_active_0.85": round(r["mean_n_active_tau_sparse"], 2),
            "mean_n_active_0.00": round(r["mean_n_active_tau_dense"], 2),
            "extract_wall_s": (
                None if r.get("extract_wall_s") is None else round(r["extract_wall_s"], 2)
            ),
            "n": r["n"],
            "seed_pass": ok,
        })

    with (EXP / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    gap_mean = float(np.mean(gaps))
    gap_std = float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0
    s_mean = float(np.mean(a_s))
    s_std = float(np.std(a_s, ddof=1)) if len(a_s) > 1 else 0.0
    d_mean = float(np.mean(a_d))
    d_std = float(np.std(a_d, ddof=1)) if len(a_d) > 1 else 0.0
    every = all(per_seed_ok.values())
    mean_ok = (
        gap_mean <= float(thr["max_auroc_gap"])
        and s_mean >= float(thr["min_auroc_sparse"])
    )

    if every:
        status = "PASS"
    elif mean_ok:
        status = "PARTIAL"
    else:
        status = "FAIL"

    status_obj = {
        "status": status,
        "auroc_sparse_mean": s_mean,
        "auroc_sparse_std": s_std,
        "auroc_dense_mean": d_mean,
        "auroc_dense_std": d_std,
        "auroc_gap_mean": gap_mean,
        "auroc_gap_std": gap_std,
        "every_seed_pass": every,
        "mean_pass": mean_ok,
        "per_seed_pass": per_seed_ok,
        "max_auroc_gap": thr["max_auroc_gap"],
        "min_auroc_sparse": thr["min_auroc_sparse"],
        "pass_condition": thr["pass_condition"],
        "status_rule": thr["status_rule"],
        "load_wall_s": raw.get("load_wall_s"),
    }
    (EXP / "status.json").write_text(json.dumps(status_obj, indent=2))

    lines = [
        f"# exp1_cpu_auditable results — **{status}**",
        "",
        "Primary metric: `auroc_gap = AUROC(tau=0.00) − AUROC(tau=0.85)` "
        f"on {rows[0]['n']}-example balanced subsamples "
        f"({cfg['data']['dataset']}).",
        "",
        f"- Mean AUROC τ=0.85: **{s_mean:.4f} ± {s_std:.4f}**",
        f"- Mean AUROC τ=0.00: **{d_mean:.4f} ± {d_std:.4f}**",
        f"- Mean gap: **{gap_mean:.4f} ± {gap_std:.4f}** "
        f"(threshold ≤ {thr['max_auroc_gap']})",
        f"- Every-seed pass: **{every}**",
        "",
        "| seed | AUROC τ=0.85 | AUROC τ=0.00 | gap | n_active 0.85 | pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['auroc_tau_0.85']:.4f} | {r['auroc_tau_0.00']:.4f} | "
            f"{r['auroc_gap']:.4f} | {r['mean_n_active_0.85']:.1f} | "
            f"{'yes' if r['seed_pass'] else 'no'} |"
        )
    lines += ["", f"Status: **{status}**", ""]
    (EXP / "results.md").write_text("\n".join(lines) + "\n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    xs = np.arange(len(seeds))
    w = 0.35
    axes[0].bar(xs - w / 2, a_s, w, label="τ=0.85 (auditable)", color="#1b4f72")
    axes[0].bar(xs + w / 2, a_d, w, label="τ=0.00 (all features)", color="#b9770e")
    axes[0].axhline(
        float(thr["min_auroc_sparse"]),
        color="#922b21",
        ls="--",
        lw=1.2,
        label=f"min AUROC {thr['min_auroc_sparse']}",
    )
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([f"seed {s}" for s in seeds])
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.5, 1.02)
    axes[0].set_title("DetectorScope AUROC by seed")
    axes[0].legend(fontsize=9, loc="lower right", prop={"weight": "bold"})
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(xs, gaps, color="#1e8449" if every else "#a04000")
    axes[1].axhline(
        float(thr["max_auroc_gap"]),
        color="#922b21",
        ls="--",
        lw=1.2,
        label=f"max gap {thr['max_auroc_gap']}",
    )
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"seed {s}" for s in seeds])
    axes[1].set_ylabel("AUROC gap (0.00 − 0.85)")
    axes[1].set_title("Primary metric: AUROC gap")
    axes[1].legend(fontsize=9, prop={"weight": "bold"})
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"exp1_cpu_auditable — {status}  |  gap {gap_mean:.3f}±{gap_std:.3f}",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(EXP / "plot.png")
    plt.close(fig)

    # Breakpoints tables
    bp = raw.get("breakpoints") or {}
    bp_lines = ["# Breakpoints (seed 0)", ""]
    if bp.get("truncate"):
        bp_lines += [
            "## Truncation (max chars)",
            "",
            "| max_chars | AUROC τ=0.85 | AUROC τ=0.00 | gap | n_active |",
            "|---:|---:|---:|---:|---:|",
        ]
        with (EXP / "breakpoints.csv").open("w", newline="") as f:
            wri = csv.DictWriter(
                f,
                fieldnames=[
                    "family",
                    "setting",
                    "auroc_sparse",
                    "auroc_dense",
                    "auroc_gap",
                    "auroc",
                    "mean_n_active",
                ],
            )
            wri.writeheader()
            for row in bp["truncate"]:
                bp_lines.append(
                    f"| {row['max_chars']} | {row['auroc_sparse']:.4f} | "
                    f"{row['auroc_dense']:.4f} | {row['auroc_gap']:.4f} | "
                    f"{row['mean_n_active_sparse']:.1f} |"
                )
                wri.writerow({
                    "family": "truncate",
                    "setting": row["max_chars"],
                    "auroc_sparse": row["auroc_sparse"],
                    "auroc_dense": row["auroc_dense"],
                    "auroc_gap": row["auroc_gap"],
                    "auroc": row["auroc_sparse"],
                    "mean_n_active": row["mean_n_active_sparse"],
                })
            bp_lines += ["", "## Interpretability threshold τ", "",
                         "| τ | AUROC | accuracy | mean n_active |",
                         "|---:|---:|---:|---:|"]
            for row in bp.get("tau_sweep", []):
                bp_lines.append(
                    f"| {row['tau']:.2f} | {row['auroc']:.4f} | "
                    f"{row['accuracy']:.4f} | {row['mean_n_active_features']:.1f} |"
                )
                wri.writerow({
                    "family": "tau",
                    "setting": row["tau"],
                    "auroc_sparse": "",
                    "auroc_dense": "",
                    "auroc_gap": "",
                    "auroc": row["auroc"],
                    "mean_n_active": row["mean_n_active_features"],
                })
            bp_lines += ["", "## Top-k active features (τ=0.85 rescore)", "",
                         "| k | AUROC |", "|---:|---:|"]
            for row in bp.get("topk", []):
                bp_lines.append(f"| {row['k']} | {row['auroc']:.4f} |")
                wri.writerow({
                    "family": "topk",
                    "setting": row["k"],
                    "auroc_sparse": "",
                    "auroc_dense": "",
                    "auroc_gap": "",
                    "auroc": row["auroc"],
                    "mean_n_active": row["k"],
                })
        (EXP / "breakpoints.md").write_text("\n".join(bp_lines) + "\n")

    print(f"status={status} gap_mean={gap_mean:.4f} sparse_mean={s_mean:.4f}")


if __name__ == "__main__":
    main()
