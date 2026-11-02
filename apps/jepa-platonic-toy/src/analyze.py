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
EXP = APP / "experiments" / "exp1_cpu_views"

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
    jepa_cks: list[float] = []
    recon_cks: list[float] = []
    random_cks: list[float] = []
    per_seed_beats_recon: dict[str, bool] = {}
    per_seed_beats_random: dict[str, bool] = {}

    for s in seeds:
        kj = raw["seeds"][s]["kernels"]["jepa"]
        kr = raw["seeds"][s]["kernels"]["recon"]
        kn = raw["seeds"][s]["kernels"]["random"]
        cj, cr, cn = kj["procrustes_cka"], kr["procrustes_cka"], kn["procrustes_cka"]
        jepa_cks.append(cj)
        recon_cks.append(cr)
        random_cks.append(cn)
        per_seed_beats_recon[s] = cj > cr
        per_seed_beats_random[s] = cj > cn
        for name, k in (("jepa", kj), ("recon", kr), ("random", kn)):
            rows.append({
                "seed": s,
                "kind": name,
                "procrustes_cka": round(k["procrustes_cka"], 6),
                "mutual_knn_overlap": round(k["mutual_knn_overlap"], 6),
                "procrustes_rmse": round(k["procrustes_rmse"], 6),
                "n_eval": k["n_eval"],
                "train_wall_s": round(k["train_wall_s"], 2),
                "n_params_a": k["n_params_a"],
                "n_params_b": k["n_params_b"],
            })

    with (EXP / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    j_mean = float(np.mean(jepa_cks))
    j_std = float(np.std(jepa_cks, ddof=1)) if len(jepa_cks) > 1 else 0.0
    r_mean = float(np.mean(recon_cks))
    r_std = float(np.std(recon_cks, ddof=1)) if len(recon_cks) > 1 else 0.0
    n_mean = float(np.mean(random_cks))
    n_std = float(np.std(random_cks, ddof=1)) if len(random_cks) > 1 else 0.0
    gap_recon = j_mean - r_mean
    gap_random = j_mean - n_mean

    every_recon = all(per_seed_beats_recon.values())
    every_random = all(per_seed_beats_random.values())
    abs_ok = j_mean >= float(thr["cka_jepa_mean_min"])
    gap_r_ok = gap_recon >= float(thr["gap_vs_recon_min"])
    gap_n_ok = gap_random >= float(thr["gap_vs_random_min"])

    if every_recon and every_random and abs_ok and gap_r_ok and gap_n_ok:
        status = "PASS"
    elif j_mean > r_mean and j_mean > n_mean:
        status = "PARTIAL"
    else:
        status = "FAIL"

    status_obj = {
        "status": status,
        "mean_cka_jepa": j_mean,
        "std_cka_jepa": j_std,
        "mean_cka_recon": r_mean,
        "std_cka_recon": r_std,
        "mean_cka_random": n_mean,
        "std_cka_random": n_std,
        "gap_vs_recon": gap_recon,
        "gap_vs_random": gap_random,
        "every_seed_beats_recon": every_recon,
        "every_seed_beats_random": every_random,
        "checks": {
            "cka_jepa_mean_min": {
                "threshold": thr["cka_jepa_mean_min"],
                "value": j_mean,
                "ok": abs_ok,
            },
            "gap_vs_recon_min": {
                "threshold": thr["gap_vs_recon_min"],
                "value": gap_recon,
                "ok": gap_r_ok,
            },
            "gap_vs_random_min": {
                "threshold": thr["gap_vs_random_min"],
                "value": gap_random,
                "ok": gap_n_ok,
            },
            "every_seed_beats_recon": every_recon,
            "every_seed_beats_random": every_random,
        },
        "per_seed_cka": {
            s: {
                "jepa": raw["seeds"][s]["kernels"]["jepa"]["procrustes_cka"],
                "recon": raw["seeds"][s]["kernels"]["recon"]["procrustes_cka"],
                "random": raw["seeds"][s]["kernels"]["random"]["procrustes_cka"],
            }
            for s in seeds
        },
        "wall_s_total": raw.get("wall_s_total"),
    }
    (EXP / "status.json").write_text(json.dumps(status_obj, indent=2))

    # results.md
    lines = [
        f"# Results — `{cfg['exp_id']}`",
        "",
        f"**Status: {status}**",
        "",
        "Primary metric: Procrustes-aligned linear CKA (higher is better).",
        "",
        "| seed | JEPA CKA | recon CKA | random CKA | JEPA−recon | JEPA−random | JEPA kNN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in seeds:
        kj = raw["seeds"][s]["kernels"]["jepa"]
        kr = raw["seeds"][s]["kernels"]["recon"]
        kn = raw["seeds"][s]["kernels"]["random"]
        lines.append(
            f"| {s} | {kj['procrustes_cka']:.4f} | {kr['procrustes_cka']:.4f} | "
            f"{kn['procrustes_cka']:.4f} | {kj['procrustes_cka']-kr['procrustes_cka']:+.4f} | "
            f"{kj['procrustes_cka']-kn['procrustes_cka']:+.4f} | {kj['mutual_knn_overlap']:.4f} |"
        )
    lines += [
        "",
        f"**Mean ± std:** JEPA **{j_mean:.4f} ± {j_std:.4f}**, "
        f"recon **{r_mean:.4f} ± {r_std:.4f}**, "
        f"random **{n_mean:.4f} ± {n_std:.4f}**.",
        "",
        f"Gaps: vs recon **{gap_recon:+.4f}** (need ≥ {thr['gap_vs_recon_min']}); "
        f"vs random **{gap_random:+.4f}** (need ≥ {thr['gap_vs_random_min']}); "
        f"JEPA mean need ≥ {thr['cka_jepa_mean_min']}.",
        "",
        f"Every-seed beats recon: **{every_recon}**; beats random: **{every_random}**.",
        "",
    ]
    (EXP / "results.md").write_text("\n".join(lines) + "\n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    x = np.arange(len(seeds))
    w = 0.25
    ax.bar(x - w, jepa_cks, width=w, label="JEPA (good views)", color="#1f6f5a")
    ax.bar(x, recon_cks, width=w, label="Reconstruction", color="#b86b2a")
    ax.bar(x + w, random_cks, width=w, label="Random init", color="#6b7280")
    ax.axhline(thr["cka_jepa_mean_min"], color="#1f6f5a", ls="--", lw=1.2, label="JEPA mean bar")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Procrustes-aligned linear CKA")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-modal alignment by seed")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    means = [j_mean, r_mean, n_mean]
    stds = [j_std, r_std, n_std]
    colors = ["#1f6f5a", "#b86b2a", "#6b7280"]
    labels = ["JEPA", "recon", "random"]
    ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.9)
    ax.axhline(thr["cka_jepa_mean_min"], color="#1f6f5a", ls="--", lw=1.2)
    ax.set_ylabel("Mean Procrustes CKA")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Aggregate (status={status})")
    ax.grid(True, axis="y", alpha=0.3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Platonic JEPA toy — good-view cross-modal alignment", fontweight="bold")
    fig.tight_layout()
    fig.savefig(EXP / "plot.png")
    plt.close(fig)

    # Breakpoints tables
    bp = raw.get("breakpoints") or {}
    bp_lines = [
        f"# Breakpoints — `{cfg['exp_id']}` (seed {cfg['breakpoints']['seed']})",
        "",
    ]
    bp_csv_rows: list[dict] = []

    if "bad_views" in bp:
        b = bp["bad_views"]
        primary0 = raw["seeds"][str(cfg["breakpoints"]["seed"])]["kernels"]["jepa"]["procrustes_cka"]
        bp_lines += [
            "## Bad views (corruptions of one observation)",
            "",
            f"- Good-view JEPA (primary, seed {cfg['breakpoints']['seed']}): CKA **{primary0:.4f}**",
            f"- Bad-view JEPA: CKA **{b['procrustes_cka']:.4f}** "
            f"(noise_std={b['noise_std']}, dropout_p={b['dropout_p']}, steps={b['steps']})",
            f"- Δ (good − bad): **{primary0 - b['procrustes_cka']:+.4f}**",
            "",
        ]
        bp_csv_rows.append({
            "breakpoint": "bad_views",
            "setting": f"noise={b['noise_std']},drop={b['dropout_p']}",
            "procrustes_cka": b["procrustes_cka"],
            "mutual_knn_overlap": b["mutual_knn_overlap"],
        })

    if "tiny_emb_dim" in bp:
        bp_lines += [
            "## Embedding dimension",
            "",
            "| emb_dim | steps | Procrustes CKA | kNN overlap |",
            "|---:|---:|---:|---:|",
        ]
        for row in bp["tiny_emb_dim"]:
            bp_lines.append(
                f"| {row['emb_dim']} | {row['steps']} | {row['procrustes_cka']:.4f} | "
                f"{row['mutual_knn_overlap']:.4f} |"
            )
            bp_csv_rows.append({
                "breakpoint": "tiny_emb_dim",
                "setting": f"emb={row['emb_dim']}",
                "procrustes_cka": row["procrustes_cka"],
                "mutual_knn_overlap": row["mutual_knn_overlap"],
            })
        bp_lines.append("")

    if "short_training" in bp:
        bp_lines += [
            "## Training length",
            "",
            "| steps | Procrustes CKA | kNN overlap |",
            "|---:|---:|---:|",
        ]
        for row in bp["short_training"]:
            bp_lines.append(
                f"| {row['steps']} | {row['procrustes_cka']:.4f} | {row['mutual_knn_overlap']:.4f} |"
            )
            bp_csv_rows.append({
                "breakpoint": "short_training",
                "setting": f"steps={row['steps']}",
                "procrustes_cka": row["procrustes_cka"],
                "mutual_knn_overlap": row["mutual_knn_overlap"],
            })
        bp_lines.append("")

    (EXP / "breakpoints.md").write_text("\n".join(bp_lines) + "\n")
    if bp_csv_rows:
        with (EXP / "breakpoints.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bp_csv_rows[0].keys()))
            w.writeheader()
            w.writerows(bp_csv_rows)

    print(f"status={status} jepa={j_mean:.4f} recon={r_mean:.4f} random={n_mean:.4f}")


if __name__ == "__main__":
    main()
