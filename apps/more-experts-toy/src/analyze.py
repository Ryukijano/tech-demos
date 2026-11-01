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
EXP = APP / "experiments" / "exp1_cpu_charlm"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "figure.dpi": 140,
    }
)

COLORS = {
    "dense": "#7570b3",
    "moe": "#d95f02",
    "more": "#1b9e77",
    "more_nodepth": "#66a61e",
}
LABELS = {
    "dense": "dense",
    "moe": "MoE",
    "more": "MoRE (R=2)",
    "more_nodepth": "MoRE no-depth",
}


def _std(xs: list[float]) -> float:
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def main() -> None:
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    raw = json.loads((EXP / "raw.json").read_text())
    seeds = sorted(raw["seeds"].keys(), key=int)
    kernels = list(cfg["kernels"])

    rows = []
    series: dict[str, list[float]] = {k: [] for k in kernels}
    for s in seeds:
        for kind in kernels:
            k = raw["seeds"][s]["kernels"][kind]
            series[kind].append(k["val_ce"])
            route = k.get("routing") or {}
            rows.append(
                {
                    "seed": s,
                    "kind": kind,
                    "val_ce": round(k["val_ce"], 6),
                    "bits_per_char": round(k["bits_per_char"], 6),
                    "train_wall_s": round(k["train_wall_s"], 1),
                    "final_train_ce": round(k["final_train_ce"], 5),
                    "n_params": k["n_params"],
                    "n_active_params_per_token": k["n_active_params_per_token"],
                    "adjacent_expert_reuse_rate": route.get(
                        "adjacent_expert_reuse_rate"
                    ),
                    "random_reuse_baseline": route.get("random_reuse_baseline"),
                    "reuse_ratio_vs_random": route.get("ratio_vs_random"),
                }
            )

    with (EXP / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    means = {k: float(np.mean(series[k])) for k in kernels}
    stds = {k: _std(series[k]) for k in kernels}
    gap = means["moe"] - means["more"]
    gap_ok = gap >= stds["more"]
    more_wins_mean = means["more"] < means["moe"]

    if more_wins_mean and gap_ok:
        status = "PASS"
    elif more_wins_mean:
        status = "PARTIAL"
    else:
        status = "FAIL"

    # Routing aggregates for MoRE
    more_reuse = []
    more_rand = []
    for s in seeds:
        r = raw["seeds"][s]["kernels"]["more"].get("routing") or {}
        if r.get("adjacent_expert_reuse_rate") is not None:
            more_reuse.append(r["adjacent_expert_reuse_rate"])
            more_rand.append(r["random_reuse_baseline"])

    status_obj = {
        "status": status,
        "means_val_ce": means,
        "stds_val_ce": stds,
        "moe_minus_more_gap": gap,
        "more_std": stds["more"],
        "gap_ge_more_std": gap_ok,
        "more_beats_moe_mean": more_wins_mean,
        "more_reuse_rate_mean": float(np.mean(more_reuse)) if more_reuse else None,
        "more_random_baseline_mean": float(np.mean(more_rand)) if more_rand else None,
        "param_table": raw.get("param_table"),
        "pass_condition": cfg["thresholds"]["pass_condition"],
        "status_rule": cfg["thresholds"]["status_rule"],
        "wall_s_total": raw.get("wall_s_total"),
    }
    (EXP / "status.json").write_text(json.dumps(status_obj, indent=2))

    lines = [
        f"# exp1_cpu_charlm results — **{status}**",
        "",
        "Primary metric: validation cross-entropy (nats) on held-out characters "
        f"(seq_len={cfg['data']['seq_len']}, "
        f"steps={cfg['training']['steps']}). Lower is better.",
        "",
        "| seed | dense | MoE | MoRE | MoRE no-depth | MoE−MoRE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in seeds:
        kd = raw["seeds"][s]["kernels"]
        gap_s = kd["moe"]["val_ce"] - kd["more"]["val_ce"]
        lines.append(
            f"| {s} | {kd['dense']['val_ce']:.4f} | {kd['moe']['val_ce']:.4f} | "
            f"{kd['more']['val_ce']:.4f} | {kd['more_nodepth']['val_ce']:.4f} | "
            f"{gap_s:+.4f} |"
        )
    lines += [
        "",
        "**Mean ± std (val CE):**",
        f"- dense **{means['dense']:.4f} ± {stds['dense']:.4f}**",
        f"- MoE **{means['moe']:.4f} ± {stds['moe']:.4f}**",
        f"- MoRE **{means['more']:.4f} ± {stds['more']:.4f}**",
        f"- MoRE no-depth **{means['more_nodepth']:.4f} ± {stds['more_nodepth']:.4f}**",
        f"- gap MoE−MoRE **{gap:+.4f}** (need ≥ MoRE std {stds['more']:.4f})",
        "",
        f"Status rule: PASS needs mean(MoRE) < mean(MoE) AND gap ≥ std(MoRE). "
        f"Got more_beats={more_wins_mean}, gap_ok={gap_ok} → **{status}**.",
        "",
        f"Params: {raw.get('param_table')}.",
        "",
        "## Routing (MoRE adjacent shared-pool reuse)",
        "",
    ]
    if more_reuse:
        lines.append(
            f"Mean adjacent reuse rate **{float(np.mean(more_reuse)):.4f}** vs "
            f"random baseline **{float(np.mean(more_rand)):.4f}** "
            f"(ratio **{float(np.mean(more_reuse))/max(1e-9,float(np.mean(more_rand))):.2f}×**)."
        )
    else:
        lines.append("No MoRE routing stats recorded.")
    lines.append("")
    (EXP / "results.md").write_text("\n".join(lines) + "\n")

    # Plot: per-seed bars + loss curves for seed 0
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    x = np.arange(len(seeds))
    width = 0.2
    for i, kind in enumerate(kernels):
        ax.bar(
            x + (i - 1.5) * width,
            series[kind],
            width,
            label=LABELS[kind],
            color=COLORS[kind],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Validation CE (nats)")
    ax.set_title("Primary comparison (char LM)")
    ax.legend(loc="best", prop={"weight": "bold"}, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    # Loss curves from seed 0 histories
    s0 = seeds[0]
    for kind in kernels:
        hist = raw["seeds"][s0]["kernels"][kind].get("history") or []
        if not hist:
            continue
        steps = [h["step"] for h in hist]
        vals = [h["val_ce"] for h in hist]
        ax.plot(
            steps,
            vals,
            label=LABELS[kind],
            color=COLORS[kind],
            linewidth=2.0,
            marker="o",
            markersize=3,
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation CE")
    ax.set_title(f"Learning curves (seed {s0}) — status {status}")
    ax.legend(loc="best", prop={"weight": "bold"}, fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(
        "MoRE toy vs matched MoE / dense (char LM, CPU)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(EXP / "plot.png", bbox_inches="tight")
    plt.close(fig)

    # Breakpoints markdown / csv
    bp_lines = ["# Breakpoints (seed 0)", ""]
    bp_csv_rows = []
    bp = raw.get("breakpoints") or {}

    if "reuse_factor_extreme" in bp:
        bp_lines += [
            "## Reuse factor extremes",
            "",
            "| kind | R | steps | val_ce | bpc | reuse_rate | random |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in bp["reuse_factor_extreme"]:
            route = r.get("routing") or {}
            bp_lines.append(
                f"| {r['kind']} | {r['R']} | {r['steps']} | {r['val_ce']:.4f} | "
                f"{r['bits_per_char']:.4f} | "
                f"{route.get('adjacent_expert_reuse_rate')} | "
                f"{route.get('random_reuse_baseline')} |"
            )
            bp_csv_rows.append(
                {
                    "kind_bp": "reuse_factor",
                    "model": r["kind"],
                    "R": r["R"],
                    "n_experts": "",
                    "top_k": "",
                    "steps": r["steps"],
                    "val_ce": r["val_ce"],
                }
            )
        bp_lines.append("")

    if "few_experts" in bp:
        bp_lines += [
            "## Few experts (N=2)",
            "",
            "| kind | N | steps | val_ce | bpc |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in bp["few_experts"]:
            bp_lines.append(
                f"| {r['kind']} | {r['n_experts']} | {r['steps']} | "
                f"{r['val_ce']:.4f} | {r['bits_per_char']:.4f} |"
            )
            bp_csv_rows.append(
                {
                    "kind_bp": "few_experts",
                    "model": r["kind"],
                    "R": "",
                    "n_experts": r["n_experts"],
                    "top_k": "",
                    "steps": r["steps"],
                    "val_ce": r["val_ce"],
                }
            )
        bp_lines.append("")

    if "top1" in bp:
        bp_lines += [
            "## Top-1 routing",
            "",
            "| kind | K | steps | val_ce | bpc |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in bp["top1"]:
            bp_lines.append(
                f"| {r['kind']} | {r['top_k']} | {r['steps']} | "
                f"{r['val_ce']:.4f} | {r['bits_per_char']:.4f} |"
            )
            bp_csv_rows.append(
                {
                    "kind_bp": "top1",
                    "model": r["kind"],
                    "R": "",
                    "n_experts": "",
                    "top_k": r["top_k"],
                    "steps": r["steps"],
                    "val_ce": r["val_ce"],
                }
            )
        bp_lines.append("")

    bp_lines.append(
        "Honest note: this toy uses a ~100k-param network on a 50k-char "
        "public-domain excerpt — far from the paper's 114M–1.15B C4 runs. "
        "Breakpoints show where the *toy* saturates or breaks."
    )
    (EXP / "breakpoints.md").write_text("\n".join(bp_lines) + "\n")
    if bp_csv_rows:
        with (EXP / "breakpoints.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bp_csv_rows[0].keys()))
            w.writeheader()
            w.writerows(bp_csv_rows)

    print(
        f"status={status} more={means['more']:.4f}±{stds['more']:.4f} "
        f"moe={means['moe']:.4f}±{stds['moe']:.4f} gap={gap:+.4f}"
    )


if __name__ == "__main__":
    main()
