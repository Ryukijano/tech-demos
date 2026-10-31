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
EXP = APP / "experiments" / "exp1_cpu_addition"

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

    rows = []
    looped_rates = []
    baseline_rates = []
    per_seed_looped_wins = {}
    for s in seeds:
        kl = raw["seeds"][s]["kernels"]["looped"]
        kb = raw["seeds"][s]["kernels"]["baseline"]
        gap = kl["exact_match"] - kb["exact_match"]
        per_seed_looped_wins[s] = kl["exact_match"] > kb["exact_match"]
        looped_rates.append(kl["exact_match"])
        baseline_rates.append(kb["exact_match"])
        for name, k in (("looped", kl), ("baseline", kb)):
            rows.append({
                "seed": s,
                "kind": name,
                "exact_match": k["exact_match"],
                "digit_accuracy": k["digit_accuracy"],
                "n": k["n"],
                "n_digits": k["n_digits"],
                "n_loops": k["n_loops"],
                "train_wall_s": round(k["train_wall_s"], 1),
                "eval_wall_s": round(k["eval_wall_s"], 2),
                "final_loss": round(k["final_loss"], 5),
                "n_params": k["n_params"],
            })

    with (EXP / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    l_mean = float(np.mean(looped_rates))
    l_std = float(np.std(looped_rates, ddof=1)) if len(looped_rates) > 1 else 0.0
    b_mean = float(np.mean(baseline_rates))
    b_std = float(np.std(baseline_rates, ddof=1)) if len(baseline_rates) > 1 else 0.0
    mean_gap = l_mean - b_mean
    every_seed = all(per_seed_looped_wins.values())
    gap_ok = mean_gap >= 0.10

    if every_seed and gap_ok:
        status = "PASS"
    elif l_mean > b_mean:
        status = "PARTIAL"
    else:
        status = "FAIL"

    status_obj = {
        "status": status,
        "looped_mean_exact_match": l_mean,
        "looped_std_exact_match": l_std,
        "baseline_mean_exact_match": b_mean,
        "baseline_std_exact_match": b_std,
        "mean_gap": mean_gap,
        "every_seed_looped_wins": every_seed,
        "mean_gap_ge_0.10": gap_ok,
        "per_seed_looped_wins": per_seed_looped_wins,
        "n_params": raw.get("n_params"),
        "pass_condition": cfg["thresholds"]["pass_condition"],
        "status_rule": cfg["thresholds"]["status_rule"],
        "wall_s_total": raw.get("wall_s_total"),
    }
    (EXP / "status.json").write_text(json.dumps(status_obj, indent=2))

    lines = [
        f"# exp1_cpu_addition results — **{status}**",
        "",
        f"Primary metric: exact-match accuracy on {rows[0]['n']} held-out "
        f"{cfg['data']['n_digits_eval']}-digit additions "
        f"(looped K={cfg['model']['n_loops_train']} vs baseline K=1, matched params).",
        "",
        "| seed | looped exact | baseline exact | gap | digit_acc looped | digit_acc baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in seeds:
        kl = raw["seeds"][s]["kernels"]["looped"]
        kb = raw["seeds"][s]["kernels"]["baseline"]
        lines.append(
            f"| {s} | {kl['exact_match']:.3f} | {kb['exact_match']:.3f} | "
            f"{kl['exact_match']-kb['exact_match']:+.3f} | "
            f"{kl['digit_accuracy']:.3f} | {kb['digit_accuracy']:.3f} |"
        )
    lines += [
        "",
        f"**Mean ± std:** looped **{l_mean:.3f} ± {l_std:.3f}**, "
        f"baseline **{b_mean:.3f} ± {b_std:.3f}**, gap **{mean_gap:+.3f}**.",
        "",
        f"Status rule: PASS needs looped > baseline on every seed AND mean gap ≥ 0.10. "
        f"Got every_seed={every_seed}, mean_gap={mean_gap:.3f} → **{status}**.",
        "",
        f"Params (matched): {raw.get('n_params')}.",
    ]
    (EXP / "results.md").write_text("\n".join(lines) + "\n")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    ax = axes[0]
    x = np.arange(len(seeds))
    wbar = 0.35
    ax.bar(x - wbar / 2, looped_rates, wbar, label="looped (K=4)", color="#1b9e77")
    ax.bar(x + wbar / 2, baseline_rates, wbar, label="baseline (K=1)", color="#d95f02")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Primary comparison (4-digit addition)")
    ax.legend(loc="best", prop={"weight": "bold"})
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(
        ["looped", "baseline"],
        [l_mean, b_mean],
        yerr=[l_std, b_std],
        color=["#1b9e77", "#d95f02"],
        capsize=6,
        width=0.55,
    )
    ax.set_ylabel("Exact-match (mean ± std)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Aggregate — status {status}")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Looped transformer toy vs param-matched baseline (addition, CPU)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(EXP / "plot.png", bbox_inches="tight")
    plt.close(fig)

    # breakpoints
    bp_lines = ["# Breakpoints (seed 0)", ""]
    bp_csv_rows = []
    if "test_time_loops" in raw.get("breakpoints", {}):
        bp_lines += [
            "## Test-time loops (trained looped model, K_train=4)",
            "",
            "| n_loops | looped exact | digit_acc |",
            "|---:|---:|---:|",
        ]
        for r in raw["breakpoints"]["test_time_loops"]:
            bp_lines.append(
                f"| {r['n_loops']} | {r['looped_exact_match']:.3f} | "
                f"{r['looped_digit_accuracy']:.3f} |"
            )
            bp_csv_rows.append({
                "kind": "test_time_loops",
                "n_loops": r["n_loops"],
                "n_digits": "",
                "steps": "",
                "looped_exact_match": r["looped_exact_match"],
                "baseline_exact_match": "",
            })
        bp_lines.append("")

    if "longer_digits" in raw.get("breakpoints", {}):
        bp_lines += [
            "## Digit length (ID=4; OOD shorter/longer)",
            "",
            "| n_digits | looped exact | baseline exact | gap |",
            "|---:|---:|---:|---:|",
        ]
        for r in raw["breakpoints"]["longer_digits"]:
            gap = r["looped_exact_match"] - r["baseline_exact_match"]
            bp_lines.append(
                f"| {r['n_digits']} | {r['looped_exact_match']:.3f} | "
                f"{r['baseline_exact_match']:.3f} | {gap:+.3f} |"
            )
            bp_csv_rows.append({
                "kind": "longer_digits",
                "n_loops": "",
                "n_digits": r["n_digits"],
                "steps": "",
                "looped_exact_match": r["looped_exact_match"],
                "baseline_exact_match": r["baseline_exact_match"],
            })
        bp_lines.append("")

    if "train_steps_ablation" in raw.get("breakpoints", {}):
        bp_lines += [
            "## Training-step ablation",
            "",
            "| steps | looped exact | baseline exact | gap |",
            "|---:|---:|---:|---:|",
        ]
        for r in raw["breakpoints"]["train_steps_ablation"]:
            gap = r["looped_exact_match"] - r["baseline_exact_match"]
            bp_lines.append(
                f"| {r['steps']} | {r['looped_exact_match']:.3f} | "
                f"{r['baseline_exact_match']:.3f} | {gap:+.3f} |"
            )
            bp_csv_rows.append({
                "kind": "train_steps",
                "n_loops": "",
                "n_digits": "",
                "steps": r["steps"],
                "looped_exact_match": r["looped_exact_match"],
                "baseline_exact_match": r["baseline_exact_match"],
            })
        bp_lines.append("")

    bp_lines.append(
        "Honest note: this toy uses a ~50–100k-param network on synthetic "
        "addition — far from the paper's up-to-1B LM experiments. Breakpoints "
        "show where the *toy* saturates or breaks, not where the paper's method fails."
    )
    (EXP / "breakpoints.md").write_text("\n".join(bp_lines) + "\n")
    if bp_csv_rows:
        with (EXP / "breakpoints.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bp_csv_rows[0].keys()))
            w.writeheader()
            w.writerows(bp_csv_rows)

    print(
        f"status={status} looped={l_mean:.3f}±{l_std:.3f} "
        f"baseline={b_mean:.3f}±{b_std:.3f} gap={mean_gap:+.3f}"
    )


if __name__ == "__main__":
    main()
