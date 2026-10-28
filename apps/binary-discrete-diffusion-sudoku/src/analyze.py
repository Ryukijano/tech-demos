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
EXP = APP / "experiments" / "exp1_cpu_sudoku"

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
    binary_rates = []
    masked_rates = []
    per_seed_binary_wins = {}
    for s in seeds:
        kb = raw["seeds"][s]["kernels"]["binary"]
        km = raw["seeds"][s]["kernels"]["masked"]
        gap = kb["solve_rate"] - km["solve_rate"]
        per_seed_binary_wins[s] = kb["solve_rate"] > km["solve_rate"]
        binary_rates.append(kb["solve_rate"])
        masked_rates.append(km["solve_rate"])
        for name, k in (("binary", kb), ("masked", km)):
            rows.append({
                "seed": s,
                "kernel": name,
                "solve_rate": k["solve_rate"],
                "valid_board_rate": k["valid_board_rate"],
                "cell_accuracy": k["cell_accuracy"],
                "n_solved": k["n_solved"],
                "n": k["n"],
                "nfe": k["nfe"],
                "train_wall_s": round(k["train_wall_s"], 1),
                "sample_wall_s": round(k["sample_wall_s"], 2),
                "final_loss": round(k["final_loss"], 5),
                "n_params": k["n_params"],
            })

    with (EXP / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    b_mean, b_std = float(np.mean(binary_rates)), float(np.std(binary_rates, ddof=1)) if len(binary_rates) > 1 else 0.0
    m_mean, m_std = float(np.mean(masked_rates)), float(np.std(masked_rates, ddof=1)) if len(masked_rates) > 1 else 0.0
    mean_gap = b_mean - m_mean
    every_seed = all(per_seed_binary_wins.values())
    gap_ok = mean_gap >= 0.05

    if every_seed and gap_ok:
        status = "PASS"
    elif b_mean > m_mean:
        status = "PARTIAL"
    else:
        status = "FAIL"

    status_obj = {
        "status": status,
        "binary_mean_solve_rate": b_mean,
        "binary_std_solve_rate": b_std,
        "masked_mean_solve_rate": m_mean,
        "masked_std_solve_rate": m_std,
        "mean_gap": mean_gap,
        "every_seed_binary_wins": every_seed,
        "mean_gap_ge_0.05": gap_ok,
        "per_seed_binary_wins": per_seed_binary_wins,
        "n_params": raw.get("n_params"),
        "pass_condition": cfg["thresholds"]["pass_condition"],
        "status_rule": cfg["thresholds"]["status_rule"],
    }
    (EXP / "status.json").write_text(json.dumps(status_obj, indent=2))

    # results.md
    lines = [
        f"# exp1_cpu_sudoku results — **{status}**",
        "",
        f"Primary metric: full-puzzle solve rate on {rows[0]['n']} held-out puzzles "
        f"(36 clues, NFE={rows[0]['nfe']}).",
        "",
        f"| seed | binary solve | masked solve | gap | cell_acc binary | cell_acc masked |",
        f"|---|---:|---:|---:|---:|---:|",
    ]
    for s in seeds:
        kb = raw["seeds"][s]["kernels"]["binary"]
        km = raw["seeds"][s]["kernels"]["masked"]
        lines.append(
            f"| {s} | {kb['solve_rate']:.3f} | {km['solve_rate']:.3f} | "
            f"{kb['solve_rate']-km['solve_rate']:+.3f} | "
            f"{kb['cell_accuracy']:.3f} | {km['cell_accuracy']:.3f} |"
        )
    lines += [
        "",
        f"**Mean ± std:** binary **{b_mean:.3f} ± {b_std:.3f}**, "
        f"masked **{m_mean:.3f} ± {m_std:.3f}**, gap **{mean_gap:+.3f}**.",
        "",
        f"Status rule: PASS needs binary > masked on every seed AND mean gap ≥ 0.05. "
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
    ax.bar(x - wbar / 2, binary_rates, wbar, label="binary symmetric", color="#2166ac")
    ax.bar(x + wbar / 2, masked_rates, wbar, label="masked", color="#b2182b")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Full-puzzle solve rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Primary comparison (50 clues, NFE=32)")
    ax.legend(loc="upper right", prop={"weight": "bold"})
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(
        ["binary", "masked"],
        [b_mean, m_mean],
        yerr=[b_std, m_std],
        color=["#2166ac", "#b2182b"],
        capsize=6,
        width=0.55,
    )
    ax.set_ylabel("Solve rate (mean ± std)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Aggregate — status {status}")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Binary Discrete Diffusion toy vs masked baseline (Sudoku CPU)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(EXP / "plot.png", bbox_inches="tight")
    plt.close(fig)

    # breakpoints
    bp_lines = ["# Breakpoints (seed 0)", ""]
    bp_csv_rows = []
    if "fewer_clues" in raw.get("breakpoints", {}):
        bp_lines += [
            "## Fewer clues (harder puzzles)",
            "",
            "| n_clues | binary solve | masked solve | gap |",
            "|---:|---:|---:|---:|",
        ]
        for r in raw["breakpoints"]["fewer_clues"]:
            gap = r["binary_solve_rate"] - r["masked_solve_rate"]
            bp_lines.append(
                f"| {r['n_clues']} | {r['binary_solve_rate']:.3f} | "
                f"{r['masked_solve_rate']:.3f} | {gap:+.3f} |"
            )
            bp_csv_rows.append({
                "kind": "fewer_clues",
                "n_clues": r["n_clues"],
                "nfe": "",
                "steps": "",
                "binary_solve_rate": r["binary_solve_rate"],
                "masked_solve_rate": r["masked_solve_rate"],
            })
        bp_lines.append("")

    if "fewer_steps" in raw.get("breakpoints", {}):
        bp_lines += [
            "## Fewer denoising steps (NFE)",
            "",
            "| NFE | binary solve | masked solve | gap |",
            "|---:|---:|---:|---:|",
        ]
        for r in raw["breakpoints"]["fewer_steps"]:
            gap = r["binary_solve_rate"] - r["masked_solve_rate"]
            bp_lines.append(
                f"| {r['nfe']} | {r['binary_solve_rate']:.3f} | "
                f"{r['masked_solve_rate']:.3f} | {gap:+.3f} |"
            )
            bp_csv_rows.append({
                "kind": "fewer_steps",
                "n_clues": "",
                "nfe": r["nfe"],
                "steps": "",
                "binary_solve_rate": r["binary_solve_rate"],
                "masked_solve_rate": r["masked_solve_rate"],
            })
        bp_lines.append("")

    if "train_steps_ablation" in raw.get("breakpoints", {}):
        bp_lines += [
            "## Training-step ablation",
            "",
            "| steps | binary solve | masked solve | gap |",
            "|---:|---:|---:|---:|",
        ]
        for r in raw["breakpoints"]["train_steps_ablation"]:
            gap = r["binary_solve_rate"] - r["masked_solve_rate"]
            bp_lines.append(
                f"| {r['steps']} | {r['binary_solve_rate']:.3f} | "
                f"{r['masked_solve_rate']:.3f} | {gap:+.3f} |"
            )
            bp_csv_rows.append({
                "kind": "train_steps",
                "n_clues": "",
                "nfe": "",
                "steps": r["steps"],
                "binary_solve_rate": r["binary_solve_rate"],
                "masked_solve_rate": r["masked_solve_rate"],
            })
        bp_lines.append("")

    bp_lines.append(
        "Honest note: this toy uses a ~150k-param network and synthetic 36-clue "
        "boards — far from the paper's 43M DiT on Sudoku-Extreme. Breakpoints show "
        "where the *toy* comparison collapses, not where the paper's method fails."
    )
    (EXP / "breakpoints.md").write_text("\n".join(bp_lines) + "\n")
    if bp_csv_rows:
        with (EXP / "breakpoints.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bp_csv_rows[0].keys()))
            w.writeheader()
            w.writerows(bp_csv_rows)

    print(f"status={status} binary={b_mean:.3f}±{b_std:.3f} masked={m_mean:.3f}±{m_std:.3f}")


if __name__ == "__main__":
    main()
