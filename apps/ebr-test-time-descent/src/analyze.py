"""Tables, the pre-registered status, and the figure. Reads raw.json; does not re-roll."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import bars_from_runs, mean_std  # noqa: E402

EXP = ROOT / "experiments" / "exp1_cpu_sudoku_extreme"

# Wong colorblind palette.
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
GOLD = "#E69F00"
PURPLE = "#CC79A7"
BLACK = "#222222"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": True,
        "axes.spines.right": True,
    })


def _load() -> tuple[dict, dict, dict | None]:
    with (EXP / "config.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    raw = json.loads((EXP / "raw.json").read_text())
    bp_path = EXP / "breakpoints_raw.json"
    bp = json.loads(bp_path.read_text()) if bp_path.is_file() else None
    return cfg, raw, bp


def _curve(runs: list[dict], key: str, segment: int) -> tuple[float, float]:
    return mean_std([float(run[key][segment - 1]) for run in runs])


def _class_energy(run: dict, segment: int) -> tuple[float, float]:
    labels = np.asarray(run["exact_mask"][segment - 1], dtype=bool)
    energy = np.asarray(run["energy"][segment - 1], dtype=np.float64)
    correct = float(energy[labels].mean()) if labels.any() else float("nan")
    incorrect = float(energy[~labels].mean()) if (~labels).any() else float("nan")
    return correct, incorrect


def write_tables(cfg: dict, raw: dict, status: dict) -> None:
    runs = raw["runs"]
    depth = int(cfg["eval_segments"])
    rows = []
    for segment in range(1, depth + 1):
        mean, std = _curve(runs, "exact", segment)
        cell_mean, cell_std = _curve(runs, "cell", segment)
        rows.append({
            "segment": segment,
            "exact_mean": mean,
            "exact_std": std,
            "cell_mean": cell_mean,
            "cell_std": cell_std,
            "microsteps": segment * 21,
        })
    with (EXP / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Sudoku-Extreme subset, exact-board accuracy",
        "",
        "Mean ± sample standard deviation over inference seeds 0/1/2. "
        "One released checkpoint. 96 highest-rated test puzzles, one trajectory each.",
        "",
        "| Segment | Microsteps | Exact accuracy | Cell accuracy |",
        "|---|---|---|---|",
    ]
    for row in rows:
        if row["segment"] in (1, 2, 4, 8, 16) or row["segment"] == depth:
            lines.append(
                f"| {row['segment']} | {row['microsteps']} | "
                f"{row['exact_mean']:.3f} ± {row['exact_std']:.3f} | "
                f"{row['cell_mean']:.3f} ± {row['cell_std']:.3f} |"
            )
    lines.extend([
        "",
        f"Status: **{status['status']}** ({status['n_bars_passed']}/4 bars).",
        "",
        f"Gain segment 16 − 4: {status['gain_16_minus_4']['mean']:.3f} ± "
        f"{status['gain_16_minus_4']['std']:.3f} "
        f"(bar ≥ {cfg['thresholds']['gain_16_minus_4_min']}, "
        f"{'met' if status['bars']['gain'] else 'not met'}).",
        "",
        "| Seed | Energy depth | AUROC(−E) | gtan incorrect / correct |",
        "|---|---|---|---|",
    ])
    for record in status["energy"]:
        auroc = "—" if record["auroc"] is None else f"{record['auroc']:.3f}"
        ratio = "—" if record["gtan_ratio"] is None else f"{record['gtan_ratio']:.2f}"
        lines.append(f"| {record['seed']} | {record['depth']} | {auroc} | {ratio} |")
    lines.append("")
    (EXP / "results.md").write_text("\n".join(lines) + "\n")


def write_breakpoints(bp: dict | None) -> None:
    if bp is None:
        return
    lines = [
        "# Breakpoints (seed 0, not part of the status)",
        "",
        "Init-std and Langevin noise use the 32 hardest boards and 8 segments. "
        "Rating bands use 32 boards and 16 segments. "
        "The unit-RMS projection removes the scale of the initial Gaussian, so "
        "init_std is a weak knob. noise_scale is added after that projection.",
        "",
    ]
    for kind, title in (("init_std", "Initial Gaussian std"), ("noise_scale", "Langevin noise scale")):
        rows = [s for s in bp["studies"] if s["kind"] == kind]
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Value | Accuracy at segment 1 | Accuracy at segment 8 |")
        lines.append("|---|---|---|")
        for row in rows:
            lines.append(f"| {row['value']} | {row['exact'][0]:.3f} | {row['exact'][7]:.3f} |")
        lines.append("")
    lines.append("## Rating band")
    lines.append("")
    lines.append("| Band | Segment 1 | Segment 4 | Segment 16 |")
    lines.append("|---|---|---|---|")
    for row in bp["studies"]:
        if row["kind"] != "band":
            continue
        lines.append(
            f"| {row['name']} | {row['exact'][0]:.3f} | {row['exact'][3]:.3f} | {row['exact'][15]:.3f} |"
        )
    lines.append("")
    (EXP / "breakpoints.md").write_text("\n".join(lines) + "\n")
    with (EXP / "breakpoints.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "name", "value", "segment", "exact"])
        for row in bp["studies"]:
            label = row.get("name", "")
            value = row.get("value", "")
            for i, exact in enumerate(row["exact"], start=1):
                writer.writerow([row["kind"], label, value, i, f"{exact:.6f}"])


def _acc_matrix(runs: list[dict], depth: int) -> np.ndarray:
    return np.stack([np.asarray(run["exact"][:depth], dtype=np.float64) for run in runs], axis=0)


def make_plot(cfg: dict, raw: dict, status: dict, bp: dict | None) -> None:
    _style()
    runs = raw["runs"]
    depth = int(cfg["eval_segments"])
    segments = np.arange(1, depth + 1)
    acc = _acc_matrix(runs, depth)
    mean = acc.mean(axis=0)
    std = acc.std(axis=0, ddof=1)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.4), constrained_layout=True)

    ax = axes[0, 0]
    for row in acc:
        ax.plot(segments, row, color=BLUE, alpha=0.35, lw=1.0)
    ax.errorbar(
        segments, mean, yerr=std, color=BLUE, lw=2.2, marker="o", ms=4.5,
        capsize=3, label="EBR, seeds 0/1/2",
    )
    seed0 = next(run for run in runs if int(run["seed"]) == 0)
    if len(seed0["exact"]) > depth:
        extra_x = np.arange(depth, len(seed0["exact"]) + 1)
        extra_y = np.asarray(seed0["exact"][depth - 1 :], dtype=np.float64)
        ax.plot(extra_x, extra_y, color=BLUE, lw=1.6, ls="--", label="Seed 0, past 16")
    ax.axvline(int(cfg["train_segments"]), color=BLACK, lw=1.0, ls=":", label="Training halt (4)")
    ax.set_xlabel("Descent segments")
    ax.set_ylabel("Exact-board accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("A  Test-time segments")
    ax.legend(prop={"weight": "bold", "size": 9}, loc="lower right")

    ax = axes[0, 1]
    e_ok, e_bad = [], []
    for segment in segments:
        oks, bads = [], []
        for run in runs:
            correct, incorrect = _class_energy(run, int(segment))
            oks.append(correct)
            bads.append(incorrect)
        e_ok.append(oks)
        e_bad.append(bads)
    e_ok_arr = np.asarray(e_ok, dtype=np.float64)
    e_bad_arr = np.asarray(e_bad, dtype=np.float64)
    ax.plot(segments, np.nanmean(e_ok_arr, axis=1), color=GREEN, lw=2.2, marker="o", ms=4, label="Correct")
    ax.plot(segments, np.nanmean(e_bad_arr, axis=1), color=ORANGE, lw=2.2, marker="s", ms=4, label="Incorrect")
    ax.set_xlabel("Descent segments")
    ax.set_ylabel("Mean energy")
    ax.set_title("B  Energy by correctness")
    ax.legend(prop={"weight": "bold", "size": 9})

    ax = axes[1, 0]
    if bp is not None:
        for kind, color, label in (
            ("init_std", BLUE, r"Init std (before RMS)"),
            ("noise_scale", PURPLE, r"Langevin $\sigma$"),
        ):
            rows = [s for s in bp["studies"] if s["kind"] == kind]
            xs = [float(s["value"]) for s in rows]
            ys = [float(s["exact"][7]) for s in rows]
            ax.plot(xs, ys, color=color, lw=2.2, marker="o", ms=5, label=label)
        ax.set_xscale("symlog", linthresh=0.02)
    ax.set_xlabel("Noise parameter")
    ax.set_ylabel("Exact accuracy at segment 8")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("C  Init scale and Langevin noise")
    ax.legend(prop={"weight": "bold", "size": 9})

    ax = axes[1, 1]
    if bp is not None:
        band_colors = {"easiest": GREEN, "median": GOLD, "hardest": ORANGE}
        for row in bp["studies"]:
            if row["kind"] != "band":
                continue
            xs = np.arange(1, len(row["exact"]) + 1)
            ax.plot(
                xs, row["exact"], color=band_colors[row["name"]], lw=2.2, marker="o", ms=3.5,
                label=row["name"],
            )
        ax.axvline(int(cfg["train_segments"]), color=BLACK, lw=1.0, ls=":")
    ax.set_xlabel("Descent segments")
    ax.set_ylabel("Exact-board accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("D  Rating bands, seed 0")
    ax.legend(prop={"weight": "bold", "size": 9}, loc="lower right")

    fig.suptitle(
        f"EBR Sudoku-Extreme, 96 hardest boards  ·  {status['status']}",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(EXP / "plot.png", dpi=160)
    fig.savefig(EXP / "plot.pdf")
    plt.close(fig)


def write_summary(cfg: dict, raw: dict, status: dict) -> None:
    runs = raw["runs"]
    depth = int(cfg["eval_segments"])
    acc = _acc_matrix(runs, depth)
    summary = {
        "status": status["status"],
        "bars": status["bars"],
        "n_bars_passed": status["n_bars_passed"],
        "n_boards": int(cfg["n_boards"]),
        "train_segments": int(cfg["train_segments"]),
        "segments": list(range(1, depth + 1)),
        "exact_mean": acc.mean(axis=0).tolist(),
        "exact_std": acc.std(axis=0, ddof=1).tolist(),
        "gain_16_minus_4": status["gain_16_minus_4"]["mean"],
        "energy_auroc_mean": status["energy_auroc_mean"],
        "thresholds": cfg["thresholds"],
        "example": raw["example"],
        "blank_token": int(cfg["blank_token"]),
        "ratings": {
            "min": min(raw["hard_ratings"]),
            "max": max(raw["hard_ratings"]),
        },
    }
    text = json.dumps(summary)
    (EXP / "summary.json").write_text(text)
    viewer = ROOT / "viewer"
    viewer.mkdir(parents=True, exist_ok=True)
    (viewer / "summary.json").write_text(text)
    (EXP / "status.json").write_text(json.dumps(status, indent=2))


def main() -> None:
    cfg, raw, bp = _load()
    thresholds = cfg["thresholds"]
    status = bars_from_runs(
        raw["runs"],
        n_boards=int(cfg["n_boards"]),
        min_class=int(cfg["min_class"]),
        gain_min=float(thresholds["gain_16_minus_4_min"]),
        regression_boards_max=int(thresholds["regression_boards_max"]),
        auroc_min=float(thresholds["energy_auroc_min"]),
        gtan_ratio_min=float(thresholds["gtan_ratio_min"]),
    )
    write_tables(cfg, raw, status)
    write_breakpoints(bp)
    make_plot(cfg, raw, status, bp)
    write_summary(cfg, raw, status)
    print(f"status {status['status']} bars {status['bars']}")


if __name__ == "__main__":
    main()
