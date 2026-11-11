"""Turn raw.json into the status, tables, and the result figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_emeans"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Liberation Serif", "Times"],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {"mixture": "#0072B2", "stretched": "#D55E00", "mnist": "#009E73"}


def _bold_legend(ax: plt.Axes, **kwargs: Any) -> None:
    legend = ax.legend(**kwargs)
    for text in legend.get_texts():
        text.set_fontweight("bold")


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def _fmt(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _grouped(raw: dict[str, Any], role: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in raw["runs"]:
        if run["role"] != role:
            continue
        groups.setdefault(run["name"], []).append(run)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["seed"]))
    return groups


def _dataset_stats(runs: list[dict[str, Any]], gate: str, budgets: list[str]) -> dict[str, Any]:
    ratios = [float(run["emeans"][gate]["train_mse_ratio"]) for run in runs]
    walls = [float(run["emeans"][gate]["wall_ratio"]) for run in runs]
    lloyd = [float(run["lloyd"]["train_mse"]) for run in runs]
    emean = [float(run["emeans"][gate]["train_mse"]) for run in runs]
    lloyd_test = [float(run["lloyd"]["test_mse"]) for run in runs]
    emean_test = [float(run["emeans"][gate]["test_mse"]) for run in runs]
    lloyd_s = [float(run["lloyd"]["seconds"]) for run in runs]
    emean_s = [float(run["emeans"][gate]["seconds"]) for run in runs]
    ratio_mean, ratio_std = _mean_std(ratios)
    wall_mean, wall_std = _mean_std(walls)
    curves = {}
    for budget in budgets:
        curve_ratios = [float(run["emeans"][budget]["train_mse_ratio"]) for run in runs]
        curve_seconds = [float(run["emeans"][budget]["seconds"]) for run in runs]
        curves[budget] = {
            "train_mse_ratio_mean": _mean_std(curve_ratios)[0],
            "train_mse_ratio_std": _mean_std(curve_ratios)[1],
            "seconds_mean": _mean_std(curve_seconds)[0],
            "seconds_std": _mean_std(curve_seconds)[1],
        }
    stats = {
        "n": int(runs[0]["n"]),
        "d": int(runs[0]["d"]),
        "k": int(runs[0]["k"]),
        "train_mse_ratio_mean": ratio_mean,
        "train_mse_ratio_std": ratio_std,
        "wall_ratio_mean": wall_mean,
        "wall_ratio_std": wall_std,
        "lloyd_train_mse_mean": _mean_std(lloyd)[0],
        "lloyd_train_mse_std": _mean_std(lloyd)[1],
        "emeans_train_mse_mean": _mean_std(emean)[0],
        "emeans_train_mse_std": _mean_std(emean)[1],
        "lloyd_test_mse_mean": _mean_std(lloyd_test)[0],
        "lloyd_test_mse_std": _mean_std(lloyd_test)[1],
        "emeans_test_mse_mean": _mean_std(emean_test)[0],
        "emeans_test_mse_std": _mean_std(emean_test)[1],
        "lloyd_seconds_mean": _mean_std(lloyd_s)[0],
        "lloyd_seconds_std": _mean_std(lloyd_s)[1],
        "emeans_seconds_mean": _mean_std(emean_s)[0],
        "emeans_seconds_std": _mean_std(emean_s)[1],
        "curves": curves,
        "per_seed": [
            {
                "seed": int(run["seed"]),
                "train_mse_ratio": float(run["emeans"][gate]["train_mse_ratio"]),
                "wall_ratio": float(run["emeans"][gate]["wall_ratio"]),
                "lloyd_train_mse": float(run["lloyd"]["train_mse"]),
                "emeans_train_mse": float(run["emeans"][gate]["train_mse"]),
                "batch_size": int(run["emeans"][gate]["batch_size"]),
                "alpha0": float(run["emeans"][gate]["alpha0"]),
                "batch_law": run["emeans"][gate]["batch_law"],
                "train_purity_lloyd": run["lloyd"].get("train_purity"),
                "train_purity_emeans": run["emeans"][gate].get("train_purity"),
            }
            for run in runs
        ],
    }
    return stats


def _status(flags: dict[str, dict[str, bool]]) -> str:
    mse_ok = all(item["mse"] for item in flags.values())
    wall_ok = all(item["wall"] for item in flags.values())
    if mse_ok and wall_ok:
        return "PASS"
    if mse_ok and not wall_ok:
        return "PARTIAL"
    return "FAIL"


def _plot(stats: dict[str, Any], status: str, mse_bar: float, wall_bar: float, path_png: Path, path_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.35))
    ax = axes[0]
    for name, block in stats.items():
        budgets = sorted(block["curves"], key=float)
        xs = np.array([float(budget) for budget in budgets])
        means = np.array([block["curves"][budget]["train_mse_ratio_mean"] for budget in budgets])
        stds = np.array([block["curves"][budget]["train_mse_ratio_std"] for budget in budgets])
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            marker="o",
            linewidth=1.6,
            capsize=2,
            color=COLORS.get(name, "#333333"),
            label=name,
        )
    ax.axhline(1.0, color="#444444", linewidth=0.8, linestyle="--", label="Lloyd @ 25")
    ax.axhline(mse_bar, color="#888888", linewidth=0.8, linestyle=":", label=f"{mse_bar:.2f} bar")
    ax.set_xlabel("e-means epochs (separate runs)")
    ax.set_ylabel("Train MSE / Lloyd @ 25")
    ax.set_title(f"Objective ratio · {status}")
    ax.set_xticks([1, 2, 2.5, 5])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _bold_legend(ax, frameon=False, loc="best")

    ax = axes[1]
    names = list(stats)
    x = np.arange(len(names))
    width = 0.36
    lloyd = [stats[name]["lloyd_seconds_mean"] for name in names]
    lloyd_err = [stats[name]["lloyd_seconds_std"] for name in names]
    emean = [stats[name]["emeans_seconds_mean"] for name in names]
    emean_err = [stats[name]["emeans_seconds_std"] for name in names]
    ax.bar(x - width / 2, lloyd, width, yerr=lloyd_err, color="#D55E00", label="Lloyd 25 iter", capsize=2)
    ax.bar(x + width / 2, emean, width, yerr=emean_err, color="#0072B2", label="e-means 2.5 ep", capsize=2)
    ax.set_xticks(x, names)
    ax.set_ylabel("Train wall-clock (s)")
    ax.set_title("CPU time")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _bold_legend(ax, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path_png, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def _write_tables(cfg: dict[str, Any], stats: dict[str, Any], flags: dict[str, dict[str, bool]], status: str, breakpoints: list[dict[str, Any]], control: dict[str, Any] | None) -> None:
    gate = str(cfg["gate_epochs"])
    lines = [
        f"# exp1 CPU e-means vs Lloyd — {status}",
        "",
        f"Primary metric: train MSE ratio at {gate} epochs versus FAISS Lloyd at {cfg['lloyd_iterations']} iterations.",
        f"Bar: ratio ≤ {cfg['thresholds']['train_mse_ratio_max']} and wall-clock ratio ≤ {cfg['thresholds']['wall_time_ratio_max']}.",
        "Mean ± sample std (ddof=1) over seeds 0, 1, 2.",
        "",
        "| dataset | n, d, k | Lloyd train MSE | e-means train MSE | ratio | wall ratio | MSE bar | time bar |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for name, block in stats.items():
        lines.append(
            "| {name} | {n}, {d}, {k} | {lloyd} | {emeans} | {ratio} | {wall} | {mse} | {time} |".format(
                name=name,
                n=block["n"],
                d=block["d"],
                k=block["k"],
                lloyd=_fmt(block["lloyd_train_mse_mean"], block["lloyd_train_mse_std"]),
                emeans=_fmt(block["emeans_train_mse_mean"], block["emeans_train_mse_std"]),
                ratio=_fmt(block["train_mse_ratio_mean"], block["train_mse_ratio_std"]),
                wall=_fmt(block["wall_ratio_mean"], block["wall_ratio_std"], 3),
                mse="met" if flags[name]["mse"] else "miss",
                time="met" if flags[name]["wall"] else "miss",
            )
        )
    lines.extend(["", "Test MSE is report-only.", "", "| dataset | Lloyd test MSE | e-means test MSE |", "|---|---:|---:|"])
    for name, block in stats.items():
        lines.append(
            f"| {name} | {_fmt(block['lloyd_test_mse_mean'], block['lloyd_test_mse_std'])} | {_fmt(block['emeans_test_mse_mean'], block['emeans_test_mse_std'])} |"
        )
    (EXP / "results.md").write_text("\n".join(lines) + "\n")

    with (EXP / "results.csv").open("w", newline="") as handle:
        fieldnames = [
            "dataset",
            "seed",
            "train_mse_ratio",
            "wall_ratio",
            "lloyd_train_mse",
            "emeans_train_mse",
            "batch_size",
            "alpha0",
            "batch_law",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, block in stats.items():
            for row in block["per_seed"]:
                writer.writerow({"dataset": name, **{key: row[key] for key in fieldnames if key != "dataset"}})

    bp_lines = [
        "# Breakpoints (seed 0, not in the PASS rule)",
        "",
        "| case | n, d, k | train MSE ratio @ 2.5 | wall ratio |",
        "|---|---|---:|---:|",
    ]
    for row in breakpoints:
        bp_lines.append(
            f"| {row['name']} | {row['n']}, {row['d']}, {row['k']} | {row['train_mse_ratio']:.4f} | {row['wall_ratio']:.3f} |"
        )
    if control is not None:
        full = control["full_train_mse"]
        default = control["default_train_mse"]
        bp_lines.extend(
            [
                "",
                "Library-default caps on the mixture, seed 0, same init, 25 Lloyd iterations.",
                f"Full-data train MSE {full:.6f}. Default max 256 / min 39 train MSE {default:.6f}.",
                f"Default / full = {default / full:.4f}.",
            ]
        )
    (EXP / "breakpoints.md").write_text("\n".join(bp_lines) + "\n")
    with (EXP / "breakpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "n", "d", "k", "train_mse_ratio", "wall_ratio", "lloyd_seconds", "emeans_seconds"])
        writer.writeheader()
        writer.writerows(breakpoints)


def main() -> None:
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    raw = json.loads((EXP / "raw.json").read_text())
    gate = str(cfg["gate_epochs"])
    budgets = [str(item) for item in cfg["epoch_budgets"]]
    mse_bar = float(cfg["thresholds"]["train_mse_ratio_max"])
    wall_bar = float(cfg["thresholds"]["wall_time_ratio_max"])
    grouped = _grouped(raw, "gated")
    stats = {name: _dataset_stats(runs, gate, budgets) for name, runs in grouped.items()}
    flags = {
        name: {
            "mse": block["train_mse_ratio_mean"] <= mse_bar,
            "wall": block["wall_ratio_mean"] <= wall_bar,
        }
        for name, block in stats.items()
    }
    status = _status(flags)
    breakpoints = []
    for run in sorted((row for row in raw["runs"] if row["role"] == "breakpoint"), key=lambda row: row["name"]):
        gate_row = run["emeans"][gate]
        breakpoints.append(
            {
                "name": run["name"],
                "n": run["n"],
                "d": run["d"],
                "k": run["k"],
                "train_mse_ratio": float(gate_row["train_mse_ratio"]),
                "wall_ratio": float(gate_row["wall_ratio"]),
                "lloyd_seconds": float(run["lloyd"]["seconds"]),
                "emeans_seconds": float(gate_row["seconds"]),
            }
        )
    control_runs = [row for row in raw["runs"] if row["role"] == "control"]
    control = None
    if control_runs:
        control_run = control_runs[0]
        full = next(
            row
            for row in raw["runs"]
            if row["role"] == "gated" and row["name"] == cfg["control"]["dataset"] and int(row["seed"]) == int(cfg["control"]["seed"])
        )
        control = {
            "full_train_mse": float(full["lloyd"]["train_mse"]),
            "default_train_mse": float(control_run["lloyd"]["train_mse"]),
            "default_seconds": float(control_run["lloyd"]["seconds"]),
        }
    payload = {
        "status": status,
        "primary_metric": cfg["metrics"]["primary"],
        "gate_epochs": gate,
        "thresholds": cfg["thresholds"],
        "datasets": stats,
        "flags": flags,
        "breakpoints": breakpoints,
        "control": control,
        "faiss": raw.get("faiss"),
        "config_sha256": raw.get("config_sha256"),
    }
    (EXP / "status.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_tables(cfg, stats, flags, status, breakpoints, control)
    _plot(stats, status, mse_bar, wall_bar, EXP / "plot.png", EXP / "plot.pdf")
    print(f"status {status}")


if __name__ == "__main__":
    main()
