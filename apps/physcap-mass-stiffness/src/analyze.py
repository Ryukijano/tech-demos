"""Tables, the PASS rule, and the figure. Reads raw.json. Does not resimulate."""

from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path
from typing import Any

import atomic

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_probes"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Liberation Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "figure.dpi": 200,
        "savefig.dpi": 220,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
    }
)

BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
SKY = "#56B4E9"
BLACK = "#222222"


def _bold_legend(ax: plt.Axes, **kwargs: Any) -> Any:
    legend = ax.legend(**kwargs)
    if legend is None:
        return None
    for text in legend.get_texts():
        text.set_fontweight("bold")
    title = legend.get_title()
    if title is not None:
        title.set_fontweight("bold")
    return legend


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else float("nan")
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def _seed_mass(rows: list[dict[str, Any]], seed: int) -> dict[str, float]:
    picked = [row for row in rows if int(row["seed"]) == seed]
    err = np.array([abs(float(row["m_hat"]) - float(row["mass_kg"])) for row in picked], dtype=np.float64)
    rel = np.array(
        [abs(float(row["m_hat"]) - float(row["mass_kg"])) / float(row["mass_kg"]) for row in picked],
        dtype=np.float64,
    )
    heavy = [
        abs(float(row["m_hat"]) - float(row["mass_kg"])) / float(row["mass_kg"])
        for row in picked
        if float(row["mass_kg"]) >= 0.150
    ]
    grasp = float(np.mean([1.0 if row["grasped"] else 0.0 for row in picked]))
    return {
        "median_abs": float(np.median(err)),
        "median_rel": float(np.median(rel)),
        "median_rel_heavy": float(np.median(heavy)) if heavy else float("nan"),
        "grasp": grasp,
    }


def _seed_stiff(rows: list[dict[str, Any]], seed: int) -> dict[str, float]:
    picked = [row for row in rows if int(row["seed"]) == seed]
    acc = float(np.mean([1.0 if row["correct"] else 0.0 for row in picked]))
    ripe = [row for row in picked if int(row["true_level"]) == 2]
    unripe = [row for row in picked if int(row["true_level"]) in (4, 5)]
    ripe_hit = float(np.mean([1.0 if int(row["level"]) in (1, 2) else 0.0 for row in ripe])) if ripe else float("nan")
    unripe_hit = float(np.mean([1.0 if int(row["level"]) in (4, 5) else 0.0 for row in unripe])) if unripe else float("nan")
    by_level = {}
    for level in range(1, 6):
        group = [row for row in picked if int(row["true_level"]) == level]
        by_level[level] = float(np.mean([1.0 if row["correct"] else 0.0 for row in group])) if group else float("nan")
    return {"accuracy": acc, "ripe": ripe_hit, "unripe": unripe_hit, "by_level": by_level}


def _seed_explore(rows: list[dict[str, Any]], seed: int, policy: str) -> dict[str, float]:
    picked = [row for row in rows if int(row["seed"]) == seed and row["policy"] == policy]
    return {
        "oi": float(np.mean([row["probes"] for row in picked])),
        "success": float(np.mean([1.0 if row["success"] else 0.0 for row in picked])),
    }


def _judge(summary: dict[str, float], thresholds: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    mass_pass = summary["mass_median_abs"] <= float(thresholds["mass_median_abs_err_kg"]) and summary["grasp"] >= float(
        thresholds["grasp_success_min"]
    )
    stiff_pass = (
        summary["stiffness_accuracy"] >= float(thresholds["stiffness_accuracy_min"])
        and summary["ripe"] >= float(thresholds["ripe_recall_min"])
        and summary["unripe"] >= float(thresholds["unripe_recall_min"])
    )
    expl_pass = (
        summary["selective_oi"] <= float(thresholds["selective_oi_max"])
        and summary["selective_oi"] <= float(thresholds["selective_oi_ratio_max"]) * summary["exhaustive_oi"]
        and summary["selective_success"] >= float(thresholds["selective_success_min"])
    )
    mass_part = summary["mass_median_abs"] <= float(thresholds["mass_partial_median_abs_err_kg"]) and summary["grasp"] >= float(
        thresholds["grasp_success_partial_min"]
    )
    stiff_part = (
        summary["stiffness_accuracy"] >= float(thresholds["stiffness_accuracy_partial_min"])
        and summary["ripe"] >= float(thresholds["separation_partial_min"])
        and summary["unripe"] >= float(thresholds["separation_partial_min"])
    )
    expl_part = summary["selective_oi"] < summary["exhaustive_oi"] and summary["selective_success"] >= float(
        thresholds["selective_success_partial_min"]
    )
    checks = {
        "mass_median_abs": mass_pass,
        "stiffness": stiff_pass,
        "explore": expl_pass,
        "mass_partial": mass_part,
        "stiffness_partial": stiff_part,
        "explore_partial": expl_part,
    }
    if mass_pass and stiff_pass and expl_pass:
        return "PASS", checks
    if mass_part and stiff_part and expl_part:
        return "PARTIAL", checks
    return "FAIL", checks


def _group_mean(rows: list[dict[str, Any]], key: str, value: str) -> list[tuple[float, float, float]]:
    keys = sorted({float(row[key]) for row in rows})
    out = []
    for item in keys:
        picked = [float(row[value]) for row in rows if float(row[key]) == item]
        mean, std = _mean_std(picked)
        out.append((item, mean, std))
    return out


def _fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _write_tables(raw: dict[str, Any], summary: dict[str, Any], status: str, out: Path) -> None:
    seeds = [int(s) for s in raw["seeds"]]
    lines = [
        f"# exp1_cpu_probes — {status}",
        "",
        "Pooled median absolute mass error counts a failed grasp as −1 kg.",
        "Stiffness and exploration numbers are means of the per-seed rates.",
        "",
        "| Seed | Median \\|Δm\\| (g) | Grasp | Stiffness acc. | Ripe in 1–2 | Unripe in 4–5 | Selective OI | Exhaustive OI | Selective success |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        mass = summary["per_seed"][seed]["mass"]
        stiff = summary["per_seed"][seed]["stiffness"]
        sel = summary["per_seed"][seed]["selective"]
        exh = summary["per_seed"][seed]["exhaustive"]
        lines.append(
            f"| {seed} | {mass['median_abs'] * 1000:.1f} | {mass['grasp']:.3f} | {stiff['accuracy']:.3f} | "
            f"{stiff['ripe']:.3f} | {stiff['unripe']:.3f} | {sel['oi']:.2f} | {exh['oi']:.2f} | {sel['success']:.3f} |"
        )
    lines += [
        "",
        f"Pooled median \\|Δm\\| = {summary['mass_median_abs'] * 1000:.2f} g "
        f"(bar 40 g, a median, not the paper's mean). "
        f"Mean \\|Δm\\| = {summary['mass_mean_abs'] * 1000:.2f} g, reported and not gated. "
        f"Grasp success = {summary['grasp']:.3f}.",
        f"Median relative error = {summary['mass_median_rel']:.3f}. "
        f"On masses ≥ 150 g, median relative error = {summary['mass_median_rel_heavy']:.3f}.",
        f"Stiffness accuracy = {summary['stiffness_accuracy']:.3f}. "
        f"Ripe recall = {summary['ripe']:.3f}. Unripe recall = {summary['unripe']:.3f}.",
        f"Selective OI = {summary['selective_oi']:.3f}. Exhaustive OI = {summary['exhaustive_oi']:.3f}. "
        f"Ratio = {summary['oi_ratio']:.3f}. Selective success = {summary['selective_success']:.3f}.",
        "",
    ]
    atomic.write_text(out / "results.md", "\n".join(lines))

    buffer = io.StringIO(newline="\n")
    writer = csv.DictWriter(
        buffer,
        lineterminator="\n",
        fieldnames=[
            "seed",
            "median_abs_g",
            "grasp",
            "stiffness_accuracy",
            "ripe",
            "unripe",
            "selective_oi",
            "exhaustive_oi",
            "selective_success",
        ],
    )
    writer.writeheader()
    for seed in seeds:
        mass = summary["per_seed"][seed]["mass"]
        stiff = summary["per_seed"][seed]["stiffness"]
        sel = summary["per_seed"][seed]["selective"]
        exh = summary["per_seed"][seed]["exhaustive"]
        writer.writerow(
            {
                "seed": seed,
                "median_abs_g": f"{mass['median_abs'] * 1000:.4f}",
                "grasp": f"{mass['grasp']:.4f}",
                "stiffness_accuracy": f"{stiff['accuracy']:.4f}",
                "ripe": f"{stiff['ripe']:.4f}",
                "unripe": f"{stiff['unripe']:.4f}",
                "selective_oi": f"{sel['oi']:.4f}",
                "exhaustive_oi": f"{exh['oi']:.4f}",
                "selective_success": f"{sel['success']:.4f}",
            }
        )
    atomic.write_text(out / "results.csv", buffer.getvalue())


def _break_lines(raw: dict[str, Any]) -> list[str]:
    bp = raw["breakpoints"]
    lines = [
        "# Breakpoints",
        "",
        "Seed 0 families from `config.yaml`. Not part of the PASS rule.",
        "",
        "## Very light",
        "",
        "| Mass (g) | Mean estimate (g) | Grasp |",
        "|---:|---:|---:|",
    ]
    for mass, mean, _std in _group_mean(bp["very_light"], "mass_kg", "m_hat"):
        grasp = float(np.mean([1.0 if row["grasped"] else 0.0 for row in bp["very_light"] if float(row["mass_kg"]) == mass]))
        lines.append(f"| {mass * 1000:.1f} | {mean * 1000:.2f} | {grasp:.2f} |")
    lines += ["", "## Very heavy", "", "| Mass (kg) | Mean estimate (kg) | Grasp |", "|---:|---:|---:|"]
    for mass, mean, _std in _group_mean(bp["very_heavy"], "mass_kg", "m_hat"):
        grasp = float(np.mean([1.0 if row["grasped"] else 0.0 for row in bp["very_heavy"] if float(row["mass_kg"]) == mass]))
        lines.append(f"| {mass:.1f} | {mean:.3f} | {grasp:.2f} |")
    lines += ["", "## Torque noise, 0.208 kg can", "", "| Noise (N·m) | Mean estimate (kg) |", "|---:|---:|"]
    for std, mean, stdev in _group_mean(bp["torque_noise"], "std_nm", "m_hat"):
        lines.append(f"| {std:.2f} | {mean:.3f} ± {stdev:.3f} |")
    lines += ["", "## Near-rigid springs", "", "| k (N/m) | Majority levels |", "|---:|---|"]
    ks = sorted({float(row["k"]) for row in bp["near_rigid"]})
    for k in ks:
        levels = [str(int(row["level"])) for row in bp["near_rigid"] if float(row["k"]) == k]
        lines.append(f"| {k:.0f} | {', '.join(levels)} |")
    lines += ["", "## Finger friction, 0.40 kg", "", "| μ | Mean estimate (kg) | Grasp |", "|---:|---:|---:|"]
    for mu, mean, _std in _group_mean(bp["friction_slip"], "mu", "m_hat"):
        grasp = float(np.mean([1.0 if row["grasped"] else 0.0 for row in bp["friction_slip"] if float(row["mu"]) == mu]))
        lines.append(f"| {mu:.2f} | {mean:.3f} | {grasp:.2f} |")
    lines += ["", "## Jaw frictionloss, k = 1050 N/m (level 3)", "", "| Frictionloss (N) | Levels | Median slope (N/m) |", "|---:|---|---:|"]
    losses = sorted({float(row["frictionloss_n"]) for row in bp["jaw_friction"]})
    for loss in losses:
        group = [row for row in bp["jaw_friction"] if float(row["frictionloss_n"]) == loss]
        levels = ", ".join(str(int(row["level"])) for row in group)
        slopes = [float(row["median_slope"]) for row in group if np.isfinite(row["median_slope"])]
        slope = float(np.median(slopes)) if slopes else float("nan")
        lines.append(f"| {loss:.1f} | {levels} | {_fmt(slope, 1)} |")
    lines.append("")
    return lines


def _save_fig(fig: plt.Figure, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp{path.suffix}")
    fig.savefig(tmp)
    atomic.replace_file(tmp, path)


def _write_breakpoints(raw: dict[str, Any], out: Path) -> None:
    lines = _break_lines(raw)
    atomic.write_text(out / "breakpoints.md", "\n".join(lines))
    flat: list[dict[str, str]] = []
    for family, rows in raw["breakpoints"].items():
        for row in rows:
            flat.append({"family": family, **{key: str(value) for key, value in row.items()}})
    keys: list[str] = []
    for row in flat:
        for key in row:
            if key not in keys:
                keys.append(key)
    buffer = io.StringIO(newline="\n")
    writer = csv.DictWriter(buffer, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(flat)
    atomic.write_text(out / "breakpoints.csv", buffer.getvalue())


def _plot(raw: dict[str, Any], summary: dict[str, Any], status: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.55))
    seeds = [int(s) for s in raw["seeds"]]
    colors = {0: BLUE, 1: VERM, 2: GREEN}
    ax = axes[0]
    ax.axhline(40.0, color=BLUE, linestyle="--", linewidth=1.1, label="±40 g bar")
    ax.axhline(-40.0, color=BLUE, linestyle="--", linewidth=1.1)
    ax.axhline(0.0, color=BLACK, linewidth=1.0, label="Zero error")
    for seed in seeds:
        rows = [row for row in raw["mass"] if int(row["seed"]) == seed]
        ax.scatter(
            [float(row["mass_kg"]) for row in rows],
            [(float(row["m_hat"]) - float(row["mass_kg"])) * 1000.0 for row in rows],
            s=22,
            color=colors[seed],
            alpha=0.9,
            label=f"Seed {seed}",
            zorder=3,
        )
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("True mass (kg)")
    ax.set_ylabel("Estimate − true (g)")
    ax.set_title(f"Lift torque  ·  median |Δm| {summary['mass_median_abs'] * 1000:.1f} g")
    _bold_legend(ax, fontsize=8, frameon=True, loc="upper right")

    ax = axes[1]
    levels = np.arange(1, 6)
    means = []
    stds = []
    for level in levels:
        per_seed = [summary["per_seed"][seed]["stiffness"]["by_level"][int(level)] for seed in seeds]
        mean, std = _mean_std(per_seed)
        means.append(mean)
        stds.append(std)
    ax.bar(levels, means, yerr=stds, color=SKY, edgecolor=BLACK, linewidth=0.4, capsize=3, label="Majority vote", width=0.72)
    ax.axhline(0.80, color=VERM, linestyle="--", linewidth=1.2, label="0.80 bar")
    ax.set_xticks(levels)
    ax.set_xticklabels(["1", "2", "3", "4", "5"])
    ax.set_xlabel("Level (1 Ultra-Soft … 5 Rigid)")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Classification accuracy")
    ax.set_title(f"Squeeze fit  ·  acc {summary['stiffness_accuracy']:.2f}")
    _bold_legend(ax, fontsize=8, frameon=True, loc="lower right")

    ax = axes[2]
    sel = [summary["per_seed"][seed]["selective"]["oi"] for seed in seeds]
    exh = [summary["per_seed"][seed]["exhaustive"]["oi"] for seed in seeds]
    sel_m, sel_s = _mean_std(sel)
    exh_m, exh_s = _mean_std(exh)
    xpos = np.array([0, 1])
    ax.bar(
        xpos,
        [sel_m, exh_m],
        yerr=[sel_s, exh_s],
        color=[GREEN, ORANGE],
        edgecolor=BLACK,
        linewidth=0.4,
        capsize=3,
        width=0.62,
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(["Selective", "Exhaustive"])
    ax.set_ylabel("Mass probes per episode")
    ax.set_ylim(0, max(5.5, exh_m + 1.0))
    ax.set_title(f"Empty can  ·  success {summary['selective_success']:.2f}")
    _bold_legend(
        ax,
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=GREEN, label="Stop when confident"),
            plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="Probe every can"),
        ],
        fontsize=8,
        frameon=True,
        loc="upper left",
    )
    fig.suptitle(f"PhysCaP-style probes on CPU  ·  {status}", fontweight="bold", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, out / "plot.png")
    _save_fig(fig, out / "plot.pdf")
    plt.close(fig)


def main() -> None:
    with (EXP / "config.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    raw = json.loads((EXP / "raw.json").read_text())
    seeds = [int(s) for s in raw["seeds"]]
    per_seed = {}
    for seed in seeds:
        per_seed[seed] = {
            "mass": _seed_mass(raw["mass"], seed),
            "stiffness": _seed_stiff(raw["stiffness"], seed),
            "selective": _seed_explore(raw["explore"], seed, "selective"),
            "exhaustive": _seed_explore(raw["explore"], seed, "exhaustive"),
        }
    all_err = np.array([abs(float(row["m_hat"]) - float(row["mass_kg"])) for row in raw["mass"]], dtype=np.float64)
    all_rel = np.array(
        [abs(float(row["m_hat"]) - float(row["mass_kg"])) / float(row["mass_kg"]) for row in raw["mass"]],
        dtype=np.float64,
    )
    heavy_rel = [
        abs(float(row["m_hat"]) - float(row["mass_kg"])) / float(row["mass_kg"])
        for row in raw["mass"]
        if float(row["mass_kg"]) >= 0.150
    ]
    summary: dict[str, Any] = {
        "per_seed": per_seed,
        "mass_median_abs": float(np.median(all_err)),
        "mass_mean_abs": float(np.mean(all_err)),
        "mass_median_rel": float(np.median(all_rel)),
        "mass_median_rel_heavy": float(np.median(heavy_rel)),
        "grasp": float(np.mean([per_seed[seed]["mass"]["grasp"] for seed in seeds])),
        "stiffness_accuracy": float(np.mean([per_seed[seed]["stiffness"]["accuracy"] for seed in seeds])),
        "ripe": float(np.mean([per_seed[seed]["stiffness"]["ripe"] for seed in seeds])),
        "unripe": float(np.mean([per_seed[seed]["stiffness"]["unripe"] for seed in seeds])),
        "selective_oi": float(np.mean([per_seed[seed]["selective"]["oi"] for seed in seeds])),
        "exhaustive_oi": float(np.mean([per_seed[seed]["exhaustive"]["oi"] for seed in seeds])),
        "selective_success": float(np.mean([per_seed[seed]["selective"]["success"] for seed in seeds])),
    }
    summary["oi_ratio"] = summary["selective_oi"] / summary["exhaustive_oi"]
    status, checks = _judge(summary, cfg["thresholds"])
    public = {
        "status": status,
        "checks": checks,
        "mass_median_abs_kg": summary["mass_median_abs"],
        "mass_mean_abs_kg": summary["mass_mean_abs"],
        "mass_median_rel": summary["mass_median_rel"],
        "mass_median_rel_at_least_150g": summary["mass_median_rel_heavy"],
        "grasp_success": summary["grasp"],
        "stiffness_accuracy": summary["stiffness_accuracy"],
        "ripe_recall": summary["ripe"],
        "unripe_recall": summary["unripe"],
        "selective_oi": summary["selective_oi"],
        "exhaustive_oi": summary["exhaustive_oi"],
        "oi_ratio": summary["oi_ratio"],
        "selective_success": summary["selective_success"],
        "per_seed": {
            str(seed): {
                "median_abs_kg": per_seed[seed]["mass"]["median_abs"],
                "grasp": per_seed[seed]["mass"]["grasp"],
                "stiffness_accuracy": per_seed[seed]["stiffness"]["accuracy"],
                "selective_oi": per_seed[seed]["selective"]["oi"],
                "selective_success": per_seed[seed]["selective"]["success"],
            }
            for seed in seeds
        },
    }
    names = [
        "status.json",
        "results.md",
        "results.csv",
        "breakpoints.md",
        "breakpoints.csv",
        "plot.png",
        "plot.pdf",
    ]
    with tempfile.TemporaryDirectory(prefix=".analyze-", dir=EXP) as folder:
        staging = Path(folder)
        atomic.write_text(staging / "status.json", json.dumps(public, indent=2) + "\n")
        _write_tables(raw, summary, status, staging)
        _write_breakpoints(raw, staging)
        _plot(raw, summary, status, staging)
        atomic.publish(staging, EXP, names)
    print(json.dumps({"status": status, **{k: public[k] for k in public if k != "per_seed"}}, indent=2))


if __name__ == "__main__":
    main()
