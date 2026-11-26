"""Score the pre-registered bars and write the table, status, and figure."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

SRC = Path(__file__).resolve().parent
APP = SRC.parent
EXP = APP / "experiments" / "exp1_cpu_addition"
ART = APP / "artifacts"

ONLINE = "#D55E00"
ML = "#0072B2"
NAIVE = "#CC79A7"
LASTK = "#009E73"
SHUFFLE = "#666666"
REVERSE = "#56B4E9"


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std


def _traj_mae(student: list[float], online: list[float]) -> float:
    left = np.asarray(student[1:], dtype=np.float64)
    right = np.asarray(online[1:], dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise RuntimeError("curve length mismatch")
    return float(np.mean(np.abs(left - right)))


def _bold_legend(ax: plt.Axes, **kwargs: Any) -> None:
    legend = ax.legend(**kwargs)
    for text in legend.get_texts():
        text.set_fontweight("bold")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Liberation Serif"],
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "font.size": 9,
            "axes.linewidth": 0.8,
        }
    )


def score(raw: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    seeds = [str(s) for s in cfg["seeds"]]
    rows = [raw["seeds"][s] for s in seeds]
    bases = [float(row["online"][0]) for row in rows]
    if max(bases) - min(bases) > 1e-6:
        raise RuntimeError(f"base reward is not shared across seeds: {bases}")
    online_f = [float(row["online"][-1]) for row in rows]
    ml_f = [float(row["ml"][-1]) for row in rows]
    naive_f = [float(row["naive"][-1]) for row in rows]
    last_f = [float(row["lastk"][-1]) for row in rows]
    gains = [fin - base for fin, base in zip(online_f, bases, strict=True)]
    online_gain, online_gain_std = _mean_std(gains)
    ml_mean, ml_std = _mean_std(ml_f)
    online_mean, online_std = _mean_std(online_f)
    naive_mean, naive_std = _mean_std(naive_f)
    last_mean, last_std = _mean_std(last_f)
    base_mean, base_std = _mean_std(bases)
    ml_end_gap = abs(ml_mean - online_mean)
    lastk_end_gap = abs(last_mean - online_mean)
    naive_collapse = online_mean - naive_mean
    traj = [_traj_mae(row["ml"], row["online"]) for row in rows]
    naive_traj = [_traj_mae(row["naive"], row["online"]) for row in rows]
    ml_traj_mae, ml_traj_std = _mean_std(traj)
    naive_traj_mae, _naive_traj_std = _mean_std(naive_traj)
    th = cfg["thresholds"]
    bars = {
        "online_gain": online_gain >= float(th["online_gain_min"]),
        "ml_end_gap": ml_end_gap <= float(th["ml_end_gap_max"]),
        "ml_traj_mae": ml_traj_mae <= float(th["ml_traj_mae_max"]),
        "naive_collapse": naive_collapse >= float(th["naive_collapse_min"]),
        "lastk_end_gap": lastk_end_gap <= float(th["lastk_end_gap_max"]),
    }
    if all(bars.values()):
        status = "PASS"
    elif bars["online_gain"] and bars["ml_end_gap"]:
        status = "PARTIAL"
    else:
        status = "FAIL"
    per_seed = []
    for seed, row, gain, mae in zip(seeds, rows, gains, traj, strict=True):
        per_seed.append(
            {
                "seed": int(seed),
                "base": float(row["online"][0]),
                "online": float(row["online"][-1]),
                "ml": float(row["ml"][-1]),
                "naive": float(row["naive"][-1]),
                "lastk": float(row["lastk"][-1]),
                "gain": gain,
                "ml_gap": abs(float(row["ml"][-1]) - float(row["online"][-1])),
                "ml_traj_mae": mae,
                "naive_param_gap": float(row["naive_param_gap"]),
                "seconds": float(row["seconds"]),
            }
        )
    return {
        "status": status,
        "bars": bars,
        "online_gain": online_gain,
        "online_gain_std": online_gain_std,
        "base_mean": base_mean,
        "base_std": base_std,
        "online_mean": online_mean,
        "online_std": online_std,
        "ml_mean": ml_mean,
        "ml_std": ml_std,
        "naive_mean": naive_mean,
        "naive_std": naive_std,
        "lastk_mean": last_mean,
        "lastk_std": last_std,
        "ml_end_gap": ml_end_gap,
        "ml_traj_mae": ml_traj_mae,
        "ml_traj_std": ml_traj_std,
        "naive_collapse": naive_collapse,
        "naive_traj_mae": naive_traj_mae,
        "lastk_end_gap": lastk_end_gap,
        "per_seed": per_seed,
    }


def _stack(raw: dict[str, Any], cfg: dict[str, Any], key: str) -> np.ndarray:
    return np.stack([np.asarray(raw["seeds"][str(s)][key], dtype=np.float64) for s in cfg["seeds"]], axis=0)


def _band(ax: plt.Axes, steps: np.ndarray, curves: np.ndarray, color: str, label: str) -> None:
    mean = curves.mean(axis=0)
    std = curves.std(axis=0, ddof=1)
    ax.plot(steps, mean, color=color, lw=1.8, label=label)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)


def _by_name(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in raw["breakpoints"]}


def write_plot(raw: dict[str, Any], cfg: dict[str, Any], metrics: dict[str, Any], path: Path) -> None:
    _style()
    steps = np.arange(_stack(raw, cfg, "online").shape[1])
    online = _stack(raw, cfg, "online")
    ml = _stack(raw, cfg, "ml")
    naive = _stack(raw, cfg, "naive")
    lastk = _stack(raw, cfg, "lastk")
    named = _by_name(raw)
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 6.3), dpi=300)

    ax = axes[0, 0]
    _band(ax, steps, online, ONLINE, "Online REINFORCE")
    _band(ax, steps, ml, ML, "ML replay")
    _band(ax, steps, naive, NAIVE, "Naive RL replay")
    _band(ax, steps, lastk, LASTK, "Last-4 ML")
    ax.axhline(metrics["base_mean"], color="#444444", ls=":", lw=0.8)
    ax.set_xlabel("Update")
    ax.set_ylabel("Exact-match reward")
    ax.set_title("(a) Learning curves")
    _bold_legend(ax, fontsize=7, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.02, 1.02)

    ax = axes[0, 1]
    labels = ["Base", "Online", "ML", "Naive", "Last-4"]
    means = [metrics["base_mean"], metrics["online_mean"], metrics["ml_mean"], metrics["naive_mean"], metrics["lastk_mean"]]
    stds = [metrics["base_std"], metrics["online_std"], metrics["ml_std"], metrics["naive_std"], metrics["lastk_std"]]
    colors = ["#444444", ONLINE, ML, NAIVE, LASTK]
    xpos = np.arange(len(labels))
    ax.bar(xpos, means, yerr=stds, color=colors, width=0.72, capsize=3, error_kw={"lw": 0.8})
    ax.set_xticks(xpos, labels, rotation=0)
    ax.set_ylabel("Final exact-match reward")
    ax.set_title("(b) Endpoint")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 0]
    seed0 = raw["seeds"][str(cfg["breakpoints"]["seed"])]
    x0 = np.arange(len(seed0["online"]))
    ax.plot(x0, seed0["online"], color=ONLINE, lw=1.6, label="Online")
    ax.plot(x0, seed0["ml"], color=ML, lw=1.6, label="Chronological")
    ax.plot(x0, named["shuffle"]["curve"], color=SHUFFLE, lw=1.4, label="Shuffled")
    ax.plot(x0, named["reverse"]["curve"], color=REVERSE, lw=1.4, label="Reverse")
    ax.set_xlabel("Replay update")
    ax.set_ylabel("Exact-match reward")
    ax.set_title("(c) Order, seed 0")
    _bold_legend(ax, fontsize=7, loc="best", frameon=False)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.02, 1.02)

    ax = axes[1, 1]
    sigmas = [0.0, *[float(s) for s in cfg["breakpoints"]["perturbation_std"]]]

    def _perturb(kind: str, sigma: float) -> float:
        for row in named.values():
            if str(row["name"]).startswith(f"perturb-{kind}-") and abs(float(row["sigma"]) - sigma) < 1e-12:
                return float(row["final"])
        raise KeyError(kind, sigma)

    ml_pts = [float(seed0["ml"][-1])]
    naive_pts = [float(seed0["naive"][-1])]
    for sigma in sigmas[1:]:
        ml_pts.append(_perturb("ml", sigma))
        naive_pts.append(_perturb("naive", sigma))
    ax.plot(sigmas, ml_pts, "o-", color=ML, lw=1.6, label="ML replay")
    ax.plot(sigmas, naive_pts, "s-", color=NAIVE, lw=1.6, label="Naive RL")
    ax.axhline(float(seed0["online"][-1]), color=ONLINE, ls="--", lw=1.0, label="Online final")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_xlabel("Init perturbation σ")
    ax.set_ylabel("Final exact-match reward")
    ax.set_title("(d) Drift, seed 0")
    _bold_legend(ax, fontsize=7, loc="best", frameon=False)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def write_tables(raw: dict[str, Any], cfg: dict[str, Any], metrics: dict[str, Any]) -> None:
    th = cfg["thresholds"]
    bar_rows = [
        ("online_gain", f">= {th['online_gain_min']}", metrics["online_gain"], metrics["bars"]["online_gain"]),
        ("ml_end_gap", f"<= {th['ml_end_gap_max']}", metrics["ml_end_gap"], metrics["bars"]["ml_end_gap"]),
        ("ml_traj_mae", f"<= {th['ml_traj_mae_max']}", metrics["ml_traj_mae"], metrics["bars"]["ml_traj_mae"]),
        ("naive_collapse", f">= {th['naive_collapse_min']}", metrics["naive_collapse"], metrics["bars"]["naive_collapse"]),
        ("lastk_end_gap", f"<= {th['lastk_end_gap_max']}", metrics["lastk_end_gap"], metrics["bars"]["lastk_end_gap"]),
    ]
    lines = [
        f"# Results — {metrics['status']}",
        "",
        f"Threshold commit recorded in `raw.json`: `{raw.get('threshold_commit', 'unknown')}`.",
        "",
        "| Metric | Bar | Value | |",
        "|---|---|---:|---|",
    ]
    for name, bar, value, ok in bar_rows:
        lines.append(f"| {name} | {bar} | {value:.4f} | {'met' if ok else 'miss'} |")
    lines += [
        "",
        "| Seed | Base | Online | ML | Naive | Last-4 | Gain | ML traj MAE | Naive ‖Δθ‖∞ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['base']:.3f} | {row['online']:.3f} | {row['ml']:.3f} | "
            f"{row['naive']:.3f} | {row['lastk']:.3f} | {row['gain']:.3f} | {row['ml_traj_mae']:.3f} | "
            f"{row['naive_param_gap']:.2e} |"
        )
    lines += [
        "",
        f"Seed-mean final reward: online {_fmt(metrics['online_mean'], metrics['online_std'])}, "
        f"ML {_fmt(metrics['ml_mean'], metrics['ml_std'])}, "
        f"naive {_fmt(metrics['naive_mean'], metrics['naive_std'])}, "
        f"last-4 {_fmt(metrics['lastk_mean'], metrics['lastk_std'])}.",
        "",
        "## Breakpoints (seed 0 unless noted)",
        "",
        "| Run | Final | Base |",
        "|---|---:|---:|",
    ]
    for row in raw["breakpoints"]:
        lines.append(f"| {row['name']} | {float(row['final']):.3f} | {float(row['base']):.3f} |")
    student = raw["student"]
    sub = raw["sub_online"]
    seed0 = raw["seeds"][str(cfg["breakpoints"]["seed"])]
    lines += [
        "",
        "## Specialists (report-only)",
        "",
        f"Addition specialist final {float(seed0['online'][-1]):.3f}. "
        f"Subtraction specialist {float(sub['curve'][0]):.3f} → {float(sub['curve'][-1]):.3f}. "
        f"Alternating ML student addition {float(student['add_curve'][-1]):.3f}, "
        f"subtraction {float(student['sub_curve'][-1]):.3f}.",
        "",
    ]
    (EXP / "results.md").write_text("\n".join(lines) + "\n")
    with (EXP / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics["per_seed"][0].keys()))
        writer.writeheader()
        writer.writerows(metrics["per_seed"])
    named = _by_name(raw)
    bp_lines = ["# Breakpoints", "", "Not part of PASS / PARTIAL / FAIL.", ""]
    for row in raw["breakpoints"]:
        bp_lines.append(
            f"- **{row['name']}**: final {float(row['final']):.3f} "
            f"(base {float(row['base']):.3f}, repeats {row['repeats']}, {float(row['seconds']):.1f}s)."
        )
    bp_lines += [
        "",
        f"Subtraction online: {float(sub['curve'][0]):.3f} → {float(sub['curve'][-1]):.3f}.",
        f"Student addition / subtraction finals: {float(student['add_curve'][-1]):.3f} / {float(student['sub_curve'][-1]):.3f}.",
        f"Cross-seed archive is seed {cfg['breakpoints']['cross_seed']}. "
        f"Its online final is {float(raw['seeds'][str(cfg['breakpoints']['cross_seed'])]['online'][-1]):.3f}.",
        "",
    ]
    del named
    (EXP / "breakpoints.md").write_text("\n".join(bp_lines) + "\n")


def write_summary(raw: dict[str, Any], cfg: dict[str, Any], metrics: dict[str, Any]) -> None:
    curves = {}
    stds = {}
    for key in ("online", "ml", "naive", "lastk"):
        stacked = _stack(raw, cfg, key)
        curves[key] = stacked.mean(axis=0).tolist()
        stds[key] = stacked.std(axis=0, ddof=1).tolist()
    seed0 = str(cfg["breakpoints"]["seed"])
    summary = {
        "status": metrics["status"],
        "bars": metrics["bars"],
        "metrics": {k: metrics[k] for k in (
            "online_gain", "ml_end_gap", "ml_traj_mae", "naive_collapse", "lastk_end_gap",
            "online_mean", "ml_mean", "naive_mean", "lastk_mean", "base_mean",
        )},
        "curves": curves,
        "stds": stds,
        "demo_problems": raw["demo_problems"],
        "demo_truth": raw["demo_truth"],
        "demo": raw["seeds"][seed0]["demo"],
        "n_params": raw["n_params"],
        "paper": cfg["paper"],
    }
    viewer = APP / "viewer"
    viewer.mkdir(parents=True, exist_ok=True)
    (viewer / "summary.json").write_text(json.dumps(summary) + "\n")
    (EXP / "summary.json").write_text(json.dumps(summary) + "\n")


def main() -> None:
    with (EXP / "config.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    raw = json.loads((EXP / "raw.json").read_text())
    metrics = score(raw, cfg)
    status = {
        "status": metrics["status"],
        "bars": metrics["bars"],
        "metrics": {k: v for k, v in metrics.items() if k not in {"bars", "status", "per_seed"}},
        "per_seed": metrics["per_seed"],
        "threshold_commit": raw.get("threshold_commit"),
        "thresholds": cfg["thresholds"],
    }
    (EXP / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    write_tables(raw, cfg, metrics)
    ART.mkdir(parents=True, exist_ok=True)
    write_plot(raw, cfg, metrics, EXP / "plot.png")
    write_plot(raw, cfg, metrics, ART / "rl-rollout-replay.png")
    write_summary(raw, cfg, metrics)
    print(metrics["status"], {k: round(metrics[k], 4) if isinstance(metrics[k], float) else metrics[k] for k in (
        "online_gain", "ml_end_gap", "ml_traj_mae", "naive_collapse", "lastk_end_gap"
    )}, flush=True)


if __name__ == "__main__":
    sys.exit(main())
