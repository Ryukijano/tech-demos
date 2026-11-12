"""Turn raw.json into the status, tables, and the repeatability figure."""

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
EXP = APP / "experiments" / "exp1_cpu_rotation"

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

COLORS = {"raco": "#0072B2", "aliked": "#D55E00"}


def decide(gap: float, drop: float, gap_min: float, drop_max: float) -> str:
    """Inclusive three-way rule from config.yaml. Both bars, one bar, or neither."""
    gap_ok = gap >= gap_min
    drop_ok = drop <= drop_max
    if gap_ok and drop_ok:
        return "PASS"
    if gap_ok or drop_ok:
        return "PARTIAL"
    return "FAIL"


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError(f"non-finite aggregate from {values}")
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return float(array.mean()), std


def _angle_key(angle: float) -> str:
    return f"{float(angle):g}"


def _curve(rows: list[dict[str, Any]], angles: list[float]) -> dict[str, dict[str, dict[str, float]]]:
    """detector -> angle string -> mean/std over seeds of the per-image mean."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for detector in ("raco", "aliked"):
        out[detector] = {}
        for angle in angles:
            per_seed: list[float] = []
            seeds = sorted(
                {
                    int(row["seed"])
                    for row in rows
                    if row["detector"] == detector and float(row["angle_deg"]) == float(angle)
                }
            )
            for seed in seeds:
                vals = [
                    float(row["repeatability"])
                    for row in rows
                    if row["detector"] == detector
                    and float(row["angle_deg"]) == float(angle)
                    and int(row["seed"]) == seed
                ]
                if not vals:
                    raise RuntimeError(f"missing rows for {detector} seed {seed} angle {angle}")
                per_seed.append(float(np.mean(vals)))
            mean, std = _mean_std(per_seed)
            out[detector][_angle_key(angle)] = {"mean": mean, "std": std, "per_seed": per_seed}
    return out


def _bold_legend(ax: plt.Axes, **kwargs: Any) -> None:
    legend = ax.legend(**kwargs)
    for text in legend.get_texts():
        text.set_fontweight("bold")


def _plot(status: dict[str, Any], path_png: Path, path_pdf: Path) -> None:
    angles = [float(item) for item in status["angles_deg"]]
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.15), constrained_layout=True)
    ax = axes[0]
    for detector, label in (("raco", "RaCo"), ("aliked", "ALIKED-N16")):
        means = [status["curve"][detector][_angle_key(angle)]["mean"] for angle in angles]
        stds = [status["curve"][detector][_angle_key(angle)]["std"] for angle in angles]
        ax.errorbar(
            angles,
            means,
            yerr=stds,
            color=COLORS[detector],
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            capsize=2,
            label=label,
        )
    ax.axvspan(status["gate_angle_min_deg"], status["gate_angle_max_deg"], color="#0072B2", alpha=0.08, label="30–150° gate")
    ax.set_xlim(-5, 185)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("In-plane rotation (degrees)")
    ax.set_ylabel("Repeatability @ 3 px")
    ax.set_title(f"CPU keypoints · {status['status']}")
    _bold_legend(ax, frameon=False, loc="lower left")

    gap_ax = axes[1]
    gaps = []
    gap_stds = []
    for angle in angles:
        per_seed = []
        raco_seeds = status["curve"]["raco"][_angle_key(angle)]["per_seed"]
        aliked_seeds = status["curve"]["aliked"][_angle_key(angle)]["per_seed"]
        for raco_value, aliked_value in zip(raco_seeds, aliked_seeds):
            per_seed.append(raco_value - aliked_value)
        mean, std = _mean_std(per_seed)
        gaps.append(mean)
        gap_stds.append(std)
    gap_ax.errorbar(angles, gaps, yerr=gap_stds, color="#009E73", marker="o", markersize=3.5, linewidth=1.6, capsize=2, label="RaCo − ALIKED")
    gap_ax.axhline(status["gap_min"], color="#444444", linestyle="--", linewidth=0.9, label="0.10 bar")
    gap_ax.axvspan(status["gate_angle_min_deg"], status["gate_angle_max_deg"], color="#009E73", alpha=0.08)
    gap_ax.set_xlim(-5, 185)
    gap_ax.set_xlabel("In-plane rotation (degrees)")
    gap_ax.set_ylabel("Repeatability gap")
    gap_ax.set_title(f"Gap {status['gap']:.3f} · drop {status['drop']:.3f}")
    _bold_legend(gap_ax, frameon=False, loc="best")
    fig.savefig(path_png, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def build_status(raw: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    angles = [float(item) for item in cfg["angles_deg"]]
    rows = [row for row in raw["rows"] if row["role"] == "main"]
    curve = _curve(rows, angles)
    gate = [angle for angle in angles if cfg["gate_angle_min_deg"] <= angle <= cfg["gate_angle_max_deg"]]
    if not gate or 0.0 not in angles:
        raise RuntimeError("gate angles or 0 degrees missing from the sweep")
    gap = float(
        np.mean(
            [
                curve["raco"][_angle_key(angle)]["mean"] - curve["aliked"][_angle_key(angle)]["mean"]
                for angle in gate
            ]
        )
    )
    raco_at_0 = curve["raco"][_angle_key(0)]["mean"]
    worst_key = min(curve["raco"], key=lambda key: curve["raco"][key]["mean"])
    worst = curve["raco"][worst_key]["mean"]
    drop = float(raco_at_0 - worst)
    gap_min = float(cfg["thresholds"]["gap_min"])
    drop_max = float(cfg["thresholds"]["drop_max"])
    status = decide(gap, drop, gap_min, drop_max)
    return {
        "status": status,
        "gap": gap,
        "drop": drop,
        "gap_min": gap_min,
        "drop_max": drop_max,
        "raco_at_0": raco_at_0,
        "raco_worst": worst,
        "raco_worst_angle": float(worst_key),
        "angles_deg": angles,
        "gate_angles_deg": gate,
        "gate_angle_min_deg": cfg["gate_angle_min_deg"],
        "gate_angle_max_deg": cfg["gate_angle_max_deg"],
        "curve": curve,
        "pixel_threshold": cfg["pixel_threshold"],
    }


def _write_results(status: dict[str, Any], raw: dict[str, Any]) -> None:
    angles = status["angles_deg"]
    csv_path = EXP / "results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["angle_deg", "raco_mean", "raco_std", "aliked_mean", "aliked_std", "gap"])
        for angle in angles:
            raco = status["curve"]["raco"][_angle_key(angle)]
            aliked = status["curve"]["aliked"][_angle_key(angle)]
            writer.writerow([angle, raco["mean"], raco["std"], aliked["mean"], aliked["std"], raco["mean"] - aliked["mean"]])

    per_path = EXP / "results_per_image.csv"
    with per_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "role",
                "case",
                "seed",
                "image",
                "angle_deg",
                "detector",
                "repeatability",
                "repeatability_1px",
                "repeatability_2px",
                "mnn_inlier_ratio",
                "n_eligible_0",
                "n_eligible_1",
                "n_keypoints",
                "tie_count_0",
                "tie_count_1",
                "seconds",
            ],
        )
        writer.writeheader()
        for row in raw["rows"] + raw["breakpoints"]:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    lines = [
        f"# Results — {status['status']}",
        "",
        f"Gap over 30–150°: **{status['gap']:.4f}** (bar ≥ {status['gap_min']:.2f}).",
        f"RaCo drop from 0° to {status['raco_worst_angle']:g}°: **{status['drop']:.4f}** (bar ≤ {status['drop_max']:.2f}).",
        f"RaCo at 0°: {status['raco_at_0']:.4f}. Worst RaCo angle on the sweep: {status['raco_worst']:.4f}.",
        "",
        "Mean ± sample std (ddof=1) over seeds. Each seed's value is the mean over its four images.",
        "",
        "| Angle | RaCo | ALIKED-N16 | Gap |",
        "|---:|---:|---:|---:|",
    ]
    for angle in angles:
        raco = status["curve"]["raco"][_angle_key(angle)]
        aliked = status["curve"]["aliked"][_angle_key(angle)]
        lines.append(
            f"| {angle:g} | {_fmt(raco['mean'], raco['std'])} | {_fmt(aliked['mean'], aliked['std'])} | {raco['mean'] - aliked['mean']:.3f} |"
        )
    lines.append("")
    (EXP / "results.md").write_text("\n".join(lines) + "\n")


def _write_breakpoints(raw: dict[str, Any]) -> None:
    rows = raw["breakpoints"]
    with (EXP / "breakpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "angle_deg", "scale", "blur_sigma", "image_size", "keypoint_budget", "detector", "repeatability", "mnn_inlier_ratio", "seconds"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    cases = []
    for row in rows:
        if row["case"] not in cases:
            cases.append(row["case"])
    lines = [
        "# Breakpoints",
        "",
        "Seed 0, one image. Not part of the PASS rule.",
        "",
        "| Case | Angle | Scale | Blur | Size | k | RaCo | ALIKED-N16 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        pair = [row for row in rows if row["case"] == case]
        by_det = {row["detector"]: row for row in pair}
        sample = pair[0]
        lines.append(
            "| {case} | {angle:g} | {scale:g} | {blur:g} | {size} | {k} | {raco:.3f} | {aliked:.3f} |".format(
                case=case,
                angle=float(sample["angle_deg"]),
                scale=float(sample["scale"]),
                blur=float(sample["blur_sigma"]),
                size=int(sample["image_size"]),
                k=int(sample["keypoint_budget"]),
                raco=float(by_det["raco"]["repeatability"]),
                aliked=float(by_det["aliked"]["repeatability"]),
            )
        )
    lines.append("")
    (EXP / "breakpoints.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    raw = json.loads((EXP / "raw.json").read_text())
    status = build_status(raw, cfg)
    # JSON cannot store the per-seed lists mixed with the public status if we
    # want a stable file. Keep them: the plot and the gap std need them.
    (EXP / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    _write_results(status, raw)
    _write_breakpoints(raw)
    _plot(status, EXP / "plot.png", EXP / "plot.pdf")
    print(f"status {status['status']} gap {status['gap']:.4f} drop {status['drop']:.4f}")


if __name__ == "__main__":
    main()
