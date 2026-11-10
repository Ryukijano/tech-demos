"""Turn raw.json into tables, a status file, and the result plot."""

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
EXP = APP / "experiments" / "exp1_cpu_token_quant"

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
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

METHODS = ("naive", "rotation", "uniform", "qmap")
COLORS = {"naive": "#D55E00", "rotation": "#0072B2", "uniform": "#CC79A7", "qmap": "#009E73"}


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def _bold_legend(ax: plt.Axes, **kwargs) -> None:
    legend = ax.legend(**kwargs)
    for text in legend.get_texts():
        text.set_fontweight("bold")


def _collect(raw: dict, tag: str, method: str, key: str) -> list[float]:
    return [float(raw["seeds"][s]["bits"][tag]["methods"][method][key]) for s in sorted(raw["seeds"], key=int)]


def _fp(raw: dict, key: str) -> list[float]:
    return [float(raw["seeds"][s]["fp"][key]) for s in sorted(raw["seeds"], key=int)]


def _gates(cfg: dict, stats: dict[str, float]) -> dict[str, bool]:
    thr = cfg["thresholds"]
    floor = float(thr["special_mse_floor"])
    slack = float(thr["w8a8_metric_order_slack"])
    prop1 = stats["snr_ratio_rel_change"] <= float(thr["prop1_snr_ratio_rel_change_max"]) and stats[
        "delta_shrink_qkv_fc1"
    ] >= float(thr["delta_shrink_min"])
    common = stats["w4_patch_naive"] > floor and stats["w4_patch_rot"] <= float(thr["w4a4_patch_rel_vs_naive_max"]) * stats[
        "w4_patch_naive"
    ]
    blind = stats["w4_patch_rot"] > floor and stats["w4_special_rot"] / stats["w4_patch_rot"] >= float(
        thr["w4a4_special_over_patch_min"]
    )
    restore_special = stats["w4_mse_rot"] > floor and stats["w4_mse_qmap"] <= float(
        thr["w4a4_special_mse_vs_rotation_max"]
    ) * stats["w4_mse_rot"]
    restore_metric = stats["w4_metric_qmap"] <= float(thr["w4a4_metric_rel_vs_rotation_max"]) * stats["w4_metric_rot"]
    still = stats["w4_metric_rot"] >= float(thr["w4a4_metric_rel_vs_fp_min"]) * stats["fp_metric"] and stats[
        "w4_metric_rot"
    ] >= float(thr["w4a4_rotation_metric_rel_min"])
    row = stats["w4_mse_uniform"] > floor and stats["w4_mse_qmap"] <= float(thr["w4a4_special_mse_vs_uniform_max"]) * stats[
        "w4_mse_uniform"
    ]
    speed = stats["w8_latency_ratio"] <= float(thr["w8a8_latency_vs_rotation_max"])
    order = stats["w8_metric_qmap"] <= (1.0 + slack) * stats["w8_metric_rot"] and stats["w8_metric_rot"] <= (
        1.0 + slack
    ) * stats["w8_metric_naive"]
    return {
        "prop1": prop1,
        "common_mode": common,
        "token_blind": blind,
        "restore_special": restore_special,
        "restore_metric": restore_metric,
        "still_degrades": still,
        "row_selective": row,
        "speed": speed,
        "w8_order": order,
    }


def _status(flags: dict[str, bool]) -> str:
    structural = flags["prop1"] and flags["common_mode"] and flags["token_blind"] and flags["speed"]
    if all(flags.values()):
        return "PASS"
    if structural and (flags["restore_special"] or flags["restore_metric"]):
        return "PARTIAL"
    return "FAIL"


def _plot(stats_mean: dict[str, dict[str, float]], stats_std: dict[str, dict[str, float]], path_png: Path, path_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.15), sharey=False)
    metric_keys = [("special_rel", "Special token"), ("patch_rel", "Patch token"), ("metric_rel", "Metric scale")]
    show = ("naive", "rotation", "qmap")
    for ax, tag, title in zip(axes, ("W4A4", "W8A8"), ("W4A4", "W8A8")):
        x = np.arange(len(metric_keys))
        width = 0.24
        for index, method in enumerate(show):
            means = [stats_mean[tag][method][key] for key, _ in metric_keys]
            stds = [stats_std[tag][method][key] for key, _ in metric_keys]
            ax.bar(
                x + (index - 1) * width,
                means,
                width,
                yerr=stds,
                color=COLORS[method],
                edgecolor="black",
                linewidth=0.4,
                capsize=2,
                label=method,
                error_kw={"elinewidth": 0.8},
            )
        ax.set_xticks(x, [label for _, label in metric_keys])
        ax.set_title(title)
        ax.set_ylabel("Relative error")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    _bold_legend(axes[1], frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path_png, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def _write_results(cfg: dict, raw: dict, stats: dict[str, float], flags: dict[str, bool], status: str) -> None:
    seeds = sorted(raw["seeds"], key=int)
    lines = [
        "# exp1_cpu_token_quant",
        "",
        f"Status: **{status}**",
        "",
        "Means are over seeds 0/1/2. ± is the sample standard deviation (ddof=1).",
        "",
        "## Gates",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for name, ok in flags.items():
        lines.append(f"| {name} | {'met' if ok else 'not met'} |")
    lines.extend(["", "## Per seed, W4A4", "", "| seed | method | special_rel | patch_rel | metric_rel | special_mse | latency_s |", "|---|---|---:|---:|---:|---:|---:|"])
    for seed in seeds:
        for method in METHODS:
            row = raw["seeds"][seed]["bits"]["W4A4"]["methods"][method]
            lines.append(
                f"| {seed} | {method} | {row['special_rel']:.4f} | {row['patch_rel']:.4f} | {row['metric_rel']:.4f} | {row['special_mse']:.6f} | {row['latency_s']:.6f} |"
            )
    lines.extend(["", "## Means", ""])
    for key in (
        "fp_metric",
        "snr_ratio_rel_change",
        "delta_shrink_qkv_fc1",
        "w4_special_naive",
        "w4_special_rot",
        "w4_special_qmap",
        "w4_patch_naive",
        "w4_patch_rot",
        "w4_metric_naive",
        "w4_metric_rot",
        "w4_metric_qmap",
        "w4_mse_rot",
        "w4_mse_uniform",
        "w4_mse_qmap",
        "w8_metric_naive",
        "w8_metric_rot",
        "w8_metric_qmap",
        "w8_latency_ratio",
        "inv_kappa_qkv",
        "gamma",
        "eta_w4",
    ):
        lines.append(f"- {key}: {stats[key]:.6g}")
    lines.append("")
    lines.append("Thresholds are the `thresholds` block of config.yaml. This file is generated.")
    (EXP / "results.md").write_text("\n".join(lines) + "\n")
    _ = cfg


def _write_breakpoints(raw: dict) -> None:
    bp = raw.get("breakpoints")
    if not bp:
        (EXP / "breakpoints.md").write_text("No breakpoints in raw.json.\n")
        return
    lines = [
        "# Breakpoints (seed 0, not part of the PASS rule)",
        "",
        "## Lower bits",
        "",
        "| W | A | method | special_rel | patch_rel | metric_rel |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for row in bp["lower_bits"]:
        lines.append(
            f"| {int(row['weight_bits'])} | {int(row['act_bits'])} | {row['method']} | {row['special_rel']:.4f} | {row['patch_rel']:.4f} | {row['metric_rel']:.4f} |"
        )
    lines.extend(["", "## More patch tokens (real qkv rows, per-scene absmax, 4-bit activations)", "", "| patches | special naive | patch naive | special rotation | patch rotation | delta shrink |", "|---:|---:|---:|---:|---:|---:|"])
    for row in bp["patch_counts"]:
        lines.append(
            f"| {int(row['n_patches'])} | {row['special_rel_naive']:.4f} | {row['patch_rel_naive']:.4f} | {row['special_rel_rotation']:.4f} | {row['patch_rel_rotation']:.4f} | {row['delta_shrink']:.3f} |"
        )
    matched = bp["matched_norm"]
    lines.extend(
        [
            "",
            "## Matched max-abs (grid deficit removed)",
            "",
            f"- original special/patch naive: {matched['original_special_over_patch_naive']:.3f}",
            f"- matched special/patch naive: {matched['matched_special_over_patch_naive']:.3f}",
            f"- original held-out special MSE drop: {matched['original_heldout_mse_drop']:.3f}",
            f"- matched held-out special MSE drop: {matched['matched_heldout_mse_drop']:.3f}",
            f"- original 1/kappa L2: {matched['original_inv_kappa_l2']:.3f}",
            f"- matched 1/kappa L2: {matched['matched_inv_kappa_l2']:.3f}",
            "",
            "## One global block, qmap / rotation latency",
            "",
            "| tokens | qmap/rotation |",
            "|---:|---:|",
        ]
    )
    for row in bp["speed_tokens"]:
        lines.append(f"| {int(row['tokens'])} | {row['qmap_over_rotation']:.4f} |")
    lines.append("")
    (EXP / "breakpoints.md").write_text("\n".join(lines))
    with (EXP / "breakpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bp["lower_bits"][0].keys()))
        writer.writeheader()
        writer.writerows(bp["lower_bits"])


def main() -> None:
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    raw = json.loads((EXP / "raw.json").read_text())
    stats: dict[str, float] = {}
    stats["fp_metric"], stats["fp_metric_std"] = _mean_std(_fp(raw, "metric_rel"))
    stats["snr_ratio_rel_change"], _ = _mean_std(_fp(raw, "snr_ratio_rel_change"))
    stats["delta_shrink_qkv_fc1"], stats["delta_shrink_std"] = _mean_std(_fp(raw, "delta_shrink_qkv_fc1"))
    stats["inv_kappa_qkv"], stats["inv_kappa_std"] = _mean_std(_fp(raw, "inv_kappa_qkv"))
    stats["gamma"], stats["gamma_std"] = _mean_std(_fp(raw, "gamma"))
    stats["max_norm_rel_error"], _ = _mean_std(_fp(raw, "max_norm_rel_error"))
    eta_vals = [float(raw["seeds"][s]["bits"]["W4A4"]["eta_heldout_special_mse_drop"]) for s in raw["seeds"]]
    stats["eta_w4"], stats["eta_std"] = _mean_std(eta_vals)

    mean_tables: dict[str, dict[str, dict[str, float]]] = {}
    std_tables: dict[str, dict[str, dict[str, float]]] = {}
    for tag in ("W4A4", "W8A8", "W4A8"):
        mean_tables[tag] = {}
        std_tables[tag] = {}
        for method in METHODS:
            mean_tables[tag][method] = {}
            std_tables[tag][method] = {}
            for key in ("special_rel", "patch_rel", "metric_rel", "special_mse", "latency_s", "upscale_rel", "metric_points_rel"):
                vals = _collect(raw, tag, method, key)
                mean, std = _mean_std(vals)
                mean_tables[tag][method][key] = mean
                std_tables[tag][method][key] = std

    stats["w4_patch_naive"] = mean_tables["W4A4"]["naive"]["patch_rel"]
    stats["w4_patch_rot"] = mean_tables["W4A4"]["rotation"]["patch_rel"]
    stats["w4_special_rot"] = mean_tables["W4A4"]["rotation"]["special_rel"]
    stats["w4_special_naive"] = mean_tables["W4A4"]["naive"]["special_rel"]
    stats["w4_special_qmap"] = mean_tables["W4A4"]["qmap"]["special_rel"]
    stats["w4_mse_rot"] = mean_tables["W4A4"]["rotation"]["special_mse"]
    stats["w4_mse_uniform"] = mean_tables["W4A4"]["uniform"]["special_mse"]
    stats["w4_mse_qmap"] = mean_tables["W4A4"]["qmap"]["special_mse"]
    stats["w4_metric_naive"] = mean_tables["W4A4"]["naive"]["metric_rel"]
    stats["w4_metric_rot"] = mean_tables["W4A4"]["rotation"]["metric_rel"]
    stats["w4_metric_qmap"] = mean_tables["W4A4"]["qmap"]["metric_rel"]
    stats["w4_metric_uniform"] = mean_tables["W4A4"]["uniform"]["metric_rel"]
    stats["w8_metric_naive"] = mean_tables["W8A8"]["naive"]["metric_rel"]
    stats["w8_metric_rot"] = mean_tables["W8A8"]["rotation"]["metric_rel"]
    stats["w8_metric_qmap"] = mean_tables["W8A8"]["qmap"]["metric_rel"]
    stats["w8_latency_ratio"] = mean_tables["W8A8"]["qmap"]["latency_s"] / max(mean_tables["W8A8"]["rotation"]["latency_s"], 1e-12)
    stats["w4_latency_ratio"] = mean_tables["W4A4"]["qmap"]["latency_s"] / max(mean_tables["W4A4"]["rotation"]["latency_s"], 1e-12)
    flags = _gates(cfg, stats)
    status = _status(flags)
    _write_results(cfg, raw, stats, flags, status)
    _write_breakpoints(raw)
    clean = []
    for seed in sorted(raw["seeds"], key=int):
        for tag in ("W4A4", "W8A8", "W4A8"):
            for method in METHODS:
                row = {"seed": seed, "bits": tag, "method": method}
                row.update(raw["seeds"][seed]["bits"][tag]["methods"][method])
                clean.append(row)
    with (EXP / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clean[0].keys()))
        writer.writeheader()
        writer.writerows(clean)
    status_obj = {
        "status": status,
        "gates": flags,
        "means": {key: stats[key] for key in stats},
        "per_bit": {
            tag: {method: mean_tables[tag][method] for method in METHODS} for tag in ("W4A4", "W8A8", "W4A8")
        },
        "per_bit_std": {
            tag: {method: std_tables[tag][method] for method in METHODS} for tag in ("W4A4", "W8A8", "W4A8")
        },
        "paper_code_available": False,
        "substitution": "clean-room CPU toy; MapAnything weights not run; QMapAnything code not released",
    }
    (EXP / "status.json").write_text(json.dumps(status_obj, indent=2))
    _plot(mean_tables, std_tables, EXP / "plot.png", EXP / "plot.pdf")
    print(status)


if __name__ == "__main__":
    main()
