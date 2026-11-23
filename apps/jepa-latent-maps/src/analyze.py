"""Turn raw.json into results, status, breakpoint tables, and the figure."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_grid"

METHODS = ("vicreg", "svreg", "slow_svreg")
METHOD_LABEL = {
    "vicreg": "VICReg",
    "svreg": "SVReg",
    "slow_svreg": "Slow SVReg",
}
METHOD_COLOR = {
    "vicreg": "#0072B2",
    "svreg": "#E69F00",
    "slow_svreg": "#009E73",
}
PRIMARY_KEYS = ("agent_r2", "geodesic_spearman", "free_object_r2")
PRIMARY_LABEL = {
    "agent_r2": "Agent R²",
    "geodesic_spearman": "Geodesic ρ",
    "free_object_r2": "Free-object R²",
}

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _bold_legend(ax: plt.Axes, **kwargs) -> None:
    legend = ax.legend(**kwargs)
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontweight("bold")
    if legend.get_title() is not None:
        legend.get_title().set_fontweight("bold")


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(arr)):
        return float("nan"), float("nan")
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _fmt(value: float, digits: int = 4) -> str:
    if not _finite(value):
        return "nan"
    return f"{value:.{digits}f}"


def main() -> None:
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    raw = json.loads((EXP / "raw.json").read_text())
    thr = cfg["thresholds"]
    seeds = sorted(raw["seeds"].keys(), key=int)

    rows: list[dict[str, object]] = []
    by_method: dict[str, dict[str, list[float]]] = {
        method: {key: [] for key in PRIMARY_KEYS} for method in METHODS
    }
    for seed in seeds:
        for method in METHODS:
            metrics = raw["seeds"][seed][method]
            for key in PRIMARY_KEYS:
                by_method[method][key].append(float(metrics[key]) if metrics[key] is not None else float("nan"))
            rows.append({
                "seed": seed,
                "method": method,
                "agent_r2": _fmt(float(metrics["agent_r2"]) if metrics["agent_r2"] is not None else float("nan")),
                "geodesic_spearman": _fmt(
                    float(metrics["geodesic_spearman"]) if metrics["geodesic_spearman"] is not None else float("nan")
                ),
                "free_object_r2": _fmt(
                    float(metrics["free_object_r2"]) if metrics["free_object_r2"] is not None else float("nan")
                ),
                "object_r2_all": _fmt(
                    float(metrics["object_r2_all"]) if metrics["object_r2_all"] is not None else float("nan")
                ),
                "jaggedness": _fmt(float(metrics["jaggedness"]) if metrics["jaggedness"] is not None else float("nan")),
                "participation_ratio": _fmt(
                    float(metrics["participation_ratio"]) if metrics["participation_ratio"] is not None else float("nan")
                ),
                "pred_loss": _fmt(float(metrics["pred_loss"]) if metrics["pred_loss"] is not None else float("nan")),
                "train_wall_s": _fmt(float(metrics["train_wall_s"]), 2),
                "n_params": int(metrics["n_params"]),
            })

    means = {method: {key: _mean_std(by_method[method][key]) for key in PRIMARY_KEYS} for method in METHODS}

    def at(method: str, key: str, seed_index: int) -> float:
        return by_method[method][key][seed_index]

    checks: dict[str, bool] = {}
    each_agent = all(
        _finite(at("slow_svreg", "agent_r2", i)) and at("slow_svreg", "agent_r2", i) >= float(thr["slow_agent_r2_min_each"])
        for i in range(len(seeds))
    )
    each_geo = all(
        _finite(at("slow_svreg", "geodesic_spearman", i))
        and at("slow_svreg", "geodesic_spearman", i) >= float(thr["slow_geodesic_min_each"])
        for i in range(len(seeds))
    )
    each_obj = all(
        _finite(at("slow_svreg", "free_object_r2", i))
        and at("slow_svreg", "free_object_r2", i) >= float(thr["slow_object_r2_min_each"])
        for i in range(len(seeds))
    )
    each_slow_gt_vic_geo = all(
        _finite(at("slow_svreg", "geodesic_spearman", i))
        and _finite(at("vicreg", "geodesic_spearman", i))
        and at("slow_svreg", "geodesic_spearman", i) > at("vicreg", "geodesic_spearman", i)
        for i in range(len(seeds))
    )
    each_sv_gt_vic_geo = all(
        _finite(at("svreg", "geodesic_spearman", i))
        and _finite(at("vicreg", "geodesic_spearman", i))
        and at("svreg", "geodesic_spearman", i) > at("vicreg", "geodesic_spearman", i)
        for i in range(len(seeds))
    )
    each_slow_gt_sv_obj = all(
        _finite(at("slow_svreg", "free_object_r2", i))
        and _finite(at("svreg", "free_object_r2", i))
        and at("slow_svreg", "free_object_r2", i) > at("svreg", "free_object_r2", i)
        for i in range(len(seeds))
    )
    mean_agent = means["slow_svreg"]["agent_r2"][0]
    gap_sv_vic = means["svreg"]["geodesic_spearman"][0] - means["vicreg"]["geodesic_spearman"][0]
    gap_slow_vic = means["slow_svreg"]["geodesic_spearman"][0] - means["vicreg"]["geodesic_spearman"][0]
    gap_obj = means["slow_svreg"]["free_object_r2"][0] - means["svreg"]["free_object_r2"][0]
    drop_geo = means["svreg"]["geodesic_spearman"][0] - means["slow_svreg"]["geodesic_spearman"][0]
    checks = {
        "each_slow_agent_r2": each_agent,
        "each_slow_geodesic": each_geo,
        "each_slow_object": each_obj,
        "each_slow_geodesic_gt_vicreg": each_slow_gt_vic_geo,
        "each_svreg_geodesic_gt_vicreg": each_sv_gt_vic_geo,
        "each_slow_object_gt_svreg": each_slow_gt_sv_obj,
        "mean_slow_agent_r2": _finite(mean_agent) and mean_agent >= float(thr["slow_agent_r2_mean_min"]),
        "gap_svreg_geodesic_vs_vicreg": _finite(gap_sv_vic) and gap_sv_vic >= float(thr["geodesic_gap_svreg_vs_vicreg_min"]),
        "gap_slow_geodesic_vs_vicreg": _finite(gap_slow_vic) and gap_slow_vic >= float(thr["geodesic_gap_slow_vs_vicreg_min"]),
        "gap_slow_object_vs_svreg": _finite(gap_obj) and gap_obj >= float(thr["object_gap_slow_vs_svreg_min"]),
        "slow_geodesic_drop_vs_svreg": _finite(drop_geo) and drop_geo <= float(thr["slow_geodesic_drop_vs_svreg_max"]),
    }
    directional = (
        _finite(gap_slow_vic)
        and gap_slow_vic > 0.0
        and _finite(gap_obj)
        and gap_obj > 0.0
    )
    if all(checks.values()):
        status = "PASS"
    elif directional:
        status = "PARTIAL"
    else:
        status = "FAIL"

    status_obj = {
        "status": status,
        "checks": checks,
        "means": {
            method: {key: {"mean": means[method][key][0], "std": means[method][key][1]} for key in PRIMARY_KEYS}
            for method in METHODS
        },
        "gap_svreg_geodesic_vs_vicreg": gap_sv_vic,
        "gap_slow_geodesic_vs_vicreg": gap_slow_vic,
        "gap_slow_object_vs_svreg": gap_obj,
        "slow_geodesic_drop_vs_svreg": drop_geo,
        "n_seeds": len(seeds),
        "observation_ceiling": raw.get("observation_ceiling"),
        "world": raw.get("world"),
    }
    (EXP / "status.json").write_text(json.dumps(status_obj, indent=2) + "\n")

    with (EXP / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bp_rows: list[dict[str, object]] = []
    for sweep, items in raw["breakpoints"].items():
        for metrics in items:
            bp_rows.append({
                "sweep": sweep,
                "method": metrics["method"],
                "seed": int(metrics["seed"]),
                "latent_dim": int(metrics["latent_dim"]),
                "steps": int(metrics["steps"]),
                "svreg_weight": metrics["svreg_weight"],
                "vicreg_weight": metrics["vicreg_weight"],
                "slow_weight": metrics["slow_weight"],
                "agent_r2": _fmt(float(metrics["agent_r2"]) if metrics["agent_r2"] is not None else float("nan")),
                "geodesic_spearman": _fmt(
                    float(metrics["geodesic_spearman"]) if metrics["geodesic_spearman"] is not None else float("nan")
                ),
                "free_object_r2": _fmt(
                    float(metrics["free_object_r2"]) if metrics["free_object_r2"] is not None else float("nan")
                ),
                "jaggedness": _fmt(float(metrics["jaggedness"]) if metrics["jaggedness"] is not None else float("nan")),
            })
    with (EXP / "breakpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bp_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bp_rows)
    (EXP / "breakpoints_raw.json").write_text(json.dumps(raw["breakpoints"], indent=2) + "\n")

    _write_results_md(cfg, seeds, rows, means, checks, status, gap_sv_vic, gap_slow_vic, gap_obj, drop_geo, raw)
    _write_breakpoints_md(cfg, raw)
    _plot(means, raw)
    print(f"[status] {status}")


def _write_results_md(cfg, seeds, rows, means, checks, status, gap_sv_vic, gap_slow_vic, gap_obj, drop_geo, raw) -> None:
    thr = cfg["thresholds"]
    lines = [
        f"# exp1_cpu_grid — {status}",
        "",
        "Thresholds were read from `config.yaml` (frozen before this run).",
        "",
        "| Check | Bar | Result |",
        "|---|---|---|",
    ]
    descriptions = [
        ("each_slow_agent_r2", f"every seed slow agent R² ≥ {thr['slow_agent_r2_min_each']}"),
        ("each_slow_geodesic", f"every seed slow geodesic ρ ≥ {thr['slow_geodesic_min_each']}"),
        ("each_slow_object", f"every seed slow free-object R² ≥ {thr['slow_object_r2_min_each']}"),
        ("each_slow_geodesic_gt_vicreg", "every seed slow geodesic ρ > VICReg"),
        ("each_svreg_geodesic_gt_vicreg", "every seed SVReg geodesic ρ > VICReg"),
        ("each_slow_object_gt_svreg", "every seed slow free-object R² > SVReg"),
        ("mean_slow_agent_r2", f"mean slow agent R² ≥ {thr['slow_agent_r2_mean_min']}"),
        ("gap_svreg_geodesic_vs_vicreg", f"mean SVReg−VICReg geodesic ≥ {thr['geodesic_gap_svreg_vs_vicreg_min']} ({_fmt(gap_sv_vic)})"),
        ("gap_slow_geodesic_vs_vicreg", f"mean Slow−VICReg geodesic ≥ {thr['geodesic_gap_slow_vs_vicreg_min']} ({_fmt(gap_slow_vic)})"),
        ("gap_slow_object_vs_svreg", f"mean Slow−SVReg free-object R² ≥ {thr['object_gap_slow_vs_svreg_min']} ({_fmt(gap_obj)})"),
        ("slow_geodesic_drop_vs_svreg", f"mean SVReg−Slow geodesic ≤ {thr['slow_geodesic_drop_vs_svreg_max']} ({_fmt(drop_geo)})"),
    ]
    for key, label in descriptions:
        lines.append(f"| {label} | fixed | **{'met' if checks[key] else 'not met'}** |")
    lines.extend(["", "## Per seed", "", "| seed | method | agent R² | geodesic ρ | free-object R² | jaggedness | PR |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(
            f"| {row['seed']} | {METHOD_LABEL[row['method']]} | {row['agent_r2']} | {row['geodesic_spearman']} | "
            f"{row['free_object_r2']} | {row['jaggedness']} | {row['participation_ratio']} |"
        )
    lines.extend(["", "## Mean ± sample std", ""])
    for method in METHODS:
        bits = []
        for key in PRIMARY_KEYS:
            mean, std = means[method][key]
            bits.append(f"{PRIMARY_LABEL[key]} {_fmt(mean)} ± {_fmt(std)}")
        lines.append(f"- **{METHOD_LABEL[method]}:** " + "; ".join(bits))
    ceiling = raw.get("observation_ceiling") or {}
    lines.extend([
        "",
        "## Observation ceiling (report only)",
        "",
        f"Linear ridge on the noisy pixels: agent R² {_fmt(float(ceiling.get('pixel_agent_r2') or float('nan')))}, "
        f"free-object R² {_fmt(float(ceiling.get('pixel_free_object_r2') or float('nan')))}.",
        "",
    ])
    (EXP / "results.md").write_text("\n".join(lines))


def _write_breakpoints_md(cfg, raw) -> None:
    lines = ["# Breakpoints (seed 0)", ""]
    for sweep, story_key in (
        ("latent_dim", "latent_dim"),
        ("reg_weight", "reg_weight"),
        ("short_training", "short_training"),
    ):
        lines.append(f"## {sweep}")
        lines.append("")
        lines.append(cfg["breakpoints"][story_key]["story"].strip())
        lines.append("")
        lines.append("| setting | method | agent R² | geodesic ρ | free-object R² | jaggedness |")
        lines.append("|---|---|---:|---:|---:|---:|")
        items = sorted(
            raw["breakpoints"][sweep],
            key=lambda m: (float(m["latent_dim"]), float(m["svreg_weight"]), float(m["steps"]), m["method"]),
        )
        for metrics in items:
            if sweep == "latent_dim":
                setting = f"dim={int(metrics['latent_dim'])}"
            elif sweep == "reg_weight":
                setting = f"λ={metrics['svreg_weight']}"
            else:
                setting = f"steps={int(metrics['steps'])}"
            obj = metrics["free_object_r2"]
            geo = metrics["geodesic_spearman"]
            agent = metrics["agent_r2"]
            jag = metrics["jaggedness"]
            lines.append(
                f"| {setting} | {METHOD_LABEL[metrics['method']]} | "
                f"{_fmt(float(agent) if agent is not None else float('nan'))} | "
                f"{_fmt(float(geo) if geo is not None else float('nan'))} | "
                f"{_fmt(float(obj) if obj is not None else float('nan'))} | "
                f"{_fmt(float(jag) if jag is not None else float('nan'))} |"
            )
        lines.append("")
    (EXP / "breakpoints.md").write_text("\n".join(lines))


def _series(items: list[dict], method: str, x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    chosen = [m for m in items if m["method"] == method]
    chosen.sort(key=lambda m: float(m[x_key]))
    xs = np.asarray([float(m[x_key]) for m in chosen], dtype=np.float64)
    ys = np.asarray([
        float(m[y_key]) if m[y_key] is not None else np.nan
        for m in chosen
    ], dtype=np.float64)
    return xs, ys


def _plot(means, raw) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1))
    ax_bar, ax_dim_obj = axes[0]
    ax_dim_geo, ax_lam = axes[1]

    x = np.arange(len(PRIMARY_KEYS))
    width = 0.24
    for offset, method in enumerate(METHODS):
        heights = [means[method][key][0] for key in PRIMARY_KEYS]
        errors = [means[method][key][1] for key in PRIMARY_KEYS]
        ax_bar.bar(
            x + (offset - 1) * width,
            heights,
            width,
            yerr=errors,
            color=METHOD_COLOR[method],
            label=METHOD_LABEL[method],
            capsize=2,
            error_kw={"elinewidth": 0.8, "ecolor": "black"},
        )
    ax_bar.set_xticks(x, [PRIMARY_LABEL[key] for key in PRIMARY_KEYS])
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("A  Held-out scores (mean ± s.d.)")
    # Include the negative free-object R². A floor at −0.05 would clip it.
    ax_bar.set_ylim(-0.40, 1.05)
    ax_bar.axhline(0.0, color="0.5", linewidth=0.4)
    _bold_legend(ax_bar, loc="upper right", frameon=True)

    for method in METHODS:
        xs, ys = _series(raw["breakpoints"]["latent_dim"], method, "latent_dim", "free_object_r2")
        ax_dim_obj.plot(
            xs, ys, marker="o", color=METHOD_COLOR[method], label=METHOD_LABEL[method], linewidth=1.8,
        )
    ax_dim_obj.axhline(0.45, color="0.35", linestyle=":", linewidth=0.9, label="Floor 0.45")
    ax_dim_obj.set_xlabel("Latent dimension")
    ax_dim_obj.set_ylabel("Free-object R²")
    ax_dim_obj.set_title("B  Capacity sweep, content")
    ax_dim_obj.set_xticks([2, 3, 8, 16])
    ax_dim_obj.set_ylim(-0.45, 0.60)
    _bold_legend(ax_dim_obj, loc="best", frameon=True)

    for method in METHODS:
        xs, ys = _series(raw["breakpoints"]["latent_dim"], method, "latent_dim", "geodesic_spearman")
        ax_dim_geo.plot(
            xs, ys, marker="s", color=METHOD_COLOR[method], label=METHOD_LABEL[method], linewidth=1.8,
        )
    ax_dim_geo.axhline(0.45, color="0.35", linestyle=":", linewidth=0.9, label="Floor 0.45")
    ax_dim_geo.set_xlabel("Latent dimension")
    ax_dim_geo.set_ylabel("Geodesic Spearman ρ")
    ax_dim_geo.set_title("C  Capacity sweep, geometry")
    ax_dim_geo.set_xticks([2, 3, 8, 16])
    ax_dim_geo.set_ylim(0.0, 0.55)
    _bold_legend(ax_dim_geo, loc="best", frameon=True)

    for method in METHODS:
        xs, ys = _series(raw["breakpoints"]["reg_weight"], method, "svreg_weight", "free_object_r2")
        ax_lam.plot(
            xs, ys, marker="D", color=METHOD_COLOR[method], label=METHOD_LABEL[method], linewidth=1.8,
        )
    ax_lam.set_xscale("log")
    ax_lam.set_xlabel("Regularizer weight λ")
    ax_lam.set_ylabel("Free-object R²")
    ax_lam.set_title("D  Weight sweep, content")
    ax_lam.set_ylim(-0.45, 0.60)
    ax_lam.axhline(0.45, color="0.35", linestyle=":", linewidth=0.9, label="Floor 0.45")
    _bold_legend(ax_lam, loc="best", frameon=True)

    for ax in axes.ravel():
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(EXP / "plot.png")
    fig.savefig(EXP / "plot.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
