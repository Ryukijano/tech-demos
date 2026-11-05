"""Turn raw.json into results, status, breakpoint tables, and the results figure."""

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
EXP = APP / "experiments" / "exp1_cpu_mrp"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Liberation Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
    }
)

COLORS = {"vf": "#0072B2", "endpoint": "#D55E00", "qr": "#009E73", "truth": "#222222"}
LABELS = {"vf": "Value Forcing", "endpoint": "Endpoint flow", "qr": "QR-DQN"}
METHOD_ORDER = ("vf", "endpoint", "qr")


def _bold_legend(ax, **kwargs):
    legend = ax.legend(**kwargs)
    if legend is None:
        return None
    for text in legend.get_texts():
        text.set_fontweight("bold")
    title = legend.get_title()
    if title is not None:
        title.set_fontweight("bold")
    return legend


def _mean_std(xs: list[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def _status(vf_w1: float, ep_w1: float, qr_w1: float, vf_ratio: float, thr: dict) -> tuple[str, dict]:
    reduction = float(thr["w1_reduction_vs_endpoint"])
    ratio_lo = float(thr["std_ratio_min"])
    ratio_hi = float(thr["std_ratio_max"])
    loose_lo = float(thr["partial_std_ratio_min"])
    loose_hi = float(thr["partial_std_ratio_max"])
    beats_endpoint = vf_w1 < ep_w1
    reduction_ok = vf_w1 <= (1.0 - reduction) * ep_w1
    beats_qr = vf_w1 < qr_w1
    ratio_ok = ratio_lo <= vf_ratio <= ratio_hi
    ratio_loose = loose_lo <= vf_ratio <= loose_hi
    checks = {
        "vf_w1_at_least_20pct_below_endpoint": reduction_ok,
        "vf_w1_below_qr": beats_qr,
        "vf_std_ratio_in_0.8_1.2": ratio_ok,
        "vf_w1_below_endpoint": beats_endpoint,
        "vf_std_ratio_in_0.5_1.5": ratio_loose,
    }
    if reduction_ok and beats_qr and ratio_ok:
        status = "PASS"
    elif beats_endpoint and ratio_loose:
        status = "PARTIAL"
    else:
        status = "FAIL"
    return status, checks


def _write_results(raw: dict, cfg: dict, status: str, checks: dict, summary: dict) -> None:
    seeds = sorted(raw["seeds"], key=int)
    rows = []
    for seed in seeds:
        for method in METHOD_ORDER:
            rec = raw["seeds"][seed][method]
            row = {
                "seed": seed,
                "method": method,
                "mean_w1": round(rec["mean_w1"], 6),
                "std_ratio": round(rec["std_ratio"], 6),
                "farthest_state": rec["farthest_state"],
                "farthest_w1": round(rec["farthest_w1"], 6),
                "mix_high_fraction": round(rec["mix_high_fraction"], 6),
                "final_loss": round(rec["final_loss"], 6),
                "train_wall_s": round(rec["train_wall_s"], 3),
                "n_params": rec["n_params"],
            }
            for state in rec["per_state"]:
                row[f"w1_{state['state']}"] = round(state["w1"], 6)
            rows.append(row)
    with (EXP / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# exp1_cpu_mrp results",
        "",
        f"Status: **{status}**",
        "",
        "Pre-registered pass rule: mean W1(VF) ≤ 0.80 × mean W1(endpoint), mean W1(VF) < mean W1(QR), and VF std ratio inside [0.8, 1.2], averaged over seeds 0/1/2.",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, ok in checks.items():
        lines.append(f"| `{name}` | {'met' if ok else 'not met'} |")
    lines += [
        "",
        "## Mean over seeds",
        "",
        "| Method | mean W1 | std ratio |",
        "|---|---:|---:|",
    ]
    for method in METHOD_ORDER:
        lines.append(
            f"| {LABELS[method]} | {summary[method]['w1_mean']:.4f} ± {summary[method]['w1_std']:.4f} | "
            f"{summary[method]['ratio_mean']:.3f} ± {summary[method]['ratio_std']:.3f} |"
        )
    lines += [
        "",
        f"Relative W1 reduction vs endpoint: **{summary['reduction_vs_endpoint']:.1%}** (threshold 20%).",
        "",
        "## Per seed",
        "",
        "| Seed | VF W1 | Endpoint W1 | QR W1 | VF std ratio | Endpoint std ratio | QR std ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        cells = [seed]
        for method in METHOD_ORDER:
            cells.append(f"{raw['seeds'][seed][method]['mean_w1']:.4f}")
        for method in METHOD_ORDER:
            cells.append(f"{raw['seeds'][seed][method]['std_ratio']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Per-state W1 (mean over seeds)", "", "| State | true mean | true std | VF | Endpoint | QR |", "|---|---:|---:|---:|---:|---:|"]
    truth = {row["state"]: row for row in raw["truth"]["states"]}
    state_names = [row["state"] for row in raw["truth"]["states"]]
    for name in state_names:
        w1s = []
        for method in METHOD_ORDER:
            vals = []
            for seed in seeds:
                match = next(s for s in raw["seeds"][seed][method]["per_state"] if s["state"] == name)
                vals.append(match["w1"])
            w1s.append(float(np.mean(vals)))
        lines.append(
            f"| {name} | {truth[name]['mean']:.3f} | {truth[name]['std']:.3f} | {w1s[0]:.4f} | {w1s[1]:.4f} | {w1s[2]:.4f} |"
        )
    lines.append("")
    (EXP / "results.md").write_text("\n".join(lines) + "\n")


def _write_breakpoints(raw: dict) -> None:
    sections = [
        ("few_transitions", ["n_transitions", "gamma", "n_euler", "chain_depth"], "Few transitions (seed 0)"),
        ("gamma_near_one", ["gamma", "n_transitions", "n_euler", "chain_depth"], "Discount near 1 (seed 0)"),
        ("long_horizon", ["chain_depth", "n_transitions", "gamma", "n_euler"], "Chain length into MIX (seed 0)"),
        ("flow_steps", ["n_euler", "gamma", "n_transitions", "chain_depth"], "Euler steps (seed 0; QR has no ODE)"),
    ]
    # Include the primary seed-0 row in each table so the registered setting is visible.
    primary = []
    if "0" in raw["seeds"]:
        for method, rec in raw["seeds"]["0"].items():
            primary.append(rec)
    lines = ["# Breakpoints", "", "Seed 0 unless noted. Primary setting is included as the reference row.", ""]
    flat_csv = []
    for tag, fields, title in sections:
        recs = list(raw["breakpoints"][tag]) + primary
        if tag == "flow_steps":
            recs = [r for r in recs if r["method"] != "qr"]
        recs = sorted(recs, key=lambda r: (r[fields[0]], r["method"]))
        lines.append(f"## {title}")
        lines.append("")
        header = "| " + " | ".join(fields + ["method", "mean W1", "std ratio", "farthest W1"]) + " |"
        sep = "|" + "|".join(["---:" if f != "method" else "---" for f in fields]) + "|---|---:|---:|---:|"
        # fix sep to match columns
        sep = "|" + "|".join(["---:"] * len(fields)) + "|---|---:|---:|---:|"
        lines.append(header)
        lines.append(sep)
        for rec in recs:
            cells = []
            for field in fields:
                value = rec[field]
                cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
            cells += [rec["method"], f"{rec['mean_w1']:.4f}", f"{rec['std_ratio']:.3f}", f"{rec['farthest_w1']:.4f}"]
            lines.append("| " + " | ".join(cells) + " |")
            flat = {"breakpoint": tag, "method": rec["method"]}
            for field in fields:
                flat[field] = rec[field]
            flat["mean_w1"] = rec["mean_w1"]
            flat["std_ratio"] = rec["std_ratio"]
            flat["farthest_w1"] = rec["farthest_w1"]
            flat_csv.append(flat)
        lines.append("")
    (EXP / "breakpoints.md").write_text("\n".join(lines) + "\n")
    if flat_csv:
        # Union of keys
        keys = ["breakpoint", "method", "n_transitions", "gamma", "n_euler", "chain_depth", "mean_w1", "std_ratio", "farthest_w1"]
        with (EXP / "breakpoints.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in flat_csv:
                writer.writerow({k: row.get(k, "") for k in keys})


def _density_from_hist(hist: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.asarray(hist["edges"], dtype=np.float64)
    width = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pred = np.asarray(hist["pred_count"], dtype=np.float64) / (hist["n_samples"] * width)
    truth = np.asarray(hist["true_mass"], dtype=np.float64) / width
    return centers, pred, truth


def _plot(raw: dict, summary: dict) -> None:
    seed0 = raw["seeds"]["0"]
    names = [row["state"] for row in raw["truth"]["states"]]
    fig = plt.figure(figsize=(11.4, 8.8))
    outer = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.15], hspace=0.48, wspace=0.38)
    for i, name in enumerate(names):
        ax = fig.add_subplot(outer[i // 4, i % 4])
        for method in METHOD_ORDER:
            hist = seed0[method]["hist"][name]
            centers, pred, truth = _density_from_hist(hist)
            if method == "vf":
                ax.fill_between(centers, truth, step="mid", color=COLORS["truth"], alpha=0.18, label="True")
                ax.step(centers, truth, where="mid", color=COLORS["truth"], lw=1.1, label="_nolegend_")
            ax.plot(centers, pred, color=COLORS[method], lw=1.5, label=LABELS[method])
        ax.set_title(name)
        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
        ax.tick_params(labelsize=8)
    _bold_legend(
        fig.axes[0],
        fontsize=7,
        frameon=True,
        loc="upper right",
    )

    ax_bar = fig.add_subplot(outer[2, 0:2])
    x = np.arange(len(METHOD_ORDER))
    means = [summary[m]["w1_mean"] for m in METHOD_ORDER]
    stds = [summary[m]["w1_std"] for m in METHOD_ORDER]
    ax_bar.bar(x, means, yerr=stds, color=[COLORS[m] for m in METHOD_ORDER], width=0.72, capsize=3, error_kw={"lw": 1})
    threshold = 0.8 * summary["endpoint"]["w1_mean"]
    ax_bar.axhline(threshold, color="#333333", ls="--", lw=1.0, label="20% below endpoint")
    ax_bar.set_xticks(x, [LABELS[m] for m in METHOD_ORDER])
    ax_bar.set_ylabel("Mean W1 across states")
    ax_bar.set_title("Limited data, seeds 0–2")
    _bold_legend(ax_bar, fontsize=8, loc="upper right")

    ax_n = fig.add_subplot(outer[2, 2:4])
    curve: dict[str, list[tuple[int, float]]] = {m: [] for m in METHOD_ORDER}
    for rec in raw["breakpoints"]["few_transitions"]:
        curve[rec["method"]].append((int(rec["n_transitions"]), float(rec["mean_w1"])))
    for method, rec in seed0.items():
        curve[method].append((int(rec["n_transitions"]), float(rec["mean_w1"])))
    for method in METHOD_ORDER:
        pts = sorted(curve[method])
        ax_n.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4.5, lw=1.6, color=COLORS[method], label=LABELS[method])
    ax_n.set_xscale("log", base=2)
    ax_n.set_xlabel("Offline transitions")
    ax_n.set_ylabel("Mean W1")
    ax_n.set_title("W1 vs dataset size (seed 0)")
    _bold_legend(ax_n, fontsize=8, loc="upper right")

    fig.suptitle("Value Forcing on a closed-form branching MRP", fontweight="bold", fontsize=13)
    fig.savefig(EXP / "plot.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    raw = json.loads((EXP / "raw.json").read_text())
    seeds = sorted(raw["seeds"], key=int)
    summary = {}
    for method in METHOD_ORDER:
        w1s = [raw["seeds"][s][method]["mean_w1"] for s in seeds]
        ratios = [raw["seeds"][s][method]["std_ratio"] for s in seeds]
        w1_mean, w1_std = _mean_std(w1s)
        ratio_mean, ratio_std = _mean_std(ratios)
        summary[method] = {
            "w1_mean": w1_mean,
            "w1_std": w1_std,
            "ratio_mean": ratio_mean,
            "ratio_std": ratio_std,
        }
    ep = summary["endpoint"]["w1_mean"]
    reduction = (ep - summary["vf"]["w1_mean"]) / ep if ep > 0 else float("nan")
    summary["reduction_vs_endpoint"] = reduction
    status, checks = _status(
        summary["vf"]["w1_mean"],
        summary["endpoint"]["w1_mean"],
        summary["qr"]["w1_mean"],
        summary["vf"]["ratio_mean"],
        cfg["thresholds"],
    )
    _write_results(raw, cfg, status, checks, summary)
    _write_breakpoints(raw)
    _plot(raw, summary)
    payload = {
        "status": status,
        "checks": checks,
        "reduction_vs_endpoint": reduction,
        "summary": summary,
        "rule": cfg["thresholds"]["status_rule"].strip(),
    }
    (EXP / "status.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"status={status}  VF W1={summary['vf']['w1_mean']:.4f}  endpoint={ep:.4f}  QR={summary['qr']['w1_mean']:.4f}  ratio={summary['vf']['ratio_mean']:.3f}  reduction={reduction:.1%}")


if __name__ == "__main__":
    main()
