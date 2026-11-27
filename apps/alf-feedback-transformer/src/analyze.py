"""Score the pre-registered bars and write the table, status, and figure."""

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
EXP = APP / "experiments" / "exp1_cpu_register"
ART = APP / "artifacts"
MODES = ("vanilla", "fbt", "alf")
COLORS = {"vanilla": "#0072B2", "fbt": "#E69F00", "alf": "#009E73"}
LABELS = {"vanilla": "Vanilla", "fbt": "FBT (last state)", "alf": "ALF (flexible)"}


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std


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
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def _final(raw: dict[str, Any], seed: str, mode: str) -> float:
    return float(raw["seeds"][seed][mode]["final"]["answer_nats"])


def score(raw: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    seeds = [str(s) for s in cfg["seeds"]]
    gap_v_bar = float(cfg["thresholds"]["gap_vs_vanilla_nats"])
    gap_f_bar = float(cfg["thresholds"]["gap_vs_fbt_nats"])
    per_seed = []
    for seed in seeds:
        row = {mode: _final(raw, seed, mode) for mode in MODES}
        row["seed"] = int(seed)
        row["acc"] = {mode: float(raw["seeds"][seed][mode]["final"]["answer_acc"]) for mode in MODES}
        row["full_nats"] = {mode: float(raw["seeds"][seed][mode]["final"]["full_nats"]) for mode in MODES}
        row["n_params"] = {mode: int(raw["seeds"][seed][mode]["n_params"]) for mode in MODES}
        per_seed.append(row)
    means = {}
    stds = {}
    for mode in MODES:
        means[mode], stds[mode] = _mean_std([row[mode] for row in per_seed])
    gap_v = [row["vanilla"] - row["alf"] for row in per_seed]
    gap_f = [row["fbt"] - row["alf"] for row in per_seed]
    gap_v_mean, gap_v_std = _mean_std(gap_v)
    gap_f_mean, gap_f_std = _mean_std(gap_f)
    every = all(row["alf"] < row["fbt"] and row["alf"] < row["vanilla"] for row in per_seed)
    mean_best = means["alf"] < means["vanilla"] and means["alf"] < means["fbt"]
    gaps_ok = gap_v_mean >= gap_v_bar and gap_f_mean >= gap_f_bar
    if every and gaps_ok:
        status = "PASS"
    elif mean_best:
        status = "PARTIAL"
    else:
        status = "FAIL"
    return {
        "status": status,
        "primary_metric": "answer_nats",
        "seeds": per_seed,
        "mean": means,
        "std": stds,
        "gap_vs_vanilla": gap_v_mean,
        "gap_vs_vanilla_std": gap_v_std,
        "gap_vs_fbt": gap_f_mean,
        "gap_vs_fbt_std": gap_f_std,
        "gap_vs_vanilla_bar": gap_v_bar,
        "gap_vs_fbt_bar": gap_f_bar,
        "every_seed_alf_best": every,
        "chance_nats": float(raw["chance_nats"]),
        "wall_s": float(raw["wall_s"]),
    }


def _write_tables(status: dict[str, Any], raw: dict[str, Any]) -> None:
    lines = [
        "| seed | vanilla nats | FBT nats | ALF nats | vanilla − ALF | FBT − ALF | ALF acc |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    csv_rows = []
    for row in status["seeds"]:
        lines.append(
            f"| {row['seed']} | {row['vanilla']:.4f} | {row['fbt']:.4f} | {row['alf']:.4f} | "
            f"{row['vanilla'] - row['alf']:.4f} | {row['fbt'] - row['alf']:.4f} | {row['acc']['alf']:.4f} |"
        )
        for mode in MODES:
            csv_rows.append(
                {
                    "seed": row["seed"],
                    "model": mode,
                    "answer_nats": f"{row[mode]:.6f}",
                    "answer_acc": f"{row['acc'][mode]:.6f}",
                    "full_nats": f"{row['full_nats'][mode]:.6f}",
                    "n_params": row["n_params"][mode],
                }
            )
    lines.append(
        f"| mean | {status['mean']['vanilla']:.4f} ± {status['std']['vanilla']:.4f} | "
        f"{status['mean']['fbt']:.4f} ± {status['std']['fbt']:.4f} | "
        f"{status['mean']['alf']:.4f} ± {status['std']['alf']:.4f} | "
        f"{status['gap_vs_vanilla']:.4f} | {status['gap_vs_fbt']:.4f} |  |"
    )
    (EXP / "results.md").write_text("\n".join(lines) + "\n")
    with (EXP / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    bp = raw["breakpoints"]
    bp_lines = [
        "# Breakpoints (seed 0 unless noted). Not part of the status.",
        "",
        "## Sequence length",
        "",
        "Fresh traces, not the status set.",
        "",
        "| rounds | length | vanilla | FBT | ALF |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in bp["sequence_length"]:
        bp_lines.append(
            f"| {row['n_rounds']} | {row['length']} | {row['vanilla']:.4f} | {row['fbt']:.4f} | {row['alf']:.4f} |"
        )
    bp_lines += [
        "",
        "## Jacobi passes",
        "",
        "Vanilla stays at 1 pass. FBT and ALF use the listed pass count; the metric is the last pass.",
        "",
        "| passes | vanilla | FBT | ALF |",
        "|---:|---:|---:|---:|",
    ]
    for row in bp["eval_passes"]:
        bp_lines.append(f"| {row['eval_passes']} | {row['vanilla']:.4f} | {row['fbt']:.4f} | {row['alf']:.4f} |")
    bp_lines += [
        "",
        "## Inference read (trained ALF, not retrained)",
        "",
        "`nearest` is a uniform window of four eligible sources. `last` reads only the deepest sublayer output. Input injection stays on.",
        "",
        "| read | vanilla | FBT | ALF |",
        "|---|---:|---:|---:|",
    ]
    for row in bp["read_mode"]:
        bp_lines.append(f"| {row['read_mode']} | {row['vanilla']:.4f} | {row['fbt']:.4f} | {row['alf']:.4f} |")
    bp_lines += [
        "",
        "## Depth (retrained, seed 0, same step budget)",
        "",
        "Layer count 4 is the main run, copied here. Other depths are separate trainings.",
        "",
        "| layers | vanilla | FBT | ALF |",
        "|---:|---:|---:|---:|",
    ]
    main = raw["seeds"]["0"]
    depth_rows = [
        {
            "n_layers": int(main["n_layers"]),
            "answer_nats": {mode: float(main[mode]["final"]["answer_nats"]) for mode in MODES},
        },
        *bp["depth"],
    ]
    depth_rows.sort(key=lambda row: int(row["n_layers"]))
    for row in depth_rows:
        ans = row["answer_nats"]
        bp_lines.append(
            f"| {row['n_layers']} | {float(ans['vanilla']):.4f} | {float(ans['fbt']):.4f} | {float(ans['alf']):.4f} |"
        )
    bp_lines += [
        "",
        "## Low-rank memory (ALF retrained, seed 0)",
        "",
        "Rank 4 is the main run. Ranks 1 and 16 are separate ALF trainings at the same width and step budget. FBT and vanilla are the main seed-0 numbers.",
        "",
        "| rank | ALF nats | ALF params |",
        "|---:|---:|---:|",
        f"| 4 | {float(main['alf']['final']['answer_nats']):.4f} | {int(main['alf']['n_params'])} |",
    ]
    for row in bp["rank"]:
        bp_lines.append(f"| {row['rank']} | {float(row['answer_nats']):.4f} | {int(row['n_params'])} |")
    (EXP / "breakpoints.md").write_text("\n".join(bp_lines) + "\n")


def _curves(raw: dict[str, Any], seeds: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    for mode in MODES:
        stacks = []
        steps = None
        for seed in seeds:
            curve = raw["seeds"][seed][mode]["curve"]
            xs = np.asarray([row["step"] for row in curve], dtype=np.float64)
            ys = np.asarray([row["answer_nats"] for row in curve], dtype=np.float64)
            if steps is None:
                steps = xs
            elif not np.array_equal(steps, xs):
                raise RuntimeError(f"curve steps differ for {mode} seed {seed}")
            stacks.append(ys)
        mat = np.stack(stacks, axis=0)
        out[mode] = (steps, mat.mean(axis=0), mat.std(axis=0, ddof=1))
    return out


def _plot(raw: dict[str, Any], status: dict[str, Any], cfg: dict[str, Any]) -> None:
    _style()
    seeds = [str(s) for s in cfg["seeds"]]
    curves = _curves(raw, seeds)
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), dpi=160)
    ax = axes[0, 0]
    for mode in MODES:
        xs, mean, std = curves[mode]
        ax.plot(xs, mean, color=COLORS[mode], lw=2.0, label=LABELS[mode])
        ax.fill_between(xs, mean - std, mean + std, color=COLORS[mode], alpha=0.18, linewidth=0)
    ax.axhline(status["chance_nats"], color="#666666", ls="--", lw=0.8, label="Chance")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Probe answer NLL (nats)")
    ax.set_title("Probe answer loss")
    _bold_legend(ax, frameon=False, fontsize=8)

    ax = axes[0, 1]
    xs = np.arange(len(MODES))
    means = [status["mean"][mode] for mode in MODES]
    stds = [status["std"][mode] for mode in MODES]
    ax.bar(xs, means, yerr=stds, color=[COLORS[m] for m in MODES], width=0.72, capsize=3, error_kw={"elinewidth": 1})
    ax.axhline(status["chance_nats"], color="#666666", ls="--", lw=0.8)
    ax.set_xticks(xs, [LABELS[m] for m in MODES])
    ax.set_ylabel("Held-out answer NLL (nats)")
    ax.set_title(f"Status {status['status']}")
    for x, mean in zip(xs, means, strict=True):
        ax.text(x, mean + 0.03, f"{mean:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax = axes[1, 0]
    length = raw["breakpoints"]["sequence_length"]
    lx = [int(row["length"]) for row in length]
    for mode in MODES:
        ax.plot(lx, [float(row[mode]) for row in length], color=COLORS[mode], lw=2.0, marker="o", ms=4, label=LABELS[mode])
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Answer NLL (nats)")
    ax.set_title("Breakpoint: length (seed 0)")
    _bold_legend(ax, frameon=False, fontsize=8)

    ax = axes[1, 1]
    beta = np.asarray(raw["seeds"]["0"]["alf"]["beta"], dtype=np.float64)
    masked = np.ma.masked_where(np.tri(beta.shape[0], beta.shape[1], k=-1, dtype=bool), beta)
    image = ax.imshow(masked, cmap="viridis", vmin=0.0, vmax=max(0.2, float(np.nanmax(beta))))
    ax.set_xlabel("Source sublayer")
    ax.set_ylabel("Receiving sublayer")
    ax.set_title("ALF depth weights, seed 0")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Mean β")
    fig.suptitle(
        "Attention Latent Feedback toy — register checksum",
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout()
    ART.mkdir(exist_ok=True)
    for folder in (EXP, ART):
        fig.savefig(folder / "plot.png", bbox_inches="tight")
        fig.savefig(folder / "plot.pdf", bbox_inches="tight")
    fig.savefig(ART / "alf-feedback-transformer.png", bbox_inches="tight")
    fig.savefig(ART / "alf-feedback-transformer.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with (EXP / "config.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    raw = json.loads((EXP / "raw.json").read_text())
    status = score(raw, cfg)
    _write_tables(status, raw)
    _plot(raw, status, cfg)
    (EXP / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({k: status[k] for k in ("status", "mean", "gap_vs_vanilla", "gap_vs_fbt", "every_seed_alf_best")}, indent=2))


if __name__ == "__main__":
    main()
