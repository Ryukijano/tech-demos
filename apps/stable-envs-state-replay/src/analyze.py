"""Apply the frozen pass rule and write the table, plot, and status."""

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
EXP = APP / "experiments" / "exp1_cpu_state_replay"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif", "Times New Roman"],
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
    }
)

PNG_COLOR = "#0072B2"
NPZ_COLOR = "#E69F00"
REPLAY_COLOR = "#009E73"
LOAD_COLOR = "#D55E00"


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1))


def judge(env_rows: list[dict]) -> tuple[str, dict]:
    """Return status and the per-check booleans. Rule is copied from config.yaml."""
    h3 = all(row["lossless_min"] == 1.0 for row in env_rows)
    per_env = []
    for row in env_rows:
        h1 = row["saving_png_mean"] >= 90.0
        h2 = row["saving_npz_mean"] >= 90.0
        h4 = row["replay_fps_mean"] >= row["load_fps_mean"]
        per_env.append({"env": row["env"], "h1": h1, "h2": h2, "h4": h4})
    passed = h3 and all(item["h1"] and item["h2"] and item["h4"] for item in per_env)
    disk_both_missed = any((not item["h1"]) and (not item["h2"]) for item in per_env)
    if passed:
        status = "PASS"
    elif (not h3) or disk_both_missed:
        status = "FAIL"
    else:
        status = "PARTIAL"
    return status, {"h3": h3, "per_env": per_env, "disk_both_missed": disk_both_missed}


def bold_legend(ax, **kwargs):
    legend = ax.legend(frameon=True, prop={"weight": "bold", "size": 8}, **kwargs)
    legend.get_frame().set_linewidth(0.4)
    return legend


def plot(env_rows: list[dict], path_png: Path, path_pdf: Path) -> None:
    names = [row["env"] for row in env_rows]
    x = np.arange(len(names))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.15), dpi=300)

    ax = axes[0]
    png = [row["saving_png_mean"] for row in env_rows]
    npz = [row["saving_npz_mean"] for row in env_rows]
    png_err = [row["saving_png_std"] for row in env_rows]
    npz_err = [row["saving_npz_std"] for row in env_rows]
    ax.bar(x - width / 2, png, width, yerr=png_err, color=PNG_COLOR, label="vs PNG", capsize=2, ecolor="black")
    ax.bar(x + width / 2, npz, width, yerr=npz_err, color=NPZ_COLOR, label="vs npz", capsize=2, ecolor="black")
    ax.axhline(90, color="#000000", linewidth=0.8, linestyle="--", label="90% bar")
    ax.set_ylabel("Disk saving (%)", fontweight="bold")
    ax.set_xticks(x, names)
    low = min([0.0, *png, *npz]) - 5
    ax.set_ylim(low, 100)
    ax.set_title("State file vs stored frames", fontweight="bold")
    bold_legend(ax, loc="lower right")

    ax = axes[1]
    replay = [row["replay_fps_mean"] for row in env_rows]
    load = [row["load_fps_mean"] for row in env_rows]
    replay_err = [row["replay_fps_std"] for row in env_rows]
    load_err = [row["load_fps_std"] for row in env_rows]
    ax.bar(
        x - width / 2,
        replay,
        width,
        yerr=replay_err,
        color=REPLAY_COLOR,
        label="State replay",
        capsize=2,
        ecolor="black",
    )
    ax.bar(
        x + width / 2,
        load,
        width,
        yerr=load_err,
        color=LOAD_COLOR,
        label="Faster frame load",
        capsize=2,
        ecolor="black",
    )
    ax.set_ylabel("Throughput (frames/s)", fontweight="bold")
    ax.set_xticks(x, names)
    ax.set_title("Warm-cache replay vs load", fontweight="bold")
    bold_legend(ax, loc="best")

    fig.tight_layout()
    fig.savefig(path_png, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def results_md(cfg: dict, raw: dict, env_rows: list[dict], status: str, checks: dict) -> str:
    lines = [
        f"# {cfg['exp_id']}",
        "",
        f"**Status: {status}**",
        "",
        cfg["hypothesis"].strip(),
        "",
        f"Config SHA-256 `{raw['config_sha256']}`.",
        "",
        "| Env | PNG saving % | npz saving % | Lossless min | Replay fps | Load fps | H1 | H2 | H4 |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    by_name = {item["env"]: item for item in checks["per_env"]}
    for row in env_rows:
        flag = by_name[row["env"]]
        lines.append(
            "| {env} | {png} ± {pngs} | {npz} ± {npzs} | {los} | {rep} ± {reps} | {load} ± {loads} | {h1} | {h2} | {h4} |".format(
                env=row["env"],
                png=fmt(row["saving_png_mean"]),
                pngs=fmt(row["saving_png_std"]),
                npz=fmt(row["saving_npz_mean"]),
                npzs=fmt(row["saving_npz_std"]),
                los=fmt(row["lossless_min"], 3),
                rep=fmt(row["replay_fps_mean"], 1),
                reps=fmt(row["replay_fps_std"], 1),
                load=fmt(row["load_fps_mean"], 1),
                loads=fmt(row["load_fps_std"], 1),
                h1="yes" if flag["h1"] else "no",
                h2="yes" if flag["h2"] else "no",
                h4="yes" if flag["h4"] else "no",
            )
        )
    lines.append("")
    lines.append(f"H3 (every cell lossless): {'yes' if checks['h3'] else 'no'}.")
    lines.append("")
    lines.append("## Cells")
    lines.append("")
    lines.append("| Env | Seed | Frames | State B | PNG B | npz B | PNG % | npz % | Lossless | Replay fps | npz fps | PNG fps |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cell in raw["cells"]:
        lines.append(
            "| {env} | {seed} | {n} | {sb} | {pb} | {zb} | {ps:.2f} | {zs:.2f} | {lf:.3f} | {rf:.1f} | {nf:.1f} | {pf:.1f} |".format(
                env=cell["env"],
                seed=cell["seed"],
                n=cell["n_frames"],
                sb=cell["state_bytes"],
                pb=cell["png_bytes"],
                zb=cell["npz_bytes"],
                ps=cell["disk_saving_vs_png_pct"],
                zs=cell["disk_saving_vs_npz_pct"],
                lf=cell["lossless_fraction"],
                rf=cell["state_replay_fps"],
                nf=cell["npz_load_fps"],
                pf=cell["png_load_fps"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def breakpoints_md(raw: dict) -> str:
    bp = raw["breakpoints"]
    lines = ["# Breakpoints (report-only)", ""]
    lines.append("## Non-native resolution")
    lines.append("")
    for item in bp["non_native_resolution"]:
        if item["accepted"]:
            lines.append(f"- {item['env']} size {item['size']}: accepted, shape {item['shape']}")
        else:
            lines.append(f"- {item['env']} size {item['size']}: refused. `{item['error']}`")
    up = bp["upsample_tworoom_seed0"]
    lines.append("")
    lines.append(
        f"Nearest 2× upsample of TwoRoom seed 0 is {up['up_shape']} and {up['up_npz_bytes']} bytes "
        f"versus native npz {up['native_npz_bytes']} bytes. {up['note']}"
    )
    lines.append("")
    for key in ("long_episode", "many_envs", "third_family"):
        cell = bp[key]
        lines.append(f"## {key}")
        lines.append("")
        lines.append(
            f"- {cell['env']} B={cell['num_envs']} T={cell['steps']} frames={cell['n_frames']}: "
            f"PNG saving {cell['disk_saving_vs_png_pct']:.2f}%, npz saving {cell['disk_saving_vs_npz_pct']:.2f}%, "
            f"lossless {cell['lossless_fraction']:.3f}, replay {cell['state_replay_fps']:.1f} fps, "
            f"load {cell['frame_load_fps']:.1f} fps, max abs {cell['max_abs']}."
        )
        if "repeated_position_fraction" in cell:
            lines.append(f"- Repeated-position fraction: {cell['repeated_position_fraction']}.")
        lines.append("")
    var = bp["variation_omitted"]
    lines.append("## Variation omitted")
    lines.append("")
    lines.append(
        f"- TwoRoom collected with agent.color {var['agent_color']}. "
        f"Replay at the default appearance matches {var['default_replay']['lossless_fraction']:.3f} of frames "
        f"(max abs {var['default_replay']['max_abs']}). "
        f"Passing the stored variation row back to render matches {var['variation_restored']['lossless_fraction']:.3f} "
        f"(max abs {var['variation_restored']['max_abs']})."
    )
    lines.append(
        f"- State file {var['state_bytes']} bytes; state plus variation row {var['state_plus_variation_bytes']} bytes."
    )
    lines.append("")
    fresh = bp["fresh_process"]
    lines.append("## Fresh process")
    lines.append("")
    lines.append(
        f"- Parent collection hash `{fresh['parent_sha256']}` vs a new process re-render `{fresh['child_sha256']}`: "
        f"{'match' if fresh['match'] else 'DIFFER'}."
    )
    lines.append("")
    extreme = bp["png_compress_level_extreme"]
    lines.append("## PNG compress level 9")
    lines.append("")
    lines.append(
        f"- TwoRoom seed 0 at compress_level {extreme['compress_level']}: "
        f"{extreme['png_bytes']} bytes, saving vs states {extreme['disk_saving_vs_png_pct']:.2f}%."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    raw = json.loads((EXP / "raw.json").read_text())
    names = list(cfg["envs"])
    env_rows = []
    for name in names:
        cells = [cell for cell in raw["cells"] if cell["env"] == name]
        if len(cells) != 3:
            raise RuntimeError(f"{name} has {len(cells)} cells, expected 3 seeds")
        png_m, png_s = mean_std([cell["disk_saving_vs_png_pct"] for cell in cells])
        npz_m, npz_s = mean_std([cell["disk_saving_vs_npz_pct"] for cell in cells])
        rep_m, rep_s = mean_std([cell["state_replay_fps"] for cell in cells])
        load_m, load_s = mean_std([cell["frame_load_fps"] for cell in cells])
        env_rows.append(
            {
                "env": name,
                "saving_png_mean": png_m,
                "saving_png_std": png_s,
                "saving_npz_mean": npz_m,
                "saving_npz_std": npz_s,
                "replay_fps_mean": rep_m,
                "replay_fps_std": rep_s,
                "load_fps_mean": load_m,
                "load_fps_std": load_s,
                "lossless_min": min(cell["lossless_fraction"] for cell in cells),
            }
        )
    status, checks = judge(env_rows)
    (EXP / "results.md").write_text(results_md(cfg, raw, env_rows, status, checks))
    (EXP / "breakpoints.md").write_text(breakpoints_md(raw))
    plot(env_rows, EXP / "plot.png", EXP / "plot.pdf")
    with (EXP / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw["cells"][0].keys()))
        writer.writeheader()
        for cell in raw["cells"]:
            flat = {key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in cell.items()}
            writer.writerow(flat)
    status_doc = {
        "status": status,
        "config_sha256": raw["config_sha256"],
        "checks": checks,
        "envs": env_rows,
        "wall_seconds": raw["wall_seconds"],
    }
    (EXP / "status.json").write_text(json.dumps(status_doc, indent=2) + "\n")
    with (EXP / "breakpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "env", "n_frames", "disk_saving_vs_png_pct", "disk_saving_vs_npz_pct", "lossless_fraction", "state_replay_fps", "frame_load_fps", "max_abs"],
        )
        writer.writeheader()
        for label in ("long_episode", "many_envs", "third_family"):
            cell = raw["breakpoints"][label]
            writer.writerow({key: cell.get(key, "") for key in writer.fieldnames} | {"label": label})
    print(json.dumps({"status": status, "envs": env_rows}, indent=2))


if __name__ == "__main__":
    main()
