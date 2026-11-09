"""Tables, PASS/FAIL, and the tunneling-rate figure from raw.json."""

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
EXP = APP / "experiments" / "exp1_cpu_tunnel"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Liberation Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
    }
)

COLOR = {"on": "#0072B2", "off": "#D55E00", "kinematic": "#4D4D4D"}
LABEL = {"on": "Speculative on", "off": "Speculative off", "kinematic": "Kinematic discrete skip"}


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
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def _primary(raw: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in raw["primary"] if row.get("family", "primary") == "primary"]


def _cell(rows: list[dict[str, Any]], seed: int, mode: str, dt: float) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if int(row["seed"]) == seed and row["mode"] == mode and abs(float(row["dt"]) - dt) < 1e-12
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one cell for seed={seed} mode={mode} dt={dt}, found {len(matches)}")
    return matches[0]


def _at(timesteps: list[float], multiple: int) -> float:
    target = 0.002 * multiple
    return min(timesteps, key=lambda dt: abs(dt - target))


def _judge(summary: dict[str, Any], thresholds: dict[str, Any], timesteps: list[float]) -> tuple[str, dict[str, bool]]:
    on_max = float(thresholds["on_max_rate"])
    off_1x = float(thresholds["off_max_rate_at_1x"])
    off_4x = float(thresholds["off_min_rate_at_4x"])
    off_8x = float(thresholds["off_min_rate_at_8x"])
    loose = float(thresholds["partial_on_max_rate"])
    gap_need = float(thresholds["partial_min_gap_at_8x"])
    dt1 = _at(timesteps, 1)
    dt4 = _at(timesteps, 4)
    dt8 = _at(timesteps, 8)
    on_ok = all(summary["on"][dt]["rate_mean"] <= on_max for dt in timesteps)
    hold_1x = summary["off"][dt1]["rate_mean"] <= off_1x
    floor_4x = summary["off"][dt4]["rate_mean"] >= off_4x
    floor_8x = summary["off"][dt8]["rate_mean"] >= off_8x
    on_loose = all(summary["on"][dt]["rate_mean"] <= loose for dt in timesteps)
    gap_8x = (summary["off"][dt8]["rate_mean"] - summary["on"][dt8]["rate_mean"]) >= gap_need
    gap_4x = summary["off"][dt4]["rate_mean"] > summary["on"][dt4]["rate_mean"]
    checks = {
        "on_rate_at_most_0.05_all_primary_dts": on_ok,
        "off_rate_at_most_0.10_at_1x": hold_1x,
        "off_rate_at_least_0.35_at_4x": floor_4x,
        "off_rate_at_least_0.60_at_8x": floor_8x,
        "partial_on_rate_at_most_0.20": on_loose,
        "partial_gap_at_least_0.25_at_8x": gap_8x,
        "partial_off_above_on_at_4x": gap_4x,
    }
    if on_ok and hold_1x and floor_4x and floor_8x:
        status = "PASS"
    elif on_loose and gap_8x and gap_4x:
        status = "PARTIAL"
    else:
        status = "FAIL"
    return status, checks


def _summary(raw: dict[str, Any], timesteps: list[float]) -> dict[str, Any]:
    rows = _primary(raw)
    seeds = [int(seed) for seed in raw["seeds"]]
    out: dict[str, Any] = {"on": {}, "off": {}, "per_seed": []}
    for dt in timesteps:
        for mode in ("off", "on"):
            rates = []
            pens = []
            drifts = []
            contacts = []
            kins = []
            for seed in seeds:
                cell = _cell(rows, seed, mode, dt)["summary"]
                rates.append(float(cell["tunnel_rate"]))
                pens.append(float(cell["mean_max_penetration_m"]))
                drifts.append(float(cell["mean_ke_rel_drift"]))
                contacts.append(float(cell["contact_rate"]))
                kins.append(float(cell["kinematic_skip_rate"]))
                out["per_seed"].append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "dt": dt,
                        "multiplier": dt / 0.002,
                        **{k: cell[k] for k in (
                            "n_shots",
                            "tunnel_rate",
                            "contact_rate",
                            "mean_max_penetration_m",
                            "mean_ke_rel_drift",
                            "kinematic_skip_rate",
                            "wall_s",
                            "n_diverged",
                        )},
                    }
                )
            rate_mean, rate_std = _mean_std(rates)
            pen_mean, pen_std = _mean_std(pens)
            drift_mean, drift_std = _mean_std(drifts)
            contact_mean, _ = _mean_std(contacts)
            kin_mean, kin_std = _mean_std(kins)
            out[mode][dt] = {
                "rate_mean": rate_mean,
                "rate_std": rate_std,
                "pen_mean": pen_mean,
                "pen_std": pen_std,
                "drift_mean": drift_mean,
                "drift_std": drift_std,
                "contact_mean": contact_mean,
                "kin_mean": kin_mean,
                "kin_std": kin_std,
            }
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary: dict[str, Any], timesteps: list[float], path: Path) -> None:
    xs = np.array([dt * 1e3 for dt in timesteps], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), constrained_layout=True)
    ax = axes[0]
    for mode in ("off", "on"):
        means = np.array([summary[mode][dt]["rate_mean"] for dt in timesteps])
        stds = np.array([summary[mode][dt]["rate_std"] for dt in timesteps])
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            color=COLOR[mode],
            marker="o",
            markersize=7,
            linewidth=2.2,
            capsize=3.5,
            label=LABEL[mode],
        )
    kin = np.array([summary["off"][dt]["kin_mean"] for dt in timesteps])
    ax.plot(
        xs,
        kin,
        color=COLOR["kinematic"],
        linestyle="--",
        linewidth=1.8,
        marker="s",
        markersize=5,
        label=LABEL["kinematic"],
    )
    ax.set_xlabel("Timestep (ms)")
    ax.set_ylabel("Tunneling rate")
    ax.set_title("Sphere vs 4 mm wall")
    ax.set_ylim(-0.05, 1.55)
    ax.set_xticks(xs)
    _bold_legend(ax, frameon=False, loc="upper left", fontsize=9, borderpad=0.2)

    ax = axes[1]
    for mode in ("off", "on"):
        means = np.array([summary[mode][dt]["pen_mean"] * 1e3 for dt in timesteps])
        stds = np.array([summary[mode][dt]["pen_std"] * 1e3 for dt in timesteps])
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            color=COLOR[mode],
            marker="o",
            markersize=7,
            linewidth=2.2,
            capsize=3.5,
            label=LABEL[mode],
        )
    ax.set_xlabel("Timestep (ms)")
    ax.set_ylabel("Mean peak contact penetration (mm)")
    ax.set_title("Contact penetration")
    ax.set_ylim(-0.15, 7.4)
    ax.set_xticks(xs)
    _bold_legend(ax, frameon=False, loc="upper right", fontsize=9, borderpad=0.2)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _fmt_rate(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def _write_results_md(
    cfg: dict[str, Any],
    raw: dict[str, Any],
    summary: dict[str, Any],
    status: str,
    checks: dict[str, bool],
    timesteps: list[float],
) -> None:
    lines = [
        "# exp1_cpu_tunnel results",
        "",
        f"Status: **{status}**",
        "",
        f"MuJoCo {raw['mujoco']['version']}. `mjENBL_SPECULATIVE` = {raw['mujoco']['mjENBL_SPECULATIVE']} (XML `speculative`, default `{raw['mujoco']['default']}`).",
        "",
        "Pre-registered PASS rule: speculative ON tunneling rate ≤ 0.05 at 2, 4, 8 and 16 ms; speculative OFF ≤ 0.10 at 2 ms, ≥ 0.35 at 8 ms, and ≥ 0.60 at 16 ms. Means over seeds 0/1/2.",
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
        "| dt (ms) | × default | OFF rate | ON rate | Kinematic skip | OFF penetration (mm) | ON penetration (mm) | OFF KE drift | ON KE drift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dt in timesteps:
        off = summary["off"][dt]
        on = summary["on"][dt]
        lines.append(
            f"| {dt * 1e3:.0f} | {dt / 0.002:.0f}× | {_fmt_rate(off['rate_mean'], off['rate_std'])} | "
            f"{_fmt_rate(on['rate_mean'], on['rate_std'])} | {_fmt_rate(off['kin_mean'], off['kin_std'])} | "
            f"{off['pen_mean'] * 1e3:.3f} ± {off['pen_std'] * 1e3:.3f} | "
            f"{on['pen_mean'] * 1e3:.3f} ± {on['pen_std'] * 1e3:.3f} | "
            f"{off['drift_mean']:.3f} ± {off['drift_std']:.3f} | "
            f"{on['drift_mean']:.3f} ± {on['drift_std']:.3f} |"
        )
    lines += [
        "",
        "Kinematic skip is the fraction of the same shots whose evenly spaced free-flight samples miss the wall. It is not a MuJoCo result. Contact penetration is the mean, over shots, of the deepest negative contact distance seen by `mj_step1` (0 if that mode never touched).",
        "",
        "## Per seed",
        "",
        "| Seed | dt (ms) | OFF | ON | Kinematic | OFF contact | ON contact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in raw["seeds"]:
        for dt in timesteps:
            off = _cell(_primary(raw), int(seed), "off", dt)["summary"]
            on = _cell(_primary(raw), int(seed), "on", dt)["summary"]
            lines.append(
                f"| {seed} | {dt * 1e3:.0f} | {off['tunnel_rate']:.3f} | {on['tunnel_rate']:.3f} | "
                f"{off['kinematic_skip_rate']:.3f} | {off['contact_rate']:.3f} | {on['contact_rate']:.3f} |"
            )
    clocks = raw["wallclock"]
    lines += [
        "",
        "## Wall-clock (`mj_step`, after warmup)",
        "",
        "| Spheres | dt (ms) | OFF µs/step | ON µs/step | ON / OFF |",
        "|---|---:|---:|---:|---:|",
    ]
    pairs: dict[tuple[int, float], dict[str, float]] = {}
    for row in clocks:
        key = (int(row["n_spheres"]), float(row["dt"]))
        pairs.setdefault(key, {})
        pairs[key]["on" if row["speculative"] else "off"] = float(row["us_per_step"])
    for (n_spheres, dt), modes in sorted(pairs.items()):
        ratio = modes["on"] / modes["off"] if modes["off"] else float("nan")
        lines.append(
            f"| {n_spheres} | {dt * 1e3:.0f} | {modes['off']:.2f} | {modes['on']:.2f} | {ratio:.2f} |"
        )
    lines.append("")
    (EXP / "results.md").write_text("\n".join(lines))


def _breakpoint_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group in raw["breakpoints"]:
        shot = group["shots"][0]
        summary = group["summary"]
        rows.append(
            {
                "family": group["family"],
                "mode": group["mode"],
                "parameter": group["parameter"],
                "dt": group["dt"],
                "tunneled": int(bool(shot["tunneled"])),
                "contacted": int(bool(shot["contacted"])),
                "diverged": int(bool(shot["diverged"])),
                "max_penetration_m": shot["max_penetration_m"],
                "max_x": shot["max_x"],
                "x_after_first": shot["x_after_first"],
                "ke_rel_drift": shot["ke_rel_drift"],
                "kinematic_skip_rate": summary["kinematic_skip_rate"],
                "wall_s": summary["wall_s"],
            }
        )
    return rows


def _write_breakpoints(raw: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    def pick(family: str) -> list[dict[str, Any]]:
        return [row for row in rows if row["family"] == family]

    def yn(value: int) -> str:
        return "yes" if int(value) else "no"

    lines = [
        "# Breakpoints",
        "",
        "Seed 0. One controlled projectile per cell, not the random primary box. These cells do not change the PASS rule. `tunneled` means the sphere finished strictly on the far side of the wall (or the state diverged).",
        "",
        "## Huge initial speed",
        "",
        "4 mm wall, radius 7 mm, the media-shot geometry. Speculative contacts sweep pre-step velocity, so a large constant speed is the case they are built for.",
        "",
        "| Speed (m/s) | dt (ms) | OFF tunneled | ON tunneled | OFF contact | ON contact |",
        "|---:|---:|---|---|---|---|",
    ]
    for row_off in pick("huge_velocity"):
        if row_off["mode"] != "off":
            continue
        row_on = next(
            row
            for row in pick("huge_velocity")
            if row["mode"] == "on" and row["parameter"] == row_off["parameter"] and abs(row["dt"] - row_off["dt"]) < 1e-12
        )
        lines.append(
            f"| {row_off['parameter']:.0f} | {row_off['dt'] * 1e3:.0f} | {yn(row_off['tunneled'])} | {yn(row_on['tunneled'])} | "
            f"{yn(row_off['contacted'])} | {yn(row_on['contacted'])} |"
        )
    lines += [
        "",
        "## Huge timestep",
        "",
        "Speed 4 m/s, radius 7 mm, 4 mm wall. No applied acceleration, so a miss here is numerical rather than the documented acceleration gap.",
        "",
        "| dt (ms) | × default | OFF tunneled | ON tunneled | OFF contact | ON contact | diverged OFF/ON |",
        "|---:|---:|---|---|---|---|---|",
    ]
    for row_off in pick("huge_timestep"):
        if row_off["mode"] != "off":
            continue
        row_on = next(
            row
            for row in pick("huge_timestep")
            if row["mode"] == "on" and abs(float(row["parameter"]) - float(row_off["parameter"])) < 1e-12
        )
        lines.append(
            f"| {row_off['dt'] * 1e3:.0f} | {row_off['dt'] / 0.002:.0f}× | {yn(row_off['tunneled'])} | {yn(row_on['tunneled'])} | "
            f"{yn(row_off['contacted'])} | {yn(row_on['contacted'])} | {yn(row_off['diverged'])}/{yn(row_on['diverged'])} |"
        )
    lines += [
        "",
        "## Thinner wall",
        "",
        "Speed 4 m/s, radius 7 mm, timestep 16 ms. Half-thickness is the parameter.",
        "",
        "| Half-thickness (mm) | OFF tunneled | ON tunneled | OFF contact | ON contact |",
        "|---:|---|---|---|---|",
    ]
    for row_off in pick("thinner_wall"):
        if row_off["mode"] != "off":
            continue
        row_on = next(
            row
            for row in pick("thinner_wall")
            if row["mode"] == "on" and abs(float(row["parameter"]) - float(row_off["parameter"])) < 1e-15
        )
        lines.append(
            f"| {float(row_off['parameter']) * 1e3:.4f} | {yn(row_off['tunneled'])} | {yn(row_on['tunneled'])} | "
            f"{yn(row_off['contacted'])} | {yn(row_on['contacted'])} |"
        )
    lines += [
        "",
        "## Acceleration from rest",
        "",
        "Documented limit: the sweep uses pre-step `qvel` and ignores acceleration. The sphere starts at rest. The first-step Euler displacement is `a dt²` with dt = 16 ms. A 1 mm wall (half-thickness 0.5 mm) sits in front of a 6 mm sphere launched from x = −0.03 m.",
        "",
        "| Gravity (m/s²) | First-step x (m) | OFF tunneled | ON tunneled | OFF contact | ON contact |",
        "|---:|---:|---|---|---|---|",
    ]
    for row_off in pick("accel_from_rest"):
        if row_off["mode"] != "off":
            continue
        row_on = next(
            row
            for row in pick("accel_from_rest")
            if row["mode"] == "on" and abs(float(row["parameter"]) - float(row_off["parameter"])) < 1e-12
        )
        lines.append(
            f"| {float(row_off['parameter']):.0f} | {float(row_off['x_after_first']):.4f} | {yn(row_off['tunneled'])} | "
            f"{yn(row_on['tunneled'])} | {yn(row_off['contacted'])} | {yn(row_on['contacted'])} |"
        )
    lines += [
        "",
        "## Wall-clock",
        "",
        "See the table in `results.md`. Cost is microseconds per `mj_step` on this CPU, flag off versus on.",
        "",
    ]
    (EXP / "breakpoints.md").write_text("\n".join(lines))


def main() -> None:
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    raw = json.loads((EXP / "raw.json").read_text())
    timesteps = [float(dt) for dt in cfg["scene"]["timesteps_s"]]
    summary = _summary(raw, timesteps)
    status, checks = _judge(summary, cfg["thresholds"], timesteps)
    _write_csv(EXP / "results.csv", summary["per_seed"])
    _write_results_md(cfg, raw, summary, status, checks, timesteps)
    _plot(summary, timesteps, EXP / "plot.png")
    bp_rows = _breakpoint_rows(raw)
    _write_csv(EXP / "breakpoints.csv", bp_rows)
    _write_breakpoints(raw, bp_rows)
    status_doc = {
        "status": status,
        "checks": checks,
        "mujoco": raw["mujoco"],
        "on_rate_mean": {str(dt): summary["on"][dt]["rate_mean"] for dt in timesteps},
        "off_rate_mean": {str(dt): summary["off"][dt]["rate_mean"] for dt in timesteps},
        "on_rate_std": {str(dt): summary["on"][dt]["rate_std"] for dt in timesteps},
        "off_rate_std": {str(dt): summary["off"][dt]["rate_std"] for dt in timesteps},
        "kinematic_skip_mean": {str(dt): summary["off"][dt]["kin_mean"] for dt in timesteps},
        "thresholds": cfg["thresholds"],
    }
    (EXP / "status.json").write_text(json.dumps(status_doc, indent=2) + "\n")
    print("status", status)


if __name__ == "__main__":
    main()
