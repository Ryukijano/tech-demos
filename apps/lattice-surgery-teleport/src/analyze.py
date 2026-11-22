"""Tables, the pre-registered verdict, and the publication figure."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXP = ROOT / "experiments" / "exp1_cpu_teleport"

# Okabe-Ito, colorblind-safe.
DIST_COLOR = {3: "#0072B2", 5: "#E69F00", 7: "#009E73"}


def _load() -> tuple[dict, dict, dict | None]:
    with (EXP / "config.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    raw = json.loads((EXP / "raw.json").read_text())
    bp_path = EXP / "breakpoints_raw.json"
    bp = json.loads(bp_path.read_text()) if bp_path.exists() else None
    return cfg, raw, bp


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Noto Serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
        }
    )


def _decade_label(value: float, _pos: object = None) -> str:
    if value <= 0:
        return ""
    exp = int(round(math.log10(value)))
    if abs(math.log10(value) - exp) > 1e-6:
        return ""
    return f"$10^{{{exp}}}$"


def _log_axis(ax, which: str = "both") -> None:
    fmt = FuncFormatter(_decade_label)
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_formatter(NullFormatter())
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(NullFormatter())


def _rows(records: list[dict], **match: object) -> list[dict]:
    found = []
    for row in records:
        if all(row.get(key) == value for key, value in match.items()):
            found.append(row)
    return found


def _rate(fails: int, shots: int) -> float:
    return fails / shots


def _upper(shots: int) -> float:
    return 3.0 / shots


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _basis_table(records: list[dict], p: float) -> dict:
    """Per distance, basis, seed raw rates for the scaling study."""
    table: dict[tuple[int, str, int], dict] = {}
    for row in records:
        if row["study"] != "scaling" or float(row["p"]) != float(p):
            continue
        key = (int(row["distance"]), str(row["basis"]), int(row["seed"]))
        table[key] = row
    return table


def _pbar_for_seed(
    table: dict,
    distance: int,
    seed: int,
    substitute_zeros: bool,
) -> tuple[float, bool]:
    used_bound = False
    rates = []
    for basis in ("Z", "X"):
        row = table[(distance, basis, seed)]
        fails = int(row["fails"])
        shots = int(row["shots"])
        if fails == 0 and substitute_zeros:
            rates.append(_upper(shots))
            used_bound = True
        else:
            rates.append(_rate(fails, shots))
    return 0.5 * (rates[0] + rates[1]), used_bound


def verdict(cfg: dict, raw: dict) -> dict:
    p = float(cfg["physical_error_rate"])
    table = _basis_table(raw["records"], p)
    distances = [int(d) for d in cfg["distances"]]
    seeds = [int(s) for s in cfg["seeds"]]
    bar = float(cfg["thresholds"]["lambda_step_min"])

    per_seed = []
    strict_ok = True
    for seed in seeds:
        entry: dict[str, object] = {"seed": seed, "bases": {}}
        for basis in ("Z", "X"):
            rates = []
            for distance in distances:
                row = table[(distance, basis, seed)]
                rates.append(_rate(int(row["fails"]), int(row["shots"])))
            decreasing = rates[0] > rates[1] > rates[2]
            if not decreasing:
                strict_ok = False
            entry["bases"][basis] = {
                "rates": rates,
                "strictly_decreasing": decreasing,
                "fails": [int(table[(d, basis, seed)]["fails"]) for d in distances],
                "shots": [int(table[(d, basis, seed)]["shots"]) for d in distances],
            }
        per_seed.append(entry)

    def lambda_step(left: int, right: int) -> dict:
        num = []
        den = []
        bounded = False
        for seed in seeds:
            p_left, _ = _pbar_for_seed(table, left, seed, substitute_zeros=False)
            p_right, used = _pbar_for_seed(table, right, seed, substitute_zeros=True)
            num.append(p_left)
            den.append(p_right)
            bounded = bounded or used
        num_mean, num_std = _mean_std(num)
        den_mean, den_std = _mean_std(den)
        ratio = num_mean / den_mean if den_mean > 0 else math.inf
        return {
            "from": left,
            "to": right,
            "lambda": ratio,
            "numerator_mean": num_mean,
            "numerator_std": num_std,
            "denominator_mean": den_mean,
            "denominator_std": den_std,
            "lower_bound": bounded,
            "per_seed_numerator": num,
            "per_seed_denominator": den,
        }

    step_35 = lambda_step(3, 5)
    step_57 = lambda_step(5, 7)
    ge = step_35["lambda"] >= bar and step_57["lambda"] >= bar
    both_above_one = step_35["lambda"] > 1 and step_57["lambda"] > 1
    if ge and strict_ok:
        status = "PASS"
    elif both_above_one:
        status = "PARTIAL"
    else:
        status = "FAIL"

    # Report-only per-basis lambdas, same zero rule on the larger distance.
    per_basis = {}
    for basis in ("Z", "X"):
        for left, right in ((3, 5), (5, 7)):
            nums = []
            dens = []
            bounded = False
            for seed in seeds:
                nrow = table[(left, basis, seed)]
                drow = table[(right, basis, seed)]
                nums.append(_rate(int(nrow["fails"]), int(nrow["shots"])))
                if int(drow["fails"]) == 0:
                    dens.append(_upper(int(drow["shots"])))
                    bounded = True
                else:
                    dens.append(_rate(int(drow["fails"]), int(drow["shots"])))
            n_mean, _ = _mean_std(nums)
            d_mean, _ = _mean_std(dens)
            per_basis[f"{basis}_{left}_{right}"] = {
                "lambda": n_mean / d_mean,
                "lower_bound": bounded,
            }

    summary_rows = []
    for distance in distances:
        for basis in ("Z", "X"):
            rates = []
            fails = 0
            shots = 0
            for seed in seeds:
                row = table[(distance, basis, seed)]
                rates.append(_rate(int(row["fails"]), int(row["shots"])))
                fails += int(row["fails"])
                shots += int(row["shots"])
            mean, std = _mean_std(rates)
            summary_rows.append(
                {
                    "distance": distance,
                    "basis": basis,
                    "mean": mean,
                    "std": std,
                    "pooled_fails": fails,
                    "pooled_shots": shots,
                    "pooled": fails / shots,
                }
            )

    return {
        "status": status,
        "bar": bar,
        "p": p,
        "lambda_3_to_5": step_35,
        "lambda_5_to_7": step_57,
        "strictly_decreasing_every_seed": strict_ok,
        "per_seed": per_seed,
        "per_basis_lambda": per_basis,
        "summary": summary_rows,
    }


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    if value >= 0.01:
        return f"{value:.4f}"
    return f"{value:.3e}"


def write_tables(cfg: dict, status: dict, bp: dict | None) -> None:
    lines = [
        "# Scaling results",
        "",
        f"Status: **{status['status']}**",
        "",
        f"Pre-registered bar: Lambda >= {status['bar']} at p = {status['p']}.",
        "",
        "| d | basis | mean ± sample std (seeds 0/1/2) | pooled |",
        "|---|---|---|---|",
    ]
    for row in status["summary"]:
        lines.append(
            f"| {row['distance']} | {row['basis']} | {_fmt(row['mean'])} ± {_fmt(row['std'])} "
            f"| {row['pooled_fails']}/{row['pooled_shots']} = {_fmt(row['pooled'])} |"
        )
    lines += ["", "## Lambda", ""]
    for key in ("lambda_3_to_5", "lambda_5_to_7"):
        step = status[key]
        bound = " (lower bound)" if step["lower_bound"] else ""
        lines.append(
            f"- {step['from']}→{step['to']}: Lambda = {step['lambda']:.3f}{bound} "
            f"from p_bar {_fmt(step['numerator_mean'])} ± {_fmt(step['numerator_std'])} "
            f"/ {_fmt(step['denominator_mean'])} ± {_fmt(step['denominator_std'])}"
        )
    lines.append("")
    lines.append(f"Strict decrease on every seed and both bases: {status['strictly_decreasing_every_seed']}")
    (EXP / "results.md").write_text("\n".join(lines) + "\n")

    with (EXP / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["distance", "basis", "mean", "std", "pooled_fails", "pooled_shots", "pooled"],
        )
        writer.writeheader()
        writer.writerows(status["summary"])

    if bp is None:
        return
    _write_breakpoints(cfg, bp)


def _write_breakpoints(cfg: dict, bp: dict) -> None:
    del cfg
    records = bp["records"]
    sweep = [r for r in records if r["study"] == "p_sweep"]
    ps = sorted({float(r["p"]) for r in sweep})
    csv_rows = []
    cross_2: dict[str, float | None] = {"3_5": None, "5_7": None}
    cross_1: dict[str, float | None] = {"3_5": None, "5_7": None}
    prev: dict[str, float | None] = {"3_5": None, "5_7": None}
    for p in ps:
        means: dict[int, float] = {}
        bounds: dict[int, bool] = {}
        for distance in (3, 5, 7):
            basis_rates = []
            bounded = False
            for basis in ("Z", "X"):
                row = next(
                    r
                    for r in sweep
                    if int(r["distance"]) == distance and r["basis"] == basis and float(r["p"]) == p
                )
                if int(row["fails"]) == 0:
                    basis_rates.append(_upper(int(row["shots"])))
                    bounded = True
                else:
                    basis_rates.append(_rate(int(row["fails"]), int(row["shots"])))
            means[distance] = 0.5 * (basis_rates[0] + basis_rates[1])
            bounds[distance] = bounded
        l35 = means[3] / means[5]
        l57 = means[5] / means[7]
        csv_rows.append(
            {
                "p": p,
                "p_bar_3": means[3],
                "p_bar_5": means[5],
                "p_bar_7": means[7],
                "lambda_3_5": l35,
                "lambda_5_7": l57,
                "bound_5": bounds[5],
                "bound_7": bounds[7],
            }
        )
        for name, value, bounded_den in (("3_5", l35, bounds[5]), ("5_7", l57, bounds[7])):
            if bounded_den:
                continue
            if prev[name] is not None and prev[name] >= 2 and value < 2 and cross_2[name] is None:
                cross_2[name] = p
            if prev[name] is not None and prev[name] >= 1 and value < 1 and cross_1[name] is None:
                cross_1[name] = p
            prev[name] = value
    (EXP / "breakpoints.md").write_text(_breakpoint_text(records, csv_rows, cross_2, cross_1))
    with (EXP / "breakpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)


def _breakpoint_text(records: list[dict], csv_rows: list[dict], cross_2: dict, cross_1: dict) -> str:
    lines = [
        "# Breakpoints (seed 0)",
        "",
        "Uniform circuit-level noise. p_bar is the mean of the Z-basis and X-basis failure rates.",
        "When a basis has zero failures, its rate in p_bar is the 3/shots upper bound and Lambda is a lower bound (marked *).",
        "",
        "## Failure versus p, rounds = d",
        "",
        "| p | p_bar d=3 | p_bar d=5 | p_bar d=7 | Lambda 3→5 | Lambda 5→7 |",
        "|---|---|---|---|---|---|",
    ]
    for row in csv_rows:
        def cell(value: float, bound: bool) -> str:
            return _fmt(value) + ("*" if bound else "")
        lines.append(
            f"| {row['p']:g} | {cell(row['p_bar_3'], False)} | {cell(row['p_bar_5'], row['bound_5'])} "
            f"| {cell(row['p_bar_7'], row['bound_7'])} | {row['lambda_3_5']:.3f}{'*' if row['bound_5'] else ''} "
            f"| {row['lambda_5_7']:.3f}{'*' if row['bound_7'] else ''} |"
        )
    lines += [
        "",
        f"First measured point (not a lower bound) where Lambda(3→5) falls below 2: {cross_2['3_5']}.",
        f"First measured point where Lambda(5→7) falls below 2: {cross_2['5_7']}.",
        f"First measured point where Lambda(3→5) falls below 1: {cross_1['3_5']}.",
        f"First measured point where Lambda(5→7) falls below 1: {cross_1['5_7']}.",
        "",
        "If a crossing is None, the sweep did not show that drop on a fully measured (non-star) ratio.",
        "",
        "## Missing merge rounds",
        "",
        "init_rounds = d. Compared with merge_rounds = d at the same p.",
        "",
        "| d | p | merge | basis | fails | rate |",
        "|---|---|---|---|---|---|",
    ]
    merges = sorted(
        [r for r in records if r["study"] == "merge_rounds"],
        key=lambda r: (int(r["distance"]), float(r["p"]), int(r["merge_rounds"]), r["basis"]),
    )
    for row in merges:
        lines.append(
            f"| {row['distance']} | {float(row['p']):g} | {row['merge_rounds']} | {row['basis']} "
            f"| {row['fails']}/{row['shots']} | {_fmt(_rate(int(row['fails']), int(row['shots'])))} |"
        )
    lines += [
        "",
        "## Very few rounds (init = 1, merge = 1)",
        "",
        "| d | p | basis | fails | rate |",
        "|---|---|---|---|---|",
    ]
    few = sorted(
        [r for r in records if r["study"] == "few_rounds"],
        key=lambda r: (int(r["distance"]), float(r["p"]), r["basis"]),
    )
    for row in few:
        lines.append(
            f"| {row['distance']} | {float(row['p']):g} | {row['basis']} "
            f"| {row['fails']}/{row['shots']} | {_fmt(_rate(int(row['fails']), int(row['shots'])))} |"
        )
    lines += [
        "",
        "## d = 3 with the paper's modular round counts (init = 2, merge = 3)",
        "",
        "Same uniform noise. Not the device, and not part of the pass bar.",
        "",
        "| p | basis | fails | rate |",
        "|---|---|---|---|",
    ]
    paper = sorted(
        [r for r in records if r["study"] == "paper_d3_rounds"],
        key=lambda r: (float(r["p"]), r["basis"]),
    )
    for row in paper:
        lines.append(
            f"| {float(row['p']):g} | {row['basis']} | {row['fails']}/{row['shots']} "
            f"| {_fmt(_rate(int(row['fails']), int(row['shots'])))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _binomial_err(fails: int, shots: int) -> float:
    p = fails / shots
    return math.sqrt(p * (1.0 - p) / shots)


def write_plot(cfg: dict, raw: dict, status: dict, bp: dict | None) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.3), dpi=300)
    ax_z, ax_x, ax_l, ax_b = axes.ravel()
    if bp is not None:
        sweep = [r for r in bp["records"] if r["study"] == "p_sweep"]
        for basis, ax in (("Z", ax_z), ("X", ax_x)):
            for distance in (3, 5, 7):
                rows = sorted(
                    [
                        r
                        for r in sweep
                        if r["basis"] == basis and int(r["distance"]) == distance
                    ],
                    key=lambda r: float(r["p"]),
                )
                xs = [float(r["p"]) for r in rows]
                ys = []
                lo = []
                hi = []
                for r in rows:
                    fails = int(r["fails"])
                    shots = int(r["shots"])
                    if fails == 0:
                        ys.append(_upper(shots))
                        lo.append(0.0)
                        hi.append(0.0)
                    else:
                        rate = fails / shots
                        err = _binomial_err(fails, shots)
                        ys.append(rate)
                        lo.append(min(err, rate * 0.9))
                        hi.append(err)
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[lo, hi],
                    fmt="o-",
                    color=DIST_COLOR[distance],
                    ms=3.5,
                    lw=1.4,
                    capsize=2,
                    label=f"d = {distance}",
                )
            _mark_official(ax, status, basis)
            ax.set_xscale("log")
            ax.set_yscale("log")
            _log_axis(ax)
            ax.set_xlabel("Physical error rate p")
            ax.set_ylabel("Logical teleportation failure")
            ax.set_title(f"{basis} basis", fontweight="bold")
            ax.legend(prop={"weight": "bold", "size": 8})
        lam = _lambda_from_sweep(sweep)
        ax_l.plot(
            [r["p"] for r in lam],
            [r["lambda_3_5"] for r in lam],
            "o-",
            color=DIST_COLOR[3],
            lw=1.6,
            ms=3.5,
            label="3 to 5",
        )
        ax_l.plot(
            [r["p"] for r in lam],
            [r["lambda_5_7"] for r in lam],
            "s-",
            color=DIST_COLOR[7],
            lw=1.6,
            ms=3.5,
            label="5 to 7",
        )
        ax_l.axhline(status["bar"], color="#333333", lw=0.8, ls="--", label="bar = 2")
        ax_l.axhline(1.0, color="#666666", lw=0.7, ls=":")
        ax_l.set_xscale("log")
        ax_l.set_yscale("log")
        _log_axis(ax_l)
        ax_l.set_xlabel("Physical error rate p")
        ax_l.set_ylabel("Suppression factor Lambda")
        ax_l.set_title("Seed 0, rounds = d", fontweight="bold")
        ax_l.legend(prop={"weight": "bold", "size": 8})
    # Bars at the pre-registered point.
    width = 0.36
    distances = [3, 5, 7]
    for offset, basis, color in ((-width / 2, "Z", "#0072B2"), (width / 2, "X", "#D55E00")):
        means = []
        stds = []
        for distance in distances:
            row = next(r for r in status["summary"] if r["distance"] == distance and r["basis"] == basis)
            means.append(row["mean"])
            stds.append(row["std"])
        xpos = np.arange(len(distances)) + offset
        ax_b.bar(
            xpos,
            means,
            width=width,
            yerr=stds,
            color=color,
            ecolor="#222222",
            capsize=2,
            label=f"{basis} basis",
            error_kw={"elinewidth": 0.8},
        )
    ax_b.set_xticks(np.arange(len(distances)))
    ax_b.set_xticklabels([f"d = {d}" for d in distances])
    ax_b.set_yscale("log")
    _log_axis(ax_b, "y")
    ax_b.set_ylabel("Failure at p = 0.001")
    ax_b.set_title(f"Seeds 0/1/2 · {status['status']}", fontweight="bold")
    ax_b.legend(prop={"weight": "bold", "size": 8})
    fig.tight_layout()
    fig.savefig(EXP / "plot.png", dpi=300)
    fig.savefig(EXP / "plot.pdf")
    plt.close(fig)
    _ = cfg


def _mark_official(ax, status: dict, basis: str) -> None:
    for distance, marker in ((3, "*"), (5, "*"), (7, "*")):
        row = next(r for r in status["summary"] if r["distance"] == distance and r["basis"] == basis)
        y = row["mean"] if row["mean"] > 0 else 9.0 / row["pooled_shots"]
        err = min(row["std"], y * 0.9) if y > 0 else 0.0
        ax.errorbar(
            [status["p"]],
            [y],
            yerr=[[err], [row["std"]]],
            fmt=marker,
            color=DIST_COLOR[distance],
            ms=9,
            mew=0.6,
            mec="#111111",
            zorder=5,
            capsize=2,
        )


def _lambda_from_sweep(sweep: list[dict]) -> list[dict]:
    ps = sorted({float(r["p"]) for r in sweep})
    rows = []
    for p in ps:
        means = {}
        for distance in (3, 5, 7):
            rates = []
            for basis in ("Z", "X"):
                row = next(
                    r
                    for r in sweep
                    if int(r["distance"]) == distance and r["basis"] == basis and float(r["p"]) == p
                )
                if int(row["fails"]) == 0:
                    rates.append(_upper(int(row["shots"])))
                else:
                    rates.append(int(row["fails"]) / int(row["shots"]))
            means[distance] = 0.5 * (rates[0] + rates[1])
        rows.append(
            {
                "p": p,
                "lambda_3_5": means[3] / means[5],
                "lambda_5_7": means[5] / means[7],
            }
        )
    return rows


def main() -> None:
    cfg, raw, bp = _load()
    status = verdict(cfg, raw)
    (EXP / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    write_tables(cfg, status, bp)
    write_plot(cfg, raw, status, bp)
    print(f"status {status['status']}")
    print(
        f"Lambda 3→5 {status['lambda_3_to_5']['lambda']:.3f} "
        f"5→7 {status['lambda_5_to_7']['lambda']:.3f}"
    )


if __name__ == "__main__":
    main()
