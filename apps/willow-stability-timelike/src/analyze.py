"""Tables, pass/fail status, and the publication figure from raw counts."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.decode import per_cycle_error, rule_of_three_upper  # noqa: E402

EXP = ROOT / "experiments" / "exp1_cpu_stability"

# Wong colorblind palette.
PAULI = "#0072B2"
LEAK = "#D55E00"
WALK = "#009E73"
DIST = {3: "#0072B2", 5: "#E69F00", 7: "#CC79A7"}


def _load() -> tuple[dict, dict, dict | None]:
    with (EXP / "config.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    raw = json.loads((EXP / "raw.json").read_text())
    bp_path = EXP / "breakpoints_raw.json"
    bp = json.loads(bp_path.read_text()) if bp_path.exists() else None
    return cfg, raw, bp


def _rows(raw: dict, study: str, **match: object) -> list[dict]:
    found = []
    for row in raw["records"]:
        if row["study"] != study:
            continue
        if all(row.get(key) == value for key, value in match.items()):
            found.append(row)
    return found


def _pool(rows: list[dict]) -> tuple[int, int]:
    return sum(int(r["fails"]) for r in rows), sum(int(r["shots"]) for r in rows)


def _eps_from_counts(fails: int, shots: int, cycles: int, upper_if_zero: bool) -> tuple[float, bool]:
    if fails == 0 and upper_if_zero:
        return per_cycle_error(rule_of_three_upper(shots), cycles), True
    return per_cycle_error(fails / shots, cycles), False


def _lambda_pair(small: list[dict], large: list[dict]) -> tuple[float, bool]:
    """One seed. Denominator uses a 3/N upper bound when the larger patch has no failures."""

    fs, ns = _pool(small)
    fl, nl = _pool(large)
    cycles_s = int(small[0]["cycles"])
    cycles_l = int(large[0]["cycles"])
    eps_s, _ = _eps_from_counts(fs, ns, cycles_s, upper_if_zero=False)
    eps_l, bounded = _eps_from_counts(fl, nl, cycles_l, upper_if_zero=True)
    if eps_l <= 0.0:
        return math.inf, True
    return eps_s / eps_l, bounded


def lambda_by_seed(raw: dict, study: str = "lambda") -> list[dict]:
    seeds = sorted({int(r["seed"]) for r in _rows(raw, study)})
    distances = sorted({int(r["distance"]) for r in _rows(raw, study)})
    out = []
    for seed in seeds:
        by_d = {
            d: _rows(raw, study, seed=seed, distance=d)
            for d in distances
        }
        step = {}
        bounded = False
        for left, right in zip(distances, distances[1:]):
            value, is_bound = _lambda_pair(by_d[left], by_d[right])
            step[f"{left}->{right}"] = value
            bounded = bounded or is_bound
        values = list(step.values())
        out.append({
            "seed": seed,
            "steps": step,
            "Lambda_T_mean": float(np.mean(values)),
            "lower_bound": bounded,
            "by_distance": {
                d: by_d[d][0] for d in distances
            },
        })
    return out


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def knee_and_gain(raw: dict, cfg: dict) -> dict:
    knee = cfg["knee"]
    early_a, early_b = (int(c) for c in knee["early_cycles"])
    late_a, late_b = (int(c) for c in knee["late_cycles"])
    ref = int(knee["reference_cycle"])
    longest = int(knee["longest_cycle"])

    def pooled(variant: str, cycles: int) -> tuple[float, int, int]:
        rows = _rows(raw, "curve", variant=variant, cycles=cycles)
        fails, shots = _pool(rows)
        return fails / shots, fails, shots

    p2, f2, n2 = pooled("leakage", early_a)
    p4, f4, n4 = pooled("leakage", early_b)
    p_ref, f_ref, n_ref = pooled("leakage", ref)
    p_long, f_long, n_long = pooled("leakage", longest)
    p_late_a, _, _ = pooled("leakage", late_a)
    p_walk, f_walk, n_walk = pooled("walking", longest)
    p_pauli, f_pauli, n_pauli = pooled("pauli", longest)
    pauli_bound = rule_of_three_upper(n_pauli) if f_pauli == 0 else p_pauli
    walk_bound = rule_of_three_upper(n_walk) if f_walk == 0 else p_walk
    return {
        "early_drop": p2 / p4 if p4 else math.inf,
        "late_over_reference": p_long / p_ref if p_ref else math.inf,
        "floor_over_pauli": p_long / pauli_bound if pauli_bound else math.inf,
        "walking_gain": p_long / walk_bound if walk_bound else math.inf,
        "walking_gain_is_lower_bound": f_walk == 0,
        "pauli_floor_is_upper_bound": f_pauli == 0,
        "counts": {
            "leak_early_a": [f2, n2, early_a],
            "leak_early_b": [f4, n4, early_b],
            "leak_ref": [f_ref, n_ref, ref],
            "leak_late_a": pooled("leakage", late_a)[1:],
            "leak_long": [f_long, n_long, longest],
            "walk_long": [f_walk, n_walk, longest],
            "pauli_long": [f_pauli, n_pauli, longest],
        },
        "p": {
            "early_a": p2,
            "early_b": p4,
            "reference": p_ref,
            "late_a": p_late_a,
            "long": p_long,
            "walk_long": p_walk,
            "pauli_long": p_pauli,
        },
    }


def decide(cfg: dict, lam: list[dict], knee: dict) -> dict:
    bars = cfg["thresholds"]
    mean_l, std_l = _mean_std([row["Lambda_T_mean"] for row in lam])
    steps_ok = all(
        all(value > bars["lambda_t_step_min"] for value in row["steps"].values())
        for row in lam
    )
    lambda_ok = mean_l > bars["lambda_t_mean_min"] and steps_ok
    knee_ok = (
        knee["early_drop"] >= bars["early_drop_min"]
        and knee["late_over_reference"] >= bars["late_over_reference_min"]
        and knee["floor_over_pauli"] >= bars["floor_over_pauli_min"]
    )
    walk_ok = knee["walking_gain"] >= bars["walking_gain_min"]
    if lambda_ok and knee_ok and walk_ok:
        status = "PASS"
    elif mean_l > 1.0 and (knee_ok or walk_ok):
        status = "PARTIAL"
    else:
        status = "FAIL"
    return {
        "status": status,
        "Lambda_T_mean": mean_l,
        "Lambda_T_std": std_l,
        "lambda_steps_all_seeds_gt_1": steps_ok,
        "lambda_ok": lambda_ok,
        "knee_ok": knee_ok,
        "walking_ok": walk_ok,
    }


def _fmt(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if value >= 100:
        return f"{value:.1f}"
    if value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.3e}"


def write_tables(cfg: dict, raw: dict, lam: list[dict], knee: dict, verdict: dict, fixed: list[dict]) -> None:
    records = raw["records"]
    with (EXP / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# exp1 results",
        "",
        f"Status: **{verdict['status']}**",
        "",
        "Lambda_T at cycles = patch width, p = 1e-3, no leakage. "
        "eps = 1 - (1 - P_shot)^(1/cycles). A zero-failure denominator uses P < 3/N.",
        "",
        "| seed | d | cycles | fails/shots | P_L | eps |",
        "|---|---|---|---|---|---|",
    ]
    for row in lam:
        for distance, rec in row["by_distance"].items():
            lines.append(
                f"| {row['seed']} | {distance} | {rec['cycles']} | "
                f"{rec['fails']}/{rec['shots']} | {rec['p_L']:.3e} | {rec['eps']:.3e} |"
            )
    lines += ["", "| seed | Lambda_T 3→5 | Lambda_T 5→7 | mean | lower bound? |", "|---|---|---|---|---|"]
    for row in lam:
        s = row["steps"]
        lines.append(
            f"| {row['seed']} | {_fmt(s['3->5'])} | {_fmt(s['5->7'])} | "
            f"{_fmt(row['Lambda_T_mean'])} | {row['lower_bound']} |"
        )
    mean_l, std_l = verdict["Lambda_T_mean"], verdict["Lambda_T_std"]
    lines.append("")
    lines.append(f"Seed mean of Lambda_T_mean: **{mean_l:.3f} ± {std_l:.3f}** (bar > 1.2).")
    lines += ["", "## Same-cycle ratio at c = 4 (reported, not a pass bar)", ""]
    lines.append("| seed | Lambda_fixed 3→5 | Lambda_fixed 5→7 | mean |")
    lines.append("|---|---|---|---|")
    for row in fixed:
        s = row["steps"]
        lines.append(
            f"| {row['seed']} | {_fmt(s.get('3->5', math.nan))} | {_fmt(s.get('5->7', math.nan))} | "
            f"{_fmt(row['Lambda_T_mean'])} |"
        )
    lines += [
        "",
        "## Leakage knee and walking gain (pooled over seeds, d = 3)",
        "",
        f"- early drop P(2)/P(4) = {_fmt(knee['early_drop'])} (bar ≥ 2)",
        f"- late floor P(32)/P(8) = {_fmt(knee['late_over_reference'])} (bar ≥ 0.5)",
        f"- leakage floor / Pauli bound at 32 cycles = {_fmt(knee['floor_over_pauli'])} (bar ≥ 10)",
        f"- walking gain at 32 cycles = {_fmt(knee['walking_gain'])}"
        + (" (lower bound, walking had 0 failures)" if knee["walking_gain_is_lower_bound"] else "")
        + " (bar ≥ 3)",
        "",
        "Per-cycle logical error, mean ± sample std over seeds:",
        "",
        "| variant | cycles | P_L mean | P_L std | eps mean |",
        "|---|---|---|---|---|",
    ]
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rec in _rows(raw, "curve"):
        grouped[(rec["variant"], int(rec["cycles"]))].append(rec)
    for key in sorted(grouped, key=lambda item: (item[0], item[1])):
        variant, cycles = key
        rows = grouped[key]
        mean_p, std_p = _mean_std([r["p_L"] for r in rows])
        mean_e, _ = _mean_std([r["eps"] for r in rows])
        lines.append(f"| {variant} | {cycles} | {mean_p:.4e} | {std_p:.4e} | {mean_e:.4e} |")
    if raw.get("adaptive") and raw["adaptive"].get("headline"):
        head = raw["adaptive"]["headline"]
        lines += [
            "",
            "## Adaptive stopping (report-only, not in the status)",
            "",
            f"Pre-registered gap threshold T = {head['threshold']} nats.",
            f"Mean stopping cycle {head['mean_cycles']:.2f} versus best fixed cycle "
            f"{head['best_fixed_cycles']} (time change {head['time_change']:+.1%}).",
            f"Spacetime error {head['p_st']:.4e} versus best fixed {head['best_fixed_p_st']:.4e} "
            f"(spacetime change {head['spacetime_change']:+.1%}).",
            "",
            "Toy spacetime error is 0.5 * (memory P_L(c) + stability failure). "
            "The hardware 38% / 34% figures are QUOTED and are not this run.",
            "",
            "| T (nats) | mean cycles | P_stab | P_ST | best fixed c | best fixed P_ST | spacetime change | time change |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in raw["adaptive"]["sweep"]:
            lines.append(
                f"| {row['threshold']} | {row['mean_cycles']:.2f} | {row['p_stab']:.3e} | "
                f"{row['p_st']:.3e} | {row['best_fixed_cycles']} | {row['best_fixed_p_st']:.3e} | "
                f"{row['spacetime_change']:+.1%} | {row['time_change']:+.1%} |"
            )
    lines += [
        "",
        "## QUOTED hardware numbers (not measured here)",
        "",
        "- mu_T = 1.47 ± 0.25 (Pauli model 1.97 ± 0.44), block suppression on Willow",
        "- walking stability floor ≤ 3.20e-8 at 60 cycles",
        "- paper: two orders of magnitude below the standard construction at 60 cycles",
        "- bookmark post: four orders of magnitude below any experiment in the literature at 60 cycles",
        "- adaptive stopping: about 38% lower spacetime logical error and 34% lower mean stopping time",
        "",
    ]
    (EXP / "results.md").write_text("\n".join(lines))


def _curve_stats(raw: dict, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _rows(raw, "curve", variant=variant)
    cycles = sorted({int(r["cycles"]) for r in rows})
    means, stds = [], []
    for cycles_i in cycles:
        vals = [r["p_L"] for r in rows if int(r["cycles"]) == cycles_i]
        mean, std = _mean_std(vals)
        means.append(mean)
        stds.append(std)
    return np.asarray(cycles), np.asarray(means), np.asarray(stds)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.framealpha": 0.95,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "figure.dpi": 140,
        "savefig.dpi": 300,
    })


def write_plot(raw: dict, bp: dict | None, verdict: dict) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.6))
    fig.suptitle(
        "Surface-code stability toy  ·  stim + PyMatching  ·  "
        f"status {verdict['status']}",
        fontweight="bold",
    )

    ax = axes[0, 0]
    for variant, color, label in (
        ("pauli", PAULI, "Pauli only"),
        ("leakage", LEAK, r"Leakage bursts ($\tau=8$)"),
        ("walking", WALK, "Walking analogue (clear every 2)"),
    ):
        cycles, mean, std = _curve_stats(raw, variant)
        # Log axis cannot show exact zeros. Plot a 3/N stand-in only when the mean is 0.
        y = mean.copy()
        yerr = std.copy()
        upper = np.zeros(len(mean), dtype=bool)
        for i, (m, cyc) in enumerate(zip(mean, cycles)):
            if m <= 0.0:
                shots = next(
                    r["shots"] for r in _rows(raw, "curve", variant=variant, cycles=int(cyc))
                )
                y[i] = rule_of_three_upper(int(shots))
                yerr[i] = 0.0
                upper[i] = True
        ax.errorbar(
            cycles[~upper], y[~upper], yerr=yerr[~upper], fmt="o-", color=color,
            lw=2.0, ms=5.5, capsize=3, label=label,
        )
        if np.any(upper):
            ax.plot(cycles[upper], y[upper], marker="v", ls="none", color=color, ms=7)
    ax.set_yscale("log")
    ax.set_xlabel("Cycles")
    ax.set_ylabel(r"Logical error per shot $P_L$")
    ax.set_title(r"$d=3$, $p=10^{-3}$, seeds 0/1/2")
    ax.legend(prop={"weight": "bold", "size": 8}, loc="upper right")

    ax = axes[0, 1]
    # Per-seed eps at c = d.
    for distance, color in DIST.items():
        eps, labels = [], []
        for rec in sorted(_rows(raw, "lambda", distance=distance), key=lambda r: r["seed"]):
            eps.append(rec["eps"] if rec["fails"] else rule_of_three_upper(rec["shots"]))
            labels.append(rec["seed"])
        ax.plot([distance] * len(eps), eps, "o", color=color, ms=8, label=f"d={distance}")
    ax.set_yscale("log")
    ax.set_xticks([3, 5, 7])
    ax.set_xlabel("Patch width d  (cycles = d)")
    ax.set_ylabel(r"Per-cycle error $\varepsilon$")
    ax.set_title(rf"$\Lambda_T$ mean {verdict['Lambda_T_mean']:.2f} ± {verdict['Lambda_T_std']:.2f}")
    ax.legend(prop={"weight": "bold"})

    ax = axes[1, 0]
    if bp is not None:
        ps = sorted({float(r["p"]) for r in bp["records"] if r["study"] == "breakpoint_p"})
        means = []
        for p in ps:
            rows = [r for r in bp["records"] if r["study"] == "breakpoint_p" and float(r["p"]) == p]
            # Single seed: build a fake raw and reuse lambda_by_seed.
            fake = {"records": [{**r, "study": "lambda"} for r in rows]}
            lam = lambda_by_seed(fake)
            means.append(lam[0]["Lambda_T_mean"] if lam else math.nan)
        ax.plot(ps, means, "o-", color="#332288", lw=2.0, ms=6)
        ax.axhline(1.0, color="k", lw=1.0)
        ax.axhline(1.2, color="#882255", ls="--", lw=1.2, label=r"pass bar 1.2")
        ax.set_xscale("log")
        ax.set_xlabel(r"Physical error $p$")
        ax.set_ylabel(r"$\Lambda_T$ mean")
        ax.set_title(r"Breakpoint: $\Lambda_T$ versus $p$ (seed 0)")
        ax.legend(prop={"weight": "bold"})
    else:
        ax.set_title("Breakpoint p-sweep not run")

    ax = axes[1, 1]
    if bp is not None:
        starts = sorted({
            float(r["p_start"]) for r in bp["records"] if r["study"] == "breakpoint_leak"
        })
        for variant, color, label in (
            ("leakage", LEAK, "Leakage"),
            ("walking", WALK, "Walking"),
        ):
            ys = []
            for p_start in starts:
                rows = [
                    r for r in bp["records"]
                    if r["study"] == "breakpoint_leak"
                    and r["variant"] == variant
                    and float(r["p_start"]) == p_start
                ]
                ys.append(rows[0]["p_L"] if rows and rows[0]["fails"] else (
                    rule_of_three_upper(rows[0]["shots"]) if rows else math.nan
                ))
            ax.plot(starts, ys, "o-", color=color, lw=2.0, ms=6, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Burst start probability $p_{\mathrm{start}}$")
        ax.set_ylabel(r"$P_L$ at 32 cycles")
        ax.set_title("Breakpoint: walking stops helping")
        ax.legend(prop={"weight": "bold"})
    else:
        ax.set_title("Breakpoint leakage sweep not run")

    fig.tight_layout()
    fig.savefig(EXP / "plot.png")
    fig.savefig(EXP / "plot.pdf")
    plt.close(fig)


def write_breakpoints(bp: dict | None) -> None:
    if bp is None:
        return
    records = bp["records"]
    with (EXP / "breakpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# Breakpoints (seed 0)",
        "",
        "Lambda_T uses cycles = d. A value below 1 means the wider patch has the higher per-cycle error.",
        "",
        "| p | Lambda_T 3→5 | Lambda_T 5→7 | mean |",
        "|---|---|---|---|",
    ]
    ps = sorted({float(r["p"]) for r in records if r["study"] == "breakpoint_p"})
    cross = None
    for p in ps:
        rows = [r for r in records if r["study"] == "breakpoint_p" and float(r["p"]) == p]
        fake = {"records": [{**r, "study": "lambda"} for r in rows]}
        lam = lambda_by_seed(fake)[0]
        lines.append(
            f"| {p:.4g} | {_fmt(lam['steps']['3->5'])} | {_fmt(lam['steps']['5->7'])} | "
            f"{_fmt(lam['Lambda_T_mean'])} |"
        )
        if cross is None and lam["Lambda_T_mean"] < 1.0:
            cross = p
    lines.append("")
    if cross is None:
        lines.append("Lambda_T mean stayed at or above 1 on every grid point.")
    else:
        lines.append(f"Lambda_T mean first drops below 1 at p = {cross:.4g} on this grid.")
        later = [p for p in ps if p > cross]
        # Recomputed below from the table we just wrote: scan records again.
        rebound = []
        for p in later:
            rows = [r for r in records if r["study"] == "breakpoint_p" and float(r["p"]) == p]
            fake = {"records": [{**r, "study": "lambda"} for r in rows]}
            if lambda_by_seed(fake)[0]["Lambda_T_mean"] >= 1.0:
                rebound.append(p)
        if rebound:
            lines.append(
                "At higher p the ratio climbs back through 1 "
                f"(p = {', '.join(f'{p:.4g}' for p in rebound)}). "
                "All three distances are then in the high-error regime, so the ratio "
                "is no longer a clean threshold diagnostic."
            )
    lines += [
        "",
        "## Walking gain versus burst rate (d=3, 32 cycles, tau=8)",
        "",
        "| p_start | P_leak | P_walk | gain |",
        "|---|---|---|---|",
    ]
    gain_cross = None
    starts = sorted({float(r["p_start"]) for r in records if r["study"] == "breakpoint_leak"})
    for p_start in starts:
        leak = next(
            r for r in records
            if r["study"] == "breakpoint_leak" and r["variant"] == "leakage" and float(r["p_start"]) == p_start
        )
        walk = next(
            r for r in records
            if r["study"] == "breakpoint_leak" and r["variant"] == "walking" and float(r["p_start"]) == p_start
        )
        p_l = leak["p_L"]
        p_w = walk["p_L"] if walk["fails"] else rule_of_three_upper(walk["shots"])
        gain = p_l / p_w if p_w else math.inf
        lines.append(f"| {p_start:.4g} | {leak['p_L']:.3e} | {walk['p_L']:.3e} | {_fmt(gain)} |")
        if gain_cross is None and gain < 3.0:
            gain_cross = p_start
    lines.append("")
    if gain_cross is None:
        lines.append("Walking gain stayed at or above 3x on every p_start grid point.")
    else:
        lines.append(f"Walking gain first falls below 3x at p_start = {gain_cross:.4g}.")
    lines += [
        "",
        "## Walking gain versus lifetime (p_start = 3e-4)",
        "",
        "| tau | P_leak | P_walk | gain |",
        "|---|---|---|---|",
    ]
    taus = sorted({float(r["tau"]) for r in records if r["study"] == "breakpoint_tau"})
    below_three = []
    for tau in taus:
        leak = next(
            r for r in records
            if r["study"] == "breakpoint_tau" and r["variant"] == "leakage" and float(r["tau"]) == tau
        )
        walk = next(
            r for r in records
            if r["study"] == "breakpoint_tau" and r["variant"] == "walking" and float(r["tau"]) == tau
        )
        if leak["fails"] == 0 and walk["fails"] == 0:
            gain = 0.0
            below_three.append((tau, "both zero"))
        else:
            p_w = walk["p_L"] if walk["fails"] else rule_of_three_upper(walk["shots"])
            gain = leak["p_L"] / p_w if p_w else math.inf
            if gain < 3.0:
                below_three.append((tau, _fmt(gain)))
        lines.append(f"| {tau:.3g} | {leak['p_L']:.3e} | {walk['p_L']:.3e} | {_fmt(gain)} |")
    lines.append("")
    if not below_three:
        lines.append("Walking gain stayed at or above 3x on every tau grid point.")
    else:
        listed = ", ".join(f"tau={tau:.3g} ({why})" for tau, why in below_three)
        lines.append(
            f"Walking gain is below 3x at {listed}. "
            "Short lifetimes produce too few logical failures for the 2-cycle clear to matter; "
            "from tau = 8 upward the clear removes a visible floor."
        )
    lines.append("")
    (EXP / "breakpoints.md").write_text("\n".join(lines))


def main() -> None:
    cfg, raw, bp = _load()
    saved = raw.get("config_thresholds")
    if saved != cfg["thresholds"]:
        raise SystemExit("config.yaml thresholds do not match the thresholds stored with the run")
    lam = lambda_by_seed(raw, "lambda")
    fixed = lambda_by_seed(raw, "fixed_cycle")
    knee = knee_and_gain(raw, cfg)
    verdict = decide(cfg, lam, knee)
    write_tables(cfg, raw, lam, knee, verdict, fixed)
    write_plot(raw, bp, verdict)
    write_breakpoints(bp)
    status = {
        "status": verdict["status"],
        "verdict": verdict,
        "knee": {k: v for k, v in knee.items() if k != "counts"},
        "lambda_seeds": [
            {"seed": row["seed"], "steps": row["steps"], "mean": row["Lambda_T_mean"]}
            for row in lam
        ],
        "seconds": raw.get("seconds"),
    }
    (EXP / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
