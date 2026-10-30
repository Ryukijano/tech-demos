"""Turn raw.json into results / status / plot / breakpoints tables."""

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
EXP = APP / "experiments" / "exp1_cpu_minatar"

plt.rcParams.update({
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.fontsize": 10,
    "figure.dpi": 140,
})


def _mean_std(xs: list[float]) -> tuple[float, float]:
    a = np.asarray(xs, dtype=np.float64)
    if len(a) == 0:
        return 0.0, 0.0
    if len(a) == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def main() -> None:
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    raw = json.loads((EXP / "raw.json").read_text())
    games = list(cfg["env"]["games"])

    rows = []
    game_summaries = {}
    for game in games:
        seeds_data = raw["games"][game]["seeds"]
        seeds = sorted(seeds_data.keys(), key=int)
        agent_means = []
        random_means = []
        per_seed_wins = {}
        for s in seeds:
            d = seeds_data[s]
            agent_means.append(d["agent_mean_return"])
            random_means.append(d["random_mean_return"])
            per_seed_wins[s] = d["agent_mean_return"] > d["random_mean_return"]
            rows.append({
                "game": game,
                "seed": s,
                "agent_mean_return": round(d["agent_mean_return"], 4),
                "agent_std_return": round(d["agent_std_return"], 4),
                "random_mean_return": round(d["random_mean_return"], 4),
                "random_std_return": round(d["random_std_return"], 4),
                "ratio": round(
                    d["agent_mean_return"] / max(1e-9, d["random_mean_return"]), 4
                ),
                "n_params": d["n_params"],
                "policy_steps": d["policy_steps"],
                "train_wall_s": round(d["train_wall_s"], 1),
                "eval_wall_s": round(d["eval_wall_s"], 2),
            })
        a_mean, a_std = _mean_std(agent_means)
        r_mean, r_std = _mean_std(random_means)
        every = all(per_seed_wins.values())
        two_x = a_mean >= 2.0 * r_mean
        game_summaries[game] = {
            "agent_mean": a_mean,
            "agent_std": a_std,
            "random_mean": r_mean,
            "random_std": r_std,
            "every_seed_beats_random": every,
            "mean_ge_2x_random": two_x,
            "mean_beats_random": a_mean > r_mean,
            "per_seed_wins": per_seed_wins,
            "n_params": seeds_data[seeds[0]]["n_params"],
        }

    with (EXP / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    both_pass = all(
        g["every_seed_beats_random"] and g["mean_ge_2x_random"]
        for g in game_summaries.values()
    )
    both_mean_win = all(g["mean_beats_random"] for g in game_summaries.values())
    if both_pass:
        status = "PASS"
    elif both_mean_win:
        status = "PARTIAL"
    else:
        status = "FAIL"

    status_obj = {
        "status": status,
        "games": game_summaries,
        "pass_condition": cfg["thresholds"]["pass_condition"],
        "status_rule": cfg["thresholds"]["status_rule"],
    }
    (EXP / "status.json").write_text(json.dumps(status_obj, indent=2))

    # results.md
    lines = [
        f"# exp1_cpu_minatar results — **{status}**",
        "",
        "Primary metric: mean episode return over 100 evaluation episodes vs uniform-random.",
        "",
    ]
    for game in games:
        g = game_summaries[game]
        lines += [
            f"## {game}",
            "",
            "| seed | agent mean | random mean | ratio |",
            "|---|---:|---:|---:|",
        ]
        for s in sorted(raw["games"][game]["seeds"].keys(), key=int):
            d = raw["games"][game]["seeds"][s]
            ratio = d["agent_mean_return"] / max(1e-9, d["random_mean_return"])
            lines.append(
                f"| {s} | {d['agent_mean_return']:.3f} | {d['random_mean_return']:.3f} | {ratio:.2f}× |"
            )
        lines += [
            "",
            f"**Mean ± std:** agent **{g['agent_mean']:.3f} ± {g['agent_std']:.3f}**, "
            f"random **{g['random_mean']:.3f} ± {g['random_std']:.3f}** "
            f"(every_seed_win={g['every_seed_beats_random']}, "
            f"mean≥2×random={g['mean_ge_2x_random']}).",
            "",
        ]
    lines += [
        f"Status rule → **{status}**. Params (example): {game_summaries[games[0]]['n_params']}.",
        "",
    ]
    (EXP / "results.md").write_text("\n".join(lines))

    # Learning-curve plot (publication-friendly)
    fig, axes = plt.subplots(1, len(games), figsize=(5.2 * len(games), 4.2), squeeze=False)
    colors = ["#1b4f72", "#117a65", "#b9770e"]
    for ax, game in zip(axes[0], games):
        for i, s in enumerate(sorted(raw["games"][game]["seeds"].keys(), key=int)):
            curve = np.asarray(raw["games"][game]["seeds"][s]["learning_curve"], dtype=float)
            if len(curve) == 0:
                continue
            # Moving average for readability
            win = max(1, min(20, len(curve) // 5))
            kernel = np.ones(win) / win
            smooth = np.convolve(curve, kernel, mode="valid")
            xs = np.arange(len(smooth))
            ax.plot(
                xs,
                smooth,
                color=colors[i % len(colors)],
                lw=2.0,
                label=f"seed {s} (MA-{win})",
            )
            rmean = raw["games"][game]["seeds"][s]["random_mean_return"]
            ax.axhline(
                rmean,
                color=colors[i % len(colors)],
                ls="--",
                lw=1.2,
                alpha=0.7,
                label=f"random seed {s}",
            )
        ax.set_title(f"{game}: training episode returns")
        ax.set_xlabel("Episode index (during training)")
        ax.set_ylabel("Episode return")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", prop={"weight": "bold"})
    fig.tight_layout()
    fig.savefig(EXP / "plot.png")
    plt.close(fig)

    # Breakpoints tables
    if "breakpoints" in raw:
        bp = raw["breakpoints"]
        bp_rows = []
        md = [
            f"# Breakpoints (seed {bp['seed']}, game {bp['game']})",
            "",
            "## Tiny step budget",
            "",
            "| policy_steps | agent mean | random mean | ratio |",
            "|---:|---:|---:|---:|",
        ]
        for r in bp["tiny_step_budget"]:
            ratio = r["agent_mean"] / max(1e-9, r["random_mean"])
            md.append(
                f"| {r['policy_steps']} | {r['agent_mean']:.3f} | {r['random_mean']:.3f} | {ratio:.2f}× |"
            )
            bp_rows.append({
                "kind": "tiny_step_budget",
                "policy_steps": r["policy_steps"],
                "imagine_horizon": "",
                "n_params": r["n_params"],
                "agent_mean": round(r["agent_mean"], 4),
                "random_mean": round(r["random_mean"], 4),
                "train_wall_s": round(r["train_wall_s"], 1),
            })
        md += [
            "",
            "## Imagination horizon",
            "",
            "| horizon | policy_steps | agent mean | random mean | ratio |",
            "|---:|---:|---:|---:|---:|",
        ]
        for r in bp["imagination_horizon"]:
            ratio = r["agent_mean"] / max(1e-9, r["random_mean"])
            md.append(
                f"| {r['imagine_horizon']} | {r['policy_steps']} | {r['agent_mean']:.3f} | {r['random_mean']:.3f} | {ratio:.2f}× |"
            )
            bp_rows.append({
                "kind": "imagination_horizon",
                "policy_steps": r["policy_steps"],
                "imagine_horizon": r["imagine_horizon"],
                "n_params": r["n_params"],
                "agent_mean": round(r["agent_mean"], 4),
                "random_mean": round(r["random_mean"], 4),
                "train_wall_s": round(r["train_wall_s"], 1),
            })
        tm = bp["tiny_model"]
        ratio = tm["agent_mean"] / max(1e-9, tm["random_mean"])
        md += [
            "",
            "## Tiny model",
            "",
            f"Overrides: `{tm['overrides']}`",
            "",
            f"| policy_steps | n_params | agent mean | random mean | ratio |",
            f"|---:|---:|---:|---:|---:|",
            f"| {tm['policy_steps']} | {tm['n_params']} | {tm['agent_mean']:.3f} | {tm['random_mean']:.3f} | {ratio:.2f}× |",
            "",
            "Honest note: breakpoints use 40 eval episodes on Breakout (seed 0) and a "
            "reduced step budget — they show where *this toy* collapses, not where the "
            "upstream 500k-step NanoDreamer fails.",
            "",
        ]
        bp_rows.append({
            "kind": "tiny_model",
            "policy_steps": tm["policy_steps"],
            "imagine_horizon": "",
            "n_params": tm["n_params"],
            "agent_mean": round(tm["agent_mean"], 4),
            "random_mean": round(tm["random_mean"], 4),
            "train_wall_s": round(tm["train_wall_s"], 1),
        })
        (EXP / "breakpoints.md").write_text("\n".join(md))
        with (EXP / "breakpoints.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bp_rows[0].keys()))
            w.writeheader()
            w.writerows(bp_rows)

    print(f"[analyze] status={status}", flush=True)


if __name__ == "__main__":
    main()
