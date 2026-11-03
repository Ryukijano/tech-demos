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
EXP = APP / "experiments" / "exp1_cpu_pendulum"

plt.rcParams.update({
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "figure.dpi": 140,
})


def main() -> None:
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    raw = json.loads((EXP / "raw.json").read_text())
    seeds = sorted(raw["seeds"].keys(), key=int)
    thr = cfg["thresholds"]

    rows = []
    w_lrc: list[float] = []
    d_lrc: list[float] = []
    r_lrc: list[float] = []
    w_probe: list[float] = []
    d_probe: list[float] = []
    per_seed_win: dict[str, bool] = {}

    for s in seeds:
        kw = raw["seeds"][s]["kernels"]["world2vec"]
        kd = raw["seeds"][s]["kernels"]["discrete"]
        kr = raw["seeds"][s]["kernels"]["recon"]
        w_lrc.append(kw["lrc_h"])
        d_lrc.append(kd["lrc_h"])
        r_lrc.append(kr["lrc_h"])
        w_probe.append(kw["probe_r2"])
        d_probe.append(kd["probe_r2"])
        per_seed_win[s] = kw["lrc_h"] > kd["lrc_h"]
        for name, k in (("world2vec", kw), ("discrete", kd), ("recon", kr)):
            rows.append({
                "seed": s,
                "kind": name,
                "lrc_h": round(k["lrc_h"], 6),
                "lrc_h1": round(k["lrc_h1"], 6),
                "lrc_h4": round(k["lrc_h4"], 6),
                "probe_r2": round(k["probe_r2"], 6),
                "horizon_state_r2": round(k["horizon_state_r2"], 6),
                "n_eval": k["n_eval"],
                "train_wall_s": round(k["train_wall_s"], 2),
                "n_params": k["n_params"],
            })

    with (EXP / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    w_mean = float(np.mean(w_lrc))
    w_std = float(np.std(w_lrc, ddof=1)) if len(w_lrc) > 1 else 0.0
    d_mean = float(np.mean(d_lrc))
    d_std = float(np.std(d_lrc, ddof=1)) if len(d_lrc) > 1 else 0.0
    r_mean = float(np.mean(r_lrc))
    r_std = float(np.std(r_lrc, ddof=1)) if len(r_lrc) > 1 else 0.0
    gap = w_mean - d_mean
    probe_mean = float(np.mean(w_probe))
    probe_std = float(np.std(w_probe, ddof=1)) if len(w_probe) > 1 else 0.0

    every_win = all(per_seed_win.values())
    gap_ok = gap >= float(thr["lrc_gap_min"])
    probe_ok = probe_mean >= float(thr["probe_r2_mean_min"])

    if every_win and gap_ok and probe_ok:
        status = "PASS"
    elif w_mean > d_mean and probe_mean >= 0.40:
        status = "PARTIAL"
    else:
        status = "FAIL"

    status_obj = {
        "status": status,
        "mean_lrc_world2vec": w_mean,
        "std_lrc_world2vec": w_std,
        "mean_lrc_discrete": d_mean,
        "std_lrc_discrete": d_std,
        "mean_lrc_recon": r_mean,
        "std_lrc_recon": r_std,
        "gap_lrc_vs_discrete": gap,
        "mean_probe_r2_world2vec": probe_mean,
        "std_probe_r2_world2vec": probe_std,
        "mean_probe_r2_discrete": float(np.mean(d_probe)),
        "every_seed_lrc_win": every_win,
        "per_seed_lrc_win": per_seed_win,
        "gap_ok": gap_ok,
        "probe_ok": probe_ok,
        "thresholds": {
            "lrc_gap_min": thr["lrc_gap_min"],
            "probe_r2_mean_min": thr["probe_r2_mean_min"],
            "eval_horizon": thr["eval_horizon"],
        },
    }
    (EXP / "status.json").write_text(json.dumps(status_obj, indent=2))

    # results.md
    lines = [
        f"# Results — `{cfg['exp_id']}`",
        "",
        f"**Status: {status}**",
        "",
        f"Primary metric: LRC at horizon H={thr['eval_horizon']} "
        "(higher is better). Anti-collapse gate: linear-probe R² of "
        "`[sin θ, cos θ, ω]` from latents.",
        "",
        "| seed | world2vec LRC | discrete LRC | recon LRC | w2v probe R² | discrete probe R² | Δ LRC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in seeds:
        kw = raw["seeds"][s]["kernels"]["world2vec"]
        kd = raw["seeds"][s]["kernels"]["discrete"]
        kr = raw["seeds"][s]["kernels"]["recon"]
        lines.append(
            f"| {s} | **{kw['lrc_h']:.4f}** | {kd['lrc_h']:.4f} | {kr['lrc_h']:.4f} | "
            f"{kw['probe_r2']:.4f} | {kd['probe_r2']:.4f} | {kw['lrc_h'] - kd['lrc_h']:+.4f} |"
        )
    lines += [
        "",
        f"**Mean ± std LRC:** world2vec **{w_mean:.4f} ± {w_std:.4f}**, "
        f"discrete **{d_mean:.4f} ± {d_std:.4f}**, recon **{r_mean:.4f} ± {r_std:.4f}** "
        f"(gap vs discrete **{gap:+.4f}**; bar ≥ {thr['lrc_gap_min']}).",
        "",
        f"**Mean probe R² (world2vec):** **{probe_mean:.4f} ± {probe_std:.4f}** "
        f"(bar ≥ {thr['probe_r2_mean_min']}).",
        "",
        f"Every-seed LRC win vs discrete: **{every_win}**.",
        "",
    ]
    (EXP / "results.md").write_text("\n".join(lines) + "\n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    x = np.arange(len(seeds))
    width = 0.25
    ax = axes[0]
    ax.bar(x - width, w_lrc, width, label="world2vec", color="#1b6ca8")
    ax.bar(x, d_lrc, width, label="discrete", color="#c47a2c")
    ax.bar(x + width, r_lrc, width, label="recon", color="#6b6b6b")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("LRC @ H")
    ax.set_title("Latent Rollout Consistency")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", prop={"weight": "bold"})
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - width / 2, w_probe, width, label="world2vec", color="#1b6ca8")
    ax.bar(x + width / 2, d_probe, width, label="discrete", color="#c47a2c")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel(r"Linear-probe $R^2$")
    ax.set_title("True-state probe from latents")
    ax.set_ylim(0, 1.05)
    ax.axhline(float(thr["probe_r2_mean_min"]), color="#1b6ca8", ls="--", lw=1.2, label="probe bar")
    ax.legend(loc="lower right", prop={"weight": "bold"})
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"world2vec-toy · status {status} · gap LRC {gap:+.3f}",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(EXP / "plot.png")
    plt.close(fig)

    # Breakpoints tables
    bp = raw.get("breakpoints") or {}
    bp_rows = []
    if "long_horizon" in bp:
        for kind, rows_k in bp["long_horizon"].items():
            for row in rows_k:
                bp_rows.append({"breakpoint": "long_horizon", "kind": kind, **row})
    if "observation_noise" in bp:
        for row in bp["observation_noise"]:
            bp_rows.append({"breakpoint": "observation_noise", **row})
    if "irregular_dt" in bp:
        for row in bp["irregular_dt"]:
            bp_rows.append({"breakpoint": "irregular_dt", **row})

    if bp_rows:
        # flatten keys
        keys = sorted({k for r in bp_rows for k in r.keys()})
        with (EXP / "breakpoints.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in bp_rows:
                w.writerow(r)

        md = ["# Breakpoints", ""]
        if "long_horizon" in bp:
            md += [
                "## Long horizon LRC",
                "",
                "| kind | H | LRC | probe R² | horizon state R² |",
                "|---|---:|---:|---:|---:|",
            ]
            for kind, rows_k in bp["long_horizon"].items():
                for row in rows_k:
                    md.append(
                        f"| {kind} | {row['horizon']} | {row['lrc_h']:.4f} | "
                        f"{row['probe_r2']:.4f} | {row['horizon_state_r2']:.4f} |"
                    )
            md.append("")
        if "observation_noise" in bp:
            md += [
                "## Observation noise",
                "",
                "| kind | noise_std | LRC | probe R² |",
                "|---|---:|---:|---:|",
            ]
            for row in bp["observation_noise"]:
                md.append(
                    f"| {row['kind']} | {row['noise_std']} | {row['lrc_h']:.4f} | {row['probe_r2']:.4f} |"
                )
            md.append("")
        if "irregular_dt" in bp:
            md += [
                "## Irregular Δt (Euler step scale)",
                "",
                "| kind | dt_scale | LRC | horizon state R² |",
                "|---|---:|---:|---:|",
            ]
            for row in bp["irregular_dt"]:
                md.append(
                    f"| {row['kind']} | {row['dt_scale']} | {row['lrc_h']:.4f} | "
                    f"{row['horizon_state_r2']:.4f} |"
                )
            md.append("")
        (EXP / "breakpoints.md").write_text("\n".join(md) + "\n")

    print(json.dumps(status_obj, indent=2))


if __name__ == "__main__":
    main()
