"""Screenshot (PNG) + short video (MP4) of MoRE vs MoE loss / routing."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

APP = Path(__file__).resolve().parents[1]
ART = APP / "artifacts"
EXP = APP / "experiments" / "exp1_cpu_charlm"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "figure.dpi": 140,
    }
)

COLORS = {
    "dense": "#7570b3",
    "moe": "#d95f02",
    "more": "#1b9e77",
    "more_nodepth": "#66a61e",
}
LABELS = {
    "dense": "dense",
    "moe": "MoE",
    "more": "MoRE (R=2)",
    "more_nodepth": "MoRE no-depth",
}


def main() -> None:
    ART.mkdir(exist_ok=True)
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    raw = json.loads((EXP / "raw.json").read_text())
    status = json.loads((EXP / "status.json").read_text())
    seeds = sorted(raw["seeds"].keys(), key=int)
    kernels = list(cfg["kernels"])

    # Panel PNG: mean±std bars + seed-0 curves (mirrors analyze plot, branded)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax = axes[0]
    means = [status["means_val_ce"][k] for k in kernels]
    stds = [status["stds_val_ce"][k] for k in kernels]
    ax.bar(
        [LABELS[k] for k in kernels],
        means,
        yerr=stds,
        color=[COLORS[k] for k in kernels],
        capsize=5,
        width=0.65,
    )
    ax.set_ylabel("Validation CE (nats, mean ± std)")
    ax.set_title(f"Aggregate — status {status['status']}")
    ax.grid(axis="y", alpha=0.3)
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)

    ax = axes[1]
    s0 = seeds[0]
    for kind in kernels:
        hist = raw["seeds"][s0]["kernels"][kind].get("history") or []
        if not hist:
            continue
        ax.plot(
            [h["step"] for h in hist],
            [h["val_ce"] for h in hist],
            label=LABELS[kind],
            color=COLORS[kind],
            linewidth=2.2,
            marker="o",
            markersize=3.5,
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation CE")
    ax.set_title(f"Learning curves (seed {s0})")
    ax.legend(loc="best", prop={"weight": "bold"}, fontsize=8)
    ax.grid(alpha=0.3)

    gap = status["moe_minus_more_gap"]
    fig.suptitle(
        f"MoRE toy (arXiv:2609.18176) — MoE−MoRE gap {gap:+.4f}  [{status['status']}]",
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout()
    out_png = ART / "more-experts-toy.png"
    fig.savefig(out_png, bbox_inches="tight")
    # Also refresh experiment plot alias for Pages copy convenience
    fig.savefig(EXP / "plot.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")

    # Video: animate learning curves growing step-by-step + final reuse bar
    tmp = Path(tempfile.mkdtemp(prefix="more_frames_"))
    try:
        # Build aligned step axis from MoRE history
        hist_more = raw["seeds"][s0]["kernels"]["more"].get("history") or []
        n_frames = max(1, len(hist_more))
        frames = []
        for fi in range(n_frames):
            fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
            ax = axes[0]
            for kind in kernels:
                hist = raw["seeds"][s0]["kernels"][kind].get("history") or []
                if not hist:
                    continue
                cut = hist[: min(fi + 1, len(hist))]
                ax.plot(
                    [h["step"] for h in cut],
                    [h["val_ce"] for h in cut],
                    label=LABELS[kind],
                    color=COLORS[kind],
                    linewidth=2.0,
                    marker="o",
                    markersize=3,
                )
            ax.set_xlabel("Step")
            ax.set_ylabel("Val CE")
            ax.set_title("Loss curves (growing)", fontweight="bold")
            ax.legend(loc="best", fontsize=8, prop={"weight": "bold"})
            ax.grid(alpha=0.3)

            ax = axes[1]
            # Reuse rate vs random for MoRE across seeds (static bars that
            # "reveal" after halfway)
            rates = []
            rands = []
            for s in seeds:
                r = raw["seeds"][s]["kernels"]["more"].get("routing") or {}
                rates.append(r.get("adjacent_expert_reuse_rate") or 0.0)
                rands.append(r.get("random_reuse_baseline") or 0.0)
            reveal = 1.0 if fi >= n_frames // 2 else (fi + 1) / max(1, n_frames // 2)
            x = np.arange(len(seeds))
            ax.bar(
                x - 0.2,
                [v * reveal for v in rates],
                0.4,
                label="MoRE reuse",
                color=COLORS["more"],
            )
            ax.bar(
                x + 0.2,
                [v * reveal for v in rands],
                0.4,
                label="random baseline",
                color="#999999",
            )
            ax.set_xticks(x)
            ax.set_xticklabels([f"seed {s}" for s in seeds])
            ax.set_ylabel("Adjacent expert overlap / K")
            ax.set_title("Routing reuse vs random", fontweight="bold")
            ax.set_ylim(0, max(0.05, max(rates + rands) * 1.25))
            ax.legend(loc="best", fontsize=8, prop={"weight": "bold"})
            ax.grid(axis="y", alpha=0.3)

            fig.suptitle(
                f"MoRE CPU toy — frame {fi+1}/{n_frames}  status {status['status']}",
                fontweight="bold",
            )
            fig.tight_layout()
            fp = tmp / f"frame_{fi:03d}.png"
            fig.savefig(fp)
            plt.close(fig)
            frames.append(fp)

        # Hold last frame
        for j in range(6):
            shutil.copy(frames[-1], tmp / f"frame_{len(frames)+j:03d}.png")

        out = ART / "more-experts-toy.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "2",
            "-i",
            str(tmp / "frame_%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"wrote {out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
