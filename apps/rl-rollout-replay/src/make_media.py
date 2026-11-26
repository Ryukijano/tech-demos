"""Short mp4: mean learning curves and six held-out sums on seed 0."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent
APP = SRC.parent
ART = APP / "artifacts"
EXP = APP / "experiments" / "exp1_cpu_addition"

COLORS = {
    "online": "#D55E00",
    "ml": "#0072B2",
    "naive": "#CC79A7",
    "lastk": "#009E73",
}
LABELS = {
    "online": "Online",
    "ml": "ML replay",
    "naive": "Naive RL",
    "lastk": "Last-4 ML",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Liberation Serif"],
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "font.size": 10,
        }
    )


def _bold_legend(ax: plt.Axes) -> None:
    legend = ax.legend(fontsize=8, loc="upper left", frameon=False)
    for text in legend.get_texts():
        text.set_fontweight("bold")


def main() -> None:
    summary = json.loads((EXP / "summary.json").read_text())
    curves = {key: np.asarray(summary["curves"][key], dtype=np.float64) for key in COLORS}
    demo = summary["demo"]
    problems = summary["demo_problems"]
    truth = summary["demo_truth"]
    n_steps = curves["online"].shape[0]
    # Every 4 updates, then hold the final frame.
    frames = list(range(0, n_steps, 4))
    if frames[-1] != n_steps - 1:
        frames.append(n_steps - 1)
    frames.extend([n_steps - 1] * 8)
    _style()
    ART.mkdir(parents=True, exist_ok=True)
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found")
    with tempfile.TemporaryDirectory(prefix="rl-replay-") as tmp_name:
        tmp = Path(tmp_name)
        for frame_id, step in enumerate(frames):
            fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
            ax = axes[0]
            xs = np.arange(step + 1)
            for key, color in COLORS.items():
                ax.plot(xs, curves[key][: step + 1], color=color, lw=2.0, label=LABELS[key])
            ax.set_xlim(0, n_steps - 1)
            ax.set_ylim(0.0, 0.50)
            ax.set_xlabel("Update")
            ax.set_ylabel("Exact-match reward")
            ax.set_title(f"Seed-mean reward, update {step}")
            ax.grid(True, alpha=0.25)
            _bold_legend(ax)
            ax = axes[1]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.set_title("Seed 0 greedy answers")
            header = f"{'sum':<8}{'true':>5}{'on':>6}{'ml':>6}{'naive':>7}{'last4':>7}"
            ax.text(0.0, 0.92, header, family="monospace", fontsize=9, fontweight="bold", transform=ax.transAxes)
            for i, problem in enumerate(problems):
                vals = {
                    "online": int(demo["online"][step][i]),
                    "ml": int(demo["ml"][step][i]),
                    "naive": int(demo["naive"][step][i]),
                    "lastk": int(demo["lastk"][step][i]),
                }
                y = 0.78 - i * 0.12
                ax.text(0.0, y, f"{problem:<8}{truth[i]:>5}", family="monospace", fontsize=9, transform=ax.transAxes)
                x_pos = {"online": 0.46, "ml": 0.60, "naive": 0.74, "lastk": 0.88}
                for key, xpos in x_pos.items():
                    ok = vals[key] == int(truth[i])
                    ax.text(
                        xpos,
                        y,
                        f"{vals[key]:>4}",
                        family="monospace",
                        fontsize=9,
                        color="#1b7f4a" if ok else "#a33b3b",
                        fontweight="bold",
                        transform=ax.transAxes,
                    )
            fig.tight_layout()
            fig.savefig(tmp / f"frame_{frame_id:03d}.png", dpi=120)
            plt.close(fig)
        mp4 = ART / "rl-rollout-replay.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                "8",
                "-i",
                str(tmp / "frame_%03d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                str(mp4),
            ],
            check=True,
        )
    png = ART / "rl-rollout-replay.png"
    if not png.exists():
        raise SystemExit(f"missing {png}; run analyze.py first")
    viewer = APP / "viewer"
    viewer.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(png, viewer / "plot.png")
    print(f"wrote {mp4}", flush=True)


if __name__ == "__main__":
    sys.exit(main())