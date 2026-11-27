"""Short matplotlib animation of one held-out register trace."""

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

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_register"
ART = APP / "artifacts"
MODES = ("vanilla", "fbt", "alf")
COLORS = {"vanilla": "#0072B2", "fbt": "#E69F00", "alf": "#009E73"}
LABELS = {"vanilla": "Vanilla", "fbt": "FBT", "alf": "ALF"}


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


def _frame(path: Path, rnd: dict, index: int, n_rounds: int) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=120)
    digits = np.arange(5)
    width = 0.24
    truth = int(rnd["truth"])
    for offset, mode in enumerate(MODES):
        probs = np.asarray(rnd["probs"][mode], dtype=np.float64)
        ax.bar(
            digits + (offset - 1) * width,
            probs,
            width=width,
            color=COLORS[mode],
            label=LABELS[mode],
        )
    ax.axvline(truth, color="#222222", ls="--", lw=1.0, label=f"Truth {truth}")
    marks = []
    for mode in MODES:
        pred = int(rnd["pred"][mode])
        mark = "ok" if pred == truth else f"{pred}"
        marks.append(f"{LABELS[mode]} {mark}")
    ax.set_xticks(digits, [str(d) for d in digits])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Checksum digit")
    ax.set_ylabel("Predictive probability")
    ax.set_title(f"Round {index + 1}/{n_rounds}: {rnd['op']}  →  {truth}\n" + " · ".join(marks))
    legend = ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper right")
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    _style()
    raw = json.loads((EXP / "raw.json").read_text())
    rounds = raw["demo_trace"]["rounds"]
    ART.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="alf_frames_"))
    try:
        frame_id = 0
        for index, rnd in enumerate(rounds):
            for _hold in range(2):
                _frame(tmp / f"frame_{frame_id:03d}.png", rnd, index, len(rounds))
                frame_id += 1
        for _hold in range(8):
            shutil.copy(tmp / f"frame_{frame_id - 1:03d}.png", tmp / f"frame_{frame_id:03d}.png")
            frame_id += 1
        out = ART / "alf-feedback-transformer.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "4",
                "-i",
                str(tmp / "frame_%03d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        print(f"wrote {out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
