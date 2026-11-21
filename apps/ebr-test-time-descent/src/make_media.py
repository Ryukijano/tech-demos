"""Still frame and a short mp4: the accuracy curve, then one board filling in."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analyze import BLUE, GREEN, ORANGE, _style  # noqa: E402

EXP = ROOT / "experiments" / "exp1_cpu_sudoku_extreme"
ART = ROOT / "artifacts"
DOCS = ROOT.parent.parent / "docs" / "assets"


def _digits(tokens: list[int], blank: int) -> list[str]:
    out = []
    for token in tokens:
        if int(token) <= blank:
            out.append("")
        else:
            out.append(str(int(token) - 1))
    return out


def write_still() -> None:
    src = EXP / "plot.png"
    if not src.is_file():
        raise SystemExit("missing plot; run analyze.py first")
    ART.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, ART / "ebr-test-time-descent.png")
    if DOCS.is_dir():
        shutil.copyfile(src, DOCS / "ebr-test-time-descent.png")


def write_video() -> None:
    _style()
    summary = json.loads((EXP / "summary.json").read_text())
    raw = json.loads((EXP / "raw.json").read_text())
    example = raw["example"]
    blank = int(summary["blank_token"])
    segments = np.asarray(summary["segments"], dtype=int)
    mean = np.asarray(summary["exact_mean"], dtype=float)
    std = np.asarray(summary["exact_std"], dtype=float)
    clues = _digits(example["clues"], blank)
    labels = _digits(example["labels"], blank)
    preds = [_digits(row, blank) for row in example["preds"]]
    # The example may extend past 16 segments (seed 0). The card uses 16.
    n = len(segments)
    preds = preds[:n]

    fig, (ax_c, ax_b) = plt.subplots(
        1, 2, figsize=(10.2, 5.2), gridspec_kw={"width_ratios": [1.25, 1.0]},
        constrained_layout=True,
    )
    ax_c.set_xlim(0.5, n + 0.5)
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.axvline(summary["train_segments"], color="#222222", lw=1.0, ls=":")
    ax_c.set_xlabel("Descent segments")
    ax_c.set_ylabel("Exact-board accuracy")
    ax_c.set_title("96 hardest Sudoku-Extreme boards")
    line, = ax_c.plot([], [], color=BLUE, lw=2.4, marker="o", ms=5, label="Mean ± sd")
    band = ax_c.fill_between(segments, mean, mean, color=BLUE, alpha=0.18)
    ax_c.legend(prop={"weight": "bold"}, loc="lower right")

    ax_b.set_xlim(-0.5, 8.5)
    ax_b.set_ylim(8.5, -0.5)
    ax_b.set_aspect("equal")
    ax_b.set_xticks(range(9))
    ax_b.set_yticks(range(9))
    ax_b.set_xticklabels([])
    ax_b.set_yticklabels([])
    ax_b.tick_params(length=0)
    for i in range(10):
        lw = 2.0 if i % 3 == 0 else 0.4
        ax_b.plot([-0.5, 8.5], [i - 0.5, i - 0.5], color="#222222", lw=lw)
        ax_b.plot([i - 0.5, i - 0.5], [-0.5, 8.5], color="#222222", lw=lw)
    texts = []
    for r in range(9):
        for c in range(9):
            texts.append(ax_b.text(c, r, "", ha="center", va="center", fontsize=12, fontweight="bold"))
    title = ax_b.set_title("One hard board")

    def draw_board(pred: list[str]) -> None:
        for r in range(9):
            for c in range(9):
                k = r * 9 + c
                given = clues[k] != ""
                text = clues[k] if given else pred[k]
                correct = text == labels[k] and text != ""
                texts[k].set_text(text)
                if given:
                    texts[k].set_color("#222222")
                elif correct:
                    texts[k].set_color(GREEN)
                elif text:
                    texts[k].set_color(ORANGE)
                else:
                    texts[k].set_color("#222222")

    def frame(i: int):
        nonlocal band
        k = min(i, n)
        shown = max(k, 1)
        line.set_data(segments[:shown], mean[:shown])
        band.remove()
        band = ax_c.fill_between(
            segments[:shown],
            np.clip(mean[:shown] - std[:shown], 0, 1),
            np.clip(mean[:shown] + std[:shown], 0, 1),
            color=BLUE,
            alpha=0.18,
        )
        draw_board(preds[shown - 1])
        solved = "solved" if example["exact"][shown - 1] else "unsolved"
        title.set_text(f"Segment {shown}  ·  rating {example['rating']}  ·  {solved}")
        return (line, title, *texts)

    anim = animation.FuncAnimation(fig, frame, frames=n + 8, blit=False)
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / "ebr-test-time-descent.mp4"
    anim.save(path, writer=animation.FFMpegWriter(fps=3, bitrate=1400))
    plt.close(fig)
    if DOCS.is_dir():
        shutil.copyfile(path, DOCS / "ebr-test-time-descent.mp4")
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def main() -> None:
    write_still()
    write_video()


if __name__ == "__main__":
    main()
