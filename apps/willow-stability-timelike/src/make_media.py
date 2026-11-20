"""Card still and a short mp4 of logical error versus cycles."""

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

from src.analyze import EXP, LEAK, PAULI, WALK, _curve_stats, _style  # noqa: E402

ART = ROOT / "artifacts"
DOCS = ROOT.parent.parent / "docs" / "assets"


def _series() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    raw = json.loads((EXP / "raw.json").read_text())
    out = {}
    for variant in ("pauli", "leakage", "walking"):
        cycles, mean, _std = _curve_stats(raw, variant)
        # Keep a floor so the log axis can draw the Pauli tail.
        floor = 0.5 / 40000
        out[variant] = (cycles, np.maximum(mean, floor))
    return out


def write_still() -> None:
    src = EXP / "plot.png"
    if not src.exists():
        raise SystemExit("missing experiment plot; run analyze.py first")
    ART.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, ART / "willow-stability-timelike.png")
    if DOCS.is_dir():
        shutil.copyfile(src, DOCS / "willow-stability-timelike.png")


def write_video() -> None:
    _style()
    series = _series()
    cycles = series["pauli"][0]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    styles = (
        ("pauli", PAULI, "Pauli only"),
        ("leakage", LEAK, r"Toy $|3\rangle$-analogue bursts"),
        ("walking", WALK, "Walking analogue"),
    )
    lines = []
    for _name, color, label in styles:
        line, = ax.plot([], [], "o-", color=color, lw=2.2, ms=6, label=label)
        lines.append(line)
    ax.set_xlim(cycles[0] - 1, cycles[-1] + 1)
    ax.set_ylim(3e-5, 8e-2)
    ax.set_yscale("log")
    ax.set_xlabel("Stability cycles")
    ax.set_ylabel(r"Logical error per shot $P_L$")
    ax.set_title(r"Timelike logical error, $d=3$, $p=10^{-3}$")
    ax.legend(prop={"weight": "bold"}, loc="lower left")
    note = ax.text(0.98, 0.95, "", transform=ax.transAxes, ha="right", va="top", fontsize=10)

    def frame(i: int):
        # Hold the full curve for the last second.
        k = min(i, len(cycles))
        shown = max(k, 1)
        for line, (name, _color, _label) in zip(lines, styles):
            xs, ys = series[name]
            line.set_data(xs[:shown], ys[:shown])
        cyc = int(cycles[min(shown, len(cycles)) - 1])
        note.set_text(f"cycles = {cyc}")
        return (*lines, note)

    n_frames = len(cycles) + 12
    anim = animation.FuncAnimation(fig, frame, frames=n_frames, blit=False)
    path = ART / "willow-stability-timelike.mp4"
    anim.save(
        path,
        writer=animation.FFMpegWriter(fps=4, bitrate=1200),
    )
    plt.close(fig)
    if DOCS.is_dir():
        shutil.copyfile(path, DOCS / "willow-stability-timelike.mp4")
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def main() -> None:
    write_still()
    write_video()


if __name__ == "__main__":
    main()
