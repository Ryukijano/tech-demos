"""Still of the results figure, and a short mp4 of the merge plus the sweep."""

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
from matplotlib.patches import Circle, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analyze import DIST_COLOR, EXP, _log_axis, _style  # noqa: E402
from src.circuit import patch_layout  # noqa: E402

ART = ROOT / "artifacts"
DOCS = ROOT.parent.parent / "docs" / "assets"


def write_still() -> None:
    src = EXP / "plot.png"
    if not src.exists():
        raise SystemExit("missing plot.png; run analyze.py first")
    ART.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, ART / "lattice-surgery-teleport.png")
    if DOCS.is_dir():
        shutil.copyfile(src, DOCS / "lattice-surgery-teleport.png")
    print(f"wrote {ART / 'lattice-surgery-teleport.png'}")


def _sweep_series() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    raw = json.loads((EXP / "breakpoints_raw.json").read_text())
    series = {}
    for distance in (3, 5, 7):
        rows = [
            r
            for r in raw["records"]
            if r["study"] == "p_sweep" and int(r["distance"]) == distance and r["basis"] == "Z"
        ]
        # Mean of Z and X at each p.
        ps = sorted({float(r["p"]) for r in rows})
        ys = []
        for p in ps:
            rates = []
            for basis in ("Z", "X"):
                row = next(
                    r
                    for r in raw["records"]
                    if r["study"] == "p_sweep"
                    and int(r["distance"]) == distance
                    and r["basis"] == basis
                    and float(r["p"]) == p
                )
                fails = int(row["fails"])
                shots = int(row["shots"])
                rates.append(3.0 / shots if fails == 0 else fails / shots)
            ys.append(0.5 * (rates[0] + rates[1]))
        series[distance] = (np.asarray(ps), np.asarray(ys))
    return series


def _draw_lattice(ax, phase: str) -> None:
    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")
    layout = patch_layout(3)
    merged = layout["merged"]
    l1 = {(s[0], s[1]) for s in layout["l1"]}
    l2 = {(s[0], s[1]) for s in layout["l2"]}
    seam = set(layout["seam"])
    show_iface = phase in ("merge", "merged", "split")
    show_merged_checks = phase in ("merge", "merged")
    highlight_l2 = phase == "split"

    def keep(stab) -> bool:
        coord = (stab[0], stab[1])
        if coord in seam:
            return show_merged_checks
        if show_merged_checks:
            return True
        return coord in l1 or coord in l2

    for x, y, basis, neigh in merged:
        if not keep((x, y, basis, neigh)):
            continue
        color = "#56B4E9" if basis == "X" else "#009E73"
        alpha = 0.95 if (x, y) in seam and phase == "merge" else 0.55
        xs = [q[0] for q in neigh]
        ys = [q[1] for q in neigh]
        pad = 0.85
        rect = FancyBboxPatch(
            (min(xs) - pad, min(ys) - pad),
            max(xs) - min(xs) + 2 * pad,
            max(ys) - min(ys) + 2 * pad,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor=color,
            edgecolor="none",
            alpha=alpha * 0.45,
            lw=0,
        )
        ax.add_patch(rect)
    data = list(layout["l1_data"]) + list(layout["l2_data"])
    if show_iface:
        data = data + list(layout["iface"])
    for x, y in data:
        if highlight_l2 and (x, y) in set(layout["l2_data"]):
            face = "#D55E00"
        elif (x, y) in set(layout["iface"]):
            face = "#E69F00"
        else:
            face = "#222222"
        ax.add_patch(Circle((x, y), 0.28, facecolor=face, edgecolor="none", zorder=3))
    titles = {
        "separate": "Two d = 3 patches",
        "merge": "XX merge: seam checks on",
        "merged": "One 3 × 7 patch",
        "split": "Split: state checked on L2",
    }
    ax.set_xlim(-1.2, 15.2)
    ax.set_ylim(-1.6, 7.4)
    ax.set_title(titles[phase], fontweight="bold", fontsize=11)
    ax.text(
        0.0,
        -1.35,
        "Data  ·  X checks  ·  Z checks  ·  interface",
        fontsize=8,
        color="#333333",
    )


def write_video() -> None:
    _style()
    series = _sweep_series()
    ps = series[3][0]
    fig, (ax_lat, ax) = plt.subplots(
        1, 2, figsize=(9.2, 4.4), dpi=140, gridspec_kw={"width_ratios": [1.05, 1.2]}
    )
    phases = ("separate", "merge", "merged", "split")
    for distance in (3, 5, 7):
        ax.plot([], [], "o-", color=DIST_COLOR[distance], lw=2.0, ms=4.5, label=f"d = {distance}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _log_axis(ax)
    ax.set_xlim(ps[0] * 0.8, ps[-1] * 1.2)
    ymax = max(float(series[d][1].max()) for d in (3, 5, 7))
    ymin = min(float(series[d][1].min()) for d in (3, 5, 7))
    ax.set_ylim(ymin * 0.5, min(0.6, ymax * 2))
    ax.set_xlabel("Physical error rate p")
    ax.set_ylabel("Mean logical failure")
    ax.set_title("Seed 0 sweep", fontweight="bold")
    ax.legend(prop={"weight": "bold", "size": 8}, loc="lower right")
    lines = ax.get_lines()
    note = ax.text(0.03, 0.95, "", transform=ax.transAxes, va="top", fontsize=8)

    def frame(i: int):
        phase = phases[(i // 3) % len(phases)]
        _draw_lattice(ax_lat, phase)
        shown = max(1, min(len(ps), i // 2 + 1))
        for line, distance in zip(lines, (3, 5, 7)):
            xs, ys = series[distance]
            line.set_data(xs[:shown], ys[:shown])
        note.set_text(f"p <= {ps[shown - 1]:g}")
        return lines

    anim = animation.FuncAnimation(fig, frame, frames=len(ps) * 3 + 12, blit=False)
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / "lattice-surgery-teleport.mp4"
    anim.save(path, writer=animation.FFMpegWriter(fps=5, bitrate=1400))
    plt.close(fig)
    if DOCS.is_dir():
        shutil.copyfile(path, DOCS / "lattice-surgery-teleport.mp4")
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def main() -> None:
    write_still()
    write_video()


if __name__ == "__main__":
    main()
