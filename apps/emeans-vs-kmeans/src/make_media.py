"""Still frame and a short animation of the pre-registered comparison."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_emeans"
ART = APP / "artifacts"
COLORS = {"mixture": "#0072B2", "stretched": "#D55E00", "mnist": "#009E73"}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Liberation Serif", "Times"],
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _bold_legend(ax: plt.Axes, **kwargs: Any) -> None:
    legend = ax.legend(**kwargs)
    for text in legend.get_texts():
        text.set_fontweight("bold")


def _media_run(raw: dict[str, Any]) -> dict[str, Any]:
    for run in raw["runs"]:
        if run.get("media"):
            return run
    raise RuntimeError("raw.json has no media subset")


def _draw(status: dict[str, Any], media: dict[str, Any], reveal: int, path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.4))
    ax = axes[0]
    datasets = status["datasets"]
    shown = 0
    for name, block in datasets.items():
        budgets = sorted(block["curves"], key=float)
        if reveal < 1:
            break
        take = min(reveal, len(budgets))
        xs = [float(budget) for budget in budgets[:take]]
        means = [block["curves"][budget]["train_mse_ratio_mean"] for budget in budgets[:take]]
        stds = [block["curves"][budget]["train_mse_ratio_std"] for budget in budgets[:take]]
        ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=1.8, capsize=2, color=COLORS[name], label=name)
        shown += 1
    ax.axhline(1.0, color="#444444", linewidth=0.8, linestyle="--", label="Lloyd @ 25")
    ax.axhline(1.01, color="#888888", linewidth=0.8, linestyle=":", label="1.01 bar")
    ax.set_xlim(0.7, 5.4)
    ratios = [
        block["curves"][budget]["train_mse_ratio_mean"]
        for block in status["datasets"].values()
        for budget in block["curves"]
    ]
    low = min(0.95, min(ratios) - 0.05) if ratios else 0.5
    high = max(1.15, max(ratios) + 0.08) if ratios else 1.3
    ax.set_ylim(low, high)
    ax.set_xlabel("e-means epochs")
    ax.set_ylabel("Train MSE / Lloyd @ 25")
    ax.set_title(f"CPU objective · {status['status']}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if shown:
        _bold_legend(ax, frameon=False, loc="upper right")

    ax = axes[1]
    xy = np.asarray(media["xy"], dtype=np.float64)
    labels = np.asarray(media["emeans"] if reveal >= 4 else media["lloyd"], dtype=np.int64)
    # A handful of the largest clusters keeps the panel readable at k=512.
    values, counts = np.unique(labels, return_counts=True)
    keep = set(values[np.argsort(counts)[-8:]].tolist())
    mask = np.isin(labels, list(keep))
    ax.scatter(xy[~mask, 0], xy[~mask, 1], s=6, c="#d0d0d0", linewidths=0)
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]
    for color, cluster in zip(palette, sorted(keep)):
        chosen = labels == cluster
        ax.scatter(xy[chosen, 0], xy[chosen, 1], s=8, c=color, linewidths=0)
    title = "e-means labels @ 2.5" if reveal >= 4 else "Lloyd labels @ 25"
    ax.set_title(f"Mixture PCA · {title}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _video(frame_dir: Path, dest: Path) -> str:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "2",
        "-i",
        str(frame_dir / "frame_%03d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return proc.stderr[-500:]
    return "ffmpeg libx264"


def main() -> None:
    raw = json.loads((EXP / "raw.json").read_text())
    status = json.loads((EXP / "status.json").read_text())
    media = _media_run(raw)["media"]
    ART.mkdir(parents=True, exist_ok=True)
    frame_dir = ART / "_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    reveals = [1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4]
    for index, reveal in enumerate(reveals):
        _draw(status, media, reveal, frame_dir / f"frame_{index:03d}.png")
    still = ART / "emeans-vs-kmeans.png"
    shutil.copyfile(EXP / "plot.png", still)
    video = ART / "emeans-vs-kmeans.mp4"
    backend = _video(frame_dir, video)
    (ART / "media_backend.txt").write_text(backend + "\n")
    shutil.rmtree(frame_dir)
    docs = APP.parent.parent / "docs" / "assets"
    if docs.is_dir():
        shutil.copyfile(still, docs / "emeans-vs-kmeans.png")
        if video.exists():
            shutil.copyfile(video, docs / "emeans-vs-kmeans.mp4")
    print(f"wrote {still} backend={backend}")


if __name__ == "__main__":
    main()
