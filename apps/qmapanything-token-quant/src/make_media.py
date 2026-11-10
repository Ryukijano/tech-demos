"""Still and a short video of W4A4 metric point maps versus the three quantizers."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_token_quant"
ART = APP / "artifacts"

COLORS = {"fp": "#222222", "naive": "#D55E00", "rotation": "#0072B2", "qmap": "#009E73"}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Liberation Serif", "Times"],
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _bold_legend(ax: plt.Axes, **kwargs) -> None:
    legend = ax.legend(**kwargs)
    for text in legend.get_texts():
        text.set_fontweight("bold")


def _metric_cloud(points: np.ndarray, scale: float) -> np.ndarray:
    return points * scale


def _rel(pred: float, truth: float) -> float:
    return abs(pred - truth) / max(abs(truth), 1e-8)


def _limits(media: dict) -> float:
    clouds = [_metric_cloud(np.asarray(media["points"][0]), float(media["m"][0]))]
    clouds.append(_metric_cloud(np.asarray(media["fp"]["points"][0]), float(media["fp"]["m"][0])))
    return float(np.percentile(np.abs(np.concatenate(clouds, axis=0)), 99) * 1.6 + 0.2)


def _draw_scene(ax: plt.Axes, media: dict, index: int, limit: float) -> None:
    points = np.asarray(media["points"][index], dtype=np.float64)
    truth = float(media["m"][index])
    gt = _metric_cloud(points, truth)
    ax.scatter(gt[:, 0], gt[:, 1], s=18, c="#bbbbbb", label="truth", zorder=1)
    series = [("fp", media["fp"])]
    for method in ("naive", "rotation", "qmap"):
        series.append((method, media["methods"][method]))
    for name, block in series:
        cloud = _metric_cloud(np.asarray(block["points"][index]), float(block["m"][index]))
        ax.scatter(cloud[:, 0], cloud[:, 1], s=14, c=COLORS[name], label=f"{name}  {_rel(float(block['m'][index]), truth):.2f}", zorder=2)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("metric x")
    ax.set_ylabel("metric y")
    ax.set_title(f"Scene {index} · W4A4 · fixed truth limits")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _bold_legend(ax, frameon=False, fontsize=8, loc="upper right")


def _draw_bars(ax: plt.Axes, status: dict) -> None:
    means = status["per_bit"]["W4A4"]
    stds = status["per_bit_std"]["W4A4"]
    keys = [("special_rel", "Special"), ("patch_rel", "Patch"), ("metric_rel", "Metric")]
    methods = ("naive", "rotation", "qmap")
    x = np.arange(len(keys))
    width = 0.24
    for index, method in enumerate(methods):
        vals = [means[method][key] for key, _ in keys]
        err = [stds[method][key] for key, _ in keys]
        ax.bar(
            x + (index - 1) * width,
            vals,
            width,
            yerr=err,
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.4,
            capsize=2,
            label=method,
        )
    ax.set_xticks(x, [label for _, label in keys])
    ax.set_ylabel("Relative error")
    ax.set_title(f"W4A4 means · {status['status']}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _bold_legend(ax, frameon=False, loc="upper right")


def _figure(media: dict, status: dict, scene: int, limit: float, path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 7.5))
    _draw_scene(axes[0], media, scene, limit)
    _draw_bars(axes[1], status)
    ratio = status["means"]["w8_latency_ratio"]
    fig.suptitle(
        f"QMapAnything toy · special-token quant · status {status['status']} · W8A8 qmap/rotation {ratio:.3f}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _video(frame_dir: Path, dest: Path) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return "ffmpeg missing"
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        "1",
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
    media = raw["media"]
    limit = _limits(media)
    ART.mkdir(parents=True, exist_ok=True)
    still = ART / "qmapanything-token-quant.png"
    _figure(media, status, 0, limit, still)
    frame_dir = ART / "_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    n_scenes = len(media["m"])
    index = 0
    for scene in range(n_scenes):
        _figure(media, status, scene, limit, frame_dir / f"frame_{index:03d}.png")
        index += 1
    # Hold the bar-chart reading for a moment by repeating the first scene's board
    # is already in frame 0. Repeat the last scene so the clip does not end mid-cut.
    _figure(media, status, n_scenes - 1, limit, frame_dir / f"frame_{index:03d}.png")
    video = ART / "qmapanything-token-quant.mp4"
    backend = _video(frame_dir, video)
    (ART / "media_backend.txt").write_text(backend + "\n")
    shutil.rmtree(frame_dir)
    docs = APP.parent.parent / "docs" / "assets"
    if docs.is_dir():
        shutil.copyfile(still, docs / "qmapanything-token-quant.png")
        if video.exists():
            shutil.copyfile(video, docs / "qmapanything-token-quant.mp4")
    print(f"wrote {still} backend={backend}")


if __name__ == "__main__":
    main()
