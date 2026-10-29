"""PNG screenshot + short MP4 of auditable feature attributions for one example."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

APP = Path(__file__).resolve().parents[1]
ART = APP / "artifacts"
EXP = APP / "experiments" / "exp1_cpu_auditable"
WORK = APP / "work"
sys.path.insert(0, str(APP / "src"))

from detector_util import explain_one, load_detector  # noqa: E402

plt.rcParams.update({
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "figure.dpi": 140,
})

DEMO_TEXT = (
    "In this paper, we propose a novel framework that leverages synergistic "
    "paradigms to optimize holistic outcomes across diverse domains. "
    "Furthermore, our comprehensive evaluation demonstrates robust performance."
)


def _wrap(text: str, width: int = 88) -> str:
    words = text.split()
    lines, cur = [], []
    for w in words:
        trial = (" ".join(cur + [w])).strip()
        if len(trial) > width and cur:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def _draw_panel(path: Path, detection, title: str, reveal: int | None = None) -> None:
    feats = detection.features
    if reveal is not None:
        feats = feats[:reveal]
    fig = plt.figure(figsize=(11.2, 6.4))
    fig.patch.set_facecolor("#f4f7fb")
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.82])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    label = detection.label.upper()
    color = "#922b21" if detection.label == "ai" else "#1e8449"
    ax.text(
        0.0,
        0.96,
        "DetectorScope · auditable SAE detector (CPU)",
        fontsize=13,
        fontweight="bold",
        color="#1b2631",
        transform=ax.transAxes,
        va="top",
    )
    ax.text(
        0.0,
        0.90,
        f"{title}   ·   label={label}   p(AI)={detection.p_ai:.3f}   "
        f"logit={detection.logit:.3f}   bias={detection.bias:.3f}",
        fontsize=10,
        color=color,
        fontweight="bold",
        transform=ax.transAxes,
        va="top",
    )
    ax.text(
        0.0,
        0.82,
        _wrap(DEMO_TEXT, 100),
        fontsize=9,
        color="#2c3e50",
        family="DejaVu Sans Mono",
        transform=ax.transAxes,
        va="top",
    )

    ax.text(
        0.0,
        0.58,
        "Top feature attributions (zero-ablation importance; + → AI)",
        fontsize=11,
        fontweight="bold",
        color="#1b2631",
        transform=ax.transAxes,
        va="top",
    )

    y = 0.52
    show = feats[:8]
    if not show:
        ax.text(0.0, y, "(no active features)", transform=ax.transAxes, color="#7f8c8d")
    for f in show:
        desc = (f.description or f"feature#{f.id}")[:90]
        bar = min(abs(f.importance) / 0.5, 1.0)
        c = "#c0392b" if f.importance >= 0 else "#1e8449"
        ax.plot([0.0, 0.22 * bar], [y - 0.01, y - 0.01], color=c, lw=8, solid_capstyle="butt")
        ax.text(
            0.24,
            y,
            f"{f.importance:+.3f}  {desc}",
            fontsize=9,
            color="#1b2631",
            transform=ax.transAxes,
            va="center",
            fontweight="bold" if abs(f.importance) >= 0.1 else "normal",
        )
        y -= 0.055

    ax.text(
        0.0,
        0.04,
        "Source: official detectorscope + liamdugan/detector-scope · arXiv:2610.30006",
        fontsize=8,
        color="#5d6d7e",
        transform=ax.transAxes,
    )
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    detector = load_detector(
        cfg["model"]["hub_id"], torch_threads=int(cfg["runtime"]["torch_threads"])
    )
    det = explain_one(
        detector,
        DEMO_TEXT,
        head=cfg["model"]["head"],
        tau=float(cfg["model"]["tau_sparse"]),
        highlights=True,
    )
    png = ART / "sae-text-detector.png"
    _draw_panel(png, det, title=f"τ={cfg['model']['tau_sparse']:.2f}", reveal=None)
    print(f"[media] wrote {png}", flush=True)

    # Animate progressive reveal of top features for a short MP4.
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        n = min(8, max(1, len(det.features)))
        frames = []
        for i in range(1, n + 1):
            fp = td_path / f"frame_{i:02d}.png"
            _draw_panel(fp, det, title=f"τ={cfg['model']['tau_sparse']:.2f} · reveal {i}/{n}", reveal=i)
            frames.append(fp)
        # Hold last frame
        for j in range(4):
            shutil.copy(frames[-1], td_path / f"frame_{n + j + 1:02d}.png")
        mp4 = ART / "sae-text-detector.mp4"
        cmd = [
            "ffmpeg", "-y", "-framerate", "2",
            "-i", str(td_path / "frame_%02d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            str(mp4),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[media] wrote {mp4}", flush=True)

    docs = APP.parents[1] / "docs" / "assets"
    if docs.is_dir():
        shutil.copy2(png, docs / "sae-text-detector.png")
        shutil.copy2(ART / "sae-text-detector.mp4", docs / "sae-text-detector.mp4")
        print("[media] copied into docs/assets", flush=True)

    # Persist one explained example for the Gradio defaults / README.
    dump = {
        "text": DEMO_TEXT,
        "label": det.label,
        "p_ai": det.p_ai,
        "logit": det.logit,
        "bias": det.bias,
        "features": [
            {
                "id": f.id,
                "importance": f.importance,
                "description": f.description,
            }
            for f in det.features[:12]
        ],
    }
    import json

    (WORK / "demo_detection.json").write_text(json.dumps(dump, indent=2))
    _ = torch  # silence lint if unused in some paths
    _ = np


if __name__ == "__main__":
    main()
