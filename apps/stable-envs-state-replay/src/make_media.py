"""Rollout re-rendered from stored states, plus a still and a small viewer."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import run_experiment as rx

APP = rx.APP
EXP = rx.EXP
ART = APP / "artifacts"
VIEW = APP / "viewer"


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def rollout(n_steps: int = 48):
    import stable_envs as snv

    cfg = rx.load_config()
    spec = cfg["envs"]["TwoRoom"]
    states, reference, _info = rx.collect(snv, "TwoRoom", spec, 1, n_steps, 0)
    replay = rx.replay_frames(snv, "TwoRoom", states)
    stats = rx.lossless_stats(reference, replay)
    return replay.numpy(), stats


def board(frame: np.ndarray, status: str, step: int, n_steps: int, stats: dict) -> Image.Image:
    """960×600 (16:10) so the pages card shows the whole frame."""
    canvas = Image.new("RGB", (960, 600), (11, 15, 20))
    big = np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)
    image = Image.fromarray(big, mode="RGB")
    canvas.paste(image, (36, 76))
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 24), "stable-envs · state replay", fill=(236, 232, 220), font=font(28))
    draw.text((520, 90), status, fill=(242, 196, 109), font=font(36))
    lines = [
        "TwoRoom, re-rendered from",
        "a stored simulator state.",
        f"step {step + 1} / {n_steps}   seed 0",
        f"max abs diff {stats['max_abs']}",
        "lossless on this rollout",
        "",
        "CPU only. Native 224².",
        "Non-native sizes are refused.",
    ]
    y = 160
    for line in lines:
        draw.text((520, y), line, fill=(220, 214, 200), font=font(22))
        y += 36
    return canvas


def write_video(frames: np.ndarray, stats: dict, status: str, dest: Path) -> None:
    tmp = dest.parent / "_frames"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    n_steps = int(frames.shape[0])
    for index, frame in enumerate(frames):
        board(frame, status, index, n_steps, stats).save(tmp / f"{index:04d}.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "8",
            "-i",
            str(tmp / "%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )
    shutil.rmtree(tmp)


def write_still(frames: np.ndarray, stats: dict, status: str, dest: Path) -> None:
    mid = int(frames.shape[0] // 2)
    board(frames[mid], status, mid, int(frames.shape[0]), stats).save(dest)


def write_viewer(status: str) -> None:
    summary = json.loads((EXP / "status.json").read_text())
    rows = []
    for row in summary["envs"]:
        rows.append(
            "<tr>"
            f"<td>{row['env']}</td>"
            f"<td>{row['saving_png_mean']:.2f}%</td>"
            f"<td>{row['saving_npz_mean']:.2f}%</td>"
            f"<td>{row['lossless_min']:.3f}</td>"
            f"<td>{row['replay_fps_mean']:.1f}</td>"
            f"<td>{row['load_fps_mean']:.1f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>stable-envs state replay</title>
  <style>
    body {{ margin: 0; background: #0b0f14; color: #ece8dc; font: 16px/1.45 Georgia, "Liberation Serif", serif; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 28px 20px 48px; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; }}
    .status {{ display: inline-block; padding: 2px 8px; border: 1px solid #f2c46d; color: #f2c46d; }}
    video, img {{ width: 100%; background: #000; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border-bottom: 1px solid #243040; text-align: right; padding: 6px 8px; }}
    th:first-child, td:first-child {{ text-align: left; }}
    a {{ color: #8ab4ff; }}
  </style>
</head>
<body>
<main>
  <h1>stable-envs state replay</h1>
  <p><span class="status">{status}</span> CPU experiment. Store simulator states, draw the frames again.</p>
  <video src="../artifacts/stable-envs-state-replay.mp4" poster="../artifacts/stable-envs-state-replay.png" controls muted playsinline></video>
  <table>
    <thead><tr><th>Env</th><th>vs PNG</th><th>vs npz</th><th>Lossless</th><th>Replay fps</th><th>Load fps</th></tr></thead>
    <tbody>
    {''.join(rows)}
    </tbody>
  </table>
  <img src="../experiments/exp1_cpu_state_replay/plot.png" alt="Disk saving and throughput">
  <p>Library: <a href="https://github.com/galilai-group/stable-envs">galilai-group/stable-envs</a></p>
</main>
</body>
</html>
"""
    VIEW.mkdir(parents=True, exist_ok=True)
    (VIEW / "index.html").write_text(html)


def main() -> None:
    torch.set_num_threads(4)
    status = json.loads((EXP / "status.json").read_text())["status"]
    frames, stats = rollout()
    if frames.shape[1] % 2 or frames.shape[2] % 2:
        raise RuntimeError(f"odd native frame {frames.shape}")
    write_video(frames, stats, status, ART / "stable-envs-state-replay.mp4")
    write_still(frames, stats, status, ART / "stable-envs-state-replay.png")
    write_viewer(status)
    note = ART / "media_backend.txt"
    note.write_text(
        "TwoRoom CPU rollout, seed 0, 48 steps, re-rendered from torch.save states "
        f"in a new env. lossless_fraction={stats['lossless_fraction']} max_abs={stats['max_abs']}. "
        "Video via ffmpeg libx264. No GPU.\n"
    )
    print("media", stats)


if __name__ == "__main__":
    main()
