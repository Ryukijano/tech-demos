"""Screenshot (PNG) + short video (MP4) of the trained agent playing MinAtar."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dreamer_minatar import load_agent, make_env, obs_tensor, select_action

APP = Path(__file__).resolve().parents[1]
ART = APP / "artifacts"
WORK = APP / "work"

# MinAtar-ish palette (channel-tinted)
CHANNEL_COLORS = [
    np.array([0.90, 0.30, 0.25]),
    np.array([0.20, 0.55, 0.85]),
    np.array([0.95, 0.75, 0.20]),
    np.array([0.30, 0.75, 0.40]),
    np.array([0.70, 0.40, 0.85]),
    np.array([0.95, 0.55, 0.20]),
    np.array([0.40, 0.80, 0.85]),
    np.array([0.85, 0.85, 0.90]),
]


def frame_to_rgb(obs_chw: np.ndarray) -> np.ndarray:
    """obs: C,H,W float in {0,1} → H,W,3 uint8."""
    c, h, w = obs_chw.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    bg = np.array([0.08, 0.10, 0.14], dtype=np.float32)
    rgb[:] = bg
    for i in range(c):
        mask = obs_chw[i] > 0.5
        color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
        rgb[mask] = color
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def upsample(rgb: np.ndarray, scale: int = 24) -> np.ndarray:
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def render_episode(game: str, seed: int, max_steps: int = 200):
    ckpt = WORK / f"agent_{game}_seed0.pt"
    if not ckpt.exists():
        # fall back to any seed
        cands = sorted(WORK.glob(f"agent_{game}_seed*.pt"))
        if not cands:
            raise FileNotFoundError(f"no checkpoint for {game}")
        ckpt = cands[0]
    agent, _meta = load_agent(ckpt)
    env = make_env(game, sticky_action_prob=0.1)
    np.random.seed(seed)
    env.reset()
    frames = []
    hist = None
    last_a = None
    ep_ret = 0.0
    rng = np.random.default_rng(seed)
    terminated = False
    steps = 0
    while not terminated and steps < max_steps:
        obs = obs_tensor(env)
        frames.append(frame_to_rgb(obs.numpy()))
        action, hist, _z = select_action(
            agent, obs, hist, last_a, epsilon=0.0, greedy=True, rng=rng
        )
        reward, terminated = env.act(action)
        ep_ret += float(reward)
        last_a = action
        steps += 1
    frames.append(frame_to_rgb(obs_tensor(env).numpy()))
    return frames, ep_ret, ckpt.name


def main() -> None:
    ART.mkdir(exist_ok=True)
    # Prefer breakout for the hero shot; also try space_invaders
    game = "breakout"
    frames, ep_ret, ckpt_name = render_episode(game, seed=7, max_steps=180)

    # Screenshot: montage of early / mid / late frames + title panel
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    picks = [0, len(frames) // 2, len(frames) - 1]
    titles = ["start", "mid", "end"]
    for ax, idx, title in zip(axes, picks, titles):
        img = upsample(frames[idx], scale=20)
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"{game} · {title}", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"NanoDreamer-MinAtar toy · {ckpt_name} · episode return {ep_ret:.1f}",
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout()
    png_path = ART / "nanodreamer-minatar.png"
    fig.savefig(png_path)
    plt.close(fig)

    # Video via ffmpeg
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        for i, fr in enumerate(frames):
            big = upsample(fr, scale=24)
            # pad to even dims for yuv420p
            h, w, _ = big.shape
            if h % 2:
                big = np.pad(big, ((0, 1), (0, 0), (0, 0)), mode="edge")
            if w % 2:
                big = np.pad(big, ((0, 0), (0, 1), (0, 0)), mode="edge")
            plt.imsave(tdir / f"frame_{i:04d}.png", big)
        mp4 = ART / "nanodreamer-minatar.mp4"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", "12",
            "-i", str(tdir / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "20",
            str(mp4),
        ]
        subprocess.check_call(cmd)

    # Copy into docs/assets when present
    docs_assets = APP.parents[1] / "docs" / "assets"
    if docs_assets.is_dir():
        shutil.copy2(png_path, docs_assets / "nanodreamer-minatar.png")
        shutil.copy2(ART / "nanodreamer-minatar.mp4", docs_assets / "nanodreamer-minatar.mp4")

    print(f"[media] wrote {png_path} and mp4 (return={ep_ret:.1f})", flush=True)


if __name__ == "__main__":
    main()
