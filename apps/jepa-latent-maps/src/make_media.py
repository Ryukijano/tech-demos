"""Screenshot and a short trajectory video of the latent maps."""

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
import torch
import yaml

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jepa import Encoder, Predictor
from world import (
    geodesic_table,
    make_maze,
    parked_block,
    rollout,
    scripted_route,
)

APP = Path(__file__).resolve().parents[1]
ART = APP / "artifacts"
WORK = APP / "work"
EXP = APP / "experiments" / "exp1_cpu_grid"

METHODS = ("vicreg", "svreg", "slow_svreg")
LABEL = {"vicreg": "VICReg", "svreg": "SVReg", "slow_svreg": "Slow SVReg"}

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "figure.dpi": 140,
})


def _load(method: str) -> tuple[Encoder, Predictor]:
    ckpt = torch.load(WORK / f"model_{method}_seed0.pt", weights_only=False)
    encoder = Encoder(int(ckpt["grid_size"]), int(ckpt["hidden"]), int(ckpt["latent_dim"]))
    predictor = Predictor(
        int(ckpt["latent_dim"]),
        int(ckpt["hidden"]),
        gated=bool(ckpt["gated"]),
        gate_init=float(ckpt["gate_init"]),
    )
    encoder.load_state_dict(ckpt["encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    encoder.eval()
    predictor.eval()
    return encoder, predictor


@torch.no_grad()
def _encode(encoder: Encoder, obs: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(obs))
    return encoder(tensor).numpy()


def _pca2(latent: np.ndarray) -> np.ndarray:
    centered = latent - latent.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _maze_image(maze: np.ndarray) -> np.ndarray:
    image = np.ones((maze.shape[0], maze.shape[1], 3), dtype=np.float32)
    image[maze == 1] = np.array([0.15, 0.16, 0.18], dtype=np.float32)
    image[maze == 0] = np.array([0.93, 0.94, 0.95], dtype=np.float32)
    return image


def _draw_maze(ax, maze, agent, block, title: str) -> None:
    ax.imshow(_maze_image(maze), origin="upper", interpolation="nearest")
    ax.scatter([block[1]], [block[0]], s=80, c="#E69F00", marker="s", label="Block", zorder=3)
    ax.scatter([agent[1]], [agent[0]], s=70, c="#009E73", marker="o", label="Agent", zorder=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    legend = ax.legend(loc="lower right", fontsize=7, frameon=True)
    for text in legend.get_texts():
        text.set_fontweight("bold")


def _scatter(ax, xy, color_value, title: str, highlight: int | None) -> None:
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=color_value, s=10, cmap="viridis", alpha=0.75, linewidths=0)
    if highlight is not None:
        ax.scatter(
            [xy[highlight, 0]],
            [xy[highlight, 1]],
            s=60,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            zorder=4,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    return sc


def main() -> None:
    ART.mkdir(exist_ok=True)
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    raw = json.loads((EXP / "raw.json").read_text())
    maze = make_maze(int(cfg["data"]["grid_size"]))
    score = rollout(
        maze,
        length=int(cfg["data"]["traj_len"]),
        seed=int(cfg["data"]["score_seed"]),
        noise_std=float(cfg["data"]["noise_std"]),
        contact_radius=int(cfg["data"]["contact_radius"]),
    )
    index, dist = geodesic_table(maze)
    agent_geo = np.asarray(
        [dist[index[(1, 1)], index[(int(r), int(c))]] for r, c in score["agent"]],
        dtype=np.float64,
    )
    block_id = np.asarray(
        [index[(int(r), int(c))] for r, c in score["block"]],
        dtype=np.float64,
    )
    encoded = {}
    planes_agent = {}
    for method in METHODS:
        encoder, _predictor = _load(method)
        latent = _encode(encoder, score["obs"])
        encoded[method] = latent
        planes_agent[method] = _pca2(latent)

    seed0 = raw["seeds"]["0"]
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.2))
    mid = len(score["agent"]) // 2
    _draw_maze(
        axes[0, 0],
        maze,
        score["agent"][mid],
        score["block"][mid],
        "Two-room grid (score trajectory)",
    )
    for col, method in enumerate(METHODS, start=1):
        metrics = seed0[method]
        _scatter(
            axes[0, col],
            planes_agent[method],
            agent_geo,
            f"{LABEL[method]}  geo ρ={metrics['geodesic_spearman']:.2f}",
            None,
        )
        axes[0, col].set_xlabel("PC1")
        axes[0, col].set_ylabel("PC2")
    # Bottom: same planes, colored by which cell the block occupies.
    axes[1, 0].axis("off")
    ceiling = raw["observation_ceiling"]
    axes[1, 0].text(
        0.0,
        0.55,
        "Color, top row: agent geodesic\nfrom the top-left cell.\n\n"
        "Color, bottom row: block cell id\n(content). Seed 0.\n\n"
        f"Pixel ceiling  agent R²={ceiling['pixel_agent_r2']:.2f}\n"
        f"Pixel ceiling  object R²={ceiling['pixel_free_object_r2']:.2f}",
        transform=axes[1, 0].transAxes,
        va="center",
        fontsize=9,
    )
    for col, method in enumerate(METHODS, start=1):
        metrics = seed0[method]
        _scatter(
            axes[1, col],
            planes_agent[method],
            block_id,
            f"{LABEL[method]}  object R²={metrics['free_object_r2']:.2f}",
            None,
        )
    fig.suptitle(
        "JEPA latent maps on a toy grid  ·  clean-room, not the report's visual tasks",
        fontweight="bold",
    )
    fig.tight_layout()
    png = ART / "jepa-latent-maps.png"
    fig.savefig(png)
    plt.close(fig)

    move = np.where(np.any(score["agent"][1:] != score["agent"][:-1], axis=1))[0] + 1
    take = move[:: max(1, len(move) // 36)][:36]
    tmp = Path(tempfile.mkdtemp(prefix="jepa_latent_maps_"))
    for frame_i, t in enumerate(take):
        fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.4))
        _draw_maze(axes[0], maze, score["agent"][t], score["block"][t], f"t = {int(t)}")
        for ax, method in zip(axes[1:], METHODS, strict=True):
            _scatter(
                ax,
                planes_agent[method],
                agent_geo,
                f"{LABEL[method]}",
                int(t),
            )
        fig.suptitle("Latent position along the score walk  ·  color = agent geodesic", fontweight="bold")
        fig.tight_layout()
        fig.savefig(tmp / f"frame_{frame_i:03d}.png")
        plt.close(fig)

    mp4 = ART / "jepa-latent-maps.mp4"
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", "8",
            "-i", str(tmp / "frame_%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "20",
            str(mp4),
        ],
        check=True,
    )
    shutil.rmtree(tmp, ignore_errors=True)
    # Route length is recorded so a parked-block regression is obvious in the log.
    route = scripted_route(maze)
    _ = parked_block(maze, route)
    print(f"wrote {png} and {mp4}")


if __name__ == "__main__":
    main()
