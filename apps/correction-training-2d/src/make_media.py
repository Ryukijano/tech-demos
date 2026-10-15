"""Media: K-sweep scatter grid (PNG) + sampling-trajectory video (MP4, needs ffmpeg).

Loads work/model_seed0.pt saved by run_experiment.py.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ctd2d import TimeMLP, make_ring_data, sample, steps_for_budget

APP = Path(__file__).resolve().parents[1]
ART = APP / "artifacts"
NFE = 120
K_GRID = [0, 1, 2, 4, 8]

plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold",
                     "axes.labelweight": "bold", "figure.dpi": 140})


def load_models() -> tuple[TimeMLP, TimeMLP, dict]:
    ckpt = torch.load(APP / "work" / "model_seed0.pt", weights_only=False)
    mc = ckpt["model_cfg"]
    D = TimeMLP(int(mc["hidden"]) if "hidden" in mc else 128)
    C = TimeMLP(int(mc["hidden"]) if "hidden" in mc else 128)
    D.load_state_dict(ckpt["D"])
    C.load_state_dict(ckpt["C"])
    D.eval()
    C.eval()
    return D, C, ckpt["data_cfg"]


def scatter(ax, pts, color, title):
    ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.4, c=color, linewidths=0)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def make_grid(D, C, data_cfg) -> None:
    gen = torch.Generator().manual_seed(123)
    data = make_ring_data(2000, generator=gen, **data_cfg)
    fig, axes = plt.subplots(1, 1 + len(K_GRID), figsize=(3 * (1 + len(K_GRID)), 3.3))
    scatter(axes[0], data, "#444444", "data (8 Gaussians)")
    for i, K in enumerate(K_GRID):
        steps = steps_for_budget(NFE, K)
        g = torch.Generator().manual_seed(10_000)
        pts, nfe, _ = sample(D, C, 2000, steps=steps, K=K, generator=g)
        scatter(axes[1 + i], pts, "#1f77b4" if K == 0 else "#d62728",
                f"K={K} · {steps} steps · NFE {nfe}")
    fig.suptitle("Correction Training 2D toy — matched-NFE K sweep (seed 0)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(ART / "correction-training-2d-grid.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote grid png")


def make_video(D, C, data_cfg) -> None:
    """Side-by-side sampling trajectories: K=0 (120 steps) vs K=8 (13 steps)."""
    n = 1200
    runs = []
    for K in (0, 8):
        steps = steps_for_budget(NFE, K)
        g = torch.Generator().manual_seed(10_000)
        _, nfe, traj = sample(D, C, n, steps=steps, K=K, generator=g, record=True)
        runs.append((K, steps, nfe, traj))

    gen = torch.Generator().manual_seed(123)
    data = make_ring_data(2000, generator=gen, **data_cfg)

    n_frames = 150
    tmp = Path(tempfile.mkdtemp(prefix="ct2d_frames_"))
    for f in range(n_frames):
        frac = f / (n_frames - 1)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
        for ax, (K, steps, nfe, traj) in zip(axes, runs):
            idx = min(int(round(frac * steps)), steps)
            ax.scatter(data[:, 0], data[:, 1], s=2, alpha=0.12, c="#888888", linewidths=0)
            pts = traj[idx]
            ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.6,
                       c="#1f77b4" if K == 0 else "#d62728", linewidths=0)
            ax.set_xlim(-3.4, 3.4)
            ax.set_ylim(-3.4, 3.4)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"K={K} · step {idx}/{steps} · NFE budget {NFE}", fontsize=10)
        fig.suptitle("Sampling t=1 → 0 · baseline vs iterative correction "
                     f"(progress {100*frac:.0f}%)", fontweight="bold", fontsize=11)
        fig.tight_layout()
        fig.savefig(tmp / f"frame_{f:04d}.png")
        plt.close(fig)

    out = ART / "correction-training-2d.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "25",
         "-i", str(tmp / "frame_%04d.png"),
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )
    shutil.rmtree(tmp)
    print(f"wrote {out}")


def main() -> None:
    ART.mkdir(exist_ok=True)
    D, C, data_cfg = load_models()
    make_grid(D, C, data_cfg)
    make_video(D, C, data_cfg)
    # Main screenshot = the grid
    shutil.copyfile(ART / "correction-training-2d-grid.png",
                    ART / "correction-training-2d.png")


if __name__ == "__main__":
    main()
