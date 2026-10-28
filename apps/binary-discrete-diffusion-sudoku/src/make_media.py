"""Screenshot (PNG) + short video (MP4) of binary candidate-set narrowing on a puzzle."""

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

from bdd import (
    TinySudokuDiT,
    make_puzzle_pool,
    punch_clues,
    sample_binary,
)

APP = Path(__file__).resolve().parents[1]
ART = APP / "artifacts"
WORK = APP / "work"

plt.rcParams.update({
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "figure.dpi": 140,
})


def _load_binary() -> TinySudokuDiT:
    ckpt = torch.load(WORK / "model_binary_seed0.pt", weights_only=False)
    mc = ckpt["model_cfg"]
    m = TinySudokuDiT(
        d_model=int(mc["d_model"]),
        n_layers=int(mc["n_layers"]),
        n_heads=int(mc["n_heads"]),
        dropout=float(mc.get("dropout", 0.0)),
    )
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


def _draw_candidates(ax, z: np.ndarray, clues: np.ndarray, solution: np.ndarray, title: str):
    """z: (81,9) membership; clues: (9,9) with 0 empty; solution digits 1-9."""
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    for r in range(9):
        for c in range(9):
            if clues[r, c] > 0:
                ax.text(
                    c + 0.5, r + 0.5, str(int(clues[r, c])),
                    ha="center", va="center", fontsize=13, fontweight="bold", color="#111111",
                )
            else:
                bits = z[r * 9 + c]
                cands = [i + 1 for i in range(9) if bits[i] > 0.5]
                if len(cands) == 1:
                    color = "#2166ac" if cands[0] == solution[r, c] else "#b2182b"
                    ax.text(
                        c + 0.5, r + 0.5, str(cands[0]),
                        ha="center", va="center", fontsize=12, color=color, fontweight="bold",
                    )
                else:
                    # show up to 4 candidates as mini text
                    txt = "".join(str(d) for d in cands[:5])
                    if len(cands) > 5:
                        txt += "…"
                    ax.text(
                        c + 0.5, r + 0.55, txt,
                        ha="center", va="center", fontsize=7, color="#555555",
                    )
                    ax.text(
                        c + 0.5, r + 0.88, f"|{len(cands)}|",
                        ha="center", va="center", fontsize=6, color="#888888",
                    )
    for i in range(10):
        lw = 2.0 if i % 3 == 0 else 0.6
        ax.axhline(i, color="black", lw=lw)
        ax.axvline(i, color="black", lw=lw)


def main() -> None:
    ART.mkdir(exist_ok=True)
    model = _load_binary()
    sols = make_puzzle_pool(1, seed=4242)
    rng = np.random.default_rng(7)
    clues_np, mask_np = punch_clues(sols, 36, rng)
    solution = sols[0]
    clues = clues_np[0]

    clue_mask = torch.as_tensor(mask_np.reshape(1, 81))
    clean_digits = torch.as_tensor(solution.reshape(1, 81) - 1, dtype=torch.long)
    clues_t = torch.where(clue_mask, clean_digits, torch.zeros_like(clean_digits))
    g = torch.Generator().manual_seed(0)
    pred, traj = sample_binary(
        model, clues_t, clue_mask, nfe=32, t_min=0.02, generator=g, record=True
    )
    pred_board = pred.numpy().reshape(9, 9) + 1

    # Screenshot: clues | mid candidates | final decode
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    mid = traj[len(traj) // 2].numpy()[0]
    final_z = traj[-1].numpy()[0]
    empty_z = np.zeros((81, 9))
    _draw_candidates(axes[0], empty_z, clues, solution, "Clues (36 givens)")
    _draw_candidates(axes[1], mid, clues, solution, f"Binary state @ step {len(traj)//2}")
    # final panel: decoded prediction
    final_clues = pred_board.copy()
    # show all as committed digits for readability
    z_final_show = np.zeros((81, 9))
    for i, d in enumerate(pred_board.ravel()):
        z_final_show[i, d - 1] = 1.0
    _draw_candidates(
        axes[2], z_final_show, clues, solution,
        f"Decoded ({'SOLVED' if np.array_equal(pred_board, solution) else 'unsolved'})",
    )
    fig.suptitle(
        "Binary Discrete Diffusion — candidate-set narrowing on Sudoku (CPU toy)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(ART / "binary-discrete-diffusion-sudoku.png", bbox_inches="tight")
    fig.savefig(ART / "binary-discrete-diffusion-sudoku-grid.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote screenshot png")

    # Video frames along traj
    tmp = Path(tempfile.mkdtemp(prefix="bdd_frames_"))
    try:
        # include an initial fair-bit-like frame from first traj
        frames = traj
        for i, zt in enumerate(frames):
            fig, ax = plt.subplots(figsize=(5.2, 5.2))
            _draw_candidates(
                ax, zt.numpy()[0], clues, solution,
                f"Binary denoising step {i+1}/{len(frames)}",
            )
            fig.tight_layout()
            fig.savefig(tmp / f"frame_{i:03d}.png")
            plt.close(fig)
        # hold last frame
        for j in range(6):
            shutil.copy(tmp / f"frame_{len(frames)-1:03d}.png", tmp / f"frame_{len(frames)+j:03d}.png")
        out = ART / "binary-discrete-diffusion-sudoku.mp4"
        cmd = [
            "ffmpeg", "-y", "-framerate", "6",
            "-i", str(tmp / "frame_%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"wrote {out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
