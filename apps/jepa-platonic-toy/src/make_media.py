"""Screenshot (PNG) + short video (MP4) of embedding alignment animation."""

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
import yaml

from jepa_toy import (
    MLPEncoder,
    build_world,
    embed_paired,
    orthogonal_procrustes,
    _center,
)

APP = Path(__file__).resolve().parents[1]
ART = APP / "artifacts"
WORK = APP / "work"
EXP = APP / "experiments" / "exp1_cpu_views"

plt.rcParams.update({
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "figure.dpi": 140,
})


def _load_pair(kind: str, seed: int = 0):
    ckpt = torch.load(WORK / f"model_{kind}_seed{seed}.pt", weights_only=False)
    mc = ckpt["model_cfg"]
    enc_a = MLPEncoder(
        int(mc["d_mod_a"]),
        list(mc["arch_a"]),
        int(mc["emb_dim"]),
        dropout=float(mc.get("dropout", 0.0)),
    )
    enc_b = MLPEncoder(
        int(mc["d_mod_b"]),
        list(mc["arch_b"]),
        int(mc["emb_dim"]),
        dropout=float(mc.get("dropout", 0.0)),
    )
    enc_a.load_state_dict(ckpt["state_dict_a"])
    enc_b.load_state_dict(ckpt["state_dict_b"])
    enc_a.eval()
    enc_b.eval()
    return enc_a, enc_b


def _pca2(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:2].T


def main() -> None:
    ART.mkdir(exist_ok=True)
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    world = build_world(cfg["data"])

    enc_j_a, enc_j_b = _load_pair("jepa", 0)
    enc_r_a, enc_r_b = _load_pair("recon", 0)
    enc_n_a, enc_n_b = _load_pair("random", 0)

    n_plot = 400
    eval_seed = int(cfg["data"]["eval_stream_seed"])

    def get_xy(enc_a, enc_b):
        ea, eb, cls = embed_paired(world, enc_a, enc_b, n_plot, eval_seed)
        eb_al, _ = orthogonal_procrustes(ea, eb)
        # Joint PCA on concatenated centered embeddings for shared plane
        joint = torch.cat([_center(ea), eb_al], dim=0).numpy()
        plane = _pca2(joint)
        return plane[:n_plot], plane[n_plot:], cls.numpy()

    panels = [
        ("JEPA (good views)", *get_xy(enc_j_a, enc_j_b), "#1f6f5a"),
        ("Reconstruction", *get_xy(enc_r_a, enc_r_b), "#b86b2a"),
        ("Random init", *get_xy(enc_n_a, enc_n_b), "#6b7280"),
    ]

    # Static screenshot
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    cmap = plt.get_cmap("tab10")
    for ax, (title, xa, xb, cls, color) in zip(axes, panels):
        for c in range(int(cls.max()) + 1):
            m = cls == c
            ax.scatter(xa[m, 0], xa[m, 1], s=18, c=[cmap(c)], marker="o", alpha=0.75, label=f"A/{c}" if c == 0 else None)
            ax.scatter(xb[m, 0], xb[m, 1], s=18, c=[cmap(c)], marker="x", alpha=0.75, label=f"B/{c}" if c == 0 else None)
        ax.set_title(title, color=color)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=8, loc="best", title="mod A ○ / B ×")
    fig.suptitle("Procrustes-aligned embeddings (seed 0) — shared world classes", fontweight="bold")
    fig.tight_layout()
    png = ART / "jepa-platonic-toy.png"
    fig.savefig(png)
    plt.close(fig)

    # Animated frames: lerp random → JEPA alignment (visual presence)
    # Use JEPA pair; animate Procrustes blend from identity to optimal R
    ea, eb, cls = embed_paired(world, enc_j_a, enc_j_b, n_plot, eval_seed)
    Ac = _center(ea)
    Bc = _center(eb)
    _, R = orthogonal_procrustes(ea, eb)
    I = torch.eye(R.shape[0])

    tmp = Path(tempfile.mkdtemp(prefix="jepa_platonic_frames_"))
    n_frames = 36
    for fi in range(n_frames):
        t = fi / (n_frames - 1)
        # Geodesic-ish blend via matrix lerp + polar re-orth (simple SVD polar)
        M = (1 - t) * I + t * R
        U, _, Vh = torch.linalg.svd(M, full_matrices=False)
        Rt = U @ Vh
        Bt = Bc @ Rt
        joint = torch.cat([Ac, Bt], dim=0).numpy()
        plane = _pca2(joint)
        xa, xb = plane[:n_plot], plane[n_plot:]
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        for c in range(int(cls.max()) + 1):
            m = (cls.numpy() == c)
            ax.scatter(xa[m, 0], xa[m, 1], s=22, c=[cmap(c)], marker="o", alpha=0.8)
            ax.scatter(xb[m, 0], xb[m, 1], s=22, c=[cmap(c)], marker="x", alpha=0.8)
        ax.set_title(f"Orthogonal Procrustes blend  t={t:.2f}", fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        fig.savefig(tmp / f"frame_{fi:03d}.png")
        plt.close(fig)

    mp4 = ART / "jepa-platonic-toy.mp4"
    # Prefer ffmpeg; fall back to copying last frame note if missing
    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", "12",
                "-i", str(tmp / "frame_%03d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "20",
                str(mp4),
            ],
            check=True,
        )
    else:
        # Still produce something
        shutil.copy(tmp / f"frame_{n_frames-1:03d}.png", ART / "jepa-platonic-toy-last.png")
        raise SystemExit("ffmpeg not found; PNG written but MP4 missing")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {png} and {mp4}")


if __name__ == "__main__":
    main()
