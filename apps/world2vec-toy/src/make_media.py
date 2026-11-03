"""Screenshot (PNG) + short video (MP4) of pendulum rollouts / latent forecasts."""

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

from world2vec_toy import (
    WorldModel,
    collect_trajectories,
    pendulum_cfg_from_data,
    render_pendulum,
)

APP = Path(__file__).resolve().parents[1]
ART = APP / "artifacts"
WORK = APP / "work"
EXP = APP / "experiments" / "exp1_cpu_pendulum"

plt.rcParams.update({
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "figure.dpi": 140,
})


def _load(kind: str, seed: int = 0) -> WorldModel:
    ckpt = torch.load(WORK / f"model_{kind}_seed{seed}.pt", weights_only=False)
    mc = ckpt["model_cfg"]
    model = WorldModel(
        kind=kind,  # type: ignore[arg-type]
        history=int(mc["history"]),
        img_size=int(mc["img_size"]),
        latent_dim=int(mc["latent_dim"]),
        cnn_channels=list(mc["cnn_channels"]),
        dyn_hidden=list(mc["dyn_hidden"]),
        encoder_mlp_hidden=list(mc["encoder_mlp_hidden"]),
        dt=float(mc["dt"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main() -> None:
    ART.mkdir(exist_ok=True)
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    d = cfg["data"]
    pcfg = pendulum_cfg_from_data(d)
    history = int(d["history"])

    # Fresh short traj for visualization
    data = collect_trajectories(1, 40, pcfg, seed=42)
    frames = data.frames[0]  # (T,H,W)
    actions = data.actions[0]
    thetas = data.thetas[0]
    T, S, _ = frames.shape

    model_w = _load("world2vec", 0)
    model_d = _load("discrete", 0)

    # Encode + rollout from t0
    t0 = history - 1
    H_roll = 16
    x0 = torch.from_numpy(frames[t0 - history + 1 : t0 + 1]).unsqueeze(0)
    z0_w = model_w.encoder(x0)
    z0_d = model_d.encoder(x0)
    act = torch.from_numpy(actions[t0 : t0 + H_roll]).unsqueeze(0)
    with torch.no_grad():
        zh_w = model_w.rollout(z0_w, act)[0].numpy()
        zh_d = model_d.rollout(z0_d, act)[0].numpy()
        # encoded futures for overlay
        z_enc = []
        for j in range(1, H_roll + 1):
            tj = t0 + j
            xj = torch.from_numpy(frames[tj - history + 1 : tj + 1]).unsqueeze(0)
            z_enc.append(model_w.encoder(xj)[0].numpy())
        z_enc = np.stack(z_enc, axis=0)

    # PCA of latents for 2D path viz
    Z = np.concatenate([z0_w.detach().numpy(), zh_w, z_enc], axis=0)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    proj = Zc @ vt[:2].T
    p0 = proj[0]
    pw = proj[1 : 1 + H_roll]
    pe = proj[1 + H_roll :]

    Zd = np.concatenate([z0_d.detach().numpy(), zh_d], axis=0)
    Zdc = Zd - Zd.mean(axis=0, keepdims=True)
    pd = (Zdc @ vt[:2].T)  # project discrete into world2vec PCA for comparison attempt
    # Better: own PCA for discrete
    _, _, vtd = np.linalg.svd(Zdc, full_matrices=False)
    pd = Zdc @ vtd[:2].T

    # Static PNG panel
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.5))
    # row0: true frames at t0, t0+8, t0+16
    for ax, j, title in zip(
        axes[0],
        [0, 8, 16],
        ["t = t₀ (pixels)", "true @ t₀+8", "true @ t₀+16"],
    ):
        idx = min(t0 + j, T - 1)
        ax.imshow(frames[idx], cmap="magma", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"θ={thetas[idx]:.2f}")

    # latent paths
    ax = axes[1, 0]
    ax.plot(pe[:, 0], pe[:, 1], "o-", color="#2a2a2a", ms=3, label="encoded", lw=1.2)
    ax.plot(pw[:, 0], pw[:, 1], "s--", color="#1b6ca8", ms=3, label="world2vec roll", lw=1.4)
    ax.scatter([p0[0]], [p0[1]], c="#c62828", s=40, zorder=5, label="start")
    ax.set_title("Latent PCA · world2vec")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8, prop={"weight": "bold"})
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(pd[1:, 0], pd[1:, 1], "s--", color="#c47a2c", ms=3, label="discrete roll", lw=1.4)
    ax.scatter([pd[0, 0]], [pd[0, 1]], c="#c62828", s=40, zorder=5, label="start")
    ax.set_title("Latent PCA · discrete")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8, prop={"weight": "bold"})
    ax.grid(True, alpha=0.3)

    # error vs horizon
    ax = axes[1, 2]
    err_w = np.linalg.norm(zh_w - z_enc, axis=1)
    # discrete vs its own encodings
    z_enc_d = []
    with torch.no_grad():
        for j in range(1, H_roll + 1):
            tj = t0 + j
            xj = torch.from_numpy(frames[tj - history + 1 : tj + 1]).unsqueeze(0)
            z_enc_d.append(model_d.encoder(xj)[0].numpy())
    z_enc_d = np.stack(z_enc_d, axis=0)
    err_d = np.linalg.norm(zh_d - z_enc_d, axis=1)
    hs = np.arange(1, H_roll + 1)
    ax.plot(hs, err_w, "-o", color="#1b6ca8", lw=2, label="world2vec", ms=4)
    ax.plot(hs, err_d, "-s", color="#c47a2c", lw=2, label="discrete", ms=4)
    ax.set_xlabel("horizon (steps)")
    ax.set_ylabel(r"$\|z_{\mathrm{hat}}-z_{\mathrm{enc}}\|_2$")
    ax.set_title("Rollout latent error")
    ax.legend(prop={"weight": "bold"})
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "world2vec-toy · rendered pendulum · latent dynamics rollouts (seed 0)",
        fontweight="bold",
    )
    fig.tight_layout()
    png = ART / "world2vec-toy.png"
    fig.savefig(png)
    plt.close(fig)
    print(f"wrote {png}", flush=True)

    # Video: animate true pendulum + latent error curves growing
    tmp = Path(tempfile.mkdtemp(prefix="w2v_media_"))
    try:
        n_frames = min(36, T - 1)
        for fi in range(n_frames):
            fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))
            ax = axes[0]
            ax.imshow(frames[fi], cmap="magma", vmin=0, vmax=1)
            ax.set_title(f"Rendered pendulum  t={fi}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.02,
                0.98,
                f"θ={thetas[fi]:.2f}  ω={data.omegas[0, fi]:.2f}",
                transform=ax.transAxes,
                va="top",
                color="white",
                fontsize=10,
                fontweight="bold",
            )

            ax = axes[1]
            # live error up to min(fi, H_roll) if past t0
            ax.plot(hs, err_w, "-", color="#1b6ca8", lw=2, alpha=0.35)
            ax.plot(hs, err_d, "-", color="#c47a2c", lw=2, alpha=0.35)
            if fi >= t0:
                upto = min(fi - t0, H_roll)
                if upto >= 1:
                    ax.plot(hs[:upto], err_w[:upto], "-o", color="#1b6ca8", lw=2.5, ms=5, label="world2vec")
                    ax.plot(hs[:upto], err_d[:upto], "-s", color="#c47a2c", lw=2.5, ms=5, label="discrete")
            ax.set_xlim(1, H_roll)
            ax.set_ylim(0, max(err_w.max(), err_d.max()) * 1.15 + 1e-6)
            ax.set_xlabel("horizon (steps)")
            ax.set_ylabel("latent rollout error")
            ax.set_title("Forecast error vs horizon")
            ax.legend(loc="upper left", prop={"weight": "bold"})
            ax.grid(True, alpha=0.3)
            fig.suptitle("world2vec-toy · continuous-time latent dynamics", fontweight="bold")
            fig.tight_layout()
            fig.savefig(tmp / f"frame_{fi:04d}.png")
            plt.close(fig)

        mp4 = ART / "world2vec-toy.mp4"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", "8",
            "-i", str(tmp / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            str(mp4),
        ]
        subprocess.run(cmd, check=True)
        print(f"wrote {mp4}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Copy to docs/assets if present
    docs_assets = APP.parent.parent / "docs" / "assets"
    if docs_assets.is_dir():
        shutil.copy2(png, docs_assets / "world2vec-toy.png")
        shutil.copy2(ART / "world2vec-toy.mp4", docs_assets / "world2vec-toy.mp4")
        print(f"copied media → {docs_assets}", flush=True)


if __name__ == "__main__":
    main()
