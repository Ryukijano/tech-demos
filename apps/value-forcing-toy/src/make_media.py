"""Screenshot and a short flow-path video for the primary seed-0 models."""

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

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from models import QuantileNet, VelocityNet, flow_snapshot, reward_multiplier  # noqa: E402

APP = SRC.parent
ART = APP / "artifacts"
WORK = APP / "work"
EXP = APP / "experiments" / "exp1_cpu_mrp"

COLORS = {"vf": "#0072B2", "endpoint": "#D55E00", "qr": "#009E73", "truth": "#222222"}
LABELS = {"vf": "Value Forcing", "endpoint": "Endpoint flow", "qr": "QR-DQN"}


def _bold_legend(ax, **kwargs):
    legend = ax.legend(**kwargs)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    return legend


def _load_velocity(method: str) -> tuple[VelocityNet, dict]:
    ckpt = torch.load(WORK / f"primary_{method}_seed0.pt", map_location="cpu", weights_only=False)
    net = VelocityNet(int(ckpt["n_states"]), int(ckpt["embed_dim"]), int(ckpt["hidden"]))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def _load_qr() -> tuple[QuantileNet, dict]:
    ckpt = torch.load(WORK / "primary_qr_seed0.pt", map_location="cpu", weights_only=False)
    net = QuantileNet(int(ckpt["n_states"]), int(ckpt["embed_dim"]), int(ckpt["hidden"]), int(ckpt["n_quantiles"]))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def _style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Liberation Serif"],
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "font.size": 11,
        }
    )


def _panel_hist(ax, hist_by_method: dict, state: str) -> None:
    shown_truth = False
    for method in ("vf", "endpoint", "qr"):
        hist = hist_by_method[method][state]
        edges = np.asarray(hist["edges"], dtype=np.float64)
        width = np.diff(edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        pred = np.asarray(hist["pred_count"], dtype=np.float64) / (hist["n_samples"] * width)
        truth = np.asarray(hist["true_mass"], dtype=np.float64) / width
        if not shown_truth:
            ax.fill_between(centers, truth, step="mid", color=COLORS["truth"], alpha=0.15, label="True")
            ax.step(centers, truth, where="mid", color=COLORS["truth"], lw=1.4)
            shown_truth = True
        ax.plot(centers, pred, color=COLORS[method], lw=1.8, label=LABELS[method])
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)


def main() -> None:
    _style()
    ART.mkdir(exist_ok=True)
    raw = json.loads((EXP / "raw.json").read_text())
    status = json.loads((EXP / "status.json").read_text())
    seed0 = raw["seeds"]["0"]
    hists = {method: seed0[method]["hist"] for method in ("vf", "endpoint", "qr")}

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.15))
    _panel_hist(axes[0], hists, "MIX")
    axes[0].set_title("MIX return (bimodal)")
    _bold_legend(axes[0], fontsize=8, loc="upper right")
    _panel_hist(axes[1], hists, "WIDE")
    axes[1].set_title("WIDE return (skewed)")
    summary = status["summary"]
    methods = ("vf", "endpoint", "qr")
    means = [summary[m]["w1_mean"] for m in methods]
    stds = [summary[m]["w1_std"] for m in methods]
    axes[2].bar(
        np.arange(3),
        means,
        yerr=stds,
        color=[COLORS[m] for m in methods],
        capsize=3,
        width=0.72,
    )
    axes[2].axhline(0.8 * summary["endpoint"]["w1_mean"], color="#333333", ls="--", lw=1.0)
    axes[2].set_xticks(np.arange(3), [LABELS[m] for m in methods], fontsize=8)
    axes[2].set_ylabel("Mean W1")
    axes[2].set_title(f"Seeds 0–2 · {status['status']}")
    axes[2].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Value Forcing toy — limited offline MRP", fontweight="bold")
    fig.tight_layout()
    png = ART / "value-forcing-toy.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)

    vf, vf_ckpt = _load_velocity("vf")
    ep, ep_ckpt = _load_velocity("endpoint")
    gamma = float(vf_ckpt["gamma"])
    scale = reward_multiplier(gamma, bool(vf_ckpt["reward_scale"]))
    names = list(vf_ckpt["names"])
    n_euler = int(vf_ckpt["n_euler"])
    # The video is in the network's scaled coordinate. τ=1 should sit on (1-γ) × return.
    focus = ["MIX", "WIDE"]
    n_draw = 2500
    taus = np.linspace(0.0, 1.0, 28)
    tmp = Path(tempfile.mkdtemp(prefix="vf_frames_"))
    bins = 48
    # Fixed axis from the true return support, wide enough for τ=0 noise after unscaling.
    # Unscaling noise by 1/(1-γ) blows up τ=0. Keep the video in scaled coordinates
    # and draw the scaled true law (atoms approximated by scaling the stored histogram).
    def truth_scaled_density(state: str, edges: np.ndarray) -> np.ndarray:
        hist = hists["vf"][state]
        src_edges = np.asarray(hist["edges"], dtype=np.float64) * scale
        src_mass = np.asarray(hist["true_mass"], dtype=np.float64)
        # Deposit each source bin's mass into the destination bin that contains its center.
        centers = 0.5 * (src_edges[:-1] + src_edges[1:])
        dest = np.zeros(edges.size - 1, dtype=np.float64)
        idx = np.clip(np.searchsorted(edges, centers, side="right") - 1, 0, dest.size - 1)
        np.add.at(dest, idx, src_mass)
        width = np.diff(edges)
        return dest / width

    # Axis limits in scaled units: noise is O(1), scaled returns are O(1).
    lo, hi = -3.2, 3.2
    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    frame_id = 0
    hold = 6
    schedule = list(taus) + [1.0] * hold
    for tau in schedule:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
        for ax, state in zip(axes, focus):
            s_id = names.index(state)
            vf_s = flow_snapshot(vf, n_draw, s_id, float(tau), n_euler, unscale=1.0)
            ep_s = flow_snapshot(ep, n_draw, s_id, float(tau), int(ep_ckpt["n_euler"]), unscale=1.0)
            ax.hist(vf_s, bins=edges, density=True, histtype="step", lw=1.8, color=COLORS["vf"], label=LABELS["vf"])
            ax.hist(ep_s, bins=edges, density=True, histtype="step", lw=1.8, color=COLORS["endpoint"], label=LABELS["endpoint"])
            truth = truth_scaled_density(state, edges)
            ax.step(centers, truth, where="mid", color=COLORS["truth"], lw=1.3, label="True (scaled)")
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, 4.6)
            ax.set_title(f"{state}   τ = {tau:.2f}")
            ax.set_xlabel("Scaled return coordinate")
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.25)
        _bold_legend(axes[0], fontsize=8, loc="upper right")
        fig.suptitle("Flow coordinate from noise (τ=0) to the scaled return (τ=1)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(tmp / f"frame_{frame_id:03d}.png", dpi=120)
        plt.close(fig)
        frame_id += 1

    mp4 = ART / "value-forcing-toy.mp4"
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found; PNG written but MP4 missing")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "8",
            "-i",
            str(tmp / "frame_%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(mp4),
        ],
        check=True,
    )
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {png} and {mp4}")


if __name__ == "__main__":
    main()
