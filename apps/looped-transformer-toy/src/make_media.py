"""Screenshot (PNG) + short video (MP4) of looped vs baseline addition decode."""

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

from looped_transformer import (
    EQ_ID,
    PLUS_ID,
    TinyLoopedTransformer,
    encode_addition,
)

APP = Path(__file__).resolve().parents[1]
ART = APP / "artifacts"
WORK = APP / "work"

plt.rcParams.update({
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "figure.dpi": 140,
})


def _load(kind: str, seed: int = 0) -> TinyLoopedTransformer:
    ckpt = torch.load(WORK / f"model_{kind}_seed{seed}.pt", weights_only=False)
    mc = ckpt["model_cfg"]
    m = TinyLoopedTransformer(
        d_model=int(mc["d_model"]),
        n_layers=int(mc["n_layers"]),
        n_heads=int(mc["n_heads"]),
        dropout=float(mc.get("dropout", 0.0)),
        ff_mult=int(mc.get("ff_mult", 4)),
        n_loops=int(mc["n_loops"]),
    )
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


def _tok_str(tid: int) -> str:
    if tid == PLUS_ID:
        return "+"
    if tid == EQ_ID:
        return "="
    if 0 <= tid <= 9:
        return str(tid)
    return "?"


def _decode_sum(model: TinyLoopedTransformer, a: int, b: int, n_digits: int, n_loops: int):
    toks, _ = encode_addition(a, b, n_digits)
    prefix_len = 2 * n_digits + 2
    sum_len = n_digits + 1
    cur = torch.as_tensor([toks[:prefix_len]], dtype=torch.long)
    probs_traj = []
    preds = []
    with torch.no_grad():
        for _ in range(sum_len):
            logits = model(cur, n_loops=n_loops)
            probs = torch.softmax(logits[0, -1, :10], dim=-1).cpu().numpy()
            nxt = int(logits[0, -1, :].argmax().item())
            probs_traj.append(probs)
            preds.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    truth = toks[prefix_len:]
    return preds, truth, probs_traj, toks[:prefix_len]


def main() -> None:
    ART.mkdir(exist_ok=True)
    looped = _load("looped", 0)
    baseline = _load("baseline", 0)

    # A carry-heavy example that needs depth: 9999+1=10000 (LSB-first)
    a, b, n_digits = 9999, 1, 4
    lp, lt, lprobs, prefix = _decode_sum(looped, a, b, n_digits, n_loops=4)
    bp, bt, bprobs, _ = _decode_sum(baseline, a, b, n_digits, n_loops=1)

    prefix_str = " ".join(_tok_str(t) for t in prefix)
    truth_str = " ".join(_tok_str(t) for t in lt)
    looped_str = " ".join(_tok_str(t) for t in lp)
    base_str = " ".join(_tok_str(t) for t in bp)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, probs, title, pred in (
        (axes[0], lprobs, f"Looped K=4 → {looped_str}", lp),
        (axes[1], bprobs, f"Baseline K=1 → {base_str}", bp),
    ):
        mat = np.stack(probs, axis=0)  # (sum_len, 10)
        im = ax.imshow(mat.T, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1, origin="lower")
        ax.set_xlabel("Generated sum digit (LSB → MSB)")
        ax.set_ylabel("Digit id")
        ax.set_yticks(range(10))
        ax.set_xticks(range(len(pred)))
        ax.set_xticklabels([_tok_str(p) for p in pred])
        for i, (p, t) in enumerate(zip(pred, lt)):
            color = "#006d2c" if p == t else "#a50f15"
            ax.scatter([i], [t], marker="o", s=60, facecolors="none", edgecolors=color, linewidths=2)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(digit)")

    fig.suptitle(
        f"Looped transformer toy — decode {a}+{b}  (prefix LSB-first: {prefix_str} | truth {truth_str})",
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout()
    out_png = ART / "looped-transformer-toy.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")

    # Video: animate test-time loop count for the looped model on the same example
    tmp = Path(tempfile.mkdtemp(prefix="loop_frames_"))
    try:
        loop_counts = [1, 2, 4, 8, 16]
        frames = []
        for i, k in enumerate(loop_counts):
            pred, truth, probs, _ = _decode_sum(looped, a, b, n_digits, n_loops=k)
            fig, ax = plt.subplots(figsize=(6.4, 4.4))
            mat = np.stack(probs, axis=0)
            im = ax.imshow(mat.T, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1, origin="lower")
            ax.set_yticks(range(10))
            ax.set_xticks(range(len(pred)))
            ax.set_xticklabels([_tok_str(p) for p in pred])
            for j, (p, t) in enumerate(zip(pred, truth)):
                color = "#006d2c" if p == t else "#a50f15"
                ax.scatter([j], [t], marker="o", s=70, facecolors="none", edgecolors=color, linewidths=2)
            ok = "MATCH" if pred == truth else "miss"
            ax.set_title(
                f"Test-time loops K={k}  pred={''.join(_tok_str(p) for p in pred)}  ({ok})",
                fontweight="bold",
            )
            ax.set_xlabel("Generated sum digit (LSB → MSB)")
            ax.set_ylabel("Digit id")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fp = tmp / f"frame_{i:03d}.png"
            fig.savefig(fp)
            plt.close(fig)
            frames.append(fp)
        # hold last
        for j in range(8):
            shutil.copy(frames[-1], tmp / f"frame_{len(frames)+j:03d}.png")
        # also duplicate each frame for slower pacing
        out = ART / "looped-transformer-toy.mp4"
        cmd = [
            "ffmpeg", "-y", "-framerate", "2",
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
