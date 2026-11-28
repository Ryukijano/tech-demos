"""Offscreen still and a short clip of one lift and one squeeze.

OSMesa is requested before MuJoCo imports. If the renderer cannot start,
a matplotlib schematic is written instead and named in artifacts/media_backend.txt.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import atomic

os.environ.setdefault("MUJOCO_GL", "osmesa")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle, FancyBboxPatch

import scene

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_probes"
ART = APP / "artifacts"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Liberation Serif"],
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "font.size": 12,
        }
    )


def _status_line() -> str:
    path = EXP / "status.json"
    if not path.exists():
        return "PhysCaP-style lift and squeeze"
    import json

    status = json.loads(path.read_text())
    return (
        f"Status {status['status']}  ·  median |Δm| {status['mass_median_abs_kg'] * 1000:.1f} g"
        f"  ·  stiffness acc {status['stiffness_accuracy']:.2f}"
        f"  ·  selective probes {status['selective_oi']:.2f}"
    )


def _ffmpeg(frame_dir: Path, dest: Path, fps: int) -> bool:
    """Encode the clip. A missing ffmpeg leaves the still and the tables in place."""
    if shutil.which("ffmpeg") is None:
        return False
    tmp = dest.with_name(f".{dest.name}.tmp.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(tmp),
        ],
        check=True,
    )
    atomic.replace_file(tmp, dest)
    return True


def _compose(left: np.ndarray, right: np.ndarray, left_title: str, right_title: str, footer: str, path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12.80, 7.20), dpi=100)
    for ax, image, title in ((axes[0], left, left_title), (axes[1], right, right_title)):
        ax.imshow(image)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
        ax.set_axis_off()
    fig.suptitle(footer, fontsize=12, fontweight="bold", y=0.04)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=100)
    plt.close(fig)


class _Grabber:
    def __init__(self, every: int = 30) -> None:
        self.every = every
        self.n = 0
        self.frames: list[np.ndarray] = []
        self.renderer: Any = None
        self.cam: Any = None

    def __call__(self, model: Any, data: Any) -> None:
        import mujoco

        if self.renderer is None:
            self.renderer = mujoco.Renderer(model, height=480, width=640)
            self.cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.cam)
            self.cam.lookat[:] = np.array([0.22, 0.0, 0.14])
            self.cam.distance = 0.85
            self.cam.azimuth = 125
            self.cam.elevation = -18
        self.n += 1
        if self.n % self.every != 0:
            return
        self.renderer.update_scene(data, camera=self.cam)
        self.frames.append(np.asarray(self.renderer.render()).copy())

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()


class _SqueezeGrabber(_Grabber):
    def __call__(self, model: Any, data: Any) -> None:
        import mujoco

        if self.renderer is None:
            self.renderer = mujoco.Renderer(model, height=480, width=640)
            self.cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.cam)
            self.cam.lookat[:] = np.array([0.02, 0.0, 0.06])
            self.cam.distance = 0.28
            self.cam.azimuth = 90
            self.cam.elevation = -8
        self.n += 1
        if self.n % self.every != 0:
            return
        self.renderer.update_scene(data, camera=self.cam)
        self.frames.append(np.asarray(self.renderer.render()).copy())


def _mujoco_pair(cfg: dict[str, Any]) -> tuple[list[np.ndarray], list[np.ndarray], str]:
    import mujoco  # noqa: F401

    rng = np.random.default_rng(0)
    lift = _Grabber(every=25)
    scene.measure_mass(cfg["scene"], 0.208, rng, hook=lift)
    lift.close()
    squeeze = _SqueezeGrabber(every=8)
    scene.measure_stiffness(cfg["scene"], cfg["stiffness"], 630.0, rng, hook=squeeze)
    squeeze.close()
    if not lift.frames or not squeeze.frames:
        raise RuntimeError("renderer produced no frames")
    return lift.frames, squeeze.frames, "mujoco-osmesa"


def _schematic_frames(cfg: dict[str, Any]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Used only when OSMesa cannot start. Two cartoons, not a physics render."""
    _style()
    left_frames: list[np.ndarray] = []
    right_frames: list[np.ndarray] = []
    heights = np.linspace(0.02, 0.17, 16)
    gaps = np.linspace(0.03, 0.008, 16)
    for height, gap in zip(heights, gaps, strict=True):
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        ax.set_xlim(-0.05, 0.55)
        ax.set_ylim(-0.05, 0.40)
        ax.set_aspect("equal")
        ax.add_patch(FancyBboxPatch((-0.02, -0.04), 0.55, 0.04, boxstyle="square,pad=0", facecolor="#a07c58"))
        ax.plot([0, 0.18, 0.32], [0.10, 0.22, height], color="#333333", linewidth=6, solid_capstyle="round")
        ax.add_patch(Circle((0.32, height), 0.018, color="#d14b3a"))
        ax.set_axis_off()
        ax.set_title("Lift 15 cm", fontweight="bold")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        left_frames.append(buf)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        ax.set_xlim(-0.02, 0.12)
        ax.set_ylim(-0.02, 0.10)
        ax.set_aspect("equal")
        ax.add_patch(FancyBboxPatch((0.0, 0.03), 0.02, 0.04, boxstyle="square,pad=0", facecolor="#3e9e62"))
        ax.add_patch(FancyBboxPatch((0.02 + gap, 0.03), 0.02, 0.04, boxstyle="square,pad=0", facecolor="#3366b8"))
        ax.set_axis_off()
        ax.set_title("Squeeze, level 2 spring", fontweight="bold")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        right_frames.append(buf)
        plt.close(fig)
    return left_frames, right_frames


def _pad(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    if not frames:
        raise RuntimeError("no frames")
    if len(frames) >= n:
        return frames[:n]
    return frames + [frames[-1]] * (n - len(frames))


def _write(cfg: dict[str, Any], left: list[np.ndarray], right: list[np.ndarray], backend: str) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    # The squeeze finishes before the 15 cm lift. Hold its last frame so the
    # lift, including the can leaving the table, stays in the clip.
    n = max(len(left), len(right))
    still_i = min(len(left) - 1, max(0, int(0.82 * len(left))))
    left = _pad(left, n)
    right = _pad(right, n)
    footer = _status_line()
    still = ART / "physcap-mass-stiffness.png"
    still_tmp = still.with_name(".physcap-mass-stiffness.png.tmp.png")
    _compose(left[still_i], right[still_i], "get_mass  ·  15 cm lift", "get_stiffness  ·  level 2", footer, still_tmp)
    atomic.replace_file(still_tmp, still)
    plot = EXP / "plot.png"
    plot_img = plt.imread(plot) if plot.exists() else None
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        index = 0
        for i in range(n):
            frame_path = folder / f"frame_{index:03d}.png"
            _compose(
                left[i],
                right[i],
                "get_mass  ·  15 cm lift",
                "get_stiffness  ·  level 2",
                footer,
                frame_path,
            )
            index += 1
        if plot_img is not None:
            for _ in range(18):
                fig, ax = plt.subplots(figsize=(12.80, 7.20), dpi=100)
                ax.imshow(plot_img)
                ax.set_axis_off()
                ax.set_title("Pre-registered probes", fontweight="bold")
                fig.tight_layout()
                fig.savefig(folder / f"frame_{index:03d}.png", dpi=100)
                plt.close(fig)
                index += 1
        wrote_video = _ffmpeg(folder, ART / "physcap-mass-stiffness.mp4", fps=8)
    if not wrote_video:
        backend = f"{backend}; mp4 skipped, ffmpeg not found"
    atomic.write_text(ART / "media_backend.txt", backend + "\n")


def main() -> None:
    with (EXP / "config.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    try:
        left, right, backend = _mujoco_pair(cfg)
    except Exception as exc:
        left, right = _schematic_frames(cfg)
        backend = f"matplotlib-fallback: {exc}"
    _write(cfg, left, right, backend)
    print(backend)


if __name__ == "__main__":
    main()
