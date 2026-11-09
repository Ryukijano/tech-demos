"""Offscreen still and a short side-by-side video of one registered shot.

Uses MuJoCo's EGL renderer when it loads. If that fails, writes a matplotlib
trajectory animation and records the fallback in ``artifacts/media_backend.txt``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# MuJoCo chooses its GL backend from MUJOCO_GL at import time. OSMesa is the
# software path. If that import fails, the matplotlib trajectory fallback runs.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle, Rectangle

import scene

try:
    import mujoco
except Exception as gl_import_error:
    mujoco = None
    GL_IMPORT_ERROR: Exception | None = gl_import_error
else:
    GL_IMPORT_ERROR = None

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_tunnel"
ART = APP / "artifacts"


def _bold_legend(ax: plt.Axes, **kwargs: Any) -> Any:
    legend = ax.legend(**kwargs)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    return legend


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


def _media_spec(cfg: dict[str, Any]) -> dict[str, float]:
    media = cfg["media_shot"]
    return {
        "radius": float(media["radius_m"]),
        "speed": float(media["speed_m_s"]),
        "dt": float(media["timestep_s"]),
        "x0": float(cfg["scene"]["x0"]),
        "half": float(cfg["scene"]["half_thickness_m"]),
    }


def _trace(cfg: dict[str, Any], speculative: bool, spec: dict[str, float]) -> list[float]:
    row = scene.rollout_shot(
        cfg["scene"],
        {
            "radius": spec["radius"],
            "speed": spec["speed"],
            "dt": spec["dt"],
            "speculative": speculative,
            "record_x": True,
        },
    )
    return [float(x) for x in row["xs"]]


def _mujoco_frames(cfg: dict[str, Any], speculative: bool, spec: dict[str, float]) -> tuple[list[np.ndarray], list[float]]:
    scene_cfg = cfg["scene"]
    radius = spec["radius"]
    mass = scene.sphere_mass(radius, float(scene_cfg["density_kg_m3"]))
    xml = scene.make_xml(
        dt=spec["dt"],
        speculative=speculative,
        half_thickness=spec["half"],
        half_yz=float(scene_cfg["half_yz_m"]),
        radii=[radius],
        masses=[mass],
        x0=spec["x0"],
        y_positions=[0.0],
        gravity=(0.0, 0.0, 0.0),
        integrator=str(scene_cfg["integrator"]),
        solver=str(scene_cfg["solver"]),
        iterations=int(scene_cfg["iterations"]),
        ls_iterations=int(scene_cfg["ls_iterations"]),
        ccd_iterations=int(scene_cfg["ccd_iterations"]),
        solref=str(scene_cfg["solref"]),
        solimp=str(scene_cfg["solimp"]),
        condim=int(scene_cfg["condim"]),
        visual=True,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qvel[0] = spec["speed"]
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    # Wide enough to show the launch point (x = -0.08 m) and the first
    # steps after a 16 ms jump, before the missed sphere leaves the view.
    cam.lookat[:] = np.array([-0.03, 0.0, 0.0])
    cam.distance = 0.28
    cam.azimuth = 90
    cam.elevation = -10
    frames: list[np.ndarray] = []
    xs: list[float] = []

    def grab() -> None:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        frames.append(np.asarray(renderer.render()).copy())
        xs.append(float(data.qpos[0]))

    grab()
    steps = min(
        6,
        scene.n_steps_for(
            spec["speed"],
            spec["dt"],
            float(scene_cfg["travel_m"]),
            int(scene_cfg["settle_steps"]),
            None,
        ),
    )
    for _ in range(steps):
        mujoco.mj_step(model, data)
        grab()
    renderer.close()
    return frames, xs


def _compose(left: np.ndarray, right: np.ndarray, left_title: str, right_title: str, footer: str, path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12.80, 7.20), dpi=100)
    for ax, image, title in ((axes[0], left, left_title), (axes[1], right, right_title)):
        ax.imshow(image)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
        ax.axis("off")
    fig.suptitle(footer, fontsize=12, fontweight="bold", y=0.06)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _ffmpeg(frame_dir: Path, dest: Path, fps: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found")
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
            str(dest),
        ],
        check=True,
    )


def _status_line() -> str:
    status_path = EXP / "status.json"
    if not status_path.exists():
        return "MuJoCo 4.0.0 speculative contacts"
    status = json.loads(status_path.read_text())
    off = status["off_rate_mean"]
    on = status["on_rate_mean"]
    return (
        f"Status {status['status']}  ·  16 ms tunneling  "
        f"off {float(off['0.016']):.2f}   on {float(on['0.016']):.2f}"
    )


def _write_mujoco_media(cfg: dict[str, Any], spec: dict[str, float]) -> str:
    off_frames, off_xs = _mujoco_frames(cfg, False, spec)
    on_frames, on_xs = _mujoco_frames(cfg, True, spec)
    n = min(len(off_frames), len(on_frames))
    # The interesting hold is the first frame where the disabled sphere has crossed.
    cross = next((i for i, x in enumerate(off_xs[:n]) if x - spec["radius"] > spec["half"]), n // 2)
    still = ART / "mujoco4-speculative-contacts.png"
    _compose(
        off_frames[cross],
        on_frames[cross],
        "Speculative off  ·  16 ms",
        "Speculative on  ·  16 ms",
        _status_line(),
        still,
    )
    plot = EXP / "plot.png"
    plot_img = plt.imread(plot) if plot.exists() else None
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        index = 0
        hold = 6
        for i in range(n):
            frame_path = folder / f"frame_{index:03d}.png"
            _compose(
                off_frames[i],
                on_frames[i],
                "Speculative off  ·  16 ms step",
                "Speculative on  ·  16 ms step",
                f"step {i}   x off {off_xs[i]:+.3f} m    x on {on_xs[i]:+.3f} m",
                frame_path,
            )
            repeats = hold + (10 if i == cross else 0)
            for _ in range(repeats - 1):
                index += 1
                shutil.copy(frame_path, folder / f"frame_{index:03d}.png")
            index += 1
        if plot_img is not None:
            for _ in range(16):
                fig, ax = plt.subplots(figsize=(12.80, 7.20), dpi=100)
                ax.imshow(plot_img)
                ax.axis("off")
                ax.set_title("Tunneling rate vs timestep", fontweight="bold")
                fig.tight_layout()
                fig.savefig(folder / f"frame_{index:03d}.png", dpi=100)
                plt.close(fig)
                index += 1
        _ffmpeg(folder, ART / "mujoco4-speculative-contacts.mp4", fps=8)
    return "mujoco-osmesa"


def _write_matplotlib_media(cfg: dict[str, Any], spec: dict[str, float]) -> str:
    """Trajectory cartoon used only if the EGL renderer cannot start."""
    _style()
    off_x = _trace(cfg, False, spec)
    on_x = _trace(cfg, True, spec)
    n = min(len(off_x), len(on_x))
    radius = spec["radius"]
    half = spec["half"]
    cross = next((i for i in range(n) if off_x[i] - radius > half), n // 2)
    still = ART / "mujoco4-speculative-contacts.png"

    def draw(path: Path, i: int) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12.80, 7.20), dpi=100, sharey=True)
        for ax, xs, title, color in (
            (axes[0], off_x, "Speculative off  ·  16 ms", COLOR_OFF),
            (axes[1], on_x, "Speculative on  ·  16 ms", COLOR_ON),
        ):
            ax.add_patch(Rectangle((-half, -0.05), 2 * half, 0.10, color="#4C78A8", zorder=1))
            ax.add_patch(Circle((xs[i], 0.0), radius, color=color, zorder=2))
            ax.plot(xs[: i + 1], np.zeros(i + 1), color=color, lw=1.5)
            ax.set_xlim(-0.10, 0.12)
            ax.set_ylim(-0.06, 0.06)
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_title(title)
            ax.axvline(0.0, color="#999999", lw=0.6)
        fig.suptitle(_status_line() + "  ·  matplotlib fallback (no EGL frame)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)

    COLOR_OFF = "#D55E00"
    COLOR_ON = "#0072B2"
    draw(still, cross)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        index = 0
        for i in range(n):
            frame_path = folder / f"frame_{index:03d}.png"
            draw(frame_path, i)
            for _ in range(4):
                index += 1
                shutil.copy(frame_path, folder / f"frame_{index:03d}.png")
            index += 1
        _ffmpeg(folder, ART / "mujoco4-speculative-contacts.mp4", fps=8)
    return "matplotlib-fallback"


def main() -> None:
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    spec = _media_spec(cfg)
    ART.mkdir(parents=True, exist_ok=True)
    backend = "mujoco-osmesa"
    if mujoco is None:
        note = f"{type(GL_IMPORT_ERROR).__name__}: {GL_IMPORT_ERROR}"
        print("MuJoCo GL import failed, using matplotlib trajectories:", note)
        backend = _write_matplotlib_media(cfg, spec)
        (ART / "media_backend.txt").write_text(backend + "\n" + note + "\n")
        print("media", backend)
        return
    try:
        backend = _write_mujoco_media(cfg, spec)
    except Exception as exc:
        note = f"{type(exc).__name__}: {exc}"
        print("OSMesa render failed, using matplotlib trajectories:", note)
        backend = _write_matplotlib_media(cfg, spec)
        (ART / "media_backend.txt").write_text(backend + "\n" + note + "\n")
    else:
        (ART / "media_backend.txt").write_text(backend + "\n")
    print("media", backend)


if __name__ == "__main__":
    main()
