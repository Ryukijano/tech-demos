"""Thin-wall sphere scene and one rollout under MuJoCo 4.0.0.

Collision is solved in ``mj_step1`` at the pre-integration pose. ``mj_step2``
then updates ``qpos``. Contacts are read in ``mj_step1``. Positions used for
the tunneling test are read after ``mj_step2``.
"""

from __future__ import annotations

import os
import time
from typing import Any

import mujoco
import numpy as np

SPECULATIVE_BIT = int(mujoco.mjtEnableBit.mjENBL_SPECULATIVE)
ENERGY_BIT = int(mujoco.mjtEnableBit.mjENBL_ENERGY)


def require_mujoco_4() -> str:
    version = str(mujoco.__version__)
    if version != "4.0.0":
        raise RuntimeError(f"expected mujoco==4.0.0, found {version}")
    if SPECULATIVE_BIT != 128:
        raise RuntimeError(f"expected mjENBL_SPECULATIVE == 128, found {SPECULATIVE_BIT}")
    return version


def kinematic_skips(
    x0: float,
    radius: float,
    speed: float,
    dt: float,
    half_thickness: float,
    clear_eps: float,
) -> bool:
    """True when discrete samples jump the wall with no overlapping center."""
    wall_near = -half_thickness
    wall_far = half_thickness
    overlapped = False
    span = wall_far + radius + 0.05 - x0
    n_steps = int(np.ceil(span / (speed * dt))) + 2
    for k in range(n_steps + 1):
        x = x0 + speed * k * dt
        overlap = (x + radius >= wall_near) and (x - radius <= wall_far)
        past = (x - radius) > wall_far + clear_eps
        if overlap:
            overlapped = True
        if past:
            return not overlapped
    return False


def _fmt(value: float) -> str:
    return format(float(value), ".8g")


def make_xml(
    *,
    dt: float,
    speculative: bool,
    half_thickness: float,
    half_yz: float,
    radii: list[float],
    masses: list[float],
    x0: float,
    y_positions: list[float],
    gravity: tuple[float, float, float],
    integrator: str,
    solver: str,
    iterations: int,
    ls_iterations: int,
    ccd_iterations: int,
    solref: str,
    solimp: str,
    condim: int,
    visual: bool,
) -> str:
    flag = "enable" if speculative else "disable"
    gx, gy, gz = gravity
    geoms: list[str] = []
    if visual:
        geoms.append(
            '<geom name="floor" type="plane" pos="0 0 -0.03" size="0.4 0.4 0.01" '
            'rgba="0.16 0.18 0.22 1" contype="0" conaffinity="0"/>'
        )
    geoms.append(
        f'<geom name="wall" type="box" pos="0 0 0" '
        f'size="{_fmt(half_thickness)} {_fmt(half_yz)} {_fmt(half_yz)}" '
        f'rgba="0.30 0.48 0.72 1" solref="{solref}" solimp="{solimp}" '
        f'friction="0 0 0" condim="{int(condim)}"/>'
    )
    bodies: list[str] = []
    for index, (radius, mass, y) in enumerate(zip(radii, masses, y_positions)):
        bodies.append(
            f'<body name="ball{index}" pos="{_fmt(x0)} {_fmt(y)} 0">'
            f"<freejoint/>"
            f'<geom name="ball{index}" type="sphere" size="{_fmt(radius)}" '
            f'mass="{_fmt(mass)}" rgba="0.90 0.35 0.18 1" '
            f'solref="{solref}" solimp="{solimp}" friction="0 0 0" condim="{int(condim)}"/>'
            f"</body>"
        )
    visual_xml = ""
    if visual:
        visual_xml = """
  <visual>
    <headlight ambient="0.45 0.45 0.48" diffuse="0.75 0.75 0.75" specular="0.15 0.15 0.15"/>
    <rgba haze="0.12 0.14 0.18 1"/>
  </visual>
  <worldbody>
    <light pos="0.2 -0.3 0.4" dir="-0.2 0.4 -1" diffuse="0.7 0.7 0.72"/>
"""
    else:
        visual_xml = "\n  <worldbody>\n"
    return f"""<mujoco model="speculative-tunnel">
  <option timestep="{_fmt(dt)}" gravity="{_fmt(gx)} {_fmt(gy)} {_fmt(gz)}"
          integrator="{integrator}" solver="{solver}"
          iterations="{int(iterations)}" ls_iterations="{int(ls_iterations)}"
          ccd_iterations="{int(ccd_iterations)}">
    <flag speculative="{flag}" energy="enable"/>
  </option>
  {visual_xml}
    {''.join(geoms)}
    {''.join(bodies)}
  </worldbody>
</mujoco>
"""


def sphere_mass(radius: float, density: float) -> float:
    return float(density * (4.0 / 3.0) * np.pi * radius**3)


def _compile(xml: str, speculative: bool) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_string(xml)
    enabled = bool(int(model.opt.enableflags) & SPECULATIVE_BIT)
    if enabled != speculative:
        raise RuntimeError(
            f"speculative flag mismatch: xml speculative={speculative} "
            f"enableflags={int(model.opt.enableflags)} bit={SPECULATIVE_BIT}"
        )
    if not (int(model.opt.enableflags) & ENERGY_BIT):
        raise RuntimeError("energy flag was not set")
    return model


def n_steps_for(speed: float, dt: float, travel_m: float, settle_steps: int, n_steps: int | None) -> int:
    if n_steps is not None:
        return int(n_steps)
    return int(np.ceil(float(travel_m) / (float(speed) * float(dt)))) + int(settle_steps)


def rollout_shot(scene: dict[str, Any], shot: dict[str, Any]) -> dict[str, Any]:
    """Roll one sphere. ``shot`` supplies radius, speed, and optional overrides."""
    radius = float(shot["radius"])
    speed = float(shot["speed"])
    dt = float(shot["dt"])
    speculative = bool(shot["speculative"])
    half = float(shot.get("half_thickness", scene["half_thickness_m"]))
    x0 = float(shot.get("x0", scene["x0"]))
    gravity = tuple(float(v) for v in shot.get("gravity", scene["gravity"]))
    density = float(scene["density_kg_m3"])
    mass = sphere_mass(radius, density)
    clear_eps = float(scene["clear_eps_m"])
    steps = n_steps_for(
        speed,
        dt,
        float(scene["travel_m"]),
        int(scene["settle_steps"]),
        shot.get("n_steps"),
    )
    xml = make_xml(
        dt=dt,
        speculative=speculative,
        half_thickness=half,
        half_yz=float(scene["half_yz_m"]),
        radii=[radius],
        masses=[mass],
        x0=x0,
        y_positions=[0.0],
        gravity=gravity,  # type: ignore[arg-type]
        integrator=str(scene["integrator"]),
        solver=str(scene["solver"]),
        iterations=int(scene["iterations"]),
        ls_iterations=int(scene["ls_iterations"]),
        ccd_iterations=int(scene["ccd_iterations"]),
        solref=str(scene["solref"]),
        solimp=str(scene["solimp"]),
        condim=int(scene["condim"]),
        visual=False,
    )
    model = _compile(xml, speculative)
    data = mujoco.MjData(model)
    data.qvel[0] = speed
    mujoco.mj_forward(model, data)
    ke0 = float(data.energy[1])
    max_x = float(data.qpos[0])
    saw_contact = False
    max_pen = 0.0
    diverged = False
    x_after_first = None
    xs: list[float] = [float(data.qpos[0])]
    t0 = time.perf_counter()
    for k in range(steps):
        mujoco.mj_step1(model, data)
        if int(data.ncon) > 0:
            saw_contact = True
            dist = float(np.min(data.contact.dist[: int(data.ncon)]))
            if np.isfinite(dist):
                max_pen = max(max_pen, max(0.0, -dist))
        mujoco.mj_step2(model, data)
        x = float(data.qpos[0])
        if k == 0:
            x_after_first = x
        if not np.isfinite(x) or not np.isfinite(float(data.qvel[0])):
            diverged = True
            max_x = float("inf")
            break
        max_x = max(max_x, x)
        if shot.get("record_x"):
            xs.append(x)
    wall_s = time.perf_counter() - t0
    if not diverged:
        mujoco.mj_forward(model, data)
        ke1 = float(data.energy[1])
        vx = float(data.qvel[0])
    else:
        ke1 = float("nan")
        vx = float("nan")
    tunneled = bool(diverged or (max_x - radius) > (half + clear_eps))
    ke_drift = float((ke1 - ke0) / ke0) if ke0 > 0.0 and np.isfinite(ke1) else float("nan")
    out: dict[str, Any] = {
        "radius": radius,
        "speed": speed,
        "dt": dt,
        "speculative": speculative,
        "half_thickness": half,
        "tunneled": tunneled,
        "contacted": saw_contact,
        "max_penetration_m": max_pen,
        "max_x": None if not np.isfinite(max_x) else max_x,
        "x_after_first": x_after_first,
        "ke_init": ke0,
        "ke_final": None if not np.isfinite(ke1) else ke1,
        "ke_rel_drift": None if not np.isfinite(ke_drift) else ke_drift,
        "vx_final": None if not np.isfinite(vx) else vx,
        "n_steps": steps,
        "diverged": diverged,
        "wall_s": wall_s,
        "enableflags": int(model.opt.enableflags),
    }
    if shot.get("record_x"):
        out["xs"] = xs
    return out


def summarize_shots(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    tun = np.array([1.0 if row["tunneled"] else 0.0 for row in rows], dtype=np.float64)
    contact = np.array([1.0 if row["contacted"] else 0.0 for row in rows], dtype=np.float64)
    pen = np.array([float(row["max_penetration_m"]) for row in rows], dtype=np.float64)
    drift = np.array(
        [np.nan if row["ke_rel_drift"] is None else float(row["ke_rel_drift"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "n_shots": n,
        "tunnel_rate": float(tun.mean()) if n else float("nan"),
        "contact_rate": float(contact.mean()) if n else float("nan"),
        "mean_max_penetration_m": float(pen.mean()) if n else float("nan"),
        "mean_ke_rel_drift": float(np.nanmean(drift)) if n else float("nan"),
        "n_diverged": int(sum(1 for row in rows if row["diverged"])),
        "wall_s": float(sum(row["wall_s"] for row in rows)),
    }


def run_group(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker entry. One (mode, timestep, shot list) group."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    cfg_scene = payload["scene"]
    rows = [rollout_shot(cfg_scene, shot) for shot in payload["shots"]]
    for row in rows:
        row.pop("xs", None)
    summary = summarize_shots(rows)
    speeds = [float(shot["speed"]) for shot in payload["shots"]]
    if speeds and all(speed > 0.0 for speed in speeds):
        summary["kinematic_skip_rate"] = float(
            np.mean(
                [
                    kinematic_skips(
                        float(shot.get("x0", cfg_scene["x0"])),
                        float(shot["radius"]),
                        float(shot["speed"]),
                        float(shot["dt"]),
                        float(shot.get("half_thickness", cfg_scene["half_thickness_m"])),
                        float(cfg_scene["clear_eps_m"]),
                    )
                    for shot in payload["shots"]
                ]
            )
        )
    else:
        summary["kinematic_skip_rate"] = None
    return {
        "label": payload["label"],
        "seed": payload.get("seed"),
        "mode": payload["mode"],
        "dt": payload["dt"],
        "family": payload.get("family", "primary"),
        "parameter": payload.get("parameter"),
        "summary": summary,
        "shots": rows,
    }


def bench_wallclock(scene: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Time mj_step for 1 or more fast spheres. Parent process, after the pool."""
    n_spheres = int(spec["n_spheres"])
    dt = float(spec["dt"])
    speculative = bool(spec["speculative"])
    radius = float(spec["radius"])
    speed = float(spec["speed"])
    n_steps = int(spec["n_steps"])
    warmup = int(spec["warmup_steps"])
    density = float(scene["density_kg_m3"])
    mass = sphere_mass(radius, density)
    ys = [float(y) for y in np.linspace(-0.02, 0.02, n_spheres)]
    xml = make_xml(
        dt=dt,
        speculative=speculative,
        half_thickness=float(scene["half_thickness_m"]),
        half_yz=float(scene["half_yz_m"]),
        radii=[radius] * n_spheres,
        masses=[mass] * n_spheres,
        x0=float(scene["x0"]),
        y_positions=ys,
        gravity=(0.0, 0.0, 0.0),
        integrator=str(scene["integrator"]),
        solver=str(scene["solver"]),
        iterations=int(scene["iterations"]),
        ls_iterations=int(scene["ls_iterations"]),
        ccd_iterations=int(scene["ccd_iterations"]),
        solref=str(scene["solref"]),
        solimp=str(scene["solimp"]),
        condim=int(scene["condim"]),
        visual=False,
    )
    model = _compile(xml, speculative)
    data = mujoco.MjData(model)
    for index in range(n_spheres):
        data.qvel[6 * index] = speed
    for _ in range(warmup):
        mujoco.mj_step(model, data)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    elapsed = time.perf_counter() - t0
    return {
        "n_spheres": n_spheres,
        "dt": dt,
        "speculative": speculative,
        "n_steps": n_steps,
        "warmup_steps": warmup,
        "seconds": elapsed,
        "us_per_step": elapsed / n_steps * 1e6,
        "enableflags": int(model.opt.enableflags),
    }
