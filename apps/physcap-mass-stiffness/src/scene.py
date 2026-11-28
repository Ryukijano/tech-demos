"""Planar arm and jaw scenes for the clean-room PhysCaP probes.

MuJoCo is imported after MUJOCO_GL is set so a later offscreen render and
this physics path share one backend. The XML is original. It is not taken
from physcap/physcap.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np

StepHook = Callable[[mujoco.MjModel, mujoco.MjData], None]


def _f(scene: dict[str, Any], key: str) -> float:
    return float(scene[key])


def ik(x: float, z: float, l1: float, l2: float, z_shoulder: float) -> tuple[float, float]:
    """Elbow-up planar IK. Joint axis is (0, -1, 0), so +angle lifts toward +z."""
    dx = x
    dz = z - z_shoulder
    r2 = dx * dx + dz * dz
    c2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    c2 = max(-1.0, min(1.0, c2))
    s2 = -math.sqrt(max(0.0, 1.0 - c2 * c2))
    theta2 = math.atan2(s2, c2)
    k1 = l1 + l2 * c2
    k2 = l2 * s2
    theta1 = math.atan2(dz, dx) - math.atan2(k2, k1)
    return theta1, theta2


def arm_xml(scene: dict[str, Any], mass: float, friction: float, arm_force: float | None = None) -> str:
    g = _f(scene, "gravity_m_s2")
    dt = _f(scene, "timestep_s")
    l1 = _f(scene, "l1_m")
    l2 = _f(scene, "l2_m")
    z_sh = _f(scene, "shoulder_z_m")
    finger_force = _f(scene, "finger_forcerange_n")
    finger_kp = _f(scene, "finger_kp")
    arm_force = _f(scene, "arm_forcerange_nm") if arm_force is None else arm_force
    x = _f(scene, "object_x_m")
    z = _f(scene, "object_z_m")
    return f"""
<mujoco model="physcap-arm">
  <option timestep="{dt}" gravity="0 0 -{g}" integrator="{scene["integrator"]}" solver="Newton" iterations="40"/>
  <compiler angle="radian"/>
  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.7 0.7 0.7"/>
    <rgba haze="0.95 0.95 0.96 1"/>
  </visual>
  <worldbody>
    <light pos="0.35 -0.45 0.7" dir="-0.1 0.35 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 -0.04" rgba="0.93 0.93 0.94 1"/>
    <geom name="table" type="box" size="0.48 0.28 0.02" pos="0.30 0 -0.02" rgba="0.62 0.50 0.38 1"/>
    <body name="shoulder" pos="0 0 {z_sh}">
      <joint name="shoulder" type="hinge" axis="0 -1 0" range="-1.5 1.9" damping="2.0" armature="0.02"/>
      <geom type="capsule" fromto="0 0 0 {l1} 0 0" size="0.018" mass="0.45" contype="0" conaffinity="0" rgba="0.25 0.28 0.32 1"/>
      <body name="elbow" pos="{l1} 0 0">
        <joint name="elbow" type="hinge" axis="0 -1 0" range="-2.7 0.3" damping="1.2" armature="0.01"/>
        <geom type="capsule" fromto="0 0 0 {l2} 0 0" size="0.014" mass="0.28" contype="0" conaffinity="0" rgba="0.35 0.38 0.42 1"/>
        <body name="wrist" pos="{l2} 0 0">
          <joint name="wrist" type="hinge" axis="0 -1 0" range="-1.8 1.8" damping="0.4" armature="0.005"/>
          <geom type="box" size="0.016 0.012 0.012" mass="0.06" contype="0" conaffinity="0" rgba="0.45 0.48 0.52 1"/>
          <site name="grasp" pos="0 0 0" size="0.006"/>
          <body name="finger_l" pos="0 0.045 0">
            <joint name="finger_l" type="slide" axis="0 -1 0" range="0 0.04" damping="0.5"/>
            <geom name="pad_l" type="box" size="0.012 0.004 0.016" mass="0.02" rgba="0.20 0.38 0.72 1"/>
          </body>
          <body name="finger_r" pos="0 -0.045 0">
            <joint name="finger_r" type="slide" axis="0 1 0" range="0 0.04" damping="0.5"/>
            <geom name="pad_r" type="box" size="0.012 0.004 0.016" mass="0.02" rgba="0.20 0.38 0.72 1"/>
          </body>
        </body>
      </body>
    </body>
    <body name="object" pos="{x} 0 {z}">
      <freejoint name="object"/>
      <geom name="object" type="box" size="0.022 0.018 0.02" mass="{mass}" rgba="0.82 0.22 0.18 1"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="pad_l" geom2="object" condim="4" friction="{friction} {friction} 0.001 0.0001 0.0001"/>
    <pair geom1="pad_r" geom2="object" condim="4" friction="{friction} {friction} 0.001 0.0001 0.0001"/>
    <pair geom1="table" geom2="object" condim="3" friction="1 1 0.001 0.0001 0.0001"/>
  </contact>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="180" kv="16" ctrlrange="-1.5 1.9" forcerange="-{arm_force} {arm_force}"/>
    <position name="a_elbow" joint="elbow" kp="120" kv="10" ctrlrange="-2.7 0.3" forcerange="-{arm_force} {arm_force}"/>
    <position name="a_wrist" joint="wrist" kp="30" kv="3" ctrlrange="-1.8 1.8" forcerange="-4 4"/>
    <position name="a_fl" joint="finger_l" kp="{finger_kp}" kv="12" ctrlrange="0 0.04" forcerange="-{finger_force} {finger_force}"/>
    <position name="a_fr" joint="finger_r" kp="{finger_kp}" kv="12" ctrlrange="0 0.04" forcerange="-{finger_force} {finger_force}"/>
  </actuator>
</mujoco>
"""


def squeeze_xml(scene: dict[str, Any], k: float, frictionloss: float) -> str:
    g = _f(scene, "gravity_m_s2")
    dt = _f(scene, "timestep_s")
    return f"""
<mujoco model="physcap-squeeze">
  <option timestep="{dt}" gravity="0 0 -{g}" integrator="{scene["integrator"]}"/>
  <visual>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.7 0.7 0.7"/>
  </visual>
  <worldbody>
    <light pos="0.1 -0.2 0.25" dir="0 0.4 -1"/>
    <geom name="floor" type="plane" size="1 1 0.1" pos="0 0 0" rgba="0.94 0.94 0.95 1"/>
    <body name="mount" pos="0 0 0.06">
      <geom type="box" size="0.012 0.03 0.03" pos="-0.02 0 0" rgba="0.45 0.45 0.48 1" contype="0" conaffinity="0"/>
      <body name="pad" pos="0 0 0">
        <joint name="pad" type="slide" axis="1 0 0" stiffness="{k}" damping="2.0" springref="0" range="-0.05 0"/>
        <geom name="pad" type="box" size="0.008 0.02 0.02" mass="0.02" rgba="0.25 0.65 0.40 1"/>
      </body>
    </body>
    <body name="jaw" pos="0.04 0 0.06">
      <joint name="jaw" type="slide" axis="-1 0 0" range="0 0.08" damping="0.8" frictionloss="{frictionloss}"/>
      <geom name="jaw" type="box" size="0.008 0.02 0.02" mass="0.05" rgba="0.20 0.38 0.72 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="a_jaw" joint="jaw" kp="2500" kv="40" ctrlrange="0 0.08" forcerange="-40 40"/>
  </actuator>
</mujoco>
"""


def _aid(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def _step_n(model: mujoco.MjModel, data: mujoco.MjData, n: int, hook: StepHook | None) -> None:
    for _ in range(n):
        mujoco.mj_step(model, data)
        if hook is not None:
            hook(model, data)


def _settle(model: mujoco.MjModel, data: mujoco.MjData, seconds: float, hook: StepHook | None) -> None:
    dt = float(model.opt.timestep)
    steps = max(1, int(seconds / dt))
    for i in range(steps):
        mujoco.mj_step(model, data)
        if hook is not None:
            hook(model, data)
        if i > steps // 3 and float(np.max(np.abs(data.qvel))) < 2e-3:
            break


def _go(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    th1: float,
    th2: float,
    fingers: float,
    seconds: float,
    hook: StepHook | None,
) -> None:
    ids = [_aid(model, name) for name in ("a_shoulder", "a_elbow", "a_wrist", "a_fl", "a_fr")]
    start = np.array([data.ctrl[i] for i in ids], dtype=np.float64)
    goal = np.array([th1, th2, -(th1 + th2), fingers, fingers], dtype=np.float64)
    n = max(1, int(seconds / float(model.opt.timestep)))
    for i in range(n):
        cmd = (1.0 - (i + 1) / n) * start + ((i + 1) / n) * goal
        for actuator, value in zip(ids, cmd, strict=True):
            data.ctrl[actuator] = value
        mujoco.mj_step(model, data)
        if hook is not None:
            hook(model, data)


def _arm_tau(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return np.array(
        [float(data.actuator_force[_aid(model, name)]) for name in ("a_shoulder", "a_elbow", "a_wrist")],
        dtype=np.float64,
    )


def _jac_z(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp")
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site)
    return jacp[2, :3].copy()


def _mean_tau(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_samples: int,
    noise_std: float,
    rng: np.random.Generator,
    hook: StepHook | None,
) -> np.ndarray:
    acc = np.zeros(3, dtype=np.float64)
    gap = max(1, int(0.02 / float(model.opt.timestep)))
    for _ in range(n_samples):
        _step_n(model, data, gap, hook)
        sample = _arm_tau(model, data)
        if noise_std > 0.0:
            sample = sample + rng.normal(0.0, noise_std, size=3)
        acc += sample
    return acc / n_samples


def measure_mass(
    scene: dict[str, Any],
    mass: float,
    rng: np.random.Generator,
    friction: float | None = None,
    arm_force: float | None = None,
    x_offset: float = 0.0,
    hook: StepHook | None = None,
) -> dict[str, Any]:
    """15 cm empty-then-loaded lift. Returns -1 kg when the can stays down."""
    local = dict(scene)
    local["object_x_m"] = _f(scene, "object_x_m") + x_offset
    mu = _f(scene, "finger_friction") if friction is None else friction
    model = mujoco.MjModel.from_xml_string(arm_xml(local, mass, mu, arm_force))
    data = mujoco.MjData(model)
    l1 = _f(local, "l1_m")
    l2 = _f(local, "l2_m")
    z_sh = _f(local, "shoulder_z_m")
    x = _f(local, "object_x_m")
    z_obj = _f(local, "object_z_m")
    g = _f(local, "gravity_m_s2")
    close = _f(local, "finger_close")

    th1, th2 = ik(0.18, 0.28, l1, l2, z_sh)
    data.qpos[0] = th1
    data.qpos[1] = th2
    data.qpos[2] = -(th1 + th2)
    data.ctrl[:] = [th1, th2, -(th1 + th2), 0.0, 0.0]
    mujoco.mj_forward(model, data)

    lift_z = z_obj + 0.15
    th1, th2 = ik(x, lift_z, l1, l2, z_sh)
    _go(model, data, th1, th2, 0.0, _f(local, "empty_move_s"), hook)
    _settle(model, data, _f(local, "empty_settle_s"), hook)
    tau_empty = _mean_tau(model, data, int(scene["n_torque_samples"]), _f(scene, "torque_noise_std_nm"), rng, hook)

    th1, th2 = ik(x, z_obj, l1, l2, z_sh)
    _go(model, data, th1, th2, 0.0, _f(local, "descend_s"), hook)
    _settle(model, data, 0.15, hook)
    _go(model, data, th1, th2, close, _f(local, "close_s"), hook)
    _settle(model, data, _f(local, "close_settle_s"), hook)

    th1, th2 = ik(x, lift_z, l1, l2, z_sh)
    _go(model, data, th1, th2, close, _f(local, "lift_s"), hook)
    _settle(model, data, _f(local, "lift_settle_s"), hook)
    tau_loaded = _mean_tau(model, data, int(scene["n_torque_samples"]), _f(scene, "torque_noise_std_nm"), rng, hook)

    obj_z = float(data.joint("object").qpos[2])
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    lifted = finite and (obj_z - z_obj) >= _f(scene, "min_lift_m")
    if not lifted:
        return {
            "m_hat": float(scene["grasp_failure_mass_kg"]),
            "grasped": False,
            "obj_z": obj_z,
            "unstable": not finite,
        }
    jz = _jac_z(model, data)
    denom = float(np.dot(jz, jz))
    if denom < 1e-8:
        return {"m_hat": float(scene["grasp_failure_mass_kg"]), "grasped": False, "obj_z": obj_z, "unstable": True}
    delta = tau_loaded - tau_empty
    m_hat = float(np.dot(jz, delta) / (g * denom))
    return {"m_hat": m_hat, "grasped": True, "obj_z": obj_z, "unstable": False}


def _contact_normal(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    total = 0.0
    for i in range(data.ncon):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        total += abs(float(force[0]))
    return total


def measure_stiffness(
    scene: dict[str, Any],
    stiff: dict[str, Any],
    k: float,
    rng: np.random.Generator,
    frictionloss: float = 0.0,
    hook: StepHook | None = None,
) -> dict[str, Any]:
    """Incremental close, backoff, then an OLS slope. Level 0 is a failed fit."""
    model = mujoco.MjModel.from_xml_string(squeeze_xml(scene, k, frictionloss))
    data = mujoco.MjData(model)
    aid = _aid(model, "a_jaw")
    step = float(stiff["jaw_step_m"])
    noise = float(stiff["force_noise_std_n"])
    target = float(stiff["target_force_n"])
    contact_on = float(stiff["contact_force_n"])
    backoff = float(stiff["backoff_m"])
    confirmed = False
    d0 = 0.0
    xs: list[float] = []
    ys: list[float] = []
    for i in range(1, 150):
        data.ctrl[aid] = min(0.08, step * i)
        _settle(model, data, 0.035, hook)
        disp = -float(data.joint("pad").qpos[0])
        force = _contact_normal(model, data)
        if noise > 0.0:
            force += float(rng.normal(0.0, noise))
        if not confirmed and force > contact_on:
            data.ctrl[aid] = max(0.0, float(data.ctrl[aid]) - backoff)
            _settle(model, data, 0.05, hook)
            force_back = _contact_normal(model, data)
            if noise > 0.0:
                force_back += float(rng.normal(0.0, noise))
            if force_back < 0.7 * force:
                confirmed = True
                d0 = -float(data.joint("pad").qpos[0])
            data.ctrl[aid] = min(0.08, step * i)
            _settle(model, data, 0.03, hook)
            disp = -float(data.joint("pad").qpos[0])
            force = _contact_normal(model, data)
            if noise > 0.0:
                force += float(rng.normal(0.0, noise))
        if confirmed:
            xs.append(disp - d0)
            ys.append(force)
            if force >= target:
                break
    failed = {
        "k_hat": float("nan"),
        "level_ready": False,
        "n_points": len(xs),
        "confirmed": confirmed,
        "span_m": 0.0,
    }
    if len(xs) < int(stiff["min_points"]):
        return failed
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    span = float(x[-1] - x[0])
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom < 1e-12 or span < float(stiff["min_compression_m"]):
        failed["span_m"] = span
        return failed
    slope = float(np.dot(xc, y - y.mean()) / denom)
    if not math.isfinite(slope) or slope <= 0.0:
        failed["span_m"] = span
        return failed
    return {
        "k_hat": slope,
        "level_ready": True,
        "n_points": int(x.size),
        "confirmed": True,
        "span_m": span,
    }
