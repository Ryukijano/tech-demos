"""Pre-registered speculative-contact tunneling sweep. CPU only."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import scene

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_tunnel"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _shots(seed: int, n_shots: int, cfg_scene: dict[str, Any]) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    radii = rng.uniform(float(cfg_scene["radius_min_m"]), float(cfg_scene["radius_max_m"]), n_shots)
    speeds = rng.uniform(float(cfg_scene["speed_min_m_s"]), float(cfg_scene["speed_max_m_s"]), n_shots)
    return [{"radius": float(r), "speed": float(v)} for r, v in zip(radii, speeds)]


def _primary_payloads(cfg: dict[str, Any], seeds: list[int], n_shots: int, timesteps: list[float]) -> list[dict[str, Any]]:
    payloads = []
    scene_cfg = cfg["scene"]
    for seed in seeds:
        base = _shots(seed, n_shots, scene_cfg)
        for dt in timesteps:
            for speculative, mode in ((False, "off"), (True, "on")):
                shots = [
                    {
                        "radius": shot["radius"],
                        "speed": shot["speed"],
                        "dt": dt,
                        "speculative": speculative,
                    }
                    for shot in base
                ]
                payloads.append(
                    {
                        "label": f"primary-seed{seed}-{mode}-dt{dt}",
                        "seed": seed,
                        "mode": mode,
                        "dt": dt,
                        "scene": scene_cfg,
                        "shots": shots,
                    }
                )
    return payloads


def _breakpoint_payloads(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    scene_cfg = cfg["scene"]
    bp = cfg["breakpoints"]
    payloads: list[dict[str, Any]] = []
    huge_v = bp["huge_velocity"]
    for speed in huge_v["speeds_m_s"]:
        for dt in huge_v["timesteps_s"]:
            for speculative, mode in ((False, "off"), (True, "on")):
                payloads.append(
                    {
                        "label": f"huge-v-{speed}-{mode}-dt{dt}",
                        "seed": int(bp["seed"]),
                        "mode": mode,
                        "dt": float(dt),
                        "family": "huge_velocity",
                        "parameter": float(speed),
                        "scene": scene_cfg,
                        "shots": [
                            {
                                "radius": float(huge_v["radius_m"]),
                                "speed": float(speed),
                                "dt": float(dt),
                                "speculative": speculative,
                            }
                        ],
                    }
                )
    huge_dt = bp["huge_timestep"]
    for dt in huge_dt["timesteps_s"]:
        for speculative, mode in ((False, "off"), (True, "on")):
            payloads.append(
                {
                    "label": f"huge-dt-{dt}-{mode}",
                    "seed": int(bp["seed"]),
                    "mode": mode,
                    "dt": float(dt),
                    "family": "huge_timestep",
                    "parameter": float(dt),
                    "scene": scene_cfg,
                    "shots": [
                        {
                            "radius": float(huge_dt["radius_m"]),
                            "speed": float(huge_dt["speed_m_s"]),
                            "dt": float(dt),
                            "speculative": speculative,
                        }
                    ],
                }
            )
    thin = bp["thinner_wall"]
    for half in thin["half_thicknesses_m"]:
        for speculative, mode in ((False, "off"), (True, "on")):
            payloads.append(
                {
                    "label": f"thin-{half}-{mode}",
                    "seed": int(bp["seed"]),
                    "mode": mode,
                    "dt": float(thin["timestep_s"]),
                    "family": "thinner_wall",
                    "parameter": float(half),
                    "scene": scene_cfg,
                    "shots": [
                        {
                            "radius": float(thin["radius_m"]),
                            "speed": float(thin["speed_m_s"]),
                            "dt": float(thin["timestep_s"]),
                            "speculative": speculative,
                            "half_thickness": float(half),
                        }
                    ],
                }
            )
    accel = bp["accel_from_rest"]
    for gravity in accel["gravities_m_s2"]:
        for speculative, mode in ((False, "off"), (True, "on")):
            payloads.append(
                {
                    "label": f"accel-{gravity}-{mode}",
                    "seed": int(bp["seed"]),
                    "mode": mode,
                    "dt": float(accel["timestep_s"]),
                    "family": "accel_from_rest",
                    "parameter": float(gravity),
                    "scene": scene_cfg,
                    "shots": [
                        {
                            "radius": float(accel["radius_m"]),
                            "speed": 0.0,
                            "dt": float(accel["timestep_s"]),
                            "speculative": speculative,
                            "half_thickness": float(accel["half_thickness_m"]),
                            "x0": float(accel["x0"]),
                            "gravity": [float(gravity), 0.0, 0.0],
                            "n_steps": int(accel["n_steps"]),
                        }
                    ],
                }
            )
    return payloads


def _execute(payloads: list[dict[str, Any]], max_workers: int) -> list[dict[str, Any]]:
    if not payloads:
        return []
    workers = max(1, min(int(max_workers), len(payloads), os.cpu_count() or 1))
    if workers == 1:
        return [scene.run_group(payload) for payload in payloads]
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return list(pool.map(scene.run_group, payloads))


def _strip_shot_traces(result: dict[str, Any]) -> dict[str, Any]:
    """Keep per-shot outcomes, drop nothing large (no trajectories in the pool)."""
    return result


def run_smoke(cfg: dict[str, Any]) -> None:
    scene.require_mujoco_4()
    timesteps = [0.002, 0.016]
    payloads = _primary_payloads(cfg, seeds=[0], n_shots=4, timesteps=timesteps)
    results = _execute(payloads, max_workers=int(cfg["runtime"]["max_workers"]))
    out = {
        "kind": "smoke",
        "mujoco": mujoco_meta(),
        "note": "Code-path check only. Not the official result. Thresholds were not fit to this.",
        "results": [_strip_shot_traces(row) for row in results],
    }
    (EXP / "smoke.json").write_text(json.dumps(_jsonable(out), indent=2) + "\n")
    print("smoke written", EXP / "smoke.json")


def mujoco_meta() -> dict[str, Any]:
    return {
        "version": scene.require_mujoco_4(),
        "mjENBL_SPECULATIVE": scene.SPECULATIVE_BIT,
        "mjENBL_ENERGY": scene.ENERGY_BIT,
        "xml_flag": "speculative",
        "xml_values": ["disable", "enable"],
        "default": "disable",
    }


def run_full(cfg: dict[str, Any]) -> None:
    meta = mujoco_meta()
    seeds = [int(s) for s in cfg["seeds"]]
    timesteps = [float(dt) for dt in cfg["scene"]["timesteps_s"]]
    n_shots = int(cfg["scene"]["n_shots"])
    max_workers = int(cfg["runtime"]["max_workers"])
    primary = _execute(_primary_payloads(cfg, seeds, n_shots, timesteps), max_workers)
    breakpoints = _execute(_breakpoint_payloads(cfg), max_workers)
    clock_cfg = cfg["breakpoints"]["wallclock"]
    clocks = []
    for n_spheres in clock_cfg["n_spheres"]:
        for dt in clock_cfg["timesteps_s"]:
            for speculative in (False, True):
                clocks.append(
                    scene.bench_wallclock(
                        cfg["scene"],
                        {
                            "n_spheres": int(n_spheres),
                            "dt": float(dt),
                            "speculative": speculative,
                            "radius": float(clock_cfg["radius_m"]),
                            "speed": float(clock_cfg["speed_m_s"]),
                            "n_steps": int(clock_cfg["n_steps"]),
                            "warmup_steps": int(clock_cfg["warmup_steps"]),
                        },
                    )
                )
    raw = {
        "exp_id": cfg["exp_id"],
        "mujoco": meta,
        "seeds": seeds,
        "timesteps_s": timesteps,
        "n_shots": n_shots,
        "primary": primary,
        "breakpoints": breakpoints,
        "wallclock": clocks,
    }
    (EXP / "raw.json").write_text(json.dumps(_jsonable(raw), indent=2) + "\n")
    print("raw written", EXP / "raw.json", "tasks", len(primary) + len(breakpoints))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP / "config.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.smoke:
        run_smoke(cfg)
    else:
        run_full(cfg)


if __name__ == "__main__":
    main()
