"""Official seeds, then the breakpoint sweeps. Thresholds are only read."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

import atomic
import explore
import physx

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_probes"


def load_config() -> dict[str, Any]:
    with (EXP / "config.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise RuntimeError("config.yaml did not load to a mapping")
    return cfg


def _mass_block(cfg: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows = []
    scene = cfg["scene"]
    jitter = float(scene["pose_jitter_m"])
    for index, mass in enumerate(cfg["mass"]["masses_kg"]):
        for trial in range(int(cfg["mass"]["n_trials"])):
            rng = np.random.Generator(np.random.PCG64(seed * 100_000 + index * 100 + trial + 1))
            offset = float(rng.uniform(-jitter, jitter))
            row = physx.get_mass(scene, float(mass), rng, x_offset=offset)
            rows.append(
                {
                    "seed": seed,
                    "trial": trial,
                    "mass_kg": float(mass),
                    "m_hat": float(row["m_hat"]),
                    "grasped": bool(row["grasped"]),
                    "obj_z": float(row["obj_z"]),
                    "x_offset_m": offset,
                }
            )
    return rows


def _stiff_block(cfg: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows = []
    stiff = cfg["stiffness"]
    for index, k in enumerate(stiff["centers_n_per_m"]):
        for obj in range(int(stiff["n_objects_per_level"])):
            rng = np.random.Generator(np.random.PCG64(50_000 + seed * 10_000 + index * 100 + obj + 1))
            row = physx.get_stiffness(cfg["scene"], stiff, float(k), rng)
            rows.append(
                {
                    "seed": seed,
                    "object": obj,
                    "k": float(k),
                    "true_level": int(row["true_level"]),
                    "level": int(row["level"]),
                    "votes": [int(v) for v in row["votes"]],
                    "correct": int(row["level"]) == int(row["true_level"]),
                }
            )
    return rows


def _explore_block(cfg: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows = []
    for episode in range(int(cfg["explore"]["n_episodes"])):
        for policy in ("selective", "exhaustive"):
            episode_seed = 80_000 + seed * 5_000 + episode
            row = explore.run_policy(cfg["scene"], cfg["explore"], episode_seed, policy)
            rows.append(
                {
                    "seed": seed,
                    "episode": episode,
                    "policy": policy,
                    "probes": int(row["probes"]),
                    "success": bool(row["success"]),
                    "choice": row["choice"],
                    "truth": row["truth"],
                }
            )
    return rows


def run_seed(payload: tuple[dict[str, Any], int]) -> dict[str, Any]:
    cfg, seed = payload
    return {
        "seed": seed,
        "mass": _mass_block(cfg, seed),
        "stiffness": _stiff_block(cfg, seed),
        "explore": _explore_block(cfg, seed),
    }


def _light(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    spec = cfg["breakpoints"]["very_light"]
    rows = []
    for mass in spec["masses_kg"]:
        for rep in range(int(cfg["breakpoints"]["n_reps"])):
            rng = np.random.Generator(np.random.PCG64(200_000 + int(float(mass) * 1e6) + rep))
            row = physx.get_mass(cfg["scene"], float(mass), rng)
            rows.append({"mass_kg": float(mass), "m_hat": float(row["m_hat"]), "grasped": bool(row["grasped"])})
    return rows


def _heavy(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for mass in cfg["breakpoints"]["very_heavy"]["masses_kg"]:
        for rep in range(int(cfg["breakpoints"]["n_reps"])):
            rng = np.random.Generator(np.random.PCG64(210_000 + int(float(mass) * 100) + rep))
            row = physx.get_mass(cfg["scene"], float(mass), rng)
            rows.append({"mass_kg": float(mass), "m_hat": float(row["m_hat"]), "grasped": bool(row["grasped"])})
    return rows


def _noise(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    spec = cfg["breakpoints"]["torque_noise"]
    rows = []
    for std in spec["stds_nm"]:
        scene = dict(cfg["scene"])
        scene["torque_noise_std_nm"] = float(std)
        for rep in range(int(cfg["breakpoints"]["n_reps"])):
            rng = np.random.Generator(np.random.PCG64(220_000 + int(float(std) * 1000) + rep))
            row = physx.get_mass(scene, float(spec["mass_kg"]), rng)
            rows.append({"std_nm": float(std), "m_hat": float(row["m_hat"]), "grasped": bool(row["grasped"])})
    return rows


def _rigid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for k in cfg["breakpoints"]["near_rigid"]["k_n_per_m"]:
        for rep in range(int(cfg["breakpoints"]["n_reps"])):
            rng = np.random.Generator(np.random.PCG64(230_000 + int(k) + rep))
            row = physx.get_stiffness(cfg["scene"], cfg["stiffness"], float(k), rng)
            rows.append({"k": float(k), "level": int(row["level"]), "true_level": int(row["true_level"])})
    return rows


def _slip(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    spec = cfg["breakpoints"]["friction_slip"]
    rows = []
    for mu in spec["mus"]:
        for rep in range(int(cfg["breakpoints"]["n_reps"])):
            rng = np.random.Generator(np.random.PCG64(240_000 + int(float(mu) * 1000) + rep))
            row = physx.get_mass(cfg["scene"], float(spec["mass_kg"]), rng, friction=float(mu))
            rows.append({"mu": float(mu), "m_hat": float(row["m_hat"]), "grasped": bool(row["grasped"])})
    return rows


def _jaw(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    spec = cfg["breakpoints"]["jaw_friction"]
    rows = []
    for loss in spec["frictionloss_n"]:
        for rep in range(int(cfg["breakpoints"]["n_reps"])):
            rng = np.random.Generator(np.random.PCG64(250_000 + int(float(loss) * 100) + rep))
            row = physx.get_stiffness(
                cfg["scene"],
                cfg["stiffness"],
                float(spec["k_n_per_m"]),
                rng,
                frictionloss=float(loss),
            )
            finite = [s for s in row["slopes"] if np.isfinite(s)]
            rows.append(
                {
                    "frictionloss_n": float(loss),
                    "level": int(row["level"]),
                    "true_level": int(row["true_level"]),
                    "median_slope": float(np.median(finite)) if finite else float("nan"),
                }
            )
    return rows


def _merge(parts: list[dict[str, Any]]) -> dict[str, Any]:
    mass: list[dict[str, Any]] = []
    stiff: list[dict[str, Any]] = []
    expl: list[dict[str, Any]] = []
    for part in parts:
        mass.extend(part["mass"])
        stiff.extend(part["stiffness"])
        expl.extend(part["explore"])
    return {"mass": mass, "stiffness": stiff, "explore": expl}


def smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(99))
    mass = physx.get_mass(cfg["scene"], 0.208, rng)
    stiff = physx.get_stiffness(cfg["scene"], cfg["stiffness"], 630.0, rng)
    one = explore.run_policy(cfg["scene"], cfg["explore"], 99, "selective")
    return {
        "note": "Code-path check only. Not an official seed.",
        "mass": {"m_hat": mass["m_hat"], "grasped": mass["grasped"]},
        "stiffness_level": stiff["level"],
        "true_level": stiff["true_level"],
        "selective_probes": one["probes"],
        "selective_success": one["success"],
    }


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    cfg = load_config()
    EXP.mkdir(parents=True, exist_ok=True)
    if "--smoke" in args:
        payload = smoke(cfg)
        atomic.write_text(EXP / "smoke.json", json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return
    seeds = [int(s) for s in cfg["seeds"]]
    workers = int(cfg["runtime"]["max_workers"])
    parts = _map(run_seed, [(cfg, seed) for seed in seeds], workers)
    primary = _merge(parts)
    # Breakpoint families are independent. Run them across the same pool size.
    jobs = {
        "very_light": _light,
        "very_heavy": _heavy,
        "torque_noise": _noise,
        "near_rigid": _rigid,
        "friction_slip": _slip,
        "jaw_friction": _jaw,
    }
    names = list(jobs)
    results = _map(_call_break, [(name, cfg) for name in names], workers)
    breakpoints = dict(zip(names, results, strict=True))
    raw = {
        "exp_id": cfg["exp_id"],
        "seeds": seeds,
        "mass": primary["mass"],
        "stiffness": primary["stiffness"],
        "explore": primary["explore"],
        "breakpoints": breakpoints,
    }
    atomic.write_text(EXP / "raw.json", json.dumps(raw) + "\n")
    print(f"wrote {EXP / 'raw.json'}  mass={len(raw['mass'])} stiff={len(raw['stiffness'])} explore={len(raw['explore'])}")


def _map(fn: Callable[..., Any], items: list[Any], workers: int) -> list[Any]:
    """Fork on Unix. Spawn on Windows, and only when this module was imported.

    A script entry point is ``__main__``. Spawn workers cannot unpickle
    callables that live there, so that case runs in process and still writes
    results.
    """
    spawn_only = "fork" not in mp.get_all_start_methods() or os.name == "nt"
    if spawn_only and __name__ == "__main__":
        return [fn(item) for item in items]
    start = "spawn" if spawn_only else "fork"
    ctx = mp.get_context(start)
    with ctx.Pool(processes=max(1, min(workers, len(items)))) as pool:
        return pool.map(fn, items)


def _call_break(payload: tuple[str, dict[str, Any]]) -> list[dict[str, Any]]:
    name, cfg = payload
    table = {
        "very_light": _light,
        "very_heavy": _heavy,
        "torque_noise": _noise,
        "near_rigid": _rigid,
        "friction_slip": _slip,
        "jaw_friction": _jaw,
    }
    return table[name](cfg)


if __name__ == "__main__":
    mp.freeze_support()
    main(sys.argv[1:])
