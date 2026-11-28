"""Deterministic stand-in for the Planner's sufficiency test.

Appendix D says to stop once no remaining measurement could change the
answer, and not to measure an object that visual context has ruled out.
Here the ruled-out objects are the sealed cans. There is no VLM call.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import physx

# Stable per-can codes so both policies draw the same noise for a can.
_CAN_CODE = {"open_a": 1, "open_b": 2, "sealed_a": 3, "sealed_b": 4}


def _objects(explore: dict[str, Any], rng: np.random.Generator) -> list[dict[str, Any]]:
    empty_slot = int(rng.integers(0, 2))
    sealed = [float(m) for m in explore["sealed_kg"]]
    opens = []
    for index, name in enumerate(("open_a", "open_b")):
        mass = float(explore["empty_kg"] if index == empty_slot else explore["full_open_kg"])
        opens.append({"name": name, "role": "open", "mass": mass, "empty": index == empty_slot})
    sealed_rows = [
        {"name": "sealed_a", "role": "sealed", "mass": sealed[0], "empty": False},
        {"name": "sealed_b", "role": "sealed", "mass": sealed[1], "empty": False},
    ]
    return opens + sealed_rows


def _reading(scene_cfg: dict[str, Any], obj: dict[str, Any], episode_seed: int) -> dict[str, Any]:
    """One measurement per can per episode. Policies share this draw."""
    rng = np.random.Generator(np.random.PCG64(episode_seed * 16 + _CAN_CODE[obj["name"]]))
    offset = float(rng.uniform(-0.004, 0.004))
    row = physx.get_mass(scene_cfg, float(obj["mass"]), rng, x_offset=offset)
    return {
        "name": obj["name"],
        "m_hat": float(row["m_hat"]),
        "grasped": bool(row["grasped"]),
    }


def _choose(readings: list[dict[str, Any]]) -> tuple[str, bool]:
    """Lightest successful grasp. Unresolved when every probe failed to grasp."""
    ok = [rec for rec in readings if rec["grasped"]]
    if not ok:
        return "", False
    return min(ok, key=lambda rec: float(rec["m_hat"]))["name"], True


def run_policy(
    scene_cfg: dict[str, Any],
    explore: dict[str, Any],
    episode_seed: int,
    policy: str,
) -> dict[str, Any]:
    """Both policies see the same cans and the same per-can noise.

    Only the probe order and the stop rule differ. A failed grasp is not a
    mass. Appendix B.3 treats a slightly negative reading on a successful
    grasp as noise, and a large negative value as a failed grasp. The
    simulator returns that failure as grasped=False and m_hat=-1 kg.
    """
    scene_rng = np.random.Generator(np.random.PCG64(episode_seed))
    objects = _objects(explore, scene_rng)
    by_name = {obj["name"]: obj for obj in objects}
    threshold = float(explore["light_threshold_kg"])
    order_rng = np.random.Generator(np.random.PCG64(episode_seed + (11 if policy == "selective" else 29)))
    if policy == "selective":
        order = ["open_a", "open_b"]
    elif policy == "exhaustive":
        order = ["open_a", "open_b", "sealed_a", "sealed_b"]
    else:
        raise ValueError(f"unknown policy {policy}")
    order_rng.shuffle(order)

    readings: list[dict[str, Any]] = []
    choice = ""
    for name in order:
        rec = _reading(scene_cfg, by_name[name], episode_seed)
        readings.append(rec)
        # A successful grasp below the threshold, including a small negative
        # caused by torque noise, is light enough to call empty.
        if policy == "selective" and rec["grasped"] and float(rec["m_hat"]) < threshold:
            choice = name
            break
    resolved = True
    if not choice:
        choice, resolved = _choose(readings)
    truth = next(obj["name"] for obj in objects if obj["empty"])
    return {
        "policy": policy,
        "probes": len(readings),
        "choice": choice,
        "truth": truth,
        "resolved": resolved,
        "success": resolved and choice == truth,
        "readings": readings,
    }
