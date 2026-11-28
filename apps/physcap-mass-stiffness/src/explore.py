"""Deterministic stand-in for the Planner's sufficiency test.

Appendix D says to stop once no remaining measurement could change the
answer, and not to measure an object that visual context has ruled out.
Here the ruled-out objects are the sealed cans. There is no VLM call.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import physx


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


def _weigh(
    scene_cfg: dict[str, Any],
    obj: dict[str, Any],
    rng: np.random.Generator,
) -> float:
    row = physx.get_mass(scene_cfg, float(obj["mass"]), rng, x_offset=float(rng.uniform(-0.004, 0.004)))
    return float(row["m_hat"])


def run_policy(
    scene_cfg: dict[str, Any],
    explore: dict[str, Any],
    episode_seed: int,
    policy: str,
) -> dict[str, Any]:
    """Both policies see the same cans. Only the probe order and the stop rule differ."""
    scene_rng = np.random.Generator(np.random.PCG64(episode_seed))
    objects = _objects(explore, scene_rng)
    rng = np.random.Generator(np.random.PCG64(episode_seed + (11 if policy == "selective" else 29)))
    by_name = {obj["name"]: obj for obj in objects}
    threshold = float(explore["light_threshold_kg"])
    if policy == "selective":
        order = ["open_a", "open_b"]
        rng.shuffle(order)
    elif policy == "exhaustive":
        order = ["open_a", "open_b", "sealed_a", "sealed_b"]
        rng.shuffle(order)
    else:
        raise ValueError(f"unknown policy {policy}")

    readings: list[tuple[str, float]] = []
    choice = ""
    for name in order:
        m_hat = _weigh(scene_cfg, by_name[name], rng)
        readings.append((name, m_hat))
        if policy == "selective" and 0.0 <= m_hat < threshold:
            choice = name
            break
    if not choice:
        finite = [(name, m_hat) for name, m_hat in readings if m_hat >= 0.0]
        pool = finite if finite else readings
        choice = min(pool, key=lambda item: item[1])[0]
    truth = next(obj["name"] for obj in objects if obj["empty"])
    return {
        "policy": policy,
        "probes": len(readings),
        "choice": choice,
        "truth": truth,
        "success": choice == truth,
        "readings": [{"name": name, "m_hat": m_hat} for name, m_hat in readings],
    }
