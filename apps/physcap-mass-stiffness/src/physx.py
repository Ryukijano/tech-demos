"""Clean-room get_mass and get_stiffness.

The level names and the 420 N/m spring boundaries are from Appendix B of
arXiv:2608.21031. The mapping itself is the natural reading of "four
calibrated boundaries" from springs placed in parallel. The paper does not
publish the numeric cut points, so they are declared in config.yaml.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

import scene

LEVEL_NAMES = {
    0: "failed",
    1: "Ultra-Soft",
    2: "Soft",
    3: "Semi-Rigid",
    4: "Stiff",
    5: "Rigid",
}


def level_from_k(k_hat: float, bounds: list[float]) -> int:
    if not np.isfinite(k_hat) or k_hat <= 0.0:
        return 0
    level = 1
    for bound in bounds:
        if k_hat >= bound:
            level += 1
    return 5 if level > 5 else level


def true_level(k: float, bounds: list[float]) -> int:
    return level_from_k(k, bounds)


def majority(votes: list[int]) -> int:
    if not votes:
        return 0
    counts = Counter(votes)
    best = max(counts.values())
    tied = sorted(level for level, count in counts.items() if count == best)
    if len(tied) == 1:
        return tied[0]
    return int(np.median(np.asarray(votes, dtype=np.int64)))


def get_mass(
    scene_cfg: dict[str, Any],
    mass: float,
    rng: np.random.Generator,
    friction: float | None = None,
    arm_force: float | None = None,
    x_offset: float = 0.0,
) -> dict[str, Any]:
    row = scene.measure_mass(scene_cfg, mass, rng, friction=friction, arm_force=arm_force, x_offset=x_offset)
    row["mass_kg"] = mass
    return row


def get_stiffness(
    scene_cfg: dict[str, Any],
    stiff_cfg: dict[str, Any],
    k: float,
    rng: np.random.Generator,
    frictionloss: float = 0.0,
) -> dict[str, Any]:
    bounds = [float(b) for b in stiff_cfg["bounds_n_per_m"]]
    votes: list[int] = []
    slopes: list[float] = []
    for _ in range(int(stiff_cfg["votes"])):
        one = scene.measure_stiffness(scene_cfg, stiff_cfg, k, rng, frictionloss=frictionloss)
        if one["level_ready"]:
            vote = level_from_k(float(one["k_hat"]), bounds)
            slopes.append(float(one["k_hat"]))
        else:
            vote = 0
            slopes.append(float("nan"))
        votes.append(vote)
    chosen = majority(votes)
    return {
        "k": k,
        "true_level": true_level(k, bounds),
        "level": chosen,
        "votes": votes,
        "slopes": slopes,
        "name": LEVEL_NAMES[chosen],
    }
