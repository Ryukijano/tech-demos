"""Toy temporally correlated leakage, as a stim-level analogue of arXiv:2611.21578 SM S2.

Stim is a Pauli-frame simulator, so this is not a |3> transmon. A data-qubit
defect is modelled the way the supplement models it: one boundary-type
stabilizer's measurement record is flipped for N consecutive cycles, with
N drawn from the exponential lifetime in their Eq. (S1). The Pauli decoder
does not know about the defect.

The walking variant is a simplified analogue of the paper's walking circuit,
which resets every qubit every two cycles. Here the same burst is cut at the
next multiple of `clear_every` (default 2). The stabilizer circuit itself is
unchanged, so this analogue does not include the walking circuit's extra
qubit-count overhead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.circuit import StabilityExperiment


@dataclass(frozen=True)
class Leakage:
    p_start: float
    tau: float
    # None: bursts run until they decay. 2: walking-style clear every two cycles.
    clear_every: int | None


def sample_lengths(tau: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """N >= 1 with CDF(K) = 1 - exp(-K/tau), the paper's exponential lifetime."""

    if n == 0:
        return np.zeros(0, dtype=np.int64)
    u = rng.random(n)
    return np.maximum(1, np.ceil(-tau * np.log(np.maximum(1e-12, 1.0 - u)))).astype(np.int64)


def burst_mask(
    n_shots: int,
    n_stabilizers: int,
    rounds: int,
    leakage: Leakage,
    rng: np.random.Generator,
) -> np.ndarray:
    """Boolean (shots, stabilizers, rounds) of measurement flips."""

    mask = np.zeros((n_shots, n_stabilizers, rounds), dtype=bool)
    if leakage.p_start <= 0.0 or n_stabilizers == 0:
        return mask
    starts = rng.random((n_shots, n_stabilizers, rounds)) < leakage.p_start
    coords = np.argwhere(starts)
    if len(coords) == 0:
        return mask
    lengths = sample_lengths(leakage.tau, len(coords), rng)
    for (shot, stab, start), length in zip(coords, lengths):
        end = int(start) + int(length)
        if leakage.clear_every:
            end = min(end, (int(start) // leakage.clear_every + 1) * leakage.clear_every)
        end = min(end, rounds)
        if end > int(start):
            mask[int(shot), int(stab), int(start) : end] = True
    return mask


def apply_mask(measurements: np.ndarray, mask: np.ndarray, experiment: StabilityExperiment) -> np.ndarray:
    """XOR boundary-stabilizer ancilla bits. `measurements` is not copied if no flips."""

    flipped = np.argwhere(mask)
    if len(flipped) == 0:
        return measurements
    out = np.array(measurements, copy=True)
    z_index = np.asarray(experiment.z_tile_index, dtype=np.int64)
    shots = flipped[:, 0]
    cols = flipped[:, 2] * experiment.n_tiles + z_index[flipped[:, 1]]
    out[shots, cols] ^= True
    return out
