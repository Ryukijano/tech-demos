"""Sample a teleportation circuit and decode it with PyMatching."""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import pymatching

from src.circuit import Basis, build_teleport

BasisName = Literal["X", "Z"]


def stim_seed(seed: int, distance: int, basis: str, chunk: int) -> int:
    """Reproducible sampler seed. chunk indexes shot blocks inside one job."""
    basis_bit = 0 if basis == "Z" else 1
    if basis not in ("Z", "X"):
        raise ValueError(basis)
    return 100_000 * seed + 1_000 * distance + 10 * basis_bit + chunk


def count_failures(
    distance: int,
    basis: Basis,
    p: float,
    init_rounds: int,
    merge_rounds: int,
    shots: int,
    seed: int,
    chunk_size: int = 20_000,
) -> dict[str, float | int]:
    """Return decoded failures for one (distance, basis, p, rounds, seed)."""
    if shots < 1:
        raise ValueError("shots")
    started = time.perf_counter()
    circuit = build_teleport(distance, basis, p, init_rounds, merge_rounds)
    model = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(model)
    fails = 0
    done = 0
    chunk_index = 0
    while done < shots:
        take = min(chunk_size, shots - done)
        sampler = circuit.compile_detector_sampler(
            seed=stim_seed(seed, distance, basis, chunk_index)
        )
        det, obs = sampler.sample(take, separate_observables=True)
        pred = matcher.decode_batch(det)
        fails += int(np.sum(pred[:, 0] != obs[:, 0]))
        done += take
        chunk_index += 1
    elapsed = time.perf_counter() - started
    return {
        "fails": fails,
        "shots": shots,
        "seconds": round(elapsed, 4),
        "detectors": int(circuit.num_detectors),
        "qubits": int(circuit.num_qubits),
    }


def raw_observable_rate(
    distance: int,
    basis: Basis,
    p: float,
    init_rounds: int,
    merge_rounds: int,
    shots: int,
    seed: int,
) -> tuple[int, int]:
    """Uncorrected observable flips. Used only as a decoder sanity check."""
    circuit = build_teleport(distance, basis, p, init_rounds, merge_rounds)
    sampler = circuit.compile_detector_sampler(seed=stim_seed(seed, distance, basis, 0))
    _det, obs = sampler.sample(shots, separate_observables=True)
    return int(np.sum(obs[:, 0])), shots
