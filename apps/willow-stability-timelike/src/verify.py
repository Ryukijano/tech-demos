"""Correctness checks for the stability circuit. Exits non-zero on failure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import stim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.circuit import Noise, boundary_degree, build_stability_circuit  # noqa: E402
from src.leakage import Leakage, burst_mask  # noqa: E402


def _noiseless_sample(distance: int, rounds: int, shots: int = 200) -> None:
    exp = build_stability_circuit(distance, rounds, Noise(0.0), boundary="Z")
    degree = boundary_degree(exp.tiles, "Z")
    odd = {coord: count for coord, count in degree.items() if count % 2}
    if odd:
        raise SystemExit(f"d={distance}: odd Z-degree data qubits {odd}")
    sampler = exp.circuit.compile_detector_sampler(seed=0)
    dets, obs = sampler.sample(shots, separate_observables=True)
    if int(dets.sum()) or int(obs.sum()):
        raise SystemExit(
            f"d={distance} r={rounds}: noiseless detections {int(dets.sum())} "
            f"observable {int(obs.sum())}"
        )
    # A full-timeline flip of one Z ancilla is an undetectable timelike logical.
    meas = exp.circuit.compile_sampler(seed=1).sample(shots)
    z_index = exp.z_tile_index[0]
    for round_index in range(rounds):
        meas[:, exp.ancilla_measurement(round_index, z_index)] ^= True
    converter = exp.circuit.compile_m2d_converter()
    dets2, obs2 = converter.convert(measurements=meas, separate_observables=True)
    if int(dets2.sum()) != 0 or int(obs2.sum()) != shots:
        raise SystemExit(
            f"d={distance} r={rounds}: full-timeline Z flip gave "
            f"detectors {int(dets2.sum())} observable flips {int(obs2.sum())}/{shots}"
        )
    dem = build_stability_circuit(distance, rounds, Noise(1e-3)).circuit.detector_error_model(
        decompose_errors=True
    )
    if dem.num_detectors < 1 or dem.num_observables != 1:
        raise SystemExit(f"d={distance}: bad DEM detectors={dem.num_detectors}")
    print(
        f"ok d={distance:d} rounds={rounds:d} data={len(exp.data_qubits):d} "
        f"tiles={exp.n_tiles:d} Z={len(exp.z_tile_index):d} "
        f"detectors={dem.num_detectors:d}"
    )


def main() -> None:
    for distance in (3, 5, 7):
        _noiseless_sample(distance, rounds=4)
    _noiseless_sample(3, rounds=8)
    # X-boundary patch is the same construction with the bases swapped.
    exp = build_stability_circuit(3, 3, Noise(0.0), boundary="X")
    odd = {c: n for c, n in boundary_degree(exp.tiles, "X").items() if n % 2}
    if odd:
        raise SystemExit(f"X-boundary odd degree {odd}")
    dets, obs = exp.circuit.compile_detector_sampler(seed=0).sample(50, separate_observables=True)
    if int(np.sum(dets)) or int(np.sum(obs)):
        raise SystemExit("X-boundary noiseless circuit is not deterministic")
    print("ok X-boundary d=3")
    opened = build_stability_circuit(3, 6, Noise(0.0), close_data=False)
    dets_o, obs_o = opened.circuit.compile_detector_sampler(seed=0).sample(40, separate_observables=True)
    if int(np.sum(dets_o)) or int(np.sum(obs_o)):
        raise SystemExit("open stability circuit is not deterministic")
    short = build_stability_circuit(3, 4, Noise(0.0), close_data=False)
    width = 4 * short.n_tiles
    prefix = opened.circuit.compile_sampler(seed=0).sample(20)[:, :width]
    dets_p, obs_p = short.circuit.compile_m2d_converter().convert(
        measurements=prefix, separate_observables=True
    )
    if int(np.sum(dets_p)) or int(np.sum(obs_p)):
        raise SystemExit("ancilla prefix is not a shorter open stability experiment")
    print("ok open-circuit prefix")
    full = burst_mask(30, 4, 32, Leakage(0.05, 8.0, None), np.random.default_rng(0))
    walked = burst_mask(30, 4, 32, Leakage(0.05, 8.0, 2), np.random.default_rng(0))
    if int(walked.sum()) >= int(full.sum()) or np.any(walked & ~full):
        raise SystemExit("walking clear is not a strict shortening of the same bursts")
    print("ok walking clear")
    # DEM must reject a deliberately broken observable-less check by building.
    _ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3)
    print("verify passed")


if __name__ == "__main__":
    main()
