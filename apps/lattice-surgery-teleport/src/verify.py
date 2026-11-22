"""Correctness checks that do not touch the pre-registered pass bar.

Noiseless teleportation, graphlike distance, the d=3 layout counts from
arXiv:2611.23165, and a small noisy shot that shows decoding beats the raw
observable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.circuit import Basis, build_teleport, data_coords, patch_layout, stabilizers  # noqa: E402
from src.decode import count_failures, raw_observable_rate  # noqa: E402


def _commutes(left: dict[tuple[int, int], str], right: dict[tuple[int, int], str]) -> bool:
    odd = 0
    for qubit, pauli in left.items():
        other = right.get(qubit)
        if other is None or other == pauli:
            continue
        odd += 1
    return odd % 2 == 0


def _mul(ops: list[dict[tuple[int, int], str]]) -> dict[tuple[int, int], str]:
    acc: dict[tuple[int, int], str] = {}
    mix = {"XZ": "Y", "ZX": "Y", "XY": "Z", "YX": "Z", "YZ": "X", "ZY": "X"}
    for op in ops:
        for qubit, pauli in op.items():
            if qubit not in acc:
                acc[qubit] = pauli
            elif acc[qubit] == pauli:
                del acc[qubit]
            else:
                acc[qubit] = mix[acc[qubit] + pauli]
    return acc


def check_layout() -> None:
    for distance in (3, 5, 7):
        layout = patch_layout(distance)
        merged = layout["merged"]
        data = (2 * distance + 1) * distance
        assert len(merged) == data - 1, (distance, len(merged), data)
        paulis = [{q: basis for q in neigh} for _x, _y, basis, neigh in merged]
        for i, left in enumerate(paulis):
            for right in paulis[i + 1 :]:
                assert _commutes(left, right)
        z_logical = {(2 * c + 1, 1): "Z" for c in range(2 * distance + 1)}
        assert all(_commutes(z_logical, op) for op in paulis)
    layout = patch_layout(3)
    merged = layout["merged"]
    n_x = sum(1 for stab in merged if stab[2] == "X")
    n_z = sum(1 for stab in merged if stab[2] == "Z")
    n_w2 = sum(1 for stab in merged if len(stab[3]) == 2)
    n_w4 = sum(1 for stab in merged if len(stab[3]) == 4)
    assert (n_x, n_z, n_w2, n_w4, len(layout["seam"])) == (12, 8, 8, 12, 4)
    seam_ops = []
    for x, y, basis, neigh in merged:
        if (x, y) in layout["seam"]:
            assert basis == "X"
            seam_ops.append({q: "X" for q in neigh})
    seam_product = _mul(seam_ops)
    x_l1 = {(5, 2 * r + 1): "X" for r in range(3)}
    x_l2 = {(9, 2 * r + 1): "X" for r in range(3)}
    assert seam_product == _mul([x_l1, x_l2])
    # Square d=3 patch is the usual 9 data + 8 ancilla code.
    assert len(stabilizers(3, 3)) == 8
    assert len(data_coords(3, 3)) == 9


def check_noiseless(distance: int, basis: Basis, init_rounds: int, merge_rounds: int, shots: int) -> None:
    circuit = build_teleport(distance, basis, 0.0, init_rounds, merge_rounds)
    det, obs = circuit.compile_detector_sampler().sample(shots, separate_observables=True)
    fires = int(det.sum()) + int(obs.sum())
    if fires:
        raise SystemExit(
            f"noiseless failure d={distance} {basis} init={init_rounds} "
            f"merge={merge_rounds} fires={fires}"
        )
    if distance == 3 and init_rounds == 3 and merge_rounds == 3 and circuit.num_qubits != 41:
        raise SystemExit(f"d=3 qubit count {circuit.num_qubits}, expected 41")


def check_distance() -> None:
    for distance in (3, 5, 7):
        for basis in ("Z", "X"):
            circuit = build_teleport(distance, basis, 0.001, distance, distance)
            found = circuit.shortest_graphlike_error(ignore_ungraphlike_errors=True)
            if len(found) != distance:
                raise SystemExit(f"graphlike distance {len(found)} != {distance} for {basis}")


def check_logical_flip() -> None:
    # A full X column is a Z-logical flip and commutes with every stabilizer.
    flip_z = {(1, 2 * r + 1): "X" for r in range(3)}
    circuit = build_teleport(3, "Z", 0.0, 3, 3, pauli_before_measure=flip_z)
    det, obs = circuit.compile_detector_sampler().sample(8, separate_observables=True)
    if int(det.sum()) != 0 or not np.all(obs[:, 0] == 1):
        raise SystemExit("Z logical flip was detected or did not flip the observable")
    # A full horizontal Z row flips X_L2 once and commutes with X stabilizers.
    flip_x = {(2 * c + 1, 1): "Z" for c in range(7)}
    circuit = build_teleport(3, "X", 0.0, 3, 3, pauli_before_measure=flip_x)
    det, obs = circuit.compile_detector_sampler().sample(8, separate_observables=True)
    if int(det.sum()) != 0 or not np.all(obs[:, 0] == 1):
        raise SystemExit("X logical flip was detected or did not flip the observable")


def check_decoder_helps() -> None:
    for basis in ("Z", "X"):
        decoded = count_failures(3, basis, 0.005, 3, 3, 800, seed=0)
        raw, shots = raw_observable_rate(3, basis, 0.005, 3, 3, 800, seed=0)
        if decoded["fails"] >= raw:
            raise SystemExit(f"decoder did not beat raw {basis}: {decoded['fails']} vs {raw}")
        if decoded["fails"] / shots > 0.15:
            raise SystemExit(f"d=3 p=0.005 {basis} failure {decoded['fails'] / shots} is unexpectedly high")


def main() -> None:
    check_layout()
    for distance in (3, 5, 7):
        for basis in ("Z", "X"):
            check_noiseless(distance, basis, distance, distance, 12)
    check_noiseless(3, "Z", 2, 3, 12)
    check_noiseless(3, "X", 1, 1, 12)
    check_distance()
    check_logical_flip()
    check_decoder_helps()
    print("verify ok")


if __name__ == "__main__":
    main()
