"""Rotated-surface-code lattice-surgery teleportation circuits for stim.

The d=3 merged patch is the same shape as Hofele et al., arXiv:2611.23165:
21 data qubits, 20 ancillas (41 qubits), 12 X and 8 Z stabilizers, 8 weight-2
and 12 weight-4 checks, and 4 seam X checks whose product is X_L1 X_L2.

The CNOT layer order matches stim's rotated-memory generator. It is not the
device's pipelined 1664 ns schedule. Pauli frame updates are folded into one
observable; nothing is applied as a physical feedforward gate.
"""

from __future__ import annotations

from typing import Literal

import stim

Basis = Literal["X", "Z"]
Policy = Literal["compare", "determin", "none"]
Coord = tuple[int, int]
Stab = tuple[int, int, str, tuple[Coord, ...]]

X_DIRS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
Z_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _unexpected(value: str) -> None:
    raise ValueError(f"unhandled basis {value}")


def data_coords(width: int, height: int, x0: int = 0) -> list[Coord]:
    return [(x0 + 2 * c + 1, 2 * r + 1) for r in range(height) for c in range(width)]


def stabilizers(width: int, height: int, x0: int = 0) -> list[Stab]:
    """Rotated surface-code checks on a width-by-height block of data qubits.

    Z boundaries are left and right, so a horizontal data row is a Z logical.
    X boundaries are top and bottom, so a vertical data column is an X logical.
    """
    data = set(data_coords(width, height, x0))
    x_max = x0 + 2 * width
    y_max = 2 * height
    found: list[Stab] = []
    for y in range(0, y_max + 1, 2):
        for x in range(x0, x_max + 1, 2):
            kind = (x + y) % 4
            if kind == 0:
                basis = "Z"
                if not (0 < y < y_max):
                    continue
            elif kind == 2:
                basis = "X"
                if not (x0 < x < x_max):
                    continue
            else:
                continue
            neigh = tuple(
                sorted(
                    q
                    for q in (
                        (x + 1, y + 1),
                        (x + 1, y - 1),
                        (x - 1, y + 1),
                        (x - 1, y - 1),
                    )
                    if q in data
                )
            )
            if len(neigh) < 2:
                continue
            found.append((x, y, basis, neigh))
    return found


def patch_layout(distance: int) -> dict[str, object]:
    """Two d×d patches plus one interface column. Merged block is d × (2d+1)."""
    d = distance
    l1 = stabilizers(d, d, 0)
    l2 = stabilizers(d, d, 2 * d + 2)
    merged = stabilizers(2 * d + 1, d, 0)
    l1_data = data_coords(d, d, 0)
    l2_data = data_coords(d, d, 2 * d + 2)
    iface = [(2 * d + 1, 2 * r + 1) for r in range(d)]
    l1x = {q[0] for q in l1_data}
    l2x = {q[0] for q in l2_data}
    seam = tuple(
        (x, y)
        for x, y, basis, neigh in merged
        if basis == "X" and not ({q[0] for q in neigh} <= l1x or {q[0] for q in neigh} <= l2x)
    )
    return {
        "l1": l1,
        "l2": l2,
        "merged": merged,
        "l1_data": l1_data,
        "l2_data": l2_data,
        "iface": iface,
        "seam": seam,
        "l2_left_x": 2 * d + 3,
        "z_row_y": 1,
    }


class _Builder:
    def __init__(self) -> None:
        self.c = stim.Circuit()
        self.q: dict[Coord, int] = {}
        self.n_meas = 0
        # coord -> (basis, neighbors, absolute measurement index)
        self.last: dict[Coord, tuple[str, tuple[Coord, ...], int]] = {}

    def qubit(self, coord: Coord) -> int:
        if coord not in self.q:
            idx = len(self.q)
            self.q[coord] = idx
            self.c.append("QUBIT_COORDS", idx, [coord[0], coord[1]])
        return self.q[coord]

    def _rec(self, absolute: int) -> int:
        return absolute - self.n_meas

    def reset(self, coords: list[Coord], p: float, plus: bool) -> None:
        if not coords:
            return
        qs = [self.qubit(c) for c in coords]
        self.c.append("R", qs)
        if p > 0:
            self.c.append("X_ERROR", qs, p)
        if plus:
            self.c.append("H", qs)
            if p > 0:
                self.c.append("DEPOLARIZE1", qs, p)

    def syndrome_round(
        self,
        stabs: list[Stab],
        p: float,
        policy_fn,
        time_index: int,
    ) -> None:
        data = sorted({q for stab in stabs for q in stab[3]})
        if p > 0 and data:
            self.c.append("DEPOLARIZE1", [self.qubit(q) for q in data], p)
        x_anc = [(x, y) for x, y, basis, _n in stabs if basis == "X"]
        if x_anc:
            self.c.append("H", [self.qubit(c) for c in x_anc])
            if p > 0:
                self.c.append("DEPOLARIZE1", [self.qubit(c) for c in x_anc], p)
        self.c.append("TICK")
        for layer in range(4):
            pairs: list[int] = []
            used: set[Coord] = set()
            for x, y, basis, neigh in stabs:
                dirs = X_DIRS if basis == "X" else Z_DIRS
                dx, dy = dirs[layer]
                partner = (x + dx, y + dy)
                if partner not in neigh:
                    continue
                if basis == "X":
                    control, target = (x, y), partner
                elif basis == "Z":
                    control, target = partner, (x, y)
                else:
                    _unexpected(basis)
                    continue
                if control in used or target in used:
                    raise RuntimeError(f"overlapping CX {control} {target}")
                used.add(control)
                used.add(target)
                pairs.extend((self.qubit(control), self.qubit(target)))
            if pairs:
                self.c.append("CX", pairs)
                if p > 0:
                    self.c.append("DEPOLARIZE2", pairs, p)
            self.c.append("TICK")
        if x_anc:
            self.c.append("H", [self.qubit(c) for c in x_anc])
            if p > 0:
                self.c.append("DEPOLARIZE1", [self.qubit(c) for c in x_anc], p)
        anc = sorted({(x, y) for x, y, _b, _n in stabs})
        if p > 0:
            self.c.append("X_ERROR", [self.qubit(c) for c in anc], p)
        self.c.append("MR", [self.qubit(c) for c in anc])
        new_idx: dict[Coord, int] = {}
        for coord in anc:
            new_idx[coord] = self.n_meas
            self.n_meas += 1
        if p > 0:
            self.c.append("X_ERROR", [self.qubit(c) for c in anc], p)
        by_coord = {(x, y): (x, y, basis, neigh) for x, y, basis, neigh in stabs}
        for coord in anc:
            stab = by_coord[coord]
            prev = self.last.get(coord)
            policy: Policy = policy_fn(stab, prev)
            if policy == "compare":
                if prev is None:
                    raise RuntimeError("compare without a previous measurement")
                targets = [
                    stim.target_rec(self._rec(new_idx[coord])),
                    stim.target_rec(self._rec(prev[2])),
                ]
                self.c.append("DETECTOR", targets, [coord[0], coord[1], time_index])
            elif policy == "determin":
                self.c.append(
                    "DETECTOR",
                    [stim.target_rec(self._rec(new_idx[coord]))],
                    [coord[0], coord[1], time_index],
                )
            elif policy == "none":
                pass
            else:
                _unexpected(policy)
            self.last[coord] = (stab[2], stab[3], new_idx[coord])
        self.c.append("TICK")


def _init_policy(stab: Stab, prev, prepared: str) -> Policy:
    det_basis = "X" if prepared == "X" else "Z"
    if prepared not in ("X", "Z"):
        _unexpected(prepared)
    if prev is not None and prev[0] == stab[2] and prev[1] == stab[3]:
        return "compare"
    if prev is None and stab[2] == det_basis:
        return "determin"
    return "none"


def _merge_policy(stab: Stab, prev, fresh: set[Coord]) -> Policy:
    if prev is None:
        return "none"
    basis, neigh, _idx = prev
    if basis == stab[2] and neigh == stab[3]:
        return "compare"
    # Boundary Z checks grow onto the freshly reset interface column.
    if (
        stab[2] == "Z"
        and basis == "Z"
        and set(neigh).issubset(stab[3])
        and set(stab[3]).difference(neigh).issubset(fresh)
    ):
        return "compare"
    return "none"


def build_teleport(
    distance: int,
    basis: Basis,
    p: float,
    init_rounds: int,
    merge_rounds: int,
    pauli_before_measure: dict[Coord, str] | None = None,
) -> stim.Circuit:
    """XX-merge teleportation of a Z or X eigenstate from patch L1 onto L2.

    L2 is prepared in |0>. L1 is prepared in |0> (basis Z) or |+> (basis X).
    The checked observable is deterministic and equals 0 with no noise:
    Z basis — merged logical Z, which is the paper's m_S m_Z Z_L2.
    X basis — last-round seam X product XOR X on L2's facing column,
    which is the paper's m_X X_L2 when X_L1 = +1.
    """
    if basis not in ("X", "Z"):
        _unexpected(basis)
    if distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be an odd integer >= 3")
    if init_rounds < 1 or merge_rounds < 1:
        raise ValueError("init_rounds and merge_rounds must be >= 1")
    if p < 0 or p > 1:
        raise ValueError("p out of range")

    layout = patch_layout(distance)
    l1: list[Stab] = layout["l1"]  # type: ignore[assignment]
    l2: list[Stab] = layout["l2"]  # type: ignore[assignment]
    merged: list[Stab] = layout["merged"]  # type: ignore[assignment]
    l1_data: list[Coord] = layout["l1_data"]  # type: ignore[assignment]
    l2_data: list[Coord] = layout["l2_data"]  # type: ignore[assignment]
    iface: list[Coord] = layout["iface"]  # type: ignore[assignment]
    seam: tuple[Coord, ...] = layout["seam"]  # type: ignore[assignment]
    l2_left = int(layout["l2_left_x"])  # type: ignore[arg-type]

    b = _Builder()
    anc = sorted({(x, y) for x, y, _basis, _n in merged})
    for coord in l1_data + l2_data + iface + anc:
        b.qubit(coord)

    b.reset(l1_data, p, plus=(basis == "X"))
    b.reset(l2_data, p, plus=False)
    b.reset(anc, p, plus=False)
    b.c.append("TICK")

    l1_keys = {(s[0], s[1], s[2], s[3]) for s in l1}

    def policy_init(stab: Stab, prev) -> Policy:
        prepared = basis if (stab[0], stab[1], stab[2], stab[3]) in l1_keys else "Z"
        return _init_policy(stab, prev, prepared)

    separate = l1 + l2
    for round_index in range(init_rounds):
        b.syndrome_round(separate, p, policy_init, round_index)

    b.reset(iface, p, plus=False)
    b.c.append("TICK")
    fresh = set(iface)

    def policy_first_merge(stab: Stab, prev) -> Policy:
        return _merge_policy(stab, prev, fresh)

    def policy_later(stab: Stab, prev) -> Policy:
        if prev is not None and prev[0] == stab[2] and prev[1] == stab[3]:
            return "compare"
        return "none"

    for step in range(merge_rounds):
        policy = policy_first_merge if step == 0 else policy_later
        b.syndrome_round(merged, p, policy, init_rounds + step)

    all_data = sorted(set(l1_data + l2_data + iface))
    if pauli_before_measure:
        # Probability-1 errors, not Cliffords: stim's detector sampler folds
        # guaranteed X/Z gates into the expected parity and would hide the flip.
        xs = [q for q, pauli in pauli_before_measure.items() if pauli == "X"]
        zs = [q for q, pauli in pauli_before_measure.items() if pauli == "Z"]
        other = [pauli for pauli in pauli_before_measure.values() if pauli not in ("X", "Z")]
        if other:
            _unexpected(other[0])
        if xs:
            b.c.append("X_ERROR", [b.qubit(q) for q in xs], 1)
        if zs:
            b.c.append("Z_ERROR", [b.qubit(q) for q in zs], 1)
    if basis == "X":
        b.c.append("H", [b.qubit(q) for q in all_data])
        if p > 0:
            b.c.append("DEPOLARIZE1", [b.qubit(q) for q in all_data], p)
    if p > 0:
        b.c.append("X_ERROR", [b.qubit(q) for q in all_data], p)
    b.c.append("M", [b.qubit(q) for q in all_data])
    data_idx: dict[Coord, int] = {}
    for coord in all_data:
        data_idx[coord] = b.n_meas
        b.n_meas += 1

    close = "X" if basis == "X" else "Z"
    final_time = init_rounds + merge_rounds
    for x, y, stab_basis, neigh in merged:
        if stab_basis != close:
            continue
        prev = b.last.get((x, y))
        if prev is None or prev[1] != neigh or prev[0] != stab_basis:
            continue
        targets = [stim.target_rec(b._rec(prev[2]))]
        targets.extend(stim.target_rec(b._rec(data_idx[q])) for q in neigh)
        b.c.append("DETECTOR", targets, [x, y, final_time])

    if basis == "Z":
        obs_qs = [(2 * c + 1, 1) for c in range(2 * distance + 1)]
        b.c.append(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(b._rec(data_idx[q])) for q in obs_qs],
            0,
        )
    else:
        obs = [stim.target_rec(b._rec(b.last[coord][2])) for coord in seam]
        obs.extend(
            stim.target_rec(b._rec(data_idx[(l2_left, 2 * r + 1)])) for r in range(distance)
        )
        b.c.append("OBSERVABLE_INCLUDE", obs, 0)
    return b.c
