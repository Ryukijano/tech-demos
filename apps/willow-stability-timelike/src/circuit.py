"""Rotated-surface-code stability experiment (Gidney, Quantum 6, 786, 2022).

Stim ships memory-experiment generators only. This module builds the stability
experiment from that paper's definition:

1. A patch whose four spatial boundaries are the same type (default Z).
2. Data qubits prepared in the opposite basis, so those boundary checks are
   not fixed by initialization.
3. `rounds` cycles of X and Z stabilizer measurement.
4. Data qubits measured in that same opposite basis.
5. The timelike observable is the product of every boundary-type stabilizer
   in the last cycle. Corner data qubits that would sit in only one tile are
   removed, leaving every remaining data qubit in an even number of
   boundary-type tiles, so the product is the identity.

The Willow experiment (arXiv:2611.21578, SM S1) uses a ZXXZ gate schedule and
|+Y> preparation. This circuit is the textbook opposite-basis construction,
not that device schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import stim

# Hook order relative to a tile's top-left corner. The middle two data qubits
# are swapped between X and Z so a single hook fault does not point the same way
# for both checks.
_Z_OFFSETS: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (0, 1), (1, 1))
_X_OFFSETS: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))


@dataclass(frozen=True)
class Tile:
    basis: str
    ancilla: int
    ancilla_coord: tuple[float, float]
    # Four hook-order slots. None means off-patch or a clipped corner.
    slots: tuple[int | None, int | None, int | None, int | None]
    slot_coords: tuple[tuple[int, int] | None, ...]

    @property
    def data(self) -> tuple[int, ...]:
        return tuple(q for q in self.slots if q is not None)

    @property
    def data_coords(self) -> tuple[tuple[int, int], ...]:
        return tuple(c for c in self.slot_coords if c is not None)


@dataclass(frozen=True)
class Noise:
    """Uniform circuit-level Pauli noise. Not SI1000 and not a Willow error budget."""

    p: float

    @property
    def clifford(self) -> float:
        return self.p

    @property
    def reset_flip(self) -> float:
        return self.p

    @property
    def measure_flip(self) -> float:
        return self.p

    @property
    def data_idle(self) -> float:
        return self.p


@dataclass(frozen=True)
class StabilityExperiment:
    circuit: stim.Circuit
    tiles: tuple[Tile, ...]
    data_coords: tuple[tuple[int, int], ...]
    data_qubits: tuple[int, ...]
    rounds: int
    boundary: str
    distance: int

    @property
    def n_tiles(self) -> int:
        return len(self.tiles)

    @property
    def z_tile_index(self) -> tuple[int, ...]:
        return tuple(i for i, tile in enumerate(self.tiles) if tile.basis == self.boundary)

    def ancilla_measurement(self, round_index: int, tile_index: int) -> int:
        return round_index * self.n_tiles + tile_index


def _basis_of(x: int, y: int) -> str:
    return "X" if (x + y) % 2 == 0 else "Z"


def boundary_degree(tiles: tuple[Tile, ...], boundary: str) -> dict[tuple[int, int], int]:
    degree: dict[tuple[int, int], int] = {}
    for tile in tiles:
        if tile.basis != boundary:
            continue
        for coord in tile.data_coords:
            degree[coord] = degree.get(coord, 0) + 1
    return degree


def _tiles_for(distance: int, boundary: str) -> tuple[Tile, ...]:
    if distance < 2:
        raise ValueError("distance must be >= 2")
    if boundary not in ("X", "Z"):
        raise ValueError("boundary must be X or Z")

    raw: list[tuple[str, tuple[tuple[int, int] | None, ...], tuple[float, float]]] = []
    for x in range(-1, distance):
        for y in range(-1, distance):
            basis = _basis_of(x, y)
            if x in (-1, distance - 1) and basis != boundary:
                continue
            if y in (-1, distance - 1) and basis != boundary:
                continue
            offsets = _X_OFFSETS if basis == "X" else _Z_OFFSETS
            coords: list[tuple[int, int] | None] = []
            for dx, dy in offsets:
                qx, qy = x + dx, y + dy
                if 0 <= qx < distance and 0 <= qy < distance:
                    coords.append((qx, qy))
                else:
                    coords.append(None)
            if sum(c is not None for c in coords) < 2:
                continue
            raw.append((basis, tuple(coords), (x + 0.5, y + 0.5)))

    usage: dict[tuple[int, int], int] = {}
    for _basis, coords, _anc in raw:
        for coord in coords:
            if coord is not None:
                usage[coord] = usage.get(coord, 0) + 1
    # A data qubit in only one tile is an exposed corner. Removing it turns
    # that square into a triangle and drops an odd factor from the global product.
    clipped = {coord for coord, count in usage.items() if count < 2}
    kept_coords = sorted(
        (coord for coord in usage if coord not in clipped),
        key=lambda c: (c[1], c[0]),
    )
    coord_to_q = {coord: index for index, coord in enumerate(kept_coords)}
    n_data = len(kept_coords)

    tiles: list[Tile] = []
    ordered = sorted(raw, key=lambda item: (item[2][1], item[2][0], item[0]))
    for basis, coords, anc in ordered:
        slot_coords = tuple(None if (c is None or c in clipped) else c for c in coords)
        if not any(c is not None for c in slot_coords):
            continue
        slots = tuple(None if c is None else coord_to_q[c] for c in slot_coords)
        tiles.append(
            Tile(
                basis=basis,
                ancilla=n_data + len(tiles),
                ancilla_coord=anc,
                slots=(slots[0], slots[1], slots[2], slots[3]),
                slot_coords=slot_coords,
            )
        )
    if not tiles:
        raise RuntimeError(f"no tiles for distance={distance} boundary={boundary}")
    return tuple(tiles)


def _rec(measure_index: int, n_meas: int) -> stim.GateTarget:
    return stim.target_rec(-(n_meas - measure_index))


def build_stability_circuit(
    distance: int,
    rounds: int,
    noise: Noise,
    boundary: str = "Z",
    close_data: bool = True,
) -> StabilityExperiment:
    """`close_data=False` omits the final data measurement.

    Ancilla records of a longer circuit are then a prefix of the same experiment,
    which the adaptive-stopping bonus uses. The timelike observable is still the
    last cycle's boundary-stabilizer product.
    """

    if rounds < 2:
        raise ValueError("rounds must be >= 2")
    if boundary not in ("X", "Z"):
        raise ValueError(boundary)
    time_basis = "X" if boundary == "Z" else "Z"
    tiles = _tiles_for(distance, boundary)
    data_coords = tuple(
        sorted({c for tile in tiles for c in tile.data_coords}, key=lambda c: (c[1], c[0]))
    )
    data_qubits = tuple(range(len(data_coords)))
    n_tiles = len(tiles)
    circ = stim.Circuit()

    for coord, qubit in zip(data_coords, data_qubits):
        circ.append("QUBIT_COORDS", [qubit], [float(coord[0]), float(coord[1])])
    for tile in tiles:
        circ.append("QUBIT_COORDS", [tile.ancilla], [tile.ancilla_coord[0], tile.ancilla_coord[1]])

    def channel(name: str, qubits: list[int], prob: float) -> None:
        if prob > 0.0 and qubits:
            circ.append(name, qubits, prob)

    circ.append("R", list(data_qubits))
    channel("X_ERROR", list(data_qubits), noise.reset_flip)
    if time_basis == "X":
        circ.append("H", list(data_qubits))
        channel("DEPOLARIZE1", list(data_qubits), noise.clifford)
    circ.append("TICK")

    x_anc = [tile.ancilla for tile in tiles if tile.basis == "X"]
    n_meas = 0
    for round_index in range(rounds):
        channel("DEPOLARIZE1", list(data_qubits), noise.data_idle)
        ancillas = [tile.ancilla for tile in tiles]
        circ.append("R", ancillas)
        channel("X_ERROR", ancillas, noise.reset_flip)
        if x_anc:
            circ.append("H", x_anc)
            channel("DEPOLARIZE1", x_anc, noise.clifford)
        circ.append("TICK")

        for layer in range(4):
            controls: list[int] = []
            targets: list[int] = []
            for tile in tiles:
                data_q = tile.slots[layer]
                if data_q is None:
                    continue
                if tile.basis == "X":
                    controls.append(tile.ancilla)
                    targets.append(data_q)
                else:
                    controls.append(data_q)
                    targets.append(tile.ancilla)
            if not controls:
                continue
            used = controls + targets
            if len(used) != len(set(used)):
                raise RuntimeError("stabilizer schedule uses one qubit twice in a layer")
            pairs = [qubit for pair in zip(controls, targets) for qubit in pair]
            circ.append("CX", pairs)
            channel("DEPOLARIZE2", pairs, noise.clifford)
            circ.append("TICK")

        if x_anc:
            circ.append("H", x_anc)
            channel("DEPOLARIZE1", x_anc, noise.clifford)
        channel("X_ERROR", ancillas, noise.measure_flip)
        circ.append("M", ancillas)
        n_meas += n_tiles

        if round_index == 0:
            for tile_index, tile in enumerate(tiles):
                if tile.basis != time_basis:
                    continue
                circ.append(
                    "DETECTOR",
                    [_rec(tile_index, n_meas)],
                    [tile.ancilla_coord[0], tile.ancilla_coord[1], 0.0],
                )
        else:
            for tile_index, tile in enumerate(tiles):
                now = round_index * n_tiles + tile_index
                prev = now - n_tiles
                circ.append(
                    "DETECTOR",
                    [_rec(now, n_meas), _rec(prev, n_meas)],
                    [tile.ancilla_coord[0], tile.ancilla_coord[1], float(round_index)],
                )
        circ.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
        circ.append("TICK")

    if close_data:
        if time_basis == "X":
            circ.append("H", list(data_qubits))
            channel("DEPOLARIZE1", list(data_qubits), noise.clifford)
        channel("X_ERROR", list(data_qubits), noise.measure_flip)
        circ.append("M", list(data_qubits))
        n_meas += len(data_qubits)
        data_measure_at = {qubit: rounds * n_tiles + i for i, qubit in enumerate(data_qubits)}
        for tile_index, tile in enumerate(tiles):
            if tile.basis != time_basis:
                continue
            targets = [_rec(data_measure_at[q], n_meas) for q in tile.data]
            targets.append(_rec((rounds - 1) * n_tiles + tile_index, n_meas))
            circ.append(
                "DETECTOR",
                targets,
                [tile.ancilla_coord[0], tile.ancilla_coord[1], float(rounds)],
            )

    observable = [
        _rec((rounds - 1) * n_tiles + tile_index, n_meas)
        for tile_index, tile in enumerate(tiles)
        if tile.basis == boundary
    ]
    circ.append("OBSERVABLE_INCLUDE", observable, 0)

    return StabilityExperiment(
        circuit=circ,
        tiles=tiles,
        data_coords=data_coords,
        data_qubits=data_qubits,
        rounds=rounds,
        boundary=boundary,
        distance=distance,
    )
