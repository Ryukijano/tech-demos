"""Closed-form Markov reward process used as the Value Forcing toy.

The graph is small enough that every return law is a finite mixture of atoms
(a geometric series for the self-loop state, two-point mixtures elsewhere).
Means are checked against the linear Bellman equation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Primary state order. Extra chain states, when chain_depth != 1, are appended
# and named C0 (farthest from MIX) ... C{L-1} (the state that transitions to MIX).
PRIMARY_CORE = ("MIX", "HIGH", "HIGH2", "LOW", "LOW2", "SINK", "WIDE")


@dataclass(frozen=True)
class MRP:
    gamma: float
    names: tuple[str, ...]
    # edges[s] = tuple of (prob, reward, next_index)
    edges: tuple[tuple[tuple[float, float, int], ...], ...]
    values: tuple[np.ndarray, ...]
    probs: tuple[np.ndarray, ...]
    mean: np.ndarray
    std: np.ndarray
    chain_depth: int

    @property
    def n_states(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def farthest_name(self) -> str | None:
        """State whose return must bootstrap through the longest chain."""
        if self.chain_depth <= 0:
            return "MIX"
        if self.chain_depth == 1 and "CHAIN" in self.names:
            return "CHAIN"
        return "C0"


def _merge(values: np.ndarray, probs: np.ndarray, decimals: int = 10) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        raise ValueError("empty atom list")
    key = np.round(values, decimals=decimals)
    order = np.argsort(key, kind="mergesort")
    key = key[order]
    probs = probs[order]
    uniq, start = np.unique(key, return_index=True)
    merged_p = np.add.reduceat(probs, start)
    merged_v = np.add.reduceat(values[order], start) / np.maximum(np.diff(np.append(start, len(key))), 1)
    # values were identical within a bucket up to rounding; use the key
    merged_v = uniq
    total = float(merged_p.sum())
    if total <= 0:
        raise ValueError("non-positive atom mass")
    return merged_v.astype(np.float64), (merged_p / total).astype(np.float64)


def _shift_scale(values: np.ndarray, probs: np.ndarray, reward: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    return _merge(reward + gamma * values, probs.copy())


def _mixture(parts: list[tuple[float, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    vs: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for weight, values, probs in parts:
        vs.append(values)
        ps.append(weight * probs)
    return _merge(np.concatenate(vs), np.concatenate(ps))


def _geometric(p_stay: float, r_stay: float, r_exit: float, gamma: float, mass_eps: float = 1e-15) -> tuple[np.ndarray, np.ndarray]:
    """K self-loops then exit. P(K=k) = p_stay^k * (1-p_stay)."""
    if not 0.0 <= p_stay < 1.0:
        raise ValueError(f"p_stay must be in [0, 1), got {p_stay}")
    q = 1.0 - p_stay
    if p_stay == 0.0:
        return np.array([r_exit], dtype=np.float64), np.array([1.0], dtype=np.float64)
    max_k = int(np.ceil(np.log(mass_eps) / np.log(p_stay))) + 1
    max_k = int(min(max(max_k, 1), 20000))
    ks = np.arange(max_k + 1, dtype=np.float64)
    probs = (p_stay ** ks) * q
    if abs(gamma - 1.0) < 1e-15:
        partial = ks * r_stay
    else:
        # sum_{i=0}^{k-1} γ^i r_stay = r_stay (1-γ^k)/(1-γ); k=0 -> 0
        geom = np.where(ks == 0.0, 0.0, (1.0 - np.power(gamma, ks)) / (1.0 - gamma))
        partial = r_stay * geom
    values = partial + np.power(gamma, ks) * r_exit
    tail = float(p_stay ** (max_k + 1))
    if tail > 0:
        # Dump the unreachable tail onto the last atom, then the merge renormalizes.
        probs = probs.copy()
        probs[-1] += tail
    return _merge(values, probs)


def _mean_std(values: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    mean = float(np.sum(probs * values))
    var = float(np.sum(probs * (values - mean) ** 2))
    return mean, float(np.sqrt(max(var, 0.0)))


def bellman_means(edges: tuple[tuple[tuple[float, float, int], ...], ...], gamma: float) -> np.ndarray:
    n = len(edges)
    trans = np.zeros((n, n), dtype=np.float64)
    reward = np.zeros(n, dtype=np.float64)
    for s, outs in enumerate(edges):
        mass = 0.0
        for prob, r, nxt in outs:
            trans[s, nxt] += prob
            reward[s] += prob * r
            mass += prob
        if abs(mass - 1.0) > 1e-9:
            raise ValueError(f"state {s} transition mass {mass}")
    eye = np.eye(n, dtype=np.float64)
    return np.linalg.solve(eye - gamma * trans, reward)


def build_mrp(
    gamma: float,
    *,
    high_reward: float = 1.5,
    chain_reward: float = 0.4,
    wide_p_stay: float = 0.55,
    wide_r_stay: float = 1.0,
    wide_r_exit: float = 0.0,
    chain_depth: int = 1,
) -> MRP:
    """Build the branching MRP and its exact return atoms.

    MIX splits 50/50 into HIGH (two reward steps of ``high_reward``) and LOW
    (two zero-reward steps). WIDE is a geometric self-loop. A deterministic
    chain of length ``chain_depth`` feeds MIX; the farthest state carries
    ``chain_reward`` and the intermediate edges carry 0.
    """
    if chain_depth < 0:
        raise ValueError("chain_depth must be >= 0")
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie in (0, 1)")

    names: list[str] = list(PRIMARY_CORE)
    if chain_depth == 1:
        names.append("CHAIN")
    elif chain_depth > 1:
        names.extend(f"C{k}" for k in range(chain_depth))

    idx = {name: i for i, name in enumerate(names)}
    n = len(names)
    raw: list[list[tuple[float, float, int]]] = [[] for _ in range(n)]
    raw[idx["SINK"]] = [(1.0, 0.0, idx["SINK"])]
    raw[idx["HIGH2"]] = [(1.0, high_reward, idx["SINK"])]
    raw[idx["HIGH"]] = [(1.0, high_reward, idx["HIGH2"])]
    raw[idx["LOW2"]] = [(1.0, 0.0, idx["SINK"])]
    raw[idx["LOW"]] = [(1.0, 0.0, idx["LOW2"])]
    raw[idx["MIX"]] = [(0.5, 0.0, idx["HIGH"]), (0.5, 0.0, idx["LOW"])]
    raw[idx["WIDE"]] = [
        (wide_p_stay, wide_r_stay, idx["WIDE"]),
        (1.0 - wide_p_stay, wide_r_exit, idx["SINK"]),
    ]
    if chain_depth == 1:
        raw[idx["CHAIN"]] = [(1.0, chain_reward, idx["MIX"])]
    elif chain_depth > 1:
        raw[idx["C0"]] = [(1.0, chain_reward, idx["C1"])]
        for k in range(1, chain_depth - 1):
            raw[idx[f"C{k}"]] = [(1.0, 0.0, idx[f"C{k + 1}"])]
        raw[idx[f"C{chain_depth - 1}"]] = [(1.0, 0.0, idx["MIX"])]

    edges = tuple(tuple(outs) for outs in raw)

    sink_v, sink_p = np.array([0.0]), np.array([1.0])
    high2_v, high2_p = _shift_scale(sink_v, sink_p, high_reward, gamma)
    high_v, high_p = _shift_scale(high2_v, high2_p, high_reward, gamma)
    low2_v, low2_p = _shift_scale(sink_v, sink_p, 0.0, gamma)
    low_v, low_p = _shift_scale(low2_v, low2_p, 0.0, gamma)
    mix_v, mix_p = _mixture(
        [
            (0.5, *_shift_scale(high_v, high_p, 0.0, gamma)),
            (0.5, *_shift_scale(low_v, low_p, 0.0, gamma)),
        ]
    )
    wide_v, wide_p = _geometric(wide_p_stay, wide_r_stay, wide_r_exit, gamma)

    atoms: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "SINK": (sink_v, sink_p),
        "HIGH2": (high2_v, high2_p),
        "HIGH": (high_v, high_p),
        "LOW2": (low2_v, low2_p),
        "LOW": (low_v, low_p),
        "MIX": (mix_v, mix_p),
        "WIDE": (wide_v, wide_p),
    }
    if chain_depth == 1:
        atoms["CHAIN"] = _shift_scale(mix_v, mix_p, chain_reward, gamma)
    elif chain_depth > 1:
        # C{L-1} backs up MIX with reward 0; walk back to C0, which holds chain_reward.
        dist_v, dist_p = mix_v, mix_p
        for k in range(chain_depth - 1, -1, -1):
            reward = chain_reward if k == 0 else 0.0
            dist_v, dist_p = _shift_scale(dist_v, dist_p, reward, gamma)
            atoms[f"C{k}"] = (dist_v, dist_p)

    values = tuple(atoms[name][0] for name in names)
    probs = tuple(atoms[name][1] for name in names)
    means = np.array([_mean_std(v, p)[0] for v, p in zip(values, probs)], dtype=np.float64)
    stds = np.array([_mean_std(v, p)[1] for v, p in zip(values, probs)], dtype=np.float64)
    solved = bellman_means(edges, gamma)
    gap = float(np.max(np.abs(solved - means)))
    if gap > 1e-6:
        raise AssertionError(f"atom means disagree with the linear Bellman solution by {gap:.3e}")

    return MRP(
        gamma=float(gamma),
        names=tuple(names),
        edges=edges,
        values=values,
        probs=probs,
        mean=means,
        std=stds,
        chain_depth=int(chain_depth),
    )


def sample_dataset(mrp: MRP, n_transitions: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Stratified offline transitions: about the same count from every state.

    The behavior kernel is the MRP itself (on-policy evaluation of the chain).
    """
    if n_transitions < mrp.n_states:
        raise ValueError(f"need at least one transition per state, got {n_transitions} for {mrp.n_states} states")
    base = n_transitions // mrp.n_states
    extra = n_transitions % mrp.n_states
    counts = np.full(mrp.n_states, base, dtype=int)
    if extra:
        pick = rng.choice(mrp.n_states, size=extra, replace=False)
        counts[pick] += 1
    states: list[int] = []
    rewards: list[float] = []
    nxts: list[int] = []
    for s, count in enumerate(counts):
        outs = mrp.edges[s]
        probs = np.array([e[0] for e in outs], dtype=np.float64)
        choice = rng.choice(len(outs), size=int(count), p=probs)
        for k in choice:
            _p, r, sp = outs[int(k)]
            states.append(s)
            rewards.append(r)
            nxts.append(sp)
    s_arr = np.asarray(states, dtype=np.int64)
    r_arr = np.asarray(rewards, dtype=np.float64)
    sp_arr = np.asarray(nxts, dtype=np.int64)
    mix = mrp.index("MIX")
    high = mrp.index("HIGH")
    mix_rows = s_arr == mix
    mix_n = int(mix_rows.sum())
    mix_high = float(np.mean(sp_arr[mix_rows] == high)) if mix_n else float("nan")
    return {
        "s": s_arr,
        "r": r_arr,
        "sp": sp_arr,
        "counts": counts,
        "mix_high_fraction": np.array([mix_high], dtype=np.float64),
    }


def atom_quantiles(values: np.ndarray, probs: np.ndarray, us: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    v = values[order]
    cdf = np.cumsum(probs[order])
    cdf[-1] = 1.0
    idx = np.searchsorted(cdf, us, side="left")
    idx = np.clip(idx, 0, v.shape[0] - 1)
    return v[idx]


def w1_to_atoms(samples: np.ndarray, values: np.ndarray, probs: np.ndarray) -> float:
    """1-Wasserstein distance via mid-quantile Riemann sum. Samples are the prediction."""
    ordered = np.sort(np.asarray(samples, dtype=np.float64))
    n = ordered.shape[0]
    us = (np.arange(n, dtype=np.float64) + 0.5) / n
    truth = atom_quantiles(values, probs, us)
    return float(np.mean(np.abs(ordered - truth)))


def describe(mrp: MRP) -> str:
    lines = [f"gamma={mrp.gamma:.4f}  states={mrp.n_states}  chain_depth={mrp.chain_depth}"]
    for i, name in enumerate(mrp.names):
        lines.append(f"  {name:6s}  mean={mrp.mean[i]:8.4f}  std={mrp.std[i]:8.4f}  atoms={mrp.values[i].shape[0]}")
    return "\n".join(lines)
