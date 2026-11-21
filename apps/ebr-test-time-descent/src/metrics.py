"""AUROC and the pre-registered pass rule. No experiment I/O."""

from __future__ import annotations

import math

import numpy as np


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """AUROC of `scores` for binary `labels` (1 = positive). Ties count as 0.5.

    Returns None when either class is empty.
    """

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    n = scores.shape[0]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * ((i + 1) + (j + 1))
        i = j + 1
    sum_pos = float(ranks[labels == 1].sum())
    u = sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def energy_depth(exact: np.ndarray, candidates: list[int], min_class: int) -> int | None:
    """Deepest 1-based segment in `candidates` with enough correct and incorrect boards."""

    exact = np.asarray(exact, dtype=bool)
    for segment in candidates:
        row = exact[segment - 1]
        n_pos = int(row.sum())
        n_neg = int(row.size - n_pos)
        if n_pos >= min_class and n_neg >= min_class:
            return segment
    return None


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def bars_from_runs(
    runs: list[dict],
    *,
    n_boards: int,
    min_class: int,
    gain_min: float,
    regression_boards_max: int,
    auroc_min: float,
    gtan_ratio_min: float,
) -> dict:
    """Apply the four pre-registered bars. `runs` entries need exact, energy, gtan."""

    def acc(run: dict, segment: int) -> float:
        return float(run["exact"][segment - 1])

    seeds = [int(run["seed"]) for run in runs]
    gain = [acc(run, 16) - acc(run, 4) for run in runs]
    gain_mean, gain_std = mean_std(gain)
    acc4 = [acc(run, 4) for run in runs]
    acc8 = [acc(run, 8) for run in runs]
    acc16 = [acc(run, 16) for run in runs]
    m4, s4 = mean_std(acc4)
    m8, s8 = mean_std(acc8)
    m16, s16 = mean_std(acc16)
    tol = regression_boards_max / float(n_boards)
    bar_gain = gain_mean >= gain_min
    bar_mono = (m8 >= m4 - tol - 1e-12) and (m16 >= m8 - tol - 1e-12)

    per_seed_energy = []
    bar_auroc = True
    bar_gtan = True
    for run in runs:
        exact = np.asarray(run["exact_mask"], dtype=bool)
        depth = energy_depth(exact, [16, 8, 4, 2, 1], min_class)
        record = {"seed": int(run["seed"]), "depth": depth, "auroc": None, "gtan_ratio": None}
        if depth is None:
            bar_auroc = False
            bar_gtan = False
            per_seed_energy.append(record)
            continue
        labels = exact[depth - 1].astype(np.int32)
        score = -np.asarray(run["energy"][depth - 1], dtype=np.float64)
        value = auroc(score, labels)
        record["auroc"] = value
        gtan = np.asarray(run["gtan"][depth - 1], dtype=np.float64)
        g_ok = float(gtan[labels == 1].mean())
        g_bad = float(gtan[labels == 0].mean())
        ratio = g_bad / max(g_ok, 1e-8)
        record["gtan_correct"] = g_ok
        record["gtan_incorrect"] = g_bad
        record["gtan_ratio"] = ratio
        record["n_correct"] = int(labels.sum())
        record["n_incorrect"] = int(labels.size - labels.sum())
        if value is None or value < auroc_min:
            bar_auroc = False
        if not math.isfinite(ratio) or ratio < gtan_ratio_min:
            bar_gtan = False
        per_seed_energy.append(record)

    flags = {
        "gain": bar_gain,
        "nondecreasing": bar_mono,
        "energy_auroc": bar_auroc,
        "gtan_ratio": bar_gtan,
    }
    n_pass = sum(bool(v) for v in flags.values())
    if n_pass == 4:
        status = "PASS"
    elif n_pass >= 2:
        status = "PARTIAL"
    else:
        status = "FAIL"
    aurocs = [r["auroc"] for r in per_seed_energy if r["auroc"] is not None]
    auroc_mean, auroc_std = mean_std([float(v) for v in aurocs]) if aurocs else (math.nan, math.nan)
    return {
        "status": status,
        "n_bars_passed": n_pass,
        "bars": flags,
        "seeds": seeds,
        "acc_segment_4": {"mean": m4, "std": s4, "per_seed": acc4},
        "acc_segment_8": {"mean": m8, "std": s8, "per_seed": acc8},
        "acc_segment_16": {"mean": m16, "std": s16, "per_seed": acc16},
        "gain_16_minus_4": {"mean": gain_mean, "std": gain_std, "per_seed": gain},
        "regression_tolerance": tol,
        "energy": per_seed_energy,
        "energy_auroc_mean": auroc_mean,
        "energy_auroc_std": auroc_std,
    }
