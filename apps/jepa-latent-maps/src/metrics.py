"""Held-out content and geometry metrics from Ren et al., Section 3.

Content: ridge R² of state factors (here, free-block position), fit and scored
on separate trajectories. Geometry: Spearman correlation of latent L2 distance
with maze geodesic distance, plus path jaggedness (Equation 4) and the
participation ratio (Equation 5).
"""

from __future__ import annotations

import numpy as np

from world import free_object_mask, geodesic_table, normalize_rc


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks for ties. 0-based average rank is enough for Pearson-on-ranks."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.shape[0], dtype=np.float64)
    sorted_x = x[order]
    n = x.shape[0]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + (j - 1))
        i = j
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ. A constant input (collapsed latent distances) returns 0."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or a.size != b.size:
        return float("nan")
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    ra = rankdata(a)
    rb = rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom < 1e-12:
        return 0.0
    return float((ra * rb).sum() / denom)


def ridge_r2(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_score: np.ndarray,
    y_score: np.ndarray,
    alpha: float,
) -> float:
    """Pooled multi-output R² of a ridge readout. Intercept is unpenalized.

    Latent columns are standardized on the fit split. The penalty α multiplies
    the identity on those standardized columns (the report uses ridge λ = 1).
    """
    x_fit = np.asarray(x_fit, dtype=np.float64)
    x_score = np.asarray(x_score, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_fit.ndim == 1:
        y_fit = y_fit[:, None]
        y_score = y_score[:, None]
    if x_fit.shape[0] < x_fit.shape[1] + 2 or x_score.shape[0] < 5:
        return float("nan")
    mu = x_fit.mean(axis=0)
    sd = x_fit.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    xf = (x_fit - mu) / sd
    xs = (x_score - mu) / sd
    y_mu = y_fit.mean(axis=0)
    yc = y_fit - y_mu
    dim = xf.shape[1]
    gram = xf.T @ xf + float(alpha) * np.eye(dim)
    weight = np.linalg.solve(gram, xf.T @ yc)
    pred = xs @ weight + y_mu
    ss_res = float(((y_score - pred) ** 2).sum())
    ss_tot = float(((y_score - y_score.mean(axis=0)) ** 2).sum())
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def jaggedness(latent: np.ndarray, window: int) -> float:
    """Path-length ratio of Equation 4. Smooth paths are near 1; larger is more jagged.

    The moving average is trailing, and the raw path is aligned to the same frames.
    """
    z = np.asarray(latent, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < window + 2 or window < 2:
        return float("nan")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.stack(
        [np.convolve(z[:, i], kernel, mode="valid") for i in range(z.shape[1])],
        axis=1,
    )
    raw = z[window - 1 :]
    if raw.shape[0] != smoothed.shape[0]:
        return float("nan")
    num = float(np.linalg.norm(np.diff(raw, axis=0), axis=1).sum())
    den = float(np.linalg.norm(np.diff(smoothed, axis=0), axis=1).sum())
    if den < 1e-8:
        return float("nan")
    return num / den


def participation_ratio(latent: np.ndarray) -> float:
    z = np.asarray(latent, dtype=np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    evals = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    denom = float((evals ** 2).sum())
    if denom < 1e-18:
        return float("nan")
    return float((evals.sum() ** 2) / denom)


def geodesic_spearman(
    latent: np.ndarray,
    agent: np.ndarray,
    maze: np.ndarray,
    *,
    n_pairs: int,
    pair_seed: int,
) -> float:
    index, dist = geodesic_table(maze)
    ids = np.asarray([index[(int(r), int(c))] for r, c in agent], dtype=np.int64)
    rng = np.random.default_rng(pair_seed)
    n = latent.shape[0]
    left = rng.integers(0, n, size=n_pairs)
    right = rng.integers(0, n, size=n_pairs)
    keep = left != right
    left = left[keep]
    right = right[keep]
    latent_d = np.linalg.norm(latent[left] - latent[right], axis=1)
    maze_d = dist[ids[left], ids[right]]
    return spearman(latent_d, maze_d)


def evaluate_latents(
    latent_fit: np.ndarray,
    agent_fit: np.ndarray,
    block_fit: np.ndarray,
    latent_score: np.ndarray,
    agent_score: np.ndarray,
    block_score: np.ndarray,
    latent_route: np.ndarray,
    maze: np.ndarray,
    *,
    ridge_alpha: float,
    min_chebyshev: int,
    jag_window: int,
    n_pairs: int,
    pair_seed: int,
) -> dict[str, float]:
    n = int(maze.shape[0])
    y_agent_fit = normalize_rc(agent_fit, n)
    y_agent_score = normalize_rc(agent_score, n)
    y_block_fit = normalize_rc(block_fit, n)
    y_block_score = normalize_rc(block_score, n)
    free_fit = free_object_mask(agent_fit, block_fit, min_chebyshev)
    free_score = free_object_mask(agent_score, block_score, min_chebyshev)
    return {
        "agent_r2": ridge_r2(latent_fit, y_agent_fit, latent_score, y_agent_score, ridge_alpha),
        "free_object_r2": ridge_r2(
            latent_fit[free_fit],
            y_block_fit[free_fit],
            latent_score[free_score],
            y_block_score[free_score],
            ridge_alpha,
        ),
        "object_r2_all": ridge_r2(
            latent_fit, y_block_fit, latent_score, y_block_score, ridge_alpha
        ),
        "geodesic_spearman": geodesic_spearman(
            latent_score, agent_score, maze, n_pairs=n_pairs, pair_seed=pair_seed
        ),
        "jaggedness": jaggedness(latent_route, jag_window),
        "participation_ratio": participation_ratio(latent_score),
        "n_free_fit": float(free_fit.sum()),
        "n_free_score": float(free_score.sum()),
    }


def observation_ceiling(
    obs_fit: np.ndarray,
    agent_fit: np.ndarray,
    block_fit: np.ndarray,
    obs_score: np.ndarray,
    agent_score: np.ndarray,
    block_score: np.ndarray,
    *,
    ridge_alpha: float,
    min_chebyshev: int,
    grid_size: int,
) -> dict[str, float]:
    """Linear readout of the noisy pixels themselves. Report-only sanity check."""
    x_fit = obs_fit.reshape(obs_fit.shape[0], -1)
    x_score = obs_score.reshape(obs_score.shape[0], -1)
    free_fit = free_object_mask(agent_fit, block_fit, min_chebyshev)
    free_score = free_object_mask(agent_score, block_score, min_chebyshev)
    return {
        "pixel_agent_r2": ridge_r2(
            x_fit,
            normalize_rc(agent_fit, grid_size),
            x_score,
            normalize_rc(agent_score, grid_size),
            ridge_alpha,
        ),
        "pixel_free_object_r2": ridge_r2(
            x_fit[free_fit],
            normalize_rc(block_fit, grid_size)[free_fit],
            x_score[free_score],
            normalize_rc(block_score, grid_size)[free_score],
            ridge_alpha,
        ),
    }


def self_check() -> None:
    perfect = spearman(np.arange(6, dtype=np.float64), np.arange(6, dtype=np.float64))
    reverse = spearman(np.arange(6, dtype=np.float64), np.arange(5, -1, -1, dtype=np.float64))
    if abs(perfect - 1.0) > 1e-9 or abs(reverse + 1.0) > 1e-9:
        raise AssertionError(f"spearman self-check failed: {perfect}, {reverse}")
    if spearman(np.zeros(8), np.arange(8.0)) != 0.0:
        raise AssertionError("collapsed spearman should be 0")
    line = np.linspace(0.0, 1.0, 40)[:, None] * np.array([[1.0, -0.25]])
    jag = jaggedness(line, 3)
    if not 0.98 <= jag <= 1.02:
        raise AssertionError(f"line jaggedness {jag} is not ~1")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    coef = np.array([[0.4, -0.2], [0.1, 0.7], [-0.3, 0.2]])
    y = x @ coef + 0.05
    got = ridge_r2(x[:120], y[:120], x[120:], y[120:], 1.0)
    if got < 0.95:
        raise AssertionError(f"ridge self-check R² {got}")
