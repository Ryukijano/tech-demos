"""FAISS Lloyd and FAISS e-means training, scored the same way.

e-means is the released NumpyEmeans. This module does not reimplement it.
Assignment for the reported objective is a fresh faiss.knn, so both methods
are scored on the centroids they return.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

# OpenMP reads the thread count when the native library loads. run_all.sh and
# the experiment entry point set OMP_NUM_THREADS before this import. Workers
# also call faiss.omp_set_num_threads.
import faiss
from faiss.contrib.e_means import NumpyEmeans

from data import load_split


def as_f32(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)


def initial_centroids(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if k > len(x):
        raise ValueError(f"k={k} is larger than n={len(x)}")
    pick = rng.choice(len(x), size=k, replace=False)
    return as_f32(x[pick])


def mean_squared_error(x: np.ndarray, centroids: np.ndarray) -> tuple[float, np.ndarray]:
    distances, labels = faiss.knn(as_f32(x), as_f32(centroids), 1)
    mse = float(distances.ravel().sum(dtype=np.float64) / len(x))
    return mse, labels.ravel().astype(np.int64, copy=False)


def purity(y: np.ndarray, labels: np.ndarray) -> float:
    total = 0
    for cluster in np.unique(labels):
        members = y[labels == cluster]
        if members.size == 0:
            continue
        _, counts = np.unique(members, return_counts=True)
        total += int(counts.max())
    return float(total) / float(len(y))


def _lloyd(
    x: np.ndarray,
    k: int,
    niter: int,
    seed: int,
    init: np.ndarray,
    *,
    max_points_per_centroid: int | None,
    min_points_per_centroid: int | None,
) -> tuple[np.ndarray, float]:
    kwargs: dict[str, Any] = {
        "niter": int(niter),
        "nredo": 1,
        "seed": int(seed),
        "verbose": False,
        "spherical": False,
        "gpu": False,
    }
    if max_points_per_centroid is not None:
        kwargs["max_points_per_centroid"] = int(max_points_per_centroid)
    if min_points_per_centroid is not None:
        kwargs["min_points_per_centroid"] = int(min_points_per_centroid)
    model = faiss.Kmeans(x.shape[1], k, **kwargs)
    start = time.perf_counter()
    model.train(as_f32(x), init_centroids=as_f32(init))
    elapsed = time.perf_counter() - start
    return as_f32(model.centroids), float(elapsed)


def _emeans(
    x: np.ndarray,
    k: int,
    nepochs: float,
    seed: int,
    init: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    model = NumpyEmeans(
        x.shape[1],
        k,
        nepochs=float(nepochs),
        batch_law="auto",
        seed=int(seed),
        verbose=False,
    )
    start = time.perf_counter()
    model.train(as_f32(x), init_centroids=as_f32(init))
    elapsed = time.perf_counter() - start
    recipe = {key: _jsonable(value) for key, value in model.recipe.items()}
    recipe["n_splits"] = int(getattr(model, "n_splits", 0))
    return as_f32(model.centroids), float(elapsed), recipe


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, bool)):
        return value
    return str(value)


def _score(
    x_train: np.ndarray,
    x_test: np.ndarray,
    centroids: np.ndarray,
    seconds: float,
    y_train: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray]:
    train_mse, train_labels = mean_squared_error(x_train, centroids)
    test_mse, _ = mean_squared_error(x_test, centroids)
    row: dict[str, Any] = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "seconds": seconds,
    }
    if y_train is not None:
        row["train_purity"] = purity(y_train, train_labels)
    return row, train_labels


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    """Train Lloyd once and e-means at each budget on one dataset and seed."""
    threads = int(task["omp_threads"])
    os.environ["OMP_NUM_THREADS"] = str(threads)
    faiss.omp_set_num_threads(threads)
    spec = task["spec"]
    x_train, x_test, y_train = load_split(spec, task.get("mnist"))
    k = int(spec["k"])
    seed = int(task["seed"])
    init = initial_centroids(x_train, k, seed)
    lloyd_centroids, lloyd_seconds = _lloyd(
        x_train,
        k,
        int(task["lloyd_iterations"]),
        seed,
        init,
        max_points_per_centroid=task["max_points_per_centroid"],
        min_points_per_centroid=task["min_points_per_centroid"],
    )
    lloyd_row, lloyd_labels = _score(x_train, x_test, lloyd_centroids, lloyd_seconds, y_train)
    emeans: dict[str, Any] = {}
    gate_labels: np.ndarray | None = None
    if not task.get("run_emeans", True):
        return {
            "role": task["role"],
            "name": task["name"],
            "seed": seed,
            "n": int(x_train.shape[0]),
            "d": int(x_train.shape[1]),
            "k": k,
            "n_test": int(x_test.shape[0]),
            "lloyd": lloyd_row,
            "emeans": {},
        }
    for budget in task["epoch_budgets"]:
        centroids, seconds, recipe = _emeans(x_train, k, float(budget), seed, init)
        row, labels = _score(x_train, x_test, centroids, seconds, y_train)
        row["batch_size"] = int(recipe["batch_size"])
        row["alpha0"] = float(recipe["alpha0"])
        row["batch_law"] = str(recipe["batch_law"])
        row["n_splits"] = int(recipe["n_splits"])
        row["train_mse_ratio"] = row["train_mse"] / lloyd_row["train_mse"]
        row["wall_ratio"] = row["seconds"] / lloyd_row["seconds"] if lloyd_row["seconds"] > 0 else None
        emeans[str(budget)] = row
        if str(budget) == str(task["gate_epochs"]):
            gate_labels = labels
    result: dict[str, Any] = {
        "role": task["role"],
        "name": task["name"],
        "seed": seed,
        "n": int(x_train.shape[0]),
        "d": int(x_train.shape[1]),
        "k": k,
        "n_test": int(x_test.shape[0]),
        "lloyd": lloyd_row,
        "emeans": emeans,
    }
    if task.get("capture_media") and gate_labels is not None:
        result["media"] = _media(x_train, lloyd_labels, gate_labels, seed=seed)
    return result


def _media(
    x_train: np.ndarray,
    lloyd_labels: np.ndarray,
    emeans_labels: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(10_000 + seed)
    take = min(1500, len(x_train))
    choice = np.sort(rng.choice(len(x_train), size=take, replace=False))
    subset = x_train[choice].astype(np.float64)
    subset -= subset.mean(axis=0, keepdims=True)
    # Thin SVD of the Gram matrix keeps the 784-d MNIST case cheap.
    gram = subset.T @ subset
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order[:2]]
    xy = subset @ basis
    return {
        "xy": np.round(xy, 5).tolist(),
        "lloyd": lloyd_labels[choice].astype(int).tolist(),
        "emeans": emeans_labels[choice].astype(int).tolist(),
    }
