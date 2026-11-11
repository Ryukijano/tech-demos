"""Datasets for the e-means comparison.

Families follow Appendix D.2 of arXiv:2611.09501 for the synthetic draws.
MNIST is a flat pixel subset, cached under the app and not committed.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np

Family = Literal["mixture", "stretched", "mnist"]


def _never(family: Family) -> None:
    raise AssertionError(f"unhandled dataset family {family!r}")


def stretched_scales(d: int, exponent: float) -> np.ndarray:
    """Per-coordinate std so the variances sum to d (Appendix D.2)."""
    index = np.arange(1, d + 1, dtype=np.float64)
    weights = np.power(index, -exponent)
    variance = d * weights / weights.sum()
    return np.sqrt(variance).astype(np.float64)


def _draw_mixture(
    n: int,
    centers: np.ndarray,
    sigma_w: float,
    rng: np.random.Generator,
    zipf: bool,
) -> np.ndarray:
    k = centers.shape[0]
    if zipf:
        weights = 1.0 / np.arange(1, k + 1, dtype=np.float64)
        weights /= weights.sum()
        assign = rng.choice(k, size=n, p=weights)
    else:
        assign = rng.integers(0, k, size=n)
    noise = rng.standard_normal((n, centers.shape[1])) * sigma_w
    return np.ascontiguousarray(centers[assign] + noise, dtype=np.float32)


def make_synthetic(spec: dict, n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, None]:
    family: Family = spec["family"]
    d = int(spec["d"])
    k = int(spec["k"])
    rng = np.random.default_rng(int(spec["dataset_seed"]))
    if family == "mixture":
        # sigma_mu^2 + sigma_w^2 = 1 and sigma_mu = sqrt(2) sigma_w.
        sigma_w = float(np.sqrt(1.0 / 3.0))
        sigma_mu = float(np.sqrt(2.0 / 3.0))
        centers = rng.standard_normal((k, d)) * sigma_mu
        zipf = bool(spec.get("zipf", False))
        train = _draw_mixture(n_train, centers, sigma_w, rng, zipf)
        test = _draw_mixture(n_test, centers, sigma_w, rng, zipf)
        return train, test, None
    if family == "stretched":
        scale = stretched_scales(d, float(spec["stretch_a"]))
        train_z = rng.standard_normal((n_train, d)) * scale
        test_z = rng.standard_normal((n_test, d)) * scale
        train = np.ascontiguousarray(train_z, dtype=np.float32)
        test = np.ascontiguousarray(test_z, dtype=np.float32)
        return train, test, None
    if family == "mnist":
        _never(family)
    _never(family)
    raise AssertionError("unreachable")


def _download(urls: list[str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    errors: list[str] = []
    for url in urls:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "emeans-vs-kmeans/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                dest.write_bytes(response.read())
            return
        except Exception as exc:  # noqa: BLE001 — try the next mirror
            errors.append(f"{url}: {exc}")
            if dest.exists():
                dest.unlink()
    raise RuntimeError("MNIST download failed: " + "; ".join(errors))


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic = struct.unpack(">I", handle.read(4))[0]
        if magic == 2051:
            count, rows, cols = struct.unpack(">III", handle.read(12))
            raw = np.frombuffer(handle.read(), dtype=np.uint8)
            return raw.reshape(count, rows * cols).copy()
        if magic == 2049:
            count = struct.unpack(">I", handle.read(4))[0]
            raw = np.frombuffer(handle.read(), dtype=np.uint8)
            return raw.reshape(count).copy()
    raise ValueError(f"unrecognized IDX magic in {path}")


def ensure_mnist(cache: Path, image_urls: list[str], label_urls: list[str]) -> tuple[np.ndarray, np.ndarray]:
    images_path = cache / "train-images-idx3-ubyte.gz"
    labels_path = cache / "train-labels-idx1-ubyte.gz"
    _download(image_urls, images_path)
    _download(label_urls, labels_path)
    images = _read_idx(images_path)
    labels = _read_idx(labels_path)
    if images.shape != (60000, 784) or labels.shape != (60000,):
        raise RuntimeError(f"unexpected MNIST shapes {images.shape} {labels.shape}")
    return images, labels


def make_mnist(
    spec: dict,
    images: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(spec["n_train"])
    n_test = int(spec["n_test"])
    if n_train + n_test > len(images):
        raise ValueError("MNIST subset is larger than the training file")
    rng = np.random.default_rng(int(spec["dataset_seed"]))
    order = rng.permutation(len(images))
    train_idx = order[:n_train]
    test_idx = order[n_train : n_train + n_test]
    train = np.ascontiguousarray(images[train_idx], dtype=np.float32) / 255.0
    test = np.ascontiguousarray(images[test_idx], dtype=np.float32) / 255.0
    y_train = labels[train_idx].astype(np.int64)
    return train, test, y_train


def load_split(
    spec: dict,
    mnist: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    family: Family = spec["family"]
    if family == "mnist":
        if mnist is None:
            raise RuntimeError("MNIST cache was not loaded")
        return make_mnist(spec, mnist[0], mnist[1])
    if family in ("mixture", "stretched"):
        return make_synthetic(spec, int(spec["n_train"]), int(spec["n_test"]))
    _never(family)
    raise AssertionError("unreachable")
