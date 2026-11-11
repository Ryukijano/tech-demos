"""Locks on the released FAISS e-means recipe and the shared scorer."""

from __future__ import annotations

import math
import unittest

import faiss
import numpy as np
from faiss.contrib.e_means import NumpyEmeans

from cluster import as_f32, initial_centroids, mean_squared_error
from data import stretched_scales


def _round_power_of_two(value: float) -> int:
    if value <= 1:
        return 1
    low = 2 ** int(math.floor(math.log2(value)))
    high = 2 * low
    return low if value / low < high / value else high


def _published_batch(n: int, k: int, nepochs: float, law: str) -> int:
    laws = {
        "throughput": (0.035134, 0.134749, 0.876473, 0.813344),
        "quality": (0.00439175, 0.134749, 0.876473, 0.813344),
    }
    coefficient, k_exp, n_exp, iter_exp = laws[law]
    raw = coefficient * k**k_exp * n**n_exp * nepochs**iter_exp
    batch = _round_power_of_two(raw)
    if law == "throughput":
        batch = min(batch, _round_power_of_two(n / 2))
    else:
        batch = min(batch, n // 2)
    return max(1, min(batch, n))


class RecipeTest(unittest.TestCase):
    def test_knn_is_squared_l2(self) -> None:
        points = as_f32(np.array([[0.0], [3.0]]))
        centroids = as_f32(np.array([[0.0], [1.0]]))
        mse, labels = mean_squared_error(points, centroids)
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertAlmostEqual(mse, 2.0, places=5)

    def test_stretched_variances_sum_to_d(self) -> None:
        scale = stretched_scales(64, 1.0)
        self.assertAlmostEqual(float(np.sum(scale**2)), 64.0, places=6)

    def test_throughput_law_matches_library(self) -> None:
        rng = np.random.default_rng(0)
        x = as_f32(rng.standard_normal((5000, 32)))
        model = NumpyEmeans(32, 64, nepochs=2.5, seed=0, verbose=False)
        init = initial_centroids(x, 64, 0)
        model.train(x, init_centroids=init)
        self.assertEqual(model.recipe["batch_law"], "throughput")
        self.assertEqual(int(model.recipe["batch_size"]), _published_batch(5000, 64, 2.5, "throughput"))
        self.assertGreater(float(model.recipe["alpha0"]), 0.0)
        self.assertLessEqual(float(model.recipe["alpha0"]), 1.0)

    def test_low_dimension_selects_quality(self) -> None:
        rng = np.random.default_rng(1)
        x = as_f32(rng.standard_normal((800, 8)))
        model = NumpyEmeans(8, 8, nepochs=1.0, seed=1, verbose=False)
        model.train(x, init_centroids=initial_centroids(x, 8, 1))
        self.assertEqual(model.recipe["batch_law"], "quality")
        self.assertEqual(int(model.recipe["batch_size"]), _published_batch(800, 8, 1.0, "quality"))

    def test_same_seed_is_deterministic(self) -> None:
        rng = np.random.default_rng(2)
        x = as_f32(rng.standard_normal((600, 16)))
        init = initial_centroids(x, 12, 2)
        first = NumpyEmeans(16, 12, nepochs=1.0, seed=2, verbose=False).train(x, init_centroids=init)[0]
        second = NumpyEmeans(16, 12, nepochs=1.0, seed=2, verbose=False).train(x, init_centroids=init)[0]
        np.testing.assert_allclose(first, second, rtol=0, atol=0)

    def test_faiss_import_is_cpu_build(self) -> None:
        self.assertTrue(hasattr(faiss, "Kmeans"))
        model = faiss.Kmeans(4, 2, niter=1, gpu=False)
        self.assertFalse(model.gpu)


if __name__ == "__main__":
    unittest.main()
