"""Geometry and the pass rule. No weights and no network."""

from __future__ import annotations

import unittest

import numpy as np

from analyze import decide
from geometry import (
    add_noise,
    crop_window,
    homography,
    mnn_inlier_ratio,
    output_center,
    render_view,
    symmetric_repeatability,
    tie_count,
    warp_points,
)


class GeometryTest(unittest.TestCase):
    def test_square_side_matches_the_inscribed_diagonal(self) -> None:
        x0, y0, side = crop_window(480, 640)
        self.assertEqual(side, int(np.floor(480 / np.sqrt(2.0))))
        self.assertEqual(side, 339)
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x0 + side, 640)
        self.assertLessEqual(y0 + side, 480)

    def test_homography_agrees_with_the_warp_on_a_smooth_field(self) -> None:
        height, width, out = 90, 120, 64
        yy, xx = np.mgrid[0:height, 0:width]
        image = np.stack(
            [
                (xx * 255 / (width - 1)).astype(np.uint8),
                (yy * 255 / (height - 1)).astype(np.uint8),
                np.full((height, width), 128, dtype=np.uint8),
            ],
            axis=-1,
        )
        _x0, _y0, side = crop_window(height, width)
        view_0 = render_view(image, 0, out)
        for angle in (45, 90, 180):
            view_angle = render_view(image, angle, out)
            matrix = homography(angle, out, side)
            xs = np.linspace(8, out - 9, 8)
            ys = np.linspace(8, out - 9, 8)
            grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
            points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
            warped = warp_points(points, matrix)
            samples_0 = _bilinear(view_0, points)
            samples_angle = _bilinear(view_angle, warped)
            mean_abs = float(np.mean(np.abs(samples_0.astype(np.float64) - samples_angle.astype(np.float64))))
            self.assertLess(mean_abs, 8.0, msg=f"angle {angle} mean abs {mean_abs}")

    def test_scale_homography_moves_a_center_offset_point(self) -> None:
        center = output_center(64, 40)
        matrix = homography(0.0, 64, 40, scale=0.5)
        warped = warp_points(np.array([[center + 10.0, center]]), matrix)[0]
        self.assertAlmostEqual(float(warped[1]), center, places=4)
        self.assertAlmostEqual(float(warped[0]), center + 5.0, places=4)

    def test_identical_points_are_fully_repeatable(self) -> None:
        rng = np.random.default_rng(0)
        points = rng.uniform(10, 54, size=(30, 2))
        matrix = np.eye(3)
        score = symmetric_repeatability(points, points.copy(), matrix, 3.0, 64, 64)
        self.assertEqual(score["repeatability"], 1.0)
        self.assertEqual(mnn_inlier_ratio(points, points.copy(), matrix, 3.0, 64, 64), 1.0)

    def test_a_ten_pixel_shift_is_not_repeatable_at_3px(self) -> None:
        points = np.array([[20.0, 20.0], [40.0, 30.0], [25.0, 45.0]])
        shifted = points + np.array([10.0, 0.0])
        score = symmetric_repeatability(points, shifted, np.eye(3), 3.0, 64, 64)
        self.assertEqual(score["repeatability"], 0.0)
        self.assertEqual(mnn_inlier_ratio(points, shifted, np.eye(3), 3.0, 64, 64), 0.0)

    def test_noise_is_deterministic_and_clipped(self) -> None:
        image = np.full((8, 8, 3), 250, dtype=np.uint8)
        a = add_noise(image, np.random.default_rng(1), 10)
        b = add_noise(image, np.random.default_rng(1), 10)
        self.assertTrue(np.array_equal(a, b))
        self.assertGreaterEqual(int(a.min()), 0)
        self.assertLessEqual(int(a.max()), 255)

    def test_tie_count(self) -> None:
        self.assertEqual(tie_count(np.array([0.1, 0.2, 0.1, 0.3])), 2)
        self.assertEqual(tie_count(np.array([0.1, 0.2])), 0)

    def test_pass_rule_is_inclusive_and_three_way(self) -> None:
        bars = (0.10, 0.10)
        self.assertEqual(decide(0.10, 0.10, *bars), "PASS")
        self.assertEqual(decide(0.10, 0.0, *bars), "PASS")
        self.assertEqual(decide(0.20, 0.11, *bars), "PARTIAL")
        self.assertEqual(decide(0.09, 0.05, *bars), "PARTIAL")
        self.assertEqual(decide(0.09, 0.11, *bars), "FAIL")


def _bilinear(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    samples = np.zeros((len(points), image.shape[2]), dtype=np.float64)
    for index, (x, y) in enumerate(points):
        x = float(np.clip(x, 0, width - 1.001))
        y = float(np.clip(y, 0, height - 1.001))
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
        tx, ty = x - x0, y - y0
        patch = (
            image[y0, x0] * (1 - tx) * (1 - ty)
            + image[y0, x1] * tx * (1 - ty)
            + image[y1, x0] * (1 - tx) * ty
            + image[y1, x1] * tx * ty
        )
        samples[index] = patch
    return samples


if __name__ == "__main__":
    unittest.main()
