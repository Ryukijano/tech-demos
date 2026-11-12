"""Crop, rotate, and score keypoints. No learned models."""

from __future__ import annotations

import cv2
import numpy as np


def crop_window(height: int, width: int) -> tuple[int, int, int]:
    """Largest center square that stays inside the photo at every angle.

    The square's diagonal equals the short side of the photograph, so
    rotating it about its center never asks for pixels outside the photo.
    """
    side = int(np.floor(min(height, width) / np.sqrt(2.0)))
    if side < 8:
        raise ValueError(f"image {width}x{height} is too small for a rotatable square")
    x0 = (width - side) // 2
    y0 = (height - side) // 2
    return x0, y0, side


def output_center(out_size: int, side: int) -> float:
    """Where cv2.resize puts the crop center, in output pixels.

    OpenCV maps a source coordinate ``s`` to ``(s + 0.5) * (out / side) - 0.5``.
    """
    if out_size < 8 or side < 8:
        raise ValueError(f"out_size {out_size} and side {side} must be at least 8")
    return (side / 2.0 + 0.5) * (out_size / float(side)) - 0.5


def homography(angle_deg: float, out_size: int, side: int, scale: float = 1.0) -> np.ndarray:
    """Map view 0 onto the rotated (and optionally scaled) view, 3x3."""
    center = output_center(out_size, side)
    affine = cv2.getRotationMatrix2D((float(center), float(center)), float(angle_deg), float(scale))
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2] = affine
    return matrix


def render_view(
    image_rgb: np.ndarray,
    angle_deg: float,
    out_size: int,
    scale: float = 1.0,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """Rotate about the crop center, crop, resize, then optional scale and blur."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3, got {image_rgb.shape}")
    height, width = image_rgb.shape[:2]
    x0, y0, side = crop_window(height, width)
    center = (x0 + side / 2.0, y0 + side / 2.0)
    affine = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    rotated = cv2.warpAffine(
        image_rgb,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    crop = rotated[y0 : y0 + side, x0 : x0 + side]
    view = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    if scale != 1.0:
        mapped = output_center(out_size, side)
        scale_affine = cv2.getRotationMatrix2D((float(mapped), float(mapped)), 0.0, float(scale))
        view = cv2.warpAffine(
            view,
            scale_affine,
            (out_size, out_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    if blur_sigma > 0:
        view = cv2.GaussianBlur(view, (0, 0), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))
    return view


def add_noise(image_u8: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    noise = rng.normal(0.0, float(sigma), size=image_u8.shape)
    noisy = image_u8.astype(np.float32) + noise.astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def warp_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    hom = np.concatenate([points.astype(np.float64), np.ones((len(points), 1))], axis=1)
    mapped = hom @ matrix.T
    return mapped[:, :2] / mapped[:, 2:3]


def _inside(points: np.ndarray, height: int, width: int, margin: float) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0,), dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    return (x >= margin) & (x < width - margin) & (y >= margin) & (y < height - margin)


def nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(target) == 0:
        return np.full((len(source),), np.inf)
    delta = source[:, None, :].astype(np.float64) - target[None, :, :].astype(np.float64)
    return np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))


def symmetric_repeatability(
    keypoints_0: np.ndarray,
    keypoints_1: np.ndarray,
    matrix: np.ndarray,
    threshold: float,
    height: int,
    width: int,
) -> dict[str, float]:
    """Fraction of eligible points that have a neighbor within ``threshold``."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    warped_0 = warp_points(keypoints_0, matrix)
    warped_1 = warp_points(keypoints_1, np.linalg.inv(matrix))
    eligible_0 = _inside(keypoints_0, height, width, threshold) & _inside(warped_0, height, width, threshold)
    eligible_1 = _inside(keypoints_1, height, width, threshold) & _inside(warped_1, height, width, threshold)
    in_image_1 = _inside(keypoints_1, height, width, 0.0)
    in_image_0 = _inside(keypoints_0, height, width, 0.0)
    distances_0 = nearest_distances(warped_0[eligible_0], keypoints_1[in_image_1])
    distances_1 = nearest_distances(warped_1[eligible_1], keypoints_0[in_image_0])
    n0 = int(eligible_0.sum())
    n1 = int(eligible_1.sum())
    hits_0 = int(np.sum(distances_0 < threshold)) if n0 else 0
    hits_1 = int(np.sum(distances_1 < threshold)) if n1 else 0
    denom = n0 + n1
    if denom == 0:
        score = float("nan")
    else:
        score = (hits_0 + hits_1) / denom
    return {
        "repeatability": float(score),
        "n_eligible_0": n0,
        "n_eligible_1": n1,
        "n_repeatable_0": hits_0,
        "n_repeatable_1": hits_1,
    }


def mnn_inlier_ratio(
    keypoints_0: np.ndarray,
    keypoints_1: np.ndarray,
    matrix: np.ndarray,
    threshold: float,
    height: int,
    width: int,
) -> float:
    """Mutual nearest neighbors within ``threshold``, over the smaller set."""
    keep_0 = _inside(keypoints_0, height, width, 0.0)
    keep_1 = _inside(keypoints_1, height, width, 0.0)
    points_0 = keypoints_0[keep_0]
    points_1 = keypoints_1[keep_1]
    if len(points_0) == 0 or len(points_1) == 0:
        return 0.0
    warped = warp_points(points_0, matrix)
    delta = warped[:, None, :] - points_1[None, :, :]
    dist = np.sqrt(np.sum(delta * delta, axis=2))
    nn_01 = np.argmin(dist, axis=1)
    nn_10 = np.argmin(dist, axis=0)
    mutual = nn_10[nn_01] == np.arange(len(points_0))
    close = dist[np.arange(len(points_0)), nn_01] < threshold
    inliers = int(np.sum(mutual & close))
    return inliers / float(min(len(points_0), len(points_1)))


def tie_count(scores: np.ndarray) -> int:
    if len(scores) == 0:
        return 0
    _unique, counts = np.unique(scores.astype(np.float64), return_counts=True)
    return int(np.sum(counts[counts > 1]))
