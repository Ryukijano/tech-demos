"""Keypoints from the VisMatch 1.4.0 RaCo and ALIKED extractors."""

from __future__ import annotations

import os

# Pin threads before torch initializes its pools. run_all.sh exports the same values.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import cv2
import numpy as np
import torch
from vismatch import get_matcher

DETECTORS = ("raco", "aliked")


class CpuDetectors:
    """Two matchers from one VisMatch install. Scoring uses the extractors only."""

    def __init__(self, torch_threads: int, opencv_threads: int) -> None:
        torch.set_num_threads(int(torch_threads))
        cv2.setNumThreads(int(opencv_threads))
        # 2048 is the largest budget in the pre-registered breakpoints.
        # The live budget is written onto the extractor before each call.
        self.raco_matcher = get_matcher("raco-lightglue", device="cpu", max_num_keypoints=2048)
        self.aliked_matcher = get_matcher("aliked-lightglue", device="cpu", max_num_keypoints=2048)
        model_name = self.aliked_matcher.extractor.conf.model_name
        if model_name != "aliked-n16":
            raise RuntimeError(f"expected aliked-n16, VisMatch built {model_name}")
        # Top-k mode. The shipped default threshold of 0.2 sets top_k to -1
        # and can return fewer than the budget. See config.yaml substitution.
        self.aliked_matcher.extractor.dkd.scores_th = -1
        self.aliked_matcher.extractor.dkd.top_k = 200
        self._budget = 200

    def set_budget(self, budget: int) -> None:
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        self._budget = int(budget)
        self.raco_matcher.extractor.conf.max_num_keypoints = int(budget)
        self.aliked_matcher.extractor.dkd.top_k = int(budget)
        self.aliked_matcher.extractor.dkd.n_limit = int(budget)

    def extract(self, name: str, image_rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if name not in DETECTORS:
            raise ValueError(f"unknown detector {name}")
        tensor = _to_unit_tensor(image_rgb_u8)
        if name == "raco":
            feats = self.raco_matcher.extractor.extract(tensor)
        else:
            # resize=None keeps detection on this canvas. The ALIKED default
            # would upscale the long edge to 1024 and scale coordinates back.
            feats = self.aliked_matcher.extractor.extract(tensor, resize=None)
        keypoints = _as_numpy(feats["keypoints"])
        scores = _as_numpy(feats["keypoint_scores"])
        if keypoints.ndim != 2 or keypoints.shape[1] != 2:
            raise RuntimeError(f"{name} keypoints have shape {keypoints.shape}")
        if len(keypoints) != self._budget:
            raise RuntimeError(f"{name} returned {len(keypoints)} keypoints, budget is {self._budget}")
        if not np.isfinite(keypoints).all():
            raise RuntimeError(f"{name} returned non-finite keypoints")
        return keypoints.astype(np.float64), scores.astype(np.float64)


def _to_unit_tensor(image_rgb_u8: np.ndarray) -> torch.Tensor:
    if image_rgb_u8.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB, got {image_rgb_u8.dtype}")
    array = np.ascontiguousarray(image_rgb_u8)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)
    return tensor


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2 and array.shape[0] == 1 and array.shape[1] != 2:
        array = array[0]
    return array
