"""Shared helpers for the loma-matcher CPU clear-experiment demo.

Thin wrapper around upstream `lomatch` (`from loma import LoMa, LoMaB, LoMaR`).
Weights and sample assets download at run time into a git-ignored cache/work dir.
"""
from __future__ import annotations

import os
import time
import urllib.request
from typing import Literal

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
WORK = os.path.join(APP, "work")
CACHE = os.path.join(WORK, "cache")
ASSETS = os.path.join(WORK, "assets")

ModelName = Literal["LoMa-B", "LoMa-R"]

GITHUB_RAW = {
    "github:davnords/LoMa/assets/0015_A.jpg": "https://raw.githubusercontent.com/davnords/LoMa/main/assets/0015_A.jpg",
    "github:davnords/LoMa/assets/0015_B.jpg": "https://raw.githubusercontent.com/davnords/LoMa/main/assets/0015_B.jpg",
    "github:davnords/LoMa/assets/0022_A.jpg": "https://raw.githubusercontent.com/davnords/LoMa/main/assets/0022_A.jpg",
    "github:davnords/LoMa/assets/0022_B.jpg": "https://raw.githubusercontent.com/davnords/LoMa/main/assets/0022_B.jpg",
    "github:davnords/LoMa/assets/toronto_A.jpg": "https://raw.githubusercontent.com/davnords/LoMa/main/assets/toronto_A.jpg",
    "github:davnords/LoMa/assets/toronto_B.jpg": "https://raw.githubusercontent.com/davnords/LoMa/main/assets/toronto_B.jpg",
}


def ensure_dirs() -> None:
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(ASSETS, exist_ok=True)


def fetch(src: str) -> str:
    """Resolve a github:asset key or local path to a local file path."""
    ensure_dirs()
    if src.startswith("github:"):
        url = GITHUB_RAW[src]
        name = src.split("/")[-1]
        path = os.path.join(ASSETS, name)
        if not os.path.isfile(path) or os.path.getsize(path) < 1000:
            urllib.request.urlretrieve(url, path)
        return path
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    return src


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def load_model(name: ModelName, *, num_keypoints: int = 2048, filter_threshold: float = 0.1):
    """Load LoMa-B or LoMa-R via the upstream API. Weights auto-download."""
    from dataclasses import replace

    from loma import LoMa, LoMaB, LoMaR

    ensure_dirs()
    # Keep torch hub / hub cache under work/ so weights stay out of the repo.
    torch.hub.set_dir(os.path.join(CACHE, "torch_hub"))
    base = LoMaB() if name == "LoMa-B" else LoMaR()
    cfg = replace(base, num_keypoints=num_keypoints, filter_threshold=filter_threshold, compile=False)
    t0 = time.perf_counter()
    model = LoMa(cfg)
    load_s = time.perf_counter() - t0
    model.eval()
    return model, load_s


def match_timed(model, path_a: str, path_b: str):
    """Run model.match and return (kptsA, kptsB, latency_s)."""
    t0 = time.perf_counter()
    kpts_a, kpts_b = model.match(path_a, path_b)
    return kpts_a, kpts_b, time.perf_counter() - t0


def magsac_inliers(kpts_a: np.ndarray, kpts_b: np.ndarray, cfg: dict) -> tuple[int, float]:
    """Return (n_inliers, inlier_ratio) via OpenCV USAC_MAGSAC fundamental matrix."""
    n = int(len(kpts_a))
    if n < 8:
        return 0, 0.0
    method = getattr(cv2, cfg.get("method", "USAC_MAGSAC"))
    _F, mask = cv2.findFundamentalMat(
        kpts_a.astype(np.float32),
        kpts_b.astype(np.float32),
        ransacReprojThreshold=float(cfg.get("ransacReprojThreshold", 0.2)),
        method=method,
        confidence=float(cfg.get("confidence", 0.999999)),
        maxIters=int(cfg.get("maxIters", 10000)),
    )
    if mask is None:
        return 0, 0.0
    nin = int(mask.ravel().sum())
    return nin, nin / max(n, 1)


def rotate_image_file(src_path: str, angle_deg: float, out_path: str) -> str:
    """Rotate image around center; expand canvas; write RGB JPEG; return path."""
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read {src_path}")
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += (nw / 2) - w / 2
    m[1, 2] += (nh / 2) - h / 2
    rot = cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, rot, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return out_path


def draw_matches(path_a: str, path_b: str, kpts_a: np.ndarray, kpts_b: np.ndarray, max_lines: int = 200, seed: int = 0):
    """Side-by-side RGB image with match lines (numpy BGR uint8)."""
    a = cv2.imread(path_a, cv2.IMREAD_COLOR)
    b = cv2.imread(path_b, cv2.IMREAD_COLOR)
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = max(ha, hb)
    canvas = np.zeros((h, wa + wb, 3), dtype=np.uint8)
    canvas[:ha, :wa] = a
    canvas[:hb, wa : wa + wb] = b
    n = min(len(kpts_a), max_lines)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(kpts_a))
    if len(kpts_a) > max_lines:
        idx = rng.choice(len(kpts_a), size=max_lines, replace=False)
    for i in idx[:n]:
        color = tuple(int(c) for c in rng.integers(40, 255, 3))
        x1, y1 = map(int, kpts_a[i])
        x2, y2 = int(kpts_b[i][0] + wa), int(kpts_b[i][1])
        cv2.line(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x2, y2), 2, color, -1, cv2.LINE_AA)
    return canvas


def label_bar(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return out
