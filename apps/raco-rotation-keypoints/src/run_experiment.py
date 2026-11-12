"""Run the pre-registered RaCo vs ALIKED rotation comparison.

Usage:
    python src/run_experiment.py
    python src/run_experiment.py --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from data import ensure_reference_pool
from detect import DETECTORS, CpuDetectors
from geometry import (
    add_noise,
    crop_window,
    homography,
    mnn_inlier_ratio,
    render_view,
    symmetric_repeatability,
    tie_count,
)

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_rotation"


def _load() -> dict[str, Any]:
    return yaml.safe_load((EXP / "config.yaml").read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _checkerboard(height: int = 96, width: int = 128, cell: int = 8) -> np.ndarray:
    yy, xx = np.indices((height, width))
    board = (((xx // cell) + (yy // cell)) % 2) * 255
    return np.stack([board, board, board], axis=-1).astype(np.uint8)


def _sample(paths: list[Path], seed: int, count: int) -> list[Path]:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(paths), size=count, replace=False)
    chosen.sort()
    return [paths[int(index)] for index in chosen]


def _noise(seed: int, image_index: int, angle_deg: float, view: int, case: str) -> np.random.Generator:
    token = sum(ord(char) for char in case) % 10000
    sequence = np.random.SeedSequence(
        [int(seed), int(image_index), int(round(float(angle_deg) * 1000)), int(view), int(token)]
    )
    return np.random.default_rng(sequence)


def _score_pair(
    detectors: CpuDetectors,
    image: np.ndarray,
    angle_deg: float,
    scale: float,
    blur_sigma: float,
    out_size: int,
    budget: int,
    threshold: float,
    sigma: float,
    seed: int,
    image_index: int,
    case: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detectors.set_budget(budget)
    _x0, _y0, side = crop_window(int(image.shape[0]), int(image.shape[1]))
    clean_0 = render_view(image, 0.0, out_size, scale=1.0, blur_sigma=0.0)
    clean_1 = render_view(image, angle_deg, out_size, scale=scale, blur_sigma=blur_sigma)
    view_0 = add_noise(clean_0, _noise(seed, image_index, angle_deg, 0, case), sigma)
    view_1 = add_noise(clean_1, _noise(seed, image_index, angle_deg, 1, case), sigma)
    matrix = homography(angle_deg, out_size, side, scale=scale)
    rows: list[dict[str, Any]] = []
    packed: dict[str, Any] = {"view_0": view_0, "view_1": view_1, "side": side}
    for name in DETECTORS:
        started = time.perf_counter()
        keypoints_0, scores_0 = detectors.extract(name, view_0)
        keypoints_1, scores_1 = detectors.extract(name, view_1)
        elapsed = time.perf_counter() - started
        stats = symmetric_repeatability(keypoints_0, keypoints_1, matrix, threshold, out_size, out_size)
        if not np.isfinite(stats["repeatability"]):
            raise RuntimeError(f"{name} repeatability was not finite at {angle_deg} degrees")
        at_1 = symmetric_repeatability(keypoints_0, keypoints_1, matrix, 1.0, out_size, out_size)
        at_2 = symmetric_repeatability(keypoints_0, keypoints_1, matrix, 2.0, out_size, out_size)
        rows.append(
            {
                "detector": name,
                "angle_deg": float(angle_deg),
                "scale": float(scale),
                "blur_sigma": float(blur_sigma),
                "image_size": int(out_size),
                "keypoint_budget": int(budget),
                "repeatability": stats["repeatability"],
                "repeatability_1px": at_1["repeatability"],
                "repeatability_2px": at_2["repeatability"],
                "mnn_inlier_ratio": mnn_inlier_ratio(
                    keypoints_0, keypoints_1, matrix, threshold, out_size, out_size
                ),
                "n_eligible_0": stats["n_eligible_0"],
                "n_eligible_1": stats["n_eligible_1"],
                "n_keypoints": int(len(keypoints_0)),
                "tie_count_0": tie_count(scores_0),
                "tie_count_1": tie_count(scores_1),
                "seconds": elapsed,
            }
        )
        packed[f"{name}_0"] = keypoints_0
        packed[f"{name}_1"] = keypoints_1
    return rows, packed


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": str(os_cpu()),
        "vismatch": version("vismatch"),
        "torch": version("torch"),
        "kornia": version("kornia"),
        "numpy": version("numpy"),
        "opencv": version("opencv-python-headless"),
    }


def os_cpu() -> int:
    return os.cpu_count() or 1


def _smoke(cfg: dict[str, Any]) -> None:
    detectors = CpuDetectors(cfg["runtime"]["torch_threads"], cfg["runtime"]["opencv_threads"])
    image = _checkerboard()
    rows: list[dict[str, Any]] = []
    for angle in (0.0, 45.0):
        scored, _packed = _score_pair(
            detectors,
            image,
            angle,
            scale=1.0,
            blur_sigma=0.0,
            out_size=64,
            budget=32,
            threshold=float(cfg["pixel_threshold"]),
            sigma=float(cfg["noise_sigma"]),
            seed=0,
            image_index=0,
            case="smoke",
        )
        for row in scored:
            row["role"] = "smoke"
            row["case"] = "smoke"
            row["seed"] = 0
            row["image"] = "checkerboard"
            rows.append(row)
    payload = {
        "note": "Synthetic checkerboard. Not a gate. Thresholds in config.yaml were not fit to these numbers.",
        "versions": _versions(),
        "rows": rows,
    }
    (EXP / "smoke.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"smoke wrote {len(rows)} rows")


def _official(cfg: dict[str, Any]) -> None:
    if _versions()["vismatch"] != cfg["upstream"]["vismatch_version"]:
        raise RuntimeError(f"vismatch {_versions()['vismatch']} is not {cfg['upstream']['vismatch_version']}")
    cache = APP / cfg["hpatches_cache"]
    pool = ensure_reference_pool(cache, int(cfg["pool_size"]))
    detectors = CpuDetectors(cfg["runtime"]["torch_threads"], cfg["runtime"]["opencv_threads"])
    angles = [float(item) for item in cfg["angles_deg"]]
    rows: list[dict[str, Any]] = []
    breakpoints: list[dict[str, Any]] = []
    samples: dict[str, list[str]] = {}
    media: dict[str, Any] | None = None
    media_name = ""
    for seed in cfg["seeds"]:
        chosen = _sample(pool, int(seed), int(cfg["images_per_seed"]))
        samples[str(int(seed))] = [path.stem for path in chosen]
        for image_index, path in enumerate(chosen):
            image = _load_rgb(path)
            for angle in angles:
                scored, packed = _score_pair(
                    detectors,
                    image,
                    angle,
                    scale=1.0,
                    blur_sigma=0.0,
                    out_size=int(cfg["image_size"]),
                    budget=int(cfg["keypoint_budget"]),
                    threshold=float(cfg["pixel_threshold"]),
                    sigma=float(cfg["noise_sigma"]),
                    seed=int(seed),
                    image_index=image_index,
                    case="main",
                )
                for row in scored:
                    row["role"] = "main"
                    row["case"] = "main"
                    row["seed"] = int(seed)
                    row["image"] = path.stem
                    rows.append(row)
                if int(seed) == 0 and image_index == 0:
                    if media is None:
                        media = {
                            "angles": [],
                            "views": [],
                            "raco_0": [],
                            "raco_1": [],
                            "aliked_0": [],
                            "aliked_1": [],
                            "side": packed["side"],
                        }
                        media_name = path.stem
                    media["angles"].append(angle)
                    media["views"].append(np.stack([packed["view_0"], packed["view_1"]], axis=0))
                    for name in DETECTORS:
                        media[f"{name}_0"].append(packed[f"{name}_0"])
                        media[f"{name}_1"].append(packed[f"{name}_1"])
            print(f"seed {seed} {path.stem} done", flush=True)
        if int(seed) == int(cfg["breakpoints"]["seed"]):
            image = _load_rgb(chosen[0])
            for case in cfg["breakpoints"]["cases"]:
                scored, _packed = _score_pair(
                    detectors,
                    image,
                    float(case["angle_deg"]),
                    scale=float(case["scale"]),
                    blur_sigma=float(case["blur_sigma"]),
                    out_size=int(case["image_size"]),
                    budget=int(case["keypoint_budget"]),
                    threshold=float(cfg["pixel_threshold"]),
                    sigma=float(cfg["noise_sigma"]),
                    seed=int(seed),
                    image_index=0,
                    case=str(case["name"]),
                )
                for row in scored:
                    row["role"] = "breakpoint"
                    row["case"] = str(case["name"])
                    row["seed"] = int(seed)
                    row["image"] = chosen[0].stem
                    breakpoints.append(row)
                print(f"breakpoint {case['name']} done", flush=True)

    if media is None:
        raise RuntimeError("no media image was captured")
    npz = EXP / "media_cache.npz"
    np.savez_compressed(
        npz,
        angles=np.asarray(media["angles"], dtype=np.float64),
        views=np.stack(media["views"], axis=0),
        raco_0=np.stack(media["raco_0"], axis=0),
        raco_1=np.stack(media["raco_1"], axis=0),
        aliked_0=np.stack(media["aliked_0"], axis=0),
        aliked_1=np.stack(media["aliked_1"], axis=0),
        side=np.asarray([media["side"]], dtype=np.int32),
        image_size=np.asarray([int(cfg["image_size"])], dtype=np.int32),
        sequence=np.asarray(media_name),
    )
    payload = {
        "config_sha256": _sha256(EXP / "config.yaml"),
        "versions": _versions(),
        "pool": [path.stem for path in pool],
        "samples": samples,
        "media_sequence": media_name,
        "rows": rows,
        "breakpoints": breakpoints,
    }
    (EXP / "raw.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(rows)} main rows and {len(breakpoints)} breakpoint rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = _load()
    if args.smoke:
        _smoke(cfg)
        return
    _official(cfg)


if __name__ == "__main__":
    main()
