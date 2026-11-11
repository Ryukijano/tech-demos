"""Run the pre-registered e-means vs Lloyd comparison and write raw.json.

Usage:
    python src/run_experiment.py
    python src/run_experiment.py --smoke
"""

from __future__ import annotations

import os

# Pin a default before NumPy/FAISS load. main() overwrites this from config
# only when the native pools have not started yet; run_all.sh exports the
# same values first.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import hashlib
import json
import multiprocessing as mp
import platform
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version as dist_version
from pathlib import Path
from typing import Any

import yaml

from cluster import run_task

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_emeans"


def _load() -> dict[str, Any]:
    return yaml.safe_load((EXP / "config.yaml").read_text())


def _pin_threads(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)


def _task(
    cfg: dict[str, Any],
    name: str,
    spec: dict[str, Any],
    seed: int,
    role: str,
    *,
    mnist: tuple | None,
    capture_media: bool,
    default_caps: bool,
) -> dict[str, Any]:
    baseline = cfg["baseline"]
    return {
        "role": role,
        "name": name,
        "spec": spec,
        "seed": int(seed),
        "omp_threads": int(cfg["runtime"]["omp_threads"]),
        "epoch_budgets": [str(item) for item in cfg["epoch_budgets"]],
        "gate_epochs": str(cfg["gate_epochs"]),
        "lloyd_iterations": int(cfg["lloyd_iterations"]),
        "max_points_per_centroid": None if default_caps else int(baseline["max_points_per_centroid"]),
        "min_points_per_centroid": None if default_caps else int(baseline["min_points_per_centroid"]),
        "mnist": mnist if spec["family"] == "mnist" else None,
        "capture_media": capture_media,
        "run_emeans": role != "control",
    }


def _smoke_spec() -> dict[str, Any]:
    return {
        "family": "mixture",
        "n_train": 400,
        "n_test": 80,
        "d": 16,
        "k": 8,
        "dataset_seed": 7,
        "zipf": False,
    }


def _execute(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1 or len(tasks) == 1:
        return [run_task(task) for task in tasks]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        return list(pool.map(run_task, tasks))


def _faiss_version() -> str:
    return dist_version("faiss-cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = _load()
    threads = int(cfg["runtime"]["omp_threads"])
    _pin_threads(threads)
    if args.smoke:
        smoke_cfg = {
            "runtime": cfg["runtime"],
            "baseline": cfg["baseline"],
            "epoch_budgets": ["1"],
            "gate_epochs": "1",
            "lloyd_iterations": 2,
        }
        task = _task(smoke_cfg, "smoke", _smoke_spec(), 0, "smoke", mnist=None, capture_media=False, default_caps=False)
        result = _execute([task], workers=1)[0]
        payload = {"smoke": True, "result": result, "faiss": _faiss_version()}
        (EXP / "smoke.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(
            "smoke train_mse lloyd %.4f e-means %.4f"
            % (result["lloyd"]["train_mse"], result["emeans"]["1"]["train_mse"])
        )
        return

    from data import ensure_mnist

    cache = APP / str(cfg["mnist_cache"])
    mnist_spec = cfg["datasets"]["mnist"]
    mnist = ensure_mnist(cache, list(mnist_spec["urls"]), list(mnist_spec["label_urls"]))
    tasks: list[dict[str, Any]] = []
    for name, spec in cfg["datasets"].items():
        for seed in cfg["seeds"]:
            tasks.append(
                _task(
                    cfg,
                    name,
                    spec,
                    int(seed),
                    "gated",
                    mnist=mnist,
                    capture_media=name == "mixture" and int(seed) == 0,
                    default_caps=False,
                )
            )
    bp_seed = int(cfg["breakpoints"]["seed"])
    for spec in cfg["breakpoints"]["cases"]:
        tasks.append(
            _task(
                cfg,
                str(spec["name"]),
                spec,
                bp_seed,
                "breakpoint",
                mnist=None,
                capture_media=False,
                default_caps=False,
            )
        )
    control = cfg["control"]
    tasks.append(
        _task(
            cfg,
            "library_default_caps",
            cfg["datasets"][control["dataset"]],
            int(control["seed"]),
            "control",
            mnist=mnist,
            capture_media=False,
            default_caps=True,
        )
    )
    workers = int(cfg["runtime"]["workers"])
    print(f"running {len(tasks)} jobs with {workers} processes, {threads} FAISS threads each")
    results = _execute(tasks, workers=workers)
    config_bytes = (EXP / "config.yaml").read_bytes()
    raw = {
        "exp_id": cfg["exp_id"],
        "faiss": _faiss_version(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "machine": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "workers": workers,
            "omp_threads": threads,
        },
        "runs": results,
    }
    (EXP / "raw.json").write_text(json.dumps(raw) + "\n")
    print(f"wrote {EXP / 'raw.json'} ({len(results)} runs)")


if __name__ == "__main__":
    main()
