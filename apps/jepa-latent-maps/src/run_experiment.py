"""Run the pre-registered gridworld JEPA experiment.

Usage:
    python src/run_experiment.py                  # official run + breakpoints
    python src/run_experiment.py --smoke          # disclosed tiny code-path check
    python src/run_experiment.py --skip-breakpoints
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import yaml

from jepa import METHODS, penalty_self_check, train_and_eval
from metrics import observation_ceiling, self_check
from world import (
    assert_maze_geometry,
    make_maze,
    rollout,
    scripted_route,
    world_report,
)

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_grid"
WORK = APP / "work"


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, str)) or value is None:
        return value
    raise TypeError(type(value))


def _base_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    data = cfg["data"]
    model = cfg["model"]
    training = cfg["training"]
    evaluation = cfg["evaluation"]
    svreg_weight = float(training["svreg_weight"])
    return {
        "grid_size": int(data["grid_size"]),
        "traj_len": int(data["traj_len"]),
        "noise_std": float(data["noise_std"]),
        "contact_radius": int(data["contact_radius"]),
        "train_seed": int(data["train_seed"]),
        "fit_seed": int(data["fit_seed"]),
        "score_seed": int(data["score_seed"]),
        "route_noise_seed": int(data["route_noise_seed"]),
        "hidden": int(model["hidden"]),
        "latent_dim": int(model["latent_dim"]),
        "gate_init": float(model["gate_init"]),
        "steps": int(training["steps"]),
        "batch_size": int(training["batch_size"]),
        "lr": float(training["lr"]),
        "weight_decay": float(training["weight_decay"]),
        "grad_clip": float(training["grad_clip"]),
        "log_every": int(training["log_every"]),
        "svreg_weight": svreg_weight,
        "svreg_n_projections": int(training["svreg_n_projections"]),
        "vicreg_std_coef": float(training["vicreg_std_coef"]),
        "vicreg_cov_coef": float(training["vicreg_cov_coef"]),
        "vicreg_weight": float(training["vicreg_weight"]),
        "slow_weight": float(training["slow_weight"]),
        "std_target": float(training["std_target"]),
        "ridge_alpha": float(evaluation["ridge_alpha"]),
        "min_chebyshev": int(evaluation["min_chebyshev"]),
        "jag_window": int(evaluation["jag_window"]),
        "n_pairs": int(evaluation["n_pairs"]),
        "pair_seed": int(evaluation["pair_seed"]),
        "return_history": False,
        "return_state": False,
    }


def _with_lambda(spec: dict[str, Any], svreg_weight: float, std_coef: float) -> dict[str, Any]:
    """Keep the pre-registered coupling: slow weight is 1/λ, VICReg hinge matches λ at collapse."""
    out = dict(spec)
    out["svreg_weight"] = float(svreg_weight)
    out["slow_weight"] = 1.0 / float(svreg_weight)
    out["vicreg_weight"] = float(svreg_weight) / float(std_coef)
    return out


def _jobs(cfg: dict[str, Any], *, smoke: bool, skip_breakpoints: bool) -> list[dict[str, Any]]:
    training = cfg["training"]
    jobs: list[dict[str, Any]] = []
    if smoke:
        spec = _base_spec(cfg)
        spec.update({
            "method": "vicreg",
            "seed": 0,
            "steps": int(cfg["disclosure_smoke"]["steps"]),
            "kind": "smoke",
            "return_history": True,
            "return_state": False,
        })
        return [spec]

    for seed in cfg["seeds"]:
        for method in METHODS:
            spec = _base_spec(cfg)
            spec.update({
                "method": method,
                "seed": int(seed),
                "kind": "primary",
                "return_history": True,
                "return_state": int(seed) == 0,
            })
            jobs.append(spec)
    if skip_breakpoints:
        return jobs

    bp = cfg["breakpoints"]
    bp_seed = int(bp["seed"])
    for dim in bp["latent_dim"]["dims"]:
        for method in METHODS:
            spec = _base_spec(cfg)
            spec.update({
                "method": method,
                "seed": bp_seed,
                "latent_dim": int(dim),
                "steps": int(bp["latent_dim"]["steps"]),
                "kind": "latent_dim",
            })
            jobs.append(spec)
    std_coef = float(training["vicreg_std_coef"])
    for weight in bp["reg_weight"]["weights"]:
        for method in METHODS:
            spec = _with_lambda(_base_spec(cfg), float(weight), std_coef)
            spec.update({
                "method": method,
                "seed": bp_seed,
                "steps": int(bp["reg_weight"]["steps"]),
                "kind": "reg_weight",
            })
            jobs.append(spec)
    for steps in bp["short_training"]["steps"]:
        for method in METHODS:
            spec = _base_spec(cfg)
            spec.update({
                "method": method,
                "seed": bp_seed,
                "steps": int(steps),
                "kind": "short_training",
            })
            jobs.append(spec)
    return jobs


def _world_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    data = cfg["data"]
    maze = make_maze(int(data["grid_size"]))
    reports = {}
    bundles = {}
    for name, seed_key in ("train", "train_seed"), ("fit", "fit_seed"), ("score", "score_seed"):
        traj = rollout(
            maze,
            length=int(data["traj_len"]),
            seed=int(data[seed_key]),
            noise_std=float(data["noise_std"]),
            contact_radius=int(data["contact_radius"]),
        )
        reports[name] = world_report(traj, int(cfg["evaluation"]["min_chebyshev"]))
        bundles[name] = traj
    ceiling = observation_ceiling(
        bundles["fit"]["obs"],
        bundles["fit"]["agent"],
        bundles["fit"]["block"],
        bundles["score"]["obs"],
        bundles["score"]["agent"],
        bundles["score"]["block"],
        ridge_alpha=float(cfg["evaluation"]["ridge_alpha"]),
        min_chebyshev=int(cfg["evaluation"]["min_chebyshev"]),
        grid_size=int(data["grid_size"]),
    )
    return {
        "reports": reports,
        "ceiling": ceiling,
        "route_len": len(scripted_route(maze)),
        "n_free_cells": int((maze == 0).sum()),
    }


def _run_pool(jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    ctx_workers = max(1, workers)
    print(f"[pool] {len(jobs)} jobs, workers={ctx_workers}", flush=True)
    with ProcessPoolExecutor(max_workers=ctx_workers) as pool:
        future_map = {pool.submit(train_and_eval, job): job for job in jobs}
        done = 0
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                payload = future.result()
            except Exception:
                print(
                    f"[fail] {job['kind']} {job['method']} seed={job['seed']} "
                    f"dim={job['latent_dim']} steps={job['steps']}",
                    flush=True,
                )
                raise
            done += 1
            metrics = payload["metrics"]
            print(
                f"[{done}/{len(jobs)}] {job['kind']} {job['method']} seed={job['seed']} "
                f"dim={int(job['latent_dim'])} steps={int(job['steps'])} "
                f"agentR2={metrics['agent_r2']:.3f} geo={metrics['geodesic_spearman']:.3f} "
                f"obj={metrics['free_object_r2']:.3f} wall={metrics['train_wall_s']:.1f}s",
                flush=True,
            )
            results.append(payload)
    return results


def _save_checkpoint(payload: dict[str, Any], cfg: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    if metrics["kind"] != "primary" or int(metrics["seed"]) != 0 or "encoder" not in payload:
        return
    WORK.mkdir(parents=True, exist_ok=True)
    method = str(metrics["method"])
    torch.save(
        {
            "encoder": payload["encoder"],
            "predictor": payload["predictor"],
            "gated": payload["gated"],
            "method": method,
            "grid_size": int(cfg["data"]["grid_size"]),
            "hidden": int(cfg["model"]["hidden"]),
            "latent_dim": int(cfg["model"]["latent_dim"]),
            "gate_init": float(cfg["model"]["gate_init"]),
        },
        WORK / f"model_{method}_seed0.pt",
    )


def _assemble(cfg: dict[str, Any], payloads: list[dict[str, Any]], world: dict[str, Any]) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "exp_id": cfg["exp_id"],
        "world": {
            "reports": world["reports"],
            "route_len": world["route_len"],
            "n_free_cells": world["n_free_cells"],
        },
        "observation_ceiling": world["ceiling"],
        "seeds": {},
        "breakpoints": {"latent_dim": [], "reg_weight": [], "short_training": []},
    }
    for payload in payloads:
        metrics = dict(payload["metrics"])
        if payload.get("history") is not None and metrics["kind"] == "primary":
            metrics["history"] = payload["history"]
        kind = str(metrics["kind"])
        if kind == "primary":
            seed = str(int(metrics["seed"]))
            raw["seeds"].setdefault(seed, {})
            raw["seeds"][seed][str(metrics["method"])] = metrics
        elif kind in raw["breakpoints"]:
            raw["breakpoints"][kind].append(metrics)
        else:
            raise ValueError(kind)
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-breakpoints", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    training = cfg["training"]
    coupled_vic = float(training["svreg_weight"]) / float(training["vicreg_std_coef"])
    coupled_slow = 1.0 / float(training["svreg_weight"])
    if abs(float(training["vicreg_weight"]) - coupled_vic) > 1e-9:
        raise SystemExit("vicreg_weight must equal svreg_weight / vicreg_std_coef")
    if abs(float(training["slow_weight"]) - coupled_slow) > 1e-9:
        raise SystemExit("slow_weight must equal 1 / svreg_weight")
    assert_maze_geometry(int(cfg["data"]["grid_size"]))
    self_check()
    penalty_self_check()
    world = _world_snapshot(cfg)
    print("[world]", json.dumps(world["reports"]), flush=True)
    print("[ceiling]", json.dumps(world["ceiling"]), flush=True)

    jobs = _jobs(cfg, smoke=args.smoke, skip_breakpoints=args.skip_breakpoints)
    payloads = _run_pool(jobs, args.workers)

    if args.smoke:
        smoke = {
            "note": cfg["disclosure_smoke"]["note"],
            "metrics": payloads[0]["metrics"],
            "history": payloads[0].get("history", []),
        }
        (EXP / "smoke.json").write_text(json.dumps(_jsonable(smoke), indent=2) + "\n")
        print(f"[smoke] wrote {EXP / 'smoke.json'}", flush=True)
        return

    for payload in payloads:
        _save_checkpoint(payload, cfg)
    raw = _assemble(cfg, payloads, world)
    (EXP / "raw.json").write_text(json.dumps(_jsonable(raw), indent=2) + "\n")
    print(f"[done] wrote {EXP / 'raw.json'}", flush=True)


if __name__ == "__main__":
    main()
