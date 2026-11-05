"""Run exp1_cpu_mrp: Value Forcing vs endpoint flow TD vs QR-DQN.

Usage:
    python src/run_experiment.py                  # official run + breakpoints
    python src/run_experiment.py --smoke          # disclosed tiny code-path check
    python src/run_experiment.py --skip-breakpoints
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import yaml

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
os.environ["PYTHONPATH"] = str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")

from jobs import run_job  # noqa: E402
from mrp import build_mrp, describe  # noqa: E402

APP = SRC.parent
EXP = APP / "experiments" / "exp1_cpu_mrp"
WORK = APP / "work"
METHODS = ("vf", "endpoint", "qr")


def _base_job(cfg: dict, **overrides) -> dict:
    mrp = cfg["mrp"]
    model = cfg["model"]
    train = cfg["training"]
    ev = cfg["eval"]
    runtime = cfg["runtime"]
    job = {
        "gamma": float(mrp["gamma"]),
        "high_reward": float(mrp["high_reward"]),
        "chain_reward": float(mrp["chain_reward"]),
        "wide_p_stay": float(mrp["wide_p_stay"]),
        "wide_r_stay": float(mrp["wide_r_stay"]),
        "wide_r_exit": float(mrp["wide_r_exit"]),
        "chain_depth": int(mrp["chain_depth"]),
        "n_transitions": int(mrp["n_transitions"]),
        "data_seed_offset": int(cfg.get("data_seed_offset", 0)),
        "hidden": int(model["hidden"]),
        "embed_dim": int(model["embed_dim"]),
        "n_quantiles": int(model["n_quantiles"]),
        "huber_kappa": float(model["huber_kappa"]),
        "steps": int(train["steps"]),
        "batch_size": int(train["batch_size"]),
        "lr": float(train["lr"]),
        "weight_decay": float(train["weight_decay"]),
        "grad_clip": float(train["grad_clip"]),
        "ema": float(train["ema"]),
        "n_euler": int(train["n_euler"]),
        "reward_scale": bool(train["reward_scale"]),
        "log_every": int(train["log_every"]),
        "n_samples": int(ev["n_samples"]),
        "std_floor": float(ev["std_floor"]),
        "threads": int(runtime["torch_threads_per_worker"]),
        "save_hist": False,
        "ckpt_path": None,
    }
    job.update(overrides)
    return job


def _state_count(chain_depth: int) -> int:
    if chain_depth == 1:
        return 8
    return 7 + chain_depth


def build_jobs(cfg: dict, *, smoke: bool, skip_breakpoints: bool) -> list[dict]:
    seeds = [0] if smoke else [int(s) for s in cfg["seeds"]]
    steps = 40 if smoke else None
    n_samples = 512 if smoke else None
    jobs: list[dict] = []
    for seed in seeds:
        for method in METHODS:
            extra = {}
            if steps is not None:
                extra["steps"] = steps
                extra["log_every"] = 20
            if n_samples is not None:
                extra["n_samples"] = n_samples
            save = (not smoke) and seed == 0
            ckpt = None
            if save:
                ckpt = str(WORK / f"primary_{method}_seed{seed}.pt")
            jobs.append(
                _base_job(
                    cfg,
                    tag="primary",
                    method=method,
                    seed=seed,
                    save_hist=save,
                    ckpt_path=ckpt,
                    **extra,
                )
            )
    if smoke or skip_breakpoints:
        return jobs

    bp = cfg["breakpoints"]
    seed = int(bp["seed"])
    for n in bp["few_transitions"]["n_transitions"]:
        for method in METHODS:
            jobs.append(_base_job(cfg, tag="few_transitions", method=method, seed=seed, n_transitions=int(n)))
    for gamma in bp["gamma_near_one"]["gamma"]:
        for method in METHODS:
            jobs.append(_base_job(cfg, tag="gamma_near_one", method=method, seed=seed, gamma=float(gamma)))
    per_state = int(bp["long_horizon"]["transitions_per_state"])
    for depth in bp["long_horizon"]["chain_depth"]:
        depth = int(depth)
        n_trans = per_state * _state_count(depth)
        for method in METHODS:
            jobs.append(
                _base_job(
                    cfg,
                    tag="long_horizon",
                    method=method,
                    seed=seed,
                    chain_depth=depth,
                    n_transitions=n_trans,
                )
            )
    for n_euler in bp["flow_steps"]["n_euler"]:
        for method in ("vf", "endpoint"):
            jobs.append(_base_job(cfg, tag="flow_steps", method=method, seed=seed, n_euler=int(n_euler)))
    return jobs


def _primary_truth(cfg: dict) -> dict:
    m = cfg["mrp"]
    mrp = build_mrp(
        float(m["gamma"]),
        high_reward=float(m["high_reward"]),
        chain_reward=float(m["chain_reward"]),
        wide_p_stay=float(m["wide_p_stay"]),
        wide_r_stay=float(m["wide_r_stay"]),
        wide_r_exit=float(m["wide_r_exit"]),
        chain_depth=int(m["chain_depth"]),
    )
    rows = []
    for i, name in enumerate(mrp.names):
        rows.append(
            {
                "state": name,
                "mean": float(mrp.mean[i]),
                "std": float(mrp.std[i]),
                "n_atoms": int(mrp.values[i].shape[0]),
            }
        )
    return {"summary": describe(mrp), "states": rows}


def _assemble(cfg: dict, rows: list[dict]) -> dict:
    seeds: dict[str, dict] = {}
    breakpoints: dict[str, list] = {
        "few_transitions": [],
        "gamma_near_one": [],
        "long_horizon": [],
        "flow_steps": [],
    }
    for row in rows:
        if row["tag"] == "primary":
            seeds.setdefault(str(row["seed"]), {})[row["method"]] = row
        else:
            breakpoints[row["tag"]].append(row)
    return {
        "exp_id": cfg["exp_id"],
        "truth": _primary_truth(cfg),
        "seeds": seeds,
        "breakpoints": breakpoints,
    }


def _preflight(cfg: dict) -> None:
    """Build every breakpoint MRP once so a closed-form bug fails before training."""
    m = cfg["mrp"]
    common = dict(
        high_reward=float(m["high_reward"]),
        chain_reward=float(m["chain_reward"]),
        wide_p_stay=float(m["wide_p_stay"]),
        wide_r_stay=float(m["wide_r_stay"]),
        wide_r_exit=float(m["wide_r_exit"]),
    )
    gammas = [float(m["gamma"]), *[float(g) for g in cfg["breakpoints"]["gamma_near_one"]["gamma"]]]
    depths = [int(m["chain_depth"]), *[int(d) for d in cfg["breakpoints"]["long_horizon"]["chain_depth"]]]
    for gamma in gammas:
        build_mrp(gamma, chain_depth=int(m["chain_depth"]), **common)
    for depth in depths:
        build_mrp(float(m["gamma"]), chain_depth=depth, **common)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-breakpoints", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    _preflight(cfg)
    jobs = build_jobs(cfg, smoke=args.smoke, skip_breakpoints=args.skip_breakpoints)
    workers = 1 if args.smoke else min(int(cfg["runtime"]["max_workers"]), len(jobs), os.cpu_count() or 1)
    print(f"[exp] {len(jobs)} jobs on {workers} workers  smoke={args.smoke}", flush=True)
    ctx = mp.get_context("spawn")
    rows: list[dict] = []
    with ctx.Pool(workers) as pool:
        for row in pool.imap_unordered(run_job, jobs):
            rows.append(row)
            print(
                f"[done] {row['tag']:16s} {row['method']:8s} seed={row['seed']} "
                f"n={row['n_transitions']:4d} γ={row['gamma']:.3f} euler={row['n_euler']:2d} "
                f"depth={row['chain_depth']}  W1={row['mean_w1']:.4f}  "
                f"std_ratio={row['std_ratio']:.3f}  {row['train_wall_s']:.1f}s",
                flush=True,
            )
    payload = _assemble(cfg, rows)
    EXP.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        # Code-path check only. Do not write raw.json and do not touch thresholds.
        smoke_seeds = {
            seed: {
                method: {
                    "mean_w1": rec["mean_w1"],
                    "std_ratio": rec["std_ratio"],
                    "final_loss": rec["final_loss"],
                }
                for method, rec in methods.items()
            }
            for seed, methods in payload["seeds"].items()
        }
        smoke_out = {
            "disclosure": "smoke: tiny step budget, not used to set or edit thresholds",
            "steps": 40,
            "seeds": smoke_seeds,
        }
        (EXP / "smoke.json").write_text(json.dumps(smoke_out, indent=2) + "\n")
        print(f"wrote {EXP / 'smoke.json'}", flush=True)
        return
    (EXP / "raw.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {EXP / 'raw.json'}", flush=True)


if __name__ == "__main__":
    main()
