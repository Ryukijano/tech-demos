"""Pre-registered CPU experiment: recover on-policy REINFORCE from its rollouts.

    python src/run_experiment.py --probe     # budget check, not a result
    python src/run_experiment.py --smoke     # tiny code path, not a result
    python src/run_experiment.py             # seeds 0/1/2 and breakpoints
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jobs import primary_job, replay_job, student_job, sub_online_job  # noqa: E402
from task import demo_batch, holdout_keys, sample_train_batch, self_check_task, to_torch  # noqa: E402
from train import (  # noqa: E402
    adam_step,
    build_policy,
    exact_reward,
    make_opt,
    ml_loss,
    prepare_eval,
    restore,
    run_online,
    run_sft,
    self_check_train,
    snapshot,
)

APP = SRC.parent
EXP = APP / "experiments" / "exp1_cpu_addition"
WORK = APP / "work"
REPO = APP.parents[1]


def load_cfg() -> dict[str, Any]:
    with (EXP / "config.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise RuntimeError("config.yaml did not load to a mapping")
    return cfg


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _holdout_and_eval(cfg: dict[str, Any]):
    prompts, truth, keys = prepare_eval(cfg["eval"], "add")
    demo_p, demo_a, _ = demo_batch("add")
    demo_prompts, _ = to_torch(demo_p, demo_a)
    return prompts, truth, keys | holdout_keys(demo_p)


def train_base(cfg: dict[str, Any], *, log_every: int) -> tuple[Any, dict[str, torch.Tensor]]:
    model = cfg["model"]
    sft = cfg["sft"]
    policy = build_policy(int(model["emb"]), int(model["hidden"]), init_seed=int(sft["init_seed"]))
    prompts, truth, holdout = _holdout_and_eval(cfg)
    print(f"parameters {policy.n_parameters()}", flush=True)
    curve = run_sft(
        policy,
        steps=int(sft["steps"]),
        batch_prompts=int(sft["batch_prompts"]),
        lr=float(sft["lr"]),
        weight_decay=float(sft["weight_decay"]),
        grad_clip=float(sft["grad_clip"]),
        data_seed=int(sft["data_seed"]),
        holdout=holdout,
        eval_prompts=prompts,
        eval_truth=truth,
        eval_samples=int(cfg["eval"]["n_samples"]),
        eval_temperature=float(cfg["eval"]["temperature"]),
        eval_seed=50_000,
        log_every=log_every,
    )
    print(f"sft exact_reward {curve[0]:.3f} -> {curve[-1]:.3f}", flush=True)
    return policy, snapshot(policy), curve


def self_check() -> None:
    self_check_task()
    self_check_train()
    print("self-check ok", flush=True)


def run_probe(cfg: dict[str, Any]) -> None:
    """SFT curve, then short REINFORCE from a few prefixes. Not a measured seed."""
    self_check()
    torch.set_num_threads(4)
    probe = deepcopy(cfg)
    probe["eval"]["n_problems"] = 128
    probe["eval"]["n_samples"] = 4
    probe["sft"]["steps"] = 30
    model = probe["model"]
    sft = probe["sft"]
    policy = build_policy(int(model["emb"]), int(model["hidden"]), init_seed=int(sft["init_seed"]))
    prompts, truth, holdout = _holdout_and_eval(probe)
    print(f"probe parameters {policy.n_parameters()}", flush=True)
    snaps: dict[int, dict[str, torch.Tensor]] = {}
    opt_steps = 30
    opt = make_opt(policy, float(sft["lr"]), float(sft["weight_decay"]))
    rng = np.random.default_rng(int(sft["data_seed"]))

    def draw(step: int) -> float:
        generator = torch.Generator()
        generator.manual_seed(50_000 + step)
        return exact_reward(policy, prompts, truth, 4, 1.0, generator)

    print(f"probe sft 0 {draw(0):.3f}", flush=True)
    for step in range(opt_steps):
        batch_p, batch_a = sample_train_batch(rng, int(sft["batch_prompts"]), "add", holdout)
        tp, ta = to_torch(batch_p, batch_a)
        policy.train()
        adam_step(policy, opt, ml_loss(policy, tp, ta[:, None, :]), float(sft["grad_clip"]))
        reward = draw(step + 1)
        if (step + 1) % 5 == 0:
            snaps[step + 1] = snapshot(policy)
            print(f"probe sft {step + 1} {reward:.3f}", flush=True)
    rl = probe["rl"]
    for prefix, state in snaps.items():
        restore(policy, state)
        curve, _arch, _demo = run_online(
            policy,
            steps=15,
            batch_prompts=int(rl["batch_prompts"]),
            group_size=int(rl["group_size"]),
            lr=float(rl["lr"]),
            weight_decay=float(rl["weight_decay"]),
            grad_clip=float(rl["grad_clip"]),
            advantage=rl["advantage"],
            reward_name=rl["reward"],
            data_seed=10_000,
            holdout=holdout,
            op="add",
            eval_prompts=prompts,
            eval_truth=truth,
            eval_samples=4,
            eval_temperature=1.0,
            eval_seed=50_000,
            demo_prompts=None,
            log_every=5,
        )
        print(f"probe rl from sft {prefix}: {curve[0]:.3f} -> {curve[-1]:.3f}", flush=True)


def run_smoke(cfg: dict[str, Any]) -> None:
    self_check()
    torch.set_num_threads(2)
    smoke = deepcopy(cfg)
    smoke["sft"]["steps"] = 2
    smoke["sft"]["batch_prompts"] = 16
    smoke["rl"]["steps"] = 2
    smoke["rl"]["batch_prompts"] = 4
    smoke["rl"]["group_size"] = 4
    smoke["eval"]["n_problems"] = 8
    smoke["eval"]["n_samples"] = 2
    smoke["last_k"] = 1
    WORK.mkdir(parents=True, exist_ok=True)
    _policy, state, _sft_curve = train_base(smoke, log_every=1)
    base_path = WORK / "smoke_base.pt"
    torch.save(state, base_path)
    result = primary_job(
        {
            "cfg": smoke,
            "seed": 99,
            "threads": 2,
            "base_path": str(base_path),
            "archive_path": str(WORK / "smoke_archive.pt"),
            "collect_demo": True,
            "log_every": 1,
        }
    )
    lengths = {key: len(result[key]) for key in ("online", "ml", "naive", "lastk")}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(lengths)
    payload = {"smoke": True, "not_a_result": True, "seed": 99, "lengths": lengths, "final": {k: result[k][-1] for k in lengths}}
    (EXP / "smoke.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("smoke ok", payload, flush=True)


def _replay_payloads(cfg: dict[str, Any], base_path: Path, archives: dict[int, Path]) -> list[dict[str, Any]]:
    bp = cfg["breakpoints"]
    seed0 = archives[int(bp["seed"])]
    cross = archives[int(bp["cross_seed"])]
    threads = int(cfg["runtime"]["torch_threads_per_worker"])
    names = ["shuffle", "reverse", "repeats-ml", "repeats-naive", "lastk-tiny", "foreign-init", "cross-seed"]
    payloads = []
    for name in names:
        payloads.append(
            {
                "name": name,
                "cfg": cfg,
                "threads": threads,
                "base_path": str(base_path),
                "archive_path": str(cross if name == "cross-seed" else seed0),
                "sigma": 0.0,
            }
        )
    for sigma in bp["perturbation_std"]:
        for kind in ("ml", "naive"):
            payloads.append(
                {
                    "name": f"perturb-{kind}-{sigma}",
                    "cfg": cfg,
                    "threads": threads,
                    "base_path": str(base_path),
                    "archive_path": str(seed0),
                    "sigma": float(sigma),
                }
            )
    return payloads


def run_official(cfg: dict[str, Any]) -> None:
    self_check()
    torch.set_num_threads(4)
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    policy, state, sft_curve = train_base(cfg, log_every=max(1, int(cfg["sft"]["steps"]) // 4))
    base_path = WORK / "base.pt"
    torch.save(state, base_path)
    n_params = int(policy.n_parameters())
    threads = int(cfg["runtime"]["torch_threads_per_worker"])
    workers = int(cfg["runtime"]["workers"])
    archives = {int(seed): WORK / f"archive_add_seed{seed}.pt" for seed in cfg["seeds"]}
    sub_path = WORK / "archive_sub_seed0.pt"
    primary_payloads = [
        {
            "cfg": cfg,
            "seed": int(seed),
            "threads": threads,
            "base_path": str(base_path),
            "archive_path": str(archives[int(seed)]),
            "collect_demo": int(seed) == int(cfg["breakpoints"]["seed"]),
            "log_every": 10,
        }
        for seed in cfg["seeds"]
    ]
    sub_payload = {
        "cfg": cfg,
        "threads": threads,
        "base_path": str(base_path),
        "archive_path": str(sub_path),
    }
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        primary_async = pool.map_async(primary_job, primary_payloads)
        sub_async = pool.apply_async(sub_online_job, (sub_payload,))
        primaries = primary_async.get()
        sub = sub_async.get()
    replay_payloads = _replay_payloads(cfg, base_path, archives)
    student_payload = {
        "cfg": cfg,
        "threads": threads,
        "base_path": str(base_path),
        "add_archive": str(archives[int(cfg["breakpoints"]["seed"])]),
        "sub_archive": str(sub_path),
    }
    with ctx.Pool(workers) as pool:
        replay_async = pool.map_async(replay_job, replay_payloads)
        student_async = pool.apply_async(student_job, (student_payload,))
        replays = replay_async.get()
        student = student_async.get()
    raw = {
        "experiment": cfg["experiment"],
        "threshold_commit": git_sha(),
        "n_params": n_params,
        "sft_curve": sft_curve,
        "seeds": {str(row["seed"]): row for row in primaries},
        "sub_online": sub,
        "breakpoints": replays,
        "student": student,
        "seconds_total": time.perf_counter() - t0,
        "demo_problems": ["12+35", "48+27", "99+1", "7+8", "50+50", "63+19"],
        "demo_truth": [47, 75, 100, 15, 100, 82],
    }
    (EXP / "raw.json").write_text(json.dumps(raw) + "\n")
    print(f"wrote {EXP / 'raw.json'} in {raw['seconds_total']:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_cfg()
    if args.probe:
        run_probe(cfg)
        return
    if args.smoke:
        run_smoke(cfg)
        return
    run_official(cfg)


if __name__ == "__main__":
    main()
