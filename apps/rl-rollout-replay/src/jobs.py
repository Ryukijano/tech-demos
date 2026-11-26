"""Process-pool jobs. Imported by name so spawn can unpickle them."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from task import Op, demo_batch, holdout_keys, to_torch
from train import (
    Advantage,
    Objective,
    adam_step,
    build_policy,
    chronological,
    exact_reward,
    last_k_order,
    load_archive,
    make_opt,
    max_param_gap,
    ml_loss,
    perturb_,
    prepare_eval,
    replay_batches,
    restore,
    run_online,
    save_archive,
    snapshot,
)


def _threads(n: int) -> None:
    torch.set_num_threads(n)


def _eval_pack(cfg: dict[str, Any], op: Op) -> tuple[torch.Tensor, torch.Tensor, frozenset[tuple[int, ...]], torch.Tensor]:
    prompts, truth, keys = prepare_eval(cfg["eval"], op)
    demo_p, demo_a, _ = demo_batch("add")
    demo_prompts, _ = to_torch(demo_p, demo_a)
    if op == "add":
        keys = keys | holdout_keys(demo_p)
    return prompts, truth, keys, demo_prompts


def _load_base(cfg: dict[str, Any], path: str) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    model = cfg["model"]
    policy = build_policy(int(model["emb"]), int(model["hidden"]), init_seed=None)
    state = torch.load(path, map_location="cpu", weights_only=False)
    restore(policy, state)
    return policy, state


def _rl(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["rl"]


def primary_job(payload: dict[str, Any]) -> dict[str, Any]:
    _threads(int(payload["threads"]))
    cfg = payload["cfg"]
    seed = int(payload["seed"])
    rl = _rl(cfg)
    advantage: Advantage = rl["advantage"]
    steps = int(rl["steps"])
    collect = bool(payload["collect_demo"])
    prompts, truth, keys, demo_prompts = _eval_pack(cfg, "add")
    policy, _base = _load_base(cfg, payload["base_path"])
    t0 = time.perf_counter()
    online, archive, online_demo = run_online(
        policy,
        steps=steps,
        batch_prompts=int(rl["batch_prompts"]),
        group_size=int(rl["group_size"]),
        lr=float(rl["lr"]),
        weight_decay=float(rl["weight_decay"]),
        grad_clip=float(rl["grad_clip"]),
        advantage=advantage,
        reward_name=rl["reward"],
        data_seed=10_000 + seed,
        holdout=keys,
        op="add",
        eval_prompts=prompts,
        eval_truth=truth,
        eval_samples=int(cfg["eval"]["n_samples"]),
        eval_temperature=float(cfg["eval"]["temperature"]),
        eval_seed=50_000,
        demo_prompts=demo_prompts if collect else None,
        log_every=int(payload["log_every"]),
    )
    online_state = snapshot(policy)
    save_archive(payload["archive_path"], archive, "add")
    common = dict(
        advantage=advantage,
        lr=float(rl["lr"]),
        weight_decay=float(rl["weight_decay"]),
        grad_clip=float(rl["grad_clip"]),
        eval_prompts=prompts,
        eval_truth=truth,
        eval_samples=int(cfg["eval"]["n_samples"]),
        eval_temperature=float(cfg["eval"]["temperature"]),
        eval_seed=50_000,
    )

    def _replay(objective: Objective, order: list[int], repeats: int, demo: bool) -> tuple[list[float], list[list[int]] | None]:
        restore(policy, _base)
        return replay_batches(
            policy,
            archive,
            objective=objective,
            order=order,
            repeats=repeats,
            demo_prompts=demo_prompts if demo else None,
            **common,
        )

    order = chronological(steps)
    ml, ml_demo = _replay("ml", order, 1, collect)
    naive, naive_demo = _replay("rl", order, 1, collect)
    naive_gap = max_param_gap(online_state, snapshot(policy))
    last_order = last_k_order(steps, int(cfg["last_k"]), steps)
    lastk, lastk_demo = _replay("ml", last_order, 1, collect)
    seconds = time.perf_counter() - t0
    print(
        f"seed {seed} base {online[0]:.3f} online {online[-1]:.3f} ml {ml[-1]:.3f} "
        f"naive {naive[-1]:.3f} lastk {lastk[-1]:.3f} naive_param_gap {naive_gap:.3e} ({seconds:.1f}s)",
        flush=True,
    )
    return {
        "seed": seed,
        "online": online,
        "ml": ml,
        "naive": naive,
        "lastk": lastk,
        "naive_param_gap": naive_gap,
        "seconds": seconds,
        "demo": None
        if not collect
        else {
            "online": online_demo,
            "ml": ml_demo,
            "naive": naive_demo,
            "lastk": lastk_demo,
        },
    }


def sub_online_job(payload: dict[str, Any]) -> dict[str, Any]:
    _threads(int(payload["threads"]))
    cfg = payload["cfg"]
    rl = _rl(cfg)
    prompts, truth, keys, _demo = _eval_pack(cfg, "sub")
    policy, _base = _load_base(cfg, payload["base_path"])
    t0 = time.perf_counter()
    curve, archive, _ = run_online(
        policy,
        steps=int(rl["steps"]),
        batch_prompts=int(rl["batch_prompts"]),
        group_size=int(rl["group_size"]),
        lr=float(rl["lr"]),
        weight_decay=float(rl["weight_decay"]),
        grad_clip=float(rl["grad_clip"]),
        advantage=rl["advantage"],
        reward_name=rl["reward"],
        data_seed=20_000,
        holdout=keys,
        op="sub",
        eval_prompts=prompts,
        eval_truth=truth,
        eval_samples=int(cfg["eval"]["n_samples"]),
        eval_temperature=float(cfg["eval"]["temperature"]),
        eval_seed=60_000,
        demo_prompts=None,
        log_every=0,
    )
    save_archive(payload["archive_path"], archive, "sub")
    seconds = time.perf_counter() - t0
    print(f"sub specialist base {curve[0]:.3f} final {curve[-1]:.3f} ({seconds:.1f}s)", flush=True)
    return {"name": "sub-online", "curve": curve, "seconds": seconds}


def _order_for(name: str, steps: int, cfg: dict[str, Any]) -> tuple[list[int], int]:
    bp = cfg["breakpoints"]
    if name == "shuffle":
        perm = np.random.default_rng(int(bp["shuffle_seed"])).permutation(steps).tolist()
        return [int(i) for i in perm], 1
    if name == "reverse":
        return list(range(steps - 1, -1, -1)), 1
    if name == "repeats-ml" or name == "repeats-naive":
        return chronological(steps), int(bp["repeats"])
    if name == "lastk-tiny":
        k = int(bp["last_k_tiny"])
        return last_k_order(steps, k, steps), 1
    if name.startswith("perturb-") or name in {"foreign-init", "cross-seed"}:
        return chronological(steps), 1
    raise ValueError(name)


def replay_job(payload: dict[str, Any]) -> dict[str, Any]:
    _threads(int(payload["threads"]))
    cfg = payload["cfg"]
    name = str(payload["name"])
    rl = _rl(cfg)
    steps = int(rl["steps"])
    prompts, truth, _keys, demo_prompts = _eval_pack(cfg, "add")
    model = cfg["model"]
    if name == "foreign-init":
        policy = build_policy(int(model["emb"]), int(model["hidden"]), init_seed=int(cfg["breakpoints"]["foreign_init_seed"]))
    else:
        policy, _base = _load_base(cfg, payload["base_path"])
        if name.startswith("perturb-"):
            sigma = float(payload["sigma"])
            perturb_(policy, sigma, int(cfg["breakpoints"]["perturbation_seed"]))
    archive = load_archive(payload["archive_path"])
    order, repeats = _order_for(name, steps, cfg)
    objective: Objective = "rl" if name == "repeats-naive" or name.startswith("perturb-naive") else "ml"
    t0 = time.perf_counter()
    curve, _demo = replay_batches(
        policy,
        archive,
        objective=objective,
        advantage=rl["advantage"],
        order=order,
        repeats=repeats,
        lr=float(rl["lr"]),
        weight_decay=float(rl["weight_decay"]),
        grad_clip=float(rl["grad_clip"]),
        eval_prompts=prompts,
        eval_truth=truth,
        eval_samples=int(cfg["eval"]["n_samples"]),
        eval_temperature=float(cfg["eval"]["temperature"]),
        eval_seed=50_000,
        demo_prompts=None,
    )
    seconds = time.perf_counter() - t0
    print(f"{name} base {curve[0]:.3f} final {curve[-1]:.3f} repeats {repeats} ({seconds:.1f}s)", flush=True)
    del demo_prompts
    return {
        "name": name,
        "curve": curve,
        "final": curve[-1],
        "base": curve[0],
        "repeats": repeats,
        "seconds": seconds,
        "sigma": payload.get("sigma", 0.0),
    }


def student_job(payload: dict[str, Any]) -> dict[str, Any]:
    """One ML update on the addition batch, then one on the subtraction batch, each online step."""
    _threads(int(payload["threads"]))
    cfg = payload["cfg"]
    rl = _rl(cfg)
    steps = int(rl["steps"])
    add_p, add_t, _k, _d = _eval_pack(cfg, "add")
    sub_p, sub_t, _sk, _sd = _eval_pack(cfg, "sub")
    policy, _base = _load_base(cfg, payload["base_path"])
    add_arch = load_archive(payload["add_archive"])
    sub_arch = load_archive(payload["sub_archive"])
    opt = make_opt(policy, float(rl["lr"]), float(rl["weight_decay"]))
    n_samples = int(cfg["eval"]["n_samples"])
    temperature = float(cfg["eval"]["temperature"])
    add_curve: list[float] = []
    sub_curve: list[float] = []

    def record(step: int) -> None:
        generator_seed_add = 50_000 + step
        generator_seed_sub = 60_000 + step
        add_curve.append(_draw(policy, add_p, add_t, n_samples, temperature, generator_seed_add))
        sub_curve.append(_draw(policy, sub_p, sub_t, n_samples, temperature, generator_seed_sub))

    t0 = time.perf_counter()
    record(0)
    for step in range(steps):
        policy.train()
        adam_step(
            policy,
            opt,
            ml_loss(policy, add_arch["prompts"][step], add_arch["answers"][step]),
            float(rl["grad_clip"]),
        )
        policy.train()
        adam_step(
            policy,
            opt,
            ml_loss(policy, sub_arch["prompts"][step], sub_arch["answers"][step]),
            float(rl["grad_clip"]),
        )
        record(step + 1)
    seconds = time.perf_counter() - t0
    print(
        f"student add {add_curve[-1]:.3f} sub {sub_curve[-1]:.3f} ({seconds:.1f}s)",
        flush=True,
    )
    return {"name": "specialists", "add_curve": add_curve, "sub_curve": sub_curve, "seconds": seconds}


def _draw(policy, prompts, truth, n_samples: int, temperature: float, seed: int) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return exact_reward(policy, prompts, truth, n_samples, temperature, generator)
