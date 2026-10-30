"""Run exp1_cpu_minatar: clean-room Dreamer toy vs random on MinAtar.

Usage:
    python src/run_experiment.py                  # official run + breakpoints
    python src/run_experiment.py --smoke          # disclosed tiny code-path check
    python src/run_experiment.py --skip-breakpoints
"""

from __future__ import annotations

import argparse
import json
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import yaml

from dreamer_minatar import (
    evaluate_policy,
    evaluate_random,
    make_env,
    save_agent,
    set_threads,
    train_agent,
)

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_minatar"
WORK = APP / "work"


def _log(msg: dict) -> None:
    print(
        f"[log] {msg.get('game')} seed={msg.get('seed')} "
        f"step={msg.get('policy_step')} wm={msg.get('wm_loss', float('nan')):.3f} "
        f"actor={msg.get('actor_loss', float('nan')):.3f}",
        flush=True,
    )


def _train_and_eval(args: tuple) -> dict:
    """Worker for one (game, seed). Must be top-level for ProcessPoolExecutor."""
    game, seed, cfg, smoke = args
    set_threads(max(1, int(cfg["runtime"]["torch_threads"]) // 2))
    print(f"[train] {game} seed={seed}", flush=True)
    tr = train_agent(game, seed, cfg, log_fn=_log if not smoke else None)
    agent = tr["agent"]
    ev = cfg["evaluation"]
    t0 = time.perf_counter()
    agent_metrics = evaluate_policy(
        agent,
        game,
        float(cfg["env"]["sticky_action_prob"]),
        int(ev["n_episodes"]),
        int(ev["max_episode_steps"]),
        seed=seed + 123,
        greedy=bool(ev["greedy"]),
    )
    random_metrics = evaluate_random(
        game,
        float(cfg["env"]["sticky_action_prob"]),
        int(ev["n_episodes"]),
        int(ev["max_episode_steps"]),
        seed=seed + 456,
        n_actions=tr["n_actions"],
    )
    eval_wall = time.perf_counter() - t0
    ckpt_path = WORK / f"agent_{game}_seed{seed}.pt"
    save_agent(ckpt_path, tr, game, seed)
    return {
        "game": game,
        "seed": seed,
        "n_params": tr["n_params"],
        "train_wall_s": tr["train_wall_s"],
        "eval_wall_s": eval_wall,
        "policy_steps": tr["policy_steps"],
        "imagine_horizon": tr["imagine_horizon"],
        "learning_curve": tr["learning_curve"],
        "agent": agent_metrics,
        "random": random_metrics,
        "ckpt": str(ckpt_path.name),
    }


def _run_breakpoints(cfg: dict) -> dict:
    bcfg = cfg["breakpoints"]
    seed = int(bcfg["seed"])
    game = str(bcfg["game"])
    set_threads(int(cfg["runtime"]["torch_threads"]))
    out: dict = {"seed": seed, "game": game}

    # Tiny step budget
    rows = []
    for steps in bcfg["tiny_step_budget"]["policy_steps"]:
        print(f"[breakpoint] tiny_step_budget steps={steps}", flush=True)
        # scale initial random with budget
        local = yaml.safe_load(yaml.dump(cfg))
        local["training"]["initial_random_steps"] = min(
            int(cfg["training"]["initial_random_steps"]), max(100, int(steps) // 4)
        )
        local["evaluation"]["n_episodes"] = 40  # cheaper for breakpoints
        tr = train_agent(game, seed, local, policy_steps_override=int(steps))
        am = evaluate_policy(
            tr["agent"],
            game,
            float(cfg["env"]["sticky_action_prob"]),
            40,
            int(cfg["evaluation"]["max_episode_steps"]),
            seed=seed + 999,
            greedy=bool(cfg["evaluation"]["greedy"]),
        )
        rm = evaluate_random(
            game,
            float(cfg["env"]["sticky_action_prob"]),
            40,
            int(cfg["evaluation"]["max_episode_steps"]),
            seed=seed + 1001,
            n_actions=tr["n_actions"],
        )
        rows.append(
            {
                "policy_steps": int(steps),
                "agent_mean": am["mean_return"],
                "random_mean": rm["mean_return"],
                "n_params": tr["n_params"],
                "train_wall_s": tr["train_wall_s"],
            }
        )
    out["tiny_step_budget"] = rows

    # Imagination horizon
    rows = []
    hz_steps = int(bcfg["imagination_horizon"]["policy_steps"])
    for hz in bcfg["imagination_horizon"]["imagine_horizon"]:
        print(f"[breakpoint] imagination_horizon={hz}", flush=True)
        local = yaml.safe_load(yaml.dump(cfg))
        local["training"]["initial_random_steps"] = min(
            int(cfg["training"]["initial_random_steps"]), max(100, hz_steps // 4)
        )
        local["evaluation"]["n_episodes"] = 40
        tr = train_agent(
            game,
            seed,
            local,
            policy_steps_override=hz_steps,
            imagine_horizon_override=int(hz),
        )
        am = evaluate_policy(
            tr["agent"],
            game,
            float(cfg["env"]["sticky_action_prob"]),
            40,
            int(cfg["evaluation"]["max_episode_steps"]),
            seed=seed + 1111,
            greedy=bool(cfg["evaluation"]["greedy"]),
        )
        rm = evaluate_random(
            game,
            float(cfg["env"]["sticky_action_prob"]),
            40,
            int(cfg["evaluation"]["max_episode_steps"]),
            seed=seed + 1222,
            n_actions=tr["n_actions"],
        )
        rows.append(
            {
                "imagine_horizon": int(hz),
                "policy_steps": hz_steps,
                "agent_mean": am["mean_return"],
                "random_mean": rm["mean_return"],
                "n_params": tr["n_params"],
                "train_wall_s": tr["train_wall_s"],
            }
        )
    out["imagination_horizon"] = rows

    # Tiny model
    print("[breakpoint] tiny_model", flush=True)
    tm = bcfg["tiny_model"]
    local = yaml.safe_load(yaml.dump(cfg))
    local["training"]["initial_random_steps"] = min(
        int(cfg["training"]["initial_random_steps"]),
        max(100, int(tm["policy_steps"]) // 4),
    )
    local["evaluation"]["n_episodes"] = 40
    overrides = {
        "latent_dim": int(tm["latent_dim"]),
        "transformer_width": int(tm["transformer_width"]),
        "transformer_layers": int(tm["transformer_layers"]),
        "actor_hidden": int(tm["actor_hidden"]),
        "critic_hidden": int(tm["critic_hidden"]),
        # keep heads divisible
        "transformer_heads": 2,
        "cnn_channels": [4, 8, 16],
    }
    tr = train_agent(
        game,
        seed,
        local,
        model_overrides=overrides,
        policy_steps_override=int(tm["policy_steps"]),
    )
    am = evaluate_policy(
        tr["agent"],
        game,
        float(cfg["env"]["sticky_action_prob"]),
        40,
        int(cfg["evaluation"]["max_episode_steps"]),
        seed=seed + 1333,
        greedy=bool(cfg["evaluation"]["greedy"]),
    )
    rm = evaluate_random(
        game,
        float(cfg["env"]["sticky_action_prob"]),
        40,
        int(cfg["evaluation"]["max_episode_steps"]),
        seed=seed + 1444,
        n_actions=tr["n_actions"],
    )
    out["tiny_model"] = {
        "policy_steps": int(tm["policy_steps"]),
        "agent_mean": am["mean_return"],
        "random_mean": rm["mean_return"],
        "n_params": tr["n_params"],
        "train_wall_s": tr["train_wall_s"],
        "overrides": overrides,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-breakpoints", action="store_true")
    ap.add_argument("--jobs", type=int, default=2, help="parallel game-seed workers")
    args = ap.parse_args()

    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    set_threads(int(cfg["runtime"]["torch_threads"]))
    WORK.mkdir(exist_ok=True)

    games = list(cfg["env"]["games"])
    seeds = list(cfg["seeds"])
    smoke = bool(args.smoke)

    if smoke:
        # Disclosed tiny code-path check — does not tune thresholds.
        cfg["training"]["policy_steps"] = 200
        cfg["training"]["initial_random_steps"] = 50
        cfg["training"]["wm_batch"] = 8
        cfg["training"]["imagine_batch"] = 8
        cfg["evaluation"]["n_episodes"] = 8
        games = [games[0]]
        seeds = [0]
        args.skip_breakpoints = True

    raw: dict = {
        "smoke": smoke,
        "torch": torch.__version__,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "games": {},
    }

    jobs = [(g, s, cfg, smoke) for g in games for s in seeds]
    results = []
    # Parallelize independent (game, seed) runs across CPU processes.
    n_workers = 1 if smoke else max(1, min(int(args.jobs), len(jobs)))
    if n_workers == 1:
        for job in jobs:
            results.append(_train_and_eval(job))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futs = {ex.submit(_train_and_eval, job): job for job in jobs}
            for fut in as_completed(futs):
                results.append(fut.result())

    for r in results:
        g = r["game"]
        raw["games"].setdefault(g, {"seeds": {}})
        raw["games"][g]["seeds"][str(r["seed"])] = {
            "n_params": r["n_params"],
            "train_wall_s": r["train_wall_s"],
            "eval_wall_s": r["eval_wall_s"],
            "policy_steps": r["policy_steps"],
            "imagine_horizon": r["imagine_horizon"],
            "learning_curve": r["learning_curve"],
            "agent_mean_return": r["agent"]["mean_return"],
            "agent_std_return": r["agent"]["std_return"],
            "random_mean_return": r["random"]["mean_return"],
            "random_std_return": r["random"]["std_return"],
            "ckpt": r["ckpt"],
        }
        print(
            f"[eval] {g} seed={r['seed']}: agent={r['agent']['mean_return']:.3f} "
            f"random={r['random']['mean_return']:.3f} "
            f"params={r['n_params']} train={r['train_wall_s']:.1f}s",
            flush=True,
        )

    if not args.skip_breakpoints:
        print("[breakpoints] starting ...", flush=True)
        raw["breakpoints"] = _run_breakpoints(cfg)

    raw["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_name = "smoke.json" if smoke else "raw.json"
    (EXP / out_name).write_text(json.dumps(raw, indent=2))
    print(f"[done] wrote {out_name}", flush=True)


if __name__ == "__main__":
    # Probe MinAtar early for clearer errors
    _ = make_env("breakout")
    main()
