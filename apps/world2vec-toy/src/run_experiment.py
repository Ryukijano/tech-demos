"""Run exp1_cpu_pendulum: world2vec continuous-time latent dynamics vs discrete.

Usage:
    python src/run_experiment.py                  # official run + breakpoints
    python src/run_experiment.py --smoke          # disclosed tiny code-path check
    python src/run_experiment.py --skip-breakpoints
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import yaml

from world2vec_toy import (
    Kind,
    collect_trajectories,
    evaluate_model,
    pendulum_cfg_from_data,
    set_threads,
    train_model,
)

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_pendulum"
WORK = APP / "work"

KINDS: list[Kind] = ["world2vec", "discrete", "recon"]


def _train_one(
    kind: Kind,
    seed: int,
    cfg: dict,
    train_data,
    *,
    steps_override: int | None = None,
    noise_std: float = 0.0,
):
    print(
        f"[train] {kind} seed={seed} steps="
        f"{steps_override or cfg['training']['steps']} noise={noise_std}",
        flush=True,
    )
    tr = train_model(
        kind,
        seed,
        cfg,
        train_data,
        steps_override=steps_override,
        noise_std=noise_std,
    )
    return kind, seed, tr


def _eval_pack(tr, cfg, eval_data, *, horizon: int | None = None, noise_std: float = 0.0, dt_scale: float = 1.0):
    h = int(cfg["thresholds"]["eval_horizon"] if horizon is None else horizon)
    metrics = evaluate_model(
        tr.model,
        eval_data,
        horizon=h,
        history=int(cfg["data"]["history"]),
        noise_std=noise_std,
        dt_scale=dt_scale,
    )
    return {
        **metrics,
        "final_loss": tr.final_loss,
        "train_wall_s": tr.train_wall_s,
        "n_params": tr.n_params,
        "history": tr.history,
    }


def run_official(cfg: dict, *, smoke: bool, skip_breakpoints: bool) -> dict:
    set_threads(int(cfg["runtime"]["torch_threads"]))
    WORK.mkdir(exist_ok=True)
    d = cfg["data"]
    pcfg = pendulum_cfg_from_data(d)

    if smoke:
        seeds = [0]
        steps_override = 40
        n_train, n_eval = 32, 16
        traj_len = 32
        skip_breakpoints = True
    else:
        seeds = list(cfg["seeds"])
        steps_override = None
        n_train = int(d["n_train_traj"])
        n_eval = int(d["n_eval_traj"])
        traj_len = int(d["traj_len"])

    print("[data] collecting train/eval pendulum rollouts…", flush=True)
    t_data = time.perf_counter()
    train_data = collect_trajectories(
        n_train, traj_len, pcfg, int(d["train_stream_seed"])
    )
    eval_data = collect_trajectories(
        n_eval, traj_len, pcfg, int(d["eval_stream_seed"])
    )
    print(f"[data] done in {time.perf_counter() - t_data:.1f}s", flush=True)

    raw: dict = {
        "exp_id": cfg["exp_id"],
        "smoke": smoke,
        "seeds": {},
        "breakpoints": {},
        "meta": {
            "n_train_traj": n_train,
            "n_eval_traj": n_eval,
            "traj_len": traj_len,
            "torch": torch.__version__,
        },
    }

    jobs = [(k, s) for s in seeds for k in KINDS]
    results: dict[tuple[str, int], object] = {}

    # Parallelize across kinds×seeds (CPU-bound; modest workers)
    workers = min(4, len(jobs))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(_train_one, k, s, cfg, train_data, steps_override=steps_override)
            for k, s in jobs
        ]
        for fut in as_completed(futs):
            kind, seed, tr = fut.result()
            results[(kind, seed)] = tr
            # persist weights for media
            path = WORK / f"model_{kind}_seed{seed}.pt"
            torch.save(
                {
                    "kind": kind,
                    "seed": seed,
                    "state_dict": tr.model.state_dict(),
                    "model_cfg": {
                        "history": int(d["history"]),
                        "img_size": int(d["img_size"]),
                        "latent_dim": int(cfg["model"]["latent_dim"]),
                        "cnn_channels": list(cfg["model"]["cnn_channels"]),
                        "dyn_hidden": list(cfg["model"]["dyn_hidden"]),
                        "encoder_mlp_hidden": list(cfg["model"]["encoder_mlp_hidden"]),
                        "dt": float(d["dt"]),
                    },
                },
                path,
            )

    for s in seeds:
        raw["seeds"][str(s)] = {"kernels": {}}
        for k in KINDS:
            tr = results[(k, s)]
            pack = _eval_pack(tr, cfg, eval_data)
            raw["seeds"][str(s)]["kernels"][k] = pack
            print(
                f"[eval] {k} seed={s} lrc_h={pack['lrc_h']:.4f} "
                f"probe_r2={pack['probe_r2']:.4f} params={pack['n_params']}",
                flush=True,
            )

    if not skip_breakpoints:
        bp_seed = int(cfg["breakpoints"]["seed"])
        raw["breakpoints"] = run_breakpoints(cfg, train_data, eval_data, bp_seed, results)

    return raw


def run_breakpoints(cfg, train_data, eval_data, seed: int, results: dict) -> dict:
    out: dict = {}
    bp = cfg["breakpoints"]
    history = int(cfg["data"]["history"])

    # Long horizon on already-trained primary models
    print("[bp] long_horizon…", flush=True)
    lh = {}
    for kind in ("world2vec", "discrete"):
        tr = results[(kind, seed)]
        rows = []
        for h in bp["long_horizon"]["horizons"]:
            # need traj long enough
            if h >= train_data.frames.shape[1] - history:
                continue
            m = evaluate_model(
                tr.model, eval_data, horizon=int(h), history=history
            )
            rows.append({"horizon": int(h), **{kk: m[kk] for kk in ("lrc_h", "probe_r2", "horizon_state_r2")}})
            print(f"  {kind} H={h} lrc={m['lrc_h']:.4f}", flush=True)
        lh[kind] = rows
    out["long_horizon"] = lh

    # Observation noise: retrain short
    print("[bp] observation_noise…", flush=True)
    noise_rows = []
    for ns in bp["observation_noise"]["noise_stds"]:
        for kind in ("world2vec", "discrete"):
            tr = train_model(
                kind,
                seed,
                cfg,
                train_data,
                steps_override=int(bp["observation_noise"]["steps"]),
                noise_std=float(ns),
            )
            m = evaluate_model(
                tr.model,
                eval_data,
                horizon=int(cfg["thresholds"]["eval_horizon"]),
                history=history,
                noise_std=float(ns),
            )
            noise_rows.append(
                {
                    "kind": kind,
                    "noise_std": float(ns),
                    "lrc_h": m["lrc_h"],
                    "probe_r2": m["probe_r2"],
                }
            )
            print(f"  {kind} noise={ns} lrc={m['lrc_h']:.4f}", flush=True)
    out["observation_noise"] = noise_rows

    # Irregular dt: continuous model can change Euler step; discrete cannot
    print("[bp] irregular_dt…", flush=True)
    dt_rows = []
    h_nom = int(bp["irregular_dt"]["horizon_steps_nominal"])
    for kind in ("world2vec", "discrete"):
        tr = results[(kind, seed)]
        for scale in bp["irregular_dt"]["dt_scales"]:
            # For continuous: roll h_nom steps with dt*scale — covers different physical time.
            # Compare LRC against encoded state at the same number of env steps
            # (scale=1) for a fair latent-match; also report horizon_state_r2.
            m = evaluate_model(
                tr.model,
                eval_data,
                horizon=h_nom,
                history=history,
                dt_scale=float(scale),
            )
            dt_rows.append(
                {
                    "kind": kind,
                    "dt_scale": float(scale),
                    "lrc_h": m["lrc_h"],
                    "horizon_state_r2": m["horizon_state_r2"],
                }
            )
            print(f"  {kind} dt_scale={scale} lrc={m['lrc_h']:.4f}", flush=True)
    out["irregular_dt"] = dt_rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-breakpoints", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    EXP.mkdir(parents=True, exist_ok=True)
    raw = run_official(cfg, smoke=args.smoke, skip_breakpoints=args.skip_breakpoints)

    out_name = "smoke.json" if args.smoke else "raw.json"
    (EXP / out_name).write_text(json.dumps(raw, indent=2))
    print(f"wrote {EXP / out_name}", flush=True)


if __name__ == "__main__":
    main()
