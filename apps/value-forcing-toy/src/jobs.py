"""One training job. Importable by process-pool workers (not __main__)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models import Method, evaluate, set_threads, std_ratio, train_method
from mrp import atom_quantiles, build_mrp, sample_dataset


def _finite(value: float) -> float | None:
    number = float(value)
    if math.isfinite(number):
        return number
    return None


def _histograms(samples: list[np.ndarray], mrp, n_bins: int = 40) -> dict[str, Any]:
    packed: dict[str, Any] = {}
    for s_id, drawn in enumerate(samples):
        name = mrp.names[s_id]
        us = np.array([0.005, 0.995], dtype=np.float64)
        true_q = atom_quantiles(mrp.values[s_id], mrp.probs[s_id], us)
        lo = float(min(true_q[0], np.quantile(drawn, 0.005)))
        hi = float(max(true_q[1], np.quantile(drawn, 0.995)))
        if hi - lo < 1e-3:
            lo -= 0.5
            hi += 0.5
        pad = 0.08 * (hi - lo)
        edges = np.linspace(lo - pad, hi + pad, n_bins + 1)
        pred_count, _ = np.histogram(drawn, bins=edges)
        true_mass = np.zeros(n_bins, dtype=np.float64)
        bin_id = np.clip(np.searchsorted(edges, mrp.values[s_id], side="right") - 1, 0, n_bins - 1)
        np.add.at(true_mass, bin_id, mrp.probs[s_id])
        packed[name] = {
            "edges": [float(x) for x in edges],
            "pred_count": [int(c) for c in pred_count],
            "true_mass": [float(m) for m in true_mass],
            "n_samples": int(drawn.shape[0]),
        }
    return packed


def run_job(job: dict) -> dict[str, Any]:
    set_threads(int(job["threads"]))
    mrp = build_mrp(
        float(job["gamma"]),
        high_reward=float(job["high_reward"]),
        chain_reward=float(job["chain_reward"]),
        wide_p_stay=float(job["wide_p_stay"]),
        wide_r_stay=float(job["wide_r_stay"]),
        wide_r_exit=float(job["wide_r_exit"]),
        chain_depth=int(job["chain_depth"]),
    )
    rng = np.random.default_rng(int(job["seed"]) + int(job["data_seed_offset"]))
    data = sample_dataset(mrp, int(job["n_transitions"]), rng)
    method: Method = job["method"]
    trained = train_method(
        method,
        mrp,
        data,
        seed=int(job["seed"]),
        hidden=int(job["hidden"]),
        embed_dim=int(job["embed_dim"]),
        n_quantiles=int(job["n_quantiles"]),
        huber_kappa=float(job["huber_kappa"]),
        steps=int(job["steps"]),
        batch_size=int(job["batch_size"]),
        lr=float(job["lr"]),
        weight_decay=float(job["weight_decay"]),
        grad_clip=float(job["grad_clip"]),
        ema=float(job["ema"]),
        n_euler=int(job["n_euler"]),
        reward_scale=bool(job["reward_scale"]),
        log_every=int(job["log_every"]),
    )
    ev = evaluate(
        method,
        trained["module"],
        mrp,
        n_samples=int(job["n_samples"]),
        n_euler=int(job["n_euler"]),
        reward_scale=bool(job["reward_scale"]),
        n_quantiles=int(job["n_quantiles"]),
    )
    ratio = std_ratio(ev["per_state"], float(job["std_floor"]))
    per_state = []
    for row in ev["per_state"]:
        per_state.append(
            {
                "state": row["state"],
                "w1": float(row["w1"]),
                "pred_mean": float(row["pred_mean"]),
                "true_mean": float(row["true_mean"]),
                "pred_std": float(row["pred_std"]),
                "true_std": float(row["true_std"]),
                "std_ratio": _finite(float(row["std_ratio"])),
            }
        )
    farthest = mrp.farthest_name()
    farthest_w1 = next(row["w1"] for row in per_state if row["state"] == farthest)
    out: dict[str, Any] = {
        "tag": job["tag"],
        "method": method,
        "seed": int(job["seed"]),
        "mean_w1": float(ev["mean_w1"]),
        "std_ratio": float(ratio),
        "per_state": per_state,
        "n_params": int(trained["n_params"]),
        "train_wall_s": float(trained["train_wall_s"]),
        "final_loss": float(trained["final_loss"]),
        "history": trained["history"],
        "mix_high_fraction": float(data["mix_high_fraction"][0]),
        "gamma": float(mrp.gamma),
        "n_transitions": int(job["n_transitions"]),
        "n_euler": int(job["n_euler"]),
        "chain_depth": int(mrp.chain_depth),
        "n_states": int(mrp.n_states),
        "farthest_state": farthest,
        "farthest_w1": float(farthest_w1),
        "names": list(mrp.names),
    }
    if job.get("save_hist"):
        out["hist"] = _histograms(ev["samples"], mrp)
    ckpt = job.get("ckpt_path")
    if ckpt:
        path = Path(ckpt)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "method": method,
                "state_dict": trained["module"].state_dict(),
                "n_states": mrp.n_states,
                "names": list(mrp.names),
                "hidden": int(job["hidden"]),
                "embed_dim": int(job["embed_dim"]),
                "n_quantiles": int(job["n_quantiles"]),
                "gamma": float(mrp.gamma),
                "n_euler": int(job["n_euler"]),
                "reward_scale": bool(job["reward_scale"]),
            },
            path,
        )
        out["ckpt"] = str(path)
    # Drop samples before returning across the process boundary.
    return out
