"""Rebuild raw.json + breakpoints from cached SAE features (CPU recovery path)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_auditable"
WORK = APP / "work"
sys.path.insert(0, str(APP / "src"))

from data_util import load_pair_pool, stratified_subsample  # noqa: E402
from detector_util import (  # noqa: E402
    extract_features,
    load_detector,
    metrics_for,
    rescore_topk,
    score_cached,
)


def main() -> None:
    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    detector = load_detector(
        cfg["model"]["hub_id"], torch_threads=int(cfg["runtime"]["torch_threads"])
    )
    head = cfg["model"]["head"]
    tau_s = float(cfg["model"]["tau_sparse"])
    tau_d = float(cfg["model"]["tau_dense"])

    raw = {
        "exp_id": cfg["exp_id"],
        "hypothesis": cfg["hypothesis"],
        "load_wall_s": None,
        "device": "cpu",
        "torch": torch.__version__,
        "recovery": "finish_from_cache",
        "seeds": {},
        "breakpoints": None,
    }

    pool = load_pair_pool(
        cfg["data"]["dataset"],
        int(cfg["data"]["pool_pairs"]),
        int(cfg["data"]["pool_seed"]),
        int(cfg["data"]["min_chars"]),
        int(cfg["data"]["max_chars"]),
    )

    cache_x0 = cache_y0 = cache_ex0 = None
    for seed in cfg["seeds"]:
        blob = torch.load(WORK / f"features_seed{seed}.pt", weights_only=False)
        x, y = blob["x"], np.asarray(blob["y"], dtype=int)
        examples = stratified_subsample(
            pool, int(cfg["data"]["n_per_class"]), int(seed)
        )
        # Sanity: labels must match the cached y order.
        y_check = np.asarray([e.label for e in examples], dtype=int)
        if y_check.shape != y.shape or not np.array_equal(y_check, y):
            # Prefer cached labels; examples only needed for truncation texts.
            print(f"[warn] seed {seed}: example order/labels differ from cache; "
                  f"using cached y for metrics", flush=True)
        sparse = score_cached(detector, x, head, tau_s)
        dense = score_cached(detector, x, head, tau_d)
        m_s = metrics_for(y, sparse.probs, sparse.n_active)
        m_d = metrics_for(y, dense.probs, dense.n_active)
        gap = m_d["auroc"] - m_s["auroc"]
        raw["seeds"][str(seed)] = {
            "n": int(len(y)),
            "n_human": int((y == 0).sum()),
            "n_ai": int((y == 1).sum()),
            "extract_wall_s": None,
            "score_wall_s": sparse.wall_s + dense.wall_s,
            "auroc_tau_sparse": m_s["auroc"],
            "auroc_tau_dense": m_d["auroc"],
            "auroc_gap": gap,
            "accuracy_tau_sparse": m_s["accuracy"],
            "accuracy_tau_dense": m_d["accuracy"],
            "mean_n_active_tau_sparse": m_s["mean_n_active_features"],
            "mean_n_active_tau_dense": m_d["mean_n_active_features"],
            "tau_sparse": tau_s,
            "tau_dense": tau_d,
            "probs_sparse": sparse.probs.tolist(),
            "probs_dense": dense.probs.tolist(),
            "labels": y.tolist(),
        }
        print(
            f"[seed] {seed} auroc_s={m_s['auroc']:.4f} auroc_d={m_d['auroc']:.4f} "
            f"gap={gap:.4f}",
            flush=True,
        )
        if int(seed) == int(cfg["breakpoints"]["seed"]):
            cache_x0, cache_y0, cache_ex0 = x, y, examples

    # Breakpoints: tau + topk on full cached features; truncate on 32 texts.
    assert cache_x0 is not None and cache_ex0 is not None
    bp = {"truncate": [], "tau_sweep": [], "topk": []}
    for tau in cfg["breakpoints"]["tau_sweep"]:
        sc = score_cached(detector, cache_x0, head, float(tau))
        m = metrics_for(cache_y0, sc.probs, sc.n_active)
        bp["tau_sweep"].append(
            {
                "tau": float(tau),
                "auroc": m["auroc"],
                "accuracy": m["accuracy"],
                "mean_n_active_features": m["mean_n_active_features"],
            }
        )
    for k in cfg["breakpoints"]["topk_features"]:
        logits = rescore_topk(detector, cache_x0, head, tau_s, int(k))
        probs = 1.0 / (1.0 + np.exp(-logits))
        bp["topk"].append({"k": int(k), "auroc": float(roc_auc_score(cache_y0, probs))})

    # Truncation on a fixed 32-example subset of seed 0 (balanced).
    rng = np.random.default_rng(0)
    humans = [e for e in cache_ex0 if e.label == 0]
    ais = [e for e in cache_ex0 if e.label == 1]
    take_h = [humans[i] for i in rng.choice(len(humans), size=16, replace=False)]
    take_a = [ais[i] for i in rng.choice(len(ais), size=16, replace=False)]
    sub = take_h + take_a
    y_sub = np.asarray([e.label for e in sub], dtype=int)
    for n_chars in cfg["breakpoints"]["truncate_chars"]:
        trunc = [" ".join(e.text.split())[: int(n_chars)] for e in sub]
        xt, wall = extract_features(
            detector, trunc, batch_size=int(cfg["runtime"]["batch_size"])
        )
        s = score_cached(detector, xt, head, tau_s)
        d = score_cached(detector, xt, head, tau_d)
        ms = metrics_for(y_sub, s.probs, s.n_active)
        md = metrics_for(y_sub, d.probs, d.n_active)
        bp["truncate"].append(
            {
                "max_chars": int(n_chars),
                "auroc_sparse": ms["auroc"],
                "auroc_dense": md["auroc"],
                "auroc_gap": md["auroc"] - ms["auroc"],
                "mean_n_active_sparse": ms["mean_n_active_features"],
                "extract_wall_s": wall,
                "n": int(len(sub)),
            }
        )
        print(f"[truncate] {n_chars} gap={md['auroc']-ms['auroc']:.4f}", flush=True)

    raw["breakpoints"] = bp
    raw["breakpoints_note"] = (
        "tau/topk on full seed-0 cached features; truncate on a balanced "
        "32-example subset for CPU time"
    )
    (EXP / "raw.json").write_text(json.dumps(raw, indent=2))
    print("[write]", EXP / "raw.json", flush=True)


if __name__ == "__main__":
    main()
