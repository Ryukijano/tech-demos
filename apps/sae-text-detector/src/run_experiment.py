"""Run exp1_cpu_auditable: DetectorScope tau=0.85 vs tau=0.00 on CPU.

Usage:
    python src/run_experiment.py                  # official run + breakpoints
    python src/run_experiment.py --smoke          # disclosed tiny code-path check
    python src/run_experiment.py --skip-breakpoints
"""

from __future__ import annotations

import argparse
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

from data_util import examples_frame, load_pair_pool, stratified_subsample  # noqa: E402
from detector_util import (  # noqa: E402
    extract_features,
    load_detector,
    metrics_for,
    rescore_topk,
    score_cached,
)


def _run_seed(detector, examples, cfg) -> dict:
    texts = [e.text for e in examples]
    y = np.asarray([e.label for e in examples], dtype=int)
    x, extract_wall = extract_features(
        detector, texts, batch_size=int(cfg["runtime"]["batch_size"])
    )
    head = cfg["model"]["head"]
    sparse = score_cached(detector, x, head, float(cfg["model"]["tau_sparse"]))
    dense = score_cached(detector, x, head, float(cfg["model"]["tau_dense"]))
    m_s = metrics_for(y, sparse.probs, sparse.n_active)
    m_d = metrics_for(y, dense.probs, dense.n_active)
    gap = m_d["auroc"] - m_s["auroc"]
    return {
        "n": int(len(examples)),
        "n_human": int((y == 0).sum()),
        "n_ai": int((y == 1).sum()),
        "extract_wall_s": extract_wall,
        "score_wall_s": sparse.wall_s + dense.wall_s,
        "auroc_tau_sparse": m_s["auroc"],
        "auroc_tau_dense": m_d["auroc"],
        "auroc_gap": gap,
        "accuracy_tau_sparse": m_s["accuracy"],
        "accuracy_tau_dense": m_d["accuracy"],
        "mean_n_active_tau_sparse": m_s["mean_n_active_features"],
        "mean_n_active_tau_dense": m_d["mean_n_active_features"],
        "tau_sparse": float(cfg["model"]["tau_sparse"]),
        "tau_dense": float(cfg["model"]["tau_dense"]),
        "probs_sparse": sparse.probs.tolist(),
        "probs_dense": dense.probs.tolist(),
        "labels": y.tolist(),
        "feature_cache_path": None,
    }, x, y


def _breakpoints(detector, examples, x, y, cfg) -> dict:
    bp = cfg["breakpoints"]
    head = cfg["model"]["head"]
    tau_s = float(cfg["model"]["tau_sparse"])
    out = {"truncate": [], "tau_sweep": [], "topk": []}

    # Truncation on a balanced 32-example subset (CPU time); tau/topk use full x.
    rng = np.random.default_rng(0)
    humans = [e for e in examples if e.label == 0]
    ais = [e for e in examples if e.label == 1]
    n_take = min(16, len(humans), len(ais))
    sub = (
        [humans[i] for i in rng.choice(len(humans), size=n_take, replace=False)]
        + [ais[i] for i in rng.choice(len(ais), size=n_take, replace=False)]
    )
    y_sub = np.asarray([e.label for e in sub], dtype=int)
    for n_chars in bp["truncate_chars"]:
        trunc = [" ".join(e.text.split())[:n_chars] for e in sub]
        xt, wall = extract_features(
            detector, trunc, batch_size=int(cfg["runtime"]["batch_size"])
        )
        s = score_cached(detector, xt, head, tau_s)
        d = score_cached(detector, xt, head, float(cfg["model"]["tau_dense"]))
        ms = metrics_for(y_sub, s.probs, s.n_active)
        md = metrics_for(y_sub, d.probs, d.n_active)
        out["truncate"].append(
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

    for tau in bp["tau_sweep"]:
        sc = score_cached(detector, x, head, float(tau))
        m = metrics_for(y, sc.probs, sc.n_active)
        out["tau_sweep"].append(
            {
                "tau": float(tau),
                "auroc": m["auroc"],
                "accuracy": m["accuracy"],
                "mean_n_active_features": m["mean_n_active_features"],
            }
        )

    for k in bp["topk_features"]:
        logits = rescore_topk(detector, x, head, tau_s, int(k))
        probs = 1.0 / (1.0 + np.exp(-logits))
        auroc = float(roc_auc_score(y, probs))
        out["topk"].append({"k": int(k), "auroc": auroc})

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-breakpoints", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    WORK.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        cfg = dict(cfg)
        cfg["data"] = dict(cfg["data"])
        cfg["data"]["pool_pairs"] = 20
        cfg["data"]["n_per_class"] = 8
        cfg["seeds"] = [0]
        out_path = EXP / "smoke.json"
    else:
        out_path = EXP / "raw.json"

    print("[load] dataset pool", flush=True)
    pool = load_pair_pool(
        cfg["data"]["dataset"],
        int(cfg["data"]["pool_pairs"]),
        int(cfg["data"]["pool_seed"]),
        int(cfg["data"]["min_chars"]),
        int(cfg["data"]["max_chars"]),
    )
    examples_frame(pool).to_csv(WORK / "pool.csv", index=False)

    print("[load] detector", cfg["model"]["hub_id"], flush=True)
    t_load = time.perf_counter()
    detector = load_detector(
        cfg["model"]["hub_id"], torch_threads=int(cfg["runtime"]["torch_threads"])
    )
    load_wall = time.perf_counter() - t_load
    print(f"[load] done in {load_wall:.1f}s", flush=True)

    raw = {
        "exp_id": cfg["exp_id"],
        "hypothesis": cfg["hypothesis"],
        "load_wall_s": load_wall,
        "device": "cpu",
        "torch": torch.__version__,
        "seeds": {},
        "breakpoints": None,
    }

    cache_x0 = None
    cache_y0 = None
    cache_ex0 = None

    for seed in cfg["seeds"]:
        print(f"[seed] {seed}", flush=True)
        examples = stratified_subsample(
            pool, int(cfg["data"]["n_per_class"]), int(seed)
        )
        examples_frame(examples).to_csv(WORK / f"seed{seed}_examples.csv", index=False)
        result, x, y = _run_seed(detector, examples, cfg)
        # Drop bulky per-example probs from printed summary but keep in raw.
        print(
            f"  auroc_sparse={result['auroc_tau_sparse']:.4f} "
            f"auroc_dense={result['auroc_tau_dense']:.4f} "
            f"gap={result['auroc_gap']:.4f} "
            f"n_active_s={result['mean_n_active_tau_sparse']:.1f}",
            flush=True,
        )
        torch.save({"x": x, "y": y}, WORK / f"features_seed{seed}.pt")
        raw["seeds"][str(seed)] = result
        if int(seed) == int(cfg["breakpoints"]["seed"]):
            cache_x0, cache_y0, cache_ex0 = x, y, examples

    # Checkpoint seed metrics before the slower breakpoint sweep.
    out_path.write_text(json.dumps(raw, indent=2))
    print(f"[write] seeds → {out_path}", flush=True)

    if not args.skip_breakpoints and cache_x0 is not None and not args.smoke:
        print("[breakpoints] seed", cfg["breakpoints"]["seed"], flush=True)
        raw["breakpoints"] = _breakpoints(
            detector, cache_ex0, cache_x0, cache_y0, cfg
        )
        out_path.write_text(json.dumps(raw, indent=2))
        print(f"[write] +breakpoints → {out_path}", flush=True)


if __name__ == "__main__":
    main()
