"""Run exp1_cpu_addition: looped vs parameter-matched baseline on addition.

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

from looped_transformer import (
    evaluate_exact_match,
    set_threads,
    train_model,
)

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_addition"
WORK = APP / "work"


def _train_one(kind: str, seed: int, cfg: dict, steps_override: int | None = None):
    m = cfg["model"]
    t = cfg["training"]
    d = cfg["data"]
    n_loops = int(m["n_loops_train"]) if kind == "looped" else int(m["n_loops_baseline"])
    steps = int(t["steps"]) if steps_override is None else int(steps_override)
    print(f"[train] {kind} seed={seed} loops={n_loops} steps={steps}", flush=True)
    tr = train_model(
        kind,
        seed,
        n_digits=int(d["n_digits_train"]),
        steps=steps,
        batch_size=int(t["batch_size"]),
        lr=float(t["lr"]),
        d_model=int(m["d_model"]),
        n_layers=int(m["n_layers"]),
        n_heads=int(m["n_heads"]),
        dropout=float(m["dropout"]),
        ff_mult=int(m["ff_mult"]),
        n_loops=n_loops,
        weight_decay=float(t["weight_decay"]),
        grad_clip=float(t["grad_clip"]),
        log_every=int(t["log_every"]),
        stream_seed=int(d["train_stream_seed"]),
    )
    return kind, seed, tr


def _eval_model(model, cfg, n_digits=None, n_loops=None, n_eval=None, eval_seed=None):
    d = cfg["data"]
    return evaluate_exact_match(
        model,
        n_digits=int(d["n_digits_eval"] if n_digits is None else n_digits),
        n_eval=int(d["n_eval"] if n_eval is None else n_eval),
        eval_seed=int(d["eval_stream_seed"] if eval_seed is None else eval_seed),
        n_loops=n_loops,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-breakpoints", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    set_threads(int(cfg["runtime"]["torch_threads"]))
    WORK.mkdir(exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)

    seeds = list(cfg["seeds"])
    t_wall0 = time.perf_counter()

    if args.smoke:
        # Disclosed tiny path check — does not change thresholds.
        smoke_cfg = json.loads(json.dumps(cfg))  # deep-ish copy via json
        smoke_cfg["training"]["steps"] = 40
        smoke_cfg["data"]["n_eval"] = 32
        smoke_cfg["training"]["batch_size"] = 16
        smoke_cfg["training"]["log_every"] = 20
        kind, seed, tr = _train_one("looped", 0, smoke_cfg)
        metrics = _eval_model(tr.model, smoke_cfg, n_eval=32)
        smoke = {
            "kind": kind,
            "seed": seed,
            "steps": 40,
            "final_loss": tr.final_loss,
            "n_params": tr.n_params,
            "train_wall_s": tr.train_wall_s,
            "metrics": metrics,
            "note": "Disclosed smoke only; thresholds unchanged.",
        }
        (EXP / "smoke.json").write_text(json.dumps(smoke, indent=2))
        print("smoke done:", json.dumps(smoke, indent=2))
        return

    # Parallelize independent (kind, seed) trains — CPU-friendly.
    jobs = [(k, s) for s in seeds for k in ("looped", "baseline")]
    results: dict[str, dict] = {str(s): {"kernels": {}} for s in seeds}
    n_params = None

    # Sequential is more predictable for CPU contention; still train two at a
    # time when cores allow.
    max_workers = 2
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_train_one, k, s, cfg): (k, s) for k, s in jobs}
        for fut in as_completed(futs):
            kind, seed, tr = fut.result()
            ckpt_path = WORK / f"model_{kind}_seed{seed}.pt"
            torch.save(
                {
                    "state_dict": tr.model.state_dict(),
                    "model_cfg": {
                        "d_model": cfg["model"]["d_model"],
                        "n_layers": cfg["model"]["n_layers"],
                        "n_heads": cfg["model"]["n_heads"],
                        "dropout": cfg["model"]["dropout"],
                        "ff_mult": cfg["model"]["ff_mult"],
                        "n_loops": tr.model.n_loops,
                    },
                    "kind": kind,
                    "seed": seed,
                    "final_loss": tr.final_loss,
                    "train_wall_s": tr.train_wall_s,
                    "n_params": tr.n_params,
                    "history": tr.history,
                },
                ckpt_path,
            )
            metrics = _eval_model(tr.model, cfg)
            metrics["train_wall_s"] = tr.train_wall_s
            metrics["final_loss"] = tr.final_loss
            metrics["n_params"] = tr.n_params
            results[str(seed)]["kernels"][kind] = metrics
            n_params = tr.n_params
            print(
                f"[eval] {kind} seed={seed} exact={metrics['exact_match']:.3f} "
                f"digit={metrics['digit_accuracy']:.3f}",
                flush=True,
            )

    raw: dict = {
        "exp_id": cfg["exp_id"],
        "n_params": n_params,
        "seeds": results,
        "breakpoints": {},
        "wall_s_total": None,
        "config_snapshot": {
            "n_digits_train": cfg["data"]["n_digits_train"],
            "n_eval": cfg["data"]["n_eval"],
            "steps": cfg["training"]["steps"],
            "n_loops_train": cfg["model"]["n_loops_train"],
            "n_loops_baseline": cfg["model"]["n_loops_baseline"],
        },
    }

    if not args.skip_breakpoints:
        bp_seed = int(cfg["breakpoints"]["seed"])
        # Load trained looped + baseline for seed 0
        from looped_transformer import TinyLoopedTransformer

        def _load(kind: str, seed: int):
            ckpt = torch.load(WORK / f"model_{kind}_seed{seed}.pt", weights_only=False)
            mc = ckpt["model_cfg"]
            m = TinyLoopedTransformer(
                d_model=int(mc["d_model"]),
                n_layers=int(mc["n_layers"]),
                n_heads=int(mc["n_heads"]),
                dropout=float(mc.get("dropout", 0.0)),
                ff_mult=int(mc.get("ff_mult", 4)),
                n_loops=int(mc["n_loops"]),
            )
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            return m

        looped = _load("looped", bp_seed)
        baseline = _load("baseline", bp_seed)

        # Test-time loops on looped model
        ttl = []
        for k in cfg["breakpoints"]["test_time_loops"]["n_loops"]:
            m = _eval_model(looped, cfg, n_loops=int(k), eval_seed=cfg["data"]["eval_stream_seed"] + int(k))
            ttl.append(
                {
                    "n_loops": int(k),
                    "looped_exact_match": m["exact_match"],
                    "looped_digit_accuracy": m["digit_accuracy"],
                }
            )
            print(f"[bp] test-time loops={k} exact={m['exact_match']:.3f}", flush=True)
        raw["breakpoints"]["test_time_loops"] = ttl

        # Longer / shorter digits (both models at their train K)
        longer = []
        for nd in cfg["breakpoints"]["longer_digits"]["n_digits"]:
            ml = _eval_model(looped, cfg, n_digits=int(nd), eval_seed=910000 + int(nd))
            mb = _eval_model(baseline, cfg, n_digits=int(nd), eval_seed=910000 + int(nd))
            longer.append(
                {
                    "n_digits": int(nd),
                    "looped_exact_match": ml["exact_match"],
                    "baseline_exact_match": mb["exact_match"],
                    "looped_digit_accuracy": ml["digit_accuracy"],
                    "baseline_digit_accuracy": mb["digit_accuracy"],
                }
            )
            print(
                f"[bp] digits={nd} looped={ml['exact_match']:.3f} "
                f"baseline={mb['exact_match']:.3f}",
                flush=True,
            )
        raw["breakpoints"]["longer_digits"] = longer

        # Train-steps ablation (retrain seed 0 both kinds at 500 steps)
        tsa = []
        for steps in cfg["breakpoints"]["train_steps_ablation"]["steps"]:
            if int(steps) == int(cfg["training"]["steps"]):
                # reuse primary
                kl = results[str(bp_seed)]["kernels"]["looped"]
                kb = results[str(bp_seed)]["kernels"]["baseline"]
                tsa.append(
                    {
                        "steps": int(steps),
                        "looped_exact_match": kl["exact_match"],
                        "baseline_exact_match": kb["exact_match"],
                    }
                )
                continue
            row = {}
            for kind in ("looped", "baseline"):
                _, _, tr = _train_one(kind, bp_seed, cfg, steps_override=int(steps))
                met = _eval_model(tr.model, cfg, eval_seed=820000 + int(steps))
                row[f"{kind}_exact_match"] = met["exact_match"]
            row["steps"] = int(steps)
            tsa.append(row)
            print(
                f"[bp] steps={steps} looped={row['looped_exact_match']:.3f} "
                f"baseline={row['baseline_exact_match']:.3f}",
                flush=True,
            )
        raw["breakpoints"]["train_steps_ablation"] = tsa

    raw["wall_s_total"] = time.perf_counter() - t_wall0
    (EXP / "raw.json").write_text(json.dumps(raw, indent=2))
    print(f"wrote {EXP / 'raw.json'} total_wall_s={raw['wall_s_total']:.1f}")


if __name__ == "__main__":
    main()
