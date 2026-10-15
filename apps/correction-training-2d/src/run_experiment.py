"""Run exp1_cpu_2d: pre-registered K>0 vs K=0 matched-NFE comparison + breakpoints.

Writes experiments/exp1_cpu_2d/raw.json and work/model_seed0.pt (for media).

Usage:
    python src/run_experiment.py                  # official run (seeds 0/1/2 + breakpoints)
    python src/run_experiment.py --smoke          # tiny disclosed smoke to verify the code path
    python src/run_experiment.py --skip-breakpoints
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from ctd2d import (
    make_ring_data,
    median_heuristic_gamma,
    mmd2_unbiased,
    nearest_mode_stats,
    sample,
    set_threads,
    sliced_w1,
    steps_for_budget,
    train_joint,
)

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_2d"
WORK = APP / "work"


def evaluate(D, C, *, K, nfe_budget, n_samples, eval_data, gamma, data_cfg, sample_seed):
    steps = steps_for_budget(nfe_budget, K)
    gen = torch.Generator().manual_seed(sample_seed)
    t0 = time.perf_counter()
    samples, nfe, _ = sample(D, C, n_samples, steps=steps, K=K, generator=gen)
    wall = time.perf_counter() - t0
    nm_mse, coverage = nearest_mode_stats(samples, data_cfg["n_modes"], data_cfg["radius"])
    return {
        "K": K,
        "steps": steps,
        "nfe": nfe,
        "nfe_budget": nfe_budget,
        "mmd2": mmd2_unbiased(samples, eval_data, gamma),
        "sliced_w1": sliced_w1(samples, eval_data),
        "nearest_mode_mse": nm_mse,
        "mode_coverage": coverage,
        "sample_wall_s": wall,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny disclosed smoke run")
    ap.add_argument("--skip-breakpoints", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    set_threads(int(cfg["runtime"]["torch_threads"]))
    WORK.mkdir(exist_ok=True)

    data_cfg = {k: cfg["data"][k] for k in ("n_modes", "radius", "mode_std")}
    model_cfg = cfg["model"]
    train_kw = dict(
        n_modes=data_cfg["n_modes"],
        radius=data_cfg["radius"],
        mode_std=data_cfg["mode_std"],
        iters=int(model_cfg["iters"]),
        batch=int(model_cfg["batch"]),
        lr=float(model_cfg["lr"]),
        corr_weight=float(model_cfg["corr_weight"]),
    )
    nfe_budget = int(cfg["sampling"]["nfe_budget"])
    k_values = list(cfg["sampling"]["K_values"])
    n_samples = int(cfg["sampling"]["n_samples"])
    seeds = list(cfg["seeds"])

    if args.smoke:
        train_kw["iters"] = 100
        n_samples = 200
        seeds = [0]
        args.skip_breakpoints = True

    # Fixed eval set + kernel bandwidth, independent of training seeds (pre-registered).
    eval_gen = torch.Generator().manual_seed(9999)
    eval_data = make_ring_data(2000 if not args.smoke else 400, generator=eval_gen, **data_cfg)
    gamma = median_heuristic_gamma(eval_data)

    raw: dict = {
        "smoke": args.smoke,
        "gamma": gamma,
        "torch": torch.__version__,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": {},
        "breakpoints": {},
    }

    # ---------------- main pre-registered comparison ----------------
    for seed in seeds:
        print(f"[seed {seed}] training (iters={train_kw['iters']}) ...", flush=True)
        tr = train_joint(seed, **train_kw)
        print(f"[seed {seed}] trained in {tr.wall_s:.1f}s "
              f"(denoise loss {tr.final_denoise_loss:.4f}, corr loss {tr.final_corr_loss:.4f})")
        if seed == 0 and not args.smoke:
            torch.save(
                {"D": tr.denoiser.state_dict(), "C": tr.corrector.state_dict(),
                 "data_cfg": data_cfg, "model_cfg": model_cfg},
                WORK / "model_seed0.pt",
            )
        rows = []
        for K in k_values:
            r = evaluate(
                tr.denoiser, tr.corrector, K=K, nfe_budget=nfe_budget,
                n_samples=n_samples, eval_data=eval_data, gamma=gamma,
                data_cfg=data_cfg, sample_seed=10_000 + seed,
            )
            rows.append(r)
            print(f"  K={K:>2} steps={r['steps']:>3} nfe={r['nfe']:>3} "
                  f"mmd2={r['mmd2']:.6f} w1={r['sliced_w1']:.4f} "
                  f"nm_mse={r['nearest_mode_mse']:.4f} cov={r['mode_coverage']:.2f} "
                  f"({r['sample_wall_s']:.1f}s)", flush=True)
        raw["seeds"][str(seed)] = {
            "train_wall_s": tr.wall_s,
            "final_denoise_loss": tr.final_denoise_loss,
            "final_corr_loss": tr.final_corr_loss,
            "rows": rows,
        }

    # ---------------- breakpoints (seed 0) ----------------
    if not args.skip_breakpoints:
        bp_cfg = cfg["breakpoints"]
        bp_seed = int(bp_cfg["seed"])
        print("[breakpoints] retraining seed-0 model reference ...", flush=True)
        tr0 = train_joint(bp_seed, **train_kw)

        def ev(D, C, K, budget, e_data, e_gamma, d_cfg):
            return evaluate(D, C, K=K, nfe_budget=budget, n_samples=n_samples,
                            eval_data=e_data, gamma=e_gamma, data_cfg=d_cfg,
                            sample_seed=20_000 + bp_seed)

        raw["breakpoints"]["k_extreme"] = [
            ev(tr0.denoiser, tr0.corrector, K, nfe_budget, eval_data, gamma, data_cfg)
            for K in bp_cfg["k_extreme"]["K_values"]
        ]
        print("[breakpoints] k_extreme done", flush=True)

        raw["breakpoints"]["nfe_sweep"] = [
            ev(tr0.denoiser, tr0.corrector, K, budget, eval_data, gamma, data_cfg)
            for budget in bp_cfg["nfe_sweep"]["budgets"]
            for K in bp_cfg["nfe_sweep"]["K_values"]
        ]
        print("[breakpoints] nfe_sweep done", flush=True)

        difficulty = []
        diff = bp_cfg["difficulty"]
        settings = [("mode_std", v) for v in diff["mode_std_values"]] + [
            ("n_modes", v) for v in diff["n_modes_values"]
        ]
        for key, val in settings:
            d_cfg = dict(data_cfg)
            d_cfg[key] = val
            kw = dict(train_kw)
            kw.update(n_modes=d_cfg["n_modes"], radius=d_cfg["radius"], mode_std=d_cfg["mode_std"])
            print(f"[breakpoints] difficulty {key}={val}: training ...", flush=True)
            tr = train_joint(bp_seed, **kw)
            e_gen = torch.Generator().manual_seed(9999)
            e_data = make_ring_data(2000, generator=e_gen, **d_cfg)
            e_gamma = median_heuristic_gamma(e_data)
            for K in [0, 1, 2, 4, 8]:
                r = ev(tr.denoiser, tr.corrector, K, nfe_budget, e_data, e_gamma, d_cfg)
                r["setting"] = f"{key}={val}"
                difficulty.append(r)
        raw["breakpoints"]["difficulty"] = difficulty

    raw["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = EXP / ("smoke.json" if args.smoke else "raw.json")
    out.write_text(json.dumps(raw, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
