"""Run exp1_cpu_sudoku: binary-symmetric vs masked discrete diffusion on Sudoku.

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

import numpy as np
import torch
import yaml

from bdd import (
    evaluate_predictions,
    make_puzzle_pool,
    punch_clues,
    sample_binary,
    sample_masked,
    set_threads,
    train_kernel,
)

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_sudoku"
WORK = APP / "work"


def _eval_kernel(model, kernel, solutions, n_clues, nfe, t_min, sample_seed):
    rng = np.random.default_rng(sample_seed)
    clues_np, mask_np = punch_clues(solutions, n_clues, rng)
    b = solutions.shape[0]
    clue_mask = torch.as_tensor(mask_np.reshape(b, 81))
    # Only clue cells carry digit ids; blanks are 0 placeholders (ignored via mask).
    clean_digits = torch.as_tensor(solutions.reshape(b, 81) - 1, dtype=torch.long)
    clues = torch.where(clue_mask, clean_digits, torch.zeros_like(clean_digits))

    g = torch.Generator().manual_seed(sample_seed)
    t0 = time.perf_counter()
    if kernel == "binary":
        pred_t, _ = sample_binary(
            model, clues, clue_mask, nfe=nfe, t_min=t_min, generator=g
        )
    else:
        pred_t, _ = sample_masked(
            model, clues, clue_mask, nfe=nfe, t_min=t_min, generator=g
        )
    wall = time.perf_counter() - t0
    pred = pred_t.cpu().numpy().reshape(b, 9, 9) + 1
    metrics = evaluate_predictions(pred, solutions, mask_np)
    metrics["sample_wall_s"] = wall
    metrics["nfe"] = nfe
    metrics["n_clues"] = n_clues
    return metrics


def _train_one(kernel, seed, pool, cfg):
    mcfg = cfg["model"]
    tcfg = cfg["training"]
    print(f"[train] {kernel} seed={seed}", flush=True)
    tr = train_kernel(
        kernel,
        seed,
        pool,
        steps=int(tcfg["steps"]),
        batch_size=int(tcfg["batch_size"]),
        lr=float(tcfg["lr"]),
        n_clues=int(cfg["data"]["n_clues_train"]),
        d_model=int(mcfg["d_model"]),
        n_layers=int(mcfg["n_layers"]),
        n_heads=int(mcfg["n_heads"]),
        dropout=float(mcfg["dropout"]),
    )
    return kernel, seed, tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-breakpoints", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    set_threads(int(cfg["runtime"]["torch_threads"]))
    WORK.mkdir(exist_ok=True)

    data = cfg["data"]
    seeds = list(cfg["seeds"])
    nfe = int(cfg["sampling"]["nfe"])
    t_min = float(cfg["sampling"]["t_min"])
    train_steps = int(cfg["training"]["steps"])
    n_train = int(data["n_train_puzzles"])
    n_eval = int(data["n_eval_puzzles"])

    if args.smoke:
        train_steps = 40
        cfg["training"]["steps"] = train_steps
        n_train = 32
        n_eval = 8
        seeds = [0]
        nfe = 8
        args.skip_breakpoints = True

    print(f"[data] building train pool ({n_train}) ...", flush=True)
    train_pool = make_puzzle_pool(n_train, int(data["train_grid_seed"]))
    print(f"[data] building eval pool ({n_eval}) ...", flush=True)
    eval_pool = make_puzzle_pool(n_eval, int(data["eval_grid_seed"]))

    raw: dict = {
        "smoke": args.smoke,
        "torch": torch.__version__,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_params": None,
        "seeds": {},
        "breakpoints": {},
    }

    # Train 2 kernels × N seeds. Parallelize across (kernel, seed) pairs with
    # threads (GIL released in torch ops) for wall-time savings on multi-core.
    jobs = [(k, s) for s in seeds for k in ("binary", "masked")]
    results = {}
    # Cap parallelism to avoid oversubscription with torch threads.
    max_workers = 2 if not args.smoke else 1
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(_train_one, k, s, train_pool, cfg): (k, s) for k, s in jobs
        }
        for fut in as_completed(futs):
            k, s, tr = fut.result()
            results[(k, s)] = tr
            if raw["n_params"] is None:
                raw["n_params"] = tr.n_params
            if s == 0 and k == "binary" and not args.smoke:
                torch.save(
                    {
                        "state_dict": tr.model.state_dict(),
                        "kernel": k,
                        "model_cfg": cfg["model"],
                        "seed": s,
                    },
                    WORK / "model_binary_seed0.pt",
                )
            if s == 0 and k == "masked" and not args.smoke:
                torch.save(
                    {
                        "state_dict": tr.model.state_dict(),
                        "kernel": k,
                        "model_cfg": cfg["model"],
                        "seed": s,
                    },
                    WORK / "model_masked_seed0.pt",
                )

    for s in seeds:
        entry = {"kernels": {}}
        for k in ("binary", "masked"):
            tr = results[(k, s)]
            metrics = _eval_kernel(
                tr.model,
                k,
                eval_pool,
                n_clues=int(data["n_clues_eval"]),
                nfe=nfe,
                t_min=t_min,
                sample_seed=10_000 + s,
            )
            entry["kernels"][k] = {
                "train_wall_s": tr.wall_s,
                "final_loss": tr.final_loss,
                "n_params": tr.n_params,
                **metrics,
            }
            print(
                f"[eval] seed={s} {k}: solve={metrics['solve_rate']:.3f} "
                f"cell={metrics['cell_accuracy']:.3f} valid={metrics['valid_board_rate']:.3f} "
                f"train={tr.wall_s:.1f}s sample={metrics['sample_wall_s']:.1f}s",
                flush=True,
            )
        raw["seeds"][str(s)] = entry

    if not args.skip_breakpoints:
        bp = cfg["breakpoints"]
        bp_seed = int(bp["seed"])
        binary_m = results[("binary", bp_seed)].model
        masked_m = results[("masked", bp_seed)].model
        # fewer clues
        clue_rows = []
        for nc in bp["fewer_clues"]["n_clues"]:
            row = {"n_clues": nc}
            for k, m in (("binary", binary_m), ("masked", masked_m)):
                met = _eval_kernel(
                    m, k, eval_pool, n_clues=int(nc), nfe=nfe, t_min=t_min,
                    sample_seed=20_000 + int(nc),
                )
                row[f"{k}_solve_rate"] = met["solve_rate"]
                row[f"{k}_cell_accuracy"] = met["cell_accuracy"]
            clue_rows.append(row)
            print(f"[bp clues={nc}] binary={row['binary_solve_rate']:.3f} "
                  f"masked={row['masked_solve_rate']:.3f}", flush=True)
        raw["breakpoints"]["fewer_clues"] = clue_rows

        # fewer steps
        step_rows = []
        for nf in bp["fewer_steps"]["nfe"]:
            row = {"nfe": nf}
            for k, m in (("binary", binary_m), ("masked", masked_m)):
                met = _eval_kernel(
                    m, k, eval_pool, n_clues=int(data["n_clues_eval"]),
                    nfe=int(nf), t_min=t_min, sample_seed=30_000 + int(nf),
                )
                row[f"{k}_solve_rate"] = met["solve_rate"]
                row[f"{k}_cell_accuracy"] = met["cell_accuracy"]
            step_rows.append(row)
            print(f"[bp nfe={nf}] binary={row['binary_solve_rate']:.3f} "
                  f"masked={row['masked_solve_rate']:.3f}", flush=True)
        raw["breakpoints"]["fewer_steps"] = step_rows

        # train-steps ablation (retrain both kernels at reduced budget)
        abl_rows = []
        for st in bp["train_steps_ablation"]["steps"]:
            if int(st) == int(cfg["training"]["steps"]):
                # reuse primary
                bsr = raw["seeds"][str(bp_seed)]["kernels"]["binary"]["solve_rate"]
                msr = raw["seeds"][str(bp_seed)]["kernels"]["masked"]["solve_rate"]
                abl_rows.append({
                    "steps": st,
                    "binary_solve_rate": bsr,
                    "masked_solve_rate": msr,
                    "reused_primary": True,
                })
                continue
            cfg_ab = json.loads(json.dumps(cfg))
            cfg_ab["training"]["steps"] = int(st)
            _, _, tr_b = _train_one("binary", bp_seed, train_pool, cfg_ab)
            _, _, tr_m = _train_one("masked", bp_seed, train_pool, cfg_ab)
            mb = _eval_kernel(
                tr_b.model, "binary", eval_pool,
                n_clues=int(data["n_clues_eval"]), nfe=nfe, t_min=t_min,
                sample_seed=40_000 + int(st),
            )
            mm = _eval_kernel(
                tr_m.model, "masked", eval_pool,
                n_clues=int(data["n_clues_eval"]), nfe=nfe, t_min=t_min,
                sample_seed=40_000 + int(st),
            )
            abl_rows.append({
                "steps": st,
                "binary_solve_rate": mb["solve_rate"],
                "masked_solve_rate": mm["solve_rate"],
                "reused_primary": False,
            })
            print(f"[bp steps={st}] binary={mb['solve_rate']:.3f} "
                  f"masked={mm['solve_rate']:.3f}", flush=True)
        raw["breakpoints"]["train_steps_ablation"] = abl_rows

    raw["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = EXP / ("smoke.json" if args.smoke else "raw.json")
    out.write_text(json.dumps(raw, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
