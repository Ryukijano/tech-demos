"""Run exp1_cpu_views: good-view JEPA cross-modal alignment vs baselines.

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

from jepa_toy import (
    alignment_metrics,
    build_world,
    init_random_encoder,
    set_threads,
    train_jepa,
    train_recon,
)

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_views"
WORK = APP / "work"


def _pack(tr_a, tr_b, metrics: dict) -> dict:
    return {
        "cka": metrics["procrustes_cka"],
        "procrustes_cka": metrics["procrustes_cka"],
        "procrustes_rmse": metrics["procrustes_rmse"],
        "mutual_knn_overlap": metrics["mutual_knn_overlap"],
        "n_eval": metrics["n_eval"],
        "final_loss_a": tr_a.final_loss,
        "final_loss_b": tr_b.final_loss,
        "train_wall_s": tr_a.train_wall_s + tr_b.train_wall_s,
        "n_params_a": tr_a.n_params,
        "n_params_b": tr_b.n_params,
        "history_a": tr_a.history,
        "history_b": tr_b.history,
    }


def _train_pair(
    kind: str,
    seed: int,
    cfg: dict,
    world,
    *,
    steps_override: int | None = None,
    emb_dim_override: int | None = None,
    view_mode: str = "good",
    bad_noise_std: float | None = None,
    bad_dropout_p: float | None = None,
):
    m = cfg["model"]
    t = cfg["training"]
    d = cfg["data"]
    emb = int(m["emb_dim"] if emb_dim_override is None else emb_dim_override)
    steps_j = int(t["steps"] if steps_override is None else steps_override)
    steps_r = int(t["recon_steps"] if steps_override is None else steps_override)
    arch_a = list(m["arch_a"])
    arch_b = list(m["arch_b"])
    drop = float(m.get("dropout", 0.0))
    bp = cfg.get("breakpoints", {}).get("bad_views", {})
    noise = float(bad_noise_std if bad_noise_std is not None else bp.get("noise_std", 0.5))
    drop_p = float(bad_dropout_p if bad_dropout_p is not None else bp.get("dropout_p", 0.3))

    common = dict(
        emb_dim=emb,
        batch_size=int(t["batch_size"]),
        weight_decay=float(t["weight_decay"]),
        grad_clip=float(t["grad_clip"]),
        log_every=int(t["log_every"]),
        stream_seed=int(d["train_stream_seed"]),
        dropout=drop,
    )

    print(
        f"[train] {kind} seed={seed} emb={emb} steps={steps_j} views={view_mode}",
        flush=True,
    )

    if kind == "jepa":
        tr_a = train_jepa(
            world,
            "a",
            seed,
            hidden=arch_a,
            steps=steps_j,
            lr=float(t["lr"]),
            sim_coef=float(t["sim_coef"]),
            std_coef=float(t["std_coef"]),
            cov_coef=float(t["cov_coef"]),
            std_target=float(t["std_target"]),
            view_mode=view_mode,
            bad_noise_std=noise,
            bad_dropout_p=drop_p,
            **common,
        )
        tr_b = train_jepa(
            world,
            "b",
            seed + 1000,
            hidden=arch_b,
            steps=steps_j,
            lr=float(t["lr"]),
            sim_coef=float(t["sim_coef"]),
            std_coef=float(t["std_coef"]),
            cov_coef=float(t["cov_coef"]),
            std_target=float(t["std_target"]),
            view_mode=view_mode,
            bad_noise_std=noise,
            bad_dropout_p=drop_p,
            **common,
        )
    elif kind == "recon":
        tr_a = train_recon(
            world,
            "a",
            seed,
            hidden=arch_a,
            steps=steps_r,
            lr=float(t["recon_lr"]),
            **common,
        )
        tr_b = train_recon(
            world,
            "b",
            seed + 1000,
            hidden=arch_b,
            steps=steps_r,
            lr=float(t["recon_lr"]),
            **common,
        )
    elif kind == "random":
        tr_a = init_random_encoder(world, "a", seed, arch_a, emb, dropout=drop)
        tr_b = init_random_encoder(world, "b", seed + 1000, arch_b, emb, dropout=drop)
    else:
        raise ValueError(kind)

    return kind, seed, tr_a, tr_b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-breakpoints", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    set_threads(int(cfg["runtime"]["torch_threads"]))
    WORK.mkdir(exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)

    world = build_world(cfg["data"])
    print(
        f"world states={world.n_states} classes={world.n_classes} "
        f"d_a={world.d_mod_a} d_b={world.d_mod_b} emb={cfg['model']['emb_dim']}",
        flush=True,
    )

    seeds = list(cfg["seeds"])
    t_wall0 = time.perf_counter()

    if args.smoke:
        smoke_cfg = json.loads(json.dumps(cfg))
        smoke_cfg["training"]["steps"] = 40
        smoke_cfg["training"]["recon_steps"] = 40
        smoke_cfg["training"]["batch_size"] = 64
        smoke_cfg["training"]["log_every"] = 20
        smoke_cfg["data"]["n_eval"] = 128
        kind, seed, tr_a, tr_b = _train_pair("jepa", 0, smoke_cfg, world)
        met = alignment_metrics(
            world, tr_a.encoder, tr_b.encoder, n_eval=128, eval_seed=int(cfg["data"]["eval_stream_seed"])
        )
        smoke = {
            "kind": kind,
            "seed": seed,
            "steps": 40,
            "metrics": met,
            "n_params_a": tr_a.n_params,
            "n_params_b": tr_b.n_params,
            "train_wall_s": tr_a.train_wall_s + tr_b.train_wall_s,
            "note": "Disclosed smoke only; thresholds unchanged.",
        }
        (EXP / "smoke.json").write_text(json.dumps(smoke, indent=2))
        print("smoke done:", json.dumps(smoke, indent=2))
        return

    kinds = ["jepa", "recon", "random"]
    jobs = [(k, s) for s in seeds for k in kinds]
    results: dict[str, dict] = {str(s): {"kernels": {}} for s in seeds}

    # Sequential is fine on CPU for this tiny MLP; keep mild parallelism on pairs
    max_workers = 2
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_train_pair, k, s, cfg, world): (k, s) for k, s in jobs}
        for fut in as_completed(futs):
            kind, seed, tr_a, tr_b = fut.result()
            ckpt = {
                "kind": kind,
                "seed": seed,
                "state_dict_a": tr_a.encoder.state_dict(),
                "state_dict_b": tr_b.encoder.state_dict(),
                "model_cfg": {
                    "arch_a": cfg["model"]["arch_a"],
                    "arch_b": cfg["model"]["arch_b"],
                    "emb_dim": cfg["model"]["emb_dim"],
                    "d_mod_a": world.d_mod_a,
                    "d_mod_b": world.d_mod_b,
                    "dropout": cfg["model"].get("dropout", 0.0),
                },
                "final_loss_a": tr_a.final_loss,
                "final_loss_b": tr_b.final_loss,
                "n_params_a": tr_a.n_params,
                "n_params_b": tr_b.n_params,
                "train_wall_s": tr_a.train_wall_s + tr_b.train_wall_s,
                "history_a": tr_a.history,
                "history_b": tr_b.history,
            }
            torch.save(ckpt, WORK / f"model_{kind}_seed{seed}.pt")
            met = alignment_metrics(
                world,
                tr_a.encoder,
                tr_b.encoder,
                n_eval=int(cfg["data"]["n_eval"]),
                eval_seed=int(cfg["data"]["eval_stream_seed"]) + seed,
            )
            packed = _pack(tr_a, tr_b, met)
            results[str(seed)]["kernels"][kind] = packed
            print(
                f"[eval] {kind} seed={seed} cka={met['procrustes_cka']:.4f} "
                f"knn={met['mutual_knn_overlap']:.4f}",
                flush=True,
            )

    raw: dict = {
        "exp_id": cfg["exp_id"],
        "world": {
            "n_states": world.n_states,
            "n_classes": world.n_classes,
            "d_latent": world.d_latent,
            "d_mod_a": world.d_mod_a,
            "d_mod_b": world.d_mod_b,
        },
        "seeds": results,
        "breakpoints": {},
        "wall_s_total": None,
        "config_snapshot": {
            "steps": cfg["training"]["steps"],
            "emb_dim": cfg["model"]["emb_dim"],
            "arch_a": cfg["model"]["arch_a"],
            "arch_b": cfg["model"]["arch_b"],
            "n_eval": cfg["data"]["n_eval"],
            "nuisance_std": cfg["data"]["nuisance_std"],
        },
    }

    if not args.skip_breakpoints:
        bp_seed = int(cfg["breakpoints"]["seed"])
        bps: dict = {}

        # Bad views
        print("[breakpoint] bad_views", flush=True)
        _, _, tr_a, tr_b = _train_pair(
            "jepa",
            bp_seed,
            cfg,
            world,
            view_mode="bad",
            steps_override=int(cfg["breakpoints"]["bad_views"].get("steps", cfg["training"]["steps"])),
        )
        met_bad = alignment_metrics(
            world,
            tr_a.encoder,
            tr_b.encoder,
            n_eval=int(cfg["data"]["n_eval"]),
            eval_seed=int(cfg["data"]["eval_stream_seed"]) + bp_seed,
        )
        bps["bad_views"] = {
            "procrustes_cka": met_bad["procrustes_cka"],
            "mutual_knn_overlap": met_bad["mutual_knn_overlap"],
            "procrustes_rmse": met_bad["procrustes_rmse"],
            "noise_std": cfg["breakpoints"]["bad_views"]["noise_std"],
            "dropout_p": cfg["breakpoints"]["bad_views"]["dropout_p"],
            "steps": cfg["breakpoints"]["bad_views"].get("steps", cfg["training"]["steps"]),
        }

        # Tiny / large emb dims
        print("[breakpoint] tiny_emb_dim", flush=True)
        emb_rows = []
        for ed in cfg["breakpoints"]["tiny_emb_dim"]["emb_dims"]:
            _, _, tr_a, tr_b = _train_pair(
                "jepa",
                bp_seed,
                cfg,
                world,
                emb_dim_override=int(ed),
                steps_override=int(cfg["breakpoints"]["tiny_emb_dim"]["steps"]),
            )
            met = alignment_metrics(
                world,
                tr_a.encoder,
                tr_b.encoder,
                n_eval=int(cfg["data"]["n_eval"]),
                eval_seed=int(cfg["data"]["eval_stream_seed"]) + bp_seed,
            )
            emb_rows.append(
                {
                    "emb_dim": int(ed),
                    "steps": int(cfg["breakpoints"]["tiny_emb_dim"]["steps"]),
                    "procrustes_cka": met["procrustes_cka"],
                    "mutual_knn_overlap": met["mutual_knn_overlap"],
                }
            )
            print(f"  emb={ed} cka={met['procrustes_cka']:.4f}", flush=True)
        bps["tiny_emb_dim"] = emb_rows

        # Short training
        print("[breakpoint] short_training", flush=True)
        step_rows = []
        for st in cfg["breakpoints"]["short_training"]["steps"]:
            _, _, tr_a, tr_b = _train_pair(
                "jepa",
                bp_seed,
                cfg,
                world,
                steps_override=int(st),
            )
            met = alignment_metrics(
                world,
                tr_a.encoder,
                tr_b.encoder,
                n_eval=int(cfg["data"]["n_eval"]),
                eval_seed=int(cfg["data"]["eval_stream_seed"]) + bp_seed,
            )
            step_rows.append(
                {
                    "steps": int(st),
                    "procrustes_cka": met["procrustes_cka"],
                    "mutual_knn_overlap": met["mutual_knn_overlap"],
                }
            )
            print(f"  steps={st} cka={met['procrustes_cka']:.4f}", flush=True)
        bps["short_training"] = step_rows

        raw["breakpoints"] = bps

    raw["wall_s_total"] = time.perf_counter() - t_wall0
    (EXP / "raw.json").write_text(json.dumps(raw, indent=2))
    print(f"wrote {EXP / 'raw.json'} wall_s={raw['wall_s_total']:.1f}", flush=True)


if __name__ == "__main__":
    main()
