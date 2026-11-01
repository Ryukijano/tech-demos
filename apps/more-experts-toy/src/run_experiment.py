"""Run exp1_cpu_charlm: MoRE vs MoE vs dense vs MoRE-no-depth on char LM.

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

from more_model import (
    adjacent_reuse_stats,
    evaluate_ce,
    load_char_data,
    set_threads,
    train_model,
)

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_charlm"
WORK = APP / "work"


def _model_kwargs(cfg: dict, overrides: dict | None = None) -> dict:
    m = dict(cfg["model"])
    t = cfg["training"]
    d = cfg["data"]
    if overrides:
        m.update(overrides)
    return {
        "steps": int(t["steps"]),
        "batch_size": int(t["batch_size"]),
        "lr": float(t["lr"]),
        "seq_len": int(d["seq_len"]),
        "d_model": int(m["d_model"]),
        "n_layers": int(m["n_layers"]),
        "n_heads": int(m["n_heads"]),
        "dropout": float(m["dropout"]),
        "n_experts": int(m["n_experts"]),
        "top_k": int(m["top_k"]),
        "reuse_factor": int(m["reuse_factor"]),
        "d_ff_expert": int(m["d_ff_expert"]),
        "d_ff_dense": int(m["d_ff_dense"]),
        "aux_loss_coef": float(m["aux_loss_coef"]),
        "weight_decay": float(t["weight_decay"]),
        "grad_clip": float(t["grad_clip"]),
        "log_every": int(t["log_every"]),
        "eval_every": int(t["eval_every"]),
        "eval_batches": int(d["eval_batches"]),
    }


def _train_one(kind: str, seed: int, cfg: dict, data, overrides: dict | None = None, steps_override: int | None = None):
    kw = _model_kwargs(cfg, overrides)
    if steps_override is not None:
        kw["steps"] = int(steps_override)
        kw["eval_every"] = min(kw["eval_every"], max(50, kw["steps"] // 4))
        kw["log_every"] = min(kw["log_every"], kw["eval_every"])
    print(
        f"[train] {kind} seed={seed} steps={kw['steps']} "
        f"N={kw['n_experts']} K={kw['top_k']} R={kw['reuse_factor']}",
        flush=True,
    )
    tr = train_model(kind, seed, data, **kw)
    return kind, seed, tr, overrides


def _pack_metrics(tr, cfg, data, seed: int) -> dict:
    d = cfg["data"]
    met = evaluate_ce(
        tr.model,
        data.val_ids,
        seq_len=int(d["seq_len"]),
        n_batches=int(d["eval_batches"]),
        batch_size=int(cfg["training"]["batch_size"]),
        eval_seed=int(820_000 + seed),
    )
    route = adjacent_reuse_stats(
        tr.model,
        data.val_ids,
        seq_len=int(d["seq_len"]),
        n_batches=8,
        batch_size=int(cfg["training"]["batch_size"]),
        eval_seed=int(830_000 + seed),
    )
    return {
        "val_ce": met["val_ce"],
        "bits_per_char": met["bits_per_char"],
        "train_wall_s": tr.train_wall_s,
        "final_train_ce": tr.final_loss,
        "n_params": tr.n_params,
        "n_active_params_per_token": tr.model.n_active_params_per_token(),
        "history": tr.history,
        "routing": route,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-breakpoints", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((EXP / "config.yaml").read_text())
    set_threads(int(cfg["runtime"]["torch_threads"]))
    WORK.mkdir(exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)

    corpus = APP / cfg["data"]["corpus_path"]
    data = load_char_data(corpus, train_frac=float(cfg["data"]["train_frac"]))
    print(
        f"corpus chars={len(data.text)} vocab={data.vocab_size} "
        f"train={len(data.train_ids)} val={len(data.val_ids)}",
        flush=True,
    )

    seeds = list(cfg["seeds"])
    kernels = list(cfg["kernels"])
    t_wall0 = time.perf_counter()

    if args.smoke:
        smoke_cfg = json.loads(json.dumps(cfg))
        smoke_cfg["training"]["steps"] = 40
        smoke_cfg["training"]["batch_size"] = 16
        smoke_cfg["training"]["log_every"] = 20
        smoke_cfg["training"]["eval_every"] = 20
        smoke_cfg["data"]["eval_batches"] = 4
        kind, seed, tr, _ = _train_one("more", 0, smoke_cfg, data)
        metrics = _pack_metrics(tr, smoke_cfg, data, seed)
        smoke = {
            "kind": kind,
            "seed": seed,
            "steps": 40,
            "final_train_ce": tr.final_loss,
            "n_params": tr.n_params,
            "train_wall_s": tr.train_wall_s,
            "metrics": {
                k: metrics[k]
                for k in (
                    "val_ce",
                    "bits_per_char",
                    "n_params",
                    "n_active_params_per_token",
                    "routing",
                )
            },
            "note": "Disclosed smoke only; thresholds unchanged.",
        }
        (EXP / "smoke.json").write_text(json.dumps(smoke, indent=2))
        print("smoke done:", json.dumps(smoke, indent=2))
        return

    jobs = [(k, s) for s in seeds for k in kernels]
    results: dict[str, dict] = {str(s): {"kernels": {}} for s in seeds}
    param_table: dict[str, int] = {}

    max_workers = 2
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_train_one, k, s, cfg, data): (k, s) for k, s in jobs}
        for fut in as_completed(futs):
            kind, seed, tr, _ = fut.result()
            ckpt_path = WORK / f"model_{kind}_seed{seed}.pt"
            torch.save(
                {
                    "state_dict": tr.model.state_dict(),
                    "model_cfg": {
                        "kind": kind,
                        "vocab_size": data.vocab_size,
                        "d_model": cfg["model"]["d_model"],
                        "n_layers": cfg["model"]["n_layers"],
                        "n_heads": cfg["model"]["n_heads"],
                        "dropout": cfg["model"]["dropout"],
                        "n_experts": cfg["model"]["n_experts"],
                        "top_k": cfg["model"]["top_k"],
                        "reuse_factor": cfg["model"]["reuse_factor"],
                        "d_ff_expert": cfg["model"]["d_ff_expert"],
                        "d_ff_dense": cfg["model"]["d_ff_dense"],
                        "aux_loss_coef": cfg["model"]["aux_loss_coef"],
                    },
                    "kind": kind,
                    "seed": seed,
                    "final_loss": tr.final_loss,
                    "final_val_ce": tr.final_val_ce,
                    "train_wall_s": tr.train_wall_s,
                    "n_params": tr.n_params,
                    "history": tr.history,
                },
                ckpt_path,
            )
            metrics = _pack_metrics(tr, cfg, data, seed)
            # Drop bulky history duplicate in kernels summary (keep in ckpt + raw)
            results[str(seed)]["kernels"][kind] = metrics
            param_table[kind] = tr.n_params
            print(
                f"[eval] {kind} seed={seed} val_ce={metrics['val_ce']:.4f} "
                f"bpc={metrics['bits_per_char']:.4f} params={tr.n_params}",
                flush=True,
            )

    raw: dict = {
        "exp_id": cfg["exp_id"],
        "param_table": param_table,
        "vocab_size": data.vocab_size,
        "corpus_chars": len(data.text),
        "seeds": results,
        "breakpoints": {},
        "wall_s_total": None,
        "config_snapshot": {
            "steps": cfg["training"]["steps"],
            "n_experts": cfg["model"]["n_experts"],
            "top_k": cfg["model"]["top_k"],
            "reuse_factor": cfg["model"]["reuse_factor"],
            "n_layers": cfg["model"]["n_layers"],
            "d_model": cfg["model"]["d_model"],
            "seq_len": cfg["data"]["seq_len"],
        },
    }

    if not args.skip_breakpoints:
        bp_seed = int(cfg["breakpoints"]["seed"])
        bp_steps = max(200, int(cfg["training"]["steps"]) // 2)
        bps: dict = {}

        # Extreme reuse R=n_layers
        r_rows = []
        for r in cfg["breakpoints"]["reuse_factor_extreme"]["R"]:
            overrides = {"reuse_factor": int(r)}
            # Only MoRE variants use R
            for kind in ("more", "more_nodepth"):
                _, _, tr, _ = _train_one(
                    kind, bp_seed, cfg, data, overrides=overrides, steps_override=bp_steps
                )
                met = _pack_metrics(tr, cfg, data, bp_seed)
                r_rows.append(
                    {
                        "kind": kind,
                        "R": int(r),
                        "steps": bp_steps,
                        "val_ce": met["val_ce"],
                        "bits_per_char": met["bits_per_char"],
                        "n_params": met["n_params"],
                        "routing": met["routing"],
                    }
                )
                print(
                    f"[bp] R={r} {kind} val_ce={met['val_ce']:.4f}",
                    flush=True,
                )
        # Also record primary R=2 MoRE from main run for comparison
        primary = results[str(bp_seed)]["kernels"]["more"]
        r_rows.insert(
            0,
            {
                "kind": "more",
                "R": int(cfg["model"]["reuse_factor"]),
                "steps": int(cfg["training"]["steps"]),
                "val_ce": primary["val_ce"],
                "bits_per_char": primary["bits_per_char"],
                "n_params": primary["n_params"],
                "routing": primary["routing"],
                "note": "primary run (full steps)",
            },
        )
        bps["reuse_factor_extreme"] = r_rows

        # Few experts
        few_rows = []
        for n_exp in cfg["breakpoints"]["few_experts"]["n_experts"]:
            overrides = {"n_experts": int(n_exp)}
            for kind in ("moe", "more"):
                _, _, tr, _ = _train_one(
                    kind, bp_seed, cfg, data, overrides=overrides, steps_override=bp_steps
                )
                met = _pack_metrics(tr, cfg, data, bp_seed)
                few_rows.append(
                    {
                        "kind": kind,
                        "n_experts": int(n_exp),
                        "steps": bp_steps,
                        "val_ce": met["val_ce"],
                        "bits_per_char": met["bits_per_char"],
                        "n_params": met["n_params"],
                    }
                )
                print(
                    f"[bp] N={n_exp} {kind} val_ce={met['val_ce']:.4f}",
                    flush=True,
                )
        bps["few_experts"] = few_rows

        # Top-1
        top_rows = []
        for k in cfg["breakpoints"]["top1"]["top_k"]:
            overrides = {"top_k": int(k)}
            for kind in ("moe", "more"):
                _, _, tr, _ = _train_one(
                    kind, bp_seed, cfg, data, overrides=overrides, steps_override=bp_steps
                )
                met = _pack_metrics(tr, cfg, data, bp_seed)
                top_rows.append(
                    {
                        "kind": kind,
                        "top_k": int(k),
                        "steps": bp_steps,
                        "val_ce": met["val_ce"],
                        "bits_per_char": met["bits_per_char"],
                        "n_params": met["n_params"],
                    }
                )
                print(
                    f"[bp] K={k} {kind} val_ce={met['val_ce']:.4f}",
                    flush=True,
                )
        bps["top1"] = top_rows
        raw["breakpoints"] = bps

    raw["wall_s_total"] = time.perf_counter() - t_wall0
    (EXP / "raw.json").write_text(json.dumps(raw, indent=2))
    print(f"wrote {EXP / 'raw.json'} total_wall_s={raw['wall_s_total']:.1f}")


if __name__ == "__main__":
    main()
