"""Pre-registered CPU comparison: vanilla vs FBT vs ALF."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from model import Mode, ReadMode, build_model, build_trio, set_threads
from task import VOCAB, trace_rounds
from train import demo_record, evaluate, heldout, make_batches, train_model

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_register"
MODES: tuple[Mode, ...] = ("vanilla", "fbt", "alf")


def load_cfg() -> dict[str, Any]:
    with (EXP / "config.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise RuntimeError("config.yaml did not load as a mapping")
    return cfg


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _train_trio(
    cfg: dict[str, Any],
    seed: int,
    n_layers: int,
    rank: int,
    steps: int,
    n_rounds: int,
    probe: np.ndarray,
    eval_tokens: np.ndarray,
) -> tuple[dict[str, Any], dict[Mode, Any]]:
    model_cfg = cfg["model"]
    data = cfg["data"]
    train_cfg = cfg["training"]
    rng = np.random.default_rng(int(data["train_stream_seed"]) + seed * 10007 + n_layers * 17 + rank)
    batches = make_batches(steps, int(train_cfg["batch_size"]), n_rounds, rng)
    models, counts = build_trio(
        vocab=VOCAB,
        d=int(model_cfg["d_model"]),
        n_layers=n_layers,
        n_heads=int(model_cfg["n_heads"]),
        d_ff=int(model_cfg["d_ff"]),
        rank=rank,
        init_share=float(model_cfg["init_share"]),
        seed=seed,
    )
    out: dict[str, Any] = {"counts": counts, "n_layers": n_layers, "rank": rank, "steps": steps}
    for mode in MODES:
        print(f"seed {seed} {mode} L={n_layers} r={rank} params={models[mode].n_parameters()} d_ff={counts[f'd_ff_{mode}']}", flush=True)
        out[mode] = train_model(
            models[mode],
            batches,
            probe,
            eval_tokens,
            lr=float(train_cfg["lr"]),
            grad_clip=float(train_cfg["grad_clip"]),
            log_every=int(train_cfg["log_every"]),
            branch_fraction=float(train_cfg["branch_fraction"]),
            eval_pass_count=int(train_cfg["eval_passes"]),
            batch_size=int(train_cfg["batch_size"]),
        )
    return out, models


def _score_modes(
    models: dict[Mode, Any],
    tokens: np.ndarray,
    passes: dict[Mode, int],
    batch_size: int,
    read_mode: ReadMode = "full",
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for mode in MODES:
        models[mode].read_mode = read_mode if mode == "alf" else "full"
        scored = evaluate(models[mode], tokens, passes[mode], batch_size)
        scores[mode] = float(scored["answer_nats"])
        models[mode].read_mode = "full"
    return scores


def run_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    smoke = cfg["smoke"]
    seed = int(smoke["seed"])
    steps = int(smoke["steps"])
    n_rounds = int(cfg["data"]["n_rounds"])
    probe = heldout(int(smoke["n_seq"]), n_rounds, int(smoke["seed"]) + 1)
    ev = heldout(int(smoke["n_seq"]), n_rounds, int(smoke["seed"]) + 2)
    print(f"smoke seed {seed} steps {steps}", flush=True)
    out, _models = _train_trio(cfg, seed, int(cfg["model"]["n_layers"]), int(cfg["model"]["rank"]), steps, n_rounds, probe, ev)
    summary = {
        "seed": seed,
        "steps": steps,
        "answer_nats": {mode: out[mode]["final"]["answer_nats"] for mode in MODES},
        "finite": all(np.isfinite(out[mode]["final"]["answer_nats"]) for mode in MODES),
    }
    if not summary["finite"]:
        raise RuntimeError("smoke produced a non-finite answer NLL")
    return summary


def breakpoints(
    cfg: dict[str, Any],
    models: dict[Mode, Any],
    seed: int,
) -> dict[str, Any]:
    bp = cfg["breakpoints"]
    data = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    batch_size = int(train_cfg["batch_size"])
    n_seq = int(bp["n_seq"])
    base_rounds = int(data["n_rounds"])
    passes = {
        "vanilla": 1,
        "fbt": int(train_cfg["eval_passes"]),
        "alf": int(train_cfg["eval_passes"]),
    }
    length_rows = []
    for n_rounds in bp["n_rounds"]:
        tokens = heldout(n_seq, int(n_rounds), int(data["eval_stream_seed"]) + 1000 + int(n_rounds))
        scores = _score_modes(models, tokens, passes, batch_size)
        length_rows.append({"n_rounds": int(n_rounds), "length": 1 + 3 * int(n_rounds), **scores})
        print(f"breakpoint length rounds={n_rounds} {scores}", flush=True)

    pass_rows = []
    tokens = heldout(n_seq, base_rounds, int(data["eval_stream_seed"]) + 2000)
    for n_pass in bp["eval_passes"]:
        use = {"vanilla": 1, "fbt": int(n_pass), "alf": int(n_pass)}
        scores = _score_modes(models, tokens, use, batch_size)
        pass_rows.append({"eval_passes": int(n_pass), **scores})
        print(f"breakpoint passes={n_pass} {scores}", flush=True)

    read_rows = []
    for read_mode in ("full", "nearest", "last"):
        scores = _score_modes(models, tokens, passes, batch_size, read_mode=read_mode)  # type: ignore[arg-type]
        read_rows.append({"read_mode": read_mode, **scores})
        print(f"breakpoint read={read_mode} {scores}", flush=True)

    depth_rows = []
    main_layers = int(model_cfg["n_layers"])
    bp_steps = int(bp["steps"])
    for n_layers in bp["depth_layers"]:
        n_layers = int(n_layers)
        if n_layers == main_layers and bp_steps == int(train_cfg["steps"]):
            continue
        probe = heldout(int(data["probe_size"]), base_rounds, int(data["probe_stream_seed"]) + n_layers)
        ev = heldout(int(data["n_eval"]), base_rounds, int(data["eval_stream_seed"]))
        trained, _extra = _train_trio(cfg, seed, n_layers, int(model_cfg["rank"]), bp_steps, base_rounds, probe, ev)
        depth_rows.append(
            {
                "n_layers": n_layers,
                "steps": bp_steps,
                "answer_nats": {mode: trained[mode]["final"]["answer_nats"] for mode in MODES},
                "n_params": {mode: trained[mode]["n_params"] for mode in MODES},
            }
        )

    rank_rows = []
    for rank in bp["ranks"]:
        rank = int(rank)
        probe = heldout(int(data["probe_size"]), base_rounds, int(data["probe_stream_seed"]) + 50 + rank)
        ev = heldout(int(data["n_eval"]), base_rounds, int(data["eval_stream_seed"]))
        alf = build_model(
            VOCAB,
            int(model_cfg["d_model"]),
            main_layers,
            int(model_cfg["n_heads"]),
            int(model_cfg["d_ff"]),
            "alf",
            rank,
            float(model_cfg["init_share"]),
            seed,
        )
        rng = np.random.default_rng(int(data["train_stream_seed"]) + seed * 10007 + 300 + rank)
        batches = make_batches(bp_steps, batch_size, base_rounds, rng)
        print(f"breakpoint rank={rank} params={alf.n_parameters()}", flush=True)
        trained = train_model(
            alf,
            batches,
            probe,
            ev,
            lr=float(train_cfg["lr"]),
            grad_clip=float(train_cfg["grad_clip"]),
            log_every=int(train_cfg["log_every"]),
            branch_fraction=float(train_cfg["branch_fraction"]),
            eval_pass_count=int(train_cfg["eval_passes"]),
            batch_size=batch_size,
        )
        rank_rows.append(
            {
                "rank": rank,
                "steps": bp_steps,
                "answer_nats": trained["final"]["answer_nats"],
                "n_params": trained["n_params"],
            }
        )
    return {
        "sequence_length": length_rows,
        "eval_passes": pass_rows,
        "read_mode": read_rows,
        "depth": depth_rows,
        "rank": rank_rows,
    }


def main() -> None:
    cfg = load_cfg()
    set_threads(int(cfg["runtime"]["torch_threads"]))
    torch.manual_seed(0)
    t0 = time.perf_counter()
    smoke = run_smoke(cfg)
    (EXP / "smoke.json").write_text(json.dumps(_json_ready(smoke), indent=2) + "\n")
    print("smoke finite", smoke["answer_nats"], flush=True)

    data = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    n_rounds = int(data["n_rounds"])
    raw: dict[str, Any] = {
        "seeds": {},
        "primary_metric": "answer_nats",
        "chance_nats": float(np.log(5.0)),
        "eval_passes": {"vanilla": 1, "fbt": int(train_cfg["eval_passes"]), "alf": int(train_cfg["eval_passes"])},
    }
    seed0_models: dict[Mode, Any] | None = None
    for seed in cfg["seeds"]:
        seed = int(seed)
        probe = heldout(int(data["probe_size"]), n_rounds, int(data["probe_stream_seed"]) + seed)
        ev = heldout(int(data["n_eval"]), n_rounds, int(data["eval_stream_seed"]))
        trained, models = _train_trio(
            cfg,
            seed,
            int(model_cfg["n_layers"]),
            int(model_cfg["rank"]),
            int(train_cfg["steps"]),
            n_rounds,
            probe,
            ev,
        )
        if seed == int(cfg["breakpoints"]["seed"]):
            seed0_models = models
            raw["demo_trace"] = demo_record(ev[0], {mode: trained[mode] for mode in MODES})
            raw["demo_rounds_check"] = len(trace_rounds(ev[0]))
        # Drop per-step demo fields from the stored seed record; curves stay.
        for mode in MODES:
            trained[mode].pop("demo_probs", None)
            trained[mode].pop("demo_pred", None)
        raw["seeds"][str(seed)] = trained
        print(f"seed {seed} done", {mode: trained[mode]["final"]["answer_nats"] for mode in MODES}, flush=True)

    if seed0_models is None:
        raise RuntimeError("breakpoint seed was not trained")
    raw["breakpoints"] = breakpoints(cfg, seed0_models, int(cfg["breakpoints"]["seed"]))
    raw["wall_s"] = time.perf_counter() - t0
    (EXP / "raw.json").write_text(json.dumps(_json_ready(raw), indent=2) + "\n")
    print(f"wrote raw.json in {raw['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
