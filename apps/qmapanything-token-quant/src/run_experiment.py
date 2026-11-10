"""Train the toy, fake-quantize it, and write raw.json.

Usage:
    python src/run_experiment.py
    python src/run_experiment.py --smoke
    python src/run_experiment.py --skip-breakpoints
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from breakpoints import block_latency_vs_tokens, layer_inputs, matched_norm_probe, patch_count_probe
from model import TinyGeoTransformer, sample_batch, task_loss
from quant import (
    LayerPack,
    RecordCtx,
    RunCtx,
    build_layer_pack,
    held_out_special_mse_drop,
    qmax_for_bits,
    random_signs,
    rotate_activations,
    snr_proxy_ratio,
    walsh_hadamard,
)

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_token_quant"
METHODS = ("naive", "rotation", "uniform", "qmap")


def _load() -> dict[str, Any]:
    return yaml.safe_load((EXP / "config.yaml").read_text())


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _batch(cfg: dict[str, Any], n: int, seed: int, device: torch.device):
    return sample_batch(cfg, n, _generator(seed), device)


def train_one(cfg: dict[str, Any], seed: int, steps: int, device: torch.device) -> tuple[TinyGeoTransformer, float, float]:
    torch.manual_seed(seed)
    model = TinyGeoTransformer(cfg).to(device)
    train_cfg = cfg["training"]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    model.train()
    start = time.perf_counter()
    last = 0.0
    batch_n = int(train_cfg["batch_size"])
    for step in range(steps):
        patches, scale, outliers, points, m = _batch(cfg, batch_n, 30_000 + seed * 100_000 + step, device)
        opt.zero_grad(set_to_none=True)
        out = model(patches, scale, outliers, None)
        loss = task_loss(out, points, m, model.range_loss_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
        opt.step()
        last = float(loss.detach().item())
    elapsed = time.perf_counter() - start
    model.eval()
    return model, elapsed, last


def _weights(model: TinyGeoTransformer) -> dict[str, tuple[torch.Tensor, bool]]:
    found: dict[str, tuple[torch.Tensor, bool]] = {}
    for index, block in enumerate(model.blocks):
        special = block.kind == "global"
        for part, layer in (("qkv", block.qkv), ("proj", block.proj), ("fc1", block.fc1), ("fc2", block.fc2)):
            found[f"b{index}.{part}"] = (layer.weight, special)
    return found


def _record(model: TinyGeoTransformer, cfg: dict[str, Any], n: int, seed: int, device: torch.device) -> tuple[RecordCtx, tuple]:
    batch = _batch(cfg, n, seed, device)
    ctx = RecordCtx()
    with torch.no_grad():
        model(batch[0], batch[1], batch[2], ctx)
    return ctx, batch


def calibrate(
    model: TinyGeoTransformer,
    recorded: RecordCtx,
    *,
    seed: int,
    weight_bits: int,
    act_bits: int,
) -> dict[str, LayerPack]:
    weights = _weights(model)
    sign_g = _generator(seed + 17)
    hadamards: dict[int, torch.Tensor] = {}
    packs: dict[str, LayerPack] = {}
    for name, pieces in recorded.inputs.items():
        weight, special = weights[name]
        inputs = torch.cat(pieces, dim=0)
        width = weight.shape[1]
        if width not in hadamards:
            hadamards[width] = walsh_hadamard(width, device=weight.device, dtype=weight.dtype)
        signs = random_signs(width, sign_g, device=weight.device, dtype=weight.dtype)
        packs[name] = build_layer_pack(
            name,
            weight,
            inputs,
            has_special=special and recorded.has_special[name],
            signs=signs,
            hadamard=hadamards[width],
            weight_bits=weight_bits,
            act_bits=act_bits,
        )
    return packs


def _mean_rel(pred: torch.Tensor, ref: torch.Tensor) -> float:
    num = (pred - ref).norm(dim=-1)
    den = ref.norm(dim=-1).clamp_min(1e-8)
    return float((num / den).mean().item())


def _score(out_q, out_fp, m_true: torch.Tensor, points: torch.Tensor) -> dict[str, float]:
    special_rel = _mean_rel(out_q.h[:, -1], out_fp.h[:, -1])
    patch_rel = _mean_rel(out_q.h[:, :-1], out_fp.h[:, :-1])
    special_mse = float((out_q.h[:, -1] - out_fp.h[:, -1]).pow(2).mean().item())
    metric_rel = float(((out_q.m_hat - m_true).abs() / m_true.abs().clamp_min(1e-8)).mean().item())
    metric_vs_fp = float(((out_q.m_hat - out_fp.m_hat).abs() / out_fp.m_hat.abs().clamp_min(1e-8)).mean().item())
    upscale = float(
        ((out_q.points_hat - points).flatten(1).norm(dim=-1) / points.flatten(1).norm(dim=-1).clamp_min(1e-8)).mean().item()
    )
    metric_pts = out_q.m_hat[:, None, None] * out_q.points_hat
    truth_pts = m_true[:, None, None] * points
    metric_pts_rel = float(
        ((metric_pts - truth_pts).flatten(1).norm(dim=-1) / truth_pts.flatten(1).norm(dim=-1).clamp_min(1e-8)).mean().item()
    )
    return {
        "special_rel": special_rel,
        "patch_rel": patch_rel,
        "special_mse": special_mse,
        "metric_rel": metric_rel,
        "metric_rel_vs_fp": metric_vs_fp,
        "upscale_rel": upscale,
        "metric_points_rel": metric_pts_rel,
    }


def _forward(model, batch, ctx):
    with torch.no_grad():
        return model(batch[0], batch[1], batch[2], ctx)


def _bench(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    rounds = 8
    chunk = max(repeats // rounds, 1)
    samples: list[float] = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(chunk):
            fn()
        samples.append((time.perf_counter() - start) / chunk)
    samples.sort()
    return samples[len(samples) // 2]


def _prop1(recorded: RecordCtx, packs: dict[str, LayerPack]) -> dict[str, float]:
    changes: list[float] = []
    shrinks: list[float] = []
    norm_err: list[float] = []
    heavy: list[float] = []
    for name, pack in packs.items():
        if not pack.has_special:
            continue
        tokens = torch.cat(recorded.inputs[name], dim=0)
        rotated = rotate_activations(tokens, pack.signs, pack.hadamard)
        pre = snr_proxy_ratio(tokens, pack.delta_naive)
        post = snr_proxy_ratio(rotated, pack.delta_rot)
        changes.append(abs(post / max(pre, 1e-12) - 1.0))
        ratio = pack.delta_naive / max(pack.delta_rot, 1e-12)
        shrinks.append(ratio)
        base = tokens.norm(dim=-1).clamp_min(1e-8)
        norm_err.append(float(((rotated.norm(dim=-1) - base).abs() / base).max().item()))
        if name.endswith(".qkv") or name.endswith(".fc1"):
            heavy.append(ratio)
    return {
        "snr_ratio_rel_change": float(sum(changes) / max(len(changes), 1)),
        "delta_shrink_all": float(sum(shrinks) / max(len(shrinks), 1)),
        "delta_shrink_qkv_fc1": float(sum(heavy) / max(len(heavy), 1)),
        "max_norm_rel_error": float(max(norm_err) if norm_err else 0.0),
    }


def _audit(model: TinyGeoTransformer, cfg: dict[str, Any], seed: int, device: torch.device) -> dict[str, float]:
    batch = _batch(cfg, 64, 60_000 + seed, device)
    model.eval()
    out = model(batch[0], batch[1], batch[2], None)
    loss = (out.m_hat - batch[4]).square().mean()
    grad = torch.autograd.grad(loss, out.h)[0]
    lam_s = grad[:, -1].norm(dim=-1).mean()
    lam_p = grad[:, :-1].norm(dim=-1).mean()
    ctx = RecordCtx()
    with torch.no_grad():
        model(batch[0], batch[1], batch[2], ctx)
    invs: list[torch.Tensor] = []
    tails: list[float] = []
    for name, pieces in ctx.inputs.items():
        if not (name.endswith(".qkv") and ctx.has_special[name]):
            continue
        tokens = pieces[0]
        norms = tokens.norm(dim=-1)
        invs.append((norms[:, :-1].amax(dim=-1) / norms[:, -1].clamp_min(1e-8)).mean())
        flat = tokens.abs().reshape(-1)
        p99 = torch.quantile(flat, 0.99).clamp_min(1e-12)
        tails.append(float((flat.max() / p99).item()))
    inv = torch.stack(invs).mean()
    gamma = (lam_s / lam_p.clamp_min(1e-8)) * torch.clamp(inv - 1.0, min=0.0)
    h = out.h.detach()
    res_inv = (h[:, :-1].norm(dim=-1).amax(dim=-1) / h[:, -1].norm(dim=-1).clamp_min(1e-8)).mean()
    frame_deltas = [d for d, block in zip(out.scale_deltas, model.blocks) if block.kind == "frame"]
    global_deltas = [d for d, block in zip(out.scale_deltas, model.blocks) if block.kind == "global"]
    return {
        "lambda_s": float(lam_s.item()),
        "lambda_patch": float(lam_p.item()),
        "inv_kappa_qkv": float(inv.item()),
        "inv_kappa_residual": float(res_inv.item()),
        "gamma": float(gamma.item()),
        "qkv_max_over_p99": float(sum(tails) / len(tails)),
        "frame_scale_delta": float(sum(frame_deltas) / max(len(frame_deltas), 1)),
        "global_scale_delta": float(sum(global_deltas) / max(len(global_deltas), 1)),
    }


def _eta(recorded: RecordCtx, packs: dict[str, LayerPack], act_bits: int) -> float:
    drops: list[float] = []
    qmax = qmax_for_bits(act_bits)
    for name, pack in packs.items():
        if not pack.has_special:
            continue
        drop = held_out_special_mse_drop(pack, torch.cat(recorded.inputs[name], dim=0), qmax)
        if drop == drop:
            drops.append(drop)
    if not drops:
        return float("nan")
    return float(sum(drops) / len(drops))


def _eval_bits(
    model: TinyGeoTransformer,
    cfg: dict[str, Any],
    recorded: RecordCtx,
    test_batch,
    fp_out,
    seed: int,
    weight_bits: int,
    act_bits: int,
    *,
    time_it: bool,
) -> dict[str, Any]:
    packs = calibrate(model, recorded, seed=seed, weight_bits=weight_bits, act_bits=act_bits)
    methods: dict[str, Any] = {}
    for method in METHODS:
        ctx = RunCtx(packs, method, qmax_for_bits(act_bits))
        pred = _forward(model, test_batch, ctx)
        row = _score(pred, fp_out, test_batch[4], test_batch[3])
        if time_it:
            def _call(method=method, ctx_packs=packs) -> None:
                _forward(model, test_batch, RunCtx(ctx_packs, method, qmax_for_bits(act_bits)))

            row["latency_s"] = _bench(
                _call,
                int(cfg["runtime"]["speed_warmup"]) if not time_it else int(cfg["runtime"]["speed_warmup"]),
                int(cfg["runtime"]["speed_repeats"]),
            )
        methods[method] = row
    tag = f"W{weight_bits}A{act_bits}"
    return {
        "tag": tag,
        "methods": methods,
        "eta_heldout_special_mse_drop": _eta(recorded, packs, act_bits) if (weight_bits, act_bits) == (4, 4) else None,
        "packs": packs,
    }


def _media(test_batch, fp_out, preds: dict[str, Any], n_scenes: int = 6) -> dict[str, Any]:
    def _list(t: torch.Tensor) -> list:
        return t.detach().cpu().float().numpy().round(6).tolist()

    n = min(n_scenes, test_batch[4].shape[0])
    payload: dict[str, Any] = {
        "points": _list(test_batch[3][:n]),
        "m": _list(test_batch[4][:n]),
        "fp": {"points": _list(fp_out.points_hat[:n]), "m": _list(fp_out.m_hat[:n])},
        "methods": {},
    }
    for method, pred in preds.items():
        payload["methods"][method] = {"points": _list(pred.points_hat[:n]), "m": _list(pred.m_hat[:n])}
    return payload


def run_seed(cfg: dict[str, Any], seed: int, steps: int, device: torch.device, *, time_it: bool, keep_media: bool) -> dict[str, Any]:
    model, train_s, train_loss = train_one(cfg, seed, steps, device)
    n_calib = int(cfg["data"]["n_calib"])
    n_test = int(cfg["data"]["n_test"])
    recorded, _ = _record(model, cfg, n_calib, 20_000 + seed, device)
    test_batch = _batch(cfg, n_test, 10_000 + seed, device)
    fp_out = _forward(model, test_batch, None)
    fp_metric = float(((fp_out.m_hat - test_batch[4]).abs() / test_batch[4].abs().clamp_min(1e-8)).mean().item())
    fp_points = float(
        ((fp_out.points_hat - test_batch[3]).flatten(1).norm(dim=-1) / test_batch[3].flatten(1).norm(dim=-1).clamp_min(1e-8)).mean().item()
    )
    audit = _audit(model, cfg, seed, device)
    # Prop. 1 uses any bit-width's deltas; the ratio is invariant to qmax.
    # Build W8 packs only to read deltas, then the gated loop rebuilds them.
    ref_packs = calibrate(model, recorded, seed=seed, weight_bits=8, act_bits=8)
    prop1 = _prop1(recorded, ref_packs)
    bits_out: dict[str, Any] = {}
    media = None
    w4_packs = None
    for pair in list(cfg["bitwidths"]["gated"]) + list(cfg["bitwidths"]["report"]):
        w_bits, a_bits = int(pair[0]), int(pair[1])
        result = _eval_bits(
            model, cfg, recorded, test_batch, fp_out, seed, w_bits, a_bits, time_it=time_it,
        )
        if (w_bits, a_bits) == (4, 4):
            w4_packs = result["packs"]
            if keep_media:
                preds = {}
                for method in METHODS:
                    preds[method] = _forward(model, test_batch, RunCtx(result["packs"], method, qmax_for_bits(4)))
                media = _media(test_batch, fp_out, preds)
        bits_out[result["tag"]] = {
            "methods": result["methods"],
            "eta_heldout_special_mse_drop": result["eta_heldout_special_mse_drop"],
        }
    n_params = sum(p.numel() for p in model.parameters())
    payload = {
        "seed": seed,
        "train_s": train_s,
        "train_loss": train_loss,
        "n_params": n_params,
        "fp": {"metric_rel": fp_metric, "upscale_rel": fp_points, **audit, **prop1},
        "bits": bits_out,
    }
    return {
        "payload": payload,
        "model": model,
        "recorded": recorded,
        "w4_packs": w4_packs,
        "media": media,
    }


def _lower_bits(model, cfg, seed, device) -> list[dict[str, float]]:
    recorded, _ = _record(model, cfg, int(cfg["data"]["n_calib"]), 20_000 + seed, device)
    test_batch = _batch(cfg, int(cfg["data"]["n_test"]), 10_000 + seed, device)
    fp_out = _forward(model, test_batch, None)
    rows: list[dict[str, float]] = []
    for pair in cfg["breakpoints"]["lower_bits"]:
        result = _eval_bits(
            model,
            cfg,
            recorded,
            test_batch,
            fp_out,
            seed,
            int(pair[0]),
            int(pair[1]),
            time_it=False,
        )
        for method, scores in result["methods"].items():
            if method == "uniform":
                continue
            rows.append(
                {
                    "weight_bits": float(pair[0]),
                    "act_bits": float(pair[1]),
                    "method": method,
                    "special_rel": scores["special_rel"],
                    "patch_rel": scores["patch_rel"],
                    "metric_rel": scores["metric_rel"],
                    "special_mse": scores["special_mse"],
                }
            )
    return rows


def run(smoke: bool, skip_breakpoints: bool) -> None:
    cfg = _load()
    device = torch.device("cpu")
    threads = int(cfg["runtime"]["torch_threads"])
    torch.set_num_threads(threads)
    steps = 40 if smoke else int(cfg["training"]["steps"])
    if smoke:
        cfg = json.loads(json.dumps(cfg))
        cfg["data"]["n_calib"] = 64
        cfg["data"]["n_test"] = 32
        cfg["runtime"]["speed_warmup"] = 2
        cfg["runtime"]["speed_repeats"] = 8
        cfg["training"]["batch_size"] = 64
    seeds = [0] if smoke else [int(s) for s in cfg["seeds"]]
    started = time.perf_counter()
    seed_rows: dict[str, Any] = {}
    kept = None
    for seed in seeds:
        print(f"seed {seed} training {steps} steps", flush=True)
        result = run_seed(cfg, seed, steps, device, time_it=True, keep_media=(seed == 0 and not smoke))
        seed_rows[str(seed)] = result["payload"]
        fp = result["payload"]["fp"]
        w4 = result["payload"]["bits"]["W4A4"]["methods"]
        print(
            f"  fp_metric={fp['metric_rel']:.4f} inv_kappa_qkv={fp['inv_kappa_qkv']:.2f} "
            f"shrink={fp['delta_shrink_qkv_fc1']:.2f} "
            f"W4 special naive/rot/qmap="
            f"{w4['naive']['special_rel']:.3f}/{w4['rotation']['special_rel']:.3f}/{w4['qmap']['special_rel']:.3f} "
            f"metric={w4['naive']['metric_rel']:.3f}/{w4['rotation']['metric_rel']:.3f}/{w4['qmap']['metric_rel']:.3f}",
            flush=True,
        )
        if seed == 0:
            kept = result
    raw: dict[str, Any] = {
        "exp_id": cfg["exp_id"],
        "smoke": smoke,
        "seeds": seed_rows,
        "wall_s": time.perf_counter() - started,
        "torch": torch.__version__,
        "threads": threads,
    }
    if kept is not None and kept["media"] is not None:
        raw["media"] = kept["media"]
    if not smoke and not skip_breakpoints and kept is not None:
        print("breakpoints", flush=True)
        bp = cfg["breakpoints"]
        layer_name = str(bp["patch_probe_layer"])
        inputs = layer_inputs(kept["recorded"].inputs, layer_name)
        weight = _weights(kept["model"])[layer_name][0]
        global_index = next(i for i, block in enumerate(kept["model"].blocks) if block.kind == "global")
        raw["breakpoints"] = {
            "lower_bits": _lower_bits(kept["model"], cfg, int(bp["seed"]), device),
            "patch_counts": patch_count_probe(
                inputs,
                counts=[int(c) for c in bp["patch_counts"]],
                trials=int(bp["patch_probe_trials"]),
                seed=int(bp["seed"]),
                act_bits=4,
            ),
            "matched_norm": matched_norm_probe(
                inputs,
                weight,
                seed=int(bp["seed"]),
                weight_bits=4,
                act_bits=4,
            ),
            "speed_tokens": block_latency_vs_tokens(
                kept["model"].blocks[global_index],
                kept["w4_packs"],
                block_index=global_index,
                width=int(cfg["model"]["d"]),
                tokens=[int(t) for t in bp["speed_tokens"]],
                batch=int(bp["speed_batch"]),
                repeats=int(bp["speed_repeats"]),
                outlier_gain=float(cfg["model"]["outlier_gain"]),
            ),
        }
    out_name = "smoke.json" if smoke else "raw.json"
    (EXP / out_name).write_text(json.dumps(raw, indent=2))
    print(f"wrote {EXP / out_name} in {raw['wall_s']:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-breakpoints", action="store_true")
    args = parser.parse_args()
    run(smoke=args.smoke, skip_breakpoints=args.skip_breakpoints)


if __name__ == "__main__":
    main()
