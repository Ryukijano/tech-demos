"""Report-only stresses. They do not enter the PASS rule."""

from __future__ import annotations

import time
from typing import Any

import torch

from model import Block
from quant import (
    RunCtx,
    activation_rel_errors,
    build_layer_pack,
    held_out_special_mse_drop,
    qmax_for_bits,
    random_signs,
    rotate_activations,
    walsh_hadamard,
)


def _rel_pair(tokens: torch.Tensor, qmax: int) -> tuple[float, float, float]:
    """Per-row shared absmax, then special rel, patch rel, and step shrink after rotation."""
    peak = tokens.detach().abs().flatten(1).amax(dim=-1).clamp_min(1e-12)
    delta = peak / qmax
    hat = torch.round(tokens / delta[:, None, None]).clamp(-qmax, qmax) * delta[:, None, None]
    spec = (hat[:, -1] - tokens[:, -1]).norm(dim=-1) / tokens[:, -1].norm(dim=-1).clamp_min(1e-8)
    patch = (hat[:, :-1] - tokens[:, :-1]).norm(dim=-1) / tokens[:, :-1].norm(dim=-1).clamp_min(1e-8)
    return float(spec.mean().item()), float(patch.mean().item()), float(delta.mean().item())


def patch_count_probe(
    inputs: torch.Tensor,
    *,
    counts: list[int],
    trials: int,
    seed: int,
    act_bits: int,
) -> list[dict[str, float]]:
    """Resample real qkv rows. More patches compete for one per-scene absmax."""
    pool = inputs[:, :-1].reshape(-1, inputs.shape[-1])
    specials = inputs[:, -1]
    qmax = qmax_for_bits(act_bits)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(40_000 + seed)
    signs = random_signs(pool.shape[-1], generator, device=pool.device, dtype=pool.dtype)
    hadamard = walsh_hadamard(pool.shape[-1], device=pool.device, dtype=pool.dtype)
    rows: list[dict[str, float]] = []
    for count in counts:
        patch_idx = torch.randint(0, pool.shape[0], (trials, count), generator=generator)
        spec_idx = torch.randint(0, specials.shape[0], (trials,), generator=generator)
        tokens = torch.cat([pool[patch_idx], specials[spec_idx, None, :]], dim=1)
        spec_n, patch_n, delta_n = _rel_pair(tokens, qmax)
        rotated = rotate_activations(tokens, signs, hadamard)
        spec_r, patch_r, delta_r = _rel_pair(rotated, qmax)
        rows.append(
            {
                "n_patches": float(count),
                "special_rel_naive": spec_n,
                "patch_rel_naive": patch_n,
                "special_rel_rotation": spec_r,
                "patch_rel_rotation": patch_r,
                "delta_shrink": delta_n / max(delta_r, 1e-12),
            }
        )
    return rows


def matched_norm_probe(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    *,
    seed: int,
    weight_bits: int,
    act_bits: int,
) -> dict[str, float]:
    """Rescale the special row so its max-abs matches the median patch max-abs.

    This removes the S1 occupancy gap. ``b_s`` is refit. The VGGT row of
    Table 5 is the qualitative control: without a grid deficit the extra
    row term should matter less.
    """
    peak_patch = inputs[:, :-1].abs().amax(dim=-1).median(dim=-1).values
    peak_scale = inputs[:, -1].abs().amax(dim=-1).clamp_min(1e-8)
    matched = inputs.clone()
    matched[:, -1] = inputs[:, -1] * (peak_patch / peak_scale).unsqueeze(-1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(50_000 + seed)
    signs = random_signs(weight.shape[1], generator, device=weight.device, dtype=weight.dtype)
    hadamard = walsh_hadamard(weight.shape[1], device=weight.device, dtype=weight.dtype)
    qmax = qmax_for_bits(act_bits)

    def _summarize(tokens: torch.Tensor) -> dict[str, float]:
        pack = build_layer_pack(
            "probe",
            weight,
            tokens,
            has_special=True,
            signs=signs,
            hadamard=hadamard,
            weight_bits=weight_bits,
            act_bits=act_bits,
        )
        spec_n, patch_n = activation_rel_errors(tokens, pack.delta_naive, qmax)
        rotated = rotate_activations(tokens, signs, hadamard)
        spec_r, patch_r = activation_rel_errors(rotated, pack.delta_rot, qmax)
        norms = tokens.norm(dim=-1)
        inv_kappa = float((norms[:, :-1].amax(dim=-1) / norms[:, -1].clamp_min(1e-8)).mean().item())
        return {
            "inv_kappa_l2": inv_kappa,
            "special_rel_naive": spec_n,
            "patch_rel_naive": patch_n,
            "special_over_patch_naive": spec_n / max(patch_n, 1e-8),
            "special_rel_rotation": spec_r,
            "patch_rel_rotation": patch_r,
            "heldout_mse_drop": held_out_special_mse_drop(pack, tokens, qmax),
            "delta": pack.delta_naive,
            "delta_rot": pack.delta_rot,
        }

    original = _summarize(inputs)
    matched_stats = _summarize(matched)
    return {
        "original_inv_kappa_l2": original["inv_kappa_l2"],
        "matched_inv_kappa_l2": matched_stats["inv_kappa_l2"],
        "original_special_over_patch_naive": original["special_over_patch_naive"],
        "matched_special_over_patch_naive": matched_stats["special_over_patch_naive"],
        "original_heldout_mse_drop": original["heldout_mse_drop"],
        "matched_heldout_mse_drop": matched_stats["heldout_mse_drop"],
        "original_special_rel_rotation": original["special_rel_rotation"],
        "matched_special_rel_rotation": matched_stats["special_rel_rotation"],
    }


def block_latency_vs_tokens(
    block: Block,
    packs: dict[str, Any],
    *,
    block_index: int,
    width: int,
    tokens: list[int],
    batch: int,
    repeats: int,
    outlier_gain: float,
) -> list[dict[str, float]]:
    """Time one global block. The row-selective add should stay a small fraction."""
    device = next(block.parameters()).device
    dtype = next(block.parameters()).dtype
    names = [f"b{block_index}.{part}" for part in ("qkv", "proj", "fc1", "fc2")]
    subset = {name: packs[name] for name in names}
    rows: list[dict[str, float]] = []
    for length in tokens:
        outliers = torch.zeros(batch, length - 1, width, device=device, dtype=dtype)
        outliers[:, :, -2:] = torch.randn(batch, length - 1, 2, device=device, dtype=dtype)
        hidden = torch.randn(batch, length, width, device=device, dtype=dtype)

        def _once(method: str) -> None:
            ctx = RunCtx(subset, method, qmax_for_bits(8))  # type: ignore[arg-type]
            block(hidden, outliers, f"b{block_index}", ctx, outlier_gain, 6.02)

        for _ in range(5):
            _once("rotation")
        samples: dict[str, list[float]] = {"rotation": [], "qmap": []}
        chunk = max(repeats // 5, 1)
        for method in ("rotation", "qmap"):
            for _ in range(5):
                start = time.perf_counter()
                for _ in range(chunk):
                    _once(method)
                samples[method].append((time.perf_counter() - start) / chunk)
        rot = sorted(samples["rotation"])[len(samples["rotation"]) // 2]
        qmap = sorted(samples["qmap"])[len(samples["qmap"]) // 2]
        rows.append({"tokens": float(length), "rotation_s": rot, "qmap_s": qmap, "qmap_over_rotation": qmap / max(rot, 1e-12)})
    return rows


def layer_inputs(recorded: dict[str, list[torch.Tensor]], name: str) -> torch.Tensor:
    return torch.cat(recorded[name], dim=0)
