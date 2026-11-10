"""Symmetric absmax fake-quant and randomized Hadamard rotation.

Faithful to arXiv:2611.07955 Eq. (2), Eq. (8), and Appendix D:
signed absmax, per-output-channel weights, per-tensor activations,
R = D H with signs applied first. No upstream code is copied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Never

import torch
import torch.nn.functional as F

MethodName = Literal["naive", "rotation", "uniform", "qmap"]


def qmax_for_bits(bits: int) -> int:
    if bits < 2:
        raise ValueError(f"bit-width must be >= 2, got {bits}")
    return (1 << (bits - 1)) - 1


def walsh_hadamard(n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard matrix. ``n`` must be a power of two."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard order must be a power of two, got {n}")
    eye = torch.ones((1, 1), device=device, dtype=dtype)
    current = eye
    while current.shape[0] < n:
        current = torch.cat(
            [
                torch.cat([current, current], dim=1),
                torch.cat([current, -current], dim=1),
            ],
            dim=0,
        )
    return current / math.sqrt(current.shape[0])


def random_signs(n: int, generator: torch.Generator, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    bits = torch.randint(0, 2, (n,), generator=generator, device="cpu")
    signs = bits.to(dtype=dtype, device=device) * 2 - 1
    return signs


def rotate_activations(x: torch.Tensor, signs: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    """``x @ diag(signs) @ H``. Signs are applied first, as in Appendix D."""
    return (x * signs) @ hadamard


def fold_weight(weight: torch.Tensor, signs: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    """Fold ``R`` into a PyTorch ``[out, in]`` weight: ``W @ R``."""
    return (weight * signs.unsqueeze(0)) @ hadamard


def quantize(x: torch.Tensor, delta: torch.Tensor | float, qmax: int) -> torch.Tensor:
    """Symmetric round-to-nearest. ``delta`` broadcasts with ``x``."""
    if isinstance(delta, float):
        scale = max(delta, 1e-12)
    else:
        scale = delta.clamp_min(1e-12)
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q * scale


def quantize_weight_per_channel(weight: torch.Tensor, qmax: int) -> torch.Tensor:
    """Per-output-channel absmax. ``weight`` is ``[out, in]``."""
    delta = weight.detach().abs().amax(dim=1).clamp_min(1e-12) / qmax
    return quantize(weight.detach(), delta.unsqueeze(1), qmax)


def per_tensor_delta(x: torch.Tensor, qmax: int) -> float:
    peak = float(x.detach().abs().max().item())
    return max(peak, 1e-12) / qmax


@dataclass
class AffineFit:
    beta: torch.Tensor
    b_s: torch.Tensor | None


def fit_affine(error: torch.Tensor, *, has_special: bool) -> AffineFit:
    """Closed form of Eq. (8).

    ``error`` is ``h_hat - h`` with shape ``[N, T, C]``.
    ``beta = -E_t[e_bar_t]`` and, when a special row is last,
    ``b_s = -e_bar_s - beta``, so the special row is shifted by ``-e_bar_s``.
    """
    e_token = error.mean(dim=0)
    e_bar = e_token.mean(dim=0)
    beta = -e_bar
    b_s = None
    if has_special:
        b_s = -e_token[-1] - beta
    return AffineFit(beta=beta, b_s=b_s)


def rotation_identity_error(x: torch.Tensor, weight: torch.Tensor, signs: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    """Max abs error of the FP-preserving fold. Zero up to numerics."""
    reference = F.linear(x, weight)
    rotated = rotate_activations(x, signs, hadamard)
    folded = fold_weight(weight, signs, hadamard)
    rebuilt = F.linear(rotated, folded)
    return (rebuilt - reference).abs().max()


@dataclass
class LayerPack:
    name: str
    has_special: bool
    signs: torch.Tensor
    hadamard: torch.Tensor
    weight_naive: torch.Tensor
    weight_rot: torch.Tensor
    delta_naive: float
    delta_rot: float
    beta: torch.Tensor
    b_s: torch.Tensor | None
    weight_fp: torch.Tensor


class RecordCtx:
    """FP forward that stores each linear's input."""

    def __init__(self) -> None:
        self.inputs: dict[str, list[torch.Tensor]] = {}
        self.has_special: dict[str, bool] = {}

    def record(self, name: str, x: torch.Tensor, has_special: bool) -> None:
        self.inputs.setdefault(name, []).append(x.detach())
        self.has_special[name] = has_special


class RunCtx:
    def __init__(self, packs: dict[str, LayerPack], method: MethodName, act_qmax: int) -> None:
        self.packs = packs
        self.method: MethodName = method
        self.act_qmax = act_qmax


def apply_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    name: str,
    ctx: RecordCtx | RunCtx | None,
    has_special: bool,
) -> torch.Tensor:
    """FP, recording, or fake-quant linear. ``weight`` is PyTorch ``[out, in]``."""
    if ctx is None or isinstance(ctx, RecordCtx):
        if isinstance(ctx, RecordCtx):
            ctx.record(name, x, has_special)
        return F.linear(x, weight)

    method: MethodName = ctx.method
    pack = ctx.packs[name]
    if method == "naive":
        x_q = quantize(x, pack.delta_naive, ctx.act_qmax)
        return F.linear(x_q, pack.weight_naive)
    if method == "rotation":
        x_q = quantize(rotate_activations(x, pack.signs, pack.hadamard), pack.delta_rot, ctx.act_qmax)
        return F.linear(x_q, pack.weight_rot)
    if method == "uniform" or method == "qmap":
        x_q = quantize(rotate_activations(x, pack.signs, pack.hadamard), pack.delta_rot, ctx.act_qmax)
        y = F.linear(x_q, pack.weight_rot) + pack.beta
        if method == "qmap" and has_special and pack.b_s is not None:
            extra = torch.zeros_like(y)
            extra[:, -1] = pack.b_s
            y = y + extra
        return y
    never: Never = method
    raise AssertionError(never)


def build_layer_pack(
    name: str,
    weight: torch.Tensor,
    inputs: torch.Tensor,
    *,
    has_special: bool,
    signs: torch.Tensor,
    hadamard: torch.Tensor,
    weight_bits: int,
    act_bits: int,
) -> LayerPack:
    """Calibrate one linear on FP inputs ``[N, T, in]``."""
    w_qmax = qmax_for_bits(weight_bits)
    a_qmax = qmax_for_bits(act_bits)
    rotated = rotate_activations(inputs, signs, hadamard)
    delta_naive = per_tensor_delta(inputs, a_qmax)
    delta_rot = per_tensor_delta(rotated, a_qmax)
    weight_naive = quantize_weight_per_channel(weight, w_qmax)
    weight_rot = quantize_weight_per_channel(fold_weight(weight, signs, hadamard), w_qmax)
    y_fp = F.linear(inputs, weight)
    y_q = F.linear(quantize(rotated, delta_rot, a_qmax), weight_rot)
    fit = fit_affine(y_q - y_fp, has_special=has_special)
    return LayerPack(
        name=name,
        has_special=has_special,
        signs=signs,
        hadamard=hadamard,
        weight_naive=weight_naive,
        weight_rot=weight_rot,
        delta_naive=delta_naive,
        delta_rot=delta_rot,
        beta=fit.beta,
        b_s=fit.b_s,
        weight_fp=weight.detach(),
    )


def snr_proxy_ratio(tokens: torch.Tensor, delta: float) -> float:
    """``SNR(special) / mean SNR(patch)`` under the shared-step proxy in Prop. 1.

    ``tokens`` is ``[N, T, D]`` with the special row last. The ratio equals the
    power ratio and does not depend on ``delta`` when ``delta`` is shared; the
    argument is kept so callers use the same definition as the paper.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    power = tokens.detach().pow(2).sum(dim=-1).mean(dim=0)
    width = tokens.shape[-1]
    snr = power / (width * (delta ** 2) / 12.0)
    return float((snr[-1] / snr[:-1].mean()).item())


def activation_rel_errors(tokens: torch.Tensor, delta: float, qmax: int) -> tuple[float, float]:
    """Mean special-token and patch-token relative L2 after per-tensor quant."""
    hat = quantize(tokens, delta, qmax)
    spec = _mean_rel(hat[:, -1], tokens[:, -1])
    patch = _mean_rel(hat[:, :-1], tokens[:, :-1])
    return spec, patch


def _mean_rel(pred: torch.Tensor, ref: torch.Tensor) -> float:
    num = (pred - ref).flatten(0, -2).norm(dim=-1) if pred.ndim >= 2 else (pred - ref).norm(dim=-1)
    den = ref.flatten(0, -2).norm(dim=-1).clamp_min(1e-8) if ref.ndim >= 2 else ref.norm(dim=-1).clamp_min(1e-8)
    # pred is [N, D] or [N, T, D]. flatten(0, -2) collapses leading dims.
    return float((num / den).mean().item())


def held_out_special_mse_drop(pack: LayerPack, inputs: torch.Tensor, act_qmax: int) -> float:
    """Split-half drop in special-row MSE from ``(beta, b_s)`` versus rotation.

    Scales and weights stay the full-set calibration. Only the affine vectors
    are fit on one half and scored on the other, then the halves swap.
    Returns the mean fractional drop, or ``nan`` if the layer has no special row.
    """
    if not pack.has_special or inputs.shape[0] < 4:
        return float("nan")
    half = inputs.shape[0] // 2
    drops: list[float] = []
    folds = ((inputs[:half], inputs[half:]), (inputs[half:], inputs[:half]))
    for fit_x, eval_x in folds:
        y_fp = F.linear(fit_x, pack.weight_fp)
        y_q = F.linear(
            quantize(rotate_activations(fit_x, pack.signs, pack.hadamard), pack.delta_rot, act_qmax),
            pack.weight_rot,
        )
        fit = fit_affine(y_q - y_fp, has_special=True)
        y_fp_e = F.linear(eval_x, pack.weight_fp)
        y_q_e = F.linear(
            quantize(rotate_activations(eval_x, pack.signs, pack.hadamard), pack.delta_rot, act_qmax),
            pack.weight_rot,
        )
        before = (y_q_e[:, -1] - y_fp_e[:, -1]).pow(2).mean()
        corrected = y_q_e[:, -1] + fit.beta + fit.b_s
        after = (corrected - y_fp_e[:, -1]).pow(2).mean()
        denom = float(before.item())
        if denom <= 1e-12:
            continue
        drops.append(1.0 - float(after.item()) / denom)
    if not drops:
        return float("nan")
    return float(sum(drops) / len(drops))
