"""Tiny pre-LN transformer with one metric scale token.

Clean-room stand-in for the structural setting in arXiv:2611.07955 (S1–S4).
Not MapAnything. Heads stay outside the quantized scope, as in Sec. 3.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from quant import RecordCtx, RunCtx, apply_linear

BlockKind = Literal["frame", "global"]


@dataclass
class ForwardOut:
    m_hat: torch.Tensor
    points_hat: torch.Tensor
    h: torch.Tensor
    range_loss: torch.Tensor
    scale_deltas: list[float]


def sample_batch(
    cfg: dict,
    n: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic scenes.

    Returns patches ``[N,P,D]``, scale token ``[N,1,D]``, outlier injection
    ``[N,P,D]``, local points ``[N,P,2]``, metric scale ``[N]``.
    The scale token's RMS is the median patch RMS divided by the Table 5
    target ``1/kappa = 6.02``. ``m`` is stored in the token's direction.
    """
    model = cfg["model"]
    d = int(model["d"])
    p = int(model["n_patches"])
    n_out = int(model["n_outlier_channels"])
    m_lo = float(model["m_lo"])
    m_hi = float(model["m_hi"])
    m = torch.empty(n, device=device).uniform_(m_lo, m_hi, generator=generator)
    points = torch.empty(n, p, 2, device=device).normal_(generator=generator)
    patches = torch.empty(n, p, d, device=device).normal_(generator=generator) * float(model["patch_std"])
    patches[:, :, 0:2] = points

    prob = torch.empty(n, p, n_out, device=device).uniform_(1e-6, 1.0 - 1e-6, generator=generator)
    lap = -float(model["outlier_scale"]) * torch.sign(prob - 0.5) * torch.log1p(-2.0 * (prob - 0.5).abs())
    outliers = torch.zeros(n, p, d, device=device)
    outliers[:, :, -n_out:] = lap
    patches[:, :, -n_out:] = lap

    feat_rms = patches.pow(2).mean(dim=-1).sqrt()
    target = feat_rms.median() / float(model["kappa_inv_target"])
    unit = (m - m_lo) / (m_hi - m_lo)
    c0 = 0.85 * (2.0 * unit - 1.0)
    c1 = torch.sqrt((1.0 - c0.square()).clamp_min(0.0))
    scale = torch.zeros(n, 1, d, device=device)
    scale[:, 0, 0] = c0 * target
    scale[:, 0, 1] = c1 * target
    return patches, scale, outliers, points, m


class Block(nn.Module):
    def __init__(self, d: int, ff: int, n_heads: int, kind: BlockKind) -> None:
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d={d} is not divisible by n_heads={n_heads}")
        self.kind: BlockKind = kind
        self.n_heads = n_heads
        self.d = d
        self.dh = d // n_heads
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, ff, bias=False)
        self.fc2 = nn.Linear(ff, d, bias=False)

    def _mix(
        self,
        h: torch.Tensor,
        outliers: torch.Tensor,
        which: int,
        has_special: bool,
        gain: float,
        kappa_inv: float,
    ) -> torch.Tensor:
        norm = self.ln1 if which == 1 else self.ln2
        if not has_special:
            return norm(h) + gain * outliers
        patches = norm(h[:, :-1]) + gain * outliers
        mixed = torch.cat([patches, h[:, -1:]], dim=1)
        # Table 5 measures 1/kappa on the activations that share a grid.
        # A 4-block toy does not learn MapAnything's gap (a probe saw ~95
        # once outliers were added after LayerNorm), so the special row is
        # rescaled onto max-patch L2 / 6.02. Direction is unchanged.
        return _pin_special_row(mixed, kappa_inv)

    def _attend(self, x: torch.Tensor, name: str, ctx: RecordCtx | RunCtx | None, has_special: bool) -> torch.Tensor:
        qkv = apply_linear(x, self.qkv.weight, f"{name}.qkv", ctx, has_special)
        b, t, _ = qkv.shape
        qkv = qkv.view(b, t, 3, self.n_heads, self.dh)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
        ctx_v = torch.matmul(torch.softmax(scores, dim=-1), v)
        mixed = ctx_v.transpose(1, 2).contiguous().view(b, t, self.d)
        return apply_linear(mixed, self.proj.weight, f"{name}.proj", ctx, has_special)

    def forward(
        self,
        h: torch.Tensor,
        outliers: torch.Tensor,
        name: str,
        ctx: RecordCtx | RunCtx | None,
        outlier_gain: float,
        kappa_inv: float,
    ) -> tuple[torch.Tensor, float]:
        has_special = self.kind == "global"
        if self.kind == "frame":
            patches = h[:, :-1]
            scale = h[:, -1:]
            x = self._mix(patches, outliers, 1, False, outlier_gain, kappa_inv)
            attn = self._attend(x, name, ctx, False)
            patches = patches + attn
            y = self._mix(patches, outliers, 2, False, outlier_gain, kappa_inv)
            hidden = F.gelu(apply_linear(y, self.fc1.weight, f"{name}.fc1", ctx, False))
            patches = patches + apply_linear(hidden, self.fc2.weight, f"{name}.fc2", ctx, False)
            return torch.cat([patches, scale], dim=1), 0.0

        before = h[:, -1]
        x = self._mix(h, outliers, 1, True, outlier_gain, kappa_inv)
        h = h + self._attend(x, name, ctx, True)
        y = self._mix(h, outliers, 2, True, outlier_gain, kappa_inv)
        hidden = F.gelu(apply_linear(y, self.fc1.weight, f"{name}.fc1", ctx, True))
        h = h + apply_linear(hidden, self.fc2.weight, f"{name}.fc2", ctx, True)
        delta = float((h[:, -1] - before).detach().norm(dim=-1).mean().item())
        return h, delta


class TinyGeoTransformer(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        model = cfg["model"]
        d = int(model["d"])
        ff = int(model["ff"])
        n_heads = int(model["n_heads"])
        kinds: list[BlockKind] = []
        for kind in model["block_kinds"]:
            if kind == "frame":
                kinds.append("frame")
            elif kind == "global":
                kinds.append("global")
            else:
                raise ValueError(f"unknown block kind {kind}")
        self.blocks = nn.ModuleList([Block(d, ff, n_heads, kind) for kind in kinds])
        self.point_head = nn.Linear(d, 2)
        self.scale_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.outlier_gain = float(model["outlier_gain"])
        self.kappa_inv_target = float(model["kappa_inv_target"])
        self.range_loss_weight = float(model["range_loss_weight"])

    def forward(
        self,
        patches: torch.Tensor,
        scale: torch.Tensor,
        outliers: torch.Tensor,
        ctx: RecordCtx | RunCtx | None = None,
    ) -> ForwardOut:
        h = torch.cat([patches, scale], dim=1)
        penalties: list[torch.Tensor] = []
        deltas: list[float] = []
        for index, block in enumerate(self.blocks):
            h, delta = block(h, outliers, f"b{index}", ctx, self.outlier_gain, self.kappa_inv_target)
            deltas.append(delta)
            if block.kind == "global":
                penalties.append(_range_penalty(h, self.kappa_inv_target))
        points = self.point_head(h[:, :-1])
        m_hat = self.scale_head(h[:, -1]).squeeze(-1)
        if penalties:
            range_loss = torch.stack(penalties).mean()
        else:
            range_loss = h.new_zeros(())
        return ForwardOut(
            m_hat=m_hat,
            points_hat=points,
            h=h,
            range_loss=range_loss,
            scale_deltas=deltas,
        )


def _pin_special_row(tokens: torch.Tensor, kappa_inv: float) -> torch.Tensor:
    """Set the last row's L2 to max-patch L2 / ``kappa_inv``. Direction is kept."""
    patch_max = tokens[:, :-1].norm(dim=-1).amax(dim=-1).clamp_min(1e-8).detach()
    target = patch_max / kappa_inv
    scale_norm = tokens[:, -1].norm(dim=-1).clamp_min(1e-8)
    pinned = tokens.clone()
    pinned[:, -1] = tokens[:, -1] / scale_norm.unsqueeze(-1) * target.unsqueeze(-1)
    return pinned


def _range_penalty(h: torch.Tensor, kappa_inv: float) -> torch.Tensor:
    patch_rms = torch.sqrt(h[:, :-1].pow(2).mean(dim=-1) + 1e-12).mean()
    scale_rms = torch.sqrt(h[:, -1].pow(2).mean(dim=-1) + 1e-12).mean()
    target = patch_rms.detach() / kappa_inv
    return (scale_rms - target).square() / (target.square() + 1e-12)


def task_loss(out: ForwardOut, points: torch.Tensor, m: torch.Tensor, range_weight: float) -> torch.Tensor:
    point = F.mse_loss(out.points_hat, points)
    scale = F.mse_loss(out.m_hat, m)
    return point + scale + range_weight * out.range_loss
