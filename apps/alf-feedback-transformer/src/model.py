"""Tiny standard-residual transformer with vanilla, FBT, and ALF modes.

Equations follow arXiv:2611.31154 §3 for the standard residual (not AttnRes):

  h = (1 - ρ) x + ρ c Σ_{i≥l} β_{i→l} M_i v_{t-1,i}
  β from a tied query on RMSNorm(M v) / sqrt(d)
  M = I + U D, U and the query initialized at 0, ρ(0) = 0.01
  v_0 = e + W_E RMSNorm(z), W_E initialized at 0
  c = RMS(x) / (RMS(n) + ε)

FBT replaces the token input by the gate product used in the official
repository's reported recipe (unit RMS, no prefix mixin):

  norm(W_s h * sigmoid(W_g norm(e)))

with h the previous position's final-normalized state. No upstream source
is copied.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

Mode = Literal["vanilla", "fbt", "alf"]
ReadMode = Literal["full", "nearest", "last"]

EPS = 1e-6


def _never(value: object) -> None:
    raise AssertionError(f"unhandled variant: {value!r}")


def set_threads(n: int) -> None:
    torch.set_num_threads(max(1, int(n)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def rms_pure(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(0.0))


def rms_normalize(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return x.float() * torch.rsqrt(var + eps)


class RmsNorm(nn.Module):
    def __init__(self, d: int, eps: float = EPS) -> None:
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_normalize(x, self.eps) * self.gain


class CausalAttention(nn.Module):
    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d={d} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.proj(y)


class FeedForward(nn.Module):
    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm_attn = RmsNorm(d)
        self.attn = CausalAttention(d, n_heads)
        self.norm_ff = RmsNorm(d)
        self.ff = FeedForward(d, d_ff)


def sinusoid(length: int, d: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    index = torch.arange(d, device=device, dtype=torch.float32)[None, :]
    angle = pos / (10000.0 ** (2.0 * (index // 2) / d))
    pe = torch.zeros(length, d, device=device)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe


def shift_right(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    out[:, 1:] = x[:, :-1]
    return out


class FeedbackTransformer(nn.Module):
    def __init__(
        self,
        vocab: int,
        d: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        mode: Mode,
        rank: int = 4,
        init_share: float = 0.01,
        eps: float = EPS,
    ) -> None:
        super().__init__()
        if mode not in ("vanilla", "fbt", "alf"):
            _never(mode)
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        if rank < 1:
            raise ValueError("rank must be positive")
        self.mode: Mode = mode
        self.d = d
        self.n_layers = n_layers
        self.n_sub = 2 * n_layers
        self.rank = rank
        self.eps = eps
        self.read_mode: ReadMode = "full"
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, d_ff) for _ in range(n_layers)])
        self.final_norm = RmsNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.query = nn.Parameter(torch.zeros(self.n_sub, d))
        self.logit = nn.Parameter(torch.full((self.n_sub,), math.log(init_share / (1.0 - init_share))))
        self.map_down = nn.Parameter(torch.empty(self.n_sub, d, rank))
        self.map_up = nn.Parameter(torch.zeros(self.n_sub, rank, d))
        self.w_e = nn.Linear(d, d, bias=False)
        self.w_s = nn.Linear(d, d, bias=False)
        self.w_g = nn.Linear(d, d, bias=False)
        self.reset_feedback()
        self.beta_last: torch.Tensor | None = None

    def reset_feedback(self) -> None:
        nn.init.normal_(self.map_down, std=0.02)
        nn.init.zeros_(self.map_up)
        nn.init.zeros_(self.query)
        nn.init.zeros_(self.w_e.weight)
        bound = math.sqrt(3.0 / self.d)
        nn.init.uniform_(self.w_s.weight, -bound, bound)
        nn.init.uniform_(self.w_g.weight, -bound, bound)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_unused(self) -> None:
        used = {id(p) for p in [*self.trunk_parameters(), *self.extra_parameters()]}
        for param in self.parameters():
            param.requires_grad_(id(param) in used)

    def trunk_parameters(self) -> list[nn.Parameter]:
        feedback = {
            id(self.query),
            id(self.logit),
            id(self.map_down),
            id(self.map_up),
            id(self.w_e.weight),
            id(self.w_s.weight),
            id(self.w_g.weight),
        }
        return [p for p in self.parameters() if id(p) not in feedback]

    def extra_parameters(self) -> list[nn.Parameter]:
        if self.mode == "vanilla":
            return []
        if self.mode == "fbt":
            return [self.w_s.weight, self.w_g.weight]
        if self.mode == "alf":
            return [
                self.query,
                self.logit,
                self.map_down,
                self.map_up,
                self.w_e.weight,
            ]
        _never(self.mode)
        return []

    def _map_sources(self, branches: torch.Tensor) -> torch.Tensor:
        # branches [B, T, S, D], M = I + U D, U is map_up, D is map_down.
        hidden = torch.einsum("btsd,sdr->btsr", branches, self.map_down)
        delta = torch.einsum("btsr,srd->btsd", hidden, self.map_up)
        return branches + delta

    def _source_span(self, dest: int) -> tuple[int, int]:
        n_src = self.n_sub
        if self.read_mode == "full":
            return dest, n_src
        if self.read_mode == "nearest":
            return dest, min(dest + 4, n_src)
        if self.read_mode == "last":
            return n_src - 1, n_src
        _never(self.read_mode)
        return dest, n_src

    def _read(self, mapped: torch.Tensor, dest: int) -> tuple[torch.Tensor, torch.Tensor]:
        start, stop = self._source_span(dest)
        window = mapped[:, :, start:stop, :]
        if self.read_mode == "full":
            u = self.query[dest]
            sq = window.float().pow(2).sum(dim=-1)
            scale = torch.rsqrt(sq + self.d * self.eps)
            dots = torch.einsum("btwd,d->btw", window.float(), u.float())
            beta = torch.softmax(dots * scale, dim=-1)
        else:
            width = window.shape[2]
            beta = window.new_full(window.shape[:3], 1.0 / width)
        n = torch.einsum("btw,btwd->btd", beta.float(), window.float())
        return n, beta

    def _mix(self, stream: torch.Tensor, mapped: torch.Tensor, dest: int, valid: torch.Tensor) -> torch.Tensor:
        n, beta = self._read(mapped, dest)
        if self.beta_last is not None and self.read_mode == "full":
            start, _stop = self._source_span(dest)
            acc = self.beta_last
            acc[dest, start : start + beta.shape[-1]] += beta.detach().mean(dim=(0, 1))
            self._beta_hits += 1
        rho = torch.sigmoid(self.logit[dest]) * valid
        ones = torch.ones_like(n)
        n_scale = torch.where(valid.bool(), n, ones)
        c = rms_pure(stream) / (rms_pure(n_scale) + self.eps)
        return (1.0 - rho) * stream.float() + rho * c * n

    def _inject(self, embed: torch.Tensor, prev_stream: torch.Tensor) -> torch.Tensor:
        z_prev = shift_right(prev_stream)
        delta = self.w_e(rms_normalize(z_prev, self.eps).to(embed.dtype))
        return embed + delta

    def _fbt_input(self, embed: torch.Tensor, prev_head: torch.Tensor) -> torch.Tensor:
        h = shift_right(prev_head)
        value = self.w_s(h)
        gate = self.w_g(rms_normalize(embed, self.eps).to(embed.dtype))
        fused = rms_normalize(value.float() * torch.sigmoid(gate.float()), self.eps)
        valid = torch.arange(embed.shape[1], device=embed.device) > 0
        valid = valid.view(1, -1, 1)
        return torch.where(valid, fused.to(embed.dtype), embed)

    def _pass(self, token_ids: torch.Tensor, prev: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = token_ids.shape[1]
        embed = self.embed(token_ids) + sinusoid(t, self.d, token_ids.device)
        if prev is None or self.mode == "vanilla":
            x = embed
            mapped = None
        elif self.mode == "fbt":
            _branches, _stream, head_in = prev
            x = self._fbt_input(embed, head_in)
            mapped = None
        elif self.mode == "alf":
            branches, stream, _head_in = prev
            x = self._inject(embed, stream)
            mapped = self._map_sources(shift_right(branches))
        else:
            _never(self.mode)
            x = embed
            mapped = None

        valid = (torch.arange(t, device=token_ids.device) > 0).view(1, t, 1).to(x.dtype)
        outs: list[torch.Tensor] = []
        for layer_i, block in enumerate(self.blocks):
            dest_attn = 2 * layer_i
            h = x if mapped is None else self._mix(x, mapped, dest_attn, valid).to(x.dtype)
            attn_out = block.attn(block.norm_attn(h))
            x = x + attn_out
            outs.append(attn_out)
            dest_ff = dest_attn + 1
            h = x if mapped is None else self._mix(x, mapped, dest_ff, valid).to(x.dtype)
            ff_out = block.ff(block.norm_ff(h))
            x = x + ff_out
            outs.append(ff_out)
        branches = torch.stack(outs, dim=2)
        head_in = self.final_norm(x)
        logits = self.head(head_in)
        return logits, branches, x

    def forward(self, token_ids: torch.Tensor, n_passes: int) -> torch.Tensor:
        if n_passes < 1:
            raise ValueError("n_passes must be positive")
        self.beta_last = None
        self._beta_hits = 0
        if self.mode == "alf" and self.read_mode == "full" and n_passes >= 2:
            self.beta_last = torch.zeros(self.n_sub, self.n_sub, device=token_ids.device)
        prev: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        logits = token_ids.new_zeros(token_ids.shape[0], token_ids.shape[1], self.head.out_features, dtype=torch.float32)
        for k in range(n_passes):
            logits, branches, stream = self._pass(token_ids, prev)
            head_in = self.final_norm(stream)
            prev = (branches, stream, head_in)
            if k + 1 == n_passes:
                break
        return logits

    def loss(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor,
        n_passes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Equation (3.8): first pass plus the mean of later passes.

        Returns (combined loss, final-pass logits).
        """
        if n_passes < 1:
            raise ValueError("n_passes must be positive")
        losses: list[torch.Tensor] = []
        prev: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        logits = token_ids.new_zeros(1)
        collect = self.mode == "alf" and self.read_mode == "full" and n_passes >= 2
        self.beta_last = torch.zeros(self.n_sub, self.n_sub, device=token_ids.device) if collect else None
        self._beta_hits = 0
        for k in range(n_passes):
            use_beta = collect and k + 1 == n_passes
            if not use_beta:
                saved = self.beta_last
                self.beta_last = None
            logits, branches, stream = self._pass(token_ids, prev)
            if not use_beta:
                self.beta_last = saved
            head_in = self.final_norm(stream)
            prev = (branches, stream, head_in)
            losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)))
        if n_passes == 1:
            total = losses[0]
        else:
            rest = torch.stack(losses[1:]).mean()
            total = losses[0] + rest
        return total, logits


def count_parameters(model: FeedbackTransformer) -> int:
    return model.n_parameters()


def build_model(
    vocab: int,
    d: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    mode: Mode,
    rank: int,
    init_share: float,
    seed: int,
) -> FeedbackTransformer:
    torch.manual_seed(seed)
    model = FeedbackTransformer(
        vocab=vocab,
        d=d,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        mode=mode,
        rank=rank,
        init_share=init_share,
    )
    model.reset_feedback()
    model.freeze_unused()
    return model


def match_width(
    vocab: int,
    d: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    mode: Mode,
    rank: int,
    init_share: float,
    seed: int,
    target: int,
) -> tuple[int, int]:
    """Pick an FFN width whose parameter count is closest to `target`."""
    best_ff = d_ff
    best_n = -1
    best_gap = 10**18
    ff = d_ff
    while ff <= d_ff + 512:
        n = build_model(vocab, d, n_layers, n_heads, ff, mode, rank, init_share, seed).n_parameters()
        gap = abs(n - target)
        if gap < best_gap:
            best_gap = gap
            best_ff = ff
            best_n = n
        if n >= target and ff > d_ff:
            break
        ff += 8
    for ff in range(max(d_ff, best_ff - 8), best_ff + 9):
        n = build_model(vocab, d, n_layers, n_heads, ff, mode, rank, init_share, seed).n_parameters()
        gap = abs(n - target)
        if gap < best_gap:
            best_gap = gap
            best_ff = ff
            best_n = n
    return best_ff, best_n


def build_trio(
    vocab: int,
    d: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    rank: int,
    init_share: float,
    seed: int,
) -> tuple[dict[Mode, FeedbackTransformer], dict[str, int]]:
    """ALF keeps `d_ff`. Vanilla and FBT widen the FFN to match its count."""
    alf = build_model(vocab, d, n_layers, n_heads, d_ff, "alf", rank, init_share, seed)
    target = alf.n_parameters()
    fbt_ff, fbt_n = match_width(vocab, d, n_layers, n_heads, d_ff, "fbt", rank, init_share, seed, target)
    van_ff, van_n = match_width(vocab, d, n_layers, n_heads, d_ff, "vanilla", rank, init_share, seed, target)
    models: dict[Mode, FeedbackTransformer] = {
        "alf": alf,
        "fbt": build_model(vocab, d, n_layers, n_heads, fbt_ff, "fbt", rank, init_share, seed),
        "vanilla": build_model(vocab, d, n_layers, n_heads, van_ff, "vanilla", rank, init_share, seed),
    }
    counts = {
        "alf": target,
        "fbt": fbt_n,
        "vanilla": van_n,
        "d_ff_alf": d_ff,
        "d_ff_fbt": fbt_ff,
        "d_ff_vanilla": van_ff,
    }
    return models, counts
