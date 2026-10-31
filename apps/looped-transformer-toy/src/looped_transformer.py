"""Clean-room tiny full-stack looped transformer for multi-digit addition.

Inspired by the full-stack looping idea in Meta/FAIR arXiv:2610.40325
("How to Loop Your Transformer"). No upstream code is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocabulary: digits 0-9, '+', '=', PAD
DIGIT_OFFSET = 0
PLUS_ID = 10
EQ_ID = 11
PAD_ID = 12
VOCAB_SIZE = 13


def set_threads(n: int) -> None:
    torch.set_num_threads(max(1, int(n)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def digits_lsb(n: int, width: int) -> list[int]:
    """LSB-first decimal digits, zero-padded to `width`."""
    out = []
    x = abs(int(n))
    for _ in range(width):
        out.append(x % 10)
        x //= 10
    return out


def encode_addition(a: int, b: int, n_digits: int) -> tuple[list[int], int]:
    """Return token ids for a+b=sum (LSB-first) and the sum length (n_digits+1)."""
    s = a + b
    a_dig = digits_lsb(a, n_digits)
    b_dig = digits_lsb(b, n_digits)
    s_dig = digits_lsb(s, n_digits + 1)
    toks = a_dig + [PLUS_ID] + b_dig + [EQ_ID] + s_dig
    return toks, n_digits + 1


def make_batch(
    n: int,
    n_digits: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample random additions. Returns (input_ids, target_ids, sum_mask).

    input_ids / target_ids are length L = 2*n_digits + 2 + (n_digits+1).
    Causal LM: predict token t+1 from prefix; sum_mask marks sum-digit positions
    in the *target* (same index as input shifted).
    """
    lo = 0
    hi = 10**n_digits
    a = rng.integers(lo, hi, size=n, dtype=np.int64)
    b = rng.integers(lo, hi, size=n, dtype=np.int64)
    seq_len = 2 * n_digits + 2 + (n_digits + 1)
    x = np.full((n, seq_len), PAD_ID, dtype=np.int64)
    for i in range(n):
        toks, _ = encode_addition(int(a[i]), int(b[i]), n_digits)
        x[i, : len(toks)] = np.asarray(toks, dtype=np.int64)

    # Causal: inputs[:-1], targets[1:]
    inp = torch.as_tensor(x[:, :-1])
    tgt = torch.as_tensor(x[:, 1:])
    # Sum digits start after a + '+' + b + '=' in the *original* sequence.
    # Original index of first sum digit: 2*n_digits + 2
    # In target (shifted by 1), that index is (2*n_digits + 2) - 1 = 2*n_digits + 1
    sum_start = 2 * n_digits + 1
    sum_len = n_digits + 1
    mask = torch.zeros_like(tgt, dtype=torch.bool)
    mask[:, sum_start : sum_start + sum_len] = True
    return inp, tgt, mask


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(a)
        x = x + self.ff(self.ln2(x))
        return x


class TinyLoopedTransformer(nn.Module):
    """Decoder-only transformer with optional full-stack looping.

    Full-stack looping: run the entire stack of `n_layers` blocks `n_loops`
    times, feeding the stack output back as the next loop's input (paper's
    Full-Stack Looping idea at toy scale).
    """

    def __init__(
        self,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
        ff_mult: int = 4,
        n_loops: int = 1,
        vocab_size: int = VOCAB_SIZE,
        max_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_loops = int(n_loops)
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, ff_mult, dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def set_loops(self, n_loops: int) -> None:
        self.n_loops = max(1, int(n_loops))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _causal_mask(self, t: int, device: torch.device) -> torch.Tensor:
        # MultiheadAttention expects float mask with -inf for disallowed
        mask = torch.zeros(t, t, device=device)
        mask = mask.masked_fill(
            torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1),
            float("-inf"),
        )
        return mask

    def forward(self, idx: torch.Tensor, n_loops: int | None = None) -> torch.Tensor:
        b, t = idx.shape
        if t > self.max_len:
            raise ValueError(f"sequence length {t} > max_len {self.max_len}")
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        attn_mask = self._causal_mask(t, idx.device)
        loops = self.n_loops if n_loops is None else max(1, int(n_loops))
        for _ in range(loops):
            for block in self.blocks:
                x = block(x, attn_mask)
        return self.head(self.ln_f(x))


@dataclass
class TrainResult:
    model: TinyLoopedTransformer
    train_wall_s: float
    final_loss: float
    n_params: int
    history: list[dict]


def train_model(
    kind: str,
    seed: int,
    *,
    n_digits: int,
    steps: int,
    batch_size: int,
    lr: float,
    d_model: int,
    n_layers: int,
    n_heads: int,
    dropout: float,
    ff_mult: int,
    n_loops: int,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    log_every: int = 500,
    stream_seed: int = 0,
) -> TrainResult:
    import time

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(stream_seed + seed * 1009)

    model = TinyLoopedTransformer(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        ff_mult=ff_mult,
        n_loops=n_loops,
        max_len=64,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    history: list[dict] = []
    t0 = time.perf_counter()
    last_loss = 0.0
    for step in range(1, steps + 1):
        inp, tgt, sum_mask = make_batch(batch_size, n_digits, np_rng)
        logits = model(inp)
        # Supervise only sum digits (algorithmic target); ignore prefix.
        loss = F.cross_entropy(
            logits[sum_mask],
            tgt[sum_mask],
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        last_loss = float(loss.item())
        if step % log_every == 0 or step == 1 or step == steps:
            history.append({"step": step, "loss": last_loss, "kind": kind})
            print(
                f"  [{kind}] seed={seed} step={step}/{steps} loss={last_loss:.4f}",
                flush=True,
            )
    wall = time.perf_counter() - t0
    return TrainResult(
        model=model,
        train_wall_s=wall,
        final_loss=last_loss,
        n_params=model.n_params(),
        history=history,
    )


@torch.no_grad()
def evaluate_exact_match(
    model: TinyLoopedTransformer,
    *,
    n_digits: int,
    n_eval: int,
    eval_seed: int,
    n_loops: int | None = None,
    batch_size: int = 128,
) -> dict:
    """Greedy decode sum digits given a+b= prefix; report exact-match + digit acc."""
    import time

    model.eval()
    rng = np.random.default_rng(eval_seed)
    loops = model.n_loops if n_loops is None else int(n_loops)
    sum_len = n_digits + 1
    prefix_len = 2 * n_digits + 2  # a digits + '+' + b digits + '='
    total = 0
    exact = 0
    digit_ok = 0
    digit_tot = 0
    t0 = time.perf_counter()
    while total < n_eval:
        bsz = min(batch_size, n_eval - total)
        a = rng.integers(0, 10**n_digits, size=bsz, dtype=np.int64)
        b = rng.integers(0, 10**n_digits, size=bsz, dtype=np.int64)
        # Build prefix tokens only; then generate sum_len tokens.
        prefixes = []
        truths = []
        for i in range(bsz):
            toks, _ = encode_addition(int(a[i]), int(b[i]), n_digits)
            prefixes.append(toks[:prefix_len])
            truths.append(toks[prefix_len:])
        cur = torch.as_tensor(np.asarray(prefixes, dtype=np.int64))
        for _ in range(sum_len):
            logits = model(cur, n_loops=loops)
            nxt = logits[:, -1, :].argmax(dim=-1)
            cur = torch.cat([cur, nxt[:, None]], dim=1)
        pred = cur[:, prefix_len:].cpu().numpy()
        truth = np.asarray(truths, dtype=np.int64)
        match = (pred == truth).all(axis=1)
        exact += int(match.sum())
        digit_ok += int((pred == truth).sum())
        digit_tot += int(truth.size)
        total += bsz
    wall = time.perf_counter() - t0
    return {
        "exact_match": exact / max(1, total),
        "digit_accuracy": digit_ok / max(1, digit_tot),
        "n": total,
        "n_digits": n_digits,
        "n_loops": loops,
        "eval_wall_s": wall,
        "n_params": model.n_params(),
    }
