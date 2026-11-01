"""Clean-room tiny MoE / MoRE character LM.

Inspired by Qiu, Acikalin et al., "MoRE: Mixture of Reused Experts"
(arXiv:2609.18176). No upstream source is vendored — official full-scale
code exists at https://github.com/EricQiu6/MoRE (Apache-2.0) but is not used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_threads(n: int) -> None:
    torch.set_num_threads(max(1, int(n)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# Corpus / charset
# ---------------------------------------------------------------------------


@dataclass
class CharData:
    text: str
    stoi: dict[str, int]
    itos: dict[int, str]
    train_ids: torch.Tensor
    val_ids: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


def load_char_data(path: Path, train_frac: float = 0.9) -> CharData:
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(len(ids) * float(train_frac))
    n = max(1, min(n, len(ids) - 1))
    return CharData(
        text=text,
        stoi=stoi,
        itos=itos,
        train_ids=ids[:n],
        val_ids=ids[n:],
    )


def sample_batch(
    ids: torch.Tensor,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    hi = len(ids) - seq_len - 1
    if hi <= 0:
        raise ValueError("corpus too short for seq_len")
    starts = rng.integers(0, hi, size=batch_size, dtype=np.int64)
    x = torch.stack([ids[s : s + seq_len] for s in starts])
    y = torch.stack([ids[s + 1 : s + seq_len + 1] for s in starts])
    return x, y


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoERouter(nn.Module):
    """Top-K softmax router over a pool of experts (Switch / Mixtral style)."""

    def __init__(self, d_model: int, n_experts: int, top_k: int):
        super().__init__()
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (combined, top_idx, top_w, aux_loss).

        x: (B, T, D)
        combined: (B, T, D) after expert mix — filled by caller; here we return
        routing tensors only via a helper. Kept for API clarity in layer.
        """
        logits = self.gate(x)  # (B, T, E)
        probs = F.softmax(logits, dim=-1)
        top_w, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # Switch-style load-balance aux: N * sum_i f_i * P_i
        # f_i = fraction of tokens routed to expert i (hard top-1 of top-k set
        # counted once per selected expert / K soft).
        b, t, e = probs.shape
        flat_idx = top_idx.reshape(-1, self.top_k)
        ones = torch.ones(flat_idx.shape[0], self.top_k, device=x.device)
        # soft count: each of K selections contributes 1/K
        frac = torch.zeros(e, device=x.device)
        frac.scatter_add_(0, flat_idx.reshape(-1), (ones / self.top_k).reshape(-1))
        frac = frac / (b * t)
        p_mean = probs.mean(dim=(0, 1))
        aux = float(e) * (frac * p_mean).sum()
        return probs, top_idx, top_w, aux


class SharedExpertPool(nn.Module):
    def __init__(self, n_experts: int, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, d_ff, dropout) for _ in range(n_experts)]
        )

    def __len__(self) -> int:
        return len(self.experts)

    def forward_selected(
        self,
        x: torch.Tensor,
        top_idx: torch.Tensor,
        top_w: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to selected experts.

        x: (B, T, D); top_idx/top_w: (B, T, K)
        Returns (B, T, D).

        For the tiny expert counts in this CPU toy, evaluating every expert on
        the full batch and gathering is faster than sparse Python scatter.
        """
        # stacked: (B, T, E, D)
        stacked = torch.stack([expert(x) for expert in self.experts], dim=2)
        # gather selected experts → (B, T, K, D)
        idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, stacked.size(-1))
        selected = torch.gather(stacked, 2, idx)
        # weight and sum over K
        return (selected * top_w.unsqueeze(-1)).sum(dim=2)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        ffn: nn.Module | None = None,
        moe_pool: SharedExpertPool | None = None,
        n_pool_experts: int | None = None,
        top_k: int = 2,
        use_depth_embed: bool = False,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = ffn
        self.moe_pool = moe_pool
        self.top_k = int(top_k)
        self.aux_loss_coef = float(aux_loss_coef)
        self.use_depth_embed = bool(use_depth_embed)
        self.depth_embed: nn.Parameter | None
        if use_depth_embed:
            self.depth_embed = nn.Parameter(torch.zeros(d_model))
        else:
            self.depth_embed = None
        self.router: MoERouter | None
        if moe_pool is not None:
            assert n_pool_experts is not None
            self.router = MoERouter(d_model, n_pool_experts, top_k)
        else:
            self.router = None
        self.last_top_idx: torch.Tensor | None = None
        self.last_aux: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(a)
        h = self.ln2(x)
        if self.router is not None and self.moe_pool is not None:
            if self.depth_embed is not None:
                h = h + self.depth_embed
            _, top_idx, top_w, aux = self.router(h)
            y = self.moe_pool.forward_selected(h, top_idx, top_w)
            self.last_top_idx = top_idx.detach()
            self.last_aux = aux
            x = x + self.dropout(y)
        else:
            assert self.ffn is not None
            x = x + self.ffn(h)
            self.last_top_idx = None
            self.last_aux = None
        return x


class TinyMoRELM(nn.Module):
    """Decoder-only LM with dense / MoE / MoRE / MoRE-no-depth variants.

    Parameter matching (paper idea at toy scale):
      - MoE: L layers × N experts each → L*N unique experts.
      - MoRE (R): L/R reuse blocks, each with pool size R*N → also L*N unique
        experts; each layer's router sees R*N candidates.
      - Dense: FLOPs-matched FFN width ≈ top_k * d_ff_expert.
    Attention / routers / depth embeddings stay layer-specific (MoRE paper).
    """

    KINDS = ("dense", "moe", "more", "more_nodepth")

    def __init__(
        self,
        *,
        kind: str,
        vocab_size: int,
        d_model: int = 48,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.0,
        n_experts: int = 4,
        top_k: int = 2,
        reuse_factor: int = 2,
        d_ff_expert: int = 96,
        d_ff_dense: int = 192,
        aux_loss_coef: float = 0.01,
        max_len: int = 128,
    ):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"unknown kind {kind}")
        self.kind = kind
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.reuse_factor = int(reuse_factor)
        self.aux_loss_coef = float(aux_loss_coef)
        self.max_len = max_len

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        self.pools = nn.ModuleList()
        self.blocks = nn.ModuleList()

        if kind == "dense":
            for _ in range(n_layers):
                self.blocks.append(
                    TransformerBlock(
                        d_model,
                        n_heads,
                        dropout,
                        ffn=DenseFFN(d_model, d_ff_dense, dropout),
                    )
                )
        elif kind == "moe":
            for _ in range(n_layers):
                pool = SharedExpertPool(n_experts, d_model, d_ff_expert, dropout)
                self.pools.append(pool)
                self.blocks.append(
                    TransformerBlock(
                        d_model,
                        n_heads,
                        dropout,
                        moe_pool=pool,
                        n_pool_experts=n_experts,
                        top_k=top_k,
                        use_depth_embed=False,
                        aux_loss_coef=aux_loss_coef,
                    )
                )
        else:
            # MoRE / MoRE without depth embeddings
            use_depth = kind == "more"
            r = max(1, int(reuse_factor))
            if n_layers % r != 0:
                raise ValueError(
                    f"n_layers={n_layers} must be divisible by reuse_factor={r}"
                )
            pool_size = r * n_experts
            n_blocks = n_layers // r
            for _ in range(n_blocks):
                pool = SharedExpertPool(pool_size, d_model, d_ff_expert, dropout)
                self.pools.append(pool)
                for _li in range(r):
                    self.blocks.append(
                        TransformerBlock(
                            d_model,
                            n_heads,
                            dropout,
                            moe_pool=pool,
                            n_pool_experts=pool_size,
                            top_k=top_k,
                            use_depth_embed=use_depth,
                            aux_loss_coef=aux_loss_coef,
                        )
                    )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Depth embeddings start at zero (paper: additive shift).
        for block in self.blocks:
            if block.depth_embed is not None:
                nn.init.zeros_(block.depth_embed)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_active_params_per_token(self) -> int:
        """Rough active expert+dense FFN params per token (ignore attention)."""
        # Count one expert's params * top_k * n_layers (+ dense path).
        if self.kind == "dense":
            # one DenseFFN
            for b in self.blocks:
                if b.ffn is not None:
                    return sum(p.numel() for p in b.ffn.parameters()) * self.n_layers
            return 0
        # expert size from first pool
        if not self.pools:
            return 0
        e0 = self.pools[0].experts[0]
        e_params = sum(p.numel() for p in e0.parameters())
        return e_params * self.top_k * self.n_layers

    def _causal_mask(self, t: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(t, t, device=device)
        mask = mask.masked_fill(
            torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1),
            float("-inf"),
        )
        return mask

    def forward(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = idx.shape
        if t > self.max_len:
            raise ValueError(f"seq len {t} > max_len {self.max_len}")
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        attn_mask = self._causal_mask(t, idx.device)
        aux_total = torch.zeros((), device=idx.device)
        n_aux = 0
        for block in self.blocks:
            x = block(x, attn_mask)
            if block.last_aux is not None:
                aux_total = aux_total + block.last_aux
                n_aux += 1
        logits = self.head(self.ln_f(x))
        if n_aux:
            aux_total = aux_total / n_aux
        return logits, self.aux_loss_coef * aux_total

    def collect_routing(self) -> list[torch.Tensor | None]:
        return [b.last_top_idx for b in self.blocks]


@dataclass
class TrainResult:
    model: TinyMoRELM
    train_wall_s: float
    final_loss: float
    final_val_ce: float
    n_params: int
    history: list[dict]


def evaluate_ce(
    model: TinyMoRELM,
    val_ids: torch.Tensor,
    *,
    seq_len: int,
    n_batches: int,
    batch_size: int,
    eval_seed: int,
) -> dict:
    model.eval()
    rng = np.random.default_rng(eval_seed)
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = sample_batch(val_ids, batch_size, seq_len, rng)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(float(loss.item()))
    ce = float(np.mean(losses)) if losses else float("nan")
    return {
        "val_ce": ce,
        "bits_per_char": ce / float(np.log(2.0)),
        "n_batches": n_batches,
    }


def adjacent_reuse_stats(
    model: TinyMoRELM,
    val_ids: torch.Tensor,
    *,
    seq_len: int,
    n_batches: int,
    batch_size: int,
    eval_seed: int,
) -> dict:
    """Rate at which adjacent layers pick overlapping experts vs random.

    reuse_rate = mean over tokens of |S_l ∩ S_{l+1}| / K for layers that share
    a pool (MoRE) or for consecutive layers (MoE — always 0 possible unique
    overlap since pools differ; we still report within-pool pairs for MoRE).
    Random baseline for independent Top-K from pool size M: K/M.
    """
    model.eval()
    rng = np.random.default_rng(eval_seed)
    # Identify adjacent layer pairs that share the same pool object.
    pairs: list[tuple[int, int, int]] = []  # (i, j, pool_size)
    for i in range(len(model.blocks) - 1):
        bi = model.blocks[i]
        bj = model.blocks[i + 1]
        if bi.moe_pool is None or bj.moe_pool is None:
            continue
        if bi.moe_pool is bj.moe_pool:
            pairs.append((i, i + 1, len(bi.moe_pool)))
    if not pairs:
        return {
            "adjacent_expert_reuse_rate": None,
            "random_reuse_baseline": None,
            "n_pairs": 0,
            "note": "no shared-pool adjacent pairs (dense or standard MoE)",
        }

    reuse_sums = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, _y = sample_batch(val_ids, batch_size, seq_len, rng)
            _logits, _ = model(x)
            routes = model.collect_routing()
            for i, j, m in pairs:
                ti = routes[i]  # (B, T, K)
                tj = routes[j]
                assert ti is not None and tj is not None
                # Intersection size per token
                # For each token, count how many of ti's experts appear in tj.
                k = ti.shape[-1]
                # broadcast compare: (B,T,K,1) vs (B,T,1,K)
                match = (ti.unsqueeze(-1) == tj.unsqueeze(-2)).any(dim=-1)  # (B,T,K)
                inter = match.float().sum(dim=-1)  # (B,T) in [0,K]
                reuse_sums.append((inter / k).mean().item())

    rate = float(np.mean(reuse_sums)) if reuse_sums else float("nan")
    # Expected |A∩B|/K for independent uniform Top-K from M is ≈ K/M
    # (exact E[|A∩B|] = K^2/M for sampling with replacement approx; without
    # replacement E = K * (K/M) still). Use K/M as paper-style random baseline.
    m0 = pairs[0][2]
    k0 = model.top_k
    random_base = float(k0) / float(m0)
    return {
        "adjacent_expert_reuse_rate": rate,
        "random_reuse_baseline": random_base,
        "n_pairs": len(pairs),
        "pool_size": m0,
        "top_k": k0,
        "ratio_vs_random": rate / random_base if random_base > 0 else None,
    }


def train_model(
    kind: str,
    seed: int,
    data: CharData,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seq_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    dropout: float,
    n_experts: int,
    top_k: int,
    reuse_factor: int,
    d_ff_expert: int,
    d_ff_dense: int,
    aux_loss_coef: float,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    log_every: int = 100,
    eval_every: int = 100,
    eval_batches: int = 16,
    stream_seed: int = 0,
) -> TrainResult:
    import time

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(stream_seed + seed * 1009)

    model = TinyMoRELM(
        kind=kind,
        vocab_size=data.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        n_experts=n_experts,
        top_k=top_k,
        reuse_factor=reuse_factor,
        d_ff_expert=d_ff_expert,
        d_ff_dense=d_ff_dense,
        aux_loss_coef=aux_loss_coef,
        max_len=max(128, seq_len + 8),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    history: list[dict] = []
    t0 = time.perf_counter()
    last_loss = 0.0
    last_val = float("nan")
    for step in range(1, steps + 1):
        x, y = sample_batch(data.train_ids, batch_size, seq_len, np_rng)
        logits, aux = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss = ce + aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        last_loss = float(ce.item())
        if step % log_every == 0 or step == 1 or step == steps:
            print(
                f"  [{kind}] seed={seed} step={step}/{steps} "
                f"ce={last_loss:.4f} aux={float(aux.item()):.5f}",
                flush=True,
            )
        if step % eval_every == 0 or step == steps:
            met = evaluate_ce(
                model,
                data.val_ids,
                seq_len=seq_len,
                n_batches=eval_batches,
                batch_size=batch_size,
                eval_seed=910_000 + seed * 17 + step,
            )
            last_val = met["val_ce"]
            history.append(
                {
                    "step": step,
                    "train_ce": last_loss,
                    "val_ce": last_val,
                    "bits_per_char": met["bits_per_char"],
                    "kind": kind,
                }
            )
            model.train()
            print(
                f"  [{kind}] seed={seed} step={step} val_ce={last_val:.4f} "
                f"bpc={met['bits_per_char']:.4f}",
                flush=True,
            )
    wall = time.perf_counter() - t0
    return TrainResult(
        model=model,
        train_wall_s=wall,
        final_loss=last_loss,
        final_val_ce=last_val,
        n_params=model.n_params(),
        history=history,
    )
