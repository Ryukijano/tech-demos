"""Minimal Binary Discrete Diffusion vs masked diffusion on Sudoku (CPU toy).

Independent re-implementation of the *idea* in Casado Noguerales et al. 2026,
arXiv:2610.28543 — NOT the authors' code (none found as of 2026-10-28).

Binary symmetric kernel (paper Sec. 3.1, simplified):
  Clean digit c is the singleton {c}. At noise level t each membership bit
  differs from the clean bit with probability t/2 (independent fair-bit limit
  as t→1). The network predicts a V-way clean-token distribution μ; candidate
  sets are embedded as E^T (z / ||z||_2) (paper Eq. 15).

Masked kernel (MDLM-style baseline, matched trunk):
  With probability t a non-clue cell is MASK; else the clean digit remains.
  Embedding is a standard token embedding (+ MASK id).

Samplers are simplified ancestral / re-corrupt style (paper Sec. 4 spirit),
not the authors' full confidence-guided revision stack. Clues stay fixed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

N = 9
V = 9
L = N * N
MASK_ID = V  # token id for masked diffusion


# ---------------------------------------------------------------- Sudoku data


def _valid_place(board: np.ndarray, r: int, c: int, dig: int) -> bool:
    if dig in board[r]:
        return False
    if dig in board[:, c]:
        return False
    br, bc = 3 * (r // 3), 3 * (c // 3)
    if dig in board[br : br + 3, bc : bc + 3]:
        return False
    return True


def _fill_grid(rng: np.random.Generator) -> np.ndarray:
    """Generate one complete valid 9x9 Sudoku via shuffled backtracking."""
    board = np.zeros((N, N), dtype=np.int8)

    def solve() -> bool:
        for i in range(L):
            r, c = divmod(i, N)
            if board[r, c] != 0:
                continue
            digits = rng.permutation(V) + 1
            for d in digits:
                if _valid_place(board, r, c, int(d)):
                    board[r, c] = int(d)
                    if solve():
                        return True
                    board[r, c] = 0
            return False
        return True

    # Seed band pattern for faster fill, then permute.
    for r in range(N):
        for c in range(N):
            board[r, c] = (3 * (r % 3) + r // 3 + c) % N + 1
    # Shuffle bands / stacks / rows / cols / digits for variety.
    rows = np.arange(N)
    cols = np.arange(N)
    for band in range(3):
        rng.shuffle(rows[band * 3 : band * 3 + 3])
        rng.shuffle(cols[band * 3 : band * 3 + 3])
    board = board[rows][:, cols]
    digit_perm = rng.permutation(V) + 1
    board = digit_perm[board - 1]
    # Ensure validity; if band shuffle broke uniqueness of digits in boxes
    # (it shouldn't for the standard construction), fall back to backtrack.
    if not is_valid_complete(board):
        board[:] = 0
        assert solve()
    return board


def is_valid_complete(board: np.ndarray) -> bool:
    if board.shape != (N, N) or board.min() < 1 or board.max() > 9:
        return False
    if np.any(board == 0):
        return False
    for i in range(N):
        if len(set(board[i].tolist())) != N:
            return False
        if len(set(board[:, i].tolist())) != N:
            return False
    for br in range(0, N, 3):
        for bc in range(0, N, 3):
            block = board[br : br + 3, bc : bc + 3].ravel()
            if len(set(block.tolist())) != N:
                return False
    return True


def make_puzzle_pool(n: int, seed: int) -> np.ndarray:
    """Return (n, 9, 9) complete grids."""
    rng = np.random.default_rng(seed)
    grids = np.stack([_fill_grid(rng) for _ in range(n)], axis=0)
    return grids


def punch_clues(
    solutions: np.ndarray, n_clues: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return clues board (0=empty) and boolean clue mask, shape (B,9,9)."""
    b = solutions.shape[0]
    clues = solutions.copy()
    mask = np.zeros((b, N, N), dtype=bool)
    for i in range(b):
        idx = rng.permutation(L)[:n_clues]
        keep = np.zeros(L, dtype=bool)
        keep[idx] = True
        keep = keep.reshape(N, N)
        mask[i] = keep
        clues[i] = np.where(keep, solutions[i], 0)
    return clues, mask


# ---------------------------------------------------------------- kernels


def corrupt_binary(
    clean: torch.Tensor, t: torch.Tensor, clue_mask: torch.Tensor, generator=None
) -> torch.Tensor:
    """clean: (B,L) int 0..8; t: (B,); clue_mask: (B,L) bool.
    Returns membership (B,L,V) float {0,1}.
    """
    b, l = clean.shape
    device = clean.device
    # one-hot clean
    z = torch.zeros(b, l, V, device=device)
    z.scatter_(2, clean.unsqueeze(-1), 1.0)
    # independent bit flips toward fair bits with P(diff)=t/2
    # Equivalent: keep clean bit with 1-t/2; other bits on with t/2
    t = t.view(b, 1, 1)
    # With prob t: replace bit with Bern(1/2); else keep clean bit.
    # P(diff from clean) = t/2, matching the paper's symmetric kernel.
    u = torch.rand(b, l, V, device=device, generator=generator)
    resample_mask = u < t
    fair = (torch.rand(b, l, V, device=device, generator=generator) < 0.5).float()
    z = torch.where(resample_mask, fair, z)
    # force clues to singleton
    cm = clue_mask.unsqueeze(-1)
    onehot = torch.zeros_like(z)
    onehot.scatter_(2, clean.unsqueeze(-1), 1.0)
    z = torch.where(cm, onehot, z)
    # ensure at least one candidate (rare empty after fair resample)
    empty = z.sum(-1, keepdim=True) == 0
    z = torch.where(empty, onehot, z)
    return z


def corrupt_masked(
    clean: torch.Tensor, t: torch.Tensor, clue_mask: torch.Tensor, generator=None
) -> torch.Tensor:
    """Returns tokens (B,L) in 0..V (V=MASK)."""
    b, l = clean.shape
    device = clean.device
    t = t.view(b, 1)
    u = torch.rand(b, l, device=device, generator=generator)
    masked = torch.where(u < t, torch.full_like(clean, MASK_ID), clean)
    return torch.where(clue_mask, clean, masked)


# ---------------------------------------------------------------- model


class TinySudokuDiT(nn.Module):
    """Tiny transformer trunk shared by both kernels; front-end differs."""

    def __init__(self, d_model: int = 64, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(V + 1, d_model)  # +MASK
        self.digit_emb = nn.Embedding(V, d_model)  # for binary set embedding
        self.pos_emb = nn.Embedding(L, d_model)
        self.clue_emb = nn.Embedding(2, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, V)

    def embed_binary(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B,L,V) membership; paper Eq. 15: E^T (z / ||z||)
        E = self.digit_emb.weight  # (V,d)
        norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        zn = z / norm
        return zn @ E  # (B,L,d)

    def embed_masked(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_emb(tokens)

    def forward(
        self,
        *,
        kernel: str,
        state: torch.Tensor,
        t: torch.Tensor,
        clue_mask: torch.Tensor,
    ) -> torch.Tensor:
        b = state.shape[0]
        if kernel == "binary":
            x = self.embed_binary(state)
        elif kernel == "masked":
            x = self.embed_masked(state)
        else:
            raise ValueError(kernel)
        pos = self.pos_emb(torch.arange(L, device=state.device)).unsqueeze(0)
        clue = self.clue_emb(clue_mask.long())
        temb = self.time_mlp(t.view(b, 1)).unsqueeze(1)
        h = x + pos + clue + temb
        h = self.encoder(h)
        return self.out(h)  # (B,L,V) logits


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------- train / sample


@dataclass
class TrainResult:
    model: TinySudokuDiT
    wall_s: float
    final_loss: float
    n_params: int


def set_threads(n: int) -> None:
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(max(1, min(n, 2)))
    except RuntimeError:
        pass


def _batch_from_pool(
    pool: np.ndarray,
    batch: int,
    n_clues: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = rng.integers(0, len(pool), size=batch)
    sols = pool[idx]
    clues, mask = punch_clues(sols, n_clues, rng)
    clean = torch.as_tensor(sols.reshape(batch, L) - 1, device=device, dtype=torch.long)
    clue_mask = torch.as_tensor(mask.reshape(batch, L), device=device)
    return clean, clue_mask


def train_kernel(
    kernel: str,
    seed: int,
    pool: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    n_clues: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    dropout: float = 0.0,
) -> TrainResult:
    device = torch.device("cpu")
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = TinySudokuDiT(d_model, n_layers, n_heads, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    t0 = time.perf_counter()
    last_loss = 0.0
    for step in range(steps):
        clean, clue_mask = _batch_from_pool(pool, batch_size, n_clues, rng, device)
        t = torch.rand(batch_size, generator=g)
        # avoid exact 0/1
        t = t.clamp(1e-3, 1 - 1e-3)
        if kernel == "binary":
            state = corrupt_binary(clean, t, clue_mask, generator=g)
        else:
            state = corrupt_masked(clean, t, clue_mask, generator=g)
        logits = model(kernel=kernel, state=state, t=t, clue_mask=clue_mask)
        # CE on non-clue cells (and for masked, especially useful on masked cells;
        # we supervise all non-clue positions for a matched objective).
        loss_tok = F.cross_entropy(
            logits.reshape(-1, V), clean.reshape(-1), reduction="none"
        ).view(batch_size, L)
        weight = (~clue_mask).float()
        if kernel == "masked":
            # emphasize currently-masked positions (paper-style masked CE)
            is_mask = state == MASK_ID
            weight = weight * (0.1 + 0.9 * is_mask.float())
        loss = (loss_tok * weight).sum() / weight.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        if (step + 1) % 500 == 0 or step == 0:
            print(f"  [{kernel} seed={seed}] step {step+1}/{steps} loss={last_loss:.4f}", flush=True)
    wall = time.perf_counter() - t0
    model.eval()
    return TrainResult(model=model, wall_s=wall, final_loss=last_loss, n_params=count_params(model))


@torch.no_grad()
def sample_binary(
    model: TinySudokuDiT,
    clues: torch.Tensor,
    clue_mask: torch.Tensor,
    *,
    nfe: int,
    t_min: float = 0.02,
    generator: torch.Generator | None = None,
    record: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Candidate-narrowing sampler for binary diffusion (CPU toy).

    Starts from the full vocabulary at blank cells (absorbing-style prior) and
    iteratively prunes low-probability candidates under the model's clean-token
    prediction, with a light symmetric re-open of near-ties early on — a
    simplified stand-in for the paper's confidence-guided revision (Sec. 4),
    not the authors' full sampler. Clue cells stay singletons throughout.
    """
    b = clues.shape[0]
    device = clues.device
    z = torch.ones(b, L, V, device=device)
    onehot = torch.zeros(b, L, V, device=device)
    onehot.scatter_(2, clues.unsqueeze(-1), 1.0)
    z = torch.where(clue_mask.unsqueeze(-1), onehot, z)
    traj = []
    for i in range(nfe):
        # noise level decreases; early steps keep larger candidate sets
        t_val = max(1.0 - (i + 1) / nfe, t_min)
        t = torch.full((b,), t_val, device=device)
        logits = model(kernel="binary", state=z, t=t, clue_mask=clue_mask)
        # Mask logits outside the current candidate set (absorbing-style)
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(z < 0.5, neg_inf)
        probs = F.softmax(masked_logits, dim=-1)
        # Target set size: V -> 1 over the schedule
        target_k = max(1, int(round(V * (1.0 - (i + 1) / nfe))))
        # Keep top-k candidates by model probability; always keep current argmax
        topk = torch.topk(probs, k=max(target_k, 1), dim=-1)
        new_z = torch.zeros_like(z)
        new_z.scatter_(2, topk.indices, 1.0)
        # Early steps: optionally re-open one extra candidate (symmetric flavor)
        if i < nfe // 3:
            flat_p = probs.clamp_min(1e-8).reshape(-1, V)
            reopen = torch.multinomial(flat_p, 1, generator=generator).view(b, L, 1)
            new_z.scatter_(2, reopen, 1.0)
        z = torch.where(clue_mask.unsqueeze(-1), onehot, new_z)
        # ensure non-empty
        empty = z.sum(-1, keepdim=True) == 0
        fill = torch.zeros_like(z)
        fill.scatter_(2, probs.argmax(-1, keepdim=True), 1.0)
        z = torch.where(empty, fill, z)
        if record:
            traj.append(z.detach().cpu().clone())
    pred = z.argmax(-1)
    pred = torch.where(clue_mask, clues, pred)
    return pred, traj


@torch.no_grad()
def sample_masked(
    model: TinySudokuDiT,
    clues: torch.Tensor,
    clue_mask: torch.Tensor,
    *,
    nfe: int,
    t_min: float = 0.02,
    generator: torch.Generator | None = None,
    record: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Simplified confidence-unmask sampler for masked diffusion (matched NFE)."""
    b = clues.shape[0]
    device = clues.device
    tokens = torch.full((b, L), MASK_ID, device=device, dtype=torch.long)
    tokens = torch.where(clue_mask, clues, tokens)
    traj = []
    # unmask ~1/nfe of remaining masks each step by confidence
    for step in range(nfe):
        t_val = 1.0 - (step + 1) / nfe
        t_val = max(t_val, t_min)
        t = torch.full((b,), t_val, device=device)
        logits = model(kernel="masked", state=tokens, t=t, clue_mask=clue_mask)
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        still = tokens == MASK_ID
        # number to unmask this step
        n_left = still.sum(dim=-1)
        # unmask roughly equal fraction of remaining
        frac = 1.0 / max(nfe - step, 1)
        for bi in range(b):
            left_idx = still[bi].nonzero(as_tuple=False).view(-1)
            if left_idx.numel() == 0:
                continue
            k = max(1, int(math.ceil(left_idx.numel() * frac)))
            if step == nfe - 1:
                k = left_idx.numel()
            scores = conf[bi, left_idx]
            topk = torch.topk(scores, k=min(k, left_idx.numel())).indices
            choose = left_idx[topk]
            tokens[bi, choose] = pred[bi, choose]
        tokens = torch.where(clue_mask, clues, tokens)
        if record:
            traj.append(tokens.detach().cpu().clone())
    return tokens, traj


def decode_board(tokens: torch.Tensor) -> np.ndarray:
    """(B,L) 0..8 -> (B,9,9) digits 1..9."""
    arr = tokens.detach().cpu().numpy().reshape(-1, N, N) + 1
    return arr


def evaluate_predictions(
    pred: np.ndarray, solutions: np.ndarray, clue_mask: np.ndarray
) -> dict:
    """pred/solutions: (B,9,9) digits; clue_mask: (B,9,9) bool."""
    b = pred.shape[0]
    solved = 0
    valid = 0
    cell_correct = 0
    cell_total = 0
    for i in range(b):
        match = np.array_equal(pred[i], solutions[i])
        ok = is_valid_complete(pred[i])
        if ok:
            valid += 1
        if match and ok:
            solved += 1
        free = ~clue_mask[i]
        cell_correct += int((pred[i][free] == solutions[i][free]).sum())
        cell_total += int(free.sum())
    return {
        "n": b,
        "solve_rate": solved / b,
        "valid_board_rate": valid / b,
        "cell_accuracy": cell_correct / max(cell_total, 1),
        "n_solved": solved,
    }
