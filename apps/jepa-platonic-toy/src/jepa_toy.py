"""Clean-room tiny whitened JEPA + synthetic two-modality world.

Independent of galilai-group/prh-proof. VICReg-style invariance + variance +
covariance anti-collapse with an identity predictor (whitened JEPA form from
the Platonic Representation Theorem paper, toy scale only).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_threads(n: int) -> None:
    torch.set_num_threads(int(n))
    try:
        torch.set_num_interop_threads(max(1, int(n) // 2))
    except RuntimeError:
        pass


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))


@dataclass
class World:
    """Finite synthetic world with two linear renderings + nuisance."""

    class_means: torch.Tensor  # (n_classes, d_latent)
    state_offsets: torch.Tensor  # (n_classes, n_states, d_latent)
    W_a: torch.Tensor  # (d_mod_a, d_latent)
    W_b: torch.Tensor  # (d_mod_b, d_latent)
    n_classes: int
    n_states_per_class: int
    d_latent: int
    d_mod_a: int
    d_mod_b: int
    nuisance_std: float
    class_sep: float

    @property
    def n_states(self) -> int:
        return self.n_classes * self.n_states_per_class


def build_world(cfg_data: dict) -> World:
    g = torch.Generator().manual_seed(int(cfg_data["world_seed"]))
    n_classes = int(cfg_data["n_classes"])
    n_states = int(cfg_data["n_states_per_class"])
    d_lat = int(cfg_data["d_latent"])
    d_a = int(cfg_data["d_mod_a"])
    d_b = int(cfg_data["d_mod_b"])
    sep = float(cfg_data["class_sep"])

    # Spread class means on a sphere-ish lattice in latent space
    means = torch.randn(n_classes, d_lat, generator=g)
    means = F.normalize(means, dim=-1) * sep

    offsets = 0.35 * torch.randn(n_classes, n_states, d_lat, generator=g)

    # Tall random projections (d_mod >= d_latent): QR on columns for conditioning.
    # Injective on the finite world in practice when noise is moderate.
    def _proj(d_out: int) -> torch.Tensor:
        M = torch.randn(d_out, d_lat, generator=g) / math.sqrt(d_lat)
        # orthonormalize the d_lat columns of M.T (size d_lat x d_out -> Q d_lat x d_lat)
        Q, _ = torch.linalg.qr(M.T, mode="reduced")  # (d_lat, d_lat)
        # lift back with a random tall factor so shape is (d_out, d_lat)
        tall = torch.randn(d_out, d_lat, generator=g)
        Qt, _ = torch.linalg.qr(tall, mode="reduced")  # (d_out, d_lat)
        return Qt @ Q.T  # (d_out, d_lat)

    W_a = _proj(d_a)
    W_b = _proj(d_b)

    return World(
        class_means=means,
        state_offsets=offsets,
        W_a=W_a,
        W_b=W_b,
        n_classes=n_classes,
        n_states_per_class=n_states,
        d_latent=d_lat,
        d_mod_a=d_a,
        d_mod_b=d_b,
        nuisance_std=float(cfg_data["nuisance_std"]),
        class_sep=sep,
    )


def _sample_states(
    world: World,
    n: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (z, class_id, state_id) for n draws uniform over the finite world."""
    n_tot = world.n_states
    flat = torch.randint(0, n_tot, (n,), generator=generator)
    class_id = flat // world.n_states_per_class
    state_id = flat % world.n_states_per_class
    z = world.class_means[class_id] + world.state_offsets[class_id, state_id]
    return z, class_id, state_id


def render(
    world: World,
    z: torch.Tensor,
    modality: str,
    generator: torch.Generator,
    nuisance_std: float | None = None,
) -> torch.Tensor:
    std = world.nuisance_std if nuisance_std is None else float(nuisance_std)
    if modality == "a":
        W = world.W_a
        d = world.d_mod_a
    elif modality == "b":
        W = world.W_b
        d = world.d_mod_b
    else:
        raise ValueError(modality)
    base = z @ W.T
    nuis = std * torch.randn(z.shape[0], d, generator=generator)
    return base + nuis


def good_view_pair(
    world: World,
    modality: str,
    batch: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two independent observations of the same world state (good views)."""
    z, class_id, _ = _sample_states(world, batch, generator)
    x1 = render(world, z, modality, generator)
    x2 = render(world, z, modality, generator)
    return x1, x2, class_id


def bad_view_pair(
    world: World,
    modality: str,
    batch: int,
    generator: torch.Generator,
    noise_std: float,
    dropout_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two corruptions of ONE observation (bad views / crops analogue)."""
    z, class_id, _ = _sample_states(world, batch, generator)
    x0 = render(world, z, modality, generator, nuisance_std=world.nuisance_std)

    def corrupt(x: torch.Tensor) -> torch.Tensor:
        noise = noise_std * torch.randn(x.shape, generator=generator)
        mask = (torch.rand(x.shape, generator=generator) > dropout_p).float()
        return x * mask + noise

    return corrupt(x0), corrupt(x0), class_id


class MLPEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        hidden: Sequence[int],
        d_out: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, d_out))
        self.net = nn.Sequential(*layers)
        self.d_in = d_in
        self.d_out = d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Autoencoder(nn.Module):
    def __init__(self, encoder: MLPEncoder, d_in: int):
        super().__init__()
        self.encoder = encoder
        hidden = []
        # Mirror last hidden width if present
        for m in encoder.net:
            if isinstance(m, nn.Linear):
                hidden.append(m.out_features)
        # decoder: emb -> reverse hidden (skip final emb width) -> d_in
        widths = list(reversed(hidden[:-1])) if len(hidden) > 1 else [max(32, d_in)]
        layers: list[nn.Module] = []
        prev = encoder.d_out
        for h in widths:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, d_in))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.decoder(z)


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coef: float,
    std_coef: float,
    cov_coef: float,
    std_target: float,
) -> tuple[torch.Tensor, dict]:
    """VICReg-style whitened JEPA with identity predictor."""
    sim = F.mse_loss(z1, z2)

    def _std_loss(z: torch.Tensor) -> torch.Tensor:
        # per-dimension std across batch
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        return torch.mean(F.relu(std_target - std))

    def _cov_loss(z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape
        zc = z - z.mean(dim=0)
        cov = (zc.T @ zc) / max(n - 1, 1)
        off = cov - torch.diag(torch.diag(cov))
        return (off**2).sum() / d

    std = 0.5 * (_std_loss(z1) + _std_loss(z2))
    cov = 0.5 * (_cov_loss(z1) + _cov_loss(z2))
    loss = sim_coef * sim + std_coef * std + cov_coef * cov
    return loss, {"sim": float(sim.item()), "std": float(std.item()), "cov": float(cov.item())}


@dataclass
class TrainResult:
    model: nn.Module
    encoder: MLPEncoder
    final_loss: float
    train_wall_s: float
    n_params: int
    history: list[dict]
    kind: str
    modality: str
    seed: int


def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_jepa(
    world: World,
    modality: str,
    seed: int,
    *,
    hidden: Sequence[int],
    emb_dim: int,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    log_every: int,
    sim_coef: float,
    std_coef: float,
    cov_coef: float,
    std_target: float,
    stream_seed: int,
    view_mode: str = "good",
    bad_noise_std: float = 0.5,
    bad_dropout_p: float = 0.3,
    dropout: float = 0.0,
) -> TrainResult:
    set_seed(seed)
    d_in = world.d_mod_a if modality == "a" else world.d_mod_b
    enc = MLPEncoder(d_in, hidden, emb_dim, dropout=dropout)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(int(stream_seed) + 10007 * seed + (0 if modality == "a" else 17))

    history: list[dict] = []
    t0 = time.perf_counter()
    last_loss = 0.0
    enc.train()
    for step in range(1, steps + 1):
        if view_mode == "good":
            x1, x2, _ = good_view_pair(world, modality, batch_size, gen)
        elif view_mode == "bad":
            x1, x2, _ = bad_view_pair(
                world, modality, batch_size, gen, bad_noise_std, bad_dropout_p
            )
        else:
            raise ValueError(view_mode)
        z1 = enc(x1)
        z2 = enc(x2)
        loss, parts = vicreg_loss(z1, z2, sim_coef, std_coef, cov_coef, std_target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(enc.parameters(), grad_clip)
        opt.step()
        last_loss = float(loss.item())
        if step % log_every == 0 or step == 1 or step == steps:
            history.append({"step": step, "loss": last_loss, **parts})
    wall = time.perf_counter() - t0
    enc.eval()
    return TrainResult(
        model=enc,
        encoder=enc,
        final_loss=last_loss,
        train_wall_s=wall,
        n_params=_count_params(enc),
        history=history,
        kind="jepa",
        modality=modality,
        seed=seed,
    )


def train_recon(
    world: World,
    modality: str,
    seed: int,
    *,
    hidden: Sequence[int],
    emb_dim: int,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    log_every: int,
    stream_seed: int,
    dropout: float = 0.0,
) -> TrainResult:
    set_seed(seed + 333)
    d_in = world.d_mod_a if modality == "a" else world.d_mod_b
    enc = MLPEncoder(d_in, hidden, emb_dim, dropout=dropout)
    ae = Autoencoder(enc, d_in)
    opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(int(stream_seed) + 20011 * seed + (1 if modality == "a" else 19))

    history: list[dict] = []
    t0 = time.perf_counter()
    last_loss = 0.0
    ae.train()
    for step in range(1, steps + 1):
        z, _, _ = _sample_states(world, batch_size, gen)
        x = render(world, z, modality, gen)
        _, x_hat = ae(x)
        loss = F.mse_loss(x_hat, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(ae.parameters(), grad_clip)
        opt.step()
        last_loss = float(loss.item())
        if step % log_every == 0 or step == 1 or step == steps:
            history.append({"step": step, "loss": last_loss})
    wall = time.perf_counter() - t0
    enc.eval()
    return TrainResult(
        model=ae,
        encoder=enc,
        final_loss=last_loss,
        train_wall_s=wall,
        n_params=_count_params(ae),
        history=history,
        kind="recon",
        modality=modality,
        seed=seed,
    )


def init_random_encoder(
    world: World,
    modality: str,
    seed: int,
    hidden: Sequence[int],
    emb_dim: int,
    dropout: float = 0.0,
) -> TrainResult:
    set_seed(seed + 777)
    d_in = world.d_mod_a if modality == "a" else world.d_mod_b
    enc = MLPEncoder(d_in, hidden, emb_dim, dropout=dropout)
    enc.eval()
    return TrainResult(
        model=enc,
        encoder=enc,
        final_loss=float("nan"),
        train_wall_s=0.0,
        n_params=_count_params(enc),
        history=[],
        kind="random",
        modality=modality,
        seed=seed,
    )


@torch.no_grad()
def embed_paired(
    world: World,
    enc_a: MLPEncoder,
    enc_b: MLPEncoder,
    n: int,
    eval_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(int(eval_seed))
    z, class_id, _ = _sample_states(world, n, gen)
    xa = render(world, z, "a", gen)
    xb = render(world, z, "b", gen)
    ea = enc_a(xa)
    eb = enc_b(xb)
    return ea, eb, class_id


def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def orthogonal_procrustes(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (B @ R, R) minimizing ||A - B R||_F over orthogonal R."""
    Ac = _center(A)
    Bc = _center(B)
    M = Bc.T @ Ac
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vh
    return Bc @ R, R


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between rows-as-examples feature matrices (already aligned)."""
    Xc = _center(X)
    Yc = _center(Y)
    hsic_xy = (Xc.T @ Yc).pow(2).sum()
    hsic_xx = (Xc.T @ Xc).pow(2).sum()
    hsic_yy = (Yc.T @ Yc).pow(2).sum()
    denom = torch.sqrt(hsic_xx * hsic_yy).clamp_min(1e-12)
    return float((hsic_xy / denom).item())


def procrustes_cka(A: torch.Tensor, B: torch.Tensor) -> tuple[float, float]:
    """Procrustes-align B to A, then linear CKA; also return whitened RMSE."""
    B_aligned, _ = orthogonal_procrustes(A, B)
    Ac = _center(A)
    # Whitened RMSE: scale-invariant residual
    a_norm = Ac.norm().clamp_min(1e-12)
    b_norm = B_aligned.norm().clamp_min(1e-12)
    rmse = float(((Ac / a_norm) - (B_aligned / b_norm)).pow(2).mean().sqrt().item())
    return linear_cka(Ac, B_aligned), rmse


def mutual_knn_overlap(A: torch.Tensor, B: torch.Tensor, k: int = 10) -> float:
    """Fraction of shared neighbors in top-k NN graphs (after Procrustes)."""
    B_aligned, _ = orthogonal_procrustes(A, B)
    Ac = _center(A)
    Bc = B_aligned

    def knn(X: torch.Tensor) -> torch.Tensor:
        # pairwise squared distances
        xx = (X * X).sum(dim=1, keepdim=True)
        dist = xx + xx.T - 2 * (X @ X.T)
        dist.fill_diagonal_(float("inf"))
        return dist.topk(k, largest=False).indices

    na = knn(Ac)
    nb = knn(Bc)
    n = A.shape[0]
    overlap = 0.0
    for i in range(n):
        sa = set(int(x) for x in na[i].tolist())
        sb = set(int(x) for x in nb[i].tolist())
        overlap += len(sa & sb) / float(k)
    return overlap / float(n)


@torch.no_grad()
def alignment_metrics(
    world: World,
    enc_a: MLPEncoder,
    enc_b: MLPEncoder,
    n_eval: int,
    eval_seed: int,
    knn_k: int = 10,
) -> dict:
    ea, eb, _ = embed_paired(world, enc_a, enc_b, n_eval, eval_seed)
    cka, rmse = procrustes_cka(ea, eb)
    knn = mutual_knn_overlap(ea, eb, k=knn_k)
    return {
        "procrustes_cka": cka,
        "procrustes_rmse": rmse,
        "mutual_knn_overlap": knn,
        "n_eval": n_eval,
    }
