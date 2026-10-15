"""Minimal 2D re-implementation of Correction Training for Diffusion Models.

Mirrors Ahn et al. 2026 (arXiv:2610.14136), Sec. 2:

  Training input (Eq. 1, flow matching / linear interpolation):
      z_t = (1 - t) * x + t * eps,   t ~ U[0, 1],  eps ~ N(0, I)

  Algorithm 1 (training loss, joint denoiser + corrector):
      pred    = D(z, t)
      refined = C(stopgrad(pred), t)
      loss    = mse(pred, x) + weight * mse(refined, x)

  Algorithm 2 (sampling with K corrections per denoising step):
      z = randn(shape)
      for t, s in zip(times, times[1:]):
          pred = D(z, t)
          for _ in range(K): pred = C(pred, t)
          v = (z - pred) / t
          z = z + (s - t) * v

Everything is CPU-only: tiny MLPs on a 2D mixture of Gaussians. No GPU, no
checkpoints, no tokens. This is a pedagogical toy, NOT the official ImageNet code
(https://github.com/sunovivid/correction-training).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------- data

def make_ring_data(
    n: int,
    n_modes: int = 8,
    radius: float = 2.0,
    mode_std: float = 0.1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """n samples from a mixture of n_modes isotropic Gaussians on a ring."""
    idx = torch.randint(0, n_modes, (n,), generator=generator)
    ang = 2.0 * math.pi * idx.to(torch.float32) / n_modes
    centers = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=1)
    noise = torch.randn(n, 2, generator=generator)
    return centers + mode_std * noise


def mode_centers(n_modes: int = 8, radius: float = 2.0) -> torch.Tensor:
    ang = 2.0 * math.pi * torch.arange(n_modes, dtype=torch.float32) / n_modes
    return torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=1)


# ---------------------------------------------------------------- model

class TimeMLP(nn.Module):
    """MLP f(x, t) -> clean-sample prediction. Fourier features embed t.

    The same architecture serves as denoiser D and corrector C; both take a
    2D point plus the noise level t and predict the clean sample (x-prediction),
    exactly the signature used in the paper's Algorithms 1-2.
    """

    def __init__(self, hidden: int = 128, depth: int = 3, n_freqs: int = 8):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freqs, dtype=torch.float32) * math.pi)
        in_dim = 2 + 2 * n_freqs
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        t = t.reshape(-1, 1)
        ft = t * self.freqs.reshape(1, -1)
        emb = torch.cat([torch.sin(ft), torch.cos(ft)], dim=1)
        return self.net(torch.cat([x, emb], dim=1))


# ---------------------------------------------------------------- training

@dataclass
class TrainResult:
    denoiser: TimeMLP
    corrector: TimeMLP
    wall_s: float
    final_denoise_loss: float
    final_corr_loss: float


def train_joint(
    seed: int,
    *,
    n_modes: int = 8,
    radius: float = 2.0,
    mode_std: float = 0.1,
    hidden: int = 128,
    depth: int = 3,
    iters: int = 3000,
    batch: int = 256,
    lr: float = 1e-3,
    corr_weight: float = 1.0,
) -> TrainResult:
    """Algorithm 1: jointly train denoiser D and stop-gradient corrector C."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    D = TimeMLP(hidden, depth)
    C = TimeMLP(hidden, depth)
    opt = torch.optim.Adam(list(D.parameters()) + list(C.parameters()), lr=lr)
    mse = nn.MSELoss()

    t0 = time.perf_counter()
    dl = cl = float("nan")
    for _ in range(iters):
        x = make_ring_data(batch, n_modes, radius, mode_std, generator=gen)
        t = torch.rand(batch, generator=gen)
        e = torch.randn(batch, 2, generator=gen)
        z = (1.0 - t.unsqueeze(1)) * x + t.unsqueeze(1) * e

        pred = D(z, t)
        refined = C(pred.detach(), t)  # stop-gradient: corr loss never trains D
        loss_d = mse(pred, x)
        loss_c = mse(refined, x)
        loss = loss_d + corr_weight * loss_c

        opt.zero_grad()
        loss.backward()
        opt.step()
        dl, cl = float(loss_d), float(loss_c)
    wall = time.perf_counter() - t0
    D.eval()
    C.eval()
    return TrainResult(D, C, wall, dl, cl)


# ---------------------------------------------------------------- sampling

@torch.no_grad()
def sample(
    D: TimeMLP,
    C: TimeMLP,
    n: int,
    *,
    steps: int,
    K: int,
    generator: torch.Generator | None = None,
    record: bool = False,
) -> tuple[torch.Tensor, int, list[torch.Tensor]]:
    """Algorithm 2. Returns (samples, nfe, trajectory[z after each step])."""
    z = torch.randn(n, 2, generator=generator)
    times = torch.linspace(1.0, 0.0, steps + 1)
    traj: list[torch.Tensor] = [z.clone()] if record else []
    nfe = 0
    for i in range(steps):
        t, s = times[i], times[i + 1]
        pred = D(z, t)
        nfe += 1
        for _ in range(K):
            pred = C(pred, t)
            nfe += 1
        v = (z - pred) / t
        z = z + (s - t) * v
        if record:
            traj.append(z.clone())
    return z, nfe, traj


def steps_for_budget(nfe_budget: int, K: int) -> int:
    """Matched-NFE rule (pre-registered): floor(budget / (1+K)), at least 1 step."""
    return max(1, nfe_budget // (1 + K))


# ---------------------------------------------------------------- metrics

def mmd2_unbiased(x: torch.Tensor, y: torch.Tensor, gamma: float) -> float:
    """Unbiased MMD^2 estimate with RBF kernel k(a,b)=exp(-gamma*||a-b||^2)."""
    def pdist2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cdist(a, b) ** 2

    n, m = x.shape[0], y.shape[0]
    kxx = torch.exp(-gamma * pdist2(x, x))
    kyy = torch.exp(-gamma * pdist2(y, y))
    kxy = torch.exp(-gamma * pdist2(x, y))
    sum_xx = (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
    sum_yy = (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
    sum_xy = kxy.mean()
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def median_heuristic_gamma(data: torch.Tensor) -> float:
    """gamma = 1 / (2 * median of squared pairwise distances) on the eval data."""
    d2 = (torch.cdist(data, data) ** 2).flatten()
    d2 = d2[d2 > 0]
    med = float(d2.median())
    return 1.0 / (2.0 * med)


def sliced_w1(x: torch.Tensor, y: torch.Tensor, n_proj: int = 128, seed: int = 0) -> float:
    """Sliced Wasserstein-1 with fixed seeded projections (report-only)."""
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(n_proj, 2, generator=g)
    theta = theta / theta.norm(dim=1, keepdim=True)
    px = (x @ theta.T).sort(dim=0).values
    py = (y @ theta.T).sort(dim=0).values
    n = min(px.shape[0], py.shape[0])
    return float((px[:n] - py[:n]).abs().mean())


def nearest_mode_stats(
    samples: torch.Tensor, n_modes: int, radius: float
) -> tuple[float, float]:
    """(mean squared distance to nearest mode centre, mode coverage).

    Coverage = fraction of modes that receive >= 1% of samples (report-only).
    """
    centers = mode_centers(n_modes, radius)
    d2 = torch.cdist(samples, centers) ** 2
    mins, idx = d2.min(dim=1)
    counts = torch.bincount(idx, minlength=n_modes).to(torch.float32)
    coverage = float((counts / samples.shape[0] >= 0.01).to(torch.float32).mean())
    return float(mins.mean()), coverage


def set_threads(n: int) -> None:
    torch.set_num_threads(n)
    np.random.seed(0)  # numpy only used for plotting jitter; determinism anchor
