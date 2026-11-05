"""Velocity and quantile critics for the MRP toy.

Value Forcing implements the practical flow-Bellman loss (paper §4.2): the
reward is scaled by (1-γ), and the Gaussian correction sqrt(1-γ²) is added at
every noise level. The endpoint baseline is FlowCritic-style conditional flow
matching onto a single τ=1 Bellman sample (paper appendix Eq. 25). QR-DQN is
pairwise Huber quantile regression (Dabney et al., 2018) with the same reward
scaling. Generated samples are divided by (1-γ) before they are compared to
the unscaled closed-form return.
"""

from __future__ import annotations

import math
import time
from typing import Literal, NoReturn

import numpy as np
import torch
import torch.nn as nn

from mrp import MRP, w1_to_atoms

Method = Literal["vf", "endpoint", "qr"]


def _never(value: Method) -> NoReturn:
    raise AssertionError(f"unhandled method {value!r}")


def set_threads(n: int) -> None:
    torch.set_num_threads(max(1, int(n)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


class VelocityNet(nn.Module):
    """Scalar velocity v(z | s, τ)."""

    def __init__(self, n_states: int, embed_dim: int, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_states, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.embed.weight, std=0.05)
        last = self.mlp[-1]
        assert isinstance(last, nn.Linear)
        nn.init.xavier_uniform_(last.weight, gain=0.1)
        nn.init.zeros_(last.bias)

    def forward(self, z: torch.Tensor, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([z[:, None], tau[:, None], self.embed(state)], dim=-1)
        return self.mlp(feat).squeeze(-1)


class QuantileNet(nn.Module):
    """QR-DQN quantile heads. Output is in scaled-return units."""

    def __init__(self, n_states: int, embed_dim: int, hidden: int, n_quantiles: int) -> None:
        super().__init__()
        self.n_quantiles = n_quantiles
        self.embed = nn.Embedding(n_states, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_quantiles),
        )
        nn.init.normal_(self.embed.weight, std=0.05)
        last = self.mlp[-1]
        assert isinstance(last, nn.Linear)
        nn.init.xavier_uniform_(last.weight, gain=1.0)
        with torch.no_grad():
            last.bias.copy_(torch.linspace(-0.5, 0.5, n_quantiles))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(state))


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def correction(gamma: float) -> float:
    return math.sqrt(max(0.0, 1.0 - gamma * gamma))


def reward_multiplier(gamma: float, enabled: bool) -> float:
    return (1.0 - gamma) if enabled else 1.0


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, decay: float) -> None:
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(decay).add_(po, alpha=1.0 - decay)


def integrate(net: VelocityNet, z: torch.Tensor, state: torch.Tensor, tau: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Explicit Euler from time 0 to per-sample time ``tau``."""
    steps = max(1, int(n_steps))
    dt = tau / steps
    for i in range(steps):
        t = tau * (i / steps)
        z = z + dt * net(z, state, t)
    return z


def _batch(s: torch.Tensor, r: torch.Tensor, sp: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, s.shape[0], (batch_size,))
    return s[idx], r[idx], sp[idx]


def vf_loss(
    net: VelocityNet,
    target: VelocityNet,
    s: torch.Tensor,
    r: torch.Tensor,
    sp: torch.Tensor,
    gamma: float,
    n_euler: int,
    scale: float,
) -> torch.Tensor:
    """Flow Bellman regression at a random noise level (practical, reward-scaled)."""
    batch = s.shape[0]
    tau = torch.rand(batch)
    eps = torch.randn(batch)
    z0 = torch.randn(batch)
    corr = correction(gamma)
    with torch.no_grad():
        z_tau = integrate(target, z0, sp, tau, n_euler)
        v_next = target(z_tau, sp, tau)
        r_s = scale * r
        z_s = tau * r_s + gamma * z_tau + (1.0 - tau) * corr * eps
        velocity = r_s + gamma * v_next - corr * eps
    pred = net(z_s, s, tau)
    return torch.mean((pred - velocity) ** 2)


def endpoint_loss(
    net: VelocityNet,
    target: VelocityNet,
    s: torch.Tensor,
    r: torch.Tensor,
    sp: torch.Tensor,
    gamma: float,
    n_euler: int,
    scale: float,
) -> torch.Tensor:
    """Conditional flow matching onto one clean Bellman sample. No intermediate Bellman constraint."""
    batch = s.shape[0]
    with torch.no_grad():
        z_next = integrate(target, torch.randn(batch), sp, torch.ones(batch), n_euler)
        y = scale * r + gamma * z_next
    z0 = torch.randn(batch)
    tau = torch.rand(batch)
    z_tau = (1.0 - tau) * z0 + tau * y
    pred = net(z_tau, s, tau)
    return torch.mean((pred - (y - z0)) ** 2)


def qr_loss(
    net: QuantileNet,
    target: QuantileNet,
    s: torch.Tensor,
    r: torch.Tensor,
    sp: torch.Tensor,
    gamma: float,
    taus: torch.Tensor,
    kappa: float,
    scale: float,
) -> torch.Tensor:
    """Pairwise quantile Huber loss. ``taus`` has shape [N]."""
    with torch.no_grad():
        y = scale * r[:, None] + gamma * target(sp)
    pred = net(s)
    diff = y[:, None, :] - pred[:, :, None]
    abs_diff = diff.abs()
    huber = torch.where(abs_diff <= kappa, 0.5 * diff * diff, kappa * (abs_diff - 0.5 * kappa))
    weight = (taus.view(1, -1, 1) - (diff < 0).to(diff.dtype)).abs()
    return torch.mean(weight * huber)


def _optim(module: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)


def train_method(
    method: Method,
    mrp: MRP,
    data: dict[str, np.ndarray],
    *,
    seed: int,
    hidden: int,
    embed_dim: int,
    n_quantiles: int,
    huber_kappa: float,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    ema: float,
    n_euler: int,
    reward_scale: bool,
    log_every: int,
) -> dict:
    torch.manual_seed(seed)
    scale = reward_multiplier(mrp.gamma, reward_scale)
    s = torch.from_numpy(np.ascontiguousarray(data["s"]))
    r = torch.from_numpy(np.ascontiguousarray(data["r"])).float()
    sp = torch.from_numpy(np.ascontiguousarray(data["sp"]))
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()

    if method == "vf" or method == "endpoint":
        net = VelocityNet(mrp.n_states, embed_dim, hidden)
        tgt = VelocityNet(mrp.n_states, embed_dim, hidden)
        tgt.load_state_dict(net.state_dict())
        opt = _optim(net, lr, weight_decay)
        params = n_params(net)
        net.train()
        tgt.eval()
        for step in range(1, steps + 1):
            bs, br, bsp = _batch(s, r, sp, batch_size)
            if method == "vf":
                loss = vf_loss(net, tgt, bs, br, bsp, mrp.gamma, n_euler, scale)
            else:
                loss = endpoint_loss(net, tgt, bs, br, bsp, mrp.gamma, n_euler, scale)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{method} seed {seed} non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()
            ema_update(tgt, net, ema)
            if step == 1 or step % log_every == 0 or step == steps:
                history.append({"step": step, "loss": float(loss.detach())})
        module: nn.Module = net
    elif method == "qr":
        net_q = QuantileNet(mrp.n_states, embed_dim, hidden, n_quantiles)
        tgt_q = QuantileNet(mrp.n_states, embed_dim, hidden, n_quantiles)
        tgt_q.load_state_dict(net_q.state_dict())
        opt = _optim(net_q, lr, weight_decay)
        params = n_params(net_q)
        taus = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles
        net_q.train()
        tgt_q.eval()
        for step in range(1, steps + 1):
            bs, br, bsp = _batch(s, r, sp, batch_size)
            loss = qr_loss(net_q, tgt_q, bs, br, bsp, mrp.gamma, taus, huber_kappa, scale)
            if not torch.isfinite(loss):
                raise RuntimeError(f"qr seed {seed} non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net_q.parameters(), grad_clip)
            opt.step()
            ema_update(tgt_q, net_q, ema)
            if step == 1 or step % log_every == 0 or step == steps:
                history.append({"step": step, "loss": float(loss.detach())})
        module = net_q
    else:
        _never(method)

    wall = time.perf_counter() - t0
    return {
        "module": module,
        "n_params": params,
        "train_wall_s": wall,
        "history": history,
        "final_loss": history[-1]["loss"] if history else float("nan"),
        "scale": scale,
    }


@torch.no_grad()
def evaluate(
    method: Method,
    module: nn.Module,
    mrp: MRP,
    *,
    n_samples: int,
    n_euler: int,
    reward_scale: bool,
    n_quantiles: int,
) -> dict:
    module.eval()
    scale = reward_multiplier(mrp.gamma, reward_scale)
    unscale = 1.0 / scale
    per_state: list[dict] = []
    samples: list[np.ndarray] = []

    if method in ("vf", "endpoint"):
        assert isinstance(module, VelocityNet)
        for s_id in range(mrp.n_states):
            z0 = torch.randn(n_samples)
            state = torch.full((n_samples,), s_id, dtype=torch.long)
            tau = torch.ones(n_samples)
            drawn = integrate(module, z0, state, tau, n_euler).numpy() * unscale
            samples.append(drawn.astype(np.float64))
    elif method == "qr":
        assert isinstance(module, QuantileNet)
        taus = (np.arange(n_quantiles, dtype=np.float64) + 0.5) / n_quantiles
        us = (np.arange(n_samples, dtype=np.float64) + 0.5) / n_samples
        for s_id in range(mrp.n_states):
            state = torch.tensor([s_id], dtype=torch.long)
            q = module(state).numpy().reshape(-1).astype(np.float64) * unscale
            # Monotone rearrangement: the induced law is the sorted quantile estimates
            # placed on the fixed grid (i+0.5)/N.
            q_sorted = np.sort(q, kind="mergesort")
            drawn = np.interp(us, taus, q_sorted)
            samples.append(drawn)
    else:
        _never(method)

    w1s: list[float] = []
    for s_id, drawn in enumerate(samples):
        w1 = w1_to_atoms(drawn, mrp.values[s_id], mrp.probs[s_id])
        pred_std = float(np.std(drawn))
        true_std = float(mrp.std[s_id])
        ratio = pred_std / true_std if true_std > 0 else float("nan")
        w1s.append(w1)
        per_state.append(
            {
                "state": mrp.names[s_id],
                "w1": w1,
                "pred_mean": float(np.mean(drawn)),
                "true_mean": float(mrp.mean[s_id]),
                "pred_std": pred_std,
                "true_std": true_std,
                "std_ratio": ratio,
            }
        )
    return {
        "per_state": per_state,
        "mean_w1": float(np.mean(w1s)),
        "samples": samples,
    }


def std_ratio(per_state: list[dict], std_floor: float) -> float:
    ratios = [float(row["std_ratio"]) for row in per_state if float(row["true_std"]) >= std_floor]
    if not ratios:
        raise ValueError("no state cleared std_floor; collapse metric is undefined")
    return float(np.mean(ratios))


def flow_snapshot(net: VelocityNet, n_samples: int, state: int, tau: float, n_euler: int, unscale: float) -> np.ndarray:
    """Push noise to time ``tau`` and map into unscaled return units.

    Unscaling is only meaningful at τ=1 (the network lives in scaled units at
    every τ). Callers that animate intermediate τ should pass unscale=1 and
    label the axis as scaled return, or pass the true unscale only at τ=1.
    """
    net.eval()
    with torch.no_grad():
        z0 = torch.randn(n_samples)
        st = torch.full((n_samples,), state, dtype=torch.long)
        t = torch.full((n_samples,), float(tau))
        z = integrate(net, z0, st, t, n_euler)
        return (z.numpy() * unscale).astype(np.float64)
