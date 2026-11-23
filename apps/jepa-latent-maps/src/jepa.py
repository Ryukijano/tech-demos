"""Tiny action-conditioned JEPA: VICReg, SVReg, and Slow SVReg.

Clean-room transcription of the report's formulas, not a copy of any
repository. No stop-gradient on the target, matching the report's
no-stop-gradient column for these three objectives.

SVReg (Equation 1):
    R = E_u [max(0, 1 - std(z · u))^2], u uniform on the sphere.
VICReg: element-wise variance hinge and off-diagonal covariance
    (Bardes et al., 2022), applied to every frame in the batch, with no
    term that pulls consecutive latents together.
Slow channels (Equation 2):
    R_slow = E ||b_{t+1} - b_t||^2 / (2 tr Var(b))
    on the last 1/8 of the coordinates (at least one).
Gated residual predictor (Equation 3):
    ẑ_{t+1} = z_t + c ⊙ f(z_t, a_t), c initialized at 0.1.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from metrics import evaluate_latents
from world import (
    N_ACTIONS,
    make_maze,
    parked_block,
    render_path,
    rollout,
    scripted_route,
)

METHODS = ("vicreg", "svreg", "slow_svreg")


class Encoder(nn.Module):
    def __init__(self, grid_size: int, hidden: int, latent_dim: int) -> None:
        super().__init__()
        in_dim = 2 * grid_size * grid_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Predictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden: int,
        *,
        gated: bool,
        gate_init: float,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + N_ACTIONS, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        if gated:
            self.gate: nn.Parameter | None = nn.Parameter(torch.full((latent_dim,), float(gate_init)))
        else:
            self.gate = None

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(action, N_ACTIONS).to(dtype=z.dtype)
        delta = self.mlp(torch.cat([z, one_hot], dim=-1))
        if self.gate is None:
            return delta
        return z + self.gate * delta


def n_slow_channels(latent_dim: int) -> int:
    """Last eighth of the latent, at least one channel. The report uses 1/8."""
    return max(1, int(latent_dim) // 8)


def vicreg_terms(z: torch.Tensor, std_target: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance hinge and covariance penalty. z is [N, D], already the full frame stack."""
    std = torch.sqrt(z.var(dim=0, unbiased=True) + 1e-4)
    var_loss = torch.relu(std_target - std).mean()
    centered = z - z.mean(dim=0)
    n, dim = z.shape
    cov = (centered.T @ centered) / float(n - 1)
    off = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    cov_loss = off / dim
    return var_loss, cov_loss


def svreg_penalty(z: torch.Tensor, n_proj: int, generator: torch.Generator) -> torch.Tensor:
    """Monte Carlo estimate of Equation 1. Population std over the sample."""
    dim = z.shape[1]
    direction = torch.randn(n_proj, dim, generator=generator)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
    projected = z @ direction.T
    std = torch.sqrt(projected.var(dim=0, unbiased=False) + 1e-4)
    return torch.relu(1.0 - std).pow(2).mean()


def slow_penalty(z_t: torch.Tensor, z_tp1: torch.Tensor, n_slow: int) -> torch.Tensor:
    """Equation 2 on the last n_slow coordinates. Scale-invariant in those coordinates."""
    b_t = z_t[:, -n_slow:]
    b_tp1 = z_tp1[:, -n_slow:]
    numerator = (b_tp1 - b_t).pow(2).sum(dim=1).mean()
    pooled = torch.cat([b_t, b_tp1], dim=0)
    trace_var = pooled.var(dim=0, unbiased=False).sum().clamp_min(1e-4)
    return numerator / (2.0 * trace_var)


def penalty_self_check() -> None:
    generator = torch.Generator().manual_seed(0)
    collapsed = svreg_penalty(torch.zeros(256, 4), 128, generator)
    if float(collapsed) < 0.95:
        raise AssertionError(f"collapsed SVReg penalty {float(collapsed)} is not ~1")
    spread = svreg_penalty(torch.randn(8192, 4), 64, torch.Generator().manual_seed(1))
    if float(spread) > 0.05:
        raise AssertionError(f"unit-gaussian SVReg penalty {float(spread)} is not ~0")
    var_loss, _cov = vicreg_terms(torch.zeros(128, 4), 1.0)
    # std = sqrt(0 + 1e-4) = 0.01, so the hinge is 0.99 rather than 1.
    if abs(float(var_loss) - 0.99) > 1e-3:
        raise AssertionError(f"collapsed VICReg variance {float(var_loss)}")
    base = torch.randn(64, 3)
    doubled_t = torch.cat([base, base], dim=0)
    doubled_tp = torch.cat([base + 0.1, base + 0.1], dim=0)
    r1 = float(slow_penalty(doubled_t, doubled_tp, 3))
    r2 = float(slow_penalty(2.0 * doubled_t, 2.0 * doubled_tp, 3))
    if abs(r1 - r2) > 1e-4:
        raise AssertionError(f"slowness is not scale-invariant: {r1} vs {r2}")


def _build(spec: dict[str, Any]) -> tuple[Encoder, Predictor]:
    latent_dim = int(spec["latent_dim"])
    hidden = int(spec["hidden"])
    grid_size = int(spec["grid_size"])
    gated = spec["method"] == "slow_svreg"
    torch.manual_seed(int(spec["seed"]))
    encoder = Encoder(grid_size, hidden, latent_dim)
    predictor = Predictor(
        latent_dim,
        hidden,
        gated=gated,
        gate_init=float(spec["gate_init"]),
    )
    return encoder, predictor


def _regularizer(
    method: str,
    z_t: torch.Tensor,
    z_tp1: torch.Tensor,
    spec: dict[str, Any],
    generator: torch.Generator,
) -> torch.Tensor:
    if method == "vicreg":
        var_loss, cov_loss = vicreg_terms(
            torch.cat([z_t, z_tp1], dim=0),
            float(spec["std_target"]),
        )
        return float(spec["vicreg_weight"]) * float(spec["vicreg_std_coef"]) * var_loss + float(
            spec["vicreg_cov_coef"]
        ) * cov_loss
    if method == "svreg":
        return float(spec["svreg_weight"]) * svreg_penalty(
            torch.cat([z_t, z_tp1], dim=0),
            int(spec["svreg_n_projections"]),
            generator,
        )
    if method == "slow_svreg":
        sliced = svreg_penalty(
            torch.cat([z_t, z_tp1], dim=0),
            int(spec["svreg_n_projections"]),
            generator,
        )
        slow = slow_penalty(z_t, z_tp1, n_slow_channels(int(spec["latent_dim"])))
        return float(spec["svreg_weight"]) * sliced + float(spec["slow_weight"]) * slow
    raise ValueError(f"unknown method {method}")


@torch.no_grad()
def _encode(encoder: Encoder, obs: np.ndarray) -> np.ndarray:
    encoder.eval()
    tensor = torch.from_numpy(np.ascontiguousarray(obs))
    pieces = []
    for start in range(0, tensor.shape[0], 512):
        pieces.append(encoder(tensor[start : start + 512]))
    return torch.cat(pieces, dim=0).numpy()


def train_and_eval(spec: dict[str, Any]) -> dict[str, Any]:
    """Train one objective on the fixed grid trajectories and score held-out metrics.

    Data seeds are part of the spec, so every method and model seed sees the
    same transitions. The model seed changes initialization and minibatch order.
    """
    torch.set_num_threads(1)
    method = str(spec["method"])
    if method not in METHODS:
        raise ValueError(f"unknown method {method}")
    maze = make_maze(int(spec["grid_size"]))
    train = rollout(
        maze,
        length=int(spec["traj_len"]),
        seed=int(spec["train_seed"]),
        noise_std=float(spec["noise_std"]),
        contact_radius=int(spec["contact_radius"]),
    )
    fit = rollout(
        maze,
        length=int(spec["traj_len"]),
        seed=int(spec["fit_seed"]),
        noise_std=float(spec["noise_std"]),
        contact_radius=int(spec["contact_radius"]),
    )
    score = rollout(
        maze,
        length=int(spec["traj_len"]),
        seed=int(spec["score_seed"]),
        noise_std=float(spec["noise_std"]),
        contact_radius=int(spec["contact_radius"]),
    )
    route = scripted_route(maze)
    parked = parked_block(maze, route)
    route_traj = render_path(
        maze,
        route,
        parked,
        seed=int(spec["route_noise_seed"]),
        noise_std=float(spec["noise_std"]),
    )

    encoder, predictor = _build(spec)
    params = list(encoder.parameters()) + list(predictor.parameters())
    n_params = int(sum(p.numel() for p in params))
    optimizer = torch.optim.Adam(
        params,
        lr=float(spec["lr"]),
        weight_decay=float(spec["weight_decay"]),
    )
    obs = torch.from_numpy(np.ascontiguousarray(train["obs"]))
    action = torch.from_numpy(np.ascontiguousarray(train["action"]))
    n_frames = int(obs.shape[0])
    batch_rng = torch.Generator().manual_seed(int(spec["seed"]))
    proj_rng = torch.Generator().manual_seed(int(spec["seed"]) + 10_000)
    steps = int(spec["steps"])
    batch_size = int(spec["batch_size"])
    log_every = int(spec["log_every"])
    history: list[dict[str, float]] = []
    last_pred = float("nan")
    last_reg = float("nan")
    t0 = time.perf_counter()
    encoder.train()
    predictor.train()
    for step in range(1, steps + 1):
        index = torch.randint(0, n_frames - 1, (batch_size,), generator=batch_rng)
        optimizer.zero_grad(set_to_none=True)
        z_t = encoder(obs[index])
        z_tp1 = encoder(obs[index + 1])
        z_hat = predictor(z_t, action[index])
        pred = F.mse_loss(z_hat, z_tp1)
        reg = _regularizer(method, z_t, z_tp1, spec, proj_rng)
        loss = pred + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(spec["grad_clip"]))
        optimizer.step()
        last_pred = float(pred.detach())
        last_reg = float(reg.detach())
        if step % log_every == 0 or step == steps:
            history.append({
                "step": float(step),
                "pred": last_pred,
                "reg": last_reg,
                "total": last_pred + last_reg,
            })
    wall = time.perf_counter() - t0
    latent_fit = _encode(encoder, fit["obs"])
    latent_score = _encode(encoder, score["obs"])
    latent_route = _encode(encoder, route_traj["obs"])
    metrics = evaluate_latents(
        latent_fit,
        fit["agent"],
        fit["block"],
        latent_score,
        score["agent"],
        score["block"],
        latent_route,
        maze,
        ridge_alpha=float(spec["ridge_alpha"]),
        min_chebyshev=int(spec["min_chebyshev"]),
        jag_window=int(spec["jag_window"]),
        n_pairs=int(spec["n_pairs"]),
        pair_seed=int(spec["pair_seed"]),
    )
    metrics.update({
        "pred_loss": last_pred,
        "reg_loss": last_reg,
        "train_wall_s": wall,
        "n_params": float(n_params),
        "latent_dim": float(spec["latent_dim"]),
        "steps": float(steps),
        "svreg_weight": float(spec["svreg_weight"]),
        "vicreg_weight": float(spec["vicreg_weight"]),
        "slow_weight": float(spec["slow_weight"]),
        "method": method,
        "seed": int(spec["seed"]),
        "kind": str(spec["kind"]),
    })
    out: dict[str, Any] = {"metrics": metrics}
    if spec.get("return_history"):
        out["history"] = history
    if spec.get("return_state"):
        out["encoder"] = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
        out["predictor"] = {k: v.detach().cpu() for k, v in predictor.state_dict().items()}
        out["gated"] = predictor.gate is not None
    return out
