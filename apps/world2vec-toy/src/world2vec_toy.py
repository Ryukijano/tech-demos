"""Clean-room world2vec toy: continuous-time latent dynamics from pixels+actions.

No reward. No pixel reconstruction loss. Multi-step latent matching + VICReg-style
anti-collapse. Independent of dynamical-inference/world2vec (code not released).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Kind = Literal["world2vec", "discrete", "recon"]


def set_threads(n: int) -> None:
    torch.set_num_threads(max(1, int(n)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# Pendulum environment (CPU numpy render)
# ---------------------------------------------------------------------------


@dataclass
class PendulumConfig:
    img_size: int = 24
    dt: float = 0.05
    gravity: float = 9.81
    length: float = 1.0
    mass: float = 1.0
    damping: float = 0.15
    action_scale: float = 2.0


def _wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
    return (theta + math.pi) % (2 * math.pi) - math.pi


def pendulum_step(
    theta: float,
    omega: float,
    action: float,
    cfg: PendulumConfig,
) -> tuple[float, float]:
    """Semi-implicit Euler for damped pendulum with torque."""
    g, L, m, b, dt = cfg.gravity, cfg.length, cfg.mass, cfg.damping, cfg.dt
    torque = float(np.clip(action, -cfg.action_scale, cfg.action_scale))
    alpha = -(g / L) * math.sin(theta) - (b / (m * L * L)) * omega + torque / (m * L * L)
    omega_n = omega + dt * alpha
    theta_n = _wrap_angle(theta + dt * omega_n)
    return float(theta_n), float(omega_n)


def render_pendulum(theta: float, size: int) -> np.ndarray:
    """Grayscale HxW float32 in [0, 1]: rod + bob on dark field."""
    img = np.zeros((size, size), dtype=np.float32)
    cx = cy = (size - 1) / 2.0
    length_px = 0.42 * size
    # θ=0 hangs down (standard), screen y grows downward
    x1 = cx + length_px * math.sin(theta)
    y1 = cy + length_px * math.cos(theta)
    n = max(size * 2, 16)
    xs = np.linspace(cx, x1, n)
    ys = np.linspace(cy, y1, n)
    for x, y in zip(xs, ys):
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < size and 0 <= iy < size:
            img[iy, ix] = 1.0
            if ix + 1 < size:
                img[iy, ix + 1] = max(img[iy, ix + 1], 0.6)
            if iy + 1 < size:
                img[iy + 1, ix] = max(img[iy + 1, ix], 0.6)
    # bob
    br = max(1, size // 16)
    for dy in range(-br, br + 1):
        for dx in range(-br, br + 1):
            if dx * dx + dy * dy <= br * br + 1:
                ix, iy = int(round(x1 + dx)), int(round(y1 + dy))
                if 0 <= ix < size and 0 <= iy < size:
                    img[iy, ix] = 1.0
    # pivot
    pr = max(1, size // 24)
    for dy in range(-pr, pr + 1):
        for dx in range(-pr, pr + 1):
            if dx * dx + dy * dy <= pr * pr:
                ix, iy = int(round(cx + dx)), int(round(cy + dy))
                if 0 <= ix < size and 0 <= iy < size:
                    img[iy, ix] = 1.0
    return img


def _ou_actions(n: int, scale: float, rng: np.random.Generator, theta: float = 0.15) -> np.ndarray:
    a = np.zeros(n, dtype=np.float32)
    x = 0.0
    for t in range(n):
        x = (1 - theta) * x + math.sqrt(2 * theta) * rng.normal()
        a[t] = float(np.clip(x * scale * 0.7, -scale, scale))
    return a


@dataclass
class TrajectoryBatch:
    frames: np.ndarray       # (N, T, H, W) float32
    actions: np.ndarray      # (N, T-1) float32
    states: np.ndarray       # (N, T, 3) = [sinθ, cosθ, ω]
    thetas: np.ndarray       # (N, T)
    omegas: np.ndarray       # (N, T)


def collect_trajectories(
    n_traj: int,
    traj_len: int,
    cfg: PendulumConfig,
    seed: int,
    noise_std: float = 0.0,
) -> TrajectoryBatch:
    rng = np.random.default_rng(seed)
    H = cfg.img_size
    frames = np.zeros((n_traj, traj_len, H, H), dtype=np.float32)
    actions = np.zeros((n_traj, traj_len - 1), dtype=np.float32)
    states = np.zeros((n_traj, traj_len, 3), dtype=np.float32)
    thetas = np.zeros((n_traj, traj_len), dtype=np.float32)
    omegas = np.zeros((n_traj, traj_len), dtype=np.float32)

    for i in range(n_traj):
        theta = float(rng.uniform(-math.pi, math.pi))
        omega = float(rng.uniform(-2.5, 2.5))
        acts = _ou_actions(traj_len - 1, cfg.action_scale, rng)
        actions[i] = acts
        for t in range(traj_len):
            img = render_pendulum(theta, H)
            if noise_std > 0:
                img = np.clip(img + rng.normal(0, noise_std, img.shape).astype(np.float32), 0, 1)
            frames[i, t] = img
            states[i, t] = (math.sin(theta), math.cos(theta), omega)
            thetas[i, t] = theta
            omegas[i, t] = omega
            if t < traj_len - 1:
                theta, omega = pendulum_step(theta, omega, float(acts[t]), cfg)
    return TrajectoryBatch(frames, actions, states, thetas, omegas)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FrameEncoder(nn.Module):
    """CNN over H stacked grayscale frames → latent z."""

    def __init__(
        self,
        history: int,
        img_size: int,
        latent_dim: int,
        channels: list[int],
        mlp_hidden: list[int],
    ):
        super().__init__()
        self.history = history
        layers: list[nn.Module] = []
        in_c = history
        s = img_size
        for c in channels:
            layers += [
                nn.Conv2d(in_c, c, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_c = c
            s = (s + 1) // 2
        self.cnn = nn.Sequential(*layers)
        flat = in_c * s * s
        mlp: list[nn.Module] = []
        d = flat
        for h in mlp_hidden:
            mlp += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            d = h
        mlp.append(nn.Linear(d, latent_dim))
        self.head = nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, S, S) or (B, H*S*S) — expect (B, history, S, S)
        return self.head(self.cnn(x).flatten(1))


class ContinuousDynamics(nn.Module):
    """ż = f(z, a); Euler Φ(z,a) = z + dt * f(z,a)."""

    def __init__(self, latent_dim: int, hidden: list[int], dt: float):
        super().__init__()
        self.dt = float(dt)
        layers: list[nn.Module] = []
        d = latent_dim + 1
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, latent_dim))
        self.net = nn.Sequential(*layers)

    def field(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        return self.net(torch.cat([z, a], dim=-1))

    def step(self, z: torch.Tensor, a: torch.Tensor, dt: float | None = None) -> torch.Tensor:
        h = self.dt if dt is None else float(dt)
        return z + h * self.field(z, a)


class DiscreteDynamics(nn.Module):
    """z' = MLP(z, a) — discrete next-latent baseline (paper §3.1)."""

    def __init__(self, latent_dim: int, hidden: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        d = latent_dim + 1
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, latent_dim))
        self.net = nn.Sequential(*layers)

    def step(self, z: torch.Tensor, a: torch.Tensor, dt: float | None = None) -> torch.Tensor:
        del dt
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        return self.net(torch.cat([z, a], dim=-1))


class PixelDecoder(nn.Module):
    """Tiny decoder for reconstruction baseline only."""

    def __init__(self, latent_dim: int, history: int, img_size: int):
        super().__init__()
        self.history = history
        self.img_size = img_size
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, history * img_size * img_size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).view(-1, self.history, self.img_size, self.img_size)


class WorldModel(nn.Module):
    def __init__(
        self,
        kind: Kind,
        history: int,
        img_size: int,
        latent_dim: int,
        cnn_channels: list[int],
        dyn_hidden: list[int],
        encoder_mlp_hidden: list[int],
        dt: float,
    ):
        super().__init__()
        self.kind = kind
        self.history = history
        self.dt = dt
        self.encoder = FrameEncoder(
            history, img_size, latent_dim, cnn_channels, encoder_mlp_hidden
        )
        if kind == "world2vec":
            self.dynamics: nn.Module = ContinuousDynamics(latent_dim, dyn_hidden, dt)
        else:
            self.dynamics = DiscreteDynamics(latent_dim, dyn_hidden)
        self.decoder = (
            PixelDecoder(latent_dim, history, img_size) if kind == "recon" else None
        )

    def encode_window(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, S, S); take last `history` frames ending at index t handled by caller
        return self.encoder(frames)

    def rollout(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,
        dt: float | None = None,
    ) -> torch.Tensor:
        """actions: (B, K); returns zhats (B, K, D) after each step."""
        zs = []
        z = z0
        k = actions.shape[1]
        for j in range(k):
            z = self.dynamics.step(z, actions[:, j], dt=dt)
            zs.append(z)
        return torch.stack(zs, dim=1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Losses / metrics
# ---------------------------------------------------------------------------


def vicreg_reg(z: torch.Tensor, std_coef: float, cov_coef: float, std_target: float) -> torch.Tensor:
    # z: (B, D)
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    std_loss = F.relu(std_target - std).mean()
    if z.shape[0] < 2:
        return std_coef * std_loss
    cov = (z.T @ z) / (z.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off ** 2).sum() / z.shape[1]
    return std_coef * std_loss + cov_coef * cov_loss


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Multi-output mean R² (uniform average over dims)."""
    yt = y_true.reshape(y_true.shape[0], -1)
    yp = y_pred.reshape(y_pred.shape[0], -1)
    scores = []
    for i in range(yt.shape[1]):
        ss_res = float(np.sum((yt[:, i] - yp[:, i]) ** 2))
        ss_tot = float(np.sum((yt[:, i] - yt[:, i].mean()) ** 2))
        if ss_tot < 1e-12:
            scores.append(0.0)
        else:
            scores.append(1.0 - ss_res / ss_tot)
    return float(np.mean(scores))


def linear_probe_r2(z: np.ndarray, y: np.ndarray) -> float:
    """Ridge-ish least squares via lstsq; R² on same data (in-sample probe)."""
    zc = z - z.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    # augment with bias already handled by centering; use ridge for stability
    d = zc.shape[1]
    reg = 1e-3 * np.eye(d, dtype=np.float64)
    a = zc.astype(np.float64)
    b = yc.astype(np.float64)
    w, _, _, _ = np.linalg.lstsq(a.T @ a + reg, a.T @ b, rcond=None)
    pred = a @ w
    return r2_score(yc, pred)


def lrc_r2(z_true: np.ndarray, z_pred: np.ndarray) -> float:
    return r2_score(z_true, z_pred)


# ---------------------------------------------------------------------------
# Training / eval
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    kind: str
    seed: int
    model: WorldModel
    final_loss: float
    train_wall_s: float
    n_params: int
    history: list[dict] = field(default_factory=list)


def _history_windows(
    frames: torch.Tensor,
    t: torch.Tensor,
    history: int,
) -> torch.Tensor:
    """frames (B,T,S,S), t (B,) end indices → (B, history, S, S)."""
    b = frames.shape[0]
    outs = []
    for i in range(history):
        idx = (t - (history - 1 - i)).clamp(min=0)
        gather = frames[torch.arange(b), idx]
        outs.append(gather)
    return torch.stack(outs, dim=1)


def build_model(kind: Kind, cfg: dict) -> WorldModel:
    d = cfg["data"]
    m = cfg["model"]
    return WorldModel(
        kind=kind,
        history=int(d["history"]),
        img_size=int(d["img_size"]),
        latent_dim=int(m["latent_dim"]),
        cnn_channels=list(m["cnn_channels"]),
        dyn_hidden=list(m["dyn_hidden"]),
        encoder_mlp_hidden=list(m["encoder_mlp_hidden"]),
        dt=float(d["dt"]),
    )


def train_model(
    kind: Kind,
    seed: int,
    cfg: dict,
    train_data: TrajectoryBatch,
    *,
    steps_override: int | None = None,
    noise_std: float = 0.0,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(kind, cfg)
    model.train()
    tcfg = cfg["training"]
    dcfg = cfg["data"]
    steps = int(tcfg["steps"] if steps_override is None else steps_override)
    bs = int(tcfg["batch_size"])
    k = int(tcfg["rollout_k"])
    history = int(dcfg["history"])
    traj_len = train_data.frames.shape[1]
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )

    frames = torch.from_numpy(train_data.frames)
    actions = torch.from_numpy(train_data.actions)
    n_traj = frames.shape[0]
    t0 = time.perf_counter()
    hist: list[dict] = []
    last_loss = 0.0

    for step in range(1, steps + 1):
        bi = torch.randint(0, n_traj, (bs,))
        # start index must leave room for history-1 past and k future steps
        t_max = traj_len - 1 - k
        t_min = history - 1
        t = torch.randint(t_min, t_max + 1, (bs,))
        x0 = _history_windows(frames[bi], t, history)
        if noise_std > 0:
            x0 = (x0 + noise_std * torch.randn_like(x0)).clamp(0, 1)

        z0 = model.encoder(x0)
        # gather actions a_t ... a_{t+k-1}
        act = torch.stack([actions[bi, t + j] for j in range(k)], dim=1)
        zhat = model.rollout(z0, act)  # (B,K,D)

        # targets: encoded futures (stop-grad for JEPA-ish stability)
        targets = []
        for j in range(1, k + 1):
            xj = _history_windows(frames[bi], t + j, history)
            if noise_std > 0:
                xj = (xj + noise_std * torch.randn_like(xj)).clamp(0, 1)
            with torch.no_grad():
                targets.append(model.encoder(xj))
        z_tgt = torch.stack(targets, dim=1)

        pred_loss = F.mse_loss(zhat, z_tgt)
        # anti-collapse on online encodings of x0 (and optionally mix)
        reg = vicreg_reg(
            z0,
            float(tcfg["std_coef"]),
            float(tcfg["cov_coef"]),
            float(tcfg["std_target"]),
        )
        loss = float(tcfg["pred_coef"]) * pred_loss + reg

        if kind == "recon" and model.decoder is not None:
            recon = model.decoder(z0)
            loss = loss + F.mse_loss(recon, x0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
        opt.step()
        last_loss = float(loss.detach())
        if step % int(tcfg["log_every"]) == 0 or step == 1:
            hist.append({"step": step, "loss": last_loss, "pred": float(pred_loss.detach())})

    wall = time.perf_counter() - t0
    model.eval()
    return TrainResult(
        kind=kind,
        seed=seed,
        model=model,
        final_loss=last_loss,
        train_wall_s=wall,
        n_params=count_params(model),
        history=hist,
    )


@torch.no_grad()
def evaluate_model(
    model: WorldModel,
    data: TrajectoryBatch,
    *,
    horizon: int,
    history: int,
    noise_std: float = 0.0,
    dt_scale: float = 1.0,
    max_starts: int = 512,
) -> dict:
    model.eval()
    frames = torch.from_numpy(data.frames)
    actions = torch.from_numpy(data.actions)
    states = data.states
    n_traj, traj_len = frames.shape[0], frames.shape[1]
    k = int(horizon)
    t_min = history - 1
    t_max = traj_len - 1 - k
    if t_max < t_min:
        raise ValueError(f"traj_len={traj_len} too short for history={history}, H={k}")

    # sample starts
    starts = []
    for i in range(n_traj):
        for t in range(t_min, t_max + 1, 2):
            starts.append((i, t))
    rng = np.random.default_rng(0)
    if len(starts) > max_starts:
        pick = rng.choice(len(starts), size=max_starts, replace=False)
        starts = [starts[j] for j in pick]

    bi = torch.tensor([s[0] for s in starts], dtype=torch.long)
    t = torch.tensor([s[1] for s in starts], dtype=torch.long)
    x0 = _history_windows(frames[bi], t, history)
    if noise_std > 0:
        x0 = (x0 + noise_std * torch.randn_like(x0)).clamp(0, 1)
    z0 = model.encoder(x0)

    # encode all needed futures for LRC
    z_true_h = []
    for j in (1, 4, k):
        jj = min(j, k)
        xj = _history_windows(frames[bi], t + jj, history)
        if noise_std > 0:
            xj = (xj + noise_std * torch.randn_like(xj)).clamp(0, 1)
        z_true_h.append(model.encoder(xj).cpu().numpy())

    act = torch.stack([actions[bi, t + j] for j in range(k)], dim=1)
    dt = model.dt * float(dt_scale)
    # For discrete models, dt_scale != 1 means we still take `k` discrete steps
    # but compare to encoded state at round(k * dt_scale) — handled by caller for
    # irregular breakpoint; here we always roll k steps with optional dt for continuous.
    zhat = model.rollout(z0, act, dt=dt if model.kind == "world2vec" else None)
    zhat_np = zhat.cpu().numpy()
    z0_np = z0.cpu().numpy()

    # state labels at t and t+k
    st0 = states[bi.numpy(), t.numpy()]
    stk = states[bi.numpy(), (t + k).numpy()]

    probe = linear_probe_r2(z0_np, st0)
    # fit probe on z0→state, apply to zhat at horizon
    zc = z0_np - z0_np.mean(axis=0, keepdims=True)
    yc = st0 - st0.mean(axis=0, keepdims=True)
    d = zc.shape[1]
    w, _, _, _ = np.linalg.lstsq(
        zc.astype(np.float64).T @ zc.astype(np.float64) + 1e-3 * np.eye(d),
        zc.astype(np.float64).T @ yc.astype(np.float64),
        rcond=None,
    )
    zhk = zhat_np[:, -1] - z0_np.mean(axis=0, keepdims=True)
    pred_state = zhk @ w + st0.mean(axis=0, keepdims=True)
    horizon_state_r2 = r2_score(stk, pred_state)

    lrc_h = lrc_r2(z_true_h[-1], zhat_np[:, -1])
    # also H=1 and H=4 if k>=4
    lrc_h1 = lrc_r2(z_true_h[0], zhat_np[:, 0])
    lrc_h4 = lrc_r2(z_true_h[1], zhat_np[:, min(3, k - 1)]) if k >= 4 else lrc_h1

    return {
        "lrc_h": float(lrc_h),
        "lrc_h1": float(lrc_h1),
        "lrc_h4": float(lrc_h4),
        "probe_r2": float(probe),
        "horizon_state_r2": float(horizon_state_r2),
        "n_eval": int(len(starts)),
        "horizon": int(k),
    }


def pendulum_cfg_from_data(d: dict) -> PendulumConfig:
    return PendulumConfig(
        img_size=int(d["img_size"]),
        dt=float(d["dt"]),
        gravity=float(d["gravity"]),
        length=float(d["length"]),
        mass=float(d["mass"]),
        damping=float(d["damping"]),
        action_scale=float(d["action_scale"]),
    )
