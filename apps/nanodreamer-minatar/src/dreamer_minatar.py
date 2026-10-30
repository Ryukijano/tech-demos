"""Clean-room tiny Dreamer-style agent for MinAtar (PyTorch, CPU).

Inspired by Benjamin Bai's NanoDreamer write-up. No upstream source is copied.
Actor updates use imagined rollouts only; the world model is trained on replay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from minatar import Environment


def set_threads(n: int) -> None:
    torch.set_num_threads(int(n))
    try:
        torch.set_num_interop_threads(max(1, int(n) // 2))
    except RuntimeError:
        pass


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

GAME_ALIASES = {
    "breakout": "breakout",
    "space_invaders": "space_invaders",
    "spaceinvaders": "space_invaders",
}


def make_env(game: str, sticky_action_prob: float = 0.1) -> Environment:
    g = GAME_ALIASES.get(game.lower().replace("-", "_"), game)
    return Environment(g, sticky_action_prob=sticky_action_prob)


def obs_tensor(env: Environment) -> torch.Tensor:
    """Return float32 CHW tensor in {0,1}."""
    s = env.state()  # H, W, C bool
    return torch.from_numpy(np.asarray(s, dtype=np.float32)).permute(2, 0, 1)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], latent_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        self.cnn = nn.Sequential(*layers)
        self.fc = nn.Linear(c_in * 10 * 10, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x)
        return self.fc(h.flatten(1))


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int, channels: list[int]):
        super().__init__()
        rev = list(reversed(channels))
        self.fc = nn.Linear(latent_dim, rev[0] * 10 * 10)
        layers: list[nn.Module] = []
        for i, c_out in enumerate(rev[1:] + [out_channels]):
            c_in = rev[i]
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.ReLU(inplace=True) if i < len(rev) - 1 else nn.Identity(),
            ]
        self.cnn = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.shape[0], -1, 10, 10)
        return self.cnn(h)


class CausalTransformer(nn.Module):
    """Tiny causal transformer over (latent + action) tokens."""

    def __init__(self, width: int, n_layers: int, n_heads: int, history: int):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=n_heads,
            dim_feedforward=width * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pos = nn.Parameter(torch.zeros(1, history, width))
        nn.init.normal_(self.pos, std=0.02)
        self.history = history
        mask = torch.triu(torch.ones(history, history), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: B, T, W
        t = tokens.shape[1]
        x = tokens + self.pos[:, :t]
        return self.blocks(x, mask=self.causal_mask[:t, :t])


class WorldModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_actions: int,
        cnn_channels: list[int],
        latent_dim: int,
        width: int,
        n_layers: int,
        n_heads: int,
        history: int,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.width = width
        self.history = history
        self.n_actions = n_actions
        self.encoder = ConvEncoder(in_channels, cnn_channels, latent_dim)
        self.decoder = ConvDecoder(latent_dim, in_channels, cnn_channels)
        self.z_proj = nn.Linear(latent_dim, width)
        self.a_emb = nn.Embedding(n_actions, width)
        self.transformer = CausalTransformer(width, n_layers, n_heads, history)
        self.delta_head = nn.Linear(width, latent_dim)
        self.reward_head = nn.Linear(width, 1)
        self.cont_head = nn.Linear(width, 1)
        self.latent_norm = nn.LayerNorm(latent_dim)
        # Bound residual steps so open-loop imagination cannot explode.
        self.delta_scale = 0.5

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.latent_norm(self.encoder(obs))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def embed_step(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.z_proj(z) + self.a_emb(action)

    def _next_latent(self, z: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = self.delta_scale * torch.tanh(self.delta_head(h))
        # MinAtar rewards are non-negative; softplus avoids imaginary negative loot.
        reward = F.softplus(self.reward_head(h).squeeze(-1))
        cont_logit = self.cont_head(h).squeeze(-1)
        z_next = self.latent_norm(z + delta)
        return z_next, reward, cont_logit

    def forward_history(
        self, z_seq: torch.Tensor, a_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """z_seq, a_seq: B,T,* → predictions at each step for the *next* transition."""
        tokens = self.embed_step(z_seq, a_seq)
        h = self.transformer(tokens)
        z_next, reward, cont_logit = self._next_latent(z_seq, h)
        return z_next, reward, cont_logit, h

    def imagine_step(
        self, z: torch.Tensor, action: torch.Tensor, h_tokens: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One imagination step from current latent + action, with rolling history tokens."""
        tok = self.embed_step(z, action).unsqueeze(1)  # B,1,W
        if h_tokens is None:
            hist = tok
        else:
            hist = torch.cat([h_tokens, tok], dim=1)
            if hist.shape[1] > self.history:
                hist = hist[:, -self.history :]
        out = self.transformer(hist)
        h = out[:, -1]
        z_next, reward, cont_logit = self._next_latent(z, h)
        return z_next, reward, cont_logit, h, hist

class Actor(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def dist(self, x: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.forward(x).clamp(-10.0, 10.0)
        # 95% learned + 5% uniform mixing — stronger floor than the write-up's
        # 1% so a tiny CPU budget cannot fully collapse the policy.
        probs = F.softmax(logits, dim=-1)
        n = probs.shape[-1]
        mixed = 0.95 * probs + 0.05 / n
        return torch.distributions.Categorical(probs=mixed)

class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class AgentBundle:
    world: WorldModel
    actor: Actor
    critic: Critic
    target_critic: Critic
    n_actions: int
    in_channels: int
    n_params: int


def build_agent(
    in_channels: int,
    n_actions: int,
    cnn_channels: list[int],
    latent_dim: int,
    transformer_width: int,
    transformer_layers: int,
    transformer_heads: int,
    history: int,
    actor_hidden: int,
    critic_hidden: int,
) -> AgentBundle:
    world = WorldModel(
        in_channels,
        n_actions,
        cnn_channels,
        latent_dim,
        transformer_width,
        transformer_layers,
        transformer_heads,
        history,
    )
    # Actor/critic read [z, summary h] when available; use width+latent for input.
    in_dim = latent_dim + transformer_width
    actor = Actor(in_dim, n_actions, actor_hidden)
    critic = Critic(in_dim, critic_hidden)
    target_critic = Critic(in_dim, critic_hidden)
    target_critic.load_state_dict(critic.state_dict())
    for p in target_critic.parameters():
        p.requires_grad_(False)
    n_params = count_params(world) + count_params(actor) + count_params(critic)
    return AgentBundle(world, actor, critic, target_critic, n_actions, in_channels, n_params)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(self, capacity: int, channels: int, history: int):
        self.capacity = int(capacity)
        self.channels = channels
        self.history = history
        self.obs = np.zeros((capacity, channels, 10, 10), dtype=np.float32)
        self.next_obs = np.zeros((capacity, channels, 10, 10), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.cont = np.zeros(capacity, dtype=np.float32)
        self.is_reward_event = np.zeros(capacity, dtype=np.bool_)
        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        cont: float,
    ) -> None:
        i = self.idx
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.action[i] = action
        self.reward[i] = reward
        self.cont[i] = cont
        self.is_reward_event[i] = reward > 0
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample_starts(self, batch: int, seq_len: int, rng: np.random.Generator):
        n = len(self)
        if n < seq_len + 1:
            raise RuntimeError("replay too small")
        # 40% reward-event starts + 60% uniform (slightly denser than write-up
        # 25/75 — sparse MinAtar rewards need more positive samples at toy budget)
        event_idx = np.flatnonzero(self.is_reward_event[:n])
        n_event = max(1, (2 * batch) // 5)
        starts = []
        if len(event_idx) > 0:
            # start so the reward event falls inside the window
            chosen = rng.choice(event_idx, size=min(n_event, len(event_idx)), replace=True)
            for e in chosen:
                s = int(max(0, e - rng.integers(0, seq_len)))
                s = min(s, n - seq_len)
                starts.append(s)
        while len(starts) < batch:
            starts.append(int(rng.integers(0, n - seq_len + 1)))
        starts = np.asarray(starts[:batch], dtype=np.int64)

        # Build sequences
        obs = np.stack([self.obs[s : s + seq_len] for s in starts])
        next_obs = np.stack([self.next_obs[s : s + seq_len] for s in starts])
        action = np.stack([self.action[s : s + seq_len] for s in starts])
        reward = np.stack([self.reward[s : s + seq_len] for s in starts])
        cont = np.stack([self.cont[s : s + seq_len] for s in starts])
        # Importance weight: event-biased vs uniform (simple correction)
        # P_mix(i) ≈ 0.25 * 1/N_event * 1_event + 0.75 / N ; weight = 1/(N P_mix)
        weights = np.ones(batch, dtype=np.float32)
        if len(event_idx) > 0:
            for j, s in enumerate(starts):
                window = self.is_reward_event[s : s + seq_len]
                if window.any():
                    weights[j] = 0.75 + 0.25 * (n / max(1, len(event_idx)))
                else:
                    weights[j] = 0.75
            weights = 1.0 / np.clip(weights, 1e-3, None)
            weights = weights / weights.mean()
        return {
            "obs": torch.from_numpy(obs),
            "next_obs": torch.from_numpy(next_obs),
            "action": torch.from_numpy(action),
            "reward": torch.from_numpy(reward),
            "cont": torch.from_numpy(cont),
            "weight": torch.from_numpy(weights.astype(np.float32)),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _actor_features(z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return torch.cat([z, h], dim=-1)


def train_world_model(
    agent: AgentBundle,
    batch: dict[str, torch.Tensor],
    opt: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> dict[str, float]:
    world = agent.world
    world.train()
    obs = batch["obs"]  # B,T,C,H,W
    next_obs = batch["next_obs"]
    action = batch["action"]
    reward = batch["reward"]
    cont = batch["cont"]
    weight = batch["weight"].unsqueeze(1)  # B,1
    b, t, c, h, w = obs.shape

    obs_flat = obs.reshape(b * t, c, h, w)
    next_flat = next_obs.reshape(b * t, c, h, w)
    # Encoder must receive reconstruction grads; freeze only the *target* latents.
    z = world.encode(obs_flat).view(b, t, -1)
    with torch.no_grad():
        z_next_tgt = world.encode(next_flat).view(b, t, -1)

    # Reconstruction on current frames
    recon = world.decode(z.reshape(b * t, -1))
    recon_loss = F.binary_cross_entropy_with_logits(recon, obs_flat, reduction="none")
    # Occupied cells are rare — weight positives (write-up idea, simplified)
    pos_w = 1.0 + 4.0 * obs_flat
    recon_loss = (recon_loss * pos_w).mean(dim=(1, 2, 3)).view(b, t)
    recon_loss = (recon_loss * weight).mean()

    # Dynamics heads train on detached latents (representation–dynamics split)
    z_det = z.detach()
    z_pred, r_pred, c_logit, _h = world.forward_history(z_det, action)
    latent_loss = F.smooth_l1_loss(z_pred, z_next_tgt, reduction="none").mean(-1)
    latent_loss = (latent_loss * weight).mean()
    # Decode one-step predicted latents
    frame_pred = world.decode(z_pred.reshape(b * t, -1))
    frame_loss = F.binary_cross_entropy_with_logits(
        frame_pred, next_flat, reduction="none"
    )
    frame_loss = (frame_loss * (1.0 + 4.0 * next_flat)).mean(dim=(1, 2, 3)).view(b, t)
    frame_loss = (frame_loss * weight).mean()
    reward_loss = F.smooth_l1_loss(r_pred, reward, reduction="none")
    reward_loss = (reward_loss * weight).mean()
    cont_loss = F.binary_cross_entropy_with_logits(c_logit, cont, reduction="none")
    cont_loss = (cont_loss * weight).mean()

    # Multi-step generated rollouts (wm_horizon). Detach the rolled latent when
    # feeding the next step so open-loop prediction cannot explode gradients.
    horizon = int(cfg["wm_horizon"])
    gen_losses = []
    z_roll = z_det[:, 0]
    hist = None
    for k in range(horizon):
        a_k = action[:, min(k, t - 1)]
        z_pred_k, r_k, c_k, _hk, hist = world.imagine_step(z_roll, a_k, hist)
        tgt_idx = min(k, t - 1)
        if k < t:
            gl = F.smooth_l1_loss(z_pred_k, z_next_tgt[:, tgt_idx])
            gl = gl + float(cfg["reward_loss_coef"]) * F.smooth_l1_loss(
                r_k, reward[:, tgt_idx]
            )
            gl = gl + float(cfg["continue_loss_coef"]) * F.binary_cross_entropy_with_logits(
                c_k, cont[:, tgt_idx]
            )
            gen_losses.append(gl)
        z_roll = z_pred_k.detach()
        hist = hist.detach() if hist is not None else None
    gen_loss = torch.stack(gen_losses).mean() if gen_losses else torch.tensor(0.0)

    loss = (
        float(cfg["frame_loss_coef"]) * recon_loss
        + float(cfg["latent_loss_coef"]) * latent_loss
        + float(cfg["frame_loss_coef"]) * frame_loss
        + float(cfg["reward_loss_coef"]) * reward_loss
        + float(cfg["continue_loss_coef"]) * cont_loss
        + gen_loss
    )
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(world.parameters(), float(cfg["grad_clip"]))
    opt.step()
    return {
        "wm_loss": float(loss.item()),
        "recon": float(recon_loss.item()),
        "latent": float(latent_loss.item()),
        "reward": float(reward_loss.item()),
    }


def train_actor_critic_imagined(
    agent: AgentBundle,
    batch: dict[str, torch.Tensor],
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Actor trains entirely on imagined rollouts (no real-env policy gradients).

    Critic mixes imagined λ-returns with a real one-step TD target (write-up).
    """
    world = agent.world
    actor = agent.actor
    critic = agent.critic
    target_critic = agent.target_critic
    world.eval()
    for p in world.parameters():
        p.requires_grad_(False)

    obs = batch["obs"]  # B,T,C,H,W
    action = batch["action"]
    real_reward = batch["reward"]
    real_cont = batch["cont"]
    next_obs = batch["next_obs"]
    b, t, c, h, w = obs.shape
    horizon = int(cfg["imagine_horizon"])
    gamma = float(cfg["gamma"])
    lam = float(cfg["lambda_gae"])

    with torch.no_grad():
        # Warm history with real prefix (up to history length)
        prefix = min(t, world.history)
        z_seq = world.encode(obs[:, :prefix].reshape(b * prefix, c, h, w)).view(
            b, prefix, -1
        )
        a_seq = action[:, :prefix]
        tokens = world.embed_step(z_seq, a_seq)
        hist = tokens
        z = z_seq[:, -1]
        h_feat = world.transformer(hist)[:, -1]

    rewards = []
    conts = []
    logps = []
    ents = []
    values = []
    target_values = []

    actor.train()
    critic.train()
    for _ in range(horizon):
        feat = _actor_features(z, h_feat)
        dist = actor.dist(feat)
        a = dist.sample()
        logps.append(dist.log_prob(a))
        ents.append(dist.entropy())
        values.append(critic(feat))
        with torch.no_grad():
            target_values.append(target_critic(feat))
            z, r, c_logit, h_feat, hist = world.imagine_step(z, a, hist)
            # Actor does not differentiate through the world model.
            rewards.append(r)
            conts.append(torch.sigmoid(c_logit))

    # λ-returns with target critic (Dreamer-style)
    with torch.no_grad():
        feat_last = _actor_features(z, h_feat)
        bootstrap = target_critic(feat_last)
        returns = []
        g = bootstrap
        for t_i in reversed(range(horizon)):
            g = rewards[t_i] + gamma * conts[t_i] * (
                (1.0 - lam) * target_values[t_i] + lam * g
            )
            returns.append(g)
        returns = list(reversed(returns))

    logps_t = torch.stack(logps, dim=1)
    ents_t = torch.stack(ents, dim=1)
    values_t = torch.stack(values, dim=1)
    returns_t = torch.stack(returns, dim=1)
    adv = returns_t - values_t.detach()
    # Normalize advantages per batch for stable imagination updates
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    actor_loss = -(logps_t * adv).mean() - float(cfg["entropy_coef"]) * ents_t.mean()
    critic_imagine = F.mse_loss(values_t, returns_t.detach())

    # Real one-step TD critic (write-up: real replay weight 1, imagined 0.3)
    with torch.no_grad():
        z_real = world.encode(obs[:, -1])
        a_ctx = action[:, -1]
        tok = world.embed_step(z_real, a_ctx).unsqueeze(1)
        h_real = world.transformer(tok)[:, -1]
        feat_real = _actor_features(z_real, h_real)
        z_next = world.encode(next_obs[:, -1])
        tok_n = world.embed_step(z_next, a_ctx).unsqueeze(1)
        h_next = world.transformer(tok_n)[:, -1]
        feat_next = _actor_features(z_next, h_next)
        td_target = real_reward[:, -1] + gamma * real_cont[:, -1] * target_critic(
            feat_next
        )
    v_real = critic(feat_real)
    critic_real = F.mse_loss(v_real, td_target)
    critic_loss = critic_real + float(cfg.get("critic_imagine_coef", 0.3)) * critic_imagine

    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), float(cfg["grad_clip"]))
    actor_opt.step()

    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), float(cfg["grad_clip"]))
    critic_opt.step()

    # EMA target critic (~2% toward online, write-up)
    with torch.no_grad():
        for p, tp in zip(critic.parameters(), target_critic.parameters()):
            tp.data.mul_(0.98).add_(p.data, alpha=0.02)

    for p in world.parameters():
        p.requires_grad_(True)

    return {
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "entropy": float(ents_t.mean().item()),
        "mean_imag_reward": float(torch.stack(rewards).mean().item()),
    }


def epsilon_at(step: int, cfg: dict[str, Any]) -> float:
    start = float(cfg["explore_start"])
    end = float(cfg["explore_end"])
    decay = max(1, int(cfg["explore_decay_steps"]))
    if step >= decay:
        return end
    return start + (end - start) * (step / decay)


@torch.no_grad()
def select_action(
    agent: AgentBundle,
    obs: torch.Tensor,
    hist_tokens: torch.Tensor | None,
    last_action: int | None,
    epsilon: float,
    greedy: bool,
    rng: np.random.Generator,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Choose an action from the current observation.

    History tokens store past (z, action) embeddings. The current latent is
    paired with a placeholder / last action only to obtain a transformer
    summary *before* committing the new action; the returned history then
    appends the actually chosen action.
    """
    world = agent.world
    actor = agent.actor
    world.eval()
    actor.eval()
    z = world.encode(obs.unsqueeze(0))
    a_ctx = 0 if last_action is None else int(last_action)
    a0 = torch.tensor([a_ctx], dtype=torch.long)
    tok = world.embed_step(z, a0).unsqueeze(1)
    if hist_tokens is None:
        hist = tok
    else:
        hist = torch.cat([hist_tokens, tok], dim=1)
        if hist.shape[1] > world.history:
            hist = hist[:, -world.history :]
    h = world.transformer(hist)[:, -1]
    feat = _actor_features(z, h)
    if greedy:
        logits = actor.forward(feat)
        action = int(torch.argmax(logits, dim=-1).item())
    else:
        if rng.random() < epsilon:
            action = int(rng.integers(0, agent.n_actions))
        else:
            dist = actor.dist(feat)
            action = int(dist.sample().item())
    # Commit history with the chosen action (replace the context token).
    a_t = torch.tensor([action], dtype=torch.long)
    committed = world.embed_step(z, a_t).unsqueeze(1)
    if hist_tokens is None:
        new_hist = committed
    else:
        new_hist = torch.cat([hist_tokens, committed], dim=1)
        if new_hist.shape[1] > world.history:
            new_hist = new_hist[:, -world.history :]
    return action, new_hist, z.squeeze(0)

def evaluate_policy(
    agent: AgentBundle,
    game: str,
    sticky_action_prob: float,
    n_episodes: int,
    max_episode_steps: int,
    seed: int,
    greedy: bool = True,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    returns = []
    lengths = []
    for ep in range(n_episodes):
        env = make_env(game, sticky_action_prob)
        # MinAtar seeds via numpy global; set per-episode
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        env.reset()
        hist = None
        last_a = None
        ep_ret = 0.0
        steps = 0
        terminated = False
        while not terminated and steps < max_episode_steps:
            obs = obs_tensor(env)
            action, hist, _z = select_action(
                agent, obs, hist, last_a, epsilon=0.0, greedy=greedy, rng=rng
            )
            reward, terminated = env.act(action)
            ep_ret += float(reward)
            last_a = action
            steps += 1
        returns.append(ep_ret)
        lengths.append(steps)
    arr = np.asarray(returns, dtype=np.float64)
    return {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "mean_length": float(np.mean(lengths)),
        "n_episodes": float(n_episodes),
    }


def evaluate_random(
    game: str,
    sticky_action_prob: float,
    n_episodes: int,
    max_episode_steps: int,
    seed: int,
    n_actions: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed + 10_000)
    returns = []
    for _ in range(n_episodes):
        env = make_env(game, sticky_action_prob)
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        env.reset()
        ep_ret = 0.0
        steps = 0
        terminated = False
        while not terminated and steps < max_episode_steps:
            a = int(rng.integers(0, n_actions))
            reward, terminated = env.act(a)
            ep_ret += float(reward)
            steps += 1
        returns.append(ep_ret)
    arr = np.asarray(returns, dtype=np.float64)
    return {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n_episodes": float(n_episodes),
    }


def train_agent(
    game: str,
    seed: int,
    cfg: dict[str, Any],
    model_overrides: dict[str, Any] | None = None,
    policy_steps_override: int | None = None,
    imagine_horizon_override: int | None = None,
    log_fn=None,
) -> dict[str, Any]:
    """Train one agent on one game; return metrics + state dicts."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    env_cfg = cfg["env"]
    mcfg = dict(cfg["model"])
    tcfg = dict(cfg["training"])
    if model_overrides:
        mcfg.update(model_overrides)
    if policy_steps_override is not None:
        tcfg["policy_steps"] = int(policy_steps_override)
    if imagine_horizon_override is not None:
        tcfg["imagine_horizon"] = int(imagine_horizon_override)

    env = make_env(game, float(env_cfg["sticky_action_prob"]))
    env.reset()
    in_channels = int(env.n_channels)
    n_actions = int(env.num_actions())

    agent = build_agent(
        in_channels=in_channels,
        n_actions=n_actions,
        cnn_channels=list(mcfg["cnn_channels"]),
        latent_dim=int(mcfg["latent_dim"]),
        transformer_width=int(mcfg["transformer_width"]),
        transformer_layers=int(mcfg["transformer_layers"]),
        transformer_heads=int(mcfg["transformer_heads"]),
        history=int(mcfg["history"]),
        actor_hidden=int(mcfg["actor_hidden"]),
        critic_hidden=int(mcfg["critic_hidden"]),
    )
    assert agent.n_params < 1_000_000, f"params {agent.n_params} exceed 1M budget"

    wm_opt = torch.optim.Adam(agent.world.parameters(), lr=float(tcfg["wm_lr"]))
    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=float(tcfg["actor_lr"]))
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=float(tcfg["critic_lr"]))

    replay = ReplayBuffer(
        int(tcfg["replay_capacity"]), in_channels, int(mcfg["history"])
    )
    hist = None
    last_a = None
    obs = obs_tensor(env)
    curve = []
    ep_ret = 0.0
    ep_count = 0

    import time as _time

    t0 = _time.perf_counter()
    total_steps = int(tcfg["initial_random_steps"]) + int(tcfg["policy_steps"])
    policy_step = 0

    for step in range(total_steps):
        exploring = step < int(tcfg["initial_random_steps"])
        if exploring:
            action = int(rng.integers(0, n_actions))
            # still maintain history for later
            with torch.no_grad():
                z = agent.world.encode(obs.unsqueeze(0))
                a_t = torch.tensor([action], dtype=torch.long)
                tok = agent.world.embed_step(z, a_t).unsqueeze(1)
                hist = tok if hist is None else torch.cat([hist, tok], dim=1)
                if hist.shape[1] > agent.world.history:
                    hist = hist[:, -agent.world.history :]
        else:
            eps = epsilon_at(policy_step, tcfg)
            action, hist, _z = select_action(
                agent, obs, hist, last_a, epsilon=eps, greedy=False, rng=rng
            )
            policy_step += 1

        reward, terminated = env.act(action)
        next_obs = obs_tensor(env)
        cont = 0.0 if terminated else 1.0
        replay.add(
            obs.numpy(),
            action,
            float(reward),
            next_obs.numpy(),
            cont,
        )
        ep_ret += float(reward)
        obs = next_obs
        last_a = action

        if terminated:
            env.reset()
            obs = obs_tensor(env)
            hist = None
            last_a = None
            curve.append(ep_ret)
            ep_ret = 0.0
            ep_count += 1

        # Updates after enough replay and past initial random fill
        if (
            not exploring
            and policy_step > 0
            and policy_step % int(tcfg["update_every"]) == 0
            and len(replay) >= int(tcfg["wm_batch"]) * int(mcfg["history"])
        ):
            seq_len = int(mcfg["history"])
            wm_stats: dict = {}
            ac_stats: dict = {}
            try:
                b_wm = replay.sample_starts(int(tcfg["wm_batch"]), seq_len, rng)
                wm_stats = train_world_model(agent, b_wm, wm_opt, tcfg)
                # Delay actor updates until the world model has a warm start.
                if policy_step >= int(tcfg.get("actor_delay_steps", 0)):
                    b_ac = replay.sample_starts(int(tcfg["imagine_batch"]), seq_len, rng)
                    ac_stats = train_actor_critic_imagined(
                        agent, b_ac, actor_opt, critic_opt, tcfg
                    )
            except RuntimeError:
                pass
            if log_fn and policy_step % int(tcfg["log_every"]) == 0:
                log_fn(
                    {
                        "game": game,
                        "seed": seed,
                        "policy_step": policy_step,
                        **wm_stats,
                        **ac_stats,
                    }
                )

    wall = _time.perf_counter() - t0
    # close dangling episode
    if ep_ret != 0.0 or not curve:
        curve.append(ep_ret)

    return {
        "agent": agent,
        "n_params": agent.n_params,
        "train_wall_s": wall,
        "learning_curve": curve,
        "policy_steps": int(tcfg["policy_steps"]),
        "imagine_horizon": int(tcfg["imagine_horizon"]),
        "model_cfg": mcfg,
        "in_channels": in_channels,
        "n_actions": n_actions,
    }


def save_agent(path, train_result: dict[str, Any], game: str, seed: int) -> None:
    agent: AgentBundle = train_result["agent"]
    torch.save(
        {
            "game": game,
            "seed": seed,
            "model_cfg": train_result["model_cfg"],
            "in_channels": train_result["in_channels"],
            "n_actions": train_result["n_actions"],
            "n_params": train_result["n_params"],
            "world": agent.world.state_dict(),
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "learning_curve": train_result["learning_curve"],
        },
        path,
    )


def load_agent(path) -> tuple[AgentBundle, dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mc = ckpt["model_cfg"]
    agent = build_agent(
        in_channels=int(ckpt["in_channels"]),
        n_actions=int(ckpt["n_actions"]),
        cnn_channels=list(mc["cnn_channels"]),
        latent_dim=int(mc["latent_dim"]),
        transformer_width=int(mc["transformer_width"]),
        transformer_layers=int(mc["transformer_layers"]),
        transformer_heads=int(mc["transformer_heads"]),
        history=int(mc["history"]),
        actor_hidden=int(mc["actor_hidden"]),
        critic_hidden=int(mc["critic_hidden"]),
    )
    agent.world.load_state_dict(ckpt["world"])
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.target_critic.load_state_dict(ckpt["critic"])
    agent.world.eval()
    agent.actor.eval()
    agent.critic.eval()
    return agent, ckpt
