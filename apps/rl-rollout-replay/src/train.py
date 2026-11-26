"""On-policy REINFORCE and the two replays from arXiv:2611.33102.

Online step: sample a group of answers from the current policy, score them,
and take one AdamW step on the score-function gradient. The advantage is the
group-mean baseline in Table 1 (REINFORCE / RLOO row, prompt weight 1):

    A = r - mean(r)

Maximum-likelihood replay ignores r and minimizes token NLL on every archived
answer. Naive replay applies the same REINFORCE update to the archived batch,
with scores taken at the replay policy (paper Eq. 2).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn

from policy import DigitPolicy
from task import ANSWER_LEN, RewardName, demo_batch, make_batch, sample_train_batch, score_answers, to_torch

Advantage = Literal["reinforce", "maxrl", "grpo"]
Objective = Literal["ml", "rl"]


def advantages(rewards: torch.Tensor, kind: Advantage) -> torch.Tensor:
    """rewards [B, G] -> advantages [B, G]."""
    mean = rewards.mean(dim=1, keepdim=True)
    if kind == "reinforce":
        return rewards - mean
    if kind == "maxrl":
        return (rewards - mean) / (mean + 1.0e-6)
    if kind == "grpo":
        var = rewards.var(dim=1, keepdim=True, unbiased=False)
        return (rewards - mean) / (torch.sqrt(var) + 1.0e-6)
    raise ValueError(kind)


def flatten_group(prompts: torch.Tensor, answers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    batch, group, length = answers.shape
    flat_prompts = prompts.repeat_interleave(group, dim=0)
    flat_answers = answers.reshape(batch * group, length)
    return flat_prompts, flat_answers, batch, group


def ml_loss(policy: DigitPolicy, prompts: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
    flat_prompts, flat_answers, _, _ = flatten_group(prompts, answers)
    return -policy.token_logprobs(flat_prompts, flat_answers).mean()


def reinforce_loss(
    policy: DigitPolicy,
    prompts: torch.Tensor,
    answers: torch.Tensor,
    rewards: torch.Tensor,
    kind: Advantage,
) -> torch.Tensor:
    flat_prompts, flat_answers, batch, group = flatten_group(prompts, answers)
    token_logp = policy.token_logprobs(flat_prompts, flat_answers)
    adv = advantages(rewards, kind).reshape(batch * group, 1)
    return -(adv.detach() * token_logp).mean()


def adam_step(policy: DigitPolicy, opt: torch.optim.Optimizer, loss: torch.Tensor, clip: float) -> None:
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss {float(loss)}")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), clip)
    opt.step()


def make_opt(policy: DigitPolicy, lr: float, weight_decay: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)


def snapshot(policy: DigitPolicy) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}


def restore(policy: DigitPolicy, state: dict[str, torch.Tensor]) -> None:
    policy.load_state_dict(state)


def build_policy(emb: int, hidden: int, init_seed: int | None) -> DigitPolicy:
    if init_seed is not None:
        torch.manual_seed(init_seed)
    policy = DigitPolicy(emb, hidden)
    policy.cpu()
    return policy


@torch.no_grad()
def exact_reward(
    policy: DigitPolicy,
    prompts: torch.Tensor,
    truth: torch.Tensor,
    n_samples: int,
    temperature: float,
    generator: torch.Generator,
) -> float:
    policy.eval()
    answers = policy.sample_len(prompts, n_samples, ANSWER_LEN, generator, temperature)
    match = (answers == truth[:, None, :]).all(dim=-1)
    return float(match.to(dtype=torch.float32).mean())


@torch.no_grad()
def greedy_ints(policy: DigitPolicy, prompts: torch.Tensor) -> list[int]:
    policy.eval()
    tokens = policy.greedy(prompts, ANSWER_LEN).cpu().tolist()
    out: list[int] = []
    for row in tokens:
        value = 0
        scale = 1
        for digit in row:
            value += int(digit) * scale
            scale *= 10
        out.append(value)
    return out


def _eval_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


class Archive:
    def __init__(self) -> None:
        self.prompts: list[torch.Tensor] = []
        self.answers: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []

    def add(self, prompts: torch.Tensor, answers: torch.Tensor, rewards: torch.Tensor) -> None:
        self.prompts.append(prompts.detach().cpu().contiguous())
        self.answers.append(answers.detach().cpu().contiguous())
        self.rewards.append(rewards.detach().cpu().contiguous())

    def stack(self) -> dict[str, torch.Tensor]:
        return {
            "prompts": torch.stack(self.prompts, dim=0),
            "answers": torch.stack(self.answers, dim=0),
            "rewards": torch.stack(self.rewards, dim=0),
        }


def save_archive(path: str, archive: dict[str, torch.Tensor], task: str) -> None:
    torch.save({"task": task, **archive}, path)


def load_archive(path: str) -> dict[str, torch.Tensor]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return {"prompts": blob["prompts"], "answers": blob["answers"], "rewards": blob["rewards"]}


def run_sft(
    policy: DigitPolicy,
    *,
    steps: int,
    batch_prompts: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    data_seed: int,
    holdout: frozenset[tuple[int, ...]],
    eval_prompts: torch.Tensor,
    eval_truth: torch.Tensor,
    eval_samples: int,
    eval_temperature: float,
    eval_seed: int,
    log_every: int,
) -> list[float]:
    opt = make_opt(policy, lr, weight_decay)
    rng = np.random.default_rng(data_seed)
    curve: list[float] = []

    def record(step: int) -> None:
        curve.append(
            exact_reward(
                policy,
                eval_prompts,
                eval_truth,
                eval_samples,
                eval_temperature,
                _eval_generator(eval_seed + step),
            )
        )

    record(0)
    for step in range(steps):
        prompts_np, answers_np = sample_train_batch(rng, batch_prompts, "add", holdout)
        prompts, answers = to_torch(prompts_np, answers_np)
        policy.train()
        # Teacher forcing is a group of size 1.
        loss = ml_loss(policy, prompts, answers[:, None, :])
        adam_step(policy, opt, loss, grad_clip)
        record(step + 1)
        if log_every and (step + 1) % log_every == 0:
            print(f"sft step {step + 1}/{steps} exact_reward {curve[-1]:.3f}", flush=True)
    return curve


def run_online(
    policy: DigitPolicy,
    *,
    steps: int,
    batch_prompts: int,
    group_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    advantage: Advantage,
    reward_name: RewardName,
    data_seed: int,
    holdout: frozenset[tuple[int, ...]],
    op: Literal["add", "sub"],
    eval_prompts: torch.Tensor,
    eval_truth: torch.Tensor,
    eval_samples: int,
    eval_temperature: float,
    eval_seed: int,
    demo_prompts: torch.Tensor | None,
    log_every: int,
) -> tuple[list[float], dict[str, torch.Tensor], list[list[int]] | None]:
    opt = make_opt(policy, lr, weight_decay)
    rng = np.random.default_rng(data_seed)
    train_gen = torch.Generator(device="cpu")
    train_gen.manual_seed(data_seed + 17)
    archive = Archive()
    curve: list[float] = []
    demos: list[list[int]] | None = [] if demo_prompts is not None else None

    def record(step: int) -> None:
        curve.append(
            exact_reward(
                policy,
                eval_prompts,
                eval_truth,
                eval_samples,
                eval_temperature,
                _eval_generator(eval_seed + step),
            )
        )
        if demos is not None and demo_prompts is not None:
            demos.append(greedy_ints(policy, demo_prompts))

    record(0)
    for step in range(steps):
        prompts_np, truth_np = sample_train_batch(rng, batch_prompts, op, holdout)
        prompts, truth = to_torch(prompts_np, truth_np)
        policy.eval()
        answers = policy.sample_len(prompts, group_size, ANSWER_LEN, train_gen, temperature=1.0)
        rewards = score_answers(answers, truth, reward_name)
        archive.add(prompts, answers, rewards)
        policy.train()
        loss = reinforce_loss(policy, prompts, answers, rewards, advantage)
        adam_step(policy, opt, loss, grad_clip)
        record(step + 1)
        if log_every and (step + 1) % log_every == 0:
            print(
                f"online {op} step {step + 1}/{steps} exact_reward {curve[-1]:.3f} batch_r {float(rewards.mean()):.3f}",
                flush=True,
            )
    return curve, archive.stack(), demos


def replay_batches(
    policy: DigitPolicy,
    archive: dict[str, torch.Tensor],
    *,
    objective: Objective,
    advantage: Advantage,
    order: list[int],
    repeats: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    eval_prompts: torch.Tensor,
    eval_truth: torch.Tensor,
    eval_samples: int,
    eval_temperature: float,
    eval_seed: int,
    demo_prompts: torch.Tensor | None,
) -> tuple[list[float], list[list[int]] | None]:
    opt = make_opt(policy, lr, weight_decay)
    curve: list[float] = []
    demos: list[list[int]] | None = [] if demo_prompts is not None else None

    def record(step: int) -> None:
        curve.append(
            exact_reward(
                policy,
                eval_prompts,
                eval_truth,
                eval_samples,
                eval_temperature,
                _eval_generator(eval_seed + step),
            )
        )
        if demos is not None and demo_prompts is not None:
            demos.append(greedy_ints(policy, demo_prompts))

    record(0)
    for step, index in enumerate(order):
        prompts = archive["prompts"][index]
        answers = archive["answers"][index]
        rewards = archive["rewards"][index]
        for _ in range(repeats):
            policy.train()
            if objective == "ml":
                loss = ml_loss(policy, prompts, answers)
            elif objective == "rl":
                loss = reinforce_loss(policy, prompts, answers, rewards, advantage)
            else:
                raise ValueError(objective)
            adam_step(policy, opt, loss, grad_clip)
        record(step + 1)
    return curve, demos


def chronological(n_steps: int) -> list[int]:
    return list(range(n_steps))


def last_k_order(n_steps: int, k: int, n_updates: int) -> list[int]:
    if k < 1 or k > n_steps:
        raise ValueError(f"last_k {k} outside 1..{n_steps}")
    start = n_steps - k
    return [start + (i % k) for i in range(n_updates)]


def perturb_(policy: DigitPolicy, sigma: float, seed: int) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        for param in policy.parameters():
            noise = torch.randn(param.shape, generator=generator, dtype=param.dtype)
            param.add_(noise, alpha=sigma)


def max_param_gap(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    gap = 0.0
    for key in a:
        gap = max(gap, float((a[key] - b[key]).abs().max()))
    return gap


def prepare_eval(cfg_eval: dict, op: Literal["add", "sub"]) -> tuple[torch.Tensor, torch.Tensor, frozenset[tuple[int, ...]]]:
    seed = int(cfg_eval["seed"]) + (0 if op == "add" else 1)
    prompts_np, answers_np = make_batch(np.random.default_rng(seed), int(cfg_eval["n_problems"]), op)
    prompts, answers = to_torch(prompts_np, answers_np)
    keys = frozenset(tuple(int(x) for x in row) for row in prompts_np)
    return prompts, answers, keys


def self_check_train() -> None:
    torch.manual_seed(0)
    policy = build_policy(16, 16, init_seed=0)
    prompts = torch.randint(0, 10, (4, 6))
    prompts[:, 2] = 10
    prompts[:, 5] = 12
    answers = torch.randint(0, 10, (4, 3))
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 1.0]])
    # Group size 4, so expand answers.
    grouped = answers[:, None, :].expand(4, 4, 3).contiguous()
    adv = advantages(rewards, "reinforce")
    if not torch.allclose(adv.sum(dim=1), torch.zeros(4), atol=1e-6):
        raise RuntimeError("REINFORCE advantages must sum to zero inside a group")
    maxrl = advantages(rewards[:1, :2], "maxrl")
    # mean of [1, 0] is 0.5; advantages are +1 and -1
    if not torch.allclose(maxrl, torch.tensor([[1.0, -1.0]]), atol=1e-5):
        raise RuntimeError(maxrl)
    other = build_policy(16, 16, init_seed=0)
    loss_a = reinforce_loss(policy, prompts, grouped, rewards, "reinforce")
    loss_b = reinforce_loss(other, prompts, grouped, rewards, "reinforce")
    loss_a.backward()
    loss_b.backward()
    for left, right in zip(policy.parameters(), other.parameters(), strict=True):
        if left.grad is None or right.grad is None or not torch.allclose(left.grad, right.grad, atol=1e-6):
            raise RuntimeError("identical policies produced different REINFORCE gradients")
    fresh = build_policy(16, 16, init_seed=1)
    opt = make_opt(fresh, lr=0.05, weight_decay=0.0)
    target = torch.randint(0, 10, (8, 3))
    batch_p = prompts[:1].expand(8, 6).contiguous()
    before = float(ml_loss(fresh, batch_p, target[:, None, :]).detach())
    for _ in range(8):
        fresh.train()
        adam_step(fresh, opt, ml_loss(fresh, batch_p, target[:, None, :]), clip=1.0)
    after = float(ml_loss(fresh, batch_p, target[:, None, :]).detach())
    if not after < before - 0.05:
        raise RuntimeError(f"NLL did not fall on a fixed batch ({before:.3f} -> {after:.3f})")
    _ = demo_batch("add")
