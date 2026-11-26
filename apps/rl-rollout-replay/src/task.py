"""Two-digit addition and subtraction, least-significant digit first.

The reported reward is exact match of the three answer digits. Training can
score a rollout with that same bit, or with the fraction of correct digits.
Prompts are six tokens: tens, ones, operator, tens, ones, equals.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

Op = Literal["add", "sub"]
RewardName = Literal["exact", "digit"]

PLUS = 10
MINUS = 11
EQ = 12
VOCAB = 13
N_DIGITS = 10
PROMPT_LEN = 6
ANSWER_LEN = 3

DEMO_A = (12, 48, 99, 7, 50, 63)
DEMO_B = (35, 27, 1, 8, 50, 19)
DEMO_TRUTH = (47, 75, 100, 15, 100, 82)


def _answers_from_sums(sums: np.ndarray) -> np.ndarray:
    sums = np.asarray(sums, dtype=np.int64)
    return np.stack((sums % 10, (sums // 10) % 10, sums // 100), axis=1)


def make_batch(rng: np.random.Generator, n: int, op: Op, low: int = 0, high: int = 100) -> tuple[np.ndarray, np.ndarray]:
    a = rng.integers(low, high, size=n, dtype=np.int64)
    b = rng.integers(low, high, size=n, dtype=np.int64)
    op_tok = np.full(n, PLUS if op == "add" else MINUS, dtype=np.int64)
    prompts = np.stack((a // 10, a % 10, op_tok, b // 10, b % 10, np.full(n, EQ, dtype=np.int64)), axis=1)
    sums = a + b if op == "add" else np.abs(a - b)
    return prompts, _answers_from_sums(sums)


def demo_batch(op: Op = "add") -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    a = np.asarray(DEMO_A, dtype=np.int64)
    b = np.asarray(DEMO_B, dtype=np.int64)
    n = a.shape[0]
    op_tok = np.full(n, PLUS if op == "add" else MINUS, dtype=np.int64)
    prompts = np.stack((a // 10, a % 10, op_tok, b // 10, b % 10, np.full(n, EQ, dtype=np.int64)), axis=1)
    sums = a + b if op == "add" else np.abs(a - b)
    truths = tuple(int(x) for x in sums.tolist())
    if op == "add" and truths != DEMO_TRUTH:
        raise RuntimeError(f"demo truths changed: {truths}")
    return prompts, _answers_from_sums(sums), truths


def holdout_keys(prompts: np.ndarray) -> frozenset[tuple[int, ...]]:
    return frozenset(tuple(int(x) for x in row) for row in prompts)


def sample_train_batch(
    rng: np.random.Generator,
    n: int,
    op: Op,
    holdout: frozenset[tuple[int, ...]],
    low: int = 0,
    high: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    got_p: list[np.ndarray] = []
    got_a: list[np.ndarray] = []
    guard = 0
    while len(got_p) < n:
        guard += 1
        if guard > 1000:
            raise RuntimeError("holdout filter rejected too many prompts")
        prompts, answers = make_batch(rng, n, op, low, high)
        for i in range(n):
            key = tuple(int(x) for x in prompts[i])
            if key in holdout:
                continue
            got_p.append(prompts[i])
            got_a.append(answers[i])
            if len(got_p) == n:
                break
    return np.stack(got_p), np.stack(got_a)


def to_torch(prompts: np.ndarray, answers: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.ascontiguousarray(prompts, dtype=np.int64)),
        torch.from_numpy(np.ascontiguousarray(answers, dtype=np.int64)),
    )


def score_answers(answers: torch.Tensor, truth: torch.Tensor, reward: RewardName) -> torch.Tensor:
    """answers [B, G, A], truth [B, A] -> [B, G]."""
    match = answers == truth[:, None, :]
    if reward == "exact":
        return match.all(dim=-1).to(dtype=torch.float32)
    if reward == "digit":
        return match.to(dtype=torch.float32).mean(dim=-1)
    raise ValueError(reward)


def decode_int(tokens: list[int] | tuple[int, ...] | np.ndarray) -> int:
    digits = [int(x) for x in tokens]
    value = 0
    scale = 1
    for digit in digits:
        value += digit * scale
        scale *= 10
    return value


def human_prompt(tokens: list[int] | np.ndarray, op: Op) -> str:
    row = [int(x) for x in tokens]
    a = row[0] * 10 + row[1]
    b = row[3] * 10 + row[4]
    sym = "+" if op == "add" else "-"
    return f"{a}{sym}{b}"


def self_check_task() -> None:
    if decode_int([7, 4, 0]) != 47 or decode_int([0, 0, 1]) != 100:
        raise RuntimeError("LSD decode failed")
    prompts, answers, truths = demo_batch("add")
    if truths != DEMO_TRUTH:
        raise RuntimeError(truths)
    decoded = [decode_int(answers[i]) for i in range(answers.shape[0])]
    if decoded != list(DEMO_TRUTH):
        raise RuntimeError(decoded)
    sub_p, sub_a = make_batch(np.random.default_rng(0), 1, "sub")
    #  rebuild a known subtraction
    known_p, known_a = make_batch(np.random.default_rng(1), 4, "sub")
    del sub_p
    for i in range(known_p.shape[0]):
        a = int(known_p[i, 0]) * 10 + int(known_p[i, 1])
        b = int(known_p[i, 3]) * 10 + int(known_p[i, 4])
        if decode_int(known_a[i]) != abs(a - b):
            raise RuntimeError("subtraction encoding failed")
    ev_p, ev_a = make_batch(np.random.default_rng(12345), 32, "add")
    keys = holdout_keys(ev_p)
    train_p, _ = sample_train_batch(np.random.default_rng(3), 64, "add", keys)
    if any(tuple(int(x) for x in row) in keys for row in train_p):
        raise RuntimeError("holdout leaked into the train batch")
    t_prompts, t_answers = to_torch(ev_p[:4], ev_a[:4])
    grouped = t_answers[:, None, :].expand(4, 2, ANSWER_LEN).contiguous()
    exact = score_answers(grouped, t_answers, "exact")
    digit = score_answers(grouped, t_answers, "digit")
    if not torch.equal(exact, torch.ones(4, 2)) or not torch.allclose(digit, torch.ones(4, 2)):
        raise RuntimeError("perfect rollouts must score 1")
    if ev_p.shape[1] != PROMPT_LEN or ev_a.shape[1] != ANSWER_LEN:
        raise RuntimeError("shape")
