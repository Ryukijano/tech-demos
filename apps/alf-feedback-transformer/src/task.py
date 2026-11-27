"""Register-trace language for the ALF toy.

Each document is BOS, then `n_rounds` of (op, Q, digit). The digit is a
checksum of a 3-register machine in Z/5Z. Ops are the only state updates.
One checksum does not determine the register file, so the next digit still
depends on the latent state.

The X post was not fetched. This task is not in the paper.
"""

from __future__ import annotations

import numpy as np

MOD = 5
N_REG = 3
N_ADD = 12  # 3 registers × deltas 1..4
N_MUL = 3
N_CROSS = 3
N_OPS = N_ADD + N_MUL + N_CROSS  # 18

BOS_ID = 0
Q_ID = 1
DIGIT0 = 2
OP0 = DIGIT0 + MOD  # 7
VOCAB = OP0 + N_OPS  # 25

WEIGHTS = np.asarray([1, 2, 3], dtype=np.int64)


def checksum(state: np.ndarray) -> np.ndarray:
    """state [..., 3] -> checksum in 0..MOD-1."""
    return (state * WEIGHTS).sum(axis=-1) % MOD


def apply_ops(state: np.ndarray, ops: np.ndarray) -> np.ndarray:
    """Apply one op per row. `state` is [N, 3], `ops` is [N] in 0..N_OPS-1."""
    s = np.array(state, dtype=np.int64, copy=True)
    ops = np.asarray(ops, dtype=np.int64)
    add = ops < N_ADD
    reg = ops % N_REG
    delta = ops // N_REG + 1
    for r in range(N_REG):
        m = add & (reg == r)
        s[m, r] = (s[m, r] + delta[m]) % MOD
    for r in range(N_REG):
        m = ops == (N_ADD + r)
        s[m, r] = (s[m, r] * 3 + 1) % MOD
    for r in range(N_REG):
        m = ops == (N_ADD + N_MUL + r)
        others = [i for i in range(N_REG) if i != r]
        s[m, r] = (s[m, r] + s[m, others[0]] * s[m, others[1]]) % MOD
    return s


def trace_length(n_rounds: int) -> int:
    return 1 + 3 * int(n_rounds)


def generate_traces(n: int, n_rounds: int, rng: np.random.Generator) -> np.ndarray:
    """Return int64 token ids [n, 1 + 3 * n_rounds]."""
    if n_rounds < 1:
        raise ValueError("n_rounds must be positive")
    length = trace_length(n_rounds)
    tokens = np.empty((n, length), dtype=np.int64)
    tokens[:, 0] = BOS_ID
    state = np.zeros((n, N_REG), dtype=np.int64)
    for rnd in range(n_rounds):
        ops = rng.integers(0, N_OPS, size=n, dtype=np.int64)
        state = apply_ops(state, ops)
        base = 1 + 3 * rnd
        tokens[:, base] = OP0 + ops
        tokens[:, base + 1] = Q_ID
        tokens[:, base + 2] = DIGIT0 + checksum(state)
    return tokens


def lm_batch(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal LM split. Answer mask is 1 where the input token is Q."""
    inp = tokens[:, :-1]
    tgt = tokens[:, 1:]
    answer = (inp == Q_ID).astype(np.float64)
    return inp, tgt, answer


def op_name(token_id: int) -> str:
    if token_id == BOS_ID:
        return "BOS"
    if token_id == Q_ID:
        return "Q"
    if DIGIT0 <= token_id < OP0:
        return str(token_id - DIGIT0)
    op = token_id - OP0
    if op < 0 or op >= N_OPS:
        return "?"
    if op < N_ADD:
        return f"ADD r{op % N_REG}+{op // N_REG + 1}"
    if op < N_ADD + N_MUL:
        return f"MUL r{op - N_ADD}"
    return f"CROSS r{op - N_ADD - N_MUL}"


def trace_rounds(tokens: np.ndarray) -> list[dict[str, int | str]]:
    """One sequence [T] -> per-round op name and true digit."""
    rounds: list[dict[str, int | str]] = []
    t = 1
    while t + 2 < tokens.shape[0]:
        rounds.append(
            {
                "op": op_name(int(tokens[t])),
                "truth": int(tokens[t + 2] - DIGIT0),
            }
        )
        t += 3
    return rounds
