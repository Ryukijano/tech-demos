"""Checks for the register trace and the three feedback modes."""

from __future__ import annotations

import math

import numpy as np
import torch

from model import build_model, build_trio
from task import (
    MOD,
    N_OPS,
    N_REG,
    OP0,
    Q_ID,
    VOCAB,
    apply_ops,
    checksum,
    generate_traces,
    lm_batch,
)


def _reference_state(ops: list[int]) -> np.ndarray:
    state = np.zeros((1, N_REG), dtype=np.int64)
    for op in ops:
        state = apply_ops(state, np.asarray([op], dtype=np.int64))
    return state[0]


def test_trace_matches_reference() -> None:
    rng = np.random.default_rng(0)
    tokens = generate_traces(32, 6, rng)
    assert tokens.shape == (32, 1 + 18)
    for row in tokens:
        state = np.zeros(N_REG, dtype=np.int64)
        ops: list[int] = []
        t = 1
        while t + 2 < row.shape[0]:
            assert int(row[t + 1]) == Q_ID
            op = int(row[t]) - OP0
            assert 0 <= op < N_OPS
            ops.append(op)
            state = _reference_state(ops)
            assert int(row[t + 2]) == int(checksum(state)) + 2
            t += 3
    inp, tgt, answer = lm_batch(tokens)
    assert inp.shape == tgt.shape
    assert int(answer.sum()) == 32 * 6
    assert np.all(tgt[answer > 0.5] >= 2)
    assert np.all(tgt[answer > 0.5] < 2 + MOD)


def test_position_zero_ignores_feedback() -> None:
    tokens = torch.randint(0, VOCAB, (4, 19))
    for mode in ("alf", "fbt"):
        model = build_model(VOCAB, 32, 2, 4, 64, mode, 4, 0.01, 0)  # type: ignore[arg-type]
        one = model.forward(tokens, 1)
        two = model.forward(tokens, 2)
        if not torch.allclose(one[:, 0], two[:, 0], atol=1e-5):
            raise AssertionError(f"{mode} position 0 changed when a second pass was added")


def test_init_beta_is_uniform() -> None:
    model = build_model(VOCAB, 32, 2, 4, 64, "alf", 4, 0.01, 1)
    tokens = torch.randint(0, VOCAB, (3, 13))
    model.forward(tokens, 2)
    beta = model.beta_last
    if beta is None:
        raise AssertionError("ALF did not record depth weights")
    n_sub = beta.shape[0]
    for dest in range(n_sub):
        row = beta[dest, dest:]
        expected = 1.0 / row.numel()
        if not torch.allclose(row, torch.full_like(row, expected), atol=1e-5):
            raise AssertionError(f"dest {dest} beta {row.tolist()} expected {expected}")
        if dest > 0 and not torch.all(beta[dest, :dest] == 0):
            raise AssertionError("shallower sources received weight")
    share = torch.sigmoid(model.logit.detach())
    if not torch.allclose(share, torch.full_like(share, 0.01), atol=1e-6):
        raise AssertionError(f"initial share {share.tolist()}")


def test_params_match_and_loss_is_finite() -> None:
    models, counts = build_trio(VOCAB, 32, 4, 4, 128, 4, 0.01, 0)
    target = counts["alf"]
    for mode in ("vanilla", "fbt", "alf"):
        gap = abs(counts[mode] - target) / target
        if gap > 0.02:
            raise AssertionError(f"{mode} param gap {gap:.3f} counts={counts}")
    model = models["alf"]
    tokens = generate_traces(8, 4, np.random.default_rng(2))
    inp, tgt, _answer = lm_batch(tokens)
    loss, logits = model.loss(torch.from_numpy(inp), torch.from_numpy(tgt), 2)
    loss.backward()
    if not torch.isfinite(loss) or not torch.isfinite(logits).all():
        raise AssertionError("non-finite ALF loss")
    if model.w_e.weight.grad is None or model.query.grad is None:
        raise AssertionError("ALF feedback parameters received no gradient on a 2-pass loss")


def test_chance_line() -> None:
    if not math.isclose(math.log(MOD), 1.6094379124341003, rel_tol=0, abs_tol=1e-12):
        raise AssertionError("checksum alphabet is no longer 5-way")


def main() -> None:
    test_trace_matches_reference()
    test_position_zero_ignores_feedback()
    test_init_beta_is_uniform()
    test_params_match_and_loss_is_finite()
    test_chance_line()
    print("tests ok")


if __name__ == "__main__":
    main()
