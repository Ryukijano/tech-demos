"""Train and score one feedback transformer on register traces."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from model import FeedbackTransformer, Mode
from task import DIGIT0, MOD, VOCAB, generate_traces, lm_batch, trace_rounds


def eval_passes(mode: Mode, configured: int) -> int:
    if mode == "vanilla":
        return 1
    if mode in ("fbt", "alf"):
        return configured
    raise AssertionError(mode)


def train_passes(mode: Mode, step: int, steps: int, branch_fraction: float) -> int:
    if mode == "vanilla":
        return 1
    if mode in ("fbt", "alf"):
        branch_at = int(branch_fraction * steps)
        return 1 if step < branch_at else 2
    raise AssertionError(mode)


def make_optimizer(model: FeedbackTransformer, lr: float) -> torch.optim.Adam:
    groups: list[dict[str, Any]] = [{"params": model.trunk_parameters(), "lr": lr}]
    extra = [p for p in model.extra_parameters() if p.requires_grad]
    if model.mode == "alf":
        rest = [p for p in extra if p is not model.w_e.weight]
        if rest:
            groups.append({"params": rest, "lr": lr})
        groups.append({"params": [model.w_e.weight], "lr": lr / model.d})
    elif extra:
        groups.append({"params": extra, "lr": lr})
    return torch.optim.Adam(groups)


@torch.inference_mode()
def evaluate(
    model: FeedbackTransformer,
    tokens: np.ndarray,
    n_passes: int,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    nll_sum = 0.0
    token_count = 0
    answer_nll = 0.0
    answer_count = 0.0
    answer_correct = 0.0
    beta_acc: torch.Tensor | None = None
    beta_batches = 0
    first_probs: list[list[float]] | None = None
    first_pred: list[int] | None = None
    for start in range(0, tokens.shape[0], batch_size):
        batch = tokens[start : start + batch_size]
        inp, tgt, answer = lm_batch(batch)
        inp_t = torch.from_numpy(np.ascontiguousarray(inp))
        tgt_t = torch.from_numpy(np.ascontiguousarray(tgt))
        logits = model.forward(inp_t, n_passes)
        log_prob = F.log_softmax(logits, dim=-1)
        nll = -log_prob.gather(-1, tgt_t.unsqueeze(-1)).squeeze(-1)
        nll_sum += float(nll.sum())
        token_count += int(nll.numel())
        weight = torch.from_numpy(np.ascontiguousarray(answer))
        answer_nll += float((nll * weight).sum())
        answer_count += float(weight.sum())
        pred = logits.argmax(dim=-1)
        answer_correct += float(((pred == tgt_t).float() * weight).sum())
        if model.beta_last is not None:
            beta_acc = model.beta_last.float().cpu() if beta_acc is None else beta_acc + model.beta_last.float().cpu()
            beta_batches += 1
        if first_probs is None:
            probs = torch.softmax(logits[0], dim=-1)
            positions = np.flatnonzero(answer[0] > 0.5)
            first_probs = probs[positions, DIGIT0 : DIGIT0 + MOD].cpu().tolist()
            first_pred = [int(v) - DIGIT0 for v in pred[0, positions].cpu().tolist()]
    if answer_count <= 0:
        raise RuntimeError("evaluation batch has no Q positions")
    beta = None
    if beta_acc is not None and beta_batches > 0:
        beta = (beta_acc / beta_batches).tolist()
    model.train()
    return {
        "answer_nats": answer_nll / answer_count,
        "answer_acc": answer_correct / answer_count,
        "full_nats": nll_sum / token_count,
        "n_answer": int(answer_count),
        "beta": beta,
        "demo_probs": first_probs,
        "demo_pred": first_pred,
    }


def train_model(
    model: FeedbackTransformer,
    batches: list[np.ndarray],
    probe: np.ndarray,
    eval_tokens: np.ndarray,
    lr: float,
    grad_clip: float,
    log_every: int,
    branch_fraction: float,
    eval_pass_count: int,
    batch_size: int,
) -> dict[str, Any]:
    opt = make_optimizer(model, lr)
    curve: list[dict[str, float]] = []
    steps = len(batches)
    passes = eval_passes(model.mode, eval_pass_count)
    t0 = time.perf_counter()
    for step, tokens in enumerate(batches):
        n_pass = train_passes(model.mode, step, steps, branch_fraction)
        inp, tgt, _answer = lm_batch(tokens)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, _logits = model.loss(
            torch.from_numpy(np.ascontiguousarray(inp)),
            torch.from_numpy(np.ascontiguousarray(tgt)),
            n_pass,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"{model.mode} non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trunk_parameters() + model.extra_parameters(), grad_clip)
        opt.step()
        if step % log_every == 0 or step + 1 == steps:
            scored = evaluate(model, probe, passes, batch_size)
            curve.append(
                {
                    "step": step + 1,
                    "answer_nats": scored["answer_nats"],
                    "answer_acc": scored["answer_acc"],
                    "full_nats": scored["full_nats"],
                    "train_loss": float(loss.detach()),
                    "train_passes": n_pass,
                }
            )
            print(
                f"  {model.mode:7s} step {step + 1:4d}/{steps} "
                f"train {float(loss.detach()):.3f}  probe answer nats {scored['answer_nats']:.3f} "
                f"acc {scored['answer_acc']:.3f}",
                flush=True,
            )
    final = evaluate(model, eval_tokens, passes, batch_size)
    wall = time.perf_counter() - t0
    final_public = {k: v for k, v in final.items() if k not in ("demo_probs", "demo_pred", "beta")}
    print(
        f"  {model.mode:7s} held-out answer nats {final['answer_nats']:.4f} "
        f"acc {final['answer_acc']:.4f}  ({wall:.1f}s)",
        flush=True,
    )
    return {
        "curve": curve,
        "final": final_public,
        "beta": final["beta"],
        "demo_probs": final["demo_probs"],
        "demo_pred": final["demo_pred"],
        "wall_s": wall,
        "n_params": model.n_parameters(),
        "d_ff": int(model.blocks[0].ff.up.out_features),
    }


def make_batches(n_batches: int, batch_size: int, n_rounds: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [generate_traces(batch_size, n_rounds, rng) for _ in range(n_batches)]


def heldout(n: int, n_rounds: int, seed: int) -> np.ndarray:
    return generate_traces(n, n_rounds, np.random.default_rng(seed))


def demo_record(tokens_row: np.ndarray, by_mode: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rounds = trace_rounds(tokens_row)
    packed = []
    for index, rnd in enumerate(rounds):
        packed.append(
            {
                "op": rnd["op"],
                "truth": rnd["truth"],
                "probs": {mode: by_mode[mode]["demo_probs"][index] for mode in by_mode},
                "pred": {mode: by_mode[mode]["demo_pred"][index] for mode in by_mode},
            }
        )
    return {"tokens": tokens_row.tolist(), "rounds": packed, "vocab": VOCAB}
