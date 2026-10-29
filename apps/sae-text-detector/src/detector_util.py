"""Thin wrappers around the official detectorscope AuditableDetector."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def load_detector(hub_id: str, torch_threads: int = 4):
    torch.set_num_threads(int(torch_threads))
    from detectorscope import AuditableDetector

    return AuditableDetector.from_pretrained(hub_id)


@dataclass
class ScoreBundle:
    logits: np.ndarray
    probs: np.ndarray
    n_active: np.ndarray
    wall_s: float


def extract_features(
    detector, texts: list[str], batch_size: int = 4
) -> tuple[torch.Tensor, float]:
    """Max-pooled SAE feature matrix [n, d] on CPU, plus wall seconds."""
    chunks = []
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        x = detector.get_feature_vectors(batch).detach().cpu()
        chunks.append(x)
    wall = time.perf_counter() - t0
    return torch.cat(chunks, dim=0), wall


def score_cached(detector, x: torch.Tensor, head: str, tau: float) -> ScoreBundle:
    """Score cached features with a (head, tau) and count active features."""
    t0 = time.perf_counter()
    h = detector.load_head(head, tau)
    logits = h.predict(x).detach().cpu().float().numpy()
    # Importance / active count via zero-ablation scores when available.
    try:
        imp = h.get_importance_scores(x).detach().cpu().float().numpy()
        n_active = (np.abs(imp) > 0).sum(axis=1).astype(np.float64)
    except Exception:
        # Fallback: non-zero feature activations among head support.
        w = getattr(h, "weight", None)
        if w is None:
            n_active = np.full(len(logits), np.nan)
        else:
            ww = w.detach().cpu().float().numpy().reshape(-1)
            support = np.abs(ww) > 0
            acts = x.numpy()
            n_active = ((np.abs(acts) > 0) & support[None, :]).sum(axis=1).astype(
                np.float64
            )
    probs = 1.0 / (1.0 + np.exp(-logits))
    return ScoreBundle(logits, probs, n_active, time.perf_counter() - t0)


def metrics_for(y_true: np.ndarray, probs: np.ndarray, n_active: np.ndarray) -> dict:
    y = y_true.astype(int)
    auroc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else float("nan")
    pred = (probs >= 0.5).astype(int)
    return {
        "auroc": auroc,
        "accuracy": float(accuracy_score(y, pred)),
        "mean_n_active_features": float(np.nanmean(n_active)),
        "median_n_active_features": float(np.nanmedian(n_active)),
    }


def rescore_topk(
    detector, x: torch.Tensor, head: str, tau: float, k: int
) -> np.ndarray:
    """Logits using only the top-|importance| active features per row."""
    h = detector.load_head(head, tau)
    base = h.predict(x).detach().cpu()
    imp = h.get_importance_scores(x).detach().cpu()
    out = []
    for i in range(x.shape[0]):
        row_imp = imp[i]
        nz = (row_imp.abs() > 0).nonzero(as_tuple=False).flatten()
        if nz.numel() == 0 or k <= 0:
            # bias-only: ablate everything
            out.append(h.predict(torch.zeros_like(x[i : i + 1]))[0].item())
            continue
        order = torch.argsort(row_imp[nz].abs(), descending=True)
        keep = nz[order[:k]]
        mask = torch.zeros_like(x[i])
        mask[keep] = x[i, keep]
        out.append(h.predict(mask.unsqueeze(0))[0].item())
    _ = base  # base unused; kept for debugging parity
    return np.asarray(out, dtype=np.float64)


def explain_one(detector, text: str, head: str, tau: float, highlights: bool = True):
    """Return an official Detection for UI / media."""
    return detector.explain(
        detector.extract(text, highlights=highlights), head=head, tau=tau
    )
