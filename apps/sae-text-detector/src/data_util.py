"""Load and subsample the public human-vs-AI text dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasets import load_dataset


@dataclass(frozen=True)
class Example:
    text: str
    label: int  # 1 = AI, 0 = human
    pair_id: int
    source: str


def _truncate(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def load_pair_pool(
    dataset: str,
    pool_pairs: int,
    pool_seed: int,
    min_chars: int,
    max_chars: int,
) -> list[Example]:
    """Download dataset and return a fixed pool of balanced (human, AI) examples.

    Scans the train split in order, keeping every pair that clears ``min_chars``.
    The pool is then a ``pool_seed`` subsample of size ``pool_pairs`` from that
    filtered list (deterministic given the dataset revision).
    """
    # Stream the train split so we never materialize 1M rows as a DataFrame.
    # Deterministic: every qualifying row in order is a candidate; pool_seed then
    # draws ``pool_pairs`` of them.
    ds = load_dataset(dataset, split="train", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        if "human_text" not in row or "ai_text" not in row:
            raise ValueError(
                f"{dataset} missing human_text/ai_text; keys={list(row.keys())}"
            )
        h = " ".join(str(row["human_text"]).split())
        a = " ".join(str(row["ai_text"]).split())
        if len(h) < min_chars or len(a) < min_chars:
            continue
        rows.append((i, h, a))
        # Early stop once we have a large enough candidate set for stable draws.
        if len(rows) >= max(5000, pool_pairs * 12):
            break

    rng = np.random.default_rng(pool_seed)
    if len(rows) < pool_pairs:
        raise RuntimeError(f"only {len(rows)} usable pairs; need {pool_pairs}")
    pick = rng.choice(len(rows), size=pool_pairs, replace=False)
    examples: list[Example] = []
    for idx in sorted(int(x) for x in pick):
        pair_id, h, a = rows[idx]
        examples.append(
            Example(_truncate(h, max_chars), 0, pair_id, f"pair{pair_id}:human")
        )
        examples.append(
            Example(_truncate(a, max_chars), 1, pair_id, f"pair{pair_id}:ai")
        )
    return examples


def stratified_subsample(
    pool: list[Example], n_per_class: int, seed: int
) -> list[Example]:
    """Draw n_per_class human and AI examples without replacement."""
    rng = np.random.default_rng(seed)
    humans = [e for e in pool if e.label == 0]
    ais = [e for e in pool if e.label == 1]
    if len(humans) < n_per_class or len(ais) < n_per_class:
        raise RuntimeError("pool too small for requested n_per_class")
    h_idx = rng.choice(len(humans), size=n_per_class, replace=False)
    a_idx = rng.choice(len(ais), size=n_per_class, replace=False)
    out = [humans[i] for i in sorted(int(x) for x in h_idx)]
    out += [ais[i] for i in sorted(int(x) for x in a_idx)]
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def examples_frame(examples: list[Example]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [e.text for e in examples],
            "label": [e.label for e in examples],
            "pair_id": [e.pair_id for e in examples],
            "source": [e.source for e in examples],
            "n_chars": [len(e.text) for e in examples],
        }
    )
