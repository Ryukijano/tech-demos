#!/usr/bin/env python3
"""CLI similarity search against the local corpus (no web UI)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from app import SearchRequest, get_doc_matrix, get_embedder, cosine_rank, Hit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    emb = get_embedder()
    corpus, mat = get_doc_matrix()
    q = emb.encode_query(args.query)
    scores = cosine_rank(q, mat)
    order = scores.argsort()[::-1][: args.top_k]
    hits = [
        {
            "id": corpus[i]["id"],
            "title": corpus[i]["title"],
            "score": float(scores[i]),
            "text": corpus[i]["text"],
        }
        for i in order
    ]
    print(json.dumps({"mode": emb.mode, "model": emb.model_name, "hits": hits}, indent=2))


if __name__ == "__main__":
    main()
