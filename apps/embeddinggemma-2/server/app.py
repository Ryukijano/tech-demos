#!/usr/bin/env python3
"""EmbeddingGemma 2 text similarity API.

Loads SentenceTransformer("google/embeddinggemma-2") with vision/audio dropped
for a smaller text-only footprint. Falls back to a deterministic hash embedder
if the real model cannot be downloaded (clearly labeled MOCK).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "corpus.json"
HOST = os.environ.get("EMBED_HOST", "127.0.0.1")
PORT = int(os.environ.get("EMBED_PORT", "43122"))
FORCE_MOCK = os.environ.get("EMBED_FORCE_MOCK", "").lower() in {"1", "true", "yes"}
MODEL_ID = os.environ.get("EMBED_MODEL_ID", "google/embeddinggemma-2")

app = FastAPI(title="EmbeddingGemma 2 demo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class Hit(BaseModel):
    id: str
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    mode: str
    model: str
    dim: int
    hits: list[Hit]


def load_corpus() -> list[dict[str, str]]:
    return json.loads(CORPUS_PATH.read_text())


def mock_embed(texts: list[str], dim: int = 768) -> np.ndarray:
    """Deterministic bag-of-bytes embedding for offline demos."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hashlib.sha256(t.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        # Mix token hashes for a bit of lexical signal
        vec = rng.standard_normal(dim).astype(np.float32) * 0.15
        for tok in t.lower().split():
            th = hashlib.md5(tok.encode()).digest()
            idx = int.from_bytes(th[:4], "little") % dim
            vec[idx] += 1.0
            vec[(idx * 7) % dim] += 0.5
        n = np.linalg.norm(vec) + 1e-9
        out[i] = vec / n
    return out


class Embedder:
    def __init__(self) -> None:
        self.mode = "mock"
        self.model_name = "mock/hash-768"
        self.dim = 768
        self._st = None
        if FORCE_MOCK:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(
                MODEL_ID,
                config_kwargs={"vision_config": None, "audio_config": None},
            )
            self.mode = "live"
            self.model_name = MODEL_ID
            # probe dim
            probe = self._st.encode(["ping"], prompt_name="Document")
            self.dim = int(np.asarray(probe).shape[-1])
        except Exception as exc:  # noqa: BLE001
            print(f"[embed] falling back to MOCK: {exc}")
            self.mode = "mock"
            self.model_name = "mock/hash-768"
            self._st = None

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if self._st is None:
            return mock_embed(texts, self.dim)
        return np.asarray(
            self._st.encode(texts, prompt_name="Document", normalize_embeddings=True),
            dtype=np.float32,
        )

    def encode_query(self, text: str) -> np.ndarray:
        if self._st is None:
            return mock_embed([text], self.dim)[0]
        return np.asarray(
            self._st.encode([text], prompt_name="SearchQuery", normalize_embeddings=True),
            dtype=np.float32,
        )[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


@lru_cache(maxsize=1)
def get_doc_matrix() -> tuple[list[dict[str, str]], np.ndarray]:
    corpus = load_corpus()
    docs = [
        f"title: {row['title']} | text: {row['text']}"
        for row in corpus
    ]
    mat = get_embedder().encode_documents(docs)
    return corpus, mat


def cosine_rank(query_vec: np.ndarray, doc_mat: np.ndarray) -> np.ndarray:
    # vectors are L2-normalized → cosine = dot
    return doc_mat @ query_vec


@app.get("/health")
def health() -> dict[str, Any]:
    emb = get_embedder()
    return {
        "ok": True,
        "mode": emb.mode,
        "model": emb.model_name,
        "dim": emb.dim,
        "corpus_size": len(load_corpus()),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    emb = get_embedder()
    corpus, mat = get_doc_matrix()
    q = emb.encode_query(req.query)
    scores = cosine_rank(q, mat)
    order = np.argsort(-scores)[: req.top_k]
    hits = [
        Hit(
            id=corpus[i]["id"],
            title=corpus[i]["title"],
            text=corpus[i]["text"],
            score=float(scores[i]),
        )
        for i in order
    ]
    return SearchResponse(
        query=req.query,
        mode=emb.mode,
        model=emb.model_name,
        dim=emb.dim,
        hits=hits,
    )


@app.get("/corpus")
def corpus() -> list[dict[str, str]]:
    return load_corpus()


def main() -> None:
    import uvicorn

    # Warm embedder at startup so first request is snappier
    print(f"Loading embedder (FORCE_MOCK={FORCE_MOCK})…")
    emb = get_embedder()
    print(f"mode={emb.mode} model={emb.model_name} dim={emb.dim}")
    get_doc_matrix()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
