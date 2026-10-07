# apps/embeddinggemma-2

On-device-style multimodal embeddings demo for [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (Google DeepMind).

This MVP is **text → 768-d cosine similarity** over a tiny local corpus, with an optional Vite UI.

## What it is

- Python FastAPI server loads `SentenceTransformer("google/embeddinggemma-2")` with
  `config_kwargs={"vision_config": None, "audio_config": None}` (text-only ~270M).
- If the HF model cannot download, the server falls back to a labeled **MOCK** hash embedder.
- Bun/Vite UI: paste a query, see ranked matches with cosine scores.

Sources: [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/), [model card](https://huggingface.co/google/embeddinggemma-2), Sentence Transformers docs.

## Prerequisites

- Python 3.10+
- Bun (for the web UI)
- ~1–2 GB disk for the text-only model weights (first run)
- Network access to Hugging Face on first download

Optional multimodal / MediaPipe LiteRT paths are heavier; this demo stays on text-only ST.

## How to run

```bash
cd apps/embeddinggemma-2
python3 -m venv .venv && source .venv/bin/activate   # or: pip install --user -r requirements.txt
pip install -r requirements.txt

# API (default http://127.0.0.1:43122) — loads google/embeddinggemma-2 text-only
python server/app.py

# CLI
python scripts/cli_search.py "What causes the northern lights?"

# Web UI (another shell)
cd web && bun install && bun run dev -- --host 127.0.0.1 --port 43123
```

Needs `pillow` + `torchvision` even for text-only (processor import chain). Force offline mock:

```bash
EMBED_FORCE_MOCK=1 python server/app.py
```

## Known limits

| Limit | Detail |
| --- | --- |
| Text-only MVP | Vision/audio/video encoders dropped via `config_kwargs` |
| First download | Needs HF access; gated? model is Apache-2.0 / public as of writing — still may need `hf auth login` in some envs |
| MOCK mode | Hash embeddings — not semantically meaningful; UI banners it |
| RAM | Text-only load is much lighter than full 740M multimodal |
| MediaPipe | LiteRT `.litertlm` path not wired in this MVP |

## Artifacts

Put UI screenshots / recordings in `artifacts/`.
