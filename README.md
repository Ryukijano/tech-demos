# Sticky tech-demos

**[View on GitHub Pages →](https://ryukijano.github.io/tech-demos/)**

One monorepo for sticky / bookmark-worthy tech demos. Prefer adding a new `apps/<slug>/` forever — do **not** create one repo per demo.

## Layout

| Path | What |
| --- | --- |
| `apps/llamacpp-decision-models/` | Decision Models UI against llama.cpp `POST /v1/systemone` (mock offline) |
| `apps/jepa-anything/` | JEPA-Anything Colab notebook + dry-run / light-checkpoint scripts |
| `apps/pixelumm/` | PixelUMM image-understanding kit (GPU + noncommercial weights) |
| `apps/embeddinggemma-2/` | EmbeddingGemma 2 text similarity search (Python API + Vite UI) |
| `tracking/seen-bookmarks.json` | Stub bookmark tracker (empty array) |
| `docs/` | GitHub Pages showcase site (static HTML/CSS) |
| `AGENTS.md` | Rules for future cloud agents |
| `bunfig.toml` | Bun install policy (`minimumReleaseAge = 259200`) |

## Prerequisites

- [Bun](https://bun.sh) for the JS/TS UIs
- Python 3.10+ for EmbeddingGemma, JEPA, and PixelUMM scripts
- Optional: recent [llama.cpp](https://github.com/ggml-org/llama.cpp) / `llama` CLI for a real Decision Model backend
- Optional: NVIDIA GPU + large disk for JEPA domain checkpoints and PixelUMM (~15B params)

## Run each app

### 1. Decision Models UI

```bash
cd apps/llamacpp-decision-models
bun install
bun run dev -- --host 127.0.0.1 --port 43121
```

Open http://127.0.0.1:43121 — mock mode works with no GPU. For a real backend see `apps/llamacpp-decision-models/scripts/serve-example.sh`.

### 2. JEPA-Anything

```bash
cd apps/jepa-anything
python3 scripts/dry_run.py          # no weights required
# With GPU / HF download:
python3 scripts/run_light_inference.py --download
```

Colab: open `apps/jepa-anything/notebooks/jepa_anything_demo.ipynb`.

### 3. PixelUMM

```bash
cd apps/pixelumm
python3 scripts/smoke_test.py       # fails gracefully without GPU/weights
# Full image understanding (GPU + checkpoint):
bash scripts/run_image_vlm.sh
```

### 4. EmbeddingGemma 2

```bash
cd apps/embeddinggemma-2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python server/app.py                # API on :43122
# In another shell:
cd web && bun install && bun run dev -- --host 127.0.0.1 --port 43123
```

Open http://127.0.0.1:43123 — paste a query and rank a tiny local corpus.

## Agent notes

See [AGENTS.md](./AGENTS.md). Screenshot + video artifacts belong with each runnable UI under `apps/<slug>/artifacts/`.
