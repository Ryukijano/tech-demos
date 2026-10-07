# apps/llamacpp-decision-models

Web UI for [Decision Models in llama.cpp](https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp).

You send a **state** plus typed questions (`choice` / `score` / `noul`) to `POST /v1/systemone`. The model returns option probabilities in one forward pass.

## What this ships

- Bun + Vite + React UI (typed state → probability bars)
- Configurable base URL (default `http://127.0.0.1:8080`)
- **MOCK mode** (default on) so the UI demos without a GPU / server
- Auto-fallback to MOCK if the live server is down
- `scripts/serve-example.sh` with Julia-1 serve + curl examples

## Prerequisites

- [Bun](https://bun.sh)
- Optional: recent llama.cpp / `llama` CLI with Decision Models (`/v1/systemone`)

## How to run (UI)

```bash
cd apps/llamacpp-decision-models
bun install
bun run dev -- --host 127.0.0.1 --port 43121
```

Open http://127.0.0.1:43121 — leave **Force MOCK mode** checked for offline demos.

## How to run (real backend)

Prefer the smallest model for demos:

```bash
# Julia-1 — 144M, ~3 ms/question on high-end GPU, Apache-2.0
./scripts/serve-example.sh
# or:
llama serve -hf ggml-org/Julia-1-GGUF:Q8_0
```

Then uncheck MOCK in the UI (or set base URL to your server).

Curl:

```bash
curl http://127.0.0.1:8080/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ggml-org/Julia-1-GGUF:Q8_0",
    "state": "Customer message: I was charged twice for my order last week and nobody has replied.",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "payments, charges, refunds, invoices",
          "shipping": "delivery, tracking, lost or late parcels",
          "technical": "bugs, errors, login problems"
        }
      }
    }
  }'
```

Other GGUFs: `ggml-org/Laya-GGUF`, `ggml-org/Kev-4B-GGUF`, `ggml-org/lev-GGUF`, `ggml-org/OpenJev-GGUF` (vision, CC BY-NC).

## Known limits

| Limit | Detail |
| --- | --- |
| MOCK ≠ model | Offline scores are keyword heuristics, clearly labeled **MOCK** |
| Julia-1 | Text only, 144M — describe options; bare labels can misroute |
| Server | Needs a llama.cpp build with `/v1/systemone` (PR #29818+) |
| Vision | OpenJev / Clef only; this UI is text-state for the MVP |
| GPU/RAM | Julia-1 is CPU-friendly; Kev/lev/OpenJev need more VRAM |

## Artifacts

Screenshots / recordings of the running UI live in `artifacts/`.
