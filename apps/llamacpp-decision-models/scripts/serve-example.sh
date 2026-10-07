#!/usr/bin/env bash
# Start a Decision Model with llama.cpp (prefer Julia-1 — 144M, fastest for demos).
# Requires a recent llama.cpp build with /v1/systemone (PR #29818+).
set -euo pipefail

MODEL="${MODEL:-ggml-org/Julia-1-GGUF:Q8_0}"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"

echo "Serving ${MODEL} on http://${HOST}:${PORT}"
echo "Docs: https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp"
echo

# Prefer `llama serve` (llama.app / recent CLI). Fallbacks for older installs.
if command -v llama >/dev/null 2>&1; then
  exec llama serve -hf "${MODEL}" --host "${HOST}" --port "${PORT}"
elif command -v llama-server >/dev/null 2>&1; then
  exec llama-server -hf "${MODEL}" --host "${HOST}" --port "${PORT}"
else
  cat <<'EOF'
Neither `llama` nor `llama-server` was found on PATH.

Install a recent llama.cpp / llama.app build that includes Decision Models, then re-run:

  MODEL=ggml-org/Julia-1-GGUF:Q8_0 ./scripts/serve-example.sh

Alternatives:
  llama serve -hf ggml-org/Julia-1-GGUF:Q8_0
  llama serve -hf ggml-org/Kev-4B-GGUF

Curl example (after the server is up):

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
        },
        "angry": { "type": "noul", "instructions": "Is the customer angry?" },
        "urgency": {
          "type": "score",
          "instructions": "How urgent is this?",
          "criteria": ["can wait", "this week", "today", "right now"]
        }
      }
    }'
EOF
  exit 127
fi
