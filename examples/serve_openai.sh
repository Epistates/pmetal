#!/usr/bin/env bash
# OpenAI-compatible server example.

set -euo pipefail

PMETAL_BIN="${PMETAL_BIN:-./target/release/pmetal}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

echo "=== PMetal OpenAI-Compatible Server ==="
echo "Model: $MODEL"
echo "Address: http://$HOST:$PORT"
echo ""

if ! "$PMETAL_BIN" serve --help >/dev/null 2>&1; then
    echo "This pmetal binary was built without the serve feature." >&2
    echo "Build it with: cargo build -p pmetal --release --features serve,mcp" >&2
    exit 1
fi

exec "$PMETAL_BIN" serve \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --continuous-batch
