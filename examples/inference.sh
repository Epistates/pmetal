#!/bin/bash
# Inference Example
# Run text generation with a base model

set -e

MODEL="qwen/Qwen3-0.6B-Base"
PROMPT="Explain the concept of machine learning in simple terms."

echo "=== PMetal Inference ==="
echo "Model: $MODEL"
echo ""

./target/release/pmetal infer \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --max-tokens 256 \
    --temperature 0.7 \
    --top-p 0.9
