#!/usr/bin/env bash
# LoRA Inference Example
# Run text generation with a fine-tuned LoRA adapter.
# LORA_PATH is the adapter directory `pmetal train` wrote (it holds
# adapter_model.safetensors and adapter_config.json).

set -euo pipefail

PMETAL_BIN="${PMETAL_BIN:-./target/release/pmetal}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LORA_PATH="${LORA_PATH:-./output/lora_finetune}"
PROMPT="${PROMPT:-What are the benefits of machine learning?}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"

echo "=== PMetal LoRA Inference ==="
echo "Model: $MODEL"
echo "LoRA: $LORA_PATH"
echo ""

"$PMETAL_BIN" infer \
    --model "$MODEL" \
    --lora "$LORA_PATH" \
    --prompt "$PROMPT" \
    --chat \
    --no-thinking \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P"
