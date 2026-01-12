#!/bin/bash
# LoRA Inference Example
# Run text generation with a fine-tuned LoRA adapter

set -e

MODEL="qwen/Qwen3-0.6B-Base"
LORA_PATH="./output/lora_finetune/lora_weights.safetensors"
PROMPT="What are the benefits of machine learning?"

echo "=== PMetal LoRA Inference ==="
echo "Model: $MODEL"
echo "LoRA: $LORA_PATH"
echo ""

./target/release/pmetal infer \
    --model "$MODEL" \
    --lora "$LORA_PATH" \
    --prompt "$PROMPT" \
    --max-tokens 256 \
    --temperature 0.7 \
    --top-p 0.9
