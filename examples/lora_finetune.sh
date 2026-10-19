#!/usr/bin/env bash
# LoRA Fine-tuning Example
# Fine-tune a model using Low-Rank Adaptation

set -euo pipefail

# Configuration
PMETAL_BIN="${PMETAL_BIN:-./target/release/pmetal}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
DATASET="${DATASET:-./examples/sample_dataset.jsonl}"
OUTPUT="${OUTPUT:-./output/lora_finetune}"

# Training hyperparameters
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
EPOCHS="${EPOCHS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-0}" # 0 = auto-detect from model context size

echo "=== PMetal LoRA Fine-tuning ==="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "LoRA rank: $LORA_R"
echo ""

# Sequence Packing and Gradient Checkpointing are ENABLED BY DEFAULT.
# Use --no-sequence-packing or --no-gradient-checkpointing to disable.

"$PMETAL_BIN" train \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --max-seq-len "$MAX_SEQ_LEN"

echo ""
echo "Training complete! Adapter saved to: $OUTPUT"
echo "Run inference with: ./examples/inference_lora.sh"
