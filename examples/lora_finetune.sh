#!/bin/bash
# LoRA Fine-tuning Example
# Fine-tune a model using Low-Rank Adaptation

set -e

# Configuration
MODEL="qwen/Qwen3-0.6B-Base"
DATASET="./examples/sample_dataset.jsonl"
OUTPUT="./output/lora_finetune"

# Training hyperparameters
LORA_R=16
LORA_ALPHA=32
BATCH_SIZE=4
LEARNING_RATE=2e-4
EPOCHS=1
MAX_SEQ_LEN=2048

echo "=== PMetal LoRA Fine-tuning ==="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "LoRA rank: $LORA_R"
echo ""

./target/release/pmetal train \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --max-seq-len $MAX_SEQ_LEN \
    --use-metal-flash-attention \
    --use-sequence-packing

echo ""
echo "Training complete! Adapter saved to: $OUTPUT"
echo "Run inference with: ./examples/inference_lora.sh"
