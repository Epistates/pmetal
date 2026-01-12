#!/bin/bash
# QLoRA Fine-tuning Example
# Fine-tune with 4-bit quantized base weights and LoRA adapters

set -e

# Configuration
MODEL="unsloth/Llama-3.2-3B-bnb-4bit"
DATASET="./examples/sample_dataset.jsonl"
OUTPUT="./output/qlora_finetune"

# Training hyperparameters
LORA_R=32
LORA_ALPHA=64
BATCH_SIZE=2
LEARNING_RATE=1e-4
EPOCHS=1
MAX_SEQ_LEN=4096

echo "=== PMetal QLoRA Fine-tuning ==="
echo "Model: $MODEL (4-bit quantized)"
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
    --use-sequence-packing \
    --gradient-checkpointing

echo ""
echo "Training complete! Adapter saved to: $OUTPUT"
