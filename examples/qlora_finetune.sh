#!/usr/bin/env bash
# QLoRA Fine-tuning Example
# Fine-tune with 4-bit quantized base weights and LoRA adapters

set -euo pipefail

# Configuration
PMETAL_BIN="${PMETAL_BIN:-./target/release/pmetal}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
DATASET="${DATASET:-./examples/sample_dataset.jsonl}"
OUTPUT="${OUTPUT:-./output/qlora_finetune}"

# Training hyperparameters
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
EPOCHS="${EPOCHS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"

echo "=== PMetal QLoRA Fine-tuning ==="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "LoRA rank: $LORA_R"
echo ""

# Sequence Packing and Gradient Checkpointing are ENABLED BY DEFAULT.
# QLoRA requires specifying the quantization method (nf4, fp4, or int8).

"$PMETAL_BIN" train \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --quantization nf4 \
    --double-quant

echo ""
echo "Training complete! Adapter saved to: $OUTPUT"
echo "Run inference with: LORA_PATH=\"$OUTPUT\" ./examples/inference_lora.sh"
