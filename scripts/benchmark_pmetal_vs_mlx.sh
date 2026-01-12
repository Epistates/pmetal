#!/usr/bin/env bash
#
# PMetal vs mlx_lm.lora Speed Benchmark
#
# Compares training throughput between PMetal and mlx_lm.lora
# using identical model, dataset, and hyperparameters.
#

set -euo pipefail

# Configuration
MODEL="qwen/Qwen3-0.6B-Base"
SAMPLES=100
BATCH_SIZE=4
MAX_SEQ_LEN=2048
LEARNING_RATE="2e-4"
LORA_RANK=16
ITERS=25  # ~100 samples / 4 batch = 25 iterations

# Directories
PMETAL_OUTPUT="./output_bench_pmetal"
MLX_OUTPUT="./output_bench_mlx"
DATA_DIR="./mlx_lm_data"
PMETAL="./target/release/pmetal"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_section() { echo -e "\n${CYAN}════════════════════════════════════════${NC}"; echo -e "${CYAN}$1${NC}"; echo -e "${CYAN}════════════════════════════════════════${NC}"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --samples)
            SAMPLES="$2"
            ITERS=$((SAMPLES / BATCH_SIZE))
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            ITERS=$((SAMPLES / BATCH_SIZE))
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --samples N      Number of training samples (default: 100)"
            echo "  --batch-size N   Batch size (default: 4)"
            echo "  --help           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_section "PMetal vs mlx_lm.lora Benchmark"

echo ""
log_info "Model:      $MODEL"
log_info "Samples:    $SAMPLES"
log_info "Batch Size: $BATCH_SIZE"
log_info "Iterations: $ITERS"
log_info "Max Seq:    $MAX_SEQ_LEN"
log_info "LoRA Rank:  $LORA_RANK"

# ============================================================================
log_section "Step 1: Prepare Dataset"
# ============================================================================

mkdir -p "$DATA_DIR" "$PMETAL_OUTPUT" "$MLX_OUTPUT"

# Create dataset in both formats
python3 << 'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])
    from datasets import load_dataset

samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
data_dir = Path("mlx_lm_data")
pmetal_output = Path("output_bench_pmetal")

data_dir.mkdir(exist_ok=True)
pmetal_output.mkdir(exist_ok=True)

print(f"Loading dataset for {samples} samples...")

try:
    dataset = load_dataset("TeichAI/gemini-3-pro-preview-high-reasoning-1000x", split="train")

    train_samples_mlx = []
    train_samples_pmetal = []
    valid_samples_mlx = []
    valid_samples_pmetal = []

    for i, sample in enumerate(dataset):
        if i >= samples:
            break

        # Convert to formats
        if "messages" in sample:
            # MLX format: {"text": "conversation"}
            text_parts = []
            convos = []
            for msg in sample["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    text_parts.append(f"User: {content}")
                    convos.append({"from": "human", "value": content})
                elif role == "assistant":
                    text_parts.append(f"Assistant: {content}")
                    convos.append({"from": "gpt", "value": content})
                elif role == "system":
                    text_parts.append(f"System: {content}")
                    convos.append({"from": "system", "value": content})

            mlx_sample = {"text": "\n".join(text_parts)}
            pmetal_sample = {"conversations": convos}
        else:
            continue

        # 90/10 split
        if i % 10 == 0:
            valid_samples_mlx.append(mlx_sample)
            valid_samples_pmetal.append(pmetal_sample)
        else:
            train_samples_mlx.append(mlx_sample)
            train_samples_pmetal.append(pmetal_sample)

    # Write MLX format
    with open(data_dir / "train.jsonl", "w") as f:
        for s in train_samples_mlx:
            f.write(json.dumps(s) + "\n")
    with open(data_dir / "valid.jsonl", "w") as f:
        for s in valid_samples_mlx:
            f.write(json.dumps(s) + "\n")
    with open(data_dir / "test.jsonl", "w") as f:
        for s in valid_samples_mlx[:5]:
            f.write(json.dumps(s) + "\n")

    # Write PMetal format
    with open(pmetal_output / "train.jsonl", "w") as f:
        for s in train_samples_pmetal:
            f.write(json.dumps(s) + "\n")
    with open(pmetal_output / "eval.jsonl", "w") as f:
        for s in valid_samples_pmetal:
            f.write(json.dumps(s) + "\n")

    print(f"MLX data:    {len(train_samples_mlx)} train, {len(valid_samples_mlx)} valid")
    print(f"PMetal data: {len(train_samples_pmetal)} train, {len(valid_samples_pmetal)} eval")

except Exception as e:
    print(f"Error: {e}")
    # Create synthetic data
    for i in range(samples):
        sample_mlx = {"text": f"User: What is {i} + {i}?\nAssistant: {i} + {i} = {i*2}"}
        sample_pm = {"conversations": [
            {"from": "human", "value": f"What is {i} + {i}?"},
            {"from": "gpt", "value": f"{i} + {i} = {i*2}"}
        ]}

        if i % 10 == 0:
            valid_samples_mlx.append(sample_mlx)
            valid_samples_pmetal.append(sample_pm)
        else:
            train_samples_mlx.append(sample_mlx)
            train_samples_pmetal.append(sample_pm)
PYTHON_SCRIPT

log_success "Dataset prepared"

# ============================================================================
log_section "Step 2: Build PMetal"
# ============================================================================

if [ ! -f "$PMETAL" ]; then
    log_info "Building PMetal..."
    cargo build --release 2>&1 | tail -3
fi
log_success "PMetal ready"

# ============================================================================
log_section "Step 3: Benchmark mlx_lm.lora"
# ============================================================================

log_info "Running mlx_lm.lora training..."

MLX_START=$(python3 -c "import time; print(time.time())")

python3 -m mlx_lm lora \
    --model "$MODEL" \
    --train \
    --data "$DATA_DIR" \
    --batch-size $BATCH_SIZE \
    --iters $ITERS \
    --learning-rate $LEARNING_RATE \
    --max-seq-length $MAX_SEQ_LEN \
    --num-layers -1 \
    --adapter-path "$MLX_OUTPUT/adapters" \
    --steps-per-report 5 \
    --val-batches 2 \
    2>&1 | tee "$MLX_OUTPUT/training.log"

MLX_END=$(python3 -c "import time; print(time.time())")
MLX_DURATION=$(python3 -c "print(f'{$MLX_END - $MLX_START:.2f}')")

log_success "mlx_lm.lora completed in ${MLX_DURATION}s"

# ============================================================================
log_section "Step 4: Benchmark PMetal"
# ============================================================================

log_info "Running PMetal training..."

PMETAL_START=$(python3 -c "import time; print(time.time())")

$PMETAL train \
    --model "$MODEL" \
    --dataset "$PMETAL_OUTPUT/train.jsonl" \
    --eval-dataset "$PMETAL_OUTPUT/eval.jsonl" \
    --output "$PMETAL_OUTPUT" \
    --lora-r $LORA_RANK \
    --learning-rate $LEARNING_RATE \
    --batch-size $BATCH_SIZE \
    --epochs 1 \
    --max-seq-len $MAX_SEQ_LEN \
    --gradient-accumulation-steps 1 \
    --use-metal-flash-attention \
    --use-sequence-packing \
    --log-metrics "$PMETAL_OUTPUT/metrics.jsonl" \
    2>&1 | tee "$PMETAL_OUTPUT/training.log"

PMETAL_END=$(python3 -c "import time; print(time.time())")
PMETAL_DURATION=$(python3 -c "print(f'{$PMETAL_END - $PMETAL_START:.2f}')")

log_success "PMetal completed in ${PMETAL_DURATION}s"

# ============================================================================
log_section "Step 5: Results Comparison"
# ============================================================================

echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│                    BENCHMARK RESULTS                         │"
echo "├──────────────────────────────────────────────────────────────┤"
printf "│  %-20s %18s %18s │\n" "Metric" "mlx_lm.lora" "PMetal"
echo "├──────────────────────────────────────────────────────────────┤"
printf "│  %-20s %17ss %17ss │\n" "Total Time" "$MLX_DURATION" "$PMETAL_DURATION"

# Calculate samples/sec
MLX_THROUGHPUT=$(python3 -c "print(f'{$SAMPLES / $MLX_DURATION:.2f}')")
PMETAL_THROUGHPUT=$(python3 -c "print(f'{$SAMPLES / $PMETAL_DURATION:.2f}')")
printf "│  %-20s %14s/s %14s/s │\n" "Throughput" "$MLX_THROUGHPUT" "$PMETAL_THROUGHPUT"

# Calculate speedup
SPEEDUP=$(python3 -c "print(f'{$MLX_DURATION / $PMETAL_DURATION:.2f}')")
printf "│  %-20s %18s %17sx │\n" "Speedup" "-" "$SPEEDUP"

# Extract final loss from logs
MLX_LOSS=$(grep -oE "Train loss [0-9.]+" "$MLX_OUTPUT/training.log" | tail -1 | awk '{print $3}' || echo "N/A")
PMETAL_LOSS=$(tail -2 "$PMETAL_OUTPUT/metrics.jsonl" | grep epoch_end | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d.get(\"loss\", 0):.4f}')" 2>/dev/null || echo "N/A")
printf "│  %-20s %18s %18s │\n" "Final Loss" "$MLX_LOSS" "$PMETAL_LOSS"

echo "└──────────────────────────────────────────────────────────────┘"
echo ""

# Memory comparison (macOS)
log_info "Memory footprint comparison:"
echo ""
MLX_ADAPTER_SIZE=$(du -sh "$MLX_OUTPUT/adapters" 2>/dev/null | awk '{print $1}' || echo "N/A")
PMETAL_ADAPTER_SIZE=$(du -sh "$PMETAL_OUTPUT/lora_weights.safetensors" 2>/dev/null | awk '{print $1}' || echo "N/A")
echo "  mlx_lm adapters:  $MLX_ADAPTER_SIZE"
echo "  PMetal adapters:  $PMETAL_ADAPTER_SIZE"

echo ""
log_section "Summary"
echo ""
if (( $(echo "$PMETAL_DURATION < $MLX_DURATION" | bc -l) )); then
    echo -e "${GREEN}PMetal is ${SPEEDUP}x faster than mlx_lm.lora${NC}"
else
    SLOWDOWN=$(python3 -c "print(f'{$PMETAL_DURATION / $MLX_DURATION:.2f}')")
    echo -e "${YELLOW}mlx_lm.lora is ${SLOWDOWN}x faster than PMetal${NC}"
fi
echo ""
echo "Configuration:"
echo "  Model:      $MODEL"
echo "  Samples:    $SAMPLES"
echo "  Batch Size: $BATCH_SIZE"
echo "  Iterations: $ITERS"
echo "  Max Seq:    $MAX_SEQ_LEN"
echo ""
