# pmetal-cli

Command-line interface for the PMetal framework.

## Overview

This crate provides the `pmetal` command-line tool for training and inference with LLMs on Apple Silicon.

## Installation

```bash
cargo install --path crates/pmetal-cli
```

Or build from source:

```bash
cargo build --release -p pmetal-cli
./target/release/pmetal --help
```

## Commands

### Train

Fine-tune a model with LoRA:

```bash
pmetal train \
  --model qwen/Qwen3-0.6B-Base \
  --dataset train.jsonl \
  --output ./output \
  --lora-r 16 \
  --batch-size 4 \
  --learning-rate 2e-4 \
  --epochs 1
```

#### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model ID or path | Required |
| `--dataset` | Training data (JSONL) | Required |
| `--output` | Output directory | Required |
| `--lora-r` | LoRA rank | 8 |
| `--lora-alpha` | LoRA alpha | 16.0 |
| `--batch-size` | Batch size | 4 |
| `--gradient-accumulation-steps` | Grad accumulation | 1 |
| `--learning-rate` | Learning rate | 2e-4 |
| `--epochs` | Training epochs | 1 |
| `--max-seq-len` | Max sequence length | 2048 |
| `--max-grad-norm` | Gradient clipping | 1.0 |
| `--use-metal-flash-attention` | Enable FlashAttention | false |
| `--use-sequence-packing` | Enable packing | false |
| `--gradient-checkpointing` | Save memory | false |
| `--resume` | Resume from checkpoint | false |

### Infer

Run inference with optional LoRA adapter:

```bash
pmetal infer \
  --model qwen/Qwen3-0.6B-Base \
  --lora ./output/lora_weights.safetensors \
  --prompt "What is machine learning?" \
  --max-tokens 256 \
  --temperature 0.7
```

#### Inference Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model ID or path | Required |
| `--lora` | LoRA adapter path | None |
| `--prompt` | Input prompt | Required |
| `--max-tokens` | Max tokens | 256 |
| `--temperature` | Sampling temp | Model default |
| `--top-k` | Top-k sampling | Model default |
| `--top-p` | Nucleus sampling | Model default |
| `--repetition-penalty` | Rep penalty | 1.0 |
| `--chat` | Use chat template | Auto |
| `--no-thinking` | Disable thinking | false |
| `--stream` | Stream output | false |

### Benchmark

Run performance benchmarks:

```bash
pmetal benchmark \
  --model qwen/Qwen3-0.6B-Base \
  --batch-size 4 \
  --seq-len 512
```

## Examples

### LoRA Training with Sequence Packing

```bash
pmetal train \
  --model unsloth/Llama-3.2-1B \
  --dataset conversations.jsonl \
  --output ./llama-lora \
  --lora-r 16 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-4 \
  --epochs 1 \
  --max-seq-len 2048 \
  --use-metal-flash-attention \
  --use-sequence-packing
```

### Interactive Chat

```bash
pmetal infer \
  --model qwen/Qwen3-0.6B-Base \
  --chat \
  --prompt "Hello! How are you?" \
  --max-tokens 512 \
  --temperature 0.7
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token |
| `RUST_LOG` | Log level (info, debug, trace) |

## License

MIT OR Apache-2.0
