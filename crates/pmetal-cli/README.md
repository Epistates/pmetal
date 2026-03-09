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

# With ANE and dashboard support (default)
cargo build --release -p pmetal-cli --features "ane dashboard"
```

## Commands

### `train`

Fine-tune a model with LoRA or QLoRA:

```bash
pmetal train \
  --model Qwen/Qwen3-0.6B-Base \
  --dataset train.jsonl \
  --output ./output \
  --lora-r 16 \
  --batch-size 4 \
  --learning-rate 2e-4
```

#### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model ID or path | Required |
| `--dataset` | Training data (JSONL) | Required |
| `--output` | Output directory | `./output` |
| `--lora-r` | LoRA rank | 16 |
| `--lora-alpha` | LoRA alpha | 32.0 |
| `--batch-size` | Micro-batch size | 1 |
| `--gradient-accumulation-steps` | Grad accumulation | 4 |
| `--learning-rate` | Learning rate | 2e-4 |
| `--epochs` | Training epochs | 1 |
| `--max-seq-len` | Max sequence length | 0 (Auto) |
| `--no-flash-attention` | Disable FlashAttention | false |
| `--no-sequence-packing` | Disable packing | false |
| `--no-gradient-checkpointing` | Disable memory savings | false |
| `--quantization` | QLoRA method (nf4, fp4, int8) | none |
| `--no-ane` | Disable Apple Neural Engine | false |

### `infer`

Run inference with optional LoRA adapter:

```bash
pmetal infer \
  --model Qwen/Qwen3-0.6B-Base \
  --lora ./output/lora_weights.safetensors \
  --prompt "Does absolute truth exist?" \
  --chat \
  --show-thinking
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
| `--min-p` | Min-p dynamic sampling | Model default |
| `--chat` | Apply chat template | false |
| `--show-thinking` | Show reasoning content | false |
| `--fp8` | Use FP8 weights | false |
| `--no-ane` | Disable ANE inference | false |

### `dashboard`

Real-time TUI dashboard for monitoring training progress (requires `dashboard` feature):

```bash
pmetal dashboard --metrics-file ./output/metrics.jsonl
```

### `bench`

Benchmark training performance:

```bash
pmetal bench \
  --model Qwen/Qwen3-0.6B-Base \
  --batch-size 4 \
  --seq-len 512
```

Use `bench-ffi` for overhead analysis and `bench-gen` for generation loop profiling.

### `distill`

Knowledge distillation from teacher to student model:

```bash
pmetal distill \
  --teacher Qwen/Qwen3-4B \
  --student unsloth/Qwen3.5-0.8B-Base \
  --dataset train.jsonl \
  --output ./output/distilled \
  --method online \
  --loss-type kl_divergence \
  --temperature 2.0
```

Supports cross-vocabulary distillation (teacher and student can have different vocab sizes).

#### Distillation Options

| Option | Description | Default |
|--------|-------------|---------|
| `--teacher` | Teacher model ID | Required |
| `--student` | Student model ID | Required |
| `--dataset` | Training data (JSONL) | Required |
| `--method` | online, offline, progressive | online |
| `--loss-type` | kl_divergence, jensen_shannon, soft_cross_entropy | kl_divergence |
| `--temperature` | Softmax temperature | 2.0 |
| `--alpha` | Hard/soft label balance | 0.5 |
| `--rationale` | Reasoning-aware distillation | false |
| `--lora-r` | Student LoRA rank | 16 |

### `dataset`

Dataset utilities for preparing and analyzing training data:

```bash
pmetal dataset analyze --path train.jsonl
pmetal dataset validate --path train.jsonl --model Qwen/Qwen3-0.6B
pmetal dataset prepare TeichAI/dataset-id --output-dir ./data --model Qwen/Qwen3-0.6B
```

### `grpo`

Group Relative Policy Optimization for reasoning models:

```bash
pmetal grpo \
  --model unsloth/Qwen3-0.6B-Base \
  --dataset problems.jsonl \
  --output ./output/grpo
```

Use `--dapo` for Decoupled Clip and Dynamic Sampling Policy Optimization.

### Other Commands

| Command | Description |
|---------|-------------|
| `download` | Download model from HuggingFace |
| `memory` | Show memory usage and capacity |
| `quantize` | Quantize model to GGUF (Dynamic 2.0) |
| `ollama` | Export trained model for Ollama |
| `init` | Generate sample config file |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token |
| `RUST_LOG` | Log level (info, debug, trace) |

## License

MIT OR Apache-2.0
