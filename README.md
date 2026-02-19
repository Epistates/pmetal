# PMetal

**Powdered Metal** — High-performance LLM fine-tuning framework for Apple Silicon, written in Rust.

PMetal is a  machine learning framework that brings [Unsloth](https://github.com/unslothai/unsloth)-style optimizations to macOS. It leverages custom Metal shaders and the MLX framework to achieve state-of-the-art training throughput on Apple Silicon GPUs.

[![Rust](https://img.shields.io/badge/rust-1.82+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://www.apple.com/macos)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/epistates/pmetal.git
cd pmetal

# Build in release mode
cargo build --release
```

### Fine-tune a Model

```bash
# LoRA fine-tuning with auto-detected max-seq-len and sequence packing
./target/release/pmetal train \
  --model qwen/Qwen3-0.6B-Base \
  --dataset path/to/train.jsonl \
  --output ./output \
  --lora-r 16 \
  --batch-size 4 \
  --learning-rate 2e-4
```

### Run Reasoning Inference

```bash
# Inference with thinking mode enabled
./target/release/pmetal infer \
  --model qwen/Qwen3-0.6B-Base \
  --lora ./output/lora_weights.safetensors \
  --prompt "Does absolute truth exist?" \
  --chat \
  --show-thinking
```

## Architecture

PMetal is organized as a Rust workspace with 15 specialized crates:

```
pmetal/
├── pmetal-core         # Foundation: configs, traits, types
├── pmetal-metal        # Custom Metal GPU kernels
├── pmetal-mlx          # MLX backend integration (KV cache, RoPE, etc.)
├── pmetal-models       # LLM architectures (Llama, Qwen, DeepSeek, etc.)
├── pmetal-lora         # LoRA/QLoRA training implementations
├── pmetal-trainer      # Training loops (SFT, DPO, GRPO)
├── pmetal-data         # Dataset loading and preprocessing
├── pmetal-hub          # HuggingFace Hub integration
├── pmetal-distill      # Knowledge distillation
├── pmetal-merge        # Model merging (SLERP, TIES, DARE)
├── pmetal-gguf         # GGUF format with imatrix quantization
├── pmetal-mhc          # Memory-efficient Hierarchical Context
├── pmetal-distributed  # Distributed training support
├── pmetal-vocoder      # BigVGAN neural vocoder
└── pmetal-cli          # Command-line interface
```

### Dependency Graph

```
                    ┌─────────────────┐
                    │  pmetal-cli   │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ pmetal-trainer│ │ pmetal-lora   │ │ pmetal-data   │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ pmetal-models │ │  pmetal-mlx   │ │ pmetal-metal  │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  pmetal-core  │
                    └─────────────────┘
```

## Supported Models

| Family | Variants | LoRA | QLoRA | Full FT |
|--------|----------|------|-------|---------|
| Llama | 2, 3, 3.1, 3.2, 3.3, 4 | ✓ | ✓ | ✓ |
| Qwen | 2, 2.5, 3, 3-MoE | ✓ | - | ✓ |
| DeepSeek | V3, V3.2, V3.2-Speciale | ✓ | - | ✓ |
| Mistral | 7B, 8x7B | ✓ | ✓ | ✓ |
| Gemma | 2, 3 | ✓ | - | ✓ |
| Phi | 3, 4 | ✓ | - | ✓ |
| GPT-OSS | 20B, 120B | ✓ | - | - |
| Granite | 3.0, 3.1 | ✓ | - | ✓ |
| Cohere | Command R | ✓ | - | ✓ |

### Vision Models

| Family | Variants | Inference |
|--------|----------|-----------|
| Pixtral | 12B | ✓ |
| Qwen2-VL | 2B, 7B | ✓ |
| MLlama | 3.2-Vision | ✓ |

## Training Methods

- **Supervised Fine-Tuning (SFT)**: Standard next-token prediction
- **LoRA**: Low-Rank Adaptation with configurable rank and alpha
- **QLoRA**: 4-bit quantized base weights with LoRA adapters
- **DPO**: Direct Preference Optimization for RLHF
- **GRPO**: Group Relative Policy Optimization

## Key Features

### Metal Optimizations

Custom Metal shaders provide significant speedups:

- **FlashAttention**: O(n) memory attention with fused softmax
- **Fused LoRA**: Combined forward pass for adapter layers
- **Fused Cross-Entropy**: Unsloth-style chunked loss computation
- **Fused RoPE**: Rotary position embeddings in-kernel
- **Fused Sampler**: JIT-compiled token sampling

### Sequence Packing

Efficiently pack multiple sequences into single batches:

```bash
--use-sequence-packing  # Enable packing (99.7% efficiency)
--max-seq-len 2048      # Maximum packed sequence length
```

### Gradient Checkpointing

Trade compute for memory on large models:

```bash
--gradient-checkpointing  # Enable memory-efficient training
```

### Dataset Formats

Supported formats for training data:

**ShareGPT (conversations)**:
```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

**Alpaca (instruction)**:
```json
{"instruction": "...", "input": "...", "output": "..."}
```

**Messages (chat)**:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32.0 | LoRA scaling factor (2x rank) |
| `--batch-size` | 4 | Micro-batch size |
| `--learning-rate` | 2e-4 | Learning rate |
| `--max-seq-len` | 0 | Max seq len (0 = auto-detect) |
| `--epochs` | 1 | Number of training epochs |
| `--max-grad-norm` | 1.0 | Gradient clipping |

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--temperature` | Model default | Sampling temperature |
| `--top-k` | Model default | Top-k sampling |
| `--top-p` | Model default | Nucleus sampling |
| `--max-tokens` | 256 | Maximum generation length |
| `--repetition-penalty` | 1.0 | Repetition penalty |

## Development

### Building

```bash
# Debug build
cargo build

# Release build with optimizations
cargo build --release

# Run tests
cargo test --all

# Run clippy
cargo clippy --all
```

### Adding a New Model Architecture

1. Implement the `CausalLMModel` trait in `pmetal-models`
2. Add architecture detection in `dispatcher.rs`
3. Create LoRA wrapper in `pmetal-lora` if needed
4. Update the model registry

## Benchmarks

Run the included benchmarks:

```bash
# FFI overhead benchmark
cargo bench --bench ffi_overhead
```

## Troubleshooting

### Metal Toolchain Missing

If you see "cannot execute tool 'metal'":

```bash
xcodebuild -downloadComponent MetalToolchain
```

### Out of Memory

Try these options:
- Reduce `--batch-size`
- Enable `--gradient-checkpointing`
- Use `--use-sequence-packing` for variable-length data
- Reduce `--max-seq-len`

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [mlx-rs](https://github.com/oxideai/mlx-rs) - Rust bindings for MLX
- [Unsloth](https://github.com/unslothai/unsloth) - Inspiration for fused kernel optimizations
- [HuggingFace](https://huggingface.co) - Model hub and tokenizers
