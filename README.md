# PMetal

**Powdered Metal** — High-performance LLM fine-tuning framework for Apple Silicon, written in Rust.

PMetal is a machine learning framework that brings [Unsloth](https://github.com/unslothai/unsloth)-style optimizations to macOS. It leverages custom Metal shaders, the MLX framework, and native Apple Neural Engine (ANE) integration to achieve state-of-the-art training and inference throughput on Apple Silicon.

[![Rust](https://img.shields.io/badge/rust-1.85+-orange.svg)](https://www.rust-lang.org)
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
  --model Qwen/Qwen3-0.6B-Base \
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
  --model Qwen/Qwen3-0.6B-Base \
  --lora ./output/lora_weights.safetensors \
  --prompt "Does absolute truth exist?" \
  --chat \
  --show-thinking
```

### ANE (Apple Neural Engine) Training & Inference

PMetal includes a native ANE pipeline behind the `ane` feature flag. This uses private `AppleNeuralEngine.framework` APIs to run MIL 1.3 programs directly on the Neural Engine with zero-copy IOSurface data transfer. **ANE support is enabled by default** when the `ane` feature is included.

```bash
# ANE is enabled by default. Just build:
cargo build --release

# Train on ANE (dynamic weight pipeline — 9 kernels, compiled once)
./target/release/pmetal train \
  --model Qwen/Qwen3-0.6B-Base \
  --dataset path/to/train.jsonl \
  --output ./output
  # ANE is used automatically. Use --no-ane to disable.

# Inference on ANE (hybrid ANE prefill + CPU decode with KV cache)
./target/release/pmetal infer \
  --model Qwen/Qwen3-0.6B \
  --prompt "Explain quantum entanglement" \
  --chat
  # ANE is used automatically. Use --no-ane to disable.

# Knowledge distillation (supports cross-vocabulary teacher/student)
./target/release/pmetal distill \
  --teacher Qwen/Qwen3-4B \
  --student unsloth/Qwen3.5-0.8B-Base \
  --dataset train.jsonl

# Real-time training dashboard (TUI)
./target/release/pmetal dashboard --metrics-file ./output/metrics.jsonl
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
├── pmetal-trainer      # Training loops (SFT, DPO, GRPO, GSPO, DAPO)
├── pmetal-data         # Dataset loading and preprocessing
├── pmetal-hub          # HuggingFace Hub integration
├── pmetal-distill      # Knowledge distillation
├── pmetal-merge        # Model merging (SLERP, TIES, DARE, ModelStock)
├── pmetal-gguf         # GGUF format with imatrix quantization
├── pmetal-mhc          # Manifold-Constrained Hyper-Connections
├── pmetal-distributed  # Distributed training support (mDNS, Ring All-Reduce)
├── pmetal-vocoder      # BigVGAN neural vocoder
└── pmetal-cli          # Command-line interface
```

## Supported Models

| Family | Variants | LoRA | QLoRA | Full FT |
|--------|----------|------|-------|---------|
| Llama | 2, 3, 3.1, 3.2, 3.3 | ✓ | ✓ | ✓ |
| Llama 4 | Scout, Maverick | ✓ | - | ✓ |
| Qwen | 2, 2.5, 3, 3.5 (Next) | ✓ | ✓ | ✓ |
| Qwen MoE | 3-MoE | ✓ | - | ✓ |
| DeepSeek | V3, V3.2, V3.2-Speciale | ✓ | - | ✓ |
| Mistral | 7B, 8x7B | ✓ | ✓ | ✓ |
| Gemma | 2, 3 | ✓ | - | ✓ |
| Phi | 3, 4 | ✓ | - | ✓ |
| Cohere | Command R | ✓ | - | ✓ |
| Granite | 3.0, 3.1 | ✓ | - | ✓ |
| NemotronH | Hybrid (Mamba+Attention) | ✓ | - | ✓ |
| StarCoder2 | 3B, 7B, 15B | ✓ | - | ✓ |
| RecurrentGemma | Griffin | ✓ | - | ✓ |
| Jamba | 1.5 | ✓ | - | ✓ |
| GPT-OSS | 20B, 120B | ✓ | - | - |

### Vision & Multimodal Models (In Progress)

Architecture implementations exist but are not yet integrated into the CLI dispatcher.

| Family | Variants | Status |
|--------|----------|--------|
| Pixtral | 12B | Architecture implemented |
| Qwen2-VL | 2B, 7B | Architecture implemented |
| MLlama | 3.2-Vision | Architecture implemented |
| CLIP | ViT-L/14 | Architecture implemented |
| Whisper | Base, Small, Medium, Large | Architecture implemented |

### Diffusion Models

| Family | Variants | Status |
|--------|----------|--------|
| Flux | 1-dev, 1-schnell | Dispatcher + pipeline implemented |

## Training Methods

The PMetal framework supports a wide range of training methods. Methods marked with **(CLI)** are directly available via subcommands.

- **Supervised Fine-Tuning (SFT) (CLI)**: Standard next-token prediction
- **LoRA (CLI)**: Low-Rank Adaptation with configurable rank and alpha
- **QLoRA (CLI)**: 4-bit quantized base weights with LoRA adapters
- **GRPO (CLI)**: Group Relative Policy Optimization for reasoning models
- **DAPO (CLI)**: Decoupled Clip and Dynamic Sampling Policy Optimization (available via `grpo --dapo`)
- **Knowledge Distillation (CLI)**: Online, Offline, and Progressive methods with reasoning (rationale) support. Cross-vocabulary distillation via sparse top-k alignment.
- **GSPO**: Group Sequence Policy Optimization (fixes GRPO length bias)
- **ANE Training**: Native Apple Neural Engine training with dynamic weight pipeline
- **DoRA**: Weight-Decomposed Low-Rank Adaptation
- **DPO**: Direct Preference Optimization for RLHF
- **PPO**: Proximal Policy Optimization
- **ORPO**: Odds Ratio Preference Optimization (reference-free)
- **SimPO**: Simple Preference Optimization
- **KTO**: Kahneman-Tversky Optimization (unpaired preference data)
- **Online DPO**: Online Direct Preference Optimization with reward models
- **Diffusion Training**: LLaDA-style masked diffusion for language models

## Key Features

### Metal GPU Optimizations

Custom Metal shaders provide significant speedups:

- **FlashAttention**: O(n) memory attention with fused softmax
- **Fused LoRA**: Combined forward pass for adapter layers
- **Fused Cross-Entropy**: Unsloth-style chunked loss computation
- **Fused RoPE**: Rotary position embeddings in-kernel
- **Fused Sampler**: JIT-compiled token sampling

### ANE (Neural Engine) Pipeline

Native ANE integration for power-efficient training and inference:

- **Dynamic Weight Pipeline**: 9 MIL kernels compiled once at startup; weights packed alongside activations in IOSurface spatial dimension. Zero recompilation during training.
- **Hybrid Inference**: ANE prefill + CPU decode with KV cache for autoregressive generation.
- **CPU RMSNorm**: RMSNorm computed in f32 on CPU to avoid fp16 overflow on ANE (saturation arithmetic). Per-head QK-norm stays on ANE.
- **IOSurface Zero-Copy**: fp32 shared memory surfaces for CPU↔ANE data transfer with no serialization overhead.
- **GQA/MQA Support**: Grouped-query and multi-query attention via MIL KV head expansion (replaces unreliable `tile` ops).

### Training Dashboard (TUI)

Real-time terminal dashboard via `pmetal dashboard`:

- Loss curve visualization (braille characters)
- Learning rate schedule tracking
- Per-component timing breakdown (ANE, Adam, RMSNorm, cblas)
- Token throughput monitoring

### Sequence Packing

Efficiently pack multiple sequences into single batches for 2-5x throughput improvement. **Enabled by default**.

```bash
--no-sequence-packing  # Disable packing
--max-seq-len 2048      # Maximum packed sequence length
```

### Gradient Checkpointing

Trade compute for memory on large models. Enabled by default with configurable layer grouping:

```bash
--gradient-checkpointing-layers 4  # Layers per checkpoint block (default)
--no-gradient-checkpointing        # Disable
```

Note: Currently implemented for Llama-family architectures. Qwen3 will log a warning that checkpointing is not yet applied.

### Dataset Formats

Supported formats for training data (Auto-detected):

- **ShareGPT**: `{"conversations": [{"from": "human", "value": "..."}, ...]}`
- **Alpaca**: `{"instruction": "...", "input": "...", "output": "..."}`
- **OpenAI/Messages**: `{"messages": [{"role": "user", "content": "..."}, ...]}`
- **Reasoning**: `{"problem": "...", "thinking": "...", "solution": "..."}`
- **Simple**: `{"text": "..."}`
- **Parquet**: Supports both standard text columns and reasoning formats.

## Configuration

### `pmetal train` Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32.0 | LoRA scaling factor (2x rank) |
| `--batch-size` | 1 | Micro-batch size |
| `--learning-rate` | 2e-4 | Learning rate |
| `--max-seq-len` | 0 | Max seq len (0 = auto-detect) |
| `--epochs` | 1 | Number of training epochs |
| `--max-grad-norm` | 1.0 | Gradient clipping |
| `--quantization` | none | QLoRA method (nf4, fp4, int8) |
| `--gradient-accumulation-steps` | 4 | Gradient accumulation steps |
| `--no-ane` | false | Disable ANE training |
| `--embedding-lr` | None | Separate LR for embeddings |
| `--no-metal-fused-optimizer` | false | Disable Metal fused optimizer |

### `pmetal infer` Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--temperature` | Model default | Sampling temperature |
| `--top-k` | Model default | Top-k sampling |
| `--top-p` | Model default | Nucleus sampling |
| `--min-p` | Model default | Min-p dynamic sampling |
| `--max-tokens` | 256 | Maximum generation length |
| `--repetition-penalty`| 1.0 | Repetition penalty |
| `--chat` | false | Apply chat template |
| `--show-thinking` | false | Show reasoning content |
| `--fp8` | false | Use FP8 weights (~2x mem reduction) |
| `--compiled` | false | Use JIT-compiled sampling |
| `--no-ane` | false | Disable ANE inference |
| `--ane-max-seq-len` | 1024 | Max ANE kernel sequence length |

## Development

### Building

```bash
# Release build (default features: ANE + Dashboard)
cargo build --release

# Build without ANE
cargo build --release --no-default-features --features dashboard

# Run tests (single-threaded for Metal compatibility)
just test
```

## License

Licensed under either of MIT or Apache-2.0.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [mlx-rs](https://github.com/oxideai/mlx-rs) - Rust bindings for MLX
- [Unsloth](https://github.com/unslothai/unsloth) - Inspiration for fused kernels
