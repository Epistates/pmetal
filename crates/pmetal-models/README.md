# pmetal-models

LLM architecture implementations with dynamic dispatch.

## Overview

This crate provides implementations of popular LLM architectures optimized for Apple Silicon. It includes a dynamic dispatch system that automatically detects and loads models based on their configuration.

## Supported Architectures

| Family | Variants | Status |
|--------|----------|--------|
| **Llama** | 2, 3, 3.1, 3.2, 3.3, 4 | Production |
| **Qwen** | 2, 2.5, 3, 3-MoE | Production |
| **DeepSeek** | V3, V3.2, V3.2-Speciale | Production |
| **Mistral** | 7B, 8x7B (MoE) | Production |
| **Gemma** | 2, 3 | Production |
| **Phi** | 3, 4 | Production |
| **GPT-OSS** | 20B, 120B | Production |
| **Granite** | 3.0, 3.1 | Production |
| **Cohere** | Command R | Production |

### Vision Models

| Family | Variants | Status |
|--------|----------|--------|
| **Pixtral** | 12B | Inference |
| **Qwen2-VL** | 2B, 7B | Inference |
| **MLlama** | 3.2-Vision | Inference |

## Features

- **Dynamic Model Loading**: Auto-detect architecture from `config.json`
- **Unified Generation API**: Common interface for all models
- **Advanced Sampling**: Temperature, top-k, top-p, repetition penalty
- **Metal-Accelerated Sampling**: Fused GPU sampler kernel
- **KV Cache Management**: Efficient inference with caching

## Usage

```rust
use pmetal_models::{DynamicModel, GenerationConfig, generate};

// Load model with auto-detection
let model = DynamicModel::from_pretrained("unsloth/Llama-3.2-1B")?;

// Configure generation
let config = GenerationConfig::sampling(256, 0.7)
    .with_top_k(40)
    .with_top_p(0.95);

// Generate tokens
let output = generate(
    |input| model.forward(input, None),
    &input_tokens,
    config,
)?;
```

## Architecture Detection

The `DynamicModel` automatically detects model architecture:

```rust
use pmetal_models::ModelArchitecture;

let arch = ModelArchitecture::detect("path/to/model")?;
// Returns: Llama, Qwen3, Mistral, Gemma, Phi, etc.
```

## Generation Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_tokens` | Maximum tokens to generate | Required |
| `temperature` | Sampling temperature (0 = greedy) | Model default |
| `top_k` | Top-k sampling (0 = disabled) | Model default |
| `top_p` | Nucleus sampling threshold | Model default |
| `repetition_penalty` | Penalty for repeated tokens | 1.0 |
| `stop_tokens` | Tokens that stop generation | EOS |

## Modules

| Module | Description |
|--------|-------------|
| `architectures/` | Model implementations (Llama, Qwen, etc.) |
| `dispatcher` | Dynamic model loading and dispatch |
| `generation` | Token generation with sampling |
| `loader` | HuggingFace model loading |
| `sampling/` | Sampling strategy implementations |
| `traits` | `CausalLMModel`, `Quantizable` traits |

## License

MIT OR Apache-2.0
