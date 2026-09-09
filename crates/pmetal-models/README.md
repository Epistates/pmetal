# pmetal-models

LLM architecture implementations with dynamic dispatch.

## Overview

This crate provides implementations of popular LLM architectures optimized for Apple Silicon. It includes a dynamic dispatch system that automatically detects and loads models based on their configuration.

## Supported Architectures

### Dispatched Models (via `DynamicModel`)

These architectures are wired into the `ModelArchitecture` dispatcher and can be loaded automatically from `config.json`:

| Architecture | Family | Variants |
|-------------|--------|----------|
| `Llama` | Llama | 2, 3, 3.1, 3.2, 3.3 |
| `Llama4` | Llama 4 | Scout, Maverick |
| `Qwen2` | Qwen | 2, 2.5 |
| `Qwen3` | Qwen | 3 |
| `Qwen3MoE` | Qwen | 3-MoE |
| `Qwen3Next` | Qwen | 3.5 (Next) |
| `DeepSeek` | DeepSeek | V3, V3.2, V3.2-Speciale |
| `Mistral` | Mistral | 7B, Mixtral 8x7B (MoE) |
| `Gemma` | Gemma | 2, 3 |
| `Phi` | Phi | 3, 3.5 |
| `Phi4` | Phi | 4 |
| `Cohere` | Cohere | Command R |
| `Granite` | Granite | 3.0, 3.1, Hybrid MoE |
| `NemotronH` | NemotronH | Hybrid (Mamba+Attention) |
| `StarCoder2` | StarCoder2 | 3B, 7B, 15B |
| `RecurrentGemma` | RecurrentGemma | Griffin |
| `Jamba` | Jamba | 1.5 |
| `Flux` | Flux | 1-dev, 1-schnell (diffusion) |

### Architecture Modules (Not Dispatched)

These have implementations but are not wired into `DynamicModel` — use their types directly:

| Module | Family | Notes |
|--------|--------|-------|
| `gpt_oss` | GPT-OSS | 20B, 120B MoE |
| `pixtral` | Pixtral | 12B vision-language |
| `qwen2_vl` | Qwen2-VL | 2B, 7B vision-language |
| `mllama` | MLlama | 3.2-Vision |
| `clip` | CLIP | ViT-L/14 vision encoder |
| `whisper` | Whisper | Base, Small, Medium, Large |
| `t5` | T5 | Encoder-decoder |

## Features

- **Dynamic Model Loading**: Auto-detect architecture from `config.json`
- **Unified Generation API**: Common interface for all models
- **Advanced Sampling**: Temperature, top-k, top-p, repetition penalty
- **Metal-Accelerated Sampling**: Fused GPU sampler kernel
- **KV Cache Management**: Efficient inference with caching

## Usage

`DynamicModel::load` detects the architecture from `config.json`. It takes a **local directory**;
resolve HuggingFace ids with `pmetal_hub::download_model` first.

```rust,no_run
use pmetal_bridge::compat::Exception;
use pmetal_models::{DynamicModel, GenerationConfig, generate};

fn run(model_dir: &str, input_tokens: &[u32]) -> Result<Vec<u32>, Exception> {
    let mut model = DynamicModel::load(model_dir)?;

    let config = GenerationConfig::sampling(256, 0.7)
        .with_top_k(40)
        .with_top_p(0.95);

    let output = generate(|input| model.forward(input, None), input_tokens, config)?;
    Ok(output.token_ids)
}
```

## Architecture Detection

The `DynamicModel` automatically detects model architecture:

```rust,no_run
use pmetal_bridge::compat::Exception;
use pmetal_models::ModelArchitecture;

fn detect(model_dir: &str) -> Result<ModelArchitecture, Exception> {
    // Reads config.json: Llama, Qwen3, Mistral, Gemma, Phi, and so on.
    ModelArchitecture::detect(model_dir)
}
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
