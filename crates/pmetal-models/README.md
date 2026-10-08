# pmetal-models

LLM architecture implementations with dynamic dispatch.

## Overview

This crate provides implementations of popular LLM architectures optimized for Apple Silicon. It includes a dynamic dispatch system that automatically detects and loads models based on their configuration.

## Supported Architectures

### Dispatched Models (via `DynamicModel`)

These are the `ModelArchitecture` variants. Each is detected from `config.json` (its `model_type`, or failing that its `architectures`), and every one but `Flux` loads as a `DynamicModel`:

| Architecture | `model_type` | Models |
|-------------|--------------|--------|
| `Llama` | `llama`, `llama3` | Llama 2, 3, 3.1, 3.2, 3.3 |
| `Llama4` | `llama4`, `llama4_text` | Llama 4 Scout, Maverick |
| `Mllama` | `mllama` | Llama 3.2 Vision |
| `Qwen2` | `qwen2`, `qwen2_5` | Qwen 2, 2.5 |
| `Qwen3` | `qwen3` | Qwen 3 |
| `Qwen3MoE` | `qwen3_moe` | Qwen 3 MoE |
| `Qwen3Next` | `qwen3_next`, `qwen3_5`, `qwen3_5_moe`, `qwen3_6`, `qwen3_6_moe` | Qwen 3.5, 3.6, 3.8 (hybrid Gated DeltaNet and attention) |
| `Qwen4Exp` | `qwen4_exp` | Qwen3.8-Flash-Next |
| `Gemma` | `gemma`, `gemma2`, `gemma3` | Gemma, Gemma 2, Gemma 3 |
| `Gemma4` | `gemma4`, `gemma4_text`, `gemma4_unified` | Gemma 4, dense and MoE |
| `DiffusionGemma` | `diffusion_gemma` | DiffusionGemma (block-autoregressive diffusion) |
| `Mistral` | `mistral`, `mixtral` | Mistral 7B, Mixtral (MoE) |
| `Phi` | `phi3` | Phi 3, 3.5 |
| `Phi4` | `phi4` | Phi 4 |
| `DeepSeek` | `deepseek`, `deepseek_v2`, `deepseek_v3` | DeepSeek V2, V3, R1 |
| `Cohere` | `cohere`, `cohere2`, `command_r` | Command R |
| `Granite` | `granite`, `granitemoe`, `granitemoeshared`, `granitemoehybrid` | Granite 3.x dense and MoE, 4.0 / 4.0-H (Mamba-2 hybrid), 4.1, 4.2 |
| `NemotronH` | `nemotron_h` | Nemotron-H (hybrid Mamba and attention) |
| `GptOss` | `gpt_oss` | GPT-OSS 20B, 120B (MoE) |
| `Bert` | `bert` | BERT (encoder only) |
| `Flux` | `flux` | Flux.1 (diffusion): detected, but built with `pipelines::FluxPipeline` rather than `DynamicModel` |

### Pipeline Components (Not Dispatched)

These are loaded by the Flux pipeline rather than by `DynamicModel`:

| Module | Family | Notes |
|--------|--------|-------|
| `clip` | CLIP | Text encoder |
| `t5` | T5 | Text encoder |
| `vae` | VAE | Latent decoder |

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
| `decision` | Decision models (Clef): record encoding, the joint schema head, `/v1/systemone` |
| `dispatcher` | Dynamic model loading and dispatch |
| `generation` | Token generation with sampling |
| `loader` | HuggingFace model loading |
| `sampling/` | Sampling strategy implementations |
| `traits` | `CausalLMModel`, `Quantizable` traits |

## License

MIT OR Apache-2.0
