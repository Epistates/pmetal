# pmetal-lora

LoRA and QLoRA training implementations with Metal acceleration.

## Overview

This crate provides efficient Low-Rank Adaptation (LoRA) and Quantized LoRA (QLoRA) training for LLMs on Apple Silicon. It includes architecture-specific optimizations and a dynamic model system for seamless multi-architecture support.

## Features

- **Standard LoRA**: Low-rank adaptation with configurable rank and alpha
- **QLoRA**: 4-bit quantized base weights with full-precision adapters
- **Dynamic Architecture**: Auto-detect and load any supported model
- **Fused Training**: Metal-accelerated forward/backward passes (~2x speedup)
- **Gradient Checkpointing**: Memory-efficient training for large models
- **Sequence Packing**: Efficient training on variable-length data

## Usage

### Basic LoRA Training

`from_pretrained` takes a **local directory**; resolve HuggingFace ids with
`pmetal_hub::download_model` first.

```rust,no_run
use pmetal_bridge::compat::Array;
use pmetal_core::LoraConfig;
use pmetal_lora::{DynamicLoraModel, TrainableModel};

fn train(model_dir: &str, batches: &[Array]) -> Result<(), Box<dyn std::error::Error>> {
    let config = LoraConfig {
        r: 16,
        alpha: 16.0,
        dropout: 0.0,
        ..Default::default()
    };

    let mut model = DynamicLoraModel::from_pretrained(model_dir, config)?;

    for input_ids in batches {
        let _logits = model.forward(input_ids, None)?;
        // Compute loss and backprop...
    }

    model.save_lora_weights("output/lora_weights.safetensors")?;
    Ok(())
}
```

### Loading Trained Adapters

```rust,no_run
use pmetal_bridge::compat::Array;
use pmetal_core::LoraConfig;
use pmetal_lora::{DynamicLoraModel, TrainableModel};

fn infer(model_dir: &str, input_ids: &Array) -> Result<Array, Box<dyn std::error::Error>> {
    let config = LoraConfig::default();

    // Build the base model with LoRA structure, then fill the adapters.
    let mut model = DynamicLoraModel::from_pretrained(model_dir, config)?;
    model.load_lora_weights("output/lora_weights.safetensors")?;

    Ok(model.forward(input_ids, None)?)
}
```

## Architecture Support

The following architectures are supported via `DynamicLoraModel` (auto-detection + loading):

| Architecture | LoRA | QLoRA | Notes |
|--------------|------|-------|-------|
| Llama (2, 3, 3.1, 3.2, 3.3) | Yes | Yes | Gradient checkpointing supported |
| Qwen 2 (2, 2.5) | Yes | — | Uses Qwen3 LoRA implementation internally |
| Qwen 3 | Yes | Yes | Gradient checkpointing supported |
| Qwen 3.5 (Next) | Yes | — | Hybrid GDN + Attention, nested text_config |
| Mistral (7B, Mixtral 8x7B) | Yes | Yes | Sliding window attention |
| Gemma (2, 3) | Yes | Yes | GeGLU activation, special RMSNorm |
| Phi (3, 3.5) | Yes | — | Partial RoPE, fused gate_up |

Architectures not listed (Llama 4, Qwen3MoE, DeepSeek, Phi4, Cohere, Granite, NemotronH, StarCoder2, RecurrentGemma, Jamba) return `DynamicLoraError::NotImplemented`. The `generic_lora` module provides reusable LoRA attention and MLP components for building custom LoRA models for these architectures.

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `r` | LoRA rank | 8 |
| `alpha` | Scaling factor | 16.0 |
| `dropout` | Dropout rate | 0.0 |
| `target_modules` | Modules to adapt | All attention + MLP |

## Modules

| Module | Description |
|--------|-------------|
| `dynamic` | `DynamicLoraModel` with auto-detection |
| `llama_lora` | LLaMA-specific LoRA (also covers Granite, Cohere, StarCoder2) |
| `qwen3_lora` | Qwen3-specific LoRA |
| `qwen3_next_lora` | Qwen 3.5 (Next) hybrid LoRA |
| `mistral_lora` | Mistral-specific LoRA |
| `gemma_lora` | Gemma-specific LoRA |
| `phi_lora` | Phi-specific LoRA |
| `generic_lora` | Generic LoRA for architectures without dedicated implementations |
| `trainable` | `TrainableModel` trait definition |
| `arch_config` | Per-architecture LoRA configuration |

## Performance

Compared to mlx-lm on identical hardware:

| Metric | pmetal-lora | mlx-lm |
|--------|---------------|--------|
| Steps/sec | 1.33 | 0.62 |
| Memory | ~10 GB | 19 GB |

## License

MIT OR Apache-2.0
