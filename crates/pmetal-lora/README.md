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

`DynamicLoraModel` wraps an `AdaptedModel`: the same `DynamicModel` the inference path loads, with
adapters attached to its projections. Every architecture `DynamicModel` loads can be LoRA-trained
this way, so a fine-tune trains exactly the forward pass `pmetal infer` and `pmetal serve` run.

QLoRA (`DynamicQloraModel`) has dedicated modules for Llama, Mistral, Granite, Qwen 2 / 3,
Qwen 3.5 (Next), Qwen3 MoE, Gemma, Gemma 4, GPT-OSS, Llama 4, DeepSeek, NemotronH, Phi and Cohere.

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
| `adapted` | `AdaptedModel`: LoRA adapters attached to a `DynamicModel` |
| `dynamic` | `DynamicLoraModel`, the trainer-facing shell over `AdaptedModel` |
| `dynamic_qlora` | `DynamicQloraModel` with per-architecture QLoRA dispatch |
| `lora` / `dora` / `qlora` | `LoraLinear`, `DoraLinear`, `QLoraLinear` |
| `*_qlora`, `*_lora` | Per-architecture QLoRA models and the LoRA stacks they build on |
| `autograd` | Hand-written LoRA and fused-MLP backward passes |
| `lora_helpers` | Shared parameter collection and adapter save/load |
| `trainable` | `TrainableModel` trait definition |

## License

MIT OR Apache-2.0
