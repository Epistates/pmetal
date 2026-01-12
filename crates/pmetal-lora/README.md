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

```rust
use pmetal_lora::{DynamicLoraModel, TrainableModel};
use pmetal_core::LoraConfig;

// Configure LoRA
let config = LoraConfig {
    r: 16,
    alpha: 16.0,
    dropout: 0.0,
    ..Default::default()
};

// Load model with LoRA adapters
let mut model = DynamicLoraModel::from_pretrained("path/to/model", config)?;

// Training loop
for batch in dataloader {
    let logits = model.forward(&batch.input_ids, None)?;
    // Compute loss and backprop...
}

// Save adapters
model.save_lora_weights("output/lora_weights.safetensors")?;
```

### Loading Trained Adapters

```rust
// Load base model with LoRA structure
let mut model = DynamicLoraModel::from_pretrained("path/to/model", config)?;

// Load trained adapter weights
model.load_lora_weights("output/lora_weights.safetensors")?;

// Run inference
let logits = model.forward(&input_ids, None)?;
```

## Architecture Support

| Architecture | LoRA | QLoRA | Grad Checkpoint |
|--------------|------|-------|-----------------|
| Llama | Yes | Yes | Yes |
| Qwen | Yes | No | Yes |
| DeepSeek | Yes | No | Yes |
| Mistral | Yes | Yes | Yes |
| Gemma | Yes | No | Yes |
| Phi | Yes | No | Yes |
| GPT-OSS | Yes | No | No |
| Granite | Yes | No | Yes |
| Cohere | Yes | No | Yes |

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
| `llama_lora` | LLaMA-specific LoRA implementation |
| `llama_qlora` | LLaMA QLoRA with 4-bit quantization |
| `qwen3_lora` | Qwen3-specific LoRA with grad checkpointing |
| `trainable` | `TrainableModel` trait definition |
| `fused_training` | Metal-accelerated training utilities |

## Performance

Compared to mlx-lm on identical hardware:

| Metric | pmetal-lora | mlx-lm |
|--------|---------------|--------|
| Steps/sec | 1.33 | 0.62 |
| Memory | ~10 GB | 19 GB |

## License

MIT OR Apache-2.0
