# pmetal-lora

LoRA and QLoRA training implementations with Metal acceleration.

## Overview

This crate provides Low-Rank Adaptation (LoRA) and Quantized LoRA (QLoRA) training for LLMs on Apple Silicon. Adapters attach to the projections of the same model the inference path loads, so every architecture it loads can be fine-tuned.

## Features

- **Standard LoRA**: Low-rank adaptation with configurable rank and alpha
- **QLoRA**: base weights packed to NF4, NVFP4 or 8-bit integers, adapters in full precision
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

### QLoRA

Pack the base before the adapters go on. NF4 is the QLoRA paper's data type; `fp4` (NVFP4)
and `int8` run MLX's fused quantized matmul.

```rust,no_run
use pmetal_core::LoraConfig;
use pmetal_lora::{DynamicLoraModel, QLoraConfig, QLoraScheme, TrainableModel};

fn qlora(model_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let qlora = QLoraConfig {
        scheme: QLoraScheme::Nf4,
        group_size: 64,
        double_quant: true,
    };
    let (model, packed) =
        DynamicLoraModel::from_pretrained_quantized(model_dir, LoraConfig::default(), &qlora)?;
    println!(
        "{} projections packed, {} -> {} bytes",
        packed.packed, packed.dense_bytes, packed.packed_bytes
    );
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

QLoRA is the same model with its projections packed first (`quantize_base`), so it covers the same
architectures. The LM head, MoE routers and routed experts stay as loaded.

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
| `qlora` | `quantize_base` and the QLoRA schemes |
| `lora` | `LoraError` and adapter-file IO |
| `trainable` | `TrainableModel` trait definition |

## License

MIT OR Apache-2.0
