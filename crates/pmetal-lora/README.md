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
use pmetal_runtime::compat::Array;
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

    // A PEFT adapter directory: adapter_model.safetensors + adapter_config.json.
    model.save_adapter("output", Some(model_dir))?;
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
    model.save_adapter("output", Some(model_dir))?;
    Ok(())
}
```

### Loading Trained Adapters

```rust,no_run
use pmetal_runtime::compat::Array;
use pmetal_core::LoraConfig;
use pmetal_lora::{DynamicLoraModel, TrainableModel};

fn infer(model_dir: &str, input_ids: &Array) -> Result<Array, Box<dyn std::error::Error>> {
    // Build the model from the adapter's own config, then fill the adapters.
    let config = pmetal_lora::peft::lora_config_from_peft(&serde_json::from_str(
        &std::fs::read_to_string("output/adapter_config.json")?,
    )?)?;
    let mut model = DynamicLoraModel::from_pretrained(model_dir, config)?;
    model.load_lora_weights("output")?;

    Ok(model.forward(input_ids, None)?)
}
```

Adapters are saved and read in the layout `peft` uses: `PeftModel.from_pretrained(base, "output")`
loads one trained here, and one `peft` wrote loads here (a plain, rsLoRA or DoRA LoRA with one
rank and alpha; per-module ranks and `modules_to_save` are refused rather than loaded wrong).
Adapters from earlier releases (`lora_weights.safetensors`) still load.

## Architecture Support

`DynamicLoraModel` wraps an `AdaptedModel`: the same `DynamicModel` the inference path loads, with
adapters attached to its projections. Every architecture `DynamicModel` loads can be LoRA-trained
this way, so a fine-tune trains exactly the forward pass `pmetal infer` and `pmetal serve` run.

QLoRA is the same model with its projections packed first (`quantize_base`), so it covers the same
architectures. The LM head, MoE routers and routed experts stay as loaded. A checkpoint that is
already packed (NVIDIA ModelOpt NVFP4 or FP8) keeps its projections packed on load, and an adapter
on one is QLoRA with no dense copy of the base.

The default adapts every linear layer (attention and MLP), and the routed experts of Qwen3-MoE,
DeepSeek and Nemotron-H with one adapter per expert at rank `r / top_k`. MoE routers and the LM head
are left alone.

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `r` | LoRA rank | 16 |
| `alpha` | Scaling factor | 32.0 |
| `dropout` | Dropout rate | 0.0 |
| `target_modules` | Projections to adapt, by name or full module name; empty (or `all-linear`) adapts every linear layer but the LM head | empty |
| `use_rslora` | Scale by `alpha / sqrt(r)` | false |
| `use_dora` | Weight-decomposed LoRA (dense projections) | false |
| `init_lora_weights` | `true` (A random, B zero), `pissa`, `olora`, `loftq` | `true` |

## Modules

| Module | Description |
|--------|-------------|
| `adapted` | `AdaptedModel`: LoRA adapters attached to a `DynamicModel` |
| `dynamic` | `DynamicLoraModel`, the trainer-facing shell over `AdaptedModel` |
| `qlora` | `quantize_base` and the QLoRA schemes |
| `peft` | The PEFT adapter layout: `adapter_config.json`, tensor names, `lora_config_from_peft` |
| `lora` | `LoraError` and adapter-file IO |
| `trainable` | `TrainableModel` trait definition |

## License

MIT OR Apache-2.0
