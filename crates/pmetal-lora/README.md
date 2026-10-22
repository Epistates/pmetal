# pmetal-lora

LoRA and QLoRA training implementations with Metal acceleration.

## Overview

This crate provides Low-Rank Adaptation (LoRA) and Quantized LoRA (QLoRA) training for LLMs on Apple Silicon. Adapters attach to the projections of the same model the inference path loads, so every architecture it loads can be fine-tuned.

`pmetal-trainer` trains the models built here, and the `pmetal` CLI and the Python bindings reach it through the trainer. It builds on `pmetal-models` (the `DynamicModel` it adapts), `pmetal-cache` and `pmetal-runtime`, where the adapter itself lives as a property of `Linear`.

## Features

- **LoRA, rsLoRA and DoRA**: configurable rank, alpha and target modules; the default adapts every linear layer (attention and MLP)
- **Initializations**: standard, PiSSA, OLoRA and LoftQ, each tested against its paper's definition
- **Routed experts**: one adapter per expert on Qwen3-MoE, DeepSeek and Nemotron-H
- **QLoRA**: base weights packed to NF4, NVFP4 or 8-bit integers, adapters in full precision; a checkpoint already packed (NVIDIA ModelOpt NVFP4 or FP8) trains with no dense copy of its base
- **Any architecture the dispatcher loads**, encoders included (the embedding trainer takes an adapted BERT-family model)
- **PEFT layout**: adapters are written and read as `adapter_config.json` and `adapter_model.safetensors`, in both directions
- **Packed sequences**: `forward_with_positions` takes explicit positions, so documents packed into one row stay apart
- **Merge and unmerge** of the adapter into the base weights

## Installation

```toml
[dependencies]
pmetal-lora = "0.7"
```

## Usage

The adapter's configuration round-trips through the PEFT layout without any weights:

```rust
use pmetal_core::LoraConfig;
use pmetal_lora::peft::{adapter_base_model, lora_config_from_peft, peft_config};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = LoraConfig {
        r: 8,
        alpha: 16.0,
        use_rslora: true,
        target_modules: vec!["q_proj".into(), "v_proj".into()],
        ..Default::default()
    };

    // What goes in `adapter_config.json`...
    let json = peft_config(&config, &config.target_modules, Some("Qwen/Qwen3-0.6B"));
    assert_eq!(json["peft_type"], "LORA");
    assert_eq!(adapter_base_model(&json).as_deref(), Some("Qwen/Qwen3-0.6B"));

    // ...and what reading it back gives.
    let back = lora_config_from_peft(&json)?;
    assert_eq!((back.r, back.alpha, back.use_rslora), (8, 16.0, true));
    Ok(())
}
```

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

## Feature flags

None.

## Requirements

- Rust 1.95 or later
- macOS 14 or later on Apple Silicon

## License

MIT OR Apache-2.0
