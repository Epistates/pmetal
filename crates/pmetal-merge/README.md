# pmetal-merge

Model merging toolkit for Apple Silicon.

## Overview

This crate provides utilities for merging multiple fine-tuned models into a single model. It supports various merging strategies and is optimized for memory-efficient processing of large models.

## Merge Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **Linear** | Weighted averaging | Simple blending |
| **SLERP** | Spherical interpolation | Smooth transitions between 2 models |
| **Multi-SLERP** | Multi-model spherical interpolation | Smooth blending of 3+ models |
| **TIES** | Task arithmetic + sparsification + sign consensus | Multi-task merging |
| **DARE (TIES)** | Random pruning + rescaling (TIES variant) | Reducing interference |
| **DARE (Linear)** | Random pruning + rescaling (linear variant) | Reducing interference |
| **Task Arithmetic** | Direct task vector addition | Combining capabilities |
| **DELLA** | Adaptive magnitude-based pruning | Quality preservation |
| **DELLA (Linear)** | Adaptive magnitude pruning (linear variant) | Quality preservation |
| **Breadcrumbs** | Breadcrumbs merge strategy | Preserving training trajectory |
| **Model Stock** | Geometric interpolation based on task vector similarity | Robust averaging |
| **Nearswap** | Near-swap merge strategy | Layer-level blending |
| **Passthrough** | Layer passthrough composition | Frankenstein merging |
| **RAM** | RAM merge strategy | Robust merging |
| **RAM+** | Enhanced RAM merge | Improved robustness |

## Features

- **Lazy Loading**: Stream weights without loading full models
- **Memory Efficient**: Process layer-by-layer for large models
- **Multiple Formats**: SafeTensors, PyTorch, GGUF support
- **GPU-Accelerated Merging**: Metal-based merge operations for large models
- **FP8-Aware Merging**: Merge with FP8 quantization for memory efficiency
- **Async Merge Pipeline**: Double-buffered streaming merge for large models
- **LoRA Merge**: Fuse LoRA adapters into base weights (standard and accurate modes)
- **Configurable**: Fine-grained control over merge parameters

## Usage

### Linear Merge

```rust,no_run
use std::path::PathBuf;

use pmetal_merge::{
    MergeConfig, MergeMethodConfig, MergeParameters, ModelConfig, ParameterSetting, run_merge,
};

fn weighted(path: &str, weight: f32) -> ModelConfig {
    ModelConfig {
        model: path.to_string(),
        parameters: MergeParameters {
            weight: Some(ParameterSetting::Scalar(weight)),
            ..Default::default()
        },
    }
}

fn linear() -> Result<(), Box<dyn std::error::Error>> {
    let config = MergeConfig {
        merge_method: MergeMethodConfig::Linear,
        models: vec![weighted("model_a", 0.7), weighted("model_b", 0.3)],
        output_path: Some(PathBuf::from("merged_model")),
        ..Default::default()
    };

    run_merge(&config)?;
    Ok(())
}
```

### SLERP Merge

Method-specific knobs like SLERP's `t` and TIES's `density` live in `MergeParameters`, either
globally or per model.

```rust,no_run
use std::path::PathBuf;

use pmetal_merge::{
    MergeConfig, MergeMethodConfig, MergeParameters, ModelConfig, ParameterSetting,
};

fn model(path: &str) -> ModelConfig {
    ModelConfig { model: path.to_string(), parameters: MergeParameters::default() }
}

fn slerp() -> MergeConfig {
    MergeConfig {
        merge_method: MergeMethodConfig::Slerp,
        models: vec![model("model_a"), model("model_b")],
        parameters: MergeParameters {
            t: Some(ParameterSetting::Scalar(0.5)),
            ..Default::default()
        },
        output_path: Some(PathBuf::from("merged_model")),
        ..Default::default()
    }
}
```

### TIES Merge

TIES and the other task-vector methods need a `base_model` to subtract.

```rust,no_run
use std::path::PathBuf;

use pmetal_merge::{
    MergeConfig, MergeMethodConfig, MergeParameters, ModelConfig, ParameterSetting,
};

fn model(path: &str) -> ModelConfig {
    ModelConfig { model: path.to_string(), parameters: MergeParameters::default() }
}

fn ties() -> MergeConfig {
    MergeConfig {
        merge_method: MergeMethodConfig::Ties,
        models: vec![model("task_a"), model("task_b"), model("task_c")],
        base_model: Some("base_model".to_string()),
        parameters: MergeParameters {
            // Keep the top 50% of each task vector by magnitude.
            density: Some(ParameterSetting::Scalar(0.5)),
            ..Default::default()
        },
        output_path: Some(PathBuf::from("merged_model")),
        ..Default::default()
    }
}
```

## Merge Methods Explained

### Linear
Simple weighted average: `merged = w1*m1 + w2*m2 + ...`

### SLERP
Spherical linear interpolation for smooth blending between two models. Parameter `t` controls interpolation (0.0 = model A, 1.0 = model B).

### TIES
Task Arithmetic with Interference Elimination:
1. Compute task vectors (fine-tuned - base)
2. Trim low-magnitude weights
3. Resolve sign conflicts by majority vote
4. Merge remaining weights

### DARE
Drop And REscale:
1. Randomly drop weights with probability p
2. Rescale remaining weights by 1/(1-p)
3. Reduces interference between models. Available in TIES and Linear variants.

### Task Arithmetic
Direct task vector addition: `merged = base + w1*(m1-base) + w2*(m2-base) + ...`

### DELLA
Adaptive magnitude-based pruning. Prunes weights based on their magnitude relative to the base model, preserving important changes.

### Model Stock
Geometric interpolation using task vector similarity. Computes merge weights based on geometric properties of the fine-tuning directions.

### Passthrough
Layer passthrough composition — select layers from different models to build a "Frankenstein" model.

## Modules

| Module | Description |
|--------|-------------|
| `methods` | All merge strategy implementations |
| `config` | Configuration types and method enum |
| `async_merge` | Async double-buffered merge pipeline |
| `batched` | Batched tensor merging |
| `consensus` | Sparsification and sign consensus |
| `fp8_merge` | FP8 quantization-aware merging |
| `gpu_merge` | GPU-accelerated merging |
| `loader` | Model weight loading |
| `lora_merge` | LoRA adapter merging (standard + accurate) |
| `sparsify` | Sparsification utilities |

## License

MIT OR Apache-2.0
