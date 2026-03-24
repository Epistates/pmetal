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

```rust
use pmetal_merge::{MergeConfig, LinearMerge, run_merge};

let config = MergeConfig {
    method: MergeMethod::Linear,
    models: vec![
        ModelWeight { path: "model_a", weight: 0.7 },
        ModelWeight { path: "model_b", weight: 0.3 },
    ],
    output: "merged_model",
};

run_merge(&config)?;
```

### SLERP Merge

```rust
use pmetal_merge::{MergeConfig, MergeMethod};

let config = MergeConfig {
    method: MergeMethod::Slerp { t: 0.5 },
    models: vec![
        ModelWeight { path: "model_a", weight: 1.0 },
        ModelWeight { path: "model_b", weight: 1.0 },
    ],
    output: "merged_model",
};
```

### TIES Merge

```rust
use pmetal_merge::{MergeConfig, MergeMethod};

let config = MergeConfig {
    method: MergeMethod::Ties {
        density: 0.5,      // Keep top 50% of weights
        majority_sign: true,
    },
    models: vec![
        ModelWeight { path: "task_a", weight: 1.0 },
        ModelWeight { path: "task_b", weight: 1.0 },
        ModelWeight { path: "task_c", weight: 1.0 },
    ],
    base_model: Some("base_model"),
    output: "merged_model",
};
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
