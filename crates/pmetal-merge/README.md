# pmetal-merge

Model merging toolkit inspired by MergeKit.

## Overview

This crate provides utilities for merging multiple fine-tuned models into a single model. It supports various merging strategies and is optimized for memory-efficient processing of large models.

## Merge Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **Linear** | Weighted averaging | Simple blending |
| **SLERP** | Spherical interpolation | Smooth transitions |
| **TIES** | Task arithmetic + sparsification | Multi-task merging |
| **DARE** | Random pruning + rescaling | Reducing interference |
| **DELLA** | Adaptive magnitude pruning | Quality preservation |
| **Model Stock** | Geometric interpolation | Robust averaging |

## Features

- **Lazy Loading**: Stream weights without loading full models
- **Memory Efficient**: Process layer-by-layer for large models
- **Multiple Formats**: SafeTensors, PyTorch, GGUF support
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
Spherical linear interpolation for smooth blending between two models.

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
3. Reduces interference between models

## Modules

| Module | Description |
|--------|-------------|
| `linear` | Linear/weighted averaging |
| `slerp` | Spherical interpolation |
| `ties` | TIES merging |
| `dare` | DARE pruning |
| `della` | DELLA adaptive pruning |
| `model_stock` | Geometric merging |
| `config` | Configuration types |

## License

MIT OR Apache-2.0
