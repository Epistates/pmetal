# pmetal-merge

Model merging toolkit for Apple Silicon.

## Overview

This crate provides utilities for merging multiple fine-tuned models into a single model. It supports various merging strategies and is optimized for memory-efficient processing of large models.

It computes on `pmetal-runtime` arrays and needs only `pmetal-hub` to resolve model ids. `pmetal-trainer` borrows its checkpoint helpers (behind the trainer's `ane` feature), and the `pmetal` CLI exposes it as `pmetal merge` and `pmetal fuse`.

## Merge Methods

| Method | Description | Reference |
|--------|-------------|-----------|
| **Linear** | Weighted averaging (model soups) | Wortsman et al., 2022 |
| **SLERP** | Spherical interpolation of two models, per-layer `t` | Shoemake, 1985 |
| **Multi-SLERP** | Weighted Karcher mean on the sphere for 3+ models | |
| **Task Arithmetic** | Weighted sum of task vectors; negative weights subtract a skill | Ilharco et al., 2023 |
| **TIES** | Trim, elect the sign by mass, disjoint mean | Yadav et al., 2023 |
| **DARE (TIES / Linear)** | Random drop and rescale, then TIES or a weighted sum | Yu et al., 2024 |
| **DELLA (TIES / Linear)** | Magnitude-ranked drop and rescale | Deep et al., 2024 |
| **Breadcrumbs** | Masks the smallest entries and the largest outliers | Davari and Belilovsky, 2024 |
| **Consensus** | TALL masks keep only coordinates several tasks claim | Wang et al., 2024 |
| **Model Stock** | Interpolates toward the pre-trained weights by task-vector angle | Jang et al., 2024 |
| **Fisher** | Fisher-weighted averaging from precomputed diagonal Fisher | Matena and Raffel, 2022 |
| **RegMean** | Closed-form merge from precomputed Gram matrices | Jin et al., 2023 |
| **RAM / RAM+** | Unique vs shared update handling for RL-trained agents | arXiv:2601.13572 |
| **Nearswap** | Per-parameter selection of the value nearest the base | |
| **Passthrough** | Layer stacking (frankenmerges) with `slices` | |

## Features

- **Streaming**: checkpoints are memory-mapped and one tensor is merged at a time, in f32
  and written in the target dtype; peak memory is a few tensors, not the models
- **Atomic, indexed output**: shards are written to scratch names and renamed into place with
  a `model.safetensors.index.json`; an interrupted merge leaves no torn model
- **Frankenmerges**: `slices` stack layer ranges of several models, with `config.json` updated
- **Per-tensor parameters**: `weight`, `density`, `t` and `lambda` take `filter` lists
  (`self_attn` vs `mlp`) and layer gradients (`t: [0, 0.5, 1]`)
- **Reproducible**: DARE and DELLA masks are keyed by `seed`, tensor name and model
- **FP8 checkpoints**: E4M3 weights with block-wise `weight_scale_inv` are dequantized on load
- **Vocabulary mismatches**: `vocab_mismatch: truncate | extend` reconciles embedding/head rows
- **LoRA fusing**: PEFT and `pmetal train` adapters, rsLoRA, DoRA, `alpha_pattern`, and
  MLX-quantized bases (dequantize, fold, requantize)

## Installation

```toml
[dependencies]
pmetal-merge = "0.7"
```

## Usage

### A merge, end to end

`MergeBuilder` assembles a configuration and runs it. This one merges two tiny models written
into a temporary directory, so it needs no downloaded weights; a real merge names model
directories (or Hugging Face ids) instead.

```rust
use std::collections::HashMap;
use std::path::Path;

use pmetal_merge::{MergeBuilder, MergeMethodConfig, SafetensorsLoader, TensorLoader};
use safetensors::{Dtype, tensor::TensorView};

fn write_model(dir: &Path, weights: [f32; 4]) {
    std::fs::create_dir_all(dir).unwrap();
    let bytes: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view = TensorView::new(Dtype::F32, vec![1, 4], &bytes).unwrap();
    let file = safetensors::serialize(HashMap::from([("w", view)]), None).unwrap();
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
    std::fs::write(dir.join("config.json"), r#"{"model_type": "test", "hidden_size": 4}"#).unwrap();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let work = tempfile::tempdir()?;
    let (a, b) = (work.path().join("a"), work.path().join("b"));
    write_model(&a, [0.0, 2.0, 4.0, 6.0]);
    write_model(&b, [2.0, 4.0, 6.0, 8.0]);

    let merged = MergeBuilder::new()
        .method(MergeMethodConfig::Linear)
        .add_model(a.to_string_lossy())
        .add_model(b.to_string_lossy())
        .output(work.path().join("merged"))
        .dtype("float32")
        .run()?;

    let mut w = SafetensorsLoader::new(&merged)?.load_tensor("w")?;
    assert_eq!(w.to_f32_vec(4).unwrap(), [1.0, 3.0, 5.0, 7.0]);
    Ok(())
}
```

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
Weighted average: `merged = Σ wᵢ·mᵢ / Σ wᵢ` (`normalize: false` skips the division).

### SLERP
Spherical linear interpolation of two models: the direction follows the arc between them and the
norm is interpolated linearly. `t` (0 = model A, 1 = model B) may be a number, a list of
`value`/`filter` entries, or a list of anchors spread over the layers.

### TIES
1. Compute task vectors (fine-tuned − base)
2. Trim each to its top `density` fraction by magnitude
3. Elect, per parameter, the sign with the larger total mass
4. Average only the entries that carry the elected sign, and add `lambda` times that to the base

### DARE
Drop And REscale: drop each task-vector entry with probability `1 − density`, multiply the
survivors by `1/density`, then combine linearly or with TIES. Masks are drawn from `seed`.

### DELLA
Like DARE, but the keep probability of an entry rises linearly with its magnitude rank inside
its row, from `density − epsilon` to `density + epsilon`, and each survivor is divided by its own
keep probability.

### Breadcrumbs
Per layer, removes the smallest `1 − density` fraction of a task vector and the largest `gamma`
fraction (outliers), without rescaling.

### Consensus
TALL masks mark where a task's own update dominates (`|τₜ| ≥ tall_lambda·|τ_MTL − τₜ|`); only
coordinates claimed by at least `min_tasks` tasks are kept.

### Task Arithmetic
`merged = base + lambda·Σ wᵢ·(mᵢ − base)`

### Model Stock
`t = N·cosθ / (1 + (N−1)·cosθ)` with `cosθ` the mean pairwise cosine of the task vectors;
`merged = base + t·(mean(mᵢ) − base)`.

### Passthrough
Layer passthrough composition — select layers from different models to build a "Frankenstein"
model.

## Modules

| Module | Description |
|--------|-------------|
| `methods` | All merge strategy implementations |
| `config` | Configuration types and method enum |
| `consensus` | Sign election and the disjoint merge |
| `loader` | Memory-mapped model loading and the sharded writer |
| `lora_merge` | LoRA adapter folding |
| `moe_align` | Expert permutation alignment for MoE merges |
| `sanity` | NaN/inf checks on merged tensors |
| `sparsify` | Magnitude, DARE, DELLA and Breadcrumbs sparsifiers |

## Feature flags

None.

## Requirements

- Rust 1.95 or later
- macOS 14 or later on Apple Silicon (merging computes on `pmetal-runtime` arrays)

## License

MIT OR Apache-2.0
