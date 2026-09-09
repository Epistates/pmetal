# pmetal-distill

Knowledge distillation with GPU-accelerated loss computation.

## Overview

This crate provides knowledge distillation utilities for training smaller student models to mimic larger teacher models. It includes Metal-optimized loss functions for efficient training on Apple Silicon.

## Loss Functions

| Loss | Description | Memory |
|------|-------------|--------|
| **KL Divergence** | Standard KL distance | O(vocab) |
| **Jensen-Shannon** | Symmetric divergence | O(vocab) |
| **Soft Cross-Entropy** | Temperature-scaled CE | O(1) |
| **TVD** | Total Variation Distance | O(vocab) |
| **Hinge Ranking** | Margin-based ranking loss | O(vocab) |
| **Logistic Ranking** | Logistic ranking loss | O(vocab) |
| **Hidden State MSE** | Layer alignment | O(hidden) |
| **Hidden State Cosine** | Direction alignment | O(hidden) |
| **Hidden State L1** | L1 layer alignment | O(hidden) |

## Features

- **TAID**: Temporally Adaptive Interpolated Distillation (ICLR 2025 SOTA) — `TaidDistiller` with configurable schedules
- **Online Softmax**: O(1) memory per token via streaming computation
- **Fused Operations**: Temperature scaling + softmax + loss in one kernel
- **Cross-Vocabulary Distillation**: Sparse top-k alignment for teacher/student vocab mismatch (e.g. Qwen3 to Qwen3.5)
- **Progressive Distillation**: Temperature annealing schedules
- **Offline Distillation**: Compressed logit caching for large teachers (`LogitCache`, `LogitCompressor`)
- **Layer Matching**: Align intermediate representations
- **Reasoning-Aware**: Rationale distillation with weighted reasoning tokens

## Usage

`pmetal-distill` is the low-level loss/cache crate. It provides `Distiller`,
`TaidDistiller`, `LogitCache`, and `LogitCompressor` for use inside a higher-level
training loop.

```rust,no_run
use pmetal_distill::{DistillConfig, Distiller, Result};

fn build() -> Result<Distiller> {
    let config = DistillConfig::from_yaml_file("distill_config.yaml")?;
    // Call `distiller.compute_loss(..)` inside your training loop.
    Distiller::new(config)
}
```

For end-to-end model training, use the `pmetal distill` CLI or
`pmetal_trainer::DistillationTrainer`.

## Metal Optimizations

The distillation losses are optimized for Apple Silicon:

- **Online Softmax**: Streaming max/sum computation avoids materializing full probability tensors
- **Fused Temperature**: Temperature division happens once in the kernel
- **SIMD Parallelization**: Efficient handling of large vocabularies

## Modules

| Module | Description |
|--------|-------------|
| `losses` | Loss function implementations (KL, JS, Soft CE, MSE, Cosine, L1, TVD, Hinge, Logistic) |
| `taid` | Temporally Adaptive Interpolated Distillation (ICLR 2025 SOTA) |
| `reasoning` | Rationale distillation for reasoning models |
| Config/Builder | `DistillConfig`, `OfflineConfig`, `DistillerBuilder`, distillation method types |

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `temperature` | Softmax temperature | 2.0 |
| `alpha` | Soft/hard label balance | 0.5 |
| `hidden_loss` | Hidden state loss type | None |
| `hidden_weight` | Hidden loss weight | 0.1 |

## License

MIT OR Apache-2.0
