# pmetal-distill

Knowledge distillation losses for training on Apple Silicon.

## Overview

This crate provides knowledge distillation utilities for training smaller student models to mimic larger teacher models. Every loss is an MLX expression, so it runs on the GPU and the student is trained by differentiating it.

## Loss Functions

| Loss | Description |
|------|-------------|
| **KL Divergence** | Forward or reverse KL at temperature `T` |
| **Jensen-Shannon** | Symmetric divergence |
| **Soft Cross-Entropy** | Temperature-scaled CE |
| **TVD** | Total Variation Distance |
| **Hinge Ranking** | Margin-based ranking loss |
| **Logistic Ranking** | Logistic ranking loss |
| **Hidden State MSE** | Layer alignment |
| **Hidden State Cosine** | Direction alignment |
| **Hidden State L1** | L1 layer alignment |

`Distiller::compute_loss` multiplies the soft loss by `T²`, as in Hinton et
al. (2015), so its gradient keeps the same scale as the temperature changes.

## Features

- **TAID**: Temporally Adaptive Interpolated Distillation (ICLR 2025 SOTA) — `TaidDistiller` with configurable schedules
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
