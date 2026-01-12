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
| **Hidden State MSE** | Layer alignment | O(hidden) |
| **Hidden State Cosine** | Direction alignment | O(hidden) |

## Features

- **Online Softmax**: O(1) memory per token via streaming computation
- **Fused Operations**: Temperature scaling + softmax + loss in one kernel
- **Progressive Distillation**: Temperature annealing schedules
- **Offline Distillation**: Compressed logit caching for large teachers
- **Layer Matching**: Align intermediate representations

## Usage

### Basic Distillation

```rust
use pmetal_distill::{Distiller, DistillConfig, KLDivergenceLoss};

let config = DistillConfig {
    temperature: 2.0,
    alpha: 0.5,  // Balance between soft and hard labels
    ..Default::default()
};

let distiller = Distiller::new(teacher, student, config)?;

// Distillation training loop
for batch in dataloader {
    let loss = distiller.step(&batch)?;
}
```

### With Hidden State Alignment

```rust
use pmetal_distill::{DistillConfig, HiddenStateLoss};

let config = DistillConfig {
    temperature: 2.0,
    alpha: 0.5,
    hidden_loss: Some(HiddenStateLoss::Cosine),
    hidden_layers: vec![6, 12, 18, 24],  // Layers to align
    ..Default::default()
};
```

### Progressive Temperature

```rust
use pmetal_distill::TemperatureSchedule;

let schedule = TemperatureSchedule::linear(
    start: 4.0,
    end: 1.0,
    steps: 1000,
);
```

## Metal Optimizations

The distillation losses are optimized for Apple Silicon:

- **Online Softmax**: Streaming max/sum computation avoids materializing full probability tensors
- **Fused Temperature**: Temperature division happens once in the kernel
- **SIMD Parallelization**: Efficient handling of large vocabularies

## Modules

| Module | Description |
|--------|-------------|
| `losses` | Loss function implementations |
| `distiller` | Main distillation trainer |
| `config` | Configuration types |
| `schedule` | Temperature scheduling |
| `offline` | Logit caching for offline distillation |

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `temperature` | Softmax temperature | 2.0 |
| `alpha` | Soft/hard label balance | 0.5 |
| `hidden_loss` | Hidden state loss type | None |
| `hidden_weight` | Hidden loss weight | 0.1 |

## License

MIT OR Apache-2.0
