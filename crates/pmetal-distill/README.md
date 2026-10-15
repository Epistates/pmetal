# pmetal-distill

Knowledge distillation losses for training on Apple Silicon.

## Overview

This crate provides knowledge distillation utilities for training smaller student models to mimic larger teacher models. Every loss is an MLX expression, so it runs on the GPU and the student is trained by differentiating it. The teacher is always a constant: no loss sends gradient into it.

## Loss Functions

| Loss | Description |
|------|-------------|
| **KL Divergence** | Forward or reverse KL at temperature `T` (Hinton et al., 2015) |
| **Jensen-Shannon** | Symmetric divergence |
| **Generalized JSD** (`JsdSkewedLoss`) | `JSD(β)` of GKD (Agarwal et al., 2024); `β = 0` / `1` are forward / reverse KL |
| **Skew KL** (`SkewKlLoss`) | DistiLLM's `KL(T ‖ αT + (1-α)S)` and its reverse (Ko et al., 2024) |
| **Soft Cross-Entropy** | Temperature-scaled CE |
| **TVD** | Total Variation Distance |
| **Universal Logit** (`UniversalLogitLoss`) | ULD (Boizard et al., 2024): L1 between sorted distributions, for teachers and students with different tokenizers |
| **MiniLLM** (`MiniLlmLoss`) | Policy-gradient reverse KL on student samples (Gu et al., 2024) |
| **GKD** (`GkdLoss`) | On-policy distillation with a chosen divergence (Agarwal et al., 2024) |
| **Sparse KL** (`SparseKlLoss`) | Forward KL against a top-k teacher read from an offline cache, without a dense teacher |
| **Hinge / Logistic Ranking** | Pairwise ordering of the teacher's top-k tokens |
| **Hidden State MSE / Cosine / L1** | Layer alignment, with a token mask |
| **Attention transfer** | MSE, KL or Zagoruyko-Komodakis maps, wired through `LossConfig::attention` |

`Distiller::compute_loss` multiplies the soft loss by `T²`, as in Hinton et
al. (2015), so its gradient keeps the same scale as the temperature changes
(a loss that never softens a distribution, such as logit MSE, opts out).

## Features

- **TAID**: Temporally Adaptive Interpolated Distillation (Shing et al., ICLR 2025) — `TaidDistiller`: a logit-space intermediate target whose interpolation `t` adapts to the student's progress
- **Offline Distillation**: top-k, sampled, int8 or int4 teacher logits cached per sequence (`LogitCache`, `LogitCompressor`); a top-k cache is read back sparse (`SparseLogits`) and trained on with `Distiller::compute_loss_sparse`; a sampled cache (Random-Sampling KD, Anshumann et al., 2025: draws from the teacher, an unbiased estimate where the top-k is biased) is read back as `SparseTargets` and trained on with `Distiller::compute_loss_sampled`
- **Cross-Vocabulary Distillation**: for models that share a tokenizer but differ in embedding padding, the losses compare on the teacher's top-k ids (or `align_vocab_truncate` keeps the whole common vocabulary); for models with different tokenizers use `UniversalLogitLoss`
- **Progressive Distillation**: temperature annealing schedules
- **Layer Matching**: align intermediate representations and attention maps
- **Reasoning-Aware**: entropy-weighted rationale distillation and a reasoning-span mask (`generate_reasoning_mask`)

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
| `losses` | Loss function implementations (KL, JS, skew KL, ULD, MiniLLM, GKD, sparse KL, Soft CE, MSE, Cosine, L1, TVD, Hinge, Logistic) |
| `taid` | Temporally Adaptive Interpolated Distillation |
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
