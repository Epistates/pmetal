# pmetal-distill

Knowledge distillation losses for training on Apple Silicon.

## Overview

This crate provides knowledge distillation utilities for training smaller student models to mimic larger teacher models. Every loss is an expression over `pmetal-runtime` arrays, so it runs on the GPU and the student is trained by differentiating it. The teacher is always a constant: no loss sends gradient into it.

`pmetal-trainer` builds its distillation trainers on it (`DistillationTrainer`, the RLKD and cross-vocabulary paths), and the `pmetal` CLI exposes them as `pmetal distill`. It depends only on `pmetal-runtime` from the workspace.

## Installation

```toml
[dependencies]
pmetal-distill = "0.7"
```

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
training loop. A loss takes the teacher's and the student's logits, `[batch, seq, vocab]`,
and returns a lazy scalar to differentiate:

```rust
use pmetal_distill::{DistillConfig, Distiller};
use pmetal_runtime::compat::Array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `DistillConfig::from_yaml_file` reads the same thing from disk.
    let config = DistillConfig::from_yaml("teacher: Qwen/Qwen3-8B\nstudent: Qwen/Qwen3-0.6B\n")?;
    let distiller = Distiller::new(config)?;

    let teacher = Array::from_f32_slice(&[2.0, 0.5, 0.1, 0.1, 0.1, 2.0, 0.5, 0.1], &[1, 2, 4]);
    let flat_student = Array::from_f32_slice(&[0.0; 8], &[1, 2, 4]);

    // (labels, per-token weights, step, total steps)
    let far = distiller.compute_loss(&teacher, &flat_student, None, None, 0, 100)?;
    let same = distiller.compute_loss(&teacher, &teacher, None, None, 0, 100)?;
    let (far, same): (f32, f32) = (far.total.item(), same.total.item());

    assert!(far > 0.01); // the student is not the teacher
    assert!(same.abs() < 1e-5); // a student that is the teacher owes nothing
    Ok(())
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

`DistillConfig` has `teacher`, `student`, `method` (`online`, `offline` or `progressive`),
`loss`, `offline`, `output_path` and `training`. The `loss` section:

| Key | Description | Default |
|-----|-------------|---------|
| `loss_type` | Which loss | `kl_divergence` |
| `temperature` | Softmax temperature | 2.0 |
| `alpha` | Soft/hard label balance | 0.5 |
| `reverse_kl` | Reverse KL (student ‖ teacher) | false |
| `rationale`, `rationale_weight` | Reasoning-aware distillation, and the weight of high-entropy tokens | false, 1.0 |
| `hidden_state` | Layer mapping, loss type and `weight` (default 0.1) for hidden-state alignment | none |
| `attention` | Attention transfer | none |

## Feature flags

None.

## Requirements

- Rust 1.95 or later
- macOS 14 or later on Apple Silicon

## License

MIT OR Apache-2.0
