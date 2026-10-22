//! Knowledge distillation toolkit for PMetal.
//!
//! This crate provides knowledge distillation capabilities with GPU-first architecture
//! optimized for Apple Silicon:
//!
//! - **Loss Functions**: KL Divergence, Jensen-Shannon / generalized JSD, skew KL,
//!   Soft Cross-Entropy, Universal Logit Distillation, MiniLLM, GKD
//! - **Hidden State Alignment**: MSE, Cosine, L1 losses between teacher/student layers
//! - **Offline Distillation**: Compressed logit caching for efficient training
//! - **Progressive Distillation**: Temperature annealing support
//! - **TAID**: Temporally Adaptive Interpolated Distillation (Shing et al., ICLR 2025)
//!
//! # TAID
//!
//! TAID (Shing et al., ICLR 2025) trains the student against an intermediate
//! distribution that starts at the student and moves to the teacher:
//!
//! - the target is the softmax of a logit-space mixture of the (detached)
//!   student and the teacher, `softmax((1 - t)·z_S + t·z_T)`;
//! - `t` advances with the student's measured improvement, never slower than
//!   a linear schedule.
//!
//! ```
//! use pmetal_distill::{TaidConfig, TaidDistiller};
//! use pmetal_runtime::compat::Array;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let teacher = Array::from_f32_slice(&[2.0, 0.5, 0.1, 0.1, 0.1, 2.0, 0.5, 0.1], &[1, 2, 4]);
//! let student = Array::from_f32_slice(&[0.0; 8], &[1, 2, 4]);
//!
//! let mut taid = TaidDistiller::new(TaidConfig::default())?;
//! let out = taid.compute_loss(&teacher, &student, None, None)?;
//! // ... differentiate `out.total`, step the optimizer, then report progress:
//! let objective: f32 = out.distillation.item();
//! taid.observe(f64::from(objective), 0, 100);
//! # Ok(())
//! # }
//! ```
//!
//! # Differentiability
//!
//! Every loss is an expression over `pmetal-runtime` arrays, so it runs on the
//! GPU and the student is trained by differentiating it directly.
//!
//! # Example
//!
//! ```
//! use pmetal_distill::{DistillConfig, Distiller};
//! use pmetal_runtime::compat::Array;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // `DistillConfig::from_yaml_file` reads the same thing from disk.
//! let config = DistillConfig::from_yaml("teacher: Qwen/Qwen3-8B\nstudent: Qwen/Qwen3-0.6B\n")?;
//! let distiller = Distiller::new(config)?;
//!
//! let teacher = Array::from_f32_slice(&[2.0, 0.5, 0.1, 0.1, 0.1, 2.0, 0.5, 0.1], &[1, 2, 4]);
//! let student = Array::from_f32_slice(&[0.0; 8], &[1, 2, 4]);
//! // Use `distiller.compute_loss(..)` inside your own training loop:
//! // (labels, per-token weights, step, total steps)
//! let out = distiller.compute_loss(&teacher, &student, None, None, 0, 100)?;
//! assert!(out.total.item::<f32>() > 0.0);
//! # Ok(())
//! # }
//! ```

mod config;
mod distill;
mod error;
pub mod losses;
mod offline;
pub mod reasoning;
pub mod taid;

pub use config::{
    AttentionConfig, AttentionHeadReduction, AttentionLossType, CompressionMethod, DistillConfig,
    DistillMethod, HiddenStateConfig, HiddenStateLossType, LossConfig, LossType, OfflineConfig,
    TrainingConfig,
};
pub use distill::{DistillLossOutput, Distiller, DistillerBuilder};
pub use error::{DistillError, Result};
pub use losses::{
    AttentionTransferLoss, DistillLoss, GkdLoss, GreedySampler, HiddenStateLoss, HingeRankingLoss,
    JensenShannonLoss, JsdSkewedLoss, KlDivergenceLoss, LogisticRankingLoss, MiniLlmLoss,
    MiniLlmOutput, MseLoss, OnPolicySampler, SampledTokenKl, SampledTokenKlOutput,
    SampledTokenKlTarget, SkewKlLoss, SoftCrossEntropyLoss, SparseKlLoss, TemperatureSampler,
    TvdLoss, UniversalLogitLoss,
};
pub use offline::{
    CacheMetadata, CompressedLogits, LogitCache, LogitCompressor, SparseLogits, SparseTargets,
};
pub use reasoning::RationaleLoss;
pub use taid::{TaidConfig, TaidDistiller, TaidError, TaidLossOutput};

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
