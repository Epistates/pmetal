//! Knowledge distillation toolkit for PMetal.
//!
//! This crate provides knowledge distillation capabilities with GPU-first architecture
//! optimized for Apple Silicon:
//!
//! - **Loss Functions**: KL Divergence, Jensen-Shannon, Soft Cross-Entropy
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
//! ```rust,ignore
//! use pmetal_distill::{TaidConfig, TaidDistiller};
//!
//! let mut taid = TaidDistiller::new(TaidConfig::default())?;
//! let out = taid.compute_loss(&teacher_logits, &student_logits, Some(&mask), None)?;
//! // ... differentiate `out.total`, step the optimizer ...
//! taid.observe(out.distillation.item::<f32>() as f64, step, total_steps);
//! ```
//!
//! # Differentiability
//!
//! Every loss is an MLX expression, so it runs on the GPU through MLX and the
//! student is trained by differentiating it directly.
//!
//! # Example
//!
//! ```rust,ignore
//! use pmetal_distill::{DistillConfig, Distiller};
//!
//! let config = DistillConfig::from_yaml_file("distill_config.yaml")?;
//! let distiller = Distiller::new(config)?;
//! // Use `distiller.compute_loss(...)` inside your own training loop.
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
pub use distill::{DistillLossOutput, Distiller, DistillerBuilder, run_distillation};
pub use error::{DistillError, Result};
pub use losses::{
    AttentionTransferLoss, DistillLoss, GkdLoss, GreedySampler, HiddenStateLoss, HingeRankingLoss,
    JensenShannonLoss, JsdSkewedLoss, KlDivergenceLoss, LogisticRankingLoss, MiniLlmLoss,
    MiniLlmOutput, MseLoss, OnPolicySampler, SkewKlLoss, SoftCrossEntropyLoss, TemperatureSampler,
    TvdLoss, UniversalLogitLoss,
};
pub use offline::{LogitCache, LogitCompressor};
pub use reasoning::RationaleLoss;
pub use taid::{TaidConfig, TaidDistiller, TaidError, TaidLossOutput};

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
