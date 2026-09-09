//! Knowledge distillation toolkit for PMetal.
//!
//! This crate provides knowledge distillation capabilities with GPU-first architecture
//! optimized for Apple Silicon:
//!
//! - **Loss Functions**: KL Divergence, Jensen-Shannon, Soft Cross-Entropy
//! - **Hidden State Alignment**: MSE, Cosine, L1 losses between teacher/student layers
//! - **Offline Distillation**: Compressed logit caching for efficient training
//! - **Progressive Distillation**: Temperature annealing support
//! - **TAID**: Temporally Adaptive Interpolated Distillation (ICLR 2025)
//!
//! # Q4 2025 SOTA: TAID
//!
//! TAID (Temporally Adaptive Interpolated Distillation) is an ICLR 2025 Spotlight
//! paper that prevents mode collapse through adaptive intermediate distributions:
//!
//! - Creates an interpolated target between teacher and student
//! - Adapts interpolation factor based on training progress
//! - Per-sample difficulty awareness for better guidance
//!
//! ```rust,ignore
//! use pmetal_distill::{TaidConfig, TaidDistiller};
//!
//! let distiller = TaidDistiller::new(TaidConfig::default())?;
//! let loss = distiller.compute_loss(&teacher_logits, &student_logits, step, total_steps, None)?;
//! ```
//!
//! # GPU Acceleration
//!
//! When the `metal` feature is enabled (default), all loss implementations
//! automatically use custom Metal kernels with these optimizations:
//!
//! - **Online softmax**: O(1) memory per token instead of O(vocab) probability tensors
//! - **Fused operations**: Temperature scaling + softmax + loss in single kernel pass
//! - **SIMD parallelization**: Optimized for large vocabularies (>1024 tokens)
//!
//! No API changes are needed - GPU acceleration is transparent to the user.
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

// Crate-level lint configuration for ML/GPU code patterns
#![allow(missing_docs)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::type_complexity)]

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
    JensenShannonLoss, JsdSkewedLoss, KlDivergenceLoss, LogisticRankingLoss, MiniLlmLoss, MseLoss,
    OnPolicySampler, SoftCrossEntropyLoss, TvdLoss, UniversalLogitLoss, is_gpu_available,
};
pub use offline::{LogitCache, LogitCompressor};
pub use reasoning::RationaleLoss;
pub use taid::{TaidConfig, TaidDistiller, TaidError, TaidLossOutput, TaidLossType, TaidSchedule};

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
