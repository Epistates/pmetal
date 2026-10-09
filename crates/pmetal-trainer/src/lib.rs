//! Training loops and optimization for PMetal.
//!
//! This crate provides:
//! - Supervised Fine-Tuning (SFT)
//! - LoRA fine-tuning
//! - Offline preference optimization ([`preference`]): DPO, IPO, hinge,
//!   SimPO and ORPO on preference pairs, and KTO on labelled completions
//! - Group Relative Policy Optimization (GRPO), with DAPO's clip-higher,
//!   dynamic sampling and overlong penalty as a mode of the same trainer
//! - Learning rate schedulers
//! - Training callbacks
//! - Parameter grouping for per-layer learning rates
//!
//! # Separate Embedding Learning Rates
//!
//! PMetal supports separate learning rates for embeddings for improved training stability:
//!
//! - Embeddings use a lower learning rate (default 5e-5 vs 2e-4 for LoRA)
//! - Use [`ParamGroupOptimizer`] or the `--embedding-lr` CLI flag
//!
//! ```ignore
//! use pmetal_core::OptimizerType;
//! use pmetal_trainer::ParamGroupOptimizerBuilder;
//!
//! let optimizer = ParamGroupOptimizerBuilder::new(OptimizerType::AdamW, 2e-4)
//!     .with_embedding_lr(5e-5)  // recommended default
//!     .with_weight_decay(0.01)
//!     .build();
//! ```

pub mod orchestrator;

pub mod adaptive_lr;
pub mod ane_reward;
pub mod callbacks;
pub mod checkpoint;
pub mod contrastive_loss;
pub mod dflash_training;
pub mod diffusion_gemma_train;
pub mod distillation;
pub mod embedding_trainer;
pub mod grpo;
pub mod logprob_utils;
pub mod mlx_metal_optimizer;
pub mod mtp_training;
pub mod optimizer;
pub mod param_groups;
pub mod preference;
pub mod pretrain;
pub mod reward_model;
pub mod rlkd;
pub mod scheduler;
pub mod sft;
pub mod step_check;
pub mod tensorboard;
pub mod training_loop;

#[cfg(feature = "distributed")]
pub mod distributed_bridge;

#[cfg(feature = "ane")]
pub mod ane_training;
#[cfg(feature = "distributed")]
pub use distributed_bridge::{DistributedGradientSync, create_distributed_context};

#[cfg(feature = "ane")]
pub use ane_training::{AneTrainingLoop, AneTrainingLoopConfig};
#[cfg(feature = "ane")]
pub use pmetal_metal::ane::dynamic_trainer::{
    DynamicAneTrainer, DynamicAneTrainerConfig, IGNORE_TARGET, VocabMap,
};

pub use adaptive_lr::{AdaptiveLrConfig, AdaptiveLrController, LrControlCommand, LrEvent};
pub use ane_reward::{AsyncRewardModel, PendingRewards, PipelinedGrpoSession};
pub use callbacks::*;
pub use checkpoint::*;
pub use diffusion_gemma_train::{
    DiffusionGemmaStepStats, DiffusionGemmaTrainConfig, DiffusionGemmaTrainer,
    diffusion_denoising_loss, encoder_ar_loss, quantize_diffusion_gemma_base,
    uniform_categorical_noise,
};
pub use distillation::*;
pub use embedding_trainer::{
    EmbeddingLossType, EmbeddingResult, EmbeddingTrainer, EmbeddingTrainerConfig,
    EmbeddingTrainerError,
};
pub use grpo::*;
pub use mlx_metal_optimizer::{
    MlxMetalOptimizer, MlxMetalOptimizerBuilder, MlxMetalOptimizerConfig, MlxMetalOptimizerError,
    MlxMetalOptimizerResult, is_mlx_metal_optimizer_available,
};
pub use mtp_training::*;
pub use optimizer::{
    Adafactor, Lion, MomentumSgd, ParamGroupOptimizer, ParamGroupOptimizerBuilder, SecondMoment,
    TrainOptimizer,
};
pub use param_groups::*;
pub use preference::{
    KtoConfig, KtoSample, KtoTrainer, PreferenceError, PreferenceLoss, PreferencePair,
    PreferenceResult, PreferenceStepMetrics, PreferenceTrainer,
};
pub use rlkd::{RlkdConfig, RlkdStepStats, RlkdTrainer};
pub use scheduler::*;
pub use sft::*;
pub use training_loop::*;

// Orchestrator re-exports
pub use orchestrator::{
    DispatchConfig, FullTrainingConfig, PhaseCallback, QLoraOrchConfig, QuantizationScheme,
    TrainingJobConfig, TrainingPhase, TrainingResult, resolve_dataset_path,
};

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
