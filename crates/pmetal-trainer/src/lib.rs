//! Training loops and optimization for PMetal.
//!
//! This crate provides:
//! - Supervised Fine-Tuning (SFT)
//! - LoRA fine-tuning
//! - Direct Preference Optimization (DPO)
//! - Group Relative Policy Optimization (GRPO), with DAPO's clip-higher,
//!   dynamic sampling and overlong penalty as a mode of the same trainer
//! - ORPO (Odds Ratio Preference Optimization)
//! - SimPO (Simple Preference Optimization)
//! - KTO (Kahneman-Tversky Optimization)
//! - Learning rate schedulers
//! - Training callbacks
//! - Parameter grouping for per-layer learning rates
//!
//! # Separate Embedding Learning Rates
//!
//! PMetal supports separate learning rates for embeddings for improved training stability:
//!
//! - Embeddings use a lower learning rate (default 5e-5 vs 2e-4 for LoRA)
//! - Use [`AdamWGroups`] optimizer or the `--embedding-lr` CLI flag
//!
//! ```ignore
//! use pmetal_trainer::{AdamWGroups, AdamWGroupsBuilder};
//!
//! let optimizer = AdamWGroupsBuilder::new(2e-4)
//!     .with_embedding_lr(5e-5)  // recommended default
//!     .with_weight_decay(0.01)
//!     .build()?;
//! ```

// Crate-level lint configuration
#![allow(missing_docs)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::type_complexity)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::manual_saturating_arithmetic)]
#![allow(clippy::unnecessary_unwrap)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::useless_vec)]
#![allow(clippy::io_other_error)]
#![allow(clippy::map_clone)]
#![allow(clippy::borrow_deref_ref)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::derivable_impls)]
#![allow(ambiguous_glob_reexports)]

pub mod orchestrator;
pub mod preference_data;

pub mod adamw_groups;
pub mod adaptive_lr;
pub mod ane_reward;
pub mod callbacks;
pub mod checkpoint;
pub mod contrastive_loss;
pub mod dflash_training;
pub mod diffusion_gemma_train;
pub mod distillation;
pub mod dpo;
pub mod embedding_trainer;
pub mod grpo;
#[cfg(feature = "experimental-trainers")]
pub mod kto;
pub mod logprob_utils;
pub mod mlx_metal_optimizer;
pub mod mtp_training;
#[cfg(feature = "experimental-trainers")]
pub mod orpo;
pub mod paired_preference;
pub mod param_groups;
mod preference_batch;
pub mod pretrain;
pub mod reward_model;
pub mod rlkd;
pub mod scheduler;
pub mod sft;
#[cfg(feature = "experimental-trainers")]
pub mod simpo;
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

pub use adamw_groups::*;
pub use adaptive_lr::{AdaptiveLrConfig, AdaptiveLrController, LrControlCommand, LrEvent};
pub use ane_reward::{AsyncRewardModel, PendingRewards, PipelinedGrpoSession};
pub use callbacks::*;
pub use checkpoint::*;
pub use diffusion_gemma_train::{
    DiffusionGemmaStepStats, DiffusionGemmaTrainConfig, DiffusionGemmaTrainer,
    diffusion_denoising_loss, encoder_ar_loss, load_lora_adapters, save_lora_adapters,
    uniform_categorical_noise,
};
pub use distillation::*;
pub use dpo::*;
pub use embedding_trainer::{
    EmbeddingLossType, EmbeddingResult, EmbeddingTrainer, EmbeddingTrainerConfig,
    EmbeddingTrainerError,
};
pub use grpo::*;
#[cfg(feature = "experimental-trainers")]
pub use kto::*;
pub use mlx_metal_optimizer::{
    MlxMetalOptimizer, MlxMetalOptimizerBuilder, MlxMetalOptimizerConfig, MlxMetalOptimizerError,
    MlxMetalOptimizerResult, is_mlx_metal_optimizer_available,
};
pub use mtp_training::*;
#[cfg(feature = "experimental-trainers")]
pub use orpo::*;
pub use param_groups::*;
pub use rlkd::{RlkdConfig, RlkdStepStats, RlkdTrainer};
pub use scheduler::*;
pub use sft::*;
#[cfg(feature = "experimental-trainers")]
pub use simpo::*;
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
