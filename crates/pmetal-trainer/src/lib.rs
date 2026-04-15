//! Training loops and optimization for PMetal.
//!
//! This crate provides:
//! - Supervised Fine-Tuning (SFT)
//! - LoRA fine-tuning
//! - Direct Preference Optimization (DPO)
//! - Group Relative Policy Optimization (GRPO)
//! - DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)
//! - GSPO (Group Sequence Policy Optimization)
//! - PPO (Proximal Policy Optimization)
//! - ORPO (Odds Ratio Preference Optimization)
//! - SimPO (Simple Preference Optimization)
//! - KTO (Kahneman-Tversky Optimization)
//! - Online DPO with reward models
//! - LLaDA-style Diffusion Training
//! - Learning rate schedulers
//! - Training callbacks
//! - Parameter grouping for per-layer learning rates
//!
//! # Q4 2025 SOTA Algorithms
//!
//! - **DAPO**: ByteDance's algorithm with Clip-Higher, Dynamic Sampling,
//!   Token-Level Policy Gradient, and Overlong Reward Penalty
//! - **GSPO**: Group Sequence Policy Optimization with equal token weighting
//!   to fix GRPO length bias
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

#[cfg(feature = "experimental-trainers")]
pub mod adam8bit;
pub mod adamw_groups;
pub mod adaptive_lr;
pub mod ane_reward;
pub mod callbacks;
pub mod checkpoint;
pub mod checkpointing;
pub mod contrastive_loss;
pub mod dapo;
pub mod dflash_training;
#[cfg(feature = "experimental-trainers")]
pub mod diffusion;
pub mod distillation;
pub mod dpo;
pub mod embedding_trainer;
pub mod explicit_state_compile;
pub mod ffi_compile;
pub mod grpo;
pub mod gspo;
pub mod jit_compile;
#[cfg(feature = "experimental-trainers")]
pub mod kto;
pub mod logprob_utils;
pub mod lora_trainer;
pub mod metal_fused;
pub mod mlx_metal_optimizer;
#[cfg(feature = "experimental-trainers")]
pub mod online_dpo;
#[cfg(feature = "experimental-trainers")]
pub mod orpo;
pub mod param_groups;
#[cfg(feature = "experimental-trainers")]
pub mod ppo;
mod preference_batch;
pub mod reasoning_template;
pub mod reward_model;
pub mod rlkd;
#[cfg(feature = "experimental-trainers")]
pub mod schedule_free;
pub mod scheduler;
pub mod sft;
#[cfg(feature = "experimental-trainers")]
pub mod simpo;
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
    DynamicAneTrainer, DynamicAneTrainerConfig, VocabMap,
};

#[cfg(feature = "experimental-trainers")]
pub use adam8bit::*;
pub use adamw_groups::*;
pub use adaptive_lr::{AdaptiveLrConfig, AdaptiveLrController, LrControlCommand, LrEvent};
pub use ane_reward::{AsyncRewardModel, PendingRewards, PipelinedGrpoSession};
pub use callbacks::*;
pub use checkpoint::*;
pub use checkpointing::*;
pub use dapo::*;
#[cfg(feature = "experimental-trainers")]
pub use diffusion::*;
pub use distillation::*;
pub use dpo::*;
pub use explicit_state_compile::*;
pub use ffi_compile::*;
pub use gspo::*;
pub use jit_compile::*;
pub use metal_fused::*;
pub use mlx_metal_optimizer::{
    MlxMetalOptimizer, MlxMetalOptimizerBuilder, MlxMetalOptimizerConfig, MlxMetalOptimizerError,
    MlxMetalOptimizerResult, is_mlx_metal_optimizer_available,
};
// Re-export online_dpo selectively to avoid ambiguous RewardFunction with grpo
pub use embedding_trainer::{
    EmbeddingLossType, EmbeddingResult, EmbeddingTrainer, EmbeddingTrainerConfig,
    EmbeddingTrainerError,
};
pub use grpo::*;
#[cfg(feature = "experimental-trainers")]
pub use kto::*;
pub use lora_trainer::*;
#[cfg(feature = "experimental-trainers")]
pub use online_dpo::{
    LengthRewardFunction, OnlineDpoConfig, OnlineDpoIterationStats, OnlineDpoTrainer,
    OnlinePreferencePair, RewardFunction as OnlineRewardFunction,
};
#[cfg(feature = "experimental-trainers")]
pub use orpo::*;
pub use param_groups::*;
#[cfg(feature = "experimental-trainers")]
pub use ppo::*;
pub use rlkd::{RlkdConfig, RlkdStepStats, RlkdTrainer};
#[cfg(feature = "experimental-trainers")]
pub use schedule_free::{
    ScheduleFreeConfig, ScheduleFreeError, ScheduleFreeOptimizer, ScheduleFreeResult,
};
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
