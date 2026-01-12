//! Training loops and optimization for PMetal.
//!
//! This crate provides:
//! - Supervised Fine-Tuning (SFT)
//! - LoRA fine-tuning
//! - Direct Preference Optimization (DPO)
//! - Group Relative Policy Optimization (GRPO)
//! - DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)
//! - GSPO (Group Sequence Policy Optimization)
//! - LLaDA-style Diffusion Training
//! - Learning rate schedulers
//! - Training callbacks
//! - Parameter grouping for per-layer learning rates
//!
//! # Q4 2025 SOTA Algorithms
//!
//! This crate includes the latest SOTA algorithms from Q4 2025:
//!
//! - **DAPO**: ByteDance's algorithm with Clip-Higher, Dynamic Sampling,
//!   Token-Level Policy Gradient, and Overlong Reward Penalty
//! - **GSPO**: Group Sequence Policy Optimization with equal token weighting
//!   to fix GRPO length bias
//!
//! # Separate Embedding Learning Rates
//!
//! PMetal supports separate learning rates for embeddings, matching Unsloth's
//! approach for improved training stability:
//!
//! - Embeddings use a lower learning rate (default 5e-5 vs 2e-4 for LoRA)
//! - Use [`AdamWGroups`] optimizer or the `--embedding-lr` CLI flag
//!
//! ```ignore
//! use pmetal_trainer::{AdamWGroups, AdamWGroupsBuilder};
//!
//! let optimizer = AdamWGroupsBuilder::new(2e-4)
//!     .with_embedding_lr(5e-5)  // Unsloth's default
//!     .with_weight_decay(0.01)
//!     .build()?;
//! ```

#![warn(missing_docs)]

pub mod adam8bit;
pub mod adamw_groups;
pub mod callbacks;
pub mod checkpoint;
pub mod checkpointing;
pub mod dapo;
pub mod diffusion;
pub mod distillation;
pub mod explicit_state_compile;
pub mod ffi_compile;
pub mod jit_compile;
pub mod dpo;
pub mod gspo;
pub mod metal_fused;
pub mod mlx_metal_optimizer;
pub mod online_dpo;
pub mod grpo;
pub mod kto;
pub mod lora_trainer;
pub mod orpo;
pub mod param_groups;
pub mod ppo;
pub mod scheduler;
pub mod sft;
pub mod simpo;
pub mod training_loop;

pub use adam8bit::*;
pub use adamw_groups::*;
pub use callbacks::*;
pub use checkpoint::*;
pub use checkpointing::*;
pub use dapo::*;
pub use diffusion::*;
pub use distillation::*;
pub use dpo::*;
pub use explicit_state_compile::*;
pub use ffi_compile::*;
pub use jit_compile::*;
pub use gspo::*;
pub use metal_fused::*;
pub use mlx_metal_optimizer::{
    MlxMetalOptimizer, MlxMetalOptimizerBuilder, MlxMetalOptimizerConfig,
    MlxMetalOptimizerError, MlxMetalOptimizerResult, is_mlx_metal_optimizer_available,
};
// Re-export online_dpo selectively to avoid ambiguous RewardFunction with grpo
pub use online_dpo::{
    OnlineDpoConfig, OnlineDpoTrainer, OnlinePreferencePair, OnlineDpoIterationStats,
    RewardFunction as OnlineRewardFunction, LengthRewardFunction,
};
pub use grpo::*;
pub use kto::*;
pub use lora_trainer::*;
pub use orpo::*;
pub use param_groups::*;
pub use ppo::*;
pub use scheduler::*;
pub use sft::*;
pub use simpo::*;
pub use training_loop::*;
