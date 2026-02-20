//! LoRA and QLoRA implementations for PMetal.
//!
//! This crate provides:
//! - Standard LoRA (Low-Rank Adaptation)
//! - QLoRA (Quantized LoRA with 4-bit base weights)
//! - Q-BLoRA (Quantized Balanced LoRA - addresses underfitting in QLoRA)
//! - DoRA (Weight-Decomposed Low-Rank Adaptation)
//! - Fused training with Metal acceleration (~2x speedup)
//! - Adapter management utilities
//! - LoRA-enabled model architectures
//! - Dynamic model dispatch for architecture-agnostic training
//!
//! # Feature Flags
//!
//! - `metal-fused`: Enable Metal fused kernels for accelerated training
//!
//! # Architecture-Agnostic Training
//!
//! Use [`DynamicLoraModel`] to automatically detect and load the correct
//! model architecture for training:
//!
//! ```ignore
//! use pmetal_lora::DynamicLoraModel;
//! use pmetal_core::LoraConfig;
//!
//! // Auto-detects Llama, Qwen2, Qwen3, etc.
//! let model = DynamicLoraModel::from_pretrained("/path/to/model", lora_config)?;
//! ```

// Crate-level lint configuration for ML/LoRA code patterns
#![allow(missing_docs)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::type_complexity)]

mod adapter;
pub mod arch_config;
pub mod autograd;
pub mod custom_autograd_trainer;
pub mod custom_backward;
pub mod custom_training;
pub mod custom_training_step;
mod dora;
mod dynamic;
pub mod fused_training;
pub mod galore;
pub mod gemma_lora;
pub mod gemma_qlora;
pub mod generic_lora;
pub mod llama_lora;
pub mod llama_qlora;
mod lora;
pub mod mistral_lora;
pub mod mistral_qlora;
mod patcher;
pub mod phi_lora;
mod qblora;
mod qlora;
pub mod qwen3_lora;
pub mod qwen3_qlora;
mod trainable;

pub use adapter::*;
pub use arch_config::LoraArchitectureConfig;
pub use autograd::{
    fused_mlp_backward, fused_mlp_forward, lora_backward, lora_forward_with_grad,
    AccumulatedLoraGrads, LoraForwardSaved, LoraGradContext, LoraGrads, MlpForwardSaved,
    MlpLoraGrads,
};
pub use custom_autograd_trainer::{
    mlp_backward, CustomAutogradTrainer, LayerForwardState, LayerGradients, ModelForwardState,
};
pub use custom_backward::{
    attention_backward, attention_forward_with_grad, rmsnorm_backward, rmsnorm_forward_with_grad,
    rope_backward, rope_forward_with_grad, silu_backward, silu_forward_with_grad, AttentionSaved,
    DecoderLayerGrads, DecoderLayerSaved, RmsNormSaved, RopeSaved, SiluSaved,
};
pub use custom_training::{
    CustomLoraTrainer, LayerSavedState, LoraGradAccumulator, ModelSavedState,
};
pub use custom_training_step::{Qwen3CustomTrainer, Qwen3LayerSaved, Qwen3ModelSaved};
pub use dora::*;
pub use dynamic::*;
pub use fused_training::*;
pub use galore::{
    GaloreConfig, GaloreParamState, GaloreProjectionState, GaloreProjectionType, GaloreProjector,
};
pub use gemma_lora::*;
pub use gemma_qlora::*;
pub use llama_lora::*;
pub use llama_qlora::*;
pub use lora::*;
pub use mistral_lora::*;
pub use mistral_qlora::*;
pub use patcher::*;
pub use phi_lora::*;
pub use qblora::*;
pub use qlora::*;
pub use qwen3_lora::*;
pub use qwen3_qlora::*;
pub use trainable::*;
