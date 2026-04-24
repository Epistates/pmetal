//! LoRA and QLoRA implementations for PMetal.
//!
//! This crate provides:
//! - Standard LoRA (Low-Rank Adaptation)
//! - QLoRA (Quantized LoRA with 4-bit base weights)
//! - Q-BLoRA (Quantized Balanced LoRA - addresses underfitting in QLoRA)
//! - DoRA (Weight-Decomposed Low-Rank Adaptation)
//! - GaLore (Gradient Low-Rank Projection, ICML 2024)
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
mod dora;
mod dynamic;
mod dynamic_qlora;
pub mod galore;
pub mod deepseek_lora;
pub mod deepseek_qlora;
pub mod gemma4_lora;
pub mod gemma4_qlora;
pub mod gemma_lora;
pub mod gemma_qlora;
pub mod gpt_oss_lora;
pub mod gpt_oss_qlora;
pub mod granite_lora;
pub mod granite_qlora;
pub mod llama4_lora;
pub mod llama4_qlora;
pub mod llama_lora;
pub mod llama_qlora;
mod lora;
pub mod lora_helpers;
pub mod mistral_lora;
pub mod mistral_qlora;
pub mod mllama_lora;
pub mod nemotron_h_lora;
pub mod nemotron_h_qlora;
mod patcher;
pub mod phi_lora;
mod qblora;
mod qlora;
pub mod qwen3_lora;
pub mod qwen3_moe_lora;
pub mod qwen3_moe_qlora;
pub mod qwen3_next_lora;
pub mod qwen3_next_qlora;
pub mod qwen3_qlora;
mod trainable;

pub use adapter::*;
pub use arch_config::LoraArchitectureConfig;
pub use autograd::{
    AccumulatedLoraGrads, LoraForwardSaved, LoraGradContext, LoraGrads, MlpForwardSaved,
    MlpLoraGrads, fused_mlp_backward, fused_mlp_forward, lora_backward, lora_forward_with_grad,
};
pub use dora::*;
pub use dynamic::*;
pub use dynamic_qlora::DynamicQloraModel;
pub use galore::{
    GaloreConfig, GaloreParamState, GaloreProjectionState, GaloreProjectionType, GaloreProjector,
};
pub use deepseek_lora::*;
pub use deepseek_qlora::*;
pub use gemma_lora::*;
pub use gemma_qlora::*;
pub use gemma4_lora::*;
pub use gemma4_qlora::*;
pub use gpt_oss_lora::*;
pub use gpt_oss_qlora::*;
pub use granite_lora::*;
pub use granite_qlora::*;
pub use llama4_lora::*;
pub use llama4_qlora::*;
pub use llama_lora::*;
pub use llama_qlora::*;
pub use lora::*;
pub use lora_helpers::{
    LoraDecoderStack, collect_lora_parameters, count_trainable_params, load_lora_weights_impl,
    save_lora_weights_impl, set_lora_parameters,
};
pub use mistral_lora::*;
pub use mistral_qlora::*;
pub use mllama_lora::*;
pub use nemotron_h_lora::*;
pub use nemotron_h_qlora::*;
pub use patcher::*;
pub use phi_lora::*;
pub use qblora::*;
pub use qlora::*;
pub use qwen3_lora::*;
pub use qwen3_moe_lora::*;
pub use qwen3_moe_qlora::*;
pub use qwen3_next_lora::*;
pub use qwen3_next_qlora::*;
pub use qwen3_qlora::*;
pub use trainable::*;
