//! LoRA and QLoRA implementations for PMetal.
//!
//! This crate provides:
//! - LoRA (Low-Rank Adaptation) and DoRA adapters on any architecture the
//!   dispatcher loads ([`AdaptedModel`])
//! - QLoRA: the same, over a base packed to NF4, NVFP4 or 8-bit integers
//!   ([`quantize_base`])
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

pub mod adapted;
pub mod autograd;
mod dora;
mod dynamic;
pub mod gemma4_lora;
pub mod gpt_oss_lora;
pub mod llama4_lora;
mod lora;
pub mod lora_helpers;
pub mod nemotron_h_lora;
mod qlora;
pub mod qwen3_moe_lora;
pub mod qwen3_next_lora;
mod trainable;

pub use adapted::AdaptedModel;
pub use autograd::{
    AccumulatedLoraGrads, LoraForwardSaved, LoraGradContext, LoraGrads, MlpForwardSaved,
    MlpLoraGrads, fused_mlp_backward, fused_mlp_forward, lora_backward, lora_forward_with_grad,
};
pub use dora::*;
pub use dynamic::*;
pub use gemma4_lora::*;
pub use gpt_oss_lora::*;
pub use llama4_lora::*;
pub use lora::*;
pub use lora_helpers::{
    LoraDecoderStack, collect_lora_parameters, count_trainable_params, load_lora_weights_impl,
    save_lora_weights_impl, set_lora_parameters,
};
pub use nemotron_h_lora::*;
pub use qlora::*;
pub use qwen3_moe_lora::*;
pub use qwen3_next_lora::*;
pub use trainable::*;

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
