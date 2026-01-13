//! MLX backend implementation for PMetal LLM fine-tuning.
//!
//! This crate provides the MLX-based implementation of PMetal's core
//! abstractions, including:
//!
//! - Optimized kernels for attention, RoPE, and activations
//! - Gradient checkpointing for memory-efficient training
//! - NF4/FP4/Int8 quantization implementations
//! - Memory management utilities for Apple Silicon
//! - KV caching for efficient inference
//! - Sequence packing for efficient SFT training
//! - NEFTune for improved fine-tuning quality
//! - Mixture of Experts (MoE) for sparse models
//! - Speculative decoding for faster inference

#![warn(missing_docs)]
#![allow(unsafe_code)]
#![allow(ambiguous_glob_reexports)]
#![allow(unused_imports)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::expect_fun_call)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::type_complexity)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::arc_with_non_send_sync)]
#![allow(dead_code)]

pub mod attention;
pub mod bridge;
pub mod error;
pub mod fp8_quantization;
pub mod gradient_checkpoint;
pub mod grouped_gemm_moe;
pub mod kernels;
pub mod kv_cache;
pub mod memory;
pub mod moe;
pub mod neftune;
pub mod offloading;
pub mod prefix_cache;
pub mod quantization;
pub mod sequence_packing;
pub mod smart_checkpoint;
pub mod speculative;

mod array_ext;

pub use array_ext::*;
pub use bridge::MlxMetalBridge;
pub use fp8_quantization::*;
pub use gradient_checkpoint::*;
pub use grouped_gemm_moe::*;
pub use kv_cache::*;
pub use moe::*;
pub use neftune::*;
pub use offloading::*;
pub use prefix_cache::*;
pub use sequence_packing::*;
pub use smart_checkpoint::*;
pub use speculative::*;

// Re-export mlx-rs types for convenience
pub use mlx_rs::builder::Builder;
pub use mlx_rs::error::{Exception, Result};
pub use mlx_rs::module::{Module, ModuleParameters, Param};
pub use mlx_rs::nn::Linear;
pub use mlx_rs::{Array, Dtype};

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::array_ext::*;
    pub use crate::attention::*;
    pub use crate::fp8_quantization::*;
    pub use crate::gradient_checkpoint::*;
    pub use crate::grouped_gemm_moe::*;
    pub use crate::kernels::*;
    pub use crate::kv_cache::*;
    pub use crate::memory::*;
    pub use crate::moe::*;
    pub use crate::neftune::*;
    pub use crate::offloading::*;
    pub use crate::prefix_cache::*;
    pub use crate::quantization::*;
    pub use crate::sequence_packing::*;
    pub use crate::smart_checkpoint::*;
    pub use crate::speculative::*;
    pub use mlx_rs::builder::Builder;
    pub use mlx_rs::module::{Module, ModuleParameters, Param};
    pub use mlx_rs::{Array, Dtype};
}
