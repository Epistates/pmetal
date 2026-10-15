//! Metal GPU compute kernels for PMetal.
//!
//! This crate provides Metal compute kernels for the operations the tensor
//! runtime has no op for, optimized for Apple Silicon:
//!
//! - **Quantized and MoE GEMMs**: dequantizing matmuls, grouped expert GEMMs
//!   and fused expert MLPs
//! - **Training kernels**: fused cross-entropy (including the chunked linear +
//!   cross-entropy that never materializes the logits), fused AdamW, FP8
//! - **DwGemm**: the f32 weight-gradient GEMM the ANE trainer batches into one
//!   command buffer
//! - **Merging and KV-cache transforms**: model merging, TurboQuant
//! - **Device Management**: Thread-safe Metal device and command queue handling
//! - **Buffer Abstractions**: Type-safe GPU buffer management with unified memory
//!
//! Dense attention, RoPE, SwiGLU, LoRA and GEMM are deliberately absent: the
//! runtime runs them in its lazy graph, with autodiff, faster than the
//! hand-written kernels this crate used to carry for them.
//!
//! # Architecture
//!
//! The crate is designed to work alongside MLX, sharing unified memory:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Unified Memory                           │
//! │                                                             │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
//! │  │  MLX Array  │◄──►│ Metal Buffer│◄──►│ CPU Access  │     │
//! │  └─────────────┘    └─────────────┘    └─────────────┘     │
//! │                                                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```

#![warn(missing_docs)]
#![cfg(target_os = "macos")]

pub mod accelerate;
#[cfg(feature = "ane")]
pub mod ane;
pub mod buffer;
pub(crate) mod chip;
pub mod context;
pub mod error;
pub mod expert_buffer;
pub mod kernels;
pub(crate) mod msl_layout;
pub mod neon_convert;
pub mod pipeline;
pub mod tuna;
pub mod view;

pub use buffer::{AsMetalBuffer, BufferUsage, MetalBuffer};
pub use context::MetalContext;
pub use error::{MetalError, Result};
pub use kernels::fused_cross_entropy::{
    FusedCrossEntropy,
    FusedCrossEntropyConfig,
    FusedCrossEntropyOutput,
    // Chunked vocabulary optimization: fused linear + cross-entropy (skips logits materialization)
    FusedLinearCrossEntropy,
    FusedLinearCrossEntropyConfig,
    FusedLinearCrossEntropyOutput,
};
pub use kernels::fused_merge::{
    FusedMergeMetal, MergeConfig as FusedMergeConfig, TensorInfo, build_merge_config,
    build_tensor_info,
};
pub use kernels::fused_moe::{
    ExpertBits, ExpertWeightBuffers, FusedMoeExpert, FusedMoeExpertConfig,
};
pub use kernels::mpp_quantized::{MppQuantizedGemm, MppQuantizedGemmConfig};
pub use kernels::turboquant::TurboQuantTransform;
pub use pipeline::{FunctionConstant, PipelineCache};
pub use view::{MetalBufferView, MetalBufferViewF16, MetalBufferViewF32, metal_buffer_from_ptr};

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::buffer::{BufferUsage, MetalBuffer};
    pub use crate::context::MetalContext;
    pub use crate::error::{MetalError, Result};
    pub use crate::kernels::fused_cross_entropy::{
        FusedCrossEntropy, FusedCrossEntropyConfig, FusedLinearCrossEntropy,
        FusedLinearCrossEntropyConfig,
    };
    pub use crate::pipeline::PipelineCache;
}

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
