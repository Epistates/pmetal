//! Metal GPU compute kernels for PMetal.
//!
//! This crate provides high-performance Metal compute kernels optimized for
//! Apple Silicon, including:
//!
//! - **FlashAttention**: Memory-efficient attention with O(n) memory complexity
//!   - Forward pass with online softmax
//!   - Backward pass (dQ, dK, dV) for training
//!   - Support for GQA/MQA, causal masking, sliding window
//!
//! - **FusedLora**: Optimized LoRA training kernels (~2x speedup)
//!   - Fused forward: y = x @ W.T + scale * (x @ A.T) @ B.T
//!   - Fused backward: Gradient computation for A, B, and x
//!   - Intermediate caching in threadgroup memory
//!
//! - **Device Management**: Thread-safe Metal device and command queue handling
//!
//! - **Buffer Abstractions**: Type-safe GPU buffer management with unified memory
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
//!
//! # Example
//!
//! ```ignore
//! use pmetal_metal::{MetalContext, FlashAttention, FlashAttentionConfig};
//!
//! // Initialize Metal context (cached globally)
//! let ctx = MetalContext::global();
//!
//! // Create FlashAttention instance
//! let config = FlashAttentionConfig {
//!     num_heads: 32,
//!     num_kv_heads: 8,  // GQA
//!     head_dim: 128,
//!     is_causal: true,
//!     ..Default::default()
//! };
//!
//! let flash_attn = FlashAttention::new(&ctx, config)?;
//!
//! // Run forward pass
//! let (output, logsumexp) = flash_attn.forward(&q, &k, &v)?;
//!
//! // Run backward pass (for training)
//! let (dq, dk, dv) = flash_attn.backward(&q, &k, &v, &output, &d_out, &logsumexp)?;
//! ```

#![warn(missing_docs)]
#![cfg(target_os = "macos")]

pub mod async_scheduler;
pub mod bridge;
pub mod buffer;
pub mod context;
pub mod error;
pub mod kernels;
pub mod pipeline;
pub mod tuna;

pub use bridge::{metal_buffer_from_ptr, MetalBufferView, MetalBufferViewF16, MetalBufferViewF32};
pub use buffer::{BufferUsage, MetalBuffer};
pub use context::MetalContext;
pub use error::{MetalError, Result};
pub use kernels::flash_attention::{
    FlashAttention, FlashAttentionConfig, FlashAttentionOutput, FlashAttentionVarlen,
    FlashAttentionVarlenConfig, FlashAttentionVarlenOutput,
};
pub use kernels::fused_cross_entropy::{
    FusedCrossEntropy,
    FusedCrossEntropyConfig,
    FusedCrossEntropyOutput,
    // Key unsloth optimization: fused linear + cross-entropy (skips logits materialization)
    FusedLinearCrossEntropy,
    FusedLinearCrossEntropyConfig,
    FusedLinearCrossEntropyOutput,
};
pub use kernels::fused_lora::{FusedLora, FusedLoraConfig, FusedLoraOutput};
pub use kernels::fused_norm_lora::{FusedNormLora, FusedNormLoraConfig, FusedNormLoraOutput};
pub use kernels::fused_rope::{FusedRoPE, FusedRoPEConfig, RoPECache};
pub use kernels::fused_sampler::{AsMetalBuffer, FusedSampler, FusedSamplerConfig, SamplingParams};
pub use kernels::fused_swiglu::{
    FusedMLP, FusedMLPOutput, FusedSwiGLU, FusedSwiGLUConfig, FusedSwiGLUOutput,
};
pub use pipeline::{FunctionConstant, PipelineCache};

// Async command buffer scheduling
pub use async_scheduler::{
    AsyncBatchBuilder, AsyncScheduler, CompletionToken, DoubleBuffer, GpuCompletionToken,
    InFlightBuffer, SchedulerStats, TripleBuffer, DEFAULT_GPU_TIMEOUT,
};

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::buffer::{BufferUsage, MetalBuffer};
    pub use crate::context::MetalContext;
    pub use crate::error::{MetalError, Result};
    pub use crate::kernels::flash_attention::{
        FlashAttention, FlashAttentionConfig, FlashAttentionVarlen, FlashAttentionVarlenConfig,
    };
    pub use crate::kernels::fused_cross_entropy::{
        FusedCrossEntropy, FusedCrossEntropyConfig, FusedLinearCrossEntropy,
        FusedLinearCrossEntropyConfig,
    };
    pub use crate::kernels::fused_lora::{FusedLora, FusedLoraConfig};
    pub use crate::kernels::fused_sampler::{FusedSampler, FusedSamplerConfig};
    pub use crate::pipeline::PipelineCache;
}
