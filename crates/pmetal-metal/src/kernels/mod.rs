//! GPU compute kernels for ML operations.
//!
//! This module contains optimized Metal kernels for machine learning operations,
//! with a focus on transformer attention mechanisms and efficient training
//! operations.

pub mod batch_encoder;
pub mod dispatch;
pub mod dw_gemm;
pub mod flash_attention;
pub mod fused_lora;
pub mod fused_moe;
pub mod fused_norm_lora;
pub mod fused_rope;
pub mod fused_swiglu;
pub mod fused_training;
pub mod mpp_bench;
pub mod mpp_dispatch;
pub mod mpp_flash_attention;
pub mod mpp_gemm;
pub mod turboquant;

#[cfg(has_metal4)]
pub mod mpp_dw_gemm;
#[cfg(has_metal4)]
pub mod mpp_fused_lora;
#[cfg(has_metal4)]
pub mod mpp_fused_norm_lora;
#[cfg(has_metal4)]
pub mod mpp_fused_rope;
#[cfg(has_metal4)]
pub mod mpp_fused_swiglu;

// Re-export main types
pub use dw_gemm::{DwGemm, GPU_DW_MIN_DIM, ScratchPool};
pub use flash_attention::{
    FlashAttention, FlashAttentionConfig, FlashAttentionOutput, FlashAttentionVarlen,
    FlashAttentionVarlenConfig, FlashAttentionVarlenOutput,
};
pub use fused_lora::{FusedLora, FusedLoraConfig, FusedLoraOutput};
pub use fused_moe::{ExpertBits, ExpertWeightBuffers, FusedMoeExpert, FusedMoeExpertConfig};
pub use fused_norm_lora::{FusedNormLora, FusedNormLoraConfig, FusedNormLoraOutput};
pub use fused_rope::{FusedRoPE, FusedRoPEConfig, RoPECache};
pub use fused_swiglu::{
    FusedMLP, FusedMLPOutput, FusedSwiGLU, FusedSwiGLUConfig, FusedSwiGLUOutput,
};
pub use fused_training::{
    AdamWConfig, AdamWScalars, BatchedCommandBuffer, FusedAdamW, FusedGradientClipping, ParamInfo,
};
pub use mpp_flash_attention::{
    MppFlashAttention, MppFlashAttentionConfig, MppFlashAttentionOutput,
};
pub use turboquant::TurboQuantTransform;
