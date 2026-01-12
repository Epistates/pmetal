//! GPU compute kernels for ML operations.
//!
//! This module contains optimized Metal kernels for machine learning operations,
//! with a focus on transformer attention mechanisms, high-performance sampling,
//! and efficient training operations.

pub mod flash_attention;
pub mod fp8_training;
pub mod fused_cross_entropy;
pub mod fused_distill;
pub mod fused_lora;
pub mod fused_norm_lora;
pub mod fused_rope;
pub mod fused_sampler;
pub mod fused_swiglu;
pub mod fused_training;
pub mod moe;

// Re-export main types
pub use flash_attention::{
    FlashAttention, FlashAttentionConfig, FlashAttentionOutput,
    FlashAttentionVarlen, FlashAttentionVarlenConfig, FlashAttentionVarlenOutput,
};
pub use fused_cross_entropy::{
    FusedCrossEntropy, FusedCrossEntropyConfig, FusedCrossEntropyOutput,
    // The key unsloth optimization: fused linear + cross-entropy
    FusedLinearCrossEntropy, FusedLinearCrossEntropyConfig, FusedLinearCrossEntropyOutput,
};
pub use fused_distill::{
    FusedDistill, FusedDistillConfig, FusedDistillOutput, DistillLossType,
    FusedHiddenAlign, HiddenAlignConfig, HiddenAlignLossType,
};
pub use fused_lora::{FusedLora, FusedLoraConfig, FusedLoraOutput};
pub use fused_norm_lora::{FusedNormLora, FusedNormLoraConfig, FusedNormLoraOutput};
pub use fused_rope::{FusedRoPE, FusedRoPEConfig, RoPECache};
pub use fused_sampler::{FusedSampler, FusedSamplerConfig, SamplingParams};
pub use fused_swiglu::{FusedMLP, FusedMLPOutput, FusedSwiGLU, FusedSwiGLUConfig, FusedSwiGLUOutput};
pub use moe::{MoeConfig, MoeGemmOutput, MoeKernel, MoeRouting};
pub use fp8_training::{
    Fp8Format, Fp8TrainingConfig, Fp8TrainingKernel, Fp8QuantOutput, Fp8GemmOutput, Fp8DynamicScale,
};
pub use fused_training::{
    AdamWConfig, BatchedCommandBuffer, BatchCompletionToken, FusedAdamW,
    FusedCrossEntropyTraining, FusedGradientClipping, FusedTrainingCoordinator, ParamInfo,
};
