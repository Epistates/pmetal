//! GPU compute kernels for ML operations.
//!
//! Dense attention, RoPE, SwiGLU, LoRA and GEMM are not here: the tensor
//! runtime's own ops run them inside the lazy graph, with autodiff, faster
//! than any kernel this crate carried for them. What remains are the kernels
//! the runtime has no op for (quantized and MoE expert GEMMs, fused losses and
//! optimizers, merging, KV-cache transforms) and the weight-gradient GEMM the
//! ANE trainer batches into its own command buffer.

pub mod batch_encoder;
pub mod dequant;
pub mod dispatch;
pub mod dw_gemm;
pub mod fp8_training;
pub mod fused_cross_entropy;
pub mod fused_merge;
pub mod fused_moe;
pub mod fused_training;
pub mod moe;
pub mod mpp_bench;
pub mod mpp_dispatch;
pub mod mpp_quantized;
pub mod turboquant;

#[cfg(has_metal4)]
pub mod mpp_fused_cross_entropy;
#[cfg(has_metal4)]
pub mod mpp_fused_moe;
#[cfg(has_metal4)]
pub mod mpp_fused_training;
#[cfg(has_metal4)]
pub mod mpp_grouped_gemm;

// Re-export main types
pub use dw_gemm::{DwGemm, GPU_DW_MIN_DIM, ScratchPool};
pub use fp8_training::{
    Fp8DynamicScale, Fp8Format, Fp8GemmOutput, Fp8QuantOutput, Fp8TrainingConfig, Fp8TrainingKernel,
};
pub use fused_cross_entropy::{
    FusedCrossEntropy,
    FusedCrossEntropyConfig,
    FusedCrossEntropyOutput,
    // Chunked vocabulary optimization: fused linear + cross-entropy
    FusedLinearCrossEntropy,
    FusedLinearCrossEntropyConfig,
    FusedLinearCrossEntropyOutput,
};
pub use fused_merge::{
    FusedMergeMetal, MergeConfig, TensorInfo, build_merge_config, build_tensor_info,
};
pub use fused_moe::{ExpertBits, ExpertWeightBuffers, FusedMoeExpert, FusedMoeExpertConfig};
pub use fused_training::{
    AdamWConfig, AdamWScalars, BatchedCommandBuffer, FusedAdamW, FusedCrossEntropyTraining,
    FusedGradientClipping, ParamInfo,
};
pub use moe::{MoeConfig, MoeGemmOutput, MoeKernel, MoeRouting};
pub use mpp_quantized::{MppQuantizedGemm, MppQuantizedGemmConfig};
pub use turboquant::TurboQuantTransform;
