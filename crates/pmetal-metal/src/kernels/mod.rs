//! GPU compute kernels for ML operations.
//!
//! Dense attention, RoPE, SwiGLU, LoRA and GEMM are not here: the tensor
//! runtime's own ops run them inside the lazy graph, with autodiff, faster
//! than any kernel this crate carried for them, and so do the loss, the MoE
//! expert dispatch, the quantized matmuls, merging and the KV-cache transforms.
//! What remains is the fused AdamW update, the single-expert MoE MLP the
//! offloaded Qwen3-Next forward runs straight out of a read buffer, and the
//! weight-gradient GEMM the ANE trainer batches into its own command buffer.

pub mod dw_gemm;
pub mod fused_moe;
pub mod fused_training;

// Re-export main types
pub use dw_gemm::{DwGemm, GPU_DW_MIN_DIM, ScratchPool};
pub use fused_moe::{ExpertBits, ExpertWeightBuffers, FusedMoeExpert, FusedMoeExpertConfig};
pub use fused_training::{AdamWConfig, AdamWScalars, BatchedCommandBuffer, FusedAdamW, ParamInfo};
