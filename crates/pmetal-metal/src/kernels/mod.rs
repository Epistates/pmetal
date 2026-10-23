//! GPU compute kernels for ML operations.
//!
//! Dense attention, RoPE, SwiGLU, LoRA and GEMM are not here: the tensor
//! runtime's own ops run them inside the lazy graph, with autodiff, faster
//! than any kernel this crate carried for them, and so do the loss, the MoE
//! expert dispatch (SSD-offloaded experts included), the quantized matmuls,
//! merging and the KV-cache transforms, and so does the AdamW update (a fused
//! kernel in `pmetal-runtime`). What remains is the weight-gradient GEMM the
//! ANE trainer batches into its own command buffer.

pub mod batched_command_buffer;
pub mod dw_gemm;

// Re-export main types
pub use batched_command_buffer::BatchedCommandBuffer;
pub use dw_gemm::{DwGemm, GPU_DW_MIN_DIM, ScratchPool};
