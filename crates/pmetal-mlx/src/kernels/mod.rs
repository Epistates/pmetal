//! Optimized MLX kernels for LLM training.
//!
//! This module contains optimized implementations of common LLM operations:
//! - Fused attention (Metal-optimized SDPA with GQA/MQA support)
//! - Training-mode flag shared by trainers and model forwards
//! - Fused LoRA forward/backward
//! - Rotary position embeddings (RoPE)
//! - RMS layer normalization
//! - Metal-optimized fused SwiGLU MLP
//! - Cut Cross Entropy (loss without materializing the full logits)
//! - Gated DeltaNet recurrence

pub mod cut_cross_entropy;
pub mod fast_lora;
pub mod fused_attention;
pub mod fused_moe;
pub mod gated_delta;
pub mod metal_swiglu;
mod persistent_cache;
pub mod quantized_matmul;
pub mod rms_norm;
pub mod rope;
pub mod training_context;
pub mod utils;

pub use cut_cross_entropy::*;
pub use fast_lora::*;
pub use fused_attention::*;
pub use fused_moe::*;
pub use gated_delta::*;
pub use metal_swiglu::*;
pub use quantized_matmul::*;
pub use rms_norm::*;
pub use rope::*;
pub use training_context::*;
pub use utils::*;
