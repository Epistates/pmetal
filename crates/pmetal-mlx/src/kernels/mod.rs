//! Optimized MLX kernels for LLM training.
//!
//! This module contains optimized implementations of common LLM operations:
//! - Fused attention (Metal-optimized SDPA with GQA/MQA support)
//! - Differentiable attention with Metal FlashAttention backward pass
//! - Fused LoRA forward/backward
//! - Rotary position embeddings (RoPE)
//! - RMS layer normalization (including novel fused RMSNorm+LoRA)
//! - SwiGLU/GEGLU activations (including Metal-optimized fused MLP)
//! - Cross-entropy loss computation
//! - Cut Cross Entropy for 13x longer context
//! - Metal-accelerated fused linear + cross-entropy (key unsloth optimization)

pub mod cross_entropy;
pub mod cut_cross_entropy;
pub mod differentiable_attention;
pub mod fast_lora;
pub mod fused_attention;
pub mod metal_cross_entropy;
pub mod metal_norm_lora;
pub mod metal_swiglu;
pub mod rms_norm;
pub mod rope;
pub mod swiglu;
pub mod training_attention;
pub mod utils;

pub use cross_entropy::*;
pub use cut_cross_entropy::*;
pub use differentiable_attention::*;
pub use fast_lora::*;
pub use fused_attention::*;
pub use metal_cross_entropy::*;
pub use metal_norm_lora::*;
pub use metal_swiglu::*;
pub use rms_norm::*;
pub use rope::*;
pub use swiglu::*;
pub use training_attention::*;
pub use utils::*;
