//! Sampling strategies for token generation.
//!
//! This module provides high-performance sampling implementations:
//!
//! - **compiled_sampler**: filtered sampling as MLX ops on the GPU (recommended;
//!   not passed to `mx.compile`, see the module docs)
//! - **metal_sampler**: Fused Metal kernel for single-launch sampling

pub mod compiled_sampler;
pub mod diffusion;

#[cfg(target_os = "macos")]
pub mod metal_sampler;

pub use compiled_sampler::{CompiledSampler, SamplerState};
pub use diffusion::*;

#[cfg(target_os = "macos")]
pub use metal_sampler::MetalSampler;
