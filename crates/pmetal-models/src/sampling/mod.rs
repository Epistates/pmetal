//! Sampling strategies for token generation.
//!
//! Filtered sampling as MLX ops is [`crate::generation::Sampler`]; this
//! module holds the others:
//!
//! - **metal_sampler**: Fused Metal kernel for single-launch sampling
//! - **diffusion**: samplers for diffusion language models

pub mod diffusion;

#[cfg(target_os = "macos")]
pub mod metal_sampler;

pub use diffusion::*;

#[cfg(target_os = "macos")]
pub use metal_sampler::MetalSampler;
