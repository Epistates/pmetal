//! Sampling strategies for token generation.
//!
//! Filtered sampling as MLX ops is [`crate::generation::Sampler`]; this
//! module holds the samplers for diffusion language models.

pub mod diffusion;

pub use diffusion::*;
