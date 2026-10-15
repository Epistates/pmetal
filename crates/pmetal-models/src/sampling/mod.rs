//! Sampling strategies for token generation.
//!
//! Filtered sampling is the runtime's `DecodeSampler`; this module holds the
//! noise schedulers of diffusion models.

pub mod diffusion;

pub use diffusion::*;
