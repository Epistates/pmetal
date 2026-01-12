//! Sampling strategies for token generation.
//!
//! This module provides high-performance sampling implementations:
//!
//! - **compiled_sampler**: JIT-compiled sampling using MLX's compile transform (recommended)
//! - **metal_sampler**: Fused Metal kernel for single-launch sampling
//!
//! The compiled sampler is preferred as it matches mlx_lm's approach of using
//! `@partial(mx.compile, inputs=state, outputs=state)` to fuse operations into
//! optimized Metal kernels with proper state tracking.
//!
//! # State Tracking
//!
//! The key insight from Python's mlx-lm is that sampling functions must properly
//! track random state. This is done via:
//!
//! ```python
//! @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
//! def categorical_sampling(logits, temp):
//!     return mx.random.categorical(logits * (1 / temp))
//! ```
//!
//! In Rust, we achieve the same with `compile_with_state` and the `Updatable` trait:
//!
//! ```rust,ignore
//! let mut compiled = compile_with_state(categorical_with_state, None);
//! compiled(&mut sampler_state, &logits)?;
//! ```

pub mod compiled_sampler;

#[cfg(target_os = "macos")]
pub mod metal_sampler;

pub use compiled_sampler::{CompiledSampler, SamplerState};

#[cfg(target_os = "macos")]
pub use metal_sampler::MetalSampler;
