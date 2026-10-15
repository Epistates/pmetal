//! Model-side operations, KV caches and decoding state over the PMetal
//! runtime.
//!
//! `pmetal-runtime` owns the arrays, the ops and the fused kernels; this crate
//! is what the model forwards build on top of them:
//!
//! - [`kernels`]: attention (masks, sinks, logit softcapping), the gated delta
//!   rule with a differentiable chunkwise form, the cut cross-entropy loss,
//!   rotary embeddings at explicit positions, the MoE combine, and the
//!   training-mode flag
//! - [`kv_cache`] and [`prefix_cache`]: KV caches (dense, rotating,
//!   quantized, TurboQuant, recurrent state) and prefix reuse
//! - [`speculative`]: the state a target model records for speculative
//!   decoding
//! - [`memory`]: allocator and system memory figures
//! - [`bridge`]: array to Metal buffer, for the kernels in `pmetal-metal`

#![warn(missing_docs)]

pub mod bridge;
pub mod kernels;
pub mod kv_cache;
pub mod memory;
pub mod moe;
pub mod prefix_cache;
pub mod speculative;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

#[cfg(test)]
mod f64_reference;

pub use bridge::MlxMetalBridge;
pub use kv_cache::*;
pub use moe::*;
pub use prefix_cache::*;
pub use speculative::*;

// The runtime's array types, for the callers that reach them here.
pub use pmetal_runtime::compat::builder::Builder;
pub use pmetal_runtime::compat::{Array, Dtype, Exception, Module, ModuleParameters, Param};
/// Crate-level result type alias using [`Exception`] as the error.
pub type Result<T> = std::result::Result<T, Exception>;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::kernels::*;
    pub use crate::kv_cache::*;
    pub use crate::memory::*;
    pub use crate::moe::*;
    pub use crate::prefix_cache::*;
    pub use crate::speculative::*;
    pub use pmetal_runtime::compat::{Array, Dtype, Exception, Module, ModuleParameters, Param};
}

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
