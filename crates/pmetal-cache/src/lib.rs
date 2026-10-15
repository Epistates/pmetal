//! KV caches, prefix reuse and speculative-decoding state for PMetal's
//! decoder models.
//!
//! `pmetal-runtime` owns the arrays, the ops and the fused kernels; this crate
//! holds the state a decoder carries from one step to the next:
//!
//! - [`kv_cache`]: KV caches (dense, rotating, quantized, TurboQuant,
//!   recurrent state) and the batched view of several sequences' caches
//! - [`prefix_cache`]: reuse of a shared prompt prefix across requests
//! - [`speculative`]: the state a target model records for speculative
//!   decoding

#![warn(missing_docs)]

pub mod kv_cache;
pub mod prefix_cache;
pub mod speculative;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

pub use kv_cache::*;
pub use prefix_cache::*;
pub use speculative::*;

/// Compiles the code blocks in `README.md` as doctests, so the crate's front
/// page cannot drift away from its API. `cfg(doctest)` keeps the item out of
/// the rendered docs and out of every normal build.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
