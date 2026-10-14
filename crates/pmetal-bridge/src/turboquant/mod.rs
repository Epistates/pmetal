//! TurboQuant KV cache: data-oblivious, rotation-based KV-cache quantization
//! (arXiv:2504.19874) with attention computed straight from the packed cache.
//!
//! # Algorithm
//!
//! Each key or value vector `x` is stored as its norm `|x|` and the indices of
//! the nearest Lloyd-Max centroids of the coordinates of `Π x / |x|`, where `Π`
//! is a fixed Haar-random orthogonal matrix. After the rotation every
//! coordinate has the same Beta density whatever `x` is, so one codebook per
//! `(head dim, bits)` serves every vector and the codebooks are not calibrated
//! on data; the squared error is within a constant (about 2.7) of the best any
//! `bits`-bit code can do. See [`Quantizer`].
//!
//! Attention scores come out as `scale_k <Π q, ĉ_k>`: the query is rotated once
//! per step and the cached rows stay packed and rotated. The values' weighted
//! sum is accumulated in the rotated basis and rotated back once. The scale
//! is the vector's norm times the least-squares gain of its reconstruction, an
//! addition to the paper that removes the per-vector shrinkage of a Lloyd-Max
//! codebook (see [`Quantizer`]).
//!
//! The paper's second stage, the one-bit QJL sketch of the residual that
//! makes the inner-product estimate unbiased, is not used: it is unbiased
//! only to feed a softmax the variance that outweighs its bias. Measured on
//! attention output against f64 attention it loses to spending the same bit
//! on the codebook, as independent implementations have also found.
//!
//! The quantizer's error is as large in every direction, which is what the
//! rotation buys and what real keys punish: they share a large component
//! across tokens and vary far more in some channels than in others, so a
//! uniform error swamps the logits' small spread. Each KV head's keys are
//! therefore put in a frame before they are quantized, calibrated on the
//! first tokens stored: the mean is taken off and each channel is divided by
//! its spread (`TurboQuantConfig::center_keys`, `scale_key_channels`). The
//! logits are unchanged: the query is multiplied by the same spreads, and the
//! mean adds one constant to a head's logits, which the softmax does not
//! see, so the dense tokens' scores and the sink logits just have it taken
//! off. On Qwen3-0.6B this is the difference between +273% and +0.3%
//! perplexity at 4 bits.
//!
//! # Cache layout
//!
//! [`QuantizedKvCache`] keeps the first few tokens (attention sinks) and the
//! newest ones uncompressed and the tokens between them packed. A decode step
//! attends to all of them in two kernel launches. See `cache.rs`.
//!
//! # Module map
//!
//! - `config`: [`TurboQuantConfig`] and friends.
//! - `math`: codebooks and rotation matrices.
//! - `quantizer`: [`Quantizer`], the bit-exact packed format.
//! - `kernels`: safe wrappers over `cpp/bridge_turboquant_*.cpp`.
//! - `store`: the packed cold store and the dense sink/recent store.
//! - `attention`: decode attention launch planning.
//! - `cache`: [`QuantizedKvCache`].
//! - `state`: [`TurboQuantState`], the quantizers shared by a model's layers.

use std::sync::Arc;

mod attention;
mod cache;
mod config;
mod kernels;
mod math;
mod quantizer;
mod state;
mod store;

pub use cache::QuantizedKvCache;
pub use config::{
    DEFAULT_RECENT_WINDOW, DEFAULT_SINK_TOKENS, TurboQuantConfig, TurboQuantTensorConfig,
};
pub use math::{beta_codebook, generate_rademacher_signs, signed_fwht_forward};
pub(crate) use math::{generate_random_orthogonal, generate_random_projection};
pub use quantizer::{Encoded, Quantizer, words_per_row};
pub use state::TurboQuantState;

#[cfg(test)]
mod attention_tests;
#[cfg(test)]
mod bench_tests;
#[cfg(test)]
mod cache_tests;
#[cfg(test)]
mod codebook_tests;
#[cfg(test)]
mod kernel_tests;
#[cfg(test)]
mod quality_tests;
#[cfg(test)]
mod reference_tests;

/// Build the shared [`TurboQuantState`] for `key_dim`-wide keys and
/// `value_dim`-wide values. Costs a rotation and a codebook per distinct
/// `(dim, bits)`: call once when the model loads and share the `Arc`.
///
/// # Panics
///
/// On a configuration the cache does not support (see
/// [`TurboQuantConfig::validate`]).
pub fn build_state(
    key_dim: usize,
    value_dim: usize,
    config: TurboQuantConfig,
) -> Arc<TurboQuantState> {
    match TurboQuantState::new(key_dim, value_dim, config) {
        Ok(state) => Arc::new(state),
        Err(err) => panic!("{err}"),
    }
}

/// A cache that builds its state on the first append.
pub fn new_cache(config: TurboQuantConfig) -> QuantizedKvCache {
    QuantizedKvCache::new(config)
}

/// A cache that uses a state built by [`build_state`].
pub fn new_cache_with_state(
    config: TurboQuantConfig,
    state: Arc<TurboQuantState>,
) -> QuantizedKvCache {
    QuantizedKvCache::with_state(config, state)
}
