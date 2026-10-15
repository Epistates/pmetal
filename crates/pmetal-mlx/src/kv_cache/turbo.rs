//! One layer of the TurboQuant KV cache: the runtime's compressed caches
//! behind the shape the shared path's [`KVCache`](super::KVCache) needs.
//!
//! The quantizer, the packed format, the key frame, the sinks and recent ring
//! and the attention kernels are `pmetal_runtime::turboquant`'s, the same
//! ones the native engines run. A configuration with mixed-precision tensors
//! (the 2.5- and 3.5-bit presets) is held by a
//! [`MixedQuantizedKvCache`]; every other one by a [`QuantizedKvCache`].

use std::sync::Arc;

use pmetal_runtime::compat::{Array, Exception};
use pmetal_runtime::turboquant::{
    MixedQuantizedKvCache, MixedTurboQuantState, QuantizedKvCache, TurboQuantConfig,
    TurboQuantState,
};

/// The quantizers a model's layers share.
#[derive(Debug, Clone)]
pub(super) enum TurboShared {
    Uniform(Arc<TurboQuantState>),
    Mixed(Arc<MixedTurboQuantState>),
    /// The configuration cannot be built; every append says why.
    Refused,
}

impl TurboShared {
    pub(super) fn new(key_dim: usize, value_dim: usize, config: TurboQuantConfig) -> Self {
        if config.is_mixed() {
            MixedTurboQuantState::new(key_dim, value_dim, config)
                .map_or(Self::Refused, |s| Self::Mixed(Arc::new(s)))
        } else {
            TurboQuantState::new(key_dim, value_dim, config)
                .map_or(Self::Refused, |s| Self::Uniform(Arc::new(s)))
        }
    }
}

/// One layer's compressed keys and values.
#[derive(Debug, Clone)]
pub(super) enum TurboLayer {
    Uniform(Box<QuantizedKvCache>),
    Mixed(Box<MixedQuantizedKvCache>),
}

fn failure(err: String) -> Exception {
    Exception::custom(err)
}

impl TurboLayer {
    pub(super) fn new(config: TurboQuantConfig, shared: &TurboShared) -> Self {
        match shared {
            TurboShared::Uniform(state) => Self::Uniform(Box::new(QuantizedKvCache::with_state(
                config,
                Arc::clone(state),
            ))),
            TurboShared::Mixed(state) => Self::Mixed(Box::new(MixedQuantizedKvCache::with_state(
                config,
                Arc::clone(state),
            ))),
            // Built on the first append, which reports what is wrong.
            TurboShared::Refused if config.is_mixed() => {
                Self::Mixed(Box::new(MixedQuantizedKvCache::new(config)))
            }
            TurboShared::Refused => Self::Uniform(Box::new(QuantizedKvCache::new(config))),
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Uniform(c) => c.len(),
            Self::Mixed(c) => c.len(),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Tokens held quantized.
    pub(super) fn cold_len(&self) -> usize {
        match self {
            Self::Uniform(c) => c.cold_len(),
            Self::Mixed(c) => c.cold_len(),
        }
    }

    pub(super) fn nbytes(&self) -> usize {
        match self {
            Self::Uniform(c) => c.nbytes(),
            Self::Mixed(c) => c.nbytes(),
        }
    }

    pub(super) fn reset(&mut self) {
        match self {
            Self::Uniform(c) => c.reset(),
            Self::Mixed(c) => c.reset(),
        }
    }

    pub(super) fn truncate(&mut self, n: usize) -> usize {
        match self {
            Self::Uniform(c) => c.truncate(n),
            Self::Mixed(c) => c.truncate(n),
        }
    }

    fn append(&mut self, keys: &Array, values: &Array) -> Result<(), Exception> {
        match self {
            Self::Uniform(c) => c.append(keys, values),
            Self::Mixed(c) => c.append(keys, values),
        }
        .map_err(failure)
    }

    /// Store the new tokens and return every key and value dequantized: the
    /// sinks and the recent tokens exact, the history within the code's error.
    pub(super) fn update(
        &mut self,
        keys: &Array,
        values: &Array,
    ) -> Result<(Array, Array), Exception> {
        self.append(keys, values)?;
        let (k, v) = match self {
            Self::Uniform(c) => (c.dequantize_keys(), c.dequantize_values()),
            Self::Mixed(c) => (c.dequantize_keys(), c.dequantize_values()),
        };
        k.zip(v)
            .ok_or_else(|| Exception::custom("TurboQuant: nothing cached after an append"))
    }

    /// Whether [`Self::attend`] serves this layer: the packed-attention
    /// kernels score a vector against one cache, so only a uniform layer has
    /// them.
    pub(super) fn attends_packed(&self) -> bool {
        matches!(self, Self::Uniform(_))
    }

    /// Store one new token and attend `queries` `[B, Hq, 1, D]` to the whole
    /// cache without dequantizing it. `first_token` is the absolute position
    /// of the oldest token the query may see.
    pub(super) fn attend(
        &mut self,
        queries: &Array,
        keys: &Array,
        values: &Array,
        scale: f32,
        window: Option<usize>,
    ) -> Result<Array, Exception> {
        let Self::Uniform(cache) = self else {
            return Err(Exception::custom(
                "TurboQuant: a mixed-precision cache attends through dequantization",
            ));
        };
        cache.append(keys, values).map_err(failure)?;
        let first_token = window.map_or(0, |w| cache.len().saturating_sub(w));
        cache
            .compute_attention(queries, scale, first_token, None)
            .map_err(failure)
    }
}
