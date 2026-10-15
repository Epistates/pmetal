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

/// The quantizers a model's layers share, built for one head geometry.
#[derive(Debug, Clone)]
pub(super) struct TurboShared {
    /// The key and value head dimensions the quantizers were built for.
    dims: (usize, usize),
    state: SharedState,
}

#[derive(Debug, Clone)]
enum SharedState {
    Uniform(Arc<TurboQuantState>),
    Mixed(Arc<MixedTurboQuantState>),
    /// The configuration cannot be built; every append says why.
    Refused,
}

impl TurboShared {
    pub(super) fn new(key_dim: usize, value_dim: usize, config: TurboQuantConfig) -> Self {
        let state = if config.is_mixed() {
            MixedTurboQuantState::new(key_dim, value_dim, config)
                .map_or(SharedState::Refused, |s| SharedState::Mixed(Arc::new(s)))
        } else {
            TurboQuantState::new(key_dim, value_dim, config)
                .map_or(SharedState::Refused, |s| SharedState::Uniform(Arc::new(s)))
        };
        Self {
            dims: (key_dim, value_dim),
            state,
        }
    }
}

/// The cache behind a layer.
#[derive(Debug, Clone)]
enum Cache {
    Uniform(Box<QuantizedKvCache>),
    Mixed(Box<MixedQuantizedKvCache>),
}

impl Cache {
    /// A cache that builds its own quantizers on the first append.
    fn unshared(config: TurboQuantConfig) -> Self {
        if config.is_mixed() {
            Self::Mixed(Box::new(MixedQuantizedKvCache::new(config)))
        } else {
            Self::Uniform(Box::new(QuantizedKvCache::new(config)))
        }
    }
}

/// One layer's compressed keys and values.
#[derive(Debug, Clone)]
pub(super) struct TurboLayer {
    config: TurboQuantConfig,
    /// The head dimensions the shared quantizers fit, `None` once the layer
    /// has built its own.
    shared_dims: Option<(usize, usize)>,
    cache: Cache,
}

fn failure(err: String) -> Exception {
    Exception::custom(err)
}

impl TurboLayer {
    pub(super) fn new(config: TurboQuantConfig, shared: &TurboShared) -> Self {
        let cache =
            match &shared.state {
                SharedState::Uniform(state) => Cache::Uniform(Box::new(
                    QuantizedKvCache::with_state(config, Arc::clone(state)),
                )),
                SharedState::Mixed(state) => Cache::Mixed(Box::new(
                    MixedQuantizedKvCache::with_state(config, Arc::clone(state)),
                )),
                // Built on the first append, which reports what is wrong.
                SharedState::Refused => Cache::unshared(config),
            };
        Self {
            config,
            shared_dims: Some(shared.dims),
            cache,
        }
    }

    pub(super) fn len(&self) -> usize {
        match &self.cache {
            Cache::Uniform(c) => c.len(),
            Cache::Mixed(c) => c.len(),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Tokens held quantized.
    pub(super) fn cold_len(&self) -> usize {
        match &self.cache {
            Cache::Uniform(c) => c.cold_len(),
            Cache::Mixed(c) => c.cold_len(),
        }
    }

    pub(super) fn nbytes(&self) -> usize {
        match &self.cache {
            Cache::Uniform(c) => c.nbytes(),
            Cache::Mixed(c) => c.nbytes(),
        }
    }

    pub(super) fn reset(&mut self) {
        match &mut self.cache {
            Cache::Uniform(c) => c.reset(),
            Cache::Mixed(c) => c.reset(),
        }
    }

    pub(super) fn truncate(&mut self, n: usize) -> usize {
        match &mut self.cache {
            Cache::Uniform(c) => c.truncate(n),
            Cache::Mixed(c) => c.truncate(n),
        }
    }

    /// A layer whose head geometry differs from the one the model's shared
    /// quantizers were built for (a model with global and local attention
    /// layers of different head dimensions) builds quantizers of its own.
    fn fit_geometry(&mut self, keys: &Array, values: &Array) {
        let Some(dims) = self.shared_dims else {
            return;
        };
        if keys.ndim() != 4 || values.ndim() != 4 {
            return;
        }
        let given = (keys.dim(3) as usize, values.dim(3) as usize);
        if self.is_empty() && given != dims {
            self.cache = Cache::unshared(self.config);
            self.shared_dims = None;
        }
    }

    fn append(&mut self, keys: &Array, values: &Array) -> Result<(), Exception> {
        self.fit_geometry(keys, values);
        match &mut self.cache {
            Cache::Uniform(c) => c.append(keys, values),
            Cache::Mixed(c) => c.append(keys, values),
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
        let (k, v) = match &self.cache {
            Cache::Uniform(c) => (c.dequantize_keys(), c.dequantize_values()),
            Cache::Mixed(c) => (c.dequantize_keys(), c.dequantize_values()),
        };
        k.zip(v)
            .ok_or_else(|| Exception::custom("TurboQuant: nothing cached after an append"))
    }

    /// Every key and value now cached, dequantized; `None` while empty.
    pub(super) fn current(&self) -> Option<(Array, Array)> {
        match &self.cache {
            Cache::Uniform(c) => c.dequantize_keys().zip(c.dequantize_values()),
            Cache::Mixed(c) => c.dequantize_keys().zip(c.dequantize_values()),
        }
    }

    /// Whether [`Self::attend`] serves this layer: the packed-attention
    /// kernels score a vector against one cache, so only a uniform layer has
    /// them.
    pub(super) fn attends_packed(&self) -> bool {
        matches!(self.cache, Cache::Uniform(_))
    }

    /// Store one new token and attend `queries` `[B, Hq, 1, D]` to the whole
    /// cache without dequantizing it. `window` limits the keys to the newest
    /// ones.
    pub(super) fn attend(
        &mut self,
        queries: &Array,
        keys: &Array,
        values: &Array,
        scale: f32,
        window: Option<usize>,
    ) -> Result<Array, Exception> {
        self.fit_geometry(keys, values);
        let Cache::Uniform(cache) = &mut self.cache else {
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
