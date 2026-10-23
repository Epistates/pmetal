//! Mixed-precision TurboQuant: a wider code for the channels that carry the
//! most energy and a narrower one for the rest (the outlier-aware schemes of
//! arXiv:2504.19874, "2.5-bit" and "3.5-bit" caches).
//!
//! # Construction
//!
//! The channels of a vector are split into two groups and each group is a
//! full TurboQuant cache of its own, with its own rotation, codebook, sinks,
//! recent ring and key frame: [`QuantizedKvCache`] twice, the outlier channels
//! at `outlier_bits` and the others at `regular_bits`. Nothing about the
//! packed format or the kernels changes, and each group keeps the guarantees
//! of the uniform cache (a rotation makes its error the same in every
//! direction of the group, so the group's error is what its width says).
//!
//! # Which channels are outliers
//!
//! Chosen per KV head from the first tokens appended, by the spread of each
//! channel: its variance about the head's mean for keys that are centred
//! (the mean is taken off before quantizing, so it costs the code nothing),
//! its energy otherwise and for values. The channels are put in the order
//! `[outliers | rest]` before the split and back in the original order when
//! the cache is read, so the permutation is invisible outside this type.
//!
//! # Attention
//!
//! The groups are separate caches, so the packed-attention kernels (which
//! score a vector against one cache) do not apply: a decode step reads the
//! cache with [`MixedQuantizedKvCache::dequantize_keys`] and
//! [`dequantize_values`](MixedQuantizedKvCache::dequantize_values) and
//! attends densely. The cache is still compressed at rest, which is what the
//! mixed widths are for.

use crate::{RuntimeError, RuntimeResult};
use std::sync::Arc;

use crate::InlineArray;
use crate::compat::Dtype;

use super::cache::QuantizedKvCache;
use super::config::{TurboQuantConfig, TurboQuantTensorConfig};
use super::state::TurboQuantState;

/// Tokens the channel ranking needs to be a spread and not one token's
/// values.
const RANKING_MIN_TOKENS: usize = 16;

/// How one tensor's channels split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Split {
    dim: usize,
    outliers: usize,
    regular_bits: u8,
    outlier_bits: u8,
}

impl Split {
    /// The split of a `dim`-wide tensor. Both groups are a multiple of
    /// `multiple` wide (the kernels read keys four channels at a time) and at
    /// least two channels wide. A uniform tensor splits a quarter off at the
    /// same width: equivalent to leaving it whole, and what lets one
    /// tensor be mixed while the other is not.
    fn resolve(
        what: &str,
        dim: usize,
        config: TurboQuantTensorConfig,
        multiple: usize,
    ) -> RuntimeResult<Self> {
        let floor = multiple.max(2);
        if dim < 2 * floor {
            return Err(RuntimeError::Config(format!(
                "TurboQuant: a {dim}-wide {what} cannot be split into two groups of {floor} or more"
            )));
        }
        let (regular_bits, outlier_bits, wanted) = match config {
            TurboQuantTensorConfig::Uniform { bits } => (bits, bits, dim / 4),
            TurboQuantTensorConfig::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } => (regular_bits, outlier_bits, outlier_count),
        };
        for bits in [regular_bits, outlier_bits] {
            if !(1..=8).contains(&bits) {
                return Err(RuntimeError::Config(format!(
                    "TurboQuant {what} bits must be in 1..=8, got {bits}"
                )));
            }
        }
        let outliers = (wanted.div_ceil(multiple) * multiple).clamp(floor, dim - floor);
        let outliers = outliers / multiple * multiple;
        if (dim - outliers) % multiple != 0 || dim % multiple != 0 {
            return Err(RuntimeError::Config(format!(
                "TurboQuant: the {what} dimension {dim} must be a multiple of {multiple}"
            )));
        }
        Ok(Self {
            dim,
            outliers,
            regular_bits,
            outlier_bits,
        })
    }

    fn regular(&self) -> usize {
        self.dim - self.outliers
    }
}

/// Both tensors' splits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Plan {
    keys: Split,
    values: Split,
}

impl Plan {
    fn resolve(key_dim: usize, value_dim: usize, config: &TurboQuantConfig) -> RuntimeResult<Self> {
        if matches!(
            (config.keys, config.values),
            (
                TurboQuantTensorConfig::Uniform { .. },
                TurboQuantTensorConfig::Uniform { .. }
            )
        ) {
            return Err(RuntimeError::Config(
                "TurboQuant: a mixed-precision cache needs a mixed key or value configuration"
                    .to_string(),
            ));
        }
        Ok(Self {
            keys: Split::resolve("key", key_dim, config.keys, 4)?,
            values: Split::resolve("value", value_dim, config.values, 1)?,
        })
    }

    /// The configuration of the group of regular (`false`) or outlier
    /// (`true`) channels.
    fn group_config(&self, config: &TurboQuantConfig, outlier: bool) -> TurboQuantConfig {
        let bits = |split: &Split| {
            TurboQuantTensorConfig::uniform(if outlier {
                split.outlier_bits
            } else {
                split.regular_bits
            })
        };
        TurboQuantConfig {
            keys: bits(&self.keys),
            values: bits(&self.values),
            ..*config
        }
    }
}

/// The quantizers of a mixed cache: one [`TurboQuantState`] per channel
/// group. Built once for a model and shared by its layers.
#[derive(Debug, Clone)]
pub struct MixedTurboQuantState {
    plan: Plan,
    regular: Arc<TurboQuantState>,
    outlier: Arc<TurboQuantState>,
}

impl MixedTurboQuantState {
    /// The state for `key_dim`-wide keys and `value_dim`-wide values.
    pub fn new(key_dim: usize, value_dim: usize, config: TurboQuantConfig) -> RuntimeResult<Self> {
        let plan = Plan::resolve(key_dim, value_dim, &config)?;
        let group = |outlier: bool| -> RuntimeResult<Arc<TurboQuantState>> {
            let (k, v) = if outlier {
                (plan.keys.outliers, plan.values.outliers)
            } else {
                (plan.keys.regular(), plan.values.regular())
            };
            Ok(Arc::new(TurboQuantState::new(
                k,
                v,
                plan.group_config(&config, outlier),
            )?))
        };
        Ok(Self {
            plan,
            regular: group(false)?,
            outlier: group(true)?,
        })
    }
}

/// Where one tensor's channels go: for each KV head, the outliers first.
#[derive(Debug, Clone)]
struct Channels {
    /// `[1, H, 1, D]` int32: `to_split[..., j]` is the channel placed at `j`.
    to_split: InlineArray,
    /// `[1, H, 1, D]` int32: where channel `i` was placed.
    to_original: InlineArray,
}

impl Channels {
    /// Rank the channels of `x` `[B, H, S, D]` by their spread and make the
    /// `outliers` widest of each head the first ones.
    fn rank(x: &InlineArray, outliers: usize, centred: bool) -> Self {
        let (b, h, s, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3));
        let tokens = (b * s) as usize;
        let x = x
            .as_dtype(Dtype::Float32.as_i32())
            .transpose_axes(&[1, 0, 2, 3])
            .reshape(&[h, b * s, d]);
        let energy = x.square().mean_axis(1, false);
        let spread = if centred && tokens >= RANKING_MIN_TOKENS {
            let mean = x.mean_axis(1, false);
            energy.subtract(&mean.multiply(&mean))
        } else {
            energy
        };
        let mut spread = spread;
        let spread = spread
            .to_f32_vec((h * d) as usize)
            .expect("channel spreads are readable");
        let (h, d) = (h as usize, d as usize);

        let mut to_split = Vec::with_capacity(h * d);
        let mut to_original = vec![0i32; h * d];
        for head in 0..h {
            let row = &spread[head * d..(head + 1) * d];
            let mut order: Vec<usize> = (0..d).collect();
            // Widest first; ties (and NaNs) by index, so the choice is a
            // function of the data alone.
            order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            let mut chosen = order[..outliers].to_vec();
            let mut rest = order[outliers..].to_vec();
            chosen.sort_unstable();
            rest.sort_unstable();
            for (placed, channel) in chosen.into_iter().chain(rest).enumerate() {
                to_split.push(channel as i32);
                to_original[head * d + channel] = placed as i32;
            }
        }
        let shape = [1, h as i32, 1, d as i32];
        Self {
            to_split: InlineArray::from_i32_slice_shaped(&to_split, &shape),
            to_original: InlineArray::from_i32_slice_shaped(&to_original, &shape),
        }
    }

    fn gather(index: &InlineArray, x: &InlineArray) -> InlineArray {
        let shape = [x.dim(0), x.dim(1), x.dim(2), x.dim(3)];
        x.take_along_axis(&index.broadcast_to(&shape), 3)
    }

    /// `x` with its channels in the order `[outliers | rest]`.
    fn split_order(&self, x: &InlineArray) -> InlineArray {
        Self::gather(&self.to_split, x)
    }

    /// The inverse of [`Self::split_order`].
    fn original_order(&self, x: &InlineArray) -> InlineArray {
        Self::gather(&self.to_original, x)
    }
}

/// Mixed-precision compressed KV cache of one attention layer; see the
/// module docs.
#[derive(Debug, Clone)]
pub struct MixedQuantizedKvCache {
    config: TurboQuantConfig,
    state: Option<Arc<MixedTurboQuantState>>,
    channels: Option<(Channels, Channels)>,
    regular: QuantizedKvCache,
    outlier: QuantizedKvCache,
}

impl MixedQuantizedKvCache {
    /// An empty cache that builds its quantizers on the first append.
    ///
    /// The configuration must have a mixed key or value tensor; a uniform one
    /// is a [`QuantizedKvCache`].
    pub fn new(config: TurboQuantConfig) -> Self {
        Self {
            config,
            state: None,
            channels: None,
            regular: QuantizedKvCache::new(config),
            outlier: QuantizedKvCache::new(config),
        }
    }

    /// An empty cache that uses a state built once for the model.
    pub fn with_state(config: TurboQuantConfig, state: Arc<MixedTurboQuantState>) -> Self {
        let mut cache = Self::new(config);
        cache.adopt(state);
        cache
    }

    fn adopt(&mut self, state: Arc<MixedTurboQuantState>) {
        let plan = state.plan;
        self.regular = QuantizedKvCache::with_state(
            plan.group_config(&self.config, false),
            Arc::clone(&state.regular),
        );
        self.outlier = QuantizedKvCache::with_state(
            plan.group_config(&self.config, true),
            Arc::clone(&state.outlier),
        );
        self.state = Some(state);
    }

    /// Tokens cached.
    pub fn len(&self) -> usize {
        self.regular.len()
    }

    /// True when nothing is cached.
    pub fn is_empty(&self) -> bool {
        self.regular.is_empty()
    }

    /// Tokens held uncompressed: the sinks and the recent ring.
    pub fn hot_len(&self) -> usize {
        self.regular.hot_len()
    }

    /// Tokens held quantized.
    pub fn cold_len(&self) -> usize {
        self.regular.cold_len()
    }

    /// Bytes of device memory the cached tokens occupy.
    pub fn nbytes(&self) -> usize {
        self.regular.nbytes() + self.outlier.nbytes()
    }

    /// Dtype of the uncompressed tokens, once something has been appended.
    pub fn dtype(&self) -> Option<Dtype> {
        self.regular.dtype()
    }

    /// Forget every cached token, keeping the configuration and the state.
    pub fn reset(&mut self) {
        self.regular.reset();
        self.outlier.reset();
        self.channels = None;
    }

    /// An independent copy that shares storage with this cache until either
    /// side appends.
    pub fn fork(&self) -> Self {
        self.clone()
    }

    /// Cut the lazy graph behind the cached arrays by evaluating them.
    pub fn eval_and_detach_gpu_state(&mut self) {
        self.regular.eval_and_detach_gpu_state();
        self.outlier.eval_and_detach_gpu_state();
    }

    /// Take back the newest `n` tokens; see [`QuantizedKvCache::truncate`].
    pub fn truncate(&mut self, n: usize) -> usize {
        let n = self.regular.truncate(n);
        self.outlier.truncate(n);
        if self.is_empty() {
            self.channels = None;
        }
        n
    }

    /// Append `[B, H, S, D]` keys and `[B, H, S, Dv]` values.
    pub fn append(&mut self, keys: &InlineArray, values: &InlineArray) -> RuntimeResult<()> {
        if keys.ndim() != 4 || values.ndim() != 4 {
            return Err(RuntimeError::Config(format!(
                "TurboQuant: expected [B, H, S, D] keys and values, got {} and {} dimensions",
                keys.ndim(),
                values.ndim()
            )));
        }
        let (key_dim, value_dim) = (keys.dim(3) as usize, values.dim(3) as usize);
        if self.state.is_none() {
            let state = MixedTurboQuantState::new(key_dim, value_dim, self.config)?;
            self.adopt(Arc::new(state));
        }
        let plan = self.state.as_ref().expect("adopted above").plan;
        if (plan.keys.dim, plan.values.dim) != (key_dim, value_dim) {
            return Err(RuntimeError::Config(format!(
                "TurboQuant: the shared state splits {}-wide keys and {}-wide values, \
             the cache is being given {key_dim} and {value_dim}",
                plan.keys.dim, plan.values.dim
            )));
        }
        if keys.dim(2) == 0 {
            return Ok(());
        }
        if let Some((held, _)) = &self.channels
            && held.to_split.dim(1) != keys.dim(1)
        {
            return Err(RuntimeError::Config(format!(
                "TurboQuant: the cache holds {} KV heads, cannot append {:?} keys",
                held.to_split.dim(1),
                keys.shape()
            )));
        }
        if keys.dim(1) != values.dim(1)
            || keys.dim(0) != values.dim(0)
            || keys.dim(2) != values.dim(2)
        {
            return Err(RuntimeError::Config(format!(
                "TurboQuant: keys {:?} and values {:?} disagree on batch, heads or length",
                keys.shape(),
                values.shape()
            )));
        }
        if self.channels.is_none() {
            self.channels = Some((
                Channels::rank(keys, plan.keys.outliers, self.config.center_keys),
                Channels::rank(values, plan.values.outliers, false),
            ));
        }
        let (key_channels, value_channels) = self.channels.as_ref().expect("ranked above");
        let keys = key_channels.split_order(keys);
        let values = value_channels.split_order(values);
        let cut = |x: &InlineArray, outliers: usize| {
            let (b, h, s, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3));
            let at = outliers as i32;
            (
                x.slice(&[0, 0, 0, 0], &[b, h, s, at]),
                x.slice(&[0, 0, 0, at], &[b, h, s, d]),
            )
        };
        let (key_outliers, key_regular) = cut(&keys, plan.keys.outliers);
        let (value_outliers, value_regular) = cut(&values, plan.values.outliers);
        self.regular.append(&key_regular, &value_regular)?;
        self.outlier.append(&key_outliers, &value_outliers)?;
        Ok(())
    }

    /// Every cached key as `[B, H, T, D]`, in the dtype the keys were appended
    /// in: the sinks and recent tokens exact, the rest dequantized.
    pub fn dequantize_keys(&self) -> Option<InlineArray> {
        let (channels, _) = self.channels.as_ref()?;
        let joined = self
            .outlier
            .dequantize_keys()?
            .concatenate_2(&self.regular.dequantize_keys()?, 3);
        Some(channels.original_order(&joined))
    }

    /// Every cached value as `[B, H, T, Dv]`; see [`Self::dequantize_keys`].
    pub fn dequantize_values(&self) -> Option<InlineArray> {
        let (_, channels) = self.channels.as_ref()?;
        let joined = self
            .outlier
            .dequantize_values()?
            .concatenate_2(&self.regular.dequantize_values()?, 3);
        Some(channels.original_order(&joined))
    }
}
