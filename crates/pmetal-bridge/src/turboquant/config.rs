//! TurboQuant configuration: pure data, no runtime state.

/// How one tensor (the keys, or the values) is quantized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboQuantTensorConfig {
    /// One bit width for every coordinate.
    Uniform {
        /// Bits per coordinate, 1 to 8.
        bits: u8,
    },
    /// A wider code for the coordinates that carry the most energy and a
    /// narrower one for the rest. Not supported by the cache yet: building a
    /// cache from it fails.
    Mixed {
        /// Bits per coordinate of the regular channels.
        regular_bits: u8,
        /// Bits per coordinate of the outlier channels (at least
        /// `regular_bits`).
        outlier_bits: u8,
        /// How many channels are outliers.
        outlier_count: usize,
    },
}

impl TurboQuantTensorConfig {
    /// One bit width for the whole vector.
    pub const fn uniform(bits: u8) -> Self {
        Self::Uniform { bits }
    }

    /// A mixed-precision split; see [`TurboQuantTensorConfig::Mixed`].
    pub const fn mixed(regular_bits: u8, outlier_bits: u8, outlier_count: usize) -> Self {
        Self::Mixed {
            regular_bits,
            outlier_bits,
            outlier_count,
        }
    }

    /// Outlier channels per vector (0 for a uniform tensor).
    pub fn outlier_count(self) -> usize {
        match self {
            Self::Uniform { .. } => 0,
            Self::Mixed { outlier_count, .. } => outlier_count,
        }
    }

    /// Average bits per coordinate over a vector of `total_dim` coordinates.
    pub fn effective_bits(self, total_dim: usize) -> f32 {
        match self {
            Self::Uniform { bits } => f32::from(bits),
            Self::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } => {
                let regular = total_dim - outlier_count;
                (regular * usize::from(regular_bits) + outlier_count * usize::from(outlier_bits))
                    as f32
                    / total_dim as f32
            }
        }
    }
}

/// Tokens kept uncompressed at the head of the sequence by default. The
/// first tokens of a sequence draw attention whatever the query is, so noise
/// in them is multiplied by the largest softmax weights.
pub const DEFAULT_SINK_TOKENS: usize = 4;

/// Default size of the recent-token window kept uncompressed.
pub const DEFAULT_RECENT_WINDOW: usize = 8192;

/// How many of the oldest tokens leave the recent ring together once it is
/// full: a batch big enough to quantize efficiently and to amortise growing
/// the cold store, small enough not to bloat a small window.
pub(super) fn eviction_chunk(window: usize) -> usize {
    window.clamp(16, 1024)
}

/// A cache's configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurboQuantConfig {
    /// How the keys are quantized.
    pub keys: TurboQuantTensorConfig,
    /// How the values are quantized.
    pub values: TurboQuantTensorConfig,
    /// The newest `recent_window` tokens stay uncompressed; older history is
    /// quantized, in batches of up to `eviction_chunk` tokens. `None`
    /// quantizes every token as it arrives.
    pub recent_window: Option<usize>,
    /// The first `sink_tokens` tokens of the sequence stay uncompressed.
    pub sink_tokens: usize,
}

impl TurboQuantConfig {
    /// Uniform keys and values at independent bit widths, with the default
    /// sinks and recent window.
    pub const fn uniform(key_bits: u8, value_bits: u8) -> Self {
        Self {
            keys: TurboQuantTensorConfig::uniform(key_bits),
            values: TurboQuantTensorConfig::uniform(value_bits),
            recent_window: Some(DEFAULT_RECENT_WINDOW),
            sink_tokens: DEFAULT_SINK_TOKENS,
        }
    }

    /// Mixed-precision keys and values; see [`TurboQuantTensorConfig::Mixed`].
    pub const fn mixed(
        key_regular_bits: u8,
        key_outlier_bits: u8,
        key_outlier_count: usize,
        value_regular_bits: u8,
        value_outlier_bits: u8,
        value_outlier_count: usize,
    ) -> Self {
        Self {
            keys: TurboQuantTensorConfig::mixed(
                key_regular_bits,
                key_outlier_bits,
                key_outlier_count,
            ),
            values: TurboQuantTensorConfig::mixed(
                value_regular_bits,
                value_outlier_bits,
                value_outlier_count,
            ),
            recent_window: Some(DEFAULT_RECENT_WINDOW),
            sink_tokens: DEFAULT_SINK_TOKENS,
        }
    }

    /// Whether a cache can be built from this configuration.
    pub fn validate(&self) -> Result<(), String> {
        for (label, tensor) in [("keys", self.keys), ("values", self.values)] {
            match tensor {
                TurboQuantTensorConfig::Uniform { bits } if (1..=8).contains(&bits) => {}
                TurboQuantTensorConfig::Uniform { bits } => {
                    return Err(format!(
                        "TurboQuant {label} bits must be in 1..=8, got {bits}"
                    ));
                }
                TurboQuantTensorConfig::Mixed { .. } => {
                    return Err(format!(
                        "TurboQuant {label}: mixed-precision tensors are not supported yet"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Replace the recent window; `None` quantizes every token on arrival.
    pub const fn with_recent_window(mut self, window: Option<usize>) -> Self {
        self.recent_window = window;
        self
    }

    /// Replace the number of uncompressed sink tokens.
    pub const fn with_sink_tokens(mut self, tokens: usize) -> Self {
        self.sink_tokens = tokens;
        self
    }

    /// Outlier-aware 2.5-bit preset (a quarter of the channels at 4 bits, the
    /// rest at 2).
    pub fn preset_q2_5(total_dim: usize) -> Self {
        let outliers = recommended_outlier_count(total_dim);
        Self::mixed(2, 4, outliers, 2, 4, outliers)
    }

    /// Outlier-aware 3.5-bit preset (a quarter of the channels at 5 bits, the
    /// rest at 3).
    pub fn preset_q3_5(total_dim: usize) -> Self {
        let outliers = recommended_outlier_count(total_dim);
        Self::mixed(3, 5, outliers, 3, 5, outliers)
    }
}

fn recommended_outlier_count(total_dim: usize) -> usize {
    if total_dim <= 1 {
        0
    } else {
        total_dim.div_ceil(4).min(total_dim - 1)
    }
}
