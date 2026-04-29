//! TurboQuant precision configuration types.
//!
//! Per-tensor and per-cache configuration structs. No runtime state, no
//! GPU dependencies — pure data containers that pin the bit-widths,
//! mixed/uniform partitioning, and recent-window policy for a cache instance.
//!
//! Public surface re-exported via `crate::turboquant`.

/// Per-tensor TurboQuant precision configuration.
///
/// `Uniform` applies one codebook to the entire vector.
/// `Mixed` partitions the vector into "regular" and "outlier" coordinates
/// and applies different bit-widths to each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboQuantTensorConfig {
    /// Single bit-width across all coordinates.
    Uniform {
        /// Total effective bits per coordinate.
        bits: u8,
    },
    /// Separate bit-widths for outlier vs regular coordinates.
    Mixed {
        /// Bit-width for non-outlier coordinates.
        regular_bits: u8,
        /// Bit-width for outlier coordinates (must be >= regular_bits).
        outlier_bits: u8,
        /// How many of the highest-magnitude coordinates are "outliers".
        outlier_count: usize,
    },
}

impl TurboQuantTensorConfig {
    /// Create a uniform configuration.
    pub const fn uniform(bits: u8) -> Self {
        Self::Uniform { bits }
    }

    /// Create a mixed configuration.
    pub const fn mixed(regular_bits: u8, outlier_bits: u8, outlier_count: usize) -> Self {
        Self::Mixed {
            regular_bits,
            outlier_bits,
            outlier_count,
        }
    }

    /// Number of outlier coordinates per row (0 for Uniform).
    pub fn outlier_count(self) -> usize {
        match self {
            Self::Uniform { .. } => 0,
            Self::Mixed { outlier_count, .. } => outlier_count,
        }
    }

    /// Number of regular (non-outlier) coordinates per row.
    pub fn regular_dim(self, total_dim: usize) -> usize {
        total_dim - self.outlier_count()
    }

    /// Average effective bits per coordinate.
    pub fn effective_bits(self, total_dim: usize) -> f32 {
        match self {
            Self::Uniform { bits } => bits as f32,
            Self::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } => {
                let regular_dim = total_dim - outlier_count;
                (regular_dim * usize::from(regular_bits)
                    + outlier_count * usize::from(outlier_bits)) as f32
                    / total_dim as f32
            }
        }
    }

    pub(super) fn assert_valid(self, total_dim: usize, label: &str) {
        match self {
            Self::Uniform { bits } => {
                assert!(
                    (1..=8).contains(&bits),
                    "TurboQuant {label} bits must be in 1..=8"
                );
            }
            Self::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } => {
                assert!(
                    (1..=8).contains(&regular_bits),
                    "TurboQuant {label} regular_bits must be in 1..=8"
                );
                assert!(
                    (1..=8).contains(&outlier_bits),
                    "TurboQuant {label} outlier_bits must be in 1..=8"
                );
                assert!(
                    outlier_count > 0 && outlier_count < total_dim,
                    "TurboQuant {label} outlier_count must be in 1..{total_dim}"
                );
                assert!(
                    outlier_bits >= regular_bits,
                    "TurboQuant {label} outlier_bits must be >= regular_bits"
                );
            }
        }
    }
}

/// Centroid-index storage layout — orthogonal to bit-width, QJL mode,
/// and outlier mode.
///
/// `Bitstream` (default): centroid indices are bit-packed into a
/// contiguous LSB-first byte stream. At N bits per index the storage is
/// `ceil(N · D / 8)` bytes per slot. This is the historical layout and
/// minimises memory.
///
/// `Fullbyte`: each centroid index occupies one full byte regardless of
/// bit-width. At 4-7 bits this wastes the high bits per index but lets
/// the score kernel read indices with a single byte load instead of
/// bitstream extraction; on Apple GPUs the tradeoff favours `Fullbyte`
/// once the inner loop is bandwidth-bound (typical at decode time).
///
/// At 8 bits the two modes produce identical storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TurboQuantPackMode {
    /// LSB-first bit-stream packing — `ceil(N · D / 8)` bytes per slot.
    #[default]
    Bitstream,
    /// One byte per centroid index regardless of bit-width — SIMD-friendly.
    Fullbyte,
}

/// Per-block outlier mode (Phase E "Variant G") — orthogonal to all other
/// TurboQuant precision knobs.
///
/// `None` (default): no per-block outlier handling. The full rotated
/// vector goes to the codebook, and reconstruction error is whatever the
/// codebook resolution + per-row `slot_scale` deliver.
///
/// `PerBlock { k }`: at encode time, find the K coordinates with the
/// largest |rotated| value per slot, store them as a fp16 value plus a
/// u8 channel index, then zero those K coords before codebook
/// quantisation. Decode/score paths add the outlier values back at the
/// recorded channels, overriding the codebook reconstruction at exactly
/// those K positions. This is *strictly additive* to Uniform/Mixed and
/// QJL: K=8 with 4-bit Variant F catches the ~1% heavy-tail-channel keys
/// that even the best codebook can't represent without burning bits
/// globally.
///
/// Distinct from `TurboQuantTensorConfig::Mixed` — `Mixed` adapts
/// per-channel statistics across the whole sequence, `PerBlock` adapts
/// to local slot statistics. Both can be enabled together; `PerBlock`
/// runs after the regular/outlier split so the per-row top-K search
/// excludes channels already in the per-channel outlier set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TurboQuantOutlierMode {
    /// No per-block outliers — historical behavior.
    #[default]
    None,
    /// Variant G: store the top-K |rotated| coords per slot as
    /// `(channel: u8, value: f16)` pairs, override codebook
    /// reconstruction at those channels at score time.
    PerBlock {
        /// Number of outlier coordinates kept per slot. Practical range
        /// 4..=16; the pack format dedicates 1 byte for the channel
        /// index, so K is bounded by 255 in any case (≪ realistic).
        k: u8,
    },
}

impl TurboQuantOutlierMode {
    /// Number of per-block outliers to extract (0 when mode is `None`).
    pub const fn k(self) -> u8 {
        match self {
            Self::None => 0,
            Self::PerBlock { k } => k,
        }
    }

    /// Whether per-block outliers are enabled.
    pub const fn is_enabled(self) -> bool {
        matches!(self, Self::PerBlock { .. })
    }
}

/// QJL residual mode — orthogonal to Uniform/Mixed.
///
/// Variant E (`Standard`): `key_bits - 1` go to the codebook, 1 bit per
/// dim is the QJL sign of the residual. Decode adds a √(π/2)/D scaled
/// J^T·sign correction term to the codebook reconstruction. This is the
/// historical TurboQuant default.
///
/// Variant F (`NoQjl`): all `key_bits` go to the codebook; QJL is dropped
/// entirely. The encoded cache uses ~12.5% less memory at 4b (no
/// `qjl_signs` or `residual_norms` allocations) and the codebook gets a
/// full extra bit of resolution. Per the audit (2026-04-27) the QJL
/// residual contributes ≈0 to attention scores on rotated, normalized
/// keys; the reference `quant.cpp` ablation reproduced this.
///
/// Real-model PPL ablation (Phase A → Phase C decision gate) is required
/// before flipping the default. Until then, `NoQjl` ships opt-in via
/// `TurboQuantConfig::with_qjl_mode(NoQjl)` or `TurboQuantConfig::no_qjl_*`
/// presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TurboQuantQjlMode {
    /// Variant E: codebook at `key_bits - 1`, 1 bit per dim QJL residual.
    #[default]
    Standard,
    /// Variant F: codebook at full `key_bits`, no QJL residual.
    NoQjl,
}

/// Default size of the recent-token fp16 window. See pmetal-mlx for the
/// rationale; bridge keeps the same constant so cross-crate config plumbing
/// agrees.
pub const DEFAULT_RECENT_WINDOW: usize = 8192;

/// Eviction granularity. When the hot ring exceeds `recent_window + this`,
/// the oldest `HOT_EVICTION_CHUNK` tokens are compressed into the cold store
/// in one batch instead of churning per-token. Matches the pmetal-mlx side
/// so behavior between the two paths stays uniform.
pub(super) const HOT_EVICTION_CHUNK: usize = 1024;

/// Phase H — progressive 3-tier cache configuration.
///
/// Layered on top of the existing 2-tier (hot fp16 ring + cold compressed
/// store), the warm tier sits between them: hot-evicted tokens flow into a
/// warm cold-store at higher quality (e.g. 4-bit + outliers) while older
/// slots migrate down to the lower-quality cold store (e.g. 2-bit +
/// outliers). The result is a quality-adaptive cache: recent context stays
/// fp16, mid-recency context retains high reconstruction fidelity, deep
/// history pays maximum compression for memory savings.
///
/// `window` is the warm-tier capacity; once the warm store holds more than
/// this many slots the oldest `WARM_EVICTION_CHUNK` are dequantised and
/// re-quantised into cold at `cold_keys`/`cold_values` precision (the
/// outer `TurboQuantConfig.keys`/`.values`).
///
/// `None` keeps the legacy 2-tier behavior (everything compresses straight
/// into the existing cold store).
///
/// Distinct from quant.cpp's `tq_progressive` which uses fixed Q4/Q3
/// uniform: the TurboQuant slot_scale + Variant G outliers + RHT compose at
/// every tier, so the per-age precision steps remain quality-superior to
/// the reference's 3-tier scheme at matched bits-per-element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurboQuantWarmTier {
    /// Warm-tier slot capacity. Older slots migrate to cold beyond this.
    pub window: usize,
    /// Warm-tier key precision (typically higher quality than cold keys).
    pub keys: TurboQuantTensorConfig,
    /// Warm-tier value precision.
    pub values: TurboQuantTensorConfig,
}

impl TurboQuantWarmTier {
    /// Symmetric warm preset: same bit-width for keys and values.
    pub const fn uniform(window: usize, key_bits: u8, value_bits: u8) -> Self {
        Self {
            window,
            keys: TurboQuantTensorConfig::uniform(key_bits),
            values: TurboQuantTensorConfig::uniform(value_bits),
        }
    }
}

/// Full K/V TurboQuant configuration — one config per tensor type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurboQuantConfig {
    /// Key-cache quantisation strategy.
    pub keys: TurboQuantTensorConfig,
    /// Value-cache quantisation strategy.
    pub values: TurboQuantTensorConfig,
    /// Recent-token fp16 window. The newest `recent_window` tokens stay
    /// uncompressed; older history goes through TurboQuant. `None` disables
    /// the hot path (compress every token immediately).
    pub recent_window: Option<usize>,
    /// QJL residual mode. `Standard` (Variant E) is the historical default;
    /// `NoQjl` (Variant F) drops the QJL term and reclaims its bit for the
    /// codebook. See [`TurboQuantQjlMode`] for the trade-offs.
    pub qjl: TurboQuantQjlMode,
    /// Cold-store length above which the 1-bit Hamming skip-list pre-filter
    /// kicks in. `None` (default) keeps the full-cold-store score path. When
    /// set, encode also packs `sign(rotated)` into a per-slot hash buffer
    /// and `append_and_compute_attention` uses XOR + popcount to pick the
    /// top-M candidate slots before running exact attention on those M only.
    /// Suggested value: ~32_768; anything below `recent_window + threshold`
    /// stays on the existing fast path.
    pub skiplist_threshold: Option<usize>,
    /// Per-block outlier handling (Variant G). `None` (default) leaves
    /// the codebook to absorb heavy-tail values; `PerBlock { k }` extracts
    /// the K largest-magnitude coords per slot and stores them as
    /// `(channel, value)` pairs that override codebook reconstruction.
    pub outliers: TurboQuantOutlierMode,
    /// Centroid-index storage layout. `Bitstream` (default) bit-packs
    /// indices for minimum memory; `Fullbyte` uses one byte per index
    /// for SIMD-friendly score-kernel reads at non-8-bit widths.
    pub pack_mode: TurboQuantPackMode,
    /// Phase H — warm tier between hot fp16 and cold compressed. `None`
    /// (default) keeps the 2-tier behaviour. When `Some`, hot evictions go
    /// to a warm store at the warm-tier precision, and warm slots migrate
    /// to the cold store (sized at `keys`/`values` precision) once the
    /// warm window overflows. See [`TurboQuantWarmTier`].
    pub warm_tier: Option<TurboQuantWarmTier>,
}

impl TurboQuantConfig {
    /// Uniform K/V configuration with independent key/value bit-widths.
    pub const fn uniform(key_bits: u8, value_bits: u8) -> Self {
        Self {
            keys: TurboQuantTensorConfig::uniform(key_bits),
            values: TurboQuantTensorConfig::uniform(value_bits),
            recent_window: Some(DEFAULT_RECENT_WINDOW),
            qjl: TurboQuantQjlMode::Standard,
            skiplist_threshold: None,
            outliers: TurboQuantOutlierMode::None,
            pack_mode: TurboQuantPackMode::Bitstream,
            warm_tier: None,
        }
    }

    /// Mixed K/V configuration.
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
            qjl: TurboQuantQjlMode::Standard,
            skiplist_threshold: None,
            outliers: TurboQuantOutlierMode::None,
            pack_mode: TurboQuantPackMode::Bitstream,
            warm_tier: None,
        }
    }

    /// Override the recent fp16 window. `None` disables the hot path entirely
    /// (compress every appended token immediately — the legacy behavior).
    pub const fn with_recent_window(mut self, window: Option<usize>) -> Self {
        self.recent_window = window;
        self
    }

    /// Override the QJL residual mode. See [`TurboQuantQjlMode`] for the
    /// quality / memory trade-off.
    pub const fn with_qjl_mode(mut self, qjl: TurboQuantQjlMode) -> Self {
        self.qjl = qjl;
        self
    }

    /// Enable or disable the 1-bit Hamming skip-list pre-filter. Pass
    /// `Some(threshold)` to engage at cold-store lengths >= threshold;
    /// `None` (default) keeps the full-cold-store score path.
    pub const fn with_skiplist_threshold(mut self, threshold: Option<usize>) -> Self {
        self.skiplist_threshold = threshold;
        self
    }

    /// Override per-block outlier handling (Variant G). See
    /// [`TurboQuantOutlierMode`].
    pub const fn with_outliers(mut self, mode: TurboQuantOutlierMode) -> Self {
        self.outliers = mode;
        self
    }

    /// Override centroid-index storage layout. See [`TurboQuantPackMode`].
    pub const fn with_pack_mode(mut self, mode: TurboQuantPackMode) -> Self {
        self.pack_mode = mode;
        self
    }

    /// Phase H: enable the warm tier between hot fp16 and cold compressed.
    /// See [`TurboQuantWarmTier`] for the eviction semantics.
    pub const fn with_warm_tier(mut self, warm: Option<TurboQuantWarmTier>) -> Self {
        self.warm_tier = warm;
        self
    }

    /// Variant F preset: 4-bit codebook with QJL dropped. Use this for
    /// real-model PPL ablation runs against the Standard 4-bit baseline.
    pub const fn no_qjl_4b() -> Self {
        Self::uniform(4, 4).with_qjl_mode(TurboQuantQjlMode::NoQjl)
    }

    /// Outlier-aware 2.5-bit preset (25% outliers at 4 bits, rest at 2 bits).
    pub fn preset_q2_5(total_dim: usize) -> Self {
        let outlier_count = recommended_outlier_count(total_dim);
        Self::mixed(2, 4, outlier_count, 2, 4, outlier_count)
    }

    /// Outlier-aware 3.5-bit preset (25% outliers at 5 bits, rest at 3 bits).
    pub fn preset_q3_5(total_dim: usize) -> Self {
        let outlier_count = recommended_outlier_count(total_dim);
        Self::mixed(3, 5, outlier_count, 3, 5, outlier_count)
    }
}

fn recommended_outlier_count(total_dim: usize) -> usize {
    if total_dim <= 1 {
        0
    } else {
        total_dim.div_ceil(4).min(total_dim - 1)
    }
}
