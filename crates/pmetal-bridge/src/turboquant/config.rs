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
}

impl TurboQuantConfig {
    /// Uniform K/V configuration with independent key/value bit-widths.
    pub const fn uniform(key_bits: u8, value_bits: u8) -> Self {
        Self {
            keys: TurboQuantTensorConfig::uniform(key_bits),
            values: TurboQuantTensorConfig::uniform(value_bits),
            recent_window: Some(DEFAULT_RECENT_WINDOW),
            qjl: TurboQuantQjlMode::Standard,
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
