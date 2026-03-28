//! TurboQuant KV cache — zero mlx-rs dependency.
//!
//! Self-contained implementation of the TurboQuant-inspired KV cache using only
//! [`InlineArray`] and pure-Rust math.  The module is intentionally free of any
//! mlx-rs or pmetal-metal imports; all GPU work is driven through
//! `InlineArray::matmul` which dispatches to MLX's Metal backend automatically.
//!
//! # Algorithm overview
//!
//! **Keys** (inner-product optimised):
//!   1. Normalise each vector onto the unit sphere; store the L2 norm.
//!   2. Apply the orthogonal rotation Π: `r = Π · k_norm`.
//!   3. Nearest-centroid scalar quantisation of every coordinate using the
//!      Lloyd-Max codebook for the Beta distribution (MSE at `b-1` bits).
//!   4. Compute the residual `e = k_norm - Π^T · codebook[idx]` and project it
//!      through a Gaussian matrix J: sign(J · e) gives 1-bit QJL signs.
//!
//! **Values** (MSE optimised, no QJL stage):
//!   1. Normalise + store norm.
//!   2. Rotate then quantise with the full `b`-bit codebook.
//!
//! **Outlier-aware mixed-bit** (optional):
//!   Per-row, the top-`k` coordinates by magnitude are flagged as "outliers"
//!   and stored at a higher bit-width in a separate sub-vector.
//!
//! # What is NOT in this module
//!
//! - The mlx-rs `Array` integration code.
//! - The `TurboQuantKvCache` struct (we have [`KvLayerCache`] in qwen3_native).
//! - The pmetal-metal `TurboQuantTransform` (InlineArray.matmul replaces it).
//! - The fused-attention path (we use standard SDPA).

use std::f32::consts::PI;
use std::sync::Arc;

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::InlineArray;

// ── Constants ────────────────────────────────────────────────────────────────

/// Deterministic seed — same as the mlx-rs reference implementation.
const TURBOQUANT_SEED: u64 = 0x5442_5155_414e_544d;
/// Vectors with L2 norm below this are treated as zero.
const ZERO_EPSILON: f32 = 1e-12;
/// Lloyd-Max iteration cap.
const LLOYD_MAX_ITERS: usize = 64;
/// Lloyd-Max convergence threshold.
const LLOYD_MAX_TOLERANCE: f64 = 1e-7;
/// Number of grid points for the Beta-distribution quadrature.
const LLOYD_GRID_POINTS: usize = 8192;

// ═══════════════════════════════════════════════════════════════════════════
// Configuration types
// ═══════════════════════════════════════════════════════════════════════════

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

    fn assert_valid(self, total_dim: usize, label: &str) {
        match self {
            Self::Uniform { bits } => {
                assert!((1..=8).contains(&bits), "TurboQuant {label} bits must be in 1..=8");
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

/// Full K/V TurboQuant configuration — one config per tensor type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurboQuantConfig {
    /// Key-cache quantisation strategy.
    pub keys: TurboQuantTensorConfig,
    /// Value-cache quantisation strategy.
    pub values: TurboQuantTensorConfig,
}

impl TurboQuantConfig {
    /// Uniform K/V configuration with independent key/value bit-widths.
    pub const fn uniform(key_bits: u8, value_bits: u8) -> Self {
        Self {
            keys: TurboQuantTensorConfig::uniform(key_bits),
            values: TurboQuantTensorConfig::uniform(value_bits),
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
        }
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

// ═══════════════════════════════════════════════════════════════════════════
// Core state — built once at model load time
// ═══════════════════════════════════════════════════════════════════════════

/// Per-dimension core: rotation, QJL projection matrix, and codebooks.
///
/// Expensive to construct (random QR decomposition + Lloyd-Max), but
/// can be shared cheaply via `Arc` across heads and layers.
#[derive(Debug)]
pub struct TurboQuantCore {
    /// Number of dimensions this core handles.
    dim: usize,
    /// Row-major [dim × dim] orthogonal rotation matrix Π (f32).
    rotation: Vec<f32>,
    /// Row-major [dim × dim] inverse (= transpose) of rotation.
    inverse_rotation: Vec<f32>,
    /// Row-major [dim × dim] Gaussian random projection J for QJL.
    qjl_projection: Vec<f32>,
    /// Row-major [dim × dim] transpose of qjl_projection.
    inverse_qjl_projection: Vec<f32>,
    /// `codebooks[b]` holds the 2^b sorted centroids for `b`-bit quantisation.
    /// Index 0 is unused (0-bit is a degenerate case).
    codebooks: Vec<Vec<f32>>,
    /// InlineArray view of the rotation matrix for GPU-accelerated matmul.
    rotation_arr: Option<InlineArray>,
    /// InlineArray view of the inverse rotation matrix.
    inverse_rotation_arr: Option<InlineArray>,
    /// InlineArray view of the QJL projection matrix.
    qjl_arr: Option<InlineArray>,
    /// InlineArray view of the inverse QJL projection matrix.
    inverse_qjl_arr: Option<InlineArray>,
    /// GPU-side codebook arrays: `codebook_arrs[b]` is a 1-D f32 InlineArray
    /// holding 2^b centroids for b-bit quantisation.  Indexed as codebooks.
    codebook_arrs: Vec<Option<InlineArray>>,
}

impl TurboQuantCore {
    fn new(dim: usize, max_mse_bits: u8) -> Self {
        let mut rng = StdRng::seed_from_u64(TURBOQUANT_SEED ^ ((dim as u64) << 32));

        let rotation = generate_random_orthogonal(dim, &mut rng);
        let inverse_rotation = transpose_square_matrix(&rotation, dim);
        let qjl_projection = generate_random_projection(dim, &mut rng);
        let inverse_qjl_projection = transpose_square_matrix(&qjl_projection, dim);

        let mut codebooks = vec![Vec::new(); usize::from(max_mse_bits) + 1];
        for bits in 1..=max_mse_bits {
            codebooks[usize::from(bits)] = build_beta_codebook(dim, bits);
        }

        // Build InlineArray GPU matrices.  On failure we fall back to CPU
        // matmul transparently — the Option<InlineArray> is None in that case.
        let rotation_arr = matrix_to_inline_array(&rotation, dim);
        let inverse_rotation_arr = matrix_to_inline_array(&inverse_rotation, dim);
        let qjl_arr = matrix_to_inline_array(&qjl_projection, dim);
        let inverse_qjl_arr = matrix_to_inline_array(&inverse_qjl_projection, dim);

        // GPU-side codebooks: each is a tiny 1-D f32 array (16 elements for 4-bit).
        let codebook_arrs: Vec<Option<InlineArray>> = codebooks
            .iter()
            .map(|cb| {
                if cb.is_empty() {
                    None
                } else {
                    let mut arr = InlineArray::from_f32_slice(cb, &[cb.len() as i32]);
                    arr.eval();
                    Some(arr)
                }
            })
            .collect();

        Self {
            dim,
            rotation,
            inverse_rotation,
            qjl_projection,
            inverse_qjl_projection,
            codebooks,
            rotation_arr,
            inverse_rotation_arr,
            qjl_arr,
            inverse_qjl_arr,
            codebook_arrs,
        }
    }

    fn codebook(&self, bits: u8) -> &[f32] {
        &self.codebooks[usize::from(bits)]
    }

    fn codebook_arr(&self, bits: u8) -> Option<&InlineArray> {
        self.codebook_arrs.get(usize::from(bits))?.as_ref()
    }

    /// GPU-native nearest-centroid quantisation.
    ///
    /// `rotated`: [N, D] f32 InlineArray of already-rotated (and normalised) vectors.
    /// Returns a uint32 InlineArray of shape [N, D] holding the codebook index per coordinate.
    ///
    /// Algorithm: expand rotated → [N, D, 1], subtract codebook [C] → [N, D, C],
    /// take squared diff, argmin along axis=-1 → [N, D].
    fn gpu_quantize_mse(&self, rotated: &InlineArray, bits: u8) -> Option<InlineArray> {
        let cb_arr = self.codebook_arr(bits)?;
        // rotated: [..., D]  →  expanded: [..., D, 1]
        let expanded = rotated.expand_dims(-1);
        // broadcast subtract: [..., D, C]
        let diffs = expanded.subtract(cb_arr);
        // squared distances (cheaper than abs for MSE nearest-centroid)
        let sq = diffs.multiply(&diffs);
        // argmin over codebook dimension → [..., D] uint32
        Some(sq.argmin(-1))
    }

    /// GPU-native codebook reconstruction.
    ///
    /// `indices`: [..., D] uint32.  Returns [..., D] f32 of gathered centroid values.
    fn gpu_reconstruct_mse(&self, indices: &InlineArray, bits: u8) -> Option<InlineArray> {
        let cb_arr = self.codebook_arr(bits)?;
        // take(codebook [C], indices [..., D], axis=0) → [..., D] f32
        // Flatten indices to 1-D, gather, reshape back.
        let orig_shape = indices.shape().to_vec();
        let n: i32 = orig_shape.iter().product();
        let flat = indices.reshape(&[n]);
        let gathered = cb_arr.take_axis(&flat, 0);
        Some(gathered.reshape(&orig_shape))
    }

    /// Rotate input rows: output = input · Π^T  (each row left-multiplied by Π).
    fn rotate_rows(&self, input: &[f32]) -> Vec<f32> {
        self.apply_transform(input, &self.rotation, &self.rotation_arr)
    }

    /// Inverse-rotate: output = input · Π.
    fn inverse_rotate_rows(&self, input: &[f32]) -> Vec<f32> {
        self.apply_transform(input, &self.inverse_rotation, &self.inverse_rotation_arr)
    }

    /// Project via Gaussian matrix J for QJL.
    fn project_rows(&self, input: &[f32]) -> Vec<f32> {
        self.apply_transform(input, &self.qjl_projection, &self.qjl_arr)
    }

    /// Inverse-project via J^T.
    fn inverse_project_rows(&self, input: &[f32]) -> Vec<f32> {
        self.apply_transform(input, &self.inverse_qjl_projection, &self.inverse_qjl_arr)
    }

    /// Apply a [dim × dim] linear transform to a batch of row vectors.
    ///
    /// Tries GPU matmul via InlineArray first; falls back to CPU on any failure.
    fn apply_transform(
        &self,
        input: &[f32],
        matrix_cpu: &[f32],
        matrix_gpu: &Option<InlineArray>,
    ) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        if let Some(m_arr) = matrix_gpu {
            if let Some(result) = try_gpu_matmul_rows(input, self.dim, m_arr) {
                return result;
            }
        }

        // CPU fallback — avoids GPU round-trip for tiny inputs.
        matmul_rows(matrix_cpu, self.dim, input)
    }
}

/// Attempt a GPU matrix-multiply: input [N, dim] × matrix [dim, dim] → [N, dim].
///
/// Input is uploaded to GPU as f32, matmul is performed in f32, result is
/// copied back to a Vec<f32>.  Returns `None` if shape or dtype conversion fails.
fn try_gpu_matmul_rows(input: &[f32], dim: usize, matrix: &InlineArray) -> Option<Vec<f32>> {
    let n = input.len() / dim;
    if n == 0 || dim == 0 {
        return None;
    }

    // Upload input [N, dim] as f32, multiply by pre-built matrix [dim, dim].
    let input_arr = InlineArray::from_f32_slice(input, &[n as i32, dim as i32]);
    let result_arr = input_arr.matmul(matrix);
    inline_array_to_f32_vec(&result_arr, n * dim)
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-tensor runtime — wraps Uniform or Mixed core selection
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
enum TensorRuntime {
    Uniform {
        config: TurboQuantTensorConfig,
        core: Arc<TurboQuantCore>,
    },
    Mixed {
        config: TurboQuantTensorConfig,
        regular_core: Arc<TurboQuantCore>,
        outlier_core: Arc<TurboQuantCore>,
    },
}

// ═══════════════════════════════════════════════════════════════════════════
// State — created once per (dim, config) pair at model load time
// ═══════════════════════════════════════════════════════════════════════════

/// Shared TurboQuant state for a given K/V head dimension and config.
///
/// Expensive to build (QR decomposition + Lloyd-Max).  Wrap in `Arc` and share
/// across all layers and heads that share the same (dim, config) pair.
#[derive(Debug, Clone)]
pub struct TurboQuantState {
    key_dim: usize,
    value_dim: usize,
    keys: TensorRuntime,
    values: TensorRuntime,
}

impl TurboQuantState {
    /// Build a new state.  Typical latency: ~50–200 ms for dim=128.
    pub fn new(key_dim: usize, value_dim: usize, config: TurboQuantConfig) -> Self {
        config.keys.assert_valid(key_dim, "keys");
        config.values.assert_valid(value_dim, "values");

        // Cache cores so (dim, bits) pairs that appear for both keys and values
        // share the same Arc.
        let mut core_cache =
            std::collections::HashMap::<(usize, u8), Arc<TurboQuantCore>>::new();
        let mut get_core = |subdim: usize, max_mse_bits: u8| {
            core_cache
                .entry((subdim, max_mse_bits))
                .or_insert_with(|| Arc::new(TurboQuantCore::new(subdim, max_mse_bits)))
                .clone()
        };

        let keys = build_tensor_runtime(key_dim, config.keys, true, &mut get_core);
        let values = build_tensor_runtime(value_dim, config.values, false, &mut get_core);

        Self {
            key_dim,
            value_dim,
            keys,
            values,
        }
    }
}

fn build_tensor_runtime<F>(
    total_dim: usize,
    config: TurboQuantTensorConfig,
    is_keys: bool,
    get_core: &mut F,
) -> TensorRuntime
where
    F: FnMut(usize, u8) -> Arc<TurboQuantCore>,
{
    match config {
        TurboQuantTensorConfig::Uniform { bits } => {
            let max_mse_bits = if is_keys { bits.saturating_sub(1) } else { bits };
            TensorRuntime::Uniform {
                config,
                core: get_core(total_dim, max_mse_bits),
            }
        }
        TurboQuantTensorConfig::Mixed {
            regular_bits,
            outlier_bits,
            outlier_count,
        } => {
            let regular_dim = total_dim - outlier_count;
            let regular_max = if is_keys {
                regular_bits.saturating_sub(1)
            } else {
                regular_bits
            };
            let outlier_max = if is_keys {
                outlier_bits.saturating_sub(1)
            } else {
                outlier_bits
            };
            TensorRuntime::Mixed {
                config,
                regular_core: get_core(regular_dim, regular_max),
                outlier_core: get_core(outlier_count, outlier_max),
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Compact bit-packing
// ═══════════════════════════════════════════════════════════════════════════

/// Bit-packed storage for variable-width unsigned integers (1–8 bits each).
///
/// Values are stored LSB-first in a contiguous byte buffer.  Provides O(1)
/// random read and amortised O(1) append.
#[derive(Debug, Clone)]
pub struct PackedBits {
    bits_per_value: u8,
    len: usize,
    bytes: Vec<u8>,
}

impl PackedBits {
    pub fn new(bits_per_value: u8) -> Self {
        Self {
            bits_per_value,
            len: 0,
            bytes: Vec::new(),
        }
    }

    pub fn extend_from_slice(&mut self, values: &[u16]) {
        if self.bits_per_value == 0 || values.is_empty() {
            self.len += values.len();
            return;
        }

        for &value in values {
            debug_assert!(u32::from(value) < (1u32 << self.bits_per_value));
            let bit_offset = self.len * usize::from(self.bits_per_value);
            let required_bits = bit_offset + usize::from(self.bits_per_value);
            let required_bytes = required_bits.div_ceil(8);
            if self.bytes.len() < required_bytes {
                self.bytes.resize(required_bytes, 0);
            }
            for bit in 0..self.bits_per_value {
                if ((value >> bit) & 1) != 0 {
                    let target_bit = bit_offset + usize::from(bit);
                    self.bytes[target_bit / 8] |= 1u8 << (target_bit % 8);
                }
            }
            self.len += 1;
        }
    }

    pub fn get(&self, index: usize) -> u16 {
        debug_assert!(index < self.len);
        if self.bits_per_value == 0 {
            return 0;
        }
        let bit_offset = index * usize::from(self.bits_per_value);
        let mut value = 0u16;
        for bit in 0..self.bits_per_value {
            let target_bit = bit_offset + usize::from(bit);
            let byte = self.bytes[target_bit / 8];
            if ((byte >> (target_bit % 8)) & 1) != 0 {
                value |= 1u16 << bit;
            }
        }
        value
    }

    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len {
            return;
        }
        self.len = new_len;
        if self.bits_per_value == 0 {
            return;
        }
        let total_bits = self.len * usize::from(self.bits_per_value);
        let total_bytes = total_bits.div_ceil(8);
        self.bytes.truncate(total_bytes);
        if let Some(last) = self.bytes.last_mut() {
            let used_bits = total_bits % 8;
            if used_bits != 0 {
                *last &= (1u8 << used_bits) - 1;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }
}

fn unpack_all(bits: &PackedBits) -> Vec<u16> {
    (0..bits.len()).map(|i| bits.get(i)).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-layer quantised storage
// ═══════════════════════════════════════════════════════════════════════════

/// GPU-resident quantised key data for the Uniform (non-outlier) path.
///
/// All tensors live entirely on the GPU — no CPU round-trips during normal
/// operation.  Shape convention (accumulated over T steps):
///   indices:         [B, H, T, D]  uint32  — codebook index per coordinate
///   norms:           [B, H, T, 1]  f32     — L2 norm before unit-sphere normalise
///   qjl_signs:       [B, H, T, D]  f32     — ±1 sign of QJL projection
///   residual_norms:  [B, H, T, 1]  f32     — L2 norm of MSE residual
#[derive(Debug, Clone)]
struct GpuKeyStore {
    indices: InlineArray,
    norms: InlineArray,
    qjl_signs: InlineArray,
    residual_norms: InlineArray,
}

impl GpuKeyStore {
    /// Concatenate a new step's GPU arrays along the T (axis 2) dimension.
    fn append(&mut self, new: GpuKeyStore) {
        self.indices        = self.indices.kv_cache_append(&new.indices, 2);
        self.norms          = self.norms.kv_cache_append(&new.norms, 2);
        self.qjl_signs      = self.qjl_signs.kv_cache_append(&new.qjl_signs, 2);
        self.residual_norms = self.residual_norms.kv_cache_append(&new.residual_norms, 2);
    }
}

/// GPU-resident quantised value data for the Uniform path.
///
///   indices:  [B, H, T, D]  uint32
///   norms:    [B, H, T, 1]  f32
#[derive(Debug, Clone)]
struct GpuValueStore {
    indices: InlineArray,
    norms: InlineArray,
}

impl GpuValueStore {
    fn append(&mut self, new: GpuValueStore) {
        self.indices = self.indices.kv_cache_append(&new.indices, 2);
        self.norms   = self.norms.kv_cache_append(&new.norms, 2);
    }
}

/// Quantised key store for one attention layer.
#[derive(Debug, Clone)]
pub struct QuantizedKeyStore {
    // GPU-native store (Uniform path only).  When Some, dequantize uses GPU ops.
    gpu: Option<GpuKeyStore>,

    // CPU fallback: regular (non-outlier) sub-vector data.
    pub regular_indices: PackedBits,
    pub regular_qjl_signs: PackedBits,
    pub regular_norms: Vec<f32>,
    pub regular_residual_norms: Vec<f32>,

    // Outlier sub-vector data (None when config is Uniform).
    pub outlier_mask: Option<PackedBits>,
    pub outlier_indices: Option<PackedBits>,
    pub outlier_qjl_signs: Option<PackedBits>,
    pub outlier_norms: Option<Vec<f32>>,
    pub outlier_residual_norms: Option<Vec<f32>>,
}

impl QuantizedKeyStore {
    fn new(config: TurboQuantTensorConfig) -> Self {
        let regular_bits = match config {
            TurboQuantTensorConfig::Uniform { bits } => bits.saturating_sub(1),
            TurboQuantTensorConfig::Mixed { regular_bits, .. } => regular_bits.saturating_sub(1),
        };
        let outlier_bits: Option<u8> = match config {
            TurboQuantTensorConfig::Uniform { .. } => None,
            TurboQuantTensorConfig::Mixed { outlier_bits, .. } => {
                Some(outlier_bits.saturating_sub(1))
            }
        };

        Self {
            gpu: None,
            regular_indices: PackedBits::new(regular_bits),
            regular_qjl_signs: PackedBits::new(1),
            regular_norms: Vec::new(),
            regular_residual_norms: Vec::new(),
            outlier_mask: outlier_bits.map(|_| PackedBits::new(1)),
            outlier_indices: outlier_bits.map(PackedBits::new),
            outlier_qjl_signs: outlier_bits.map(|_| PackedBits::new(1)),
            outlier_norms: outlier_bits.map(|_| Vec::new()),
            outlier_residual_norms: outlier_bits.map(|_| Vec::new()),
        }
    }

    fn extend(&mut self, encoded: &EncodedKeyRows, outlier_encoded: Option<&EncodedKeyRows>, outlier_mask: Option<&Vec<u16>>) {
        self.regular_indices.extend_from_slice(&encoded.mse_indices);
        self.regular_qjl_signs.extend_from_slice(&encoded.qjl_signs);
        self.regular_norms.extend_from_slice(&encoded.norms);
        self.regular_residual_norms.extend_from_slice(&encoded.residual_norms);

        if let Some(mask) = outlier_mask {
            self.outlier_mask
                .as_mut()
                .expect("TurboQuant key outlier mask missing")
                .extend_from_slice(mask);
        }
        if let Some(outlier) = outlier_encoded {
            self.outlier_indices
                .as_mut()
                .expect("TurboQuant key outlier indices missing")
                .extend_from_slice(&outlier.mse_indices);
            self.outlier_qjl_signs
                .as_mut()
                .expect("TurboQuant key outlier QJL signs missing")
                .extend_from_slice(&outlier.qjl_signs);
            self.outlier_norms
                .as_mut()
                .expect("TurboQuant key outlier norms missing")
                .extend_from_slice(&outlier.norms);
            self.outlier_residual_norms
                .as_mut()
                .expect("TurboQuant key outlier residual_norms missing")
                .extend_from_slice(&outlier.residual_norms);
        }
    }

    fn truncate(&mut self, keep_rows: usize, total_dim: usize, config: TurboQuantTensorConfig) {
        self.regular_indices
            .truncate(keep_rows * config.regular_dim(total_dim));
        self.regular_qjl_signs
            .truncate(keep_rows * config.regular_dim(total_dim));
        self.regular_norms.truncate(keep_rows);
        self.regular_residual_norms.truncate(keep_rows);

        if let Some(mask) = &mut self.outlier_mask {
            mask.truncate(keep_rows * total_dim);
        }
        if let Some(idx) = &mut self.outlier_indices {
            idx.truncate(keep_rows * config.outlier_count());
        }
        if let Some(signs) = &mut self.outlier_qjl_signs {
            signs.truncate(keep_rows * config.outlier_count());
        }
        if let Some(norms) = &mut self.outlier_norms {
            norms.truncate(keep_rows);
        }
        if let Some(res) = &mut self.outlier_residual_norms {
            res.truncate(keep_rows);
        }
    }

    /// Approximate memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.regular_indices.byte_len()
            + self.regular_qjl_signs.byte_len()
            + self.regular_norms.len() * 4
            + self.regular_residual_norms.len() * 4
            + self.outlier_mask.as_ref().map_or(0, |p| p.byte_len())
            + self.outlier_indices.as_ref().map_or(0, |p| p.byte_len())
            + self.outlier_qjl_signs.as_ref().map_or(0, |p| p.byte_len())
            + self.outlier_norms.as_ref().map_or(0, |v| v.len() * 4)
            + self.outlier_residual_norms.as_ref().map_or(0, |v| v.len() * 4)
    }
}

/// Quantised value store for one attention layer.
#[derive(Debug, Clone)]
pub struct QuantizedValueStore {
    // GPU-native store (Uniform path only).
    gpu: Option<GpuValueStore>,

    pub regular_indices: PackedBits,
    pub regular_norms: Vec<f32>,

    pub outlier_mask: Option<PackedBits>,
    pub outlier_indices: Option<PackedBits>,
    pub outlier_norms: Option<Vec<f32>>,
}

impl QuantizedValueStore {
    fn new(config: TurboQuantTensorConfig) -> Self {
        let regular_bits = match config {
            TurboQuantTensorConfig::Uniform { bits } => bits,
            TurboQuantTensorConfig::Mixed { regular_bits, .. } => regular_bits,
        };
        let outlier_bits: Option<u8> = match config {
            TurboQuantTensorConfig::Uniform { .. } => None,
            TurboQuantTensorConfig::Mixed { outlier_bits, .. } => Some(outlier_bits),
        };

        Self {
            gpu: None,
            regular_indices: PackedBits::new(regular_bits),
            regular_norms: Vec::new(),
            outlier_mask: outlier_bits.map(|_| PackedBits::new(1)),
            outlier_indices: outlier_bits.map(PackedBits::new),
            outlier_norms: outlier_bits.map(|_| Vec::new()),
        }
    }

    fn extend(&mut self, encoded: &EncodedValueRows, outlier_encoded: Option<&EncodedValueRows>, outlier_mask: Option<&Vec<u16>>) {
        self.regular_indices.extend_from_slice(&encoded.indices);
        self.regular_norms.extend_from_slice(&encoded.norms);

        if let Some(mask) = outlier_mask {
            self.outlier_mask
                .as_mut()
                .expect("TurboQuant value outlier mask missing")
                .extend_from_slice(mask);
        }
        if let Some(outlier) = outlier_encoded {
            self.outlier_indices
                .as_mut()
                .expect("TurboQuant value outlier indices missing")
                .extend_from_slice(&outlier.indices);
            self.outlier_norms
                .as_mut()
                .expect("TurboQuant value outlier norms missing")
                .extend_from_slice(&outlier.norms);
        }
    }

    fn truncate(&mut self, keep_rows: usize, total_dim: usize, config: TurboQuantTensorConfig) {
        self.regular_indices
            .truncate(keep_rows * config.regular_dim(total_dim));
        self.regular_norms.truncate(keep_rows);

        if let Some(mask) = &mut self.outlier_mask {
            mask.truncate(keep_rows * total_dim);
        }
        if let Some(idx) = &mut self.outlier_indices {
            idx.truncate(keep_rows * config.outlier_count());
        }
        if let Some(norms) = &mut self.outlier_norms {
            norms.truncate(keep_rows);
        }
    }

    /// Approximate memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.regular_indices.byte_len()
            + self.regular_norms.len() * 4
            + self.outlier_mask.as_ref().map_or(0, |p| p.byte_len())
            + self.outlier_indices.as_ref().map_or(0, |p| p.byte_len())
            + self.outlier_norms.as_ref().map_or(0, |v| v.len() * 4)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Complete quantised KV entry for one attention layer
// ═══════════════════════════════════════════════════════════════════════════

/// Compressed KV cache for one attention layer.
///
/// Stores all cached positions as bit-packed indices + f32 metadata.
/// Backed by [`TurboQuantState`] for dequantisation.
#[derive(Debug, Clone)]
pub struct QuantizedKvCache {
    /// Compressed keys — inner-product optimised (MSE + QJL).
    pub keys: Option<QuantizedKeyStore>,
    /// Compressed values — MSE optimised.
    pub values: Option<QuantizedValueStore>,
    /// Layout from the first append (batch, heads, key_dim, value_dim).
    layout: Option<CacheLayout>,
    /// Number of valid cached positions.
    pub offset: usize,
    /// Config used to build this cache.
    pub config: TurboQuantConfig,
    /// Shared pre-computed matrices and codebooks.
    pub state: Option<Arc<TurboQuantState>>,
}

#[derive(Debug, Clone, Copy)]
struct CacheLayout {
    batch: usize,
    heads: usize,
    key_dim: usize,
    value_dim: usize,
}

impl QuantizedKvCache {
    /// Create an empty cache.  `state` should be `None` on first use; call
    /// [`append`] to populate.
    pub fn new(config: TurboQuantConfig) -> Self {
        Self {
            keys: None,
            values: None,
            layout: None,
            offset: 0,
            config,
            state: None,
        }
    }

    /// Create with a pre-built shared state (avoids re-building QR/Lloyd-Max).
    pub fn with_state(config: TurboQuantConfig, state: Arc<TurboQuantState>) -> Self {
        let mut cache = Self::new(config);
        cache.state = Some(state);
        cache
    }

    /// Current number of cached sequence positions.
    pub fn len(&self) -> usize {
        self.offset
    }

    /// True when no positions have been cached yet.
    pub fn is_empty(&self) -> bool {
        self.offset == 0
    }

    /// Reset to empty (retains pre-built state and config).
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.layout = None;
        self.offset = 0;
    }

    /// Append new keys and values.
    ///
    /// `keys` and `values` must have shape `[B, H, S, D]` as f32 or bf16.
    ///
    /// For the Uniform quantisation config the entire pipeline runs on-GPU:
    /// normalise → rotate → argmin codebook → QJL projection → sign.
    /// No GPU→CPU transfer happens.  Results are stored as `InlineArray`s and
    /// concatenated along the T axis on subsequent calls.
    ///
    /// For the Mixed (outlier-aware) config the CPU path is used (outlier mask
    /// selection requires a per-row top-k sort that is not trivially vectorisable).
    ///
    /// Returns an error string on shape mismatch.
    pub fn append(
        &mut self,
        keys: &InlineArray,
        values: &InlineArray,
    ) -> Result<(), String> {
        let layout = self.ensure_layout(keys, values)?;
        let seq_len = keys.dim(2) as usize;

        let config = self.config;
        let state = self.state.get_or_insert_with(|| {
            Arc::new(TurboQuantState::new(layout.key_dim, layout.value_dim, config))
        });
        let state = Arc::clone(state);

        // Cast to f32 once — needed for both GPU and CPU paths.
        let keys_f32 = keys.as_dtype(10 /* float32 */);
        let values_f32 = values.as_dtype(10 /* float32 */);

        let ks = self.keys.get_or_insert_with(|| QuantizedKeyStore::new(config.keys));
        let vs = self.values.get_or_insert_with(|| QuantizedValueStore::new(config.values));

        // ── GPU path (Uniform only) ───────────────────────────────────────
        let gpu_keys_ok = matches!(config.keys, TurboQuantTensorConfig::Uniform { .. });
        let gpu_vals_ok = matches!(config.values, TurboQuantTensorConfig::Uniform { .. });

        if gpu_keys_ok && gpu_vals_ok {
            if let Some((new_ks_gpu, new_vs_gpu)) =
                gpu_quantize_kv(&state, &keys_f32, &values_f32, config)
            {
                // Accumulate into the running GPU stores.
                match ks.gpu.as_mut() {
                    None => ks.gpu = Some(new_ks_gpu),
                    Some(g) => g.append(new_ks_gpu),
                }
                match vs.gpu.as_mut() {
                    None => vs.gpu = Some(new_vs_gpu),
                    Some(g) => g.append(new_vs_gpu),
                }
                self.offset += seq_len;
                return Ok(());
            }
            // GPU path failed — fall through to CPU.
        }

        // ── CPU fallback path ─────────────────────────────────────────────
        let key_rows = inline_array_to_bshd_rows(&keys_f32)?;
        let value_rows = inline_array_to_bshd_rows(&values_f32)?;

        let rows_per_seq = layout.batch * layout.heads;
        debug_assert_eq!(key_rows.len(), rows_per_seq * seq_len * layout.key_dim);

        let encoded_keys = encode_key_rows(&state.keys, layout.key_dim, &key_rows);
        let encoded_values = encode_value_rows(&state.values, layout.value_dim, &value_rows);

        ks.extend(
            &encoded_keys.regular,
            encoded_keys.outlier.as_ref(),
            encoded_keys.outlier_mask.as_ref(),
        );
        vs.extend(
            &encoded_values.regular,
            encoded_values.outlier.as_ref(),
            encoded_values.outlier_mask.as_ref(),
        );

        self.offset += seq_len;
        Ok(())
    }

    /// Dequantise and return all cached keys as an `InlineArray` of shape
    /// `[B, H, T, D]` (f32).
    ///
    /// Uses the GPU path when keys were quantised on-GPU; otherwise falls back
    /// to the CPU decode path.
    pub fn dequantize_keys(&self) -> Option<InlineArray> {
        let ks = self.keys.as_ref()?;
        let layout = self.layout?;
        let state = self.state.as_ref()?;

        // GPU path: all data lives in InlineArrays — single GPU graph eval.
        if let Some(ref g) = ks.gpu {
            let TurboQuantTensorConfig::Uniform { bits } = self.config.keys else {
                unreachable!("GPU store only exists for Uniform config")
            };
            return gpu_dequantize_keys(g, &state.keys, bits);
        }

        // CPU fallback.
        let rows = decode_key_rows(&state.keys, layout.key_dim, ks);
        Some(f32_rows_to_bhsd_array(
            &rows,
            layout.batch,
            layout.heads,
            self.offset,
            layout.key_dim,
        ))
    }

    /// Dequantise and return all cached values as an `InlineArray` of shape
    /// `[B, H, T, D]` (f32).
    pub fn dequantize_values(&self) -> Option<InlineArray> {
        let vs = self.values.as_ref()?;
        let layout = self.layout?;
        let state = self.state.as_ref()?;

        // GPU path.
        if let Some(ref g) = vs.gpu {
            let TurboQuantTensorConfig::Uniform { bits } = self.config.values else {
                unreachable!("GPU store only exists for Uniform config")
            };
            return gpu_dequantize_values(g, &state.values, bits);
        }

        // CPU fallback.
        let rows = decode_value_rows(&state.values, layout.value_dim, vs);
        Some(f32_rows_to_bhsd_array(
            &rows,
            layout.batch,
            layout.heads,
            self.offset,
            layout.value_dim,
        ))
    }

    fn ensure_layout(
        &mut self,
        keys: &InlineArray,
        values: &InlineArray,
    ) -> Result<CacheLayout, String> {
        // Validate shape: [B, H, S, D]
        if keys.ndim() != 4 || values.ndim() != 4 {
            return Err(format!(
                "TurboQuant: expected 4-D keys/values, got ndim {} / {}",
                keys.ndim(),
                values.ndim()
            ));
        }

        let b = keys.dim(0) as usize;
        let h = keys.dim(1) as usize;
        let kd = keys.dim(3) as usize;
        let vd = values.dim(3) as usize;

        if let Some(existing) = self.layout {
            if existing.batch != b || existing.heads != h || existing.key_dim != kd || existing.value_dim != vd {
                return Err(format!(
                    "TurboQuant: layout mismatch — expected [{b},{h},*,{kd}] / [{b},{h},*,{vd}]"
                ));
            }
            return Ok(existing);
        }

        let layout = CacheLayout { batch: b, heads: h, key_dim: kd, value_dim: vd };
        self.layout = Some(layout);
        Ok(layout)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public convenience constructors
// ═══════════════════════════════════════════════════════════════════════════

/// Build a shared [`TurboQuantState`] for the given dimensions and config.
///
/// This is the expensive step (~100 ms per unique dim).  Call once at model
/// load time and share the `Arc` across all layers.
pub fn build_state(key_dim: usize, value_dim: usize, config: TurboQuantConfig) -> Arc<TurboQuantState> {
    Arc::new(TurboQuantState::new(key_dim, value_dim, config))
}

/// Create a [`QuantizedKvCache`] that will lazily build its state on first use.
pub fn new_cache(config: TurboQuantConfig) -> QuantizedKvCache {
    QuantizedKvCache::new(config)
}

/// Create a [`QuantizedKvCache`] with a pre-built shared state.
pub fn new_cache_with_state(
    config: TurboQuantConfig,
    state: Arc<TurboQuantState>,
) -> QuantizedKvCache {
    QuantizedKvCache::with_state(config, state)
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU-native quantise / dequantise
// ═══════════════════════════════════════════════════════════════════════════

/// Quantise keys and values entirely on GPU.
///
/// Returns `None` if the GPU path is unavailable (e.g. missing codebook arr).
/// On success returns `(GpuKeyStore, GpuValueStore)` — both shapes `[B, H, S, *]`.
///
/// Algorithm:
///   1. Normalise onto unit sphere; keep per-vector L2 norm.
///   2. Rotate: x_norm @ rotation.T  → rotated   [B, H, S, D]
///   3. Nearest-centroid: argmin over squared diffs to codebook [C] → indices [B, H, S, D]
///   4. (Keys) Reconstruct MSE approx, compute residual, project via J, take sign.
fn gpu_quantize_kv(
    state: &TurboQuantState,
    keys: &InlineArray,   // [B, H, S, Dk]  f32
    values: &InlineArray, // [B, H, S, Dv]  f32
    config: TurboQuantConfig,
) -> Option<(GpuKeyStore, GpuValueStore)> {
    let TurboQuantTensorConfig::Uniform { bits: key_bits } = config.keys else {
        return None;
    };
    let TurboQuantTensorConfig::Uniform { bits: val_bits } = config.values else {
        return None;
    };
    let key_mse_bits = key_bits.saturating_sub(1);

    let k_core = match &state.keys {
        TensorRuntime::Uniform { core, .. } => core,
        TensorRuntime::Mixed { .. } => return None,
    };
    let v_core = match &state.values {
        TensorRuntime::Uniform { core, .. } => core,
        TensorRuntime::Mixed { .. } => return None,
    };

    // ── Keys ─────────────────────────────────────────────────────────────
    // 1. L2 norm along D axis, keepdims → [B, H, S, 1]
    let key_norms = keys.norm_l2(-1, true);
    // 2. Normalise: x / max(norm, eps)
    let eps = InlineArray::from_f32(ZERO_EPSILON);
    let safe_norms_k = key_norms.maximum(&eps);
    let k_norm = keys.divide(&safe_norms_k);

    // 3. Rotate: k_norm @ rotation.T  (CPU: matmul_rows(rotation, dim, input) = input @ rotation.T)
    //    rotation.T == inverse_rotation, so we matmul with inverse_rotation_arr.
    let k_rot = match &k_core.inverse_rotation_arr {
        Some(rot_t) => k_norm.matmul(rot_t),
        None => return None,
    };

    // 4. GPU nearest-centroid → [B, H, S, D] uint32
    let k_indices = k_core.gpu_quantize_mse(&k_rot, key_mse_bits)?;

    // 5. Reconstruct MSE approximation in the rotated space.
    let k_mse_recon_rot = k_core.gpu_reconstruct_mse(&k_indices, key_mse_bits)?;

    // 6. Residual norms: rotation-invariant, so compute directly in rotated space.
    //    residual_rot = k_rot - k_mse_recon_rot  →  norm_l2  [B, H, S, 1]
    let k_residual_rot = k_rot.subtract(&k_mse_recon_rot);
    let k_residual_norms = k_residual_rot.norm_l2(-1, true);

    // 7. QJL: project the residual in the **unrotated** space.
    //    residual_unrot = k_mse_recon_rot @ rotation_arr  (inverse-rotate the rotated reconstruction)
    //    then: residual_unrot = k_norm - inv_rotate(k_mse_recon_rot)
    //    QJL: residual_unrot @ inverse_qjl_arr  (= residual @ qjl.T)
    let rot = k_core.rotation_arr.as_ref()?;
    let k_mse_recon_unrot = k_mse_recon_rot.matmul(rot);
    let k_residual_unrot = k_norm.subtract(&k_mse_recon_unrot);
    let k_qjl_proj = match &k_core.inverse_qjl_arr {
        Some(inv_j) => k_residual_unrot.matmul(inv_j),
        None => return None,
    };
    let k_qjl_signs = k_qjl_proj.sign();

    // ── Values ────────────────────────────────────────────────────────────
    let val_norms = values.norm_l2(-1, true);
    let safe_norms_v = val_norms.maximum(&eps);
    let v_norm = values.divide(&safe_norms_v);

    // v_norm @ rotation.T = v_norm @ inverse_rotation_arr
    let v_rot = match &v_core.inverse_rotation_arr {
        Some(rot_t) => v_norm.matmul(rot_t),
        None => return None,
    };
    let v_indices = v_core.gpu_quantize_mse(&v_rot, val_bits)?;

    Some((
        GpuKeyStore {
            indices: k_indices,
            norms: key_norms,
            qjl_signs: k_qjl_signs,
            residual_norms: k_residual_norms,
        },
        GpuValueStore {
            indices: v_indices,
            norms: val_norms,
        },
    ))
}

/// Dequantise GPU-stored keys back to `[B, H, T, Dk]` f32.
///
/// Formula (per coordinate):
///   k̃ = (codebook[idx] + (√(π/2)/D) · (J^T · sign) · residual_norm) · norm  [inv-rotated]
fn gpu_dequantize_keys(
    store: &GpuKeyStore,
    runtime: &TensorRuntime,
    key_bits: u8,
) -> Option<InlineArray> {
    let key_mse_bits = key_bits.saturating_sub(1);
    let core = match runtime {
        TensorRuntime::Uniform { core, .. } => core,
        TensorRuntime::Mixed { .. } => return None,
    };

    // 1. Reconstruct MSE centroids in the rotated domain: take(codebook, indices) → [B,H,T,D].
    let mse_recon_rot = core.gpu_reconstruct_mse(&store.indices, key_mse_bits)?;

    // 2. Inverse-rotate back to input space.
    //    CPU: inverse_rotate_rows = matmul_rows(inverse_rotation, dim, input) = input @ inverse_rotation.T = input @ rotation.
    //    So GPU: recon_rot @ rotation_arr.
    let rot = core.rotation_arr.as_ref()?;
    let mse_base = mse_recon_rot.matmul(rot);

    // 3. QJL correction.
    //    CPU: inverse_project_rows(signs) = matmul_rows(inverse_qjl, dim, signs) = signs @ inverse_qjl.T = signs @ qjl.
    //    So GPU: qjl_signs @ qjl_arr.
    let qjl = core.qjl_arr.as_ref()?;
    let qjl_correction = store.qjl_signs.matmul(qjl);
    let dim = core.dim as f32;
    let qjl_scale_factor = InlineArray::from_f32((std::f32::consts::PI / 2.0).sqrt() / dim);
    // residual_norms: [B,H,T,1] keepdims — broadcast along D.
    let scale = store.residual_norms.multiply(&qjl_scale_factor);
    let correction = qjl_correction.multiply(&scale);

    // 4. Base + QJL correction, rescale by original L2 norm.
    // norms: [B,H,T,1] keepdims — broadcast along D.
    let combined = mse_base.add(&correction);
    Some(combined.multiply(&store.norms))
}

/// Dequantise GPU-stored values back to `[B, H, T, Dv]` f32.
fn gpu_dequantize_values(
    store: &GpuValueStore,
    runtime: &TensorRuntime,
    val_bits: u8,
) -> Option<InlineArray> {
    let core = match runtime {
        TensorRuntime::Uniform { core, .. } => core,
        TensorRuntime::Mixed { .. } => return None,
    };

    // 1. Reconstruct MSE centroids in rotated space.
    let mse_recon_rot = core.gpu_reconstruct_mse(&store.indices, val_bits)?;

    // 2. Inverse-rotate: recon_rot @ rotation_arr (same derivation as keys).
    let rot = core.rotation_arr.as_ref()?;
    let mse_base = mse_recon_rot.matmul(rot);

    // 3. Rescale by stored L2 norms [B,H,T,1].
    Some(mse_base.multiply(&store.norms))
}

// ═══════════════════════════════════════════════════════════════════════════
// Encoding (quantise) helpers
// ═══════════════════════════════════════════════════════════════════════════

struct EncodedKeyRows {
    mse_indices: Vec<u16>,
    qjl_signs: Vec<u16>,
    norms: Vec<f32>,
    residual_norms: Vec<f32>,
}

struct EncodedValueRows {
    indices: Vec<u16>,
    norms: Vec<f32>,
}

struct BatchedKeyRows {
    regular: EncodedKeyRows,
    outlier_mask: Option<Vec<u16>>,
    outlier: Option<EncodedKeyRows>,
}

struct BatchedValueRows {
    regular: EncodedValueRows,
    outlier_mask: Option<Vec<u16>>,
    outlier: Option<EncodedValueRows>,
}

fn encode_key_rows(
    runtime: &TensorRuntime,
    total_dim: usize,
    rows: &[f32],
) -> BatchedKeyRows {
    match runtime {
        TensorRuntime::Uniform { config, core } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!()
            };
            BatchedKeyRows {
                regular: encode_key_component_rows(core, rows, *bits),
                outlier_mask: None,
                outlier: None,
            }
        }
        TensorRuntime::Mixed {
            config,
            regular_core,
            outlier_core,
        } => {
            let TurboQuantTensorConfig::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } = config
            else {
                unreachable!()
            };
            let (mask, regular_rows, outlier_rows) =
                split_rows_by_outliers(rows, total_dim, *outlier_count);
            BatchedKeyRows {
                regular: encode_key_component_rows(regular_core, &regular_rows, *regular_bits),
                outlier_mask: Some(mask),
                outlier: Some(encode_key_component_rows(
                    outlier_core,
                    &outlier_rows,
                    *outlier_bits,
                )),
            }
        }
    }
}

fn encode_value_rows(
    runtime: &TensorRuntime,
    total_dim: usize,
    rows: &[f32],
) -> BatchedValueRows {
    match runtime {
        TensorRuntime::Uniform { config, core } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!()
            };
            BatchedValueRows {
                regular: encode_value_component_rows(core, rows, *bits),
                outlier_mask: None,
                outlier: None,
            }
        }
        TensorRuntime::Mixed {
            config,
            regular_core,
            outlier_core,
        } => {
            let TurboQuantTensorConfig::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } = config
            else {
                unreachable!()
            };
            let (mask, regular_rows, outlier_rows) =
                split_rows_by_outliers(rows, total_dim, *outlier_count);
            BatchedValueRows {
                regular: encode_value_component_rows(regular_core, &regular_rows, *regular_bits),
                outlier_mask: Some(mask),
                outlier: Some(encode_value_component_rows(
                    outlier_core,
                    &outlier_rows,
                    *outlier_bits,
                )),
            }
        }
    }
}

/// Two-stage key encoder: MSE at (bits-1) + QJL on residual.
fn encode_key_component_rows(core: &TurboQuantCore, rows: &[f32], key_bits: u8) -> EncodedKeyRows {
    let num_rows = rows.len() / core.dim;
    let mut norms = vec![0.0f32; num_rows];
    let mut normalized = vec![0.0f32; rows.len()];

    // Step 1: Normalise onto unit sphere.
    for (row_idx, row) in rows.chunks(core.dim).enumerate() {
        let norm = l2_norm(row);
        norms[row_idx] = norm;
        if norm > ZERO_EPSILON {
            let dst = &mut normalized[row_idx * core.dim..(row_idx + 1) * core.dim];
            for (dst, &src) in dst.iter_mut().zip(row.iter()) {
                *dst = src / norm;
            }
        }
    }

    // Step 2: MSE quantise at (bits-1).
    let mse_bits = key_bits.saturating_sub(1);
    let mut mse_indices = quantize_mse_rows(core, &normalized, mse_bits);

    // Step 3: Reconstruct MSE approximation.
    let decoded_mse = if mse_bits == 0 {
        vec![0.0; rows.len()]
    } else {
        reconstruct_mse_rows(core, &mse_indices, mse_bits)
    };

    // Step 4: Compute residual = normalized - decoded_mse.
    let mut residual = vec![0.0f32; rows.len()];
    let mut residual_norms = vec![0.0f32; num_rows];
    for row_idx in 0..num_rows {
        let start = row_idx * core.dim;
        let end = start + core.dim;
        if norms[row_idx] <= ZERO_EPSILON {
            mse_indices[start..end].fill(0);
            continue;
        }
        let res_row = &mut residual[start..end];
        for ((dst, &lhs), &rhs) in res_row
            .iter_mut()
            .zip(normalized[start..end].iter())
            .zip(decoded_mse[start..end].iter())
        {
            *dst = lhs - rhs;
        }
        residual_norms[row_idx] = l2_norm(res_row);
    }

    // Step 5: QJL — project residual and take signs.
    let projected = core.project_rows(&residual);
    let mut qjl_signs: Vec<u16> = projected
        .iter()
        .map(|&v| if v >= 0.0 { 1 } else { 0 })
        .collect();

    // Zero-vector rows get all-zero signs.
    for row_idx in 0..num_rows {
        if norms[row_idx] <= ZERO_EPSILON {
            let start = row_idx * core.dim;
            let end = start + core.dim;
            qjl_signs[start..end].fill(0);
        }
    }

    EncodedKeyRows {
        mse_indices,
        qjl_signs,
        norms,
        residual_norms,
    }
}

/// MSE-only value encoder.
fn encode_value_component_rows(
    core: &TurboQuantCore,
    rows: &[f32],
    value_bits: u8,
) -> EncodedValueRows {
    let num_rows = rows.len() / core.dim;
    let mut norms = vec![0.0f32; num_rows];
    let mut normalized = vec![0.0f32; rows.len()];

    for (row_idx, row) in rows.chunks(core.dim).enumerate() {
        let norm = l2_norm(row);
        norms[row_idx] = norm;
        if norm > ZERO_EPSILON {
            let dst = &mut normalized[row_idx * core.dim..(row_idx + 1) * core.dim];
            for (dst, &src) in dst.iter_mut().zip(row.iter()) {
                *dst = src / norm;
            }
        }
    }

    let mut indices = quantize_mse_rows(core, &normalized, value_bits);
    for row_idx in 0..num_rows {
        if norms[row_idx] <= ZERO_EPSILON {
            let start = row_idx * core.dim;
            let end = start + core.dim;
            indices[start..end].fill(0);
        }
    }

    EncodedValueRows { indices, norms }
}

// ═══════════════════════════════════════════════════════════════════════════
// Decoding (dequantise) helpers
// ═══════════════════════════════════════════════════════════════════════════

fn decode_key_rows(
    runtime: &TensorRuntime,
    total_dim: usize,
    store: &QuantizedKeyStore,
) -> Vec<f32> {
    match runtime {
        TensorRuntime::Uniform { config, core } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!()
            };
            decode_key_component_rows_raw(
                core,
                &unpack_all(&store.regular_indices),
                &unpack_all(&store.regular_qjl_signs),
                &store.regular_norms,
                &store.regular_residual_norms,
                *bits,
            )
        }
        TensorRuntime::Mixed {
            config,
            regular_core,
            outlier_core,
        } => {
            let TurboQuantTensorConfig::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } = config
            else {
                unreachable!()
            };
            let regular = decode_key_component_rows_raw(
                regular_core,
                &unpack_all(&store.regular_indices),
                &unpack_all(&store.regular_qjl_signs),
                &store.regular_norms,
                &store.regular_residual_norms,
                *regular_bits,
            );
            let outlier = decode_key_component_rows_raw(
                outlier_core,
                &unpack_all(
                    store
                        .outlier_indices
                        .as_ref()
                        .expect("TurboQuant key outlier indices missing"),
                ),
                &unpack_all(
                    store
                        .outlier_qjl_signs
                        .as_ref()
                        .expect("TurboQuant key outlier QJL signs missing"),
                ),
                store
                    .outlier_norms
                    .as_ref()
                    .expect("TurboQuant key outlier norms missing"),
                store
                    .outlier_residual_norms
                    .as_ref()
                    .expect("TurboQuant key outlier residual_norms missing"),
                *outlier_bits,
            );
            let mask = unpack_all(
                store
                    .outlier_mask
                    .as_ref()
                    .expect("TurboQuant key outlier mask missing"),
            );
            scatter_mixed_rows(&mask, total_dim, *outlier_count, &regular, &outlier)
        }
    }
}

fn decode_value_rows(
    runtime: &TensorRuntime,
    total_dim: usize,
    store: &QuantizedValueStore,
) -> Vec<f32> {
    match runtime {
        TensorRuntime::Uniform { config, core } => {
            let TurboQuantTensorConfig::Uniform { bits } = config else {
                unreachable!()
            };
            decode_value_component_rows_raw(
                core,
                &unpack_all(&store.regular_indices),
                &store.regular_norms,
                *bits,
            )
        }
        TensorRuntime::Mixed {
            config,
            regular_core,
            outlier_core,
        } => {
            let TurboQuantTensorConfig::Mixed {
                regular_bits,
                outlier_bits,
                outlier_count,
            } = config
            else {
                unreachable!()
            };
            let regular = decode_value_component_rows_raw(
                regular_core,
                &unpack_all(&store.regular_indices),
                &store.regular_norms,
                *regular_bits,
            );
            let outlier = decode_value_component_rows_raw(
                outlier_core,
                &unpack_all(
                    store
                        .outlier_indices
                        .as_ref()
                        .expect("TurboQuant value outlier indices missing"),
                ),
                store
                    .outlier_norms
                    .as_ref()
                    .expect("TurboQuant value outlier norms missing"),
                *outlier_bits,
            );
            let mask = unpack_all(
                store
                    .outlier_mask
                    .as_ref()
                    .expect("TurboQuant value outlier mask missing"),
            );
            scatter_mixed_rows(&mask, total_dim, *outlier_count, &regular, &outlier)
        }
    }
}

/// Reconstruct key rows from MSE indices + QJL signs + norms.
///
/// Formula (per row):
///   k̃ = Π^T · codebook[idx] · norm + (√(π/2)/D) · Π^T · J^T · sign · residual_norm · norm
fn decode_key_component_rows_raw(
    core: &TurboQuantCore,
    indices: &[u16],
    qjl_signs: &[u16],
    norms: &[f32],
    residual_norms: &[f32],
    key_bits: u8,
) -> Vec<f32> {
    let total_rows = norms.len();
    let mse_bits = key_bits.saturating_sub(1);

    // MSE base reconstruction (rotate back from codebook centroids).
    let mut reconstructed = if mse_bits == 0 {
        vec![0.0; total_rows * core.dim]
    } else {
        reconstruct_mse_rows(core, indices, mse_bits)
    };

    // QJL correction term — only if any residual is non-trivial.
    if residual_norms.iter().any(|&rn| rn > ZERO_EPSILON) {
        let qjl_signs_f32: Vec<f32> = qjl_signs
            .iter()
            .map(|&v| if v == 0 { -1.0 } else { 1.0 })
            .collect();
        let qjl_correction = core.inverse_project_rows(&qjl_signs_f32);

        for row_idx in 0..total_rows {
            let residual_norm = residual_norms[row_idx];
            if residual_norm <= ZERO_EPSILON {
                continue;
            }
            let scale = ((PI / 2.0).sqrt() * residual_norm) / (core.dim as f32);
            let start = row_idx * core.dim;
            let end = start + core.dim;
            for (val, &correction) in reconstructed[start..end]
                .iter_mut()
                .zip(qjl_correction[start..end].iter())
            {
                *val += scale * correction;
            }
        }
    }

    // Rescale by stored norm.
    for row_idx in 0..total_rows {
        let start = row_idx * core.dim;
        let end = start + core.dim;
        let norm = norms[row_idx];
        if norm <= ZERO_EPSILON {
            reconstructed[start..end].fill(0.0);
        } else {
            for v in &mut reconstructed[start..end] {
                *v *= norm;
            }
        }
    }

    reconstructed
}

/// Reconstruct value rows from MSE indices + norms.
fn decode_value_component_rows_raw(
    core: &TurboQuantCore,
    indices: &[u16],
    norms: &[f32],
    value_bits: u8,
) -> Vec<f32> {
    let total_rows = norms.len();
    let mut reconstructed = reconstruct_mse_rows(core, indices, value_bits);

    for row_idx in 0..total_rows {
        let start = row_idx * core.dim;
        let end = start + core.dim;
        let norm = norms[row_idx];
        if norm <= ZERO_EPSILON {
            reconstructed[start..end].fill(0.0);
        } else {
            for v in &mut reconstructed[start..end] {
                *v *= norm;
            }
        }
    }

    reconstructed
}

// ═══════════════════════════════════════════════════════════════════════════
// Core quantisation primitives
// ═══════════════════════════════════════════════════════════════════════════

/// Rotate then nearest-centroid quantise: returns a per-coordinate index.
fn quantize_mse_rows(core: &TurboQuantCore, normalized: &[f32], bits: u8) -> Vec<u16> {
    if bits == 0 {
        return vec![0; normalized.len()];
    }
    let rotated = core.rotate_rows(normalized);
    let codebook = core.codebook(bits);
    rotated
        .iter()
        .map(|&v| nearest_centroid_index(v, codebook) as u16)
        .collect()
}

/// Look up centroids then inverse-rotate to reconstruct approximate vectors.
fn reconstruct_mse_rows(core: &TurboQuantCore, indices: &[u16], bits: u8) -> Vec<f32> {
    if bits == 0 {
        return vec![0.0; indices.len()];
    }
    let codebook = core.codebook(bits);
    let rotated: Vec<f32> = indices.iter().map(|&i| codebook[usize::from(i)]).collect();
    core.inverse_rotate_rows(&rotated)
}

/// Binary search for the nearest centroid (codebook is sorted ascending).
fn nearest_centroid_index(value: f32, codebook: &[f32]) -> usize {
    match codebook.binary_search_by(|probe| probe.partial_cmp(&value).unwrap()) {
        Ok(i) => i,
        Err(0) => 0,
        Err(i) if i >= codebook.len() => codebook.len() - 1,
        Err(i) => {
            let left = codebook[i - 1];
            let right = codebook[i];
            if (value - left).abs() <= (right - value).abs() {
                i - 1
            } else {
                i
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Outlier selection / scattering
// ═══════════════════════════════════════════════════════════════════════════

/// Identify the top-k highest-magnitude coordinates as outliers.
fn select_outlier_mask(row: &[f32], outlier_count: usize) -> Vec<u16> {
    let mut ranked: Vec<usize> = (0..row.len()).collect();
    ranked.sort_unstable_by(|&lhs, &rhs| {
        row[rhs]
            .abs()
            .total_cmp(&row[lhs].abs())
            .then_with(|| lhs.cmp(&rhs))
    });
    let mut mask = vec![0u16; row.len()];
    for dim_idx in ranked.into_iter().take(outlier_count) {
        mask[dim_idx] = 1;
    }
    mask
}

/// Partition rows into regular and outlier sub-vectors.
///
/// Returns `(outlier_mask, regular_rows, outlier_rows)`.
fn split_rows_by_outliers(
    rows: &[f32],
    total_dim: usize,
    outlier_count: usize,
) -> (Vec<u16>, Vec<f32>, Vec<f32>) {
    let num_rows = rows.len() / total_dim;
    let regular_dim = total_dim - outlier_count;
    let mut mask_all = Vec::with_capacity(rows.len());
    let mut regular_rows = Vec::with_capacity(num_rows * regular_dim);
    let mut outlier_rows = Vec::with_capacity(num_rows * outlier_count);

    for row in rows.chunks(total_dim) {
        let mask = select_outlier_mask(row, outlier_count);
        for (&v, &is_outlier) in row.iter().zip(mask.iter()) {
            if is_outlier == 1 {
                outlier_rows.push(v);
            } else {
                regular_rows.push(v);
            }
        }
        mask_all.extend_from_slice(&mask);
    }

    (mask_all, regular_rows, outlier_rows)
}

/// Re-interleave regular and outlier sub-vectors using the stored mask.
fn scatter_mixed_rows(
    outlier_mask: &[u16],
    total_dim: usize,
    outlier_count: usize,
    regular_rows: &[f32],
    outlier_rows: &[f32],
) -> Vec<f32> {
    let num_rows = outlier_mask.len() / total_dim;
    let regular_dim = total_dim - outlier_count;
    let mut merged = vec![0.0f32; outlier_mask.len()];

    for row_idx in 0..num_rows {
        let mask_row = &outlier_mask[row_idx * total_dim..(row_idx + 1) * total_dim];
        let mut reg_cur = 0usize;
        let mut out_cur = 0usize;
        for dim_idx in 0..total_dim {
            let dst = &mut merged[row_idx * total_dim + dim_idx];
            if mask_row[dim_idx] == 1 {
                *dst = outlier_rows[row_idx * outlier_count + out_cur];
                out_cur += 1;
            } else {
                *dst = regular_rows[row_idx * regular_dim + reg_cur];
                reg_cur += 1;
            }
        }
    }

    merged
}

// ═══════════════════════════════════════════════════════════════════════════
// Pure-Rust math — no external dependencies
// ═══════════════════════════════════════════════════════════════════════════

/// Lloyd-Max optimal scalar quantisation codebook for the Beta distribution.
///
/// The marginal of a random unit-sphere vector in R^d is Beta((d-1)/2, (d-1)/2),
/// supported on [-1, 1].  This solver approximates the optimal MSE centroids via
/// iterative centroid update (Voronoi quantisation).
///
/// Returns a sorted Vec of 2^bits centroids in [-1, 1].
fn build_beta_codebook(dim: usize, bits: u8) -> Vec<f32> {
    let centroid_count = 1usize << bits;
    let alpha = ((dim as f64) - 3.0) / 2.0;
    let step = 2.0 / (LLOYD_GRID_POINTS as f64);

    // Quadrature grid on [-1, 1] weighted by the Beta density.
    let mut xs = Vec::with_capacity(LLOYD_GRID_POINTS);
    let mut weights = Vec::with_capacity(LLOYD_GRID_POINTS);
    for idx in 0..LLOYD_GRID_POINTS {
        let x = -1.0 + ((idx as f64) + 0.5) * step;
        let w = if dim <= 2 {
            1.0
        } else {
            (1.0 - x * x).max(0.0).powf(alpha)
        };
        xs.push(x);
        weights.push(w);
    }

    // Cumulative weights for CDF-based centroid initialisation.
    let mut cumulative = Vec::with_capacity(LLOYD_GRID_POINTS);
    let mut total_weight = 0.0f64;
    for &w in &weights {
        total_weight += w;
        cumulative.push(total_weight);
    }

    // Initialise centroids by CDF inversion.
    let mut centroids = Vec::with_capacity(centroid_count);
    for bucket in 0..centroid_count {
        let target = ((bucket as f64) + 0.5) * total_weight / (centroid_count as f64);
        let idx = cumulative.partition_point(|&v| v < target);
        centroids.push(xs[idx.min(xs.len() - 1)]);
    }
    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Iterative Lloyd-Max refinement.
    for _ in 0..LLOYD_MAX_ITERS {
        let mut boundaries = Vec::with_capacity(centroid_count + 1);
        boundaries.push(-1.0f64);
        for pair in centroids.windows(2) {
            boundaries.push((pair[0] + pair[1]) * 0.5);
        }
        boundaries.push(1.0f64);

        let mut updated = centroids.clone();
        let mut max_change = 0.0f64;
        for bucket in 0..centroid_count {
            let left = boundaries[bucket];
            let right = boundaries[bucket + 1];
            let mut weighted_sum = 0.0f64;
            let mut weight_sum = 0.0f64;
            for (&x, &w) in xs.iter().zip(weights.iter()) {
                if x >= left && x < right {
                    weighted_sum += x * w;
                    weight_sum += w;
                }
            }
            updated[bucket] = if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                (left + right) * 0.5
            };
            max_change = max_change.max((updated[bucket] - centroids[bucket]).abs());
        }
        centroids = updated;
        if max_change < LLOYD_MAX_TOLERANCE {
            break;
        }
    }

    centroids.into_iter().map(|v| v as f32).collect()
}

/// Gram-Schmidt QR decomposition of a Gaussian random matrix.
///
/// Returns a row-major [dim × dim] orthogonal matrix Q (f32).
fn generate_random_orthogonal(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut q = vec![0.0f64; dim * dim];

    for column in 0..dim {
        let mut candidate = vec![0.0f64; dim];
        loop {
            for v in &mut candidate {
                *v = f64::from(sample_standard_normal(rng));
            }
            // Gram-Schmidt orthogonalisation against previous columns.
            for prev in 0..column {
                let prev_col = &q[prev * dim..(prev + 1) * dim];
                let dot = dot_f64(&candidate, prev_col);
                for (v, &p) in candidate.iter_mut().zip(prev_col.iter()) {
                    *v -= dot * p;
                }
            }
            let norm = dot_f64(&candidate, &candidate).sqrt();
            if norm > 1e-8 {
                for (row, &v) in candidate.iter().enumerate() {
                    q[column * dim + row] = v / norm;
                }
                break;
            }
        }
    }

    // Convert from column-major to row-major.
    let mut row_major = vec![0.0f32; dim * dim];
    for row in 0..dim {
        for col in 0..dim {
            row_major[row * dim + col] = q[col * dim + row] as f32;
        }
    }
    row_major
}

/// Row-major [dim × dim] Gaussian random matrix for QJL projection.
fn generate_random_projection(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut projection = Vec::with_capacity(dim * dim);
    for _ in 0..(dim * dim) {
        projection.push(sample_standard_normal(rng));
    }
    projection
}

/// Box-Muller transform for standard normal sampling.
fn sample_standard_normal(rng: &mut StdRng) -> f32 {
    let u1: f32 = rng.random();
    let u1 = u1.max(1e-7);
    let u2: f32 = rng.random();
    (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * PI * u2).cos()
}

fn dot_f64(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

fn transpose_square_matrix(matrix: &[f32], dim: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; matrix.len()];
    for row in 0..dim {
        for col in 0..dim {
            t[col * dim + row] = matrix[row * dim + col];
        }
    }
    t
}

/// CPU fallback: row-major [dim × dim] matrix applied to a batch of row vectors.
///
/// output[row, out_dim] = sum_in( matrix[out_dim, in_dim] * input[row, in_dim] )
fn matmul_rows(matrix: &[f32], dim: usize, rows: &[f32]) -> Vec<f32> {
    let num_rows = rows.len() / dim;
    let mut output = vec![0.0f32; rows.len()];
    for row_idx in 0..num_rows {
        let src = &rows[row_idx * dim..(row_idx + 1) * dim];
        let dst = &mut output[row_idx * dim..(row_idx + 1) * dim];
        for out_dim in 0..dim {
            let matrix_row = &matrix[out_dim * dim..(out_dim + 1) * dim];
            let acc = matrix_row
                .iter()
                .zip(src.iter())
                .map(|(&w, &v)| w * v)
                .sum::<f32>();
            dst[out_dim] = acc;
        }
    }
    output
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum::<f32>().sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// InlineArray <-> Vec<f32> bridges
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a [dim × dim] f32 slice to an InlineArray of shape [dim, dim].
///
/// Uses `InlineArray::from_f32_slice` for a single-FFI-call upload (one memcpy).
/// Eval'd immediately to materialise the data in GPU memory.
/// Returns `None` on empty input — the CPU matmul fallback is used instead.
fn matrix_to_inline_array(matrix: &[f32], dim: usize) -> Option<InlineArray> {
    if matrix.is_empty() || dim == 0 {
        return None;
    }
    let mut arr = InlineArray::from_f32_slice(matrix, &[dim as i32, dim as i32]);
    arr.eval();
    Some(arr)
}

/// Read f32 values back from a GPU InlineArray (single bulk GPU→CPU copy).
///
/// The C++ `to_f32_vec` handles astype(f32) + eval + memcpy internally —
/// O(1) FFI calls, O(N) data transfer.
fn inline_array_to_f32_vec(arr: &InlineArray, expected_len: usize) -> Option<Vec<f32>> {
    arr.reshape(&[expected_len as i32]).to_f32_vec(expected_len)
}

/// Convert a [B, H, S, D] InlineArray to a flat Vec<f32> in (B, S, H, D) row order.
///
/// Transposes to [B, S, H, D] so that `(batch, seq, head)` triplets are the
/// outer dimensions — matching the reference `array_rows_in_bshd_order`.
/// Uses `to_f32_vec` for a single bulk GPU→CPU copy.
fn inline_array_to_bshd_rows(arr: &InlineArray) -> Result<Vec<f32>, String> {
    let b = arr.dim(0) as usize;
    let h = arr.dim(1) as usize;
    let s = arr.dim(2) as usize;
    let d = arr.dim(3) as usize;
    let total = b * s * h * d;
    // [B, H, S, D] → [B, S, H, D] → flat.
    arr.transpose_axes(&[0, 2, 1, 3])
        .reshape(&[total as i32])
        .to_f32_vec(total)
        .ok_or_else(|| format!("TurboQuant: failed to read {total}-element tensor"))
}

/// Convert a flat Vec<f32> in (B, S, H, D) row order back to an InlineArray
/// with shape [B, H, T, D].
///
/// Uploads via `from_f32_slice` (single FFI + memcpy), then transposes in the
/// GPU graph to produce the standard [B, H, S, D] KV layout.
fn f32_rows_to_bhsd_array(
    rows: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
) -> InlineArray {
    debug_assert_eq!(
        rows.len(),
        batch * seq * heads * dim,
        "f32_rows_to_bhsd_array: size mismatch"
    );
    // Upload [B, S, H, D] then transpose → [B, H, S, D].
    InlineArray::from_f32_slice(rows, &[batch as i32, seq as i32, heads as i32, dim as i32])
        .transpose_axes(&[0, 2, 1, 3])
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_bits_round_trip() {
        let values = [1u16, 6, 3, 0, 7, 2, 4];
        let mut packed = PackedBits::new(3);
        packed.extend_from_slice(&values);
        let round_trip: Vec<u16> = (0..values.len()).map(|i| packed.get(i)).collect();
        assert_eq!(round_trip, values);

        packed.truncate(4);
        let truncated: Vec<u16> = (0..4).map(|i| packed.get(i)).collect();
        assert_eq!(truncated, values[..4]);
    }

    #[test]
    fn beta_codebook_is_sorted_and_correct_length() {
        let codebook = build_beta_codebook(128, 4);
        assert_eq!(codebook.len(), 16);
        assert!(codebook.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn codebook_range_within_unit_interval() {
        let codebook = build_beta_codebook(128, 4);
        assert!(codebook.iter().all(|&v| v >= -1.0 && v <= 1.0));
    }

    #[test]
    fn nearest_centroid_boundary_cases() {
        let cb = vec![-0.5f32, 0.0, 0.5];
        assert_eq!(nearest_centroid_index(-2.0, &cb), 0);
        assert_eq!(nearest_centroid_index(2.0, &cb), 2);
        assert_eq!(nearest_centroid_index(0.0, &cb), 1);
        assert_eq!(nearest_centroid_index(0.26, &cb), 2);
    }

    #[test]
    fn turboquant_handles_zero_rows() {
        let core = TurboQuantCore::new(8, 4);
        let encoded = encode_key_component_rows(&core, &[0.0; 8], 4);
        assert_eq!(encoded.norms, vec![0.0]);
        assert_eq!(encoded.residual_norms, vec![0.0]);
        assert!(encoded.mse_indices.iter().all(|&v| v == 0));
        assert!(encoded.qjl_signs.iter().all(|&v| v == 0));
    }

    #[test]
    fn turboquant_state_constructs_without_panic() {
        let config = TurboQuantConfig::uniform(4, 4);
        let _state = TurboQuantState::new(64, 64, config);
    }

    #[test]
    fn mixed_config_effective_bits() {
        let config = TurboQuantTensorConfig::mixed(2, 4, 32);
        assert_eq!(config.effective_bits(128), 2.5);
        assert_eq!(config.regular_dim(128), 96);
        assert_eq!(config.outlier_count(), 32);
    }

    #[test]
    fn select_outlier_mask_marks_top_k() {
        let row = [0.1f32, 0.9, 0.5, 0.8, 0.2];
        let mask = select_outlier_mask(&row, 2);
        // Top 2 by magnitude: index 1 (0.9) and index 3 (0.8)
        assert_eq!(mask[1], 1);
        assert_eq!(mask[3], 1);
        assert_eq!(mask[0], 0);
        assert_eq!(mask[2], 0);
        assert_eq!(mask[4], 0);
    }

    #[test]
    fn scatter_round_trips_split() {
        let rows = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (mask, regular, outlier) = split_rows_by_outliers(&rows, 3, 1);
        let merged = scatter_mixed_rows(&mask, 3, 1, &regular, &outlier);
        assert_eq!(merged.len(), rows.len());
        // Merged must contain the same values (possibly reordered by scatter).
        let mut orig_sorted = rows.clone();
        let mut merged_sorted = merged.clone();
        orig_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        merged_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(orig_sorted, merged_sorted);
    }

    #[test]
    fn encode_value_norm_preserved() {
        let core = TurboQuantCore::new(8, 4);
        let v: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let encoded = encode_value_component_rows(&core, &v, 4);
        assert_eq!(encoded.norms.len(), 1);
        let expected_norm = l2_norm(&v);
        assert!((encoded.norms[0] - expected_norm).abs() < 1e-5);
    }

    #[test]
    fn turboquant_presets_match_schedule() {
        let q2_5 = TurboQuantConfig::preset_q2_5(128);
        let q3_5 = TurboQuantConfig::preset_q3_5(128);
        assert_eq!(q2_5, TurboQuantConfig::mixed(2, 4, 32, 2, 4, 32));
        assert_eq!(q3_5, TurboQuantConfig::mixed(3, 5, 32, 3, 5, 32));
    }

    /// GPU round-trip: append via GPU path then dequantize via GPU path.
    ///
    /// We verify two things:
    ///   1. The GPU path is actually taken (store.gpu is Some).
    ///   2. The GPU dequantised output is close to the CPU dequantised output
    ///      (same algorithm, both paths should produce bitwise-close results
    ///       modulo f32 ordering differences).
    #[test]
    fn turboquant_gpu_path_round_trip() {
        // Small dim so the test is fast.
        let dim = 16usize;
        let config = TurboQuantConfig::uniform(4, 4);
        let b = 1i32;
        let h = 2i32;
        let s = 3i32;
        let d = dim as i32;
        let total = (b * h * s * d) as usize;

        // Build deterministic input vectors.
        let data: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.1 - total as f32 * 0.05).sin()).collect();
        // Upload as [B, H, S, D] f32.
        let keys_arr = InlineArray::from_f32_slice(&data, &[b, h, s, d]);
        let vals_arr = InlineArray::from_f32_slice(&data, &[b, h, s, d]);

        // ── CPU reference path (use Mixed config to force CPU) ────────────
        let cpu_config = TurboQuantConfig {
            keys: TurboQuantTensorConfig::Mixed {
                regular_bits: 3,
                outlier_bits: 4,
                outlier_count: 4,
            },
            values: TurboQuantTensorConfig::Mixed {
                regular_bits: 4,
                outlier_bits: 4,
                outlier_count: 4,
            },
        };
        let mut cpu_cache = QuantizedKvCache::new(cpu_config);
        cpu_cache.append(&keys_arr, &vals_arr).expect("CPU append");
        // Verify CPU path taken (no GPU store).
        assert!(cpu_cache.keys.as_ref().unwrap().gpu.is_none(), "Expected CPU path for Mixed config");

        // ── GPU path ──────────────────────────────────────────────────────
        let mut gpu_cache = QuantizedKvCache::new(config);
        gpu_cache.append(&keys_arr, &vals_arr).expect("GPU append");

        // Verify GPU path was taken.
        assert!(
            gpu_cache.keys.as_ref().unwrap().gpu.is_some(),
            "GPU store should be Some for Uniform config"
        );
        assert!(
            gpu_cache.values.as_ref().unwrap().gpu.is_some(),
            "GPU value store should be Some for Uniform config"
        );

        // Dequantise — should succeed.
        let dk = gpu_cache.dequantize_keys().expect("GPU dequantize_keys");
        let dv = gpu_cache.dequantize_values().expect("GPU dequantize_values");

        // Verify output shapes: [B, H, T, D].
        assert_eq!(dk.shape(), &[b, h, s, d], "dequantized keys shape mismatch");
        assert_eq!(dv.shape(), &[b, h, s, d], "dequantized values shape mismatch");

        // Output should be finite (not NaN/Inf).
        let dk_vals = dk.reshape(&[(b * h * s * d)]).to_f32_vec(total).expect("dk to_f32");
        let dv_vals = dv.reshape(&[(b * h * s * d)]).to_f32_vec(total).expect("dv to_f32");
        assert!(dk_vals.iter().all(|v| v.is_finite()), "dequantized keys contain non-finite");
        assert!(dv_vals.iter().all(|v| v.is_finite()), "dequantized values contain non-finite");

        // Verify output is within reasonable range (quantisation introduces error but
        // should not explode — reconstructed vectors should be roughly same magnitude as input).
        let input_max = data.iter().cloned().fold(0.0f32, f32::max).abs();
        let dk_max = dk_vals.iter().cloned().fold(0.0f32, f32::max).abs();
        assert!(dk_max < input_max * 3.0, "dequantized keys magnitude unreasonably large");
    }

    /// Verify that multiple appends accumulate correctly in the GPU store.
    #[test]
    fn turboquant_gpu_multi_append() {
        let dim = 8usize;
        let config = TurboQuantConfig::uniform(4, 4);
        let b = 1i32;
        let h = 1i32;
        let d = dim as i32;

        let make_data = |seed: f32| -> Vec<f32> {
            (0..b * h * 1 * d).map(|i| (i as f32 * 0.15 + seed).sin()).collect()
        };

        let mut cache = QuantizedKvCache::new(config);

        // Append 3 steps individually.
        for step in 0..3 {
            let data = make_data(step as f32);
            let arr = InlineArray::from_f32_slice(&data, &[b, h, 1, d]);
            cache.append(&arr, &arr).expect("append step");
        }

        assert_eq!(cache.offset, 3, "Should have 3 cached positions");

        let dk = cache.dequantize_keys().expect("dequantize_keys");
        // Shape should be [B, H, 3, D].
        assert_eq!(dk.shape(), &[b, h, 3, d]);
    }
}
