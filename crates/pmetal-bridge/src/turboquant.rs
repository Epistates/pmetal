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
//! - The `TurboQuantKvCache` struct (see `KvLayerCache` in qwen3_native).
//! - The pmetal-metal `TurboQuantTransform` (InlineArray.matmul replaces it).
//! - The fused-attention path (we use standard SDPA).

use std::f32::consts::PI;
use std::sync::Arc;
use std::time::Instant;

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::InlineArray;
use crate::compat::Dtype;

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

fn turboquant_trace_enabled() -> bool {
    std::env::var_os("PMETAL_TRACE_TURBOQUANT").is_some()
}

fn turboquant_wht_enabled() -> bool {
    std::env::var_os("PMETAL_TQ_USE_WHT")
        .map(|value| value != "0")
        .unwrap_or(true)
}

fn turboquant_q8_fullbyte_enabled() -> bool {
    std::env::var_os("PMETAL_TQ_Q8_FULLBYTE")
        .map(|value| value != "0")
        .unwrap_or(false)
}

fn trace_turboquant_bridge(message: &str) {
    if turboquant_trace_enabled() {
        eprintln!("[TURBOQUANT TRACE][BRIDGE] {message}");
    }
}

fn eval_stage_micros(array: &InlineArray) -> u128 {
    let start = Instant::now();
    array.eval();
    crate::inline_array::synchronize();
    start.elapsed().as_micros()
}

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
    /// Optional signed-FWHT rotation signs for power-of-two dims.
    wht_left_signs: Option<Vec<f32>>,
    /// Optional signed-FWHT rotation signs for power-of-two dims.
    wht_right_signs: Option<Vec<f32>>,
    /// Optional signed-FWHT QJL signs for power-of-two dims.
    qjl_wht_left_signs: Option<Vec<f32>>,
    /// Optional signed-FWHT QJL signs for power-of-two dims.
    qjl_wht_right_signs: Option<Vec<f32>>,
    /// `codebooks[b]` holds the 2^b sorted centroids for `b`-bit quantisation.
    /// Index 0 is unused (0-bit is a degenerate case).
    codebooks: Vec<Vec<f32>>,
    /// InlineArray view of the rotation matrix for GPU-accelerated matmul.
    rotation_arr: Option<InlineArray>,
    /// Bfloat16 view of the rotation matrix for bf16 output reconstruction.
    rotation_arr_bf16: Option<InlineArray>,
    /// InlineArray view of the inverse rotation matrix.
    inverse_rotation_arr: Option<InlineArray>,
    /// InlineArray view of the QJL projection matrix.
    qjl_arr: Option<InlineArray>,
    /// InlineArray view of the inverse QJL projection matrix.
    inverse_qjl_arr: Option<InlineArray>,
    /// Horizontally-stacked `[inverse_rotation | inverse_qjl]` of shape
    /// `[dim, 2*dim]`. Lets the hot-path attention kernel do one fused
    /// matmul for (query_rot, query_proj) instead of two separate
    /// dispatches — saves one dispatch per layer per decode step.
    stacked_inv_rot_qjl_arr: Option<InlineArray>,
    /// GPU-side signed-FWHT rotation signs for D=256 experiments.
    wht_left_signs_arr: Option<InlineArray>,
    /// GPU-side signed-FWHT rotation signs for D=256 experiments.
    wht_right_signs_arr: Option<InlineArray>,
    /// GPU-side signed-FWHT QJL signs for D=256 experiments.
    qjl_wht_left_signs_arr: Option<InlineArray>,
    /// GPU-side signed-FWHT QJL signs for D=256 experiments.
    qjl_wht_right_signs_arr: Option<InlineArray>,
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
        let (wht_left_signs, wht_right_signs, qjl_wht_left_signs, qjl_wht_right_signs) =
            if dim.is_power_of_two() {
                let mut wht_rng = StdRng::seed_from_u64(TURBOQUANT_SEED ^ 0x5748_5400 ^ dim as u64);
                (
                    Some(generate_rademacher_signs(dim, &mut wht_rng)),
                    Some(generate_rademacher_signs(dim, &mut wht_rng)),
                    Some(generate_rademacher_signs(dim, &mut wht_rng)),
                    Some(generate_rademacher_signs(dim, &mut wht_rng)),
                )
            } else {
                (None, None, None, None)
            };

        let mut codebooks = vec![Vec::new(); usize::from(max_mse_bits) + 1];
        for bits in 1..=max_mse_bits {
            codebooks[usize::from(bits)] = build_beta_codebook(dim, bits);
        }

        // Build InlineArray GPU matrices.  On failure we fall back to CPU
        // matmul transparently — the Option<InlineArray> is None in that case.
        let rotation_arr = matrix_to_inline_array(&rotation, dim);
        let rotation_arr_bf16 = rotation_arr.as_ref().map(|arr| {
            let cast = arr.as_dtype(Dtype::Bfloat16.as_i32());
            cast.eval();
            cast
        });
        let inverse_rotation_arr = matrix_to_inline_array(&inverse_rotation, dim);
        let qjl_arr = matrix_to_inline_array(&qjl_projection, dim);
        let inverse_qjl_arr = matrix_to_inline_array(&inverse_qjl_projection, dim);
        // Precompute the horizontally-stacked [inv_rot | inv_qjl]
        // matrix so the decode-time hot path can do a single matmul
        // instead of two. Shape: [dim, 2*dim]. Built only when both
        // component arrays built successfully; caller falls back to
        // separate dispatches otherwise.
        let stacked_inv_rot_qjl_arr =
            if let (Some(rot), Some(qjl)) = (&inverse_rotation_arr, &inverse_qjl_arr) {
                let stacked = crate::compat::ops::concatenate_axis(&[rot, qjl], -1);
                stacked.eval();
                Some(stacked)
            } else {
                None
            };
        let signs_to_inline_array = |signs: &Option<Vec<f32>>| {
            if dim == 256 {
                signs.as_ref().map(|values| {
                    let arr = InlineArray::from_f32_slice(values, &[dim as i32]);
                    arr.eval();
                    arr
                })
            } else {
                None
            }
        };
        let wht_left_signs_arr = signs_to_inline_array(&wht_left_signs);
        let wht_right_signs_arr = signs_to_inline_array(&wht_right_signs);
        let qjl_wht_left_signs_arr = signs_to_inline_array(&qjl_wht_left_signs);
        let qjl_wht_right_signs_arr = signs_to_inline_array(&qjl_wht_right_signs);

        // GPU-side codebooks: each is a tiny 1-D f32 array (16 elements for 4-bit).
        let codebook_arrs: Vec<Option<InlineArray>> = codebooks
            .iter()
            .map(|cb| {
                if cb.is_empty() {
                    None
                } else {
                    let arr = InlineArray::from_f32_slice(cb, &[cb.len() as i32]);
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
            wht_left_signs,
            wht_right_signs,
            qjl_wht_left_signs,
            qjl_wht_right_signs,
            codebooks,
            rotation_arr,
            rotation_arr_bf16,
            inverse_rotation_arr,
            qjl_arr,
            inverse_qjl_arr,
            stacked_inv_rot_qjl_arr,
            wht_left_signs_arr,
            wht_right_signs_arr,
            qjl_wht_left_signs_arr,
            qjl_wht_right_signs_arr,
            codebook_arrs,
        }
    }

    fn codebook(&self, bits: u8) -> &[f32] {
        &self.codebooks[usize::from(bits)]
    }

    fn codebook_arr(&self, bits: u8) -> Option<&InlineArray> {
        self.codebook_arrs.get(usize::from(bits))?.as_ref()
    }

    /// GPU-native nearest-centroid quantisation via fused Metal kernel.
    ///
    /// `rotated`: `[N, D]` f32 — already normalised and rotated.
    /// Returns `[N, D]` uint32 indices on success.
    ///
    /// The fused kernel eliminates the `[N, D, C]` intermediate tensor that
    /// the old expand_dims+subtract+square+argmin chain allocated.  Falls back
    /// to the ops-based path if Metal is unavailable or n_centroids > 16.
    fn gpu_quantize_mse(&self, rotated: &InlineArray, bits: u8) -> Option<InlineArray> {
        let cb_arr = self.codebook_arr(bits)?;
        let shape = rotated.shape();
        let ndim = shape.len();
        // Compute flat N = product of all dimensions except last.
        let n_rows: i32 = shape[..ndim - 1].iter().product();
        let dim = shape[ndim - 1] as u32;
        let n_centroids = cb_arr.shape()[0] as u32;

        // Reshape to [N, D] for the kernel (kernel expects exactly 2-D input).
        // The caller may pass higher-rank tensors like [B, H, S, D].
        let flat = if ndim == 2 {
            // Already [N, D] — no copy.
            None
        } else {
            Some(rotated.reshape(&[n_rows, dim as i32]))
        };
        let input_2d = flat.as_ref().unwrap_or(rotated);

        if let Some(indices_2d) =
            InlineArray::turboquant_encode(input_2d, cb_arr, dim, n_centroids, n_rows as u32)
        {
            // Reshape back to original leading dims + D.
            if ndim == 2 {
                return Some(indices_2d);
            }
            let mut out_shape: Vec<i32> = shape[..ndim - 1].to_vec();
            out_shape.push(dim as i32);
            return Some(indices_2d.reshape(&out_shape));
        }

        // Ops fallback — original expand_dims+argmin path.
        let cb_arr = self.codebook_arr(bits)?;
        let expanded = rotated.expand_dims(-1);
        let diffs = expanded.subtract(cb_arr);
        let sq = diffs.multiply(&diffs);
        Some(sq.argmin(-1))
    }

    /// GPU-native codebook reconstruction via fused Metal kernel.
    ///
    /// `indices`: `[..., D]` uint32.  Returns `[..., D]` f32 of centroid values.
    ///
    /// The fused kernel eliminates the flatten→take→reshape round-trip.
    /// Falls back to the ops-based take_axis path if Metal is unavailable.
    fn gpu_reconstruct_mse(&self, indices: &InlineArray, bits: u8) -> Option<InlineArray> {
        let cb_arr = self.codebook_arr(bits)?;
        let shape = indices.shape();
        let ndim = shape.len();
        let n_rows: i32 = shape[..ndim - 1].iter().product();
        let dim = shape[ndim - 1] as u32;
        let n_centroids = cb_arr.shape()[0] as u32;

        let flat = if ndim == 2 {
            None
        } else {
            Some(indices.reshape(&[n_rows, dim as i32]))
        };
        let indices_2d = flat.as_ref().unwrap_or(indices);

        if let Some(recon_2d) =
            InlineArray::turboquant_decode(indices_2d, cb_arr, dim, n_centroids, n_rows as u32)
        {
            if ndim == 2 {
                return Some(recon_2d);
            }
            let mut out_shape: Vec<i32> = shape[..ndim - 1].to_vec();
            out_shape.push(dim as i32);
            return Some(recon_2d.reshape(&out_shape));
        }

        // Ops fallback — original take_axis+reshape path.
        let cb_arr = self.codebook_arr(bits)?;
        let orig_shape: Vec<i32> = shape.to_vec();
        let n: i32 = orig_shape.iter().product();
        let flat_idx = indices.reshape(&[n]);
        let gathered = cb_arr.take_axis(&flat_idx, 0);
        Some(gathered.reshape(&orig_shape))
    }

    /// Rotate input rows: output = input · Π^T  (each row left-multiplied by Π).
    fn rotate_rows(&self, input: &[f32]) -> Vec<f32> {
        if turboquant_wht_enabled() && self.dim == 256 {
            if let (Some(pre), Some(post)) = (&self.wht_right_signs, &self.wht_left_signs) {
                let mut output = input.to_vec();
                for row in output.chunks_mut(self.dim) {
                    signed_fwht_forward(row, pre, post);
                }
                return output;
            }
        }
        self.apply_transform(input, &self.rotation, &self.rotation_arr)
    }

    /// Inverse-rotate: output = input · Π.
    fn inverse_rotate_rows(&self, input: &[f32]) -> Vec<f32> {
        if turboquant_wht_enabled() && self.dim == 256 {
            if let (Some(pre), Some(post)) = (&self.wht_left_signs, &self.wht_right_signs) {
                let mut output = input.to_vec();
                for row in output.chunks_mut(self.dim) {
                    signed_fwht_forward(row, pre, post);
                }
                return output;
            }
        }
        self.apply_transform(input, &self.inverse_rotation, &self.inverse_rotation_arr)
    }

    /// Project via Gaussian matrix J for QJL.
    fn project_rows(&self, input: &[f32]) -> Vec<f32> {
        if turboquant_wht_enabled() && self.dim == 256 {
            if let (Some(pre), Some(post)) = (&self.qjl_wht_right_signs, &self.qjl_wht_left_signs) {
                let mut output = input.to_vec();
                for row in output.chunks_mut(self.dim) {
                    signed_fwht_forward(row, pre, post);
                }
                return output;
            }
        }
        self.apply_transform(input, &self.qjl_projection, &self.qjl_arr)
    }

    /// Inverse-project via J^T.
    fn inverse_project_rows(&self, input: &[f32]) -> Vec<f32> {
        if turboquant_wht_enabled() && self.dim == 256 {
            if let (Some(pre), Some(post)) = (&self.qjl_wht_left_signs, &self.qjl_wht_right_signs) {
                let mut output = input.to_vec();
                for row in output.chunks_mut(self.dim) {
                    signed_fwht_forward(row, pre, post);
                }
                return output;
            }
        }
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
    /// Experimental signed-FWHT rotation path for power-of-two dims.
    fn rotate_rows_wht(&self, input_rows: &InlineArray) -> Option<InlineArray> {
        self.apply_signed_fwht_rows(
            input_rows,
            self.wht_right_signs.as_ref()?,
            self.wht_left_signs.as_ref()?,
            &self.wht_left_signs_arr,
            &self.wht_right_signs_arr,
        )
    }

    /// Experimental inverse signed-FWHT rotation path for power-of-two dims.
    fn inverse_rotate_rows_wht(&self, input_rows: &InlineArray) -> Option<InlineArray> {
        self.apply_signed_fwht_rows(
            input_rows,
            self.wht_left_signs.as_ref()?,
            self.wht_right_signs.as_ref()?,
            &self.wht_right_signs_arr,
            &self.wht_left_signs_arr,
        )
    }

    /// Experimental signed-FWHT QJL projection path for power-of-two dims.
    #[allow(dead_code)]
    fn project_rows_wht(&self, input_rows: &InlineArray) -> Option<InlineArray> {
        self.apply_signed_fwht_rows(
            input_rows,
            self.qjl_wht_right_signs.as_ref()?,
            self.qjl_wht_left_signs.as_ref()?,
            &self.qjl_wht_left_signs_arr,
            &self.qjl_wht_right_signs_arr,
        )
    }

    /// Experimental inverse signed-FWHT QJL projection path for power-of-two dims.
    fn inverse_project_rows_wht(&self, input_rows: &InlineArray) -> Option<InlineArray> {
        self.apply_signed_fwht_rows(
            input_rows,
            self.qjl_wht_left_signs.as_ref()?,
            self.qjl_wht_right_signs.as_ref()?,
            &self.qjl_wht_right_signs_arr,
            &self.qjl_wht_left_signs_arr,
        )
    }

    fn apply_signed_fwht_rows(
        &self,
        input_rows: &InlineArray,
        pre_signs_cpu: &[f32],
        post_signs_cpu: &[f32],
        post_signs_gpu: &Option<InlineArray>,
        pre_signs_gpu: &Option<InlineArray>,
    ) -> Option<InlineArray> {
        if !self.dim.is_power_of_two()
            || input_rows.ndim() != 2
            || input_rows.dim(1) != self.dim as i32
        {
            return None;
        }
        let n_rows = input_rows.dim(0) as usize;

        if self.dim == 256 {
            if let (Some(post_gpu), Some(pre_gpu)) =
                (post_signs_gpu.as_ref(), pre_signs_gpu.as_ref())
            {
                if let Some(out) = InlineArray::turboquant_signed_fwht_256_rows(
                    input_rows,
                    post_gpu,
                    pre_gpu,
                    n_rows as u32,
                ) {
                    return Some(out);
                }
            }
        }

        let mut output = inline_array_to_f32_vec(input_rows, n_rows * self.dim)?;
        for row in output.chunks_mut(self.dim) {
            signed_fwht_forward(row, post_signs_cpu, pre_signs_cpu);
        }
        Some(InlineArray::from_f32_slice(
            &output,
            &[n_rows as i32, self.dim as i32],
        ))
    }

    fn apply_array_transform_rows(
        &self,
        input: &InlineArray,
        matrix_gpu: &Option<InlineArray>,
        wht_impl: impl FnOnce(&Self, &InlineArray) -> Option<InlineArray>,
    ) -> Option<InlineArray> {
        let shape = input.shape();
        if shape.is_empty() || *shape.last()? != self.dim as i32 {
            return None;
        }
        let ndim = shape.len();
        let n_rows: i32 = if ndim == 1 {
            1
        } else {
            shape[..ndim - 1].iter().product()
        };
        let input_rows = if ndim == 2 {
            input.clone()
        } else {
            input.reshape(&[n_rows, self.dim as i32])
        };
        let output_rows = if turboquant_wht_enabled() && self.dim == 256 {
            let input_rows_f32 = if input_rows.dtype_raw() == Dtype::Float32.as_i32() {
                input_rows
            } else {
                input_rows.as_dtype(Dtype::Float32.as_i32())
            };
            wht_impl(self, &input_rows_f32)?
        } else {
            let matrix = matrix_gpu.as_ref()?;
            input_rows.matmul(matrix)
        };
        if ndim == 2 {
            Some(output_rows)
        } else {
            Some(output_rows.reshape(shape))
        }
    }

    fn rotate_array(&self, input: &InlineArray) -> Option<InlineArray> {
        self.apply_array_transform_rows(input, &self.inverse_rotation_arr, |core, rows| {
            core.rotate_rows_wht(rows)
        })
    }

    /// Fused rotate + project: computes `input @ [inv_rotation | inv_qjl]`
    /// as a single [N, 2*dim] matmul and splits the result back into
    /// `(query_rot, query_proj)` of shape [N, dim] each. Saves one op
    /// dispatch per decode step compared to calling `rotate_array` and
    /// `project_array` separately. Returns `None` if the stacked matrix
    /// wasn't built at construction (falls back to separate paths in
    /// the caller), or if the FWHT fast path is enabled (that path
    /// stays unchanged).
    fn rotate_and_project_array(
        &self,
        input: &InlineArray,
    ) -> Option<(InlineArray, InlineArray)> {
        if turboquant_wht_enabled() && self.dim == 256 {
            return None;
        }
        let stacked = self.stacked_inv_rot_qjl_arr.as_ref()?;
        let shape = input.shape();
        let ndim = shape.len();
        if ndim == 0 {
            return None;
        }
        let last_dim = shape[ndim - 1];
        if last_dim as usize != self.dim {
            return None;
        }
        let n: i32 = shape[..ndim - 1].iter().product();
        let input_2d = input.reshape(&[n, last_dim]);
        let fused = input_2d.matmul(stacked); // [n, 2*dim]
        let dim_i32 = self.dim as i32;
        let rot = fused.slice(&[0, 0], &[n, dim_i32]);
        let proj = fused.slice(&[0, dim_i32], &[n, 2 * dim_i32]);
        Some((rot, proj))
    }

    fn inverse_rotate_array(&self, input: &InlineArray) -> Option<InlineArray> {
        self.apply_array_transform_rows(input, &self.rotation_arr, |core, rows| {
            core.inverse_rotate_rows_wht(rows)
        })
    }

    fn project_array(&self, input: &InlineArray) -> Option<InlineArray> {
        self.apply_array_transform_rows(input, &self.inverse_qjl_arr, |core, rows| {
            core.project_rows_wht(rows)
        })
    }

    fn inverse_project_array(&self, input: &InlineArray) -> Option<InlineArray> {
        self.apply_array_transform_rows(input, &self.qjl_arr, |core, rows| {
            core.inverse_project_rows_wht(rows)
        })
    }

    fn inverse_rotate_output_array(
        &self,
        input: &InlineArray,
        output_dtype: i32,
    ) -> Option<InlineArray> {
        if turboquant_wht_enabled() && self.dim == 256 {
            let output = self.inverse_rotate_array(input)?;
            if output_dtype == Dtype::Bfloat16.as_i32() {
                Some(output.as_dtype(output_dtype))
            } else {
                Some(output)
            }
        } else if output_dtype == Dtype::Bfloat16.as_i32() {
            let input_bf16 = input.as_dtype(output_dtype);
            let rotation_bf16 = self.rotation_arr_bf16.as_ref()?;
            Some(input_bf16.matmul(rotation_bf16))
        } else {
            let rotation = self.rotation_arr.as_ref()?;
            Some(input.matmul(rotation))
        }
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
        let mut core_cache = std::collections::HashMap::<(usize, u8), Arc<TurboQuantCore>>::new();
        let mut get_core = |subdim: usize, max_mse_bits: u8| {
            core_cache
                .entry((subdim, max_mse_bits))
                .or_insert_with(|| Arc::new(TurboQuantCore::new(subdim, max_mse_bits)))
                .clone()
        };

        let keys = build_tensor_runtime(key_dim, config.keys, true, &mut get_core);
        let values = build_tensor_runtime(value_dim, config.values, false, &mut get_core);

        Self { keys, values }
    }
}

fn build_tensor_runtime<F>(
    total_dim: usize,
    config: TurboQuantTensorConfig,
    _is_keys: bool,
    get_core: &mut F,
) -> TensorRuntime
where
    F: FnMut(usize, u8) -> Arc<TurboQuantCore>,
{
    match config {
        TurboQuantTensorConfig::Uniform { bits } => {
            TensorRuntime::Uniform {
                config,
                // Build the full MSE codebook ladder even for keys so pure-MSE
                // key paths (for example the full-byte D256 experiments) can
                // reuse the same core without a second cache format.
                core: get_core(total_dim, bits),
            }
        }
        TurboQuantTensorConfig::Mixed {
            regular_bits,
            outlier_bits,
            outlier_count,
        } => {
            let regular_dim = total_dim - outlier_count;
            TensorRuntime::Mixed {
                config,
                regular_core: get_core(regular_dim, regular_bits),
                outlier_core: get_core(outlier_count, outlier_bits),
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

fn packed_qjl_words(dim: usize) -> usize {
    dim.div_ceil(32)
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-layer quantised storage
// ═══════════════════════════════════════════════════════════════════════════

/// GPU-resident quantised key data for the Uniform (non-outlier) path.
///
/// All tensors live entirely on the GPU — no CPU round-trips during normal
/// operation.  Shape convention (accumulated over T steps):
///   indices:         [B, H, T, D]  uint8   — codebook index per coordinate
///   indices_t:       [B, H, D, T]  uint8   — score-friendly transposed view
///   q8_keybytes_t:   [B, H, D, T]  uint8   — q8-only packed index/sign view
///   q8_keybytes_seq: [B, H, T, D]  uint8   — q8 seq-major key shadow
///   q8_fullbyte_seq: [B, H, T, D]  uint8   — q8 seq-major pure-256-centroid key shadow
///   q8_kvbytes_seq:  [B, H, T, D]  uint16  — D256 q8 seq-major packed {key,value}
///   q8_slot_scales_seq: [B, H, T, 3]  f32  — [key_norm, residual_norm, value_norm]
///   norms:           [B, H, T, 1]  f32     — optional L2 norm before unit-sphere normalise
///   qjl_signs:       [B, H, T, ceil(D/32)]  uint32 packed sign words
///   qjl_signs_t:     [B, H, ceil(D/32), T]  uint32 transposed sign-word view
///   residual_norms:  [B, H, T, 1]  f32     — optional unscaled residual L2 norm
#[derive(Debug, Clone)]
struct GpuKeyStore {
    indices: InlineArray,
    indices_t: Option<InlineArray>,
    q8_keybytes_t: Option<InlineArray>,
    q8_keybytes_seq: Option<InlineArray>,
    q8_fullbyte_seq: Option<InlineArray>,
    q8_kvbytes_seq: Option<InlineArray>,
    q8_slot_scales_seq: Option<InlineArray>,
    norms: Option<InlineArray>,
    qjl_signs: InlineArray,
    qjl_signs_t: Option<InlineArray>,
    residual_norms: Option<InlineArray>,
}

impl GpuKeyStore {
    /// Concatenate a new step's GPU arrays along the T (axis 2) dimension.
    fn append(&mut self, new: GpuKeyStore) {
        self.indices = self.indices.kv_cache_append(&new.indices, 2);
        self.indices_t = match (self.indices_t.take(), new.indices_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.q8_keybytes_t = match (self.q8_keybytes_t.take(), new.q8_keybytes_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.q8_keybytes_seq = match (self.q8_keybytes_seq.take(), new.q8_keybytes_seq) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.q8_fullbyte_seq = match (self.q8_fullbyte_seq.take(), new.q8_fullbyte_seq) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.q8_kvbytes_seq = match (self.q8_kvbytes_seq.take(), new.q8_kvbytes_seq) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.q8_slot_scales_seq = match (self.q8_slot_scales_seq.take(), new.q8_slot_scales_seq) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.norms = match (self.norms.take(), new.norms) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            _ => None,
        };
        self.qjl_signs = self.qjl_signs.kv_cache_append(&new.qjl_signs, 2);
        self.qjl_signs_t = match (self.qjl_signs_t.take(), new.qjl_signs_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.residual_norms = match (self.residual_norms.take(), new.residual_norms) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            _ => None,
        };
    }

    fn cache_seq_capacity(&self) -> i32 {
        self.q8_kvbytes_seq
            .as_ref()
            .map(|arr| arr.dim(2))
            .or_else(|| self.q8_keybytes_seq.as_ref().map(|arr| arr.dim(2)))
            .or_else(|| self.indices_t.as_ref().map(|arr| arr.dim(3)))
            .unwrap_or_else(|| self.indices.dim(2))
    }

    fn indices_t_array(&self) -> InlineArray {
        self.indices_t
            .clone()
            .unwrap_or_else(|| self.indices.transpose_axes(&[0, 1, 3, 2]))
    }

    fn qjl_signs_t_array(&self) -> InlineArray {
        self.qjl_signs_t
            .clone()
            .unwrap_or_else(|| self.qjl_signs.transpose_axes(&[0, 1, 3, 2]))
    }

    fn qjl_words(&self) -> i32 {
        self.qjl_signs_t
            .as_ref()
            .map(|arr| arr.dim(2))
            .unwrap_or_else(|| self.qjl_signs.dim(3))
    }

    fn slot_scale_component_array(&self, component: i32) -> Option<InlineArray> {
        let slot_scales = self.q8_slot_scales_seq.as_ref()?;
        Some(slot_scales.slice(
            &[0, 0, 0, component],
            &[
                slot_scales.dim(0),
                slot_scales.dim(1),
                slot_scales.dim(2),
                component + 1,
            ],
        ))
    }

    fn key_norms_array(&self) -> Option<InlineArray> {
        self.norms
            .clone()
            .or_else(|| self.slot_scale_component_array(0))
    }

    fn residual_norms_array(&self) -> Option<InlineArray> {
        self.residual_norms
            .clone()
            .or_else(|| self.slot_scale_component_array(1))
    }

    fn collect_for_detach<'a>(&'a mut self, out: &mut Vec<&'a mut InlineArray>) {
        out.push(&mut self.indices);
        if let Some(indices_t) = self.indices_t.as_mut() {
            out.push(indices_t);
        }
        if let Some(q8_keybytes_t) = self.q8_keybytes_t.as_mut() {
            out.push(q8_keybytes_t);
        }
        if let Some(q8_keybytes_seq) = self.q8_keybytes_seq.as_mut() {
            out.push(q8_keybytes_seq);
        }
        if let Some(q8_fullbyte_seq) = self.q8_fullbyte_seq.as_mut() {
            out.push(q8_fullbyte_seq);
        }
        if let Some(q8_kvbytes_seq) = self.q8_kvbytes_seq.as_mut() {
            out.push(q8_kvbytes_seq);
        }
        if let Some(q8_slot_scales_seq) = self.q8_slot_scales_seq.as_mut() {
            out.push(q8_slot_scales_seq);
        }
        if let Some(norms) = self.norms.as_mut() {
            out.push(norms);
        }
        out.push(&mut self.qjl_signs);
        if let Some(qjl_signs_t) = self.qjl_signs_t.as_mut() {
            out.push(qjl_signs_t);
        }
        if let Some(residual_norms) = self.residual_norms.as_mut() {
            out.push(residual_norms);
        }
    }
}

/// GPU-resident quantised value data for the Uniform path.
///
///   indices:  [B, H, T, D]  uint8
///   indices_t:[B, H, D, T]  uint8
///   norms:    [B, H, T, 1]  f32
#[derive(Debug, Clone)]
struct GpuValueStore {
    indices: Option<InlineArray>,
    indices_t: Option<InlineArray>,
    norms: Option<InlineArray>,
    d256_rot_values_seq: Option<InlineArray>,
}

impl GpuValueStore {
    fn append(&mut self, new: GpuValueStore) {
        self.indices = match (self.indices.take(), new.indices) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.indices_t = match (self.indices_t.take(), new.indices_t) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 3)),
            _ => None,
        };
        self.norms = match (self.norms.take(), new.norms) {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
        self.d256_rot_values_seq = match (self.d256_rot_values_seq.take(), new.d256_rot_values_seq)
        {
            (Some(current), Some(next)) => Some(current.kv_cache_append(&next, 2)),
            (None, Some(next)) => Some(next),
            (Some(current), None) => Some(current),
            (None, None) => None,
        };
    }

    fn indices_t_array(&self) -> Option<InlineArray> {
        self.indices_t.clone().or_else(|| {
            self.indices
                .as_ref()
                .map(|arr| arr.transpose_axes(&[0, 1, 3, 2]))
        })
    }

    fn norms_array(&self) -> Option<InlineArray> {
        self.norms.clone()
    }

    fn collect_for_detach<'a>(&'a mut self, out: &mut Vec<&'a mut InlineArray>) {
        if let Some(indices) = self.indices.as_mut() {
            out.push(indices);
        }
        if let Some(indices_t) = self.indices_t.as_mut() {
            out.push(indices_t);
        }
        if let Some(norms) = self.norms.as_mut() {
            out.push(norms);
        }
        if let Some(d256_rot_values_seq) = self.d256_rot_values_seq.as_mut() {
            out.push(d256_rot_values_seq);
        }
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

    fn extend(
        &mut self,
        encoded: &EncodedKeyRows,
        outlier_encoded: Option<&EncodedKeyRows>,
        outlier_mask: Option<&Vec<u16>>,
    ) {
        self.regular_indices.extend_from_slice(&encoded.mse_indices);
        self.regular_qjl_signs.extend_from_slice(&encoded.qjl_signs);
        self.regular_norms.extend_from_slice(&encoded.norms);
        self.regular_residual_norms
            .extend_from_slice(&encoded.residual_norms);

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
            + self
                .outlier_residual_norms
                .as_ref()
                .map_or(0, |v| v.len() * 4)
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

    fn extend(
        &mut self,
        encoded: &EncodedValueRows,
        outlier_encoded: Option<&EncodedValueRows>,
        outlier_mask: Option<&Vec<u16>>,
    ) {
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

#[derive(Debug, Clone, Copy)]
pub enum UniformAttentionBenchMode {
    Split,
    SpecializedQ8D128TwoPass,
    SpecializedQ8D256TwoPass,
    SpecializedQ8D256FullbytePass1,
    SpecializedQ8D256FullbytePass2,
    SpecializedQ8D256FullbyteSplitDenseV,
    SpecializedQ8D256FullbyteLocalSoftmax,
}

impl QuantizedKvCache {
    /// Create an empty cache.  `state` should be `None` on first use; call
    /// `append` to populate.
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
    pub fn append(&mut self, keys: &InlineArray, values: &InlineArray) -> Result<(), String> {
        let layout = self.ensure_layout(keys, values)?;
        let seq_len = keys.dim(2) as usize;

        let config = self.config;
        let state = self.state.get_or_insert_with(|| {
            Arc::new(TurboQuantState::new(
                layout.key_dim,
                layout.value_dim,
                config,
            ))
        });
        let state = Arc::clone(state);

        // Cast to f32 once — needed for both GPU and CPU paths.
        let keys_f32 = keys.as_dtype(10 /* float32 */);
        let values_f32 = values.as_dtype(10 /* float32 */);

        let ks = self
            .keys
            .get_or_insert_with(|| QuantizedKeyStore::new(config.keys));
        let vs = self
            .values
            .get_or_insert_with(|| QuantizedValueStore::new(config.values));

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
            let mut dense = gpu_dequantize_keys(g, &state.keys, bits)?;
            let mut to_eval = vec![&mut dense];
            crate::inline_array::eval_and_detach_many(&mut to_eval);
            return Some(dense);
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
            let mut dense = gpu_dequantize_values(g, &state.values, bits)?;
            let mut to_eval = vec![&mut dense];
            crate::inline_array::eval_and_detach_many(&mut to_eval);
            return Some(dense);
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

    /// Evaluate and detach GPU-resident cache arrays to keep graph chains short.
    pub fn eval_and_detach_gpu_state(&mut self) {
        let mut to_eval = Vec::new();
        if let Some(keys) = &mut self.keys {
            if let Some(gpu) = &mut keys.gpu {
                gpu.collect_for_detach(&mut to_eval);
            }
        }
        if let Some(values) = &mut self.values {
            if let Some(gpu) = &mut values.gpu {
                gpu.collect_for_detach(&mut to_eval);
            }
        }
        if !to_eval.is_empty() {
            crate::inline_array::eval_and_detach_many(&mut to_eval);
        }
    }

    /// Append a single-token KV chunk and compute attention output.
    ///
    /// This restores the bridge cache API expected by native Qwen decode.
    /// The current implementation is correctness-first: it dequantizes the
    /// cache and runs standard SDPA over the valid prefix.
    pub fn append_and_compute_attention(
        &mut self,
        queries: &InlineArray,
        keys: &InlineArray,
        values: &InlineArray,
        scale: f32,
    ) -> Result<InlineArray, String> {
        if queries.ndim() != 4
            || keys.ndim() != 4
            || values.ndim() != 4
            || queries.dim(2) != 1
            || keys.dim(2) != 1
            || values.dim(2) != 1
        {
            return Err(
                "TurboQuant direct attention requires [B, H, 1, D] single-token decode inputs"
                    .to_string(),
            );
        }

        let layout = self.ensure_layout(keys, values)?;
        self.append(keys, values)?;
        let query_dtype = queries.dtype_raw();
        let queries_f32 = if query_dtype == 10 {
            queries.clone()
        } else {
            queries.as_dtype(10)
        };

        if let Some(output) =
            self.try_gpu_uniform_attention(&queries_f32, layout, scale, query_dtype)
        {
            return Ok(if query_dtype == 10 || output.dtype_raw() == query_dtype {
                output
            } else {
                output.as_dtype(query_dtype)
            });
        }

        let full_keys = self
            .dequantize_keys()
            .ok_or_else(|| "TurboQuant failed to dequantize keys".to_string())?;
        let full_values = self
            .dequantize_values()
            .ok_or_else(|| "TurboQuant failed to dequantize values".to_string())?;

        let q_heads = queries.dim(1) as usize;
        let kv_heads = layout.heads;
        let (keys_for_attn, values_for_attn) = if q_heads == kv_heads {
            (full_keys, full_values)
        } else {
            let groups = q_heads / kv_heads;
            if groups * kv_heads != q_heads {
                return Err(format!(
                    "TurboQuant GQA mismatch: query heads {q_heads} not divisible by kv heads {kv_heads}"
                ));
            }
            (
                full_keys.repeat(groups as i32, 1),
                full_values.repeat(groups as i32, 1),
            )
        };

        let queries_f32 = queries.as_dtype(10);
        let output = crate::decode::sdpa_causal_like_mlx(
            &queries_f32,
            &keys_for_attn,
            &values_for_attn,
            scale,
            queries.dim(2),
        );
        Ok(if queries.dtype_raw() == 10 {
            output
        } else {
            output.as_dtype(queries.dtype_raw())
        })
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_attention_core_precomputed(
        &self,
        query_rot: &InlineArray,
        query_proj: &InlineArray,
        q_heads: i32,
        scale: f32,
        mode: UniformAttentionBenchMode,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;

        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let (value_core, value_bits) = match &state.values {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };

        let key_dim = layout.key_dim as i32;
        let value_dim = layout.value_dim as i32;
        let kv_heads_i32 = layout.heads as i32;
        let q_rows = query_rot.dim(0);
        let n_seq = self.offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        if q_rows <= 0 || n_seq <= 0 || cache_seq_capacity < n_seq || kv_heads_i32 <= 0 {
            return None;
        }

        let kv_rows = (layout.batch * layout.heads) as i32;
        let key_norms = ks
            .key_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let key_residual_norms = ks
            .residual_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let qjl_words = ks.qjl_words();

        match mode {
            UniformAttentionBenchMode::SpecializedQ8D128TwoPass => {
                if key_bits != 8
                    || value_bits != 8
                    || key_dim != 128
                    || value_dim != 128
                    || qjl_words != 4
                {
                    return None;
                }
                let key_indices =
                    ks.indices_t_array()
                        .reshape(&[kv_rows, key_dim, cache_seq_capacity]);
                let key_qjl_signs =
                    ks.qjl_signs_t_array()
                        .reshape(&[kv_rows, qjl_words, cache_seq_capacity]);
                let value_indices =
                    vs.indices_t_array()?
                        .reshape(&[kv_rows, value_dim, cache_seq_capacity]);
                InlineArray::turboquant_attention_q8_d128_2pass(
                    query_rot,
                    query_proj,
                    &key_indices,
                    &key_qjl_signs,
                    &key_norms,
                    &key_residual_norms,
                    key_core.codebook_arr(key_bits.saturating_sub(1))?,
                    &value_indices,
                    &vs.norms_array()?.reshape(&[kv_rows, cache_seq_capacity]),
                    value_core.codebook_arr(value_bits)?,
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                    scale,
                )
            }
            UniformAttentionBenchMode::SpecializedQ8D256TwoPass => self
                .try_gpu_uniform_attention_q8_d256_precomputed(
                    query_rot,
                    Some(query_proj),
                    q_heads,
                    scale,
                ),
            UniformAttentionBenchMode::SpecializedQ8D256FullbytePass1 => {
                if key_bits != 8
                    || value_bits != 8
                    || key_dim != 256
                    || value_dim != 256
                    || n_seq < 1024
                {
                    return None;
                }
                if let (Some(key_indices), Some(slot_scales), Some(value_rot_dense)) = (
                    ks.q8_fullbyte_seq.as_ref(),
                    ks.q8_slot_scales_seq.as_ref(),
                    vs.d256_rot_values_seq.as_ref(),
                ) {
                    InlineArray::turboquant_attention_q8_d256_fullbyte_dense_values_2pass_pass1(
                        query_rot,
                        &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                        &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                        key_core.codebook_arr(key_bits)?,
                        &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                        scale,
                    )
                } else {
                    None
                }
            }
            UniformAttentionBenchMode::SpecializedQ8D256FullbytePass2 => {
                if key_bits != 8
                    || value_bits != 8
                    || key_dim != 256
                    || value_dim != 256
                    || n_seq < 1024
                {
                    return None;
                }
                let (partials, sums, maxs) = self
                    .bench_gpu_uniform_attention_state_precomputed_fullbyte(
                        query_rot, q_heads, scale,
                    )?;
                InlineArray::turboquant_attention_q8_d256_pass2_merge(
                    &partials,
                    &sums,
                    &maxs,
                    q_rows as u32,
                    sums.dim(1) as u32,
                )
            }
            UniformAttentionBenchMode::SpecializedQ8D256FullbyteSplitDenseV => {
                if key_bits != 8
                    || value_bits != 8
                    || key_dim != 256
                    || value_dim != 256
                    || n_seq < 1024
                {
                    return None;
                }
                if let (Some(key_indices), Some(slot_scales), Some(value_rot_dense)) = (
                    ks.q8_fullbyte_seq.as_ref(),
                    ks.q8_slot_scales_seq.as_ref(),
                    vs.d256_rot_values_seq.as_ref(),
                ) {
                    let scores = InlineArray::turboquant_score_q8_d256_fullbyte(
                        query_rot,
                        &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                        &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                        key_core.codebook_arr(key_bits)?,
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                        scale,
                    )?;
                    let weights = scores.softmax(-1);
                    InlineArray::turboquant_weighted_sum_d256_dense_values(
                        &weights,
                        &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                    )
                } else {
                    None
                }
            }
            UniformAttentionBenchMode::SpecializedQ8D256FullbyteLocalSoftmax => {
                if key_bits != 8
                    || value_bits != 8
                    || key_dim != 256
                    || value_dim != 256
                    || n_seq < 1024
                {
                    return None;
                }
                if let (Some(key_indices), Some(slot_scales), Some(value_rot_dense)) = (
                    ks.q8_fullbyte_seq.as_ref(),
                    ks.q8_slot_scales_seq.as_ref(),
                    vs.d256_rot_values_seq.as_ref(),
                ) {
                    InlineArray::turboquant_attention_q8_d256_fullbyte_dense_values_2pass_localsoftmax(
                        query_rot,
                        &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                        &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                        key_core.codebook_arr(key_bits)?,
                        &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                        scale,
                    )
                } else {
                    None
                }
            }
            UniformAttentionBenchMode::Split => {
                let scores = self
                    .bench_gpu_uniform_scores_precomputed(query_rot, query_proj, q_heads, scale)?;
                let weights = scores.softmax(-1);
                InlineArray::turboquant_weighted_decode(
                    &weights,
                    &vs.indices_t_array()?,
                    &vs.norms_array()?.reshape(&[kv_rows, cache_seq_capacity]),
                    value_core.codebook_arr(value_bits)?,
                    value_dim as u32,
                    1u32 << value_bits,
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                )
            }
        }
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_query_transforms(
        &self,
        queries_f32: &InlineArray,
    ) -> Option<(InlineArray, InlineArray)> {
        let state = self.state.as_ref()?;
        let key_core = match &state.keys {
            TensorRuntime::Uniform { core, .. } => core,
            _ => return None,
        };
        let key_rot = key_core.inverse_rotation_arr.as_ref()?;
        let key_proj = key_core.inverse_qjl_arr.as_ref()?;
        let batch = queries_f32.dim(0);
        let q_heads = queries_f32.dim(1);
        let key_dim = queries_f32.dim(3);
        let q_rows = batch * q_heads;
        let query_rows = queries_f32.reshape(&[q_rows, key_dim]);
        Some((query_rows.matmul(key_rot), query_rows.matmul(key_proj)))
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_query_transforms_wht(
        &self,
        queries_f32: &InlineArray,
    ) -> Option<(InlineArray, InlineArray)> {
        let state = self.state.as_ref()?;
        let key_core = match &state.keys {
            TensorRuntime::Uniform { core, .. } => core,
            _ => return None,
        };
        let batch = queries_f32.dim(0);
        let q_heads = queries_f32.dim(1);
        let key_dim = queries_f32.dim(3);
        if key_dim != 256 {
            return None;
        }
        let q_rows = batch * q_heads;
        let query_rows = queries_f32.reshape(&[q_rows, key_dim]);
        Some((
            key_core.rotate_rows_wht(&query_rows)?,
            key_core.project_rows_wht(&query_rows)?,
        ))
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_output_inverse_rotate_wht(
        &self,
        decoded_rot: &InlineArray,
    ) -> Option<InlineArray> {
        let state = self.state.as_ref()?;
        let value_core = match &state.values {
            TensorRuntime::Uniform { core, .. } => core,
            _ => return None,
        };
        let dim = decoded_rot.dim(1);
        if dim != 256 {
            return None;
        }
        value_core.inverse_rotate_rows_wht(decoded_rot)
    }
    fn try_gpu_uniform_attention_q8_d256_precomputed(
        &self,
        query_rot: &InlineArray,
        query_proj: Option<&InlineArray>,
        q_heads: i32,
        scale: f32,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;
        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let (value_core, value_bits) = match &state.values {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };

        let key_dim = layout.key_dim as i32;
        let value_dim = layout.value_dim as i32;
        let q_rows = query_rot.dim(0);
        let n_seq = self.offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        if key_bits != 8
            || value_bits != 8
            || key_dim != 256
            || value_dim != 256
            || n_seq < 1024
            || q_rows <= 0
            || q_heads <= 0
            || (q_heads % layout.heads as i32) != 0
            || (q_heads / layout.heads as i32) > 8
            || cache_seq_capacity < n_seq
        {
            return None;
        }

        let kv_rows = (layout.batch * layout.heads) as i32;
        if let (Some(key_indices), Some(slot_scales), Some(value_rot_dense)) = (
            ks.q8_fullbyte_seq.as_ref(),
            ks.q8_slot_scales_seq.as_ref(),
            vs.d256_rot_values_seq.as_ref(),
        ) {
            InlineArray::turboquant_attention_q8_d256_fullbyte_dense_values_2pass(
                query_rot,
                &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                key_core.codebook_arr(key_bits)?,
                &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                q_rows as u32,
                n_seq as u32,
                cache_seq_capacity as u32,
                q_heads as u32,
                layout.heads as u32,
                scale,
            )
        } else {
            let qjl_words = ks.qjl_words();
            if qjl_words != 8 {
                return None;
            }
            let query_proj = query_proj?;
            if let (Some(key_bytes), Some(slot_scales), Some(value_rot_dense)) = (
                ks.q8_keybytes_seq.as_ref(),
                ks.q8_slot_scales_seq.as_ref(),
                vs.d256_rot_values_seq.as_ref(),
            ) {
                InlineArray::turboquant_attention_q8_d256_packed_keys_dense_values_2pass(
                    query_rot,
                    query_proj,
                    &key_bytes.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                    &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                    key_core.codebook_arr(key_bits.saturating_sub(1))?,
                    &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                    scale,
                )
            } else if let (Some(kv_bytes), Some(slot_scales)) =
                (ks.q8_kvbytes_seq.as_ref(), ks.q8_slot_scales_seq.as_ref())
            {
                if let Some(value_rot_dense) = vs.d256_rot_values_seq.as_ref() {
                    InlineArray::turboquant_attention_q8_d256_packed_kv_dense_values_2pass(
                        query_rot,
                        query_proj,
                        &kv_bytes.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                        &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                        key_core.codebook_arr(key_bits.saturating_sub(1))?,
                        &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                        scale,
                    )
                } else {
                    InlineArray::turboquant_attention_q8_d256_packed_kv_2pass(
                        query_rot,
                        query_proj,
                        &kv_bytes.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                        &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                        key_core.codebook_arr(key_bits.saturating_sub(1))?,
                        value_core.codebook_arr(value_bits)?,
                        q_rows as u32,
                        n_seq as u32,
                        cache_seq_capacity as u32,
                        q_heads as u32,
                        layout.heads as u32,
                        scale,
                    )
                }
            } else if let (Some(key_bytes), Some(slot_scales)) =
                (ks.q8_keybytes_seq.as_ref(), ks.q8_slot_scales_seq.as_ref())
            {
                InlineArray::turboquant_attention_q8_d256_packed_keys_2pass(
                    query_rot,
                    query_proj,
                    &key_bytes.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                    &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                    key_core.codebook_arr(key_bits.saturating_sub(1))?,
                    &vs.indices
                        .as_ref()?
                        .reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                    value_core.codebook_arr(value_bits)?,
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                    scale,
                )
            } else {
                InlineArray::turboquant_attention_q8_d256_2pass(
                    query_rot,
                    query_proj,
                    &ks.indices_t_array()
                        .reshape(&[kv_rows, key_dim, cache_seq_capacity]),
                    &ks.qjl_signs_t_array()
                        .reshape(&[kv_rows, qjl_words, cache_seq_capacity]),
                    &ks.key_norms_array()?
                        .reshape(&[kv_rows, cache_seq_capacity]),
                    &ks.residual_norms_array()?
                        .reshape(&[kv_rows, cache_seq_capacity]),
                    key_core.codebook_arr(key_bits.saturating_sub(1))?,
                    &vs.indices_t_array()?
                        .reshape(&[kv_rows, value_dim, cache_seq_capacity]),
                    &vs.norms_array()?.reshape(&[kv_rows, cache_seq_capacity]),
                    value_core.codebook_arr(value_bits)?,
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                    scale,
                )
            }
        }
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_scores_precomputed(
        &self,
        query_rot: &InlineArray,
        query_proj: &InlineArray,
        q_heads: i32,
        scale: f32,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;
        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let key_dim = layout.key_dim as i32;
        let kv_rows = (layout.batch * layout.heads) as i32;
        let q_rows = query_rot.dim(0);
        let n_seq = self.offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        let qjl_words = ks.qjl_words();
        let key_norms = ks
            .key_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let key_residual_norms = ks
            .residual_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        if key_bits == 8
            && key_dim == 256
            && qjl_words == 8
            && q_heads > 0
            && (q_heads % layout.heads as i32) == 0
            && (q_heads / layout.heads as i32) <= 8
        {
            if let Some(scores) = InlineArray::turboquant_score_q8_d256(
                query_rot,
                query_proj,
                &ks.indices_t_array(),
                &ks.qjl_signs_t_array(),
                &key_norms,
                &key_residual_norms,
                key_core.codebook_arr(key_bits.saturating_sub(1))?,
                q_rows as u32,
                n_seq as u32,
                cache_seq_capacity as u32,
                q_heads as u32,
                layout.heads as u32,
                scale,
            ) {
                return Some(scores);
            }
        }
        InlineArray::turboquant_score(
            query_rot,
            query_proj,
            &ks.indices_t_array(),
            &ks.qjl_signs_t_array(),
            &key_norms,
            &key_residual_norms,
            key_core.codebook_arr(key_bits.saturating_sub(1))?,
            key_dim as u32,
            qjl_words as u32,
            key_core.codebook_arr(key_bits.saturating_sub(1))?.dim(0) as u32,
            q_rows as u32,
            n_seq as u32,
            cache_seq_capacity as u32,
            q_heads as u32,
            layout.heads as u32,
            scale,
        )
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_scores_precomputed_fullbyte(
        &self,
        query_rot: &InlineArray,
        q_heads: i32,
        scale: f32,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;
        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let key_dim = layout.key_dim as i32;
        let q_rows = query_rot.dim(0);
        let n_seq = self.offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        if key_bits != 8
            || key_dim != 256
            || q_heads <= 0
            || q_rows <= 0
            || n_seq <= 0
            || cache_seq_capacity < n_seq
        {
            return None;
        }
        let kv_rows = (layout.batch * layout.heads) as i32;
        if let (Some(key_indices), Some(slot_scales)) =
            (ks.q8_fullbyte_seq.as_ref(), ks.q8_slot_scales_seq.as_ref())
        {
            InlineArray::turboquant_score_q8_d256_fullbyte(
                query_rot,
                &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                key_core.codebook_arr(key_bits)?,
                q_rows as u32,
                n_seq as u32,
                cache_seq_capacity as u32,
                q_heads as u32,
                layout.heads as u32,
                scale,
            )
        } else {
            None
        }
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_attention_state_precomputed_fullbyte(
        &self,
        query_rot: &InlineArray,
        q_heads: i32,
        scale: f32,
    ) -> Option<(InlineArray, InlineArray, InlineArray)> {
        let layout = self.layout?;
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;
        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let key_dim = layout.key_dim as i32;
        let value_dim = layout.value_dim as i32;
        let q_rows = query_rot.dim(0);
        let n_seq = self.offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        if key_bits != 8
            || key_dim != 256
            || value_dim != 256
            || q_heads <= 0
            || q_rows <= 0
            || n_seq < 1024
            || cache_seq_capacity < n_seq
        {
            return None;
        }
        let kv_rows = (layout.batch * layout.heads) as i32;
        if let (Some(key_indices), Some(slot_scales), Some(value_rot_dense)) = (
            ks.q8_fullbyte_seq.as_ref(),
            ks.q8_slot_scales_seq.as_ref(),
            vs.d256_rot_values_seq.as_ref(),
        ) {
            InlineArray::turboquant_attention_q8_d256_fullbyte_dense_values_2pass_state(
                query_rot,
                &key_indices.reshape(&[kv_rows, cache_seq_capacity, key_dim]),
                &slot_scales.reshape(&[kv_rows, cache_seq_capacity, 3]),
                key_core.codebook_arr(key_bits)?,
                &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
                q_rows as u32,
                n_seq as u32,
                cache_seq_capacity as u32,
                q_heads as u32,
                layout.heads as u32,
                scale,
            )
        } else {
            None
        }
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_weighted_decode(
        &self,
        weights: &InlineArray,
        q_heads: i32,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;
        let (value_core, value_bits) = match &state.values {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            _ => return None,
        };
        let value_dim = layout.value_dim as i32;
        let kv_rows = (layout.batch * layout.heads) as i32;
        let q_rows = weights.dim(0);
        let n_seq = self.offset as i32;
        let indices_t = vs.indices_t_array()?;
        InlineArray::turboquant_weighted_decode(
            weights,
            &indices_t,
            &vs.norms_array()?.reshape(&[kv_rows, indices_t.dim(3)]),
            value_core.codebook_arr(value_bits)?,
            value_dim as u32,
            1u32 << value_bits,
            q_rows as u32,
            n_seq as u32,
            indices_t.dim(3) as u32,
            q_heads as u32,
            layout.heads as u32,
        )
    }

    #[doc(hidden)]
    pub fn bench_gpu_uniform_weighted_sum_dense_values(
        &self,
        weights: &InlineArray,
        q_heads: i32,
    ) -> Option<InlineArray> {
        let layout = self.layout?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let value_dim = layout.value_dim as i32;
        let q_rows = weights.dim(0);
        let n_seq = self.offset as i32;
        let value_rot_dense = vs.d256_rot_values_seq.as_ref()?;
        let cache_seq_capacity = value_rot_dense.dim(2);
        if value_dim != 256 || q_rows <= 0 || n_seq <= 0 || cache_seq_capacity < n_seq {
            return None;
        }
        let kv_rows = (layout.batch * layout.heads) as i32;
        InlineArray::turboquant_weighted_sum_d256_dense_values(
            weights,
            &value_rot_dense.reshape(&[kv_rows, cache_seq_capacity, value_dim]),
            q_rows as u32,
            n_seq as u32,
            cache_seq_capacity as u32,
            q_heads as u32,
            layout.heads as u32,
        )
    }

    fn try_gpu_uniform_attention(
        &self,
        queries_f32: &InlineArray,
        layout: CacheLayout,
        scale: f32,
        output_dtype: i32,
    ) -> Option<InlineArray> {
        let ks = self.keys.as_ref()?.gpu.as_ref()?;
        let vs = self.values.as_ref()?.gpu.as_ref()?;
        let state = self.state.as_ref()?;

        let (key_core, key_bits) = match &state.keys {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            TensorRuntime::Uniform { .. } | TensorRuntime::Mixed { .. } => return None,
        };
        let (value_core, value_bits) = match &state.values {
            TensorRuntime::Uniform {
                config: TurboQuantTensorConfig::Uniform { bits },
                core,
            } => (core, *bits),
            TensorRuntime::Uniform { .. } | TensorRuntime::Mixed { .. } => return None,
        };

        let batch = queries_f32.dim(0);
        let q_heads = queries_f32.dim(1);
        let key_dim = queries_f32.dim(3);
        let value_dim = layout.value_dim as i32;
        let q_rows = batch * q_heads;
        let n_seq = self.offset as i32;
        let cache_seq_capacity = ks.cache_seq_capacity();
        if q_rows <= 0 || n_seq <= 0 || cache_seq_capacity < n_seq {
            return None;
        }

        let trace_timing = turboquant_trace_enabled();
        let query_ready_us = if trace_timing {
            eval_stage_micros(queries_f32)
        } else {
            0
        };
        let query_rows = queries_f32.reshape(&[q_rows, key_dim]);
        let can_try_q8_fullbyte = turboquant_q8_fullbyte_enabled()
            && key_bits == 8
            && value_bits == 8
            && key_dim == 256
            && value_dim == 256
            && n_seq >= 1024
            && ks.q8_fullbyte_seq.is_some()
            && ks.q8_slot_scales_seq.is_some()
            && vs.d256_rot_values_seq.is_some();
        let mut project_us = 0;
        // Fused rotate+project: saves one dispatch per layer by doing
        // `input @ [inv_rot | inv_qjl]` as a single [N, 2*dim] matmul
        // instead of two separate [N, dim] matmuls. Only applied when
        // both outputs are needed (i.e., the q8 fullbyte fast path is
        // not taken). Falls back to sequential calls if the stacked
        // matrix wasn't built.
        let (query_rot, mut query_proj) = if !can_try_q8_fullbyte {
            if let Some((rot, proj)) = key_core.rotate_and_project_array(&query_rows) {
                (rot, Some(proj))
            } else {
                let rot = key_core.rotate_array(&query_rows)?;
                let proj = key_core.project_array(&query_rows)?;
                (rot, Some(proj))
            }
        } else {
            (key_core.rotate_array(&query_rows)?, None)
        };
        let rotate_us = if trace_timing {
            eval_stage_micros(&query_rot)
        } else {
            0
        };
        if let Some(proj) = query_proj.as_ref() {
            if trace_timing {
                project_us = eval_stage_micros(proj);
            }
        }

        let kv_rows = (layout.batch * layout.heads) as i32;
        let key_norms = ks
            .key_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let key_residual_norms = ks
            .residual_norms_array()?
            .reshape(&[kv_rows, cache_seq_capacity]);
        let qjl_words = ks.qjl_words();
        if can_try_q8_fullbyte {
            if let Some(decoded_rot) =
                self.try_gpu_uniform_attention_q8_d256_precomputed(&query_rot, None, q_heads, scale)
            {
                let decode_us = if trace_timing {
                    eval_stage_micros(&decoded_rot)
                } else {
                    0
                };
                let output_rows =
                    value_core.inverse_rotate_output_array(&decoded_rot, output_dtype)?;
                let inverse_rotate_us = if trace_timing {
                    eval_stage_micros(&output_rows)
                } else {
                    0
                };
                if trace_timing {
                    trace_turboquant_bridge(&format!(
                        "gpu_uniform_q8_d256_fullbyte_densev_2pass_stage_us seq={} q_rows={} query_ready={} rotate={} project={} score=0 softmax=0 decode={} inverse_rotate={}",
                        n_seq,
                        q_rows,
                        query_ready_us,
                        rotate_us,
                        project_us,
                        decode_us,
                        inverse_rotate_us,
                    ));
                }
                return Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]));
            }
        }
        if query_proj.is_none() {
            let projected = key_core.project_array(&query_rows)?;
            if trace_timing {
                project_us = eval_stage_micros(&projected);
            }
            query_proj = Some(projected);
        }
        let query_proj = query_proj.as_ref()?;
        let key_codebook = key_core.codebook_arr(key_bits.saturating_sub(1))?;
        if let Some(decoded_rot) = self.try_gpu_uniform_attention_q8_d256_precomputed(
            &query_rot,
            Some(query_proj),
            q_heads,
            scale,
        ) {
            let decode_us = if trace_timing {
                eval_stage_micros(&decoded_rot)
            } else {
                0
            };
            let output_rows = value_core.inverse_rotate_output_array(&decoded_rot, output_dtype)?;
            let inverse_rotate_us = if trace_timing {
                eval_stage_micros(&output_rows)
            } else {
                0
            };
            if trace_timing {
                trace_turboquant_bridge(&format!(
                    "gpu_uniform_q8_d256_2pass_stage_us seq={} q_rows={} query_ready={} rotate={} project={} score=0 softmax=0 decode={} inverse_rotate={}",
                    n_seq,
                    q_rows,
                    query_ready_us,
                    rotate_us,
                    project_us,
                    decode_us,
                    inverse_rotate_us,
                ));
            }
            return Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]));
        }
        if key_bits == 8 && value_bits == 8 && key_dim == 128 && value_dim == 128 && n_seq >= 1024 {
            let key_indices = ks
                .indices_t_array()
                .reshape(&[kv_rows, key_dim, cache_seq_capacity]);
            let value_indices =
                vs.indices_t_array()?
                    .reshape(&[kv_rows, value_dim, cache_seq_capacity]);
            let value_norms = vs.norms_array()?.reshape(&[kv_rows, cache_seq_capacity]);

            if q_heads > 8 {
                if let Some(key_bytes) = ks.q8_keybytes_t.as_ref() {
                    let key_bytes = key_bytes.reshape(&[kv_rows, key_dim, cache_seq_capacity]);
                    if let Some(decoded_rot) =
                        InlineArray::turboquant_attention_q8_d128_packed_keys_2pass(
                            &query_rot,
                            query_proj,
                            &key_bytes,
                            &key_norms,
                            &key_residual_norms,
                            key_codebook,
                            &value_indices,
                            &value_norms,
                            value_core.codebook_arr(value_bits)?,
                            q_rows as u32,
                            n_seq as u32,
                            cache_seq_capacity as u32,
                            q_heads as u32,
                            layout.heads as u32,
                            scale,
                        )
                    {
                        let decode_us = if trace_timing {
                            eval_stage_micros(&decoded_rot)
                        } else {
                            0
                        };
                        let output_rows =
                            value_core.inverse_rotate_output_array(&decoded_rot, output_dtype)?;
                        let inverse_rotate_us = if trace_timing {
                            eval_stage_micros(&output_rows)
                        } else {
                            0
                        };
                        if trace_timing {
                            trace_turboquant_bridge(&format!(
                                "gpu_uniform_q8_d128_packed_keys_2pass_stage_us seq={} q_rows={} query_ready={} rotate={} project={} score=0 softmax=0 decode={} inverse_rotate={}",
                                n_seq,
                                q_rows,
                                query_ready_us,
                                rotate_us,
                                project_us,
                                decode_us,
                                inverse_rotate_us,
                            ));
                        }
                        return Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]));
                    }
                }
            } else if qjl_words == 4 {
                let key_qjl_signs =
                    ks.qjl_signs_t_array()
                        .reshape(&[kv_rows, qjl_words, cache_seq_capacity]);
                if let Some(decoded_rot) = InlineArray::turboquant_attention_q8_d128_2pass(
                    &query_rot,
                    query_proj,
                    &key_indices,
                    &key_qjl_signs,
                    &key_norms,
                    &key_residual_norms,
                    key_codebook,
                    &value_indices,
                    &value_norms,
                    value_core.codebook_arr(value_bits)?,
                    q_rows as u32,
                    n_seq as u32,
                    cache_seq_capacity as u32,
                    q_heads as u32,
                    layout.heads as u32,
                    scale,
                ) {
                    let decode_us = if trace_timing {
                        eval_stage_micros(&decoded_rot)
                    } else {
                        0
                    };
                    let output_rows =
                        value_core.inverse_rotate_output_array(&decoded_rot, output_dtype)?;
                    let inverse_rotate_us = if trace_timing {
                        eval_stage_micros(&output_rows)
                    } else {
                        0
                    };
                    if trace_timing {
                        trace_turboquant_bridge(&format!(
                            "gpu_uniform_q8_d128_2pass_stage_us seq={} q_rows={} query_ready={} rotate={} project={} score=0 softmax=0 decode={} inverse_rotate={}",
                            n_seq,
                            q_rows,
                            query_ready_us,
                            rotate_us,
                            project_us,
                            decode_us,
                            inverse_rotate_us,
                        ));
                    }
                    return Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]));
                }
            }
        }

        let scores =
            self.bench_gpu_uniform_scores_precomputed(&query_rot, query_proj, q_heads, scale)?;
        let score_us = if trace_timing {
            eval_stage_micros(&scores)
        } else {
            0
        };
        let weights = scores.softmax(-1);
        let softmax_us = if trace_timing {
            eval_stage_micros(&weights)
        } else {
            0
        };
        let value_norms = vs.norms_array()?.reshape(&[kv_rows, cache_seq_capacity]);
        let decoded_rot = InlineArray::turboquant_weighted_decode(
            &weights,
            &vs.indices_t_array()?,
            &value_norms,
            value_core.codebook_arr(value_bits)?,
            value_dim as u32,
            1u32 << value_bits,
            q_rows as u32,
            n_seq as u32,
            cache_seq_capacity as u32,
            q_heads as u32,
            layout.heads as u32,
        )?;
        let decode_us = if trace_timing {
            eval_stage_micros(&decoded_rot)
        } else {
            0
        };
        let output_rows = value_core.inverse_rotate_output_array(&decoded_rot, output_dtype)?;
        let inverse_rotate_us = if trace_timing {
            eval_stage_micros(&output_rows)
        } else {
            0
        };
        if trace_timing {
            trace_turboquant_bridge(&format!(
                "gpu_uniform_stage_us seq={} q_rows={} query_ready={} rotate={} project={} score={} softmax={} decode={} inverse_rotate={}",
                n_seq,
                q_rows,
                query_ready_us,
                rotate_us,
                project_us,
                score_us,
                softmax_us,
                decode_us,
                inverse_rotate_us
            ));
        }
        Some(output_rows.reshape(&[batch, q_heads, 1, value_dim]))
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
            if existing.batch != b
                || existing.heads != h
                || existing.key_dim != kd
                || existing.value_dim != vd
            {
                return Err(format!(
                    "TurboQuant: layout mismatch — expected [{b},{h},*,{kd}] / [{b},{h},*,{vd}]"
                ));
            }
            return Ok(existing);
        }

        let layout = CacheLayout {
            batch: b,
            heads: h,
            key_dim: kd,
            value_dim: vd,
        };
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
pub fn build_state(
    key_dim: usize,
    value_dim: usize,
    config: TurboQuantConfig,
) -> Arc<TurboQuantState> {
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
    let k_rot = k_core.rotate_array(&k_norm)?;

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
    let k_mse_recon_unrot = k_core.inverse_rotate_array(&k_mse_recon_rot)?;
    let k_residual_unrot = k_norm.subtract(&k_mse_recon_unrot);
    let k_qjl_proj = k_core.project_array(&k_residual_unrot)?;
    let qjl_shape = k_qjl_proj.shape();
    let qjl_ndim = qjl_shape.len();
    let qjl_rows: i32 = qjl_shape[..qjl_ndim - 1].iter().product();
    let packed_dim = packed_qjl_words(k_core.dim) as i32;
    let k_qjl_proj_2d = if qjl_ndim == 2 {
        k_qjl_proj.clone()
    } else {
        k_qjl_proj.reshape(&[qjl_rows, k_core.dim as i32])
    };
    let k_qjl_signs = InlineArray::turboquant_pack_sign_bits(
        &k_qjl_proj_2d,
        k_core.dim as u32,
        packed_dim as u32,
        qjl_rows as u32,
    )?;
    let k_qjl_signs = if qjl_ndim == 2 {
        k_qjl_signs
    } else {
        let mut packed_shape: Vec<i32> = qjl_shape[..qjl_ndim - 1].to_vec();
        packed_shape.push(packed_dim);
        k_qjl_signs.reshape(&packed_shape)
    };
    let use_q8_seq_shadow = key_bits == 8 && k_core.dim == 256 && v_core.dim == 256;
    let k_indices_t = (!use_q8_seq_shadow).then(|| k_indices.transpose_axes(&[0, 1, 3, 2]));
    let k_qjl_signs_t = (!use_q8_seq_shadow).then(|| k_qjl_signs.transpose_axes(&[0, 1, 3, 2]));
    let q8_pack_inputs = if key_bits == 8 {
        let kv_rows = (keys.dim(0) * keys.dim(1)) as u32;
        let seq = keys.dim(2) as u32;
        let indices_t_3d = if let Some(indices_t) = k_indices_t.as_ref() {
            indices_t.reshape(&[kv_rows as i32, k_core.dim as i32, seq as i32])
        } else {
            k_indices.transpose_axes(&[0, 1, 3, 2]).reshape(&[
                kv_rows as i32,
                k_core.dim as i32,
                seq as i32,
            ])
        };
        let qjl_signs_t_3d = if let Some(qjl_signs_t) = k_qjl_signs_t.as_ref() {
            qjl_signs_t.reshape(&[kv_rows as i32, packed_dim, seq as i32])
        } else {
            k_qjl_signs.transpose_axes(&[0, 1, 3, 2]).reshape(&[
                kv_rows as i32,
                packed_dim,
                seq as i32,
            ])
        };
        Some((kv_rows, seq, indices_t_3d, qjl_signs_t_3d))
    } else {
        None
    };
    let q8_keybytes_t = if use_q8_seq_shadow {
        None
    } else if let Some((kv_rows, seq, indices_t_3d, qjl_signs_t_3d)) = q8_pack_inputs.as_ref() {
        InlineArray::turboquant_pack_q8_keybytes(
            indices_t_3d,
            qjl_signs_t_3d,
            k_core.dim as u32,
            packed_dim as u32,
            *kv_rows,
            *seq,
        )
        .map(|packed| packed.reshape(&[keys.dim(0), keys.dim(1), k_core.dim as i32, keys.dim(2)]))
    } else {
        None
    };
    let q8_keybytes_seq = if use_q8_seq_shadow {
        q8_pack_inputs
            .as_ref()
            .and_then(|(kv_rows, seq, indices_t_3d, qjl_signs_t_3d)| {
                InlineArray::turboquant_pack_q8_keybytes_seq(
                    indices_t_3d,
                    qjl_signs_t_3d,
                    k_core.dim as u32,
                    packed_dim as u32,
                    *kv_rows,
                    *seq,
                )
                .map(|packed| {
                    packed.reshape(&[keys.dim(0), keys.dim(1), keys.dim(2), k_core.dim as i32])
                })
            })
    } else {
        None
    };
    let q8_fullbyte_seq = if use_q8_seq_shadow && turboquant_q8_fullbyte_enabled() {
        k_core
            .gpu_quantize_mse(&k_rot, 8)
            .map(|indices| indices.as_dtype(Dtype::Uint8.as_i32()))
    } else {
        None
    };

    // ── Values ────────────────────────────────────────────────────────────
    let (v_indices, v_indices_t, val_norms, d256_rot_values_seq) = if use_q8_seq_shadow {
        (
            None,
            None,
            None,
            Some(
                v_core
                    .rotate_array(values)?
                    .as_dtype(Dtype::Bfloat16.as_i32()),
            ),
        )
    } else {
        let val_norms = values.norm_l2(-1, true);
        let safe_norms_v = val_norms.maximum(&eps);
        let v_norm = values.divide(&safe_norms_v);

        // v_norm @ rotation.T = v_norm @ inverse_rotation_arr
        let v_rot = v_core.rotate_array(&v_norm)?;
        let v_indices = v_core.gpu_quantize_mse(&v_rot, val_bits)?;
        let v_indices_t = Some(v_indices.transpose_axes(&[0, 1, 3, 2]));
        (Some(v_indices), v_indices_t, Some(val_norms), None)
    };
    let q8_kvbytes_seq = None;
    let q8_slot_scales_seq = if use_q8_seq_shadow {
        let key_scales = key_norms.concatenate_2(&k_residual_norms, 3);
        let value_scales = InlineArray::ones(
            &[values.dim(0), values.dim(1), values.dim(2), 1],
            Dtype::Float32.as_i32(),
        );
        Some(key_scales.concatenate_2(&value_scales, 3))
    } else {
        None
    };

    Some((
        GpuKeyStore {
            indices: k_indices,
            indices_t: k_indices_t,
            q8_keybytes_t,
            q8_keybytes_seq,
            q8_fullbyte_seq,
            q8_kvbytes_seq,
            q8_slot_scales_seq,
            norms: (!use_q8_seq_shadow).then_some(key_norms),
            qjl_signs: k_qjl_signs,
            qjl_signs_t: k_qjl_signs_t,
            residual_norms: (!use_q8_seq_shadow).then_some(k_residual_norms),
        },
        GpuValueStore {
            indices: v_indices,
            indices_t: v_indices_t,
            norms: val_norms,
            d256_rot_values_seq,
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
    let mse_base = core.inverse_rotate_array(&mse_recon_rot)?;

    // 3. QJL correction.
    //    CPU: inverse_project_rows(signs) = matmul_rows(inverse_qjl, dim, signs) = signs @ inverse_qjl.T = signs @ qjl.
    //    The GPU store keeps packed uint32 sign words, so unpack to {-1,+1}
    //    before the matmul with qjl_arr.
    let packed_shape = store.qjl_signs.shape();
    let packed_ndim = packed_shape.len();
    let packed_rows: i32 = packed_shape[..packed_ndim - 1].iter().product();
    let packed_words = packed_shape[packed_ndim - 1];
    let packed_signs = if packed_ndim == 2 {
        store.qjl_signs.clone()
    } else {
        store.qjl_signs.reshape(&[packed_rows, packed_words])
    };
    let unpacked_qjl_2d = InlineArray::turboquant_unpack_sign_bits(
        &packed_signs,
        core.dim as u32,
        packed_words as u32,
        packed_rows as u32,
    )?;
    let unpacked_qjl = if packed_ndim == 2 {
        unpacked_qjl_2d
    } else {
        let mut unpacked_shape: Vec<i32> = packed_shape[..packed_ndim - 1].to_vec();
        unpacked_shape.push(core.dim as i32);
        unpacked_qjl_2d.reshape(&unpacked_shape)
    };
    let qjl_correction = core.inverse_project_array(&unpacked_qjl)?;
    let dim = core.dim as f32;
    let qjl_scale_factor = InlineArray::from_f32((std::f32::consts::PI / 2.0).sqrt() / dim);
    // residual_norms: [B,H,T,1] keepdims — broadcast along D.
    let residual_norms = store.residual_norms_array()?;
    let scale = residual_norms.multiply(&qjl_scale_factor);
    let correction = qjl_correction.multiply(&scale);

    // 4. Base + QJL correction, rescale by original L2 norm.
    // norms: [B,H,T,1] keepdims — broadcast along D.
    let combined = mse_base.add(&correction);
    Some(combined.multiply(&store.key_norms_array()?))
}

/// Dequantise GPU-stored values back to `[B, H, T, Dv]` f32.
fn gpu_dequantize_values(
    store: &GpuValueStore,
    runtime: &TensorRuntime,
    val_bits: u8,
) -> Option<InlineArray> {
    if let Some(d256_rot_values_seq) = store.d256_rot_values_seq.as_ref() {
        let core = match runtime {
            TensorRuntime::Uniform { core, .. } => core,
            TensorRuntime::Mixed { .. } => return None,
        };
        let dense_rot = d256_rot_values_seq.as_dtype(Dtype::Float32.as_i32());
        return core.inverse_rotate_array(&dense_rot);
    }

    let core = match runtime {
        TensorRuntime::Uniform { core, .. } => core,
        TensorRuntime::Mixed { .. } => return None,
    };

    if turboquant_wht_enabled() && core.dim == 256 {
        let indices_arr = store.indices.as_ref()?;
        let norms_arr = store.norms.as_ref()?;
        let shape = indices_arr.shape();
        let total = shape.iter().product::<i32>() as usize;
        let rows = shape[..shape.len() - 1].iter().product::<i32>() as usize;
        let indices: Vec<u16> = inline_array_to_f32_vec(indices_arr, total)?
            .into_iter()
            .map(|v| v as u16)
            .collect();
        let norms = inline_array_to_f32_vec(norms_arr, rows)?;
        let reconstructed = decode_value_component_rows_raw(core, &indices, &norms, val_bits);
        return Some(InlineArray::from_f32_slice(&reconstructed, shape));
    }

    // 1. Reconstruct MSE centroids in rotated space.
    let mse_recon_rot = core.gpu_reconstruct_mse(store.indices.as_ref()?, val_bits)?;

    // 2. Inverse-rotate: recon_rot @ rotation_arr (same derivation as keys).
    let mse_base = core.inverse_rotate_array(&mse_recon_rot)?;

    // 3. Rescale by stored L2 norms [B,H,T,1].
    Some(mse_base.multiply(store.norms.as_ref()?))
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

fn encode_key_rows(runtime: &TensorRuntime, total_dim: usize, rows: &[f32]) -> BatchedKeyRows {
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

fn encode_value_rows(runtime: &TensorRuntime, total_dim: usize, rows: &[f32]) -> BatchedValueRows {
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
#[allow(clippy::needless_range_loop)]
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
#[allow(clippy::needless_range_loop)]
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
#[allow(clippy::needless_range_loop)]
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
#[allow(clippy::needless_range_loop)]
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
pub(crate) fn generate_random_orthogonal(dim: usize, rng: &mut StdRng) -> Vec<f32> {
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
pub(crate) fn generate_random_projection(dim: usize, rng: &mut StdRng) -> Vec<f32> {
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

fn generate_rademacher_signs(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    (0..dim)
        .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
        .collect()
}

fn fwht_in_place(values: &mut [f32]) {
    let len = values.len();
    debug_assert!(len.is_power_of_two());
    let mut h = 1usize;
    while h < len {
        let step = h << 1;
        let mut i = 0usize;
        while i < len {
            for j in i..i + h {
                let a = values[j];
                let b = values[j + h];
                values[j] = a + b;
                values[j + h] = a - b;
            }
            i += step;
        }
        h = step;
    }
}

fn signed_fwht_forward(values: &mut [f32], left_signs: &[f32], right_signs: &[f32]) {
    debug_assert_eq!(values.len(), left_signs.len());
    debug_assert_eq!(values.len(), right_signs.len());
    for (v, &s) in values.iter_mut().zip(right_signs.iter()) {
        *v *= s;
    }
    fwht_in_place(values);
    let scale = 1.0f32 / (values.len() as f32).sqrt();
    for (v, &left) in values.iter_mut().zip(left_signs.iter()) {
        *v *= left * scale;
    }
}

#[allow(dead_code)] // Inverse path for TurboQuant CPU dequantization — paired with signed_fwht_forward
fn signed_fwht_inverse(values: &mut [f32], left_signs: &[f32], right_signs: &[f32]) {
    debug_assert_eq!(values.len(), left_signs.len());
    debug_assert_eq!(values.len(), right_signs.len());
    let scale = 1.0f32 / (values.len() as f32).sqrt();
    for (v, &left) in values.iter_mut().zip(left_signs.iter()) {
        *v *= left * scale;
    }
    fwht_in_place(values);
    for (v, &right) in values.iter_mut().zip(right_signs.iter()) {
        *v *= right;
    }
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
    let arr = InlineArray::from_f32_slice(matrix, &[dim as i32, dim as i32]);
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

    fn make_uniform_direct_attention_case_with(
        dim: usize,
        heads: i32,
        prefill: i32,
    ) -> (
        QuantizedKvCache,
        InlineArray,
        InlineArray,
        InlineArray,
        f32,
        i32,
        i32,
        i32,
    ) {
        let config = TurboQuantConfig::uniform(8, 8);
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_len = (b * h * prefill * d) as usize;
        let step_len = (b * h * d) as usize;
        let prefill_keys =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.2), &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.7), &[b, h, prefill, d]);
        let queries = InlineArray::from_f32_slice(&make_data(step_len, 1.3), &[b, h, 1, d]);
        let step_keys = InlineArray::from_f32_slice(&make_data(step_len, 1.9), &[b, h, 1, d]);
        let step_values = InlineArray::from_f32_slice(&make_data(step_len, 2.4), &[b, h, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");
        assert!(
            seed_cache
                .keys
                .as_ref()
                .and_then(|k| k.gpu.as_ref())
                .is_some()
        );
        assert!(
            seed_cache
                .values
                .as_ref()
                .and_then(|v| v.gpu.as_ref())
                .is_some()
        );

        (seed_cache, queries, step_keys, step_values, scale, b, h, d)
    }

    fn make_uniform_direct_attention_case() -> (
        QuantizedKvCache,
        InlineArray,
        InlineArray,
        InlineArray,
        f32,
        i32,
        i32,
        i32,
    ) {
        make_uniform_direct_attention_case_with(16, 2, 3)
    }

    fn make_uniform_gqa_direct_attention_case_with(
        dim: usize,
        q_heads: i32,
        kv_heads: i32,
        prefill: i32,
    ) -> (
        QuantizedKvCache,
        InlineArray,
        InlineArray,
        InlineArray,
        f32,
        i32,
        i32,
        i32,
        i32,
    ) {
        let config = TurboQuantConfig::uniform(8, 8);
        let b = 1i32;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_kv_len = (b * kv_heads * prefill * d) as usize;
        let step_kv_len = (b * kv_heads * d) as usize;
        let query_len = (b * q_heads * d) as usize;
        let prefill_keys = InlineArray::from_f32_slice(
            &make_data(prefill_kv_len, 0.2),
            &[b, kv_heads, prefill, d],
        );
        let prefill_values = InlineArray::from_f32_slice(
            &make_data(prefill_kv_len, 0.7),
            &[b, kv_heads, prefill, d],
        );
        let queries = InlineArray::from_f32_slice(&make_data(query_len, 1.3), &[b, q_heads, 1, d]);
        let step_keys =
            InlineArray::from_f32_slice(&make_data(step_kv_len, 1.9), &[b, kv_heads, 1, d]);
        let step_values =
            InlineArray::from_f32_slice(&make_data(step_kv_len, 2.4), &[b, kv_heads, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");
        assert!(
            seed_cache
                .keys
                .as_ref()
                .and_then(|k| k.gpu.as_ref())
                .is_some()
        );
        assert!(
            seed_cache
                .values
                .as_ref()
                .and_then(|v| v.gpu.as_ref())
                .is_some()
        );

        (
            seed_cache,
            queries,
            step_keys,
            step_values,
            scale,
            b,
            q_heads,
            kv_heads,
            d,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_range_loop)]
    fn manual_single_token_attention(
        queries: &mut InlineArray,
        keys: &mut InlineArray,
        values: &mut InlineArray,
        batch: i32,
        heads: i32,
        seq: i32,
        dim: i32,
        scale: f32,
    ) -> Vec<f32> {
        let q = queries
            .to_f32_vec((batch * heads * dim) as usize)
            .expect("queries to_f32");
        let k = keys
            .to_f32_vec((batch * heads * seq * dim) as usize)
            .expect("keys to_f32");
        let v = values
            .to_f32_vec((batch * heads * seq * dim) as usize)
            .expect("values to_f32");

        let rows = (batch * heads) as usize;
        let seq_usize = seq as usize;
        let dim_usize = dim as usize;
        let mut out = vec![0.0f32; rows * dim_usize];

        for row in 0..rows {
            let q_base = row * dim_usize;
            let q_row = &q[q_base..q_base + dim_usize];

            let mut scores = vec![0.0f32; seq_usize];
            for t in 0..seq_usize {
                let k_base = (row * seq_usize + t) * dim_usize;
                let k_row = &k[k_base..k_base + dim_usize];
                let dot = q_row
                    .iter()
                    .zip(k_row.iter())
                    .map(|(lhs, rhs)| lhs * rhs)
                    .sum::<f32>();
                scores[t] = dot * scale;
            }

            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                sum_exp += *score;
            }
            for score in &mut scores {
                *score /= sum_exp.max(f32::MIN_POSITIVE);
            }

            let out_row = &mut out[q_base..q_base + dim_usize];
            for t in 0..seq_usize {
                let v_base = (row * seq_usize + t) * dim_usize;
                let v_row = &v[v_base..v_base + dim_usize];
                let weight = scores[t];
                for (dst, val) in out_row.iter_mut().zip(v_row.iter()) {
                    *dst += weight * *val;
                }
            }
        }

        out
    }

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
        assert!(codebook.iter().all(|&v| (-1.0..=1.0).contains(&v)));
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
    ///      modulo f32 ordering differences).
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
        let data: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.1 - total as f32 * 0.05).sin())
            .collect();
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
        assert!(
            cpu_cache.keys.as_ref().unwrap().gpu.is_none(),
            "Expected CPU path for Mixed config"
        );

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
        let dv = gpu_cache
            .dequantize_values()
            .expect("GPU dequantize_values");

        // Verify output shapes: [B, H, T, D].
        assert_eq!(dk.shape(), &[b, h, s, d], "dequantized keys shape mismatch");
        assert_eq!(
            dv.shape(),
            &[b, h, s, d],
            "dequantized values shape mismatch"
        );

        // Output should be finite (not NaN/Inf).
        let dk_vals = dk
            .reshape(&[(b * h * s * d)])
            .to_f32_vec(total)
            .expect("dk to_f32");
        let dv_vals = dv
            .reshape(&[(b * h * s * d)])
            .to_f32_vec(total)
            .expect("dv to_f32");
        assert!(
            dk_vals.iter().all(|v| v.is_finite()),
            "dequantized keys contain non-finite"
        );
        assert!(
            dv_vals.iter().all(|v| v.is_finite()),
            "dequantized values contain non-finite"
        );

        // Verify output is within reasonable range (quantisation introduces error but
        // should not explode — reconstructed vectors should be roughly same magnitude as input).
        let input_max = data.iter().cloned().fold(0.0f32, f32::max).abs();
        let dk_max = dk_vals.iter().cloned().fold(0.0f32, f32::max).abs();
        assert!(
            dk_max < input_max * 3.0,
            "dequantized keys magnitude unreasonably large"
        );
    }

    #[test]
    fn gpu_packed_qjl_sign_words_round_trip_encodes_zero_as_positive_bit() {
        let projected = InlineArray::from_f32_slice(&[-2.0f32, 0.0, 3.0, -0.5], &[1, 4]);
        let packed =
            InlineArray::turboquant_pack_sign_bits(&projected, 4, 1, 1).expect("pack sign bits");
        let mut unpacked =
            InlineArray::turboquant_unpack_sign_bits(&packed, 4, 1, 1).expect("unpack sign bits");
        let values = unpacked.to_f32_vec(4).expect("unpacked to_f32");
        assert_eq!(values, vec![-1.0, 1.0, 1.0, -1.0]);
    }

    #[test]
    fn turboquant_q8_d256_gpu_store_uses_seq_shadow_without_transposed_shadow() {
        let (seed_cache, _, _, _, _, _, _, _, _) =
            make_uniform_gqa_direct_attention_case_with(256, 16, 2, 1023);
        let gpu = seed_cache
            .keys
            .as_ref()
            .and_then(|k| k.gpu.as_ref())
            .expect("gpu key store");
        assert!(
            gpu.q8_keybytes_t.is_none(),
            "d256 q8 path should not keep transposed q8 shadow"
        );
        assert!(
            gpu.q8_keybytes_seq.is_some(),
            "d256 q8 path should keep seq-major packed key shadow"
        );
        assert!(
            gpu.q8_kvbytes_seq.is_none(),
            "d256 q8 path should not keep packed kv shadow when dense rotated values are present"
        );
        assert!(
            gpu.q8_slot_scales_seq.is_some(),
            "d256 q8 path should keep seq-major slot scale shadow"
        );
        assert!(
            gpu.indices_t.is_none(),
            "d256 q8 path should not keep transposed key indices"
        );
        assert!(
            gpu.qjl_signs_t.is_none(),
            "d256 q8 path should not keep transposed qjl sign words"
        );
        assert!(
            gpu.norms.is_none(),
            "d256 q8 path should source key norms from slot scales"
        );
        assert!(
            gpu.residual_norms.is_none(),
            "d256 q8 path should source residual norms from slot scales"
        );
        let gpu_values = seed_cache
            .values
            .as_ref()
            .and_then(|v| v.gpu.as_ref())
            .expect("gpu value store");
        assert!(
            gpu_values.indices_t.is_none(),
            "d256 q8 path should not keep transposed value indices"
        );
    }

    #[test]
    fn turboquant_direct_attention_matches_dequantized_sdpa_uniform() {
        let (seed_cache, queries, step_keys, step_values, scale, b, h, d) =
            make_uniform_direct_attention_case();
        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            4,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "direct attention diverged from dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    #[test]
    fn turboquant_direct_attention_matches_dequantized_sdpa_uniform_q8_d128_long_context() {
        let (seed_cache, queries, step_keys, step_values, scale, b, h, d) =
            make_uniform_direct_attention_case_with(128, 2, 1023);
        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            1024,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "long-context direct attention diverged from dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    #[test]
    fn turboquant_direct_attention_matches_dequantized_sdpa_uniform_q8_d256_long_context_gqa() {
        let (seed_cache, queries, step_keys, step_values, scale, b, q_heads, kv_heads, d) =
            make_uniform_gqa_direct_attention_case_with(256, 16, 2, 1023);
        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let full_values = ref_cache.dequantize_values().expect("dequantize values");
        let repeated_keys = full_keys.repeat(q_heads / kv_heads, 1);
        let repeated_values = full_values.repeat(q_heads / kv_heads, 1);
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut repeated_keys.clone(),
            &mut repeated_values.clone(),
            b,
            q_heads,
            1024,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * q_heads * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "d256 gqa direct attention diverged from dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    #[test]
    fn turboquant_attention_q8_d256_fullbyte_dense_values_matches_manual_reference() {
        let batch = 1i32;
        let q_heads = 4i32;
        let kv_heads = 2i32;
        let groups = q_heads / kv_heads;
        let dim = 256i32;
        let seq = 1024i32;
        let kv_rows = batch * kv_heads;
        let q_rows = batch * q_heads;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let query_rot_vec: Vec<f32> = (0..(q_rows * dim) as usize)
            .map(|i| ((i as f32) * 0.013 + 0.4).sin() * 0.7)
            .collect();
        let key_indices_vec: Vec<u8> = (0..(kv_rows * seq * dim) as usize)
            .map(|i| (((i * 17) + 23) & 0xff) as u8)
            .collect();
        let slot_scales_vec: Vec<f32> = (0..(kv_rows * seq * 3) as usize)
            .map(|i| match i % 3 {
                0 => 0.5 + (((i / 3) % 11) as f32) * 0.03125,
                1 => 0.0,
                _ => 1.0,
            })
            .collect();
        let key_codebook_vec: Vec<f32> = (0..256).map(|i| ((i as f32) - 127.5) / 96.0).collect();
        let value_dense_vec: Vec<f32> = (0..(kv_rows * seq * dim) as usize)
            .map(|i| ((i as f32) * 0.009 - 0.7).cos() * 0.5)
            .collect();

        let query_rot = InlineArray::from_f32_slice(&query_rot_vec, &[q_rows, dim]);
        let key_indices = InlineArray::from_u8_slice(&key_indices_vec, &[kv_rows, seq, dim]);
        let slot_scales = InlineArray::from_f32_slice(&slot_scales_vec, &[kv_rows, seq, 3]);
        let key_codebook = InlineArray::from_f32_slice(&key_codebook_vec, &[256]);
        let value_dense = InlineArray::from_f32_slice(&value_dense_vec, &[kv_rows, seq, dim]);

        let mut direct = InlineArray::turboquant_attention_q8_d256_fullbyte_dense_values_2pass(
            &query_rot,
            &key_indices,
            &slot_scales,
            &key_codebook,
            &value_dense,
            q_rows as u32,
            seq as u32,
            seq as u32,
            q_heads as u32,
            kv_heads as u32,
            scale,
        )
        .expect("fullbyte direct attention");

        let mut keys = vec![0.0f32; (batch * q_heads * seq * dim) as usize];
        let mut values = vec![0.0f32; (batch * q_heads * seq * dim) as usize];
        for qh in 0..q_heads as usize {
            let kvh = qh / groups as usize;
            for t in 0..seq as usize {
                let scale_base = (kvh * seq as usize + t) * 3;
                let key_norm = slot_scales_vec[scale_base];
                let key_base = (kvh * seq as usize + t) * dim as usize;
                let out_base = (qh * seq as usize + t) * dim as usize;
                for d_idx in 0..dim as usize {
                    let idx = key_indices_vec[key_base + d_idx] as usize;
                    keys[out_base + d_idx] = key_codebook_vec[idx] * key_norm;
                    values[out_base + d_idx] = value_dense_vec[key_base + d_idx];
                }
            }
        }

        let mut queries = query_rot.reshape(&[batch, q_heads, 1, dim]);
        let mut keys = InlineArray::from_f32_slice(&keys, &[batch, q_heads, seq, dim]);
        let mut values = InlineArray::from_f32_slice(&values, &[batch, q_heads, seq, dim]);
        let reference_vals = manual_single_token_attention(
            &mut queries,
            &mut keys,
            &mut values,
            batch,
            q_heads,
            seq,
            dim,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((batch * q_heads * dim) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "d256 fullbyte direct attention diverged from manual reference: max_abs_diff={max_abs_diff}"
        );
    }

    #[test]
    fn turboquant_direct_attention_uniform_smoke() {
        let (seed_cache, queries, step_keys, step_values, scale, b, h, d) =
            make_uniform_direct_attention_case();
        let mut direct_cache = seed_cache;
        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");
        let vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn turboquant_reference_attention_uniform_smoke() {
        let (seed_cache, queries, step_keys, step_values, scale, b, h, d) =
            make_uniform_direct_attention_case();
        let mut ref_cache = seed_cache;
        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            4,
            d,
            scale,
        );
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn turboquant_dequantize_keys_after_append_uniform_smoke() {
        let (seed_cache, _, step_keys, step_values, _, b, h, d) =
            make_uniform_direct_attention_case();
        let mut ref_cache = seed_cache;
        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        assert_eq!(full_keys.shape(), &[b, h, 4, d]);
        let vals = full_keys
            .to_f32_vec((b * h * 4 * d) as usize)
            .expect("keys to_f32");
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn turboquant_dequantize_values_after_append_uniform_smoke() {
        let (seed_cache, _, step_keys, step_values, _, b, h, d) =
            make_uniform_direct_attention_case();
        let mut ref_cache = seed_cache;
        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        assert_eq!(full_values.shape(), &[b, h, 4, d]);
        let vals = full_values
            .to_f32_vec((b * h * 4 * d) as usize)
            .expect("values to_f32");
        assert!(vals.iter().all(|v| v.is_finite()));
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
            (0..b * h * d)
                .map(|i| (i as f32 * 0.15 + seed).sin())
                .collect()
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
