//! Pure-Rust math primitives for the TurboQuant cache.
//!
//! No external dependencies beyond `rand` and `mlx-rs`-free InlineArray
//! marshalling — every function here is deterministic in its inputs so
//! call sites can rely on byte-identical results across crates (notably
//! `pmetal-mlx` mirrors several of these).

use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};

use rand::{RngExt, rngs::StdRng};

use crate::InlineArray;

type BetaCodebookKey = (usize, u8);
type BetaCodebookCache = Mutex<HashMap<BetaCodebookKey, Arc<Vec<f32>>>>;

/// Lloyd-Max iteration cap.
const LLOYD_MAX_ITERS: usize = 64;
/// Lloyd-Max convergence threshold.
const LLOYD_MAX_TOLERANCE: f64 = 1e-7;
/// Number of grid points for the Beta-distribution quadrature.
const LLOYD_GRID_POINTS: usize = 8192;

/// Lloyd-Max optimal scalar quantisation codebook for the Beta distribution.
///
/// The marginal of a random unit-sphere vector in R^d is Beta((d-1)/2, (d-1)/2),
/// supported on [-1, 1].  This solver approximates the optimal MSE centroids via
/// iterative centroid update (Voronoi quantisation).
///
/// Returns a sorted Vec of 2^bits centroids in [-1, 1].
///
/// Memoized across calls — Lloyd-Max with 8192-pt grid × 64 iterations is
/// deterministic in `(dim, bits)`, so we cache the result and reuse it for
/// every TurboQuant core that shares the same head dim and bit-width. This
/// also lets `pmetal-mlx` reuse the bridge's tables instead of recomputing.
pub fn beta_codebook(dim: usize, bits: u8) -> Arc<Vec<f32>> {
    static CACHE: OnceLock<BetaCodebookCache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache
        .lock()
        .expect("beta_codebook cache poisoned")
        .get(&(dim, bits))
    {
        return Arc::clone(hit);
    }
    let computed = Arc::new(build_beta_codebook(dim, bits));
    cache
        .lock()
        .expect("beta_codebook cache poisoned")
        .insert((dim, bits), Arc::clone(&computed));
    computed
}

pub(super) fn build_beta_codebook(dim: usize, bits: u8) -> Vec<f32> {
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

pub(super) fn transpose_square_matrix(matrix: &[f32], dim: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; matrix.len()];
    for row in 0..dim {
        for col in 0..dim {
            t[col * dim + row] = matrix[row * dim + col];
        }
    }
    t
}

/// Sample `dim` independent Rademacher (±1) signs into an `f32` vector.
///
/// Used to randomize the signed-FWHT rotation so it matches the statistical
/// guarantees of a Haar-random orthogonal matrix in `O(d log d)` time. Exposed
/// publicly so `pmetal-mlx` can build identical sign vectors for its own
/// TurboQuant core (deterministic seed → byte-identical signs across crates).
pub fn generate_rademacher_signs(dim: usize, rng: &mut StdRng) -> Vec<f32> {
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

/// Apply a randomized signed Walsh–Hadamard transform to one row in place.
///
/// Computes `output = D_left · H · D_right · input`, where `D_*` are diagonal
/// Rademacher (±1) sign matrices and `H` is the unnormalized Walsh–Hadamard
/// matrix. With the included `1/sqrt(d)` scale this produces a uniform random
/// orthogonal transform — the same statistical guarantee as a Haar-random
/// rotation for TurboQuant, but in `O(d log d)` time and `O(d)` memory instead
/// of the `O(d²)` cost of a dense matmul.
///
/// `values.len()` must be a power of two; both sign vectors must match.
pub fn signed_fwht_forward(values: &mut [f32], left_signs: &[f32], right_signs: &[f32]) {
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
pub(super) fn matmul_rows(matrix: &[f32], dim: usize, rows: &[f32]) -> Vec<f32> {
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

pub(super) fn l2_norm(values: &[f32]) -> f32 {
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
pub(super) fn matrix_to_inline_array(matrix: &[f32], dim: usize) -> Option<InlineArray> {
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
pub(super) fn inline_array_to_f32_vec(arr: &InlineArray, expected_len: usize) -> Option<Vec<f32>> {
    arr.reshape(&[expected_len as i32]).to_f32_vec(expected_len)
}

/// Convert a [B, H, S, D] InlineArray to a flat Vec<f32> in (B, S, H, D) row order.
///
/// Transposes to [B, S, H, D] so that `(batch, seq, head)` triplets are the
/// outer dimensions — matching the reference `array_rows_in_bshd_order`.
/// Uses `to_f32_vec` for a single bulk GPU→CPU copy.
pub(super) fn inline_array_to_bshd_rows(arr: &InlineArray) -> Result<Vec<f32>, String> {
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
///
/// The flatten→reshape pair after `transpose_axes` forces MLX to materialize
/// the data contiguously in the new [B, H, S, D] order. Without it, the
/// returned tensor is a lazy strided view: the *logical* shape is correct,
/// but `to_f32_vec` (raw memcpy of `data<float>()`) yields the pre-transpose
/// [B, S, H, D] memory order. Any caller that reads the buffer directly —
/// rather than consuming it via stride-aware MLX ops — would see scrambled
/// rows. This was a latent bug for the Mixed-precision dequantize path
/// (Uniform paths already produce contiguous output via GPU matmul).
pub(super) fn f32_rows_to_bhsd_array(
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
    let total = (batch * seq * heads * dim) as i32;
    InlineArray::from_f32_slice(rows, &[batch as i32, seq as i32, heads as i32, dim as i32])
        .transpose_axes(&[0, 2, 1, 3])
        .reshape(&[total])
        .reshape(&[batch as i32, heads as i32, seq as i32, dim as i32])
}
