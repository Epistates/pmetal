//! Pure-Rust math primitives for the TurboQuant cache.
//!
//! No external dependencies beyond `rand` and InlineArray marshalling —
//! every function here is deterministic in its inputs so
//! call sites can rely on byte-identical results across crates (notably
//! `pmetal-mlx` mirrors several of these).

use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};

use rand::{RngExt, rngs::StdRng};

type BetaCodebookKey = (usize, u8);
type BetaCodebookCache = Mutex<HashMap<BetaCodebookKey, Arc<Vec<f32>>>>;

/// Seed of the rotation matrices: fixed, so a cache built today decodes the
/// same everywhere.
const ROTATION_SEED: u64 = 0x5442_5155_414e_544d;

/// Lloyd-Max iteration cap. Convergence is linear and slows as the codebook
/// grows: 8 bits needs a few thousand sweeps, each a few hundred lookups.
const LLOYD_MAX_ITERS: usize = 20_000;
/// Lloyd-Max convergence threshold, in standard deviations of a coordinate.
const LLOYD_MAX_TOLERANCE: f64 = 1e-9;
/// Grid cells for the Beta-density quadrature.
const LLOYD_GRID_POINTS: usize = 1 << 20;
/// Half-width of the quadrature grid in standard deviations of a coordinate.
const LLOYD_SUPPORT_SIGMAS: f64 = 9.0;

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

/// Cumulative mass and first moment of the coordinate density on a fine grid,
/// with linear interpolation inside a cell, so a Lloyd-Max step costs one
/// lookup per cell boundary however many centroids there are.
struct BetaMoments {
    lo: f64,
    step: f64,
    /// `mass[i]`, `first[i]`: integrals over `[lo, lo + i * step)`.
    mass: Vec<f64>,
    first: Vec<f64>,
}

impl BetaMoments {
    fn new(dim: usize) -> Self {
        // The tails past nine standard deviations carry less than 1e-17 of
        // the mass, far below the resolution of an 8-bit codebook.
        let half = (LLOYD_SUPPORT_SIGMAS / (dim as f64).sqrt()).min(1.0);
        let alpha = ((dim as f64) - 3.0) / 2.0;
        let step = 2.0 * half / (LLOYD_GRID_POINTS as f64);
        let mut mass = Vec::with_capacity(LLOYD_GRID_POINTS + 1);
        let mut first = Vec::with_capacity(LLOYD_GRID_POINTS + 1);
        let (mut m, mut f) = (0.0f64, 0.0f64);
        mass.push(m);
        first.push(f);
        for i in 0..LLOYD_GRID_POINTS {
            let x = -half + ((i as f64) + 0.5) * step;
            let w = (1.0 - x * x).max(0.0).powf(alpha);
            m += w;
            f += w * x;
            mass.push(m);
            first.push(f);
        }
        // Normalise to a unit total mass.
        let total = m;
        for v in mass.iter_mut().chain(first.iter_mut()) {
            *v /= total;
        }
        Self {
            lo: -half,
            step,
            mass,
            first,
        }
    }

    /// `(mass, first moment)` of `[lo, x)`.
    fn up_to(&self, x: f64) -> (f64, f64) {
        let pos = ((x - self.lo) / self.step).clamp(0.0, (self.mass.len() - 1) as f64);
        let cell = (pos.floor() as usize).min(self.mass.len() - 2);
        let frac = pos - cell as f64;
        (
            self.mass[cell] + frac * (self.mass[cell + 1] - self.mass[cell]),
            self.first[cell] + frac * (self.first[cell + 1] - self.first[cell]),
        )
    }

    /// Smallest grid position whose cumulative mass reaches `target`.
    fn quantile(&self, target: f64) -> f64 {
        let cell = self
            .mass
            .partition_point(|&m| m < target)
            .clamp(1, self.mass.len() - 1);
        let (m0, m1) = (self.mass[cell - 1], self.mass[cell]);
        let frac = if m1 > m0 {
            (target - m0) / (m1 - m0)
        } else {
            0.5
        };
        self.lo + ((cell - 1) as f64 + frac) * self.step
    }
}

pub(super) fn build_beta_codebook(dim: usize, bits: u8) -> Vec<f32> {
    let centroid_count = 1usize << bits;
    let moments = BetaMoments::new(dim);

    // Start from the centroids of equal-probability cells.
    let mut centroids = Vec::with_capacity(centroid_count);
    for bucket in 0..centroid_count {
        let left = moments.quantile(bucket as f64 / centroid_count as f64);
        let right = moments.quantile((bucket as f64 + 1.0) / centroid_count as f64);
        centroids.push(0.5 * (left + right));
    }

    // Lloyd-Max: move every centroid to the mean of its cell until none moves
    // by more than a millionth of a coordinate's standard deviation. The
    // density is symmetric, so the codebook is kept symmetric.
    let tolerance = LLOYD_MAX_TOLERANCE / (dim as f64).sqrt();
    for _ in 0..LLOYD_MAX_ITERS {
        let mut boundaries = Vec::with_capacity(centroid_count + 1);
        boundaries.push(f64::NEG_INFINITY);
        boundaries.extend(centroids.windows(2).map(|pair| 0.5 * (pair[0] + pair[1])));
        boundaries.push(f64::INFINITY);

        let mut max_change = 0.0f64;
        let mut updated = centroids.clone();
        for bucket in 0..centroid_count {
            let (m_left, f_left) = moments.up_to(boundaries[bucket]);
            let (m_right, f_right) = moments.up_to(boundaries[bucket + 1]);
            let cell_mass = m_right - m_left;
            if cell_mass > 0.0 {
                updated[bucket] = (f_right - f_left) / cell_mass;
            }
        }
        for bucket in 0..centroid_count / 2 {
            let mirror = centroid_count - 1 - bucket;
            let symmetric = 0.5 * (updated[mirror] - updated[bucket]);
            updated[bucket] = -symmetric;
            updated[mirror] = symmetric;
        }
        for (new, old) in updated.iter().zip(&centroids) {
            max_change = max_change.max((new - old).abs());
        }
        centroids = updated;
        if max_change < tolerance {
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

/// The fixed Haar-random orthogonal matrix `Π` for vectors of `dim`
/// coordinates, row-major `[dim, dim]` f32. Deterministic in `dim`; computed
/// once per process.
pub(crate) fn rotation_matrix(dim: usize) -> Arc<Vec<f32>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<Vec<f32>>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("rotation cache poisoned");
    Arc::clone(guard.entry(dim).or_insert_with(|| {
        let mut rng =
            <StdRng as rand::SeedableRng>::seed_from_u64(ROTATION_SEED ^ ((dim as u64) << 32));
        Arc::new(generate_random_orthogonal(dim, &mut rng))
    }))
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
