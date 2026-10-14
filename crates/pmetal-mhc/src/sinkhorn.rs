//! Sinkhorn-Knopp projection onto the doubly stochastic matrices.
//!
//! A doubly stochastic matrix has non-negative entries and every row and
//! column summing to 1: the Birkhoff polytope. mHC constrains each layer's
//! residual mixing matrix to it, which keeps the layer's gain on the residual
//! stream at most 1 and closed under composition.
//!
//! # Algorithm
//!
//! The projection follows the released DeepSeek-V4 hyper-connection kernel
//! (`hc_split_sinkhorn`), which is the paper's `M⁽⁰⁾ = exp(H̃)`,
//! `M⁽ᵗ⁾ = T_r(T_c(M⁽ᵗ⁻¹⁾))` with the first step done as a softmax:
//!
//! ```text
//! M = softmax_rows(H̃) + ε                       (exp, then the first T_r)
//! M = M / (colsum(M) + ε)                       (T_c)
//! repeat iterations - 1 times:
//!     M = M / (rowsum(M) + ε)                   (T_r)
//!     M = M / (colsum(M) + ε)                   (T_c)
//! ```
//!
//! That is `iterations` row/column pairs in all (the paper's `t_max = 20`),
//! ending on a column step, so columns sum to 1 up to `ε` and rows converge
//! towards it. Starting from the row softmax rather than a bare `exp` is the
//! numerically stable form: `exp` overflows `f32` for logits above 88, while the
//! softmax subtracts the row maximum first and the first row normalization
//! would have divided that shift out anyway. The iteration count is fixed, with
//! no early exit, so the map is a smooth, deterministic function of its input
//! and [`sinkhorn_knopp_backward`] differentiates exactly what the forward ran.
//!
//! # Gradients
//!
//! Every step is a normalization `y = q / (sum(q) + ε)` along one axis, whose
//! adjoint is `g_q = (g_y − Σ g_y ⊙ y) / (sum(q) + ε)` along that axis, and the
//! softmax's is `g_z = p ⊙ (g_p − Σ g_p ⊙ p)`. The backward replays the forward,
//! keeping each step's output and denominators, and walks them in reverse: O(n²)
//! per iteration.
//!
//! # References
//!
//! - Sinkhorn & Knopp (1967): "Concerning nonnegative matrices and doubly stochastic matrices"
//! - arXiv:2512.24880: mHC paper

use crate::error::MhcError;
use ndarray::{Array2, Array3};

/// Configuration for the Sinkhorn-Knopp projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SinkhornConfig {
    /// Number of row/column normalization pairs. The paper uses 20.
    pub iterations: usize,

    /// Added to every element after the softmax and to every normalizing sum,
    /// so a row or column that underflows to zero cannot divide by zero.
    /// The release uses `1e-6`.
    pub epsilon: f32,
}

impl Default for SinkhornConfig {
    fn default() -> Self {
        Self {
            iterations: crate::DEFAULT_SINKHORN_ITERATIONS,
            epsilon: 1e-6,
        }
    }
}

impl SinkhornConfig {
    /// Check the configuration: at least one iteration, a positive finite `epsilon`.
    pub fn validate(&self) -> Result<(), MhcError> {
        if self.iterations == 0 {
            return Err(crate::MhcConfigError::InvalidSinkhornIterations(0).into());
        }
        if !(self.epsilon.is_finite() && self.epsilon > 0.0) {
            return Err(crate::MhcConfigError::InvalidEpsilon(self.epsilon).into());
        }
        Ok(())
    }
}

/// Result of a Sinkhorn-Knopp projection.
#[derive(Debug, Clone)]
pub struct SinkhornResult {
    /// The projected, approximately doubly stochastic matrix.
    pub matrix: Array2<f32>,

    /// Maximum deviation of a row sum from 1.
    pub row_error: f32,

    /// Maximum deviation of a column sum from 1.
    pub col_error: f32,
}

// ---------------------------------------------------------------------------
// Kernels on flat row-major `n × n` slices
// ---------------------------------------------------------------------------

/// `out = softmax_rows(z)`, with the row maximum subtracted.
fn softmax_rows(z: &[f32], n: usize, out: &mut [f32]) {
    for (zr, or) in z.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        let max = zr.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (o, &v) in or.iter_mut().zip(zr) {
            *o = (v - max).exp();
            sum += *o;
        }
        let inv = 1.0 / sum;
        or.iter_mut().for_each(|o| *o *= inv);
    }
}

/// Divide every row by `rowsum + eps`; the denominators go to `d` (`n` of them).
fn normalize_rows(m: &mut [f32], n: usize, eps: f32, d: &mut [f32]) {
    for (row, d_i) in m.chunks_exact_mut(n).zip(d.iter_mut()) {
        *d_i = row.iter().sum::<f32>() + eps;
        let inv = 1.0 / *d_i;
        row.iter_mut().for_each(|v| *v *= inv);
    }
}

/// Divide every column by `colsum + eps`; the denominators go to `d` (`n` of them).
fn normalize_cols(m: &mut [f32], n: usize, eps: f32, d: &mut [f32]) {
    d.fill(eps);
    for row in m.chunks_exact(n) {
        for (d_j, &v) in d.iter_mut().zip(row) {
            *d_j += v;
        }
    }
    for row in m.chunks_exact_mut(n) {
        for (v, &d_j) in row.iter_mut().zip(d.iter()) {
            *v /= d_j;
        }
    }
}

/// One saved normalization step, kept for the backward pass.
struct Step {
    /// The step's output `y`, `n × n`.
    y: Vec<f32>,
    /// The `n` denominators `sum + ε` it divided by.
    d: Vec<f32>,
    /// `true` for a row normalization, `false` for a column one.
    rows: bool,
}

/// Run the projection of `z` into `out`; with `trace`, record what the backward
/// pass needs.
fn project(
    z: &[f32],
    n: usize,
    cfg: &SinkhornConfig,
    out: &mut [f32],
    mut trace: Option<(&mut Vec<f32>, &mut Vec<Step>)>,
) {
    softmax_rows(z, n, out);
    if let Some((p, _)) = trace.as_mut() {
        p.clear();
        p.extend_from_slice(out);
    }
    out.iter_mut().for_each(|v| *v += cfg.epsilon);

    let mut d = vec![0.0f32; n];
    let mut step = |out: &mut [f32], rows: bool, d: &mut Vec<f32>| {
        if rows {
            normalize_rows(out, n, cfg.epsilon, d);
        } else {
            normalize_cols(out, n, cfg.epsilon, d);
        }
        if let Some((_, steps)) = trace.as_mut() {
            steps.push(Step {
                y: out.to_vec(),
                d: d.clone(),
                rows,
            });
        }
    };

    step(out, false, &mut d);
    for _ in 1..cfg.iterations {
        step(out, true, &mut d);
        step(out, false, &mut d);
    }
}

/// Gradient of `project` with respect to `z`, from the trace of its forward run.
fn project_backward(n: usize, p: &[f32], steps: &[Step], grad_out: &[f32], grad_z: &mut [f32]) {
    grad_z.copy_from_slice(grad_out);
    for s in steps.iter().rev() {
        if s.rows {
            for ((g_row, y_row), &d_i) in grad_z
                .chunks_exact_mut(n)
                .zip(s.y.chunks_exact(n))
                .zip(&s.d)
            {
                let dot: f32 = g_row.iter().zip(y_row).map(|(g, y)| g * y).sum();
                g_row.iter_mut().for_each(|g| *g = (*g - dot) / d_i);
            }
        } else {
            let mut dot = vec![0.0f32; n];
            for (g_row, y_row) in grad_z.chunks_exact(n).zip(s.y.chunks_exact(n)) {
                for ((acc, g), y) in dot.iter_mut().zip(g_row).zip(y_row) {
                    *acc += g * y;
                }
            }
            for g_row in grad_z.chunks_exact_mut(n) {
                for ((g, &acc), &d_j) in g_row.iter_mut().zip(&dot).zip(&s.d) {
                    *g = (*g - acc) / d_j;
                }
            }
        }
    }
    // Softmax: g_z = p ⊙ (g − Σ g ⊙ p), per row. (The `+ ε` before the first
    // normalization is a constant shift and passes the gradient through.)
    for (g_row, p_row) in grad_z.chunks_exact_mut(n).zip(p.chunks_exact(n)) {
        let dot: f32 = g_row.iter().zip(p_row).map(|(g, p)| g * p).sum();
        g_row
            .iter_mut()
            .zip(p_row)
            .for_each(|(g, &p)| *g = p * (*g - dot));
    }
}

fn square_size(shape: &[usize], what: &str) -> usize {
    let n = shape[shape.len() - 1];
    assert!(
        shape[shape.len() - 2] == n,
        "{what} must be square, got {}x{}",
        shape[shape.len() - 2],
        n
    );
    n
}

fn assert_config(cfg: &SinkhornConfig) {
    if let Err(e) = cfg.validate() {
        panic!("invalid Sinkhorn configuration: {e}");
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Project a matrix of logits onto the doubly stochastic manifold.
///
/// See the [module docs](self) for the exact iteration.
///
/// # Arguments
///
/// * `h_tilde` - Unconstrained logits `H̃`. Shape `[n, n]`, finite.
/// * `config` - Iteration count and epsilon.
///
/// # Panics
///
/// If `h_tilde` is not square or `config` is invalid
/// ([`SinkhornConfig::validate`]).
pub fn sinkhorn_knopp(h_tilde: &Array2<f32>, config: &SinkhornConfig) -> SinkhornResult {
    assert_config(config);
    let n = square_size(h_tilde.shape(), "Sinkhorn input");
    let z = h_tilde.as_standard_layout();
    let mut out = Array2::<f32>::zeros((n, n));
    project(
        z.as_slice().expect("standard layout is contiguous"),
        n,
        config,
        out.as_slice_mut().expect("fresh array is contiguous"),
        None,
    );
    let (row_error, col_error) = compute_errors(&out);
    SinkhornResult {
        matrix: out,
        row_error,
        col_error,
    }
}

/// Batched [`sinkhorn_knopp`].
///
/// # Arguments
///
/// * `h_tilde_batch` - Batch of logit matrices. Shape `[batch, n, n]`
/// * `config` - Iteration count and epsilon.
///
/// # Panics
///
/// If the matrices are not square or `config` is invalid.
pub fn sinkhorn_knopp_batch(h_tilde_batch: &Array3<f32>, config: &SinkhornConfig) -> Array3<f32> {
    assert_config(config);
    let n = square_size(h_tilde_batch.shape(), "Sinkhorn inputs");
    let z = h_tilde_batch.as_standard_layout();
    let z = z.as_slice().expect("standard layout is contiguous");
    let mut out = Array3::<f32>::zeros(h_tilde_batch.raw_dim());
    let o = out.as_slice_mut().expect("fresh array is contiguous");
    for (zb, ob) in z.chunks_exact(n * n).zip(o.chunks_exact_mut(n * n)) {
        project(zb, n, config, ob, None);
    }
    out
}

/// Backward pass through [`sinkhorn_knopp`]: the gradient of a loss with
/// respect to the logits `H̃`, given its gradient with respect to the output.
///
/// This is the exact adjoint of the forward (same `iterations`, same `epsilon`),
/// so it matches a finite-difference derivative of [`sinkhorn_knopp`] to
/// `f32` rounding.
///
/// # Arguments
///
/// * `h_tilde` - The input the forward pass was given. Shape `[n, n]`
/// * `grad_output` - Gradient with respect to the projected matrix. Shape `[n, n]`
/// * `config` - The configuration the forward pass used.
///
/// # Panics
///
/// If the shapes differ, `h_tilde` is not square or `config` is invalid.
pub fn sinkhorn_knopp_backward(
    h_tilde: &Array2<f32>,
    grad_output: &Array2<f32>,
    config: &SinkhornConfig,
) -> Array2<f32> {
    assert_config(config);
    let n = square_size(h_tilde.shape(), "Sinkhorn input");
    assert_eq!(
        h_tilde.shape(),
        grad_output.shape(),
        "grad_output must match the Sinkhorn input's shape"
    );
    let z = h_tilde.as_standard_layout();
    let g = grad_output.as_standard_layout();
    let mut grad_z = Array2::<f32>::zeros((n, n));
    backward_one(
        z.as_slice().expect("standard layout is contiguous"),
        g.as_slice().expect("standard layout is contiguous"),
        n,
        config,
        grad_z.as_slice_mut().expect("fresh array is contiguous"),
    );
    grad_z
}

/// Batched [`sinkhorn_knopp_backward`]. Shapes `[batch, n, n]`.
pub fn sinkhorn_knopp_backward_batch(
    h_tilde_batch: &Array3<f32>,
    grad_output: &Array3<f32>,
    config: &SinkhornConfig,
) -> Array3<f32> {
    assert_config(config);
    let n = square_size(h_tilde_batch.shape(), "Sinkhorn inputs");
    assert_eq!(
        h_tilde_batch.shape(),
        grad_output.shape(),
        "grad_output must match the Sinkhorn inputs' shape"
    );
    let z = h_tilde_batch.as_standard_layout();
    let g = grad_output.as_standard_layout();
    let (z, g) = (
        z.as_slice().expect("standard layout is contiguous"),
        g.as_slice().expect("standard layout is contiguous"),
    );
    let mut out = Array3::<f32>::zeros(h_tilde_batch.raw_dim());
    let o = out.as_slice_mut().expect("fresh array is contiguous");
    for ((zb, gb), ob) in z
        .chunks_exact(n * n)
        .zip(g.chunks_exact(n * n))
        .zip(o.chunks_exact_mut(n * n))
    {
        backward_one(zb, gb, n, config, ob);
    }
    out
}

fn backward_one(z: &[f32], g: &[f32], n: usize, cfg: &SinkhornConfig, grad_z: &mut [f32]) {
    let mut out = vec![0.0f32; n * n];
    let mut p = Vec::with_capacity(n * n);
    let mut steps = Vec::with_capacity(2 * cfg.iterations);
    project(z, n, cfg, &mut out, Some((&mut p, &mut steps)));
    project_backward(n, &p, &steps, g, grad_z);
}

/// Maximum deviation of the row sums and of the column sums from 1.
fn compute_errors(m: &Array2<f32>) -> (f32, f32) {
    let row_error = m
        .rows()
        .into_iter()
        .map(|r| (r.sum() - 1.0).abs())
        .fold(0.0f32, f32::max);
    let col_error = m
        .columns()
        .into_iter()
        .map(|c| (c.sum() - 1.0).abs())
        .fold(0.0f32, f32::max);
    (row_error, col_error)
}

/// Verify that a matrix is doubly stochastic.
///
/// # Arguments
///
/// * `m` - Matrix to verify
/// * `tolerance` - Maximum allowed deviation from 1.0 for row/column sums, and
///   from 0.0 for a negative entry
///
/// # Returns
///
/// True if the matrix is square, non-negative and its row and column sums are
/// within `tolerance` of 1. `NaN` entries make it false.
pub fn is_doubly_stochastic(m: &Array2<f32>, tolerance: f32) -> bool {
    if m.nrows() != m.ncols() {
        return false;
    }
    if m.iter().any(|&x| x.is_nan() || x < -tolerance) {
        return false;
    }
    let (row_error, col_error) = compute_errors(m);
    row_error <= tolerance && col_error <= tolerance
}

/// The Amax gain magnitude of the forward signal: the largest absolute row sum
/// of `m`, the worst-case amplification of a residual-stream component.
pub fn amax_gain_forward(m: &Array2<f32>) -> f32 {
    m.rows()
        .into_iter()
        .map(|r| r.iter().map(|x| x.abs()).sum::<f32>())
        .fold(0.0f32, f32::max)
}

/// The Amax gain magnitude of the backward gradient: the largest absolute
/// column sum of `m`, the worst-case amplification of a gradient component.
pub fn amax_gain_backward(m: &Array2<f32>) -> f32 {
    m.columns()
        .into_iter()
        .map(|c| c.iter().map(|x| x.abs()).sum::<f32>())
        .fold(0.0f32, f32::max)
}

/// Compose mappings across layers: the ordered product `M_1 M_2 ... M_k`.
///
/// Products of doubly stochastic matrices are doubly stochastic, so this is
/// the effective residual mapping of the stack and its [`amax_gain_forward`]
/// stays near 1 however deep it is.
///
/// # Errors
///
/// [`MhcError::EmptyComposite`] for an empty list and
/// [`MhcError::ShapeMismatch`] if the matrices are not all the same square size.
pub fn composite_mapping(matrices: &[Array2<f32>]) -> Result<Array2<f32>, MhcError> {
    let Some(first) = matrices.first() else {
        return Err(MhcError::EmptyComposite);
    };
    let n = first.nrows();
    for m in matrices {
        if m.shape() != [n, n] {
            return Err(MhcError::shape(
                "composite matrix",
                format!("[{n}, {n}]"),
                m.shape(),
            ));
        }
    }
    let mut result = Array2::eye(n);
    for m in matrices {
        result = result.dot(m);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn cfg(iterations: usize) -> SinkhornConfig {
        SinkhornConfig {
            iterations,
            epsilon: 1e-6,
        }
    }

    fn logits3() -> Array2<f32> {
        array![[0.5, -1.0, 2.0], [1.5, 0.25, -0.75], [-2.0, 0.8, 1.1]]
    }

    /// Values from a float64 evaluation of the release's `hc_split_sinkhorn`
    /// recipe (row softmax, `+ε`, column step, then row/column pairs).
    #[test]
    fn matches_the_release_recipe_on_hand_computed_values() {
        let expected: [(usize, [[f64; 3]; 3]); 3] = [
            (
                1,
                [
                    [0.190749726, 0.059283127, 0.552741337],
                    [0.781798168, 0.311981133, 0.053278921],
                    [0.027451017, 0.628734224, 0.393979039],
                ],
            ),
            (
                2,
                [
                    [0.251356824, 0.078184431, 0.620222777],
                    [0.720990441, 0.287955653, 0.041839770],
                    [0.027651678, 0.633858857, 0.337936553],
                ],
            ),
            (
                20,
                [
                    [0.278624043, 0.084663294, 0.636711663],
                    [0.692566163, 0.270211820, 0.037221017],
                    [0.028808794, 0.645123886, 0.326066320],
                ],
            ),
        ];
        for (iters, want) in expected {
            let got = sinkhorn_knopp(&logits3(), &cfg(iters)).matrix;
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (f64::from(got[[i, j]]) - want[i][j]).abs() < 2e-6,
                        "iterations {iters} [{i},{j}]: {} vs {}",
                        got[[i, j]],
                        want[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn result_is_doubly_stochastic_with_reported_errors() {
        let r = sinkhorn_knopp(&logits3(), &cfg(20));
        assert!(is_doubly_stochastic(&r.matrix, 1e-3));
        assert!(r.col_error < 1e-5 && r.row_error < 1e-3, "{r:?}");
        assert!(r.matrix.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn large_logits_do_not_overflow() {
        // exp(100) is inf in f32; a bare exp-then-normalize gives NaN here.
        let z = array![[100.0, -100.0], [-100.0, 100.0]];
        let m = sinkhorn_knopp(&z, &cfg(20)).matrix;
        assert!(m.iter().all(|v| v.is_finite()), "{m:?}");
        assert!((m[[0, 0]] - 1.0).abs() < 1e-5 && m[[0, 1]] < 1e-5);
        let z = array![[1e30, 0.0], [0.0, 1e30]];
        assert!(
            sinkhorn_knopp(&z, &cfg(5))
                .matrix
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn uniform_logits_give_the_uniform_matrix() {
        let m = sinkhorn_knopp(&Array2::from_elem((4, 4), 3.0), &cfg(20)).matrix;
        for &v in &m {
            assert!((v - 0.25).abs() < 1e-6, "{v}");
        }
    }

    #[test]
    fn batch_matches_the_single_matrix_path() {
        let batch = Array3::from_shape_fn((5, 4, 4), |(b, i, j)| {
            ((b * 7 + i * 3 + j * 5) % 11) as f32 * 0.4 - 2.0
        });
        let out = sinkhorn_knopp_batch(&batch, &cfg(20));
        let grad = Array3::from_shape_fn((5, 4, 4), |(b, i, j)| ((b + 2 * i + j) % 5) as f32 - 2.0);
        let gout = sinkhorn_knopp_backward_batch(&batch, &grad, &cfg(20));
        for b in 0..5 {
            let z = batch.index_axis(ndarray::Axis(0), b).to_owned();
            let g = grad.index_axis(ndarray::Axis(0), b).to_owned();
            assert_eq!(
                sinkhorn_knopp(&z, &cfg(20)).matrix,
                out.index_axis(ndarray::Axis(0), b)
            );
            assert_eq!(
                sinkhorn_knopp_backward(&z, &g, &cfg(20)),
                gout.index_axis(ndarray::Axis(0), b)
            );
        }
    }

    #[test]
    fn composite_closure_and_amax_gain() {
        let mats: Vec<Array2<f32>> = (0..60)
            .map(|k| {
                let z = Array2::from_shape_fn((4, 4), |(i, j)| {
                    (((k * 13 + i * 5 + j * 3) % 17) as f32) * 0.25 - 2.0
                });
                sinkhorn_knopp(&z, &cfg(20)).matrix
            })
            .collect();
        let c = composite_mapping(&mats).unwrap();
        assert!(is_doubly_stochastic(&c, 1e-3));
        assert!(amax_gain_forward(&c) < 1.0 + 1e-3);
        assert!(amax_gain_backward(&c) < 1.0 + 1e-3);
    }

    #[test]
    fn composite_rejects_empty_and_mismatched() {
        assert!(matches!(
            composite_mapping(&[]),
            Err(MhcError::EmptyComposite)
        ));
        let a = Array2::<f32>::eye(3);
        let b = Array2::<f32>::eye(4);
        assert!(matches!(
            composite_mapping(&[a, b]),
            Err(MhcError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn invalid_configs_are_reported() {
        assert!(cfg(0).validate().is_err());
        assert!(
            SinkhornConfig {
                iterations: 3,
                epsilon: 0.0
            }
            .validate()
            .is_err()
        );
        assert!(cfg(1).validate().is_ok());
    }
}
