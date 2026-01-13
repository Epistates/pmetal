//! Sinkhorn-Knopp algorithm for doubly stochastic matrix projection.
//!
//! This module implements the Sinkhorn-Knopp algorithm that projects matrices
//! onto the Birkhoff polytope (set of doubly stochastic matrices).
//!
//! A doubly stochastic matrix has:
//! - All entries non-negative
//! - All rows sum to 1
//! - All columns sum to 1
//!
//! # Algorithm
//!
//! Given an input matrix H̃, the algorithm:
//! 1. Exponentiates to ensure positivity: M⁽⁰⁾ = exp(H̃)
//! 2. Iteratively alternates row and column normalization:
//!    M⁽ᵗ⁾ = T_r(T_c(M⁽ᵗ⁻¹⁾))
//! 3. Converges to a doubly stochastic matrix as t → ∞
//!
//! # References
//!
//! - Sinkhorn & Knopp (1967): "Concerning nonnegative matrices and doubly stochastic matrices"
//! - arXiv:2512.24880: mHC paper

use ndarray::{Array2, Array3, Axis};

/// Configuration for Sinkhorn-Knopp algorithm.
#[derive(Debug, Clone, Copy)]
pub struct SinkhornConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,

    /// Epsilon for numerical stability.
    pub epsilon: f32,

    /// Early stopping tolerance (if row/col sums are within this of 1.0).
    pub tolerance: f32,
}

impl Default for SinkhornConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            epsilon: 1e-8,
            tolerance: 1e-6,
        }
    }
}

/// Result of Sinkhorn-Knopp iteration.
#[derive(Debug, Clone)]
pub struct SinkhornResult {
    /// The projected doubly stochastic matrix.
    pub matrix: Array2<f32>,

    /// Number of iterations performed.
    pub iterations: usize,

    /// Whether convergence was achieved.
    pub converged: bool,

    /// Maximum deviation from row sum = 1.
    pub row_error: f32,

    /// Maximum deviation from column sum = 1.
    pub col_error: f32,
}

/// Apply Sinkhorn-Knopp algorithm to project a matrix onto the doubly stochastic manifold.
///
/// # Arguments
///
/// * `h_tilde` - Input matrix (unconstrained). Shape: [n, n]
/// * `config` - Algorithm configuration
///
/// # Returns
///
/// Doubly stochastic matrix where rows and columns sum to 1.
pub fn sinkhorn_knopp(h_tilde: &Array2<f32>, config: &SinkhornConfig) -> SinkhornResult {
    let n = h_tilde.nrows();
    assert_eq!(n, h_tilde.ncols(), "Matrix must be square");

    // Step 1: Exponentiate to make all entries positive
    let mut m: Array2<f32> = h_tilde.mapv(|x| x.exp());

    let mut converged = false;
    let mut iterations = 0;

    // Step 2: Iterate row and column normalization
    for t in 0..config.max_iterations {
        iterations = t + 1;

        // Row normalization: M[i,:] = M[i,:] / sum(M[i,:])
        for i in 0..n {
            let row_sum: f32 = m.row(i).iter().sum();
            let inv_sum = 1.0 / (row_sum + config.epsilon);
            for j in 0..n {
                m[[i, j]] *= inv_sum;
            }
        }

        // Column normalization: M[:,j] = M[:,j] / sum(M[:,j])
        for j in 0..n {
            let col_sum: f32 = m.column(j).iter().sum();
            let inv_sum = 1.0 / (col_sum + config.epsilon);
            for i in 0..n {
                m[[i, j]] *= inv_sum;
            }
        }

        // Check convergence
        let (row_error, col_error) = compute_errors(&m);
        if row_error < config.tolerance && col_error < config.tolerance {
            converged = true;
            break;
        }
    }

    let (row_error, col_error) = compute_errors(&m);

    SinkhornResult {
        matrix: m,
        iterations,
        converged,
        row_error,
        col_error,
    }
}

/// Batched Sinkhorn-Knopp for multiple matrices.
///
/// # Arguments
///
/// * `h_tilde_batch` - Batch of input matrices. Shape: [batch, n, n]
/// * `config` - Algorithm configuration
///
/// # Returns
///
/// Batch of doubly stochastic matrices.
pub fn sinkhorn_knopp_batch(h_tilde_batch: &Array3<f32>, config: &SinkhornConfig) -> Array3<f32> {
    let batch_size = h_tilde_batch.shape()[0];
    let n = h_tilde_batch.shape()[1];
    assert_eq!(n, h_tilde_batch.shape()[2], "Matrices must be square");

    let mut output = Array3::zeros((batch_size, n, n));

    // Process each matrix in the batch
    // Note: In production, this would be parallelized or done in a GPU kernel
    for b in 0..batch_size {
        let h_tilde = h_tilde_batch.slice(ndarray::s![b, .., ..]).to_owned();
        let result = sinkhorn_knopp(&h_tilde, config);
        output
            .slice_mut(ndarray::s![b, .., ..])
            .assign(&result.matrix);
    }

    output
}

/// Compute row and column sum errors.
fn compute_errors(m: &Array2<f32>) -> (f32, f32) {
    let n = m.nrows();

    // Maximum deviation from row sum = 1
    let row_error = (0..n)
        .map(|i| (m.row(i).iter().sum::<f32>() - 1.0).abs())
        .fold(0.0f32, f32::max);

    // Maximum deviation from column sum = 1
    let col_error = (0..n)
        .map(|j| (m.column(j).iter().sum::<f32>() - 1.0).abs())
        .fold(0.0f32, f32::max);

    (row_error, col_error)
}

/// Verify that a matrix is doubly stochastic.
///
/// # Arguments
///
/// * `m` - Matrix to verify
/// * `tolerance` - Maximum allowed deviation from 1.0 for row/column sums
///
/// # Returns
///
/// True if the matrix is doubly stochastic within tolerance.
pub fn is_doubly_stochastic(m: &Array2<f32>, tolerance: f32) -> bool {
    let n = m.nrows();
    if n != m.ncols() {
        return false;
    }

    // Check non-negativity
    if m.iter().any(|&x| x < -tolerance) {
        return false;
    }

    // Check row sums
    for i in 0..n {
        let row_sum: f32 = m.row(i).iter().sum();
        if (row_sum - 1.0).abs() > tolerance {
            return false;
        }
    }

    // Check column sums
    for j in 0..n {
        let col_sum: f32 = m.column(j).iter().sum();
        if (col_sum - 1.0).abs() > tolerance {
            return false;
        }
    }

    true
}

/// Compute the Amax Gain Magnitude for forward signal.
///
/// This is the maximum absolute row sum, capturing worst-case signal amplification.
pub fn amax_gain_forward(m: &Array2<f32>) -> f32 {
    let n = m.nrows();
    (0..n)
        .map(|i| m.row(i).iter().map(|x| x.abs()).sum::<f32>())
        .fold(0.0f32, f32::max)
}

/// Compute the Amax Gain Magnitude for backward gradient.
///
/// This is the maximum absolute column sum, capturing worst-case gradient amplification.
pub fn amax_gain_backward(m: &Array2<f32>) -> f32 {
    let n = m.ncols();
    (0..n)
        .map(|j| m.column(j).iter().map(|x| x.abs()).sum::<f32>())
        .fold(0.0f32, f32::max)
}

/// Compute composite mapping across multiple layers.
///
/// # Arguments
///
/// * `matrices` - List of H^res matrices from consecutive layers
///
/// # Returns
///
/// The product ∏ H^res, which should remain doubly stochastic.
pub fn composite_mapping(matrices: &[Array2<f32>]) -> Array2<f32> {
    if matrices.is_empty() {
        panic!("Cannot compute composite of empty list");
    }

    let n = matrices[0].nrows();
    let mut result = Array2::eye(n);

    for m in matrices {
        result = result.dot(m);
    }

    result
}

/// Backward pass through Sinkhorn-Knopp.
///
/// Computes the gradient of the loss with respect to the input H̃
/// given the gradient with respect to the output M.
///
/// # Arguments
///
/// * `h_tilde` - Original input to Sinkhorn-Knopp
/// * `grad_output` - Gradient w.r.t. the output doubly stochastic matrix
/// * `config` - Algorithm configuration
///
/// # Returns
///
/// Gradient w.r.t. the input H̃.
pub fn sinkhorn_knopp_backward(
    h_tilde: &Array2<f32>,
    grad_output: &Array2<f32>,
    config: &SinkhornConfig,
) -> Array2<f32> {
    let n = h_tilde.nrows();

    // We need to backprop through the Sinkhorn iterations.
    // This is done by:
    // 1. Recomputing the forward pass, storing intermediates
    // 2. Backpropping through each iteration in reverse

    // Forward pass with intermediate storage
    let mut intermediates: Vec<Array2<f32>> = Vec::with_capacity(config.max_iterations + 1);

    // Initial: M⁰ = exp(H̃)
    let m_0: Array2<f32> = h_tilde.mapv(|x| x.exp());
    intermediates.push(m_0.clone());

    let mut m = m_0;

    for _ in 0..config.max_iterations {
        // Row normalization
        let mut m_row = m.clone();
        for i in 0..n {
            let row_sum: f32 = m_row.row(i).iter().sum();
            let inv_sum = 1.0 / (row_sum + config.epsilon);
            for j in 0..n {
                m_row[[i, j]] *= inv_sum;
            }
        }

        // Column normalization
        let mut m_col = m_row.clone();
        for j in 0..n {
            let col_sum: f32 = m_col.column(j).iter().sum();
            let inv_sum = 1.0 / (col_sum + config.epsilon);
            for i in 0..n {
                m_col[[i, j]] *= inv_sum;
            }
        }

        m = m_col;
        intermediates.push(m.clone());
    }

    // Backward pass
    let mut grad = grad_output.clone();

    for t in (0..config.max_iterations).rev() {
        let m_prev = &intermediates[t];

        // Backward through column normalization
        let mut m_row = m_prev.clone();
        for i in 0..n {
            let row_sum: f32 = m_row.row(i).iter().sum();
            let inv_sum = 1.0 / (row_sum + config.epsilon);
            for j in 0..n {
                m_row[[i, j]] *= inv_sum;
            }
        }

        let mut grad_m_row: Array2<f32> = Array2::zeros((n, n));
        for j in 0..n {
            let col_sum: f32 = m_row.column(j).iter().sum();
            let inv_sum = 1.0 / (col_sum + config.epsilon);
            let inv_sum_sq = inv_sum * inv_sum;

            for i in 0..n {
                let grad_ij = grad[[i, j]];
                // Gradient from direct term
                grad_m_row[[i, j]] += grad_ij * inv_sum;
                // Gradient from normalization term
                for k in 0..n {
                    grad_m_row[[k, j]] -= grad_ij * m_row[[k, j]] * m_row[[i, j]] * inv_sum_sq;
                }
            }
        }

        // Backward through row normalization
        let mut grad_m_prev: Array2<f32> = Array2::zeros((n, n));
        for i in 0..n {
            let row_sum: f32 = m_prev.row(i).iter().sum();
            let inv_sum = 1.0 / (row_sum + config.epsilon);
            let inv_sum_sq = inv_sum * inv_sum;

            for j in 0..n {
                let grad_ij = grad_m_row[[i, j]];
                // Gradient from direct term
                grad_m_prev[[i, j]] += grad_ij * inv_sum;
                // Gradient from normalization term
                for k in 0..n {
                    grad_m_prev[[i, k]] -= grad_ij * m_prev[[i, k]] * m_prev[[i, j]] * inv_sum_sq;
                }
            }
        }

        grad = grad_m_prev;
    }

    // Backward through exp: d/dH̃[exp(H̃)] = exp(H̃) ⊙ grad
    let m_0 = &intermediates[0];
    let grad_h_tilde = m_0 * &grad;

    grad_h_tilde
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sinkhorn_convergence() {
        let h_tilde = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let config = SinkhornConfig::default();

        let result = sinkhorn_knopp(&h_tilde, &config);

        assert!(result.converged);
        assert!(is_doubly_stochastic(&result.matrix, 1e-4));
    }

    #[test]
    fn test_sinkhorn_non_negativity() {
        let h_tilde = array![[-1.0, -2.0], [3.0, 4.0]];
        let config = SinkhornConfig::default();

        let result = sinkhorn_knopp(&h_tilde, &config);

        // All entries should be non-negative
        assert!(result.matrix.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_sinkhorn_row_sums() {
        let h_tilde = array![
            [0.5, 1.5, 2.0, 0.3],
            [1.0, 2.0, 3.0, 1.5],
            [0.2, 0.8, 1.2, 0.9],
            [2.1, 1.3, 0.7, 1.8]
        ];
        let config = SinkhornConfig::default();

        let result = sinkhorn_knopp(&h_tilde, &config);

        // Check row sums
        for i in 0..4 {
            let row_sum: f32 = result.matrix.row(i).iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-4, "Row {} sum: {}", i, row_sum);
        }
    }

    #[test]
    fn test_sinkhorn_col_sums() {
        let h_tilde = array![
            [0.5, 1.5, 2.0, 0.3],
            [1.0, 2.0, 3.0, 1.5],
            [0.2, 0.8, 1.2, 0.9],
            [2.1, 1.3, 0.7, 1.8]
        ];
        let config = SinkhornConfig::default();

        let result = sinkhorn_knopp(&h_tilde, &config);

        // Check column sums
        for j in 0..4 {
            let col_sum: f32 = result.matrix.column(j).iter().sum();
            assert!(
                (col_sum - 1.0).abs() < 1e-4,
                "Column {} sum: {}",
                j,
                col_sum
            );
        }
    }

    #[test]
    fn test_compositional_closure() {
        // Property: Product of doubly stochastic matrices is doubly stochastic
        let config = SinkhornConfig::default();
        let mut rng = rand::thread_rng();

        let matrices: Vec<Array2<f32>> = (0..10)
            .map(|_| {
                let h_tilde = Array2::from_shape_fn((4, 4), |_| {
                    use rand::Rng;
                    rng.gen_range(-2.0..2.0)
                });
                sinkhorn_knopp(&h_tilde, &config).matrix
            })
            .collect();

        let composite = composite_mapping(&matrices);

        // Composite should still be doubly stochastic
        assert!(
            is_doubly_stochastic(&composite, 1e-3),
            "Composite is not doubly stochastic"
        );
    }

    #[test]
    fn test_amax_gain_bounded() {
        // Doubly stochastic matrices should have Amax gain ≤ 1 (approximately)
        let h_tilde = array![
            [1.0, 2.0, 0.5, 1.5],
            [0.8, 1.2, 2.1, 0.9],
            [1.5, 0.7, 1.3, 2.0],
            [2.0, 1.8, 0.6, 0.4]
        ];
        let config = SinkhornConfig::default();

        let result = sinkhorn_knopp(&h_tilde, &config);

        let forward_gain = amax_gain_forward(&result.matrix);
        let backward_gain = amax_gain_backward(&result.matrix);

        // For a doubly stochastic matrix, Amax gain should be close to 1
        assert!(forward_gain <= 1.0 + 1e-4, "Forward gain: {}", forward_gain);
        assert!(
            backward_gain <= 1.0 + 1e-4,
            "Backward gain: {}",
            backward_gain
        );
    }

    #[test]
    fn test_spectral_norm_bounded() {
        // The spectral norm of a doubly stochastic matrix is bounded by 1
        let h_tilde = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 4.0, 3.0],
            [3.0, 4.0, 1.0, 2.0],
            [4.0, 3.0, 2.0, 1.0]
        ];
        let config = SinkhornConfig::default();

        let result = sinkhorn_knopp(&h_tilde, &config);

        // Approximate spectral norm via power iteration
        let mut v = Array2::from_elem((4, 1), 0.5);
        for _ in 0..100 {
            let av = result.matrix.dot(&v);
            let norm: f32 = av.iter().map(|x| x * x).sum::<f32>().sqrt();
            v = av / norm;
        }
        let av = result.matrix.dot(&v);
        let spectral_norm: f32 = av.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            spectral_norm <= 1.0 + 1e-4,
            "Spectral norm: {}",
            spectral_norm
        );
    }

    #[test]
    fn test_batched_sinkhorn() {
        let batch = Array3::from_shape_fn((8, 4, 4), |(_, i, j)| ((i + j) as f32) * 0.5);
        let config = SinkhornConfig::default();

        let output = sinkhorn_knopp_batch(&batch, &config);

        assert_eq!(output.shape(), &[8, 4, 4]);

        // Check each matrix in batch is doubly stochastic
        for b in 0..8 {
            let m = output.slice(ndarray::s![b, .., ..]).to_owned();
            assert!(
                is_doubly_stochastic(&m, 1e-4),
                "Batch {} is not doubly stochastic",
                b
            );
        }
    }

    #[test]
    fn test_deep_composition_stability() {
        // Test that even 60 layers deep, the composite remains stable
        let config = SinkhornConfig::default();
        let mut rng = rand::thread_rng();

        let matrices: Vec<Array2<f32>> = (0..60)
            .map(|_| {
                let h_tilde = Array2::from_shape_fn((4, 4), |_| {
                    use rand::Rng;
                    rng.gen_range(-2.0..2.0)
                });
                sinkhorn_knopp(&h_tilde, &config).matrix
            })
            .collect();

        let composite = composite_mapping(&matrices);

        let forward_gain = amax_gain_forward(&composite);
        let backward_gain = amax_gain_backward(&composite);

        // Paper shows max ~1.6 for 60 layers
        assert!(
            forward_gain < 3.0,
            "Forward gain too high: {}",
            forward_gain
        );
        assert!(
            backward_gain < 3.0,
            "Backward gain too high: {}",
            backward_gain
        );

        // Should still be approximately doubly stochastic
        assert!(
            is_doubly_stochastic(&composite, 0.1),
            "Deep composite lost doubly stochastic property"
        );
    }

    #[test]
    fn test_backward_gradient_shape() {
        let h_tilde = array![[1.0, 2.0], [3.0, 4.0]];
        let grad_output = array![[0.1, 0.2], [0.3, 0.4]];
        let config = SinkhornConfig::default();

        let grad_input = sinkhorn_knopp_backward(&h_tilde, &grad_output, &config);

        assert_eq!(grad_input.shape(), h_tilde.shape());
    }

    #[test]
    fn test_backward_gradient_nonzero() {
        // Verify backward pass produces non-zero gradients
        let h_tilde = array![[0.5, 1.0], [1.5, 0.8]];
        let config = SinkhornConfig::default();

        // Loss = sum of all elements (simple test loss)
        let grad_output = Array2::ones((2, 2));

        // Analytical gradient
        let grad_analytical = sinkhorn_knopp_backward(&h_tilde, &grad_output, &config);

        // Gradient should be non-zero for at least some elements
        let grad_norm: f32 = grad_analytical.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            grad_norm > 0.0,
            "Gradient should be non-zero, got norm: {}",
            grad_norm
        );

        // Gradient should have same shape as input
        assert_eq!(grad_analytical.shape(), h_tilde.shape());
    }
}
