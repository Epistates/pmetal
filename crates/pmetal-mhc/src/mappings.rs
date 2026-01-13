//! Computation of mHC mappings (H^pre, H^post, H^res).
//!
//! This module implements the mapping computation pipeline:
//! 1. Flatten and normalize input: x̃'_l = RMSNorm(vec(x_l))
//! 2. Compute unconstrained mappings via linear projections
//! 3. Apply constraints (sigmoid for pre/post, Sinkhorn-Knopp for res)
//!
//! # Paper Reference
//!
//! From arXiv:2512.24880, Equations 7-8:
//!
//! ```text
//! H̃^pre_l  = α^pre_l  · (x̃'_l · φ^pre_l)  + b^pre_l
//! H̃^post_l = α^post_l · (x̃'_l · φ^post_l) + b^post_l
//! H̃^res_l  = α^res_l  · mat(x̃'_l · φ^res_l) + b^res_l
//!
//! H^pre_l  = σ(H̃^pre_l)
//! H^post_l = 2σ(H̃^post_l)
//! H^res_l  = Sinkhorn-Knopp(H̃^res_l)
//! ```

use crate::config::MhcConfig;
use crate::params::{MhcGradients, MhcMappings, MhcParams};
use crate::sinkhorn::{sinkhorn_knopp_backward, SinkhornConfig};
use ndarray::{Array1, Array2, Array3, Axis};

/// Compute mHC mappings from input and parameters.
///
/// # Arguments
///
/// * `x` - Input tensor. Shape: [batch, n, C]
/// * `params` - Layer parameters
/// * `config` - Configuration
///
/// # Returns
///
/// Computed mappings (H^pre, H^post, H^res).
pub fn compute_mappings(x: &Array3<f32>, params: &MhcParams, config: &MhcConfig) -> MhcMappings {
    let batch_size = x.shape()[0];
    let n = config.expansion_rate;
    let c = config.hidden_dim;
    let nc = n * c;

    // Step 1: Flatten input to [batch, nC]
    let x_flat = x.to_shape((batch_size, nc)).unwrap().to_owned();

    // Step 2: Apply RMSNorm
    let x_norm = rmsnorm(&x_flat, &params.rmsnorm_weight, config.epsilon);

    // Step 3: Compute unconstrained mappings
    let (h_tilde_pre, h_tilde_post, h_tilde_res) =
        compute_unconstrained_mappings(&x_norm, params, config);

    // Step 4: Apply constraints
    let h_pre = apply_sigmoid(&h_tilde_pre);
    let h_post = apply_scaled_sigmoid(&h_tilde_post, 2.0);
    let h_res = apply_sinkhorn(&h_tilde_res, config);

    MhcMappings {
        h_pre,
        h_post,
        h_res,
    }
}

/// Compute unconstrained mappings (before constraint application).
///
/// Returns (H̃^pre, H̃^post, H̃^res).
fn compute_unconstrained_mappings(
    x_norm: &Array2<f32>,
    params: &MhcParams,
    config: &MhcConfig,
) -> (Array2<f32>, Array2<f32>, Array3<f32>) {
    let batch_size = x_norm.nrows();
    let n = config.expansion_rate;

    // H̃^pre = α^pre · (x' · φ^pre) + b^pre
    let h_tilde_pre = if config.use_dynamic_mappings {
        let proj = x_norm.dot(&params.phi_pre); // [batch, n]
        proj * params.alpha_pre + &params.b_pre
    } else {
        // Static only: broadcast bias
        let mut h = Array2::zeros((batch_size, n));
        for i in 0..batch_size {
            h.row_mut(i).assign(&params.b_pre);
        }
        h
    };

    // H̃^post = α^post · (x' · φ^post) + b^post
    let h_tilde_post = if config.use_dynamic_mappings {
        let proj = x_norm.dot(&params.phi_post); // [batch, n]
        proj * params.alpha_post + &params.b_post
    } else {
        let mut h = Array2::zeros((batch_size, n));
        for i in 0..batch_size {
            h.row_mut(i).assign(&params.b_post);
        }
        h
    };

    // H̃^res = α^res · mat(x' · φ^res) + b^res
    let h_tilde_res = if config.use_dynamic_mappings {
        let proj = x_norm.dot(&params.phi_res); // [batch, n²]
        let proj_scaled = proj * params.alpha_res;

        // Reshape to [batch, n, n] and add bias
        let mut h = Array3::zeros((batch_size, n, n));
        for b in 0..batch_size {
            for i in 0..n {
                for j in 0..n {
                    h[[b, i, j]] = proj_scaled[[b, i * n + j]] + params.b_res[[i, j]];
                }
            }
        }
        h
    } else {
        // Static only: broadcast bias
        let mut h = Array3::zeros((batch_size, n, n));
        for b in 0..batch_size {
            for i in 0..n {
                for j in 0..n {
                    h[[b, i, j]] = params.b_res[[i, j]];
                }
            }
        }
        h
    };

    (h_tilde_pre, h_tilde_post, h_tilde_res)
}

/// Apply RMSNorm to input.
///
/// RMSNorm(x) = x / sqrt(mean(x²) + ε) * weight
fn rmsnorm(x: &Array2<f32>, weight: &Array1<f32>, epsilon: f32) -> Array2<f32> {
    let batch_size = x.nrows();
    let dim = x.ncols();

    let mut output = Array2::zeros((batch_size, dim));

    for b in 0..batch_size {
        // Compute RMS
        let sum_sq: f32 = x.row(b).iter().map(|v| v * v).sum();
        let rms = (sum_sq / dim as f32 + epsilon).sqrt();
        let rms_inv = 1.0 / rms;

        // Normalize and scale
        for i in 0..dim {
            output[[b, i]] = x[[b, i]] * rms_inv * weight[i];
        }
    }

    output
}

/// Apply sigmoid function element-wise.
fn apply_sigmoid(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

/// Apply scaled sigmoid function element-wise.
fn apply_scaled_sigmoid(x: &Array2<f32>, scale: f32) -> Array2<f32> {
    x.mapv(|v| scale / (1.0 + (-v).exp()))
}

/// Apply Sinkhorn-Knopp to each matrix in batch.
fn apply_sinkhorn(h_tilde: &Array3<f32>, config: &MhcConfig) -> Array3<f32> {
    let sk_config = SinkhornConfig {
        max_iterations: config.sinkhorn_iterations,
        epsilon: config.epsilon,
        tolerance: 1e-6,
    };

    crate::sinkhorn::sinkhorn_knopp_batch(h_tilde, &sk_config)
}

/// Apply H^pre mapping: aggregate n streams into single layer input.
///
/// h_in = H^pre @ x  (sum over streams dimension)
///
/// # Arguments
///
/// * `x` - Input tensor. Shape: [batch, n, C]
/// * `h_pre` - Pre-mapping weights. Shape: [batch, n]
///
/// # Returns
///
/// Layer input. Shape: [batch, C]
pub fn apply_pre_mapping(x: &Array3<f32>, h_pre: &Array2<f32>) -> Array2<f32> {
    let batch_size = x.shape()[0];
    let n = x.shape()[1];
    let c = x.shape()[2];

    let mut output = Array2::zeros((batch_size, c));

    for b in 0..batch_size {
        for i in 0..c {
            let mut sum = 0.0f32;
            for s in 0..n {
                sum += h_pre[[b, s]] * x[[b, s, i]];
            }
            output[[b, i]] = sum;
        }
    }

    output
}

/// Apply H^post and H^res mappings: fused post-mapping and residual merge.
///
/// x_{l+1} = H^res @ x_l + H^post^T @ h_out
///
/// # Arguments
///
/// * `x` - Input tensor (residual stream). Shape: [batch, n, C]
/// * `h_out` - Layer output. Shape: [batch, C]
/// * `h_post` - Post-mapping weights. Shape: [batch, n]
/// * `h_res` - Residual mapping (doubly stochastic). Shape: [batch, n, n]
///
/// # Returns
///
/// Updated residual stream. Shape: [batch, n, C]
pub fn apply_post_res_mapping(
    x: &Array3<f32>,
    h_out: &Array2<f32>,
    h_post: &Array2<f32>,
    h_res: &Array3<f32>,
) -> Array3<f32> {
    let batch_size = x.shape()[0];
    let n = x.shape()[1];
    let c = x.shape()[2];

    let mut output = Array3::zeros((batch_size, n, c));

    for b in 0..batch_size {
        for i in 0..n {
            for k in 0..c {
                // H^res @ x component
                let mut res_val = 0.0f32;
                for j in 0..n {
                    res_val += h_res[[b, i, j]] * x[[b, j, k]];
                }

                // H^post^T @ h_out component (outer product)
                let post_val = h_post[[b, i]] * h_out[[b, k]];

                output[[b, i, k]] = res_val + post_val;
            }
        }
    }

    output
}

/// Backward pass for mapping computation.
///
/// Computes gradients for parameters and input.
pub fn compute_mappings_backward(
    x: &Array3<f32>,
    params: &MhcParams,
    config: &MhcConfig,
    grad_h_pre: &Array2<f32>,
    grad_h_post: &Array2<f32>,
    grad_h_res: &Array3<f32>,
) -> (Array3<f32>, MhcGradients) {
    let batch_size = x.shape()[0];
    let n = config.expansion_rate;
    let c = config.hidden_dim;
    let nc = n * c;

    let mut gradients = MhcGradients::zeros(config);

    // Flatten input
    let x_flat = x.to_shape((batch_size, nc)).unwrap().to_owned();

    // Forward pass to get intermediates
    let x_norm = rmsnorm(&x_flat, &params.rmsnorm_weight, config.epsilon);
    let (h_tilde_pre, h_tilde_post, h_tilde_res) =
        compute_unconstrained_mappings(&x_norm, params, config);

    // Backward through constraints
    // grad_h_tilde_pre = grad_h_pre * sigmoid'(h_tilde_pre)
    let grad_h_tilde_pre = {
        let sig = apply_sigmoid(&h_tilde_pre);
        grad_h_pre * &sig * &(1.0 - &sig)
    };

    // grad_h_tilde_post = grad_h_post * 2 * sigmoid'(h_tilde_post)
    let grad_h_tilde_post = {
        let sig = apply_sigmoid(&h_tilde_post);
        grad_h_post * &sig * &(1.0 - &sig) * 2.0
    };

    // grad_h_tilde_res = sinkhorn_backward(...)
    let sk_config = SinkhornConfig {
        max_iterations: config.sinkhorn_iterations,
        epsilon: config.epsilon,
        tolerance: 1e-6,
    };
    let mut grad_h_tilde_res = Array3::zeros(h_tilde_res.raw_dim());
    for b in 0..batch_size {
        let h_t = h_tilde_res.slice(ndarray::s![b, .., ..]).to_owned();
        let g_out = grad_h_res.slice(ndarray::s![b, .., ..]).to_owned();
        let g_in = sinkhorn_knopp_backward(&h_t, &g_out, &sk_config);
        grad_h_tilde_res
            .slice_mut(ndarray::s![b, .., ..])
            .assign(&g_in);
    }

    // Backward through unconstrained mapping computation
    if config.use_dynamic_mappings {
        // Gradient for phi_pre: d/d(phi_pre) = x_norm^T @ (grad_h_tilde_pre * alpha_pre)
        let grad_proj_pre = &grad_h_tilde_pre * params.alpha_pre;
        gradients.d_phi_pre = x_norm.t().dot(&grad_proj_pre);

        // Gradient for alpha_pre
        let proj_pre = x_norm.dot(&params.phi_pre);
        gradients.d_alpha_pre = (&grad_h_tilde_pre * &proj_pre).sum();

        // Gradient for phi_post
        let grad_proj_post = &grad_h_tilde_post * params.alpha_post;
        gradients.d_phi_post = x_norm.t().dot(&grad_proj_post);

        // Gradient for alpha_post
        let proj_post = x_norm.dot(&params.phi_post);
        gradients.d_alpha_post = (&grad_h_tilde_post * &proj_post).sum();

        // Gradient for phi_res (reshape grad_h_tilde_res to [batch, n²])
        let grad_h_tilde_res_flat = grad_h_tilde_res
            .to_shape((batch_size, n * n))
            .unwrap()
            .to_owned();
        let grad_proj_res = &grad_h_tilde_res_flat * params.alpha_res;
        gradients.d_phi_res = x_norm.t().dot(&grad_proj_res);

        // Gradient for alpha_res
        let proj_res = x_norm.dot(&params.phi_res);
        gradients.d_alpha_res = (&grad_h_tilde_res_flat * &proj_res).sum();
    }

    // Gradient for biases (sum over batch)
    gradients.d_b_pre = grad_h_tilde_pre.sum_axis(Axis(0));
    gradients.d_b_post = grad_h_tilde_post.sum_axis(Axis(0));
    gradients.d_b_res = grad_h_tilde_res.sum_axis(Axis(0));

    // Gradient for input x (through dynamic mappings and RMSNorm)
    let grad_x_norm = if config.use_dynamic_mappings {
        // grad_x_norm from pre
        let grad_from_pre = grad_h_tilde_pre.dot(&params.phi_pre.t()) * params.alpha_pre;

        // grad_x_norm from post
        let grad_from_post = grad_h_tilde_post.dot(&params.phi_post.t()) * params.alpha_post;

        // grad_x_norm from res
        let grad_h_tilde_res_flat = grad_h_tilde_res
            .to_shape((batch_size, n * n))
            .unwrap()
            .to_owned();
        let grad_from_res = grad_h_tilde_res_flat.dot(&params.phi_res.t()) * params.alpha_res;

        grad_from_pre + grad_from_post + grad_from_res
    } else {
        Array2::zeros((batch_size, nc))
    };

    // Backward through RMSNorm
    let grad_x_flat = rmsnorm_backward(
        &x_flat,
        &grad_x_norm,
        &params.rmsnorm_weight,
        config.epsilon,
    );

    // Gradient for rmsnorm weight
    let x_normalized = rmsnorm_normalize_only(&x_flat, config.epsilon);
    gradients.d_rmsnorm_weight = (&grad_x_norm * &x_normalized).sum_axis(Axis(0));

    // Reshape gradient back to [batch, n, C]
    let grad_x = grad_x_flat.to_shape((batch_size, n, c)).unwrap().to_owned();

    (grad_x, gradients)
}

/// RMSNorm backward pass.
fn rmsnorm_backward(
    x: &Array2<f32>,
    grad_output: &Array2<f32>,
    weight: &Array1<f32>,
    epsilon: f32,
) -> Array2<f32> {
    let batch_size = x.nrows();
    let dim = x.ncols();

    let mut grad_input = Array2::zeros((batch_size, dim));

    for b in 0..batch_size {
        // Compute RMS
        let sum_sq: f32 = x.row(b).iter().map(|v| v * v).sum();
        let rms_sq = sum_sq / dim as f32 + epsilon;
        let rms = rms_sq.sqrt();
        let rms_inv = 1.0 / rms;

        // Gradient computation
        // d/dx[x * w / rms] = w / rms - x * w * x / (rms^3 * dim)
        let mut sum_grad = 0.0f32;
        for i in 0..dim {
            sum_grad += grad_output[[b, i]] * weight[i] * x[[b, i]];
        }
        sum_grad /= rms_sq * dim as f32;

        for i in 0..dim {
            let grad_norm = grad_output[[b, i]] * weight[i] * rms_inv;
            let grad_rms = -x[[b, i]] * sum_grad;
            grad_input[[b, i]] = grad_norm + grad_rms;
        }
    }

    grad_input
}

/// RMSNorm without weight multiplication (for gradient computation).
fn rmsnorm_normalize_only(x: &Array2<f32>, epsilon: f32) -> Array2<f32> {
    let batch_size = x.nrows();
    let dim = x.ncols();

    let mut output = Array2::zeros((batch_size, dim));

    for b in 0..batch_size {
        let sum_sq: f32 = x.row(b).iter().map(|v| v * v).sum();
        let rms = (sum_sq / dim as f32 + epsilon).sqrt();
        let rms_inv = 1.0 / rms;

        for i in 0..dim {
            output[[b, i]] = x[[b, i]] * rms_inv;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_compute_mappings_shapes() {
        let config = MhcConfig {
            expansion_rate: 4,
            hidden_dim: 64,
            ..Default::default()
        };
        let params = MhcParams::new(&config);

        let batch = 8;
        let x = Array3::from_shape_fn((batch, 4, 64), |(b, i, j)| ((b + i + j) as f32) * 0.01);

        let mappings = compute_mappings(&x, &params, &config);

        assert_eq!(mappings.h_pre.shape(), &[8, 4]);
        assert_eq!(mappings.h_post.shape(), &[8, 4]);
        assert_eq!(mappings.h_res.shape(), &[8, 4, 4]);
    }

    #[test]
    fn test_h_pre_non_negative() {
        let config = MhcConfig::default();
        let params = MhcParams::new(&config);

        let x = Array3::from_shape_fn((4, 4, config.hidden_dim), |_| rand::random::<f32>() - 0.5);
        let mappings = compute_mappings(&x, &params, &config);

        // H^pre should be non-negative (sigmoid output)
        assert!(mappings.h_pre.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_h_post_range() {
        let config = MhcConfig::default();
        let params = MhcParams::new(&config);

        let x = Array3::from_shape_fn((4, 4, config.hidden_dim), |_| rand::random::<f32>() - 0.5);
        let mappings = compute_mappings(&x, &params, &config);

        // H^post should be in [0, 2] (2*sigmoid output)
        assert!(mappings.h_post.iter().all(|&v| (0.0..=2.0).contains(&v)));
    }

    #[test]
    fn test_h_res_doubly_stochastic() {
        let config = MhcConfig::default();
        let params = MhcParams::new(&config);

        let x = Array3::from_shape_fn((4, 4, config.hidden_dim), |_| rand::random::<f32>() - 0.5);
        let mappings = compute_mappings(&x, &params, &config);

        // Each H^res should be doubly stochastic
        for b in 0..4 {
            let m = mappings.h_res.slice(ndarray::s![b, .., ..]).to_owned();
            assert!(
                crate::sinkhorn::is_doubly_stochastic(&m, 1e-4),
                "H^res for batch {} is not doubly stochastic",
                b
            );
        }
    }

    #[test]
    fn test_apply_pre_mapping() {
        let batch = 2;
        let n = 4;
        let c = 8;

        let x = Array3::from_shape_fn((batch, n, c), |(b, i, j)| (b + i + j) as f32);
        let h_pre = Array2::from_shape_fn((batch, n), |(_, _i)| 0.25); // Uniform

        let h_in = apply_pre_mapping(&x, &h_pre);

        assert_eq!(h_in.shape(), &[batch, c]);

        // With uniform weights, output should be average over streams
        for b in 0..batch {
            for j in 0..c {
                let expected: f32 = (0..n).map(|i| x[[b, i, j]] * 0.25).sum();
                assert!((h_in[[b, j]] - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_apply_post_res_mapping() {
        let batch = 2;
        let n = 4;
        let c = 8;

        let x = Array3::from_shape_fn((batch, n, c), |(b, i, j)| ((b + i + j) as f32) * 0.1);
        let h_out = Array2::from_shape_fn((batch, c), |(b, j)| ((b + j) as f32) * 0.1);
        let h_post = Array2::from_shape_fn((batch, n), |(_, _i)| 0.5);

        // Identity-like H^res
        let mut h_res = Array3::zeros((batch, n, n));
        for b in 0..batch {
            for i in 0..n {
                h_res[[b, i, i]] = 0.9;
                for j in 0..n {
                    if i != j {
                        h_res[[b, i, j]] = 0.1 / (n - 1) as f32;
                    }
                }
            }
        }

        let output = apply_post_res_mapping(&x, &h_out, &h_post, &h_res);

        assert_eq!(output.shape(), &[batch, n, c]);
    }

    #[test]
    fn test_rmsnorm() {
        let x = ndarray::array![[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]];
        let weight = ndarray::Array1::ones(4);
        let epsilon = 1e-6;

        let output = rmsnorm(&x, &weight, epsilon);

        // Check that output has unit RMS (approximately)
        for b in 0..2 {
            let sum_sq: f32 = output.row(b).iter().map(|v| v * v).sum();
            let rms = (sum_sq / 4.0).sqrt();
            assert!((rms - 1.0).abs() < 0.01, "RMS: {}", rms);
        }
    }

    #[test]
    fn test_static_mappings() {
        let config = MhcConfig {
            expansion_rate: 4,
            hidden_dim: 64,
            use_dynamic_mappings: false,
            ..Default::default()
        };
        let params = MhcParams::new(&config);

        let x1 = Array3::from_shape_fn((4, 4, 64), |_| rand::random::<f32>());
        let x2 = Array3::from_shape_fn((4, 4, 64), |_| rand::random::<f32>());

        let mappings1 = compute_mappings(&x1, &params, &config);
        let mappings2 = compute_mappings(&x2, &params, &config);

        // With static mappings, same params should give same mappings
        // (they don't depend on input)
        // Note: H^res goes through Sinkhorn which is deterministic
        for b in 0..4 {
            for i in 0..4 {
                assert!(
                    (mappings1.h_pre[[b, i]] - mappings2.h_pre[[b, i]]).abs() < 1e-5,
                    "H^pre differs for different inputs with static mappings"
                );
            }
        }
    }
}
