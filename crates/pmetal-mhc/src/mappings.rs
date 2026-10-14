//! Computation of mHC mappings (H^pre, H^post, H^res) and their application.
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
//!
//! The released models add `ε` to `H^pre` (`σ(·) + ε`); so does this module,
//! with [`MhcConfig::epsilon`]. The RMSNorm is unweighted (its scale is part of φ).
//!
//! Every function here works on a batch of tokens: `x` is `[batch, n, C]`, one
//! `n × C` residual stream per row. Shape mismatches are programmer errors and
//! panic with a message naming the shapes; [`MhcLayer`](crate::MhcLayer) checks
//! them first and returns [`MhcError`] instead.

use crate::config::MhcConfig;
use crate::error::MhcError;
use crate::params::{MhcGradients, MhcMappings, MhcParams};
use crate::sinkhorn::{sinkhorn_knopp_backward_batch, sinkhorn_knopp_batch};
use ndarray::{Array2, Array3, ArrayView3, Axis, s};

/// Check that `x` is `[batch, n, C]` for `config`.
pub(crate) fn check_stream_shape(
    x: &ArrayView3<'_, f32>,
    config: &MhcConfig,
) -> Result<(), MhcError> {
    let n = config.expansion_rate;
    let c = config.hidden_dim;
    if x.shape()[1] != n || x.shape()[2] != c {
        return Err(MhcError::shape(
            "residual stream",
            format!("[batch, {n}, {c}]"),
            x.shape(),
        ));
    }
    Ok(())
}

fn expect_ok<T>(r: Result<T, MhcError>) -> T {
    r.unwrap_or_else(|e| panic!("{e}"))
}

/// Intermediates of the mapping computation, kept for the backward pass.
struct Forward {
    /// `RMSNorm(vec(x))`, `[batch, nC]`.
    x_norm: Array2<f32>,
    /// `1 / sqrt(mean(x²) + ε)` per row.
    inv_rms: Vec<f32>,
    /// `x_norm · φ`, `[batch, 2n + n²]`; `None` for static mappings.
    proj: Option<Array2<f32>>,
    /// Pre-activation logits of H^pre and H^post, `[batch, n]`.
    a_pre: Array2<f32>,
    a_post: Array2<f32>,
    /// H̃^res, `[batch, n, n]`.
    z_res: Array3<f32>,
}

fn forward(x: &ArrayView3<'_, f32>, params: &MhcParams, config: &MhcConfig) -> Forward {
    let batch = x.shape()[0];
    let n = config.expansion_rate;
    let nc = config.expanded_dim();

    let x_flat = x
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((batch, nc))
        .expect("[batch, n, C] holds batch·n·C elements");
    let (x_norm, inv_rms) = rmsnorm(&x_flat, config.epsilon);

    if !config.use_dynamic_mappings {
        let a_pre = Array2::from_shape_fn((batch, n), |(_, i)| params.b_pre[i]);
        let a_post = Array2::from_shape_fn((batch, n), |(_, i)| params.b_post[i]);
        let z_res = Array3::from_shape_fn((batch, n, n), |(_, i, j)| params.b_res[[i, j]]);
        return Forward {
            x_norm,
            inv_rms,
            proj: None,
            a_pre,
            a_post,
            z_res,
        };
    }

    let proj = x_norm.dot(&params.phi); // [batch, 2n + n²]
    let a_pre = &proj.slice(s![.., ..n]) * params.alpha_pre + &params.b_pre;
    let a_post = &proj.slice(s![.., n..2 * n]) * params.alpha_post + &params.b_post;
    let z_res = Array3::from_shape_fn((batch, n, n), |(b, i, j)| {
        params.alpha_res * proj[[b, 2 * n + i * n + j]] + params.b_res[[i, j]]
    });
    Forward {
        x_norm,
        inv_rms,
        proj: Some(proj),
        a_pre,
        a_post,
        z_res,
    }
}

/// Compute mHC mappings from input and parameters.
///
/// # Arguments
///
/// * `x` - Input tensor. Shape: `[batch, n, C]`
/// * `params` - Layer parameters
/// * `config` - Configuration
///
/// # Returns
///
/// Computed mappings (H^pre, H^post, H^res).
///
/// # Panics
///
/// If `x` or `params` do not match `config`.
pub fn compute_mappings(x: &Array3<f32>, params: &MhcParams, config: &MhcConfig) -> MhcMappings {
    expect_ok(check_stream_shape(&x.view(), config));
    expect_ok(params.validate(config));
    let f = forward(&x.view(), params, config);
    let h_pre = f.a_pre.mapv(|v| sigmoid(v) + config.epsilon);
    let h_post = f.a_post.mapv(|v| 2.0 * sigmoid(v));
    let h_res = sinkhorn_knopp_batch(&f.z_res, &config.sinkhorn_config());
    MhcMappings {
        h_pre,
        h_post,
        h_res,
    }
}

/// `RMSNorm(x) = x / sqrt(mean(x²) + ε)`, with the mean accumulated in `f64`
/// (a row has `n·C` ≈ 10⁴ entries). Returns the normalized rows and each row's
/// `1 / sqrt(mean(x²) + ε)`.
fn rmsnorm(x: &Array2<f32>, epsilon: f32) -> (Array2<f32>, Vec<f32>) {
    let dim = x.ncols() as f64;
    let mut out = x.to_owned();
    let mut inv_rms = Vec::with_capacity(x.nrows());
    for mut row in out.rows_mut() {
        let ms = row
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / dim;
        let r = (1.0 / (ms + f64::from(epsilon)).sqrt()) as f32;
        row.mapv_inplace(|v| v * r);
        inv_rms.push(r);
    }
    (out, inv_rms)
}

/// Backward of the unweighted RMSNorm: given `y = x · r`,
/// `g_x = r · (g_y − y · (Σ g_y ⊙ y) / dim)`.
fn rmsnorm_backward(x_norm: &Array2<f32>, inv_rms: &[f32], grad_y: &Array2<f32>) -> Array2<f32> {
    let dim = x_norm.ncols() as f64;
    let mut out = grad_y.to_owned();
    for ((mut g_row, y_row), &r) in out.rows_mut().into_iter().zip(x_norm.rows()).zip(inv_rms) {
        let dot = g_row
            .iter()
            .zip(&y_row)
            .map(|(&g, &y)| f64::from(g) * f64::from(y))
            .sum::<f64>()
            / dim;
        let dot = dot as f32;
        g_row.zip_mut_with(&y_row, |g, &y| *g = r * (*g - y * dot));
    }
    out
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// Apply H^pre mapping: aggregate n streams into single layer input.
///
/// h_in = H^pre @ x  (sum over streams dimension)
///
/// # Arguments
///
/// * `x` - Input tensor. Shape: `[batch, n, C]`
/// * `h_pre` - Pre-mapping weights. Shape: `[batch, n]`
///
/// # Returns
///
/// Layer input. Shape: `[batch, C]`
pub fn apply_pre_mapping(x: &Array3<f32>, h_pre: &Array2<f32>) -> Array2<f32> {
    let (batch, n, c) = x.dim();
    assert_eq!(h_pre.dim(), (batch, n), "h_pre must be [batch, n]");
    let mut out = Array2::zeros((batch, c));
    for b in 0..batch {
        let mut o = out.row_mut(b);
        for s in 0..n {
            o.scaled_add(h_pre[[b, s]], &x.slice(s![b, s, ..]));
        }
    }
    out
}

/// Gradients of [`apply_pre_mapping`].
///
/// Returns `(grad_x, grad_h_pre)` for `h_in[b, k] = Σ_s h_pre[b, s] x[b, s, k]`:
/// `grad_x[b, s, k] = h_pre[b, s] g[b, k]` and `grad_h_pre[b, s] = Σ_k g[b, k] x[b, s, k]`.
pub fn apply_pre_mapping_backward(
    x: &Array3<f32>,
    h_pre: &Array2<f32>,
    grad_h_in: &Array2<f32>,
) -> (Array3<f32>, Array2<f32>) {
    let (batch, n, c) = x.dim();
    assert_eq!(h_pre.dim(), (batch, n), "h_pre must be [batch, n]");
    assert_eq!(grad_h_in.dim(), (batch, c), "grad_h_in must be [batch, C]");
    let mut grad_x = Array3::zeros((batch, n, c));
    let mut grad_h_pre = Array2::zeros((batch, n));
    for b in 0..batch {
        let g = grad_h_in.row(b);
        for s in 0..n {
            grad_x.slice_mut(s![b, s, ..]).scaled_add(h_pre[[b, s]], &g);
            grad_h_pre[[b, s]] = g.dot(&x.slice(s![b, s, ..]));
        }
    }
    (grad_x, grad_h_pre)
}

/// Apply H^post and H^res mappings: fused post-mapping and residual merge.
///
/// x_{l+1} = H^res @ x_l + H^post^T @ h_out
///
/// # Arguments
///
/// * `x` - Input tensor (residual stream). Shape: `[batch, n, C]`
/// * `h_out` - Layer output. Shape: `[batch, C]`
/// * `h_post` - Post-mapping weights. Shape: `[batch, n]`
/// * `h_res` - Residual mapping (doubly stochastic). Shape: `[batch, n, n]`
///
/// # Returns
///
/// Updated residual stream. Shape: `[batch, n, C]`
pub fn apply_post_res_mapping(
    x: &Array3<f32>,
    h_out: &Array2<f32>,
    h_post: &Array2<f32>,
    h_res: &Array3<f32>,
) -> Array3<f32> {
    let (batch, n, c) = x.dim();
    assert_eq!(h_out.dim(), (batch, c), "h_out must be [batch, C]");
    assert_eq!(h_post.dim(), (batch, n), "h_post must be [batch, n]");
    assert_eq!(h_res.dim(), (batch, n, n), "h_res must be [batch, n, n]");
    let mut out = Array3::zeros((batch, n, c));
    for b in 0..batch {
        for i in 0..n {
            let mut o = out.slice_mut(s![b, i, ..]);
            for j in 0..n {
                o.scaled_add(h_res[[b, i, j]], &x.slice(s![b, j, ..]));
            }
            o.scaled_add(h_post[[b, i]], &h_out.row(b));
        }
    }
    out
}

/// Gradients of [`apply_post_res_mapping`].
#[derive(Debug, Clone)]
pub struct PostResGrads {
    /// With respect to the residual stream through `H^res`. `[batch, n, C]`
    pub grad_x: Array3<f32>,
    /// With respect to the layer output. `[batch, C]`
    pub grad_h_out: Array2<f32>,
    /// With respect to `H^post`. `[batch, n]`
    pub grad_h_post: Array2<f32>,
    /// With respect to `H^res`. `[batch, n, n]`
    pub grad_h_res: Array3<f32>,
}

/// Backward of [`apply_post_res_mapping`] for `out[i] = Σ_j H^res[i,j] x[j] + H^post[i] h_out`:
///
/// - `grad_x[j] = Σ_i H^res[i,j] g[i]`
/// - `grad_h_out = Σ_i H^post[i] g[i]`
/// - `grad_h_res[i,j] = ⟨g[i], x[j]⟩`
/// - `grad_h_post[i] = ⟨g[i], h_out⟩`
pub fn apply_post_res_mapping_backward(
    x: &Array3<f32>,
    h_out: &Array2<f32>,
    h_post: &Array2<f32>,
    h_res: &Array3<f32>,
    grad_output: &Array3<f32>,
) -> PostResGrads {
    let (batch, n, c) = x.dim();
    assert_eq!(
        grad_output.dim(),
        (batch, n, c),
        "grad_output must be [batch, n, C]"
    );
    assert_eq!(h_out.dim(), (batch, c), "h_out must be [batch, C]");
    assert_eq!(h_post.dim(), (batch, n), "h_post must be [batch, n]");
    assert_eq!(h_res.dim(), (batch, n, n), "h_res must be [batch, n, n]");
    let mut grad_x = Array3::zeros((batch, n, c));
    let mut grad_h_out = Array2::zeros((batch, c));
    let mut grad_h_res = Array3::zeros((batch, n, n));
    let mut grad_h_post = Array2::zeros((batch, n));
    for b in 0..batch {
        let mut gho = grad_h_out.row_mut(b);
        for i in 0..n {
            let g = grad_output.slice(s![b, i, ..]);
            gho.scaled_add(h_post[[b, i]], &g);
            grad_h_post[[b, i]] = g.dot(&h_out.row(b));
            for j in 0..n {
                grad_x
                    .slice_mut(s![b, j, ..])
                    .scaled_add(h_res[[b, i, j]], &g);
                grad_h_res[[b, i, j]] = g.dot(&x.slice(s![b, j, ..]));
            }
        }
    }
    PostResGrads {
        grad_x,
        grad_h_out,
        grad_h_post,
        grad_h_res,
    }
}

/// Backward pass for the mapping computation.
///
/// Given the gradients of a loss with respect to H^pre, H^post and H^res,
/// returns its gradient with respect to the input `x` (through the dynamic
/// mappings and the RMSNorm; zero for static mappings) and to every parameter.
/// The forward intermediates are recomputed from `x`, as the paper does to
/// avoid storing them.
///
/// # Panics
///
/// If the arguments do not match `config`.
pub fn compute_mappings_backward(
    x: &Array3<f32>,
    params: &MhcParams,
    config: &MhcConfig,
    grad_h_pre: &Array2<f32>,
    grad_h_post: &Array2<f32>,
    grad_h_res: &Array3<f32>,
) -> (Array3<f32>, MhcGradients) {
    expect_ok(check_stream_shape(&x.view(), config));
    expect_ok(params.validate(config));
    let batch = x.shape()[0];
    let n = config.expansion_rate;
    assert_eq!(
        grad_h_pre.dim(),
        (batch, n),
        "grad_h_pre must be [batch, n]"
    );
    assert_eq!(
        grad_h_post.dim(),
        (batch, n),
        "grad_h_post must be [batch, n]"
    );
    assert_eq!(
        grad_h_res.dim(),
        (batch, n, n),
        "grad_h_res must be [batch, n, n]"
    );

    let f = forward(&x.view(), params, config);
    let mut grads = MhcGradients::zeros(config);

    // Through the constraints: σ' = σ(1 − σ); H^post carries a factor 2.
    let g_a_pre = Array2::from_shape_fn((batch, n), |(b, i)| {
        let sg = sigmoid(f.a_pre[[b, i]]);
        grad_h_pre[[b, i]] * sg * (1.0 - sg)
    });
    let g_a_post = Array2::from_shape_fn((batch, n), |(b, i)| {
        let sg = sigmoid(f.a_post[[b, i]]);
        2.0 * grad_h_post[[b, i]] * sg * (1.0 - sg)
    });
    let g_z = sinkhorn_knopp_backward_batch(&f.z_res, grad_h_res, &config.sinkhorn_config());

    // Biases: the logits' gradient summed over the batch.
    grads.d_b_pre = g_a_pre.sum_axis(Axis(0));
    grads.d_b_post = g_a_post.sum_axis(Axis(0));
    grads.d_b_res = g_z.sum_axis(Axis(0));

    let Some(proj) = f.proj else {
        return (Array3::zeros(x.raw_dim()), grads);
    };

    // Through the gated projection: logits = α · proj + b.
    let g_proj = {
        let mut g = Array2::zeros(proj.raw_dim());
        g.slice_mut(s![.., ..n])
            .assign(&(&g_a_pre * params.alpha_pre));
        g.slice_mut(s![.., n..2 * n])
            .assign(&(&g_a_post * params.alpha_post));
        for b in 0..batch {
            for i in 0..n {
                for j in 0..n {
                    g[[b, 2 * n + i * n + j]] = params.alpha_res * g_z[[b, i, j]];
                }
            }
        }
        g
    };
    grads.d_alpha_pre = (&g_a_pre * &proj.slice(s![.., ..n])).sum();
    grads.d_alpha_post = (&g_a_post * &proj.slice(s![.., n..2 * n])).sum();
    grads.d_alpha_res = (0..batch)
        .flat_map(|b| (0..n * n).map(move |k| (b, k)))
        .map(|(b, k)| g_z[[b, k / n, k % n]] * proj[[b, 2 * n + k]])
        .sum();

    grads.d_phi = f.x_norm.t().dot(&g_proj);
    let g_x_norm = g_proj.dot(&params.phi.t());
    let g_x_flat = rmsnorm_backward(&f.x_norm, &f.inv_rms, &g_x_norm);
    let grad_x = g_x_flat
        .into_shape_with_order(x.raw_dim())
        .expect("[batch, nC] holds batch·n·C elements");
    (grad_x, grads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

    fn small() -> MhcConfig {
        MhcConfig {
            expansion_rate: 3,
            hidden_dim: 2,
            ..Default::default()
        }
    }

    /// `φ[r][c] = sin(0.7 r + 1.3 c) / 2` — a deterministic, non-trivial matrix
    /// that the float64 reference in the test below reproduces exactly.
    fn phi_small() -> Array2<f32> {
        Array2::from_shape_fn((6, 15), |(r, c)| {
            ((0.7 * r as f64 + 1.3 * c as f64).sin() / 2.0) as f32
        })
    }

    fn params_small() -> MhcParams {
        let c = small();
        let mut p = MhcParams::identity_init(&c);
        p.alpha_pre = 0.5;
        p.alpha_post = 0.4;
        p.alpha_res = 0.3;
        p.b_pre = Array1::from(vec![0.1, -0.2, 0.3]);
        p.b_post = Array1::from(vec![0.2, 0.0, -0.1]);
        p.b_res = array![[0.5, -0.3, 0.1], [0.0, 0.4, -0.2], [-0.1, 0.2, 0.3]];
        p.phi = phi_small();
        p
    }

    fn x_small() -> Array3<f32> {
        // one token: stream i, channel k.
        array![[[1.0, -2.0], [0.5, 3.0], [-1.5, 0.25]]]
    }

    /// Hand-computed with float64 from the paper's equations (RMSNorm with
    /// ε = 1e-6, σ + ε for H^pre, 2σ for H^post, the release's Sinkhorn):
    /// see the module docs. The reference is a plain-Python transcription.
    #[test]
    fn mappings_match_the_paper_equations() {
        let m = compute_mappings(&x_small(), &params_small(), &small());
        let want_pre: [f64; 3] = [0.569823702, 0.434224036, 0.521060802];
        let want_post: [f64; 3] = [1.079526430, 1.075278413, 1.010654994];
        let want_res: [[f64; 3]; 3] = [
            [0.460401895, 0.197056911, 0.342540194],
            [0.320382709, 0.458016960, 0.221599330],
            [0.219214396, 0.344925128, 0.435859476],
        ];
        for i in 0..3 {
            assert!(
                (f64::from(m.h_pre[[0, i]]) - want_pre[i]).abs() < 2e-6,
                "pre[{i}] {}",
                m.h_pre[[0, i]]
            );
            assert!(
                (f64::from(m.h_post[[0, i]]) - want_post[i]).abs() < 2e-6,
                "post[{i}] {}",
                m.h_post[[0, i]]
            );
            for (j, &want) in want_res[i].iter().enumerate() {
                assert!(
                    (f64::from(m.h_res[[0, i, j]]) - want).abs() < 2e-6,
                    "res[{i},{j}] {}",
                    m.h_res[[0, i, j]]
                );
            }
        }
    }

    #[test]
    fn rmsnorm_has_unit_rms_and_is_scale_invariant() {
        let x = array![[1.0f32, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]];
        let (y, _) = rmsnorm(&x, 1e-6);
        for row in y.rows() {
            let rms = (row.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
            assert!((rms - 1.0).abs() < 1e-5, "{rms}");
        }
        for k in 0..4 {
            assert!((y[[0, k]] - y[[1, k]]).abs() < 1e-5);
        }
    }

    #[test]
    fn static_mappings_ignore_the_input() {
        let config = MhcConfig {
            use_dynamic_mappings: false,
            ..small()
        };
        let p = params_small();
        let x1 = x_small();
        let x2 = x1.mapv(|v| v * -3.0 + 1.0);
        let a = compute_mappings(&x1, &p, &config);
        let b = compute_mappings(&x2, &p, &config);
        assert_eq!(a.h_pre, b.h_pre);
        assert_eq!(a.h_res, b.h_res);
        // and `b_pre` alone decides them.
        let want = sigmoid(0.1) + config.epsilon;
        assert!((a.h_pre[[0, 0]] - want).abs() < 1e-7);
    }

    #[test]
    fn apply_mappings_match_hand_computation() {
        let x = array![[[1.0f32, 2.0], [3.0, 4.0]]];
        let h_pre = array![[0.25f32, 0.75]];
        assert_eq!(apply_pre_mapping(&x, &h_pre), array![[2.5f32, 3.5]]);

        let h_out = array![[10.0f32, 20.0]];
        let h_post = array![[1.0f32, 2.0]];
        let h_res = array![[[0.9f32, 0.1], [0.4, 0.6]]];
        let y = apply_post_res_mapping(&x, &h_out, &h_post, &h_res);
        // row 0: .9*[1,2] + .1*[3,4] + 1*[10,20]; row 1: .4*[1,2] + .6*[3,4] + 2*[10,20].
        let want = array![[[11.2f32, 22.2], [22.2, 43.2]]];
        for (a, b) in y.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-5, "{y:?}");
        }
    }

    #[test]
    #[should_panic(expected = "shape mismatch for residual stream")]
    fn wrong_stream_shape_panics_with_the_shapes() {
        let x = Array3::<f32>::zeros((1, 4, 2));
        compute_mappings(&x, &params_small(), &small());
    }
}
