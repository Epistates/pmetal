//! Learnable parameters for mHC layers.
//!
//! This module defines the parameter structures used by mHC layers: the
//! gating factors, the static biases and the fused dynamic projection.

use crate::config::MhcConfig;
use crate::error::MhcError;
use ndarray::{Array1, Array2, ArrayView2, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Logit given to every off-diagonal entry of `b_res` at initialization (the
/// diagonal gets 0). After the Sinkhorn projection that is the identity to
/// within `3e-4` per entry for `n = 4`, so a fresh layer leaves each stream
/// where it was.
pub const RES_OFF_DIAGONAL_INIT: f32 = -8.0;

/// Standard deviation of the dynamic projection's random initialization.
const PHI_INIT_STD: f32 = 0.02;

/// Learnable parameters for a single mHC layer.
///
/// These parameters define the mappings H^pre, H^post, and H^res
/// that govern information flow in the mHC architecture:
///
/// ```text
/// x' = RMSNorm(vec(x))                                    [nC]
/// proj = x' · φ                                           [2n + n²]
/// H̃^pre  = α_pre  · proj[..n]       + b_pre               [n]
/// H̃^post = α_post · proj[n..2n]     + b_post              [n]
/// H̃^res  = α_res  · mat(proj[2n..]) + b_res               [n, n]
/// ```
///
/// The three projections are one matrix, so the dynamic mappings of a token
/// cost a single `[nC] × [nC, 2n + n²]` product, and the layout is the one the
/// released models store (`fn`, transposed, with the columns ordered pre, post,
/// res). The RMSNorm carries no weight of its own: a per-channel scale of `x'`
/// is a scale of the rows of φ, so it would only duplicate φ's parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MhcParams {
    // === Gating Factors (α) ===
    /// Gating factor for H^pre mapping.
    pub alpha_pre: f32,

    /// Gating factor for H^post mapping.
    pub alpha_post: f32,

    /// Gating factor for H^res mapping.
    pub alpha_res: f32,

    // === Static Biases (b) ===
    /// Static bias for H^pre mapping. Shape: `[n]`
    pub b_pre: Array1<f32>,

    /// Static bias for H^post mapping. Shape: `[n]`
    pub b_post: Array1<f32>,

    /// Static bias for H^res mapping. Shape: `[n, n]`
    pub b_res: Array2<f32>,

    // === Dynamic Mapping Projection (φ) ===
    /// Fused projection of the normalized stream to the three mappings'
    /// logits. Shape: `[nC, 2n + n²]`; columns `0..n` feed H^pre, `n..2n`
    /// H^post and `2n..` H^res (row-major `n × n`).
    pub phi: Array2<f32>,
}

impl MhcParams {
    /// Create new parameters with the given configuration, drawing the
    /// projection from the thread's random generator.
    ///
    /// Use [`MhcParams::with_rng`] for a reproducible initialization.
    pub fn new(config: &MhcConfig) -> Self {
        Self::with_rng(config, &mut rand::rng())
    }

    /// Create new parameters, drawing the projection from `rng`.
    ///
    /// See [`MhcParams::identity_init`] for the static part: a fresh layer is
    /// a plain residual connection when its streams start out equal.
    /// The projection starts as `N(0, 0.02²)`, small enough next to the gates
    /// (`α = 0.01`) that the dynamic part begins as a perturbation.
    pub fn with_rng<R: rand::Rng + ?Sized>(config: &MhcConfig, rng: &mut R) -> Self {
        let mut params = Self::identity_init(config);
        let normal = Normal::new(0.0, PHI_INIT_STD).expect("a positive std is valid");
        params.phi.mapv_inplace(|_| normal.sample(&mut *rng));
        params
    }

    /// Parameters whose mappings are the same for every input and make the
    /// layer a plain residual connection when the `n` streams are equal:
    ///
    /// - `b_pre` gives `H^pre = 1/n` (the mean of the streams: `σ(b) + ε = 1/n`),
    /// - `b_post = 0` gives `H^post = 2σ(0) = 1`,
    /// - `b_res` is [`RES_OFF_DIAGONAL_INIT`] off the diagonal and 0 on it, so
    ///   `H^res` is the identity to within a few 1e-4,
    /// - `φ = 0`.
    ///
    /// Then `x_{l+1} = x + F(x)` in every stream.
    pub fn identity_init(config: &MhcConfig) -> Self {
        let n = config.expansion_rate;
        let nc = config.expanded_dim();

        // σ(b) + ε = 1/n  ⇒  b = logit(1/n − ε).
        let p = (1.0 / n as f64 - config.epsilon as f64).clamp(1e-12, 1.0 - 1e-12);
        let b_pre_value = (p / (1.0 - p)).ln() as f32;

        let b_res = Array2::from_shape_fn(
            (n, n),
            |(i, j)| {
                if i == j { 0.0 } else { RES_OFF_DIAGONAL_INIT }
            },
        );

        Self {
            alpha_pre: config.alpha_init,
            alpha_post: config.alpha_init,
            alpha_res: config.alpha_init,
            b_pre: Array1::from_elem(n, b_pre_value),
            b_post: Array1::zeros(n),
            b_res,
            phi: Array2::zeros((nc, config.mixing_dim())),
        }
    }

    /// The columns of φ that feed H^pre. Shape: `[nC, n]`
    pub fn phi_pre(&self) -> ArrayView2<'_, f32> {
        let n = self.b_pre.len();
        self.phi.slice(s![.., ..n])
    }

    /// The columns of φ that feed H^post. Shape: `[nC, n]`
    pub fn phi_post(&self) -> ArrayView2<'_, f32> {
        let n = self.b_pre.len();
        self.phi.slice(s![.., n..2 * n])
    }

    /// The columns of φ that feed H^res (row-major `n × n`). Shape: `[nC, n²]`
    pub fn phi_res(&self) -> ArrayView2<'_, f32> {
        let n = self.b_pre.len();
        self.phi.slice(s![.., 2 * n..])
    }

    /// Get the total number of parameters.
    pub fn num_params(&self) -> usize {
        let n = self.b_pre.len();
        // Gating factors, biases, projection.
        3 + (2 * n + n * n) + self.phi.len()
    }

    /// Get memory usage in bytes (assuming f32).
    pub fn memory_bytes(&self) -> usize {
        self.num_params() * std::mem::size_of::<f32>()
    }

    /// Validate parameter shapes against configuration.
    pub fn validate(&self, config: &MhcConfig) -> Result<(), MhcError> {
        let n = config.expansion_rate;
        let check = |what: &'static str, expected: Vec<usize>, actual: &[usize]| {
            if expected == actual {
                Ok(())
            } else {
                Err(MhcError::shape(what, format!("{expected:?}"), actual))
            }
        };
        check("b_pre", vec![n], self.b_pre.shape())?;
        check("b_post", vec![n], self.b_post.shape())?;
        check("b_res", vec![n, n], self.b_res.shape())?;
        check(
            "phi",
            vec![config.expanded_dim(), config.mixing_dim()],
            self.phi.shape(),
        )
    }
}

/// Computed mappings from a forward pass.
///
/// These are the actual H^pre, H^post, H^res matrices computed
/// from the input and parameters for a specific batch.
#[derive(Debug, Clone)]
pub struct MhcMappings {
    /// Pre-mapping H^pre. Shape: `[batch, n]`
    pub h_pre: Array2<f32>,

    /// Post-mapping H^post. Shape: `[batch, n]`
    pub h_post: Array2<f32>,

    /// Residual mapping H^res (doubly stochastic). Shape: `[batch, n, n]`
    pub h_res: ndarray::Array3<f32>,
}

impl MhcMappings {
    /// Create new mappings with given shapes.
    pub fn zeros(batch_size: usize, n: usize) -> Self {
        Self {
            h_pre: Array2::zeros((batch_size, n)),
            h_post: Array2::zeros((batch_size, n)),
            h_res: ndarray::Array3::zeros((batch_size, n, n)),
        }
    }

    /// Get batch size.
    pub fn batch_size(&self) -> usize {
        self.h_pre.nrows()
    }

    /// Get expansion rate (n).
    pub fn expansion_rate(&self) -> usize {
        self.h_pre.ncols()
    }
}

/// Gradients for mHC parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct MhcGradients {
    /// Gradient for alpha_pre.
    pub d_alpha_pre: f32,

    /// Gradient for alpha_post.
    pub d_alpha_post: f32,

    /// Gradient for alpha_res.
    pub d_alpha_res: f32,

    /// Gradient for b_pre. Shape: `[n]`
    pub d_b_pre: Array1<f32>,

    /// Gradient for b_post. Shape: `[n]`
    pub d_b_post: Array1<f32>,

    /// Gradient for b_res. Shape: `[n, n]`
    pub d_b_res: Array2<f32>,

    /// Gradient for phi. Shape: `[nC, 2n + n²]`
    pub d_phi: Array2<f32>,
}

impl MhcGradients {
    /// Create zero gradients with the given configuration.
    pub fn zeros(config: &MhcConfig) -> Self {
        let n = config.expansion_rate;

        Self {
            d_alpha_pre: 0.0,
            d_alpha_post: 0.0,
            d_alpha_res: 0.0,
            d_b_pre: Array1::zeros(n),
            d_b_post: Array1::zeros(n),
            d_b_res: Array2::zeros((n, n)),
            d_phi: Array2::zeros((config.expanded_dim(), config.mixing_dim())),
        }
    }

    /// Accumulate gradients from another gradient struct.
    pub fn accumulate(&mut self, other: &MhcGradients) {
        self.d_alpha_pre += other.d_alpha_pre;
        self.d_alpha_post += other.d_alpha_post;
        self.d_alpha_res += other.d_alpha_res;
        self.d_b_pre += &other.d_b_pre;
        self.d_b_post += &other.d_b_post;
        self.d_b_res += &other.d_b_res;
        self.d_phi += &other.d_phi;
    }

    /// Scale all gradients by a factor.
    pub fn scale(&mut self, factor: f32) {
        self.d_alpha_pre *= factor;
        self.d_alpha_post *= factor;
        self.d_alpha_res *= factor;
        self.d_b_pre *= factor;
        self.d_b_post *= factor;
        self.d_b_res *= factor;
        self.d_phi *= factor;
    }

    /// Compute gradient norm (L2), accumulated in `f64`.
    pub fn norm(&self) -> f32 {
        let sq = |a: &[f32]| a.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>();
        let scalars = [self.d_alpha_pre, self.d_alpha_post, self.d_alpha_res];
        let arrays = self
            .d_b_pre
            .iter()
            .chain(&self.d_b_post)
            .chain(&self.d_b_res)
            .chain(&self.d_phi)
            .copied();
        let total = sq(&scalars) + arrays.map(|x| f64::from(x) * f64::from(x)).sum::<f64>();
        total.sqrt() as f32
    }

    /// Clip gradients by norm.
    pub fn clip_norm(&mut self, max_norm: f32) {
        let norm = self.norm();
        if norm > max_norm {
            self.scale(max_norm / norm);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_params_creation() {
        let config = MhcConfig::default();
        let params = MhcParams::new(&config);

        assert!(params.validate(&config).is_ok());
        // 3 gates, 2·4 + 16 biases, a 10240 × 24 projection.
        assert_eq!(params.num_params(), 3 + 24 + 10240 * 24);
    }

    #[test]
    fn test_params_shapes_and_views() {
        let config = MhcConfig {
            expansion_rate: 4,
            hidden_dim: 256,
            ..Default::default()
        };
        let params = MhcParams::new(&config);

        assert_eq!(params.b_pre.len(), 4);
        assert_eq!(params.b_post.len(), 4);
        assert_eq!(params.b_res.shape(), &[4, 4]);
        assert_eq!(params.phi.shape(), &[1024, 24]);
        assert_eq!(params.phi_pre().shape(), &[1024, 4]);
        assert_eq!(params.phi_post().shape(), &[1024, 4]);
        assert_eq!(params.phi_res().shape(), &[1024, 16]);
        assert_eq!(params.phi_post()[[3, 0]], params.phi[[3, 4]]);
        assert_eq!(params.phi_res()[[3, 5]], params.phi[[3, 13]]);
    }

    #[test]
    fn validate_reports_the_offending_parameter() {
        let config = MhcConfig {
            hidden_dim: 8,
            ..Default::default()
        };
        let mut params = MhcParams::identity_init(&config);
        params.b_res = Array2::zeros((4, 3));
        match params.validate(&config) {
            Err(MhcError::ShapeMismatch { what, .. }) => assert_eq!(what, "b_res"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn seeded_initialization_is_reproducible_and_scaled() {
        let config = MhcConfig {
            hidden_dim: 128,
            ..Default::default()
        };
        let a = MhcParams::with_rng(&config, &mut rand::rngs::StdRng::seed_from_u64(7));
        let b = MhcParams::with_rng(&config, &mut rand::rngs::StdRng::seed_from_u64(7));
        let c = MhcParams::with_rng(&config, &mut rand::rngs::StdRng::seed_from_u64(8));
        assert_eq!(a, b);
        assert_ne!(a.phi, c.phi);
        let n = a.phi.len() as f64;
        let mean = a.phi.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = a
            .phi
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        assert!(
            mean.abs() < 1e-3 && (var.sqrt() - 0.02).abs() < 1e-3,
            "{mean} {var}"
        );
    }

    #[test]
    fn identity_init_biases() {
        let config = MhcConfig::default();
        let p = MhcParams::identity_init(&config);

        // σ(b_pre) + ε is 1/n.
        for &b in &p.b_pre {
            let h = 1.0 / (1.0 + (-f64::from(b)).exp()) + f64::from(config.epsilon);
            assert!((h - 0.25).abs() < 1e-7, "{h}");
        }
        assert!(p.b_post.iter().all(|&b| b == 0.0));
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j { 0.0 } else { RES_OFF_DIAGONAL_INIT };
                assert_eq!(p.b_res[[i, j]], want);
            }
        }
        assert!(p.phi.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_gradients() {
        let config = MhcConfig {
            expansion_rate: 4,
            hidden_dim: 64,
            ..Default::default()
        };

        let mut grads = MhcGradients::zeros(&config);
        grads.d_alpha_pre = 3.0;
        grads.d_b_res[[1, 2]] = 4.0;
        assert!((grads.norm() - 5.0).abs() < 1e-6);

        grads.clip_norm(1.0);
        assert!((grads.norm() - 1.0).abs() < 1e-6);
        assert!((grads.d_alpha_pre - 0.6).abs() < 1e-6);

        let mut other = MhcGradients::zeros(&config);
        other.d_phi[[2, 3]] = 2.0;
        grads.accumulate(&other);
        assert_eq!(grads.d_phi[[2, 3]], 2.0);
    }
}
