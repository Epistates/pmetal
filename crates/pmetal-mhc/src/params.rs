//! Learnable parameters for mHC layers.
//!
//! This module defines the parameter structures used by mHC layers,
//! including gating factors, biases, and projection matrices.

use crate::config::MhcConfig;
use ndarray::{Array1, Array2};
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};

/// Learnable parameters for a single mHC layer.
///
/// These parameters define the mappings H^pre, H^post, and H^res
/// that govern information flow in the mHC architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MhcParams {
    // === Gating Factors (α) ===
    /// Gating factor for H^pre mapping.
    pub alpha_pre: f32,

    /// Gating factor for H^post mapping.
    pub alpha_post: f32,

    /// Gating factor for H^res mapping.
    pub alpha_res: f32,

    // === Static Biases (b) ===
    /// Static bias for H^pre mapping. Shape: [n]
    pub b_pre: Array1<f32>,

    /// Static bias for H^post mapping. Shape: [n]
    pub b_post: Array1<f32>,

    /// Static bias for H^res mapping. Shape: [n, n]
    pub b_res: Array2<f32>,

    // === Dynamic Mapping Projections (φ) ===
    /// Projection for dynamic H^pre mapping. Shape: [nC, n]
    pub phi_pre: Array2<f32>,

    /// Projection for dynamic H^post mapping. Shape: [nC, n]
    pub phi_post: Array2<f32>,

    /// Projection for dynamic H^res mapping. Shape: [nC, n²]
    pub phi_res: Array2<f32>,

    // === RMSNorm Weight (absorbed into φ) ===
    /// RMSNorm weight for input normalization. Shape: [nC]
    pub rmsnorm_weight: Array1<f32>,
}

impl MhcParams {
    /// Create new parameters with the given configuration.
    ///
    /// Initializes parameters according to the paper's recommendations:
    /// - α values initialized to `alpha_init` (typically 0.01)
    /// - Biases initialized to produce identity-like behavior
    /// - Projections initialized with small random values
    pub fn new(config: &MhcConfig) -> Self {
        let n = config.expansion_rate;
        let c = config.hidden_dim;
        let nc = n * c;
        let n_sq = n * n;

        // Initialize gating factors to small values
        let alpha = config.alpha_init;

        // Initialize biases
        // b_pre: uniform weights (1/n) so all streams contribute equally initially
        let b_pre = Array1::from_elem(n, 1.0 / n as f32);

        // b_post: ones so output distributes to all streams
        let b_post = Array1::ones(n);

        // b_res: identity matrix (preserves streams initially)
        let mut b_res = Array2::zeros((n, n));
        for i in 0..n {
            b_res[[i, i]] = 1.0;
        }

        // Initialize projections with small random values
        let mut rng = rand::thread_rng();
        let std_dev = 0.02; // Small initialization
        let normal = Normal::new(0.0, std_dev).unwrap();

        let phi_pre = Array2::from_shape_fn((nc, n), |_| normal.sample(&mut rng) as f32);
        let phi_post = Array2::from_shape_fn((nc, n), |_| normal.sample(&mut rng) as f32);
        let phi_res = Array2::from_shape_fn((nc, n_sq), |_| normal.sample(&mut rng) as f32);

        // RMSNorm weight initialized to ones
        let rmsnorm_weight = Array1::ones(nc);

        Self {
            alpha_pre: alpha,
            alpha_post: alpha,
            alpha_res: alpha,
            b_pre,
            b_post,
            b_res,
            phi_pre,
            phi_post,
            phi_res,
            rmsnorm_weight,
        }
    }

    /// Create random parameters for testing.
    pub fn random(config: &MhcConfig) -> Self {
        let n = config.expansion_rate;
        let c = config.hidden_dim;
        let nc = n * c;
        let n_sq = n * n;

        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-1.0, 1.0);

        Self {
            alpha_pre: config.alpha_init,
            alpha_post: config.alpha_init,
            alpha_res: config.alpha_init,
            b_pre: Array1::from_shape_fn(n, |_| uniform.sample(&mut rng) as f32),
            b_post: Array1::from_shape_fn(n, |_| uniform.sample(&mut rng) as f32),
            b_res: Array2::from_shape_fn((n, n), |_| uniform.sample(&mut rng) as f32),
            phi_pre: Array2::from_shape_fn((nc, n), |_| uniform.sample(&mut rng) as f32),
            phi_post: Array2::from_shape_fn((nc, n), |_| uniform.sample(&mut rng) as f32),
            phi_res: Array2::from_shape_fn((nc, n_sq), |_| uniform.sample(&mut rng) as f32),
            rmsnorm_weight: Array1::ones(nc),
        }
    }

    /// Get the total number of parameters.
    pub fn num_params(&self) -> usize {
        let n = self.b_pre.len();

        // Gating factors: 3
        let gating = 3;

        // Biases: n + n + n²
        let biases = n + n + n * n;

        // Projections: nC×n + nC×n + nC×n²
        let nc = self.phi_pre.nrows();
        let projections = nc * n + nc * n + nc * n * n;

        // RMSNorm: nC
        let rmsnorm = nc;

        gating + biases + projections + rmsnorm
    }

    /// Get memory usage in bytes (assuming f32).
    pub fn memory_bytes(&self) -> usize {
        self.num_params() * std::mem::size_of::<f32>()
    }

    /// Validate parameter shapes against configuration.
    pub fn validate(&self, config: &MhcConfig) -> Result<(), ParamsValidationError> {
        let n = config.expansion_rate;
        let nc = config.expanded_dim();
        let n_sq = n * n;

        if self.b_pre.len() != n {
            return Err(ParamsValidationError::ShapeMismatch {
                param: "b_pre".into(),
                expected: vec![n],
                actual: vec![self.b_pre.len()],
            });
        }

        if self.b_post.len() != n {
            return Err(ParamsValidationError::ShapeMismatch {
                param: "b_post".into(),
                expected: vec![n],
                actual: vec![self.b_post.len()],
            });
        }

        if self.b_res.shape() != [n, n] {
            return Err(ParamsValidationError::ShapeMismatch {
                param: "b_res".into(),
                expected: vec![n, n],
                actual: self.b_res.shape().to_vec(),
            });
        }

        if self.phi_pre.shape() != [nc, n] {
            return Err(ParamsValidationError::ShapeMismatch {
                param: "phi_pre".into(),
                expected: vec![nc, n],
                actual: self.phi_pre.shape().to_vec(),
            });
        }

        if self.phi_post.shape() != [nc, n] {
            return Err(ParamsValidationError::ShapeMismatch {
                param: "phi_post".into(),
                expected: vec![nc, n],
                actual: self.phi_post.shape().to_vec(),
            });
        }

        if self.phi_res.shape() != [nc, n_sq] {
            return Err(ParamsValidationError::ShapeMismatch {
                param: "phi_res".into(),
                expected: vec![nc, n_sq],
                actual: self.phi_res.shape().to_vec(),
            });
        }

        if self.rmsnorm_weight.len() != nc {
            return Err(ParamsValidationError::ShapeMismatch {
                param: "rmsnorm_weight".into(),
                expected: vec![nc],
                actual: vec![self.rmsnorm_weight.len()],
            });
        }

        Ok(())
    }
}

/// Computed mappings from a forward pass.
///
/// These are the actual H^pre, H^post, H^res matrices computed
/// from the input and parameters for a specific batch.
#[derive(Debug, Clone)]
pub struct MhcMappings {
    /// Pre-mapping H^pre. Shape: [batch, n]
    pub h_pre: Array2<f32>,

    /// Post-mapping H^post. Shape: [batch, n]
    pub h_post: Array2<f32>,

    /// Residual mapping H^res (doubly stochastic). Shape: [batch, n, n]
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
#[derive(Debug, Clone)]
pub struct MhcGradients {
    /// Gradient for alpha_pre.
    pub d_alpha_pre: f32,

    /// Gradient for alpha_post.
    pub d_alpha_post: f32,

    /// Gradient for alpha_res.
    pub d_alpha_res: f32,

    /// Gradient for b_pre. Shape: [n]
    pub d_b_pre: Array1<f32>,

    /// Gradient for b_post. Shape: [n]
    pub d_b_post: Array1<f32>,

    /// Gradient for b_res. Shape: [n, n]
    pub d_b_res: Array2<f32>,

    /// Gradient for phi_pre. Shape: [nC, n]
    pub d_phi_pre: Array2<f32>,

    /// Gradient for phi_post. Shape: [nC, n]
    pub d_phi_post: Array2<f32>,

    /// Gradient for phi_res. Shape: [nC, n²]
    pub d_phi_res: Array2<f32>,

    /// Gradient for rmsnorm_weight. Shape: [nC]
    pub d_rmsnorm_weight: Array1<f32>,
}

impl MhcGradients {
    /// Create zero gradients with the given configuration.
    pub fn zeros(config: &MhcConfig) -> Self {
        let n = config.expansion_rate;
        let nc = config.expanded_dim();
        let n_sq = n * n;

        Self {
            d_alpha_pre: 0.0,
            d_alpha_post: 0.0,
            d_alpha_res: 0.0,
            d_b_pre: Array1::zeros(n),
            d_b_post: Array1::zeros(n),
            d_b_res: Array2::zeros((n, n)),
            d_phi_pre: Array2::zeros((nc, n)),
            d_phi_post: Array2::zeros((nc, n)),
            d_phi_res: Array2::zeros((nc, n_sq)),
            d_rmsnorm_weight: Array1::zeros(nc),
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
        self.d_phi_pre += &other.d_phi_pre;
        self.d_phi_post += &other.d_phi_post;
        self.d_phi_res += &other.d_phi_res;
        self.d_rmsnorm_weight += &other.d_rmsnorm_weight;
    }

    /// Scale all gradients by a factor.
    pub fn scale(&mut self, factor: f32) {
        self.d_alpha_pre *= factor;
        self.d_alpha_post *= factor;
        self.d_alpha_res *= factor;
        self.d_b_pre *= factor;
        self.d_b_post *= factor;
        self.d_b_res *= factor;
        self.d_phi_pre *= factor;
        self.d_phi_post *= factor;
        self.d_phi_res *= factor;
        self.d_rmsnorm_weight *= factor;
    }

    /// Compute gradient norm (L2).
    pub fn norm(&self) -> f32 {
        let mut sum = 0.0f32;

        sum += self.d_alpha_pre * self.d_alpha_pre;
        sum += self.d_alpha_post * self.d_alpha_post;
        sum += self.d_alpha_res * self.d_alpha_res;

        sum += self.d_b_pre.iter().map(|x| x * x).sum::<f32>();
        sum += self.d_b_post.iter().map(|x| x * x).sum::<f32>();
        sum += self.d_b_res.iter().map(|x| x * x).sum::<f32>();

        sum += self.d_phi_pre.iter().map(|x| x * x).sum::<f32>();
        sum += self.d_phi_post.iter().map(|x| x * x).sum::<f32>();
        sum += self.d_phi_res.iter().map(|x| x * x).sum::<f32>();

        sum += self.d_rmsnorm_weight.iter().map(|x| x * x).sum::<f32>();

        sum.sqrt()
    }

    /// Clip gradients by norm.
    pub fn clip_norm(&mut self, max_norm: f32) {
        let norm = self.norm();
        if norm > max_norm {
            self.scale(max_norm / norm);
        }
    }
}

/// Errors during parameter validation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ParamsValidationError {
    #[error("Shape mismatch for {param}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        param: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_creation() {
        let config = MhcConfig::default();
        let params = MhcParams::new(&config);

        assert!(params.validate(&config).is_ok());
        assert!(params.num_params() > 0);
    }

    #[test]
    fn test_params_shapes() {
        let config = MhcConfig {
            expansion_rate: 4,
            hidden_dim: 256,
            ..Default::default()
        };
        let params = MhcParams::new(&config);

        assert_eq!(params.b_pre.len(), 4);
        assert_eq!(params.b_post.len(), 4);
        assert_eq!(params.b_res.shape(), &[4, 4]);
        assert_eq!(params.phi_pre.shape(), &[1024, 4]); // 4*256 = 1024
        assert_eq!(params.phi_post.shape(), &[1024, 4]);
        assert_eq!(params.phi_res.shape(), &[1024, 16]); // 4*4 = 16
        assert_eq!(params.rmsnorm_weight.len(), 1024);
    }

    #[test]
    fn test_gradients() {
        let config = MhcConfig {
            expansion_rate: 4,
            hidden_dim: 64,
            ..Default::default()
        };

        let mut grads = MhcGradients::zeros(&config);
        grads.d_alpha_pre = 1.0;
        grads.d_alpha_post = 2.0;
        grads.d_alpha_res = 3.0;

        let norm = grads.norm();
        assert!(norm > 0.0);

        grads.clip_norm(1.0);
        assert!(grads.norm() <= 1.0 + 1e-6);
    }

    #[test]
    fn test_initial_biases() {
        let config = MhcConfig {
            expansion_rate: 4,
            ..Default::default()
        };
        let params = MhcParams::new(&config);

        // b_pre should be uniform (1/n)
        for &v in params.b_pre.iter() {
            assert!((v - 0.25).abs() < 1e-6);
        }

        // b_post should be ones
        for &v in params.b_post.iter() {
            assert!((v - 1.0).abs() < 1e-6);
        }

        // b_res should be identity
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((params.b_res[[i, j]] - expected).abs() < 1e-6);
            }
        }
    }
}
