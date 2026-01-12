//! Manifold-Constrained Hyper-Connections (mHC) for stable deep learning training.
//!
//! This crate implements the mHC framework from DeepSeek-AI, which addresses training
//! instability in Hyper-Connections by constraining residual connection matrices to
//! lie on a doubly stochastic manifold.
//!
//! # Overview
//!
//! mHC extends standard residual connections by:
//! 1. Expanding the residual stream from dimension C to n×C (expansion rate n)
//! 2. Learning mappings H^pre, H^post, H^res that mix information across streams
//! 3. Constraining these mappings to be doubly stochastic via Sinkhorn-Knopp projection
//!
//! # Key Properties
//!
//! - **Identity Mapping**: Doubly stochastic matrices preserve signal flow
//! - **Compositional Closure**: Products of doubly stochastic matrices remain doubly stochastic
//! - **Stable Gradients**: Amax gain magnitude ≈ 1 throughout training
//!
//! # Example
//!
//! ```rust,ignore
//! use pmetal_mhc::{MhcConfig, MhcLayer, MhcParams, MhcPreset};
//!
//! // Create configuration from preset
//! let config = MhcConfig::from_preset(MhcPreset::Medium);
//!
//! // Initialize parameters
//! let params = MhcParams::new(&config);
//!
//! // Create mHC layer
//! let mut layer = MhcLayer::new(params, config);
//!
//! // Forward pass with sublayer computation
//! let output = layer.forward(&input, |h_in| {
//!     // Your attention/FFN computation here
//!     sublayer.forward(h_in)
//! });
//! ```
//!
//! # Metal Acceleration
//!
//! When the `metal` feature is enabled, GPU-accelerated kernels are used for:
//! - Fused RMSNorm + projection
//! - Batched Sinkhorn-Knopp iterations
//! - Fused post-mapping + residual merge
//!
//! # References
//!
//! - [mHC Paper](https://arxiv.org/abs/2512.24880): "mHC: Manifold-Constrained Hyper-Connections"
//! - [HC Paper](https://arxiv.org/abs/2409.19606): "Hyper-Connections"

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

pub mod config;
pub mod kernels;
pub mod layer;
pub mod mappings;
pub mod params;
pub mod sinkhorn;

// Re-exports for convenience
pub use config::{MhcConfig, MhcConfigError, MhcPreset};
pub use layer::{MhcCache, MhcLayer, MhcTransformerBlock};
pub use mappings::{apply_post_res_mapping, apply_pre_mapping, compute_mappings};
pub use params::{MhcGradients, MhcMappings, MhcParams};
pub use sinkhorn::{
    amax_gain_backward, amax_gain_forward, composite_mapping, is_doubly_stochastic, sinkhorn_knopp,
    sinkhorn_knopp_backward, sinkhorn_knopp_batch, SinkhornConfig,
};

/// Version of the mHC implementation.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default expansion rate (n=4 as per paper).
pub const DEFAULT_EXPANSION_RATE: usize = 4;

/// Default number of Sinkhorn iterations (t_max=20 as per paper).
pub const DEFAULT_SINKHORN_ITERATIONS: usize = 20;

/// Prelude module for convenient imports.
pub mod prelude {
    //! Convenient re-exports for common usage.
    pub use crate::config::{MhcConfig, MhcPreset};
    pub use crate::layer::{MhcLayer, MhcTransformerBlock};
    pub use crate::params::{MhcGradients, MhcMappings, MhcParams};
    pub use crate::sinkhorn::SinkhornConfig;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_sinkhorn_produces_doubly_stochastic() {
        let n = 4;
        let matrix = Array2::<f32>::from_shape_fn((n, n), |(i, j)| {
            ((i * n + j) as f32 + 1.0) * 0.1
        });

        let config = SinkhornConfig::default();
        let result = sinkhorn_knopp(&matrix, &config);

        assert!(is_doubly_stochastic(&result.matrix, 1e-5));
    }

    #[test]
    fn test_composite_preserves_doubly_stochastic() {
        let n = 4;
        let config = SinkhornConfig::default();

        let a_input = Array2::<f32>::from_shape_fn((n, n), |(i, j)| {
            ((i + j) as f32 + 0.1) * 0.1
        });
        let b_input = Array2::<f32>::from_shape_fn((n, n), |(i, j)| {
            ((i * 2 + j) as f32 + 0.2) * 0.1
        });

        let a = sinkhorn_knopp(&a_input, &config).matrix;
        let b = sinkhorn_knopp(&b_input, &config).matrix;

        let c = composite_mapping(&[a, b]);

        assert!(is_doubly_stochastic(&c, 1e-4));
    }

    #[test]
    fn test_doubly_stochastic_preserves_sum() {
        // Key property: A doubly stochastic matrix preserves the sum of each column
        // If H is doubly stochastic, then for any x, sum(H @ x, axis=0) = sum(x, axis=0)
        let n = 4;
        let c = 64;

        // Create a random doubly stochastic matrix
        let h_input = Array2::<f32>::from_shape_fn((n, n), |(i, j)| (i + j) as f32 * 0.1);
        let config = SinkhornConfig::default();
        let h = sinkhorn_knopp(&h_input, &config).matrix;

        // Create input matrix
        let x = Array2::<f32>::from_shape_fn((n, c), |(i, j)| (i * c + j) as f32);

        // Apply mapping: y = H @ x
        let y = h.dot(&x);

        // Check that column sums are preserved
        // sum(y, axis=0) should equal sum(x, axis=0) for doubly stochastic H
        for j in 0..c {
            let x_col_sum: f32 = (0..n).map(|i| x[[i, j]]).sum();
            let y_col_sum: f32 = (0..n).map(|i| y[[i, j]]).sum();
            assert!(
                (x_col_sum - y_col_sum).abs() < 1e-3,
                "Column {} sum not preserved: x={}, y={}",
                j,
                x_col_sum,
                y_col_sum
            );
        }
    }

    #[test]
    fn test_amax_gain_near_unity() {
        let n = 4;
        let config = SinkhornConfig::default();

        let h_pre_input = Array2::<f32>::from_shape_fn((n, n), |(i, j)| {
            ((i + j) as f32 * 0.1)
        });
        let h_post_input = Array2::<f32>::from_shape_fn((n, n), |(i, j)| {
            ((i * j) as f32 * 0.05 + 0.1)
        });
        let h_res_input = Array2::<f32>::from_shape_fn((n, n), |(i, j)| {
            ((i + j * 2) as f32 * 0.08)
        });

        let h_pre = sinkhorn_knopp(&h_pre_input, &config).matrix;
        let h_post = sinkhorn_knopp(&h_post_input, &config).matrix;
        let h_res = sinkhorn_knopp(&h_res_input, &config).matrix;

        // Compute composite for layer l
        let h_post_t = h_post.t().to_owned();
        let composite = composite_mapping(&[h_res, h_post_t, h_pre]);

        // Amax gain should be close to 1 for doubly stochastic composite
        let amax = amax_gain_forward(&composite);
        assert!(
            amax >= 0.5 && amax <= 2.0,
            "Amax gain out of expected range: {}",
            amax
        );
    }

    #[test]
    fn test_config_validation() {
        // Valid config
        let config = MhcConfig::from_preset(MhcPreset::Small);
        assert!(config.validate().is_ok());

        // Invalid expansion rate
        let mut invalid = config.clone();
        invalid.expansion_rate = 0;
        assert!(invalid.validate().is_err());

        // Invalid Sinkhorn iterations
        let mut invalid = config.clone();
        invalid.sinkhorn_iterations = 0;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_params_initialization() {
        let config = MhcConfig::from_preset(MhcPreset::Small);
        let params = MhcParams::new(&config);

        let n = config.expansion_rate;

        // Check dimensions - alpha values are scalars
        assert_eq!(params.b_pre.len(), n);
        assert_eq!(params.b_post.len(), n);
        assert_eq!(params.b_res.shape(), &[n, n]);

        // Check alpha initialization (should be near config.alpha_init)
        assert!(
            (params.alpha_pre - config.alpha_init).abs() < 0.5,
            "Alpha initialization off: {} vs {}",
            params.alpha_pre,
            config.alpha_init
        );
    }

    #[test]
    fn test_gradient_accumulation() {
        let config = MhcConfig::from_preset(MhcPreset::Small);
        let mut grads = MhcGradients::zeros(&config);

        grads.d_alpha_pre = 1.0;

        let mut other_grads = MhcGradients::zeros(&config);
        other_grads.d_alpha_pre = 2.0;

        grads.accumulate(&other_grads);

        // Should have accumulated
        assert!((grads.d_alpha_pre - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_clipping() {
        let config = MhcConfig::from_preset(MhcPreset::Small);
        let mut grads = MhcGradients::zeros(&config);

        grads.d_alpha_pre = 100.0;
        grads.d_alpha_post = 100.0;
        grads.d_alpha_res = 100.0;

        let initial_norm = grads.norm();
        assert!(initial_norm > 1.0);

        grads.clip_norm(1.0);

        // Norm should be clipped
        let clipped_norm = grads.norm();
        assert!(clipped_norm <= 1.0 + 1e-5, "Gradient norm not clipped: {}", clipped_norm);
    }
}
