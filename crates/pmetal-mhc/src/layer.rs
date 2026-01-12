//! mHC layer implementation.
//!
//! This module provides the main `MhcLayer` struct that wraps the mHC
//! computation for use in transformer architectures.

use crate::config::MhcConfig;
use crate::mappings::{
    apply_post_res_mapping, apply_pre_mapping, compute_mappings, compute_mappings_backward,
};
use crate::params::{MhcGradients, MhcMappings, MhcParams};
use ndarray::{Array2, Array3};
use std::sync::Arc;
use parking_lot::RwLock;

/// mHC layer that manages residual stream expansion and information mixing.
///
/// This layer implements the core mHC operations:
/// 1. Pre-mapping: Aggregate n streams into layer input
/// 2. Post-mapping + Residual: Mix layer output back into n streams
///
/// # Usage
///
/// ```ignore
/// let config = MhcConfig::default();
/// let layer = MhcLayer::new(config);
///
/// // In forward pass:
/// let (h_in, cache) = layer.pre_layer(&x);
/// let h_out = sublayer_fn(h_in);  // Attention or FFN
/// let x_next = layer.post_res_layer(&x, &h_out, &cache);
/// ```
#[derive(Debug)]
pub struct MhcLayer {
    /// Configuration.
    config: MhcConfig,

    /// Learnable parameters.
    params: Arc<RwLock<MhcParams>>,

    /// Accumulated gradients (for training).
    gradients: Arc<RwLock<Option<MhcGradients>>>,

    /// Layer index (for debugging/logging).
    layer_idx: usize,
}

impl MhcLayer {
    /// Create a new mHC layer.
    pub fn new(config: MhcConfig) -> Self {
        let params = MhcParams::new(&config);
        Self {
            config,
            params: Arc::new(RwLock::new(params)),
            gradients: Arc::new(RwLock::new(None)),
            layer_idx: 0,
        }
    }

    /// Create a new mHC layer with specific layer index.
    pub fn with_layer_idx(config: MhcConfig, layer_idx: usize) -> Self {
        let mut layer = Self::new(config);
        layer.layer_idx = layer_idx;
        layer
    }

    /// Create a new mHC layer with pre-initialized parameters.
    pub fn with_params(config: MhcConfig, params: MhcParams) -> Self {
        Self {
            config,
            params: Arc::new(RwLock::new(params)),
            gradients: Arc::new(RwLock::new(None)),
            layer_idx: 0,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &MhcConfig {
        &self.config
    }

    /// Get read access to parameters.
    pub fn params(&self) -> impl std::ops::Deref<Target = MhcParams> + '_ {
        self.params.read()
    }

    /// Get write access to parameters.
    pub fn params_mut(&self) -> impl std::ops::DerefMut<Target = MhcParams> + '_ {
        self.params.write()
    }

    /// Pre-layer computation: compute mappings and aggregate streams.
    ///
    /// # Arguments
    ///
    /// * `x` - Input residual stream. Shape: [batch, n, C]
    ///
    /// # Returns
    ///
    /// * Layer input (aggregated from streams). Shape: [batch, C]
    /// * Mapping cache for use in post_res_layer
    pub fn pre_layer(&self, x: &Array3<f32>) -> (Array2<f32>, MhcMappings) {
        let params = self.params.read();
        let mappings = compute_mappings(x, &params, &self.config);
        let h_in = apply_pre_mapping(x, &mappings.h_pre);
        (h_in, mappings)
    }

    /// Post-layer and residual computation: mix output back into streams.
    ///
    /// # Arguments
    ///
    /// * `x` - Input residual stream. Shape: [batch, n, C]
    /// * `h_out` - Layer output. Shape: [batch, C]
    /// * `mappings` - Mapping cache from pre_layer
    ///
    /// # Returns
    ///
    /// Updated residual stream. Shape: [batch, n, C]
    pub fn post_res_layer(
        &self,
        x: &Array3<f32>,
        h_out: &Array2<f32>,
        mappings: &MhcMappings,
    ) -> Array3<f32> {
        apply_post_res_mapping(x, h_out, &mappings.h_post, &mappings.h_res)
    }

    /// Combined forward pass (for simpler use cases).
    ///
    /// Applies pre-mapping, a layer function, and post-res mapping.
    ///
    /// # Arguments
    ///
    /// * `x` - Input residual stream. Shape: [batch, n, C]
    /// * `layer_fn` - Function to apply between pre and post mappings
    ///
    /// # Returns
    ///
    /// Updated residual stream. Shape: [batch, n, C]
    pub fn forward<F>(&self, x: &Array3<f32>, layer_fn: F) -> Array3<f32>
    where
        F: FnOnce(&Array2<f32>) -> Array2<f32>,
    {
        let (h_in, mappings) = self.pre_layer(x);
        let h_out = layer_fn(&h_in);
        self.post_res_layer(x, &h_out, &mappings)
    }

    /// Backward pass through the layer.
    ///
    /// # Arguments
    ///
    /// * `x` - Original input. Shape: [batch, n, C]
    /// * `h_out` - Original layer output. Shape: [batch, C]
    /// * `mappings` - Cached mappings from forward pass
    /// * `grad_output` - Gradient from next layer. Shape: [batch, n, C]
    /// * `grad_h_out` - Additional gradient for h_out (from layer backward). Shape: [batch, C]
    ///
    /// # Returns
    ///
    /// * Gradient w.r.t. input x. Shape: [batch, n, C]
    /// * Gradient w.r.t. h_in (for layer backward). Shape: [batch, C]
    pub fn backward(
        &self,
        x: &Array3<f32>,
        h_out: &Array2<f32>,
        mappings: &MhcMappings,
        grad_output: &Array3<f32>,
        grad_h_out_additional: Option<&Array2<f32>>,
    ) -> (Array3<f32>, Array2<f32>) {
        let batch_size = x.shape()[0];
        let n = self.config.expansion_rate;
        let c = self.config.hidden_dim;

        // Backward through post_res_layer:
        // x_out = H^res @ x + H^post^T @ h_out
        //
        // grad_x = H^res^T @ grad_output
        // grad_h_out = H^post @ grad_output (summed over streams)
        // grad_H^res = grad_output @ x^T
        // grad_H^post = grad_output^T @ h_out (summed appropriately)

        let mut grad_x = Array3::zeros((batch_size, n, c));
        let mut grad_h_out = Array2::zeros((batch_size, c));
        let mut grad_h_res = Array3::zeros((batch_size, n, n));
        let mut grad_h_post = Array2::zeros((batch_size, n));

        for b in 0..batch_size {
            // grad_x from H^res
            for j in 0..n {
                for k in 0..c {
                    let mut sum = 0.0f32;
                    for i in 0..n {
                        sum += mappings.h_res[[b, i, j]] * grad_output[[b, i, k]];
                    }
                    grad_x[[b, j, k]] = sum;
                }
            }

            // grad_h_out from H^post
            for k in 0..c {
                let mut sum = 0.0f32;
                for i in 0..n {
                    sum += mappings.h_post[[b, i]] * grad_output[[b, i, k]];
                }
                grad_h_out[[b, k]] = sum;
            }

            // grad_H^res
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for k in 0..c {
                        sum += grad_output[[b, i, k]] * x[[b, j, k]];
                    }
                    grad_h_res[[b, i, j]] = sum;
                }
            }

            // grad_H^post
            for i in 0..n {
                let mut sum = 0.0f32;
                for k in 0..c {
                    sum += grad_output[[b, i, k]] * h_out[[b, k]];
                }
                grad_h_post[[b, i]] = sum;
            }
        }

        // Add any additional gradient for h_out (from layer backward)
        if let Some(grad_add) = grad_h_out_additional {
            grad_h_out = grad_h_out + grad_add;
        }

        // Backward through pre_layer:
        // h_in = H^pre @ x (over streams)
        //
        // We need grad_h_in which will be provided from the layer's backward pass
        // For now, we compute grad_x from pre_layer separately

        // Backward through mapping computation
        // This computes grad_x from the mapping computation and parameter gradients
        let (grad_x_mappings, param_gradients) = compute_mappings_backward(
            x,
            &self.params.read(),
            &self.config,
            &grad_h_post.mapv(|_| 0.0), // We'll handle H^pre gradient separately
            &grad_h_post,
            &grad_h_res,
        );

        // Accumulate parameter gradients
        {
            let mut grads = self.gradients.write();
            if let Some(ref mut g) = *grads {
                g.accumulate(&param_gradients);
            } else {
                *grads = Some(param_gradients);
            }
        }

        // Combine gradients
        let grad_x_total = grad_x + grad_x_mappings;

        // Gradient for h_in (to pass to layer backward)
        let grad_h_in = grad_h_out; // This will be used by the sublayer's backward

        (grad_x_total, grad_h_in)
    }

    /// Get accumulated gradients and reset.
    pub fn take_gradients(&self) -> Option<MhcGradients> {
        self.gradients.write().take()
    }

    /// Reset gradients to zero.
    pub fn zero_gradients(&self) {
        *self.gradients.write() = None;
    }

    /// Apply gradients to parameters using SGD.
    pub fn apply_gradients_sgd(&self, lr: f32) {
        if let Some(grads) = self.take_gradients() {
            let mut params = self.params.write();

            params.alpha_pre -= lr * grads.d_alpha_pre;
            params.alpha_post -= lr * grads.d_alpha_post;
            params.alpha_res -= lr * grads.d_alpha_res;

            params.b_pre = &params.b_pre - &(&grads.d_b_pre * lr);
            params.b_post = &params.b_post - &(&grads.d_b_post * lr);
            params.b_res = &params.b_res - &(&grads.d_b_res * lr);

            params.phi_pre = &params.phi_pre - &(&grads.d_phi_pre * lr);
            params.phi_post = &params.phi_post - &(&grads.d_phi_post * lr);
            params.phi_res = &params.phi_res - &(&grads.d_phi_res * lr);

            params.rmsnorm_weight = &params.rmsnorm_weight - &(&grads.d_rmsnorm_weight * lr);
        }
    }

    /// Get number of parameters in this layer.
    pub fn num_params(&self) -> usize {
        self.params.read().num_params()
    }

    /// Get memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.params.read().memory_bytes()
    }

    /// Clone this layer (deep copy of parameters).
    pub fn deep_clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            params: Arc::new(RwLock::new(self.params.read().clone())),
            gradients: Arc::new(RwLock::new(None)),
            layer_idx: self.layer_idx,
        }
    }
}

impl Clone for MhcLayer {
    /// Shallow clone (shares parameters).
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            params: Arc::clone(&self.params),
            gradients: Arc::new(RwLock::new(None)), // Don't share gradients
            layer_idx: self.layer_idx,
        }
    }
}

/// Cache for mHC forward pass (for backward recomputation).
#[derive(Debug, Clone)]
pub struct MhcCache {
    /// Original input.
    pub x: Array3<f32>,

    /// Layer output.
    pub h_out: Array2<f32>,

    /// Computed mappings.
    pub mappings: MhcMappings,
}

/// A complete mHC transformer block.
///
/// This wraps two mHC layers (for attention and FFN) along with
/// the sublayer functions.
pub struct MhcTransformerBlock<A, F>
where
    A: Fn(&Array2<f32>) -> Array2<f32>,
    F: Fn(&Array2<f32>) -> Array2<f32>,
{
    /// mHC layer for attention sublayer.
    pub mhc_attn: MhcLayer,

    /// mHC layer for FFN sublayer.
    pub mhc_ffn: MhcLayer,

    /// Attention function.
    pub attention: A,

    /// FFN function.
    pub ffn: F,
}

impl<A, F> MhcTransformerBlock<A, F>
where
    A: Fn(&Array2<f32>) -> Array2<f32>,
    F: Fn(&Array2<f32>) -> Array2<f32>,
{
    /// Create a new transformer block.
    pub fn new(config: MhcConfig, attention: A, ffn: F, layer_idx: usize) -> Self {
        Self {
            mhc_attn: MhcLayer::with_layer_idx(config.clone(), layer_idx * 2),
            mhc_ffn: MhcLayer::with_layer_idx(config, layer_idx * 2 + 1),
            attention,
            ffn,
        }
    }

    /// Forward pass through the block.
    pub fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        // Attention sublayer with mHC
        let x = self.mhc_attn.forward(x, &self.attention);

        // FFN sublayer with mHC
        self.mhc_ffn.forward(&x, &self.ffn)
    }
}

/// Expand input from single stream to n streams.
///
/// Replicates the input n times along a new dimension.
///
/// # Arguments
///
/// * `x` - Single-stream input. Shape: [batch, C]
/// * `n` - Expansion rate
///
/// # Returns
///
/// Expanded multi-stream tensor. Shape: [batch, n, C]
pub fn expand_to_streams(x: &Array2<f32>, n: usize) -> Array3<f32> {
    let batch_size = x.nrows();
    let c = x.ncols();

    let mut expanded = Array3::zeros((batch_size, n, c));

    for b in 0..batch_size {
        for i in 0..n {
            for j in 0..c {
                expanded[[b, i, j]] = x[[b, j]];
            }
        }
    }

    expanded
}

/// Collapse n streams back to single stream.
///
/// Takes the first stream (or averages, depending on mode).
///
/// # Arguments
///
/// * `x` - Multi-stream tensor. Shape: [batch, n, C]
/// * `mode` - Collapse mode: "first", "average", or "sum"
///
/// # Returns
///
/// Single-stream output. Shape: [batch, C]
pub fn collapse_streams(x: &Array3<f32>, mode: &str) -> Array2<f32> {
    let batch_size = x.shape()[0];
    let n = x.shape()[1];
    let c = x.shape()[2];

    let mut collapsed = Array2::zeros((batch_size, c));

    match mode {
        "first" => {
            for b in 0..batch_size {
                for j in 0..c {
                    collapsed[[b, j]] = x[[b, 0, j]];
                }
            }
        }
        "average" => {
            let scale = 1.0 / n as f32;
            for b in 0..batch_size {
                for j in 0..c {
                    let sum: f32 = (0..n).map(|i| x[[b, i, j]]).sum();
                    collapsed[[b, j]] = sum * scale;
                }
            }
        }
        "sum" => {
            for b in 0..batch_size {
                for j in 0..c {
                    let sum: f32 = (0..n).map(|i| x[[b, i, j]]).sum();
                    collapsed[[b, j]] = sum;
                }
            }
        }
        _ => panic!("Unknown collapse mode: {}", mode),
    }

    collapsed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let config = MhcConfig::default();
        let layer = MhcLayer::new(config);

        assert!(layer.num_params() > 0);
    }

    #[test]
    fn test_pre_layer() {
        let config = MhcConfig {
            expansion_rate: 4,
            hidden_dim: 64,
            ..Default::default()
        };
        let layer = MhcLayer::new(config);

        let x = Array3::from_shape_fn((8, 4, 64), |(b, i, j)| ((b + i + j) as f32) * 0.01);
        let (h_in, mappings) = layer.pre_layer(&x);

        assert_eq!(h_in.shape(), &[8, 64]);
        assert_eq!(mappings.h_pre.shape(), &[8, 4]);
        assert_eq!(mappings.h_post.shape(), &[8, 4]);
        assert_eq!(mappings.h_res.shape(), &[8, 4, 4]);
    }

    #[test]
    fn test_post_res_layer() {
        let config = MhcConfig {
            expansion_rate: 4,
            hidden_dim: 64,
            ..Default::default()
        };
        let layer = MhcLayer::new(config);

        let x = Array3::from_shape_fn((8, 4, 64), |(b, i, j)| ((b + i + j) as f32) * 0.01);
        let (_, mappings) = layer.pre_layer(&x);
        let h_out = Array2::from_shape_fn((8, 64), |(b, j)| ((b + j) as f32) * 0.02);

        let x_next = layer.post_res_layer(&x, &h_out, &mappings);

        assert_eq!(x_next.shape(), &[8, 4, 64]);
    }

    #[test]
    fn test_full_forward() {
        let config = MhcConfig {
            expansion_rate: 4,
            hidden_dim: 64,
            ..Default::default()
        };
        let layer = MhcLayer::new(config);

        let x = Array3::from_shape_fn((8, 4, 64), |(b, i, j)| ((b + i + j) as f32) * 0.01);

        // Identity layer function
        let x_next = layer.forward(&x, |h_in| h_in.clone());

        assert_eq!(x_next.shape(), &[8, 4, 64]);
    }

    #[test]
    fn test_expand_collapse() {
        let x = Array2::from_shape_fn((4, 8), |(b, j)| ((b + j) as f32));

        let expanded = expand_to_streams(&x, 4);
        assert_eq!(expanded.shape(), &[4, 4, 8]);

        // All streams should have same values
        for b in 0..4 {
            for i in 0..4 {
                for j in 0..8 {
                    assert_eq!(expanded[[b, i, j]], x[[b, j]]);
                }
            }
        }

        let collapsed = collapse_streams(&expanded, "first");
        assert_eq!(collapsed.shape(), &[4, 8]);

        // Should match original
        for b in 0..4 {
            for j in 0..8 {
                assert_eq!(collapsed[[b, j]], x[[b, j]]);
            }
        }
    }

    #[test]
    fn test_collapse_average() {
        let mut expanded = Array3::zeros((2, 4, 3));
        for b in 0..2 {
            for i in 0..4 {
                for j in 0..3 {
                    expanded[[b, i, j]] = (i + 1) as f32; // 1, 2, 3, 4
                }
            }
        }

        let collapsed = collapse_streams(&expanded, "average");

        // Average of 1, 2, 3, 4 = 2.5
        for b in 0..2 {
            for j in 0..3 {
                assert!((collapsed[[b, j]] - 2.5).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_gradient_accumulation() {
        let config = MhcConfig {
            expansion_rate: 4,
            hidden_dim: 32,
            ..Default::default()
        };
        let layer = MhcLayer::new(config);

        // Run a forward pass
        let x = Array3::from_shape_fn((4, 4, 32), |_| rand::random::<f32>() - 0.5);
        let (h_in, mappings) = layer.pre_layer(&x);
        let h_out = h_in.clone(); // Identity

        // Fake backward
        let grad_output = Array3::from_shape_fn((4, 4, 32), |_| rand::random::<f32>() - 0.5);
        let _ = layer.backward(&x, &h_out, &mappings, &grad_output, None);

        // Check that gradients were accumulated
        let grads = layer.take_gradients();
        assert!(grads.is_some());
    }

    #[test]
    fn test_deep_clone() {
        let config = MhcConfig::default();
        let layer1 = MhcLayer::new(config);
        let layer2 = layer1.deep_clone();

        // Modify layer1's params
        layer1.params_mut().alpha_pre = 0.5;

        // layer2 should be unaffected
        assert!((layer2.params().alpha_pre - 0.01).abs() < 1e-6);
    }
}
