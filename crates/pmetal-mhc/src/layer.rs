//! mHC layer implementation.
//!
//! This module provides the main `MhcLayer` struct that wraps the mHC
//! computation for use in transformer architectures.

use crate::config::MhcConfig;
use crate::error::MhcError;
use crate::mappings::{
    apply_post_res_mapping, apply_post_res_mapping_backward, apply_pre_mapping,
    apply_pre_mapping_backward, check_stream_shape, compute_mappings, compute_mappings_backward,
};
use crate::params::{MhcGradients, MhcMappings, MhcParams};
use ndarray::{Array2, Array3, Axis};
use std::str::FromStr;

/// mHC layer that manages residual stream expansion and information mixing.
///
/// A layer wraps one sublayer `F` (attention or a feed-forward block) of a
/// transformer whose residual stream has been widened to `n` streams:
///
/// ```text
/// x_{l+1} = H^res · x_l + H^post^T · F(H^pre · x_l)
/// ```
///
/// 1. Pre-mapping: aggregate the n streams into the sublayer's input
/// 2. Post-mapping + residual: mix the sublayer's output back into the n streams
///
/// All activations are `[batch, n, C]`, one residual stream per row of `batch`
/// (tokens); the layer owns its parameters and is `Send + Sync`.
///
/// # Training
///
/// The sublayer's backward pass sits between the two halves of this layer's,
/// so [`MhcLayer::backward`] takes it as a closure:
///
/// ```
/// use ndarray::{Array2, Array3};
/// use pmetal_mhc::{MhcConfig, MhcLayer};
///
/// # fn main() -> Result<(), pmetal_mhc::MhcError> {
/// let config = MhcConfig { hidden_dim: 8, ..MhcConfig::default() };
/// let mut layer = MhcLayer::new(config)?;
///
/// let x = Array3::from_elem((2, 4, 8), 0.5f32);
/// let (h_in, mappings) = layer.pre_layer(&x)?;
/// let h_out = h_in.mapv(f32::tanh); // the sublayer
/// let x_next = layer.post_res_layer(&x, &h_out, &mappings)?;
///
/// let grad_output = Array3::from_elem(x_next.raw_dim(), 1.0f32);
/// let (grad_x, grads) = layer.backward(&x, &h_out, &mappings, &grad_output, |grad_h_out| {
///     // the sublayer's own backward: d tanh(h) = 1 - tanh(h)^2
///     grad_h_out * &h_in.mapv(|v| 1.0 - v.tanh().powi(2))
/// })?;
/// assert_eq!(grad_x.shape(), x.shape());
/// layer.apply_gradients_sgd(&grads, 1e-3);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct MhcLayer {
    config: MhcConfig,
    params: MhcParams,
}

impl MhcLayer {
    /// Create a new mHC layer: [`MhcParams::new`] initialization, a plain
    /// residual connection to start with.
    ///
    /// # Errors
    ///
    /// If `config` is invalid.
    pub fn new(config: MhcConfig) -> Result<Self, MhcError> {
        config.validate()?;
        let params = MhcParams::new(&config);
        Ok(Self { config, params })
    }

    /// Create a new mHC layer with a reproducible initialization.
    ///
    /// # Errors
    ///
    /// If `config` is invalid.
    pub fn with_rng<R: rand::Rng + ?Sized>(
        config: MhcConfig,
        rng: &mut R,
    ) -> Result<Self, MhcError> {
        config.validate()?;
        let params = MhcParams::with_rng(&config, rng);
        Ok(Self { config, params })
    }

    /// Create a new mHC layer with pre-initialized parameters.
    ///
    /// # Errors
    ///
    /// If `config` is invalid or `params` do not match it.
    pub fn with_params(config: MhcConfig, params: MhcParams) -> Result<Self, MhcError> {
        config.validate()?;
        params.validate(&config)?;
        Ok(Self { config, params })
    }

    /// Get the configuration.
    pub fn config(&self) -> &MhcConfig {
        &self.config
    }

    /// Get the parameters.
    pub fn params(&self) -> &MhcParams {
        &self.params
    }

    /// Get mutable access to the parameters (an optimizer's entry point).
    ///
    /// Shapes must stay as [`MhcParams::validate`] requires.
    pub fn params_mut(&mut self) -> &mut MhcParams {
        &mut self.params
    }

    fn check_batch(
        &self,
        x: &Array3<f32>,
        h_out: Option<&Array2<f32>>,
        mappings: Option<&MhcMappings>,
    ) -> Result<(), MhcError> {
        check_stream_shape(&x.view(), &self.config)?;
        let batch = x.shape()[0];
        let (n, c) = (self.config.expansion_rate, self.config.hidden_dim);
        if let Some(h) = h_out
            && h.shape() != [batch, c]
        {
            return Err(MhcError::shape(
                "layer output",
                format!("[{batch}, {c}]"),
                h.shape(),
            ));
        }
        if let Some(m) = mappings {
            if m.h_pre.shape() != [batch, n] || m.h_post.shape() != [batch, n] {
                return Err(MhcError::shape(
                    "H^pre / H^post",
                    format!("[{batch}, {n}]"),
                    m.h_pre.shape(),
                ));
            }
            if m.h_res.shape() != [batch, n, n] {
                return Err(MhcError::shape(
                    "H^res",
                    format!("[{batch}, {n}, {n}]"),
                    m.h_res.shape(),
                ));
            }
        }
        Ok(())
    }

    /// Pre-layer computation: compute mappings and aggregate streams.
    ///
    /// # Arguments
    ///
    /// * `x` - Input residual stream. Shape: `[batch, n, C]`
    ///
    /// # Returns
    ///
    /// * Layer input (aggregated from streams). Shape: `[batch, C]`
    /// * The mappings, for [`post_res_layer`](Self::post_res_layer) and
    ///   [`backward`](Self::backward)
    ///
    /// # Errors
    ///
    /// If `x` is not `[batch, n, C]` for this layer.
    pub fn pre_layer(&self, x: &Array3<f32>) -> Result<(Array2<f32>, MhcMappings), MhcError> {
        self.check_batch(x, None, None)?;
        let mappings = compute_mappings(x, &self.params, &self.config);
        let h_in = apply_pre_mapping(x, &mappings.h_pre);
        Ok((h_in, mappings))
    }

    /// Post-layer and residual computation: mix output back into streams.
    ///
    /// # Arguments
    ///
    /// * `x` - Input residual stream. Shape: `[batch, n, C]`
    /// * `h_out` - Layer output. Shape: `[batch, C]`
    /// * `mappings` - Mappings from [`pre_layer`](Self::pre_layer)
    ///
    /// # Returns
    ///
    /// Updated residual stream. Shape: `[batch, n, C]`
    ///
    /// # Errors
    ///
    /// If any argument's shape disagrees with the layer's.
    pub fn post_res_layer(
        &self,
        x: &Array3<f32>,
        h_out: &Array2<f32>,
        mappings: &MhcMappings,
    ) -> Result<Array3<f32>, MhcError> {
        self.check_batch(x, Some(h_out), Some(mappings))?;
        Ok(apply_post_res_mapping(
            x,
            h_out,
            &mappings.h_post,
            &mappings.h_res,
        ))
    }

    /// Combined forward pass (for simpler use cases).
    ///
    /// Applies pre-mapping, a layer function, and post-res mapping.
    ///
    /// # Arguments
    ///
    /// * `x` - Input residual stream. Shape: `[batch, n, C]`
    /// * `layer_fn` - Function to apply between pre and post mappings
    ///
    /// # Returns
    ///
    /// Updated residual stream. Shape: `[batch, n, C]`
    ///
    /// # Errors
    ///
    /// If `x` or `layer_fn`'s output do not have the layer's shapes.
    pub fn forward<F>(&self, x: &Array3<f32>, layer_fn: F) -> Result<Array3<f32>, MhcError>
    where
        F: FnOnce(&Array2<f32>) -> Array2<f32>,
    {
        let (h_in, mappings) = self.pre_layer(x)?;
        let h_out = layer_fn(&h_in);
        self.post_res_layer(x, &h_out, &mappings)
    }

    /// Backward pass through the layer.
    ///
    /// Differentiates `x_out = H^res x + H^postᵀ h_out` with `h_in = H^pre x`
    /// and the mappings' own dependence on `x` and the parameters, around the
    /// sublayer `h_out = F(h_in)` whose backward pass is `sublayer_backward`.
    ///
    /// # Arguments
    ///
    /// * `x` - Original input. Shape: `[batch, n, C]`
    /// * `h_out` - The sublayer's output from the forward pass. Shape: `[batch, C]`
    /// * `mappings` - The mappings from the forward pass
    /// * `grad_output` - Gradient from the next layer. Shape: `[batch, n, C]`
    /// * `sublayer_backward` - Given the gradient with respect to the sublayer's
    ///   output (`[batch, C]`), runs the sublayer's backward pass (including
    ///   accumulating its own parameter gradients) and returns the gradient
    ///   with respect to its input (`[batch, C]`)
    ///
    /// # Returns
    ///
    /// * Gradient with respect to the input `x`. Shape: `[batch, n, C]`
    /// * Gradients of this layer's parameters, for the optimizer (add them
    ///   across micro-batches with [`MhcGradients::accumulate`])
    ///
    /// # Errors
    ///
    /// If a shape disagrees with the layer's, including what `sublayer_backward`
    /// returns.
    pub fn backward<F>(
        &self,
        x: &Array3<f32>,
        h_out: &Array2<f32>,
        mappings: &MhcMappings,
        grad_output: &Array3<f32>,
        sublayer_backward: F,
    ) -> Result<(Array3<f32>, MhcGradients), MhcError>
    where
        F: FnOnce(&Array2<f32>) -> Array2<f32>,
    {
        self.check_batch(x, Some(h_out), Some(mappings))?;
        if grad_output.shape() != x.shape() {
            return Err(MhcError::shape(
                "grad_output",
                format!("{:?}", x.shape()),
                grad_output.shape(),
            ));
        }

        // x_out = H^res x + H^postᵀ h_out
        let post_res = apply_post_res_mapping_backward(
            x,
            h_out,
            &mappings.h_post,
            &mappings.h_res,
            grad_output,
        );

        // h_out = F(h_in): the sublayer closes the loop.
        let grad_h_in = sublayer_backward(&post_res.grad_h_out);
        if grad_h_in.shape() != post_res.grad_h_out.shape() {
            return Err(MhcError::shape(
                "sublayer input gradient",
                format!("{:?}", post_res.grad_h_out.shape()),
                grad_h_in.shape(),
            ));
        }

        // h_in = H^pre x
        let (grad_x_pre, grad_h_pre) = apply_pre_mapping_backward(x, &mappings.h_pre, &grad_h_in);

        // H^pre, H^post, H^res = f(x, parameters)
        let (grad_x_mappings, grads) = compute_mappings_backward(
            x,
            &self.params,
            &self.config,
            &grad_h_pre,
            &post_res.grad_h_post,
            &post_res.grad_h_res,
        );

        let grad_x = post_res.grad_x + grad_x_pre + grad_x_mappings;
        Ok((grad_x, grads))
    }

    /// Apply gradients to the parameters with plain SGD.
    pub fn apply_gradients_sgd(&mut self, grads: &MhcGradients, lr: f32) {
        let p = &mut self.params;
        p.alpha_pre -= lr * grads.d_alpha_pre;
        p.alpha_post -= lr * grads.d_alpha_post;
        p.alpha_res -= lr * grads.d_alpha_res;
        p.b_pre.scaled_add(-lr, &grads.d_b_pre);
        p.b_post.scaled_add(-lr, &grads.d_b_post);
        p.b_res.scaled_add(-lr, &grads.d_b_res);
        p.phi.scaled_add(-lr, &grads.d_phi);
    }

    /// Get number of parameters in this layer.
    pub fn num_params(&self) -> usize {
        self.params.num_params()
    }

    /// Get memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.params.memory_bytes()
    }
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
    /// Create a new transformer block with independently initialized mHC layers.
    ///
    /// # Errors
    ///
    /// If `config` is invalid.
    pub fn new(config: MhcConfig, attention: A, ffn: F) -> Result<Self, MhcError> {
        Ok(Self {
            mhc_attn: MhcLayer::new(config.clone())?,
            mhc_ffn: MhcLayer::new(config)?,
            attention,
            ffn,
        })
    }

    /// Forward pass through the block.
    ///
    /// # Errors
    ///
    /// If `x` or a sublayer's output has the wrong shape.
    pub fn forward(&self, x: &Array3<f32>) -> Result<Array3<f32>, MhcError> {
        // Attention sublayer with mHC
        let x = self.mhc_attn.forward(x, &self.attention)?;

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
/// * `x` - Single-stream input. Shape: `[batch, C]`
/// * `n` - Expansion rate
///
/// # Returns
///
/// Expanded multi-stream tensor. Shape: `[batch, n, C]`
pub fn expand_to_streams(x: &Array2<f32>, n: usize) -> Array3<f32> {
    let (batch, c) = x.dim();
    x.view()
        .insert_axis(Axis(1))
        .broadcast((batch, n, c))
        .expect("an inserted unit axis broadcasts to any length")
        .to_owned()
}

/// Collapse mode for multi-stream to single-stream reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollapseMode {
    /// Use only the first stream.
    First,
    /// Average across all streams.
    Average,
    /// Sum across all streams.
    Sum,
}

impl FromStr for CollapseMode {
    type Err = MhcError;

    /// Parse `"first"`, `"average"` or `"sum"`.
    fn from_str(s: &str) -> Result<Self, MhcError> {
        match s {
            "first" => Ok(Self::First),
            "average" => Ok(Self::Average),
            "sum" => Ok(Self::Sum),
            _ => Err(MhcError::InvalidCollapseMode(s.to_string())),
        }
    }
}

/// Collapse n streams back to single stream.
///
/// Takes the first stream (or averages, depending on mode).
///
/// # Arguments
///
/// * `x` - Multi-stream tensor. Shape: `[batch, n, C]` with `n >= 1`
/// * `mode` - Collapse mode
///
/// # Returns
///
/// Single-stream output. Shape: `[batch, C]`
pub fn collapse_streams(x: &Array3<f32>, mode: CollapseMode) -> Array2<f32> {
    match mode {
        CollapseMode::First => x.index_axis(Axis(1), 0).to_owned(),
        CollapseMode::Average => x
            .mean_axis(Axis(1))
            .expect("collapsing needs at least one stream"),
        CollapseMode::Sum => x.sum_axis(Axis(1)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn config(c: usize) -> MhcConfig {
        MhcConfig {
            hidden_dim: c,
            ..Default::default()
        }
    }

    #[test]
    fn invalid_config_is_an_error() {
        let bad = MhcConfig {
            expansion_rate: 0,
            ..Default::default()
        };
        assert!(matches!(MhcLayer::new(bad), Err(MhcError::Config(_))));
    }

    #[test]
    fn mismatched_params_are_an_error() {
        let params = MhcParams::identity_init(&config(8));
        assert!(matches!(
            MhcLayer::with_params(config(16), params),
            Err(MhcError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn shape_errors_name_the_argument() {
        let layer = MhcLayer::new(config(8)).unwrap();
        let bad_x = Array3::<f32>::zeros((2, 4, 9));
        assert!(matches!(
            layer.pre_layer(&bad_x),
            Err(MhcError::ShapeMismatch {
                what: "residual stream",
                ..
            })
        ));
        let x = Array3::<f32>::zeros((2, 4, 8));
        let (_, m) = layer.pre_layer(&x).unwrap();
        let bad_out = Array2::<f32>::zeros((3, 8));
        assert!(matches!(
            layer.post_res_layer(&x, &bad_out, &m),
            Err(MhcError::ShapeMismatch {
                what: "layer output",
                ..
            })
        ));
        let grad = Array3::<f32>::zeros((2, 4, 8));
        let h_out = Array2::<f32>::zeros((2, 8));
        let wrong = layer.backward(&x, &h_out, &m, &grad, |_| Array2::zeros((2, 7)));
        assert!(matches!(
            wrong,
            Err(MhcError::ShapeMismatch {
                what: "sublayer input gradient",
                ..
            })
        ));
    }

    #[test]
    fn a_fresh_layer_is_a_plain_residual_connection() {
        // Equal streams: H^pre averages them to x, H^post = 1, H^res ≈ I, so
        // every output stream is x + F(x) to the identity-init tolerance.
        let layer = MhcLayer::new(config(16)).unwrap();
        let base =
            Array2::from_shape_fn((3, 16), |(b, k)| ((b * 5 + k * 3) % 7) as f32 * 0.3 - 1.0);
        let x = expand_to_streams(&base, 4);
        let f = |h: &Array2<f32>| h.mapv(|v| v * v + 0.5 * v);
        let y = layer.forward(&x, f).unwrap();
        let want = &base + &f(&base);
        for s in 0..4 {
            for b in 0..3 {
                for k in 0..16 {
                    let got = y[[b, s, k]];
                    let w = want[[b, k]];
                    assert!(
                        (got - w).abs() < 2e-2 * (1.0 + w.abs()),
                        "[{b},{s},{k}] {got} vs {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn deep_stack_keeps_the_streams_bounded() {
        // 60 layers whose sublayer is zero: the residual path alone, through
        // 60 random doubly stochastic mixes, must not grow or shrink the
        // streams (Amax gain ≤ 1 + rounding).
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(3);
        let c = 8;
        let layers: Vec<MhcLayer> = (0..60)
            .map(|_| {
                let mut l = MhcLayer::with_rng(config(c), &mut rng).unwrap();
                // Make the residual mixing genuinely non-trivial.
                l.params_mut().b_res = Array2::from_shape_fn((4, 4), |(i, j)| {
                    ((i * 3 + j * 7) % 5) as f32 * 0.5 - 1.0
                });
                l
            })
            .collect();
        let mut x =
            Array3::from_shape_fn((2, 4, c), |(b, s, k)| ((b + 2 * s + k) % 5) as f32 - 2.0);
        let norm_in = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        for l in &layers {
            x = l.forward(&x, |h| Array2::zeros(h.raw_dim())).unwrap();
        }
        let norm_out = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            norm_out <= norm_in * (1.0 + 1e-3),
            "{norm_in} -> {norm_out}"
        );
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sgd_moves_every_parameter_against_its_gradient() {
        let mut layer = MhcLayer::new(config(4)).unwrap();
        let before = layer.params().clone();
        let mut g = MhcGradients::zeros(layer.config());
        g.d_alpha_pre = 2.0;
        g.d_b_res[[0, 1]] = -1.0;
        g.d_phi[[3, 5]] = 4.0;
        layer.apply_gradients_sgd(&g, 0.5);
        assert!((layer.params().alpha_pre - (before.alpha_pre - 1.0)).abs() < 1e-7);
        assert_eq!(layer.params().b_res[[0, 1]], before.b_res[[0, 1]] + 0.5);
        assert_eq!(layer.params().phi[[3, 5]], before.phi[[3, 5]] - 2.0);
        assert_eq!(layer.params().b_pre, before.b_pre);
    }

    #[test]
    fn clone_is_independent() {
        let a = MhcLayer::new(config(4)).unwrap();
        let mut b = a.clone();
        b.params_mut().alpha_pre = 0.5;
        assert!((a.params().alpha_pre - 0.01).abs() < 1e-7);
    }

    #[test]
    fn test_expand_collapse() {
        let x = array![[1.0f32, 2.0], [3.0, 4.0]];
        let e = expand_to_streams(&x, 3);
        assert_eq!(e.shape(), &[2, 3, 2]);
        for s in 0..3 {
            assert_eq!(e.index_axis(Axis(1), s), x);
        }
        assert_eq!(collapse_streams(&e, CollapseMode::First), x);
        assert_eq!(collapse_streams(&e, CollapseMode::Sum), &x * 3.0);
        assert_eq!(collapse_streams(&e, CollapseMode::Average), x);
    }

    #[test]
    fn collapse_average_of_distinct_streams() {
        let mut e = Array3::zeros((2, 4, 3));
        for i in 0..4 {
            e.index_axis_mut(Axis(1), i).fill((i + 1) as f32);
        }
        assert!(
            collapse_streams(&e, CollapseMode::Average)
                .iter()
                .all(|&v| (v - 2.5).abs() < 1e-6)
        );
    }

    #[test]
    fn collapse_mode_parses() {
        assert_eq!(
            "first".parse::<CollapseMode>().unwrap(),
            CollapseMode::First
        );
        assert_eq!("sum".parse::<CollapseMode>().unwrap(), CollapseMode::Sum);
        assert!(matches!(
            "mean".parse::<CollapseMode>(),
            Err(MhcError::InvalidCollapseMode(s)) if s == "mean"
        ));
    }

    #[test]
    fn transformer_block_runs_both_sublayers() {
        let block = MhcTransformerBlock::new(
            config(8),
            |h: &Array2<f32>| h.mapv(|v| v + 1.0),
            |h: &Array2<f32>| h.mapv(|v| v * 2.0),
        )
        .unwrap();
        let x = Array3::<f32>::zeros((2, 4, 8));
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape(), x.shape());
        // zero streams: attention adds ≈ 1 to every stream, then the FFN doubles its input (≈ 1),
        // adding ≈ 2 on top → ≈ 3.
        assert!(y.iter().all(|&v| (v - 3.0).abs() < 0.1), "{y:?}");
    }
}
