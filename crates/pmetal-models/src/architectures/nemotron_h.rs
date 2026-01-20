//! Nemotron-H hybrid Mamba-Transformer architecture.
//!
//! Supports NVIDIA Nemotron-H and Nemotron 3 Nano models which combine:
//! - Mamba-2 SSM layers (selective state space model)
//! - Standard Transformer attention layers
//! - MLP layers with relu2 activation
//! - MoE (Mixture of Experts) layers
//!
//! The hybrid pattern is controlled by `hybrid_override_pattern` where:
//! - `M` = Mamba-2 SSM layer
//! - `*` = Attention layer
//! - `-` = MLP layer (rarely used standalone)
//! - `E` = MoE layer
//!
//! Reference: https://arxiv.org/abs/2504.03624

use std::collections::HashMap;

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters as ModuleParametersTrait, Param},
    nn,
    ops::indexing::IndexOp,
    Array, Dtype,
};
use pmetal_mlx::kernels::{fused_sdpa, rope::apply_rope, AttentionMaskType, FusedAttentionConfig};
use pmetal_mlx::kv_cache::{KVCache, MambaCache, MambaCacheEntry};
use serde::{Deserialize, Serialize};

// ============================================================================
// FP8 Quantized Linear Layer
// ============================================================================

/// FP8 quantized linear layer with scale-based weight dequantization.
///
/// Handles FP8 E4M3 quantized weights with weight scales.
/// NOTE: input_scale is stored but NOT applied during forward - it's for FP8 dynamic quantization.
/// During forward: output = input @ (weight * weight_scale).T
#[derive(Debug)]
pub struct QuantizedLinear {
    /// Quantized weight tensor (FP8 or converted to float).
    pub weight: Param<Array>,
    /// Optional bias.
    pub bias: Option<Param<Array>>,
    /// Weight scale for dequantization.
    pub weight_scale: Option<Array>,
    /// Input scale - stored but NOT applied (for FP8 dynamic quantization only).
    pub input_scale: Option<Array>,
}

impl QuantizedLinear {
    /// Create a new quantized linear layer.
    pub fn new(in_features: i32, out_features: i32) -> Result<Self, Exception> {
        let weight = Array::zeros::<f32>(&[out_features, in_features])?;
        Ok(Self {
            weight: Param::new(weight),
            bias: None,
            weight_scale: None,
            input_scale: None,
        })
    }

    /// Forward pass with FP8 weight dequantization.
    /// NOTE: input_scale is NOT applied - for float inference we only dequantize weights.
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        // Dequantize weight if scale is present
        let weight = if let Some(ref scale) = self.weight_scale {
            // weight_scale is typically a scalar, broadcast multiply
            self.weight.as_ref().multiply(scale)?
        } else {
            self.weight.as_ref().clone()
        };

        // Matrix multiply: [B, L, in] @ [out, in].T -> [B, L, out]
        // NOTE: Do NOT apply input_scale - it's for FP8 dynamic quantization
        let output = x.matmul(&weight.t())?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output.add(bias.as_ref())
        } else {
            Ok(output)
        }
    }
}

// ============================================================================
// Mamba Gated RMS Norm
// ============================================================================

/// Gated RMS norm for Mamba layers.
///
/// Unlike standard RMSNorm, this operates on groups of elements:
/// 1. Optionally multiply input by silu(gate)
/// 2. Reshape to groups: [B, L, D] -> [B, L, D/group_size, group_size]
/// 3. Apply RMS norm within each group
/// 4. Flatten back and apply learned weight
///
/// This matches the mlx-lm MambaRMSNormGated implementation.
#[derive(Debug, Clone)]
pub struct MambaRMSNormGated {
    /// Learnable scale weights [hidden_size].
    pub weight: Array,
    /// Epsilon for numerical stability.
    pub eps: f32,
    /// Size of each normalization group.
    pub group_size: i32,
    /// Total hidden size (for reshaping).
    pub hidden_size: i32,
}

impl MambaRMSNormGated {
    /// Create a new gated RMS norm.
    ///
    /// # Arguments
    /// * `hidden_size` - Total hidden dimension
    /// * `eps` - Epsilon for numerical stability
    /// * `n_groups` - Number of groups to divide hidden_size into
    pub fn new(hidden_size: i32, eps: f32, n_groups: i32) -> Result<Self, Exception> {
        let group_size = hidden_size / n_groups;
        let weight = Array::ones::<f32>(&[hidden_size])?;
        Ok(Self {
            weight,
            eps,
            group_size,
            hidden_size,
        })
    }

    /// Forward pass with optional gating.
    ///
    /// # Arguments
    /// * `x` - Input tensor [B, L, hidden_size]
    /// * `gate` - Optional gate tensor [B, L, hidden_size]
    ///
    /// # Returns
    /// Normalized tensor [B, L, hidden_size]
    pub fn forward(&self, x: &Array, gate: Option<&Array>) -> Result<Array, Exception> {
        // Debug: trace gated norm steps (disabled for performance)
        static GATED_NORM_LOG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let log_count = GATED_NORM_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let log_this = false && log_count < 2;  // Disabled

        if log_this {
            x.eval()?;
            tracing::info!(
                "GATED_NORM[{}] input x: min={:.4}, max={:.4}",
                log_count, x.min(None)?.item::<f32>(), x.max(None)?.item::<f32>()
            );
            if let Some(g) = gate {
                g.eval()?;
                tracing::info!(
                    "GATED_NORM[{}] gate: min={:.4}, max={:.4}",
                    log_count, g.min(None)?.item::<f32>(), g.max(None)?.item::<f32>()
                );
            }
            // Check specific position 3478 (Group 6, pos 406) for tokens 2 and 19
            // x is shape [B, L, hidden=4096], position 3478 = 6*512 + 406
            let seq_len = x.dim(1) as usize;
            if seq_len > 2 {
                let x_t2_pos = x.index((0, 2, 3478)).item::<f32>();
                let x_t19_pos = if seq_len > 19 {
                    x.index((0, 19, 3478)).item::<f32>()
                } else {
                    f32::NAN
                };
                tracing::info!(
                    "GATED_NORM[{}] SSM y at global pos 3478: Token 2={:.4}, Token 19={:.4}",
                    log_count, x_t2_pos, x_t19_pos
                );
                if let Some(g) = gate {
                    let z_t2_pos = g.index((0, 2, 3478)).item::<f32>();
                    let z_t19_pos = if seq_len > 19 {
                        g.index((0, 19, 3478)).item::<f32>()
                    } else {
                        f32::NAN
                    };
                    tracing::info!(
                        "GATED_NORM[{}] Gate z at global pos 3478: Token 2={:.4}, Token 19={:.4}",
                        log_count, z_t2_pos, z_t19_pos
                    );
                }
            }
        }

        // Apply gating if provided: x = x * silu(gate)
        let x = if let Some(g) = gate {
            let gate_activated = nn::silu(g)?;
            if log_this {
                gate_activated.eval()?;
                tracing::info!(
                    "GATED_NORM[{}] silu(gate): min={:.4}, max={:.4}",
                    log_count, gate_activated.min(None)?.item::<f32>(), gate_activated.max(None)?.item::<f32>()
                );
            }
            let gated = x.multiply(&gate_activated)?;
            if log_this {
                gated.eval()?;
                tracing::info!(
                    "GATED_NORM[{}] x * silu(gate): min={:.4}, max={:.4}",
                    log_count, gated.min(None)?.item::<f32>(), gated.max(None)?.item::<f32>()
                );
            }
            gated
        } else {
            x.clone()
        };

        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let num_groups = self.hidden_size / self.group_size;

        // Reshape to groups: [B, L, hidden] -> [B, L, num_groups, group_size]
        if log_this {
            tracing::info!(
                "GATED_NORM[{}] hidden_size={}, group_size={}, num_groups={}",
                log_count, self.hidden_size, self.group_size, num_groups
            );
        }
        let x_grouped = x.reshape(&[batch, seq_len, num_groups, self.group_size])?;

        // Compute RMS norm within each group (axis -1)
        // RMS = sqrt(mean(x^2) + eps)
        let x_sq = x_grouped.square()?;
        let mean_sq = x_sq.mean_axis(-1, true)?;
        let rms = mean_sq.add(&Array::from_f32(self.eps))?.sqrt()?;

        if log_this {
            rms.eval()?;
            tracing::info!(
                "GATED_NORM[{}] rms: min={:.4}, max={:.4}, mean={:.4}",
                log_count, rms.min(None)?.item::<f32>(), rms.max(None)?.item::<f32>(), rms.mean(None)?.item::<f32>()
            );
        }

        let x_normed = x_grouped.divide(&rms)?;

        if log_this {
            x_normed.eval()?;
            tracing::info!(
                "GATED_NORM[{}] x_normed (before weight): min={:.4}, max={:.4}",
                log_count, x_normed.min(None)?.item::<f32>(), x_normed.max(None)?.item::<f32>()
            );
        }

        // Flatten back: [B, L, num_groups, group_size] -> [B, L, hidden]
        let x_flat = x_normed.reshape(&[batch, seq_len, self.hidden_size])?;

        // Apply learned weight
        let result = x_flat.multiply(&self.weight)?;

        if log_this {
            // Per-group analysis: reshape weight to see per-group stats
            let weight_grouped = self.weight.reshape(&[num_groups as i32, self.group_size as i32])?;
            for g in 0..num_groups {
                // Get x_normed for this group - shape is [B, L, num_groups, group_size]
                // Take last seq position for analysis: index [0, -1, g, ..]
                let x_g = x_normed.index((0, -1, g as i32, ..));
                let w_g = weight_grouped.index(g as i32);
                x_g.eval()?;
                w_g.eval()?;
                let result_g = x_g.multiply(&w_g)?;
                result_g.eval()?;
                tracing::info!(
                    "GATED_NORM[{}] Group {}: x_normed [{:.4}, {:.4}], weight [{:.4}, {:.4}], result max={:.4}",
                    log_count, g,
                    x_g.min(None)?.item::<f32>(), x_g.max(None)?.item::<f32>(),
                    w_g.min(None)?.item::<f32>(), w_g.max(None)?.item::<f32>(),
                    result_g.max(None)?.item::<f32>()
                );
            }

            self.weight.eval()?;
            result.eval()?;
            tracing::info!(
                "GATED_NORM[{}] weight: min={:.4}, max={:.4}, mean={:.4}",
                log_count, self.weight.min(None)?.item::<f32>(), self.weight.max(None)?.item::<f32>(),
                self.weight.mean(None)?.item::<f32>()
            );
            // Print first 20 weight values
            let first_20: Vec<f32> = (0..20).map(|i| self.weight.index(i as i32).item::<f32>()).collect();
            tracing::info!(
                "GATED_NORM[{}] weight first 20: {:?}",
                log_count, first_20
            );
            tracing::info!(
                "GATED_NORM[{}] final output: min={:.4}, max={:.4}",
                log_count, result.min(None)?.item::<f32>(), result.max(None)?.item::<f32>()
            );

            // Find which token position produces the max - result is [B, L, hidden]
            for t in 0..seq_len {
                let result_t = result.index((0, t as i32, ..));
                result_t.eval()?;
                let max_t = result_t.max(None)?.item::<f32>();
                if max_t > 5.0 {
                    tracing::info!(
                        "GATED_NORM[{}] Token {} has large max={:.4}",
                        log_count, t, max_t
                    );
                    // Detailed per-group analysis for this token
                    // Focus on Group 6 where the issue occurs
                    let g = 6usize;
                    let x_g = x_normed.index((0, t as i32, g as i32, ..));
                    let w_g = weight_grouped.index(g as i32);
                    x_g.eval()?;
                    w_g.eval()?;
                    let result_g = x_g.multiply(&w_g)?;
                    result_g.eval()?;

                    // Find max weight position and check x_normed there
                    // Reference shows max weight is at position 382 with value 5.0
                    // and y_normed at 382 is ~0.0003
                    let x_at_382 = x_g.index(382).item::<f32>();
                    let w_at_382 = w_g.index(382).item::<f32>();
                    tracing::info!(
                        "GATED_NORM[{}] Token {} Group 6: x_normed [{:.4}, {:.4}], weight max={:.4}, result max={:.4}",
                        log_count, t,
                        x_g.min(None)?.item::<f32>(), x_g.max(None)?.item::<f32>(),
                        w_g.max(None)?.item::<f32>(),
                        result_g.max(None)?.item::<f32>()
                    );
                    tracing::info!(
                        "GATED_NORM[{}] Token {} Group 6 at pos 382: x_normed={:.6}, weight={:.4}, product={:.6}",
                        log_count, t, x_at_382, w_at_382, x_at_382 * w_at_382
                    );

                    // Find where the max result comes from by scanning all positions
                    let res_max = result_g.max(None)?.item::<f32>();
                    for pos in 0..self.group_size {
                        let x_val = x_g.index(pos as i32).item::<f32>();
                        let w_val = w_g.index(pos as i32).item::<f32>();
                        let prod = x_val * w_val;
                        if prod > 3.0 {  // Find positions producing large products
                            tracing::info!(
                                "GATED_NORM[{}] Token {} Group 6 pos {}: x={:.4}, w={:.4}, prod={:.4}",
                                log_count, t, pos, x_val, w_val, prod
                            );
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}

// ============================================================================
// SSM Computation Functions
// ============================================================================

/// Compute dt = softplus(dt + dt_bias) clipped to limits.
fn compute_dt(
    dt: &Array,
    dt_bias: &Array,
    time_step_min: f32,
    time_step_max: f32,
) -> Result<Array, Exception> {
    let dt_biased = dt.add(dt_bias)?;
    let dt_soft = nn::softplus(&dt_biased)?;
    mlx_rs::ops::clip(
        &dt_soft,
        (&Array::from_f32(time_step_min), &Array::from_f32(time_step_max)),
    )
}

/// Segmented cumulative sum for SSM decay computation.
/// Computes cumsum along axis for attention-like SSM formulation.
fn segsum(x: &Array) -> Result<Array, Exception> {
    let l = x.shape()[x.ndim() - 1];

    // Repeat x along new axis: [B, H, L] -> [B, H, L, L]
    let x_expanded = mlx_rs::ops::expand_dims(x, -1)?;
    let x_repeated = mlx_rs::ops::tile(&x_expanded, &[1, 1, 1, l as i32])?;

    // Create lower triangular mask (shifted by -1)
    let x_tril = mlx_rs::ops::tril(&x_repeated, -1)?;

    // Cumsum along axis -2
    x_tril.cumsum(-2, None, None)
}

/// Optimized SSM update for single-token decode (seq_len=1).
///
/// This is much faster than the full ssm_attention for incremental generation
/// because it avoids creating the full [B, H, L, L] attention matrix and segsum.
///
/// Based on Python's ssm_update_kernel but using MLX ops instead of a custom Metal kernel.
///
/// # Arguments
/// * `x` - Input tensor [B, 1, H, D]
/// * `a_log` - Log of state transition matrix [H]
/// * `b` - Input mixing [B, 1, G, N]
/// * `c` - Output mixing [B, 1, G, N]
/// * `d` - Skip connection weights [H]
/// * `dt` - Time deltas [B, 1, H]
/// * `dt_bias` - Time delta bias [H]
/// * `state` - Previous SSM state [B, H, D, N]
/// * `time_step_limit` - (min, max) for dt clipping
///
/// # Returns
/// (output [B, 1, H, D], new_state [B, H, D, N])
pub fn ssm_update_single(
    x: &Array,           // [B, 1, H, D]
    a_log: &Array,       // [H]
    b: &Array,           // [B, 1, G, N]
    c: &Array,           // [B, 1, G, N]
    d: &Array,           // [H]
    dt: &Array,          // [B, 1, H]
    dt_bias: &Array,     // [H]
    state: &Array,       // [B, H, D, N]
    time_step_limit: (f32, f32),
) -> Result<(Array, Array), Exception> {
    let shape = x.shape();
    let batch = shape[0];
    let num_heads = shape[2];
    let head_dim = shape[3];

    let b_shape = b.shape();
    let n_groups = b_shape[2];
    let state_dim = b_shape[3];

    let repeats = num_heads / n_groups;

    // Compute dt with bias and clipping: dt = clip(softplus(dt + dt_bias), min, max)
    // dt: [B, 1, H] -> squeeze to [B, H]
    let dt_squeezed = dt.squeeze_axes(&[1])?;
    let dt_full = compute_dt(&dt_squeezed, dt_bias, time_step_limit.0, time_step_limit.1)?;

    // Compute A = -exp(A_log): [H]
    let a = mlx_rs::ops::negative(&mlx_rs::ops::exp(a_log)?)?;
    let a = a.as_dtype(dt_full.dtype())?;

    // dA = exp(A * dt): [B, H] - decay factor
    let dt_a = dt_full.multiply(&a.reshape(&[1, num_heads])?)?;
    let d_a = mlx_rs::ops::exp(&dt_a)?;

    // x squeezed: [B, 1, H, D] -> [B, H, D]
    let x_squeezed = x.squeeze_axes(&[1])?;

    // dt expanded for multiplication: [B, H] -> [B, H, 1]
    let dt_expanded = mlx_rs::ops::expand_dims(&dt_full, -1)?;

    // dtx = x * dt: [B, H, D]
    let dtx = x_squeezed.multiply(&dt_expanded)?;

    // B squeezed: [B, 1, G, N] -> [B, G, N]
    let b_squeezed = b.squeeze_axes(&[1])?;

    // Repeat B to all heads: [B, G, N] -> [B, H, N]
    let b_expanded = mlx_rs::ops::expand_dims(&b_squeezed, 2)?;  // [B, G, 1, N]
    let b_tiled = mlx_rs::ops::tile(&b_expanded, &[1, 1, repeats, 1])?;  // [B, G, repeats, N]
    let b_heads = b_tiled.reshape(&[batch, num_heads, state_dim])?;  // [B, H, N]

    // dB_by_x = dtx * B: [B, H, D, 1] @ [B, H, 1, N] -> [B, H, D, N]
    // Using outer product: dtx[:,:,:,None] * B[:,:,None,:]
    let dtx_expanded = mlx_rs::ops::expand_dims(&dtx, -1)?;  // [B, H, D, 1]
    let b_expanded2 = mlx_rs::ops::expand_dims(&b_heads, 2)?;  // [B, H, 1, N]
    let db_by_x = dtx_expanded.multiply(&b_expanded2)?;  // [B, H, D, N]

    // dA expanded for state: [B, H] -> [B, H, 1, 1]
    let d_a_expanded = mlx_rs::ops::expand_dims(&d_a, -1)?;
    let d_a_expanded = mlx_rs::ops::expand_dims(&d_a_expanded, -1)?;

    // New state = dA * old_state + dB_by_x: [B, H, D, N]
    let new_state = state.multiply(&d_a_expanded)?.add(&db_by_x)?;

    // C squeezed: [B, 1, G, N] -> [B, G, N]
    let c_squeezed = c.squeeze_axes(&[1])?;

    // Repeat C to all heads: [B, G, N] -> [B, H, N]
    let c_expanded = mlx_rs::ops::expand_dims(&c_squeezed, 2)?;
    let c_tiled = mlx_rs::ops::tile(&c_expanded, &[1, 1, repeats, 1])?;
    let c_heads = c_tiled.reshape(&[batch, num_heads, state_dim])?;

    // Output = state @ C: [B, H, D, N] @ [B, H, N, 1] -> [B, H, D, 1] -> [B, H, D]
    // Or using einsum-like: sum over N: state * C
    let c_expanded2 = mlx_rs::ops::expand_dims(&c_heads, 2)?;  // [B, H, 1, N]
    let state_c = new_state.multiply(&c_expanded2)?;  // [B, H, D, N]
    let y = state_c.sum_axis(-1, false)?;  // [B, H, D]

    // Add skip connection: y + x * D
    let d_expanded = d.reshape(&[1, num_heads, 1])?;
    let skip = x_squeezed.multiply(&d_expanded)?;
    let y = y.add(&skip)?;

    // Expand output to [B, 1, H, D]
    let y = mlx_rs::ops::expand_dims(&y, 1)?;

    Ok((y, new_state))
}

/// SSM attention computation with optional state for incremental generation.
///
/// Implements the SSD-SSM forward pass based on mlx-lm's ssm_attn implementation.
/// Returns both the output and the new state for caching.
///
/// # Arguments
/// * `x` - Input tensor [B, L, H, D]
/// * `a_log` - Log of state transition matrix [H]
/// * `b` - Input mixing [B, L, G, N]
/// * `c` - Output mixing [B, L, G, N]
/// * `d` - Skip connection weights [H]
/// * `dt` - Time deltas [B, L, H]
/// * `dt_bias` - Time delta bias [H]
/// * `state` - Optional previous SSM state [B, H, D, N]
/// * `time_step_limit` - (min, max) for dt clipping
///
/// # Returns
/// (output [B, L, H, D], new_state [B, H, D, N])
pub fn ssm_attention(
    x: &Array,           // [B, L, H, D] - input
    a_log: &Array,       // [H] - log state transition
    b: &Array,           // [B, L, G, N] - input mixing
    c: &Array,           // [B, L, G, N] - output mixing
    d: &Array,           // [H] - skip connection
    dt: &Array,          // [B, L, H] - time deltas
    dt_bias: &Array,     // [H] - time delta bias
    state: Option<&Array>, // Optional previous state [B, H, D, N]
    time_step_limit: (f32, f32),
) -> Result<(Array, Array), Exception> {
    let shape = x.shape();
    let batch = shape[0];
    let seq_len = shape[1];
    let num_heads = shape[2];
    let head_dim = shape[3];

    let b_shape = b.shape();
    let n_groups = b_shape[2];
    let state_dim = b_shape[3];

    let repeats = num_heads / n_groups;

    // Compute dt with bias and clipping: dt = clip(softplus(dt + dt_bias), min, max)
    let dt_full = compute_dt(dt, dt_bias, time_step_limit.0, time_step_limit.1)?;

    // Compute A = -exp(A_log) and cast to dt dtype
    let a = mlx_rs::ops::negative(&mlx_rs::ops::exp(a_log)?)?;
    let a = a.as_dtype(dt_full.dtype())?;

    // Debug: trace SSM intermediate values (disabled for performance)
    static SSM_LOG_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let log_count = SSM_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let detailed_log = false && log_count < 2;  // Disabled - set to true to debug

    // Compute dtA = dt * A: [B, L, H]
    let dt_a = dt_full.multiply(&a.reshape(&[1, 1, num_heads])?)?;

    if detailed_log {
        x.eval()?;
        dt_full.eval()?;
        a.eval()?;
        dt_a.eval()?;
        tracing::info!(
            "SSM[{}] INPUTS: x shape={:?} min={:.4} max={:.4}",
            log_count, x.shape(), x.min(None)?.item::<f32>(), x.max(None)?.item::<f32>()
        );
        tracing::info!(
            "SSM[{}] dt shape={:?} min={:.6} max={:.6}",
            log_count, dt_full.shape(), dt_full.min(None)?.item::<f32>(), dt_full.max(None)?.item::<f32>()
        );
        tracing::info!(
            "SSM[{}] A shape={:?} min={:.4} max={:.4}",
            log_count, a.shape(), a.min(None)?.item::<f32>(), a.max(None)?.item::<f32>()
        );
        tracing::info!(
            "SSM[{}] dtA shape={:?} min={:.4} max={:.4}",
            log_count, dt_a.shape(), dt_a.min(None)?.item::<f32>(), dt_a.max(None)?.item::<f32>()
        );
    }

    // Compute dtx = dt * x: [B, L, H, D]
    let dt_expanded = dt_full.reshape(&[batch, seq_len, num_heads, 1])?;
    let dtx = x.multiply(&dt_expanded)?;

    // B: [B, L, G, N] -> [B, G, N, L]
    let b_t = b.transpose_axes(&[0, 2, 3, 1])?;

    // CB = C.swapaxes(1,2) @ B_t: [B, G, L, N] @ [B, G, N, L] = [B, G, L, L]
    let c_t = c.transpose_axes(&[0, 2, 1, 3])?;
    let cb = c_t.matmul(&b_t)?;

    // Repeat CB to all heads: [B, G, L, L] -> [B, H, L, L]
    let cb_expanded = mlx_rs::ops::expand_dims(&cb, 2)?;
    let cb_tiled = mlx_rs::ops::tile(&cb_expanded, &[1, 1, repeats, 1, 1])?;
    let cb_heads = cb_tiled.reshape(&[batch, num_heads, seq_len, seq_len])?;

    // Compute decay = exp(segsum(dtA.swapaxes(1,2))): [B, H, L, L]
    let dt_a_t = dt_a.transpose_axes(&[0, 2, 1])?;
    let segsum_result = segsum(&dt_a_t)?;
    let decay = mlx_rs::ops::exp(&segsum_result)?;

    if detailed_log {
        cb.eval()?;
        segsum_result.eval()?;
        decay.eval()?;
        tracing::info!(
            "SSM[{}] CB shape={:?} min={:.4} max={:.4}",
            log_count, cb.shape(), cb.min(None)?.item::<f32>(), cb.max(None)?.item::<f32>()
        );
        tracing::info!(
            "SSM[{}] segsum shape={:?} min={:.4} max={:.4}",
            log_count, segsum_result.shape(), segsum_result.min(None)?.item::<f32>(), segsum_result.max(None)?.item::<f32>()
        );
        tracing::info!(
            "SSM[{}] decay shape={:?} min={:.6} max={:.6}",
            log_count, decay.shape(), decay.min(None)?.item::<f32>(), decay.max(None)?.item::<f32>()
        );
    }

    // Surrogate attention = tril(CB * decay)
    let attn_weights = cb_heads.multiply(&decay)?;
    let attn_weights = mlx_rs::ops::tril(&attn_weights, 0)?;

    // y = attn @ dtx.swapaxes(1,2): [B, H, L, L] @ [B, H, L, D] = [B, H, L, D]
    let dtx_t = dtx.transpose_axes(&[0, 2, 1, 3])?;
    let mut y = attn_weights.matmul(&dtx_t)?;

    if detailed_log {
        attn_weights.eval()?;
        dtx.eval()?;
        y.eval()?;
        tracing::info!(
            "SSM[{}] attn_weights shape={:?} min={:.4} max={:.4}",
            log_count, attn_weights.shape(), attn_weights.min(None)?.item::<f32>(), attn_weights.max(None)?.item::<f32>()
        );
        tracing::info!(
            "SSM[{}] dtx shape={:?} min={:.4} max={:.4}",
            log_count, dtx.shape(), dtx.min(None)?.item::<f32>(), dtx.max(None)?.item::<f32>()
        );
        tracing::info!(
            "SSM[{}] y (after attn@dtx) shape={:?} min={:.4} max={:.4}",
            log_count, y.shape(), y.min(None)?.item::<f32>(), y.max(None)?.item::<f32>()
        );

        // Position 3478 = head 54, pos 26 within head
        // y is [B, H, L, D], head 54 is position 54 in dim 1
        // Check y at head 54, token 2, dim 26
        if seq_len > 2 {
            let y_val = y.index((0, 54, 2, 26)).item::<f32>();
            let dtx_val = dtx.index((0, 2, 54, 26)).item::<f32>();  // dtx is [B, L, H, D]
            tracing::info!(
                "SSM[{}] y[head=54, token=2, dim=26] = {:.4}, dtx[token=2, head=54, dim=26] = {:.4}",
                log_count, y_val, dtx_val
            );

            // Check the attn_weights row for head 54, token 2
            // attn_weights is [B, H, L, L], we want [0, 54, 2, :]
            let attn_row = attn_weights.index((0, 54, 2, ..));
            attn_row.eval()?;
            tracing::info!(
                "SSM[{}] attn_weights[head=54, token=2, :] min={:.4}, max={:.4}, sum={:.4}",
                log_count,
                attn_row.min(None)?.item::<f32>(),
                attn_row.max(None)?.item::<f32>(),
                attn_row.sum(None)?.item::<f32>()
            );
        }
    }

    // Compute new state for caching
    // decay_last: [B, H, 1, L] -> [B, L, H, 1]
    let decay_last = decay.index((.., .., (seq_len - 1)..seq_len, ..));
    let decay_last = decay_last.transpose_axes(&[0, 3, 1, 2])?;

    // B_expanded: [B, G, N, L] -> repeat to [B, H, N, L] -> [B, H, L, N]
    let b_expanded = mlx_rs::ops::expand_dims(&b_t, 2)?;
    let b_tiled = mlx_rs::ops::tile(&b_expanded, &[1, 1, repeats, 1, 1])?;
    let b_heads = b_tiled.reshape(&[batch, num_heads, state_dim, seq_len])?;
    let b_heads = b_heads.transpose_axes(&[0, 1, 3, 2])?; // [B, H, L, N]

    // dtxdecay = dtx * decay_last: [B, L, H, D]
    let dtxdecay = dtx.multiply(&decay_last)?;

    // dtxdecay: [B, L, H, D] -> [B, H, D, L]
    let dtxdecay_t = dtxdecay.transpose_axes(&[0, 2, 3, 1])?;

    // next_state = dtxdecay @ B_heads: [B, H, D, L] @ [B, H, L, N] = [B, H, D, N]
    let mut next_state = dtxdecay_t.matmul(&b_heads)?;

    // Debug: trace state info when present
    static STATE_PRESENT_LOG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    // If we have previous state, incorporate it
    // TEMP DEBUG: test with state contribution disabled
    let disable_state_contribution = std::env::var("DISABLE_STATE").is_ok();
    if let Some(prev_state) = state {
        if disable_state_contribution {
            tracing::warn!("SSM state contribution DISABLED for debugging");
        }
        // Separate counter for when state IS present (second+ pass)
        let state_present_log = STATE_PRESENT_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let log_state_present = state_present_log < 10;

        if log_state_present {
            prev_state.eval()?;
            tracing::info!(
                "SSM STATE PRESENT [{}]: shape={:?}, min={:.6}, max={:.6}, mean={:.6}, seq_len={}",
                state_present_log,
                prev_state.shape(),
                prev_state.min(None)?.item::<f32>(),
                prev_state.max(None)?.item::<f32>(),
                prev_state.mean(None)?.item::<f32>(),
                seq_len,
            );

            // Log y before state contribution
            y.eval()?;
            tracing::info!(
                "SSM y BEFORE state[{}]: min={:.6}, max={:.6}",
                state_present_log,
                y.min(None)?.item::<f32>(),
                y.max(None)?.item::<f32>(),
            );
        }

        // exp_dtA_cumsum = exp(cumsum(dtA, axis=-2)): [B, L, H]
        let exp_dta_cumsum = mlx_rs::ops::exp(&dt_a.cumsum(-2, None, None)?)?;

        if log_state_present {
            exp_dta_cumsum.eval()?;
            tracing::info!(
                "SSM STATE[{}]: exp_dta_cumsum shape={:?}, min={:.6}, max={:.6}",
                state_present_log,
                exp_dta_cumsum.shape(),
                exp_dta_cumsum.min(None)?.item::<f32>(),
                exp_dta_cumsum.max(None)?.item::<f32>(),
            );
        }

        // exp_dta_last: [B, 1, H] -> [B, H, 1, 1]
        let exp_dta_last = exp_dta_cumsum.index((.., (seq_len - 1)..seq_len, ..));
        let exp_dta_last = exp_dta_last.transpose_axes(&[0, 2, 1])?;
        let exp_dta_last = mlx_rs::ops::expand_dims(&exp_dta_last, -1)?;

        // next_state += exp_dta_last * prev_state
        let state_decay = prev_state.multiply(&exp_dta_last)?;
        next_state = next_state.add(&state_decay)?;

        // Compute contribution from previous state to y
        // prev_state: [B, H, D, N]
        // Reshape for group-wise processing:
        // state_grouped: [B, G, repeats, D, N] -> [B, 1, G, repeats, D, N]
        let state_grouped = prev_state.reshape(&[batch, n_groups, repeats, head_dim, state_dim])?;
        let state_grouped = mlx_rs::ops::expand_dims(&state_grouped, 1)?;

        // C_grouped: [B, L, G, N] -> [B, L, G, 1, N, 1]
        let c_grouped = c.reshape(&[batch, seq_len, n_groups, 1, state_dim, 1])?;

        // y_prev = state_grouped @ C_grouped: [B, 1, G, repeats, D, N] @ [B, L, G, 1, N, 1]
        // This is a batched matmul that broadcasts
        let y_prev_raw = state_grouped.matmul(&c_grouped)?; // [B, L, G, repeats, D, 1]

        let y_prev = y_prev_raw.squeeze_axes(&[-1])?.reshape(&[batch, seq_len, num_heads, head_dim])?;

        if log_state_present {
            y_prev.eval()?;
            c.eval()?;
            tracing::info!(
                "SSM STATE[{}]: y_prev (state@C) shape={:?}, min={:.6}, max={:.6}",
                state_present_log,
                y_prev.shape(),
                y_prev.min(None)?.item::<f32>(),
                y_prev.max(None)?.item::<f32>(),
            );
            tracing::info!(
                "SSM STATE[{}]: C shape={:?}, min={:.6}, max={:.6}",
                state_present_log,
                c.shape(),
                c.min(None)?.item::<f32>(),
                c.max(None)?.item::<f32>(),
            );
        }

        // exp_dta_cumsum: [B, L, H] -> [B, L, H, 1]
        let exp_dta_expanded = mlx_rs::ops::expand_dims(&exp_dta_cumsum, -1)?;

        // y_prev contribution: y += exp_dta_cumsum * y_prev
        if !disable_state_contribution {
            let y_prev_t = y_prev.transpose_axes(&[0, 2, 1, 3])?; // [B, H, L, D]
            let exp_dta_t = exp_dta_expanded.transpose_axes(&[0, 2, 1, 3])?; // [B, H, L, 1]
            let y_contribution = y_prev_t.multiply(&exp_dta_t)?;

            if log_state_present {
                y_contribution.eval()?;
                tracing::info!(
                    "SSM STATE[{}]: y_contribution (exp_dta*y_prev) min={:.6}, max={:.6}",
                    state_present_log,
                    y_contribution.min(None)?.item::<f32>(),
                    y_contribution.max(None)?.item::<f32>(),
                );
            }

            y = y.add(&y_contribution)?;
        }

        if log_state_present {
            y.eval()?;
            tracing::info!(
                "SSM y AFTER state[{}]: min={:.6}, max={:.6}",
                state_present_log,
                y.min(None)?.item::<f32>(),
                y.max(None)?.item::<f32>(),
            );
        }
    }

    // y = y.swapaxes(1,2): [B, L, H, D]
    let y = y.transpose_axes(&[0, 2, 1, 3])?;

    // Add skip connection: y += x * D
    let d_expanded = d.reshape(&[1, 1, num_heads, 1])?;

    // Debug: trace D values
    if detailed_log {
        d.eval()?;
        let skip = x.multiply(&d_expanded)?;
        skip.eval()?;
        y.eval()?;
        tracing::info!(
            "SSM[{}] D param: min={:.4}, max={:.4}",
            log_count, d.min(None)?.item::<f32>(), d.max(None)?.item::<f32>()
        );
        tracing::info!(
            "SSM[{}] x*D (skip): min={:.4}, max={:.4}",
            log_count, skip.min(None)?.item::<f32>(), skip.max(None)?.item::<f32>()
        );
        tracing::info!(
            "SSM[{}] y (before skip): min={:.4}, max={:.4}",
            log_count, y.min(None)?.item::<f32>(), y.max(None)?.item::<f32>()
        );

        // Check position 3478 = (head 54, dim 22) before skip
        if seq_len > 2 {
            let y_before = y.index((0, 2, 54, 22)).item::<f32>();  // y is [B, L, H, D]
            let x_val = x.index((0, 2, 54, 22)).item::<f32>();
            let d_val = d.index(54).item::<f32>();
            tracing::info!(
                "SSM[{}] POSITION 3478 (token=2, head=54, dim=22): y_before_skip={:.4}, x={:.4}, D={:.4}, x*D={:.4}",
                log_count, y_before, x_val, d_val, x_val * d_val
            );
        }
    }

    let y = y.add(&x.multiply(&d_expanded)?)?;

    if detailed_log && seq_len > 2 {
        y.eval()?;
        let y_after = y.index((0, 2, 54, 26)).item::<f32>();
        tracing::info!(
            "SSM[{}] POSITION (token=2, head=54, dim=26): y_after_skip={:.4}",
            log_count, y_after
        );

        // Position 3478 = head 54, dim 22 (3478 / 64 = 54, 3478 % 64 = 22)
        let h = 54;
        let d = 22;
        let y_val_check = y.index((0, 2, h, d)).item::<f32>();
        let x_val_check = x.index((0, 2, h, d)).item::<f32>();
        let d_val_check = d_expanded.index((0, 0, h, 0)).item::<f32>();
        tracing::info!(
            "SSM[{}] Position 3478 = (head=54, dim=22): x={:.4}, D={:.4}, x*D={:.4}, y_final={:.4}",
            log_count, x_val_check, d_val_check, x_val_check * d_val_check, y_val_check
        );
    }

    Ok((y, next_state))
}

// ============================================================================
// MoE (Mixture of Experts) Components
// ============================================================================

/// Single expert MLP with optional FP8 quantization support.
#[derive(Debug, ModuleParameters)]
pub struct Expert {
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
    /// FP8 scale factors (not parameters, just data)
    pub up_proj_weight_scale: Option<Array>,
    pub up_proj_input_scale: Option<Array>,
    pub down_proj_weight_scale: Option<Array>,
    pub down_proj_input_scale: Option<Array>,
}

impl Expert {
    /// Create a new expert.
    pub fn new(hidden_size: i32, intermediate_size: i32, bias: bool) -> Result<Self, Exception> {
        let up_proj = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(bias)
            .build()?;
        let down_proj = nn::LinearBuilder::new(intermediate_size, hidden_size)
            .bias(bias)
            .build()?;
        Ok(Self {
            up_proj,
            down_proj,
            up_proj_weight_scale: None,
            up_proj_input_scale: None,
            down_proj_weight_scale: None,
            down_proj_input_scale: None,
        })
    }

    /// Forward pass with relu2 activation and FP8 weight dequantization.
    /// Note: input_scale is NOT applied - it's for dynamic FP8 quantization which we don't use.
    /// We only dequantize weights with weight_scale for float inference.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // Up projection with weight dequantization
        // NOTE: Do NOT apply input_scale - that's for FP8 dynamic quantization
        let up = if let Some(ref scale) = self.up_proj_weight_scale {
            let weight = self.up_proj.weight.as_ref().multiply(scale)?;
            x.matmul(&weight.t())?
        } else {
            Module::forward(&mut self.up_proj, x)?
        };

        // Debug: trace up projection before ReLU
        static EXPERT_LOG_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let log_count = EXPERT_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if log_count < 3 {
            // Trace dequantized weight
            if let Some(ref scale) = self.up_proj_weight_scale {
                let raw_weight = self.up_proj.weight.as_ref();
                raw_weight.eval()?;
                scale.eval()?;
                let dequant_weight = raw_weight.multiply(scale)?;
                dequant_weight.eval()?;
                tracing::debug!(
                    "Expert weight - raw dtype={:?}, scale={:.6}, dequant min={:.4}, max={:.4}",
                    raw_weight.dtype(),
                    scale.item::<f32>(),
                    dequant_weight.min(None)?.item::<f32>(),
                    dequant_weight.max(None)?.item::<f32>()
                );
            }
            // Trace input
            x.eval()?;
            tracing::debug!(
                "Expert input - min={:.4}, max={:.4}, shape={:?}",
                x.min(None)?.item::<f32>(),
                x.max(None)?.item::<f32>(),
                x.shape()
            );
            up.eval()?;
            let positive_count = up.gt(&Array::from_f32(0.0))?.sum(None)?.item::<i32>();
            let total = up.size() as i32;
            tracing::debug!(
                "Expert up proj (before ReLU) - min={:.4}, max={:.4}, positive={}/{}",
                up.min(None)?.item::<f32>(),
                up.max(None)?.item::<f32>(),
                positive_count,
                total
            );
        }

        // ReLU² activation
        let activated = nn::relu(&up)?.square()?;

        // Down projection with weight dequantization
        // NOTE: Do NOT apply input_scale - that's for FP8 dynamic quantization
        if let Some(ref scale) = self.down_proj_weight_scale {
            let weight = self.down_proj.weight.as_ref().multiply(scale)?;
            activated.matmul(&weight.t())
        } else {
            Module::forward(&mut self.down_proj, &activated)
        }
    }
}

/// MoE router for expert selection using sigmoid-based routing.
///
/// Implements the group_expert_select algorithm from Nemotron-H:
/// 1. Compute sigmoid scores from gate logits
/// 2. Add e_score_correction_bias for selection (not final scores)
/// 3. If n_group > 1, select top groups first
/// 4. Select top-k experts within selected groups
/// 5. Optionally normalize scores (norm_topk_prob)
/// 6. Apply routed_scaling_factor
#[derive(Debug, ModuleParameters)]
pub struct MoERouter {
    #[param]
    pub gate: nn::Linear,
    /// Score correction bias for expert selection (shape: [num_experts])
    pub e_score_correction_bias: Array,
    pub num_experts: i32,
    pub top_k: i32,
    /// Number of groups for hierarchical expert selection.
    pub n_group: i32,
    /// Number of groups to select from (topk_group).
    pub topk_group: i32,
    /// Whether to normalize top-k probabilities.
    pub norm_topk_prob: bool,
    /// Scaling factor applied to routing weights after normalization.
    pub routed_scaling_factor: f32,
}

impl MoERouter {
    /// Create a new router.
    pub fn new(
        hidden_size: i32,
        num_experts: i32,
        top_k: i32,
        n_group: i32,
        topk_group: i32,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
    ) -> Result<Self, Exception> {
        let gate = nn::LinearBuilder::new(hidden_size, num_experts)
            .bias(false)
            .build()?;
        // Initialize e_score_correction_bias to zeros
        let e_score_correction_bias = Array::zeros::<f32>(&[num_experts])?;
        Ok(Self {
            gate,
            e_score_correction_bias,
            num_experts,
            top_k,
            n_group,
            topk_group,
            norm_topk_prob,
            routed_scaling_factor,
        })
    }

    /// Compute routing weights and expert indices using sigmoid-based selection.
    /// Returns (weights, indices) where:
    /// - weights: [B*L, top_k] - sigmoid scores for top-k experts (normalized if norm_topk_prob, scaled)
    /// - indices: [B*L, top_k] - indices of selected experts
    pub fn forward(&mut self, x: &Array) -> Result<(Array, Array), Exception> {
        // Compute logits: [B*L, num_experts]
        let logits = Module::forward(&mut self.gate, x)?;

        // Debug: Check e_score_correction_bias is loaded
        static LOGGED_BIAS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED_BIAS.swap(true, std::sync::atomic::Ordering::Relaxed) {
            self.e_score_correction_bias.eval()?;
            tracing::info!(
                "MoE e_score_correction_bias: shape={:?}, min={:.4}, max={:.4}, sum={:.4}",
                self.e_score_correction_bias.shape(),
                self.e_score_correction_bias.min(None)?.item::<f32>(),
                self.e_score_correction_bias.max(None)?.item::<f32>(),
                self.e_score_correction_bias.sum(None)?.item::<f32>()
            );
        }

        // Apply sigmoid (not softmax) for scoring
        let logits_f32 = logits.as_dtype(Dtype::Float32)?;
        let orig_scores = mlx_rs::ops::sigmoid(&logits_f32)?;

        // Add bias for selection (but use original scores for final weights)
        let scores_for_selection = orig_scores.add(&self.e_score_correction_bias)?;

        // Group-based selection when n_group > 1
        let scores_for_selection = if self.n_group > 1 {
            let experts_per_group = self.num_experts / self.n_group;
            // Reshape to groups: [B*L, n_group, experts_per_group]
            let grouped = scores_for_selection.reshape(&[-1, self.n_group, experts_per_group])?;

            // Get top-2 scores per group and sum them to get group scores
            let top2_scores = mlx_rs::ops::indexing::topk_axis(&grouped, 2, -1)?;
            let group_scores = top2_scores.sum_axis(-1, true)?;

            // Zero out non-selected groups (keep top topk_group groups)
            let k = self.n_group - self.topk_group;
            let group_idx = mlx_rs::ops::argpartition_axis(&group_scores, k - 1, -2)?;
            let group_idx = group_idx.index((.., ..k, ..));

            // Zero out bottom groups using put_along_axis
            let zeros = Array::from_f32(0.0);
            let group_idx_sg = mlx_rs::stop_gradient(&group_idx)?;
            let grouped = mlx_rs::ops::indexing::put_along_axis(
                &grouped,
                &group_idx_sg,
                &zeros,
                Some(-2),
            )?;

            // Flatten back: [B*L, num_experts]
            grouped.reshape(&[-1, self.num_experts])?
        } else {
            scores_for_selection
        };

        // Get top-k indices (negate for argpartition which returns smallest)
        let neg_scores = scores_for_selection.negative()?;
        let k = self.top_k - 1;
        let inds = mlx_rs::ops::argpartition_axis(&neg_scores, k, -1)?;
        let inds = inds.index((.., ..self.top_k));

        // Get original scores for selected experts (not bias-corrected)
        let scores = orig_scores.take_along_axis(&inds, -1)?;

        // Optionally normalize scores
        let scores = if self.top_k > 1 && self.norm_topk_prob {
            let denom = scores.sum_axis(-1, true)?.add(&Array::from_f32(1e-20))?;
            scores.divide(&denom)?
        } else {
            scores
        };

        // Apply routed_scaling_factor
        let weights = scores.multiply(&Array::from_f32(self.routed_scaling_factor))?;

        Ok((weights, inds))
    }
}

/// Mixture of Experts layer.
#[derive(Debug, ModuleParameters)]
pub struct MoELayer {
    #[param]
    pub router: MoERouter,
    #[param]
    pub experts: Vec<Expert>,
    #[param]
    pub shared_expert: Option<Expert>,
    pub num_experts: i32,
    pub top_k: i32,
}

impl MoELayer {
    /// Create a new MoE layer.
    pub fn new(
        hidden_size: i32,
        intermediate_size: i32,
        shared_intermediate_size: Option<i32>,
        num_experts: i32,
        top_k: i32,
        n_group: i32,
        topk_group: i32,
        norm_topk_prob: bool,
        use_shared_expert: bool,
        routed_scaling_factor: f32,
        bias: bool,
    ) -> Result<Self, Exception> {
        // Create router with sigmoid-based scoring and group selection
        let router = MoERouter::new(
            hidden_size,
            num_experts,
            top_k,
            n_group,
            topk_group,
            norm_topk_prob,
            routed_scaling_factor,
        )?;

        let mut experts = Vec::with_capacity(num_experts as usize);
        for _ in 0..num_experts {
            experts.push(Expert::new(hidden_size, intermediate_size, bias)?);
        }

        let shared_expert = if use_shared_expert {
            // Use shared_intermediate_size if provided, otherwise fall back to regular intermediate_size
            let shared_size = shared_intermediate_size.unwrap_or(intermediate_size);
            Some(Expert::new(hidden_size, shared_size, bias)?)
        } else {
            None
        };

        Ok(Self {
            router,
            experts,
            shared_expert,
            num_experts,
            top_k,
        })
    }

    /// Forward pass with expert routing.
    ///
    /// Optimized implementation:
    /// - For decode (1 token): Sync once to get routing, call only selected experts (4 vs 128)
    /// - For prefill (many tokens): Use lazy evaluation with masking
    /// - Removes per-iteration eval() calls
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];
        let num_tokens = batch * seq_len;

        // Flatten batch and sequence for routing: [B, L, H] -> [B*L, H]
        let x_flat = x.reshape(&[num_tokens, hidden_size])?;

        // Get routing weights and indices: weights [B*L, top_k], indices [B*L, top_k]
        let (weights, indices) = self.router.forward(&x_flat)?;

        // Initialize output with zeros
        let mut output = mlx_rs::ops::zeros_like(&x_flat)?;

        // For decode (single token), we can sync once and call only selected experts
        // This reduces from 128 to 4 expert calls (32x reduction)
        if num_tokens == 1 {
            // Single sync to get all routing decisions
            indices.eval()?;
            let indices_flat: Vec<u32> = indices.as_slice().to_vec();

            // Process each top-k slot
            for k in 0..self.top_k as usize {
                let expert_idx = indices_flat[k] as usize;
                let ki = k as i32;
                let expert_weight = weights.index((.., ki..ki+1));

                // Call only the selected expert
                let expert_out = self.experts[expert_idx].forward(&x_flat)?;
                let weighted_out = expert_out.multiply(&expert_weight)?;
                output = output.add(&weighted_out)?;
            }
        } else {
            // For larger batches, use lazy evaluation with masking
            // Build the entire computation graph first, then let MLX optimize
            for k in 0..self.top_k {
                let expert_indices = indices.index((.., k..k + 1)).squeeze_axes(&[1])?;
                let expert_weights = weights.index((.., k..k + 1));

                // Process each expert with masking
                for expert_idx in 0..self.num_experts {
                    // Create mask for tokens routed to this expert
                    let mask = expert_indices.eq(&Array::from_int(expert_idx))?;

                    // Process tokens through expert (MLX will optimize if mask is all false)
                    let expert_out = self.experts[expert_idx as usize].forward(&x_flat)?;

                    // Weight and mask the output using where (lazy)
                    let weighted = expert_out.multiply(&expert_weights)?;
                    let zeros = mlx_rs::ops::zeros_like(&weighted)?;
                    let mask_expanded = mask.reshape(&[-1, 1])?;
                    let masked_weighted = mlx_rs::ops::r#where(&mask_expanded, &weighted, &zeros)?;
                    output = output.add(&masked_weighted)?;
                }
            }
        }

        // Add shared expert if present (always processes all tokens)
        if let Some(ref mut shared) = self.shared_expert {
            let shared_out = shared.forward(&x_flat)?;
            output = output.add(&shared_out)?;
        }

        // Reshape back to [B, L, H]
        output.reshape(&[batch, seq_len, hidden_size])
    }

    /// Convert this MoELayer to use stacked weights for gather_mm optimization.
    /// Returns stacked weight tensors: (W_up, W_down) each of shape [num_experts, out_dim, in_dim]
    pub fn stack_expert_weights(&self) -> Result<(Array, Array), Exception> {
        let num_experts = self.experts.len();

        // Collect all up_proj weights: [hidden, intermediate] for each expert
        let up_weights: Vec<&Array> = self.experts
            .iter()
            .map(|e| e.up_proj.weight.as_ref())
            .collect();

        // Stack along new first dimension: [num_experts, intermediate, hidden]
        // Note: Linear weights are stored transposed as [out_features, in_features]
        let stacked_up = mlx_rs::ops::stack_axis(&up_weights, 0)?;

        // Collect all down_proj weights: [intermediate, hidden] for each expert
        let down_weights: Vec<&Array> = self.experts
            .iter()
            .map(|e| e.down_proj.weight.as_ref())
            .collect();

        let stacked_down = mlx_rs::ops::stack_axis(&down_weights, 0)?;

        tracing::debug!(
            "Stacked MoE weights: up={:?}, down={:?}, num_experts={}",
            stacked_up.shape(),
            stacked_down.shape(),
            num_experts
        );

        Ok((stacked_up, stacked_down))
    }

    /// Fast forward using stacked weights and gather_mm.
    /// This batches all expert computations into single kernel launches.
    pub fn forward_stacked(
        &mut self,
        x: &Array,
        stacked_up: &Array,
        stacked_down: &Array,
    ) -> Result<Array, Exception> {
        use pmetal_mlx::gather_mm;

        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];
        let num_tokens = batch * seq_len;

        // Flatten batch and sequence: [B, L, H] -> [B*L, H]
        let x_flat = x.reshape(&[num_tokens, hidden_size])?;

        // Get routing weights and indices: weights [B*L, top_k], indices [B*L, top_k]
        let (weights, indices) = self.router.forward(&x_flat)?;

        // For gather_mm with rhs_indices:
        // x: [B*L, 1, 1, hidden]
        // W_up (transposed for matmul): [num_experts, hidden, intermediate]
        // indices: [B*L, top_k]
        // Result: [B*L, top_k, intermediate]

        // Expand x for batched expert computation: [B*L, 1, 1, hidden]
        let x_expanded = x_flat.reshape(&[num_tokens, 1, 1, hidden_size])?;

        // Transpose stacked weights for matmul: [num_experts, out, in] -> [num_experts, in, out]
        let up_T = stacked_up.swap_axes(-1, -2)?;
        let down_T = stacked_down.swap_axes(-1, -2)?;

        // Use gather_mm for up projection: selects expert weights based on indices
        // x_expanded: [B*L, 1, 1, hidden]
        // up_T: [num_experts, hidden, intermediate]
        // indices: [B*L, top_k]
        // Result: [B*L, top_k, intermediate]
        let up_out = gather_mm(&x_expanded, &up_T, None, Some(&indices), false)?;

        // Apply relu2 activation (relu squared)
        let activated = mlx_rs::nn::relu(&up_out)?.square()?;

        // Use gather_mm for down projection
        // activated: [B*L, top_k, intermediate]
        // down_T: [num_experts, intermediate, hidden]
        // Result: [B*L, top_k, hidden]
        let down_out = gather_mm(&activated, &down_T, None, Some(&indices), false)?;

        // Weight and sum over top_k experts: [B*L, top_k, hidden] -> [B*L, hidden]
        // weights: [B*L, top_k] -> expand to [B*L, top_k, 1]
        let weights_expanded = weights.reshape(&[num_tokens, self.top_k, 1])?;
        let weighted = down_out.multiply(&weights_expanded)?;
        let output = weighted.sum_axis(1, None)?; // Sum over top_k

        // Add shared expert if present
        let output = if let Some(ref mut shared) = self.shared_expert {
            let shared_out = shared.forward(&x_flat)?;
            output.add(&shared_out)?
        } else {
            output
        };

        // Reshape back to [B, L, H]
        output.reshape(&[batch, seq_len, hidden_size])
    }
}

/// Nemotron-H model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NemotronHConfig {
    /// Model type identifier.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Vocabulary size.
    pub vocab_size: i32,
    /// Hidden dimension.
    pub hidden_size: i32,
    /// Intermediate size for MLP.
    pub intermediate_size: i32,
    /// Number of hidden layers (derived from hybrid_override_pattern).
    pub num_hidden_layers: i32,
    /// Maximum position embeddings.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,

    // Attention config
    /// Number of attention heads.
    pub num_attention_heads: i32,
    /// Number of key-value heads (GQA).
    pub num_key_value_heads: i32,
    /// Attention bias.
    #[serde(default)]
    pub attention_bias: bool,
    /// Head dimension for attention.
    #[serde(default)]
    pub head_dim: Option<i32>,

    // Mamba config
    /// Number of Mamba heads.
    #[serde(default = "default_mamba_num_heads")]
    pub mamba_num_heads: i32,
    /// Mamba head dimension.
    #[serde(default = "default_mamba_head_dim")]
    pub mamba_head_dim: i32,
    /// Mamba projection bias.
    #[serde(default)]
    pub mamba_proj_bias: bool,
    /// SSM state size.
    #[serde(default = "default_ssm_state_size")]
    pub ssm_state_size: i32,
    /// Convolution kernel size.
    #[serde(default = "default_conv_kernel")]
    pub conv_kernel: i32,
    /// Number of groups for Mamba.
    #[serde(default = "default_n_groups")]
    pub n_groups: i32,
    /// Time step limits [min, max] - legacy field, prefer time_step_min/max.
    #[serde(default = "default_time_step_limit")]
    pub time_step_limit: (f32, f32),
    /// Minimum time step value (overrides time_step_limit[0] if set).
    #[serde(default)]
    pub time_step_min: Option<f32>,
    /// Maximum time step value (overrides time_step_limit[1] if set).
    #[serde(default)]
    pub time_step_max: Option<f32>,

    // MLP config
    /// MLP bias.
    #[serde(default)]
    pub mlp_bias: bool,
    /// MLP hidden activation.
    #[serde(default = "default_mlp_hidden_act")]
    pub mlp_hidden_act: String,

    // General config
    /// Layer norm epsilon.
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f32,
    /// Use bias in layers.
    #[serde(default)]
    pub use_bias: bool,
    /// Use bias in conv1d.
    #[serde(default = "default_use_conv_bias")]
    pub use_conv_bias: bool,
    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,

    // Hybrid pattern (e.g., "MEMEM*EMEMEM*...")
    /// Pattern defining layer types: M=Mamba, *=Attention, -=MLP, E=MoE.
    #[serde(default)]
    pub hybrid_override_pattern: Option<String>,

    // MoE config (optional, for Nemotron 3 Nano)
    /// MoE intermediate size per expert.
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
    /// MoE shared expert intermediate size.
    #[serde(default)]
    pub moe_shared_expert_intermediate_size: Option<i32>,
    /// Number of groups for MoE routing.
    #[serde(default)]
    pub n_group: Option<i32>,
    /// Number of routed experts.
    #[serde(default)]
    pub n_routed_experts: Option<i32>,
    /// Number of shared experts.
    #[serde(default)]
    pub n_shared_experts: Option<i32>,
    /// Top-k groups for expert selection.
    #[serde(default)]
    pub topk_group: Option<i32>,
    /// Number of experts per token.
    #[serde(default)]
    pub num_experts_per_tok: Option<i32>,
    /// Normalize top-k probabilities.
    #[serde(default)]
    pub norm_topk_prob: Option<bool>,
    /// Routed scaling factor.
    #[serde(default)]
    pub routed_scaling_factor: Option<f32>,

    /// RoPE theta for attention.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

fn default_model_type() -> String {
    "nemotron_h".to_string()
}
fn default_max_position_embeddings() -> i32 {
    262144
}
fn default_mamba_num_heads() -> i32 {
    64
}
fn default_mamba_head_dim() -> i32 {
    64
}
fn default_ssm_state_size() -> i32 {
    128
}
fn default_conv_kernel() -> i32 {
    4
}
fn default_n_groups() -> i32 {
    8
}
fn default_time_step_limit() -> (f32, f32) {
    (0.0, f32::INFINITY)
}
fn default_mlp_hidden_act() -> String {
    "relu2".to_string()
}
fn default_layer_norm_epsilon() -> f32 {
    1e-5
}
fn default_use_conv_bias() -> bool {
    true
}
fn default_rope_theta() -> f32 {
    10000.0
}

impl NemotronHConfig {
    /// Parse the hybrid override pattern into a vector of layer types.
    pub fn layer_types(&self) -> Vec<char> {
        self.hybrid_override_pattern
            .as_ref()
            .map(|p| p.chars().collect())
            .unwrap_or_else(|| vec!['*'; self.num_hidden_layers as usize])
    }

    /// Get attention head dimension.
    pub fn attention_head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Check if MoE is enabled.
    pub fn has_moe(&self) -> bool {
        self.n_routed_experts.is_some() && self.n_routed_experts.unwrap() > 0
    }

    /// Get Mamba intermediate size.
    pub fn mamba_intermediate_size(&self) -> i32 {
        self.mamba_num_heads * self.mamba_head_dim
    }

    /// Get Mamba conv dimension.
    pub fn mamba_conv_dim(&self) -> i32 {
        self.mamba_intermediate_size() + 2 * self.n_groups * self.ssm_state_size
    }
}

/// Unified mixer that handles all block types (Mamba, Attention, MoE, MLP).
///
/// The field name `mixer` matches HuggingFace's weight naming convention.
/// Internally dispatches to the appropriate implementation based on block type.
#[derive(Debug, ModuleParameters)]
pub struct NemotronHMixer {
    /// Block type: 'M' (Mamba), '*' (Attention), 'E' (MoE), '-' (MLP).
    pub block_type: char,

    // Mamba components (only used for 'M' blocks)
    #[param]
    pub in_proj: Option<nn::Linear>,
    #[param]
    pub conv1d: Option<nn::Conv1d>,
    #[param]
    pub out_proj: Option<nn::Linear>,

    // FP8 scale factors for Mamba projections
    pub in_proj_weight_scale: Option<Array>,
    pub in_proj_input_scale: Option<Array>,
    pub out_proj_weight_scale: Option<Array>,
    pub out_proj_input_scale: Option<Array>,

    // SSM parameters (Mamba-2 specific)
    /// Log of state transition matrix [num_heads]
    pub a_log: Option<Array>,
    /// Skip connection weights [num_heads]
    pub d: Option<Array>,
    /// Time step bias [num_heads]
    pub dt_bias: Option<Array>,
    /// Internal gated RMS norm (not a standard nn module, handled separately)
    pub gated_norm: Option<MambaRMSNormGated>,

    // Attention components (only used for '*' blocks)
    #[param]
    pub q_proj: Option<nn::Linear>,
    #[param]
    pub k_proj: Option<nn::Linear>,
    #[param]
    pub v_proj: Option<nn::Linear>,
    #[param]
    pub o_proj: Option<nn::Linear>,

    // MLP components (only used for '-' blocks)
    #[param]
    pub up_proj: Option<nn::Linear>,
    #[param]
    pub down_proj: Option<nn::Linear>,

    // MoE components (only used for 'E' blocks)
    #[param]
    pub moe_layer: Option<MoELayer>,
    /// Stacked MoE up_proj weights: [num_experts, intermediate, hidden]
    pub stacked_moe_up: Option<Array>,
    /// Stacked MoE down_proj weights: [num_experts, hidden, intermediate]
    pub stacked_moe_down: Option<Array>,

    // Attention config
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_theta: f32,

    // Mamba config
    pub mamba_num_heads: i32,
    pub mamba_head_dim: i32,
    pub mamba_intermediate_size: i32,
    pub mamba_conv_dim: i32,
    pub ssm_state_size: i32,
    pub n_groups: i32,
    pub conv_kernel_size: i32,
    pub time_step_min: f32,
    pub time_step_max: f32,
}

impl NemotronHMixer {
    /// Create a new Mamba mixer.
    pub fn new_mamba(config: &NemotronHConfig) -> Result<Self, Exception> {
        let intermediate_size = config.mamba_intermediate_size();
        let conv_dim = config.mamba_conv_dim();
        let mamba_num_heads = config.mamba_num_heads;
        let conv_kernel_size = config.conv_kernel;

        // Compute effective time step limits
        let time_step_min = config.time_step_min.unwrap_or(config.time_step_limit.0);
        let time_step_max = config.time_step_max.unwrap_or(config.time_step_limit.1);
        tracing::debug!(
            "Mamba time_step limits: min={}, max={} (from config: min={:?}, max={:?}, limit={:?})",
            time_step_min, time_step_max, config.time_step_min, config.time_step_max, config.time_step_limit
        );

        // Input projection: hidden -> intermediate + conv_dim + num_heads
        let projection_size = intermediate_size + conv_dim + mamba_num_heads;
        let in_proj = nn::LinearBuilder::new(config.hidden_size, projection_size)
            .bias(config.mamba_proj_bias)
            .build()?;

        // Depthwise conv1d (input is conv_dim channels)
        // NOTE: padding=0 because we handle causal padding manually in _apply_conv
        let conv1d = nn::Conv1dBuilder::new(conv_dim, conv_dim, conv_kernel_size)
            .groups(conv_dim)
            .bias(config.use_conv_bias)
            .padding(0)
            .build()?;

        // Output projection
        let out_proj = nn::LinearBuilder::new(intermediate_size, config.hidden_size)
            .bias(config.mamba_proj_bias)
            .build()?;

        // SSM parameters - initialized with defaults, loaded from weights later
        let a_log = Some(Array::zeros::<f32>(&[mamba_num_heads])?);
        let d = Some(Array::ones::<f32>(&[mamba_num_heads])?);
        let dt_bias = Some(Array::ones::<f32>(&[mamba_num_heads])?);

        // Internal gated RMS norm with group-wise normalization
        let gated_norm = Some(MambaRMSNormGated::new(
            intermediate_size,
            config.layer_norm_epsilon,
            config.n_groups,
        )?);

        Ok(Self {
            block_type: 'M',
            in_proj: Some(in_proj),
            conv1d: Some(conv1d),
            out_proj: Some(out_proj),
            in_proj_weight_scale: None,
            in_proj_input_scale: None,
            out_proj_weight_scale: None,
            out_proj_input_scale: None,
            a_log,
            d,
            dt_bias,
            gated_norm,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            o_proj: None,
            up_proj: None,
            down_proj: None,
            moe_layer: None,
            stacked_moe_up: None,
            stacked_moe_down: None,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            scale: 0.0,
            rope_theta: 0.0,
            mamba_num_heads,
            mamba_head_dim: config.mamba_head_dim,
            mamba_intermediate_size: intermediate_size,
            mamba_conv_dim: conv_dim,
            ssm_state_size: config.ssm_state_size,
            n_groups: config.n_groups,
            conv_kernel_size,
            time_step_min,
            time_step_max,
        })
    }

    /// Create a new Attention mixer.
    pub fn new_attention(config: &NemotronHConfig) -> Result<Self, Exception> {
        let head_dim = config.attention_head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;

        let q_proj = nn::LinearBuilder::new(config.hidden_size, num_heads * head_dim)
            .bias(config.attention_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(config.hidden_size, num_kv_heads * head_dim)
            .bias(config.attention_bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(config.hidden_size, num_kv_heads * head_dim)
            .bias(config.attention_bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(num_heads * head_dim, config.hidden_size)
            .bias(config.attention_bias)
            .build()?;

        let scale = (head_dim as f32).sqrt().recip();

        Ok(Self {
            block_type: '*',
            in_proj: None,
            conv1d: None,
            out_proj: None,
            in_proj_weight_scale: None,
            in_proj_input_scale: None,
            out_proj_weight_scale: None,
            out_proj_input_scale: None,
            a_log: None,
            d: None,
            dt_bias: None,
            gated_norm: None,
            q_proj: Some(q_proj),
            k_proj: Some(k_proj),
            v_proj: Some(v_proj),
            o_proj: Some(o_proj),
            up_proj: None,
            down_proj: None,
            moe_layer: None,
            stacked_moe_up: None,
            stacked_moe_down: None,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            rope_theta: config.rope_theta,
            mamba_num_heads: 0,
            mamba_head_dim: 0,
            mamba_intermediate_size: 0,
            mamba_conv_dim: 0,
            ssm_state_size: 0,
            n_groups: 0,
            conv_kernel_size: 0,
            time_step_min: 0.0,
            time_step_max: 0.0,
        })
    }

    /// Create a new MLP mixer.
    pub fn new_mlp(config: &NemotronHConfig) -> Result<Self, Exception> {
        let up_proj = nn::LinearBuilder::new(config.hidden_size, config.intermediate_size)
            .bias(config.mlp_bias)
            .build()?;
        let down_proj = nn::LinearBuilder::new(config.intermediate_size, config.hidden_size)
            .bias(config.mlp_bias)
            .build()?;

        Ok(Self {
            block_type: '-',
            in_proj: None,
            conv1d: None,
            out_proj: None,
            in_proj_weight_scale: None,
            in_proj_input_scale: None,
            out_proj_weight_scale: None,
            out_proj_input_scale: None,
            a_log: None,
            d: None,
            dt_bias: None,
            gated_norm: None,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            o_proj: None,
            up_proj: Some(up_proj),
            down_proj: Some(down_proj),
            moe_layer: None,
            stacked_moe_up: None,
            stacked_moe_down: None,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            scale: 0.0,
            rope_theta: 0.0,
            mamba_num_heads: 0,
            mamba_head_dim: 0,
            mamba_intermediate_size: 0,
            mamba_conv_dim: 0,
            ssm_state_size: 0,
            n_groups: 0,
            conv_kernel_size: 0,
            time_step_min: 0.0,
            time_step_max: 0.0,
        })
    }

    /// Create a new MoE mixer with full expert routing.
    pub fn new_moe(config: &NemotronHConfig) -> Result<Self, Exception> {
        // Get MoE config values with defaults
        let num_experts = config.n_routed_experts.unwrap_or(8);
        let top_k = config.num_experts_per_tok.unwrap_or(2);
        let n_group = config.n_group.unwrap_or(1);
        let topk_group = config.topk_group.unwrap_or(1);
        let norm_topk_prob = config.norm_topk_prob.unwrap_or(false);
        let moe_intermediate_size = config.moe_intermediate_size.unwrap_or(config.intermediate_size);
        let shared_intermediate_size = config.moe_shared_expert_intermediate_size;
        let use_shared_expert = config.n_shared_experts.unwrap_or(0) > 0;
        let routed_scaling_factor = config.routed_scaling_factor.unwrap_or(1.0);

        let moe_layer = MoELayer::new(
            config.hidden_size,
            moe_intermediate_size,
            shared_intermediate_size,
            num_experts,
            top_k,
            n_group,
            topk_group,
            norm_topk_prob,
            use_shared_expert,
            routed_scaling_factor,
            config.mlp_bias,
        )?;

        Ok(Self {
            block_type: 'E',
            in_proj: None,
            conv1d: None,
            out_proj: None,
            in_proj_weight_scale: None,
            in_proj_input_scale: None,
            out_proj_weight_scale: None,
            out_proj_input_scale: None,
            a_log: None,
            d: None,
            dt_bias: None,
            gated_norm: None,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            o_proj: None,
            up_proj: None,
            down_proj: None,
            moe_layer: Some(moe_layer),
            stacked_moe_up: None,
            stacked_moe_down: None,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            scale: 0.0,
            rope_theta: 0.0,
            mamba_num_heads: 0,
            mamba_head_dim: 0,
            mamba_intermediate_size: 0,
            mamba_conv_dim: 0,
            ssm_state_size: 0,
            n_groups: 0,
            conv_kernel_size: 0,
            time_step_min: 0.0,
            time_step_max: 0.0,
        })
    }

    /// Create a mixer based on block type character.
    pub fn new(config: &NemotronHConfig, block_type: char) -> Result<Self, Exception> {
        match block_type {
            'M' => Self::new_mamba(config),
            '*' => Self::new_attention(config),
            '-' => Self::new_mlp(config),
            'E' => Self::new_moe(config),
            _ => Self::new_attention(config), // Default to attention
        }
    }

    /// Forward pass with optional KV cache and Mamba cache.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        mamba_cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        match self.block_type {
            'M' => self.forward_mamba(x, mamba_cache),
            '*' => self.forward_attention(x, mask, kv_cache),
            '-' => self.forward_mlp(x),
            'E' => self.forward_moe(x),
            _ => Ok(x.clone()),
        }
    }

    fn forward_mamba(
        &mut self,
        x: &Array,
        mut cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        let in_proj = self.in_proj.as_mut().unwrap();
        let conv1d = self.conv1d.as_mut().unwrap();
        let out_proj = self.out_proj.as_mut().unwrap();
        let gated_norm = self.gated_norm.as_ref().unwrap();

        let a_log = self.a_log.as_ref().unwrap();
        let d_param = self.d.as_ref().unwrap();
        let dt_bias = self.dt_bias.as_ref().unwrap();

        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let intermediate_size = self.mamba_intermediate_size;
        let conv_dim = self.mamba_conv_dim;
        let n_groups = self.n_groups;
        let ssm_state_size = self.ssm_state_size;
        let num_heads = self.mamba_num_heads;
        let head_dim = self.mamba_head_dim;
        let conv_kernel = self.conv_kernel_size;

        // Debug: trace Mamba forward values (disabled for performance)
        static MAMBA_TRACE_LOG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let mamba_trace = MAMBA_TRACE_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let log_mamba_trace = false && mamba_trace < 2;  // Disabled

        if log_mamba_trace {
            x.eval()?;
            tracing::info!(
                "MAMBA[{}] input x: shape={:?}, min={:.4}, max={:.4}",
                mamba_trace, x.shape(), x.min(None)?.item::<f32>(), x.max(None)?.item::<f32>()
            );
        }

        // Input projection with FP8 weight dequantization
        // NOTE: Do NOT apply input_scale - that's for FP8 dynamic quantization
        let projected = if let Some(ref ws) = self.in_proj_weight_scale {
            // Dequantize weight and compute matmul (no input scaling for float inference)
            let weight = in_proj.weight.as_ref().multiply(ws)?;
            let out = x.matmul(&weight.t())?;
            if let Some(ref bias) = in_proj.bias.as_ref() {
                out.add(bias)?
            } else {
                out
            }
        } else {
            Module::forward(in_proj, x)?
        };

        if log_mamba_trace {
            projected.eval()?;
            tracing::info!(
                "MAMBA[{}] projected: shape={:?}, min={:.4}, max={:.4}",
                mamba_trace, projected.shape(), projected.min(None)?.item::<f32>(), projected.max(None)?.item::<f32>()
            );
        }

        // Use split_sections for efficient splitting (optimization #1)
        // Split at: [intermediate_size, intermediate_size + conv_dim]
        let split_indices = &[intermediate_size, intermediate_size + conv_dim];
        let parts = mlx_rs::ops::split_sections(&projected, split_indices, -1)?;
        let gate = &parts[0];        // [B, L, intermediate_size]
        let conv_input = &parts[1];  // [B, L, conv_dim]
        let dt = &parts[2];          // [B, L, num_heads]

        // Apply conv1d with state caching for incremental generation
        let conv_activated = if let Some(ref mut mamba_cache) = cache {
            // Use cached conv state for causal convolution
            let padded_input = mamba_cache.update_conv_state(conv_input, conv_kernel)?;

            if log_mamba_trace {
                conv_input.eval()?;
                padded_input.eval()?;
                let ci_t2_3478 = if seq_len > 2 { conv_input.index((0, 2, 3478)).item::<f32>() } else { f32::NAN };
                tracing::info!(
                    "MAMBA[{}] CONV DEBUG: conv_input shape={:?}, padded_input shape={:?}, conv_input[0,2,3478]={:.4}",
                    mamba_trace, conv_input.shape(), padded_input.shape(), ci_t2_3478
                );
            }

            let conv_out = Module::forward(conv1d, &padded_input)?;
            // Output is [B, padded_len, conv_dim], truncate to [B, seq_len, conv_dim]
            let out_len = conv_out.dim(1);

            if log_mamba_trace {
                conv_out.eval()?;
                let co_t2_3478 = if out_len > 2 { conv_out.index((0, 2, 3478)).item::<f32>() } else { f32::NAN };
                tracing::info!(
                    "MAMBA[{}] CONV DEBUG: conv_out shape={:?}, out_len={}, seq_len={}, conv_out[0,2,3478]={:.4}",
                    mamba_trace, conv_out.shape(), out_len, seq_len, co_t2_3478
                );
            }

            let conv_out = conv_out.index((.., (out_len - seq_len).., ..));
            nn::silu(&conv_out)?
        } else {
            // No cache: use CAUSAL PADDING for full sequence
            // Pad (kernel_size - 1) zeros on the left of the sequence
            // This matches reference: mx.pad(conv_input, [(0, 0), (kernel_size - 1, 0), (0, 0)])
            let pad_amount = (conv_kernel - 1) as i32;
            let padded_input = mlx_rs::ops::pad(
                conv_input,
                &[(0i32, 0i32), (pad_amount, 0), (0, 0)],
                Array::from_int(0),  // pad value = 0
                None,  // mode = Constant (default)
            )?;
            let conv_out = Module::forward(conv1d, &padded_input)?;
            // Conv output has same length as padded input minus (kernel_size - 1)
            // So conv_out length = (seq_len + pad_amount) - pad_amount = seq_len
            nn::silu(&conv_out)?
        };

        // Split conv output with split_sections (optimization #1)
        let bc_size = n_groups * ssm_state_size;
        let conv_split_indices = &[intermediate_size, intermediate_size + bc_size];
        let conv_parts = mlx_rs::ops::split_sections(&conv_activated, conv_split_indices, -1)?;
        let hidden_states = &conv_parts[0]; // [B, L, intermediate_size]
        let b_proj = &conv_parts[1];        // [B, L, n_groups * ssm_state_size]
        let c_proj = &conv_parts[2];        // [B, L, n_groups * ssm_state_size]

        if log_mamba_trace {
            conv_activated.eval()?;
            hidden_states.eval()?;
            tracing::info!(
                "MAMBA[{}] conv_activated: shape={:?}, min={:.4}, max={:.4}",
                mamba_trace, conv_activated.shape(), conv_activated.min(None)?.item::<f32>(), conv_activated.max(None)?.item::<f32>()
            );
            tracing::info!(
                "MAMBA[{}] hidden_states (SSM input): shape={:?}, min={:.4}, max={:.4}",
                mamba_trace, hidden_states.shape(), hidden_states.min(None)?.item::<f32>(), hidden_states.max(None)?.item::<f32>()
            );
            // Check position 3478 (head 54, dim 26) in SSM input for tokens 2 and 19
            if seq_len > 2 {
                let hs_t2 = hidden_states.index((0, 2, 3478)).item::<f32>();
                let hs_t19 = if seq_len > 19 {
                    hidden_states.index((0, 19, 3478)).item::<f32>()
                } else {
                    f32::NAN
                };
                tracing::info!(
                    "MAMBA[{}] hidden_states at pos 3478: Token 2={:.4}, Token 19={:.4}",
                    mamba_trace, hs_t2, hs_t19
                );
            }
        }

        // Reshape for multi-head processing (combined reshape - optimization #2)
        // hidden_states: [B, L, num_heads * head_dim] -> [B, L, num_heads, head_dim]
        let x_heads = hidden_states.reshape(&[batch, seq_len, num_heads, head_dim])?;

        // B, C: [B, L, n_groups * ssm_state_size] -> [B, L, n_groups, ssm_state_size]
        let b_reshaped = b_proj.reshape(&[batch, seq_len, n_groups, ssm_state_size])?;
        let c_reshaped = c_proj.reshape(&[batch, seq_len, n_groups, ssm_state_size])?;

        // Get previous SSM state from cache
        let prev_state = cache.as_ref().and_then(|c| c.get_ssm_state());

        // Debug: trace cache state (disabled for performance)
        static MAMBA_FWD_LOG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let mamba_log = MAMBA_FWD_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if false && mamba_log < 10 {  // Disabled
            let has_cache = cache.is_some();
            let has_prev_state = prev_state.is_some();
            tracing::debug!(
                "forward_mamba[{}]: seq_len={}, has_cache={}, has_prev_state={}",
                mamba_log, seq_len, has_cache, has_prev_state
            );
        }

        // Use optimized single-token update when seq_len=1 and we have previous state
        // This avoids the expensive full SSM attention computation
        static FAST_PATH_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        static SLOW_PATH_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let (y, next_state) = if seq_len == 1 && prev_state.is_some() {
            let count = FAST_PATH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count < 5 || count % 1000 == 0 {
                tracing::info!("SSM fast path: seq_len={}, count={}", seq_len, count);
            }
            ssm_update_single(
                &x_heads,
                a_log,
                &b_reshaped,
                &c_reshaped,
                d_param,
                dt,
                dt_bias,
                prev_state.unwrap(),
                (self.time_step_min, self.time_step_max),
            )?
        } else {
            // Use full SSM attention computation
            let count = SLOW_PATH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count < 5 || count % 1000 == 0 {
                tracing::info!("SSM slow path: seq_len={}, has_state={}, count={}", seq_len, prev_state.is_some(), count);
            }
            ssm_attention(
                &x_heads,
                a_log,
                &b_reshaped,
                &c_reshaped,
                d_param,
                dt,
                dt_bias,
                prev_state,
                (self.time_step_min, self.time_step_max),
            )?
        };

        // Update SSM state in cache
        if let Some(mamba_cache) = cache {
            mamba_cache.set_ssm_state(next_state);
        }

        // Reshape back to [B, L, intermediate_size]
        let y = y.reshape(&[batch, seq_len, intermediate_size])?;

        if log_mamba_trace && seq_len > 2 {
            y.eval()?;
            // Check SSM output at position 3478 for tokens 2 and 19
            let y_t2 = y.index((0, 2, 3478)).item::<f32>();
            let y_t19 = if seq_len > 19 {
                y.index((0, 19, 3478)).item::<f32>()
            } else {
                f32::NAN
            };
            tracing::info!(
                "MAMBA[{}] SSM output y at pos 3478: Token 2={:.4}, Token 19={:.4}",
                mamba_trace, y_t2, y_t19
            );
        }

        // Gated RMS norm with group-wise normalization
        // This handles: y = norm(y * silu(gate)) with groups
        let y_normed = gated_norm.forward(&y, Some(gate))?;

        if log_mamba_trace {
            y.eval()?;
            y_normed.eval()?;
            tracing::info!(
                "MAMBA[{}] y (SSM out before norm): min={:.4}, max={:.4}",
                mamba_trace, y.min(None)?.item::<f32>(), y.max(None)?.item::<f32>()
            );
            tracing::info!(
                "MAMBA[{}] y_normed (after gated norm): min={:.4}, max={:.4}",
                mamba_trace, y_normed.min(None)?.item::<f32>(), y_normed.max(None)?.item::<f32>()
            );
        }

        // Output projection with FP8 weight dequantization
        // NOTE: Do NOT apply input_scale - that's for FP8 dynamic quantization
        if let Some(ref ws) = self.out_proj_weight_scale {
            // Dequantize weight and compute matmul (no input scaling for float inference)
            let weight = out_proj.weight.as_ref().multiply(ws)?;
            let out = y_normed.matmul(&weight.t())?;
            if let Some(ref bias) = out_proj.bias.as_ref() {
                out.add(bias)
            } else {
                Ok(out)
            }
        } else {
            Module::forward(out_proj, &y_normed)
        }
    }

    fn forward_attention(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        let q_proj = self.q_proj.as_mut().unwrap();
        let k_proj = self.k_proj.as_mut().unwrap();
        let v_proj = self.v_proj.as_mut().unwrap();
        let o_proj = self.o_proj.as_mut().unwrap();

        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Project Q, K, V
        let q = Module::forward(q_proj, x)?;
        let k = Module::forward(k_proj, x)?;
        let v = Module::forward(v_proj, x)?;

        // Reshape for multi-head attention [B, L, heads, head_dim]
        let q = q.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;

        // Transpose for attention: [B, heads, L, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE
        let (q, k, v) = if let Some((ref cache_ref, _)) = cache {
            let offset = cache_ref.rope_offset();
            let q = apply_rope(&q, self.head_dim, false, self.rope_theta, 1.0, offset)?;
            let k = apply_rope(&k, self.head_dim, false, self.rope_theta, 1.0, offset)?;
            (q, k, v)
        } else {
            let q = apply_rope(&q, self.head_dim, false, self.rope_theta, 1.0, 0)?;
            let k = apply_rope(&k, self.head_dim, false, self.rope_theta, 1.0, 0)?;
            (q, k, v)
        };

        // Update KV cache if provided
        let (k, v) = if let Some((cache, layer_idx)) = cache {
            cache.update_and_fetch(layer_idx, &k, &v)?
        } else {
            (k, v)
        };

        // Use fused attention kernel
        let attn_config =
            FusedAttentionConfig::new(self.num_heads, self.num_kv_heads, self.head_dim)
                .with_scale(self.scale)
                .with_mask_type(AttentionMaskType::Causal);

        let output = fused_sdpa(&q, &k, &v, &attn_config, mask)?;

        // Reshape and project output
        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, -1])?;
        Module::forward(o_proj, &output)
    }

    fn forward_mlp(&mut self, x: &Array) -> Result<Array, Exception> {
        let up_proj = self.up_proj.as_mut().unwrap();
        let down_proj = self.down_proj.as_mut().unwrap();

        let up = Module::forward(up_proj, x)?;
        // relu2 = relu(x)^2
        let activated = nn::relu(&up)?.square()?;
        Module::forward(down_proj, &activated)
    }

    fn forward_moe(&mut self, x: &Array) -> Result<Array, Exception> {
        let moe_layer = self.moe_layer.as_mut().unwrap();

        // Use stacked weights with gather_mm if available (much faster)
        if let (Some(stacked_up), Some(stacked_down)) = (&self.stacked_moe_up, &self.stacked_moe_down) {
            moe_layer.forward_stacked(x, stacked_up, stacked_down)
        } else {
            moe_layer.forward(x)
        }
    }

    /// Initialize stacked MoE weights for gather_mm optimization.
    /// Call this after model loading to enable fast MoE inference.
    pub fn init_stacked_moe(&mut self) -> Result<(), Exception> {
        if let Some(ref moe_layer) = self.moe_layer {
            let (stacked_up, stacked_down) = moe_layer.stack_expert_weights()?;
            // Evaluate the stacked weights once
            stacked_up.eval()?;
            stacked_down.eval()?;
            self.stacked_moe_up = Some(stacked_up);
            self.stacked_moe_down = Some(stacked_down);
            tracing::info!("Initialized stacked MoE weights for layer");
        }
        Ok(())
    }
}

/// Nemotron-H hybrid block.
#[derive(Debug, ModuleParameters)]
pub struct NemotronHBlock {
    #[param]
    pub norm: nn::RmsNorm,
    #[param]
    pub mixer: NemotronHMixer,
}

impl NemotronHBlock {
    pub fn new(config: &NemotronHConfig, block_type: char) -> Result<Self, Exception> {
        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_epsilon)
            .build()?;

        let mixer = NemotronHMixer::new(config, block_type)?;

        Ok(Self { norm, mixer })
    }

    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        mamba_cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, Exception> {
        let hidden = Module::forward(&mut self.norm, x)?;
        let output = self.mixer.forward_with_cache(&hidden, mask, kv_cache, mamba_cache)?;
        // Residual connection
        x.add(&output)
    }

    pub fn block_type(&self) -> char {
        self.mixer.block_type
    }
}

/// Nemotron-H model backbone.
#[derive(Debug, ModuleParameters)]
pub struct NemotronHModel {
    #[param]
    pub embeddings: nn::Embedding,
    #[param]
    pub layers: Vec<NemotronHBlock>,
    #[param]
    pub norm_f: nn::RmsNorm,
    pub config: NemotronHConfig,
}

impl NemotronHModel {
    pub fn new(config: NemotronHConfig) -> Result<Self, Exception> {
        let embeddings = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layer_types = config.layer_types();
        let mut layers = Vec::with_capacity(layer_types.len());
        for &block_type in &layer_types {
            layers.push(NemotronHBlock::new(&config, block_type)?);
        }

        let norm_f = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_epsilon)
            .build()?;

        Ok(Self {
            embeddings,
            layers,
            norm_f,
            config,
        })
    }

    /// Initialize stacked MoE weights for all MoE layers.
    /// Call this after model loading to enable fast gather_mm inference.
    /// This provides ~10x speedup for MoE layers.
    pub fn init_stacked_moe(&mut self) -> Result<(), Exception> {
        let mut moe_count = 0;
        for layer in &mut self.layers {
            layer.mixer.init_stacked_moe()?;
            if layer.mixer.moe_layer.is_some() {
                moe_count += 1;
            }
        }
        if moe_count > 0 {
            tracing::info!("Initialized stacked weights for {} MoE layers", moe_count);
        }
        Ok(())
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut kv_cache: Option<&mut KVCache>,
        mut mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        // Debug: check embedding weights and input (disabled for performance)
        const DEBUG_EMBEDDINGS: bool = false;
        if DEBUG_EMBEDDINGS {
            input_ids.eval()?;
            tracing::info!(
                "INPUT_IDS: shape={:?}, dtype={:?}, first few={:?}",
                input_ids.shape(),
                input_ids.dtype(),
                if input_ids.size() <= 30 {
                    input_ids.flatten(None, None)?.as_slice::<i32>().to_vec()
                } else {
                    input_ids.flatten(None, None)?.index(..30).as_slice::<i32>().to_vec()
                }
            );
            let emb_w = &self.embeddings.weight.value;
            emb_w.eval()?;
            tracing::info!(
                "EMBEDDING WEIGHTS: shape={:?}, dtype={:?}, min={:.6}, max={:.6}",
                emb_w.shape(),
                emb_w.dtype(),
                emb_w.min(None)?.item::<f32>(),
                emb_w.max(None)?.item::<f32>()
            );
            // Check specific token (22177 = "Hello")
            let tok_22177 = emb_w.index((22177, ..));
            tok_22177.eval()?;
            tracing::info!(
                "TOKEN 22177 embedding: min={:.6}, max={:.6}",
                tok_22177.min(None)?.item::<f32>(),
                tok_22177.max(None)?.item::<f32>()
            );
        }

        let mut hidden = Module::forward(&mut self.embeddings, input_ids)?;

        // Debug: check embedding output immediately (disabled for performance)
        if DEBUG_EMBEDDINGS {
            hidden.eval()?;
            let emb_min = hidden.min(None)?.item::<f32>();
            let emb_max = hidden.max(None)?.item::<f32>();
            let emb_mean = hidden.mean(None)?.item::<f32>();
            tracing::info!(
                "EMBEDDING OUTPUT: shape={:?}, dtype={:?}, min={:.6}, max={:.6}, mean={:.6}",
                hidden.shape(),
                hidden.dtype(),
                emb_min,
                emb_max,
                emb_mean
            );
        }

        // Process layers - attention layers use KV cache, Mamba layers use Mamba cache
        // Track timing for profiling
        static PROFILE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let profile_iter = PROFILE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let do_profile = false && profile_iter < 3;  // Disabled - set to true to profile
        let layer_start = if do_profile { Some(std::time::Instant::now()) } else { None };
        let mut mamba_time = std::time::Duration::ZERO;
        let mut attn_time = std::time::Duration::ZERO;
        let mut mlp_time = std::time::Duration::ZERO;
        let mut moe_time = std::time::Duration::ZERO;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let layer_mask = if layer.block_type() == '*' {
                mask
            } else {
                None
            };

            // Determine which caches to pass based on layer type
            let kv = if layer.block_type() == '*' {
                kv_cache.as_deref_mut().map(|c| (c, layer_idx))
            } else {
                None
            };
            let mamba = if layer.block_type() == 'M' {
                mamba_cache.as_deref_mut().and_then(|c| c.get_mut(layer_idx))
            } else {
                None
            };

            let op_start = if do_profile { Some(std::time::Instant::now()) } else { None };
            hidden = layer.forward_with_cache(&hidden, layer_mask, kv, mamba)?;

            // Sync and time for profiling
            if do_profile {
                hidden.eval()?;
                let elapsed = op_start.unwrap().elapsed();
                match layer.block_type() {
                    'M' => mamba_time += elapsed,
                    '*' => attn_time += elapsed,
                    '-' => mlp_time += elapsed,
                    'E' => moe_time += elapsed,
                    _ => {}
                }
            }
        }

        if do_profile {
            let total = layer_start.unwrap().elapsed();
            tracing::info!(
                "LAYER TIMING[{}]: total={:.1}ms, mamba={:.1}ms, attn={:.1}ms, mlp={:.1}ms, moe={:.1}ms",
                profile_iter,
                total.as_secs_f64() * 1000.0,
                mamba_time.as_secs_f64() * 1000.0,
                attn_time.as_secs_f64() * 1000.0,
                mlp_time.as_secs_f64() * 1000.0,
                moe_time.as_secs_f64() * 1000.0,
            );
        }

        Module::forward(&mut self.norm_f, &hidden)
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.forward_with_cache(input_ids, mask, None, None)
    }
}

/// Nemotron-H for causal language modeling.
#[derive(Debug, ModuleParameters)]
pub struct NemotronHForCausalLM {
    #[param]
    pub backbone: NemotronHModel,
    #[param]
    pub lm_head: Option<nn::Linear>,
}

impl NemotronHForCausalLM {
    pub fn new(config: NemotronHConfig) -> Result<Self, Exception> {
        let tie_weights = config.tie_word_embeddings;
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;

        let backbone = NemotronHModel::new(config)?;

        let lm_head = if !tie_weights {
            Some(
                nn::LinearBuilder::new(hidden_size, vocab_size)
                    .bias(false)
                    .build()?,
            )
        } else {
            None
        };

        Ok(Self { backbone, lm_head })
    }

    /// Get config reference.
    pub fn config(&self) -> &NemotronHConfig {
        &self.backbone.config
    }

    /// Forward pass producing logits.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let hidden = self.backbone.forward(input_ids, mask)?;

        if let Some(ref mut lm_head) = self.lm_head {
            Module::forward(lm_head, &hidden)
        } else {
            // Tie weights: use embedding weight transposed
            self.backbone.embeddings.as_linear(&hidden)
        }
    }

    /// Forward pass with optional KV cache and Mamba cache.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        let hidden = self.backbone.forward_with_cache(input_ids, mask, kv_cache, mamba_cache)?;

        // Debug: trace hidden states after backbone (disabled for performance)
        static FORWARD_LOG_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let forward_log = FORWARD_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if false && forward_log < 3 {  // Disabled
            hidden.eval()?;
            let last_hidden = hidden.index((.., -1, ..));
            last_hidden.eval()?;
            tracing::info!(
                "FINAL HIDDEN[{}]: shape={:?}, min={:.4}, max={:.4}, mean={:.6}",
                forward_log,
                hidden.shape(),
                hidden.min(None)?.item::<f32>(),
                hidden.max(None)?.item::<f32>(),
                hidden.mean(None)?.item::<f32>()
            );
            tracing::info!(
                "LAST TOKEN HIDDEN[{}]: shape={:?}, min={:.4}, max={:.4}, mean={:.6}",
                forward_log,
                last_hidden.shape(),
                last_hidden.min(None)?.item::<f32>(),
                last_hidden.max(None)?.item::<f32>(),
                last_hidden.mean(None)?.item::<f32>()
            );
        }

        let logits = if let Some(ref mut lm_head) = self.lm_head {
            Module::forward(lm_head, &hidden)?
        } else {
            // Tie weights: use embedding weight transposed
            self.backbone.embeddings.as_linear(&hidden)?
        };

        // Debug: trace logits (disabled for performance)
        if false && forward_log < 3 {
            logits.eval()?;
            let last_logits = logits.index((.., -1, ..));
            last_logits.eval()?;
            tracing::info!(
                "LOGITS[{}]: shape={:?}, min={:.4}, max={:.4}, mean={:.6}",
                forward_log,
                logits.shape(),
                logits.min(None)?.item::<f32>(),
                logits.max(None)?.item::<f32>(),
                logits.mean(None)?.item::<f32>()
            );
            // Get top-5 token predictions
            let neg_logits = last_logits.negative()?;
            let top5_indices = mlx_rs::ops::argpartition_axis(&neg_logits, 4, -1)?;
            let top5_indices = top5_indices.index((.., ..5));
            top5_indices.eval()?;
            let top5_vec: Vec<u32> = top5_indices.as_slice().to_vec();
            let top5_logits: Vec<f32> = top5_vec.iter().map(|&idx| {
                last_logits.index((.., idx as i32)).item::<f32>()
            }).collect();
            tracing::info!(
                "TOP-5 TOKENS[{}]: indices={:?}, logits={:?}",
                forward_log,
                top5_vec,
                top5_logits
            );
            // Check expected tokens: "I"=1073, "Hi"=37133, "Hello"=22177
            let expected_tokens = [1073i32, 37133, 22177, 30859, 11564];
            let expected_logits: Vec<f32> = expected_tokens.iter().map(|&idx| {
                last_logits.index((.., idx)).item::<f32>()
            }).collect();
            tracing::info!(
                "EXPECTED TOKENS[{}]: ids=[I, Hi, Hello, Thank, well], logits={:?}",
                forward_log,
                expected_logits
            );
        }

        Ok(logits)
    }

    /// Evaluate all parameters to materialize them on device.
    pub fn eval(&self) -> Result<(), Exception> {
        let params = self.parameters().flatten();
        for (_, param) in params {
            param.eval()?;
        }
        Ok(())
    }
}

/// Load weights from HuggingFace format into NemotronH model.
pub fn load_nemotron_weights(
    model: &mut NemotronHForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), crate::loader::LoadError> {
    // Load embeddings
    if let Some(w) = weights.get("backbone.embeddings.weight") {
        model.backbone.embeddings.weight = Param::new(w.clone());
    } else if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.backbone.embeddings.weight = Param::new(w.clone());
    }

    // Load transformer layers
    let layer_types = model.backbone.config.layer_types();
    tracing::debug!("Layer types pattern (first 10): {:?}", &layer_types[..10.min(layer_types.len())]);

    for (i, layer) in model.backbone.layers.iter_mut().enumerate() {
        let prefix = format!("backbone.layers.{i}");

        // Load input norm
        if let Some(w) = weights.get(&format!("{prefix}.norm.weight")) {
            layer.norm.weight = Param::new(w.clone());
        }

        let block_type = layer_types.get(i).copied().unwrap_or('*');

        match block_type {
            'M' => {
                // Mamba layer
                load_mamba_weights(&mut layer.mixer, weights, &prefix)?;
            }
            '*' => {
                // Attention layer
                load_attention_weights(&mut layer.mixer, weights, &prefix)?;
            }
            '-' => {
                // MLP layer
                load_mlp_weights(&mut layer.mixer, weights, &prefix)?;
            }
            'E' => {
                // MoE layer
                load_moe_weights(&mut layer.mixer, weights, &prefix)?;
            }
            _ => {}
        }
    }

    // Load final norm
    if let Some(w) = weights.get("backbone.norm_f.weight") {
        model.backbone.norm_f.weight = Param::new(w.clone());
    } else if let Some(w) = weights.get("model.norm.weight") {
        model.backbone.norm_f.weight = Param::new(w.clone());
    }

    // Load lm_head if not tied
    tracing::info!("lm_head is Some: {}", model.lm_head.is_some());
    if let Some(ref mut lm_head) = model.lm_head {
        if let Some(w) = weights.get("lm_head.weight") {
            tracing::info!("LOADED lm_head.weight: shape={:?}, dtype={:?}", w.shape(), w.dtype());
            lm_head.weight = Param::new(w.clone());
        } else {
            // Try without prefix
            let lm_head_keys: Vec<_> = weights.keys().filter(|k| k.contains("lm_head")).collect();
            tracing::warn!("lm_head.weight NOT FOUND! Available lm_head keys: {:?}", lm_head_keys);
        }
    } else {
        tracing::info!("lm_head is None (tie_word_embeddings=true), using embedding weights");
    }

    Ok(())
}

fn load_mamba_weights(
    mixer: &mut NemotronHMixer,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), crate::loader::LoadError> {
    tracing::debug!("Loading Mamba weights for {prefix}");

    // in_proj
    if let Some(ref mut in_proj) = mixer.in_proj {
        let key = format!("{prefix}.mixer.in_proj.weight");
        if let Some(w) = weights.get(&key) {
            tracing::debug!("  Loading {}: {:?}", key, w.shape());
            in_proj.weight = Param::new(w.clone());
        } else {
            tracing::debug!("  Key not found: {}", key);
        }
    }
    // FP8 scale factors for in_proj
    if let Some(s) = weights.get(&format!("{prefix}.mixer.in_proj.weight_scale")) {
        mixer.in_proj_weight_scale = Some(s.clone());
    }
    if let Some(s) = weights.get(&format!("{prefix}.mixer.in_proj.input_scale")) {
        mixer.in_proj_input_scale = Some(s.clone());
    }

    // conv1d - transpose weights from PyTorch format [out, in/groups, kernel]
    // to MLX format [out, kernel, in/groups]
    if let Some(ref mut conv1d) = mixer.conv1d {
        if let Some(w) = weights.get(&format!("{prefix}.mixer.conv1d.weight")) {
            // Transpose axes 1 and 2 to convert from PyTorch to MLX format
            let w_transposed = w.transpose_axes(&[0, 2, 1])?;
            conv1d.weight = Param::new(w_transposed);
        }
        if let Some(b) = weights.get(&format!("{prefix}.mixer.conv1d.bias")) {
            conv1d.bias = Param::new(Some(b.clone()));
        }
    }

    // out_proj
    if let Some(ref mut out_proj) = mixer.out_proj {
        if let Some(w) = weights.get(&format!("{prefix}.mixer.out_proj.weight")) {
            out_proj.weight = Param::new(w.clone());
        }
    }
    // FP8 scale factors for out_proj
    if let Some(s) = weights.get(&format!("{prefix}.mixer.out_proj.weight_scale")) {
        mixer.out_proj_weight_scale = Some(s.clone());
    }
    if let Some(s) = weights.get(&format!("{prefix}.mixer.out_proj.input_scale")) {
        mixer.out_proj_input_scale = Some(s.clone());
    }

    // SSM parameters
    if let Some(w) = weights.get(&format!("{prefix}.mixer.A_log")) {
        mixer.a_log = Some(w.clone());
    }
    if let Some(w) = weights.get(&format!("{prefix}.mixer.D")) {
        mixer.d = Some(w.clone());
    }
    if let Some(w) = weights.get(&format!("{prefix}.mixer.dt_bias")) {
        mixer.dt_bias = Some(w.clone());
    }

    // Internal gated norm
    if let Some(ref mut gated_norm) = mixer.gated_norm {
        if let Some(w) = weights.get(&format!("{prefix}.mixer.norm.weight")) {
            gated_norm.weight = w.clone();
        }
    }

    Ok(())
}

fn load_attention_weights(
    mixer: &mut NemotronHMixer,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), crate::loader::LoadError> {
    // q_proj
    if let Some(ref mut q_proj) = mixer.q_proj {
        if let Some(w) = weights.get(&format!("{prefix}.mixer.q_proj.weight")) {
            q_proj.weight = Param::new(w.clone());
        }
    }

    // k_proj
    if let Some(ref mut k_proj) = mixer.k_proj {
        if let Some(w) = weights.get(&format!("{prefix}.mixer.k_proj.weight")) {
            k_proj.weight = Param::new(w.clone());
        }
    }

    // v_proj
    if let Some(ref mut v_proj) = mixer.v_proj {
        if let Some(w) = weights.get(&format!("{prefix}.mixer.v_proj.weight")) {
            v_proj.weight = Param::new(w.clone());
        }
    }

    // o_proj
    if let Some(ref mut o_proj) = mixer.o_proj {
        if let Some(w) = weights.get(&format!("{prefix}.mixer.o_proj.weight")) {
            o_proj.weight = Param::new(w.clone());
        }
    }

    Ok(())
}

fn load_mlp_weights(
    mixer: &mut NemotronHMixer,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), crate::loader::LoadError> {
    // up_proj
    if let Some(ref mut up_proj) = mixer.up_proj {
        if let Some(w) = weights.get(&format!("{prefix}.mixer.up_proj.weight")) {
            up_proj.weight = Param::new(w.clone());
        }
    }

    // down_proj
    if let Some(ref mut down_proj) = mixer.down_proj {
        if let Some(w) = weights.get(&format!("{prefix}.mixer.down_proj.weight")) {
            down_proj.weight = Param::new(w.clone());
        }
    }

    Ok(())
}

fn load_moe_weights(
    mixer: &mut NemotronHMixer,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), crate::loader::LoadError> {
    if let Some(ref mut moe_layer) = mixer.moe_layer {
        // Load router gate weights
        if let Some(w) = weights.get(&format!("{prefix}.mixer.gate.weight")) {
            moe_layer.router.gate.weight = Param::new(w.clone());
        }

        // Load e_score_correction_bias for routing
        if let Some(bias) = weights.get(&format!("{prefix}.mixer.gate.e_score_correction_bias")) {
            moe_layer.router.e_score_correction_bias = bias.clone();
        }

        // Load expert weights with FP8 scale factors
        for (expert_idx, expert) in moe_layer.experts.iter_mut().enumerate() {
            // up_proj for expert
            if let Some(w) = weights.get(&format!("{prefix}.mixer.experts.{expert_idx}.up_proj.weight")) {
                expert.up_proj.weight = Param::new(w.clone());
            }
            if let Some(s) = weights.get(&format!("{prefix}.mixer.experts.{expert_idx}.up_proj.weight_scale")) {
                expert.up_proj_weight_scale = Some(s.clone());
            }
            if let Some(s) = weights.get(&format!("{prefix}.mixer.experts.{expert_idx}.up_proj.input_scale")) {
                expert.up_proj_input_scale = Some(s.clone());
            }
            // down_proj for expert
            if let Some(w) = weights.get(&format!("{prefix}.mixer.experts.{expert_idx}.down_proj.weight")) {
                expert.down_proj.weight = Param::new(w.clone());
            }
            if let Some(s) = weights.get(&format!("{prefix}.mixer.experts.{expert_idx}.down_proj.weight_scale")) {
                expert.down_proj_weight_scale = Some(s.clone());
            }
            if let Some(s) = weights.get(&format!("{prefix}.mixer.experts.{expert_idx}.down_proj.input_scale")) {
                expert.down_proj_input_scale = Some(s.clone());
            }
        }

        // Load shared expert if present (also with FP8 scale factors)
        if let Some(ref mut shared_expert) = moe_layer.shared_expert {
            if let Some(w) = weights.get(&format!("{prefix}.mixer.shared_experts.up_proj.weight")) {
                shared_expert.up_proj.weight = Param::new(w.clone());
            }
            if let Some(s) = weights.get(&format!("{prefix}.mixer.shared_experts.up_proj.weight_scale")) {
                shared_expert.up_proj_weight_scale = Some(s.clone());
            }
            if let Some(s) = weights.get(&format!("{prefix}.mixer.shared_experts.up_proj.input_scale")) {
                shared_expert.up_proj_input_scale = Some(s.clone());
            }
            if let Some(w) = weights.get(&format!("{prefix}.mixer.shared_experts.down_proj.weight")) {
                shared_expert.down_proj.weight = Param::new(w.clone());
            }
            if let Some(s) = weights.get(&format!("{prefix}.mixer.shared_experts.down_proj.weight_scale")) {
                shared_expert.down_proj_weight_scale = Some(s.clone());
            }
            if let Some(s) = weights.get(&format!("{prefix}.mixer.shared_experts.down_proj.input_scale")) {
                shared_expert.down_proj_input_scale = Some(s.clone());
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> NemotronHConfig {
        NemotronHConfig {
            model_type: "nemotron_h".to_string(),
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 4,
            max_position_embeddings: 512,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            attention_bias: false,
            head_dim: Some(32),
            mamba_num_heads: 4,
            mamba_head_dim: 32,
            mamba_proj_bias: false,
            ssm_state_size: 16,
            conv_kernel: 4,
            n_groups: 2,
            time_step_limit: (0.0, f32::INFINITY),
            time_step_min: None,
            time_step_max: None,
            mlp_bias: false,
            mlp_hidden_act: "relu2".to_string(),
            layer_norm_epsilon: 1e-5,
            use_bias: false,
            use_conv_bias: true,
            tie_word_embeddings: true,
            hybrid_override_pattern: Some("M*-E".to_string()),
            moe_intermediate_size: None,
            moe_shared_expert_intermediate_size: None,
            n_group: None,
            n_routed_experts: None,
            n_shared_experts: None,
            topk_group: None,
            num_experts_per_tok: None,
            norm_topk_prob: None,
            routed_scaling_factor: None,
            rope_theta: 10000.0,
        }
    }

    #[test]
    fn test_config_layer_types() {
        let config = small_config();
        let types = config.layer_types();
        assert_eq!(types, vec!['M', '*', '-', 'E']);
    }

    #[test]
    fn test_config_attention_head_dim() {
        let config = small_config();
        assert_eq!(config.attention_head_dim(), 32);
    }
}
