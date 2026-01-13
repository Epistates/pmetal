//! Backward Pass Implementations for Transformer Components
//!
//! This module provides explicit backward pass implementations for all
//! operations in a transformer model, enabling custom autograd training
//! that bypasses MLX's autodiff for ~50% memory reduction.
//!
//! # Components
//!
//! - `RMSNormBackward` - RMS normalization backward pass
//! - `SiLUBackward` - SiLU/Swish activation backward
//! - `RoPEBackward` - Rotary position embedding backward
//! - `AttentionBackward` - Full attention mechanism backward
//! - `TransformerLayerBackward` - Complete layer backward combining all above

use mlx_rs::error::Exception;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype};

use crate::LoraError;

// =============================================================================
// RMSNorm Backward
// =============================================================================

/// Saved state for RMSNorm backward.
#[derive(Debug, Clone)]
pub struct RmsNormSaved {
    /// Input tensor.
    pub x: Array,
    /// Normalized output (before scale).
    pub x_norm: Array,
    /// RMS values (for gradient computation).
    pub rms: Array,
    /// Scale weight.
    pub weight: Array,
    /// Epsilon used.
    pub eps: f32,
}

/// Compute RMSNorm forward with state saving for backward.
///
/// RMSNorm: y = x / rms(x) * weight
/// where rms(x) = sqrt(mean(x^2) + eps)
pub fn rmsnorm_forward_with_grad(
    x: &Array,
    weight: &Array,
    eps: f32,
) -> Result<(Array, RmsNormSaved), LoraError> {
    // Compute x^2
    let x_sq = x.multiply(x)?;

    // Mean over last axis
    let mean_sq = x_sq.mean_axis(-1, true)?;

    // RMS = sqrt(mean(x^2) + eps)
    let rms = mean_sq.add(&Array::from_f32(eps))?.sqrt()?;

    // Normalize: x / rms
    let x_norm = x.divide(&rms)?;

    // Scale by weight
    let output = x_norm.multiply(weight)?;

    let saved = RmsNormSaved {
        x: x.clone(),
        x_norm: x_norm.clone(),
        rms,
        weight: weight.clone(),
        eps,
    };

    Ok((output, saved))
}

/// Compute RMSNorm backward.
///
/// Given d_output (gradient w.r.t. output), computes d_x (gradient w.r.t. input).
///
/// The gradient is:
/// d_x = weight / rms * (d_y - x_norm * mean(d_y * x_norm))
pub fn rmsnorm_backward(d_output: &Array, saved: &RmsNormSaved) -> Result<Array, LoraError> {
    let hidden_size = saved.x.dim(-1) as f32;

    // d_y * weight (since output = x_norm * weight)
    let d_x_norm = d_output.multiply(&saved.weight)?;

    // Compute mean(d_x_norm * x_norm) over last axis
    let dot = d_x_norm.multiply(&saved.x_norm)?;
    let mean_dot = dot.mean_axis(-1, true)?;

    // d_x = (d_x_norm - x_norm * mean_dot) / rms
    let correction = saved.x_norm.multiply(&mean_dot)?;
    let d_x = d_x_norm.subtract(&correction)?.divide(&saved.rms)?;

    Ok(d_x)
}

// =============================================================================
// SiLU (Swish) Backward
// =============================================================================

/// Saved state for SiLU backward.
#[derive(Debug, Clone)]
pub struct SiluSaved {
    /// Input tensor.
    pub x: Array,
    /// Sigmoid(x) - saved to avoid recomputation.
    pub sigmoid_x: Array,
}

/// Compute SiLU forward with state saving.
///
/// SiLU(x) = x * sigmoid(x)
pub fn silu_forward_with_grad(x: &Array) -> Result<(Array, SiluSaved), LoraError> {
    // sigmoid(x) = 1 / (1 + exp(-x))
    let neg_x = x.negative()?;
    let exp_neg_x = neg_x.exp()?;
    let sigmoid_x = Array::from_f32(1.0).divide(&exp_neg_x.add(&Array::from_f32(1.0))?)?;

    // output = x * sigmoid(x)
    let output = x.multiply(&sigmoid_x)?;

    let saved = SiluSaved {
        x: x.clone(),
        sigmoid_x,
    };

    Ok((output, saved))
}

/// Compute SiLU backward.
///
/// d_silu/d_x = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
///            = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
pub fn silu_backward(d_output: &Array, saved: &SiluSaved) -> Result<Array, LoraError> {
    // 1 - sigmoid(x)
    let one_minus_sigmoid = Array::from_f32(1.0).subtract(&saved.sigmoid_x)?;

    // x * (1 - sigmoid(x))
    let x_times_deriv = saved.x.multiply(&one_minus_sigmoid)?;

    // 1 + x * (1 - sigmoid(x))
    let factor = x_times_deriv.add(&Array::from_f32(1.0))?;

    // sigmoid(x) * factor
    let d_silu = saved.sigmoid_x.multiply(&factor)?;

    // d_x = d_output * d_silu
    Ok(d_output.multiply(&d_silu)?)
}

// =============================================================================
// RoPE Backward
// =============================================================================

/// Saved state for RoPE backward.
#[derive(Debug, Clone)]
pub struct RopeSaved {
    /// Cosine values [seq_len, head_dim/2].
    pub cos: Array,
    /// Sine values [seq_len, head_dim/2].
    pub sin: Array,
}

/// Compute RoPE forward with state saving.
///
/// RoPE rotates pairs of dimensions:
/// [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
pub fn rope_forward_with_grad(
    x: &Array,
    cos: &Array,
    sin: &Array,
) -> Result<(Array, RopeSaved), LoraError> {
    let shape = x.shape();
    let head_dim = shape[shape.len() - 1];
    let half_dim = head_dim / 2;

    // Split into first half and second half
    let x1 = x.index((.., .., .., ..half_dim));
    let x2 = x.index((.., .., .., half_dim..));

    // Rotate: [x1*cos - x2*sin, x1*sin + x2*cos]
    let out1 = x1.multiply(cos)?.subtract(&x2.multiply(sin)?)?;
    let out2 = x1.multiply(sin)?.add(&x2.multiply(cos)?)?;

    // Concatenate along last dimension
    let output = mlx_rs::ops::concatenate_axis(&[&out1, &out2], -1)?;

    let saved = RopeSaved {
        cos: cos.clone(),
        sin: sin.clone(),
    };

    Ok((output, saved))
}

/// Compute RoPE backward.
///
/// The rotation is orthogonal, so backward is inverse rotation:
/// d_x = rotate_backward(d_output) = [d1*cos + d2*sin, -d1*sin + d2*cos]
pub fn rope_backward(d_output: &Array, saved: &RopeSaved) -> Result<Array, LoraError> {
    let shape = d_output.shape();
    let head_dim = shape[shape.len() - 1];
    let half_dim = head_dim / 2;

    // Split gradient
    let d1 = d_output.index((.., .., .., ..half_dim));
    let d2 = d_output.index((.., .., .., half_dim..));

    // Inverse rotation: [d1*cos + d2*sin, -d1*sin + d2*cos]
    let dx1 = d1.multiply(&saved.cos)?.add(&d2.multiply(&saved.sin)?)?;
    let dx2 = d2
        .multiply(&saved.cos)?
        .subtract(&d1.multiply(&saved.sin)?)?;

    // Concatenate
    Ok(mlx_rs::ops::concatenate_axis(&[&dx1, &dx2], -1)?)
}

// =============================================================================
// Attention Backward
// =============================================================================

/// Saved state for attention backward.
#[derive(Debug, Clone)]
pub struct AttentionSaved {
    /// Query tensor [batch, heads, seq, head_dim].
    pub q: Array,
    /// Key tensor [batch, kv_heads, seq, head_dim].
    pub k: Array,
    /// Value tensor [batch, kv_heads, seq, head_dim].
    pub v: Array,
    /// Attention weights [batch, heads, seq, seq].
    pub attn_weights: Array,
    /// Scale factor.
    pub scale: f32,
    /// Number of query heads per KV head (for GQA).
    pub num_heads_per_kv: i32,
}

/// Compute scaled dot-product attention forward with state saving.
///
/// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
pub fn attention_forward_with_grad(
    q: &Array,
    k: &Array,
    v: &Array,
    mask: Option<&Array>,
    scale: f32,
    num_heads_per_kv: i32,
) -> Result<(Array, AttentionSaved), LoraError> {
    // Expand K and V for GQA if needed
    let (k_expanded, v_expanded) = if num_heads_per_kv > 1 {
        let k_exp = expand_kv_heads(k, num_heads_per_kv)?;
        let v_exp = expand_kv_heads(v, num_heads_per_kv)?;
        (k_exp, v_exp)
    } else {
        (k.clone(), v.clone())
    };

    // Q @ K^T
    let scores = q.matmul(&k_expanded.transpose_axes(&[0, 1, 3, 2])?)?;

    // Scale
    let scores = scores.multiply(Array::from_f32(scale))?;

    // Apply mask
    let scores = if let Some(m) = mask {
        scores.add(m)?
    } else {
        scores
    };

    // Softmax
    let attn_weights = mlx_rs::ops::softmax_axis(&scores, -1, None)?;

    // Attention output: weights @ V
    let output = attn_weights.matmul(&v_expanded)?;

    let saved = AttentionSaved {
        q: q.clone(),
        k: k.clone(),
        v: v.clone(),
        attn_weights,
        scale,
        num_heads_per_kv,
    };

    Ok((output, saved))
}

/// Compute attention backward.
///
/// Returns (d_q, d_k, d_v) gradients.
pub fn attention_backward(
    d_output: &Array,
    saved: &AttentionSaved,
) -> Result<(Array, Array, Array), LoraError> {
    // Expand K and V for GQA if needed
    let (k_expanded, v_expanded) = if saved.num_heads_per_kv > 1 {
        let k_exp = expand_kv_heads(&saved.k, saved.num_heads_per_kv)?;
        let v_exp = expand_kv_heads(&saved.v, saved.num_heads_per_kv)?;
        (k_exp, v_exp)
    } else {
        (saved.k.clone(), saved.v.clone())
    };

    // d_V = weights^T @ d_output
    // [batch, heads, seq, seq]^T @ [batch, heads, seq, head_dim]
    // = [batch, heads, seq, head_dim]
    let d_v_expanded = saved
        .attn_weights
        .transpose_axes(&[0, 1, 3, 2])?
        .matmul(d_output)?;

    // d_weights = d_output @ V^T
    // [batch, heads, seq, head_dim] @ [batch, heads, head_dim, seq]
    // = [batch, heads, seq, seq]
    let d_weights = d_output.matmul(&v_expanded.transpose_axes(&[0, 1, 3, 2])?)?;

    // Softmax backward: d_scores = weights * (d_weights - sum(d_weights * weights, axis=-1, keepdims=True))
    let weighted_d = d_weights.multiply(&saved.attn_weights)?;
    let sum_weighted = weighted_d.sum_axis(-1, true)?;
    let d_scores = saved
        .attn_weights
        .multiply(&d_weights.subtract(&sum_weighted)?)?;

    // Scale backward
    let d_scores = d_scores.multiply(Array::from_f32(saved.scale))?;

    // Q @ K^T backward:
    // d_Q = d_scores @ K
    // d_K = d_scores^T @ Q = Q^T @ d_scores (then transpose)
    let d_q = d_scores.matmul(&k_expanded)?;
    let d_k_expanded = saved
        .q
        .transpose_axes(&[0, 1, 3, 2])?
        .matmul(&d_scores)?
        .transpose_axes(&[0, 1, 3, 2])?;

    // Contract GQA gradients if needed
    let (d_k, d_v) = if saved.num_heads_per_kv > 1 {
        let d_k = contract_kv_grads(&d_k_expanded, saved.num_heads_per_kv)?;
        let d_v = contract_kv_grads(&d_v_expanded, saved.num_heads_per_kv)?;
        (d_k, d_v)
    } else {
        (d_k_expanded, d_v_expanded)
    };

    Ok((d_q, d_k, d_v))
}

/// Expand KV heads for GQA.
fn expand_kv_heads(x: &Array, repeats: i32) -> Result<Array, Exception> {
    let shape = x.shape();
    let batch = shape[0];
    let n_kv_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    let x = x.reshape(&[batch, n_kv_heads, 1, seq_len, head_dim])?;
    let x = mlx_rs::ops::broadcast_to(&x, &[batch, n_kv_heads, repeats, seq_len, head_dim])?;
    x.reshape(&[batch, n_kv_heads * repeats, seq_len, head_dim])
}

/// Contract KV gradients after GQA backward (sum over repeated heads).
fn contract_kv_grads(d_expanded: &Array, repeats: i32) -> Result<Array, LoraError> {
    let shape = d_expanded.shape();
    let batch = shape[0];
    let n_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];
    let n_kv_heads = n_heads / repeats;

    // Reshape to [batch, n_kv_heads, repeats, seq_len, head_dim]
    let d_reshaped = d_expanded.reshape(&[batch, n_kv_heads, repeats, seq_len, head_dim])?;

    // Sum over repeats dimension
    let d_contracted = d_reshaped.sum_axis(2, false)?;

    Ok(d_contracted)
}

// =============================================================================
// Full Decoder Layer State
// =============================================================================

/// Complete saved state for a transformer decoder layer.
#[derive(Debug)]
pub struct DecoderLayerSaved {
    /// Input to the layer.
    pub x: Array,
    /// After input norm.
    pub x_normed: Array,
    /// Input norm saved state.
    pub input_norm_saved: RmsNormSaved,
    /// After attention (before residual).
    pub attn_out: Array,
    /// Attention saved state (includes Q, K, V, weights).
    pub attention_saved: AttentionSaved,
    /// After first residual (h = x + attn_out).
    pub h: Array,
    /// After post-attention norm.
    pub h_normed: Array,
    /// Post-attention norm saved state.
    pub post_attn_norm_saved: RmsNormSaved,
    /// Gate output (before SiLU).
    pub gate_pre_silu: Array,
    /// SiLU saved state.
    pub silu_saved: SiluSaved,
    /// Up projection output.
    pub up_out: Array,
    /// After gate * up.
    pub hidden: Array,
    /// LoRA saved states for each projection.
    pub q_proj_saved: crate::autograd::LoraForwardSaved,
    pub k_proj_saved: crate::autograd::LoraForwardSaved,
    pub v_proj_saved: crate::autograd::LoraForwardSaved,
    pub o_proj_saved: crate::autograd::LoraForwardSaved,
    pub gate_proj_saved: crate::autograd::LoraForwardSaved,
    pub up_proj_saved: crate::autograd::LoraForwardSaved,
    pub down_proj_saved: crate::autograd::LoraForwardSaved,
    /// RoPE saved states for Q and K.
    pub q_rope_saved: RopeSaved,
    pub k_rope_saved: RopeSaved,
}

/// Gradients computed for a decoder layer.
#[derive(Debug)]
pub struct DecoderLayerGrads {
    /// Gradient for input (for chain rule to previous layer).
    pub d_x: Array,
    /// Q projection LoRA gradients.
    pub q_proj_grads: crate::autograd::LoraGrads,
    /// K projection LoRA gradients.
    pub k_proj_grads: crate::autograd::LoraGrads,
    /// V projection LoRA gradients.
    pub v_proj_grads: crate::autograd::LoraGrads,
    /// O projection LoRA gradients.
    pub o_proj_grads: crate::autograd::LoraGrads,
    /// Gate projection LoRA gradients.
    pub gate_proj_grads: crate::autograd::LoraGrads,
    /// Up projection LoRA gradients.
    pub up_proj_grads: crate::autograd::LoraGrads,
    /// Down projection LoRA gradients.
    pub down_proj_grads: crate::autograd::LoraGrads,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_forward_backward() {
        let x = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();

        let (output, saved) = silu_forward_with_grad(&x).unwrap();

        // Check output shape
        assert_eq!(output.shape(), x.shape());

        // Test backward
        let d_output = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();
        let d_x = silu_backward(&d_output, &saved).unwrap();

        assert_eq!(d_x.shape(), x.shape());
    }

    #[test]
    fn test_rmsnorm_forward_backward() {
        let x = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();
        let weight = mlx_rs::Array::ones::<f32>(&[64]).unwrap();

        let (output, saved) = rmsnorm_forward_with_grad(&x, &weight, 1e-5).unwrap();
        assert_eq!(output.shape(), x.shape());

        let d_output = mlx_rs::random::normal::<f32>(&[2, 4, 64], None, None, None).unwrap();
        let d_x = rmsnorm_backward(&d_output, &saved).unwrap();

        assert_eq!(d_x.shape(), x.shape());
    }

    #[test]
    fn test_attention_forward_backward() {
        let batch = 2;
        let heads = 4;
        let kv_heads = 2;
        let seq_len = 8;
        let head_dim = 16;

        let q = mlx_rs::random::normal::<f32>(&[batch, heads, seq_len, head_dim], None, None, None)
            .unwrap();
        let k =
            mlx_rs::random::normal::<f32>(&[batch, kv_heads, seq_len, head_dim], None, None, None)
                .unwrap();
        let v =
            mlx_rs::random::normal::<f32>(&[batch, kv_heads, seq_len, head_dim], None, None, None)
                .unwrap();

        let scale = (head_dim as f32).sqrt().recip();
        let num_heads_per_kv = heads / kv_heads;

        let (output, saved) =
            attention_forward_with_grad(&q, &k, &v, None, scale, num_heads_per_kv).unwrap();
        assert_eq!(output.shape(), &[batch, heads, seq_len, head_dim]);

        let d_output =
            mlx_rs::random::normal::<f32>(&[batch, heads, seq_len, head_dim], None, None, None)
                .unwrap();
        let (d_q, d_k, d_v) = attention_backward(&d_output, &saved).unwrap();

        assert_eq!(d_q.shape(), q.shape());
        assert_eq!(d_k.shape(), k.shape());
        assert_eq!(d_v.shape(), v.shape());
    }
}
