//! Custom LoRA Autograd with In-Place Gradients
//!
//! This module implements unsloth-style custom autograd for LoRA training.
//! Instead of relying on framework autodiff (which saves all intermediate activations),
//! we compute gradients explicitly, saving only what's needed:
//!
//! - `x`: Input tensor (for dA computation)
//! - `x @ A^T`: Intermediate (for dB computation)
//!
//! # Benefits
//!
//! - ~50% memory reduction vs standard autodiff
//! - Enables larger batch sizes
//! - Works with our fused Metal kernels
//!
//! # Algorithm
//!
//! For LoRA forward: `y = x @ W^T + scale * (x @ A^T) @ B^T`
//!
//! Backward:
//! - `dB = scale * (x @ A)^T @ dY`
//! - `dA = scale * x^T @ (dY @ B^T)`
//! - `dX = dY @ W + scale * (dY @ B) @ A`
//!
//! # Usage
//!
//! ```ignore
//! let ctx = LoraGradContext::new();
//!
//! // Forward pass
//! let (output, saved) = lora_forward_with_grad(&x, &weight, &lora_a, &lora_b, scale, &ctx)?;
//!
//! // ... compute loss and upstream gradient dY ...
//!
//! // Backward pass
//! let grads = lora_backward(&dY, &saved)?;
//! ```

use mlx_rs::{nn, Array};
use std::collections::HashMap;

/// Context for tracking tensors needed in backward pass.
#[derive(Debug, Clone)]
pub struct LoraGradContext {
    /// Whether to compute input gradients (for chain rule).
    pub compute_input_grad: bool,
}

impl Default for LoraGradContext {
    fn default() -> Self {
        Self {
            compute_input_grad: true,
        }
    }
}

impl LoraGradContext {
    /// Create a new gradient context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Skip input gradient computation (for first layer or when not needed).
    pub fn without_input_grad(mut self) -> Self {
        self.compute_input_grad = false;
        self
    }
}

/// Saved tensors from forward pass needed for backward.
#[derive(Debug)]
pub struct LoraForwardSaved {
    /// Input tensor [batch, in_features].
    pub x: Array,
    /// Intermediate x @ A^T [batch, rank].
    pub x_a: Array,
    /// Base weight (reference for backward).
    pub weight: Array,
    /// LoRA A matrix.
    pub lora_a: Array,
    /// LoRA B matrix.
    pub lora_b: Array,
    /// LoRA scale.
    pub scale: f32,
    /// Whether to compute input gradient.
    pub compute_input_grad: bool,
}

/// Gradients computed during backward pass.
#[derive(Debug)]
pub struct LoraGrads {
    /// Gradient for lora_a [rank, in_features].
    pub d_lora_a: Array,
    /// Gradient for lora_b [out_features, rank].
    pub d_lora_b: Array,
    /// Gradient for input [batch, in_features] (for chain rule).
    /// None if compute_input_grad was false.
    pub d_x: Option<Array>,
}

/// Forward pass for LoRA with gradient context.
///
/// Computes: `y = x @ W^T + scale * (x @ A^T) @ B^T`
///
/// Saves minimal state needed for backward:
/// - x: for computing dA
/// - x @ A^T: for computing dB
///
/// # Arguments
///
/// * `x` - Input tensor [..., in_features] (supports 2D or 3D)
/// * `weight` - Base weight [out_features, in_features]
/// * `lora_a` - LoRA A matrix [rank, in_features]
/// * `lora_b` - LoRA B matrix [out_features, rank]
/// * `scale` - LoRA scale (alpha / rank)
/// * `ctx` - Gradient context
///
/// # Returns
///
/// Tuple of (output, saved tensors for backward)
pub fn lora_forward_with_grad(
    x: &Array,
    weight: &Array,
    lora_a: &Array,
    lora_b: &Array,
    scale: f32,
    ctx: &LoraGradContext,
) -> Result<(Array, LoraForwardSaved), mlx_rs::error::Exception> {
    // MLX matmul handles arbitrary batch dimensions:
    // [..., in_features] @ [in_features, rank] = [..., rank]
    let x_a = x.matmul(&lora_a.t())?;

    // LoRA contribution: (x @ A^T) @ B^T
    // [..., rank] @ [rank, out_features] = [..., out_features]
    let lora_out = x_a.matmul(&lora_b.t())?;
    let scaled_lora = lora_out.multiply(Array::from_f32(scale))?;

    // Base output: x @ W^T
    let base_out = x.matmul(&weight.t())?;

    // Final output
    let output = base_out.add(&scaled_lora)?;

    // Save for backward
    let saved = LoraForwardSaved {
        x: x.clone(),
        x_a,
        weight: weight.clone(),
        lora_a: lora_a.clone(),
        lora_b: lora_b.clone(),
        scale,
        compute_input_grad: ctx.compute_input_grad,
    };

    Ok((output, saved))
}

/// Backward pass for LoRA.
///
/// Computes gradients for LoRA parameters (and optionally input).
/// Handles arbitrary batch dimensions by flattening before gradient computation.
///
/// # Arguments
///
/// * `d_output` - Upstream gradient [..., out_features]
/// * `saved` - Saved tensors from forward pass
///
/// # Returns
///
/// LoRA gradients (dA, dB, and optionally dX)
pub fn lora_backward(
    d_output: &Array,
    saved: &LoraForwardSaved,
) -> Result<LoraGrads, mlx_rs::error::Exception> {
    let scale = saved.scale;
    let shape = saved.x.shape();
    let ndim = shape.len();
    let in_features = shape[ndim - 1];
    let rank = saved.lora_a.dim(0);
    let out_features = saved.lora_b.dim(0);

    // For 3D inputs [batch, seq_len, features], we need to flatten to 2D for gradient computation
    // then reshape results appropriately.
    //
    // x: [..., in_features] -> [N, in_features] where N = product of batch dims
    // x_a: [..., rank] -> [N, rank]
    // d_output: [..., out_features] -> [N, out_features]

    let (x_flat, x_a_flat, d_output_flat, original_shape) = if ndim > 2 {
        // Flatten batch dimensions
        let batch_size: i32 = shape[..ndim - 1].iter().product();
        let x_flat = saved.x.reshape(&[batch_size, in_features])?;
        let x_a_flat = saved.x_a.reshape(&[batch_size, rank])?;
        let d_output_flat = d_output.reshape(&[batch_size, out_features])?;
        (x_flat, x_a_flat, d_output_flat, Some(shape.to_vec()))
    } else {
        // Already 2D
        (saved.x.clone(), saved.x_a.clone(), d_output.clone(), None)
    };

    // dB = scale * x_a^T @ dY
    // Shape: [rank, N] @ [N, out_features] = [rank, out_features]
    // Transpose to [out_features, rank]
    let d_lora_b = x_a_flat
        .t()
        .matmul(&d_output_flat)?
        .multiply(Array::from_f32(scale))?
        .t();

    // dY @ B: [N, out_features] @ [out_features, rank] = [N, rank]
    let dy_b_flat = d_output_flat.matmul(&saved.lora_b)?;

    // dA = scale * x^T @ (dY @ B)
    // Shape: [in_features, N] @ [N, rank] = [in_features, rank]
    // Transpose to [rank, in_features]
    let d_lora_a = x_flat
        .t()
        .matmul(&dy_b_flat)?
        .multiply(Array::from_f32(scale))?
        .t();

    // Optionally compute input gradient for chain rule
    let d_x = if saved.compute_input_grad {
        // dX = dY @ W + scale * (dY @ B) @ A
        // dY @ W: [N, out_features] @ [out_features, in_features] = [N, in_features]
        let dx_base_flat = d_output_flat.matmul(&saved.weight)?;

        // (dY @ B) @ A: [N, rank] @ [rank, in_features] = [N, in_features]
        let dx_lora_flat = dy_b_flat
            .matmul(&saved.lora_a)?
            .multiply(Array::from_f32(scale))?;

        let dx_flat = dx_base_flat.add(&dx_lora_flat)?;

        // Reshape back to original batch shape if needed
        if let Some(orig_shape) = original_shape {
            Some(dx_flat.reshape(&orig_shape.iter().map(|&x| x as i32).collect::<Vec<_>>())?)
        } else {
            Some(dx_flat)
        }
    } else {
        None
    };

    Ok(LoraGrads {
        d_lora_a,
        d_lora_b,
        d_x,
    })
}

/// Saved state for fused MLP forward pass (gate + up + down + SwiGLU).
#[derive(Debug)]
pub struct MlpForwardSaved {
    /// Input tensor x [batch, hidden_size].
    pub x: Array,
    /// Gate output before SiLU: gate_proj(x) [batch, intermediate].
    pub gate_pre_silu: Array,
    /// Gate output after SiLU: silu(gate_proj(x)) [batch, intermediate].
    pub gate_activated: Array,
    /// Up output: up_proj(x) [batch, intermediate].
    pub up_out: Array,
    /// MLP hidden: gate_activated * up_out [batch, intermediate].
    pub mlp_hidden: Array,
    /// Gate LoRA A [rank, hidden_size].
    pub gate_lora_a: Array,
    /// Gate LoRA B [intermediate, rank].
    pub gate_lora_b: Array,
    /// Gate scale.
    pub gate_scale: f32,
    /// Gate base weight [intermediate, hidden_size].
    pub gate_weight: Array,
    /// Up LoRA A [rank, hidden_size].
    pub up_lora_a: Array,
    /// Up LoRA B [intermediate, rank].
    pub up_lora_b: Array,
    /// Up scale.
    pub up_scale: f32,
    /// Up base weight [intermediate, hidden_size].
    pub up_weight: Array,
    /// Down LoRA A [rank, intermediate].
    pub down_lora_a: Array,
    /// Down LoRA B [hidden_size, rank].
    pub down_lora_b: Array,
    /// Down scale.
    pub down_scale: f32,
    /// Down base weight [hidden_size, intermediate].
    pub down_weight: Array,
}

/// Gradients for all MLP LoRA parameters.
#[derive(Debug)]
pub struct MlpLoraGrads {
    /// Gate LoRA A gradient.
    pub gate_d_lora_a: Array,
    /// Gate LoRA B gradient.
    pub gate_d_lora_b: Array,
    /// Up LoRA A gradient.
    pub up_d_lora_a: Array,
    /// Up LoRA B gradient.
    pub up_d_lora_b: Array,
    /// Down LoRA A gradient.
    pub down_d_lora_a: Array,
    /// Down LoRA B gradient.
    pub down_d_lora_b: Array,
    /// Input gradient for chain rule.
    pub d_x: Array,
}

/// Fused MLP forward pass with saved state for backward.
///
/// Computes the full MLP forward:
/// ```text
/// h = silu(gate_proj(x)) * up_proj(x)
/// output = down_proj(h)
/// ```
///
/// Saves minimal state needed for fused backward pass.
pub fn fused_mlp_forward(
    x: &Array,
    gate_weight: &Array,
    gate_lora_a: &Array,
    gate_lora_b: &Array,
    gate_scale: f32,
    up_weight: &Array,
    up_lora_a: &Array,
    up_lora_b: &Array,
    up_scale: f32,
    down_weight: &Array,
    down_lora_a: &Array,
    down_lora_b: &Array,
    down_scale: f32,
) -> Result<(Array, MlpForwardSaved), mlx_rs::error::Exception> {
    // Gate projection with LoRA
    let gate_base = x.matmul(&gate_weight.t())?;
    let gate_lora = x
        .matmul(&gate_lora_a.t())?
        .matmul(&gate_lora_b.t())?
        .multiply(Array::from_f32(gate_scale))?;
    let gate_pre_silu = gate_base.add(&gate_lora)?;

    // SiLU activation
    let gate_activated = nn::silu(&gate_pre_silu)?;

    // Up projection with LoRA
    let up_base = x.matmul(&up_weight.t())?;
    let up_lora = x
        .matmul(&up_lora_a.t())?
        .matmul(&up_lora_b.t())?
        .multiply(Array::from_f32(up_scale))?;
    let up_out = up_base.add(&up_lora)?;

    // Element-wise multiply (SwiGLU)
    let mlp_hidden = gate_activated.multiply(&up_out)?;

    // Down projection with LoRA
    let down_base = mlp_hidden.matmul(&down_weight.t())?;
    let down_lora = mlp_hidden
        .matmul(&down_lora_a.t())?
        .matmul(&down_lora_b.t())?
        .multiply(Array::from_f32(down_scale))?;
    let output = down_base.add(&down_lora)?;

    let saved = MlpForwardSaved {
        x: x.clone(),
        gate_pre_silu,
        gate_activated,
        up_out,
        mlp_hidden,
        gate_lora_a: gate_lora_a.clone(),
        gate_lora_b: gate_lora_b.clone(),
        gate_scale,
        gate_weight: gate_weight.clone(),
        up_lora_a: up_lora_a.clone(),
        up_lora_b: up_lora_b.clone(),
        up_scale,
        up_weight: up_weight.clone(),
        down_lora_a: down_lora_a.clone(),
        down_lora_b: down_lora_b.clone(),
        down_scale,
        down_weight: down_weight.clone(),
    };

    Ok((output, saved))
}

/// Fused MLP backward pass (Unsloth-style optimization).
///
/// This computes gradients for all three MLP projections (gate, up, down)
/// in a single pass, minimizing intermediate tensor allocations.
///
/// # MLP Forward (SwiGLU):
/// ```text
/// h = silu(gate_proj(x)) * up_proj(x)
/// output = down_proj(h)
/// ```
///
/// # Backward chain rule:
/// ```text
/// dW_down = h.T @ dY
/// D = dY @ W_down.T  (upstream gradient to MLP hidden)
///
/// f = silu(e)  where e = gate_proj(x)
/// g = up_proj(x)
/// df/de = sigmoid(e) * (1 + e * (1 - sigmoid(e))) = sigmoid(e) + f * (1 - sigmoid(e))
///
/// d_up = D * f  (gradient for up path)
/// d_gate = D * g * df/de  (gradient for gate path)
///
/// dX = d_up @ W_up.T + d_gate @ W_gate.T (accumulate both paths)
/// ```
///
/// # Arguments
///
/// * `d_output` - Upstream gradient [batch, hidden_size]
/// * `saved` - Saved tensors from forward pass
///
/// # Returns
///
/// All MLP LoRA gradients and input gradient
pub fn fused_mlp_backward(
    d_output: &Array,
    saved: &MlpForwardSaved,
) -> Result<MlpLoraGrads, mlx_rs::error::Exception> {
    let shape = saved.x.shape();
    let ndim = shape.len();
    let hidden_size = shape[ndim - 1];
    let intermediate_size = saved.gate_activated.dim((ndim - 1) as i32);

    // Flatten for gradient computation if needed
    let (
        x_flat,
        gate_pre_silu_flat,
        gate_activated_flat,
        up_out_flat,
        mlp_hidden_flat,
        d_output_flat,
    ) = if ndim > 2 {
        let batch_size: i32 = shape[..ndim - 1].iter().product();
        (
            saved.x.reshape(&[batch_size, hidden_size])?,
            saved
                .gate_pre_silu
                .reshape(&[batch_size, intermediate_size])?,
            saved
                .gate_activated
                .reshape(&[batch_size, intermediate_size])?,
            saved.up_out.reshape(&[batch_size, intermediate_size])?,
            saved.mlp_hidden.reshape(&[batch_size, intermediate_size])?,
            d_output.reshape(&[batch_size, hidden_size])?,
        )
    } else {
        (
            saved.x.clone(),
            saved.gate_pre_silu.clone(),
            saved.gate_activated.clone(),
            saved.up_out.clone(),
            saved.mlp_hidden.clone(),
            d_output.clone(),
        )
    };

    // ============================================
    // 1. DOWN PROJECTION BACKWARD
    // ============================================
    // For down projection: output = h @ W^T + scale * (h @ A^T) @ B^T
    // where h = mlp_hidden, W = down_weight, A = down_lora_a, B = down_lora_b
    //
    // Gradient w.r.t. input h:
    // d_h = d_output @ W + scale * (d_output @ B) @ A
    let d_mlp_hidden = d_output_flat.matmul(&saved.down_weight)?;
    let d_h_a_down = d_output_flat.matmul(&saved.down_lora_b)?; // [batch, rank]
    let d_mlp_hidden_lora = d_h_a_down
        .matmul(&saved.down_lora_a)?
        .multiply(Array::from_f32(saved.down_scale))?;
    let d_mlp_hidden = d_mlp_hidden.add(&d_mlp_hidden_lora)?;

    // Down LoRA gradients:
    // h_a = h @ A^T = [batch, rank]
    // dB = d_output^T @ h_a = [hidden, batch] @ [batch, rank] = [hidden, rank]
    // dA = d_h_a^T @ h = [rank, batch] @ [batch, intermediate] = [rank, intermediate]
    let h_a_down = mlp_hidden_flat.matmul(&saved.down_lora_a.t())?; // [batch, rank]
    let down_d_lora_b = d_output_flat
        .t()
        .matmul(&h_a_down)?
        .multiply(Array::from_f32(saved.down_scale))?;

    let down_d_lora_a = d_h_a_down
        .t()
        .matmul(&mlp_hidden_flat)?
        .multiply(Array::from_f32(saved.down_scale))?;

    // ============================================
    // 2. SWIGLU BACKWARD
    // ============================================
    // mlp_hidden = gate_activated * up_out
    // d_gate_activated = D * up_out
    // d_up_out = D * gate_activated
    let d_gate_activated = d_mlp_hidden.multiply(&up_out_flat)?;
    let d_up_out = d_mlp_hidden.multiply(&gate_activated_flat)?;

    // SiLU backward: f = x * sigmoid(x), df/dx = sigmoid(x) + f * (1 - sigmoid(x))
    // = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    let sigmoid_e = mlx_rs::ops::sigmoid(&gate_pre_silu_flat)?;
    let one = Array::from_f32(1.0);
    let one_minus_sigmoid = one.subtract(&sigmoid_e)?;
    let silu_deriv = sigmoid_e.add(&gate_activated_flat.multiply(&one_minus_sigmoid)?)?;
    let d_gate_pre_silu = d_gate_activated.multiply(&silu_deriv)?;

    // ============================================
    // 3. UP PROJECTION BACKWARD
    // ============================================
    // For up projection: output = x @ W^T + scale * (x @ A^T) @ B^T
    // x_a = x @ A^T = [batch, rank]
    // dB = d_up^T @ x_a = [intermediate, batch] @ [batch, rank] = [intermediate, rank]
    // d_x_a = d_up @ B = [batch, intermediate] @ [intermediate, rank] = [batch, rank]
    // dA = d_x_a^T @ x = [rank, batch] @ [batch, hidden] = [rank, hidden]
    let x_a_up = x_flat.matmul(&saved.up_lora_a.t())?; // [batch, rank]
    let up_d_lora_b = d_up_out
        .t()
        .matmul(&x_a_up)?
        .multiply(Array::from_f32(saved.up_scale))?;

    let d_x_a_up = d_up_out.matmul(&saved.up_lora_b)?; // [batch, rank]
    let up_d_lora_a = d_x_a_up
        .t()
        .matmul(&x_flat)?
        .multiply(Array::from_f32(saved.up_scale))?;

    // ============================================
    // 4. GATE PROJECTION BACKWARD
    // ============================================
    // For gate projection: same structure as up projection
    // x_a = x @ A^T = [batch, rank]
    // dB = d_gate^T @ x_a = [intermediate, batch] @ [batch, rank] = [intermediate, rank]
    // d_x_a = d_gate @ B = [batch, intermediate] @ [intermediate, rank] = [batch, rank]
    // dA = d_x_a^T @ x = [rank, batch] @ [batch, hidden] = [rank, hidden]
    let x_a_gate = x_flat.matmul(&saved.gate_lora_a.t())?; // [batch, rank]
    let gate_d_lora_b = d_gate_pre_silu
        .t()
        .matmul(&x_a_gate)?
        .multiply(Array::from_f32(saved.gate_scale))?;

    let d_x_a_gate = d_gate_pre_silu.matmul(&saved.gate_lora_b)?; // [batch, rank]
    let gate_d_lora_a = d_x_a_gate
        .t()
        .matmul(&x_flat)?
        .multiply(Array::from_f32(saved.gate_scale))?;

    // ============================================
    // 5. INPUT GRADIENT (ACCUMULATE BOTH PATHS)
    // ============================================
    // dX = d_up @ W_up.T + d_gate @ W_gate.T
    //    + scale_up * (d_up @ B_up) @ A_up
    //    + scale_gate * (d_gate @ B_gate) @ A_gate

    // Up path base
    let d_x = d_up_out.matmul(&saved.up_weight)?;

    // Up path LoRA
    let d_x = d_x.add(
        &d_up_out
            .matmul(&saved.up_lora_b)?
            .matmul(&saved.up_lora_a)?
            .multiply(Array::from_f32(saved.up_scale))?,
    )?;

    // Gate path base
    let d_x = d_x.add(&d_gate_pre_silu.matmul(&saved.gate_weight)?)?;

    // Gate path LoRA
    let d_x = d_x.add(
        &d_gate_pre_silu
            .matmul(&saved.gate_lora_b)?
            .matmul(&saved.gate_lora_a)?
            .multiply(Array::from_f32(saved.gate_scale))?,
    )?;

    // Reshape back if needed
    let d_x = if ndim > 2 {
        d_x.reshape(&shape.iter().map(|&x| x as i32).collect::<Vec<_>>())?
    } else {
        d_x
    };

    Ok(MlpLoraGrads {
        gate_d_lora_a,
        gate_d_lora_b,
        up_d_lora_a,
        up_d_lora_b,
        down_d_lora_a,
        down_d_lora_b,
        d_x,
    })
}

/// Accumulated gradients for multiple LoRA layers.
#[derive(Debug, Default)]
pub struct AccumulatedLoraGrads {
    /// Map from parameter name to gradient.
    pub grads: HashMap<String, Array>,
}

impl AccumulatedLoraGrads {
    /// Create new empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add gradients for a named LoRA layer.
    pub fn add_layer_grads(&mut self, layer_name: &str, grads: &LoraGrads) {
        self.grads
            .insert(format!("{}.lora_a", layer_name), grads.d_lora_a.clone());
        self.grads
            .insert(format!("{}.lora_b", layer_name), grads.d_lora_b.clone());
    }

    /// Merge with another accumulator (for gradient accumulation).
    pub fn accumulate(
        &mut self,
        other: &AccumulatedLoraGrads,
    ) -> Result<(), mlx_rs::error::Exception> {
        for (name, grad) in &other.grads {
            if let Some(existing) = self.grads.get_mut(name) {
                *existing = existing.add(grad)?;
            } else {
                self.grads.insert(name.clone(), grad.clone());
            }
        }
        Ok(())
    }

    /// Scale all gradients (for gradient accumulation averaging).
    pub fn scale(&mut self, factor: f32) -> Result<(), mlx_rs::error::Exception> {
        for grad in self.grads.values_mut() {
            *grad = grad.multiply(Array::from_f32(factor))?;
        }
        Ok(())
    }

    /// Get gradient for a specific parameter.
    pub fn get(&self, name: &str) -> Option<&Array> {
        self.grads.get(name)
    }

    /// Convert to HashMap for optimizer.
    pub fn into_hashmap(self) -> HashMap<String, Array> {
        self.grads
    }
}

/// Custom training step using LoRA autograd.
///
/// This function demonstrates how to use the custom autograd for a single training step.
/// It can be used as a template for implementing custom training loops.
///
/// # Example Flow
///
/// ```ignore
/// // 1. Forward through embedding (use MLX autodiff)
/// let hidden = embed(input_ids)?;
///
/// // 2. Forward through transformer layers (custom autograd)
/// let mut saved_contexts = Vec::new();
/// for layer in &layers {
///     let (output, saved) = layer_forward_with_grad(hidden, layer)?;
///     saved_contexts.push(saved);
///     hidden = output;
/// }
///
/// // 3. LM head and loss (use MLX autodiff)
/// let logits = lm_head.forward(&hidden)?;
/// let (loss, d_logits) = cross_entropy_with_grad(&logits, &labels)?;
///
/// // 4. Backward through LM head
/// let d_hidden = lm_head_backward(&d_logits)?;
///
/// // 5. Backward through transformer layers (custom autograd)
/// let mut all_grads = AccumulatedLoraGrads::new();
/// for (i, saved) in saved_contexts.iter().rev().enumerate() {
///     let grads = layer_backward(&d_hidden, saved)?;
///     all_grads.add_layer_grads(&format!("layers.{}", layers.len() - 1 - i), &grads);
///     if let Some(dx) = grads.d_x {
///         d_hidden = dx;
///     }
/// }
///
/// // 6. Update with optimizer
/// optimizer.step(&all_grads.into_hashmap())?;
/// ```
pub fn custom_training_step_example() {
    // This is a documentation placeholder showing the intended usage pattern.
    // See the training_loop module for actual implementation.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_forward_backward_shapes() {
        let batch = 4;
        let in_features = 512;
        let out_features = 1024;
        let rank = 8;
        let scale = 2.0;

        // Create test tensors
        let x = mlx_rs::random::normal::<f32>(&[batch, in_features], None, None, None).unwrap();
        let weight =
            mlx_rs::random::normal::<f32>(&[out_features, in_features], None, None, None).unwrap();
        let lora_a = mlx_rs::random::normal::<f32>(&[rank, in_features], None, None, None).unwrap();
        let lora_b =
            mlx_rs::random::normal::<f32>(&[out_features, rank], None, None, None).unwrap();

        // Forward
        let ctx = LoraGradContext::new();
        let (output, saved) =
            lora_forward_with_grad(&x, &weight, &lora_a, &lora_b, scale, &ctx).unwrap();

        assert_eq!(output.shape(), &[batch, out_features]);
        assert_eq!(saved.x.shape(), &[batch, in_features]);
        assert_eq!(saved.x_a.shape(), &[batch, rank]);

        // Backward
        let d_output =
            mlx_rs::random::normal::<f32>(&[batch, out_features], None, None, None).unwrap();
        let grads = lora_backward(&d_output, &saved).unwrap();

        assert_eq!(grads.d_lora_a.shape(), &[rank, in_features]);
        assert_eq!(grads.d_lora_b.shape(), &[out_features, rank]);
        assert!(grads.d_x.is_some());
        assert_eq!(grads.d_x.as_ref().unwrap().shape(), &[batch, in_features]);
    }

    #[test]
    fn test_lora_without_input_grad() {
        let batch = 4;
        let in_features = 512;
        let out_features = 1024;
        let rank = 8;
        let scale = 2.0;

        let x = mlx_rs::random::normal::<f32>(&[batch, in_features], None, None, None).unwrap();
        let weight =
            mlx_rs::random::normal::<f32>(&[out_features, in_features], None, None, None).unwrap();
        let lora_a = mlx_rs::random::normal::<f32>(&[rank, in_features], None, None, None).unwrap();
        let lora_b =
            mlx_rs::random::normal::<f32>(&[out_features, rank], None, None, None).unwrap();

        // Forward without input grad
        let ctx = LoraGradContext::new().without_input_grad();
        let (_, saved) =
            lora_forward_with_grad(&x, &weight, &lora_a, &lora_b, scale, &ctx).unwrap();

        // Backward
        let d_output =
            mlx_rs::random::normal::<f32>(&[batch, out_features], None, None, None).unwrap();
        let grads = lora_backward(&d_output, &saved).unwrap();

        assert!(grads.d_x.is_none());
    }

    #[test]
    fn test_accumulated_grads() {
        let rank = 8;
        let in_features = 512;
        let out_features = 1024;

        let grads1 = LoraGrads {
            d_lora_a: mlx_rs::Array::ones::<f32>(&[rank, in_features]).unwrap(),
            d_lora_b: mlx_rs::Array::ones::<f32>(&[out_features, rank]).unwrap(),
            d_x: None,
        };

        let grads2 = LoraGrads {
            d_lora_a: mlx_rs::Array::ones::<f32>(&[rank, in_features]).unwrap(),
            d_lora_b: mlx_rs::Array::ones::<f32>(&[out_features, rank]).unwrap(),
            d_x: None,
        };

        let mut acc = AccumulatedLoraGrads::new();
        acc.add_layer_grads("layer.0.self_attn.q_proj", &grads1);
        acc.add_layer_grads("layer.0.self_attn.k_proj", &grads2);

        assert!(acc.get("layer.0.self_attn.q_proj.lora_a").is_some());
        assert!(acc.get("layer.0.self_attn.q_proj.lora_b").is_some());
        assert!(acc.get("layer.0.self_attn.k_proj.lora_a").is_some());
    }

    #[test]
    fn test_fused_mlp_forward_backward_shapes() {
        let batch = 2;
        let seq_len = 4;
        let hidden_size = 128;
        let intermediate_size = 256;
        let rank = 8;
        let scale = 2.0;

        // Create test tensors for MLP projections
        let x = mlx_rs::random::normal::<f32>(&[batch, seq_len, hidden_size], None, None, None)
            .unwrap();

        // Gate projection: hidden -> intermediate
        let gate_weight =
            mlx_rs::random::normal::<f32>(&[intermediate_size, hidden_size], None, None, None)
                .unwrap();
        let gate_lora_a =
            mlx_rs::random::normal::<f32>(&[rank, hidden_size], None, None, None).unwrap();
        let gate_lora_b =
            mlx_rs::random::normal::<f32>(&[intermediate_size, rank], None, None, None).unwrap();

        // Up projection: hidden -> intermediate
        let up_weight =
            mlx_rs::random::normal::<f32>(&[intermediate_size, hidden_size], None, None, None)
                .unwrap();
        let up_lora_a =
            mlx_rs::random::normal::<f32>(&[rank, hidden_size], None, None, None).unwrap();
        let up_lora_b =
            mlx_rs::random::normal::<f32>(&[intermediate_size, rank], None, None, None).unwrap();

        // Down projection: intermediate -> hidden
        let down_weight =
            mlx_rs::random::normal::<f32>(&[hidden_size, intermediate_size], None, None, None)
                .unwrap();
        let down_lora_a =
            mlx_rs::random::normal::<f32>(&[rank, intermediate_size], None, None, None).unwrap();
        let down_lora_b =
            mlx_rs::random::normal::<f32>(&[hidden_size, rank], None, None, None).unwrap();

        // Forward pass
        let (output, saved) = fused_mlp_forward(
            &x,
            &gate_weight,
            &gate_lora_a,
            &gate_lora_b,
            scale,
            &up_weight,
            &up_lora_a,
            &up_lora_b,
            scale,
            &down_weight,
            &down_lora_a,
            &down_lora_b,
            scale,
        )
        .unwrap();

        // Verify output shape
        assert_eq!(output.shape(), &[batch, seq_len, hidden_size]);

        // Verify saved tensors
        assert_eq!(saved.x.shape(), &[batch, seq_len, hidden_size]);
        assert_eq!(
            saved.gate_pre_silu.shape(),
            &[batch, seq_len, intermediate_size]
        );
        assert_eq!(
            saved.gate_activated.shape(),
            &[batch, seq_len, intermediate_size]
        );
        assert_eq!(saved.up_out.shape(), &[batch, seq_len, intermediate_size]);
        assert_eq!(
            saved.mlp_hidden.shape(),
            &[batch, seq_len, intermediate_size]
        );

        // Backward pass
        let d_output =
            mlx_rs::random::normal::<f32>(&[batch, seq_len, hidden_size], None, None, None)
                .unwrap();
        let grads = fused_mlp_backward(&d_output, &saved).unwrap();

        // Verify gradient shapes
        assert_eq!(grads.gate_d_lora_a.shape(), &[rank, hidden_size]);
        assert_eq!(grads.gate_d_lora_b.shape(), &[intermediate_size, rank]);
        assert_eq!(grads.up_d_lora_a.shape(), &[rank, hidden_size]);
        assert_eq!(grads.up_d_lora_b.shape(), &[intermediate_size, rank]);
        assert_eq!(grads.down_d_lora_a.shape(), &[rank, intermediate_size]);
        assert_eq!(grads.down_d_lora_b.shape(), &[hidden_size, rank]);
        assert_eq!(grads.d_x.shape(), &[batch, seq_len, hidden_size]);
    }

    #[test]
    fn test_fused_mlp_2d_input() {
        // Test with 2D input (no seq dimension)
        let batch = 8;
        let hidden_size = 64;
        let intermediate_size = 128;
        let rank = 4;
        let scale = 1.0;

        let x = mlx_rs::random::normal::<f32>(&[batch, hidden_size], None, None, None).unwrap();

        let gate_weight =
            mlx_rs::random::normal::<f32>(&[intermediate_size, hidden_size], None, None, None)
                .unwrap();
        let gate_lora_a =
            mlx_rs::random::normal::<f32>(&[rank, hidden_size], None, None, None).unwrap();
        let gate_lora_b =
            mlx_rs::random::normal::<f32>(&[intermediate_size, rank], None, None, None).unwrap();

        let up_weight =
            mlx_rs::random::normal::<f32>(&[intermediate_size, hidden_size], None, None, None)
                .unwrap();
        let up_lora_a =
            mlx_rs::random::normal::<f32>(&[rank, hidden_size], None, None, None).unwrap();
        let up_lora_b =
            mlx_rs::random::normal::<f32>(&[intermediate_size, rank], None, None, None).unwrap();

        let down_weight =
            mlx_rs::random::normal::<f32>(&[hidden_size, intermediate_size], None, None, None)
                .unwrap();
        let down_lora_a =
            mlx_rs::random::normal::<f32>(&[rank, intermediate_size], None, None, None).unwrap();
        let down_lora_b =
            mlx_rs::random::normal::<f32>(&[hidden_size, rank], None, None, None).unwrap();

        let (output, saved) = fused_mlp_forward(
            &x,
            &gate_weight,
            &gate_lora_a,
            &gate_lora_b,
            scale,
            &up_weight,
            &up_lora_a,
            &up_lora_b,
            scale,
            &down_weight,
            &down_lora_a,
            &down_lora_b,
            scale,
        )
        .unwrap();

        assert_eq!(output.shape(), &[batch, hidden_size]);

        let d_output =
            mlx_rs::random::normal::<f32>(&[batch, hidden_size], None, None, None).unwrap();
        let grads = fused_mlp_backward(&d_output, &saved).unwrap();

        // 2D case should have 2D input gradient
        assert_eq!(grads.d_x.shape(), &[batch, hidden_size]);
    }
}
