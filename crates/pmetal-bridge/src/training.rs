//! Training utilities for pmetal-bridge — zero mlx-rs dependency.
//!
//! Provides loss functions, gradient utilities, and training helpers
//! that operate purely on InlineArray.

use crate::InlineArray;
use crate::optimizer::ParamSet;
use std::collections::HashMap;
use std::hash::Hash;

// ── Loss Functions ────────────────────────────────────────────────────────

const DTYPE_INT32: i32 = 7;
const DTYPE_FLOAT32: i32 = 10;

/// Cross-entropy loss with ignore_index masking.
///
/// - `logits`: `[..., vocab_size]` — raw (un-normalised) logits.
/// - `labels`: matching leading dimensions; positions equal to `ignore_index`
///   are excluded from the mean (standard padding mask).
///
/// Returns a scalar mean loss over non-ignored tokens. The implementation uses
/// selective `logsumexp - target_logit` so it avoids materialising a full
/// `log_softmax` tensor and never gathers ignored labels.
pub fn cross_entropy_loss(
    logits: &InlineArray,
    labels: &InlineArray,
    ignore_index: i32,
) -> InlineArray {
    cross_entropy_loss_impl(logits, labels, Some(ignore_index))
}

/// Cross-entropy loss without ignore-index masking.
///
/// This is for dense batches where every label is valid. Use
/// [`cross_entropy_loss`] for padded language-model batches.
pub fn cross_entropy_loss_dense(logits: &InlineArray, labels: &InlineArray) -> InlineArray {
    cross_entropy_loss_impl(logits, labels, None)
}

/// Per-token cross-entropy with ignored positions zeroed.
///
/// Returns a flat `[N]` tensor. Callers that need custom weighting (for
/// diffusion masks, curriculum masks, etc.) should use this and apply their
/// own reduction.
pub fn per_token_cross_entropy_loss(
    logits: &InlineArray,
    labels: &InlineArray,
    ignore_index: i32,
) -> InlineArray {
    let (loss, _valid_mask) = per_token_cross_entropy_impl(logits, labels, Some(ignore_index));
    loss
}

/// Per-token cross-entropy without ignore-index masking.
pub fn per_token_cross_entropy_loss_dense(
    logits: &InlineArray,
    labels: &InlineArray,
) -> InlineArray {
    let (loss, _valid_mask) = per_token_cross_entropy_impl(logits, labels, None);
    loss
}

fn cross_entropy_loss_impl(
    logits: &InlineArray,
    labels: &InlineArray,
    ignore_index: Option<i32>,
) -> InlineArray {
    let (per_token_loss, valid_mask) = per_token_cross_entropy_impl(logits, labels, ignore_index);
    let valid_count = valid_mask.sum_all();
    let valid_count_safe = valid_count.maximum(&InlineArray::from_f32(1.0));

    per_token_loss.sum_all().divide(&valid_count_safe)
}

fn per_token_cross_entropy_impl(
    logits: &InlineArray,
    labels: &InlineArray,
    ignore_index: Option<i32>,
) -> (InlineArray, InlineArray) {
    let vocab_size = logits.dim(-1);
    let flat_logits = logits.reshape(&[-1, vocab_size]);
    let flat_labels = labels.reshape(&[-1]);

    let (gather_labels, valid_mask_f32) = if let Some(ignore) = ignore_index {
        let labels_dtype = flat_labels.dtype_raw();
        let ignore_arr = InlineArray::from_int(ignore).as_dtype(labels_dtype);
        let zero = InlineArray::from_int(0).as_dtype(labels_dtype);
        let valid_mask = flat_labels.not_equal(&ignore_arr);
        let safe_labels = valid_mask.where_cond(&flat_labels, &zero);
        (
            safe_labels.as_dtype(DTYPE_INT32),
            valid_mask.as_dtype(DTYPE_FLOAT32),
        )
    } else {
        let n_tokens = flat_labels.dim(0);
        (
            flat_labels.as_dtype(DTYPE_INT32),
            InlineArray::ones(&[n_tokens], DTYPE_FLOAT32),
        )
    };

    // MLX >= 0.32: label indices must never enter the grad trace
    let gather_indices = gather_labels.expand_dims(-1).stop_gradient();
    let selected_logits = flat_logits.take_along_axis(&gather_indices, -1).squeeze(-1);
    let logsumexp = flat_logits.logsumexp(-1, false);
    let per_token_loss = logsumexp
        .subtract(&selected_logits)
        .multiply(&valid_mask_f32);

    (per_token_loss, valid_mask_f32)
}

/// Causal language-model loss: shift logits and labels by one position, then
/// compute [`cross_entropy_loss`].
///
/// - `logits`: `[batch, seq_len, vocab_size]`
/// - `labels`: `[batch, seq_len]` int32
///
/// Returns a scalar loss.
pub fn causal_lm_loss(
    logits: &InlineArray,
    labels: &InlineArray,
    ignore_index: i32,
) -> InlineArray {
    let batch = logits.dim(0);
    let seq_len = logits.dim(1);
    let vocab_size = logits.dim(2);

    // logits[:, :-1, :] predicts labels[:, 1:]
    let shift_logits = logits.slice(&[0, 0, 0], &[batch, seq_len - 1, vocab_size]);
    let shift_labels = labels.slice(&[0, 1], &[batch, seq_len]);

    // Flatten to [batch*(seq_len-1), vocab_size] / [batch*(seq_len-1)]
    let flat_logits = shift_logits.reshape(&[-1, vocab_size]);
    let flat_labels = shift_labels.reshape(&[-1]);

    cross_entropy_loss(&flat_logits, &flat_labels, ignore_index)
}

/// Causal language-model loss without ignore-index masking.
pub fn causal_lm_loss_dense(logits: &InlineArray, labels: &InlineArray) -> InlineArray {
    let batch = logits.dim(0);
    let seq_len = logits.dim(1);
    let vocab_size = logits.dim(2);

    let shift_logits = logits.slice(&[0, 0, 0], &[batch, seq_len - 1, vocab_size]);
    let shift_labels = labels.slice(&[0, 1], &[batch, seq_len]);

    let flat_logits = shift_logits.reshape(&[-1, vocab_size]);
    let flat_labels = shift_labels.reshape(&[-1]);

    cross_entropy_loss_dense(&flat_logits, &flat_labels)
}

// ── Gradient Utilities ────────────────────────────────────────────────────

/// Clip all gradients in `grads` by the global L2 norm (in-place).
///
/// Computes `global_norm = sqrt(sum_i ||grad_i||_F^2)`, then scales every
/// gradient by `min(1, max_norm / global_norm)`.  The computation stays on the
/// GPU — no CPU sync until the caller explicitly evaluates.
///
/// Returns the (lazy) global norm scalar before clipping.
///
/// Passing `max_norm <= 0.0` is a no-op that returns `0.0`.
pub fn clip_grad_norm(grads: &mut ParamSet, max_norm: f32) -> InlineArray {
    clip_grad_norm_map(grads, max_norm)
}

/// Clip all gradients in any `HashMap<K, InlineArray>` by global L2 norm.
///
/// This supports both bridge-native [`ParamSet`] and the compat layer's
/// `FlattenedModuleParam` without duplicating the same lazy GPU expression in
/// trainer crates.
pub fn clip_grad_norm_map<K>(grads: &mut HashMap<K, InlineArray>, max_norm: f32) -> InlineArray
where
    K: Eq + Hash,
{
    if max_norm <= 0.0 {
        return InlineArray::from_f32(0.0);
    }

    // Accumulate squared Frobenius norms across all gradients.
    let mut norm_sq = InlineArray::from_f32(0.0);
    for grad in grads.values() {
        norm_sq = norm_sq.add(&grad.square().sum_all());
    }
    let norm = norm_sq.sqrt();

    // scale = max_norm / max(norm, max_norm)  ∈ (0, 1]
    let max_norm_arr = InlineArray::from_f32(max_norm);
    let norm_clamped = norm.maximum(&max_norm_arr);
    let scale = max_norm_arr.divide(&norm_clamped);

    for grad in grads.values_mut() {
        *grad = grad.multiply(&scale);
    }

    norm
}

/// Accumulate gradients into an accumulator map: `acc[k] += grads[k] * scale`.
///
/// If a key is absent from `acc` it is inserted as `grads[k] * scale`.
/// `scale` is typically `1.0 / gradient_accumulation_steps`.
pub fn accumulate_gradients(acc: &mut ParamSet, grads: &ParamSet, scale: f32) {
    let scale_arr = InlineArray::from_f32(scale);

    for (key, grad) in grads {
        let scaled = grad.multiply(&scale_arr);
        match acc.get_mut(key) {
            Some(existing) => *existing = existing.add(&scaled),
            None => {
                acc.insert(key.clone(), scaled);
            }
        }
    }
}

/// Materialise and detach all arrays in a [`ParamSet`].
///
/// Drives MLX's lazy evaluation, frees the computation graph, and prevents
/// memory growth during long training loops.
pub fn eval_params(params: &mut ParamSet) {
    let mut refs: Vec<&mut InlineArray> = params.values_mut().collect();
    if refs.is_empty() {
        return;
    }
    crate::inline_array::eval_and_detach_many(&mut refs);
}

// ── LoRA Utilities ────────────────────────────────────────────────────────

/// LoRA forward pass for a dense (fp16/bf16) base weight.
///
/// Computes `y = x @ W.T + scale * (x @ A.T) @ B.T`.
///
/// - `x`:           `[batch, seq, in_features]`
/// - `base_weight`: `[out_features, in_features]` (frozen)
/// - `lora_a`:      `[rank, in_features]` (trainable)
/// - `lora_b`:      `[out_features, rank]` (trainable)
/// - `scale`:       `alpha / rank`
///
/// Returns `[batch, seq, out_features]`.
pub fn lora_forward(
    x: &InlineArray,
    base_weight: &InlineArray,
    lora_a: &InlineArray,
    lora_b: &InlineArray,
    scale: f32,
) -> InlineArray {
    // Base projection: x @ W.T
    let y_base = x.matmul(&base_weight.t());

    // Low-rank adapter: scale * (x @ A.T) @ B.T
    let xa = x.matmul(&lora_a.t());
    let xab = xa.matmul(&lora_b.t());
    let scale_arr = InlineArray::from_f32(scale);
    let y_lora = xab.multiply(&scale_arr);

    y_base.add(&y_lora)
}

/// LoRA forward pass for a quantized base weight.
///
/// Uses `quantized_matmul` (fused dequant + matmul) for the base projection
/// and standard matmul for the low-rank adapter path.
///
/// - `x`:             `[batch, seq, in_features]`
/// - `base_weight`:   packed quantized weight `[out_features, in_features/pack]`
/// - `base_scales`:   per-group scales
/// - `base_biases`:   per-group biases (optional — pass `None` if absent)
/// - `group_size`:    quantisation group size (e.g. 64)
/// - `bits`:          bits per weight (e.g. 4)
/// - `lora_a`:        `[rank, in_features]` (trainable)
/// - `lora_b`:        `[out_features, rank]` (trainable)
/// - `scale`:         `alpha / rank`
///
/// Returns `[batch, seq, out_features]`.
#[allow(clippy::too_many_arguments)]
pub fn lora_forward_quantized(
    x: &InlineArray,
    base_weight: &InlineArray,
    base_scales: &InlineArray,
    base_biases: Option<&InlineArray>,
    group_size: i32,
    bits: i32,
    lora_a: &InlineArray,
    lora_b: &InlineArray,
    scale: f32,
) -> InlineArray {
    // Base: fused dequant + matmul (transpose=true: W is [out, in/pack])
    let y_base = x.quantized_matmul(
        base_weight,
        base_scales,
        base_biases,
        true,
        group_size,
        bits,
    );

    // Low-rank adapter: scale * (x @ A.T) @ B.T
    let xa = x.matmul(&lora_a.t());
    let xab = xa.matmul(&lora_b.t());
    let scale_arr = InlineArray::from_f32(scale);
    let y_lora = xab.multiply(&scale_arr);

    y_base.add(&y_lora)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_entropy_masks_ignored_labels_before_gather() {
        let logits = InlineArray::from_slice(
            &[
                1.0_f32, 2.0, 3.0, //
                0.0, 0.0, 0.0, //
                9.0, 9.0, 9.0,
            ],
            &[3, 3],
        );
        let labels = InlineArray::from_i32_slice_shaped(&[2, -100, 1], &[3]);

        let per_token = per_token_cross_entropy_loss(&logits, &labels, -100);
        per_token.eval();
        let vals: &[f32] = per_token.as_slice();

        let first = (1.0_f32 + (-1.0_f32).exp() + (-2.0_f32).exp()).ln();
        let third = 3.0_f32.ln();
        assert!((vals[0] - first).abs() < 1e-5, "{} vs {}", vals[0], first);
        assert_eq!(vals[1], 0.0);
        assert!((vals[2] - third).abs() < 1e-5, "{} vs {}", vals[2], third);

        let loss = cross_entropy_loss(&logits, &labels, -100);
        loss.eval();
        let expected = (first + third) / 2.0;
        assert!(
            (loss.item_f32() - expected).abs() < 1e-5,
            "{} vs {}",
            loss.item_f32(),
            expected
        );
    }
}
