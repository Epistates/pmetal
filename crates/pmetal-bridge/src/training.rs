//! Training utilities for pmetal-bridge — zero mlx-rs dependency.
//!
//! Provides loss functions, gradient utilities, and training helpers
//! that operate purely on InlineArray.

use crate::InlineArray;
use crate::optimizer::ParamSet;

// ── Loss Functions ────────────────────────────────────────────────────────

/// Cross-entropy loss with ignore_index masking.
///
/// - `logits`: `[N, vocab_size]` — raw (un-normalised) logits.
/// - `labels`: `[N]` int32 token ids; positions equal to `ignore_index` are
///   excluded from the mean (standard padding mask).
///
/// Returns a scalar mean loss over non-ignored tokens.
pub fn cross_entropy_loss(
    logits: &InlineArray,
    labels: &InlineArray,
    ignore_index: i32,
) -> InlineArray {
    // log_softmax along vocab axis → [N, vocab_size]
    let log_probs = logits.log_softmax(-1);

    // Gather log-prob at each label position:
    //   take_along_axis(log_probs, labels[:, None], axis=-1).squeeze(-1) → [N]
    let label_indices = labels.expand_dims(-1); // [N, 1]
    let nll = log_probs.take_along_axis(&label_indices, -1).squeeze(-1); // [N]
    let nll = nll.negative(); // negative log-likelihood

    // Build float mask: 1.0 where label != ignore_index, 0.0 elsewhere.
    let ignore_arr = InlineArray::from_i32(ignore_index);
    let mask = labels.not_equal(&ignore_arr); // [N] bool
    let mask_f32 = mask.as_dtype(10); // dtype 10 == float32

    // Sum masked NLL and divide by number of valid tokens.
    let masked_nll = nll.multiply(&mask_f32);
    let valid_count = mask_f32.sum_all();
    // Avoid divide-by-zero when the entire batch is padding.
    let valid_count_safe = valid_count.maximum(&InlineArray::from_f32(1.0));

    masked_nll.sum_all().divide(&valid_count_safe)
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
    let shift_logits = logits.slice(
        &[0, 0, 0],
        &[batch, seq_len - 1, vocab_size],
    );
    let shift_labels = labels.slice(
        &[0, 1],
        &[batch, seq_len],
    );

    // Flatten to [batch*(seq_len-1), vocab_size] / [batch*(seq_len-1)]
    let flat_logits = shift_logits.reshape(&[-1, vocab_size]);
    let flat_labels = shift_labels.reshape(&[-1]);

    cross_entropy_loss(&flat_logits, &flat_labels, ignore_index)
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
            None => { acc.insert(key.clone(), scaled); }
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
    let y_base = x.quantized_matmul(base_weight, base_scales, base_biases, true, group_size, bits);

    // Low-rank adapter: scale * (x @ A.T) @ B.T
    let xa = x.matmul(&lora_a.t());
    let xab = xa.matmul(&lora_b.t());
    let scale_arr = InlineArray::from_f32(scale);
    let y_lora = xab.multiply(&scale_arr);

    y_base.add(&y_lora)
}
