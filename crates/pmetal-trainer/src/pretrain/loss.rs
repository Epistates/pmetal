//! Causal language-modeling loss for full-parameter pretraining.
//!
//! Shifts logits and targets by one position so position `t` predicts token
//! `t+1`, then computes sparse softmax cross-entropy with integer targets.
//! Optionally adds z-loss for MoE router stability.

use pmetal_bridge::compat::{Array, Exception, ops};

/// Compute causal-LM loss from `[B, T, V]` logits and `[B, T]` int32 target
/// token ids.
///
/// - `ignore_index`: masks out padding positions. `None` for dense sequences.
/// - `z_loss_coef`: if `Some(c)`, adds `c * mean(logsumexp(logits)^2)` to
///   prevent router logits from growing unboundedly (Switch Transformer §3.1).
pub fn causal_lm_loss(
    logits: &Array,
    targets: &Array,
    ignore_index: Option<i32>,
    z_loss_coef: Option<f32>,
) -> Result<Array, Exception> {
    let shape = logits.shape();
    assert!(
        shape.len() == 3,
        "causal_lm_loss: logits must be [B, T, V], got {:?}",
        shape
    );
    let seq_len = shape[1];
    assert!(
        seq_len >= 2,
        "causal_lm_loss: seq_len must be >= 2 to shift, got {}",
        seq_len
    );

    let shifted_logits = ops::slice_axis(logits, 1, 0, seq_len - 1);
    let shifted_targets = ops::slice_axis(targets, 1, 1, seq_len);

    let log_probs = shifted_logits.log_softmax(-1);
    let target_idx = shifted_targets.expand_dims(-1);
    let gathered = log_probs.take_along_axis(&target_idx, -1);
    let per_token = gathered.squeeze_axes(&[-1]).negative();

    let ce_loss = if let Some(ignore) = ignore_index {
        let ignore_arr = Array::from_i32(ignore);
        let keep = shifted_targets.not_equal(&ignore_arr).as_type::<f32>();
        let masked = per_token.multiply(&keep);
        let denom = ops::maximum(&keep.sum_axes(&[0, 1], false), &Array::from_f32(1.0));
        masked.sum_axes(&[0, 1], false).divide(&denom)
    } else {
        per_token.mean_axes(&[0, 1], false)
    };

    // Optional z-loss: penalises large logits to prevent router instability.
    // z = coef * mean(logsumexp(logits, axis=-1)^2)
    if let Some(coef) = z_loss_coef {
        let lse = ops::logsumexp(&shifted_logits, -1, false);
        let z = lse.multiply(&lse).mean_axes(&[0, 1], false);
        Ok(ce_loss.add(&z.multiply(&Array::from_f32(coef))))
    } else {
        Ok(ce_loss)
    }
}
