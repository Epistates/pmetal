//! Causal language-modeling loss for full-parameter pretraining.
//!
//! Shifts logits and targets by one position so position `t` predicts token
//! `t+1`, then computes sparse softmax cross-entropy with integer targets.
//! Optionally adds z-loss for MoE router stability.

use pmetal_runtime::compat::{Array, Exception};

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

    let shifted_logits = logits.slice(&[0, 0, 0], &[shape[0], seq_len - 1, shape[2]]);
    let shifted_targets = targets.slice(&[0, 1], &[shape[0], seq_len]);

    let ce_loss = if let Some(ignore) = ignore_index {
        pmetal_runtime::training::cross_entropy_loss(&shifted_logits, &shifted_targets, ignore)
    } else {
        pmetal_runtime::training::cross_entropy_loss_dense(&shifted_logits, &shifted_targets)
    };

    // Optional z-loss: penalises large logits to prevent router instability.
    // z = coef * mean(logsumexp(logits, axis=-1)^2)
    if let Some(coef) = z_loss_coef {
        let lse = shifted_logits.logsumexp(-1, false);
        let z = lse.multiply(&lse).mean_axes(&[0, 1], false);
        Ok(ce_loss.add(&z.multiply(&Array::from_f32(coef))))
    } else {
        Ok(ce_loss)
    }
}
