//! Causal language-modeling loss for full-parameter pretraining.
//!
//! Shifts logits and targets by one position so position `t` predicts token
//! `t+1`, then computes the runtime's cross-entropy: the logits in f32,
//! ignored targets out of both the loss and its divisor, and an optional
//! z-loss over the tokens that count.

use pmetal_runtime::compat::{Array, Exception};
use pmetal_runtime::training::CrossEntropy;

/// Compute causal-LM loss from `[B, T, V]` logits and `[B, T]` int32 target
/// token ids.
///
/// - `ignore_index`: masks out padding positions. `None` for dense sequences.
/// - `z_loss_coef`: if `Some(c)`, adds `c * lse^2` to each counted token's
///   loss (PaLM's auxiliary loss, which keeps the softmax normalizer near 1).
///   An ignored position contributes nothing to it, as to the loss.
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
    assert!(
        shape[1] >= 2,
        "causal_lm_loss: seq_len must be >= 2 to shift, got {}",
        shape[1]
    );
    let mut cross_entropy = CrossEntropy::new();
    if let Some(ignore) = ignore_index {
        cross_entropy = cross_entropy.with_ignore_index(ignore);
    }
    if let Some(coef) = z_loss_coef {
        cross_entropy = cross_entropy.with_z_loss(coef);
    }
    Ok(cross_entropy.causal_lm_mean(logits, targets))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_runtime::compat::{Dtype, random};

    fn scalar(a: &Array) -> f32 {
        a.eval();
        pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
        a.item_f32()
    }

    /// The z-loss is a loss on the tokens that count. It was a mean of
    /// `lse^2` over every position, ignored ones included, so padding (whose
    /// logits nothing constrains) set the loss.
    #[test]
    fn z_loss_ignores_the_positions_the_loss_ignores() {
        random::seed(5);
        let (b, t, v) = (2, 6, 11);
        let logits = random::normal(&[b, t, v], Dtype::Float32);
        let mut targets = vec![3_i32; (b * t) as usize];
        // The last three positions of every row are padding.
        for row in 0..b as usize {
            for pos in 3..t as usize {
                targets[row * t as usize + pos] = -100;
            }
        }
        let targets = Array::from_i32_slice_shaped(&targets, &[b, t]);
        let loss = |logits: &Array| {
            scalar(&causal_lm_loss(logits, &targets, Some(-100), Some(0.05)).unwrap())
        };
        let base = loss(&logits);

        // Blow up the logits at the padded positions: nothing may change.
        let positions = Array::from_f32_slice(
            &(0..t)
                .map(|p| if p >= 3 { 1.0 } else { 0.0 })
                .collect::<Vec<_>>(),
            &[1, t, 1],
        );
        let blown = logits.add(&positions.multiply(&Array::from_f32(50.0)));
        let after = loss(&blown);
        assert!(
            (after - base).abs() <= 1e-5 * base.abs(),
            "the padded positions reached the loss: {base} -> {after}"
        );
        // The old definition, a mean of lse^2 over every position, is what
        // the padding used to move.
        let old = |logits: &Array| {
            let shifted = logits.slice(&[0, 0, 0], &[b, t - 1, v]);
            let lse = shifted.logsumexp(-1, false);
            scalar(
                &lse.multiply(&lse)
                    .mean_axes(&[0, 1], false)
                    .multiply(&Array::from_f32(0.05)),
            )
        };
        assert!(
            old(&blown) > 10.0 * old(&logits),
            "the test must be able to see the defect"
        );
        // And the z-loss is there: it adds to the plain loss.
        let plain = scalar(&causal_lm_loss(&logits, &targets, Some(-100), None).unwrap());
        assert!(base > plain, "z-loss added nothing: {plain} vs {base}");
    }

    #[test]
    fn matches_the_runtime_loss_without_the_extras() {
        random::seed(6);
        let logits = random::normal(&[2, 5, 9], Dtype::Float32);
        let targets = Array::from_i32_slice_shaped(&[1, 2, 3, 4, 5, 6, 7, 8, 0, 1], &[2, 5]);
        let ours = scalar(&causal_lm_loss(&logits, &targets, None, None).unwrap());
        let theirs = scalar(&pmetal_runtime::training::causal_lm_loss_dense(
            &logits, &targets,
        ));
        assert!((ours - theirs).abs() < 1e-6);
    }
}
