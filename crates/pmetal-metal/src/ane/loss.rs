//! Composable loss functions for ANE training.
//!
//! Decouples loss computation from the training loop, enabling pluggable
//! loss functions (cross-entropy, KL divergence, MSE) without modifying
//! `DynamicAneTrainer` internals.
//!
//! All tensors use channel-first `[V, S]` layout (vocab x sequence) to match
//! the ANE trainer convention. Each channel (vocab entry) is a contiguous
//! vector of length `seq`.

pub use crate::accelerate::IGNORE_INDEX;

/// Output from a loss computation.
#[derive(Copy, Clone)]
pub struct LossOutput {
    /// Scalar loss value: mean NLL over the counted tokens (not scaled by
    /// `loss_scale`).
    pub loss: f32,
}

/// Trait for loss functions compatible with ANE training.
///
/// Implementations compute loss + gradient in a single fused pass.
/// The gradient is written into `dlogits` (pre-allocated by the caller).
pub trait AneTrainingLoss: Send {
    /// Compute loss and gradient from logits and targets.
    ///
    /// - `logits`: `[vocab, seq]` channel-first (each vocab entry is contiguous over seq).
    /// - `targets`: `[seq]` token IDs; [`IGNORE_INDEX`] leaves a position out.
    /// - `vocab`: vocabulary size.
    /// - `seq`: sequence length.
    /// - `loss_scale`: multiplier on gradient for mixed-precision stability.
    /// - `dlogits`: output gradient buffer `[vocab, seq]`, same layout as logits.
    ///
    /// Returns the mean NLL over the counted positions, `n` of them. The
    /// gradient in `dlogits` is scaled by `loss_scale / n`.
    fn compute(
        &mut self,
        logits: &[f32],
        targets: &[u32],
        vocab: usize,
        seq: usize,
        loss_scale: f32,
        dlogits: &mut [f32],
    ) -> LossOutput;
}

/// Standard cross-entropy with numerically stable log-softmax.
///
/// Fuses forward (loss) and backward (gradient) into a single pass per
/// position. Uses vDSP for vectorized exp/sum operations on macOS.
///
/// Operates on channel-first `[V, S]` layout. Per-position softmax is computed
/// by gathering logits across channels for each position.
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Create a new cross-entropy loss function.
    pub fn new(_vocab: usize) -> Self {
        Self
    }
}

impl AneTrainingLoss for CrossEntropyLoss {
    fn compute(
        &mut self,
        logits: &[f32],
        targets: &[u32],
        vocab: usize,
        seq: usize,
        loss_scale: f32,
        dlogits: &mut [f32],
    ) -> LossOutput {
        debug_assert!(vocab > 0);
        debug_assert_eq!(logits.len(), vocab * seq);
        debug_assert_eq!(dlogits.len(), vocab * seq);
        debug_assert_eq!(targets.len(), seq);

        // Delegate to the existing accelerate::cross_entropy_loss which handles
        // channel-first [V, S] layout with vDSP-accelerated softmax.
        let loss = crate::accelerate::cross_entropy_loss(dlogits, logits, targets, vocab, seq);

        // Apply loss scaling to gradient
        if loss_scale != 1.0 {
            crate::accelerate::scale_inplace(dlogits, loss_scale);
        }

        LossOutput { loss }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ce_loss_positive() {
        let vocab = 8;
        let seq = 2;
        let mut loss_fn = CrossEntropyLoss::new(vocab);
        // Channel-first: [V=8, S=2]
        let logits = vec![0.1f32; vocab * seq];
        let targets = vec![0u32, 3];
        let mut dlogits = vec![0.0f32; vocab * seq];

        let out = loss_fn.compute(&logits, &targets, vocab, seq, 1.0, &mut dlogits);
        assert!(
            out.loss > 0.0,
            "CE loss should be positive, got {}",
            out.loss
        );
    }

    #[test]
    fn ce_loss_decreases_with_confidence() {
        let vocab = 4;
        let seq = 1;
        let mut loss_fn = CrossEntropyLoss::new(vocab);
        let targets = vec![1u32];

        // Uniform logits [V=4, S=1] — each channel has 1 element
        let logits_uniform = vec![0.0f32; vocab];
        let mut d1 = vec![0.0f32; vocab];
        let l1 = loss_fn.compute(&logits_uniform, &targets, vocab, seq, 1.0, &mut d1);

        // Confident: target=1, so logits[1*1 + 0] = 5.0
        let logits_confident = vec![0.0, 5.0, 0.0, 0.0]; // [V=4, S=1]
        let mut d2 = vec![0.0f32; vocab];
        let l2 = loss_fn.compute(&logits_confident, &targets, vocab, seq, 1.0, &mut d2);

        assert!(l2.loss < l1.loss, "confident should have lower loss");
    }

    #[test]
    fn ce_loss_scale_multiplies_gradient() {
        let vocab = 4;
        let seq = 1;
        let mut loss_fn = CrossEntropyLoss::new(vocab);
        let logits = vec![0.1f32; vocab];
        let targets = vec![0u32];

        let mut d1 = vec![0.0f32; vocab];
        let mut d2 = vec![0.0f32; vocab];
        loss_fn.compute(&logits, &targets, vocab, seq, 1.0, &mut d1);
        loss_fn.compute(&logits, &targets, vocab, seq, 2.0, &mut d2);

        // Gradient with loss_scale=2 should be ~2x gradient with loss_scale=1
        for i in 0..vocab {
            let ratio = d2[i] / d1[i];
            assert!(
                (ratio - 2.0).abs() < 0.01,
                "element {i}: ratio {ratio}, expected ~2.0"
            );
        }
    }
}
