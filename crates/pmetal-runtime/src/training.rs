//! Training utilities for pmetal-runtime.
//!
//! The language-model loss and gradient clipping, on [`InlineArray`]s. Every
//! op builds lazy MLX graph nodes; nothing here evaluates.

use crate::InlineArray;
use crate::compat::Dtype;
use crate::inline_array::{LogitRowStats, logit_row_stats};
use std::collections::HashMap;
use std::hash::Hash;

// ── Cross-entropy ─────────────────────────────────────────────────────────

/// Token-level cross-entropy of `[..., vocab]` logits against integer labels.
///
/// For a token with label `y` the loss is
///
/// ```text
/// (1 − ε)·(lse − z_y) + ε·(lse − mean_j z_j) + λ·lse²
/// ```
///
/// where `lse = logsumexp(z)`, `ε` is the label smoothing (the second term is
/// the loss against the uniform distribution, as `torch.nn.CrossEntropyLoss`'s
/// `label_smoothing` defines it) and `λ` the z-loss weight (PaLM, Chowdhery et
/// al. 2022: it keeps the normalizer near 1, which keeps bf16 logits stable
/// over long runs). With `ε = λ = 0` it is the negative log-likelihood.
///
/// Tokens whose label is the ignore index (padding, a prompt masked out of an
/// SFT loss) contribute nothing to the loss or to its gradient, and drop out
/// of the mean's divisor.
///
/// The logits are evaluated in f32 whatever their dtype. A bf16
/// `logsumexp` rounds the normalizer to 8 mantissa bits, which on a
/// 150,000-token vocabulary moved single tokens' losses by up to 0.12 nats
/// and the softmax in the backward by as much as 2%; `transformers` upcasts
/// the logits for its loss for the same reason. The upcast costs an f32 copy
/// of the logits: where that is too much (a long sequence over a large
/// vocabulary), the fused cut cross-entropy never forms them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossEntropy {
    ignore_index: Option<i32>,
    label_smoothing: f32,
    z_loss: f32,
}

impl Default for CrossEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossEntropy {
    /// The negative log-likelihood of every label, none ignored.
    pub const fn new() -> Self {
        Self {
            ignore_index: None,
            label_smoothing: 0.0,
            z_loss: 0.0,
        }
    }

    /// Ignore the tokens labelled `index` (conventionally `-100`).
    pub const fn with_ignore_index(mut self, index: i32) -> Self {
        self.ignore_index = Some(index);
        self
    }

    /// Smooth the one-hot target toward the uniform distribution by `epsilon`.
    ///
    /// # Panics
    /// If `epsilon` is not in `[0, 1)`.
    pub fn with_label_smoothing(mut self, epsilon: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&epsilon),
            "label smoothing must be in [0, 1), got {epsilon}"
        );
        self.label_smoothing = epsilon;
        self
    }

    /// Add `weight · lse²` to each token's loss (PaLM's auxiliary z-loss;
    /// `1e-4` is the paper's value).
    ///
    /// # Panics
    /// If `weight` is negative or not finite.
    pub fn with_z_loss(mut self, weight: f32) -> Self {
        assert!(
            weight.is_finite() && weight >= 0.0,
            "z-loss weight must be finite and non-negative, got {weight}"
        );
        self.z_loss = weight;
        self
    }

    /// `labels` flattened, the labels to gather with (an ignored label is out
    /// of the vocabulary, so it is replaced by 0), and the `[N]` f32 mask of
    /// the tokens that count.
    fn gather_labels_and_mask(&self, labels: &InlineArray) -> (InlineArray, InlineArray) {
        let f32_ = Dtype::Float32.as_i32();
        let flat_labels = labels.reshape(&[-1]);
        let (gather_labels, valid) = match self.ignore_index {
            Some(ignore) => {
                let dtype = flat_labels.dtype_raw();
                let ignore = InlineArray::from_int(ignore).as_dtype(dtype);
                let zero = InlineArray::from_int(0).as_dtype(dtype);
                let valid = flat_labels.not_equal(&ignore);
                let safe = valid.where_cond(&flat_labels, &zero);
                (safe, valid.as_dtype(f32_))
            }
            None => (
                flat_labels.clone(),
                InlineArray::ones(&[flat_labels.dim(0)], f32_),
            ),
        };
        // MLX >= 0.32: label indices must never enter the grad trace.
        (
            gather_labels
                .as_dtype(Dtype::Int32.as_i32())
                .stop_gradient(),
            valid,
        )
    }

    /// Each token's loss as a flat `[N]` f32 array, zero where ignored, and
    /// the `[N]` f32 mask of the tokens that count (1 where counted).
    ///
    /// For callers that weight or reduce tokens their own way (a diffusion
    /// mask, a curriculum): [`mean`](Self::mean) is this, summed and divided
    /// by the count.
    pub fn per_token(
        &self,
        logits: &InlineArray,
        labels: &InlineArray,
    ) -> (InlineArray, InlineArray) {
        let vocab = logits.dim(-1);
        let z = logits.reshape(&[-1, vocab]);
        let (targets, valid) = self.gather_labels_and_mask(labels);
        let stats = logit_row_stats(&z, &targets);
        (self.loss_from_row_stats(&stats, vocab, &valid), valid)
    }

    /// [`per_token`](Self::per_token) for statistics computed elsewhere: a
    /// fused head-and-loss kernel that never forms the logits produces each
    /// row's `lse` and target logit (and the sum of the row, when smoothing),
    /// and the loss is the same function of them.
    ///
    /// * `stats` - `[N]` rows; `logit_sum` is read only when the smoothing is
    ///   not zero, and may be a placeholder otherwise
    /// * `labels` - `[N]` integer labels, in the vocabulary or ignored
    /// * `vocab` - the number of classes the rows sum over
    pub fn per_token_from_row_stats(
        &self,
        stats: &LogitRowStats,
        labels: &InlineArray,
        vocab: i32,
    ) -> (InlineArray, InlineArray) {
        let (_, valid) = self.gather_labels_and_mask(labels);
        (self.loss_from_row_stats(stats, vocab, &valid), valid)
    }

    /// [`mean`](Self::mean) for statistics computed elsewhere; see
    /// [`per_token_from_row_stats`](Self::per_token_from_row_stats).
    pub fn mean_from_row_stats(
        &self,
        stats: &LogitRowStats,
        labels: &InlineArray,
        vocab: i32,
    ) -> InlineArray {
        let (loss, valid) = self.per_token_from_row_stats(stats, labels, vocab);
        loss.sum_all()
            .divide(&valid.sum_all().maximum(&InlineArray::from_f32(1.0)))
    }

    /// The masked per-token loss: the one definition of
    /// `(1 − ε)(lse − z_y) + ε(lse − mean z) + λ·lse²`.
    fn loss_from_row_stats(
        &self,
        stats: &LogitRowStats,
        vocab: i32,
        valid: &InlineArray,
    ) -> InlineArray {
        let LogitRowStats {
            lse,
            target_logit,
            logit_sum,
        } = stats;
        let eps = self.label_smoothing;
        let mut loss = if eps == 0.0 {
            lse.subtract(target_logit)
        } else {
            // (1 − ε)(lse − z_y) + ε(lse − mean z) = lse − (1 − ε)·z_y − ε·mean z
            lse.subtract(&target_logit.mul_scalar(1.0 - eps))
                .subtract(&logit_sum.mul_scalar(eps / vocab as f32))
        };
        if self.z_loss != 0.0 {
            loss = loss.add(&lse.square().mul_scalar(self.z_loss));
        }
        loss.multiply(valid)
    }

    /// The summed loss over the counted tokens and how many there are, both
    /// f32 scalars.
    ///
    /// Gradient accumulation weights micro-batches by their counts with
    /// these: the mean of per-batch means is not the mean over all the
    /// tokens when batches hold different numbers of them.
    pub fn sum_and_count(
        &self,
        logits: &InlineArray,
        labels: &InlineArray,
    ) -> (InlineArray, InlineArray) {
        let (loss, valid) = self.per_token(logits, labels);
        (loss.sum_all(), valid.sum_all())
    }

    /// The mean loss over the counted tokens, an f32 scalar; zero when every
    /// token is ignored.
    pub fn mean(&self, logits: &InlineArray, labels: &InlineArray) -> InlineArray {
        let (sum, count) = self.sum_and_count(logits, labels);
        sum.divide(&count.maximum(&InlineArray::from_f32(1.0)))
    }

    /// [`mean`](Self::mean) for next-token prediction: position `t`'s logits
    /// are scored against the label at `t + 1`.
    ///
    /// * `logits` - `[batch, seq_len, vocab]`
    /// * `labels` - `[batch, seq_len]`
    pub fn causal_lm_mean(&self, logits: &InlineArray, labels: &InlineArray) -> InlineArray {
        let (shift_logits, shift_labels) = shift_for_next_token(logits, labels);
        self.mean(&shift_logits, &shift_labels)
    }
}

/// `logits[:, :-1]` and `labels[:, 1:]`: what position `t` predicts is the
/// token at `t + 1`.
fn shift_for_next_token(logits: &InlineArray, labels: &InlineArray) -> (InlineArray, InlineArray) {
    let batch = logits.dim(0);
    let seq_len = logits.dim(1);
    let vocab = logits.dim(2);
    (
        logits.slice(&[0, 0, 0], &[batch, seq_len - 1, vocab]),
        labels.slice(&[0, 1], &[batch, seq_len]),
    )
}

/// Mean cross-entropy over the labels that are not `ignore_index`: a scalar.
///
/// See [`CrossEntropy`].
pub fn cross_entropy_loss(
    logits: &InlineArray,
    labels: &InlineArray,
    ignore_index: i32,
) -> InlineArray {
    CrossEntropy::new()
        .with_ignore_index(ignore_index)
        .mean(logits, labels)
}

/// Mean cross-entropy over every label, for batches with no padding.
pub fn cross_entropy_loss_dense(logits: &InlineArray, labels: &InlineArray) -> InlineArray {
    CrossEntropy::new().mean(logits, labels)
}

/// Per-token cross-entropy with ignored positions zeroed: a flat `[N]` tensor.
pub fn per_token_cross_entropy_loss(
    logits: &InlineArray,
    labels: &InlineArray,
    ignore_index: i32,
) -> InlineArray {
    CrossEntropy::new()
        .with_ignore_index(ignore_index)
        .per_token(logits, labels)
        .0
}

/// Per-token cross-entropy without an ignore index.
pub fn per_token_cross_entropy_loss_dense(
    logits: &InlineArray,
    labels: &InlineArray,
) -> InlineArray {
    CrossEntropy::new().per_token(logits, labels).0
}

/// Causal language-model loss: the logits at `t` against the label at
/// `t + 1`, then [`cross_entropy_loss`].
///
/// - `logits`: `[batch, seq_len, vocab_size]`
/// - `labels`: `[batch, seq_len]`
pub fn causal_lm_loss(
    logits: &InlineArray,
    labels: &InlineArray,
    ignore_index: i32,
) -> InlineArray {
    CrossEntropy::new()
        .with_ignore_index(ignore_index)
        .causal_lm_mean(logits, labels)
}

/// Causal language-model loss without an ignore index.
pub fn causal_lm_loss_dense(logits: &InlineArray, labels: &InlineArray) -> InlineArray {
    CrossEntropy::new().causal_lm_mean(logits, labels)
}

// ── Gradient clipping ─────────────────────────────────────────────────────

/// Clip all gradients in `grads` by the global L2 norm, in place.
///
/// Computes `global_norm = sqrt(sum_i ||grad_i||_F^2)`, then scales every
/// gradient by `min(1, max_norm / global_norm)`.  The computation stays on the
/// GPU — no CPU sync until the caller explicitly evaluates.
///
/// Returns the (lazy) f32 global norm before clipping. Squares are summed in
/// f32 whatever the gradients' dtype (an f16 gradient above 256 would
/// overflow its own square), and each gradient keeps its dtype: a bf16 or f16
/// gradient is scaled in f32 and rounded once.
///
/// A non-finite norm scales every gradient to NaN, as `torch`'s
/// `clip_grad_norm_` does; the caller decides whether to skip such a step.
///
/// Passing `max_norm <= 0.0` is a no-op that returns `0.0`.
pub fn clip_grad_norm(grads: &mut crate::optimizer::ParamSet, max_norm: f32) -> InlineArray {
    clip_grad_norm_map(grads, max_norm)
}

/// Clip all gradients in any `HashMap<K, InlineArray>` by global L2 norm.
///
/// This supports both bridge-native [`ParamSet`](crate::optimizer::ParamSet)
/// and the compat layer's `FlattenedModuleParam` without duplicating the same
/// lazy GPU expression in trainer crates. See [`clip_grad_norm`].
pub fn clip_grad_norm_map<K>(grads: &mut HashMap<K, InlineArray>, max_norm: f32) -> InlineArray
where
    K: Eq + Hash + Ord,
{
    if max_norm <= 0.0 {
        return InlineArray::from_f32(0.0);
    }

    let norm = global_grad_norm_of(grads);

    // scale = max_norm / max(norm, max_norm)  ∈ (0, 1]
    let max_norm_arr = InlineArray::from_f32(max_norm);
    let scale = max_norm_arr.divide(&norm.maximum(&max_norm_arr));

    for grad in grads.values_mut() {
        *grad = scale_in_f32(grad, &scale);
    }

    norm
}

/// The global L2 norm of a map of gradients ([`global_grad_norm`]), summed in
/// key order.
///
/// An f32 sum depends on its order, and a `HashMap` iterates in an order that
/// is different every time one is built, so summing in the map's order gave
/// the same gradients a norm (and a clipping factor) that differed in its last
/// bits from run to run. Key order is the same on every run.
pub fn global_grad_norm_of<K: Ord>(grads: &HashMap<K, InlineArray>) -> InlineArray {
    let mut ordered: Vec<(&K, &InlineArray)> = grads.iter().collect();
    ordered.sort_by(|a, b| a.0.cmp(b.0));
    global_grad_norm(ordered.into_iter().map(|(_, grad)| grad))
}

/// The global L2 norm of a set of gradients, an f32 scalar: the square root
/// of the sum of every element's square, summed in f32, in the order given.
/// For a map, [`global_grad_norm_of`] gives the order.
pub fn global_grad_norm<'a>(grads: impl IntoIterator<Item = &'a InlineArray>) -> InlineArray {
    let f32_ = Dtype::Float32.as_i32();
    let mut norm_sq = InlineArray::from_f32(0.0);
    for grad in grads {
        norm_sq = norm_sq.add(&grad.as_dtype(f32_).square().sum_all());
    }
    norm_sq.sqrt()
}

/// `grad · scale` with `scale` an f32 scalar: in f32, then back to `grad`'s
/// dtype, so a half-precision gradient is rounded once and stays half
/// precision.
fn scale_in_f32(grad: &InlineArray, scale: &InlineArray) -> InlineArray {
    let dtype = grad.dtype_raw();
    let f32_ = Dtype::Float32.as_i32();
    if dtype == f32_ {
        grad.multiply(scale)
    } else {
        grad.as_dtype(f32_).multiply(scale).as_dtype(dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check_unobserved_error;
    use crate::compat::random;

    fn read(a: &InlineArray) -> Vec<f32> {
        let mut a = a.clone();
        a.eval();
        let n = a.size();
        let v = a.to_f32_vec(n).expect("readable");
        check_unobserved_error().expect("no bridge error");
        v
    }

    /// `lse(z) - z[y]` and friends in f64, from the definition.
    struct Reference {
        lse: f64,
        target: f64,
        mean: f64,
    }

    fn reference(row: &[f32], label: usize) -> Reference {
        let m = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
        let sum: f64 = row.iter().map(|z| (*z as f64 - m).exp()).sum();
        Reference {
            lse: sum.ln() + m,
            target: row[label] as f64,
            mean: row.iter().map(|z| *z as f64).sum::<f64>() / row.len() as f64,
        }
    }

    fn token_loss(row: &[f32], label: usize, eps: f64, z_loss: f64) -> f64 {
        let r = reference(row, label);
        (1.0 - eps) * (r.lse - r.target) + eps * (r.lse - r.mean) + z_loss * r.lse * r.lse
    }

    const ROWS: [[f32; 4]; 3] = [
        [1.0, 2.0, 3.0, -1.0],
        [0.0, 0.0, 0.0, 0.0],
        [9.0, -9.0, 4.5, 0.5],
    ];

    fn logits() -> InlineArray {
        let flat: Vec<f32> = ROWS.iter().flatten().copied().collect();
        InlineArray::from_f32_slice(&flat, &[3, 4])
    }

    #[test]
    fn cross_entropy_masks_ignored_labels_before_gather() {
        let logits = InlineArray::from_f32_slice(
            &[
                1.0_f32, 2.0, 3.0, //
                0.0, 0.0, 0.0, //
                9.0, 9.0, 9.0,
            ],
            &[3, 3],
        );
        let labels = InlineArray::from_i32_slice_shaped(&[2, -100, 1], &[3]);

        let per_token = read(&per_token_cross_entropy_loss(&logits, &labels, -100));
        let first = (1.0_f32 + (-1.0_f32).exp() + (-2.0_f32).exp()).ln();
        let third = 3.0_f32.ln();
        assert!(
            (per_token[0] - first).abs() < 1e-5,
            "{} vs {first}",
            per_token[0]
        );
        assert_eq!(per_token[1], 0.0);
        assert!(
            (per_token[2] - third).abs() < 1e-5,
            "{} vs {third}",
            per_token[2]
        );

        let loss = read(&cross_entropy_loss(&logits, &labels, -100))[0];
        let expected = (first + third) / 2.0;
        assert!((loss - expected).abs() < 1e-5, "{loss} vs {expected}");
    }

    /// Label smoothing and the z-loss against their definitions in f64, with
    /// an ignored token that must stay out of the sum and the divisor.
    #[test]
    fn smoothing_and_z_loss_follow_their_definitions() {
        let labels = InlineArray::from_i32_slice_shaped(&[2, -100, 0], &[3]);
        let live = [(0, 2usize), (2, 0usize)];
        for (eps, z) in [(0.0, 0.0), (0.1, 0.0), (0.0, 1e-4), (0.2, 1e-2)] {
            let ce = CrossEntropy::new()
                .with_ignore_index(-100)
                .with_label_smoothing(eps as f32)
                .with_z_loss(z as f32);
            let per_token = read(&ce.per_token(&logits(), &labels).0);
            for &(row, label) in &live {
                let want = token_loss(&ROWS[row], label, eps, z);
                assert!(
                    (per_token[row] as f64 - want).abs() < 1e-4 * (1.0 + want.abs()),
                    "eps {eps} z {z} token {row}: {} vs {want}",
                    per_token[row]
                );
            }
            assert_eq!(per_token[1], 0.0, "an ignored token costs nothing");

            let want_mean = live
                .iter()
                .map(|&(row, label)| token_loss(&ROWS[row], label, eps, z))
                .sum::<f64>()
                / 2.0;
            let mean = read(&ce.mean(&logits(), &labels))[0] as f64;
            assert!(
                (mean - want_mean).abs() < 1e-4 * (1.0 + want_mean.abs()),
                "eps {eps} z {z}: mean {mean} vs {want_mean}"
            );
            let (sum, count) = ce.sum_and_count(&logits(), &labels);
            assert_eq!(read(&count)[0], 2.0);
            let sum = read(&sum)[0] as f64;
            assert!((sum - 2.0 * want_mean).abs() < 2e-4 * (1.0 + want_mean.abs()));
        }
    }

    /// The gradient with respect to the logits is
    /// `(1 + 2λ·lse)·softmax − (1 − ε)·onehot − ε/V`, divided by the count,
    /// and zero on an ignored token.
    #[test]
    fn the_gradient_is_softmax_minus_the_smoothed_target() {
        let (eps, z) = (0.15_f64, 1e-3_f64);
        let labels = InlineArray::from_i32_slice_shaped(&[2, -100, 0], &[3]);
        let ce = CrossEntropy::new()
            .with_ignore_index(-100)
            .with_label_smoothing(eps as f32)
            .with_z_loss(z as f32);
        let (_, grads) = crate::inline_array::value_and_grad(
            |p| ce.mean(&p[0], &p[1]),
            &[logits()],
            std::slice::from_ref(&labels),
        );
        let got = read(&grads[0]);
        for (row, label) in [(0usize, Some(2usize)), (1, None), (2, Some(0))] {
            for j in 0..4 {
                let want = match label {
                    None => 0.0,
                    Some(y) => {
                        let r = reference(&ROWS[row], y);
                        let softmax = (ROWS[row][j] as f64 - r.lse).exp();
                        let onehot = if j == y { 1.0 } else { 0.0 };
                        ((1.0 + 2.0 * z * r.lse) * softmax - (1.0 - eps) * onehot - eps / 4.0) / 2.0
                    }
                };
                let g = got[row * 4 + j] as f64;
                assert!(
                    (g - want).abs() < 1e-5 + 1e-4 * want.abs(),
                    "d/dz[{row}][{j}]: {g} vs {want}"
                );
            }
        }
    }

    /// A bf16 `logsumexp` rounds the normalizer to bf16, which moved single
    /// tokens' losses by about 0.1 nats on a large vocabulary. The loss of
    /// bf16 logits is the loss of the values they hold.
    #[test]
    fn half_precision_logits_lose_nothing_in_the_loss() {
        random::seed(1);
        let (n, v) = (32usize, 100_000usize);
        let z = random::normal(&[n as i32, v as i32], Dtype::Float32)
            .mul_scalar(3.0)
            .as_dtype(Dtype::Bfloat16.as_i32());
        let labels: Vec<i32> = (0..n as i32).map(|i| (i * 2371) % v as i32).collect();
        let labels_arr = InlineArray::from_i32_slice_shaped(&labels, &[n as i32]);
        let got = read(&per_token_cross_entropy_loss_dense(&z, &labels_arr));
        let values = read(&z.as_dtype(Dtype::Float32.as_i32()));
        for t in 0..n {
            let row = &values[t * v..(t + 1) * v];
            let r = reference(row, labels[t] as usize);
            let want = r.lse - r.target;
            assert!(
                (got[t] as f64 - want).abs() < 2e-3,
                "token {t}: {} vs {want}",
                got[t]
            );
        }
    }

    #[test]
    fn a_batch_of_ignored_labels_has_zero_loss_and_zero_gradient() {
        let labels = InlineArray::from_i32_slice_shaped(&[-100, -100, -100], &[3]);
        let ce = CrossEntropy::new()
            .with_ignore_index(-100)
            .with_z_loss(1e-3);
        let (loss, grads) = crate::inline_array::value_and_grad(
            |p| ce.mean(&p[0], &p[1]),
            &[logits()],
            std::slice::from_ref(&labels),
        );
        assert_eq!(read(&loss)[0], 0.0);
        assert!(read(&grads[0]).iter().all(|g| *g == 0.0));
    }

    /// Position `t` is scored against the label at `t + 1`.
    #[test]
    fn the_causal_loss_shifts_labels_by_one() {
        // Two positions over three classes; only position 0's logits count,
        // against label 1.
        let logits = InlineArray::from_f32_slice(&[0.5, 2.0, -1.0, 7.0, 7.0, 7.0], &[1, 2, 3]);
        let labels = InlineArray::from_i32_slice_shaped(&[2, 1], &[1, 2]);
        let want = token_loss(&[0.5, 2.0, -1.0], 1, 0.0, 0.0);
        let got = read(&causal_lm_loss(&logits, &labels, -100))[0] as f64;
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        let got = read(&causal_lm_loss_dense(&logits, &labels))[0] as f64;
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    }

    fn grads_of(dtype: Dtype, values: &[f32]) -> HashMap<&'static str, InlineArray> {
        let g =
            InlineArray::from_f32_slice(values, &[values.len() as i32]).as_dtype(dtype.as_i32());
        HashMap::from([("g", g)])
    }

    /// Clipping scales by `max_norm / norm`, and an f16 gradient above 256
    /// must not overflow the square the norm is made of.
    #[test]
    fn clipping_scales_by_the_global_norm_without_leaving_the_dtype() {
        for dtype in [Dtype::Float32, Dtype::Bfloat16, Dtype::Float16] {
            // ||(300, 400)|| = 500, clipped to 5: a factor of 0.01.
            let mut grads = grads_of(dtype, &[300.0, 400.0]);
            let norm = clip_grad_norm_map(&mut grads, 5.0);
            assert_eq!(norm.dtype(), Dtype::Float32);
            let norm = read(&norm)[0];
            assert!((norm - 500.0).abs() < 1e-2, "{dtype:?}: norm {norm}");
            let g = &grads["g"];
            assert_eq!(g.dtype(), dtype, "clipping changed the gradient's dtype");
            let g = read(&g.as_dtype(Dtype::Float32.as_i32()));
            assert!(
                (g[0] - 3.0).abs() < 0.02 && (g[1] - 4.0).abs() < 0.02,
                "{dtype:?}: {g:?}"
            );
        }
    }

    /// The global norm is a sum of f32 squares, and an f32 sum depends on its
    /// order. A `HashMap` iterates in a different order every time one is
    /// built, so the norm (and the clip it scales by) changed in its last bits
    /// from run to run on the same gradients.
    #[test]
    fn the_global_norm_does_not_depend_on_the_order_a_map_iterates_in() {
        // Magnitudes across many binades, so different orders round differently.
        let values: Vec<f32> = (0..32)
            .map(|i| 1.7f32.powi((i * 7 % 23) - 9) * if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let norms: Vec<u32> = (0..64)
            .map(|_| {
                let grads: HashMap<String, InlineArray> = values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        (
                            format!("layer.{i:02}.weight"),
                            InlineArray::from_f32_slice(&[*v, *v * 0.37], &[2]),
                        )
                    })
                    .collect();
                read(&global_grad_norm_of(&grads))[0].to_bits()
            })
            .collect();
        assert!(
            norms.windows(2).all(|pair| pair[0] == pair[1]),
            "the norm took {} different values: {norms:?}",
            norms.iter().collect::<std::collections::HashSet<_>>().len()
        );
    }

    #[test]
    fn a_gradient_inside_the_budget_is_left_alone() {
        let mut grads = grads_of(Dtype::Float32, &[0.3, 0.4]);
        let norm = clip_grad_norm_map(&mut grads, 5.0);
        assert!((read(&norm)[0] - 0.5).abs() < 1e-6);
        assert_eq!(read(&grads["g"]), vec![0.3, 0.4]);
        let off = clip_grad_norm_map(&mut grads, 0.0);
        assert_eq!(read(&off)[0], 0.0);
    }
}
