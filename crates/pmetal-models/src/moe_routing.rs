//! Shared routing helpers for Mixture-of-Experts architectures.
//!
//! Most pmetal MoE blocks (Qwen3-MoE, GptOss, …) compute expert weights the
//! same way after the activation step:
//!
//! ```text
//!   part_indices = argpartition_axis(scores, -k, -1)   // O(E) top-k
//!   top_indices  = slice_last_from(part_indices, -k).as_i32()
//!   top_weights  = scores.take_along_axis(top_indices, -1)
//!   weights      = if norm { top_weights / max(sum(top_weights, -1), eps) }
//!                  else    { top_weights }
//! ```
//!
//! They differ only in the activation used to turn raw gate logits into
//! scores (softmax vs. sigmoid). [`topk_normalize`] takes already-activated
//! scores so each architecture keeps its activation choice local while
//! sharing the selection + normalisation path.
//!
//! Note: Qwen3-Next, Qwen3-MoE, Gemma 4 and Granite route through this
//! helper. The earlier sign-flipped `argpartition(-scores, -k)[..., -k:]`
//! variant was anti-top-k (selects the k *smallest*) and has been removed —
//! see the `sign_flipped_argpartition_is_anti_topk` test below. DeepSeek's
//! router (bias-corrected, group-limited) is
//! [`DeepseekRouting`](pmetal_runtime::deepseek_native::DeepseekRouting), one
//! implementation for the shared path and the native engine; GPT-OSS's (softmax
//! over the chosen logits) is `pmetal_runtime::gpt_oss_native::route_logits`.

use pmetal_runtime::compat::{Array, Exception, ops};

/// Numerical floor for the `sum(top_weights)` denominator when
/// `norm_topk_prob` is true. Matches the value used in-line across pmetal
/// MoE blocks.
const TOPK_NORM_EPS: f32 = 1e-8;

/// Top-k selection + optional renormalisation, shared across MoE variants.
///
/// * `scores` — per-expert scores, already activation-applied (softmax or
///   sigmoid). Shape `[N, num_experts]`.
/// * `top_k` — number of experts to select per token (must be > 0 and
///   ≤ `num_experts`).
/// * `norm_topk_prob` — if true, divides each token's k weights by their
///   sum so they add to 1.
///
/// Returns `(top_indices, top_weights)` — both shape `[N, top_k]`, indices
/// as `i32`.
///
/// Fails with an `Exception` if `top_k <= 0` (a degenerate configuration
/// would otherwise produce an invalid slice range and crash in MLX).
pub fn topk_normalize(
    scores: &Array,
    top_k: i32,
    norm_topk_prob: bool,
) -> Result<(Array, Array), Exception> {
    if top_k <= 0 {
        return Err(Exception::custom(format!(
            "topk_normalize: top_k must be positive, got {top_k}"
        )));
    }
    let neg_k = -top_k;

    // argpartition with negative pivot places the k largest values at the
    // tail of the axis. We then slice them and cast to the i32 index
    // dtype expected by `take_along_axis`.
    let part_indices = ops::argpartition_axis(scores, neg_k, -1);
    // MLX >= 0.32 raises "[gather] Cannot calculate VJP with respect to
    // indices" when routing indices sit downstream of differentiable scores
    // inside a grad trace — cut it here so gradients flow only through the
    // gathered weights, never the index path.
    let top_indices =
        ops::stop_gradient(&ops::slice_last_from(&part_indices, neg_k).as_type::<i32>());
    let top_weights = scores.take_along_axis(&top_indices, -1);

    let normalized_weights = if norm_topk_prob {
        let weight_sum = top_weights.sum_axis(-1, true);
        let safe_sum = ops::maximum(&weight_sum, &Array::from_f32(TOPK_NORM_EPS));
        top_weights.divide(&safe_sum)
    } else {
        top_weights
    };

    Ok((top_indices, normalized_weights))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_runtime::compat::Array;

    #[test]
    fn topk_picks_largest_and_preserves_shape() {
        // scores row 0 has expert 3 biggest, then 1, then 0, 2.
        // scores row 1 has expert 0 biggest, then 2, then 1, 3.
        let scores = Array::from_slice(&[0.2, 0.3, 0.1, 0.4, 0.5, 0.15, 0.3, 0.05], &[2, 4]);
        let (indices, weights) = topk_normalize(&scores, 2, false).unwrap();
        assert_eq!(indices.shape(), &[2, 4 - 2]);
        assert_eq!(weights.shape(), &[2, 4 - 2]);
        // Shapes correct; exact indices depend on argpartition's internal
        // ordering (unsorted among the top-k), so we don't assert specific
        // positions — just that the sum of selected weights matches the
        // known top-2 sum per row.
        let mut w = weights.clone();
        let selected = w.to_f32_vec(4).unwrap();
        let row0_sum: f32 = selected[0] + selected[1];
        let row1_sum: f32 = selected[2] + selected[3];
        // Row 0: top 2 values are 0.4 + 0.3 = 0.7.
        assert!(
            (row0_sum - 0.7).abs() < 1e-5,
            "row 0 top-2 sum = {row0_sum}"
        );
        // Row 1: top 2 values are 0.5 + 0.3 = 0.8.
        assert!(
            (row1_sum - 0.8).abs() < 1e-5,
            "row 1 top-2 sum = {row1_sum}"
        );
    }

    #[test]
    fn topk_normalize_when_requested() {
        let scores = Array::from_slice(&[0.2, 0.3, 0.1, 0.4], &[1, 4]);
        let (_, weights) = topk_normalize(&scores, 2, true).unwrap();
        let mut w = weights.clone();
        let sel = w.to_f32_vec(2).unwrap();
        let sum: f32 = sel.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "normalized top-k should sum to 1, got {sum}"
        );
    }

    #[test]
    fn topk_no_normalize_preserves_magnitudes() {
        let scores = Array::from_slice(&[0.2, 0.3, 0.1, 0.4], &[1, 4]);
        let (_, weights) = topk_normalize(&scores, 2, false).unwrap();
        let mut w = weights.clone();
        let sel = w.to_f32_vec(2).unwrap();
        let sum: f32 = sel.iter().sum();
        // Unnormalised — sum should be 0.4 + 0.3 = 0.7, NOT 1.0.
        assert!(
            (sum - 0.7).abs() < 1e-5,
            "unnormalised top-k sum should be 0.7, got {sum}"
        );
    }

    #[test]
    fn sign_flipped_argpartition_is_anti_topk() {
        // Documents why Qwen3-Next's pre-migration pattern
        // `argpartition(-scores, -k, -1)[..., -k:]` was a latent bug:
        // it selects the k LARGEST of `-scores` = k SMALLEST of scores.
        //
        // The correct sign-flip pattern (as in the reference Hunyuan/Llama 4 routing) uses
        // a positive kth and slices from the front:
        //   argpartition(-scores, k-1, -1)[..., :k]
        //
        // This test exists to guard against accidentally reintroducing the
        // buggy combination — if argpartition semantics ever changed such
        // that the bug became a correct form, this test would flag it.
        use pmetal_runtime::compat::ops;

        let scores = Array::from_slice(&[0.2f32, 0.3, 0.1, 0.4], &[1, 4]);
        let neg_k = -2_i32;

        // Buggy pattern (what Qwen3-Next used to have):
        let part = ops::argpartition_axis(&scores.negative(), neg_k, -1);
        let top = ops::slice_last_from(&part, neg_k).as_type::<i32>();
        let w = scores.take_along_axis(&top, -1);
        let mut w_copy = w.clone();
        let vals = w_copy.to_f32_vec(2).unwrap();
        let sum: f32 = vals.iter().sum();
        // Buggy form picks the 2 SMALLEST: 0.1 + 0.2 = 0.3.
        assert!(
            (sum - 0.3).abs() < 1e-5,
            "sign-flipped form is anti-top-k: selected sum = {sum} (expected 0.3 = 0.1+0.2)"
        );

        // Correct forms via topk_normalize: 0.4 + 0.3 = 0.7.
        let (_, weights) = topk_normalize(&scores, 2, false).unwrap();
        let mut wc = weights.clone();
        let correct: f32 = wc.to_f32_vec(2).unwrap().iter().sum();
        assert!(
            (correct - 0.7).abs() < 1e-5,
            "correct top-k sum = {correct} (expected 0.7 = 0.4+0.3)"
        );
    }

    #[test]
    fn rejects_nonpositive_top_k() {
        let scores = Array::from_slice(&[0.1, 0.2], &[1, 2]);
        assert!(topk_normalize(&scores, 0, false).is_err());
        assert!(topk_normalize(&scores, -1, true).is_err());
    }
}
