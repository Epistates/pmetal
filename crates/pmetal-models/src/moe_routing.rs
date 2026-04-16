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
//! Note: Qwen3-Next uses an equivalent-but-sign-flipped variant
//! (`argpartition(-scores, -k)`) that's not covered by this helper — left
//! for a follow-up once its semantics are audited. DeepSeek uses the
//! `noaux_tc` topk method which is structurally different and out of
//! scope here.
//!
//! ## LOC accounting
//!
//! Before: Qwen3-MoE and GptOss each had a ~12 LOC copy-paste of the
//! argpartition → take → normalize pattern. After: one helper + 1-line
//! call sites.

use pmetal_bridge::compat::{Array, Exception, ops};

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
    let top_indices = ops::slice_last_from(&part_indices, neg_k).as_type::<i32>();
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

/// DeepSeek-V3 "noaux_tc" top-k routing with bias-corrected score selection.
///
/// Differs from [`topk_normalize`] in three ways that are specific to the
/// DeepSeek-V3 routing algorithm (see the arxiv:2412.19437 paper, §2.1.2
/// "Auxiliary-Loss-Free Load Balancing"):
///
/// 1. Expert **indices** are selected from `scores + e_score_correction_bias`
///    — the bias is an auxiliary-loss-free routing stability trick — but
///    **weights** are gathered from the raw `scores` without the bias.
/// 2. Renormalisation only applies when `top_k > 1` (a single expert
///    already owns full weight by definition).
/// 3. The final weights are multiplied by a `routed_scaling_factor`
///    hyperparameter that matches the training-time expert activation
///    balance.
///
/// * `scores` — per-expert post-activation scores. DeepSeek-V3 uses
///   sigmoid. Shape `[N, num_experts]`.
/// * `e_score_correction_bias` — `[num_experts]`, added to `scores` only
///   when selecting top-k indices.
/// * `top_k` — number of experts per token (must be > 0).
/// * `norm_topk_prob` — renormalise the gathered weights to sum to 1.
///   Applied only when `top_k > 1`.
/// * `routed_scaling_factor` — final multiplier on the gathered weights.
///
/// Returns `(top_indices, top_weights)` with shapes `[N, top_k]`, indices
/// as `i32`.
pub fn noaux_tc_topk(
    scores: &Array,
    e_score_correction_bias: &Array,
    top_k: i32,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
) -> Result<(Array, Array), Exception> {
    if top_k <= 0 {
        return Err(Exception::custom(format!(
            "noaux_tc_topk: top_k must be positive, got {top_k}"
        )));
    }
    let neg_k = -top_k;

    // Select top-k from bias-corrected scores …
    let scores_with_bias = scores.add(e_score_correction_bias);
    let part_indices = ops::argpartition_axis(&scores_with_bias, neg_k, -1);
    let top_indices = ops::slice_last_from(&part_indices, neg_k).as_type::<i32>();

    // … but gather weights from the ORIGINAL scores (no bias).
    let top_weights = scores.take_along_axis(&top_indices, -1);

    let normalized = if norm_topk_prob && top_k > 1 {
        top_weights.divide(&top_weights.sum_axis(-1, true))
    } else {
        top_weights
    };

    Ok((
        top_indices,
        normalized.multiply(&Array::from_f32(routed_scaling_factor)),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::Array;

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
    fn rejects_nonpositive_top_k() {
        let scores = Array::from_slice(&[0.1, 0.2], &[1, 2]);
        assert!(topk_normalize(&scores, 0, false).is_err());
        assert!(topk_normalize(&scores, -1, true).is_err());
    }

    // ── noaux_tc_topk tests ────────────────────────────────────────────

    #[test]
    fn noaux_tc_gathers_from_raw_scores_not_bias_corrected() {
        // Row: raw scores [0.2, 0.3, 0.1, 0.4]; bias [+10, 0, 0, 0].
        // Bias-corrected → top-2 by position: [0] (10.2) and [3] (0.4).
        // Raw gathered weights: scores[0]=0.2 and scores[3]=0.4 → sum 0.6.
        // (Without the bias, top-2 would be [1, 3] with sum 0.3+0.4=0.7.)
        let scores = Array::from_slice(&[0.2, 0.3, 0.1, 0.4], &[1, 4]);
        let bias = Array::from_slice(&[10.0, 0.0, 0.0, 0.0], &[4]);

        let (_, weights) = noaux_tc_topk(&scores, &bias, 2, false, 1.0).unwrap();
        let mut w = weights.clone();
        let sel = w.to_f32_vec(2).unwrap();
        let sum: f32 = sel.iter().sum();
        assert!(
            (sum - 0.6).abs() < 1e-5,
            "bias-corrected indices → raw weights: expected sum 0.6, got {sum}"
        );
    }

    #[test]
    fn noaux_tc_applies_routed_scaling_factor() {
        let scores = Array::from_slice(&[0.2, 0.3, 0.1, 0.4], &[1, 4]);
        let bias = Array::from_slice(&[0.0, 0.0, 0.0, 0.0], &[4]);

        let (_, weights) = noaux_tc_topk(&scores, &bias, 2, false, 2.5).unwrap();
        let mut w = weights.clone();
        let sel = w.to_f32_vec(2).unwrap();
        let sum: f32 = sel.iter().sum();
        // top-2 sum is 0.4 + 0.3 = 0.7, × scaling 2.5 → 1.75.
        assert!(
            (sum - 1.75).abs() < 1e-5,
            "routed_scaling_factor 2.5 should scale top-2 sum 0.7 to 1.75, got {sum}"
        );
    }

    #[test]
    fn noaux_tc_skips_normalise_when_top_k_is_one() {
        let scores = Array::from_slice(&[0.2, 0.3, 0.1, 0.4], &[1, 4]);
        let bias = Array::from_slice(&[0.0, 0.0, 0.0, 0.0], &[4]);

        // norm_topk_prob=true but top_k=1 → no normalisation per DeepSeek.
        let (_, weights) = noaux_tc_topk(&scores, &bias, 1, true, 1.0).unwrap();
        let mut w = weights.clone();
        let sel = w.to_f32_vec(1).unwrap();
        // Single weight should be the raw 0.4 (top), not 1.0.
        assert!(
            (sel[0] - 0.4).abs() < 1e-5,
            "top-1 with norm_topk_prob should keep raw value 0.4, got {}",
            sel[0]
        );
    }

    #[test]
    fn noaux_tc_rejects_nonpositive_top_k() {
        let scores = Array::from_slice(&[0.1, 0.2], &[1, 2]);
        let bias = Array::from_slice(&[0.0, 0.0], &[2]);
        assert!(noaux_tc_topk(&scores, &bias, 0, false, 1.0).is_err());
        assert!(noaux_tc_topk(&scores, &bias, -1, true, 1.0).is_err());
    }
}
