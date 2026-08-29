//! Selective log softmax utilities for RL/preference trainers.
//!
//! Computes per-token log probabilities without materializing the full
//! `[batch, seq, vocab]` log_softmax tensor. Uses the identity:
//!
//! ```text
//! log_softmax(x)[i] = x[i] - logsumexp(x)
//! ```
//!
//! to gather the logit at the target index first, then subtract the scalar
//! logsumexp — saving ~4 GB VRAM for 128K-vocab models at typical batch sizes.
//!
//! This is the technique used by DeepSeek V3/R1.

use pmetal_bridge::compat::{Array, Dtype, Exception, nn, ops};

/// Selective log softmax: compute log probabilities only at target indices.
///
/// Equivalent to `log_softmax(logits, -1).take_along_axis(indices, -1)` but
/// avoids materializing the full `[B, S, V]` intermediate tensor.
///
/// # Arguments
/// * `logits` - Model output logits `[B, S, V]`
/// * `labels` - Target label indices `[B, S]`, with `-100` for ignored positions
///
/// # Returns
/// `(per_token_logps, valid_mask)` — both `[B, S]` as `Float32`.
/// Ignored positions (label == -100) get a log-prob of 0.0 and mask of 0.0.
pub fn selective_log_softmax(logits: &Array, labels: &Array) -> Result<(Array, Array), Exception> {
    selective_log_softmax_with_temperature(logits, labels, None)
}

/// Selective log softmax with optional temperature scaling.
///
/// When `temperature` is `Some(t)` with `t != 1.0`, logits are divided by `t`
/// before computing log-softmax. This is used by RL trainers that sample at
/// non-unit temperature and need log-probs under the tempered distribution.
///
/// # Arguments
/// * `logits` - Model output logits `[B, S, V]`
/// * `labels` - Target label indices `[B, S]`, with `-100` for ignored positions
/// * `temperature` - Optional temperature scaling (logits /= temperature)
///
/// # Returns
/// `(per_token_logps, valid_mask)` — both `[B, S]` as `Float32`.
pub fn selective_log_softmax_with_temperature(
    logits: &Array,
    labels: &Array,
    temperature: Option<f32>,
) -> Result<(Array, Array), Exception> {
    // Apply temperature scaling if requested
    let logits = match temperature {
        Some(t) if (t - 1.0).abs() > 1e-8 && t > 0.0 => logits.divide(&Array::from_f32(t)),
        _ => logits.clone(),
    };
    let logits = &logits;
    // Match label dtype for comparisons (labels may be Int32 or Int64)
    let labels_dtype = labels.dtype_raw();
    let zero = Array::from_int(0).as_dtype(labels_dtype);
    let ignore_val = Array::from_int(-100).as_dtype(labels_dtype);

    // Replace -100 with 0 so gather doesn't go out-of-bounds
    let gather_labels = ops::maximum(labels, &zero);

    // [B, S] -> [B, S, 1] for take_along_axis
    let gather_indices = gather_labels.expand_dims(-1i32);

    // Gather the single logit at each target position: [B, S, 1]
    // MLX >= 0.32: label indices must not enter the grad trace
    let gather_indices = gather_indices.stop_gradient();
    let selected_logits = logits.take_along_axis(&gather_indices, -1);

    // logsumexp over vocab dim, keepdims for broadcast: [B, S, 1]
    let lse = logits.logsumexp(-1, true);

    // log_softmax(x)[i] = x[i] - logsumexp(x)  =>  [B, S, 1]
    let log_probs = selected_logits.subtract(&lse);

    // Squeeze back to [B, S]
    let log_probs = log_probs.squeeze(-1);

    // Build valid mask: 1.0 where label != -100, 0.0 otherwise
    let valid_mask = labels.ne(&ignore_val).as_dtype(Dtype::Float32 as i32);

    // Zero out ignored positions so downstream sums are correct
    let log_probs = log_probs.multiply(&valid_mask);

    Ok((log_probs, valid_mask))
}

/// Shift-then-selective-log-softmax primitive shared by DPO / KTO / ORPO /
/// OnlineDPO / GRPO.
///
/// Every preference/RL trainer open-coded this same 2-step pattern:
/// 1. Shift: `pred_logits = logits[:, :-1, :]`, `target_labels = labels[:, 1:]`
///    (next-token prediction — drop the final logit, drop the first label).
/// 2. [`selective_log_softmax_with_temperature`] on the shifted arrays.
///
/// # Arguments
/// * `logits` - Model output logits `[B, S, V]`. `S` must be > 1.
/// * `labels` - Target labels `[B, S]`, `-100` for ignored positions.
/// * `temperature` - Optional temperature scaling (used by GRPO).
///
/// # Returns
/// `(per_token_logps, valid_mask)` — both `[B, S-1]` as `Float32`.
pub fn shifted_selective_log_softmax(
    logits: &Array,
    labels: &Array,
    temperature: Option<f32>,
) -> Result<(Array, Array), Exception> {
    let seq_len = logits.dim(1);
    let pred_logits = logits.index((.., ..seq_len - 1, ..));
    let target_labels = labels.index((.., 1..));
    selective_log_softmax_with_temperature(&pred_logits, &target_labels, temperature)
}

/// Next-token-shifted sum of selective log-probs — the shared canonical shape
/// for DPO / KTO / ORPO / OnlineDPO.
///
/// Sum over the sequence axis after [`shifted_selective_log_softmax`]. Masked
/// positions (label == -100) contribute 0.
///
/// # Arguments
/// * `logits` - Model output logits `[B, S, V]`. `S` must be > 1.
/// * `labels` - Target labels `[B, S]`, `-100` for ignored positions.
pub fn compute_log_probs(logits: &Array, labels: &Array) -> Result<Array, Exception> {
    let (per_token_logps, _valid_mask) = shifted_selective_log_softmax(logits, labels, None)?;
    Ok(per_token_logps.sum_axes(&[1i32], false))
}

/// Same shift + selective-log-softmax as [`compute_log_probs`], returning both
/// the summed log-probs and a length-normalized average.
///
/// The average divides by `max(valid_count, 1.0)` per sample to avoid NaN when
/// every label at a row is `-100`. Used by SimPO and ORPO.
///
/// # Returns
/// `(sum_log_probs, avg_log_probs)` — both `[B]` as `Float32`.
pub fn compute_log_probs_with_avg(
    logits: &Array,
    labels: &Array,
) -> Result<(Array, Array), Exception> {
    let (per_token_logps, valid_mask) = shifted_selective_log_softmax(logits, labels, None)?;
    let token_sum = per_token_logps.sum_axes(&[1i32], false);
    let valid_count_raw = valid_mask
        .as_dtype(Dtype::Float32 as i32)
        .sum_axes(&[1i32], false);
    let valid_count = ops::maximum(&valid_count_raw, &Array::from_f32(1.0));
    let avg = token_sum.divide(&valid_count);
    Ok((token_sum, avg))
}

/// Memory-efficient entropy: `H = -sum(p * log(p))` over the vocab axis.
///
/// Uses the identity `H = logsumexp(x) - sum(softmax(x) * x, axis=-1)` to
/// avoid materializing both `softmax` and `log_softmax`. Only `softmax(x)` is
/// materialized — halving peak VRAM compared to the naive approach.
///
/// # Arguments
/// * `logits` - Logits tensor with vocab as the last dimension `[..., V]`
///
/// # Returns
/// Per-position entropy with the last dimension reduced: `[...]`
pub fn efficient_entropy(logits: &Array) -> Result<Array, Exception> {
    // H = logsumexp(x) - sum(softmax(x) * x, axis=-1)
    let lse = logits.logsumexp(-1, false); // [...] (no keepdims)
    let probs = ops::softmax_axis(logits, -1); // [..., V]
    let weighted = probs.multiply(logits); // [..., V]
    let weighted_sum = weighted.sum_axis(-1, false); // [...]
    Ok(lse.subtract(&weighted_sum))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selective_matches_full_log_softmax() {
        // Small vocab so we can compare against the naive approach
        // logits: [1, 3, 5]  (batch=1, seq=3, vocab=5)
        let logits_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, // position 0
            5.0, 4.0, 3.0, 2.0, 1.0, // position 1
            0.0, 0.0, 0.0, 0.0, 0.0, // position 2
        ];
        let logits = Array::from_slice(&logits_data, &[1, 3, 5]);
        let labels = Array::from_slice(&[2i32, 0, -100], &[1, 3]);

        let (log_probs, mask) = selective_log_softmax(&logits, &labels).unwrap();
        log_probs.eval();
        mask.eval();

        assert_eq!(log_probs.shape(), &[1, 3]);
        assert_eq!(mask.shape(), &[1, 3]);

        // Compare with reference: full log_softmax + gather
        let full_log_softmax = nn::log_softmax(&logits, -1);
        full_log_softmax.eval();

        let lp: &[f32] = log_probs.as_slice();
        let m: &[f32] = mask.as_slice();

        // Position 0: label=2 -> log_softmax(logits[0])[2]
        let ref_lp0 = {
            let row = &logits_data[0..5];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = row.iter().map(|x| (x - max_val).exp()).sum();
            row[2] - max_val - sum_exp.ln()
        };
        assert!(
            (lp[0] - ref_lp0).abs() < 1e-5,
            "pos 0: {} vs {}",
            lp[0],
            ref_lp0
        );
        assert_eq!(m[0], 1.0);

        // Position 1: label=0
        let ref_lp1 = {
            let row = &logits_data[5..10];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = row.iter().map(|x| (x - max_val).exp()).sum();
            row[0] - max_val - sum_exp.ln()
        };
        assert!(
            (lp[1] - ref_lp1).abs() < 1e-5,
            "pos 1: {} vs {}",
            lp[1],
            ref_lp1
        );
        assert_eq!(m[1], 1.0);

        // Position 2: label=-100 -> masked out
        assert_eq!(lp[2], 0.0);
        assert_eq!(m[2], 0.0);
    }

    #[test]
    fn test_selective_all_ignored() {
        let logits = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let labels = Array::from_slice(&[-100i32, -100], &[1, 2]);

        let (log_probs, mask) = selective_log_softmax(&logits, &labels).unwrap();
        log_probs.eval();
        mask.eval();

        let lp: &[f32] = log_probs.as_slice();
        let m: &[f32] = mask.as_slice();

        assert_eq!(lp, &[0.0, 0.0]);
        assert_eq!(m, &[0.0, 0.0]);
    }

    #[test]
    fn test_selective_multi_batch() {
        // [2, 2, 3] logits
        let logits = Array::from_slice(
            &[
                1.0f32, 2.0, 3.0, // batch 0, pos 0
                4.0, 5.0, 6.0, // batch 0, pos 1
                7.0, 8.0, 9.0, // batch 1, pos 0
                10.0, 11.0, 12.0, // batch 1, pos 1
            ],
            &[2, 2, 3],
        );
        let labels = Array::from_slice(&[1i32, 2, 0, -100], &[2, 2]);

        let (log_probs, mask) = selective_log_softmax(&logits, &labels).unwrap();
        log_probs.eval();
        mask.eval();

        assert_eq!(log_probs.shape(), &[2, 2]);
        assert_eq!(mask.shape(), &[2, 2]);

        let m: &[f32] = mask.as_slice();
        assert_eq!(m[0], 1.0);
        assert_eq!(m[1], 1.0);
        assert_eq!(m[2], 1.0);
        assert_eq!(m[3], 0.0); // -100 masked
    }

    #[test]
    fn test_efficient_entropy_matches_naive() {
        // logits: [1, 3, 5]  (batch=1, seq=3, vocab=5)
        let logits_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, // position 0
            5.0, 4.0, 3.0, 2.0, 1.0, // position 1
            0.0, 0.0, 0.0, 0.0, 0.0, // position 2 (uniform)
        ];
        let logits = Array::from_slice(&logits_data, &[1, 3, 5]);

        // Efficient entropy
        let ent = efficient_entropy(&logits).unwrap();
        ent.eval();
        assert_eq!(ent.shape(), &[1, 3]);

        // Naive reference: -sum(softmax(x) * log_softmax(x), axis=-1)
        let log_probs = nn::log_softmax(&logits, -1);
        let probs = log_probs.exp();
        let naive_ent = probs.multiply(&log_probs).sum_axis(-1, false).negative();
        naive_ent.eval();

        let ent_vals: &[f32] = ent.as_slice();
        let naive_vals: &[f32] = naive_ent.as_slice();

        for i in 0..3 {
            assert!(
                (ent_vals[i] - naive_vals[i]).abs() < 1e-5,
                "pos {}: {} vs {}",
                i,
                ent_vals[i],
                naive_vals[i]
            );
        }

        // Uniform logits should have maximum entropy = ln(5)
        let expected_uniform = (5.0_f32).ln();
        assert!(
            (ent_vals[2] - expected_uniform).abs() < 1e-5,
            "uniform: {} vs {}",
            ent_vals[2],
            expected_uniform
        );
    }

    #[test]
    fn test_efficient_entropy_2d() {
        // Test with 2D input [N, V] (used by PPO)
        let logits = Array::from_slice(&[1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0], &[2, 3]);

        let ent = efficient_entropy(&logits).unwrap();
        ent.eval();
        assert_eq!(ent.shape(), &[2]);

        let ent_vals: &[f32] = ent.as_slice();
        // Second row is uniform -> entropy = ln(3)
        let expected = (3.0_f32).ln();
        assert!(
            (ent_vals[1] - expected).abs() < 1e-5,
            "uniform: {} vs {}",
            ent_vals[1],
            expected
        );
    }

    // --- Edge case tests ---

    #[test]
    fn test_selective_single_element_vocab() {
        // Vocab size 1: only one possible token, log_prob must be 0.0
        let logits = Array::from_slice(&[5.0f32, -3.0], &[1, 2, 1]);
        let labels = Array::from_slice(&[0i32, 0], &[1, 2]);

        let (log_probs, mask) = selective_log_softmax(&logits, &labels).unwrap();
        log_probs.eval();
        mask.eval();

        let lp: &[f32] = log_probs.as_slice();
        let m: &[f32] = mask.as_slice();

        // log_softmax with vocab=1 is always 0.0 (log(1) = 0)
        assert!(
            (lp[0]).abs() < 1e-6,
            "single vocab log_prob should be 0, got {}",
            lp[0]
        );
        assert!(
            (lp[1]).abs() < 1e-6,
            "single vocab log_prob should be 0, got {}",
            lp[1]
        );
        assert_eq!(m[0], 1.0);
        assert_eq!(m[1], 1.0);
    }

    #[test]
    fn test_selective_very_large_logits() {
        // Numerical stability: logits in the thousands should not overflow.
        // Use 1e3 range where f32 has adequate precision for differences of 1.0.
        let logits = Array::from_slice(&[1e3_f32, 1e3 + 1.0, 1e3 - 1.0], &[1, 1, 3]);
        let labels = Array::from_slice(&[1i32], &[1, 1]);

        let (log_probs, _mask) = selective_log_softmax(&logits, &labels).unwrap();
        log_probs.eval();

        let lp: &[f32] = log_probs.as_slice();

        // Compare against reference log_softmax
        let ref_lp = nn::log_softmax(&logits, -1);
        ref_lp.eval();
        let ref_vals: &[f32] = ref_lp.as_slice();
        // label=1 → index 1 in the vocab dimension
        assert!(
            (lp[0] - ref_vals[1]).abs() < 1e-5,
            "large logits: selective {} vs full {}",
            lp[0],
            ref_vals[1]
        );
        assert!(lp[0].is_finite(), "result must be finite");

        // Also verify extreme magnitudes produce finite results
        let extreme = Array::from_slice(&[1e7_f32, 1e7 + 1.0, 1e7 - 1.0], &[1, 1, 3]);
        let labels_ext = Array::from_slice(&[0i32], &[1, 1]);
        let (lp_ext, _) = selective_log_softmax(&extreme, &labels_ext).unwrap();
        lp_ext.eval();
        let lp_ext_val: &[f32] = lp_ext.as_slice();
        assert!(
            lp_ext_val[0].is_finite(),
            "extreme large logits must be finite"
        );
        assert!(lp_ext_val[0] <= 0.0, "log_prob must be non-positive");
    }

    #[test]
    fn test_selective_very_small_logits() {
        // Numerical stability: very negative logits should not underflow.
        let logits = Array::from_slice(&[-1e3_f32, -1e3 + 1.0, -1e3 - 1.0], &[1, 1, 3]);
        let labels = Array::from_slice(&[1i32], &[1, 1]);

        let (log_probs, _mask) = selective_log_softmax(&logits, &labels).unwrap();
        log_probs.eval();

        let lp: &[f32] = log_probs.as_slice();

        // Compare against reference log_softmax
        let ref_lp = nn::log_softmax(&logits, -1);
        ref_lp.eval();
        let ref_vals: &[f32] = ref_lp.as_slice();
        assert!(
            (lp[0] - ref_vals[1]).abs() < 1e-5,
            "small logits: selective {} vs full {}",
            lp[0],
            ref_vals[1]
        );
        assert!(lp[0].is_finite(), "result must be finite");

        // Also verify extreme negative magnitudes produce finite results
        let extreme = Array::from_slice(&[-1e7_f32, -1e7 + 1.0, -1e7 - 1.0], &[1, 1, 3]);
        let labels_ext = Array::from_slice(&[0i32], &[1, 1]);
        let (lp_ext, _) = selective_log_softmax(&extreme, &labels_ext).unwrap();
        lp_ext.eval();
        let lp_ext_val: &[f32] = lp_ext.as_slice();
        assert!(
            lp_ext_val[0].is_finite(),
            "extreme small logits must be finite"
        );
        assert!(lp_ext_val[0] <= 0.0, "log_prob must be non-positive");
    }

    #[test]
    fn test_selective_int64_labels() {
        // Labels as Int64 (common when loaded from datasets)
        let logits = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
        let labels_i64 = Array::from_slice(&[2i32, -100], &[1, 2]).as_dtype(Dtype::Int64 as i32);

        let (log_probs, mask) = selective_log_softmax(&logits, &labels_i64).unwrap();
        log_probs.eval();
        mask.eval();

        let lp: &[f32] = log_probs.as_slice();
        let m: &[f32] = mask.as_slice();

        // First position: label=2 should give valid log_prob
        assert!(lp[0] < 0.0, "log_prob should be negative, got {}", lp[0]);
        assert_eq!(m[0], 1.0);

        // Second position: label=-100 should be masked
        assert_eq!(lp[1], 0.0);
        assert_eq!(m[1], 0.0);
    }

    #[test]
    fn test_selective_label_zero_at_ignored_position() {
        // Verify that legitimate label=0 works correctly AND that -100 positions
        // produce 0 even though the gather clamps to index 0
        let logits = Array::from_slice(
            &[
                10.0f32, 1.0, 1.0, // pos 0: logit at index 0 is dominant
                10.0, 1.0, 1.0, // pos 1: same logits, but -100 → masked
            ],
            &[1, 2, 3],
        );
        let labels = Array::from_slice(&[0i32, -100], &[1, 2]);

        let (log_probs, mask) = selective_log_softmax(&logits, &labels).unwrap();
        log_probs.eval();
        mask.eval();

        let lp: &[f32] = log_probs.as_slice();
        let m: &[f32] = mask.as_slice();

        // Position 0: label=0 should give a log_prob close to 0 (dominant logit)
        assert!(
            lp[0] < 0.0 && lp[0] > -1.0,
            "label=0 log_prob should be near 0, got {}",
            lp[0]
        );
        assert_eq!(m[0], 1.0);

        // Position 1: label=-100 → must be exactly 0, even though gather reads index 0
        assert_eq!(lp[1], 0.0, "ignored position must produce exactly 0.0");
        assert_eq!(m[1], 0.0);
    }

    #[test]
    fn test_entropy_single_element_vocab() {
        // Vocab size 1: entropy must be exactly 0 (no uncertainty)
        let logits = Array::from_slice(&[42.0f32, -7.0], &[1, 2, 1]);

        let ent = efficient_entropy(&logits).unwrap();
        ent.eval();

        let ent_vals: &[f32] = ent.as_slice();
        assert!(
            ent_vals[0].abs() < 1e-6,
            "single vocab entropy should be 0, got {}",
            ent_vals[0]
        );
        assert!(
            ent_vals[1].abs() < 1e-6,
            "single vocab entropy should be 0, got {}",
            ent_vals[1]
        );
    }

    #[test]
    fn test_entropy_extreme_peaked_distribution() {
        // One logit vastly larger than the rest → near-zero entropy (near one-hot)
        let logits = Array::from_slice(&[100.0f32, 0.0, 0.0, 0.0, 0.0], &[1, 1, 5]);

        let ent = efficient_entropy(&logits).unwrap();
        ent.eval();

        let ent_vals: &[f32] = ent.as_slice();
        // softmax([100, 0, 0, 0, 0]) ≈ [1, 0, 0, 0, 0] → entropy ≈ 0
        assert!(
            ent_vals[0] < 1e-4,
            "peaked distribution entropy should be near 0, got {}",
            ent_vals[0]
        );
        assert!(ent_vals[0] >= 0.0, "entropy must be non-negative");
    }

    #[test]
    fn test_compute_log_probs_matches_manual_shift() {
        // logits [1, 3, 4], labels [1, 3]. After shift: pred_logits [1, 2, 4],
        // target_labels [1, 2].
        let logits = Array::from_slice(
            &[
                1.0f32, 2.0, 3.0, 4.0, // pos 0 (dropped after shift)
                5.0, 6.0, 7.0, 8.0, // pos 1 (pred for pos 2)
                9.0, 10.0, 11.0, 12.0, // pos 2 (pred for pos 3)
                13.0, 14.0, 15.0, 16.0, // pos 3 (dropped — no next label)
            ],
            &[1, 4, 4],
        );
        let labels = Array::from_slice(&[0i32, 2, 3, 1], &[1, 4]);

        let summed = super::compute_log_probs(&logits, &labels).unwrap();
        summed.eval();
        assert_eq!(summed.shape(), &[1]);

        // Manual path
        let pred_logits = logits.index((.., ..3, ..));
        let target_labels = labels.index((.., 1..));
        let (per_token, _) = super::selective_log_softmax(&pred_logits, &target_labels).unwrap();
        let expected = per_token.sum_axes(&[1i32], false);
        expected.eval();

        let s: &[f32] = summed.as_slice();
        let e: &[f32] = expected.as_slice();
        assert!((s[0] - e[0]).abs() < 1e-5, "{} vs {}", s[0], e[0]);
    }

    #[test]
    fn test_compute_log_probs_with_avg_normalizes() {
        // 2-sample batch: first has 2 valid tokens, second has 1 (rest masked).
        let logits = Array::from_slice(
            &[
                // batch 0
                1.0f32, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
                // batch 1
                1.0, 1.0, 1.0, //
                2.0, 2.0, 2.0, //
                3.0, 3.0, 3.0, //
            ],
            &[2, 3, 3],
        );
        // After shift, target_labels = labels[:, 1..]  -> shape [2, 2]
        // batch 0 labels -> [1, 2] (both valid)
        // batch 1 labels -> [0, -100] (one valid)
        let labels = Array::from_slice(&[0i32, 1, 2, 0, 0, -100], &[2, 3]);

        let (sum, avg) = super::compute_log_probs_with_avg(&logits, &labels).unwrap();
        sum.eval();
        avg.eval();
        assert_eq!(sum.shape(), &[2]);
        assert_eq!(avg.shape(), &[2]);

        let s: &[f32] = sum.as_slice();
        let a: &[f32] = avg.as_slice();
        // batch 0: 2 valid → avg = sum / 2
        assert!(
            (a[0] - s[0] / 2.0).abs() < 1e-5,
            "batch0 avg mismatch: sum={} avg={}",
            s[0],
            a[0]
        );
        // batch 1: 1 valid → avg == sum
        assert!(
            (a[1] - s[1]).abs() < 1e-5,
            "batch1 avg mismatch: sum={} avg={}",
            s[1],
            a[1]
        );
    }

    #[test]
    fn test_compute_log_probs_with_avg_all_masked_no_nan() {
        // Entire sample masked after shift — denominator must clamp to 1.0.
        let logits = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let labels = Array::from_slice(&[-100i32, -100], &[1, 2]);

        let (sum, avg) = super::compute_log_probs_with_avg(&logits, &labels).unwrap();
        sum.eval();
        avg.eval();
        let s: &[f32] = sum.as_slice();
        let a: &[f32] = avg.as_slice();
        assert_eq!(s[0], 0.0);
        assert_eq!(a[0], 0.0);
        assert!(a[0].is_finite(), "avg must not be NaN when all masked");
    }

    #[test]
    fn test_entropy_large_logits_stability() {
        // Numerical stability: all logits large but with small differences.
        // Use 1e3 range for adequate f32 precision, then verify extreme ranges are finite.
        let logits = Array::from_slice(&[1e3_f32, 1e3 + 0.5, 1e3 - 0.5], &[1, 1, 3]);

        let ent = efficient_entropy(&logits).unwrap();
        ent.eval();

        // Compare against naive entropy
        let naive_log_probs = nn::log_softmax(&logits, -1);
        let naive_probs = naive_log_probs.exp();
        let naive_ent = naive_probs
            .multiply(&naive_log_probs)
            .sum_axis(-1, false)
            .negative();
        naive_ent.eval();

        let ent_vals: &[f32] = ent.as_slice();
        let naive_vals: &[f32] = naive_ent.as_slice();

        assert!(
            (ent_vals[0] - naive_vals[0]).abs() < 1e-4,
            "large logits entropy: efficient {} vs naive {}",
            ent_vals[0],
            naive_vals[0]
        );
        assert!(ent_vals[0].is_finite(), "entropy must be finite");
        assert!(
            ent_vals[0] > 0.0,
            "entropy must be positive for non-degenerate distribution"
        );

        // Verify extreme magnitudes produce finite non-negative results
        let extreme = Array::from_slice(&[1e7_f32, 1e7 + 1.0, 1e7 - 1.0], &[1, 1, 3]);
        let ent_ext = efficient_entropy(&extreme).unwrap();
        ent_ext.eval();
        let ent_ext_vals: &[f32] = ent_ext.as_slice();
        assert!(
            ent_ext_vals[0].is_finite(),
            "extreme entropy must be finite"
        );
        assert!(ent_ext_vals[0] >= 0.0, "entropy must be non-negative");
    }
}
