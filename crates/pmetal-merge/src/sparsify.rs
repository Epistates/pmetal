//! Sparsification of task vectors.
//!
//! Merging methods that combine task vectors first thin each one out so the
//! entries that survive interfere less. The strategies here differ in how
//! they choose what survives:
//!
//! - [`sparsify_by_magnitude`]: keep the largest-magnitude entries (the trim
//!   step of TIES, Yadav et al. 2023).
//! - [`sparsify_dare`]: drop entries uniformly at random and rescale the rest
//!   by `1 / density` (DARE, Yu et al. 2024).
//! - [`sparsify_della`]: drop at random with a probability that falls with
//!   magnitude rank inside each row, rescaling each survivor by its own keep
//!   probability (MAGPRUNE of DELLA, Deep et al. 2024).
//! - [`sparsify_breadcrumbs`]: keep the middle of the magnitude range, masking
//!   both the smallest entries and the largest outliers (Model Breadcrumbs,
//!   Davari and Belilovsky 2024).
//!
//! All of them run as MLX operations: thresholds come from a top-k
//! selection on the device and random masks from MLX's keyed generator, so
//! a tensor never takes a round trip through host memory and a seed fixes
//! the mask.

use crate::{MergeError, Result};
use pmetal_runtime::compat::{Array, Dtype};

fn f32_code() -> i32 {
    Dtype::Float32.as_i32()
}

/// Reject a density outside `[0, 1]` (or NaN) with a message naming it.
pub fn validate_density(density: f32) -> Result<()> {
    if (0.0..=1.0).contains(&density) {
        Ok(())
    } else {
        Err(MergeError::InvalidConfig(format!(
            "density must be in [0.0, 1.0], got {density}"
        )))
    }
}

/// How many of `n` entries a `density` fraction keeps: the nearest integer,
/// and at least one for a positive density.
fn keep_count(density: f32, n: usize) -> usize {
    let exact = f64::from(density) * n as f64;
    (exact.round() as usize).clamp(usize::from(density > 0.0), n)
}

fn zeros_like_f32(tensor: &Array) -> Array {
    Array::zeros(tensor.shape(), f32_code())
}

/// The `keep`-th largest value of `abs_flat` (`1 <= keep <= len`).
fn kth_largest(abs_flat: &Array, keep: usize) -> Result<f32> {
    let k = i32::try_from(keep)
        .map_err(|_| MergeError::InvalidConfig(format!("tensor too large to select from: {keep}")))?;
    Ok(abs_flat.topk(k, 0).min(None).item_f32())
}

/// Keep the top `density` fraction of `tensor` by magnitude and zero the rest.
///
/// Entries tied with the threshold are all kept, so a tensor with many equal
/// magnitudes can keep slightly more than `density`; zeros stay zero either
/// way.
pub fn sparsify_by_magnitude(tensor: &Array, density: f32) -> Result<Array> {
    validate_density(density)?;
    if density >= 1.0 {
        return Ok(tensor.clone());
    }
    if density <= 0.0 {
        return Ok(zeros_like_f32(tensor));
    }
    let abs = tensor.abs_val().reshape(&[-1]);
    let keep = keep_count(density, abs.size());
    let threshold = kth_largest(&abs, keep)?;
    let mask = tensor.abs_val().greater_equal(&Array::from_f32(threshold));
    Ok(mask.where_cond(tensor, &zeros_like_f32(tensor)))
}

/// Model Breadcrumbs' mask: zero the smallest `1 - density` fraction by
/// magnitude and the largest `gamma` fraction (the outliers), keep the rest.
///
/// # Errors
/// `density` or `gamma` outside `[0, 1]`, or `gamma > density`.
pub fn sparsify_breadcrumbs(tensor: &Array, density: f32, gamma: f32) -> Result<Array> {
    validate_density(density)?;
    if !(0.0..=1.0).contains(&gamma) || gamma > density {
        return Err(MergeError::InvalidConfig(format!(
            "gamma must be in [0.0, density={density}], got {gamma}"
        )));
    }
    if density <= 0.0 {
        return Ok(zeros_like_f32(tensor));
    }
    let abs = tensor.abs_val();
    let flat = abs.reshape(&[-1]);
    let n = flat.size();
    let mut mask = if density >= 1.0 {
        None
    } else {
        let floor = kth_largest(&flat, keep_count(density, n))?;
        Some(abs.greater_equal(&Array::from_f32(floor)))
    };
    let outliers = keep_count(gamma, n);
    if gamma > 0.0 && outliers > 0 {
        let ceiling = kth_largest(&flat, outliers)?;
        let below = abs.less(&Array::from_f32(ceiling));
        mask = Some(match mask {
            Some(m) => m.multiply(&below),
            None => below,
        });
    }
    Ok(match mask {
        Some(m) => m.where_cond(tensor, &zeros_like_f32(tensor)),
        None => tensor.clone(),
    })
}

/// DARE: drop each entry independently with probability `1 - density` and,
/// with `rescale`, multiply the survivors by `1 / density` so the expected
/// value is unchanged.
///
/// The mask is a function of `seed` and the tensor's shape only.
pub fn sparsify_dare(tensor: &Array, density: f32, seed: u64, rescale: bool) -> Result<Array> {
    validate_density(density)?;
    if density >= 1.0 {
        return Ok(tensor.clone());
    }
    if density <= 0.0 {
        return Ok(zeros_like_f32(tensor));
    }
    let key = Array::random_key(seed);
    let draw = Array::random_uniform_keyed(tensor.shape(), f32_code(), Some(&key));
    let keep = draw.less(&Array::from_f32(density));
    let scaled = if rescale {
        tensor.multiply(&Array::from_f32(1.0 / density))
    } else {
        tensor.clone()
    };
    Ok(keep.where_cond(&scaled, &zeros_like_f32(tensor)))
}

/// DELLA's MAGPRUNE: drop entries at random with a probability that depends
/// on the entry's magnitude rank within its row.
///
/// Rows are the leading axis (a 1-D tensor is one row; trailing axes are
/// flattened into the row). Within a row the keep probability rises linearly
/// with magnitude rank from `density - epsilon` for the smallest entry to
/// `density + epsilon` for the largest, so small updates are dropped more
/// often and the average keep rate stays `density`. With `rescale` each
/// survivor is divided by its own keep probability, which keeps the expected
/// value of every entry unchanged.
///
/// The mask is a function of `seed` and the tensor's shape only.
///
/// # Errors
/// `density` outside `[0, 1]`, or a negative or non-finite `epsilon`.
pub fn sparsify_della(
    tensor: &Array,
    density: f32,
    epsilon: f32,
    seed: u64,
    rescale: bool,
) -> Result<Array> {
    validate_density(density)?;
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(MergeError::InvalidConfig(format!(
            "DELLA epsilon must be finite and non-negative, got {epsilon}"
        )));
    }
    if density >= 1.0 {
        return Ok(tensor.clone());
    }
    if density <= 0.0 {
        return Ok(zeros_like_f32(tensor));
    }

    let shape = tensor.shape().to_vec();
    let rows = if shape.len() >= 2 { shape[0] } else { 1 };
    let cols = tensor.size() as i32 / rows.max(1);
    let matrix = tensor.reshape(&[rows, cols]);

    // The magnitude rank of every entry within its row: sorting the sort
    // order yields the inverse permutation.
    let order = matrix.abs_val().argsort(1);
    let rank = order.argsort(1).as_dtype(f32_code());
    let position = if cols > 1 {
        rank.divide(&Array::from_f32((cols - 1) as f32))
    } else {
        Array::from_f32(0.5)
    };
    let keep_prob = position
        .multiply(&Array::from_f32(2.0 * epsilon))
        .add(&Array::from_f32(density - epsilon))
        .clip(Some(&Array::from_f32(0.0)), Some(&Array::from_f32(1.0)));

    let key = Array::random_key(seed);
    let draw = Array::random_uniform_keyed(&[rows, cols], f32_code(), Some(&key));
    let keep = draw.less(&keep_prob);
    let survivors = if rescale {
        // A kept entry has keep_prob > 0 (draw < 0 never happens), so the
        // guard only keeps the dropped lanes finite.
        let safe = keep_prob.maximum(&Array::from_f32(f32::MIN_POSITIVE));
        matrix.divide(&safe)
    } else {
        matrix
    };
    let zeros = Array::zeros(&[rows, cols], f32_code());
    Ok(keep.where_cond(&survivors, &zeros).reshape(&shape))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{arr, assert_close, values};

    #[test]
    fn magnitude_keeps_the_largest_entries() {
        let t = arr(&[-4.0, 1.0, -2.0, 3.0], &[4]);
        let out = sparsify_by_magnitude(&t, 0.5).unwrap();
        assert_close(&values(&out), &[-4.0, 0.0, 0.0, 3.0], 0.0);
    }

    #[test]
    fn magnitude_density_one_and_zero_are_identity_and_zeros() {
        let t = arr(&[1.0, -2.0, 3.0, 4.0], &[2, 2]);
        let all = sparsify_by_magnitude(&t, 1.0).unwrap();
        assert_close(&values(&all), &[1.0, -2.0, 3.0, 4.0], 0.0);
        let none = sparsify_by_magnitude(&t, 0.0).unwrap();
        assert_close(&values(&none), &[0.0; 4], 0.0);
        assert_eq!(none.shape(), &[2, 2]);
    }

    #[test]
    fn magnitude_rounds_the_kept_count_to_nearest() {
        // 0.3 of 10 entries is 3 (not 2 through a float-ceil artefact).
        let t = arr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[10]);
        let out = values(&sparsify_by_magnitude(&t, 0.3).unwrap());
        assert_eq!(out.iter().filter(|v| **v != 0.0).count(), 3);
        assert_close(&out[7..], &[8.0, 9.0, 10.0], 0.0);
    }

    #[test]
    fn invalid_densities_are_rejected() {
        let t = arr(&[1.0], &[1]);
        assert!(sparsify_by_magnitude(&t, 1.5).is_err());
        assert!(sparsify_by_magnitude(&t, -0.1).is_err());
        assert!(sparsify_by_magnitude(&t, f32::NAN).is_err());
        assert!(sparsify_dare(&t, 2.0, 0, true).is_err());
        assert!(sparsify_della(&t, 2.0, 0.1, 0, true).is_err());
        assert!(sparsify_della(&t, 0.5, -0.1, 0, true).is_err());
    }

    #[test]
    fn breadcrumbs_masks_both_tails_by_magnitude() {
        // Magnitudes 0.1 .. 1.0. density 0.8 drops the smallest two (0.1,
        // 0.2); gamma 0.1 drops the largest one (1.0).
        let t = arr(&[0.5, -1.0, 0.3, 0.8, 0.1, 0.6, 0.9, -0.4, 0.2, 0.7], &[10]);
        let out = sparsify_breadcrumbs(&t, 0.8, 0.1).unwrap();
        assert_close(
            &values(&out),
            &[0.5, 0.0, 0.3, 0.8, 0.0, 0.6, 0.9, -0.4, 0.0, 0.7],
            0.0,
        );
    }

    #[test]
    fn breadcrumbs_with_zero_gamma_only_trims_the_bottom() {
        let t = arr(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let out = sparsify_breadcrumbs(&t, 0.5, 0.0).unwrap();
        assert_close(&values(&out), &[0.0, 0.0, 3.0, 4.0], 0.0);
        // density 1 with gamma 0.25 only removes the top outlier.
        let out = sparsify_breadcrumbs(&t, 1.0, 0.25).unwrap();
        assert_close(&values(&out), &[1.0, 2.0, 3.0, 0.0], 0.0);
        assert!(sparsify_breadcrumbs(&t, 0.5, 0.75).is_err());
    }

    #[test]
    fn dare_survivors_are_rescaled_and_the_rate_matches_density() {
        let n = 20_000;
        let t = arr(&vec![1.0; n], &[n as i32]);
        let out = values(&sparsify_dare(&t, 0.25, 11, true).unwrap());
        assert!(out.iter().all(|&v| v == 0.0 || v == 4.0));
        let kept = out.iter().filter(|&&v| v != 0.0).count() as f32 / n as f32;
        // Binomial sigma is about 0.003, so this is a 6-sigma bound.
        assert!((kept - 0.25).abs() < 0.02, "kept fraction {kept}");
        // Without rescale the survivors are untouched.
        let out = values(&sparsify_dare(&t, 0.25, 11, false).unwrap());
        assert!(out.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn dare_mask_is_a_function_of_the_seed() {
        let n = 4096;
        let t = arr(&vec![1.0; n], &[n as i32]);
        let a = values(&sparsify_dare(&t, 0.5, 5, false).unwrap());
        let b = values(&sparsify_dare(&t, 0.5, 5, false).unwrap());
        let c = values(&sparsify_dare(&t, 0.5, 6, false).unwrap());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn della_keeps_large_entries_more_often_and_rescales_by_keep_probability() {
        // One row of 2000 magnitudes 1..=2000, density 0.5, epsilon 0.4:
        // the keep probability runs linearly from 0.1 (rank 0) to 0.9.
        let n = 2000;
        let row: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let t = arr(&row, &[1, n as i32]);
        let out = values(&sparsify_della(&t, 0.5, 0.4, 3, true).unwrap());

        // Every survivor is x / p with p the reference probability for its
        // rank (here rank = x - 1).
        for (i, (&x, &y)) in row.iter().zip(&out).enumerate() {
            if y != 0.0 {
                let p = 0.1 + 0.8 * (i as f32) / (n as f32 - 1.0);
                assert!((y - x / p).abs() <= 1e-3 * (x / p), "entry {i}: {y} vs {}", x / p);
            }
        }
        // The lower quarter is kept about 0.2 of the time, the upper about 0.8.
        let rate = |range: std::ops::Range<usize>| {
            out[range.clone()].iter().filter(|&&v| v != 0.0).count() as f32 / range.len() as f32
        };
        let (low, high) = (rate(0..n / 4), rate(3 * n / 4..n));
        assert!((low - 0.2).abs() < 0.08, "low-magnitude keep rate {low}");
        assert!((high - 0.8).abs() < 0.08, "high-magnitude keep rate {high}");
    }

    #[test]
    fn della_ranks_each_row_on_its_own() {
        // Two rows with the same shape of magnitudes but 1000x apart: the
        // ranks, and so the keep probabilities, are identical.
        let n = 1000;
        let small: Vec<f32> = (1..=n).map(|i| i as f32 * 1e-3).collect();
        let big: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let t = arr(&[small.clone(), big.clone()].concat(), &[2, n as i32]);
        let out = values(&sparsify_della(&t, 0.5, 0.4, 9, true).unwrap());
        let (a, b) = out.split_at(n);
        for i in 0..n {
            let p = 0.1 + 0.8 * (i as f32) / (n as f32 - 1.0);
            for (row_vals, src) in [(a, &small), (b, &big)] {
                if row_vals[i] != 0.0 {
                    let want = src[i] / p;
                    assert!((row_vals[i] - want).abs() <= 1e-3 * want);
                }
            }
        }
    }

    #[test]
    fn della_without_epsilon_is_uniform_dare() {
        let n = 20_000;
        let t = arr(&vec![1.0; n], &[1, n as i32]);
        let out = values(&sparsify_della(&t, 0.25, 0.0, 1, true).unwrap());
        assert!(out.iter().all(|&v| v == 0.0 || v == 4.0));
        let kept = out.iter().filter(|&&v| v != 0.0).count() as f32 / n as f32;
        assert!((kept - 0.25).abs() < 0.02, "kept fraction {kept}");
    }
}
