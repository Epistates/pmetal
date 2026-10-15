//! Sign election and disjoint merge, the second and third steps of
//! TIES-Merging (Yadav et al., NeurIPS 2023, Algorithm 1), shared by every
//! method that combines sparsified task vectors (TIES, DARE-TIES, DELLA,
//! Breadcrumbs).
//!
//! Given the trimmed task vectors `τ̂_t`:
//!
//! 1. **Elect**: `γ[p] = sgn(Σ_t τ̂_t[p])`, the sign with the larger total
//!    mass at each position.
//! 2. **Disjoint merge**: average, at each position, only the `τ̂_t[p]` whose
//!    sign is `γ[p]` (the set `A_p`); the models that disagree are dropped.
//!
//! With per-model weights `w_t`, the weighted vectors `w_t·τ̂_t` take the
//! place of `τ̂_t` in the election and the average is divided by `Σ_{t∈A_p}
//! w_t`, which for `w_t = 1` is exactly the paper's `1/|A_p|`.

use crate::Result;
use pmetal_runtime::compat::{Array, Dtype};

/// How the sign at each position is elected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SignElection {
    /// The sign of the total signed magnitude, `sgn(Σ w_t·τ̂_t)`. This is
    /// the TIES paper's rule.
    #[default]
    Mass,
    /// The sign of the weighted vote count, `sgn(Σ w_t·sgn(τ̂_t))`: every
    /// model casts one vote whatever the size of its update.
    Count,
}

/// Elect a sign (+1, -1, or 0 on a tie) at each position of `task_vectors`.
pub fn elect_sign(
    task_vectors: &[Array],
    weights: &[f32],
    election: SignElection,
) -> Result<Array> {
    check_inputs(task_vectors, weights)?;
    let mut total = Array::zeros(task_vectors[0].shape(), Dtype::Float32.as_i32());
    for (tv, &w) in task_vectors.iter().zip(weights) {
        let ballot = match election {
            SignElection::Mass => tv.clone(),
            SignElection::Count => tv.sign(),
        };
        total = total.add(&ballot.multiply(&Array::from_f32(w)));
    }
    Ok(total.sign())
}

/// The disjoint merge of `task_vectors`: elect a sign per position, drop the
/// models that disagree with it, and combine the rest.
///
/// With `normalize` the combination is the mean over the agreeing models
/// (`Σ w_t·τ̂_t / Σ w_t` over `A_p`); without it, the weighted sum. A
/// position with no agreeing model (all zero, or an exact tie) is 0.
///
/// # Errors
/// An empty `task_vectors`, or `weights` of a different length.
pub fn disjoint_merge(
    task_vectors: &[Array],
    weights: &[f32],
    election: SignElection,
    normalize: bool,
) -> Result<Array> {
    let elected = elect_sign(task_vectors, weights, election)?;
    let zero = Array::from_f32(0.0);
    let shape = task_vectors[0].shape();
    let mut numerator = Array::zeros(shape, Dtype::Float32.as_i32());
    let mut denominator = Array::zeros(shape, Dtype::Float32.as_i32());
    for (tv, &w) in task_vectors.iter().zip(weights) {
        // Agrees when sgn(τ̂_t[p]) == γ[p] and neither is zero.
        let agrees = tv
            .sign()
            .multiply(&elected)
            .greater(&zero)
            .as_dtype(Dtype::Float32.as_i32());
        let w = Array::from_f32(w);
        numerator = numerator.add(&tv.multiply(&w).multiply(&agrees));
        if normalize {
            denominator = denominator.add(&agrees.multiply(&w));
        }
    }
    if !normalize {
        return Ok(numerator);
    }
    // No agreeing model leaves 0 / 0; the numerator is 0 there too.
    let degenerate = denominator.abs_val().less(&Array::from_f32(1e-12));
    let safe = degenerate.where_cond(&Array::ones(shape, Dtype::Float32.as_i32()), &denominator);
    Ok(numerator.divide(&safe))
}

fn check_inputs(task_vectors: &[Array], weights: &[f32]) -> Result<()> {
    if task_vectors.is_empty() {
        return Err(crate::MergeError::NotEnoughModels {
            expected: 1,
            actual: 0,
        });
    }
    if weights.len() != task_vectors.len() {
        return Err(crate::MergeError::InvalidConfig(format!(
            "{} task vectors but {} weights",
            task_vectors.len(),
            weights.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{arr, assert_close, values};

    #[test]
    fn elect_sign_by_mass_follows_the_larger_total() {
        // Column 0: +10, -1, -1 has mass +8; column 1: +1, -1, -1 has mass -1.
        let tvs = [
            arr(&[10.0, 1.0], &[2]),
            arr(&[-1.0, -1.0], &[2]),
            arr(&[-1.0, -1.0], &[2]),
        ];
        let w = [1.0; 3];
        assert_close(
            &values(&elect_sign(&tvs, &w, SignElection::Mass).unwrap()),
            &[1.0, -1.0],
            0.0,
        );
        // Counting votes instead, column 0 is two against one.
        assert_close(
            &values(&elect_sign(&tvs, &w, SignElection::Count).unwrap()),
            &[-1.0, -1.0],
            0.0,
        );
    }

    #[test]
    fn disjoint_merge_averages_only_the_agreeing_models() {
        // Column 0: all positive, mean (1 + 2 + 6) / 3 = 3.
        // Column 1: -4 and -2 beat +1, mean -3.
        // Column 2: nobody but the third model is non-zero, so it is kept.
        let tvs = [
            arr(&[1.0, -4.0, 0.0], &[3]),
            arr(&[2.0, -2.0, 0.0], &[3]),
            arr(&[6.0, 1.0, 5.0], &[3]),
        ];
        let out = disjoint_merge(&tvs, &[1.0; 3], SignElection::Mass, true).unwrap();
        assert_close(&values(&out), &[3.0, -3.0, 5.0], 1e-6);
        // The same agreeing set, summed instead of averaged.
        let out = disjoint_merge(&tvs, &[1.0; 3], SignElection::Mass, false).unwrap();
        assert_close(&values(&out), &[9.0, -6.0, 5.0], 1e-6);
    }

    #[test]
    fn a_tie_or_an_all_zero_column_merges_to_zero() {
        let tvs = [arr(&[1.0, 0.0], &[2]), arr(&[-1.0, 0.0], &[2])];
        let out = disjoint_merge(&tvs, &[1.0, 1.0], SignElection::Mass, true).unwrap();
        assert_close(&values(&out), &[0.0, 0.0], 0.0);
    }

    #[test]
    fn a_single_vector_is_its_own_majority() {
        let tvs = [arr(&[2.0, -3.0], &[2])];
        let out = disjoint_merge(&tvs, &[1.0], SignElection::Mass, true).unwrap();
        assert_close(&values(&out), &[2.0, -3.0], 1e-6);
    }

    #[test]
    fn mismatched_weights_are_an_error() {
        let tvs = [arr(&[1.0], &[1]), arr(&[2.0], &[1])];
        assert!(disjoint_merge(&tvs, &[1.0], SignElection::Mass, true).is_err());
        assert!(disjoint_merge(&[], &[], SignElection::Mass, true).is_err());
    }
}
