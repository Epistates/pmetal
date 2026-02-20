//! Sign consensus for model merging.
//!
//! Sign consensus is a key component of TIES-Merging (Yadav et al. 2023).
//! It reduces interference by:
//! 1. Computing the majority sign at each parameter position.
//! 2. Discarding contributions from models whose sign disagrees with the majority.
//! 3. Summing only the agreeing contributions.
//!
//! Reference: "TIES-Merging: Resolving Interference When Merging Models"
//! (Yadav et al., NeurIPS 2023, Algorithm 1 step 3).

use crate::Result;
use mlx_rs::ops::sign;
use mlx_rs::Array;

/// Compute sign consensus mask across multiple tensors.
///
/// Implements the TIES-Merging sign-consensus step:
/// 1. Compute the weighted sum of signs to determine the majority sign direction.
/// 2. For each position, zero out contributions from tensors whose sign disagrees
///    with the majority (i.e. `sign(tensor) != majority_sign`).
/// 3. Return the sum of the agreeing contributions.
///
/// This is a **summed** result (not a binary 0/1 mask) because the TIES paper
/// discards disagreeing model parameters and sums the remaining ones.
///
/// # Arguments
/// * `tensors` - Sparse task-vector tensors to compute consensus across.
/// * `weights` - Weight for each tensor in the consensus vote.
///
/// # Returns
/// A tensor containing the sum of contributions that agree with the majority sign,
/// with disagreeing contributions zeroed out.
pub fn sign_consensus(tensors: &[Array], weights: &[f32]) -> Result<Array> {
    if tensors.is_empty() {
        return Err(crate::MergeError::NotEnoughModels {
            expected: 1,
            actual: 0,
        });
    }

    if tensors.len() == 1 {
        // Single tensor is always its own majority; return it scaled by its weight.
        let weighted = tensors[0].multiply(Array::from_f32(weights[0]))?;
        return Ok(weighted);
    }

    // Step 1: Compute the majority sign at each position.
    // majority_sign[i] = sign( sum_m( weight_m * sign(tensor_m[i]) ) )
    let mut weighted_signs = Array::zeros::<f32>(tensors[0].shape())?;
    for (tensor, weight) in tensors.iter().zip(weights.iter()) {
        let s = sign(tensor)?;
        weighted_signs = weighted_signs.add(&s.multiply(Array::from_f32(*weight))?)?;
    }
    // The majority sign at each position (+1, -1, or 0 when tied).
    let maj_sign = sign(&weighted_signs)?;

    // Step 2: For each model, keep only the elements whose sign matches the majority.
    // A contribution agrees when sign(tensor[i]) == majority_sign[i].
    // Equivalently: sign(tensor[i]) * majority_sign[i] > 0.
    // We zero out elements where the signs differ (product <= 0).
    let mut result = Array::zeros::<f32>(tensors[0].shape())?;
    for (tensor, weight) in tensors.iter().zip(weights.iter()) {
        let s = sign(tensor)?;
        // agreement[i] = 1 if sign(tensor[i]) == majority_sign[i] else 0
        // sign product > 0 means agreement; <= 0 means disagreement or zero.
        let product = s.multiply(&maj_sign)?;
        let zero = Array::from_f32(0.0);
        let agrees = product.gt(&zero)?.as_type::<f32>()?;

        // Add weight * tensor where the model agrees with the majority sign.
        let contribution = tensor.multiply(Array::from_f32(*weight))?;
        result = result.add(&contribution.multiply(&agrees)?)?;
    }

    Ok(result)
}

/// Compute majority sign at each position.
///
/// Returns a tensor with +1, -1, or 0 at each position based on the
/// weighted majority vote of signs.
///
/// # Arguments
/// * `tensors` - Tensors to compute majority sign across
/// * `weights` - Weight for each tensor in the vote
pub fn majority_sign(tensors: &[Array], weights: &[f32]) -> Result<Array> {
    if tensors.is_empty() {
        return Err(crate::MergeError::NotEnoughModels {
            expected: 1,
            actual: 0,
        });
    }

    // Compute weighted sum of signs
    let mut weighted_signs = Array::zeros::<f32>(tensors[0].shape())?;

    for (tensor, weight) in tensors.iter().zip(weights.iter()) {
        let signs = sign(tensor)?;
        let weighted = signs.multiply(Array::from_f32(*weight))?;
        weighted_signs = weighted_signs.add(&weighted)?;
    }

    // Return sign of the weighted sum
    Ok(sign(&weighted_signs)?)
}

/// Compute element-wise agreement mask.
///
/// Returns 1.0 where ALL tensors have the same sign, 0.0 otherwise.
/// This is stricter than weighted consensus - requires unanimous agreement.
///
/// # Arguments
/// * `tensors` - Tensors to check agreement across
pub fn unanimous_agreement(tensors: &[Array]) -> Result<Array> {
    if tensors.is_empty() {
        return Err(crate::MergeError::NotEnoughModels {
            expected: 1,
            actual: 0,
        });
    }

    if tensors.len() == 1 {
        let ones = Array::ones::<f32>(tensors[0].shape())?;
        return Ok(ones);
    }

    // Get sign of first tensor
    let first_sign = sign(&tensors[0])?;

    // Check if all other tensors have the same sign
    let mut agreement = Array::ones::<f32>(tensors[0].shape())?;

    for tensor in &tensors[1..] {
        let tensor_sign = sign(tensor)?;
        let same = first_sign.eq(&tensor_sign)?;
        let same_f32 = same.as_type::<f32>()?;
        agreement = agreement.multiply(&same_f32)?;
    }

    Ok(agreement)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_consensus_all_agree() {
        // t1=[1, -1, 1], t2=[2, -2, 3] - all have the same sign per position.
        // Both models agree, so both contribute. Result = w1*t1 + w2*t2 = t1 + t2.
        let t1 = Array::from_slice(&[1.0_f32, -1.0, 1.0], &[3]);
        let t2 = Array::from_slice(&[2.0_f32, -2.0, 3.0], &[3]);
        let weights = vec![1.0, 1.0];

        let result = sign_consensus(&[t1, t2], &weights).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // All positions agree: sum = t1 + t2
        assert!((result_slice[0] - 3.0).abs() < 1e-5); // 1 + 2
        assert!((result_slice[1] - (-3.0)).abs() < 1e-5); // -1 + -2
        assert!((result_slice[2] - 4.0).abs() < 1e-5); // 1 + 3
    }

    #[test]
    fn test_sign_consensus_disagree() {
        // t1=[1, -1, 1], t2=[-1, -1, 1] with equal weights.
        // Position 0: t1=+1, t2=-1; majority sign = sign(1-1)=sign(0)=0 (tied).
        //   With tied majority (0), no model's sign equals the majority (neither +1 nor -1 == 0).
        //   Result at position 0 = 0.
        // Position 1: t1=-1, t2=-1; majority = -1. Both agree. Result = -1 + -1 = -2.
        // Position 2: t1=+1, t2=+1; majority = +1. Both agree. Result = 1 + 1 = 2.
        let t1 = Array::from_slice(&[1.0_f32, -1.0, 1.0], &[3]);
        let t2 = Array::from_slice(&[-1.0_f32, -1.0, 1.0], &[3]);
        let weights = vec![1.0, 1.0];

        let result = sign_consensus(&[t1, t2], &weights).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // Position 0: tied majority (0), both models disagree - zeroed out.
        assert!((result_slice[0]).abs() < 1e-5);
        // Position 1: both agree on negative.
        assert!((result_slice[1] - (-2.0)).abs() < 1e-5);
        // Position 2: both agree on positive.
        assert!((result_slice[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_sign_consensus_weighted() {
        // t1=[1], t2=[-1], t3=[1] with equal weights.
        // Majority sign = sign(1 - 1 + 1) = sign(1) = +1.
        // t1 and t3 agree (+1); t2 disagrees (-1).
        // Result = 1.0*t1 + 1.0*t3 = 1 + 1 = 2 (t2 is discarded).
        let t1 = Array::from_slice(&[1.0_f32], &[1]);
        let t2 = Array::from_slice(&[-1.0_f32], &[1]);
        let t3 = Array::from_slice(&[1.0_f32], &[1]);

        let result =
            sign_consensus(&[t1.clone(), t2.clone(), t3.clone()], &[1.0, 1.0, 1.0]).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();
        // Only t1 and t3 contribute (both agree with majority +1).
        assert!((result_slice[0] - 2.0).abs() < 1e-5);

        // With higher weight on t2: vote = 1 - 2 + 1 = 0 (tied).
        // Tied majority = sign(0) = 0. No model agrees with 0.
        // Result = 0.
        let result = sign_consensus(&[t1, t2, t3], &[1.0, 2.0, 1.0]).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();
        assert!((result_slice[0]).abs() < 1e-5);
    }

    #[test]
    fn test_majority_sign() {
        let t1 = Array::from_slice(&[1.0_f32, -1.0], &[2]);
        let t2 = Array::from_slice(&[1.0_f32, 1.0], &[2]);
        let t3 = Array::from_slice(&[1.0_f32, 1.0], &[2]);
        let weights = vec![1.0, 1.0, 1.0];

        let signs = majority_sign(&[t1, t2, t3], &weights).unwrap();
        let signs_slice: Vec<f32> = signs.as_slice().to_vec();

        // First position: all +1 = +3, sign = +1
        assert_eq!(signs_slice[0], 1.0);
        // Second position: -1 + 1 + 1 = 1, sign = +1
        assert_eq!(signs_slice[1], 1.0);
    }

    #[test]
    fn test_unanimous_agreement() {
        let t1 = Array::from_slice(&[1.0_f32, -1.0, 1.0], &[3]);
        let t2 = Array::from_slice(&[2.0_f32, -2.0, -1.0], &[3]);

        let agreement = unanimous_agreement(&[t1, t2]).unwrap();
        let agreement_slice: Vec<f32> = agreement.as_slice().to_vec();

        // First: both positive, agree
        assert_eq!(agreement_slice[0], 1.0);
        // Second: both negative, agree
        assert_eq!(agreement_slice[1], 1.0);
        // Third: positive and negative, disagree
        assert_eq!(agreement_slice[2], 0.0);
    }
}
