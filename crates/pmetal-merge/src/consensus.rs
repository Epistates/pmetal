//! Sign consensus for model merging.
//!
//! Sign consensus is a key component of TIES-Merging that reduces interference
//! by only keeping parameters where the models agree on the direction of change.

use crate::Result;
use mlx_rs::ops::sign;
use mlx_rs::Array;

/// Compute sign consensus mask across multiple tensors.
///
/// For each parameter position, we compute the weighted sum of signs.
/// If the weighted majority agrees on a sign (positive or negative),
/// we keep that position. Otherwise, we zero it out.
///
/// # Arguments
/// * `tensors` - Tensors to compute consensus across
/// * `weights` - Weight for each tensor in the consensus vote
///
/// # Returns
/// A mask tensor with 1.0 where consensus is achieved, 0.0 otherwise.
pub fn sign_consensus(tensors: &[Array], weights: &[f32]) -> Result<Array> {
    if tensors.is_empty() {
        return Err(crate::MergeError::NotEnoughModels {
            expected: 1,
            actual: 0,
        });
    }

    if tensors.len() == 1 {
        // Single tensor always has consensus with itself
        let ones = Array::ones::<f32>(tensors[0].shape())?;
        return Ok(ones);
    }

    // Compute weighted sum of signs
    let mut weighted_signs = Array::zeros::<f32>(tensors[0].shape())?;

    for (tensor, weight) in tensors.iter().zip(weights.iter()) {
        // Get sign: +1 for positive, -1 for negative, 0 for zero
        let signs = sign(tensor)?;
        let weighted = signs.multiply(&Array::from_f32(*weight))?;
        weighted_signs = weighted_signs.add(&weighted)?;
    }

    // Consensus achieved where |weighted_signs| > 0
    // This means the weighted majority agrees on the direction
    let abs_weighted = weighted_signs.abs()?;
    let zero = Array::from_f32(0.0);
    let mask = abs_weighted.gt(&zero)?;

    Ok(mask.as_type::<f32>()?)
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
        let weighted = signs.multiply(&Array::from_f32(*weight))?;
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
        let t1 = Array::from_slice(&[1.0_f32, -1.0, 1.0], &[3]);
        let t2 = Array::from_slice(&[2.0_f32, -2.0, 3.0], &[3]);
        let weights = vec![1.0, 1.0];

        let mask = sign_consensus(&[t1, t2], &weights).unwrap();
        let mask_slice: Vec<f32> = mask.as_slice().to_vec();

        // All positions should have consensus (same signs)
        assert_eq!(mask_slice, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sign_consensus_disagree() {
        let t1 = Array::from_slice(&[1.0_f32, -1.0, 1.0], &[3]);
        let t2 = Array::from_slice(&[-1.0_f32, -1.0, 1.0], &[3]);
        let weights = vec![1.0, 1.0];

        let mask = sign_consensus(&[t1, t2], &weights).unwrap();
        let mask_slice: Vec<f32> = mask.as_slice().to_vec();

        // First position: +1 and -1 = 0, no consensus
        // Second position: -1 and -1 = -2, consensus
        // Third position: +1 and +1 = 2, consensus
        assert_eq!(mask_slice[0], 0.0);
        assert_eq!(mask_slice[1], 1.0);
        assert_eq!(mask_slice[2], 1.0);
    }

    #[test]
    fn test_sign_consensus_weighted() {
        let t1 = Array::from_slice(&[1.0_f32], &[1]);
        let t2 = Array::from_slice(&[-1.0_f32], &[1]);
        let t3 = Array::from_slice(&[1.0_f32], &[1]);

        // Equal weights: +1 - 1 + 1 = 1 > 0, consensus
        let mask = sign_consensus(&[t1.clone(), t2.clone(), t3.clone()], &[1.0, 1.0, 1.0]).unwrap();
        let mask_slice: Vec<f32> = mask.as_slice().to_vec();
        assert_eq!(mask_slice[0], 1.0);

        // Higher weight on negative: +1 - 2 + 1 = 0, no consensus
        let mask = sign_consensus(&[t1, t2, t3], &[1.0, 2.0, 1.0]).unwrap();
        let mask_slice: Vec<f32> = mask.as_slice().to_vec();
        assert_eq!(mask_slice[0], 0.0);
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
