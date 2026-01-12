//! Sparsification strategies for model merging.
//!
//! Sparsification reduces interference when merging models by keeping only
//! the most important parameters (by some criterion) and zeroing the rest.

use mlx_rs::Array;
use crate::Result;

/// Sparsify a tensor by keeping only the top `density` fraction by magnitude.
///
/// # Arguments
/// * `tensor` - Input tensor to sparsify
/// * `density` - Fraction of elements to keep (0.0 to 1.0)
///
/// # Returns
/// A tensor with the same shape where only the top `density` elements are kept,
/// rest are zeroed.
pub fn sparsify_by_magnitude(tensor: &Array, density: f32) -> Result<Array> {
    if density >= 1.0 {
        return Ok(tensor.clone());
    }
    if density <= 0.0 {
        return Ok(Array::zeros::<f32>(tensor.shape())?);
    }

    // Flatten for processing
    let original_shape = tensor.shape().to_vec();
    let flat = tensor.reshape(&[-1])?;
    let n = flat.dim(0) as usize;

    // Compute absolute values
    let abs_vals = flat.abs()?;
    let abs_slice: Vec<f32> = abs_vals.as_slice().to_vec();

    // Find threshold value (k-th largest magnitude)
    let k = ((1.0 - density) * n as f32).ceil() as usize;
    let k = k.min(n.saturating_sub(1));

    // Get sorted magnitudes
    let mut sorted_abs: Vec<f32> = abs_slice.clone();
    sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let threshold = sorted_abs[k];

    // Create mask: 1 where |x| >= threshold, 0 otherwise
    let threshold_array = Array::from_f32(threshold);
    let mask = abs_vals.ge(&threshold_array)?;
    let mask_f32 = mask.as_type::<f32>()?;

    // Apply mask
    let result_flat = flat.multiply(&mask_f32)?;
    Ok(result_flat.reshape(&original_shape)?)
}

/// Sparsify a tensor by keeping only the middle `density` fraction.
/// This removes both the largest (gamma fraction) and smallest values.
///
/// Used by the "breadcrumbs" merge method.
///
/// # Arguments
/// * `tensor` - Input tensor to sparsify
/// * `density` - Fraction of elements to keep (0.0 to 1.0)
/// * `gamma` - Fraction of largest outliers to remove (0.0 to 1.0)
pub fn sparsify_breadcrumbs(tensor: &Array, density: f32, gamma: f32) -> Result<Array> {
    if density >= 1.0 && gamma <= 0.0 {
        return Ok(tensor.clone());
    }

    // Flatten for processing
    let original_shape = tensor.shape().to_vec();
    let flat = tensor.reshape(&[-1])?;
    let n = flat.dim(0) as usize;

    // Compute absolute values
    let abs_vals = flat.abs()?;
    let abs_slice: Vec<f32> = abs_vals.as_slice().to_vec();

    // Get sorted magnitudes with indices
    let mut indexed: Vec<(usize, f32)> = abs_slice.iter().enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Determine thresholds
    // Remove smallest (1-density) and largest (gamma) fractions
    let lower_k = ((1.0 - density) * n as f32).ceil() as usize;
    let upper_k = ((1.0 - gamma) * n as f32).floor() as usize;

    // Create mask
    let mut mask = vec![0.0_f32; n];
    for (idx, _) in indexed.iter().skip(lower_k).take(upper_k.saturating_sub(lower_k)) {
        mask[*idx] = 1.0;
    }

    let mask_array = Array::from_slice(&mask, &[n as i32]);
    let result_flat = flat.multiply(&mask_array)?;
    Ok(result_flat.reshape(&original_shape)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparsify_full_density() {
        let tensor = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
        let result = sparsify_by_magnitude(&tensor, 1.0).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        assert_eq!(result_slice, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sparsify_zero_density() {
        let tensor = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
        let result = sparsify_by_magnitude(&tensor, 0.0).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        assert_eq!(result_slice, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sparsify_half_density() {
        let tensor = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
        let result = sparsify_by_magnitude(&tensor, 0.5).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // Should keep the two largest by magnitude (3.0 and 4.0)
        assert_eq!(result_slice[0], 0.0);
        assert_eq!(result_slice[1], 0.0);
        assert_eq!(result_slice[2], 3.0);
        assert_eq!(result_slice[3], 4.0);
    }

    #[test]
    fn test_sparsify_preserves_shape() {
        let tensor = Array::from_slice(&[1.0_f32; 12], &[3, 4]);
        let result = sparsify_by_magnitude(&tensor, 0.5).unwrap();
        assert_eq!(result.shape(), &[3, 4]);
    }

    #[test]
    fn test_sparsify_handles_negative() {
        let tensor = Array::from_slice(&[-4.0_f32, 1.0, -2.0, 3.0], &[4]);
        let result = sparsify_by_magnitude(&tensor, 0.5).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // Should keep -4.0 and 3.0 (largest magnitudes)
        assert_eq!(result_slice[0], -4.0);
        assert_eq!(result_slice[1], 0.0);
        assert_eq!(result_slice[2], 0.0);
        assert_eq!(result_slice[3], 3.0);
    }

    #[test]
    fn test_breadcrumbs_removes_outliers() {
        // Values sorted by magnitude: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        let tensor = Array::from_slice(
            &[0.5_f32, 1.0, 0.3, 0.8, 0.1, 0.6, 0.9, 0.4, 0.2, 0.7],
            &[10]
        );

        // Keep middle 50% (indices 2-7 in sorted order), remove smallest and largest
        let result = sparsify_breadcrumbs(&tensor, 0.6, 0.1).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // 1.0 (largest) should be removed
        assert_eq!(result_slice[1], 0.0);
        // 0.1 (smallest) should be removed
        assert_eq!(result_slice[4], 0.0);
    }
}
