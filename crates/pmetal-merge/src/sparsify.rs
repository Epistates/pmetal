//! Sparsification strategies for model merging.
//!
//! Sparsification reduces interference when merging models by keeping only
//! the most important parameters (by some criterion) and zeroing the rest.
//!
//! # Optimizations
//!
//! This module provides two implementations:
//!
//! 1. **Standard**: Uses full sorting O(n log n) - reliable but slower
//! 2. **Online**: Uses Quickselect O(n) - faster for large tensors
//!
//! The online implementation uses the Quickselect algorithm (Floyd-Rivest variant)
//! to find the k-th percentile threshold in linear time on average.

use crate::Result;
use mlx_rs::Array;

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
    let mut indexed: Vec<(usize, f32)> =
        abs_slice.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Determine thresholds
    // Remove smallest (1-density) and largest (gamma) fractions
    let lower_k = ((1.0 - density) * n as f32).ceil() as usize;
    let upper_k = ((1.0 - gamma) * n as f32).floor() as usize;

    // Create mask
    let mut mask = vec![0.0_f32; n];
    for (idx, _) in indexed
        .iter()
        .skip(lower_k)
        .take(upper_k.saturating_sub(lower_k))
    {
        mask[*idx] = 1.0;
    }

    let mask_array = Array::from_slice(&mask, &[n as i32]);
    let result_flat = flat.multiply(&mask_array)?;
    Ok(result_flat.reshape(&original_shape)?)
}

// =============================================================================
// Online (O(n)) Sparsification
// =============================================================================

/// Find the k-th smallest element using Quickselect algorithm.
///
/// This is an O(n) average case algorithm for finding the k-th order statistic,
/// compared to O(n log n) for full sorting.
///
/// # Arguments
/// * `data` - Mutable slice of values (will be partially reordered)
/// * `k` - The index of the element to find (0-based)
///
/// # Returns
/// The k-th smallest element in the array.
pub fn quickselect(data: &mut [f32], k: usize) -> f32 {
    if data.len() == 1 {
        return data[0];
    }

    let k = k.min(data.len() - 1);
    quickselect_impl(data, 0, data.len() - 1, k)
}

fn quickselect_impl(data: &mut [f32], left: usize, right: usize, k: usize) -> f32 {
    if left == right {
        return data[left];
    }

    // Use median-of-three pivot selection for better average case
    let pivot_idx = median_of_three(data, left, right);
    let pivot_idx = partition(data, left, right, pivot_idx);

    if k == pivot_idx {
        data[k]
    } else if k < pivot_idx {
        quickselect_impl(data, left, pivot_idx.saturating_sub(1), k)
    } else {
        quickselect_impl(data, pivot_idx + 1, right, k)
    }
}

/// Select median-of-three as pivot for better performance.
fn median_of_three(data: &[f32], left: usize, right: usize) -> usize {
    let mid = left + (right - left) / 2;

    let a = data[left];
    let b = data[mid];
    let c = data[right];

    if (a <= b && b <= c) || (c <= b && b <= a) {
        mid
    } else if (b <= a && a <= c) || (c <= a && a <= b) {
        left
    } else {
        right
    }
}

/// Partition array around pivot and return final pivot position.
fn partition(data: &mut [f32], left: usize, right: usize, pivot_idx: usize) -> usize {
    let pivot_value = data[pivot_idx];

    // Move pivot to end
    data.swap(pivot_idx, right);

    let mut store_idx = left;
    for i in left..right {
        if data[i] < pivot_value {
            data.swap(i, store_idx);
            store_idx += 1;
        }
    }

    // Move pivot to final position
    data.swap(store_idx, right);
    store_idx
}

/// Sparsify a tensor using O(n) Quickselect algorithm.
///
/// This is faster than the standard implementation for large tensors,
/// as it avoids the O(n log n) full sort.
///
/// # Arguments
/// * `tensor` - Input tensor to sparsify
/// * `density` - Fraction of elements to keep (0.0 to 1.0)
///
/// # Returns
/// A tensor with the same shape where only the top `density` elements are kept.
pub fn sparsify_by_magnitude_online(tensor: &Array, density: f32) -> Result<Array> {
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
    let mut abs_slice: Vec<f32> = abs_vals.as_slice().to_vec();

    // Find threshold using O(n) Quickselect
    let k = ((1.0 - density) * n as f32).ceil() as usize;
    let k = k.min(n.saturating_sub(1));

    let threshold = quickselect(&mut abs_slice, k);

    // Create mask: 1 where |x| >= threshold, 0 otherwise
    let threshold_array = Array::from_f32(threshold);
    let mask = abs_vals.ge(&threshold_array)?;
    let mask_f32 = mask.as_type::<f32>()?;

    // Apply mask
    let result_flat = flat.multiply(&mask_f32)?;
    Ok(result_flat.reshape(&original_shape)?)
}

/// Batch sparsify multiple tensors with potentially different densities.
///
/// This processes multiple tensors efficiently, sharing overhead costs.
///
/// # Arguments
/// * `tensors` - Slice of tensors to sparsify
/// * `densities` - Density for each tensor (must match length of tensors)
///
/// # Returns
/// Vector of sparsified tensors in the same order.
pub fn sparsify_batch_by_magnitude(tensors: &[Array], densities: &[f32]) -> Result<Vec<Array>> {
    assert_eq!(
        tensors.len(),
        densities.len(),
        "Number of tensors must match number of densities"
    );

    tensors
        .iter()
        .zip(densities.iter())
        .map(|(tensor, &density)| sparsify_by_magnitude_online(tensor, density))
        .collect()
}

/// Compute threshold values for multiple tensors without applying sparsification.
///
/// This is useful when you need to inspect or reuse thresholds.
///
/// # Arguments
/// * `tensors` - Slice of tensors
/// * `densities` - Density for each tensor
///
/// # Returns
/// Vector of threshold values for each tensor.
pub fn compute_thresholds(tensors: &[Array], densities: &[f32]) -> Result<Vec<f32>> {
    assert_eq!(
        tensors.len(),
        densities.len(),
        "Number of tensors must match number of densities"
    );

    let mut thresholds = Vec::with_capacity(tensors.len());

    for (tensor, &density) in tensors.iter().zip(densities.iter()) {
        if density >= 1.0 {
            thresholds.push(0.0);
            continue;
        }
        if density <= 0.0 {
            thresholds.push(f32::MAX);
            continue;
        }

        let flat = tensor.reshape(&[-1])?;
        let n = flat.dim(0) as usize;

        let abs_vals = flat.abs()?;
        let mut abs_slice: Vec<f32> = abs_vals.as_slice().to_vec();

        let k = ((1.0 - density) * n as f32).ceil() as usize;
        let k = k.min(n.saturating_sub(1));

        let threshold = quickselect(&mut abs_slice, k);
        thresholds.push(threshold);
    }

    Ok(thresholds)
}

/// Apply pre-computed thresholds to tensors.
///
/// This separates threshold computation from application, enabling
/// Metal kernel acceleration of the masking step.
///
/// # Arguments
/// * `tensors` - Slice of tensors
/// * `thresholds` - Pre-computed threshold for each tensor
///
/// # Returns
/// Vector of sparsified tensors.
pub fn apply_thresholds(tensors: &[Array], thresholds: &[f32]) -> Result<Vec<Array>> {
    assert_eq!(
        tensors.len(),
        thresholds.len(),
        "Number of tensors must match number of thresholds"
    );

    let mut results = Vec::with_capacity(tensors.len());

    for (tensor, &threshold) in tensors.iter().zip(thresholds.iter()) {
        if threshold == 0.0 {
            // density >= 1.0, keep all
            results.push(tensor.clone());
            continue;
        }
        if threshold == f32::MAX {
            // density <= 0.0, zero all
            results.push(Array::zeros::<f32>(tensor.shape())?);
            continue;
        }

        let original_shape = tensor.shape().to_vec();
        let flat = tensor.reshape(&[-1])?;
        let abs_vals = flat.abs()?;

        let threshold_array = Array::from_f32(threshold);
        let mask = abs_vals.ge(&threshold_array)?;
        let mask_f32 = mask.as_type::<f32>()?;

        let result_flat = flat.multiply(&mask_f32)?;
        results.push(result_flat.reshape(&original_shape)?);
    }

    Ok(results)
}

/// DARE (Drop And REscale) sparsification with random masking.
///
/// Instead of keeping by magnitude, randomly drops elements and rescales
/// the remaining ones to maintain expected sum.
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `density` - Fraction of elements to keep
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
/// Sparsified and rescaled tensor.
pub fn sparsify_dare(tensor: &Array, density: f32, seed: Option<u64>) -> Result<Array> {
    if density >= 1.0 {
        return Ok(tensor.clone());
    }
    if density <= 0.0 {
        return Ok(Array::zeros::<f32>(tensor.shape())?);
    }

    let original_shape = tensor.shape().to_vec();
    let flat = tensor.reshape(&[-1])?;
    let n = flat.dim(0) as usize;

    // Generate random mask with optional seed
    use rand::{Rng, SeedableRng};

    let mut rng = if let Some(s) = seed {
        rand::rngs::StdRng::seed_from_u64(s)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

    let mut mask_data = Vec::with_capacity(n);

    for _ in 0..n {
        if rng.gen::<f32>() < density {
            // Rescale by 1/density to maintain expected sum
            mask_data.push(1.0 / density);
        } else {
            mask_data.push(0.0);
        }
    }

    let mask = Array::from_slice(&mask_data, &[n as i32]);
    let result_flat = flat.multiply(&mask)?;
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
            &[10],
        );

        // Keep middle 50% (indices 2-7 in sorted order), remove smallest and largest
        let result = sparsify_breadcrumbs(&tensor, 0.6, 0.1).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // 1.0 (largest) should be removed
        assert_eq!(result_slice[1], 0.0);
        // 0.1 (smallest) should be removed
        assert_eq!(result_slice[4], 0.0);
    }

    // =============================================================================
    // Online Sparsification Tests
    // =============================================================================

    #[test]
    fn test_quickselect_small() {
        let mut data = vec![5.0, 2.0, 8.0, 1.0, 9.0];
        let median = quickselect(&mut data, 2);
        assert!((median - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_quickselect_min() {
        let mut data = vec![5.0, 2.0, 8.0, 1.0, 9.0];
        let min = quickselect(&mut data, 0);
        assert!((min - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quickselect_max() {
        let mut data = vec![5.0, 2.0, 8.0, 1.0, 9.0];
        let max = quickselect(&mut data, 4);
        assert!((max - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_quickselect_single() {
        let mut data = vec![42.0];
        let result = quickselect(&mut data, 0);
        assert!((result - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_online_sparsify_matches_standard() {
        // Online should produce same results as standard for identical inputs
        let tensor = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8]);

        let standard = sparsify_by_magnitude(&tensor, 0.5).unwrap();
        let online = sparsify_by_magnitude_online(&tensor, 0.5).unwrap();

        let standard_slice: Vec<f32> = standard.as_slice().to_vec();
        let online_slice: Vec<f32> = online.as_slice().to_vec();

        // Both should keep the top 50% by magnitude
        assert_eq!(standard_slice, online_slice);
    }

    #[test]
    fn test_online_sparsify_full_density() {
        let tensor = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
        let result = sparsify_by_magnitude_online(&tensor, 1.0).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        assert_eq!(result_slice, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_online_sparsify_zero_density() {
        let tensor = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
        let result = sparsify_by_magnitude_online(&tensor, 0.0).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        assert_eq!(result_slice, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_batch_sparsify() {
        let t1 = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
        let t2 = Array::from_slice(&[0.5_f32, 1.5, 2.5, 3.5], &[4]);

        let results = sparsify_batch_by_magnitude(&[t1, t2], &[0.5, 0.5]).unwrap();

        assert_eq!(results.len(), 2);

        // First tensor: keep 3.0 and 4.0
        let r1: Vec<f32> = results[0].as_slice().to_vec();
        assert_eq!(r1[0], 0.0);
        assert_eq!(r1[1], 0.0);
        assert_eq!(r1[2], 3.0);
        assert_eq!(r1[3], 4.0);

        // Second tensor: keep 2.5 and 3.5
        let r2: Vec<f32> = results[1].as_slice().to_vec();
        assert_eq!(r2[0], 0.0);
        assert_eq!(r2[1], 0.0);
        assert_eq!(r2[2], 2.5);
        assert_eq!(r2[3], 3.5);
    }

    #[test]
    fn test_compute_and_apply_thresholds() {
        let t1 = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
        let t2 = Array::from_slice(&[0.5_f32, 1.5, 2.5, 3.5], &[4]);

        let thresholds = compute_thresholds(&[t1.clone(), t2.clone()], &[0.5, 0.5]).unwrap();

        assert_eq!(thresholds.len(), 2);

        // Apply thresholds
        let results = apply_thresholds(&[t1, t2], &thresholds).unwrap();

        // Verify results match direct sparsification
        let r1: Vec<f32> = results[0].as_slice().to_vec();
        assert_eq!(r1[2], 3.0);
        assert_eq!(r1[3], 4.0);
    }

    #[test]
    fn test_dare_sparsification_seeded() {
        let tensor = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8]);

        // Same seed should produce same result
        let result1 = sparsify_dare(&tensor, 0.5, Some(42)).unwrap();
        let result2 = sparsify_dare(&tensor, 0.5, Some(42)).unwrap();

        let r1: Vec<f32> = result1.as_slice().to_vec();
        let r2: Vec<f32> = result2.as_slice().to_vec();

        assert_eq!(r1, r2);
    }

    #[test]
    fn test_dare_rescaling() {
        let tensor = Array::from_slice(&[1.0_f32, 1.0, 1.0, 1.0], &[4]);

        // With density=0.5, kept values should be scaled by 1/0.5 = 2.0
        let result = sparsify_dare(&tensor, 0.5, Some(12345)).unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // Count non-zero values
        let non_zero: Vec<f32> = result_slice.iter().copied().filter(|&x| x != 0.0).collect();

        // All non-zero values should be 2.0 (1.0 * 1/0.5)
        for val in &non_zero {
            assert!((val - 2.0).abs() < 1e-6);
        }
    }
}
