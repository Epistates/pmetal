//! Nearswap - Parameter-wise nearest-to-base model selection.
//!
//! Nearswap is a conservative merging strategy that, for each individual parameter,
//! selects the value from whichever source model is *closest* to the base model at
//! that position. The intuition is that a fine-tuned weight close to the base value
//! has changed little and is therefore the least likely to conflict with other
//! specialised models when merged.
//!
//! # Algorithm
//!
//! For each element position `i`:
//! 1. Compute the absolute deviation from base for every source model:
//!    ```text
//!    d_m[i] = |W_m[i] - W_base[i]|
//!    ```
//! 2. Select the model with the minimum deviation:
//!    ```text
//!    m*(i) = argmin_m d_m[i]
//!    ```
//! 3. Take that model's value at position `i`:
//!    ```text
//!    W_merged[i] = W_{m*}[i]
//!    ```
//!
//! # Optional weighted blending
//!
//! When per-model `weight` parameters are set (or a global weight != 1.0), the
//! method optionally interpolates between the selected nearest value and the base
//! model using the global `lambda` parameter:
//! ```text
//! W_merged[i] = W_base[i] + lambda * (W_{m*}[i] - W_base[i])
//! ```
//! With `lambda = 1.0` (the default) this reduces to pure selection.
//!
//! # Properties
//!
//! - **Conservative**: only adopts changes that are small relative to the base,
//!   minimising catastrophic interference.
//! - **Deterministic**: given the same inputs always produces the same output.
//! - **Requires a base model**: the distance metric is defined relative to base.
//! - **Best for**: creating "safety-blended" models where you want each parameter
//!   to come from the model that diverged least from the foundation.
//!
//! # References
//!
//! Nearswap is an informal community method; the core idea appears in discussions
//! around MergeKit and related tooling circa 2024. It is conceptually related to
//! "model stock" selection but operates per-element rather than per-layer.

use super::MergeMethod;
use crate::{MergeError, MergeParameters, Result};
use mlx_rs::Array;

/// Nearswap merge implementation.
///
/// For each parameter element, selects the value from the source model whose
/// value is closest to the base model at that position.
#[derive(Debug, Clone, Default)]
pub struct NearswapMerge;

impl NearswapMerge {
    /// Create a new Nearswap merge method.
    pub fn new() -> Self {
        Self
    }

    /// Compute task vector δ = W_ft - W_base.
    fn task_vector(tensor: &Array, base: &Array) -> Result<Array> {
        Ok(tensor.subtract(base)?)
    }
}

impl MergeMethod for NearswapMerge {
    fn name(&self) -> &'static str {
        "nearswap"
    }

    fn description(&self) -> &'static str {
        "Parameter-wise selection of the value nearest to the base model"
    }

    fn requires_base_model(&self) -> bool {
        true
    }

    fn merge(
        &self,
        tensors: &[Array],
        base_tensor: Option<&Array>,
        params: &[MergeParameters],
        global_params: &MergeParameters,
    ) -> Result<Array> {
        if tensors.is_empty() {
            return Err(MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            });
        }

        let base = base_tensor.ok_or_else(|| MergeError::BaseModelRequired {
            method: self.name().to_string(),
        })?;

        let lambda = global_params.lambda();

        // Fast path for a single model: no selection needed.
        if tensors.len() == 1 {
            let merged_params = global_params.merge_with(&params[0]);
            let _ = merged_params; // weight not used in selection, but honour lambda
            if (lambda - 1.0).abs() < 1e-7 {
                return Ok(tensors[0].clone());
            }
            // Interpolate: base + lambda * (t - base)
            let tv = Self::task_vector(&tensors[0], base)?;
            let scaled = tv.multiply(Array::from_f32(lambda))?;
            return Ok(base.add(&scaled)?);
        }

        // General case: for each element pick the model closest to base.
        //
        // We work in flat (1-D) space and restore the original shape afterwards.
        let original_shape = base.shape().to_vec();

        // Flatten base and all source tensors for element-wise processing.
        let base_flat = base.reshape(&[-1])?;
        base_flat.eval()?;
        let base_vals: Vec<f32> = base_flat.as_slice().to_vec();
        let n = base_vals.len();

        // Flatten every source tensor and evaluate eagerly.
        let mut flat_tensors: Vec<Vec<f32>> = Vec::with_capacity(tensors.len());
        for t in tensors.iter() {
            let flat = t.reshape(&[-1])?;
            flat.eval()?;
            flat_tensors.push(flat.as_slice().to_vec());
        }

        // For each position, select the value from the nearest model.
        let mut result_vals = vec![0.0_f32; n];

        for i in 0..n {
            let base_val = base_vals[i];

            // Find the model with the smallest |W_m[i] - W_base[i]|.
            let mut best_model = 0;
            let mut best_dist = (flat_tensors[0][i] - base_val).abs();

            for (m, model_vals) in flat_tensors.iter().enumerate().skip(1) {
                let dist = (model_vals[i] - base_val).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_model = m;
                }
            }

            let selected = flat_tensors[best_model][i];

            // Apply lambda interpolation if requested.
            if (lambda - 1.0).abs() < 1e-7 {
                result_vals[i] = selected;
            } else {
                // W_merged[i] = W_base[i] + lambda * (W_{m*}[i] - W_base[i])
                result_vals[i] = base_val + lambda * (selected - base_val);
            }
        }

        let result_flat = Array::from_slice(&result_vals, &[n as i32]);
        Ok(result_flat.reshape(&original_shape)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nearswap_name_and_trait() {
        let ns = NearswapMerge::new();
        assert_eq!(ns.name(), "nearswap");
        assert!(ns.requires_base_model());
    }

    #[test]
    fn test_nearswap_single_model_passthrough() {
        let ns = NearswapMerge::new();
        let base = Array::from_slice(&[0.0_f32, 0.0, 0.0], &[3]);
        let t1 = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);

        let params = vec![MergeParameters::default()];
        let global = MergeParameters::default(); // lambda=1.0

        let result = ns
            .merge(std::slice::from_ref(&t1), Some(&base), &params, &global)
            .unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // Single model, lambda=1 → passthrough
        assert!((result_slice[0] - 1.0).abs() < 1e-5);
        assert!((result_slice[1] - 2.0).abs() < 1e-5);
        assert!((result_slice[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_nearswap_selects_nearest_per_element() {
        // base = [0, 0, 0]
        // t1   = [1, 5, 1]  — close to base at positions 0 and 2
        // t2   = [4, 1, 4]  — close to base at position 1
        //
        // Expected selection: [t1[0]=1, t2[1]=1, t1[2]=1]
        let ns = NearswapMerge::new();

        let base = Array::from_slice(&[0.0_f32, 0.0, 0.0], &[3]);
        let t1 = Array::from_slice(&[1.0_f32, 5.0, 1.0], &[3]);
        let t2 = Array::from_slice(&[4.0_f32, 1.0, 4.0], &[3]);

        let params = vec![MergeParameters::default(), MergeParameters::default()];
        let global = MergeParameters::default();

        let result = ns.merge(&[t1, t2], Some(&base), &params, &global).unwrap();
        result.eval().unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        // Position 0: |1-0|=1 vs |4-0|=4 → t1 wins → 1.0
        assert!(
            (result_slice[0] - 1.0).abs() < 1e-5,
            "pos 0: expected 1.0, got {}",
            result_slice[0]
        );
        // Position 1: |5-0|=5 vs |1-0|=1 → t2 wins → 1.0
        assert!(
            (result_slice[1] - 1.0).abs() < 1e-5,
            "pos 1: expected 1.0, got {}",
            result_slice[1]
        );
        // Position 2: |1-0|=1 vs |4-0|=4 → t1 wins → 1.0
        assert!(
            (result_slice[2] - 1.0).abs() < 1e-5,
            "pos 2: expected 1.0, got {}",
            result_slice[2]
        );
    }

    #[test]
    fn test_nearswap_with_negative_values() {
        // base = [2, 2, 2]
        // t1   = [3, 0, 3]   — distances: [1, 2, 1]
        // t2   = [2.5, 2.1, 0] — distances: [0.5, 0.1, 2]
        //
        // Expected: pos 0 → t2 (0.5 < 1), pos 1 → t2 (0.1 < 2), pos 2 → t1 (1 < 2)
        let ns = NearswapMerge::new();

        let base = Array::from_slice(&[2.0_f32, 2.0, 2.0], &[3]);
        let t1 = Array::from_slice(&[3.0_f32, 0.0, 3.0], &[3]);
        let t2 = Array::from_slice(&[2.5_f32, 2.1, 0.0], &[3]);

        let params = vec![MergeParameters::default(), MergeParameters::default()];
        let global = MergeParameters::default();

        let result = ns.merge(&[t1, t2], Some(&base), &params, &global).unwrap();
        result.eval().unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        assert!(
            (result_slice[0] - 2.5).abs() < 1e-5,
            "pos 0: expected 2.5, got {}",
            result_slice[0]
        );
        assert!(
            (result_slice[1] - 2.1).abs() < 1e-5,
            "pos 1: expected 2.1, got {}",
            result_slice[1]
        );
        assert!(
            (result_slice[2] - 3.0).abs() < 1e-5,
            "pos 2: expected 3.0, got {}",
            result_slice[2]
        );
    }

    #[test]
    fn test_nearswap_lambda_interpolation() {
        // With lambda=0, result should equal base.
        let ns = NearswapMerge::new();

        let base = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let t1 = Array::from_slice(&[5.0_f32, 6.0, 7.0], &[3]);
        let t2 = Array::from_slice(&[10.0_f32, 11.0, 12.0], &[3]);

        let params = vec![MergeParameters::default(), MergeParameters::default()];
        let global = MergeParameters {
            lambda: Some(0.0),
            ..Default::default()
        };

        let result = ns.merge(&[t1, t2], Some(&base), &params, &global).unwrap();
        result.eval().unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();
        let base_slice: Vec<f32> = base.as_slice().to_vec();

        for (r, b) in result_slice.iter().zip(base_slice.iter()) {
            assert!(
                (r - b).abs() < 1e-5,
                "lambda=0 should give base: expected {}, got {}",
                b,
                r
            );
        }
    }

    #[test]
    fn test_nearswap_lambda_half_interpolation() {
        // lambda=0.5 should interpolate halfway between base and selected.
        // base=[0], t1=[2], t2=[10]
        // nearest is t1 (|2-0|=2 < |10-0|=10)
        // result = 0 + 0.5*(2-0) = 1.0
        let ns = NearswapMerge::new();

        let base = Array::from_slice(&[0.0_f32], &[1]);
        let t1 = Array::from_slice(&[2.0_f32], &[1]);
        let t2 = Array::from_slice(&[10.0_f32], &[1]);

        let params = vec![MergeParameters::default(), MergeParameters::default()];
        let global = MergeParameters {
            lambda: Some(0.5),
            ..Default::default()
        };

        let result = ns.merge(&[t1, t2], Some(&base), &params, &global).unwrap();
        result.eval().unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        assert!(
            (result_slice[0] - 1.0).abs() < 1e-5,
            "expected 1.0, got {}",
            result_slice[0]
        );
    }

    #[test]
    fn test_nearswap_three_models() {
        // Verify three-model selection works correctly.
        // base=[0], t1=[1], t2=[3], t3=[0.5]
        // distances: [1, 3, 0.5] → t3 wins
        let ns = NearswapMerge::new();

        let base = Array::from_slice(&[0.0_f32], &[1]);
        let t1 = Array::from_slice(&[1.0_f32], &[1]);
        let t2 = Array::from_slice(&[3.0_f32], &[1]);
        let t3 = Array::from_slice(&[0.5_f32], &[1]);

        let params = vec![
            MergeParameters::default(),
            MergeParameters::default(),
            MergeParameters::default(),
        ];
        let global = MergeParameters::default();

        let result = ns
            .merge(&[t1, t2, t3], Some(&base), &params, &global)
            .unwrap();
        result.eval().unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();

        assert!(
            (result_slice[0] - 0.5).abs() < 1e-5,
            "expected t3=0.5, got {}",
            result_slice[0]
        );
    }

    #[test]
    fn test_nearswap_preserves_shape() {
        let ns = NearswapMerge::new();
        let base = Array::from_slice(&[0.0_f32; 12], &[3, 4]);
        let t1 = Array::from_slice(&[1.0_f32; 12], &[3, 4]);
        let t2 = Array::from_slice(&[2.0_f32; 12], &[3, 4]);

        let params = vec![MergeParameters::default(), MergeParameters::default()];
        let global = MergeParameters::default();

        let result = ns.merge(&[t1, t2], Some(&base), &params, &global).unwrap();
        assert_eq!(result.shape(), &[3, 4]);
    }

    #[test]
    fn test_nearswap_all_equal_to_base_returns_base() {
        // When all models equal the base, result must equal the base.
        let ns = NearswapMerge::new();

        let base = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let t1 = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
        let t2 = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);

        let params = vec![MergeParameters::default(), MergeParameters::default()];
        let global = MergeParameters::default();

        let result = ns.merge(&[t1, t2], Some(&base), &params, &global).unwrap();
        result.eval().unwrap();
        let result_slice: Vec<f32> = result.as_slice().to_vec();
        let base_slice: Vec<f32> = base.as_slice().to_vec();

        for (r, b) in result_slice.iter().zip(base_slice.iter()) {
            assert!((r - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_nearswap_empty_tensors_error() {
        let ns = NearswapMerge::new();
        let base = Array::from_slice(&[1.0_f32], &[1]);
        let result = ns.merge(&[], Some(&base), &[], &MergeParameters::default());
        assert!(result.is_err(), "should error on empty tensor list");
    }

    #[test]
    fn test_nearswap_no_base_error() {
        let ns = NearswapMerge::new();
        let t1 = Array::from_slice(&[1.0_f32], &[1]);
        let result = ns.merge(
            &[t1],
            None,
            &[MergeParameters::default()],
            &MergeParameters::default(),
        );
        assert!(result.is_err(), "should error when base model is missing");
    }
}
