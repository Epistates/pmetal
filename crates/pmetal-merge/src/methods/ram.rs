//! RAM – Reinforced Agent Merging.
//!
//! RAM (Hu et al., 2025) identifies "unique" vs "shared" parameter contributions
//! across models and handles them differently to reduce interference:
//!
//! 1. Compute task vectors: `delta_i = model_i − base`
//! 2. For each parameter position classify contributions:
//!    - **unique**: only one model has a non-trivial delta there → keep as-is
//!    - **shared**: more than one model has a non-trivial delta → average them
//! 3. Merged result:
//!    `base + sum(unique_deltas) + mean(shared_deltas)`
//!
//! **RAM+** (tensor-local variant) extends this with an adaptive rescaling
//! factor `λ = 1 + r * ρ` where `ρ = shared_count / unique_count` measures
//! how "conflicted" the tensor is, capped at `alpha`.
//!
//! Reference: <https://arxiv.org/abs/2601.13572>

use super::MergeMethod;
use crate::{MergeError, MergeParameters, Result};
use mlx_rs::Array;

// =============================================================================
// Shared tensor-preparation logic
// =============================================================================

/// Flattened task vectors and the classification masks derived from them.
struct RamVectors {
    /// Task vectors flattened to `[n_models, n_params]` (CPU f32 Vec).
    tv_flat: Vec<Vec<f32>>,
    /// Number of parameters.
    n_params: usize,
    /// Number of models.
    n_models: usize,
    /// `nonzero_mask[m][p]` – true when |delta_m[p]| > epsilon.
    nonzero_mask: Vec<Vec<bool>>,
    /// `contrib_counts[p]` – how many models have a non-trivial delta at p.
    contrib_counts: Vec<usize>,
}

impl RamVectors {
    fn prepare(tensors: &[Array], base: &Array, epsilon: f32) -> Result<(Vec<i32>, Self)> {
        let original_shape = base.shape().to_vec();
        let n_params = original_shape
            .iter()
            .map(|&d| d as usize)
            .product::<usize>();
        let n_models = tensors.len();

        let mut tv_flat = Vec::with_capacity(n_models);
        let mut nonzero_mask: Vec<Vec<bool>> = Vec::with_capacity(n_models);

        // Pull base onto CPU once
        let base_flat = base.reshape(&[-1])?;
        let base_vals: Vec<f32> = base_flat.as_slice().to_vec();

        for tensor in tensors {
            let t_flat = tensor.reshape(&[-1])?;
            let t_vals: Vec<f32> = t_flat.as_slice().to_vec();

            let delta: Vec<f32> = t_vals
                .iter()
                .zip(base_vals.iter())
                .map(|(t, b)| t - b)
                .collect();

            let mask: Vec<bool> = delta.iter().map(|d| d.abs() > epsilon).collect();

            tv_flat.push(delta);
            nonzero_mask.push(mask);
        }

        // contrib_counts[p] = number of models with nonzero delta at p
        let mut contrib_counts = vec![0usize; n_params];
        for mask in nonzero_mask.iter().take(n_models) {
            for p in 0..n_params {
                if mask[p] {
                    contrib_counts[p] += 1;
                }
            }
        }

        Ok((
            original_shape,
            RamVectors {
                tv_flat,
                n_params,
                n_models,
                nonzero_mask,
                contrib_counts,
            },
        ))
    }
}

// =============================================================================
// RAM merge
// =============================================================================

/// RAM – Reinforced Agent Merging (basic variant).
#[derive(Debug, Clone, Default)]
pub struct RamMerge {
    /// Enable RAM+ (tensor-local adaptive rescaling).
    plus: bool,
    /// RAM+ `r` parameter — scaling strength (default 0.1).
    r: f32,
    /// RAM+ `alpha` parameter — cap on `rho` (default 0.2).
    alpha: f32,
    /// Threshold below which a delta is considered zero.
    epsilon: f32,
}

impl RamMerge {
    /// Basic RAM merge.
    pub fn new() -> Self {
        Self {
            plus: false,
            r: 0.1,
            alpha: 0.2,
            epsilon: 1e-5,
        }
    }

    /// RAM+ with tensor-local adaptive rescaling.
    pub fn plus() -> Self {
        Self {
            plus: true,
            r: 0.1,
            alpha: 0.2,
            epsilon: 1e-5,
        }
    }

    /// Set the epsilon threshold (default 1e-5).
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set RAM+ `r` parameter.
    pub fn with_r(mut self, r: f32) -> Self {
        self.r = r;
        self
    }

    /// Set RAM+ `alpha` cap.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    fn merge_impl(&self, tensors: &[Array], base: &Array) -> Result<Array> {
        if tensors.is_empty() {
            return Ok(base.clone());
        }

        let (original_shape, vecs) = RamVectors::prepare(tensors, base, self.epsilon)?;
        let n_params = vecs.n_params;
        let n_models = vecs.n_models;

        // For RAM+: per-model lambda = 1 + r * clamp(rho, 0, alpha)
        // where rho = shared_count / unique_count  (both are per-model scalar counts)
        let lambdas: Vec<f32> = if self.plus {
            (0..n_models)
                .map(|m| {
                    // counts over all parameters for this model
                    let mut shared_count = 0usize;
                    let mut unique_count = 0usize;
                    for p in 0..n_params {
                        if vecs.nonzero_mask[m][p] {
                            if vecs.contrib_counts[p] > 1 {
                                shared_count += 1;
                            } else {
                                unique_count += 1;
                            }
                        }
                    }
                    let rho = if unique_count == 0 {
                        0.0
                    } else {
                        shared_count as f32 / unique_count as f32
                    };
                    1.0 + self.r * rho.clamp(0.0, self.alpha)
                })
                .collect()
        } else {
            vec![1.0f32; n_models]
        };

        // Build merged task vector on CPU
        let mut merged = vec![0.0f32; n_params];

        for (p, merged_p) in merged.iter_mut().enumerate() {
            let cnt = vecs.contrib_counts[p];
            if cnt == 0 {
                continue;
            }

            if cnt == 1 {
                // Unique: exactly one model contributes — keep it with its lambda
                for (mask, (tv, &lam)) in vecs
                    .nonzero_mask
                    .iter()
                    .zip(vecs.tv_flat.iter().zip(lambdas.iter()))
                {
                    if mask[p] {
                        *merged_p = tv[p] * lam;
                        break;
                    }
                }
            } else {
                // Shared: average over all contributing models (no lambda in basic RAM)
                let mut sum = 0.0f32;
                let mut contributing = 0usize;
                for (mask, tv) in vecs.nonzero_mask.iter().zip(vecs.tv_flat.iter()) {
                    if mask[p] {
                        sum += tv[p];
                        contributing += 1;
                    }
                }
                *merged_p = sum / contributing as f32;
            }
        }

        let delta = Array::from_slice(&merged, &[n_params as i32]);
        let delta = delta.reshape(&original_shape)?;
        Ok(base.add(&delta)?)
    }
}

impl MergeMethod for RamMerge {
    fn name(&self) -> &'static str {
        if self.plus { "ram_plus" } else { "ram" }
    }

    fn description(&self) -> &'static str {
        if self.plus {
            "Reinforced Agent Merging Plus (tensor-local adaptive rescaling)"
        } else {
            "Reinforced Agent Merging (unique/shared parameter classification)"
        }
    }

    fn requires_base_model(&self) -> bool {
        true
    }

    fn merge(
        &self,
        tensors: &[Array],
        base_tensor: Option<&Array>,
        _params: &[MergeParameters],
        _global_params: &MergeParameters,
    ) -> Result<Array> {
        let base = base_tensor.ok_or_else(|| MergeError::BaseModelRequired {
            method: self.name().to_string(),
        })?;
        self.merge_impl(tensors, base)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Array {
        Array::from_slice(&[0.0_f32, 0.0, 0.0, 0.0], &[4])
    }

    #[test]
    fn test_ram_single_model() {
        // With one model every delta is "unique": result = base + delta
        let ram = RamMerge::new();
        let base = base();
        let model = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
        let params = vec![MergeParameters::default()];
        let global = MergeParameters::default();

        let result = ram.merge(&[model], Some(&base), &params, &global).unwrap();
        let r: Vec<f32> = result.as_slice().to_vec();
        assert!((r[0] - 1.0).abs() < 1e-5);
        assert!((r[1] - 2.0).abs() < 1e-5);
        assert!((r[2] - 3.0).abs() < 1e-5);
        assert!((r[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_ram_unique_params_kept() {
        // Model A modifies position 0, model B modifies position 1 — both unique.
        let ram = RamMerge::new();
        let base = base();
        let m1 = Array::from_slice(&[1.0_f32, 0.0, 0.0, 0.0], &[4]);
        let m2 = Array::from_slice(&[0.0_f32, 2.0, 0.0, 0.0], &[4]);
        let params = vec![MergeParameters::default(); 2];
        let global = MergeParameters::default();

        let result = ram.merge(&[m1, m2], Some(&base), &params, &global).unwrap();
        let r: Vec<f32> = result.as_slice().to_vec();
        // Both unique, both kept
        assert!((r[0] - 1.0).abs() < 1e-5);
        assert!((r[1] - 2.0).abs() < 1e-5);
        assert!((r[2]).abs() < 1e-5);
        assert!((r[3]).abs() < 1e-5);
    }

    #[test]
    fn test_ram_shared_params_averaged() {
        // Both models modify position 0 → shared → averaged.
        let ram = RamMerge::new();
        let base = base();
        let m1 = Array::from_slice(&[1.0_f32, 0.0, 0.0, 0.0], &[4]);
        let m2 = Array::from_slice(&[3.0_f32, 0.0, 0.0, 0.0], &[4]);
        let params = vec![MergeParameters::default(); 2];
        let global = MergeParameters::default();

        let result = ram.merge(&[m1, m2], Some(&base), &params, &global).unwrap();
        let r: Vec<f32> = result.as_slice().to_vec();
        // Shared → average of deltas (1.0 + 3.0) / 2 = 2.0
        assert!((r[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_ram_empty_models_returns_base() {
        let ram = RamMerge::new();
        let base = base();
        let params: Vec<MergeParameters> = vec![];
        let global = MergeParameters::default();

        let result = ram.merge(&[], Some(&base), &params, &global).unwrap();
        let r: Vec<f32> = result.as_slice().to_vec();
        let b: Vec<f32> = base.as_slice().to_vec();
        assert_eq!(r, b);
    }

    #[test]
    fn test_ram_requires_base() {
        let ram = RamMerge::new();
        let model = Array::from_slice(&[1.0_f32, 2.0], &[2]);
        let params = vec![MergeParameters::default()];
        let global = MergeParameters::default();
        assert!(ram.merge(&[model], None, &params, &global).is_err());
    }

    #[test]
    fn test_ram_preserves_shape() {
        let ram = RamMerge::new();
        let base = Array::from_slice(&[0.0_f32; 12], &[3, 4]);
        let model = Array::from_slice(&[1.0_f32; 12], &[3, 4]);
        let params = vec![MergeParameters::default()];
        let global = MergeParameters::default();

        let result = ram.merge(&[model], Some(&base), &params, &global).unwrap();
        assert_eq!(result.shape(), &[3, 4]);
    }

    #[test]
    fn test_ram_plus_unique_rescaled() {
        // With RAM+, unique params get lambda > 1 when there are also shared params.
        // Here: positions 0 and 2 are shared, positions 1 and 3 are unique for m1.
        let ram_plus = RamMerge::plus();
        let base = base();
        // Both models touch position 0: shared
        // Only m1 touches position 1: unique
        let m1 = Array::from_slice(&[1.0_f32, 1.0, 1.0, 0.0], &[4]);
        let m2 = Array::from_slice(&[1.0_f32, 0.0, 1.0, 1.0], &[4]);
        let params = vec![MergeParameters::default(); 2];
        let global = MergeParameters::default();

        let result = ram_plus
            .merge(&[m1, m2], Some(&base), &params, &global)
            .unwrap();
        let r: Vec<f32> = result.as_slice().to_vec();

        // Shared positions (0, 2): averaged as in basic RAM → 1.0
        assert!((r[0] - 1.0).abs() < 1e-5, "shared pos 0: {}", r[0]);
        assert!((r[2] - 1.0).abs() < 1e-5, "shared pos 2: {}", r[2]);
        // Unique position (1 for m1, 3 for m2): lambda ≥ 1.0 so value ≥ 1.0
        assert!(r[1] >= 1.0 - 1e-5, "unique m1 pos 1: {}", r[1]);
        assert!(r[3] >= 1.0 - 1e-5, "unique m2 pos 3: {}", r[3]);
    }

    #[test]
    fn test_ram_plus_names() {
        assert_eq!(RamMerge::new().name(), "ram");
        assert_eq!(RamMerge::plus().name(), "ram_plus");
    }
}
