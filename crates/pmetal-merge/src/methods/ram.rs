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
use pmetal_runtime::compat::{Array, Dtype};

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
        let f32_code = Dtype::Float32.as_i32();
        let epsilon = Array::from_f32(self.epsilon);

        // Task vectors and, per model, where its delta is non-trivial, as 0/1.
        let deltas: Vec<Array> = tensors.iter().map(|t| t.subtract(base)).collect();
        let masks: Vec<Array> = deltas
            .iter()
            .map(|d| d.abs_val().greater(&epsilon).as_dtype(f32_code))
            .collect();
        let mut contributors = masks[0].clone();
        for mask in &masks[1..] {
            contributors = contributors.add(mask);
        }
        let unique = contributors.equal(&Array::from_f32(1.0)).as_dtype(f32_code);
        let shared = contributors
            .greater(&Array::from_f32(1.0))
            .as_dtype(f32_code);

        // RAM+: per-model lambda = 1 + r * clamp(rho, 0, alpha), rho being the
        // model's count of shared entries over its count of unique ones.
        let lambdas: Vec<f32> = if self.plus {
            masks
                .iter()
                .map(|mask| {
                    let shared_count = mask.multiply(&shared).sum_all().item_f32();
                    let unique_count = mask.multiply(&unique).sum_all().item_f32();
                    let rho = if unique_count == 0.0 {
                        0.0
                    } else {
                        shared_count / unique_count
                    };
                    1.0 + self.r * rho.clamp(0.0, self.alpha)
                })
                .collect()
        } else {
            vec![1.0; tensors.len()]
        };

        // Unique entries keep their one contributor's delta (times its lambda);
        // shared ones average over the models that moved them (no lambda in
        // basic RAM); untouched ones stay at the base.
        let mut unique_sum = deltas[0]
            .multiply(&masks[0])
            .multiply(&Array::from_f32(lambdas[0]));
        let mut shared_sum = deltas[0].multiply(&masks[0]);
        for ((delta, mask), &lambda) in deltas.iter().zip(&masks).zip(&lambdas).skip(1) {
            let moved = delta.multiply(mask);
            unique_sum = unique_sum.add(&moved.multiply(&Array::from_f32(lambda)));
            shared_sum = shared_sum.add(&moved);
        }
        let shared_mean = shared_sum.divide(&contributors.maximum(&Array::from_f32(1.0)));
        let merged = unique_sum
            .multiply(&unique)
            .add(&shared_mean.multiply(&shared));
        Ok(base.add(&merged))
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
        Array::from_f32_slice(&[0.0_f32, 0.0, 0.0, 0.0], &[4])
    }

    #[test]
    fn test_ram_single_model() {
        // With one model every delta is "unique": result = base + delta
        let ram = RamMerge::new();
        let base = base();
        let model = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
        let params = vec![MergeParameters::default()];
        let global = MergeParameters::default();

        let mut result = ram.merge(&[model], Some(&base), &params, &global).unwrap();
        let r = result.to_f32_vec(4).unwrap();
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
        let m1 = Array::from_f32_slice(&[1.0_f32, 0.0, 0.0, 0.0], &[4]);
        let m2 = Array::from_f32_slice(&[0.0_f32, 2.0, 0.0, 0.0], &[4]);
        let params = vec![MergeParameters::default(); 2];
        let global = MergeParameters::default();

        let mut result = ram.merge(&[m1, m2], Some(&base), &params, &global).unwrap();
        let r = result.to_f32_vec(4).unwrap();
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
        let m1 = Array::from_f32_slice(&[1.0_f32, 0.0, 0.0, 0.0], &[4]);
        let m2 = Array::from_f32_slice(&[3.0_f32, 0.0, 0.0, 0.0], &[4]);
        let params = vec![MergeParameters::default(); 2];
        let global = MergeParameters::default();

        let mut result = ram.merge(&[m1, m2], Some(&base), &params, &global).unwrap();
        let r = result.to_f32_vec(4).unwrap();
        // Shared → average of deltas (1.0 + 3.0) / 2 = 2.0
        assert!((r[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_ram_empty_models_returns_base() {
        let ram = RamMerge::new();
        let base = base();
        let params: Vec<MergeParameters> = vec![];
        let global = MergeParameters::default();

        let mut result = ram.merge(&[], Some(&base), &params, &global).unwrap();
        let r = result.to_f32_vec(4).unwrap();
        let mut base_clone = base.clone();
        let b = base_clone.to_f32_vec(4).unwrap();
        assert_eq!(r, b);
    }

    #[test]
    fn test_ram_requires_base() {
        let ram = RamMerge::new();
        let model = Array::from_f32_slice(&[1.0_f32, 2.0], &[2]);
        let params = vec![MergeParameters::default()];
        let global = MergeParameters::default();
        assert!(ram.merge(&[model], None, &params, &global).is_err());
    }

    #[test]
    fn test_ram_preserves_shape() {
        let ram = RamMerge::new();
        let base = Array::from_f32_slice(&[0.0_f32; 12], &[3, 4]);
        let model = Array::from_f32_slice(&[1.0_f32; 12], &[3, 4]);
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
        let m1 = Array::from_f32_slice(&[1.0_f32, 1.0, 1.0, 0.0], &[4]);
        let m2 = Array::from_f32_slice(&[1.0_f32, 0.0, 1.0, 1.0], &[4]);
        let params = vec![MergeParameters::default(); 2];
        let global = MergeParameters::default();

        let mut result = ram_plus
            .merge(&[m1, m2], Some(&base), &params, &global)
            .unwrap();
        let r = result.to_f32_vec(4).unwrap();

        // Shared positions (0, 2): averaged as in basic RAM → 1.0
        assert!((r[0] - 1.0).abs() < 1e-5, "shared pos 0: {}", r[0]);
        assert!((r[2] - 1.0).abs() < 1e-5, "shared pos 2: {}", r[2]);
        // Unique position (1 for m1, 3 for m2): lambda ≥ 1.0 so value ≥ 1.0
        assert!(r[1] >= 1.0 - 1e-5, "unique m1 pos 1: {}", r[1]);
        assert!(r[3] >= 1.0 - 1e-5, "unique m2 pos 3: {}", r[3]);
    }

    #[test]
    fn ram_plus_rescales_unique_entries_by_the_capped_conflict_ratio() {
        // m1 = [1, 1, 1, 0], m2 = [1, 0, 1, 1] over a zero base: positions 0
        // and 2 are shared, 1 (m1) and 3 (m2) unique. Each model has 2 shared
        // and 1 unique entry, so rho = 2, capped at alpha = 0.2, and the
        // unique entries are scaled by 1 + r * 0.2 = 1.02.
        let m1 = Array::from_f32_slice(&[1.0_f32, 1.0, 1.0, 0.0], &[4]);
        let m2 = Array::from_f32_slice(&[1.0_f32, 0.0, 1.0, 1.0], &[4]);
        let params = vec![MergeParameters::default(); 2];
        let result = RamMerge::plus()
            .merge(
                &[m1, m2],
                Some(&base()),
                &params,
                &MergeParameters::default(),
            )
            .unwrap();
        crate::test_util::assert_close(
            &crate::test_util::values(&result),
            &[1.0, 1.02, 1.0, 1.02],
            1e-6,
        );
    }

    #[test]
    fn test_ram_plus_names() {
        assert_eq!(RamMerge::new().name(), "ram");
        assert_eq!(RamMerge::plus().name(), "ram_plus");
    }
}
