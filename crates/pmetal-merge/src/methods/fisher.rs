//! Fisher merging (Matena & Raffel, 2022 — *Merging Models with Fisher-Weighted Averaging*).
//!
//! Each model is paired with a diagonal Fisher information tensor `F_i` of
//! the same shape as its weights. The merged tensor weights each model's
//! contribution by its Fisher value at every parameter position:
//!
//! ```text
//! θ_merged = (Σ_i λ_i (F_i + ε) ⊙ θ_i) / (Σ_i λ_i (F_i + ε))
//! ```
//!
//! The diagonal Fisher captures parameter sensitivity to the per-model
//! likelihood — positions where one model is highly confident pull the
//! merged value toward that model. Compared to plain averaging this is
//! provably better when the per-model loss landscapes overlap weakly.
//!
//! # Where Fisher tensors come from
//!
//! Fisher tensors must be precomputed offline by the caller (typically by
//! squaring per-step gradients of the log-likelihood on a calibration
//! batch and accumulating). This crate reads them from a sibling
//! `fisher.safetensors` file in each model directory; the keys must match
//! the model's tensor names exactly. Tensors without a corresponding
//! Fisher entry fall back to plain weighted mean (so embeddings, norms,
//! biases — for which Fisher is rarely useful — still merge cleanly).
//!
//! # Edge cases
//!
//! * All-zero `F_i` at some position: the `ε` floor on each Fisher value
//!   keeps the denominator positive, and the merged value there is the plain
//!   λ-weighted mean of the models.
//! * Missing `fisher.safetensors`: `FisherMerge` errors out at the first
//!   tensor it cannot resolve unless `fallback_to_mean` is set.
//!
//! Clean-room implementation from the paper's equations.

use std::path::PathBuf;
use std::sync::Mutex;

use super::{LinearMerge, MergeMethod};
use crate::loader::SafetensorsLoader;
use crate::{MergeError, MergeParameters, Result, TensorLoader};
use pmetal_runtime::compat::{Array, ops};

/// Fisher-weighted average merge.
pub struct FisherMerge {
    /// Per-model paths to a `fisher.safetensors` file. Length must match
    /// the number of models being merged.
    fisher_paths: Vec<PathBuf>,
    /// Numerical-stability ridge added to `Σ F_i` denominator.
    eps: f32,
    /// When `true`, tensors absent from a model's Fisher file are merged
    /// via plain weighted mean instead of erroring out. Default `true`
    /// (norms, biases, and embeddings rarely have meaningful Fisher).
    fallback_to_mean: bool,
    /// Lazy-loaded Fisher loaders, one per model. `Mutex` so trait method
    /// `merge_named(&self, …)` can populate the cache.
    loaders: Mutex<Option<Vec<SafetensorsLoader>>>,
}

impl FisherMerge {
    /// Construct with the given per-model Fisher safetensors paths.
    pub fn new(fisher_paths: Vec<PathBuf>) -> Self {
        Self {
            fisher_paths,
            eps: 1e-8,
            fallback_to_mean: true,
            loaders: Mutex::new(None),
        }
    }

    /// Override the numerical stability ridge.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps.max(0.0);
        self
    }

    /// When false, an absent Fisher entry surfaces as a hard error
    /// instead of silently degrading to plain mean.
    pub fn with_fallback_to_mean(mut self, on: bool) -> Self {
        self.fallback_to_mean = on;
        self
    }

    fn ensure_loaders(&self) -> Result<()> {
        let mut guard = self.loaders.lock().unwrap();
        if guard.is_some() {
            return Ok(());
        }
        let mut loaded = Vec::with_capacity(self.fisher_paths.len());
        for path in &self.fisher_paths {
            loaded.push(SafetensorsLoader::new(path)?);
        }
        *guard = Some(loaded);
        Ok(())
    }

    /// Try to load Fisher tensors for `name` from each model. Returns
    /// `None` (and triggers the mean fallback) if any model lacks the
    /// entry while `fallback_to_mean` is enabled.
    fn try_load_fisher(&self, name: &str) -> Result<Option<Vec<Array>>> {
        self.ensure_loaders()?;
        let guard = self.loaders.lock().unwrap();
        let loaders = guard.as_ref().expect("ensure_loaders populated cache");

        let mut out = Vec::with_capacity(loaders.len());
        for (idx, loader) in loaders.iter().enumerate() {
            if loader.has_tensor(name) {
                out.push(loader.load_tensor(name)?);
            } else if self.fallback_to_mean {
                return Ok(None);
            } else {
                return Err(MergeError::TensorNotFound(format!(
                    "fisher tensor '{}' missing from model[{}]",
                    name, idx
                )));
            }
        }
        Ok(Some(out))
    }

    /// `Σ λ_i (F_i + ε) ⊙ θ_i / Σ λ_i (F_i + ε)`: the ε floor on every Fisher
    /// value is what makes a position that no model has any information
    /// about (all `F_i` zero) fall back to the λ-weighted mean instead of
    /// to zero.
    fn fisher_weighted_average(
        &self,
        tensors: &[Array],
        fishers: &[Array],
        lambdas: &[f32],
    ) -> Result<Array> {
        if tensors.len() != fishers.len() || tensors.len() != lambdas.len() {
            return Err(MergeError::InvalidConfig(format!(
                "fisher: {} models, {} Fisher tensors, {} weights",
                tensors.len(),
                fishers.len(),
                lambdas.len()
            )));
        }

        let eps = Array::from_f32(self.eps);
        let mut num: Option<Array> = None;
        let mut den: Option<Array> = None;
        for ((theta, fisher), &lambda) in tensors.iter().zip(fishers).zip(lambdas) {
            let weight = fisher.add(&eps).multiply(&Array::from_f32(lambda));
            let term = weight.multiply(theta);
            num = Some(match num {
                Some(n) => n.add(&term),
                None => term,
            });
            den = Some(match den {
                Some(d) => d.add(&weight),
                None => weight,
            });
        }
        let (num, den) = (num.expect("non-empty"), den.expect("non-empty"));
        Ok(num.divide(&den))
    }
}

impl MergeMethod for FisherMerge {
    fn name(&self) -> &'static str {
        "fisher"
    }

    fn description(&self) -> &'static str {
        "Fisher-weighted averaging (Matena & Raffel, 2022)"
    }

    fn requires_base_model(&self) -> bool {
        false
    }

    fn merge(
        &self,
        tensors: &[Array],
        _base_tensor: Option<&Array>,
        params: &[MergeParameters],
        global_params: &MergeParameters,
    ) -> Result<Array> {
        // Without a tensor name we cannot key into the Fisher store —
        // degrade cleanly to the weighted mean fallback. Callers should
        // route through `merge_named` to get the actual Fisher path.
        LinearMerge::weighted_average(tensors, params, global_params)
    }

    fn merge_named(
        &self,
        name: &str,
        tensors: &[Array],
        _base_tensor: Option<&Array>,
        params: &[MergeParameters],
        global_params: &MergeParameters,
    ) -> Result<Array> {
        if tensors.is_empty() {
            return Err(MergeError::NotEnoughModels {
                expected: 1,
                actual: 0,
            });
        }
        let params: Vec<MergeParameters> = params
            .iter()
            .map(|p| p.resolved_in(name, global_params.layer_count))
            .collect();
        let global_params = &global_params.resolved(name);
        match self.try_load_fisher(name)? {
            Some(fishers) => {
                let lambdas: Vec<f32> = params
                    .iter()
                    .map(|p| global_params.merge_with(p).weight())
                    .collect();
                self.fisher_weighted_average(tensors, &fishers, &lambdas)
            }
            None => LinearMerge::weighted_average(tensors, &params, global_params),
        }
    }
}

/// Suppress unused-import linter when `ops` isn't used directly in this
/// translation unit.
const _: fn() = || {
    let _ = ops::maximum;
};

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::Dtype;
    use safetensors::tensor::TensorView;
    use std::collections::HashMap as StdHashMap;

    fn write_fisher_safetensors(path: &std::path::Path, tensors: &[(&str, Vec<f32>, Vec<usize>)]) {
        std::fs::create_dir_all(path).unwrap();
        let bytes_keep: Vec<(String, Vec<u8>, Vec<usize>)> = tensors
            .iter()
            .map(|(n, vals, shape)| {
                let bytes: Vec<u8> = vals.iter().flat_map(|f| f.to_le_bytes()).collect();
                (n.to_string(), bytes, shape.clone())
            })
            .collect();
        let views: StdHashMap<&str, TensorView<'_>> = bytes_keep
            .iter()
            .map(|(n, b, s)| {
                (
                    n.as_str(),
                    TensorView::new(Dtype::F32, s.clone(), b).unwrap(),
                )
            })
            .collect();
        let payload = safetensors::serialize(views, None).unwrap();
        std::fs::write(path.join("model.safetensors"), payload).unwrap();
    }

    /// Identical Fisher across models reduces to plain mean.
    #[test]
    fn uniform_fisher_reduces_to_mean() {
        let workdir = tempfile::tempdir().unwrap();
        let f_a = workdir.path().join("fisher_a");
        let f_b = workdir.path().join("fisher_b");
        let ones = vec![1.0_f32, 1.0, 1.0, 1.0];
        write_fisher_safetensors(&f_a, &[("w", ones.clone(), vec![4])]);
        write_fisher_safetensors(&f_b, &[("w", ones.clone(), vec![4])]);

        let merge = FisherMerge::new(vec![f_a, f_b]);
        let t1 = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[4]);
        let t2 = Array::from_f32_slice(&[5.0_f32, 6.0, 7.0, 8.0], &[4]);
        let result = merge
            .merge_named(
                "w",
                &[t1, t2],
                None,
                &[MergeParameters::default(), MergeParameters::default()],
                &MergeParameters::default(),
            )
            .unwrap();
        let v: Vec<f32> = result.clone().to_f32_vec(4).unwrap();
        // (1+5)/2 = 3, (2+6)/2 = 4, (3+7)/2 = 5, (4+8)/2 = 6
        for (got, expected) in v.iter().zip([3.0_f32, 4.0, 5.0, 6.0].iter()) {
            assert!((got - expected).abs() < 1e-4, "{} vs {}", got, expected);
        }
    }

    /// Heavy Fisher on one side pulls the merged value toward that model.
    #[test]
    fn skewed_fisher_pulls_toward_dominant_model() {
        let workdir = tempfile::tempdir().unwrap();
        let f_a = workdir.path().join("fisher_a");
        let f_b = workdir.path().join("fisher_b");
        let strong = vec![100.0_f32, 100.0];
        let weak = vec![1.0_f32, 1.0];
        write_fisher_safetensors(&f_a, &[("w", strong, vec![2])]);
        write_fisher_safetensors(&f_b, &[("w", weak, vec![2])]);

        let merge = FisherMerge::new(vec![f_a, f_b]);
        let t1 = Array::from_f32_slice(&[10.0_f32, 20.0], &[2]);
        let t2 = Array::from_f32_slice(&[0.0_f32, 0.0], &[2]);
        let result = merge
            .merge_named(
                "w",
                &[t1, t2],
                None,
                &[MergeParameters::default(), MergeParameters::default()],
                &MergeParameters::default(),
            )
            .unwrap();
        let v: Vec<f32> = result.clone().to_f32_vec(2).unwrap();
        // Expected (100*10 + 1*0) / (101 + ε) ≈ 9.9
        assert!(v[0] > 9.5 && v[0] < 10.0, "got {}", v[0]);
        assert!(v[1] > 19.0 && v[1] < 20.0, "got {}", v[1]);
    }

    /// Missing tensor with `fallback_to_mean` reverts to plain weighted mean.
    #[test]
    fn missing_fisher_falls_back_to_mean() {
        let workdir = tempfile::tempdir().unwrap();
        let f_a = workdir.path().join("fisher_a");
        let f_b = workdir.path().join("fisher_b");
        // `f_a` has Fisher for "w" but `f_b` doesn't — the absent side
        // forces the mean fallback for the whole merge of "w".
        write_fisher_safetensors(&f_a, &[("w", vec![1.0_f32, 1.0], vec![2])]);
        write_fisher_safetensors(&f_b, &[("other", vec![1.0_f32, 1.0], vec![2])]);

        let merge = FisherMerge::new(vec![f_a, f_b]);
        let t1 = Array::from_f32_slice(&[2.0_f32, 4.0], &[2]);
        let t2 = Array::from_f32_slice(&[8.0_f32, 6.0], &[2]);
        let result = merge
            .merge_named(
                "w",
                &[t1, t2],
                None,
                &[MergeParameters::default(), MergeParameters::default()],
                &MergeParameters::default(),
            )
            .unwrap();
        let v: Vec<f32> = result.clone().to_f32_vec(2).unwrap();
        for (got, expected) in v.iter().zip([5.0_f32, 5.0].iter()) {
            assert!((got - expected).abs() < 1e-4, "{} vs {}", got, expected);
        }
    }

    /// Where no model has any Fisher information the merge is the plain mean
    /// of the models, not zero.
    #[test]
    fn positions_without_fisher_information_average_the_models() {
        let workdir = tempfile::tempdir().unwrap();
        let f_a = workdir.path().join("fisher_a");
        let f_b = workdir.path().join("fisher_b");
        // Column 0 has no information in either model; column 1 only in B.
        write_fisher_safetensors(&f_a, &[("w", vec![0.0_f32, 0.0], vec![2])]);
        write_fisher_safetensors(&f_b, &[("w", vec![0.0_f32, 5.0], vec![2])]);

        let merge = FisherMerge::new(vec![f_a, f_b]);
        let t1 = Array::from_f32_slice(&[2.0_f32, 2.0], &[2]);
        let t2 = Array::from_f32_slice(&[4.0_f32, 4.0], &[2]);
        let result = merge
            .merge_named(
                "w",
                &[t1, t2],
                None,
                &[MergeParameters::default(), MergeParameters::default()],
                &MergeParameters::default(),
            )
            .unwrap();
        let v = crate::test_util::values(&result);
        // (2 + 4) / 2 = 3 where F = 0; where only B is informed it takes over.
        assert!((v[0] - 3.0).abs() < 1e-4, "got {}", v[0]);
        assert!((v[1] - 4.0).abs() < 1e-3, "got {}", v[1]);
    }

    /// Model coefficients lambda_i scale each model's Fisher weight.
    #[test]
    fn model_weights_scale_the_fisher_terms() {
        let workdir = tempfile::tempdir().unwrap();
        let f_a = workdir.path().join("fisher_a");
        let f_b = workdir.path().join("fisher_b");
        write_fisher_safetensors(&f_a, &[("w", vec![1.0_f32], vec![1])]);
        write_fisher_safetensors(&f_b, &[("w", vec![1.0_f32], vec![1])]);

        let merge = FisherMerge::new(vec![f_a, f_b]).with_eps(0.0);
        let weight = |w: f32| MergeParameters {
            weight: Some(w.into()),
            ..Default::default()
        };
        let result = merge
            .merge_named(
                "w",
                &[
                    Array::from_f32_slice(&[10.0_f32], &[1]),
                    Array::from_f32_slice(&[0.0_f32], &[1]),
                ],
                None,
                &[weight(3.0), weight(1.0)],
                &MergeParameters::default(),
            )
            .unwrap();
        // (3 * 1 * 10 + 1 * 1 * 0) / (3 + 1) = 7.5
        let v = crate::test_util::values(&result);
        assert!((v[0] - 7.5).abs() < 1e-5, "got {}", v[0]);
    }
}
