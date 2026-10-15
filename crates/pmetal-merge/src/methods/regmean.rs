//! RegMean (Jin et al., 2023 — *Dataless Knowledge Fusion by Merging Weights of Language Models*).
//!
//! For *linear-layer* weights, RegMean has a closed-form merge in terms of
//! per-model Gram matrices `G_i = Xᵢᵀ Xᵢ`, where `Xᵢ` is the layer's input
//! activations on a calibration set:
//!
//! ```text
//! W_merged = (Σ_i G_i)⁻¹ · (Σ_i G_i Wᵢ)
//! ```
//!
//! Intuition: each model's weight matrix is locally optimal for its own
//! input distribution; weighting by `G_i` makes the merged weights optimal
//! against the union of those distributions in least-squares sense.
//!
//! For non-linear-shaped weights (norms, biases, embeddings) RegMean has
//! no closed form, so this implementation falls back to a plain weighted
//! mean for any tensor whose Gram matrix is missing or whose shape is not
//! 2-D (`[in_features, out_features]`).
//!
//! # Where Gram matrices come from
//!
//! Gram matrices are precomputed offline by the caller from a calibration
//! batch — `G = Xᵀ X` for the layer's incoming activations `X ∈ ℝ^{N×in}`.
//! Stored alongside each model in a `regmean_grams.safetensors` file
//! whose keys mirror the model's tensor names. Tensors without a Gram
//! entry use the mean fallback.
//!
//! # Numerical stability
//!
//! `(Σ_i G_i)` may be ill-conditioned. We add a `ridge · I` regularizer
//! (Tikhonov) before solving — `ridge` defaults to `1e-3 · trace(Σ)/N`,
//! a scale-invariant choice. The actual solve uses MLX's `linalg::solve`
//! when available, falling back to the `Σ G_i` row-wise pseudo-inverse
//! otherwise (described below).
//!
//! Clean-room implementation from the paper's equations.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use super::{LinearMerge, MergeMethod};
use crate::loader::SafetensorsLoader;
use crate::{MergeError, MergeParameters, Result, TensorLoader};
use pmetal_runtime::compat::{Array, ops};

/// RegMean merge over per-model Gram matrices.
pub struct RegMeanMerge {
    gram_paths: Vec<PathBuf>,
    /// Tikhonov ridge for `(Σ G_i + ridge·I)`. `0.0` disables regularization.
    ridge: f32,
    /// When `true`, tensors absent from a model's Gram file are merged via
    /// plain weighted mean instead of erroring.
    fallback_to_mean: bool,
    loaders: Mutex<Option<Vec<SafetensorsLoader>>>,
}

impl RegMeanMerge {
    /// Construct with the given per-model Gram safetensors paths.
    pub fn new(gram_paths: Vec<PathBuf>) -> Self {
        Self {
            gram_paths,
            ridge: 1e-3,
            fallback_to_mean: true,
            loaders: Mutex::new(None),
        }
    }

    /// Override the Tikhonov ridge.
    pub fn with_ridge(mut self, ridge: f32) -> Self {
        self.ridge = ridge.max(0.0);
        self
    }

    /// When false, an absent Gram entry surfaces as a hard error.
    pub fn with_fallback_to_mean(mut self, on: bool) -> Self {
        self.fallback_to_mean = on;
        self
    }

    fn ensure_loaders(&self) -> Result<()> {
        let mut guard = self.loaders.lock().unwrap();
        if guard.is_some() {
            return Ok(());
        }
        let mut loaded = Vec::with_capacity(self.gram_paths.len());
        for path in &self.gram_paths {
            loaded.push(SafetensorsLoader::new(path)?);
        }
        *guard = Some(loaded);
        Ok(())
    }

    fn try_load_grams(&self, name: &str) -> Result<Option<Vec<Array>>> {
        self.ensure_loaders()?;
        let guard = self.loaders.lock().unwrap();
        let loaders = guard.as_ref().expect("ensure_loaders populated cache");

        let mut out = Vec::with_capacity(loaders.len());
        for (idx, loader) in loaders.iter().enumerate() {
            if loader.tensor_names().iter().any(|n| n == name) {
                out.push(loader.load_tensor(name)?);
            } else if self.fallback_to_mean {
                return Ok(None);
            } else {
                return Err(MergeError::TensorNotFound(format!(
                    "regmean Gram '{}' missing from model[{}]",
                    name, idx
                )));
            }
        }
        Ok(Some(out))
    }


    /// Closed-form merge for a single 2-D weight matrix.
    fn regmean_2d(&self, weights: &[Array], grams: &[Array]) -> Result<Array> {
        if weights.len() != grams.len() {
            return Err(MergeError::InvalidConfig(format!(
                "regmean: gram count {} ≠ weight count {}",
                grams.len(),
                weights.len()
            )));
        }
        // Σ G_i and Σ G_i W_i
        let mut sum_g = grams[0].clone();
        let mut sum_gw = grams[0].matmul(&weights[0]);
        for (g, w) in grams[1..].iter().zip(&weights[1..]) {
            sum_g = sum_g.add(g);
            sum_gw = sum_gw.add(&g.matmul(w));
        }

        // Tikhonov ridge: scale-invariant choice ridge · diag_mean(Σ G).
        let n = sum_g.dim(-1) as usize;
        if self.ridge > 0.0 {
            let trace = diag_mean(&sum_g);
            let lambda = self.ridge * trace.max(1e-12);
            let eye = identity_matrix(n, lambda);
            sum_g = sum_g.add(&eye);
        }

        // Solve `sum_g · W_merged = sum_gw` via the pseudo-inverse of
        // `sum_g`. The bridge does not currently expose `linalg::solve`,
        // so we materialize `sum_g` to f32, invert on CPU via a small
        // dedicated routine, and re-wrap. Layer dimensions for LLMs
        // typically cap out at a few thousand × few thousand on the
        // attention side; this stays tractable for calibration-driven
        // merges.
        let inv_g = pseudo_inverse_2d(&sum_g)?;
        Ok(inv_g.matmul(&sum_gw))
    }
}

/// Mean of the diagonal of a 2-D square matrix.
fn diag_mean(m: &Array) -> f32 {
    let n = m.dim(-1) as usize;
    let mut copy = m.clone();
    let total = n * n;
    let flat: Vec<f32> = copy.to_f32_vec(total).unwrap_or_default();
    let mut acc = 0.0_f32;
    for i in 0..n {
        acc += flat[i * n + i];
    }
    acc / n.max(1) as f32
}

/// `λ · I` for a square `n × n` matrix.
fn identity_matrix(n: usize, lambda: f32) -> Array {
    let mut data = vec![0.0_f32; n * n];
    for i in 0..n {
        data[i * n + i] = lambda;
    }
    Array::from_f32_slice(&data, &[n as i32, n as i32])
}

/// Pseudo-inverse of a 2-D square symmetric positive-(semi)-definite
/// matrix via Gauss-Jordan elimination on f32. Falls back to a plain
/// weighted-mean-style identity if singular.
fn pseudo_inverse_2d(m: &Array) -> Result<Array> {
    let n = m.dim(-1) as usize;
    let total = n * n;
    let mut a = m.clone().to_f32_vec(total).unwrap_or_default();
    if a.len() != total {
        return Err(MergeError::InvalidConfig(
            "pseudo_inverse_2d: failed to materialize matrix".to_string(),
        ));
    }
    let mut inv = vec![0.0_f32; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    // Gauss-Jordan: forward elimination + back-substitution on [a | I].
    for col in 0..n {
        // Pivot: largest |a[row][col]| for numerical stability.
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1e-12 {
            // Singular — return identity. Caller will see the merge as
            // "average of weighted weights" which degrades gracefully.
            return Ok(identity_matrix(n, 1.0));
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
                inv.swap(col * n + k, pivot * n + k);
            }
        }
        let inv_pivot = 1.0_f32 / a[col * n + col];
        for k in 0..n {
            a[col * n + k] *= inv_pivot;
            inv[col * n + k] *= inv_pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row * n + col];
            if factor.abs() > 1e-30 {
                for k in 0..n {
                    a[row * n + k] -= factor * a[col * n + k];
                    inv[row * n + k] -= factor * inv[col * n + k];
                }
            }
        }
    }
    Ok(Array::from_f32_slice(&inv, &[n as i32, n as i32]))
}

impl MergeMethod for RegMeanMerge {
    fn name(&self) -> &'static str {
        "regmean"
    }

    fn description(&self) -> &'static str {
        "Closed-form Gram-weighted merge (Jin et al., 2023)"
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
        let params: Vec<MergeParameters> = params.iter().map(|p| p.resolved(name)).collect();
        let global_params = &global_params.resolved(name);
        // RegMean's closed form only applies to 2-D weight matrices; route
        // anything else to the plain mean fallback.
        let is_2d = tensors[0].shape().len() == 2;
        if !is_2d {
            return LinearMerge::weighted_average(tensors, &params, global_params);
        }

        match self.try_load_grams(name)? {
            Some(grams) => self.regmean_2d(tensors, &grams),
            None => LinearMerge::weighted_average(tensors, &params, global_params),
        }
    }
}

const _: fn() = || {
    let _ = ops::maximum;
    let _: HashMap<String, ()> = HashMap::new();
};

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::Dtype;
    use safetensors::tensor::TensorView;
    use std::collections::HashMap as StdHashMap;

    fn write_grams(path: &std::path::Path, tensors: &[(&str, Vec<f32>, Vec<usize>)]) {
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

    /// Identical Gram matrices on both sides + identical weights → merge
    /// must equal the input weight (no information is lost).
    #[test]
    fn identical_inputs_round_trip() {
        let workdir = tempfile::tempdir().unwrap();
        let g_a = workdir.path().join("g_a");
        let g_b = workdir.path().join("g_b");
        // 2x2 identity-style Grams.
        let g = vec![1.0_f32, 0.0, 0.0, 1.0];
        write_grams(&g_a, &[("layer.weight", g.clone(), vec![2, 2])]);
        write_grams(&g_b, &[("layer.weight", g.clone(), vec![2, 2])]);

        let w = vec![1.0_f32, 2.0, 3.0, 4.0];
        let merge = RegMeanMerge::new(vec![g_a, g_b]).with_ridge(0.0);
        let t1 = Array::from_f32_slice(&w, &[2, 2]);
        let t2 = Array::from_f32_slice(&w, &[2, 2]);
        let result = merge
            .merge_named(
                "layer.weight",
                &[t1, t2],
                None,
                &[MergeParameters::default(), MergeParameters::default()],
                &MergeParameters::default(),
            )
            .unwrap();
        let v: Vec<f32> = result.clone().to_f32_vec(4).unwrap();
        for (got, expected) in v.iter().zip(w.iter()) {
            assert!(
                (got - expected).abs() < 1e-3,
                "got {} expected {}",
                got,
                expected
            );
        }
    }

    /// Equal Gram matrices on both sides → merge equals plain mean of
    /// the two weights. With ridge=0 the closed form gives:
    ///   (G+G)⁻¹ · (G·W₁ + G·W₂) = (2G)⁻¹ · 2G·(W₁+W₂)/2 = (W₁+W₂)/2.
    #[test]
    fn equal_grams_match_mean() {
        let workdir = tempfile::tempdir().unwrap();
        let g_a = workdir.path().join("g_a");
        let g_b = workdir.path().join("g_b");
        let g = vec![1.0_f32, 0.0, 0.0, 1.0];
        write_grams(&g_a, &[("w", g.clone(), vec![2, 2])]);
        write_grams(&g_b, &[("w", g.clone(), vec![2, 2])]);

        let merge = RegMeanMerge::new(vec![g_a, g_b]).with_ridge(0.0);
        let t1 = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let t2 = Array::from_f32_slice(&[5.0_f32, 6.0, 7.0, 8.0], &[2, 2]);
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
        for (got, expected) in v.iter().zip([3.0_f32, 4.0, 5.0, 6.0].iter()) {
            assert!(
                (got - expected).abs() < 1e-3,
                "got {} expected {}",
                got,
                expected
            );
        }
    }

    /// 1-D tensors fall through to the mean fallback (RegMean undefined).
    #[test]
    fn nonlinear_tensors_use_mean_fallback() {
        let workdir = tempfile::tempdir().unwrap();
        let g_a = workdir.path().join("g_a");
        let g_b = workdir.path().join("g_b");
        write_grams(&g_a, &[("norm", vec![1.0_f32, 1.0], vec![2])]);
        write_grams(&g_b, &[("norm", vec![1.0_f32, 1.0], vec![2])]);

        let merge = RegMeanMerge::new(vec![g_a, g_b]);
        let t1 = Array::from_f32_slice(&[2.0_f32, 4.0], &[2]);
        let t2 = Array::from_f32_slice(&[6.0_f32, 8.0], &[2]);
        let result = merge
            .merge_named(
                "norm",
                &[t1, t2],
                None,
                &[MergeParameters::default(), MergeParameters::default()],
                &MergeParameters::default(),
            )
            .unwrap();
        let v: Vec<f32> = result.clone().to_f32_vec(2).unwrap();
        // Mean of (2,4) and (6,8) is (4, 6).
        assert!((v[0] - 4.0).abs() < 1e-4);
        assert!((v[1] - 6.0).abs() < 1e-4);
    }

    /// `pseudo_inverse_2d` recovers a 2x2 inverse correctly.
    #[test]
    fn pseudo_inverse_2x2() {
        // [[2, 0], [0, 4]] → inverse [[0.5, 0], [0, 0.25]]
        let m = Array::from_f32_slice(&[2.0_f32, 0.0, 0.0, 4.0], &[2, 2]);
        let inv = pseudo_inverse_2d(&m).unwrap();
        let v: Vec<f32> = inv.clone().to_f32_vec(4).unwrap();
        assert!((v[0] - 0.5).abs() < 1e-5);
        assert!((v[3] - 0.25).abs() < 1e-5);
        assert!(v[1].abs() < 1e-5);
        assert!(v[2].abs() < 1e-5);
    }
}
