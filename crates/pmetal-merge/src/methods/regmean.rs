//! RegMean (Jin et al., 2023 — *Dataless Knowledge Fusion by Merging Weights of Language Models*).
//!
//! For *linear-layer* weights, RegMean has a closed-form merge in terms of
//! per-model Gram matrices `G_i = Xᵢᵀ Xᵢ`, where `Xᵢ` is the layer's input
//! activations on a calibration set. With the paper's `[in, out]` weights:
//!
//! ```text
//! W_merged = (Σ_i G̃_i)⁻¹ · (Σ_i G̃_i Wᵢ)
//! ```
//!
//! Checkpoints store the transpose, `[out, in]`, so the same merge reads
//! `W = (Σ_i Wᵢ G̃_i)(Σ_i G̃_i)⁻¹`, which is what is computed. `G̃_i` is `G_i`
//! with its off-diagonal entries scaled by `alpha` (the paper's 0.9), which
//! shrinks the cross-feature terms toward plain averaging.
//!
//! Intuition: each model's weight matrix is locally optimal for its own
//! input distribution; weighting by `G_i` makes the merged weights optimal
//! against the union of those distributions in least-squares sense.
//!
//! For non-linear-shaped weights (norms, biases, embeddings) RegMean has
//! no closed form, so this implementation falls back to a plain weighted
//! mean for any tensor whose Gram matrix is missing or whose shape is not
//! 2-D (`[out_features, in_features]`).
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
//! `(Σ_i G̃_i)` may be ill-conditioned. A `ridge · I` regularizer (Tikhonov)
//! is added before solving, with `ridge` scaled by the mean diagonal of the
//! sum so the choice is scale-invariant (default `1e-3`). The system is
//! symmetric positive definite and is solved by a Cholesky factorisation in
//! f64; a matrix that is still not positive definite is an error that names
//! the tensor rather than a silent identity.
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
    /// Factor applied to the off-diagonal entries of each Gram matrix before
    /// merging (the paper's alpha, 0.9); `1.0` is the unscaled closed form.
    alpha: f32,
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
            alpha: 0.9,
            fallback_to_mean: true,
            loaders: Mutex::new(None),
        }
    }

    /// Override the Tikhonov ridge.
    pub fn with_ridge(mut self, ridge: f32) -> Self {
        self.ridge = ridge.max(0.0);
        self
    }

    /// Set the off-diagonal scale alpha of the Gram matrices (the paper uses
    /// 0.9; `1.0` disables it).
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
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
            if loader.has_tensor(name) {
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


    /// Closed-form merge for one linear layer.
    ///
    /// `weights[i]` is the layer's weight in checkpoint layout,
    /// `[out_features, in_features]`, and `grams[i]` its Gram matrix
    /// `Gᵢ = XᵢᵀXᵢ` of the layer's inputs, `[in_features, in_features]`. The
    /// paper writes the weight as `[in, out]` and merges
    /// `(Σ G̃ᵢ)⁻¹ Σ G̃ᵢ Wᵢ`; in checkpoint layout that is
    /// `(Σ Wᵢ G̃ᵢ)(Σ G̃ᵢ + ridge·I)⁻¹`, where `G̃ᵢ` is `Gᵢ` with its
    /// off-diagonal entries scaled by `alpha`.
    fn regmean_2d(&self, name: &str, weights: &[Array], grams: &[Array]) -> Result<Array> {
        if weights.len() != grams.len() {
            return Err(MergeError::InvalidConfig(format!(
                "regmean: gram count {} ≠ weight count {}",
                grams.len(),
                weights.len()
            )));
        }
        let (out_features, in_features) = (weights[0].dim(0), weights[0].dim(1));
        let gram_shape = [in_features, in_features];
        for (weight, gram) in weights.iter().zip(grams) {
            if weight.shape() != [out_features, in_features] || gram.shape() != gram_shape {
                return Err(MergeError::ShapeMismatch {
                    name: format!("regmean '{name}' (gram for weight [out, in])"),
                    expected: gram_shape.to_vec(),
                    actual: gram.shape().to_vec(),
                });
            }
        }

        // G̃ = alpha · G + (1 - alpha) · diag(G): scale the off-diagonal part.
        let eye = Array::eye(in_features, pmetal_runtime::compat::Dtype::Float32);
        let scaled = |gram: &Array| -> Array {
            if (self.alpha - 1.0).abs() < f32::EPSILON {
                return gram.clone();
            }
            let diagonal = gram.multiply(&eye);
            gram.subtract(&diagonal)
                .multiply(&Array::from_f32(self.alpha))
                .add(&diagonal)
        };

        let mut sum_g: Option<Array> = None;
        let mut sum_wg: Option<Array> = None;
        for (weight, gram) in weights.iter().zip(grams) {
            let g = scaled(gram);
            let wg = weight.matmul(&g);
            sum_g = Some(match sum_g {
                Some(s) => s.add(&g),
                None => g,
            });
            sum_wg = Some(match sum_wg {
                Some(s) => s.add(&wg),
                None => wg,
            });
        }
        let (mut sum_g, sum_wg) = (sum_g.expect("non-empty"), sum_wg.expect("non-empty"));

        // Tikhonov ridge, scale-invariant: ridge · mean(diag(ΣG̃)).
        if self.ridge > 0.0 {
            let mean_diagonal = sum_g.multiply(&eye).sum_all().item_f32() / in_features as f32;
            let lambda = self.ridge * mean_diagonal.max(1e-12);
            sum_g = sum_g.add(&eye.multiply(&Array::from_f32(lambda)));
        }

        // X (ΣG̃) = ΣWG̃ with ΣG̃ symmetric positive definite: solve row by row.
        let n = in_features as usize;
        let a: Vec<f64> = host_values(&sum_g, n * n, name)?;
        let b: Vec<f64> = host_values(&sum_wg, out_features as usize * n, name)?;
        let x = cholesky_solve_rows(&a, n, &b).ok_or_else(|| {
            MergeError::InvalidConfig(format!(
                "regmean '{name}': the summed Gram matrix is not positive definite; \
                 raise `ridge` or calibrate on more data"
            ))
        })?;
        let x: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        Ok(Array::from_f32_slice(&x, &[out_features, in_features]))
    }
}

/// `array` as host f64, failing if the runtime cannot read it.
fn host_values(array: &Array, n: usize, name: &str) -> Result<Vec<f64>> {
    array
        .clone()
        .to_f32_vec(n)
        .map(|v| v.into_iter().map(f64::from).collect())
        .ok_or_else(|| {
            MergeError::InvalidConfig(format!("regmean '{name}': failed to read a matrix"))
        })
}

/// Solve `x · A = b_row` for every row of `b`, for symmetric positive
/// definite `A` (`n × n`, row-major) and `b` of `m × n`, by Cholesky
/// factorisation `A = L Lᵀ` in f64. Returns `None` if `A` is not positive
/// definite (a non-positive pivot).
fn cholesky_solve_rows(a: &[f64], n: usize, b: &[f64]) -> Option<Vec<f64>> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len() % n, 0);
    // Lower-triangular L (row-major) and its transpose U (also row-major),
    // so both substitution sweeps read contiguous rows.
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let dot: f64 = l[i * n..i * n + j]
                .iter()
                .zip(&l[j * n..j * n + j])
                .map(|(x, y)| x * y)
                .sum();
            let sum = a[i * n + j] - dot;
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    let mut u = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            u[j * n + i] = l[i * n + j];
        }
    }

    // x A = b  <=>  A xᵀ = bᵀ  <=>  L (Lᵀ xᵀ) = bᵀ (A is symmetric).
    let mut out = vec![0.0_f64; b.len()];
    let mut y = vec![0.0_f64; n];
    for (b_row, x_row) in b.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        for i in 0..n {
            let dot: f64 = l[i * n..i * n + i].iter().zip(&y[..i]).map(|(p, q)| p * q).sum();
            y[i] = (b_row[i] - dot) / l[i * n + i];
        }
        for i in (0..n).rev() {
            let dot: f64 = u[i * n + i + 1..(i + 1) * n]
                .iter()
                .zip(&x_row[i + 1..])
                .map(|(p, q)| p * q)
                .sum();
            x_row[i] = (y[i] - dot) / u[i * n + i];
        }
    }
    Some(out)
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
            Some(grams) => self.regmean_2d(name, tensors, &grams),
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

    /// Checkpoint weights are `[out_features, in_features]` and the Gram
    /// matrices `[in_features, in_features]`, so a non-square layer must merge
    /// as `(sum_i W_i G_i) (sum_i G_i)^-1`.
    #[test]
    fn non_square_layer_uses_the_checkpoint_layout() {
        let workdir = tempfile::tempdir().unwrap();
        let g_a = workdir.path().join("g_a");
        let g_b = workdir.path().join("g_b");
        // G1 = diag(3, 1), G2 = diag(1, 3)
        write_grams(&g_a, &[("w", vec![3.0_f32, 0.0, 0.0, 1.0], vec![2, 2])]);
        write_grams(&g_b, &[("w", vec![1.0_f32, 0.0, 0.0, 3.0], vec![2, 2])]);

        let merge = RegMeanMerge::new(vec![g_a, g_b]).with_ridge(0.0);
        // One output row: W1 = [2, 0], W2 = [0, 2].
        let t1 = Array::from_f32_slice(&[2.0_f32, 0.0], &[1, 2]);
        let t2 = Array::from_f32_slice(&[0.0_f32, 2.0], &[1, 2]);
        let result = merge
            .merge_named(
                "w",
                &[t1, t2],
                None,
                &[MergeParameters::default(), MergeParameters::default()],
                &MergeParameters::default(),
            )
            .unwrap();
        assert_eq!(result.shape(), &[1, 2]);
        // (W1 G1 + W2 G2) = [6, 0] + [0, 6]; sum G = diag(4, 4); result [1.5, 1.5].
        let v = crate::test_util::values(&result);
        assert!((v[0] - 1.5).abs() < 1e-4 && (v[1] - 1.5).abs() < 1e-4, "{v:?}");
    }

    /// The solver recovers a known 3x3 system: rows x with x A = b.
    #[test]
    fn cholesky_solves_a_known_system() {
        // A = [[4, 1, 0], [1, 3, 1], [0, 1, 2]] (symmetric positive definite).
        let a = [4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        // x = [1, 2, 3] and [0, -1, 1]; b = x A.
        let x = [[1.0, 2.0, 3.0], [0.0, -1.0, 1.0]];
        let mut b = Vec::new();
        for row in &x {
            for j in 0..3 {
                b.push((0..3).map(|i| row[i] * a[i * 3 + j]).sum::<f64>());
            }
        }
        let got = cholesky_solve_rows(&a, 3, &b).unwrap();
        for (g, w) in got.iter().zip(x.iter().flatten()) {
            assert!((g - w).abs() < 1e-12, "{got:?}");
        }
    }

    #[test]
    fn cholesky_rejects_a_matrix_that_is_not_positive_definite() {
        // Eigenvalues 3 and -1.
        assert!(cholesky_solve_rows(&[1.0, 2.0, 2.0, 1.0], 2, &[1.0, 0.0]).is_none());
    }

    /// alpha scales the off-diagonal part of each Gram matrix before merging.
    #[test]
    fn alpha_shrinks_the_cross_feature_terms() {
        let workdir = tempfile::tempdir().unwrap();
        let g_a = workdir.path().join("g_a");
        let g_b = workdir.path().join("g_b");
        // G1 = [[2, 1], [1, 1]], G2 = [[1, 0], [0, 1]]
        write_grams(&g_a, &[("w", vec![2.0_f32, 1.0, 1.0, 1.0], vec![2, 2])]);
        write_grams(&g_b, &[("w", vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2])]);
        let run = |alpha: f32| {
            let merge = RegMeanMerge::new(vec![g_a.clone(), g_b.clone()])
                .with_ridge(0.0)
                .with_alpha(alpha);
            let result = merge
                .merge_named(
                    "w",
                    &[
                        Array::from_f32_slice(&[1.0_f32, 0.0], &[1, 2]),
                        Array::from_f32_slice(&[0.0_f32, 1.0], &[1, 2]),
                    ],
                    None,
                    &[MergeParameters::default(), MergeParameters::default()],
                    &MergeParameters::default(),
                )
                .unwrap();
            crate::test_util::values(&result)
        };
        // alpha = 1: W1 G1 = [2, 1], W2 G2 = [0, 1], sum [2, 2];
        // sum G = [[3, 1], [1, 2]], inverse = [[2, -1], [-1, 3]] / 5;
        // x = [2, 2] inverse = [2/5, 4/5].
        let full = run(1.0);
        assert!((full[0] - 0.4).abs() < 1e-5 && (full[1] - 0.8).abs() < 1e-5, "{full:?}");
        // alpha = 0: G̃ are diagonal: G̃1 = diag(2, 1), G̃2 = I;
        // W1 G̃1 + W2 G̃2 = [2, 1], sum G̃ = diag(3, 2); x = [2/3, 1/2].
        let diagonal = run(0.0);
        assert!(
            (diagonal[0] - 2.0 / 3.0).abs() < 1e-5 && (diagonal[1] - 0.5).abs() < 1e-5,
            "{diagonal:?}"
        );
    }
}
