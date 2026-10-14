//! Low-rank adapters, as a property of a linear layer rather than a layer of
//! their own.
//!
//! LoRA reparameterises a frozen linear map: `y = x·Wᵀ + b` becomes
//! `y = x·Wᵀ + b + s·((x·Aᵀ)·Bᵀ)`. That is the same shape as the optional bias
//! `Linear` already carries, so [`LoraAdapter`] hangs off [`Linear`](super::Linear) the way a
//! bias does, and an architecture never has to mention LoRA to be trainable
//! with it.
//!
//! This is the Rust reading of what PEFT does at runtime. It walks a
//! built model and swaps each targeted `Linear` for an adapted one
//! (`inject_adapter_in_model`); the base architecture
//! never names the adapter. Rust has no duck-typed attribute assignment, so the
//! swap becomes an `Option` on the layer instead — same property, same
//! separation, and the architecture files stay unaware.
//!
//! What this buys, concretely: there is one forward pass per architecture
//! instead of two. The whole class of bug where a fix lands in the inference
//! copy and never reaches the training copy cannot occur, because there is no
//! second copy to miss.

use super::{Array, Dtype, Exception, random};

/// A low-rank adapter attached to a [`Linear`](super::Linear).
///
/// `B` is zero-initialised, so a freshly attached adapter is a no-op and the
/// layer computes exactly what it did before. That is the contract the
/// train-versus-serve parity test rests on.
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Down-projection, `[rank, in_features]`.
    pub a: Array,
    /// Up-projection, `[out_features, rank]`. Zero at initialisation.
    pub b: Array,
    /// Adapter rank.
    pub rank: i32,
    /// `alpha / rank`, or `alpha / sqrt(rank)` under rsLoRA.
    pub scale: f32,
    /// `scale` as an array, kept so the hot path does not rebuild it per call.
    scale_arr: Array,
    /// Dropout applied to the adapter's input while training.
    pub dropout: f32,
    /// Whether dropout is live. Inference leaves this `false`.
    pub training: bool,
    /// DoRA magnitude, `[out_features, 1]`.
    ///
    /// `Some` turns the layer into DoRA: the combined output is rescaled by
    /// `m / ‖W‖_col`, which is a post-scale rather than another additive term.
    pub magnitude: Option<Array>,
    /// Set once the adapter has been folded into the base weight, after which
    /// it contributes nothing further.
    pub merged: bool,
    /// The base weight as it stood before a merge.
    ///
    /// Only populated for DoRA, whose merge is
    /// `m·(W + s·B·A) / ‖W + s·B·A‖` — not an addition, so subtracting the
    /// delta would not put `W` back.
    unmerged_weight: Option<Array>,
}

impl LoraAdapter {
    /// Build an adapter for a layer of shape `[out_features, in_features]`.
    ///
    /// `use_rslora` switches the scale to `alpha / sqrt(rank)`, per
    /// Kalajdzievski (2023), so the update's norm stops depending on rank.
    /// `A` starts uniform on `±1/√in_features` either way, as PEFT's does.
    pub fn new(
        in_features: i32,
        out_features: i32,
        rank: i32,
        alpha: f32,
        use_rslora: bool,
    ) -> Result<Self, Exception> {
        if rank <= 0 {
            return Err(Exception::custom(format!(
                "LoRA rank must be positive, got {rank}"
            )));
        }
        let scale = if use_rslora {
            alpha / (rank as f32).sqrt()
        } else {
            alpha / rank as f32
        };
        let a_bound = 1.0_f32 / (in_features as f32).sqrt();
        Ok(Self {
            a: random::uniform_range(-a_bound, a_bound, &[rank, in_features], Dtype::Float32),
            b: super::ops::zeros(&[out_features, rank], Dtype::Float32),
            rank,
            scale,
            scale_arr: Array::from_f32(scale),
            dropout: 0.0,
            training: false,
            magnitude: None,
            merged: false,
            unmerged_weight: None,
        })
    }

    /// Turn this into a DoRA adapter, seeding the magnitude from the base
    /// weight's column norms so the layer starts out unchanged.
    pub fn with_dora(mut self, base_weight: &Array) -> Self {
        self.set_dora(base_weight);
        self
    }

    /// In-place form of [`with_dora`](Self::with_dora), for callers holding a
    /// `&mut` to an already-attached adapter.
    pub fn set_dora(&mut self, base_weight: &Array) {
        // A trainable parameter, so f32 whatever the base weight's dtype.
        self.magnitude = Some(column_norms(&base_weight.as_dtype(Dtype::Float32.as_i32())));
    }

    /// Set the adapter's input dropout.
    pub fn with_dropout(mut self, p: f32) -> Self {
        self.dropout = p;
        self
    }

    /// `scale · B·A`, the dense update this adapter represents.
    pub fn delta(&self) -> Array {
        self.b.matmul(&self.a).multiply(&self.scale_arr)
    }

    /// The weight this adapter is equivalent to, folded.
    ///
    /// For LoRA that is `W + s·B·A`. For DoRA the update is renormalised:
    /// `m·(W + s·B·A) / ‖W + s·B·A‖_col`.
    ///
    /// Computed in the adapter's width and rounded once to `weight`'s dtype,
    /// so folding an adapter into a bf16 model leaves a bf16 weight.
    pub fn merged_weight(&self, weight: &Array) -> Array {
        let combined = weight.add(&self.delta());
        let merged = match &self.magnitude {
            Some(magnitude) => {
                let norm = column_norms(&combined).add_scalar(DORA_EPS);
                combined.multiply(&magnitude.divide(&norm))
            }
            None => combined,
        };
        merged.as_dtype(weight.dtype_raw())
    }

    /// Fold this adapter into `weight`, remembering how to undo it.
    pub(crate) fn merge_into(&mut self, weight: &mut Array) {
        if self.merged {
            return;
        }
        if self.magnitude.is_some() {
            self.unmerged_weight = Some(weight.clone());
        }
        *weight = self.merged_weight(weight);
        self.merged = true;
    }

    /// Undo [`merge_into`](Self::merge_into).
    pub(crate) fn unmerge_from(&mut self, weight: &mut Array) {
        if !self.merged {
            return;
        }
        *weight = match self.unmerged_weight.take() {
            Some(original) => original,
            None => weight.subtract(&self.delta()).as_dtype(weight.dtype_raw()),
        };
        self.merged = false;
    }

    /// Recompute the cached scale array after `scale` is assigned directly.
    pub fn refresh_scale(&mut self) {
        self.scale_arr = Array::from_f32(self.scale);
    }

    /// Number of trainable elements this adapter contributes.
    pub fn num_trainable_params(&self) -> usize {
        let count = |arr: &Array| arr.shape().iter().map(|d| *d as usize).product::<usize>();
        count(&self.a) + count(&self.b) + self.magnitude.as_ref().map_or(0, count)
    }

    /// Apply the adapter on top of a frozen linear map.
    ///
    /// Never materialises `W + s·B·A`: the update is applied as two small
    /// matmuls against the activations, which is the whole point of the
    /// factorisation.
    pub fn apply(&self, x: &Array, weight: &Array, bias: Option<&Array>) -> Array {
        let y = super::layers::linear_forward_array(x, weight, None);
        self.apply_to_product(
            x,
            y,
            &|| super::layers::fp8_weight_for_compute(weight),
            bias,
        )
    }

    /// [`apply`](Self::apply) given the base product `y = x·Wᵀ` already, for
    /// a base weight stored packed. `weight` produces the dense `W`, which
    /// only DoRA's normalisation reads.
    ///
    /// The adapter's arithmetic runs at its own width (its tensors are f32
    /// whatever the base is), and the layer's output comes back at the base
    /// product's, rounded once at the end, so a bf16 model stays bf16 through
    /// an adapted layer as it does through a plain one.
    pub(crate) fn apply_to_product(
        &self,
        x: &Array,
        y: Array,
        weight: &dyn Fn() -> Array,
        bias: Option<&Array>,
    ) -> Array {
        if self.merged {
            return add_bias(y, bias);
        }
        let out_dtype = y.dtype_raw();

        let x_in = if self.training && self.dropout > 0.0 {
            dropout(x, self.dropout)
        } else {
            x.clone()
        };
        let delta = x_in
            .matmul(&self.a.t())
            .matmul(&self.b.t())
            .multiply(&self.scale_arr);
        let y = y.add(&delta);

        // DoRA rescales each output row by `m / ‖W + s·B·A‖`, the norm of the
        // *combined* weight's row, as Liu et al. define it and as PEFT
        // computes it. Scaling the output per row is the same thing as
        // scaling the weight's rows, which is what makes `merged_weight`
        // agree with this exactly.
        let y = match &self.magnitude {
            Some(magnitude) => y.multiply(&self.dora_scale(magnitude, &weight())),
            None => y,
        };
        let y = if y.dtype_raw() == out_dtype {
            y
        } else {
            y.as_dtype(out_dtype)
        };
        add_bias(y, bias)
    }

    /// DoRA's per-row factor `m / (‖W + s·B·A‖ + ε)`, `[out]`.
    ///
    /// The norm is held constant for the gradient (the DoRA paper's §4.3, and
    /// what PEFT does): `m` and the additive update carry the training
    /// signal, and the backward neither differentiates through `W + s·B·A`
    /// nor keeps a dense copy of it.
    fn dora_scale(&self, magnitude: &Array, weight: &Array) -> Array {
        let norm = self.combined_row_norms(weight).stop_gradient();
        magnitude
            .squeeze_axes(&[-1])
            .divide(&norm.add_scalar(DORA_EPS))
    }

    /// `‖W + s·B·A‖` per row, `[out]`, in f32, without forming `B·A`:
    ///
    /// ```text
    /// ‖wᵢ + s·bᵢA‖² = ‖wᵢ‖² + 2s·bᵢ·(A·wᵢ) + s²·bᵢ(AAᵀ)bᵢᵀ
    /// ```
    ///
    /// Two `[out, rank]` products and one `[rank, rank]` Gram matrix stand
    /// in for the dense `[out, in]` update.
    fn combined_row_norms(&self, weight: &Array) -> Array {
        let f32_ = Dtype::Float32.as_i32();
        let (w, a, b) = (
            weight.as_dtype(f32_),
            self.a.as_dtype(f32_),
            self.b.as_dtype(f32_),
        );
        let w_sq = w.square().sum_axis(1, false);
        let cross = w.matmul(&a.t()).multiply(&b).sum_axis(1, false);
        let gram = a.matmul(&a.t());
        let b_sq = b.matmul(&gram).multiply(&b).sum_axis(1, false);
        // Cancellation can leave a hair under zero for a row the update all
        // but cancels.
        w_sq.add(&cross.mul_scalar(2.0 * self.scale))
            .add(&b_sq.mul_scalar(self.scale * self.scale))
            .maximum(&Array::scalar_like(0.0, &w_sq))
            .sqrt()
    }
}

/// DoRA's `ε`, added to the norm so a row the update zeroes does not divide
/// by zero. The same in the live forward and in the merged weight.
const DORA_EPS: f32 = 1e-6;

fn add_bias(y: Array, bias: Option<&Array>) -> Array {
    match bias {
        Some(b) => y.add(b),
        None => y,
    }
}

/// Per-output-row L2 norm of a `[out, in]` weight, shaped `[out, 1]`.
fn column_norms(weight: &Array) -> Array {
    weight.square().sum_axis(1, true).sqrt()
}

/// Inverted dropout: zero a fraction `p` of entries and scale the rest up, so
/// the expectation is unchanged and inference needs no rescaling. The result
/// has `x`'s dtype.
fn dropout(x: &Array, p: f32) -> Array {
    let keep = 1.0 - p;
    let mask = random::dropout_mask(keep, x.shape());
    x.multiply(&mask.as_dtype(x.dtype_raw()))
        .mul_scalar(1.0 / keep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat::layers::Linear;
    use crate::compat::{ModuleParameters, ModuleParametersExt};

    fn values(mut arr: Array) -> Vec<f32> {
        arr.eval().expect("eval");
        let n: usize = arr.shape().iter().map(|d| *d as usize).product();
        arr.to_f32_vec(n).expect("readable")
    }

    fn probe() -> Array {
        Array::from_f32_slice(&[0.5, -1.25, 2.0, 0.75, -0.5, 1.5], &[2, 3])
    }

    /// The contract the whole train-versus-serve parity test rests on: attaching
    /// an adapter must not change what the layer computes.
    #[test]
    fn a_fresh_adapter_changes_nothing() {
        let mut layer = Linear::new(3, 4, true).expect("layer");
        let before = values(layer.forward(&probe()));

        layer.attach_lora(2, 4.0, false).expect("attach");
        let after = values(layer.forward(&probe()));

        assert_eq!(
            before, after,
            "a zero-initialised adapter perturbed the layer"
        );
    }

    fn trained_layer(dora: bool) -> Linear {
        let mut layer = Linear::new(3, 4, false).expect("layer");
        let weight = layer.weight.value.clone();
        {
            let adapter = layer.attach_lora(2, 4.0, false).expect("attach");
            // Give B a non-zero value so the adapter actually contributes.
            adapter.b = random::uniform_range(-0.5, 0.5, &[4, 2], Dtype::Float32);
            if dora {
                adapter.set_dora(&weight);
            }
        }
        layer
    }

    /// Folding the update into the weight keeps the layer computing what it did
    /// while adapted. That holds for DoRA too, whose merge renormalises rather
    /// than adds.
    #[test]
    fn merging_preserves_the_adapted_output() {
        for dora in [false, true] {
            let mut layer = trained_layer(dora);
            let adapted = values(layer.forward(&probe()));

            layer.merge_lora();
            let merged = values(layer.forward(&probe()));

            for (a, m) in adapted.iter().zip(&merged) {
                assert!(
                    (a - m).abs() < 1e-4,
                    "dora={dora}: merged output {m} differs from adapted {a}"
                );
            }
        }
    }

    /// Merging is reversible, so a run can merge for an evaluation pass and then
    /// carry on training.
    #[test]
    fn unmerging_restores_the_base_weight() {
        for dora in [false, true] {
            let mut layer = trained_layer(dora);
            let before = values(layer.weight.value.clone());

            layer.merge_lora();
            layer.unmerge_lora();
            let after = values(layer.weight.value.clone());

            for (b, a) in before.iter().zip(&after) {
                assert!(
                    (b - a).abs() < 1e-5,
                    "dora={dora}: unmerge left the weight at {a}, not {b}"
                );
            }
            assert!(layer.is_adapted(), "unmerge dropped the adapter");
        }
    }

    /// Fusing folds the update in and drops the adapter, which is what
    /// `pmetal fuse` produces.
    #[test]
    fn fusing_leaves_an_ordinary_dense_layer() {
        let mut layer = trained_layer(false);
        let adapted = values(layer.forward(&probe()));

        layer.fuse_lora();
        assert!(!layer.is_adapted(), "fuse left the adapter attached");
        assert!(
            layer
                .flatten_params()
                .keys()
                .all(|k| !k.starts_with("lora_")),
            "fuse left adapter tensors in the parameter tree"
        );

        let fused = values(layer.forward(&probe()));
        for (a, m) in adapted.iter().zip(&fused) {
            assert!((a - m).abs() < 1e-5, "fused output {m} differs from {a}");
        }
    }

    /// Adapter tensors sit beside the weight, not under an `adapter` level.
    /// Every adapter file pmetal has ever written uses these names.
    #[test]
    fn adapter_parameters_flatten_beside_the_weight() {
        let mut layer = Linear::new(3, 4, true).expect("layer");
        assert_eq!(
            layer.flatten_params().len(),
            2,
            "an unadapted layer should expose exactly weight and bias"
        );

        layer.attach_lora(2, 4.0, false).expect("attach");
        let params = layer.flatten_params();
        for key in ["weight", "bias", "lora_a", "lora_b"] {
            assert!(
                params.contains_key(key),
                "missing `{key}` among {:?}",
                params.keys().collect::<Vec<_>>()
            );
        }
        assert!(
            !params.keys().any(|k| k.contains("adapter")),
            "adapter leaked an `adapter` level into the parameter path"
        );
    }

    /// An adapted layer trains its adapter and nothing else; an unadapted one is
    /// an ordinary layer and trains everything.
    #[test]
    fn only_the_adapter_is_trainable_once_attached() {
        let mut layer = Linear::new(3, 4, true).expect("layer");
        assert_eq!(layer.trainable_parameters().len(), 2);

        layer.attach_lora(2, 4.0, false).expect("attach");
        let trainable: Vec<String> = layer
            .trainable_parameters()
            .keys()
            .map(|k| k.to_string())
            .collect();
        assert_eq!(trainable.len(), 2, "expected exactly lora_a and lora_b");
        assert!(trainable.iter().all(|k| k.starts_with("lora_")));
    }

    /// rsLoRA divides by `sqrt(rank)` rather than `rank`, so the update's norm
    /// stops depending on rank (Kalajdzievski 2023).
    #[test]
    fn rslora_scales_by_the_square_root_of_rank() {
        let plain = LoraAdapter::new(8, 8, 16, 32.0, false).expect("plain");
        let rs = LoraAdapter::new(8, 8, 16, 32.0, true).expect("rslora");
        assert_eq!(plain.scale, 32.0 / 16.0);
        assert_eq!(rs.scale, 32.0 / 4.0);
    }

    /// `A` starts uniform on `±1/√in_features`, which is what PEFT's
    /// `kaiming_uniform_(a=√5)` comes to, and it does
    /// not depend on the rank: rsLoRA changes the scale and nothing else
    /// (Kalajdzievski 2023 §3 takes `A`'s variance independent of `r`).
    #[test]
    fn a_starts_uniform_on_one_over_root_in_features() {
        for (in_features, rank, rslora) in [(4096, 16, false), (4096, 16, true), (512, 64, true)] {
            let adapter = LoraAdapter::new(in_features, 8, rank, 32.0, rslora).expect("adapter");
            let a = values(adapter.a.clone());
            let bound = 1.0 / (in_features as f32).sqrt();
            let max = a.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            assert!(
                max <= bound,
                "in={in_features} rank={rank} rslora={rslora}: |A| reaches {max}, bound {bound}"
            );
            // Uniform on ±bound has variance bound²/3.
            let var = a.iter().map(|v| v * v).sum::<f32>() / a.len() as f32;
            let want = bound * bound / 3.0;
            assert!(
                (var / want - 1.0).abs() < 0.05,
                "in={in_features} rank={rank} rslora={rslora}: variance {var}, want {want}"
            );
        }
    }

    /// DoRA seeds its magnitude from the base weight's column norms, so the
    /// layer starts out computing what it did before.
    #[test]
    fn a_fresh_dora_adapter_changes_almost_nothing() {
        let mut layer = Linear::new(3, 4, false).expect("layer");
        let before = values(layer.forward(&probe()));

        let weight = layer.weight.value.clone();
        let adapter = LoraAdapter::new(3, 4, 2, 4.0, false)
            .expect("adapter")
            .with_dora(&weight);
        layer.adapter = Some(Box::new(adapter));
        let after = values(layer.forward(&probe()));

        for (b, a) in before.iter().zip(&after) {
            assert!(
                (b - a).abs() < 1e-3,
                "DoRA init moved the output from {b} to {a}"
            );
        }
    }

    /// A DoRA layer with every tensor away from its initial value, so no
    /// term of the update is accidentally zero: `[out=5, in=6]`, rank 3.
    fn dora_parts() -> (Array, Array, Array, Array, Array, f32) {
        let weight = random::uniform_range(-1.0, 1.0, &[5, 6], Dtype::Float32);
        let a = random::uniform_range(-0.5, 0.5, &[3, 6], Dtype::Float32);
        let b = random::uniform_range(-0.5, 0.5, &[5, 3], Dtype::Float32);
        let magnitude = random::uniform_range(0.5, 2.0, &[5, 1], Dtype::Float32);
        let x = random::uniform_range(-1.0, 1.0, &[4, 6], Dtype::Float32);
        (weight, a, b, magnitude, x, 2.0)
    }

    /// DoRA as Liu et al. write it, spelled out densely:
    /// `y = (x·Wᵀ + s·x·Aᵀ·Bᵀ) · m/‖W + s·B·A‖`, the norm per output row and
    /// held constant for the gradient (§4.3, and what PEFT does).
    fn dora_reference(
        weight: &Array,
        a: &Array,
        b: &Array,
        magnitude: &Array,
        x: &Array,
        scale: f32,
    ) -> Array {
        let combined = weight.add(&b.matmul(a).mul_scalar(scale));
        let norm = combined
            .square()
            .sum_axis(1, false)
            .sqrt()
            .add_scalar(1e-6)
            .stop_gradient();
        let y = x
            .matmul(&weight.t())
            .add(&x.matmul(&a.t()).matmul(&b.t()).mul_scalar(scale));
        y.multiply(&magnitude.squeeze_axes(&[-1]).divide(&norm))
    }

    fn dora_layer(weight: &Array, a: &Array, b: &Array, magnitude: &Array, scale: f32) -> Linear {
        let mut layer = Linear::new(6, 5, false).expect("layer");
        layer.weight.value = weight.clone();
        let adapter = layer.attach_lora(3, scale * 3.0, false).expect("attach");
        adapter.a = a.clone();
        adapter.b = b.clone();
        adapter.magnitude = Some(magnitude.clone());
        assert_eq!(adapter.scale, scale);
        layer
    }

    #[test]
    fn dora_computes_the_definition() {
        let (weight, a, b, magnitude, x, scale) = dora_parts();
        let layer = dora_layer(&weight, &a, &b, &magnitude, scale);
        let want = values(dora_reference(&weight, &a, &b, &magnitude, &x, scale));
        let got = values(layer.forward(&x));
        crate::check_unobserved_error().expect("no bridge error");
        for (w, g) in want.iter().zip(&got) {
            assert!((w - g).abs() < 1e-5, "DoRA output {g}, definition {w}");
        }
    }

    /// The norm in DoRA's denominator is a constant to autograd. Treating it
    /// as one is what the paper prescribes to keep the backward cheap and
    /// what PEFT does; differentiating through it too gives a different
    /// gradient to `A` and `B` and holds another dense `[out, in]` copy of
    /// the update for the backward.
    #[test]
    fn dora_does_not_differentiate_its_norm() {
        let (weight, a, b, magnitude, x, scale) = dora_parts();
        let target = random::uniform_range(-1.0, 1.0, &[4, 5], Dtype::Float32);
        let params = [a.clone(), b.clone(), magnitude.clone()];

        let (_, want) = crate::inline_array::value_and_grad(
            |p| {
                let y = dora_reference(&weight, &p[0], &p[1], &p[2], &p[3], scale);
                y.multiply(&p[4]).sum_all()
            },
            &params,
            &[x.clone(), target.clone()],
        );
        let (_, got) = crate::inline_array::value_and_grad(
            |p| {
                let layer = dora_layer(&weight, &p[0], &p[1], &p[2], scale);
                layer.forward(&p[3]).multiply(&p[4]).sum_all()
            },
            &params,
            &[x, target],
        );
        for (name, (w, g)) in ["A", "B", "magnitude"].iter().zip(want.iter().zip(&got)) {
            let (w, g) = (values(w.clone()), values(g.clone()));
            crate::check_unobserved_error().expect("no bridge error");
            for (w, g) in w.iter().zip(&g) {
                assert!(
                    (w - g).abs() < 1e-4 * (1.0 + w.abs()),
                    "d/d{name}: {g}, constant-norm reference {w}"
                );
            }
        }
    }

    /// Inverted dropout: each element survives with probability `1 - p`,
    /// scaled by `1/(1 - p)`, so the expectation is the input.
    #[test]
    fn dropout_keeps_a_fraction_and_scales_it_up() {
        random::seed(3);
        let p = 0.25;
        let x = Array::ones(&[200_000], Dtype::Float32.as_i32());
        let y = values(dropout(&x, p));
        let kept = y.iter().filter(|v| **v != 0.0).count() as f32 / y.len() as f32;
        assert!((kept - (1.0 - p)).abs() < 0.01, "kept {kept}");
        assert!(
            y.iter()
                .all(|v| *v == 0.0 || (*v - 1.0 / (1.0 - p)).abs() < 1e-6)
        );
        // Seeded draws repeat; successive draws differ.
        random::seed(3);
        let again = values(dropout(&x, p));
        assert_eq!(y, again, "a seed must fix the mask");
        let next = values(dropout(&x, p));
        assert_ne!(y, next, "successive masks must differ");
    }

    /// Folding an adapter into a bf16 weight leaves a bf16 weight, and
    /// unfolding it brings one back: the adapter's f32 arithmetic is rounded
    /// to the weight's width, not carried into it.
    #[test]
    fn merging_keeps_the_weights_dtype() {
        for dora in [false, true] {
            let mut layer = trained_layer(dora);
            layer.weight.value = layer.weight.value.as_dtype(Dtype::Bfloat16.as_i32());
            layer.merge_lora();
            assert_eq!(
                layer.weight.value.dtype(),
                Dtype::Bfloat16,
                "dora={dora} merge"
            );
            layer.unmerge_lora();
            assert_eq!(
                layer.weight.value.dtype(),
                Dtype::Bfloat16,
                "dora={dora} unmerge"
            );
            crate::check_unobserved_error().expect("no bridge error");
        }
    }

    /// An FP8 base weight is its bytes until unpacked: an adapter on one
    /// computes what it does on the unpacked weight, and merging it unpacks
    /// the weight first rather than adding to the bytes.
    #[test]
    fn an_adapter_on_an_fp8_weight_computes_on_the_unpacked_one() {
        let dense = random::uniform_range(-1.0, 1.0, &[4, 8], Dtype::Bfloat16);
        let x = random::uniform_range(-1.0, 1.0, &[2, 8], Dtype::Bfloat16);
        let fp8 = crate::compat::ops::to_fp8(&dense).expect("to_fp8");
        let unpacked = crate::compat::ops::from_fp8(&fp8, Dtype::Bfloat16).expect("from_fp8");

        let adapt = |weight: &Array| {
            let mut layer = Linear::new(8, 4, false).expect("layer");
            layer.weight.value = weight.clone();
            let adapter = layer.attach_lora(2, 4.0, false).expect("attach");
            adapter.a = Array::from_f32_slice(
                &[
                    0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, 1.0, -0.1, 0.2, 0.3, -0.4, 0.5,
                    0.6,
                ],
                &[2, 8],
            );
            adapter.b =
                Array::from_f32_slice(&[0.3, -0.4, 0.5, 0.6, -0.7, 0.8, 0.2, -0.1], &[4, 2]);
            layer
        };
        let (want, mut packed) = (adapt(&unpacked), adapt(&fp8));
        let want_y = values(want.forward(&x).as_dtype(Dtype::Float32.as_i32()));
        let got_y = values(packed.forward(&x).as_dtype(Dtype::Float32.as_i32()));
        crate::check_unobserved_error().expect("no bridge error");
        for (w, g) in want_y.iter().zip(&got_y) {
            assert!((w - g).abs() < 1e-5, "fp8 base {g}, unpacked {w}");
        }

        packed.merge_lora();
        assert_eq!(packed.weight.value.dtype(), Dtype::Bfloat16);
        let merged_y = values(packed.forward(&x).as_dtype(Dtype::Float32.as_i32()));
        crate::check_unobserved_error().expect("no bridge error");
        for (w, m) in want_y.iter().zip(&merged_y) {
            assert!((w - m).abs() < 0.05, "merged {m}, adapted {w}");
        }
    }

    #[test]
    fn rank_must_be_positive() {
        assert!(LoraAdapter::new(4, 4, 0, 1.0, false).is_err());
        assert!(LoraAdapter::new(4, 4, -1, 1.0, false).is_err());
    }

    /// Packing a layer's weight (8-bit affine, as `quantize` defaults to for
    /// a drafter) keeps its shape and, to the packing's rounding, its output.
    fn packed_pair() -> (Linear, Linear, Array) {
        let dense = Linear::new(128, 8, true).expect("layer");
        let mut packed = dense.clone();
        packed
            .quantize(crate::native_weight::QuantParams {
                group_size: 64,
                bits: 8,
                mode: crate::QuantizedMode::Affine,
            })
            .expect("quantize");
        let x = random::uniform_range(-1.0, 1.0, &[2, 128], Dtype::Float32);
        (dense, packed, x)
    }

    #[test]
    fn a_packed_weight_computes_the_dense_one() {
        let (dense, packed, x) = packed_pair();
        assert_eq!(packed.shape(), (8, 128));
        let (want, got) = (values(dense.forward(&x)), values(packed.forward(&x)));
        for (w, g) in want.iter().zip(&got) {
            assert!((w - g).abs() < 2e-2, "packed {g} vs dense {w}");
        }
        // Packed, the layer saves as mlx's QuantizedLinear does, and is frozen.
        let names: Vec<String> = packed.parameters().keys().map(|k| k.to_string()).collect();
        for name in ["weight", "bias", "scales", "biases"] {
            assert!(
                names.iter().any(|n| n == name),
                "{name} missing from {names:?}"
            );
        }
        assert!(packed.trainable_parameters().is_empty());
    }

    /// QLoRA: an adapter on a packed weight computes what it does on that
    /// weight unpacked, trains alone, and merging unpacks the layer.
    #[test]
    fn an_adapter_on_a_packed_weight_is_qlora() {
        let (_, mut packed, x) = packed_pair();
        {
            let adapter = packed.attach_lora(4, 8.0, false).expect("attach");
            adapter.b = random::uniform_range(-0.5, 0.5, &[8, 4], Dtype::Float32);
        }
        let mut unpacked = packed.clone();
        unpacked.dequantize();
        let (want, got) = (values(unpacked.forward(&x)), values(packed.forward(&x)));
        for (w, g) in want.iter().zip(&got) {
            assert!((w - g).abs() < 1e-4, "packed {g} vs unpacked {w}");
        }
        let trainable: Vec<String> = packed
            .trainable_parameters()
            .keys()
            .map(|k| k.to_string())
            .collect();
        assert_eq!(trainable.len(), 2, "{trainable:?}");

        packed.merge_lora();
        assert!(packed.quant.is_none());
        let merged = values(packed.forward(&x));
        for (w, m) in want.iter().zip(&merged) {
            assert!((w - m).abs() < 1e-4, "merged {m} vs adapted {w}");
        }
    }

    /// A bf16 layer packed in `mode` computes, in bf16, close to what it did
    /// dense. The floating-point modes' scales are `uint8`, and the layer
    /// used to cast its activations to them.
    #[test]
    fn floating_point_modes_compute_in_the_weights_dtype() {
        let mut dense = Linear::new(128, 16, false).expect("layer");
        dense.weight.value = dense.weight.value.as_dtype(Dtype::Bfloat16.as_i32());
        let x = random::uniform_range(-1.0, 1.0, &[2, 128], Dtype::Bfloat16);
        let want = values(dense.forward(&x).as_dtype(Dtype::Float32.as_i32()));
        let norm = want.iter().map(|v| v * v).sum::<f32>().sqrt();
        for (mode, group_size) in [
            (crate::QuantizedMode::Mxfp4, 32),
            (crate::QuantizedMode::Nvfp4, 16),
            (crate::QuantizedMode::Mxfp8, 32),
        ] {
            let mut packed = dense.clone();
            packed
                .quantize(crate::native_weight::QuantParams {
                    group_size,
                    bits: if mode == crate::QuantizedMode::Mxfp8 {
                        8
                    } else {
                        4
                    },
                    mode,
                })
                .expect("quantize");
            let y = packed.forward(&x);
            assert_eq!(y.dtype(), Dtype::Bfloat16, "{mode:?}");
            assert_eq!(packed.dense_weight().dtype(), Dtype::Bfloat16, "{mode:?}");
            let got = values(y.as_dtype(Dtype::Float32.as_i32()));
            crate::check_unobserved_error().expect("no bridge error");
            let err = want
                .iter()
                .zip(&got)
                .map(|(w, g)| (w - g) * (w - g))
                .sum::<f32>()
                .sqrt();
            assert!(err < 0.2 * norm, "{mode:?}: relative error {}", err / norm);
        }
    }

    /// NF4's table, `[-1, 1]`, from Dettmers et al. (2023), appendix E.
    const NF4: [f32; 16] = [
        -1.0,
        -0.696_192_8,
        -0.525_073_05,
        -0.394_917_5,
        -0.284_441_38,
        -0.184_773_43,
        -0.091_050_036,
        0.0,
        0.079_580_3,
        0.160_930_2,
        0.246_112_3,
        0.337_915_24,
        0.440_709_83,
        0.562_617,
        0.722_956_84,
        1.0,
    ];

    /// Every unpacked weight is the nearest table value times its group's
    /// absmax, and the group's largest magnitude comes back exactly.
    #[test]
    fn a_codebook_weight_rounds_each_value_to_the_nearest_entry() {
        let mut layer = Linear::new(128, 4, false).expect("layer");
        let dense = values(layer.weight.value.clone());
        layer.quantize_codebook(&NF4, 64, false).expect("pack");
        assert_eq!(layer.shape(), (4, 128));
        let back = values(layer.dense_weight());
        crate::check_unobserved_error().expect("no bridge error");
        for (group, (orig, got)) in dense.chunks(64).zip(back.chunks(64)).enumerate() {
            let absmax = orig.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            // bf16 absmax: 2^-8 relative.
            let tol = absmax / 256.0;
            for (o, g) in orig.iter().zip(got) {
                let nearest = NF4
                    .iter()
                    .map(|c| c * absmax)
                    .min_by(|a, b| (a - o).abs().total_cmp(&(b - o).abs()))
                    .unwrap();
                assert!(
                    (g - nearest).abs() <= tol,
                    "group {group}: {o} came back as {g}, nearest entry is {nearest}"
                );
            }
        }
    }

    /// QLoRA on a codebook weight: the adapter computes what it does on the
    /// unpacked weight, trains alone, and double quantization of the absmax
    /// barely moves the result.
    #[test]
    fn an_adapter_on_a_codebook_weight_is_qlora() {
        // 32 rows of two groups: 64 absmax values, one run to pack.
        let base = Linear::new(128, 32, true).expect("layer");
        let x = random::uniform_range(-1.0, 1.0, &[2, 128], Dtype::Float32);
        for double_quant in [false, true] {
            let mut packed = base.clone();
            packed
                .quantize_codebook(&NF4, 64, double_quant)
                .expect("pack");
            {
                let adapter = packed.attach_lora(4, 8.0, false).expect("attach");
                adapter.b = random::uniform_range(-0.5, 0.5, &[32, 4], Dtype::Float32);
            }
            let mut unpacked = packed.clone();
            unpacked.dequantize();
            let (want, got) = (values(unpacked.forward(&x)), values(packed.forward(&x)));
            for (w, g) in want.iter().zip(&got) {
                assert!((w - g).abs() < 1e-4, "packed {g} vs unpacked {w}");
            }
            assert_eq!(packed.trainable_parameters().len(), 2);
            let names: Vec<String> = packed.parameters().keys().map(|k| k.to_string()).collect();
            assert!(names.iter().any(|n| n == "codebook"), "{names:?}");
            assert_eq!(
                names.iter().any(|n| n == "absmax_scales"),
                double_quant,
                "{names:?}"
            );
        }
        let dense_y = values(base.forward(&x));
        let mut single = base.clone();
        single.quantize_codebook(&NF4, 64, false).expect("pack");
        let mut double = base.clone();
        double.quantize_codebook(&NF4, 64, true).expect("pack");
        let err = |l: &Linear| {
            values(l.forward(&x))
                .iter()
                .zip(&dense_y)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        };
        let (e1, e2) = (err(&single), err(&double));
        crate::check_unobserved_error().expect("no bridge error");
        assert!(e2 < e1 * 1.05 + 1e-6, "double quant error {e2} vs {e1}");
    }
}
