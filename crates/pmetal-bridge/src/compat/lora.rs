//! Low-rank adapters, as a property of a linear layer rather than a layer of
//! their own.
//!
//! LoRA reparameterises a frozen linear map: `y = x·Wᵀ + b` becomes
//! `y = x·Wᵀ + b + s·((x·Aᵀ)·Bᵀ)`. That is the same shape as the optional bias
//! `Linear` already carries, so [`LoraAdapter`] hangs off [`Linear`] the way a
//! bias does, and an architecture never has to mention LoRA to be trainable
//! with it.
//!
//! This is the Rust reading of what PEFT and mlx-lm do at runtime. Both walk a
//! built model and swap each targeted `Linear` for an adapted one
//! (`LoRALinear.from_base`, `inject_adapter_in_model`); the base architecture
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
    /// `use_rslora` switches the scale to `alpha / sqrt(rank)` and widens the
    /// `A` initialisation to `sqrt(1 / rank)`, per Kalajdzievski (2023), so the
    /// update's norm stops depending on rank.
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
        let a_bound = if use_rslora {
            (1.0_f32 / rank as f32).sqrt()
        } else {
            (3.0_f32 / in_features as f32).sqrt()
        };
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
        self.magnitude = Some(column_norms(base_weight));
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
    pub fn merged_weight(&self, weight: &Array) -> Array {
        let combined = weight.add(&self.delta());
        match &self.magnitude {
            Some(magnitude) => {
                let norm = column_norms(&combined).add(&Array::from_f32(1e-6));
                combined.multiply(&magnitude.divide(&norm))
            }
            None => combined,
        }
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
            None => weight.subtract(&self.delta()),
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
    pub(crate) fn apply(&self, x: &Array, weight: &Array, bias: Option<&Array>) -> Array {
        let y = x.matmul(&weight.t());
        if self.merged {
            return add_bias(y, bias);
        }

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

        // DoRA rescales by `m / ‖W + s·B·A‖_col` — the norm of the *combined*
        // weight, as Liu et al. define it and as PEFT computes it. Scaling the
        // output per row is the same thing as scaling the weight's rows, which
        // is what makes `merged_weight` agree with this exactly.
        //
        // It does mean materialising `B·A` on the forward, which is the cost
        // DoRA carries over LoRA. Normalising by `‖W‖` instead would avoid that
        // and is a good approximation near initialisation, but it would make a
        // merged model compute something a live one does not.
        let y = match &self.magnitude {
            Some(magnitude) => {
                let combined = weight.add(&self.delta());
                let norm = column_norms(&combined).add(&Array::from_f32(1e-6));
                y.multiply(&magnitude.divide(&norm).squeeze_axes(&[-1]))
            }
            None => y,
        };
        add_bias(y, bias)
    }
}

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
/// the expectation is unchanged and inference needs no rescaling.
fn dropout(x: &Array, p: f32) -> Array {
    let keep = 1.0 - p;
    let mask = random::bernoulli(&Array::from_f32(keep), x.shape());
    x.multiply(&mask.multiply(&Array::from_f32(1.0 / keep)))
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

    #[test]
    fn rank_must_be_positive() {
        assert!(LoraAdapter::new(4, 4, 0, 1.0, false).is_err());
        assert!(LoraAdapter::new(4, 4, -1, 1.0, false).is_err());
    }
}
