//! The pieces of a mixture-of-experts block that every architecture shares.
//!
//! Each architecture routes tokens itself (top-k selection lives in
//! `pmetal_models::moe_routing`) and runs its experts through stacked or
//! gathered matmuls that keep the routing weights in the graph. What is shared
//! is the expert itself, a bias-free SwiGLU MLP whose weights a checkpoint
//! fills ([`Expert`]), and the step that sums the routed experts' outputs with
//! the gated shared expert ([`moe_combine`]).

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::compat::{
    Array, Dtype, Exception, LinearQuant, ModuleParamMut, ModuleParamRef, ModuleParameters,
    NestedValue, ops, random,
};
use crate::impl_module_params;
use crate::native_weight::LayerWeight;

/// A low-rank adapter on a stacked expert projection, one per expert.
///
/// LoRA's `y = x·Wᵀ + s·(x·Aᵀ)·Bᵀ` for each expert `e` separately:
/// `A` is `[experts, rank, in]` and `B` `[experts, out, rank]`, so the update
/// is `s·B_e·A_e` for the expert a token was routed to and nothing for the
/// others. With `top_k` experts active per token a per-expert rank of
/// `total_rank / top_k` keeps the rank a token sees at the total, which is how
/// the adapters are sized for a mixture of experts (Schulman et al., 2025,
/// "LoRA Without Regret").
///
/// `B` starts at zero, so a fresh adapter changes nothing. Both factors are
/// f32 whatever the experts' dtype, as the dense adapter's are, and the layer's
/// output keeps the experts' dtype.
#[derive(Debug, Clone)]
pub struct ExpertLora {
    /// Down-projections, `[experts, rank, in]`.
    pub a: Array,
    /// Up-projections, `[experts, out, rank]`. Zero at initialisation.
    pub b: Array,
    /// Rank of each expert's adapter.
    pub rank: i32,
    /// `alpha / rank`, or `alpha / sqrt(rank)` under rsLoRA.
    pub scale: f32,
    /// Set once folded into the experts' weight, after which it contributes
    /// nothing further.
    pub merged: bool,
}

impl ExpertLora {
    /// An adapter for `experts` projections of shape `[out, in]`, like
    /// [`LoraAdapter::new`](crate::compat::LoraAdapter::new) for each.
    pub fn new(
        experts: i32,
        in_features: i32,
        out_features: i32,
        rank: i32,
        alpha: f32,
        use_rslora: bool,
    ) -> Result<Self, Exception> {
        if rank <= 0 {
            return Err(Exception::custom(format!(
                "expert LoRA rank must be positive, got {rank}"
            )));
        }
        let scale = if use_rslora {
            alpha / (rank as f32).sqrt()
        } else {
            alpha / rank as f32
        };
        let bound = 1.0_f32 / (in_features as f32).sqrt();
        Ok(Self {
            a: random::uniform_range(-bound, bound, &[experts, rank, in_features], Dtype::Float32),
            b: ops::zeros(&[experts, out_features, rank], Dtype::Float32),
            rank,
            scale,
            merged: false,
        })
    }

    /// `s·B_e·A_e` for every expert, `[experts, out, in]`: the dense update.
    pub fn delta(&self) -> Array {
        self.b.matmul(&self.a).mul_scalar(self.scale)
    }

    /// The adapter's contribution to the rows `x` routed to experts `ids`:
    /// `s·(x·A_eᵀ)·B_eᵀ` with each row through its own expert, computed as two
    /// gather matmuls and never as a dense update. `x` and `ids` are laid out
    /// as for the base projection's gather matmul (`sorted` is the same flag),
    /// and the result has the same shape as its output, in f32.
    pub fn contribution(&self, x: &Array, ids: &Array, sorted: bool) -> Array {
        let down = self.a.swap_axes(-1, -2);
        let up = self.b.swap_axes(-1, -2);
        x.gather_mm(&down, None, Some(ids), sorted)
            .gather_mm(&up, None, Some(ids), sorted)
            .mul_scalar(self.scale)
    }

    /// Add the adapter's contribution to `base`, the projection's own output
    /// for the same rows, keeping `base`'s dtype.
    pub fn apply(&self, base: Array, x: &Array, ids: &Array, sorted: bool) -> Array {
        if self.merged {
            return base;
        }
        let dtype = base.dtype_raw();
        let y = base.add(&self.contribution(x, ids, sorted));
        if y.dtype_raw() == dtype {
            y
        } else {
            y.as_dtype(dtype)
        }
    }

    /// Number of trainable elements.
    pub fn num_trainable_params(&self) -> usize {
        let count = |arr: &Array| arr.shape().iter().map(|d| *d as usize).product::<usize>();
        count(&self.a) + count(&self.b)
    }
}

/// A stacked projection `[experts, out, in]`, or (for a single [`Expert`]) one
/// `[out, in]`, with no bias. Zero-initialised; a checkpoint fills it.
///
/// A stacked one can carry an [`ExpertLora`]. The adapter is applied by the
/// gather matmuls that read the weight ([`crate::native_moe`]), which take it
/// beside the weight; [`forward`](Self::forward) is the single expert's.
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix, shape `[out_features, in_features]`, or
    /// `[experts, out_features, in_features]` stacked. MLX's packed words
    /// (`[experts, out_features, in_features * bits / 32]`) when
    /// [`quant`](Self::quant) is set.
    pub weight: Array,
    /// Per-expert low-rank adapters, when attached to a stacked projection.
    pub adapter: Option<ExpertLora>,
    /// Set when a stacked `weight` is kept packed for MLX's quantized gather
    /// matmul, as a [`compat::Linear`](crate::compat::Linear)'s is for its
    /// quantized matmul: a decode step reads only the experts a token routes
    /// to, a quarter of the bytes at 4 bits. An adapter on it is QLoRA on the
    /// experts.
    pub quant: Option<Box<LinearQuant>>,
}

impl Linear {
    /// Create a zero-initialised linear layer.
    pub fn new(in_features: i32, out_features: i32) -> Self {
        Self {
            weight: ops::zeros(&[out_features, in_features], Dtype::Float32),
            adapter: None,
            quant: None,
        }
    }

    /// A stacked projection holding `weight`, no adapter.
    pub fn stacked(weight: Array) -> Self {
        Self {
            weight,
            adapter: None,
            quant: None,
        }
    }

    /// `y = x @ W^T`
    pub fn forward(&self, x: &Array) -> Array {
        x.matmul(&self.weight.t())
    }

    /// `(experts, out_features, in_features)` of a stacked projection, whether
    /// its weight is packed or not.
    pub fn stack_shape(&self) -> Option<(i32, i32, i32)> {
        match (self.weight.shape(), &self.quant) {
            ([e, o, i], Some(quant)) => Some((*e, *o, *i * 32 / quant.params.bits)),
            ([e, o, i], None) => Some((*e, *o, *i)),
            _ => None,
        }
    }

    /// The projection as the gather matmuls read it when it is packed: the
    /// words as stored, `[experts, out, in * bits / 32]`, with their scales.
    /// `None` for a dense weight, which the caller transposes for the dense
    /// kernel instead.
    pub fn packed_weight(&self) -> Option<LayerWeight> {
        let quant = self.quant.as_deref()?;
        Some(LayerWeight::Quantized {
            weight: self.weight.clone(),
            scales: quant.scales.clone(),
            biases: quant.biases.clone(),
            tensor_scale: quant.tensor_scale.clone(),
            params: quant.params,
        })
    }

    /// The projection as a dense `[experts, out, in]` array, unpacked if it is
    /// packed.
    pub fn dense_weight(&self) -> Array {
        let Some(quant) = self.quant.as_deref() else {
            return self.weight.clone();
        };
        let dense = self.weight.dequantize_mode(
            &quant.scales,
            quant.biases.as_ref(),
            quant.params.group_size,
            quant.params.bits,
            quant.params.mode,
        );
        let dense = match &quant.tensor_scale {
            Some(scale) => dense
                .as_dtype(Dtype::Float32.as_i32())
                .multiply(&scale.reshape(&[-1, 1, 1])),
            None => dense,
        };
        dense.as_dtype(quant.dtype.as_i32())
    }

    /// Attach a per-expert adapter of the given rank, replacing any existing
    /// one. Only a stacked projection can take one.
    pub fn attach_lora(
        &mut self,
        rank: i32,
        alpha: f32,
        use_rslora: bool,
    ) -> Result<&mut ExpertLora, Exception> {
        let (experts, out_features, in_features) = self.stack_shape().ok_or_else(|| {
            Exception::custom("an expert adapter attaches to a stacked [experts, out, in] weight")
        })?;
        let adapter = ExpertLora::new(experts, in_features, out_features, rank, alpha, use_rslora)?;
        Ok(self.adapter.insert(adapter))
    }

    /// Whether an adapter is attached.
    pub fn is_adapted(&self) -> bool {
        self.adapter.is_some()
    }

    /// Fold the adapter into the weight, keeping it attached (reversible with
    /// [`unmerge_lora`](Self::unmerge_lora)). The weight must be floating
    /// point, not packed or FP8 bytes: those are dequantized by the caller
    /// first.
    pub fn merge_lora(&mut self) -> Result<(), Exception> {
        let Some(adapter) = self.adapter.as_mut().filter(|a| !a.merged) else {
            return Ok(());
        };
        if self.quant.is_some() {
            return Err(Exception::custom(
                "an expert adapter folds into a floating-point weight; this one is packed",
            ));
        }
        if !matches!(
            self.weight.dtype(),
            Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
        ) {
            return Err(Exception::custom(
                "an expert adapter folds into a floating-point weight; this one is packed",
            ));
        }
        let dtype = self.weight.dtype_raw();
        self.weight = self.weight.add(&adapter.delta()).as_dtype(dtype);
        adapter.merged = true;
        Ok(())
    }

    /// Undo [`merge_lora`](Self::merge_lora).
    pub fn unmerge_lora(&mut self) {
        let Some(adapter) = self.adapter.as_mut().filter(|a| a.merged) else {
            return;
        };
        let dtype = self.weight.dtype_raw();
        self.weight = self.weight.subtract(&adapter.delta()).as_dtype(dtype);
        adapter.merged = false;
    }

    /// Fold the adapter in and drop it.
    pub fn fuse_lora(&mut self) -> Result<(), Exception> {
        self.merge_lora()?;
        self.adapter = None;
        Ok(())
    }
}

// Hand-written, as `compat::Linear`'s is, so the adapter flattens beside the
// weight (`gate_proj.lora_a`) and an adapted projection trains its adapter
// and nothing else.
impl ModuleParameters for Linear {
    fn num_parameters(&self) -> usize {
        1 + self.quant.as_ref().map_or(0, |q| {
            1 + usize::from(q.biases.is_some()) + usize::from(q.tensor_scale.is_some())
        }) + self.adapter.as_ref().map_or(0, |_| 2)
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut out = BTreeMap::new();
        out.insert(Rc::from("weight"), NestedValue::Value(&self.weight));
        if let Some(quant) = &self.quant {
            out.insert(Rc::from("scales"), NestedValue::Value(&quant.scales));
            if let Some(biases) = &quant.biases {
                out.insert(Rc::from("biases"), NestedValue::Value(biases));
            }
            if let Some(scale) = &quant.tensor_scale {
                out.insert(Rc::from("tensor_scale"), NestedValue::Value(scale));
            }
        }
        if let Some(adapter) = &self.adapter {
            out.insert(Rc::from("lora_a"), NestedValue::Value(&adapter.a));
            out.insert(Rc::from("lora_b"), NestedValue::Value(&adapter.b));
        }
        out
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut out = BTreeMap::new();
        out.insert(Rc::from("weight"), NestedValue::Value(&mut self.weight));
        if let Some(quant) = &mut self.quant {
            out.insert(Rc::from("scales"), NestedValue::Value(&mut quant.scales));
            if let Some(biases) = &mut quant.biases {
                out.insert(Rc::from("biases"), NestedValue::Value(biases));
            }
            if let Some(scale) = &mut quant.tensor_scale {
                out.insert(Rc::from("tensor_scale"), NestedValue::Value(scale));
            }
        }
        if let Some(adapter) = &mut self.adapter {
            out.insert(Rc::from("lora_a"), NestedValue::Value(&mut adapter.a));
            out.insert(Rc::from("lora_b"), NestedValue::Value(&mut adapter.b));
        }
        out
    }

    /// As for a [`compat::Linear`](crate::compat::Linear): an adapted
    /// projection trains its adapter alone, and a packed weight with none trains
    /// nothing.
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        let Some(adapter) = &self.adapter else {
            if self.quant.is_some() {
                return BTreeMap::new();
            }
            return self.parameters();
        };
        let mut out = BTreeMap::new();
        out.insert(Rc::from("lora_a"), NestedValue::Value(&adapter.a));
        out.insert(Rc::from("lora_b"), NestedValue::Value(&adapter.b));
        out
    }
}

// The walk for `Linear`s stops here: a stacked projection is not a
// `compat::Linear`. Adapters on it are reached by `visit_experts_mut`.
impl crate::compat::VisitLinears for Linear {
    fn visit_linears_mut(
        &mut self,
        _prefix: &str,
        _f: &mut dyn FnMut(&str, &mut crate::compat::Linear),
    ) {
    }

    fn visit_experts_mut(
        &mut self,
        prefix: &str,
        top_k: usize,
        f: &mut dyn FnMut(&str, usize, &mut Linear),
    ) {
        if self.stack_shape().is_some() {
            f(prefix, top_k, self);
        }
    }
}

/// Single expert MLP (SwiGLU).
#[derive(Debug)]
pub struct Expert {
    /// Gate projection, `[intermediate, hidden]`.
    pub w1: Linear,
    /// Up projection, `[intermediate, hidden]`.
    pub w3: Linear,
    /// Down projection, `[hidden, intermediate]`.
    pub w2: Linear,
}
impl_module_params!(Expert; w1, w3, w2);

impl Expert {
    /// Create a zero-initialised expert; a checkpoint fills the weights.
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Self {
        Self {
            w1: Linear::new(hidden_size, intermediate_size),
            w3: Linear::new(hidden_size, intermediate_size),
            w2: Linear::new(intermediate_size, hidden_size),
        }
    }

    /// `w2(silu(w1(x)) * w3(x))`.
    pub fn forward(&self, x: &Array) -> Array {
        let gate = self.w1.forward(x);
        // SwiGLU: silu(gate) * up
        let gate_activated = gate.silu();
        let up = self.w3.forward(x);
        let hidden = gate_activated.multiply(&up);
        self.w2.forward(&hidden)
    }
}

/// MoE combine: weighted expert sum + sigmoid-gated shared expert.
///
/// Computes:
/// ```text
/// y = (expert_outs * weights.unsqueeze(-1)).sum(-2)  // weighted sum
/// shared_gate = sigmoid(shared_gate_logit)
/// y += shared_gate * shared_out
/// ```
///
/// This is the MoE block's output alone. The decoder layer adds its own
/// residual, the same as it does for a dense MLP; adding the block's input here
/// as well counted the normalized hidden state twice.
///
/// All ops run asynchronously on GPU via MLX's lazy evaluation graph.
///
/// # Arguments
/// * `expert_outs` - Expert outputs `[batch_seq, K, D]`
/// * `expert_weights` - Routing weights `[batch_seq, K]`
/// * `shared_out` - Shared expert output `[batch_seq, D]`
/// * `shared_gate_logit` - Shared expert gate logit `[batch_seq, 1]` or scalar
/// * `k` - Number of active experts (top-k)
/// * `batch_seq` - Batch * sequence length
pub fn moe_combine(
    expert_outs: &Array,
    expert_weights: &Array,
    shared_out: &Array,
    shared_gate_logit: &Array,
    k: i32,
    batch_seq: i32,
) -> Result<Array, Exception> {
    let y = expert_outs
        .multiply(&expert_weights.reshape(&[batch_seq, k, 1]))
        .sum_axis(-2, false);

    // Shared expert with gate
    let shared_gate = shared_gate_logit.sigmoid();
    let shared_y = shared_gate.multiply(shared_out);

    Ok(y.add(&shared_y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f64_reference::{Rng, arr, assert_close, drain, round32, vec_of};

    /// `w2 (silu(w1 x) * w3 x)` in f64 for one token.
    fn expert_f64(x: &[f64], w1: &[f64], w3: &[f64], w2: &[f64], inter: usize) -> Vec<f64> {
        let hidden = x.len();
        let matvec = |w: &[f64], v: &[f64], rows: usize| -> Vec<f64> {
            (0..rows)
                .map(|r| (0..v.len()).map(|c| w[r * v.len() + c] * v[c]).sum())
                .collect()
        };
        let silu = |z: f64| z / (1.0 + (-z).exp());
        let gate = matvec(w1, x, inter);
        let up = matvec(w3, x, inter);
        let act: Vec<f64> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
        matvec(w2, &act, hidden)
    }

    #[test]
    fn an_expert_is_swiglu() {
        let (hidden, inter, tokens) = (6usize, 10usize, 3usize);
        let mut rng = Rng(11);
        let mut draw = |n: usize| (0..n).map(|_| round32(rng.normal())).collect::<Vec<_>>();
        let (w1, w3, w2) = (
            draw(inter * hidden),
            draw(inter * hidden),
            draw(hidden * inter),
        );
        let x = draw(tokens * hidden);

        let mut expert = Expert::new(hidden as i32, inter as i32);
        expert.w1.weight = arr(&w1, &[inter, hidden]);
        expert.w3.weight = arr(&w3, &[inter, hidden]);
        expert.w2.weight = arr(&w2, &[hidden, inter]);
        let got = vec_of(&expert.forward(&arr(&x, &[tokens, hidden])));
        drain();

        let want: Vec<f64> = (0..tokens)
            .flat_map(|t| expert_f64(&x[t * hidden..(t + 1) * hidden], &w1, &w3, &w2, inter))
            .collect();
        assert_close("expert", &got, &want, 1e-5);
    }

    #[test]
    fn a_new_expert_is_zero() {
        let expert = Expert::new(4, 8);
        let out = expert.forward(&Array::from_f32_slice(&[1.0; 8], &[2, 4]));
        assert_eq!(out.shape(), &[2, 4]);
        assert!(vec_of(&out).iter().all(|&v| v == 0.0));
    }

    /// The combine in f64: `sum_k w_k e_k + sigmoid(g) s` per token.
    fn reference(
        outs: &[f64],
        weights: &[f64],
        shared: &[f64],
        gate: &[f64],
        (tokens, k, dim): (usize, usize, usize),
    ) -> Vec<f64> {
        let mut y = vec![0.0; tokens * dim];
        for t in 0..tokens {
            let g = 1.0 / (1.0 + (-gate[t % gate.len()]).exp());
            for d in 0..dim {
                let routed: f64 = (0..k)
                    .map(|j| weights[t * k + j] * outs[(t * k + j) * dim + d])
                    .sum();
                y[t * dim + d] = routed + g * shared[t * dim + d];
            }
        }
        y
    }

    /// The block's output is the weighted expert sum plus the gated shared
    /// expert, for a gate per token and for one scalar gate, and carries no
    /// residual (the decoder layer adds its own: adding the block's input here
    /// as well counted the normalized hidden state twice).
    #[test]
    fn the_combine_is_the_weighted_sum_plus_the_gated_shared_expert() {
        let (tokens, k, dim) = (5usize, 3usize, 8usize);
        let mut rng = Rng(3);
        let mut draw = |n: usize| (0..n).map(|_| round32(rng.normal())).collect::<Vec<_>>();
        let outs = draw(tokens * k * dim);
        let weights = draw(tokens * k);
        let shared = draw(tokens * dim);
        for (what, gate, gate_shape) in [
            ("per-token gate", draw(tokens), vec![tokens, 1]),
            ("scalar gate", draw(1), vec![]),
        ] {
            let got = moe_combine(
                &arr(&outs, &[tokens, k, dim]),
                &arr(&weights, &[tokens, k]),
                &arr(&shared, &[tokens, dim]),
                &arr(&gate, &gate_shape),
                k as i32,
                tokens as i32,
            )
            .unwrap();
            drain();
            let want = reference(&outs, &weights, &shared, &gate, (tokens, k, dim));
            assert_close(what, &vec_of(&got), &want, 1e-5);
        }

        // Silent experts give nothing: no residual leaks in.
        let zeros = vec![0.0; tokens * dim];
        let got = moe_combine(
            &arr(&vec![0.0; tokens * k * dim], &[tokens, k, dim]),
            &arr(&weights, &[tokens, k]),
            &arr(&zeros, &[tokens, dim]),
            &arr(&vec![0.0; tokens], &[tokens, 1]),
            k as i32,
            tokens as i32,
        )
        .unwrap();
        assert!(vec_of(&got).iter().all(|&v| v == 0.0));
    }
}
