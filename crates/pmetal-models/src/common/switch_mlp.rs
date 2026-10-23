//! A layer's routed experts, stacked: the `switch_mlp` of an MLX checkpoint.
//!
//! DeepSeek and Qwen3-MoE route a token to its top-k experts and run each as a
//! SwiGLU MLP. The experts are held as three weights stacked over a leading
//! expert axis, `[experts, out, in]`, which is how an MLX conversion stores
//! them (`mlp.switch_mlp.{gate,up,down}_proj.weight`). A checkpoint that names
//! an expert per key, or fuses gate and up, is brought to this layout at load
//! by [`crate::loader::experts::stack_experts`].
//!
//! Stacked, not a `Linear` per expert, so a layer holds three arrays whatever
//! the expert count, the gather matmuls read them as they are (a transposed
//! view, no copy), and nothing is kept twice. Plain weights, not `Linear`s:
//! the experts are read through the gather kernels, so an adapter has to be
//! applied by them too. A stacked projection carries its own per-expert
//! low-rank adapter ([`pmetal_runtime::nn::moe::ExpertLora`]), and the gather
//! kernels add it beside the weight.

use pmetal_runtime::activation_stats::{observe_experts_per_slot, observe_experts_per_token};
use pmetal_runtime::compat::{
    Array, Dtype, Exception, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, nn,
};
use pmetal_runtime::native_moe::{self, ExpertAdapters};
use pmetal_runtime::native_weight::LayerWeight;
use pmetal_runtime::nn::moe::Linear as MoeLinear;

use crate::fp8_utils::dequantize_fp8_weight_for_compute;

/// A randomly initialised stack of `experts` projections `[out, in]`
/// (lazily: a checkpoint's weights replace the draw before anything computes
/// it).
fn init_stack(experts: i32, in_features: i32, out_features: i32) -> MoeLinear {
    let bound = (in_features as f32).sqrt().recip();
    MoeLinear::stacked(pmetal_runtime::compat::random::uniform_range(
        -bound,
        bound,
        &[experts, out_features, in_features],
        Dtype::Float32,
    ))
}

/// The projection as the gather matmuls take it: dense, `[experts, in, out]`,
/// or the packed words as stored for the quantized gather.
fn gather_weight(linear: &MoeLinear) -> Result<LayerWeight, Exception> {
    if let Some(packed) = linear.packed_weight() {
        return Ok(packed);
    }
    Ok(LayerWeight::Dense(
        dequantize_fp8_weight_for_compute(&linear.weight)?.swap_axes(-1, -2),
    ))
}

/// The named stacked projections of a block, as a parameter tree: each one's
/// weight (and adapter) under its own name.
fn named_parameters<'a>(
    projections: &[(&'static str, &'a MoeLinear)],
    trainable: bool,
) -> ModuleParamRef<'a> {
    let mut out = ModuleParamRef::new();
    for (name, linear) in projections {
        let tree = if trainable {
            linear.trainable_parameters()
        } else {
            linear.parameters()
        };
        if !tree.is_empty() {
            out.insert((*name).into(), NestedValue::Map(tree));
        }
    }
    out
}

fn named_parameters_mut<'a>(
    projections: Vec<(&'static str, &'a mut MoeLinear)>,
) -> ModuleParamMut<'a> {
    let mut out = ModuleParamMut::new();
    for (name, linear) in projections {
        out.insert(name.into(), NestedValue::Map(linear.parameters_mut()));
    }
    out
}

/// The routed experts of one MoE layer, each projection a weight
/// `[experts, out, in]`.
#[derive(Debug)]
pub struct SwitchMlp {
    /// `[experts, moe_intermediate, hidden]`.
    pub gate_proj: MoeLinear,
    /// `[experts, moe_intermediate, hidden]`.
    pub up_proj: MoeLinear,
    /// `[experts, hidden, moe_intermediate]`.
    pub down_proj: MoeLinear,
    /// How many experts a token is routed to: what sizes a per-expert adapter
    /// (the rank a token sees is the per-expert rank times this).
    pub top_k: usize,
}

impl ModuleParameters for SwitchMlp {
    fn num_parameters(&self) -> usize {
        self.gate_proj.num_parameters()
            + self.up_proj.num_parameters()
            + self.down_proj.num_parameters()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        named_parameters(
            &[
                ("gate_proj", &self.gate_proj),
                ("up_proj", &self.up_proj),
                ("down_proj", &self.down_proj),
            ],
            false,
        )
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        named_parameters(
            &[
                ("gate_proj", &self.gate_proj),
                ("up_proj", &self.up_proj),
                ("down_proj", &self.down_proj),
            ],
            true,
        )
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        named_parameters_mut(vec![
            ("gate_proj", &mut self.gate_proj),
            ("up_proj", &mut self.up_proj),
            ("down_proj", &mut self.down_proj),
        ])
    }
}

impl pmetal_runtime::compat::VisitLinears for SwitchMlp {
    fn visit_linears_mut(&mut self, _prefix: &str, _f: &mut dyn FnMut(&str, &mut nn::Linear)) {}

    fn visit_experts_mut(
        &mut self,
        prefix: &str,
        _top_k: usize,
        f: &mut dyn FnMut(&str, usize, &mut MoeLinear),
    ) {
        use pmetal_runtime::compat::child_path;
        let top_k = self.top_k;
        f(&child_path(prefix, "gate_proj"), top_k, &mut self.gate_proj);
        f(&child_path(prefix, "up_proj"), top_k, &mut self.up_proj);
        f(&child_path(prefix, "down_proj"), top_k, &mut self.down_proj);
    }
}

impl SwitchMlp {
    /// Randomly initialised experts (lazily: a checkpoint's weights replace
    /// the draw before anything computes it), `top_k` of which a token is
    /// routed to.
    pub fn new(num_experts: i32, hidden_size: i32, intermediate_size: i32, top_k: usize) -> Self {
        Self {
            gate_proj: init_stack(num_experts, hidden_size, intermediate_size),
            up_proj: init_stack(num_experts, hidden_size, intermediate_size),
            down_proj: init_stack(num_experts, intermediate_size, hidden_size),
            top_k,
        }
    }

    /// How many experts the layer holds.
    pub fn num_experts(&self) -> i32 {
        self.gate_proj.weight.dim(0)
    }

    /// `Σ_k scores[s, k] · down(silu(gate(x_s)) · up(x_s))` over each row's
    /// experts `inds[s, k]`: `x_flat` is `[S, hidden]`, `inds` and `scores`
    /// `[S, top_k]` (`scores` in the dtype of `x_flat`), the result `[S, hidden]`.
    ///
    /// Through grouped gather matmuls, the shared
    /// [`native_moe::switch_glu`] the native engines run too, with each
    /// projection's per-expert adapter added where one is attached.
    pub fn forward(
        &self,
        x_flat: &Array,
        inds: &Array,
        scores: &Array,
    ) -> Result<Array, Exception> {
        let adapters = ExpertAdapters {
            gate: self.gate_proj.adapter.as_ref(),
            up: self.up_proj.adapter.as_ref(),
            down: self.down_proj.adapter.as_ref(),
        };
        // Calibration: gate and up read the layer's input; down reads what
        // each (token, expert) pair's activation made.
        observe_experts_per_token(&self.gate_proj.weight, x_flat, inds);
        observe_experts_per_token(&self.up_proj.weight, x_flat, inds);
        Ok(native_moe::switch_glu_adapted(
            x_flat,
            &gather_weight(&self.gate_proj)?,
            &gather_weight(&self.up_proj)?,
            &gather_weight(&self.down_proj)?,
            adapters,
            inds,
            scores,
            |gate, up, expert_ids| {
                let activated = nn::silu(gate).multiply(up);
                observe_experts_per_slot(&self.down_proj.weight, &activated, expert_ids);
                activated
            },
        ))
    }
}

/// The routed experts of a layer whose experts have no gate: an up projection,
/// an activation and a down projection, as Nemotron-H's squared-ReLU experts
/// are. Stacked like [`SwitchMlp`], under the names an MLX conversion gives
/// them (`switch_mlp.fc1`, `switch_mlp.fc2`).
#[derive(Debug)]
pub struct PlainSwitchMlp {
    /// The up projection, `[experts, intermediate, in]`.
    pub fc1: MoeLinear,
    /// The down projection, `[experts, in, intermediate]`.
    pub fc2: MoeLinear,
    /// How many experts a token is routed to; see [`SwitchMlp::top_k`].
    pub top_k: usize,
}

impl ModuleParameters for PlainSwitchMlp {
    fn num_parameters(&self) -> usize {
        self.fc1.num_parameters() + self.fc2.num_parameters()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        named_parameters(&[("fc1", &self.fc1), ("fc2", &self.fc2)], false)
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        named_parameters(&[("fc1", &self.fc1), ("fc2", &self.fc2)], true)
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        named_parameters_mut(vec![("fc1", &mut self.fc1), ("fc2", &mut self.fc2)])
    }
}

impl pmetal_runtime::compat::VisitLinears for PlainSwitchMlp {
    fn visit_linears_mut(&mut self, _prefix: &str, _f: &mut dyn FnMut(&str, &mut nn::Linear)) {}

    fn visit_experts_mut(
        &mut self,
        prefix: &str,
        _top_k: usize,
        f: &mut dyn FnMut(&str, usize, &mut MoeLinear),
    ) {
        use pmetal_runtime::compat::child_path;
        let top_k = self.top_k;
        f(&child_path(prefix, "fc1"), top_k, &mut self.fc1);
        f(&child_path(prefix, "fc2"), top_k, &mut self.fc2);
    }
}

impl PlainSwitchMlp {
    /// Randomly initialised experts (lazily: a checkpoint's weights replace
    /// the draw before anything computes it).
    pub fn new(num_experts: i32, in_features: i32, intermediate_size: i32, top_k: usize) -> Self {
        Self {
            fc1: init_stack(num_experts, in_features, intermediate_size),
            fc2: init_stack(num_experts, intermediate_size, in_features),
            top_k,
        }
    }

    /// How many experts the layer holds.
    pub fn num_experts(&self) -> i32 {
        self.fc1.weight.dim(0)
    }

    /// `Σ_k scores[s, k] · fc2(act(fc1(x_s)))` over each row's experts
    /// `inds[s, k]`: `x_flat` is `[S, in]`, `inds` and `scores` `[S, top_k]`
    /// (`scores` in the dtype of `x_flat`), the result `[S, in]`.
    pub fn forward(
        &self,
        x_flat: &Array,
        inds: &Array,
        scores: &Array,
        activation: impl Fn(&Array) -> Array,
    ) -> Result<Array, Exception> {
        let adapters = ExpertAdapters {
            gate: self.fc1.adapter.as_ref(),
            up: None,
            down: self.fc2.adapter.as_ref(),
        };
        observe_experts_per_token(&self.fc1.weight, x_flat, inds);
        Ok(native_moe::switch_mlp_adapted(
            x_flat,
            &gather_weight(&self.fc1)?,
            &gather_weight(&self.fc2)?,
            adapters,
            inds,
            scores,
            |projected, expert_ids| {
                let activated = activation(projected);
                observe_experts_per_slot(&self.fc2.weight, &activated, expert_ids);
                activated
            },
        ))
    }
}

#[cfg(test)]
mod calibration_tests {
    use super::*;
    use pmetal_runtime::activation_stats::{ChannelStats, Collector};

    const EXPERTS: usize = 4;
    const HIDDEN: usize = 6;
    const INTER: usize = 8;

    fn host(a: &Array) -> Vec<f32> {
        let n = a.size();
        let mut a = a.as_dtype(Dtype::Float32.as_i32());
        a.to_f32_vec(n).expect("read array")
    }

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// `y[o] = sum_i w[e][o][i] * x[i]` for the stacked weight `w [E, out, in]`.
    fn project(w: &[f32], e: usize, out: usize, inp: usize, x: &[f32]) -> Vec<f32> {
        (0..out)
            .map(|o| {
                (0..inp)
                    .map(|i| w[(e * out + o) * inp + i] * x[i])
                    .sum::<f32>()
            })
            .collect()
    }

    /// Runs the layer over `tokens` rows routed as `routes` says and checks that
    /// the statistics are what the routed tokens give, worked out on the host
    /// from the layer's own weights.
    fn check(tokens: usize, routes: impl Fn(usize) -> [i32; 2]) {
        let mlp = SwitchMlp::new(EXPERTS as i32, HIDDEN as i32, INTER as i32, 2);
        let x: Vec<f32> = (0..tokens * HIDDEN)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) / 4.0)
            .collect();
        let ids: Vec<i32> = (0..tokens).flat_map(&routes).collect();
        let x_arr = Array::from_f32_slice(&x, &[tokens as i32, HIDDEN as i32]);
        let ids_arr = Array::from_slice(&ids, &[tokens as i32, 2]);
        let scores = Array::from_f32_slice(&vec![0.5; tokens * 2], &[tokens as i32, 2]);

        let collector = Collector::install([
            ("gate".to_string(), &mlp.gate_proj.weight),
            ("up".to_string(), &mlp.up_proj.weight),
            ("down".to_string(), &mlp.down_proj.weight),
        ])
        .unwrap();
        let y = mlp.forward(&x_arr, &ids_arr, &scores).expect("forward");
        assert_eq!(y.shape(), &[tokens as i32, HIDDEN as i32]);
        let stats: Vec<ChannelStats> = collector.finish();
        let by = |n: &str| {
            stats
                .iter()
                .find(|s| s.name == n)
                .unwrap_or_else(|| panic!("{n}"))
        };

        let (wg, wu) = (host(&mlp.gate_proj.weight), host(&mlp.up_proj.weight));
        let mut gate_sums = vec![0.0f32; EXPERTS * HIDDEN];
        let mut down_sums = vec![0.0f32; EXPERTS * INTER];
        let mut counts = vec![0.0f64; EXPERTS];
        for t in 0..tokens {
            let row = &x[t * HIDDEN..(t + 1) * HIDDEN];
            for e in routes(t) {
                let e = e as usize;
                counts[e] += 1.0;
                for j in 0..HIDDEN {
                    gate_sums[e * HIDDEN + j] += row[j] * row[j];
                }
                let gate = project(&wg, e, INTER, HIDDEN, row);
                let up = project(&wu, e, INTER, HIDDEN, row);
                for i in 0..INTER {
                    let h = silu(gate[i]) * up[i];
                    down_sums[e * INTER + i] += h * h;
                }
            }
        }

        for name in ["gate", "up"] {
            let s = by(name);
            assert_eq!(
                (s.experts, s.in_features),
                (Some(EXPERTS), HIDDEN),
                "{name}"
            );
            assert_eq!(s.counts, counts, "{name} counts");
            for (j, (g, w)) in s.sum_sq.iter().zip(&gate_sums).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-4 * w.abs().max(1e-2),
                    "{name}[{j}]: {g} vs {w}"
                );
            }
        }
        let s = by("down");
        assert_eq!((s.experts, s.in_features), (Some(EXPERTS), INTER));
        assert_eq!(s.counts, counts, "down counts");
        for (j, (g, w)) in s.sum_sq.iter().zip(&down_sums).enumerate() {
            assert!(
                (g - w).abs() <= 1e-3 * w.abs().max(1e-2),
                "down[{j}]: {g} vs {w}"
            );
        }
    }

    /// In router order: a handful of tokens, one gather matmul per slot.
    #[test]
    fn routed_experts_are_measured_in_router_order() {
        // Expert 3 is never picked, and the others are picked by two tokens each.
        check(3, |t| [[0, 2], [2, 1], [0, 1]][t]);
    }

    /// Grouped by expert, as a prefill runs: from 64 (token, expert) pairs up.
    #[test]
    fn routed_experts_are_measured_when_the_pairs_are_grouped() {
        check(40, |t| [(t % 3) as i32, (t % 3) as i32 + 1]);
    }
}
