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

/// The projection as the gather matmuls take it, `[experts, in, out]`.
fn gather_weight(linear: &MoeLinear) -> Result<LayerWeight, Exception> {
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
        Ok(native_moe::switch_glu_adapted(
            x_flat,
            &gather_weight(&self.gate_proj)?,
            &gather_weight(&self.up_proj)?,
            &gather_weight(&self.down_proj)?,
            adapters,
            inds,
            scores,
            |gate, up, _| nn::silu(gate).multiply(up),
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
        Ok(native_moe::switch_mlp_adapted(
            x_flat,
            &gather_weight(&self.fc1)?,
            &gather_weight(&self.fc2)?,
            adapters,
            inds,
            scores,
            activation,
        ))
    }
}
