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
//! the experts are read through the gather kernels, so an adapter on one would
//! never be applied.

use pmetal_runtime::compat::{Array, Dtype, Exception, nn};
use pmetal_runtime::impl_module_params;
use pmetal_runtime::native_moe;
use pmetal_runtime::native_weight::LayerWeight;
use pmetal_runtime::nn::moe::Linear as MoeLinear;

use crate::fp8_utils::dequantize_fp8_weight_for_compute;

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
}
impl_module_params!(SwitchMlp; gate_proj, up_proj, down_proj);

impl SwitchMlp {
    /// Randomly initialised experts (lazily: a checkpoint's weights replace
    /// the draw before anything computes it).
    pub fn new(num_experts: i32, hidden_size: i32, intermediate_size: i32) -> Self {
        let init = |in_features: i32, out_features: i32| {
            let bound = (in_features as f32).sqrt().recip();
            MoeLinear {
                weight: pmetal_runtime::compat::random::uniform_range(
                    -bound,
                    bound,
                    &[num_experts, out_features, in_features],
                    Dtype::Float32,
                ),
            }
        };
        Self {
            gate_proj: init(hidden_size, intermediate_size),
            up_proj: init(hidden_size, intermediate_size),
            down_proj: init(intermediate_size, hidden_size),
        }
    }

    /// How many experts the layer holds.
    pub fn num_experts(&self) -> i32 {
        self.gate_proj.weight.dim(0)
    }

    /// The projections as the gather matmuls take them, `[experts, in, out]`.
    fn dense_weights(&self) -> Result<(LayerWeight, LayerWeight, LayerWeight), Exception> {
        let weight = |linear: &MoeLinear| -> Result<LayerWeight, Exception> {
            Ok(LayerWeight::Dense(
                dequantize_fp8_weight_for_compute(&linear.weight)?.swap_axes(-1, -2),
            ))
        };
        Ok((
            weight(&self.gate_proj)?,
            weight(&self.up_proj)?,
            weight(&self.down_proj)?,
        ))
    }

    /// `Σ_k scores[s, k] · down(silu(gate(x_s)) · up(x_s))` over each row's
    /// experts `inds[s, k]`: `x_flat` is `[S, hidden]`, `inds` and `scores`
    /// `[S, top_k]` (`scores` in the dtype of `x_flat`), the result `[S, hidden]`.
    ///
    /// Through grouped gather matmuls, the shared
    /// [`native_moe::switch_glu`] the native engines run too.
    pub fn forward(
        &self,
        x_flat: &Array,
        inds: &Array,
        scores: &Array,
    ) -> Result<Array, Exception> {
        let (gate, up, down) = self.dense_weights()?;
        Ok(native_moe::switch_glu(
            x_flat,
            &gate,
            &up,
            &down,
            inds,
            scores,
            |gate, up| nn::silu(gate).multiply(up),
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
}
impl_module_params!(PlainSwitchMlp; fc1, fc2);

impl PlainSwitchMlp {
    /// Randomly initialised experts (lazily: a checkpoint's weights replace
    /// the draw before anything computes it).
    pub fn new(num_experts: i32, in_features: i32, intermediate_size: i32) -> Self {
        let init = |from: i32, to: i32| {
            let bound = (from as f32).sqrt().recip();
            MoeLinear {
                weight: pmetal_runtime::compat::random::uniform_range(
                    -bound,
                    bound,
                    &[num_experts, to, from],
                    Dtype::Float32,
                ),
            }
        };
        Self {
            fc1: init(in_features, intermediate_size),
            fc2: init(intermediate_size, in_features),
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
        let weight = |linear: &MoeLinear| -> Result<LayerWeight, Exception> {
            Ok(LayerWeight::Dense(
                dequantize_fp8_weight_for_compute(&linear.weight)?.swap_axes(-1, -2),
            ))
        };
        Ok(native_moe::switch_mlp(
            x_flat,
            &weight(&self.fc1)?,
            &weight(&self.fc2)?,
            inds,
            scores,
            activation,
        ))
    }
}
