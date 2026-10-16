//! The pieces of the Granite family beyond plain Granite: the routed experts of
//! `granitemoe` / `granitemoehybrid`, the shared MLP every `granitemoeshared` /
//! `granitemoehybrid` layer runs, and the weight loader that maps a released
//! checkpoint onto them (and onto Granite 4.0-H's Mamba-2 mixer).
//!
//! Reference: `transformers.models.granitemoehybrid.modeling_granitemoehybrid`
//! (`GraniteMoeHybridMambaLayer`, `GraniteMoeHybridMoE`, `GraniteMoeHybridMLP`).
//! `granitemoe` and `granitemoeshared` are the same layer with the Mamba and
//! shared-MLP branches removed, so one implementation serves all three.
//!
//! The Mamba-2 mixer itself is the shared [`Mamba2Mixer`](crate::common::mamba2::Mamba2Mixer)
//! (`common::mamba2`); what differs from Nemotron-H is around the scan: Granite
//! normalizes the gated output over the whole intermediate width, not per
//! group.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use pmetal_runtime::compat::{Array, Exception, Module, ModuleParametersExt, Param, nn, ops};
use pmetal_runtime::impl_module_params;

use super::granite::GraniteForCausalLM;

// =============================================================================
// Shared MLP
// =============================================================================

/// The dense SwiGLU every `granitemoeshared` / `granitemoehybrid` layer runs:
/// one fused `input_linear` whose output halves are `[gate | up]`.
#[derive(Debug)]
pub struct GraniteSharedMlp {
    pub input_linear: nn::Linear,
    pub output_linear: nn::Linear,
    pub size: i32,
}
impl_module_params!(GraniteSharedMlp; input_linear, output_linear);

impl GraniteSharedMlp {
    pub fn new(hidden: i32, size: i32) -> Result<Self, Exception> {
        Ok(Self {
            input_linear: nn::LinearBuilder::new(hidden, 2 * size)
                .bias(false)
                .build()?,
            output_linear: nn::LinearBuilder::new(size, hidden).bias(false).build()?,
            size,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let fused = Module::forward(&mut self.input_linear, x)?;
        let gate = ops::slice_axis(&fused, -1, 0, self.size);
        let up = ops::slice_axis(&fused, -1, self.size, 2 * self.size);
        Module::forward(&mut self.output_linear, &nn::silu(&gate).multiply(&up))
    }
}

// =============================================================================
// Routed experts
// =============================================================================

/// A stacked expert weight, `[experts, out, in]`, under a `weight` key so it
/// loads from `block_sparse_moe.input_linear.weight` as released.
#[derive(Debug)]
pub struct GraniteExpertBank {
    pub weight: Param<Array>,
}
impl_module_params!(GraniteExpertBank; weight);

/// The router: one bias-free linear, `router.layer.weight` `[experts, hidden]`.
#[derive(Debug)]
pub struct GraniteRouter {
    pub layer: nn::Linear,
}
impl_module_params!(GraniteRouter; layer);

/// `block_sparse_moe`: top-k routing over SwiGLU experts.
///
/// The router takes the top `k` logits and softmaxes over just those
/// (`GraniteMoeHybridTopKRouter`), so the weights always sum to one. The
/// experts are `input_linear` `[E, 2I, H]` (`[gate | up]` on the output axis)
/// and `output_linear` `[E, H, I]`, gathered per token by `gather_mm`.
#[derive(Debug)]
pub struct GraniteMoe {
    pub router: GraniteRouter,
    pub input_linear: GraniteExpertBank,
    pub output_linear: GraniteExpertBank,
    pub num_experts: i32,
    pub top_k: i32,
    pub intermediate: i32,
}
impl_module_params!(GraniteMoe; router, input_linear, output_linear);

impl GraniteMoe {
    pub fn new(
        hidden: i32,
        intermediate: i32,
        num_experts: i32,
        top_k: i32,
    ) -> Result<Self, Exception> {
        let bank = |shape: &[i32]| GraniteExpertBank {
            weight: Param::new(
                pmetal_runtime::compat::random::normal(
                    shape,
                    pmetal_runtime::compat::Dtype::Float32,
                )
                .multiply(&Array::from_f32(0.02)),
            ),
        };
        Ok(Self {
            router: GraniteRouter {
                layer: nn::LinearBuilder::new(hidden, num_experts)
                    .bias(false)
                    .build()?,
            },
            input_linear: bank(&[num_experts, 2 * intermediate, hidden]),
            output_linear: bank(&[num_experts, hidden, intermediate]),
            num_experts,
            top_k,
            intermediate,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape().to_vec();
        let hidden = *shape.last().expect("rank >= 1");
        let tokens: i32 = shape[..shape.len() - 1].iter().product();
        let flat = x.reshape(&[tokens, hidden]);

        // `F.linear(...).float()`, `topk`, then softmax over the k winners.
        let logits = Module::forward(&mut self.router.layer, &flat)?.as_type::<f32>();
        let (indices, top_logits) = crate::moe_routing::topk_normalize(&logits, self.top_k, false)?;
        let weights = ops::softmax_axis(&top_logits, -1).as_dtype(x.dtype().as_i32());

        // [N, 1, 1, H] against [E, H, 2I] gathered per row -> [N, k, 1, 2I].
        let rows = flat.expand_dims(1).expand_dims(2);
        let gate_up = rows.gather_mm(
            &self.input_linear.weight.as_ref().transpose_axes(&[0, 2, 1]),
            None,
            Some(&indices),
            false,
        );
        let i = self.intermediate;
        let gate = ops::slice_axis(&gate_up, -1, 0, i);
        let up = ops::slice_axis(&gate_up, -1, i, 2 * i);
        let activated = nn::silu(&gate).multiply(&up);
        // [N, k, 1, I] against [E, I, H] -> [N, k, H].
        let down = activated
            .gather_mm(
                &self
                    .output_linear
                    .weight
                    .as_ref()
                    .transpose_axes(&[0, 2, 1]),
                None,
                Some(&indices),
                false,
            )
            .squeeze_axes(&[2]);
        let mixed = down
            .multiply(&weights.reshape(&[tokens, self.top_k, 1]))
            .sum_axis(1, false);
        Ok(mixed.reshape(&shape))
    }
}

// =============================================================================
// Weight loading
// =============================================================================

/// What [`load_granite_weights`] did with a checkpoint.
#[derive(Debug, Default)]
pub struct GraniteLoadReport {
    /// Parameters assigned from the checkpoint.
    pub loaded: usize,
    /// Checkpoint keys deliberately not used (a tied checkpoint's spare
    /// `lm_head.weight`, rotary buffers).
    pub ignored: Vec<String>,
}

/// Load a released Granite checkpoint into `model`, all of it or not at all.
///
/// Every checkpoint tensor must land on a parameter of the same shape, and
/// every parameter must be supplied. The generic loader drops what it cannot
/// place, which for this family is the difference between a model and a
/// random one: a `granitemoe` checkpoint used to load into a dense layer with
/// every expert silently discarded. A layout this does not know is an error
/// naming the keys, not a model that parses and computes noise.
///
/// Two layouts are accepted: the released one (`input_linear`, `A_log`,
/// `conv1d.weight` as `[C, 1, K]`), and the MLX conversion of it, which splits
/// the fused expert projection into `switch_mlp.{gate,up,down}_proj`, renames
/// a dense model's `shared_mlp` to `mlp.{gate,up,down}_proj`, and stores the
/// conv as `[C, K, 1]`. MLX-quantized checkpoints are dequantized by the shared
/// reader before any of this sees them.
pub fn load_granite_weights(
    model: &mut GraniteForCausalLM,
    model_dir: &Path,
) -> Result<GraniteLoadReport, Exception> {
    let loaded = crate::loader::load_weights(model_dir)
        .map_err(|e| Exception::custom(format!("Granite: reading weights: {e:?}")))?;
    assign_granite_weights(model, loaded)
}

/// [`load_granite_weights`] on tensors already in memory.
pub fn assign_granite_weights(
    model: &mut GraniteForCausalLM,
    loaded: HashMap<String, Array>,
) -> Result<GraniteLoadReport, Exception> {
    let tied = model.lm_head.is_none();
    let dense_family = model.config.family()? == super::granite::GraniteFamily::Dense;
    let mut report = GraniteLoadReport::default();
    let weights = normalize_checkpoint_layout(loaded, dense_family, tied, &mut report)?;

    let mut params = model.flatten_params_mut();
    let mut unmatched = Vec::new();
    let mut assigned = HashSet::new();
    for (key, value) in weights {
        let Some(param) = params.get_mut(key.as_str()) else {
            unmatched.push(key);
            continue;
        };
        if param.shape() != value.shape() {
            return Err(Exception::custom(format!(
                "Granite: checkpoint tensor {key} is {:?}, the model expects {:?}",
                value.shape(),
                param.shape()
            )));
        }
        **param = value;
        assigned.insert(key);
        report.loaded += 1;
    }
    let mut missing: Vec<String> = params
        .keys()
        .filter(|k| !assigned.contains(k.as_str()))
        .cloned()
        .collect();

    if !unmatched.is_empty() || !missing.is_empty() {
        unmatched.sort();
        missing.sort();
        let sample = |v: &[String]| v.iter().take(8).cloned().collect::<Vec<_>>().join(", ");
        return Err(Exception::custom(format!(
            "Granite: checkpoint does not match the model the config describes \
             ({} unmatched checkpoint tensors: [{}]; {} parameters not in the checkpoint: [{}]). \
             Refusing to run with weights missing.",
            unmatched.len(),
            sample(&unmatched),
            missing.len(),
            sample(&missing),
        )));
    }
    Ok(report)
}

/// Rewrite an MLX-converted Granite checkpoint into the released layout, and
/// put the conv weights in MLX's `[C, K, 1]`.
fn normalize_checkpoint_layout(
    mut weights: HashMap<String, Array>,
    dense_family: bool,
    tied: bool,
    report: &mut GraniteLoadReport,
) -> Result<HashMap<String, Array>, Exception> {
    let keys: Vec<String> = weights.keys().cloned().collect();
    for key in &keys {
        if key.ends_with("rotary_emb.inv_freq") || (tied && key == "lm_head.weight") {
            weights.remove(key);
            report.ignored.push(key.clone());
        }
    }

    // Fused expert projection, split by the MLX conversion.
    let fuse = |weights: &mut HashMap<String, Array>,
                gate_key: &str,
                up_key: &str,
                fused_key: String,
                axis: i32|
     -> Result<(), Exception> {
        let gate = weights.remove(gate_key);
        let up = weights.remove(up_key);
        match (gate, up) {
            (Some(gate), Some(up)) => {
                weights.insert(fused_key, ops::concatenate_axis(&[&gate, &up], axis));
                Ok(())
            }
            (None, None) => Ok(()),
            _ => Err(Exception::custom(format!(
                "Granite: checkpoint has one of {gate_key} / {up_key} without the other"
            ))),
        }
    };
    let keys: Vec<String> = weights.keys().cloned().collect();
    for key in &keys {
        if let Some(prefix) = key.strip_suffix(".block_sparse_moe.switch_mlp.gate_proj.weight") {
            fuse(
                &mut weights,
                key,
                &format!("{prefix}.block_sparse_moe.switch_mlp.up_proj.weight"),
                format!("{prefix}.block_sparse_moe.input_linear.weight"),
                1,
            )?;
        } else if let Some(prefix) =
            key.strip_suffix(".block_sparse_moe.switch_mlp.down_proj.weight")
        {
            let value = weights.remove(key).expect("listed key");
            weights.insert(
                format!("{prefix}.block_sparse_moe.output_linear.weight"),
                value,
            );
        } else if !dense_family {
            // Plain Granite *is* `mlp.{gate,up,down}_proj`; only the MoE
            // family's dense layers were renamed from `shared_mlp`.
            if let Some(prefix) = key.strip_suffix(".mlp.gate_proj.weight") {
                fuse(
                    &mut weights,
                    key,
                    &format!("{prefix}.mlp.up_proj.weight"),
                    format!("{prefix}.shared_mlp.input_linear.weight"),
                    0,
                )?;
            } else if let Some(prefix) = key.strip_suffix(".mlp.down_proj.weight") {
                let value = weights.remove(key).expect("listed key");
                weights.insert(format!("{prefix}.shared_mlp.output_linear.weight"), value);
            }
        }
    }

    crate::common::mamba2::conv_weights_to_runtime_layout(&mut weights);
    Ok(weights)
}
