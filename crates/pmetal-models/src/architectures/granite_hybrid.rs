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
use pmetal_runtime::native_moe;
use pmetal_runtime::native_weight::LayerWeight;

use super::granite::GraniteForCausalLM;
use crate::fp8_utils::dequantize_fp8_weight_for_compute;

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

        // The experts through the grouped gather matmuls every routed layer
        // runs (`native_moe::switch_glu_fused`): `input_linear` is `[E, 2I, H]`,
        // gate then up on the output axis, and the gather reads the transposed
        // view `[E, H, 2I]` without a copy.
        let expert_weight = |bank: &GraniteExpertBank| -> Result<LayerWeight, Exception> {
            Ok(LayerWeight::Dense(
                dequantize_fp8_weight_for_compute(bank.weight.as_ref())?.swap_axes(-1, -2),
            ))
        };
        let mixed = native_moe::switch_glu_fused(
            &flat,
            &expert_weight(&self.input_linear)?,
            &expert_weight(&self.output_linear)?,
            &indices,
            &weights,
            |gate, up| nn::silu(gate).multiply(up),
        );
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
    load_granite_weights_with(model, model_dir, false)
}

/// [`load_granite_weights`], with an MLX-quantized checkpoint's projections and
/// embedding table kept packed when `keep_quantized`.
///
/// A packed module is named as the model names it (the MLX conversion's spelling
/// of anything else is unpacked to dense, as before), so it goes on its layer
/// before the audit; its words stand in the tensors, where the shape check finds
/// the shape the layer now has.
pub fn load_granite_weights_with(
    model: &mut GraniteForCausalLM,
    model_dir: &Path,
    keep_quantized: bool,
) -> Result<GraniteLoadReport, Exception> {
    let paths = keep_quantized.then(|| crate::loader::packable_paths(model));
    let (mut loaded, packed) =
        crate::loader::load_weights_keeping_packed(model_dir, paths.as_deref())
            .map_err(|e| Exception::custom(format!("Granite: reading weights: {e:?}")))?;
    loaded.extend(packed.words());
    if !packed.is_empty() {
        let count = packed
            .install(model)
            .map_err(|e| Exception::custom(format!("Granite: {e}")))?;
        tracing::info!("{count} Granite projections and tables stay packed");
    }
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
    // A packed layer's scales travel with its words (see
    // [`load_granite_weights_with`]): they are the layer's own, not tensors the
    // checkpoint was meant to carry under these names.
    let packed_aux = |name: &str| {
        name.ends_with(".scales") || name.ends_with(".biases") || name.ends_with(".tensor_scale")
    };
    let mut missing: Vec<String> = params
        .keys()
        .filter(|k| !assigned.contains(k.as_str()) && !packed_aux(k))
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
