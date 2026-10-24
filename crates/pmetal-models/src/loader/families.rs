//! Per-architecture checkpoint loaders that predate the generic loader: the
//! name-by-name assignments for the dense decoders, the Qwen3-Next loader and
//! the DeepSeek key rename.

use pmetal_runtime::compat::{Array, ModuleParametersExt};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::assign::packable_paths;
use super::quant::{QuantConfig, report_unpacked, take_packed_mlx_linears};
use super::{
    LoadError, dequantize_sidecar_weights, load_weights_filtered_raw, take_packed_linears,
    unpack_mlx_quantized_weights,
};
use crate::architectures::gemma::GemmaForCausalLM;
use crate::architectures::phi::PhiConfig;
use crate::architectures::qwen3_next::{
    PackedRoutedExperts, Qwen3NextConfig, Qwen3NextFeedForward, Qwen3NextForCausalLM,
    Qwen3NextSanitizeOptions, sanitize_weights, take_packed_experts,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Qwen3NextLoadOptions {
    pub skip_routed_experts: bool,
    /// Unpack NVIDIA ModelOpt NVFP4 routed experts to dense weights, as every
    /// other quantized tensor is, instead of keeping them packed on MLX's
    /// quantized gather kernels (a quarter of the memory, and what a trainer
    /// of a dense copy must not have).
    pub unpack_quantized_experts: bool,
    /// Keep an MLX-quantized checkpoint's projections, embedding table and
    /// routed experts packed (see [`LoadPolicy::keep_quantized`](crate::loader::LoadPolicy::keep_quantized)), where they
    /// are unpacked to dense by default. Routed experts that are offloaded are
    /// read from their own files and take no part.
    pub keep_quantized: bool,
}

fn is_non_text_auxiliary_weight(key: &str) -> bool {
    key.starts_with("model.visual.")
        || key.starts_with("visual.")
        || key.starts_with("vision_tower.")
        || key.starts_with("model.vision_tower.")
        || key.starts_with("model.audio.")
        || key.starts_with("model.multi_modal_projector.")
}

fn is_qwen3_next_raw_routed_expert_weight(key: &str) -> bool {
    key.contains(".mlp.experts.")
}

fn should_keep_qwen3_next_raw_weight(key: &str, options: Qwen3NextLoadOptions) -> bool {
    !(is_non_text_auxiliary_weight(key)
        || key.contains("mtp.")
        || options.skip_routed_experts && is_qwen3_next_raw_routed_expert_weight(key))
}

fn is_qwen3_next_routed_expert_param_key(key: &str) -> bool {
    key.ends_with(".mlp.switch_mlp_gate_proj")
        || key.ends_with(".mlp.switch_mlp_up_proj")
        || key.ends_with(".mlp.switch_mlp_down_proj")
}

fn is_qwen3_next_allowed_missing_param(key: &str, options: Qwen3NextLoadOptions) -> bool {
    options.skip_routed_experts && is_qwen3_next_routed_expert_param_key(key)
}

/// Load weights for Qwen3Next models with sanitization.
///
/// Handles expert weight stacking, (1+w) RMSNorm offset, and conv1d transposition.
pub fn load_qwen3_next_weights(
    model: &mut Qwen3NextForCausalLM,
    model_dir: impl AsRef<Path>,
    config: &Qwen3NextConfig,
) -> Result<(), LoadError> {
    load_qwen3_next_weights_with_options(model, model_dir, config, Qwen3NextLoadOptions::default())
}

pub fn load_qwen3_next_weights_with_options(
    model: &mut Qwen3NextForCausalLM,
    model_dir: impl AsRef<Path>,
    config: &Qwen3NextConfig,
    options: Qwen3NextLoadOptions,
) -> Result<(), LoadError> {
    // As `load_weights_filtered`, but the routed experts of a ModelOpt NVFP4
    // checkpoint are lifted out packed before the sidecars are unpacked.
    let model_dir = model_dir.as_ref();
    let keep = |key: &str| should_keep_qwen3_next_raw_weight(key, options);
    let mut weights = load_weights_filtered_raw(model_dir, keep)?;
    // An MLX-quantized checkpoint's projections, embedding table and routed
    // experts come out still packed, when the caller asked to keep them so;
    // what is left is unpacked to dense. The experts are named as an MLX
    // conversion stores them stacked, `mlp.switch_mlp.{gate,up,down}_proj`.
    let mut mlx_packed = if options.keep_quantized {
        let mut paths = packable_paths(model);
        if !options.skip_routed_experts {
            for layer in 0..config.num_hidden_layers.max(0) as usize {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    paths.push(format!("model.layers.{layer}.mlp.switch_mlp.{projection}"));
                }
            }
        }
        take_packed_mlx_linears(&mut weights, QuantConfig::read(model_dir)?.as_ref(), &paths)?
    } else {
        super::PackedLinears::default()
    };
    let unpacked = unpack_mlx_quantized_weights(model_dir, &mut weights)?;
    if options.keep_quantized {
        report_unpacked(&unpacked);
    }
    // The dense projections of a ModelOpt checkpoint stay packed as well,
    // unless the caller asked for everything unpacked: a 27B NVFP4 release is
    // 15 GB on its kernels and 54 GB as dense bf16. They come out first, with
    // the experts' tensors left where they are for `take_packed_experts`.
    let mut packed_linears = if options.skip_routed_experts || options.unpack_quantized_experts {
        super::PackedLinears::default()
    } else {
        take_packed_linears(&mut weights)?
    };
    packed_linears.absorb(&mut mlx_packed);
    let mut packed = if options.skip_routed_experts || options.unpack_quantized_experts {
        HashMap::new()
    } else {
        take_packed_experts(&mut weights, config.num_experts.max(0) as usize)?
    };
    // The stacked experts of an MLX conversion, onto the same packed kernels.
    for layer in 0..config.num_hidden_layers.max(0) as usize {
        let path = |projection: &str| format!("model.layers.{layer}.mlp.switch_mlp.{projection}");
        let taken = ["gate_proj", "up_proj", "down_proj"].map(|p| packed_linears.take(&path(p)));
        match taken {
            [Some(gate), Some(up), Some(down)] => {
                packed.insert(
                    layer,
                    PackedRoutedExperts {
                        gate: gate.into_layer_weight(),
                        up: up.into_layer_weight(),
                        down: down.into_layer_weight(),
                    },
                );
            }
            [None, None, None] => {}
            _ => {
                return Err(LoadError::SafeTensors(format!(
                    "layer {layer} keeps only some of its routed experts' projections packed"
                )));
            }
        }
    }
    dequantize_sidecar_weights(&mut weights)?;
    weights.retain(|key, _| keep(key));
    // `sanitize_weights` renames the checkpoint's keys (the vision-language
    // wrapper's prefix); the packed modules are named the same way.
    packed_linears.rename(|module| {
        let key = pmetal_runtime::qwen3_native::family::canonical_checkpoint_key(&format!(
            "{module}.weight"
        ));
        key.strip_suffix(".weight").unwrap_or(&key).to_string()
    });
    let packed_modules: Vec<String> = packed_linears.modules().map(str::to_owned).collect();
    sanitize_weights(
        &mut weights,
        config,
        Qwen3NextSanitizeOptions {
            skip_routed_experts: options.skip_routed_experts,
        },
    )
    .map_err(|e| LoadError::SafeTensors(format!("{:?}", e)))?;

    // Packed layers keep their experts outside the parameter tree.
    for (&layer_idx, experts) in &packed {
        let Some(Qwen3NextFeedForward::MoE(block)) = model
            .model
            .layers
            .get_mut(layer_idx)
            .map(|layer| &mut layer.mlp)
        else {
            return Err(LoadError::SafeTensors(format!(
                "packed experts for layer {layer_idx}, which is not a sparse layer"
            )));
        };
        block.use_packed_experts(experts.clone());
    }
    let packed_prefixes: Vec<String> = packed
        .keys()
        .map(|l| format!("model.layers.{l}.mlp.switch_mlp_"))
        .collect();

    // Apply sanitized weights to model parameters
    let mut params = model.flatten_params_mut();
    let expected_keys: HashSet<String> = params.keys().map(|key| key.to_string()).collect();
    let mut loaded_param_keys = HashSet::new();
    let mut matched = 0usize;
    let mut unmatched = Vec::new();
    for (key, value) in &weights {
        if let Some(param) = params.get_mut(&**key) {
            **param = value.clone();
            loaded_param_keys.insert(key.clone());
            matched += 1;
        } else {
            unmatched.push(key.clone());
        }
    }
    // The vision tower and the MTP predictor were filtered out by name before
    // sanitizing; a tensor left over now is a weight the model would run
    // without, so it is an error rather than a warning.
    if !unmatched.is_empty() {
        unmatched.sort();
        return Err(LoadError::SafeTensors(format!(
            "Qwen3Next weight loading found {} checkpoint tensor(s) no parameter consumes \
             (first 10): {:?}",
            unmatched.len(),
            &unmatched[..unmatched.len().min(10)]
        )));
    }

    drop(params);
    if !packed_linears.is_empty() {
        let count = packed_linears.install(model)?;
        tracing::info!("{count} projections and tables stay packed");
    }
    // A packed projection's `weight` is replaced by its words, not loaded
    // from the map.
    let is_packed_weight = |key: &str| {
        key.strip_suffix(".weight").is_some_and(|path| {
            packed_modules
                .iter()
                .any(|m| path == m || path.ends_with(&format!(".{m}")))
        })
    };

    let missing: Vec<String> = expected_keys
        .difference(&loaded_param_keys)
        .filter(|key| !is_qwen3_next_allowed_missing_param(key, options))
        .filter(|key| !is_packed_weight(key))
        .filter(|key| !packed_prefixes.iter().any(|p| key.starts_with(p.as_str())))
        .take(20)
        .cloned()
        .collect();
    if !missing.is_empty() {
        return Err(LoadError::SafeTensors(format!(
            "Qwen3Next weight loading is missing model parameters: {:?}",
            missing
        )));
    }

    tracing::info!("Qwen3Next weight loading: all {matched} weights matched successfully");
    Ok(())
}

/// Where Gemma 4's model tree and its checkpoint spell the same module
/// differently: `(model path segment, checkpoint segment)`. A packed module is
/// found by the checkpoint's name and placed by the model's.
const GEMMA4_SPELLINGS: [(&str, &str); 5] = [
    (".moe.router.", ".router."),
    (".per_layer_input_block.gate_proj", ".per_layer_input_gate"),
    (".per_layer_input_block.projection", ".per_layer_projection"),
    (
        "model.per_layer_inputs.model_projection",
        "model.per_layer_model_projection",
    ),
    (
        "model.per_layer_inputs.embed_tokens",
        "model.embed_tokens_per_layer",
    ),
];

/// A Gemma 4 checkpoint's tensors, with (when `keep_quantized`) the MLX-quantized
/// projections, embedding tables and routed experts the model can run packed
/// lifted out of them: [`install_gemma4_packed`] puts them on the model after
/// [`load_gemma4_weights`](crate::architectures::gemma4::load_gemma4_weights) has
/// filled the rest. Each packed module's words stand in the tensors under the
/// name the loader looks for, so the loader's account of what the checkpoint
/// holds stays whole.
pub(crate) fn read_gemma4_checkpoint(
    model: &mut crate::architectures::gemma4::Gemma4ForCausalLM,
    model_dir: &Path,
    keep_quantized: bool,
) -> Result<(HashMap<String, Array>, super::PackedLinears), LoadError> {
    let paths = keep_quantized.then(|| {
        let mut paths: Vec<String> = packable_paths(model)
            .into_iter()
            .map(|path| {
                GEMMA4_SPELLINGS
                    .iter()
                    .fold(path, |path, (model, checkpoint)| {
                        path.replace(model, checkpoint)
                    })
            })
            .collect();
        for (layer, block) in model.model.layers.iter().enumerate() {
            if block.moe.is_some() {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    paths.push(format!(
                        "model.layers.{layer}.experts.switch_glu.{projection}"
                    ));
                }
            }
        }
        paths
    });
    let (mut weights, packed) = super::load_weights_keeping_packed(model_dir, paths.as_deref())?;
    weights.extend(packed.words());
    Ok((weights, packed))
}

/// Put the packed modules [`read_gemma4_checkpoint`] lifted on the model: the
/// routed experts on their block, everything else on the `Linear` or
/// `Embedding` at its path.
pub(crate) fn install_gemma4_packed(
    model: &mut crate::architectures::gemma4::Gemma4ForCausalLM,
    mut packed: super::PackedLinears,
) -> Result<(), LoadError> {
    if packed.is_empty() {
        return Ok(());
    }
    let dtype = packed.dtype();
    for (layer, block) in model.model.layers.iter_mut().enumerate() {
        let Some(moe) = block.moe.as_mut() else {
            continue;
        };
        let path =
            |projection: &str| format!("model.layers.{layer}.experts.switch_glu.{projection}");
        let taken = ["gate_proj", "up_proj", "down_proj"].map(|p| packed.take(&path(p)));
        let [Some(gate), Some(up), Some(down)] = taken else {
            continue;
        };
        let (gate, up, down) = (gate.into_affine(), up.into_affine(), down.into_affine());
        match (gate, up, down) {
            (Some(gate), Some(up), Some(down))
                if (gate.bits, gate.group_size) == (up.bits, up.group_size)
                    && (gate.bits, gate.group_size) == (down.bits, down.group_size) =>
            {
                // The gate and up projections fuse along the output axis; each
                // output row is quantized on its own, so the words, scales and
                // biases concatenate exactly.
                let cat = |a: &Array, b: &Array| {
                    pmetal_runtime::compat::ops::concatenate_axis(&[a, b], 1)
                };
                moe.experts.install_packed(
                    (
                        cat(&gate.weight, &up.weight),
                        cat(&gate.scales, &up.scales),
                        cat(&gate.biases, &up.biases),
                    ),
                    (down.weight, down.scales, down.biases),
                    gate.group_size,
                    gate.bits,
                );
            }
            _ => {
                return Err(LoadError::SafeTensors(format!(
                    "layer {layer}: the routed experts are not all affine-quantized at one \
                     width and group size, which the packed experts need"
                )));
            }
        }
    }
    packed.rename(|module| {
        GEMMA4_SPELLINGS
            .iter()
            .fold(module.to_string(), |module, (model, checkpoint)| {
                module.replace(checkpoint, model)
            })
    });
    let count = packed.install(model)?;
    tracing::info!("{count} Gemma 4 projections and tables stay packed ({dtype:?})");
    Ok(())
}

pub fn load_gemma_weights(
    model: &mut GemmaForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = pmetal_runtime::compat::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight("model.embed_tokens.weight".into()));
    }
    if let Some(ref mut layers) = model.model.layers.gemma1 {
        for (i, layer) in layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{i}");
            load_linear_weight(
                &mut layer.self_attn.q_proj,
                weights,
                &format!("{prefix}.self_attn.q_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.k_proj,
                weights,
                &format!("{prefix}.self_attn.k_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.v_proj,
                weights,
                &format!("{prefix}.self_attn.v_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.o_proj,
                weights,
                &format!("{prefix}.self_attn.o_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.gate_proj,
                weights,
                &format!("{prefix}.mlp.gate_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.up_proj,
                weights,
                &format!("{prefix}.mlp.up_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.down_proj,
                weights,
                &format!("{prefix}.mlp.down_proj"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.input_layernorm,
                weights,
                &format!("{prefix}.input_layernorm"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.post_attention_layernorm,
                weights,
                &format!("{prefix}.post_attention_layernorm"),
            )?;
        }
    } else if let Some(ref mut layers) = model.model.layers.gemma2 {
        for (i, layer) in layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{i}");
            load_linear_weight(
                &mut layer.self_attn.q_proj,
                weights,
                &format!("{prefix}.self_attn.q_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.k_proj,
                weights,
                &format!("{prefix}.self_attn.k_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.v_proj,
                weights,
                &format!("{prefix}.self_attn.v_proj"),
            )?;
            load_linear_weight(
                &mut layer.self_attn.o_proj,
                weights,
                &format!("{prefix}.self_attn.o_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.gate_proj,
                weights,
                &format!("{prefix}.mlp.gate_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.up_proj,
                weights,
                &format!("{prefix}.mlp.up_proj"),
            )?;
            load_linear_weight(
                &mut layer.mlp.down_proj,
                weights,
                &format!("{prefix}.mlp.down_proj"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.input_layernorm,
                weights,
                &format!("{prefix}.input_layernorm"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.post_attention_layernorm,
                weights,
                &format!("{prefix}.post_attention_layernorm"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.pre_feedforward_layernorm,
                weights,
                &format!("{prefix}.pre_feedforward_layernorm"),
            )?;
            load_gemma_rms_norm_weight(
                &mut layer.post_feedforward_layernorm,
                weights,
                &format!("{prefix}.post_feedforward_layernorm"),
            )?;
            // Gemma 3's QK-norm. Present only when the config asked for it, so
            // a Gemma 2 checkpoint (which ships no such tensor) is not a
            // missing weight. `GemmaRmsNorm` inits to zeros and computes
            // `(1 + w)`, so an unloaded one is a silent identity scale rather
            // than a visible failure — hence loading it here rather than
            // trusting the generic path, which this loader does not use.
            if let Some(ref mut q_norm) = layer.self_attn.q_norm {
                load_gemma_rms_norm_weight(q_norm, weights, &format!("{prefix}.self_attn.q_norm"))?;
            }
            if let Some(ref mut k_norm) = layer.self_attn.k_norm {
                load_gemma_rms_norm_weight(k_norm, weights, &format!("{prefix}.self_attn.k_norm"))?;
            }
        }
    }
    load_gemma_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;
    Ok(())
}

/// The tensors of `weights` that [`load_gemma_weights`] does not read: a
/// checkpoint of another model, or a layout of this one the loader does not
/// know, either of which would otherwise load without a word and run without
/// the weight.
///
/// Sorted, for a stable message.
pub fn unread_gemma_weights(weights: &HashMap<String, Array>) -> Vec<String> {
    const LAYER_TENSORS: &[&str] = &[
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.bias",
        "self_attn.o_proj.bias",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
    ];
    let mut unread: Vec<String> = weights
        .keys()
        .filter(|key| {
            if matches!(
                key.as_str(),
                "model.embed_tokens.weight" | "model.norm.weight"
            ) {
                return false;
            }
            let layer_tensor = key
                .strip_prefix("model.layers.")
                .and_then(|rest| rest.split_once('.'))
                .filter(|(layer, _)| layer.parse::<usize>().is_ok())
                .map(|(_, tensor)| tensor);
            !layer_tensor.is_some_and(|tensor| LAYER_TENSORS.contains(&tensor))
        })
        .cloned()
        .collect();
    unread.sort();
    unread
}

/// Split Phi-3's fused `self_attn.qkv_proj` into `q_proj` / `k_proj` /
/// `v_proj`, in place.
///
/// Phi-3 ships one `[n_heads·head_dim + 2·n_kv_heads·head_dim, hidden]` tensor
/// where pmetal holds three projections. `Phi3Attention.forward` slices it as
/// `[:query_pos]`, `[query_pos : query_pos + kv]`, `[query_pos + kv :]`, so the
/// row order is q, k, v. Without this the fused key matches no parameter, the
/// generic loader drops it silently, and all three projections run on random
/// init — a whole model's attention, gone, behind finite plausible logits.
///
/// A no-op on checkpoints that already ship the three tensors (Phi-4-mini),
/// which is why it can run unconditionally on the Phi path.
pub fn split_phi_fused_qkv(weights: &mut HashMap<String, Array>, config: &PhiConfig) {
    let head_dim = config.head_dim();
    let q_rows = config.num_attention_heads * head_dim;
    let kv_rows = config.num_key_value_heads * head_dim;

    let fused: Vec<String> = weights
        .keys()
        .filter(|k| k.ends_with(".self_attn.qkv_proj.weight"))
        .cloned()
        .collect();

    for key in fused {
        let Some(w) = weights.remove(&key) else {
            continue;
        };
        let shape = w.shape().to_vec();
        if shape.len() != 2 || shape[0] != q_rows + 2 * kv_rows {
            // Not the geometry this config describes. Put it back rather than
            // silently dropping a tensor we failed to understand.
            weights.insert(key, w);
            continue;
        }
        let cols = shape[1];
        let prefix = key.trim_end_matches("qkv_proj.weight");
        for (name, start, stop) in [
            ("q_proj", 0, q_rows),
            ("k_proj", q_rows, q_rows + kv_rows),
            ("v_proj", q_rows + kv_rows, q_rows + 2 * kv_rows),
        ] {
            weights.insert(
                format!("{prefix}{name}.weight"),
                w.slice(&[start, 0], &[stop, cols]),
            );
        }
    }
}

fn load_linear_weight(
    linear: &mut pmetal_runtime::compat::nn::Linear,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        linear.weight = pmetal_runtime::compat::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    let bias_key = format!("{prefix}.bias");
    if let Some(b) = weights.get(&bias_key) {
        linear.bias = pmetal_runtime::compat::module::Param::new(Some(b.clone()));
    }
    Ok(())
}

fn load_gemma_rms_norm_weight(
    norm: &mut crate::architectures::gemma::GemmaRmsNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        norm.weight = pmetal_runtime::compat::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_next_raw_weight_filter_skips_auxiliary_and_routed_expert_keys() {
        let options = Qwen3NextLoadOptions {
            skip_routed_experts: true,
            ..Default::default()
        };

        assert!(should_keep_qwen3_next_raw_weight(
            "model.language_model.embed_tokens.weight",
            options
        ));
        assert!(!should_keep_qwen3_next_raw_weight(
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            options
        ));
        assert!(!should_keep_qwen3_next_raw_weight(
            "model.visual.patch_embed.weight",
            options
        ));
        assert!(!should_keep_qwen3_next_raw_weight("mtp.fc.weight", options));
    }

    #[test]
    fn qwen3_next_missing_allowlist_only_covers_routed_expert_params() {
        let options = Qwen3NextLoadOptions {
            skip_routed_experts: true,
            ..Default::default()
        };

        assert!(is_qwen3_next_allowed_missing_param(
            "model.layers.0.mlp.switch_mlp_gate_proj",
            options
        ));
        assert!(!is_qwen3_next_allowed_missing_param(
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            options
        ));
        assert!(!is_qwen3_next_allowed_missing_param(
            "model.layers.0.mlp.switch_mlp_gate_proj",
            Qwen3NextLoadOptions::default()
        ));
    }
}
