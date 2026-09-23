//! Model loading utilities for PMetal.
//!
//! Provides functionality to load model weights from safetensor files,
//! with support for HuggingFace model formats and weight name mapping.
use pmetal_bridge::compat::{
    Array, Exception, ModuleParameters, ModuleParametersExt, Param, nn, transforms,
};

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::architectures::bert::BertForEmbedding;
use crate::architectures::clip::CLIPTextModel;
use crate::architectures::flux::FluxDiT;
use crate::architectures::gemma::GemmaForCausalLM;
use crate::architectures::llama::{LlamaConfig, LlamaForCausalLM};
use crate::architectures::mistral::MistralForCausalLM;
use crate::architectures::mllama::MllamaForConditionalGeneration;
use crate::architectures::nemotron_h::{
    NemotronHForCausalLM, load_nemotron_weights as load_nemotron,
};
use crate::architectures::phi::{PhiConfig, PhiForCausalLM};
use crate::architectures::qwen2::Qwen2ForCausalLM;
use crate::architectures::qwen3::Qwen3ForCausalLM;
use crate::architectures::qwen3_next::{
    Qwen3NextConfig, Qwen3NextForCausalLM, Qwen3NextSanitizeOptions, sanitize_weights,
};
use crate::architectures::t5::T5EncoderModel;
use crate::architectures::utils::LoadReport;
use crate::architectures::vae::FluxVAE;

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("SafeTensors error: {0}")]
    SafeTensors(String),
    #[error("Missing weight: {0}")]
    MissingWeight(String),
    #[error("Shape mismatch for {key}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        key: String,
        expected: Vec<i32>,
        actual: Vec<i32>,
    },
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("MLX error: {0}")]
    Mlx(String),
    #[error("MLX IO error: {0}")]
    MlxIo(#[from] pmetal_bridge::compat::IoError),
}

impl From<pmetal_bridge::compat::Exception> for LoadError {
    fn from(e: pmetal_bridge::compat::Exception) -> Self {
        Self::Mlx(e.to_string())
    }
}

const PARAM_EVAL_BATCH_SIZE: usize = 128;

/// Load a safetensors shard file into a `HashMap<String, Array>`.
///
/// This wraps `pmetal_bridge::inline_array::load_safetensors_shard` which
/// returns `Option<Vec<(String, InlineArray)>>` and converts it into the
/// `HashMap<String, Array>` that the model loaders expect.
fn load_shard(path: &std::path::Path) -> Result<HashMap<String, Array>, LoadError> {
    let path_str = path.to_str().ok_or_else(|| {
        LoadError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Non-UTF8 path: {:?}", path),
        ))
    })?;
    pmetal_bridge::inline_array::load_safetensors_shard(path_str)
        .map(|pairs| pairs.into_iter().collect::<HashMap<_, _>>())
        .ok_or_else(|| LoadError::SafeTensors(format!("Failed to load safetensors: {}", path_str)))
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Qwen3NextLoadOptions {
    pub skip_routed_experts: bool,
}

/// Width of one quantized tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TensorQuant {
    bits: i32,
    group_size: i32,
}

#[derive(Debug, Clone)]
struct MlxQuantizationConfig {
    bits: i32,
    group_size: i32,
    /// Per-tensor widths, keyed by the tensor's own checkpoint key.
    overrides: HashMap<String, TensorQuant>,
}

impl MlxQuantizationConfig {
    /// The width to unpack `key` at, which is the model-wide one unless the
    /// checkpoint said otherwise for this tensor.
    fn for_tensor(&self, key: &str) -> TensorQuant {
        self.overrides.get(key).copied().unwrap_or(TensorQuant {
            bits: self.bits,
            group_size: self.group_size,
        })
    }
}

/// How many checkpoint keys [`detect_namespace_prefix`] probes before deciding.
/// Each probe narrows the candidate set, and one alone is enough whenever the
/// parameter tree has no repeated tails.
const NAMESPACE_PROBE_KEYS: usize = 8;

/// The namespace shift, if any, that reconciles a checkpoint's keys with a
/// model's parameter names.
///
/// Publishers disagree on whether the text tower carries its wrapper's
/// namespace. `Qwen/Qwen3-Embedding-8B` ships `embed_tokens.weight`,
/// `layers.0.…` and `norm.weight` where `Qwen3ForCausalLM` holds all three
/// under `model.`, so an exact-name load matches nothing and the model answers
/// from its random init.
///
/// The shift is detected once for the whole checkpoint rather than per key. A
/// per-key fallback would let a checkpoint carrying both `layers.0.x` and
/// `model.layers.0.x` resolve both onto the same parameter, in whatever order
/// the `HashMap` happened to yield them.
///
/// Returns `None` when the names already line up, when nothing explains them,
/// or when two prefixes explain them equally well — which leaves the mismatch
/// for [`LoadReport`] to report rather than papering over it with a guess.
fn detect_namespace_prefix(
    params: &HashMap<String, &mut Array>,
    loaded: &HashMap<String, Array>,
) -> Option<String> {
    let explains = |prefix: &str| -> usize {
        loaded
            .keys()
            .filter(|key| params.contains_key(&format!("{prefix}{key}")))
            .count()
    };

    let already = explains("");
    if already == loaded.len() {
        return None;
    }

    // A candidate is whatever the parameter tree puts in front of a checkpoint
    // key. A handful of probes surfaces every prefix in play; scoring then runs
    // each candidate against the whole checkpoint, so one stray key among the
    // probes cannot veto a prefix that explains all the rest.
    let mut probes: Vec<&String> = loaded.keys().collect();
    probes.sort_unstable();
    probes.truncate(NAMESPACE_PROBE_KEYS);

    let mut candidates: HashSet<String> = HashSet::new();
    for probe in probes {
        let tail = format!(".{probe}");
        candidates.extend(
            params
                .keys()
                .filter_map(|name| name.strip_suffix(tail.as_str()))
                .map(|prefix| format!("{prefix}.")),
        );
    }

    let mut ranked: Vec<(usize, String)> = candidates
        .into_iter()
        .map(|prefix| (explains(&prefix), prefix))
        .collect();
    ranked.sort_unstable();

    match ranked.pop() {
        // A prefix that explains no more than the bare names is not a shift,
        // and two prefixes explaining equally much is a coin flip that would
        // silently half-load the model.
        Some((best, prefix)) if best > already && !ranked.iter().any(|(n, _)| *n == best) => {
            Some(prefix)
        }
        _ => None,
    }
}

/// Assign every checkpoint tensor whose name matches a parameter path,
/// reporting the ones that matched nothing.
///
/// Matching is by exact name, after [`detect_namespace_prefix`] reconciles a
/// checkpoint that omits the model's namespace. A checkpoint laid out
/// differently in some other way still contributes nothing and leaves the model
/// on its random init, so the unmatched keys come back in the report —
/// `load_generic_weights_renamed` logs a summary, and a caller that knows the
/// checkpoint should map completely can assert on it.
fn assign_loaded_weights<M: ModuleParameters + ModuleParametersExt>(
    model: &mut M,
    loaded: HashMap<String, Array>,
) -> LoadReport {
    let mut params = model.flatten_params_mut();
    let prefix = detect_namespace_prefix(&params, &loaded);
    let mut report = LoadReport::default();
    for (key, value) in loaded {
        let name = match &prefix {
            Some(prefix) => format!("{prefix}{key}"),
            None => key.clone(),
        };
        match params.get_mut(&name) {
            Some(param) => {
                **param = value;
                report.loaded += 1;
            }
            // The checkpoint's own spelling, not the probed one, since that is
            // what the reader has in front of them.
            None => report.skipped.push(key),
        }
    }
    report
}

fn eval_loaded_parameters<M: ModuleParameters + ModuleParametersExt>(
    model: &M,
) -> Result<(), LoadError> {
    let params = model.flatten_params();
    let arrays: Vec<Array> = params.into_values().collect();
    for chunk in arrays.chunks(PARAM_EVAL_BATCH_SIZE) {
        pmetal_bridge::compat::transforms::eval(chunk.iter())
            .map_err(|e| LoadError::Mlx(e.to_string()))?;
    }
    Ok(())
}

fn load_mlx_quantization_config(
    model_dir: &Path,
) -> Result<Option<MlxQuantizationConfig>, LoadError> {
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return Ok(None);
    }

    let raw = std::fs::read_to_string(&config_path)?;
    // Through `config_value`, not `serde_json::from_str`: Python's `json.dump`
    // emits bare `Infinity`, which is not JSON. `nvidia/Nemotron-H-8B-Base-8K`
    // ships `"time_step_limit": [0.0, Infinity]`, and reading it strictly here
    // failed the *whole load* with a parse error 46 lines from anything this
    // function cares about. Architecture detection already routed around it;
    // this was the other half, and no config-only sweep could see it because
    // no config-only sweep loads weights.
    let json =
        crate::dispatcher::config_value(&raw).map_err(|e| LoadError::SafeTensors(e.to_string()))?;
    let Some(quant) = json
        .get("quantization")
        .or_else(|| json.get("quantization_config"))
        .or_else(|| {
            json.get("text_config").and_then(|text| {
                text.get("quantization")
                    .or_else(|| text.get("quantization_config"))
            })
        })
    else {
        return Ok(None);
    };

    let bits = quant
        .get("bits")
        .and_then(|value| value.as_i64())
        .unwrap_or(4) as i32;
    let group_size = quant
        .get("group_size")
        .and_then(|value| value.as_i64())
        .unwrap_or(64) as i32;
    if bits <= 0 {
        return Err(LoadError::SafeTensors(format!(
            "Invalid MLX quantization bits in config.json: {bits}"
        )));
    }
    if group_size <= 0 {
        return Err(LoadError::SafeTensors(format!(
            "Invalid MLX quantization group_size in config.json: {group_size}"
        )));
    }

    let mut overrides: HashMap<String, TensorQuant> = HashMap::new();

    // MLX's own layout, and the one mlx-community QAT releases ship: each
    // module that differs from the model-wide setting gets its own entry
    // *beside* `bits` and `group_size`, keyed by module path, e.g.
    //
    //   "language_model.model.layers.0.mlp.gate_proj": {"bits": 8, "group_size": 64}
    //
    // mlx-lm reads these in `class_predicate` when rebuilding the model. Not
    // reading them unpacks those tensors at the model-wide width, which yields
    // the right byte count and the wrong shape, so the failure surfaces several
    // ops later as a norm complaining about its input.
    //
    // A module may also map to `false`, meaning it was left unquantized; those
    // carry no `.scales`/`.biases` and are skipped on that basis below.
    if let Some(object) = quant.as_object() {
        for (module_path, value) in object {
            let Some(entry) = value.as_object() else {
                continue;
            };
            let read = |field: &str, fallback: i32| {
                entry
                    .get(field)
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32)
                    .unwrap_or(fallback)
            };
            let quant = TensorQuant {
                bits: read("bits", bits),
                group_size: read("group_size", group_size),
            };
            if quant.bits <= 0 || quant.group_size <= 0 {
                return Err(LoadError::SafeTensors(format!(
                    "Invalid MLX quantization for {module_path}: {quant:?}"
                )));
            }
            // Keyed by module; the packed tensor is that module's `weight`.
            overrides.insert(format!("{module_path}.weight"), quant);
        }
    }

    // pmetal's own mixed-precision output, which records bits per *tensor* key.
    if let Some(object) = quant
        .get("per_tensor_overrides")
        .and_then(|value| value.as_object())
    {
        for (tensor_key, value) in object {
            if let Some(per_tensor_bits) = value.as_i64() {
                overrides.insert(
                    tensor_key.clone(),
                    TensorQuant {
                        bits: per_tensor_bits as i32,
                        group_size,
                    },
                );
            }
        }
    }

    Ok(Some(MlxQuantizationConfig {
        bits,
        group_size,
        overrides,
    }))
}

fn quant_aux_base_key(key: &str) -> Option<&str> {
    key.strip_suffix(".scales")
        .or_else(|| key.strip_suffix(".biases"))
}

fn is_quant_aux_key(key: &str) -> bool {
    quant_aux_base_key(key).is_some()
}

/// The packed tensor a `.scales` / `.biases` entry belongs to, if it is present.
///
/// ⚠️ Two layouts ship, and they differ by one path segment.
///
/// **MLX**, and so every `mlx-community` release, stores a quantized module's
/// three parameters as *siblings*: `mlp.up_proj.weight` holds the packed
/// payload next to `mlp.up_proj.scales` and `mlp.up_proj.biases`.
///
/// **pmetal's own quantizer** suffixes the full tensor name instead, so the
/// same module comes out as `mlp.up_proj.weight` with
/// `mlp.up_proj.weight.scales` and `mlp.up_proj.weight.biases`.
///
/// Preferring the `.weight` child and falling back to the base name lands on
/// the right tensor either way. A module path is never itself a tensor, so the
/// two cases cannot both apply to one entry.
fn packed_weight_key_for(aux_key: &str, weights: &HashMap<String, Array>) -> Option<String> {
    let base = quant_aux_base_key(aux_key)?;
    let child = format!("{base}.weight");
    if weights.contains_key(&child) {
        Some(child)
    } else if weights.contains_key(base) {
        Some(base.to_string())
    } else {
        None
    }
}

fn dequantize_mlx_quantized_weights(
    weights: &mut HashMap<String, Array>,
    config: &MlxQuantizationConfig,
) {
    // `(packed, scales, biases)` triples, resolved before anything is replaced
    // so the lookups above see the checkpoint as it was read.
    let triples: Vec<(String, String, String)> = weights
        .keys()
        .filter(|key| key.ends_with(".scales"))
        .filter_map(|scales_key| {
            let weight_key = packed_weight_key_for(scales_key, weights)?;
            let biases_key = format!("{}.biases", quant_aux_base_key(scales_key)?);
            weights
                .contains_key(&biases_key)
                .then(|| (weight_key, scales_key.clone(), biases_key))
        })
        .collect();

    for (weight_key, scales_key, biases_key) in triples {
        let (Some(weight), Some(scales), Some(biases)) = (
            weights.get(&weight_key).cloned(),
            weights.get(&scales_key).cloned(),
            weights.get(&biases_key).cloned(),
        ) else {
            continue;
        };
        // Both the width *and* the group size can vary per tensor. Unpacking a
        // mixed-precision checkpoint at one model-wide setting silently
        // produces wrongly-shaped weights.
        let quant = config.for_tensor(&weight_key);
        let dequantized = weight.dequantize(&scales, &biases, quant.group_size, quant.bits);
        weights.insert(weight_key, dequantized);
    }

    weights.retain(|key, _| !is_quant_aux_key(key));
}

/// Unpack every per-tensor-sidecar scheme to dense, as MLX's own format is
/// unpacked above: NVIDIA ModelOpt nvfp4 and fp8, and Qwen's block-scaled FP8.
///
/// ⚠️ Before this, their sidecars fell through as unmatched keys with a
/// warning, and the packed bytes were assigned to dense `Linear` weights. The
/// load reported every weight matched, and anything trained on it trained on
/// noise.
fn dequantize_sidecar_weights(weights: &mut HashMap<String, Array>) -> Result<(), LoadError> {
    use pmetal_bridge::native_loader as nl;

    if !weights
        .keys()
        .any(|key| nl::quant_sidecar_base(key).is_some())
    {
        return Ok(());
    }
    let dtype = pmetal_bridge::native_weight::detect_model_dtype(|key| {
        weights.get(key).map(|w| w.dtype_raw())
    });

    for (base, weight) in nl::take_modelopt_weights(weights).map_err(LoadError::SafeTensors)? {
        let dense = weight
            .to_dense(dtype)
            .map_err(|e| LoadError::SafeTensors(format!("{base}.weight: {e}")))?;
        weights.insert(format!("{base}.weight"), dense);
    }

    let block_fp8: Vec<String> = weights
        .keys()
        .filter_map(|key| key.strip_suffix(".weight_scale_inv"))
        .map(ToOwned::to_owned)
        .collect();
    for base in block_fp8 {
        let weight_key = format!("{base}.weight");
        let (Some(scale_inv), Some(weight)) = (
            weights.remove(&format!("{base}.weight_scale_inv")),
            weights.remove(&weight_key),
        ) else {
            return Err(LoadError::SafeTensors(format!(
                "FP8 scale {base}.weight_scale_inv has no weight"
            )));
        };
        let dense = nl::dequantize_fp8_e4m3_scaled_weight(&weight, &scale_inv, dtype)
            .map_err(|e| LoadError::SafeTensors(format!("{weight_key}: {e}")))?;
        weights.insert(weight_key, dense);
    }

    // An activation calibration with no weight scale beside it changes no
    // weight, so it is safe to drop. A weight scale is not.
    weights.retain(|key, _| !key.ends_with(".input_scale"));
    let leftover: Vec<&String> = weights
        .keys()
        .filter(|key| nl::quant_sidecar_base(key).is_some())
        .take(10)
        .collect();
    if !leftover.is_empty() {
        return Err(LoadError::SafeTensors(format!(
            "quantization scales this loader cannot apply, so their weights would load as raw bytes: {leftover:?}"
        )));
    }
    Ok(())
}

/// Whether a sharded checkpoint carries per-tensor quantization sidecars, which
/// cannot be applied a shard at a time.
fn index_has_quant_sidecars(index: &WeightIndex) -> bool {
    index
        .weight_map
        .keys()
        .any(|key| pmetal_bridge::native_loader::quant_sidecar_base(key).is_some())
}

pub fn load_clip_weights(
    model: &mut CLIPTextModel,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    for (name, weight) in weights {
        let name = if let Some(stripped) = name.strip_prefix("text_model.") {
            stripped
        } else {
            name
        };

        match name {
            "embeddings.token_embedding.weight" => {
                model.token_embedding.weight =
                    pmetal_bridge::compat::module::Param::new(weight.clone())
            }
            "embeddings.position_embedding.weight" => {
                model.position_embedding = pmetal_bridge::compat::module::Param::new(weight.clone())
            }
            "final_layer_norm.weight" => {
                model.final_layer_norm.weight =
                    pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
            }
            "final_layer_norm.bias" => {
                model.final_layer_norm.bias =
                    pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
            }
            _ if name.starts_with("encoder.layers.") => {
                let parts: Vec<&str> = name.split('.').collect();
                let idx = parts[2].parse::<usize>().map_err(|_| {
                    LoadError::SafeTensors(format!("Invalid layer index in key: {}", name))
                })?;
                if idx >= model.layers.len() {
                    continue;
                }
                let sub_path = parts[3..].join(".");
                match sub_path.as_str() {
                    "self_attn.q_proj.weight" => {
                        model.layers[idx].attn.q_proj.weight =
                            pmetal_bridge::compat::module::Param::new(weight.clone())
                    }
                    "self_attn.q_proj.bias" => {
                        model.layers[idx].attn.q_proj.bias =
                            pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                    }
                    "self_attn.k_proj.weight" => {
                        model.layers[idx].attn.k_proj.weight =
                            pmetal_bridge::compat::module::Param::new(weight.clone())
                    }
                    "self_attn.k_proj.bias" => {
                        model.layers[idx].attn.k_proj.bias =
                            pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                    }
                    "self_attn.v_proj.weight" => {
                        model.layers[idx].attn.v_proj.weight =
                            pmetal_bridge::compat::module::Param::new(weight.clone())
                    }
                    "self_attn.v_proj.bias" => {
                        model.layers[idx].attn.v_proj.bias =
                            pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                    }
                    "self_attn.out_proj.weight" => {
                        model.layers[idx].attn.out_proj.weight =
                            pmetal_bridge::compat::module::Param::new(weight.clone())
                    }
                    "self_attn.out_proj.bias" => {
                        model.layers[idx].attn.out_proj.bias =
                            pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                    }
                    "layer_norm1.weight" => {
                        model.layers[idx].norm1.weight =
                            pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                    }
                    "layer_norm1.bias" => {
                        model.layers[idx].norm1.bias =
                            pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                    }
                    "layer_norm2.weight" => {
                        model.layers[idx].norm2.weight =
                            pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                    }
                    "layer_norm2.bias" => {
                        model.layers[idx].norm2.bias =
                            pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                    }
                    "mlp.fc1.weight" => {
                        model.layers[idx].mlp.fc1.weight =
                            pmetal_bridge::compat::module::Param::new(weight.clone())
                    }
                    "mlp.fc1.bias" => {
                        model.layers[idx].mlp.fc1.bias =
                            pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                    }
                    "mlp.fc2.weight" => {
                        model.layers[idx].mlp.fc2.weight =
                            pmetal_bridge::compat::module::Param::new(weight.clone())
                    }
                    "mlp.fc2.bias" => {
                        model.layers[idx].mlp.fc2.bias =
                            pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    Ok(())
}

pub fn load_t5_weights(
    model: &mut T5EncoderModel,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    for (name, weight) in weights {
        match name.as_str() {
            "shared.weight" => {
                model.shared.weight = pmetal_bridge::compat::module::Param::new(weight.clone())
            }
            "encoder.final_layer_norm.weight" => {
                model.final_layer_norm.weight =
                    pmetal_bridge::compat::module::Param::new(weight.clone())
            }
            _ if name.starts_with("encoder.block.") => {
                let parts: Vec<&str> = name.split('.').collect();
                let idx = parts[2].parse::<usize>().map_err(|_| {
                    LoadError::SafeTensors(format!("Invalid layer index in key: {}", name))
                })?;
                if idx >= model.blocks.len() {
                    continue;
                }
                let sub_path = parts[3..].join(".");
                match sub_path.as_str() {
                    "layer.0.SelfAttention.q.weight" => {
                        model.blocks[idx].layer_0_attn.q.weight =
                            pmetal_bridge::compat::module::Param::new(weight.clone())
                    }
                    "layer.0.SelfAttention.k.weight" => {
                        model.blocks[idx].layer_0_attn.k.weight =
                            pmetal_bridge::compat::module::Param::new(weight.clone())
                    }
                    "layer.0.SelfAttention.v.weight" => {
                        model.blocks[idx].layer_0_attn.v.weight =
                            pmetal_bridge::compat::module::Param::new(weight.clone())
                    }
                    "layer.0.SelfAttention.o.weight" => {
                        model.blocks[idx].layer_0_attn.o.weight =
                            pmetal_bridge::compat::module::Param::new(weight.clone())
                    }
                    "layer.0.SelfAttention.relative_attention_bias.weight" => {
                        if let Some(ref mut rel_bias) =
                            model.blocks[idx].layer_0_attn.relative_attention_bias
                        {
                            rel_bias.embedding.weight =
                                pmetal_bridge::compat::module::Param::new(weight.clone());
                        }
                    }
                    "layer.0.layer_norm.weight" => {
                        model.blocks[idx].layer_0_norm.weight =
                            pmetal_bridge::compat::module::Param::new(weight.clone())
                    }
                    "layer.1.DenseReluDense.wi_0.weight" => {
                        model.blocks[idx].layer_1_mlp.wi_0.weight =
                            pmetal_bridge::compat::module::Param::new(weight.clone())
                    }
                    "layer.1.DenseReluDense.wi_1.weight" => {
                        model.blocks[idx].layer_1_mlp.wi_1.weight =
                            pmetal_bridge::compat::module::Param::new(weight.clone())
                    }
                    "layer.1.DenseReluDense.wo.weight" => {
                        model.blocks[idx].layer_1_mlp.wo.weight =
                            pmetal_bridge::compat::module::Param::new(weight.clone())
                    }
                    "layer.1.layer_norm.weight" => {
                        model.blocks[idx].layer_1_norm.weight =
                            pmetal_bridge::compat::module::Param::new(weight.clone())
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    Ok(())
}

pub fn load_vae_weights(
    model: &mut FluxVAE,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    // Helper to load ResnetBlock weights from HF-style keys
    fn load_resnet_block(
        block: &mut crate::architectures::vae::ResnetBlock,
        weights: &HashMap<String, Array>,
        prefix: &str,
    ) -> Result<(), LoadError> {
        load_group_norm_weight(&mut block.norm1, weights, &format!("{prefix}.norm1"))?;
        load_conv2d_weight(&mut block.conv1, weights, &format!("{prefix}.conv1"))?;
        load_group_norm_weight(&mut block.norm2, weights, &format!("{prefix}.norm2"))?;
        load_conv2d_weight(&mut block.conv2, weights, &format!("{prefix}.conv2"))?;
        if let Some(ref mut shortcut) = block.conv_shortcut {
            // Only load if the weights exist (skip silently if they don't)
            let key = format!("{prefix}.conv_shortcut.weight");
            if weights.contains_key(&key) {
                load_conv2d_weight(shortcut, weights, &format!("{prefix}.conv_shortcut"))?;
            }
        }
        Ok(())
    }

    // Helper to load VAEAttentionBlock weights
    fn load_attn_block(
        attn: &mut crate::architectures::vae::VAEAttentionBlock,
        weights: &HashMap<String, Array>,
        prefix: &str,
    ) -> Result<(), LoadError> {
        load_group_norm_weight(&mut attn.norm, weights, &format!("{prefix}.group_norm"))?;
        load_conv2d_weight(&mut attn.q, weights, &format!("{prefix}.to_q"))?;
        load_conv2d_weight(&mut attn.k, weights, &format!("{prefix}.to_k"))?;
        load_conv2d_weight(&mut attn.v, weights, &format!("{prefix}.to_v"))?;
        load_conv2d_weight(&mut attn.proj_out, weights, &format!("{prefix}.to_out.0"))?;
        Ok(())
    }

    // Encoder weights
    if let Some(ref mut encoder) = model.encoder {
        load_conv2d_weight(&mut encoder.conv_in, weights, "encoder.conv_in")?;

        // Down blocks: block 0 has no downsampler, blocks 1-3 have downsamplers
        load_resnet_block(
            &mut encoder.down_1_0,
            weights,
            "encoder.down_blocks.0.resnets.0",
        )?;
        load_resnet_block(
            &mut encoder.down_1_1,
            weights,
            "encoder.down_blocks.0.resnets.1",
        )?;

        load_resnet_block(
            &mut encoder.down_2_0,
            weights,
            "encoder.down_blocks.1.resnets.0",
        )?;
        load_resnet_block(
            &mut encoder.down_2_1,
            weights,
            "encoder.down_blocks.1.resnets.1",
        )?;
        load_conv2d_weight(
            &mut encoder.down_2_sampler.conv,
            weights,
            "encoder.down_blocks.1.downsamplers.0.conv",
        )?;

        load_resnet_block(
            &mut encoder.down_3_0,
            weights,
            "encoder.down_blocks.2.resnets.0",
        )?;
        load_resnet_block(
            &mut encoder.down_3_1,
            weights,
            "encoder.down_blocks.2.resnets.1",
        )?;
        load_conv2d_weight(
            &mut encoder.down_3_sampler.conv,
            weights,
            "encoder.down_blocks.2.downsamplers.0.conv",
        )?;

        load_resnet_block(
            &mut encoder.down_4_0,
            weights,
            "encoder.down_blocks.3.resnets.0",
        )?;
        load_resnet_block(
            &mut encoder.down_4_1,
            weights,
            "encoder.down_blocks.3.resnets.1",
        )?;
        load_conv2d_weight(
            &mut encoder.down_4_sampler.conv,
            weights,
            "encoder.down_blocks.3.downsamplers.0.conv",
        )?;

        // Mid block
        load_resnet_block(
            &mut encoder.mid_block_1,
            weights,
            "encoder.mid_block.resnets.0",
        )?;
        load_attn_block(
            &mut encoder.mid_attn,
            weights,
            "encoder.mid_block.attentions.0",
        )?;
        load_resnet_block(
            &mut encoder.mid_block_2,
            weights,
            "encoder.mid_block.resnets.1",
        )?;

        load_group_norm_weight(&mut encoder.norm_out, weights, "encoder.conv_norm_out")?;
        load_conv2d_weight(&mut encoder.conv_out, weights, "encoder.conv_out")?;
    }

    // Decoder weights
    let decoder = &mut model.decoder;
    load_conv2d_weight(&mut decoder.conv_in, weights, "decoder.conv_in")?;

    // Mid block
    load_resnet_block(
        &mut decoder.mid_block_1,
        weights,
        "decoder.mid_block.resnets.0",
    )?;
    load_attn_block(
        &mut decoder.mid_attn,
        weights,
        "decoder.mid_block.attentions.0",
    )?;
    load_resnet_block(
        &mut decoder.mid_block_2,
        weights,
        "decoder.mid_block.resnets.1",
    )?;

    // Up blocks: blocks 0-2 have upsamplers, block 3 does not
    load_resnet_block(
        &mut decoder.up_1_0,
        weights,
        "decoder.up_blocks.0.resnets.0",
    )?;
    load_resnet_block(
        &mut decoder.up_1_1,
        weights,
        "decoder.up_blocks.0.resnets.1",
    )?;
    load_resnet_block(
        &mut decoder.up_1_2,
        weights,
        "decoder.up_blocks.0.resnets.2",
    )?;
    load_conv2d_weight(
        &mut decoder.up_1_sampler.conv,
        weights,
        "decoder.up_blocks.0.upsamplers.0.conv",
    )?;

    load_resnet_block(
        &mut decoder.up_2_0,
        weights,
        "decoder.up_blocks.1.resnets.0",
    )?;
    load_resnet_block(
        &mut decoder.up_2_1,
        weights,
        "decoder.up_blocks.1.resnets.1",
    )?;
    load_resnet_block(
        &mut decoder.up_2_2,
        weights,
        "decoder.up_blocks.1.resnets.2",
    )?;
    load_conv2d_weight(
        &mut decoder.up_2_sampler.conv,
        weights,
        "decoder.up_blocks.1.upsamplers.0.conv",
    )?;

    load_resnet_block(
        &mut decoder.up_3_0,
        weights,
        "decoder.up_blocks.2.resnets.0",
    )?;
    load_resnet_block(
        &mut decoder.up_3_1,
        weights,
        "decoder.up_blocks.2.resnets.1",
    )?;
    load_resnet_block(
        &mut decoder.up_3_2,
        weights,
        "decoder.up_blocks.2.resnets.2",
    )?;
    load_conv2d_weight(
        &mut decoder.up_3_sampler.conv,
        weights,
        "decoder.up_blocks.2.upsamplers.0.conv",
    )?;

    load_resnet_block(
        &mut decoder.up_4_0,
        weights,
        "decoder.up_blocks.3.resnets.0",
    )?;
    load_resnet_block(
        &mut decoder.up_4_1,
        weights,
        "decoder.up_blocks.3.resnets.1",
    )?;
    load_resnet_block(
        &mut decoder.up_4_2,
        weights,
        "decoder.up_blocks.3.resnets.2",
    )?;

    load_group_norm_weight(&mut decoder.norm_out, weights, "decoder.conv_norm_out")?;
    load_conv2d_weight(&mut decoder.conv_out, weights, "decoder.conv_out")?;

    Ok(())
}

pub fn load_flux_weights(
    model: &mut FluxDiT,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    for (name, weight) in weights {
        let name = if let Some(stripped) = name.strip_prefix("model.diffusion_model.") {
            stripped
        } else {
            name
        };

        if name.starts_with("double_blocks.") {
            let parts: Vec<&str> = name.split('.').collect();
            let idx = parts[1].parse::<usize>().map_err(|_| {
                LoadError::SafeTensors(format!("Invalid block index in key: {}", name))
            })?;
            if idx >= model.blocks.len() {
                continue;
            }
            let sub_path = parts[2..].join(".");

            match sub_path.as_str() {
                "img_mod.lin.weight" => {
                    model.blocks[idx].norm1_a.linear.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "img_mod.lin.bias" => {
                    model.blocks[idx].norm1_a.linear.bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }
                "txt_mod.lin.weight" => {
                    model.blocks[idx].norm1_b.linear.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "txt_mod.lin.bias" => {
                    model.blocks[idx].norm1_b.linear.bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }

                "img_attn.qkv.weight" => {
                    model.blocks[idx].attn.a_to_qkv.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "img_attn.qkv.bias" => {
                    model.blocks[idx].attn.a_to_qkv.bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }
                "txt_attn.qkv.weight" => {
                    model.blocks[idx].attn.b_to_qkv.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "txt_attn.qkv.bias" => {
                    model.blocks[idx].attn.b_to_qkv.bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }

                "img_attn.norm.query_norm.scale" => {
                    model.blocks[idx].attn.norm_q_a.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "img_attn.norm.key_norm.scale" => {
                    model.blocks[idx].attn.norm_k_a.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "txt_attn.norm.query_norm.scale" => {
                    model.blocks[idx].attn.norm_q_b.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "txt_attn.norm.key_norm.scale" => {
                    model.blocks[idx].attn.norm_k_b.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }

                "img_attn.proj.weight" => {
                    model.blocks[idx].attn.a_to_out.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "img_attn.proj.bias" => {
                    model.blocks[idx].attn.a_to_out.bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }
                "txt_attn.proj.weight" => {
                    model.blocks[idx].attn.b_to_out.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "txt_attn.proj.bias" => {
                    model.blocks[idx].attn.b_to_out.bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }

                "img_mlp.0.weight" => {
                    model.blocks[idx].ff_a[0].weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "img_mlp.0.bias" => {
                    model.blocks[idx].ff_a[0].bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }
                "img_mlp.2.weight" => {
                    model.blocks[idx].ff_a[1].weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "img_mlp.2.bias" => {
                    model.blocks[idx].ff_a[1].bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }

                "txt_mlp.0.weight" => {
                    model.blocks[idx].ff_b[0].weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "txt_mlp.0.bias" => {
                    model.blocks[idx].ff_b[0].bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }
                "txt_mlp.2.weight" => {
                    model.blocks[idx].ff_b[1].weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "txt_mlp.2.bias" => {
                    model.blocks[idx].ff_b[1].bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }
                _ => {}
            }
        } else if name.starts_with("single_blocks.") {
            let parts: Vec<&str> = name.split('.').collect();
            let idx = parts[1].parse::<usize>().map_err(|_| {
                LoadError::SafeTensors(format!("Invalid block index in key: {}", name))
            })?;
            if idx >= model.single_blocks.len() {
                continue;
            }
            let sub_path = parts[2..].join(".");

            match sub_path.as_str() {
                "modulation.lin.weight" => {
                    model.single_blocks[idx].norm.linear.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "modulation.lin.bias" => {
                    model.single_blocks[idx].norm.linear.bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }
                "linear1.weight" => {
                    model.single_blocks[idx].to_qkv_mlp.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "linear1.bias" => {
                    model.single_blocks[idx].to_qkv_mlp.bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }
                "linear2.weight" => {
                    model.single_blocks[idx].proj_out.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "linear2.bias" => {
                    model.single_blocks[idx].proj_out.bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }
                "norm.query_norm.scale" => {
                    model.single_blocks[idx].norm_q_a.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "norm.key_norm.scale" => {
                    model.single_blocks[idx].norm_k_a.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                _ => {}
            }
        } else {
            match name {
                "time_in.in_layer.weight" => {
                    model.time_embedder.linear_1.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "time_in.in_layer.bias" => {
                    model.time_embedder.linear_1.bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }
                "time_in.out_layer.weight" => {
                    model.time_embedder.linear_2.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "time_in.out_layer.bias" => {
                    model.time_embedder.linear_2.bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }

                "txt_in.weight" => {
                    model.context_embedder.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "txt_in.bias" => {
                    model.context_embedder.bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }

                "vector_in.in_layer.weight" => {
                    model.pooled_text_embedder[0].weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "vector_in.in_layer.bias" => {
                    model.pooled_text_embedder[0].bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }
                "vector_in.out_layer.weight" => {
                    model.pooled_text_embedder[1].weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "vector_in.out_layer.bias" => {
                    model.pooled_text_embedder[1].bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }

                "guidance_in.in_layer.weight" => {
                    if let Some(ref mut ge) = model.guidance_embedder {
                        ge.linear_1.weight =
                            pmetal_bridge::compat::module::Param::new(weight.clone());
                    }
                }
                "guidance_in.in_layer.bias" => {
                    if let Some(ref mut ge) = model.guidance_embedder {
                        ge.linear_1.bias =
                            pmetal_bridge::compat::module::Param::new(Some(weight.clone()));
                    }
                }
                "guidance_in.out_layer.weight" => {
                    if let Some(ref mut ge) = model.guidance_embedder {
                        ge.linear_2.weight =
                            pmetal_bridge::compat::module::Param::new(weight.clone());
                    }
                }
                "guidance_in.out_layer.bias" => {
                    if let Some(ref mut ge) = model.guidance_embedder {
                        ge.linear_2.bias =
                            pmetal_bridge::compat::module::Param::new(Some(weight.clone()));
                    }
                }

                "img_in.weight" => {
                    model.x_embedder.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "img_in.bias" => {
                    model.x_embedder.bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }

                "final_layer.adaLN_modulation.1.weight" => {
                    model.final_norm_out.linear.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "final_layer.adaLN_modulation.1.bias" => {
                    model.final_norm_out.linear.bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }
                "final_layer.linear.weight" => {
                    model.final_proj_out.weight =
                        pmetal_bridge::compat::module::Param::new(weight.clone())
                }
                "final_layer.linear.bias" => {
                    model.final_proj_out.bias =
                        pmetal_bridge::compat::module::Param::new(Some(weight.clone()))
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Weight index for sharded models.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeightIndex {
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    pub weight_map: HashMap<String, String>,
}

fn is_non_text_auxiliary_weight(key: &str) -> bool {
    key.starts_with("model.visual.")
        || key.starts_with("visual.")
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

/// Validate that a shard path does not escape the model directory (path traversal protection).
///
/// HuggingFace cache uses symlinks: snapshot files point to `../../blobs/`.
/// We validate that the canonical shard path stays within the HF repo root
/// (the common ancestor of both `snapshots/` and `blobs/`), not just the
/// snapshot directory itself.
fn validate_shard_path(
    model_dir: &Path,
    shard_file: &str,
) -> Result<std::path::PathBuf, LoadError> {
    // Reject path traversal in the shard filename itself
    if shard_file.contains("..") || shard_file.starts_with('/') {
        return Err(LoadError::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("Shard filename contains path traversal: {}", shard_file),
        )));
    }
    let shard_path = model_dir.join(shard_file);
    let canonical_dir = model_dir.canonicalize().map_err(LoadError::Io)?;
    let canonical_shard = shard_path.canonicalize().map_err(|e| {
        LoadError::Io(std::io::Error::new(
            e.kind(),
            format!(
                "Shard file not found: {} (in {})",
                shard_file,
                model_dir.display()
            ),
        ))
    })?;
    // First check: shard is directly inside model_dir (non-symlinked case)
    if canonical_shard.starts_with(&canonical_dir) {
        // Return original path to preserve .safetensors extension for mlx-rs
        return Ok(shard_path);
    }
    // Second check: HF cache layout — shard symlinks to ../../blobs/ within
    // the same repo directory (e.g. models--Org--Name/{snapshots,blobs}/)
    // Allow if both canonical paths share the same HF repo root.
    if let Some(repo_root) = find_hf_repo_root(&canonical_dir) {
        if canonical_shard.starts_with(&repo_root) {
            // Return original symlink path to preserve .safetensors extension
            return Ok(shard_path);
        }
    }
    Err(LoadError::Io(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        format!("Shard path escapes model directory: {:?}", shard_path),
    )))
}

/// Find the HuggingFace repo root directory for a given path.
///
/// HF cache layout: `~/.cache/huggingface/hub/models--Org--Name/snapshots/<hash>/`
/// The repo root is `models--Org--Name/` which contains both `snapshots/` and `blobs/`.
fn find_hf_repo_root(path: &Path) -> Option<std::path::PathBuf> {
    let mut current = Some(path);
    while let Some(p) = current {
        if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
            if name.starts_with("models--") || name.starts_with("datasets--") {
                return Some(p.to_path_buf());
            }
        }
        current = p.parent();
    }
    None
}

/// Load generic weights using safetensors.
pub fn load_generic_weights<M: ModuleParameters + ModuleParametersExt>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
) -> Result<(), LoadError> {
    load_generic_weights_renamed(model, model_dir, |_| None)
}

/// [`load_generic_weights`] with a hook that rewrites checkpoint keys into
/// pmetal parameter paths.
///
/// `rename` returns `Some(path)` for a key that needs rewriting and `None` to
/// pass it through. This exists for architectures whose parameter tree is
/// shaped differently from the checkpoint but not differently enough to justify
/// a bespoke loader — the alternative is `assign_loaded_weights` silently
/// dropping every renamed tensor, since it matches by exact name.
pub fn load_generic_weights_renamed<M: ModuleParameters + ModuleParametersExt>(
    model: &mut M,
    model_dir: impl AsRef<Path>,
    rename: impl Fn(&str) -> Option<String>,
) -> Result<(), LoadError> {
    let apply = |loaded: HashMap<String, Array>| -> HashMap<String, Array> {
        loaded
            .into_iter()
            .map(|(key, value)| match rename(&key) {
                Some(renamed) => (renamed, value),
                None => (key, value),
            })
            .collect()
    };

    let model_dir = model_dir.as_ref();
    let mut report = LoadReport::default();

    // ⚠️ A quantized checkpoint cannot be streamed a shard at a time. Unpacking
    // needs `{module}.weight` together with its `.scales` and `.biases`, which
    // are three separate tensors, and nothing in the format promises a shard
    // boundary won't fall between them. Assemble it whole, unpack, then assign.
    //
    // Without this every architecture reaching the model through
    // `load_generic_weights` — llama, qwen2, mistral, gemma, phi, cohere and
    // the rest of the `simple_load!` arms — handed the packed `uint32` payload
    // straight to the model. The first norm then reported a weight that did not
    // match the width of its input, several ops downstream of the real problem.
    if load_mlx_quantization_config(model_dir)?.is_some() {
        let loaded = apply(load_weights(model_dir)?);
        report += assign_loaded_weights(model, loaded);
        eval_loaded_parameters(model)?;
        report.log_summary(model_dir);
        return Ok(());
    }

    // One file is read whole either way; `load_weights` also unpacks any
    // per-tensor quantization it carries.
    let single_file = model_dir.join("model.safetensors");
    if single_file.exists() {
        let loaded = apply(load_weights(model_dir)?);
        report += assign_loaded_weights(model, loaded);
        eval_loaded_parameters(model)?;
        report.log_summary(model_dir);
        return Ok(());
    }
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Err(LoadError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No weights found",
        )));
    }
    let index_content = std::fs::read_to_string(&index_path).map_err(LoadError::Io)?;
    let index: WeightIndex = serde_json::from_str(&index_content)?;
    // Same reason as the MLX case above: a scale and its weight can sit in
    // different shards.
    if index_has_quant_sidecars(&index) {
        let loaded = apply(load_weights(model_dir)?);
        report += assign_loaded_weights(model, loaded);
        eval_loaded_parameters(model)?;
        report.log_summary(model_dir);
        return Ok(());
    }
    let shard_files: HashSet<&String> = index.weight_map.values().collect();
    for shard_file in shard_files {
        let shard_path = validate_shard_path(model_dir, shard_file)?;
        let loaded = apply(load_shard(&shard_path)?);
        report += assign_loaded_weights(model, loaded);
    }
    eval_loaded_parameters(model)?;
    report.log_summary(model_dir);
    Ok(())
}

/// Rewrite a DeepSeek checkpoint key into pmetal's parameter path.
///
/// Three renames, all inside a MoE layer's `mlp`:
///
/// * `mlp.gate.weight` — the router matrix. pmetal reaches it through
///   `DeepSeekMoEGate`, whose own `Linear` field is called `weight`, so the
///   flattened path double-nests to `mlp.weight.weight`.
/// * `mlp.shared_experts.*` — `DeepSeekMoE` merges the shared expert's
///   parameters into its own map with no prefix, so they sit directly at
///   `mlp.*`.
/// * `mlp.experts.N.{gate,up,down}_proj` — the shared `Expert` type spells a
///   SwiGLU expert `w1`/`w3`/`w2`.
///
/// Scoped to the `.mlp.` segment so a dense layer's own `mlp.gate_proj.weight`,
/// which already matches, is left alone. Every one of these was silently
/// dropped before, taking the entire mixture with it.
pub fn deepseek_param_name(key: &str) -> Option<String> {
    let idx = key.find(".mlp.")?;
    let (prefix, tail) = key.split_at(idx + ".mlp.".len());

    if tail == "gate.weight" {
        return Some(format!("{prefix}weight.weight"));
    }
    if let Some(rest) = tail.strip_prefix("shared_experts.") {
        return Some(format!("{prefix}{rest}"));
    }
    if let Some(rest) = tail.strip_prefix("experts.") {
        let (index, member) = rest.split_once('.')?;
        let renamed = match member {
            "gate_proj.weight" => "w1.weight",
            "up_proj.weight" => "w3.weight",
            "down_proj.weight" => "w2.weight",
            _ => return None,
        };
        return Some(format!("{prefix}experts.{index}.{renamed}"));
    }
    None
}

/// Assign in-memory, HuggingFace-named weights to a model's parameters and
/// eval them onto the GPU.
///
/// This is the in-memory dual of [`load_generic_weights`]: the safetensors path
/// memory-maps shards from disk, whereas the GGUF inference path dequantizes
/// tensors into a `HashMap` and assigns them here. Behaviour matches the
/// safetensors generic loader exactly — params whose names match are replaced,
/// unmatched map entries are ignored (e.g. `lm_head.weight` for tied models).
pub fn assign_weights<M: ModuleParameters + ModuleParametersExt>(
    model: &mut M,
    weights: HashMap<String, Array>,
) -> Result<(), LoadError> {
    assign_loaded_weights(model, weights);
    eval_loaded_parameters(model)?;
    Ok(())
}

pub(crate) fn load_weights_filtered<F>(
    model_dir: impl AsRef<Path>,
    mut keep_key: F,
) -> Result<HashMap<String, Array>, LoadError>
where
    F: FnMut(&str) -> bool,
{
    let model_dir = model_dir.as_ref();
    let quant_config = load_mlx_quantization_config(model_dir)?;
    let mut all_weights = HashMap::new();
    let single_file = model_dir.join("model.safetensors");
    if single_file.exists() {
        let mut weights = load_shard(&single_file)?;
        if let Some(config) = &quant_config {
            dequantize_mlx_quantized_weights(&mut weights, config);
        }
        dequantize_sidecar_weights(&mut weights)?;
        return Ok(weights
            .into_iter()
            .filter(|(key, _)| keep_key(key) && !is_quant_aux_key(key))
            .collect());
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Err(LoadError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No weights found",
        )));
    }

    let index_content = std::fs::read_to_string(&index_path).map_err(LoadError::Io)?;
    let index: WeightIndex = serde_json::from_str(&index_content)?;
    let wanted_keys: HashSet<String> = index
        .weight_map
        .keys()
        .filter(|key| {
            // An aux entry has to come along whenever its packed tensor does,
            // or the tensor arrives still packed. Which name that is depends on
            // the layout, so ask about both (see `packed_weight_key_for`).
            keep_key(key)
                || quant_config.is_some()
                    && quant_aux_base_key(key)
                        .is_some_and(|base| keep_key(base) || keep_key(&format!("{base}.weight")))
                || pmetal_bridge::native_loader::quant_sidecar_base(key)
                    .is_some_and(|base| keep_key(&format!("{base}.weight")))
        })
        .cloned()
        .collect();
    let shard_files: HashSet<&String> = index
        .weight_map
        .iter()
        .filter(|(key, _)| wanted_keys.contains(*key))
        .map(|(_, shard_file)| shard_file)
        .collect();

    for shard_file in shard_files {
        let shard_path = validate_shard_path(model_dir, shard_file)?;
        let shard_weights = load_shard(&shard_path)?;
        all_weights.extend(
            shard_weights
                .into_iter()
                .filter(|(key, _)| wanted_keys.contains(key)),
        );
    }

    if let Some(config) = &quant_config {
        dequantize_mlx_quantized_weights(&mut all_weights, config);
    }
    dequantize_sidecar_weights(&mut all_weights)?;

    Ok(all_weights
        .into_iter()
        .filter(|(key, _)| keep_key(key) && !is_quant_aux_key(key))
        .collect())
}

pub fn load_nemotron_weights(
    model: &mut NemotronHForCausalLM,
    model_dir: impl AsRef<Path>,
) -> Result<(), LoadError> {
    let weights = load_weights(model_dir)?;
    load_nemotron(model, &weights).map_err(|e| LoadError::SafeTensors(format!("{:?}", e)))
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
    let mut weights = load_weights_filtered(model_dir, |key| {
        should_keep_qwen3_next_raw_weight(key, options)
    })?;
    sanitize_weights(
        &mut weights,
        config,
        Qwen3NextSanitizeOptions {
            skip_routed_experts: options.skip_routed_experts,
        },
    )
    .map_err(|e| LoadError::SafeTensors(format!("{:?}", e)))?;

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
    if !unmatched.is_empty() {
        tracing::warn!(
            "Qwen3Next weight loading skipped {} unmatched weights (first 10): {:?}",
            unmatched.len(),
            &unmatched[..unmatched.len().min(10)]
        );
    }

    let missing: Vec<String> = expected_keys
        .difference(&loaded_param_keys)
        .filter(|key| !is_qwen3_next_allowed_missing_param(key, options))
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

/// Load all safetensor weights from a model directory.
pub fn load_weights(model_dir: impl AsRef<Path>) -> Result<HashMap<String, Array>, LoadError> {
    load_weights_filtered(model_dir, |_| true)
}

pub fn load_llama_weights(
    model: &mut LlamaForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = pmetal_bridge::compat::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight("model.embed_tokens.weight".into()));
    }
    for (i, layer) in model.model.layers.iter_mut().enumerate() {
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
        load_rms_norm_weight(
            &mut layer.input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm"),
        )?;
        load_rms_norm_weight(
            &mut layer.post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm"),
        )?;
    }
    load_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;
    if let Some(ref mut lm_head) = model.lm_head {
        if let Some(w) = weights.get("lm_head.weight") {
            lm_head.weight = pmetal_bridge::compat::module::Param::new(w.clone());
        }
    }
    Ok(())
}

pub fn load_mistral_weights(
    model: &mut MistralForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = pmetal_bridge::compat::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight("model.embed_tokens.weight".into()));
    }
    for (i, layer) in model.model.layers.iter_mut().enumerate() {
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
        load_rms_norm_weight(
            &mut layer.input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm"),
        )?;
        load_rms_norm_weight(
            &mut layer.post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm"),
        )?;
    }
    load_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;
    if let Some(ref mut lm_head) = model.lm_head {
        if let Some(w) = weights.get("lm_head.weight") {
            lm_head.weight = pmetal_bridge::compat::module::Param::new(w.clone());
        }
    }
    Ok(())
}

pub fn load_gemma_weights(
    model: &mut GemmaForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = pmetal_bridge::compat::module::Param::new(w.clone());
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

pub fn load_phi_weights(
    model: &mut PhiForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = pmetal_bridge::compat::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight("model.embed_tokens.weight".into()));
    }
    for (i, layer) in model.model.layers.iter_mut().enumerate() {
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
            &mut layer.mlp.gate_up_proj,
            weights,
            &format!("{prefix}.mlp.gate_up_proj"),
        )?;
        load_linear_weight(
            &mut layer.mlp.down_proj,
            weights,
            &format!("{prefix}.mlp.down_proj"),
        )?;
        load_phi_rms_norm_weight(
            &mut layer.input_layernorm,
            weights,
            &format!("{prefix}.input_layernorm"),
        )?;
        load_phi_rms_norm_weight(
            &mut layer.post_attention_layernorm,
            weights,
            &format!("{prefix}.post_attention_layernorm"),
        )?;
    }
    load_phi_rms_norm_weight(&mut model.model.norm, weights, "model.norm")?;
    // Absent on a tied checkpoint, where `lm_head` is `None` and the embedding
    // is reused instead.
    if let Some(lm_head) = model.lm_head.as_mut() {
        load_linear_weight(lm_head, weights, "lm_head")?;
    }
    Ok(())
}

fn load_linear_weight(
    linear: &mut pmetal_bridge::compat::nn::Linear,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        linear.weight = pmetal_bridge::compat::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    let bias_key = format!("{prefix}.bias");
    if let Some(b) = weights.get(&bias_key) {
        linear.bias = pmetal_bridge::compat::module::Param::new(Some(b.clone()));
    }
    Ok(())
}

fn load_rms_norm_weight(
    norm: &mut pmetal_bridge::compat::nn::RmsNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        norm.weight = pmetal_bridge::compat::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    Ok(())
}

#[allow(dead_code)] // Utility for architectures that use LayerNorm (e.g. GPT-style models)
fn load_layer_norm_weight(
    norm: &mut pmetal_bridge::compat::nn::LayerNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        norm.weight = pmetal_bridge::compat::module::Param::new(Some(w.clone()));
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    let bias_key = format!("{prefix}.bias");
    if let Some(b) = weights.get(&bias_key) {
        norm.bias = pmetal_bridge::compat::module::Param::new(Some(b.clone()));
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
        norm.weight = pmetal_bridge::compat::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    Ok(())
}

fn load_phi_rms_norm_weight(
    norm: &mut crate::architectures::phi::PhiRMSNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        norm.weight = pmetal_bridge::compat::module::Param::new(w.clone());
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    Ok(())
}

fn load_conv2d_weight(
    conv: &mut nn::Conv2d,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        // HF/PyTorch Conv2d weights are [O, I, H, W], MLX uses [O, H, W, I]
        let w = w.transpose_axes(&[0, 2, 3, 1]);
        conv.weight = pmetal_bridge::compat::module::Param::new(w);
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    let bias_key = format!("{prefix}.bias");
    if let Some(b) = weights.get(&bias_key) {
        conv.bias = pmetal_bridge::compat::module::Param::new(Some(b.clone()));
    }
    Ok(())
}

fn load_group_norm_weight(
    norm: &mut nn::GroupNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), LoadError> {
    let weight_key = format!("{prefix}.weight");
    if let Some(w) = weights.get(&weight_key) {
        norm.weight = pmetal_bridge::compat::module::Param::new(Some(w.clone()));
    } else {
        return Err(LoadError::MissingWeight(weight_key));
    }
    let bias_key = format!("{prefix}.bias");
    if let Some(b) = weights.get(&bias_key) {
        norm.bias = pmetal_bridge::compat::module::Param::new(Some(b.clone()));
    }
    Ok(())
}

/// Load weights for BERT/RoBERTa/DistilBERT models from a HuggingFace checkpoint.
///
/// HuggingFace BERT checkpoints use a different naming convention than PMetal's
/// internal parameter paths.  This function maps HF names to PMetal names before
/// loading, so standard `model.safetensors` files from the HF Hub work without
/// any prior conversion.
///
/// ## Name mapping
///
/// | HuggingFace key pattern               | PMetal parameter path                |
/// |---------------------------------------|--------------------------------------|
/// | `bert.embeddings.*`                   | `model.embeddings.*`                 |
/// | `bert.encoder.layer.{i}.*`            | `model.layers.{i}.*`                 |
/// | `attention.self.query`                | `attention.query`                    |
/// | `attention.self.key`                  | `attention.key`                      |
/// | `attention.self.value`                | `attention.value`                    |
/// | `attention.output.dense`              | `attention_output.dense`             |
/// | `intermediate.dense`                  | `intermediate.dense`                 |
/// | `output.dense` (FFN)                  | `output.dense`                       |
/// | `LayerNorm` (any position)            | `layer_norm`                         |
/// | `embeddings.position_ids`             | skipped (buffer, not a parameter)    |
/// | `pooler.*`                            | skipped (not part of BertForEmbedding)|
///
/// Both `bert.` prefixed keys and bare keys (some fine-tuned checkpoints strip
/// the top-level prefix) are handled.
///
/// This is BERT only, despite what this comment used to claim. A RoBERTa
/// checkpoint prefixes every key with `roberta.`, which nothing here strips, so
/// it matched zero parameters and failed with a message about the checkpoint
/// rather than about the gap. `ModelArchitecture::from_model_type` no longer
/// routes RoBERTa or DistilBERT here.
pub fn load_bert_weights(
    model: &mut BertForEmbedding,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    let mut params = model.flatten_params_mut();
    let mut matched: usize = 0;

    for (hf_name, weight) in weights {
        // Skip non-parameter buffers and the optional pooler head (not part of
        // BertForEmbedding which is embeddings-only).
        if hf_name.ends_with("position_ids")
            || hf_name.starts_with("pooler.")
            || hf_name.starts_with("bert.pooler.")
        {
            continue;
        }

        // Map the HF key to the PMetal parameter path.
        let pmetal_name = remap_bert_weight_name(hf_name);

        if let Some(param) = params.get_mut(&*pmetal_name) {
            **param = weight.clone();
            matched += 1;
        }
        // Silently skip keys that don't have a corresponding PMetal parameter
        // (e.g. cls.predictions.*, cls.seq_relationship.* from MLM/NSP heads).
    }

    if matched == 0 {
        return Err(LoadError::SafeTensors(
            "BERT weight loading: no parameters matched. \
             Verify that the checkpoint is a BERT/RoBERTa model."
                .into(),
        ));
    }

    tracing::info!("BERT weight loading: {matched} parameters loaded successfully");
    Ok(())
}

/// Map a single HuggingFace BERT weight name to the corresponding PMetal
/// parameter path.
///
/// Called once per key in the safetensors file; cheap string manipulation only.
fn remap_bert_weight_name(hf_name: &str) -> String {
    // 1. Strip the top-level `bert.` prefix emitted by the standard HF BERT
    //    implementation (not present in some fine-tuned variants).
    let name = hf_name.strip_prefix("bert.").unwrap_or(hf_name);

    // 2. Remap `encoder.layer.{i}` → `layers.{i}` (drop "encoder." wrapper).
    let name = if let Some(rest) = name.strip_prefix("encoder.layer.") {
        // rest = "{i}.{...}"  e.g. "0.attention.self.query.weight"
        format!("layers.{rest}")
    } else {
        name.to_string()
    };

    // 3. Remap attention sub-structure:
    //    `attention.self.query`  → `attention.query`
    //    `attention.self.key`    → `attention.key`
    //    `attention.self.value`  → `attention.value`
    //    `attention.output.dense` → `attention_output.dense`
    let name = name
        .replace("attention.self.query", "attention.query")
        .replace("attention.self.key", "attention.key")
        .replace("attention.self.value", "attention.value")
        .replace("attention.output.dense", "attention_output.dense")
        .replace("attention.output.LayerNorm", "attention_output.layer_norm");

    // 4. Remap LayerNorm everywhere: `LayerNorm` → `layer_norm`.
    //    This covers `embeddings.LayerNorm`, `output.LayerNorm`, etc.
    let name = name.replace("LayerNorm", "layer_norm");

    // 5. Prepend `model.` to match `BertForEmbedding { model: BertModel }`.
    format!("model.{name}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn max_abs_diff(a: &Array, b: &Array) -> f32 {
        let n = a.size();
        let f32_dtype = pmetal_bridge::dtype::F32;
        let mut a = a.as_dtype(f32_dtype);
        let mut b = b.as_dtype(f32_dtype);
        let va = a.to_f32_vec(n).expect("read lhs");
        let vb = b.to_f32_vec(n).expect("read rhs");
        va.iter()
            .zip(vb.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max)
    }

    /// ⚠️ A ModelOpt checkpoint used to load with its sidecars skipped as
    /// "unmatched" and its packed bytes assigned as dense weights, while
    /// reporting every weight matched. Training and eval both read this path.
    #[test]
    fn modelopt_checkpoints_unpack_to_dense() {
        use pmetal_bridge::QuantizedMode;
        let dir = tempdir().unwrap();
        let (out, in_dim) = (4i32, 64i32);
        let values: Vec<f32> = (0..out * in_dim)
            .map(|i| ((i % 13) as f32 - 6.0) / 6.0)
            .collect();
        let w = Array::from_f32_slice(&values, &[out, in_dim]);
        let (packed, scales) = w.quantize_weights_mode(16, 4, QuantizedMode::Nvfp4);
        let fp8 = w.to_fp8();
        let scalar = |v: f32| Array::from_f32_slice(&[v], &[]);

        let down = "model.layers.0.mlp.down_proj";
        let q = "model.layers.0.self_attn.q_proj";
        let mut weights = HashMap::new();
        weights.insert(
            format!("{down}.weight"),
            packed.view(pmetal_bridge::dtype::U8),
        );
        weights.insert(format!("{down}.weight_scale"), scales.clone());
        weights.insert(format!("{down}.weight_scale_2"), scalar(0.37));
        weights.insert(format!("{down}.input_scale"), scalar(0.2));
        weights.insert(format!("{q}.weight"), fp8.clone());
        weights.insert(format!("{q}.weight_scale"), scalar(0.5));
        weights.insert(format!("{q}.input_scale"), scalar(0.2));
        weights.insert(
            "model.norm.weight".to_string(),
            Array::ones(&[in_dim], pmetal_bridge::dtype::F32),
        );
        write_safetensors(&dir.path().join("model.safetensors"), &weights).unwrap();

        let loaded = load_weights(dir.path()).unwrap();

        let sidecars: Vec<&String> = loaded
            .keys()
            .filter(|k| pmetal_bridge::native_loader::quant_sidecar_base(k).is_some())
            .collect();
        assert!(sidecars.is_empty(), "sidecars survived: {sidecars:?}");

        let want_down = packed
            .dequantize_mode(&scales, None, 16, 4, QuantizedMode::Nvfp4)
            .as_dtype(pmetal_bridge::dtype::F32)
            .multiply(&scalar(0.37));
        let diff = max_abs_diff(&loaded[&format!("{down}.weight")], &want_down);
        assert!(diff < 1e-6, "nvfp4 unpacked wrongly: max|diff| = {diff}");

        let want_q = fp8
            .from_fp8(pmetal_bridge::dtype::F32)
            .multiply(&scalar(0.5));
        let diff = max_abs_diff(&loaded[&format!("{q}.weight")], &want_q);
        assert!(diff < 1e-6, "fp8 unpacked wrongly: max|diff| = {diff}");
    }

    /// A weight scale with nothing to apply it to has to stop the load.
    #[test]
    fn an_orphaned_weight_scale_is_an_error() {
        let dir = tempdir().unwrap();
        let mut weights = HashMap::new();
        weights.insert(
            "model.layers.0.mlp.down_proj.weight_scale_inv".to_string(),
            Array::from_f32_slice(&[1.0], &[1, 1]),
        );
        weights.insert(
            "model.norm.weight".to_string(),
            Array::ones(&[4], pmetal_bridge::dtype::F32),
        );
        write_safetensors(&dir.path().join("model.safetensors"), &weights).unwrap();

        let err = load_weights(dir.path()).unwrap_err().to_string();
        assert!(err.contains("weight_scale_inv"), "unexpected error: {err}");
    }

    fn write_safetensors(
        path: &Path,
        weights: &HashMap<String, Array>,
    ) -> Result<(), pmetal_bridge::compat::IoError> {
        let entries: Vec<_> = weights
            .iter()
            .map(|(key, value)| (key.as_str(), value))
            .collect();
        Array::save_safetensors(path.to_string_lossy().as_ref(), &entries);
        Ok(())
    }

    #[test]
    fn qwen3_next_raw_weight_filter_skips_auxiliary_and_routed_expert_keys() {
        let options = Qwen3NextLoadOptions {
            skip_routed_experts: true,
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

    #[test]
    fn load_weights_filtered_reads_only_needed_shards() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();
        let text_shard = model_dir.join("text.safetensors");
        let broken_shard = model_dir.join("broken.safetensors");

        let mut text_weights = HashMap::new();
        text_weights.insert(
            "model.embed_tokens.weight".to_string(),
            Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]),
        );
        write_safetensors(&text_shard, &text_weights).unwrap();
        std::fs::write(&broken_shard, b"not safetensors").unwrap();

        let index = serde_json::json!({
            "metadata": {},
            "weight_map": {
                "model.embed_tokens.weight": "text.safetensors",
                "model.language_model.layers.0.mlp.experts.gate_up_proj": "broken.safetensors"
            }
        });
        std::fs::write(
            model_dir.join("model.safetensors.index.json"),
            serde_json::to_string_pretty(&index).unwrap(),
        )
        .unwrap();

        let loaded =
            load_weights_filtered(model_dir, |key| !key.contains(".mlp.experts.")).unwrap();

        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains_key("model.embed_tokens.weight"));
    }

    #[test]
    fn load_mlx_quantization_config_accepts_alias_and_overrides() {
        let temp = tempdir().unwrap();
        std::fs::write(
            temp.path().join("config.json"),
            r#"{
                "quantization": {
                    "bits": 4,
                    "group_size": 64,
                    "per_tensor_overrides": {
                        "model.layers.0.self_attn.q_proj.weight": 8
                    }
                }
            }"#,
        )
        .unwrap();

        let config = load_mlx_quantization_config(temp.path())
            .unwrap()
            .expect("quantization config");
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 64);
        assert_eq!(
            config.for_tensor("model.layers.0.self_attn.q_proj.weight"),
            TensorQuant {
                bits: 8,
                group_size: 64
            }
        );
    }

    /// The layout MLX itself writes, and the one mlx-community QAT releases
    /// ship: per-module entries sitting beside `bits` and `group_size`, keyed
    /// by module path rather than tensor key.
    #[test]
    fn load_mlx_quantization_config_reads_mlx_per_module_overrides() {
        let temp = tempdir().unwrap();
        std::fs::write(
            temp.path().join("config.json"),
            r#"{
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                    "language_model.model.layers.0.mlp.gate_proj": {
                        "bits": 8,
                        "group_size": 32
                    },
                    "language_model.model.layers.0.self_attn.q_proj": false
                }
            }"#,
        )
        .unwrap();

        let config = load_mlx_quantization_config(temp.path())
            .unwrap()
            .expect("quantization config");

        assert_eq!(
            config.for_tensor("language_model.model.layers.0.mlp.gate_proj.weight"),
            TensorQuant {
                bits: 8,
                group_size: 32
            },
            "per-module override was not applied, so this tensor unpacks at the \
             model-wide width"
        );
        assert_eq!(
            config.for_tensor("language_model.model.layers.0.mlp.down_proj.weight"),
            TensorQuant {
                bits: 4,
                group_size: 64
            },
            "a module with no override should take the model-wide setting"
        );
        // `false` means the module was left unquantized. It carries no
        // scales/biases, so it never reaches dequantization.
        assert!(
            !config
                .overrides
                .contains_key("language_model.model.layers.0.self_attn.q_proj.weight"),
            "an unquantized module was recorded as an override"
        );
    }

    /// A per-module entry that omits `group_size` inherits the model-wide one,
    /// which is how MLX's `to_quantized(**params)` fills its defaults.
    #[test]
    fn a_partial_per_module_override_inherits_the_model_wide_group_size() {
        let temp = tempdir().unwrap();
        std::fs::write(
            temp.path().join("config.json"),
            r#"{
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "model.layers.3.mlp.up_proj": {"bits": 8}
                }
            }"#,
        )
        .unwrap();

        let config = load_mlx_quantization_config(temp.path())
            .unwrap()
            .expect("quantization config");
        assert_eq!(
            config.for_tensor("model.layers.3.mlp.up_proj.weight"),
            TensorQuant {
                bits: 8,
                group_size: 64
            }
        );
    }

    #[test]
    fn load_weights_dequantizes_mlx_affine_quantized_tensors() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();
        let weight = Array::from_slice(&[1.0f32; 128], &[2, 64]);
        let (packed, scales, biases) = weight.quantize_weights(64, 4);

        let mut weights = HashMap::new();
        weights.insert("linear.weight".to_string(), packed);
        weights.insert("linear.scales".to_string(), scales);
        weights.insert("linear.biases".to_string(), biases);
        write_safetensors(&model_dir.join("model.safetensors"), &weights).unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();

        let loaded = load_weights(model_dir).unwrap();
        assert_eq!(loaded.len(), 1);
        let restored = loaded.get("linear.weight").unwrap();
        assert_eq!(restored.shape(), &[2, 64]);
        assert!(!loaded.contains_key("linear.scales"));
        assert!(!loaded.contains_key("linear.biases"));
    }

    /// pmetal's own quantizer writes the aux tensors under the full weight name
    /// rather than beside it, so `quantize_and_save_mlx` output has to keep
    /// loading too.
    #[test]
    fn load_weights_dequantizes_pmetals_own_aux_key_layout() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();
        let weight = Array::from_slice(&[1.0f32; 128], &[2, 64]);
        let (packed, scales, biases) = weight.quantize_weights(64, 4);

        let mut weights = HashMap::new();
        weights.insert("linear.weight".to_string(), packed);
        weights.insert("linear.weight.scales".to_string(), scales);
        weights.insert("linear.weight.biases".to_string(), biases);
        write_safetensors(&model_dir.join("model.safetensors"), &weights).unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();

        let loaded = load_weights(model_dir).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.get("linear.weight").unwrap().shape(), &[2, 64]);
    }

    /// A mixed-precision checkpoint, which is what mlx-community QAT releases
    /// ship: most tensors at the model-wide width, some at another.
    ///
    /// Unpacking the 8-bit tensor at the model-wide 4 bits reads the right
    /// number of bytes and produces the wrong shape, so the load "succeeds" and
    /// the failure surfaces later as a norm complaining about its input size.
    #[test]
    fn load_weights_honours_per_module_quantization_widths() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();

        let wide = Array::from_slice(&[1.0f32; 128], &[2, 64]);
        let (wide_packed, wide_scales, wide_biases) = wide.quantize_weights(64, 8);
        let narrow = Array::from_slice(&[1.0f32; 128], &[2, 64]);
        let (narrow_packed, narrow_scales, narrow_biases) = narrow.quantize_weights(64, 4);

        let mut weights = HashMap::new();
        weights.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            wide_packed,
        );
        weights.insert(
            "model.layers.0.mlp.gate_proj.scales".to_string(),
            wide_scales,
        );
        weights.insert(
            "model.layers.0.mlp.gate_proj.biases".to_string(),
            wide_biases,
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            narrow_packed,
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.scales".to_string(),
            narrow_scales,
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.biases".to_string(),
            narrow_biases,
        );
        write_safetensors(&model_dir.join("model.safetensors"), &weights).unwrap();

        std::fs::write(
            model_dir.join("config.json"),
            r#"{
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                    "model.layers.0.mlp.gate_proj": {"bits": 8, "group_size": 64}
                }
            }"#,
        )
        .unwrap();

        let loaded = load_weights(model_dir).unwrap();
        assert_eq!(
            loaded
                .get("model.layers.0.mlp.gate_proj.weight")
                .unwrap()
                .shape(),
            &[2, 64],
            "the 8-bit tensor unpacked at the model-wide 4 bits"
        );
        assert_eq!(
            loaded
                .get("model.layers.0.self_attn.q_proj.weight")
                .unwrap()
                .shape(),
            &[2, 64],
            "the tensor with no override should still take the model-wide width"
        );
    }

    /// A checkpoint key that matches no parameter has to come back in the
    /// report. Silently dropping it is how a model ends up running on random
    /// init while looking like it loaded fine.
    #[test]
    fn unmatched_checkpoint_keys_are_reported() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();

        let mut weights = HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]),
        );
        weights.insert(
            "model.not_a_parameter.weight".to_string(),
            Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]),
        );
        weights.insert(
            "layers.0.self_attn.q_proj.weight".to_string(),
            Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]),
        );
        write_safetensors(&model_dir.join("model.safetensors"), &weights).unwrap();

        let mut model = crate::architectures::Qwen3ForCausalLM::new(Default::default()).unwrap();
        let mut report = LoadReport::default();
        report += assign_loaded_weights(
            &mut model,
            load_shard(&model_dir.join("model.safetensors")).unwrap(),
        );

        assert_eq!(
            report.loaded, 1,
            "only embed_tokens matches a real parameter"
        );
        let skipped: HashSet<&str> = report.skipped.iter().map(String::as_str).collect();
        assert_eq!(
            skipped,
            HashSet::from([
                "model.not_a_parameter.weight",
                "layers.0.self_attn.q_proj.weight"
            ]),
            "both unmatched keys must be reported, including the un-prefixed layout"
        );
    }

    /// Build a `flatten_params_mut`-shaped map without a real module, so the
    /// namespace probe can be exercised on layouts no architecture ships.
    fn param_map<'a>(names: &[&str], slots: &'a mut [Array]) -> HashMap<String, &'a mut Array> {
        assert_eq!(names.len(), slots.len());
        names
            .iter()
            .map(|name| (*name).to_string())
            .zip(slots.iter_mut())
            .collect()
    }

    fn checkpoint(keys: &[&str]) -> HashMap<String, Array> {
        keys.iter()
            .map(|key| ((*key).to_string(), Array::from_slice(&[0.0f32], &[1])))
            .collect()
    }

    /// `Qwen/Qwen3-Embedding-8B` publishes the text tower with no namespace at
    /// all, so every key missed and the model answered from random init
    /// (issue #19).
    #[test]
    fn unprefixed_checkpoint_resolves_against_the_model_namespace() {
        let mut slots = [
            Array::from_slice(&[0.0f32], &[1]),
            Array::from_slice(&[0.0f32], &[1]),
            Array::from_slice(&[0.0f32], &[1]),
            Array::from_slice(&[0.0f32], &[1]),
        ];
        let params = param_map(
            &[
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.norm.weight",
                "lm_head.weight",
            ],
            &mut slots,
        );
        let loaded = checkpoint(&[
            "embed_tokens.weight",
            "layers.0.self_attn.q_proj.weight",
            "norm.weight",
        ]);

        assert_eq!(
            detect_namespace_prefix(&params, &loaded).as_deref(),
            Some("model.")
        );
    }

    /// The checkpoint above plus an `lm_head.weight` that *does* line up. An
    /// early-out on "some key matched" would take the identity mapping here and
    /// drop every layer.
    #[test]
    fn one_already_matching_key_does_not_veto_the_shift() {
        let mut slots = [
            Array::from_slice(&[0.0f32], &[1]),
            Array::from_slice(&[0.0f32], &[1]),
            Array::from_slice(&[0.0f32], &[1]),
            Array::from_slice(&[0.0f32], &[1]),
        ];
        let params = param_map(
            &[
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.norm.weight",
                "lm_head.weight",
            ],
            &mut slots,
        );
        let loaded = checkpoint(&[
            "embed_tokens.weight",
            "layers.0.self_attn.q_proj.weight",
            "norm.weight",
            "lm_head.weight",
        ]);

        assert_eq!(
            detect_namespace_prefix(&params, &loaded).as_deref(),
            Some("model."),
            "three keys under `model.` outweigh the one that already matched"
        );
    }

    #[test]
    fn a_checkpoint_that_already_lines_up_is_left_alone() {
        let mut slots = [
            Array::from_slice(&[0.0f32], &[1]),
            Array::from_slice(&[0.0f32], &[1]),
        ];
        let params = param_map(
            &["model.embed_tokens.weight", "model.norm.weight"],
            &mut slots,
        );
        let loaded = checkpoint(&["model.embed_tokens.weight", "model.norm.weight"]);

        assert_eq!(detect_namespace_prefix(&params, &loaded), None);
    }

    /// Two towers with identically-named sublayers: prefixing onto either is a
    /// coin flip, so the load has to stay put and report the misses.
    #[test]
    fn an_ambiguous_shift_is_refused() {
        let mut slots = [
            Array::from_slice(&[0.0f32], &[1]),
            Array::from_slice(&[0.0f32], &[1]),
        ];
        let params = param_map(
            &[
                "text_model.layers.0.self_attn.q_proj.weight",
                "vision_model.layers.0.self_attn.q_proj.weight",
            ],
            &mut slots,
        );
        let loaded = checkpoint(&["layers.0.self_attn.q_proj.weight"]);

        assert_eq!(detect_namespace_prefix(&params, &loaded), None);
    }

    #[test]
    fn a_layout_no_prefix_explains_is_left_to_the_report() {
        let mut slots = [Array::from_slice(&[0.0f32], &[1])];
        let params = param_map(&["model.embed_tokens.weight"], &mut slots);
        let loaded = checkpoint(&["bert.encoder.layer.0.attention.self.query.weight"]);

        assert_eq!(detect_namespace_prefix(&params, &loaded), None);
    }

    /// End to end on the real `Qwen3ForCausalLM` parameter tree: before the
    /// shift was detected this loaded nothing at all.
    #[test]
    fn unprefixed_qwen3_checkpoint_loads_into_the_model() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();

        let mut weights = HashMap::new();
        for key in [
            "embed_tokens.weight",
            "norm.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.mlp.gate_proj.weight",
        ] {
            weights.insert(
                key.to_string(),
                Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]),
            );
        }
        write_safetensors(&model_dir.join("model.safetensors"), &weights).unwrap();

        let mut model = crate::architectures::Qwen3ForCausalLM::new(Default::default()).unwrap();
        let report = assign_loaded_weights(
            &mut model,
            load_shard(&model_dir.join("model.safetensors")).unwrap(),
        );

        assert_eq!(report.loaded, weights.len());
        assert!(report.skipped.is_empty(), "skipped: {:?}", report.skipped);
    }

    /// ⚠️ The `simple_load!` arms — llama, qwen2, mistral, gemma, phi, cohere
    /// and most of the rest — reach the model through `load_generic_weights`,
    /// which read shards straight off disk and never consulted the
    /// quantization config. Every `mlx-community` 4-bit checkpoint therefore
    /// arrived still packed, and the model built on `uint32` payloads.
    ///
    /// Asserted on the *parameter*, not on a returned map, because the gap was
    /// between reading the shard and assigning it.
    #[test]
    fn load_generic_weights_unpacks_a_quantized_checkpoint() {
        let temp = tempdir().unwrap();
        let model_dir = temp.path();

        let mut model = crate::architectures::Qwen3ForCausalLM::new(Default::default()).unwrap();
        let hidden = model.model.embed_tokens.weight.shape()[1];
        let dense = Array::from_slice(&vec![0.5f32; (hidden * 2) as usize], &[2, hidden]);
        let (packed, scales, biases) = dense.quantize_weights(64, 4);

        let mut weights = HashMap::new();
        weights.insert("model.layers.0.self_attn.q_proj.weight".to_string(), packed);
        weights.insert("model.layers.0.self_attn.q_proj.scales".to_string(), scales);
        weights.insert("model.layers.0.self_attn.q_proj.biases".to_string(), biases);
        write_safetensors(&model_dir.join("model.safetensors"), &weights).unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();

        load_generic_weights(&mut model, model_dir).unwrap();

        assert_eq!(
            model.model.layers[0].self_attn.q_proj.weight.shape(),
            &[2, hidden],
            "q_proj kept the packed uint32 width instead of being dequantized"
        );
    }
}
