//! Per-architecture checkpoint loaders that predate the generic loader: the
//! name-by-name assignments for the dense decoders, the Qwen3-Next and
//! NemotronH wrappers and the DeepSeek key rename.

use pmetal_runtime::compat::{Array, ModuleParametersExt};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::{LoadError, load_weights, load_weights_filtered};
use crate::architectures::bert::BertForEmbedding;
use crate::architectures::gemma::GemmaForCausalLM;
use crate::architectures::llama::LlamaForCausalLM;
use crate::architectures::mistral::MistralForCausalLM;
use crate::architectures::nemotron_h::{
    NemotronHForCausalLM, load_nemotron_weights as load_nemotron,
};
use crate::architectures::phi::{PhiConfig, PhiForCausalLM};
use crate::architectures::qwen3_next::{
    Qwen3NextConfig, Qwen3NextForCausalLM, Qwen3NextSanitizeOptions, sanitize_weights,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Qwen3NextLoadOptions {
    pub skip_routed_experts: bool,
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

pub fn load_llama_weights(
    model: &mut LlamaForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = pmetal_runtime::compat::module::Param::new(w.clone());
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
            lm_head.weight = pmetal_runtime::compat::module::Param::new(w.clone());
        }
    }
    Ok(())
}

pub fn load_mistral_weights(
    model: &mut MistralForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), LoadError> {
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.model.embed_tokens.weight = pmetal_runtime::compat::module::Param::new(w.clone());
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
            lm_head.weight = pmetal_runtime::compat::module::Param::new(w.clone());
        }
    }
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
        model.model.embed_tokens.weight = pmetal_runtime::compat::module::Param::new(w.clone());
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

fn load_rms_norm_weight(
    norm: &mut pmetal_runtime::compat::nn::RmsNorm,
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

fn load_phi_rms_norm_weight(
    norm: &mut crate::architectures::phi::PhiRMSNorm,
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
}
