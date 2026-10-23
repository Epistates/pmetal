//! Which GGUF architecture a Hugging Face model becomes, and how its tensors
//! are named and laid out there.
//!
//! A GGUF file is read by engines that know each architecture by name, so a
//! tensor written under the wrong name, or in the wrong layout, is not an
//! error anywhere: the engine loads it and computes with it. That is why only
//! the architectures whose mapping has been checked against what the reference
//! converter writes are exported at all, and why a tensor with no name in the
//! file is an error rather than a pass-through.

use pmetal_models::weight_format::WeightLoader;
use serde_json::Value;

/// How an architecture's tensors differ from the plain Hugging Face layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    /// `llama` (Llama, Mistral, Mixtral): the rows of every query and key
    /// projection are reordered for the interleaved rotary convention the
    /// engines use for this architecture.
    Llama,
    /// Qwen: stored as published.
    Qwen,
    /// Gemma 1: stored as published, except that every norm gains 1 (the
    /// engines multiply by the weight where Gemma multiplies by one plus it).
    Gemma,
    /// Gemma 2 and Gemma 3: Gemma's norms, and four norms per block with their
    /// own names.
    GemmaPrePostNorm,
    /// Gemma 4: the four norms, stored as they are (its norm multiplies by the
    /// weight), the language model nested in a multimodal checkpoint, and no
    /// value projection in the full-attention layers.
    Gemma4,
}

/// The multimodal Gemma 3 (4B, 12B, 27B): the Gemma 3 language model nested in
/// a wrapper with a vision tower, the way Gemma 4's is.
const GEMMA3_MULTIMODAL: &str = "Gemma3ForConditionalGeneration";

/// The Hugging Face architectures that are exported, with the GGUF
/// architecture each becomes.
const ARCHITECTURES: &[(&str, &str, Family)] = &[
    ("LlamaForCausalLM", "llama", Family::Llama),
    ("LlamaModel", "llama", Family::Llama),
    ("MistralForCausalLM", "llama", Family::Llama),
    ("MixtralForCausalLM", "llama", Family::Llama),
    ("Qwen2ForCausalLM", "qwen2", Family::Qwen),
    ("Qwen2Model", "qwen2", Family::Qwen),
    ("Qwen2MoeForCausalLM", "qwen2moe", Family::Qwen),
    ("Qwen3ForCausalLM", "qwen3", Family::Qwen),
    ("Qwen3Model", "qwen3", Family::Qwen),
    ("Qwen3MoeForCausalLM", "qwen3moe", Family::Qwen),
    ("GemmaForCausalLM", "gemma", Family::Gemma),
    ("Gemma2ForCausalLM", "gemma2", Family::GemmaPrePostNorm),
    ("Gemma3ForCausalLM", "gemma3", Family::GemmaPrePostNorm),
    (GEMMA3_MULTIMODAL, "gemma3", Family::GemmaPrePostNorm),
    ("Gemma4ForCausalLM", "gemma4", Family::Gemma4),
    ("Gemma4ForConditionalGeneration", "gemma4", Family::Gemma4),
    (
        "Gemma4UnifiedForConditionalGeneration",
        "gemma4",
        Family::Gemma4,
    ),
];

/// The architecture a model is exported as, with what its tensors need.
#[derive(Debug, Clone)]
pub(crate) struct ExportArch {
    /// `general.architecture`.
    pub name: &'static str,
    family: Family,
    /// A multimodal Gemma 3: the language model's config is nested under
    /// `text_config` (stating only what differs from `transformers`'
    /// defaults), and its tensors under `language_model`.
    gemma3_multimodal: bool,
    num_heads: usize,
    num_kv_heads: usize,
    /// The head is the embedding: no `output.weight` is written.
    tie_embeddings: bool,
    /// The file states its own RoPE scaling as per-frequency divisors in a
    /// tensor, not metadata: Llama 3.x's banded scaling.
    rope_frequency_factors: Option<Vec<f32>>,
}

impl ExportArch {
    /// The architecture of the model whose `config.json` is `config`.
    ///
    /// # Errors
    ///
    /// There is no config, or its architecture is not one that is exported.
    pub(crate) fn detect(config: Option<&Value>) -> anyhow::Result<Self> {
        let config = config.ok_or_else(|| {
            anyhow::anyhow!("config.json not found: the architecture of the model is unknown")
        })?;
        let declared = config
            .get("architectures")
            .and_then(Value::as_array)
            .and_then(|archs| archs.first())
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("config.json names no architecture"))?;
        let &(_, name, family) = ARCHITECTURES
            .iter()
            .find(|(hf, _, _)| *hf == declared)
            .ok_or_else(|| {
                let supported: Vec<&str> = ARCHITECTURES.iter().map(|(hf, _, _)| *hf).collect();
                anyhow::anyhow!(
                    "GGUF export does not support the architecture {declared} (a GGUF file \
                     names its tensors and lays them out per architecture, and none is known \
                     for this one). Supported: {}",
                    supported.join(", ")
                )
            })?;

        let gemma3_multimodal = declared == GEMMA3_MULTIMODAL;
        let text = if family == Family::Gemma4 {
            let text = text_config(config);
            reject_unsupported_gemma4(&text)?;
            text
        } else if gemma3_multimodal {
            gemma3_text_config(config)
        } else {
            config.clone()
        };
        let count = |key: &str| text.get(key).and_then(Value::as_u64).map(|v| v as usize);
        let num_heads = count("num_attention_heads").unwrap_or(1).max(1);
        let num_kv_heads = count("num_key_value_heads").unwrap_or(num_heads).max(1);
        let tie_embeddings = text
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        Ok(Self {
            name,
            family,
            gemma3_multimodal,
            num_heads,
            num_kv_heads,
            tie_embeddings,
            rope_frequency_factors: if family == Family::Gemma4 {
                gemma4_frequency_factors(&text)
            } else {
                llama3_frequency_factors(config)
            },
        })
    }

    /// Whether this is a Gemma 4 file.
    pub(crate) fn is_gemma4(&self) -> bool {
        self.family == Family::Gemma4
    }

    /// The configuration of the model proper: the language model's, nested in
    /// the multimodal wrapper a Gemma 3 or Gemma 4 checkpoint has, and the
    /// config itself for every other family.
    pub(crate) fn model_config(&self, config: Option<&Value>) -> Option<Value> {
        config.map(|config| {
            if self.family == Family::Gemma4 {
                text_config(config)
            } else if self.gemma3_multimodal {
                gemma3_text_config(config)
            } else {
                config.clone()
            }
        })
    }

    /// The configuration the tokenizer's metadata is written from: the
    /// checkpoint's own, with the vocabulary size of the language model where
    /// the checkpoint's config nests it. The file's vocabulary is as long as the
    /// embedding, and a multimodal Gemma's has rows its tokenizer has no token
    /// for.
    pub(crate) fn tokenizer_config(&self, config: Option<&Value>) -> Option<Value> {
        let mut config = config?.clone();
        let vocab_size = self
            .model_config(Some(&config))
            .and_then(|model| model.get("vocab_size").cloned());
        if let (Some(object), Some(vocab_size)) = (config.as_object_mut(), vocab_size) {
            object.entry("vocab_size").or_insert(vocab_size);
        }
        Some(config)
    }

    /// The name `hf_name` has in the file, or `None` for a tensor the file
    /// does not hold.
    ///
    /// # Errors
    ///
    /// The tensor has no name in this architecture.
    pub(crate) fn tensor_name(&self, hf_name: &str) -> anyhow::Result<Option<String>> {
        // A buffer the model computes at load, not a weight.
        if hf_name.ends_with("rotary_emb.inv_freq") {
            return Ok(None);
        }
        // The head of a model that ties it is the embedding.
        if self.tie_embeddings && hf_name == "lm_head.weight" {
            return Ok(None);
        }
        let name = match self.family {
            Family::GemmaPrePostNorm if self.gemma3_multimodal => {
                // The language model only: the vision tower and the projector
                // have no place in a text GGUF.
                let Some(text_name) = pmetal_models::dispatcher::gemma_text_key(hf_name) else {
                    return Ok(None);
                };
                if self.tie_embeddings && text_name == "lm_head.weight" {
                    return Ok(None);
                }
                WeightLoader::hf_to_gguf_name_gemma(&text_name)
            }
            Family::GemmaPrePostNorm => WeightLoader::hf_to_gguf_name_gemma(hf_name),
            Family::Gemma4 => {
                // The language model only: a multimodal checkpoint's towers
                // have no place in a text GGUF.
                let Some(text_name) =
                    pmetal_runtime::gemma4_native::normalize_checkpoint_key(hf_name)
                else {
                    return Ok(None);
                };
                if self.tie_embeddings && text_name == "lm_head.weight" {
                    return Ok(None);
                }
                WeightLoader::hf_to_gguf_name_gemma(&text_name)
            }
            Family::Llama | Family::Qwen | Family::Gemma => WeightLoader::hf_to_gguf_name(hf_name),
        };
        if !is_gguf_name(&name) {
            anyhow::bail!(
                "tensor {hf_name} has no name in a {} GGUF file; exporting it would write a \
                 tensor no engine looks for",
                self.name
            );
        }
        Ok(Some(name))
    }

    /// Put the values of the tensor `hf_name`, of shape `shape`, in the layout
    /// the file keeps them in.
    pub(crate) fn lay_out(&self, hf_name: &str, shape: &[u64], values: &mut [f32]) {
        if self.family == Family::Llama {
            let heads = if hf_name.ends_with("self_attn.q_proj.weight")
                || hf_name.ends_with("self_attn.q_proj.bias")
            {
                Some(self.num_heads)
            } else if hf_name.ends_with("self_attn.k_proj.weight")
                || hf_name.ends_with("self_attn.k_proj.bias")
            {
                Some(self.num_kv_heads)
            } else {
                None
            };
            if let Some(heads) = heads {
                permute_rotary_rows(values, shape, heads);
            }
        }
        if matches!(self.family, Family::Gemma | Family::GemmaPrePostNorm)
            && hf_name.ends_with("norm.weight")
        {
            for value in values.iter_mut() {
                *value += 1.0;
            }
        }
    }

    /// Tensors the file holds that the model has no counterpart of: name,
    /// shape and values.
    pub(crate) fn extra_tensors(&self) -> Vec<(String, Vec<u64>, Vec<f32>)> {
        self.rope_frequency_factors
            .iter()
            .map(|factors| {
                (
                    pmetal_gguf::config::ROPE_FREQS_TENSOR.to_string(),
                    vec![factors.len() as u64],
                    factors.clone(),
                )
            })
            .collect()
    }

    /// Whether the file carries its RoPE scaling as a tensor, so the metadata
    /// must not state it too.
    pub(crate) fn scales_rope_by_tensor(&self) -> bool {
        self.rope_frequency_factors.is_some()
    }
}

/// The per-block tensors the architectures exported here have.
const BLOCK_TENSORS: &[&str] = &[
    "attn_norm",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "attn_q_norm",
    "attn_k_norm",
    "post_attention_norm",
    "ffn_norm",
    "post_ffw_norm",
    "layer_output_scale",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "ffn_gate_inp",
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_gate_shexp",
    "ffn_up_shexp",
    "ffn_down_shexp",
    "ffn_gate_inp_shexp",
];

/// Whether `name` is a tensor of the GGUF tables of the architectures exported
/// here: a block tensor is `blk.N.<tensor>.weight` (or `.bias`).
fn is_gguf_name(name: &str) -> bool {
    match name.strip_prefix("blk.") {
        Some(rest) => {
            let Some((layer, tensor)) = rest.split_once('.') else {
                return false;
            };
            let stem = tensor
                .strip_suffix(".weight")
                .or_else(|| tensor.strip_suffix(".bias"));
            layer.parse::<usize>().is_ok() && stem.is_some_and(|stem| BLOCK_TENSORS.contains(&stem))
        }
        None => matches!(
            name,
            "token_embd.weight" | "output_norm.weight" | "output.weight"
        ),
    }
}

/// The language model's configuration, nested under `text_config` in a
/// multimodal Gemma 3 or Gemma 4 checkpoint. The wrapper's
/// `tie_word_embeddings` governs the text model's head when the text config
/// says nothing, as the loader reads it.
fn text_config(config: &Value) -> Value {
    let Some(mut text) = config
        .get("text_config")
        .filter(|_| config.get("hidden_size").is_none())
        .cloned()
    else {
        return config.clone();
    };
    if let (Some(object), Some(tie)) = (text.as_object_mut(), config.get("tie_word_embeddings")) {
        object
            .entry("tie_word_embeddings")
            .or_insert_with(|| tie.clone());
    }
    text
}

/// The language model's configuration of a multimodal Gemma 3: the nested
/// `text_config`, as [`text_config`] reads it, with the values `transformers`
/// gives the keys the released config leaves out.
fn gemma3_text_config(config: &Value) -> Value {
    let mut text = text_config(config);
    pmetal_models::dispatcher::with_gemma3_text_defaults(&mut text);
    text
}

/// Gemma 4 models the export does not write, refused by what they have.
///
/// The per-layer input embeddings of the E2B and E4B models, experts, per-layer
/// feed-forward widths and layers that share another's keys and values each
/// add tensors and metadata whose layout has not been checked against the
/// reference converter's.
fn reject_unsupported_gemma4(text: &Value) -> anyhow::Result<()> {
    let positive = |key: &str| text.get(key).and_then(Value::as_u64).unwrap_or(0) > 0;
    let enabled = |key: &str| text.get(key).and_then(Value::as_bool).unwrap_or(false);
    for (unsupported, what) in [
        (
            positive("hidden_size_per_layer_input"),
            "per-layer input embeddings",
        ),
        (enabled("enable_moe_block"), "mixture-of-experts layers"),
        (
            enabled("use_double_wide_mlp"),
            "double-wide feed-forward layers",
        ),
        (
            positive("num_kv_shared_layers"),
            "layers that share another layer's keys and values",
        ),
    ] {
        if unsupported {
            anyhow::bail!(
                "GGUF export of Gemma 4 does not cover {what}; it exports the dense Gemma 4 \
                 text models"
            );
        }
    }
    Ok(())
}

/// The per-frequency divisors of Gemma 4's proportional RoPE: the leading pairs
/// of a full-attention head rotate (a quarter of the head in the released
/// models), and the rest are divided by `1e30`, which leaves them unrotated.
/// `None` when the head rotates whole.
fn gemma4_frequency_factors(text: &Value) -> Option<Vec<f32>> {
    let full = text.pointer("/rope_parameters/full_attention")?;
    let fraction = full.get("partial_rotary_factor").and_then(Value::as_f64)?;
    if fraction >= 1.0 {
        return None;
    }
    let head_dim = text
        .get("global_head_dim")
        .or_else(|| text.get("head_dim"))
        .and_then(Value::as_u64)? as usize;
    // The rotated width is the largest even number of channels in the share,
    // as the model computes it.
    let rotated_pairs = ((fraction * head_dim as f64) / 2.0) as usize;
    let mut factors = vec![1e30f32; head_dim / 2];
    factors[..rotated_pairs.min(head_dim / 2)].fill(1.0);
    Some(factors)
}

/// Reorder the rows of a query or key projection (or its bias) from the
/// Hugging Face rotary pairing, `(i, i + d/2)`, to the interleaved pairing,
/// `(2i, 2i + 1)`, head by head: row `a·d/2 + b` of a head goes to row
/// `2b + a`. `shape` is `[rows, ..]`.
fn permute_rotary_rows(values: &mut [f32], shape: &[u64], heads: usize) {
    let rows = shape.first().copied().unwrap_or(0) as usize;
    if heads == 0 || rows == 0 || rows % (2 * heads) != 0 {
        return;
    }
    let width = values.len() / rows;
    let half = rows / heads / 2;
    let source = values.to_vec();
    for head in 0..heads {
        for pair in 0..2 {
            for index in 0..half {
                let from = head * 2 * half + pair * half + index;
                let to = head * 2 * half + index * 2 + pair;
                values[to * width..(to + 1) * width]
                    .copy_from_slice(&source[from * width..(from + 1) * width]);
            }
        }
    }
}

/// The per-frequency divisors Llama 3.x's banded RoPE scaling comes to, when
/// the config asks for it.
fn llama3_frequency_factors(config: &Value) -> Option<Vec<f32>> {
    let scaling = config
        .get("rope_scaling")
        .filter(|v| !v.is_null())
        .or_else(|| config.get("rope_parameters"))?;
    let kind = scaling
        .get("rope_type")
        .or_else(|| scaling.get("type"))
        .and_then(Value::as_str)?;
    if kind != "llama3" {
        return None;
    }
    let number =
        |value: Option<&Value>, default: f64| value.and_then(Value::as_f64).unwrap_or(default);
    let heads = config.get("num_attention_heads").and_then(Value::as_f64)?;
    let head_dim = config
        .get("head_dim")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .or_else(|| {
            let hidden = config.get("hidden_size").and_then(Value::as_f64)?;
            Some((hidden / heads) as usize)
        })?;
    let theta = number(
        config
            .get("rope_theta")
            .or_else(|| scaling.get("rope_theta")),
        10_000.0,
    );
    Some(pmetal_gguf::config::llama3_frequency_factors(
        theta,
        head_dim,
        number(scaling.get("factor"), 8.0),
        number(scaling.get("low_freq_factor"), 1.0),
        number(scaling.get("high_freq_factor"), 4.0),
        number(scaling.get("original_max_position_embeddings"), 8192.0),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn arch(name: &str, extra: Value) -> ExportArch {
        let mut config = json!({
            "architectures": [name],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_size": 16,
        });
        for (key, value) in extra.as_object().unwrap() {
            config[key] = value.clone();
        }
        ExportArch::detect(Some(&config)).unwrap()
    }

    #[test]
    fn mistral_and_mixtral_are_llama_files() {
        assert_eq!(arch("MistralForCausalLM", json!({})).name, "llama");
        assert_eq!(arch("MixtralForCausalLM", json!({})).name, "llama");
        assert_eq!(arch("Gemma2ForCausalLM", json!({})).name, "gemma2");
        assert_eq!(arch("Gemma3ForCausalLM", json!({})).name, "gemma3");
        assert_eq!(arch("Qwen3ForCausalLM", json!({})).name, "qwen3");
    }

    #[test]
    fn an_architecture_with_no_mapping_is_refused_not_guessed() {
        let config = json!({"architectures": ["PhiForCausalLM"]});
        let error = ExportArch::detect(Some(&config)).unwrap_err().to_string();
        assert!(error.contains("PhiForCausalLM"), "{error}");
        assert!(ExportArch::detect(None).is_err());
    }

    /// A multimodal Gemma 4 checkpoint: the language model's tensors under
    /// `model.language_model.`, the towers beside them.
    fn gemma4(extra: Value) -> ExportArch {
        let mut text = json!({
            "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "global_head_dim": 16, "tie_word_embeddings": true,
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 0.25, "rope_type": "proportional"},
                "sliding_attention": {"rope_type": "default"},
            },
        });
        for (key, value) in extra.as_object().unwrap() {
            text[key] = value.clone();
        }
        let config = json!({
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "text_config": text,
            "tie_word_embeddings": true,
        });
        ExportArch::detect(Some(&config)).unwrap()
    }

    #[test]
    fn gemma_4_names_its_language_model_and_leaves_the_towers_out() {
        let gemma = gemma4(json!({}));
        assert_eq!(gemma.name, "gemma4");
        let name = |hf: &str| gemma.tensor_name(hf).unwrap();
        let layer = "model.language_model.layers.7";
        for (hf, gguf) in [
            ("input_layernorm.weight", "attn_norm.weight"),
            (
                "post_attention_layernorm.weight",
                "post_attention_norm.weight",
            ),
            ("pre_feedforward_layernorm.weight", "ffn_norm.weight"),
            ("post_feedforward_layernorm.weight", "post_ffw_norm.weight"),
            ("self_attn.q_proj.weight", "attn_q.weight"),
            ("self_attn.k_norm.weight", "attn_k_norm.weight"),
            ("mlp.down_proj.weight", "ffn_down.weight"),
            ("layer_scalar", "layer_output_scale.weight"),
        ] {
            assert_eq!(
                name(&format!("{layer}.{hf}")).as_deref(),
                Some(format!("blk.7.{gguf}").as_str()),
                "{hf}"
            );
        }
        assert_eq!(
            name("model.language_model.embed_tokens.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            name("model.language_model.norm.weight").as_deref(),
            Some("output_norm.weight")
        );
        assert_eq!(name("model.vision_tower.encoder.layers.0.weight"), None);
        assert_eq!(name("model.embed_audio.embedding_projection.weight"), None);
        assert_eq!(name("lm_head.weight"), None);
    }

    /// A multimodal Gemma 3 as its authors released it (the 4B model): a text
    /// config of five keys, the rest left to `transformers`' defaults.
    fn gemma3_multimodal() -> (ExportArch, Value) {
        let config = json!({
            "architectures": ["Gemma3ForConditionalGeneration"],
            "model_type": "gemma3",
            "text_config": {
                "hidden_size": 2560, "intermediate_size": 10240,
                "model_type": "gemma3_text", "num_hidden_layers": 34,
                "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
                "sliding_window": 1024,
            },
            "vision_config": {"model_type": "siglip_vision_model"},
        });
        (ExportArch::detect(Some(&config)).unwrap(), config)
    }

    #[test]
    fn gemma_3_multimodal_names_its_language_model_and_leaves_the_towers_out() {
        let (gemma, _) = gemma3_multimodal();
        assert_eq!(gemma.name, "gemma3");
        let name = |hf: &str| gemma.tensor_name(hf).unwrap();
        // The authors' layout nests the model one level (`language_model.model`),
        // transformers 5's names it `model.language_model`; both are the same
        // text model.
        for layer in [
            "language_model.model.layers.7",
            "model.language_model.layers.7",
        ] {
            for (hf, gguf) in [
                ("input_layernorm.weight", "attn_norm.weight"),
                (
                    "post_attention_layernorm.weight",
                    "post_attention_norm.weight",
                ),
                ("pre_feedforward_layernorm.weight", "ffn_norm.weight"),
                ("post_feedforward_layernorm.weight", "post_ffw_norm.weight"),
                ("self_attn.q_proj.weight", "attn_q.weight"),
                ("self_attn.k_norm.weight", "attn_k_norm.weight"),
                ("mlp.down_proj.weight", "ffn_down.weight"),
            ] {
                assert_eq!(
                    name(&format!("{layer}.{hf}")).as_deref(),
                    Some(format!("blk.7.{gguf}").as_str()),
                    "{layer}.{hf}"
                );
            }
        }
        for embedding in [
            "language_model.model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
        ] {
            assert_eq!(name(embedding).as_deref(), Some("token_embd.weight"));
        }
        for norm in [
            "language_model.model.norm.weight",
            "model.language_model.norm.weight",
        ] {
            assert_eq!(name(norm).as_deref(), Some("output_norm.weight"));
        }
        for tower in [
            "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
            "model.vision_tower.vision_model.post_layernorm.weight",
            "multi_modal_projector.mm_input_projection_weight",
            "model.multi_modal_projector.mm_soft_emb_norm.weight",
        ] {
            assert_eq!(name(tower), None, "{tower}");
        }
        // Gemma ties its head to the embedding.
        assert_eq!(name("language_model.lm_head.weight"), None);
        assert_eq!(name("lm_head.weight"), None);
        // ...and a norm gains 1, as in the text-only Gemma 3.
        let mut norm = vec![0.5f32, -1.0];
        gemma.lay_out(
            "language_model.model.layers.0.input_layernorm.weight",
            &[2],
            &mut norm,
        );
        assert_eq!(norm, [1.5, 0.0]);
    }

    #[test]
    fn gemma_3_multimodal_reads_its_text_config_with_the_defaults_it_leaves_out() {
        let (gemma, config) = gemma3_multimodal();
        let text = gemma.model_config(Some(&config)).unwrap();
        let number = |key: &str| text.get(key).and_then(Value::as_f64).unwrap();
        // Stated by the config.
        assert_eq!(number("hidden_size"), 2560.0);
        assert_eq!(number("num_hidden_layers"), 34.0);
        assert_eq!(number("sliding_window"), 1024.0);
        // `Gemma3TextConfig`'s.
        assert_eq!(number("num_attention_heads"), 8.0);
        assert_eq!(number("num_key_value_heads"), 4.0);
        assert_eq!(number("head_dim"), 256.0);
        assert_eq!(number("vocab_size"), 262_208.0);
        assert_eq!(number("rope_theta"), 1_000_000.0);
        assert_eq!(number("rope_local_base_freq"), 10_000.0);
        // The head is the embedding.
        assert_eq!(gemma.tensor_name("lm_head.weight").unwrap(), None);
    }

    /// Gemma 4's norm multiplies by its weight: unlike Gemma 1 to 3, no norm
    /// gains 1.
    #[test]
    fn gemma_4_norms_are_stored_as_they_are() {
        let gemma = gemma4(json!({}));
        let mut norm = vec![0.5f32, -1.0];
        gemma.lay_out("model.layers.0.input_layernorm.weight", &[2], &mut norm);
        assert_eq!(norm, [0.5, -1.0]);
    }

    /// The unrotated pairs of a full-attention head are a `rope_freqs` tensor:
    /// the rotated quarter of 16 channels is 2 pairs of 8.
    #[test]
    fn gemma_4_rotates_a_share_of_the_full_attention_head_by_a_tensor() {
        let gemma = gemma4(json!({}));
        assert!(gemma.scales_rope_by_tensor());
        let extra = gemma.extra_tensors();
        assert_eq!(extra.len(), 1);
        assert_eq!(extra[0].0, "rope_freqs.weight");
        assert_eq!(extra[0].1, [8]);
        assert_eq!(extra[0].2[..2], [1.0, 1.0]);
        assert!(extra[0].2[2..].iter().all(|f| *f == 1e30));
        // A head rotated whole has none.
        let whole = gemma4(json!({"rope_parameters": {
            "full_attention": {"partial_rotary_factor": 1.0, "rope_type": "default"},
            "sliding_attention": {"rope_type": "default"},
        }}));
        assert!(whole.extra_tensors().is_empty());
        assert!(!whole.scales_rope_by_tensor());
    }

    /// The Gemma 4 models whose extra tensors have not been checked against
    /// the reference converter are refused, not exported without them.
    #[test]
    fn gemma_4_models_with_extra_tensors_are_refused_by_name() {
        for (extra, what) in [
            (
                json!({"hidden_size_per_layer_input": 256}),
                "per-layer input",
            ),
            (json!({"enable_moe_block": true}), "mixture-of-experts"),
            (json!({"use_double_wide_mlp": true}), "double-wide"),
            (json!({"num_kv_shared_layers": 4}), "share"),
        ] {
            let mut text = json!({"hidden_size": 64});
            for (k, v) in extra.as_object().unwrap() {
                text[k] = v.clone();
            }
            let config = json!({
                "architectures": ["Gemma4ForConditionalGeneration"], "text_config": text,
            });
            let error = ExportArch::detect(Some(&config)).unwrap_err().to_string();
            assert!(error.contains(what), "{error}");
        }
    }

    #[test]
    fn a_tied_head_is_not_written() {
        let tied = arch("Qwen3ForCausalLM", json!({"tie_word_embeddings": true}));
        assert_eq!(tied.tensor_name("lm_head.weight").unwrap(), None);
        let untied = arch("Qwen3ForCausalLM", json!({"tie_word_embeddings": false}));
        assert_eq!(
            untied.tensor_name("lm_head.weight").unwrap().as_deref(),
            Some("output.weight")
        );
    }

    #[test]
    fn qwen_biases_and_norms_get_their_names() {
        let qwen = arch("Qwen2ForCausalLM", json!({}));
        let name = |hf: &str| qwen.tensor_name(hf).unwrap().unwrap();
        assert_eq!(
            name("model.layers.3.self_attn.q_proj.bias"),
            "blk.3.attn_q.bias"
        );
        assert_eq!(
            name("model.layers.3.self_attn.k_proj.bias"),
            "blk.3.attn_k.bias"
        );
        assert_eq!(
            name("model.layers.3.self_attn.v_proj.bias"),
            "blk.3.attn_v.bias"
        );
        let qwen3 = arch("Qwen3ForCausalLM", json!({}));
        assert_eq!(
            qwen3
                .tensor_name("model.layers.0.self_attn.q_norm.weight")
                .unwrap()
                .unwrap(),
            "blk.0.attn_q_norm.weight"
        );
    }

    #[test]
    fn gemma_two_and_three_name_their_four_norms() {
        let gemma = arch("Gemma3ForCausalLM", json!({}));
        let name = |hf: &str| gemma.tensor_name(hf).unwrap().unwrap();
        assert_eq!(
            name("model.layers.2.post_attention_layernorm.weight"),
            "blk.2.post_attention_norm.weight"
        );
        assert_eq!(
            name("model.layers.2.pre_feedforward_layernorm.weight"),
            "blk.2.ffn_norm.weight"
        );
        assert_eq!(
            name("model.layers.2.post_feedforward_layernorm.weight"),
            "blk.2.post_ffw_norm.weight"
        );
        // Gemma 1 has the Llama layout: the one post-attention norm is the
        // feed-forward block's input.
        let gemma1 = arch("GemmaForCausalLM", json!({}));
        assert_eq!(
            gemma1
                .tensor_name("model.layers.2.post_attention_layernorm.weight")
                .unwrap()
                .unwrap(),
            "blk.2.ffn_norm.weight"
        );
    }

    #[test]
    fn a_tensor_with_no_name_in_the_file_is_an_error() {
        let llama = arch("LlamaForCausalLM", json!({}));
        let error = llama
            .tensor_name("model.layers.0.mystery.weight")
            .unwrap_err()
            .to_string();
        assert!(error.contains("model.layers.0.mystery.weight"), "{error}");
        assert_eq!(
            llama
                .tensor_name("model.layers.0.self_attn.rotary_emb.inv_freq")
                .unwrap(),
            None
        );
    }

    /// The converter's permutation, written out row by row.
    #[test]
    fn llama_query_and_key_rows_are_reordered_per_head() {
        let llama = arch("LlamaForCausalLM", json!({}));
        // 4 query heads of 4 rows (2 columns each): row r holds [r, 100 + r].
        let rows = 16;
        let original: Vec<f32> = (0..rows)
            .flat_map(|r| [r as f32, 100.0 + r as f32])
            .collect();
        let mut q = original.clone();
        llama.lay_out("model.layers.0.self_attn.q_proj.weight", &[16, 2], &mut q);
        // Head 0: rows [0, 1, 2, 3] -> [0, 2, 1, 3].
        let order: Vec<f32> = q.chunks(2).map(|row| row[0]).collect();
        assert_eq!(&order[..4], &[0.0, 2.0, 1.0, 3.0]);
        assert_eq!(&order[4..8], &[4.0, 6.0, 5.0, 7.0]);
        // The columns travel with their row.
        assert_eq!(q[1], 100.0);
        assert_eq!(q[3], 102.0);

        // Keys have the key-value head count: 2 heads of 8 rows.
        let mut k = original.clone();
        llama.lay_out("model.layers.0.self_attn.k_proj.weight", &[16, 2], &mut k);
        let order: Vec<f32> = k.chunks(2).map(|row| row[0]).collect();
        assert_eq!(&order[..8], &[0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]);

        // Nothing else moves.
        let mut v = original.clone();
        llama.lay_out("model.layers.0.self_attn.v_proj.weight", &[16, 2], &mut v);
        assert_eq!(v, original);
        // And a Qwen is stored as published.
        let qwen = arch("Qwen3ForCausalLM", json!({}));
        let mut q = original.clone();
        qwen.lay_out("model.layers.0.self_attn.q_proj.weight", &[16, 2], &mut q);
        assert_eq!(q, original);
    }

    #[test]
    fn the_permutation_is_what_the_loader_undoes() {
        // `unpermute_llama_qk` sends row 2b + a back to a·d/2 + b.
        let mut rows: Vec<f32> = (0..8).map(|r| r as f32).collect();
        permute_rotary_rows(&mut rows, &[8, 1], 1);
        for (to, &from) in rows.iter().enumerate() {
            let (b, a) = (to / 2, to % 2);
            assert_eq!(from as usize, a * 4 + b);
        }
    }

    #[test]
    fn gemma_norms_gain_one_and_no_other_tensor_does() {
        let gemma = arch("Gemma3ForCausalLM", json!({}));
        let mut norm = vec![0.5f32, -1.0];
        gemma.lay_out("model.layers.0.input_layernorm.weight", &[2], &mut norm);
        assert_eq!(norm, [1.5, 0.0]);
        let mut qnorm = vec![0.0f32];
        gemma.lay_out("model.layers.0.self_attn.q_norm.weight", &[1], &mut qnorm);
        assert_eq!(qnorm, [1.0]);
        let mut final_norm = vec![2.0f32];
        gemma.lay_out("model.norm.weight", &[1], &mut final_norm);
        assert_eq!(final_norm, [3.0]);
        let mut proj = vec![0.5f32, 0.5];
        gemma.lay_out("model.layers.0.mlp.up_proj.weight", &[1, 2], &mut proj);
        assert_eq!(proj, [0.5, 0.5]);
        let mut embed = vec![0.5f32];
        gemma.lay_out("model.embed_tokens.weight", &[1, 1], &mut embed);
        assert_eq!(embed, [0.5]);
    }

    #[test]
    fn llama_three_rope_becomes_a_tensor() {
        let llama = arch(
            "LlamaForCausalLM",
            json!({
                "head_dim": 64,
                "rope_theta": 500000.0,
                "rope_scaling": {
                    "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192, "rope_type": "llama3"
                }
            }),
        );
        assert!(llama.scales_rope_by_tensor());
        let extra = llama.extra_tensors();
        assert_eq!(extra.len(), 1);
        assert_eq!(extra[0].0, "rope_freqs.weight");
        assert_eq!(extra[0].1, [32]);
        assert!(
            arch("Qwen3ForCausalLM", json!({}))
                .extra_tensors()
                .is_empty()
        );
    }
}
