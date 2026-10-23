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
}

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
];

/// The architecture a model is exported as, with what its tensors need.
#[derive(Debug, Clone)]
pub(crate) struct ExportArch {
    /// `general.architecture`.
    pub name: &'static str,
    family: Family,
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

        let count = |key: &str| config.get(key).and_then(Value::as_u64).map(|v| v as usize);
        let num_heads = count("num_attention_heads").unwrap_or(1).max(1);
        let num_kv_heads = count("num_key_value_heads").unwrap_or(num_heads).max(1);
        let tie_embeddings = config
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        Ok(Self {
            name,
            family,
            num_heads,
            num_kv_heads,
            tie_embeddings,
            rope_frequency_factors: llama3_frequency_factors(config),
        })
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
            Family::GemmaPrePostNorm => WeightLoader::hf_to_gguf_name_gemma(hf_name),
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
