//! Generate HuggingFace-compatible config.json from GGUF metadata.
//!
//! GGUF files are self-contained — all architecture info lives in the file's
//! metadata. This module extracts that metadata and produces a `serde_json::Value`
//! matching the HuggingFace `config.json` schema so that downstream code
//! (model dispatch, GUI info display) works without a separate config file.

use crate::reader::GgufContent;
use crate::types::MetadataValue;

/// Extract a config.json-compatible JSON object from GGUF metadata.
///
/// Maps GGUF metadata keys (`{arch}.block_count`, etc.) to the HuggingFace
/// config.json schema (`num_hidden_layers`, `hidden_size`, etc.).
///
/// Returns `None` if the GGUF file has no `general.architecture` key.
pub fn config_from_gguf(content: &GgufContent) -> Option<serde_json::Value> {
    let arch = content.architecture()?;

    let mut config = serde_json::Map::new();

    // Model type (architecture)
    config.insert(
        "model_type".to_string(),
        serde_json::Value::String(arch.to_string()),
    );

    // Architecture string for HF compat
    let hf_arch = map_architecture_class(arch);
    config.insert("architectures".to_string(), serde_json::json!([hf_arch]));

    // Model name
    if let Some(name) = get_string(content, "general.name") {
        config.insert("_name_or_path".to_string(), serde_json::Value::String(name));
    }

    // Hidden size (embedding length)
    if let Some(v) = get_u64(content, &format!("{arch}.embedding_length")) {
        config.insert("hidden_size".to_string(), serde_json::json!(v));
    }

    // Number of layers
    if let Some(v) = get_u64(content, &format!("{arch}.block_count")) {
        config.insert("num_hidden_layers".to_string(), serde_json::json!(v));
    }

    // Attention heads
    if let Some(v) = get_u64(content, &format!("{arch}.attention.head_count")) {
        config.insert("num_attention_heads".to_string(), serde_json::json!(v));
    }

    // KV heads (GQA)
    if let Some(v) = get_u64(content, &format!("{arch}.attention.head_count_kv")) {
        config.insert("num_key_value_heads".to_string(), serde_json::json!(v));
    }

    // Intermediate size (FFN)
    if let Some(v) = get_u64(content, &format!("{arch}.feed_forward_length")) {
        config.insert("intermediate_size".to_string(), serde_json::json!(v));
    }

    // Context length
    if let Some(v) = get_u64(content, &format!("{arch}.context_length")) {
        config.insert("max_position_embeddings".to_string(), serde_json::json!(v));
    }

    // Vocab size — count tokenizer tokens if available
    if let Some(v) = get_u64(content, &format!("{arch}.vocab_size")) {
        config.insert("vocab_size".to_string(), serde_json::json!(v));
    } else if let Some(MetadataValue::Array(tokens)) = content.get_metadata("tokenizer.ggml.tokens")
    {
        config.insert("vocab_size".to_string(), serde_json::json!(tokens.len()));
    }

    // RoPE
    if let Some(v) = get_f64(content, &format!("{arch}.rope.freq_base")) {
        config.insert("rope_theta".to_string(), serde_json::json!(v));
    }

    if let Some(v) = get_u64(content, &format!("{arch}.attention.key_length"))
        .or_else(|| get_u64(content, &format!("{arch}.attention.head_dim")))
        .or_else(|| get_u64(content, &format!("{arch}.rope.dimension_count")))
    {
        config.insert("head_dim".to_string(), serde_json::json!(v));
    }

    // RMS norm epsilon
    if let Some(v) = get_f64(content, &format!("{arch}.attention.layer_norm_rms_epsilon")) {
        config.insert("rms_norm_eps".to_string(), serde_json::json!(v));
    }

    // MoE params
    if let Some(v) = get_u64(content, &format!("{arch}.expert_count")) {
        config.insert("num_local_experts".to_string(), serde_json::json!(v));
    }
    if let Some(v) = get_u64(content, &format!("{arch}.expert_used_count")) {
        config.insert("num_experts_per_tok".to_string(), serde_json::json!(v));
    }

    // Quantization info from general.file_type
    if let Some(v) = get_u64(content, "general.file_type") {
        config.insert("gguf_file_type".to_string(), serde_json::json!(v));
    }

    Some(serde_json::Value::Object(config))
}

/// Write a config.json file to disk from GGUF metadata.
///
/// Useful when downloading GGUF-only models that lack a config.json.
/// Returns the path to the written file, or `None` if the GGUF has
/// no architecture metadata.
pub fn write_config_from_gguf(
    content: &GgufContent,
    output_dir: &std::path::Path,
) -> Option<std::path::PathBuf> {
    let config = config_from_gguf(content)?;
    let path = output_dir.join("config.json");

    // Don't overwrite existing config.json
    if path.exists() {
        return Some(path);
    }

    let json = serde_json::to_string_pretty(&config).ok()?;
    std::fs::write(&path, json).ok()?;
    Some(path)
}

/// Map GGUF architecture name to HuggingFace model class name.
fn map_architecture_class(arch: &str) -> String {
    match arch {
        "llama" => "LlamaForCausalLM".to_string(),
        "mistral" => "MistralForCausalLM".to_string(),
        "gemma" | "gemma2" | "gemma3" => "GemmaForCausalLM".to_string(),
        "qwen" | "qwen2" | "qwen3" => "Qwen2ForCausalLM".to_string(),
        "phi" | "phi2" | "phi3" => "PhiForCausalLM".to_string(),
        "gpt2" => "GPT2LMHeadModel".to_string(),
        "falcon" => "FalconForCausalLM".to_string(),
        "mamba" => "MambaForCausalLM".to_string(),
        "cohere" | "command-r" => "CohereForCausalLM".to_string(),
        "deepseek" | "deepseek2" => "DeepseekV2ForCausalLM".to_string(),
        other => {
            // CamelCase fallback: "my_arch" → "MyArchForCausalLM"
            let capitalized: String = other
                .split('_')
                .map(|w| {
                    let mut c = w.chars();
                    match c.next() {
                        Some(f) => f.to_uppercase().to_string() + c.as_str(),
                        None => String::new(),
                    }
                })
                .collect();
            format!("{capitalized}ForCausalLM")
        }
    }
}

// ---------------------------------------------------------------------------
// Metadata helpers
// ---------------------------------------------------------------------------

fn get_string(content: &GgufContent, key: &str) -> Option<String> {
    match content.get_metadata(key)? {
        MetadataValue::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn get_u64(content: &GgufContent, key: &str) -> Option<u64> {
    match content.get_metadata(key)? {
        MetadataValue::Uint8(v) => Some(*v as u64),
        MetadataValue::Uint16(v) => Some(*v as u64),
        MetadataValue::Uint32(v) => Some(*v as u64),
        MetadataValue::Uint64(v) => Some(*v),
        MetadataValue::Int8(v) => Some(*v as u64),
        MetadataValue::Int16(v) => Some(*v as u64),
        MetadataValue::Int32(v) => Some(*v as u64),
        MetadataValue::Int64(v) => Some(*v as u64),
        _ => None,
    }
}

fn get_f64(content: &GgufContent, key: &str) -> Option<f64> {
    match content.get_metadata(key)? {
        MetadataValue::Float32(v) => Some(*v as f64),
        MetadataValue::Float64(v) => Some(*v),
        MetadataValue::Uint32(v) => Some(*v as f64),
        MetadataValue::Int32(v) => Some(*v as f64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_content(arch: &str, metadata: Vec<(&str, MetadataValue)>) -> GgufContent {
        let mut map: HashMap<String, MetadataValue> = HashMap::new();
        map.insert(
            "general.architecture".to_string(),
            MetadataValue::String(arch.to_string()),
        );
        for (k, v) in metadata {
            map.insert(k.to_string(), v);
        }
        GgufContent {
            version: crate::GgufVersion::V3,
            metadata: map,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
        }
    }

    #[test]
    fn test_config_from_gguf_llama() {
        let content = make_content(
            "llama",
            vec![
                ("llama.block_count", MetadataValue::Uint32(32)),
                ("llama.embedding_length", MetadataValue::Uint32(4096)),
                ("llama.attention.head_count", MetadataValue::Uint32(32)),
                ("llama.attention.head_count_kv", MetadataValue::Uint32(8)),
                ("llama.context_length", MetadataValue::Uint32(8192)),
                ("llama.feed_forward_length", MetadataValue::Uint32(11008)),
            ],
        );

        let config = config_from_gguf(&content).unwrap();
        assert_eq!(config["model_type"], "llama");
        assert_eq!(config["num_hidden_layers"], 32);
        assert_eq!(config["hidden_size"], 4096);
        assert_eq!(config["num_attention_heads"], 32);
        assert_eq!(config["num_key_value_heads"], 8);
        assert_eq!(config["max_position_embeddings"], 8192);
        assert_eq!(config["intermediate_size"], 11008);
        assert_eq!(config["architectures"][0], "LlamaForCausalLM");
    }

    #[test]
    fn test_config_from_gguf_no_architecture() {
        let content = GgufContent {
            version: crate::GgufVersion::V3,
            metadata: HashMap::new(),
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
        };
        assert!(config_from_gguf(&content).is_none());
    }

    #[test]
    fn test_vocab_size_from_tokens() {
        let tokens: Vec<MetadataValue> = (0..32000)
            .map(|i| MetadataValue::String(format!("tok_{i}")))
            .collect();
        let content = make_content(
            "llama",
            vec![("tokenizer.ggml.tokens", MetadataValue::Array(tokens))],
        );

        let config = config_from_gguf(&content).unwrap();
        assert_eq!(config["vocab_size"], 32000);
    }
}
