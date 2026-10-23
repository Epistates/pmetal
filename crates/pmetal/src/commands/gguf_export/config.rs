//! Model-config metadata of a GGUF export.

use super::{json_bool, json_f64, json_u32};

fn quantize_method_file_type(
    method: crate::cli::quantize::QuantizeMethod,
) -> pmetal_gguf::FileType {
    use pmetal_gguf::FileType;
    match method {
        crate::cli::quantize::QuantizeMethod::Dynamic => FileType::MostlyQ4KM,
        crate::cli::quantize::QuantizeMethod::Q8_0 => FileType::MostlyQ8_0,
        crate::cli::quantize::QuantizeMethod::Q8_1 => FileType::MostlyQ8_0,
        crate::cli::quantize::QuantizeMethod::Q6K => FileType::MostlyQ6K,
        crate::cli::quantize::QuantizeMethod::Q5KM => FileType::MostlyQ5KM,
        crate::cli::quantize::QuantizeMethod::Q5KS => FileType::MostlyQ5KS,
        crate::cli::quantize::QuantizeMethod::Q5_0 => FileType::MostlyQ5_0,
        crate::cli::quantize::QuantizeMethod::Q5_1 => FileType::MostlyQ5_1,
        crate::cli::quantize::QuantizeMethod::Q4KM => FileType::MostlyQ4KM,
        crate::cli::quantize::QuantizeMethod::Q4KS => FileType::MostlyQ4KS,
        crate::cli::quantize::QuantizeMethod::Q4_0 => FileType::MostlyQ4_0,
        crate::cli::quantize::QuantizeMethod::Q4_1 => FileType::MostlyQ4_1,
        crate::cli::quantize::QuantizeMethod::Iq4Nl => FileType::MostlyIq4Nl,
        crate::cli::quantize::QuantizeMethod::Iq4Xs => FileType::MostlyIq4Xs,
        crate::cli::quantize::QuantizeMethod::Q3KM => FileType::MostlyQ3KM,
        crate::cli::quantize::QuantizeMethod::Q3KS => FileType::MostlyQ3KS,
        crate::cli::quantize::QuantizeMethod::Q3KL => FileType::MostlyQ3KL,
        crate::cli::quantize::QuantizeMethod::Q2K => FileType::MostlyQ2K,
        crate::cli::quantize::QuantizeMethod::Q1_0 => FileType::MostlyQ1_0,
        crate::cli::quantize::QuantizeMethod::Tq1_0 => FileType::MostlyTq1_0,
        crate::cli::quantize::QuantizeMethod::Tq2_0 => FileType::MostlyTq2_0,
        crate::cli::quantize::QuantizeMethod::Mxfp4 => FileType::MostlyMxfp4Moe,
        crate::cli::quantize::QuantizeMethod::Nvfp4 => FileType::MostlyNvfp4,
        crate::cli::quantize::QuantizeMethod::Bf16 => FileType::MostlyBf16,
        crate::cli::quantize::QuantizeMethod::F16 => FileType::MostlyF16,
        crate::cli::quantize::QuantizeMethod::F32 => FileType::AllF32,
    }
}

fn add_arch_u32(
    builder: &mut pmetal_gguf::GgufBuilder,
    architecture: &str,
    suffix: &str,
    value: u32,
) {
    builder.add_u32(format!("{architecture}.{suffix}"), value);
}

fn add_arch_f32(
    builder: &mut pmetal_gguf::GgufBuilder,
    architecture: &str,
    suffix: &str,
    value: f32,
) {
    builder.add_f32(format!("{architecture}.{suffix}"), value);
}

fn add_arch_bool(
    builder: &mut pmetal_gguf::GgufBuilder,
    architecture: &str,
    suffix: &str,
    value: bool,
) {
    builder.add_bool(format!("{architecture}.{suffix}"), value);
}

pub(crate) fn add_hf_config_metadata(
    builder: &mut pmetal_gguf::GgufBuilder,
    architecture: &str,
    config_json: Option<&serde_json::Value>,
    method: crate::cli::quantize::QuantizeMethod,
) {
    builder.add_string("general.type", "model");
    builder.add_u32(
        pmetal_gguf::keys::GENERAL_FILE_TYPE,
        quantize_method_file_type(method) as u32,
    );

    let Some(json) = config_json else {
        return;
    };

    let mut add_u32 = |hf_keys: &[&str], gguf_suffix: &str| {
        for hf_key in hf_keys {
            if let Some(value) = json_u32(json, hf_key) {
                add_arch_u32(builder, architecture, gguf_suffix, value);
                break;
            }
        }
    };

    add_u32(&["hidden_size", "n_embd", "dim"], "embedding_length");
    add_u32(&["num_hidden_layers", "n_layer", "n_layers"], "block_count");
    add_u32(
        &["num_attention_heads", "n_head", "n_heads"],
        "attention.head_count",
    );
    add_u32(
        &["num_key_value_heads", "n_kv_heads"],
        "attention.head_count_kv",
    );
    add_u32(
        &["intermediate_size", "n_inner", "hidden_dim"],
        "feed_forward_length",
    );
    add_u32(
        &[
            "max_position_embeddings",
            "n_ctx",
            "n_positions",
            "max_length",
            "max_sequence_length",
            "model_max_length",
        ],
        "context_length",
    );
    add_u32(&["vocab_size"], "vocab_size");
    add_u32(&["num_local_experts", "num_experts"], "expert_count");
    add_u32(
        &[
            "num_experts_per_tok",
            "num_experts_per_token",
            "top_k_experts",
        ],
        "expert_used_count",
    );
    add_u32(&["n_group"], "expert_group_count");
    add_u32(&["topk_group"], "expert_group_used_count");

    if let Some(false) = json_bool(json, "is_causal") {
        add_arch_bool(builder, architecture, "attention.causal", false);
    }

    if let Some(value) = json_u32(json, "sliding_window") {
        add_arch_u32(builder, architecture, "attention.sliding_window", value);
    }

    let head_dim = json_u32(json, "head_dim").or_else(|| {
        let hidden = json_u32(json, "hidden_size")?;
        let heads = json_u32(json, "num_attention_heads")?;
        (heads != 0).then_some(hidden / heads)
    });
    if let Some(value) = head_dim {
        // The GGUF convention is key_length/value_length. Keep the older pmetal
        // compatibility key until all internal loaders consume the standard one.
        add_arch_u32(builder, architecture, "attention.head_dim", value);
        add_arch_u32(builder, architecture, "attention.key_length", value);
        add_arch_u32(builder, architecture, "attention.value_length", value);
        add_arch_u32(builder, architecture, "rope.dimension_count", value);
    }

    if let Some(value) = json_f64(json, "rope_theta") {
        add_arch_f32(builder, architecture, "rope.freq_base", value as f32);
    }
    if let Some(value) = json_f64(json, "rms_norm_eps") {
        add_arch_f32(
            builder,
            architecture,
            "attention.layer_norm_rms_epsilon",
            value as f32,
        );
    }
    if let Some(value) = json_f64(json, "layer_norm_eps")
        .or_else(|| json_f64(json, "layer_norm_epsilon"))
        .or_else(|| json_f64(json, "norm_epsilon"))
    {
        add_arch_f32(
            builder,
            architecture,
            "attention.layer_norm_epsilon",
            value as f32,
        );
    }

    if let Some(rope_scaling) = json.get("rope_scaling").and_then(|value| value.as_object()) {
        let rope_type = rope_scaling
            .get("rope_type")
            .or_else(|| rope_scaling.get("type"))
            .and_then(|value| value.as_str());
        if let Some(rope_type) = rope_type {
            let gguf_type = match rope_type.to_ascii_lowercase().as_str() {
                "linear" => Some("linear"),
                "yarn" => Some("yarn"),
                "su" | "longrope" => Some("longrope"),
                "none" => Some("none"),
                _ => None,
            };
            if let Some(gguf_type) = gguf_type {
                builder.add_string(format!("{architecture}.rope.scaling.type"), gguf_type);
            }
        }
        if let Some(value) = rope_scaling.get("factor").and_then(|value| value.as_f64()) {
            add_arch_f32(builder, architecture, "rope.scaling.factor", value as f32);
        }
        if let Some(value) = rope_scaling
            .get("original_max_position_embeddings")
            .or_else(|| rope_scaling.get("original_context_length"))
            .and_then(|value| value.as_u64())
            .and_then(|value| u32::try_from(value).ok())
        {
            add_arch_u32(
                builder,
                architecture,
                "rope.scaling.original_context_length",
                value,
            );
        }
        if let Some(value) = rope_scaling
            .get("attention_factor")
            .or_else(|| rope_scaling.get("attn_factor"))
            .and_then(|value| value.as_f64())
        {
            add_arch_f32(
                builder,
                architecture,
                "rope.scaling.yarn_attn_factor",
                value as f32,
            );
        }
        if let Some(value) = rope_scaling
            .get("beta_fast")
            .and_then(|value| value.as_f64())
        {
            add_arch_f32(
                builder,
                architecture,
                "rope.scaling.yarn_beta_fast",
                value as f32,
            );
        }
        if let Some(value) = rope_scaling
            .get("beta_slow")
            .and_then(|value| value.as_f64())
        {
            add_arch_f32(
                builder,
                architecture,
                "rope.scaling.yarn_beta_slow",
                value as f32,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_gguf::{GgufBuilder, GgufContent, MetadataValue};
    use std::io::Cursor;

    #[test]
    fn quantize_methods_have_matching_gguf_file_types() {
        use pmetal_gguf::FileType;

        assert_eq!(
            quantize_method_file_type(crate::cli::quantize::QuantizeMethod::Q4_0),
            FileType::MostlyQ4_0
        );
        assert_eq!(
            quantize_method_file_type(crate::cli::quantize::QuantizeMethod::Q5_1),
            FileType::MostlyQ5_1
        );
        assert_eq!(
            quantize_method_file_type(crate::cli::quantize::QuantizeMethod::Q1_0),
            FileType::MostlyQ1_0
        );
        assert_eq!(
            quantize_method_file_type(crate::cli::quantize::QuantizeMethod::Mxfp4),
            FileType::MostlyMxfp4Moe
        );
        assert_eq!(
            quantize_method_file_type(crate::cli::quantize::QuantizeMethod::Nvfp4),
            FileType::MostlyNvfp4
        );
    }

    #[test]
    fn hf_config_metadata_writes_standard_llama_cpp_head_keys() {
        let config = serde_json::json!({
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "max_position_embeddings": 32768,
            "vocab_size": 128256,
            "head_dim": 128,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 0.000001
        });
        let mut builder = GgufBuilder::with_model("qwen3", "test");
        add_hf_config_metadata(
            &mut builder,
            "qwen3",
            Some(&config),
            crate::cli::quantize::QuantizeMethod::Q4KM,
        );

        let bytes = builder.build_to_bytes().unwrap();
        let content = GgufContent::read(&mut Cursor::new(bytes)).unwrap();
        assert!(matches!(
            content.get_metadata("qwen3.attention.key_length"),
            Some(MetadataValue::Uint32(128))
        ));
        assert!(matches!(
            content.get_metadata("qwen3.attention.value_length"),
            Some(MetadataValue::Uint32(128))
        ));
        assert!(matches!(
            content.get_metadata("qwen3.rope.dimension_count"),
            Some(MetadataValue::Uint32(128))
        ));
        assert!(matches!(
            content.get_metadata("general.file_type"),
            Some(MetadataValue::Uint32(15))
        ));
    }
}
