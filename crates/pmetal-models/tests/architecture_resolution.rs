//! A checkpoint resolves to the architecture it is, or is declined by name.
//!
//! The dispatcher used to match loosely: a `model_type` or class name that
//! merely contained `qwen`, `mistral` or `deepseek` ran as that family. A
//! Qwen3-VL then failed the weight audit with a count of unconsumed tensors, and
//! a DeepSeek V4.1 loaded into the V3 decoder. These tests hold the two sides:
//! every family pmetal declines is refused with its `model_type` named, by type
//! and by class, and the routing keys of the released configs of every
//! supported family still reach their architecture.
//!
//! The routing keys below (`model_type`, `architectures`, a wrapper's
//! `text_config.model_type`, `index_topk`) are read off the released
//! `config.json` of one checkpoint per family. The configs themselves are
//! under their upstream licenses and are not vendored; `real_config_parity`
//! holds the full files against the same dispatcher when they are on hand.

use pmetal_core::model_type::{refused_classes, refused_model_types};
use pmetal_models::dispatcher::ModelArchitecture;
use serde_json::{Value, json};

/// A released config reduced to what routing reads.
fn routing(
    model_type: &str,
    architecture: &str,
    text_model_type: Option<&str>,
    index_topk: Option<u64>,
) -> Value {
    let mut config = json!({
        "model_type": model_type,
        "architectures": [architecture],
    });
    if let Some(text) = text_model_type {
        config["text_config"] = json!({ "model_type": text });
    }
    if let Some(k) = index_topk {
        config["index_topk"] = json!(k);
    }
    config
}

fn refusal(config: &Value) -> pmetal_models::UnsupportedModel {
    ModelArchitecture::resolve_config(config).expect_err("should be declined")
}

/// (checkpoint, released routing keys, family it is declined as)
type Refused = (
    &'static str,
    &'static str,
    &'static str,
    Option<&'static str>,
    Option<u64>,
    &'static str,
);

const REFUSED_RELEASED: &[Refused] = &[
    (
        "Qwen3-VL-8B-Instruct",
        "qwen3_vl",
        "Qwen3VLForConditionalGeneration",
        Some("qwen3_vl_text"),
        None,
        "Qwen3-VL",
    ),
    (
        "Qwen3-VL-30B-A3B-Instruct",
        "qwen3_vl_moe",
        "Qwen3VLMoeForConditionalGeneration",
        Some("qwen3_vl_moe_text"),
        None,
        "Qwen3-VL",
    ),
    (
        "Qwen2.5-VL-7B-Instruct",
        "qwen2_5_vl",
        "Qwen2_5_VLForConditionalGeneration",
        None,
        None,
        "Qwen2-VL / Qwen2.5-VL",
    ),
    (
        "Qwen3-Omni-30B-A3B-Instruct",
        "qwen3_omni_moe",
        "Qwen3OmniMoeForConditionalGeneration",
        None,
        None,
        "Qwen Omni and speech",
    ),
    (
        "Qwen1.5-MoE-A2.7B",
        "qwen2_moe",
        "Qwen2MoeForCausalLM",
        None,
        None,
        "Qwen2-MoE",
    ),
    (
        "Mistral-Small-3.2-24B-Instruct-2506",
        "mistral3",
        "Mistral3ForConditionalGeneration",
        Some("mistral"),
        None,
        "Mistral3 / Ministral / Mistral 4",
    ),
    (
        "Ministral-3-8B-Instruct-2512",
        "mistral3",
        "Mistral3ForConditionalGeneration",
        Some("ministral3"),
        None,
        "Mistral3 / Ministral / Mistral 4",
    ),
    (
        "Mistral-Small-4-119B-2603",
        "mistral3",
        "Mistral3ForConditionalGeneration",
        Some("mistral4"),
        None,
        "Mistral3 / Ministral / Mistral 4",
    ),
    (
        "Devstral-2-123B-Instruct-2512",
        "ministral3",
        "Ministral3ForCausalLM",
        None,
        None,
        "Mistral3 / Ministral / Mistral 4",
    ),
    (
        "pixtral-12b",
        "llava",
        "LlavaForConditionalGeneration",
        Some("mistral"),
        None,
        "Pixtral and LLaVA",
    ),
    (
        "llava-1.5-7b-hf",
        "llava",
        "LlavaForConditionalGeneration",
        Some("llama"),
        None,
        "Pixtral and LLaVA",
    ),
    (
        "Mixtral-8x7B-Instruct-v0.1",
        "mixtral",
        "MixtralForCausalLM",
        None,
        None,
        "Mixtral",
    ),
    (
        "DeepSeek-V3.2",
        "deepseek_v32",
        "DeepseekV32ForCausalLM",
        None,
        Some(2048),
        "DeepSeek V3.2",
    ),
    (
        "DeepSeek-V3.2-Exp",
        "deepseek_v32",
        "DeepseekV32ForCausalLM",
        None,
        Some(2048),
        "DeepSeek V3.2",
    ),
    (
        "DeepSeek-V4-Flash",
        "deepseek_v4",
        "DeepseekV4ForCausalLM",
        None,
        Some(512),
        "DeepSeek V4 / V4.1",
    ),
    (
        "DeepSeek-V4.1",
        "deepseek_v41",
        "DeepseekV41ForCausalLM",
        None,
        None,
        "DeepSeek V4 / V4.1",
    ),
    (
        "deepseek-moe-16b-base",
        "deepseek",
        "DeepseekForCausalLM",
        None,
        None,
        "DeepSeek V1",
    ),
    (
        "GLM-4-9B-0414",
        "glm4",
        "Glm4ForCausalLM",
        None,
        None,
        "GLM-4",
    ),
    (
        "GLM-4.5",
        "glm4_moe",
        "Glm4MoeForCausalLM",
        None,
        None,
        "GLM-4.5 / GLM-4.6 / GLM-4.7 MoE",
    ),
    (
        "GLM-4.7-Flash",
        "glm4_moe_lite",
        "Glm4MoeLiteForCausalLM",
        None,
        None,
        "GLM-4.5 / GLM-4.6 / GLM-4.7 MoE",
    ),
    (
        "GLM-5",
        "glm_moe_dsa",
        "GlmMoeDsaForCausalLM",
        None,
        Some(2048),
        "GLM-5",
    ),
    (
        "GLM-4.5V",
        "glm4v_moe",
        "Glm4vMoeForConditionalGeneration",
        Some("glm4v_moe_text"),
        None,
        "GLM vision-language and OCR",
    ),
    (
        "Kimi-K2-Instruct",
        "kimi_k2",
        "DeepseekV3ForCausalLM",
        None,
        None,
        "Kimi K2",
    ),
    (
        "Kimi-K2.5",
        "kimi_k25",
        "KimiK25ForConditionalGeneration",
        Some("kimi_k2"),
        None,
        "Kimi K2",
    ),
    (
        "Kimi-VL-A3B-Instruct",
        "kimi_vl",
        "KimiVLForConditionalGeneration",
        None,
        None,
        "Kimi K2",
    ),
    (
        "Kimi-Linear-48B-A3B-Instruct",
        "kimi_linear",
        "KimiLinearForCausalLM",
        None,
        None,
        "Kimi Linear",
    ),
    (
        "MiniMax-M2",
        "minimax_m2",
        "MiniMaxM2ForCausalLM",
        None,
        None,
        "MiniMax-M2",
    ),
    (
        "MiniMax-M3",
        "minimax_m3_vl",
        "MiniMaxM3SparseForConditionalGeneration",
        None,
        None,
        "MiniMax-M3 and MiniMax-01",
    ),
    (
        "Phi-2",
        "phi",
        "PhiForCausalLM",
        None,
        None,
        "Phi-1 / Phi-1.5 / Phi-2",
    ),
    (
        "Phi-3.5-MoE-instruct",
        "phimoe",
        "PhiMoEForCausalLM",
        None,
        None,
        "Phi-3.5-MoE and Phi multimodal",
    ),
    (
        "Phi-4-multimodal-instruct",
        "phi4mm",
        "Phi4MMForCausalLM",
        None,
        None,
        "Phi-3.5-MoE and Phi multimodal",
    ),
    (
        "Nemotron-4-Mini-Hindi-4B",
        "nemotron",
        "NemotronForCausalLM",
        None,
        None,
        "Nemotron",
    ),
    (
        "OLMo-2-1124-7B",
        "olmo2",
        "Olmo2ForCausalLM",
        None,
        None,
        "OLMo",
    ),
    (
        "SmolLM3-3B",
        "smollm3",
        "SmolLM3ForCausalLM",
        None,
        None,
        "SmolLM3",
    ),
];

#[test]
fn every_released_config_of_a_declined_family_is_declined_by_name() {
    for (checkpoint, model_type, class, text, topk, family) in REFUSED_RELEASED {
        let config = routing(model_type, class, *text, *topk);
        let err = refusal(&config);
        assert_eq!(err.family, Some(*family), "{checkpoint}: {err}");
        let text = err.to_string();
        assert!(
            text.contains(&format!("{model_type:?}")),
            "{checkpoint} must be named by its model_type: {text}"
        );
        assert!(text.contains("not supported yet"), "{checkpoint}: {text}");
        assert!(text.contains("Supported families:"), "{checkpoint}: {text}");
    }
}

/// The checkpoints the old loose match sent somewhere: each of these
/// `architectures` strings contains the name of a supported family.
#[test]
fn the_names_the_substring_match_misrouted_are_declined() {
    for (class, family) in [
        ("Qwen3VLForConditionalGeneration", "Qwen3-VL"),
        ("Qwen2MoeForCausalLM", "Qwen2-MoE"),
        (
            "Mistral3ForConditionalGeneration",
            "Mistral3 / Ministral / Mistral 4",
        ),
        ("MixtralForCausalLM", "Mixtral"),
        ("DeepseekV41ForCausalLM", "DeepSeek V4 / V4.1"),
        ("DeepseekV32ForCausalLM", "DeepSeek V3.2"),
        ("LlavaLlamaForCausalLM", "Pixtral and LLaVA"),
        (
            "MemTQwen3_5MoeForCausalLM",
            "Qwen3.5 with a memory-tuning head",
        ),
        ("Phi3VForCausalLM", "Phi-3.5-MoE and Phi multimodal"),
        ("EmbeddingGemma2Model", "Gemma embedding and PaliGemma"),
        ("Cohere2MoeForCausalLM", "Cohere2 MoE and vision"),
        (
            "Gemma4AssistantForCausalLM",
            "Gemma 4 assistant (MTP drafter)",
        ),
    ] {
        // By class alone, as a config with no `model_type` states it.
        let err = ModelArchitecture::resolve("", &[class.to_string()]).expect_err(class);
        assert_eq!(err.family, Some(family), "{class}: {err}");
        assert_eq!(
            ModelArchitecture::from_architectures(&[class.to_string()]),
            None,
            "{class}"
        );
    }
}

#[test]
fn every_declined_model_type_is_declined_on_its_own() {
    for (family, model_type) in refused_model_types() {
        let err = ModelArchitecture::resolve(model_type, &[]).expect_err(model_type);
        assert_eq!(err.family, Some(family), "{model_type}");
        assert!(
            err.to_string().contains(&format!("{model_type:?}")),
            "{model_type}: {err}"
        );
        assert_eq!(ModelArchitecture::from_model_type(model_type), None);
    }
}

#[test]
fn every_declined_class_is_declined_on_its_own() {
    // The table holds class names lower-cased; the config writes them in
    // camel case, which the lookup folds.
    for (family, class) in refused_classes() {
        let err = ModelArchitecture::resolve("", &[class.to_string()]).expect_err(class);
        assert_eq!(err.family, Some(family), "{class}");
        assert_eq!(
            ModelArchitecture::from_architectures(&[class.to_string()]),
            None,
            "{class}"
        );
    }
}

/// (checkpoint, released routing keys, the architecture it must reach)
type Supported = (
    &'static str,
    &'static str,
    &'static str,
    Option<&'static str>,
    ModelArchitecture,
);

const SUPPORTED_RELEASED: &[Supported] = &[
    (
        "Llama-3.2-1B-Instruct",
        "llama",
        "LlamaForCausalLM",
        None,
        ModelArchitecture::Llama,
    ),
    (
        "Meta-Llama-3.1-8B-Instruct",
        "llama",
        "LlamaForCausalLM",
        None,
        ModelArchitecture::Llama,
    ),
    (
        "Llama-4-Scout-17B-16E-Instruct",
        "llama4",
        "Llama4ForConditionalGeneration",
        Some("llama4_text"),
        ModelArchitecture::Llama4,
    ),
    (
        "Llama-3.2-11B-Vision-Instruct",
        "mllama",
        "MllamaForConditionalGeneration",
        Some("mllama_text_model"),
        ModelArchitecture::Mllama,
    ),
    (
        "Qwen2.5-0.5B-Instruct",
        "qwen2",
        "Qwen2ForCausalLM",
        None,
        ModelArchitecture::Qwen2,
    ),
    (
        "Qwen3-0.6B",
        "qwen3",
        "Qwen3ForCausalLM",
        None,
        ModelArchitecture::Qwen3,
    ),
    (
        "Qwen3-30B-A3B",
        "qwen3_moe",
        "Qwen3MoeForCausalLM",
        None,
        ModelArchitecture::Qwen3MoE,
    ),
    (
        "Qwen3.5-0.8B",
        "qwen3_5",
        "Qwen3_5ForConditionalGeneration",
        Some("qwen3_5_text"),
        ModelArchitecture::Qwen3Next,
    ),
    (
        "Qwen3.5-35B-A3B",
        "qwen3_5_moe",
        "Qwen3_5MoeForConditionalGeneration",
        Some("qwen3_5_moe_text"),
        ModelArchitecture::Qwen3Next,
    ),
    (
        "Qwen3-Next-80B-A3B-Instruct",
        "qwen3_next",
        "Qwen3NextForCausalLM",
        None,
        ModelArchitecture::Qwen3Next,
    ),
    (
        "Qwen4-Exp",
        "qwen4_exp",
        "Qwen4ExpForConditionalGeneration",
        None,
        ModelArchitecture::Qwen4Exp,
    ),
    (
        "gemma-2-2b-it",
        "gemma2",
        "Gemma2ForCausalLM",
        None,
        ModelArchitecture::Gemma,
    ),
    (
        "gemma-3-1b-it",
        "gemma3_text",
        "Gemma3ForCausalLM",
        None,
        ModelArchitecture::Gemma,
    ),
    (
        "gemma-3-4b-it",
        "gemma3",
        "Gemma3ForConditionalGeneration",
        Some("gemma3_text"),
        ModelArchitecture::Gemma,
    ),
    (
        "gemma-4-31B-it",
        "gemma4",
        "Gemma4ForConditionalGeneration",
        Some("gemma4_text"),
        ModelArchitecture::Gemma4,
    ),
    (
        "gemma-4-12B-it",
        "gemma4_unified",
        "Gemma4UnifiedForConditionalGeneration",
        Some("gemma4_unified_text"),
        ModelArchitecture::Gemma4,
    ),
    (
        "diffusiongemma-26B-A4B-it",
        "diffusion_gemma",
        "DiffusionGemmaForBlockDiffusion",
        Some("diffusion_gemma_text"),
        ModelArchitecture::DiffusionGemma,
    ),
    (
        "Mistral-7B-Instruct-v0.3",
        "mistral",
        "MistralForCausalLM",
        None,
        ModelArchitecture::Mistral,
    ),
    (
        "Phi-3-mini-4k-instruct",
        "phi3",
        "Phi3ForCausalLM",
        None,
        ModelArchitecture::Phi,
    ),
    (
        "Phi-4-mini-instruct",
        "phi3",
        "Phi3ForCausalLM",
        None,
        ModelArchitecture::Phi,
    ),
    (
        "DeepSeek-V2-Lite",
        "deepseek_v2",
        "DeepseekV2ForCausalLM",
        None,
        ModelArchitecture::DeepSeek,
    ),
    (
        "DeepSeek-V3-0324",
        "deepseek_v3",
        "DeepseekV3ForCausalLM",
        None,
        ModelArchitecture::DeepSeek,
    ),
    (
        "DeepSeek-R1",
        "deepseek_v3",
        "DeepseekV3ForCausalLM",
        None,
        ModelArchitecture::DeepSeek,
    ),
    (
        "Moonlight-16B-A3B-Instruct",
        "deepseek_v3",
        "DeepseekV3ForCausalLM",
        None,
        ModelArchitecture::DeepSeek,
    ),
    (
        "c4ai-command-r-v01",
        "cohere",
        "CohereForCausalLM",
        None,
        ModelArchitecture::Cohere,
    ),
    (
        "c4ai-command-a-03-2025",
        "cohere2",
        "Cohere2ForCausalLM",
        None,
        ModelArchitecture::Cohere,
    ),
    (
        "granite-3.1-2b-instruct",
        "granite",
        "GraniteForCausalLM",
        None,
        ModelArchitecture::Granite,
    ),
    (
        "granite-3.1-3b-a800m-instruct",
        "granitemoe",
        "GraniteMoeForCausalLM",
        None,
        ModelArchitecture::Granite,
    ),
    (
        "granite-4.0-h-small",
        "granitemoehybrid",
        "GraniteMoeHybridForCausalLM",
        None,
        ModelArchitecture::Granite,
    ),
    (
        "Nemotron-H-8B-Base-8K",
        "nemotron_h",
        "NemotronHForCausalLM",
        None,
        ModelArchitecture::NemotronH,
    ),
    (
        "gpt-oss-20b",
        "gpt_oss",
        "GptOssForCausalLM",
        None,
        ModelArchitecture::GptOss,
    ),
    (
        "bert-base-uncased",
        "bert",
        "BertForMaskedLM",
        None,
        ModelArchitecture::Bert,
    ),
    (
        "roberta-base",
        "roberta",
        "RobertaForMaskedLM",
        None,
        ModelArchitecture::Bert,
    ),
    (
        "distilbert-base-uncased",
        "distilbert",
        "DistilBertForMaskedLM",
        None,
        ModelArchitecture::Bert,
    ),
    (
        "ModernBERT-base",
        "modernbert",
        "ModernBertForMaskedLM",
        None,
        ModelArchitecture::Bert,
    ),
];

#[test]
fn every_released_config_of_a_supported_family_still_routes() {
    for (checkpoint, model_type, class, text, arch) in SUPPORTED_RELEASED {
        let config = routing(model_type, class, *text, None);
        assert_eq!(
            ModelArchitecture::resolve_config(&config)
                .unwrap_or_else(|e| panic!("{checkpoint}: {e}")),
            *arch,
            "{checkpoint}"
        );
        // And by class alone, for a config that states no usable type.
        assert_eq!(
            ModelArchitecture::resolve("not-a-known-type", &[class.to_string()])
                .unwrap_or_else(|e| panic!("{checkpoint} by class: {e}")),
            *arch,
            "{checkpoint} by class"
        );
    }
}

#[test]
fn a_bundled_mtp_checkpoint_has_no_class_list_and_still_routes() {
    // Qwen3.8-27B-MTP-4bit states `qwen3_5_mtp` and no `architectures`.
    let config = json!({"model_type": "qwen3_5_mtp"});
    assert_eq!(
        ModelArchitecture::resolve_config(&config).unwrap(),
        ModelArchitecture::Qwen3Next
    );
}

#[test]
fn an_unlisted_family_is_named_not_routed_to_a_neighbour() {
    // A family the table has never heard of, with a name that contains one
    // pmetal runs.
    let config = routing("qwen9_hyper", "Qwen9HyperForCausalLM", None, None);
    let err = refusal(&config);
    assert_eq!(err.family, None);
    let text = err.to_string();
    assert!(text.contains("\"qwen9_hyper\""), "{text}");
    assert!(text.contains("not supported yet"), "{text}");
    assert!(text.contains("Supported families:"), "{text}");
}

#[test]
fn the_wrapped_tower_does_not_decide_for_the_wrapper() {
    // Kimi-K2.5 wraps a `kimi_k2` tower and a vision model; an engine that
    // read the tower's type first ran it as the text model.
    let config = routing(
        "kimi_k25",
        "KimiK25ForConditionalGeneration",
        Some("kimi_k2"),
        None,
    );
    assert_eq!(refusal(&config).model_type, "kimi_k25");
    // A Mistral3 wrapper over a plain `mistral` tower is the wrapper.
    let config = routing(
        "mistral3",
        "Mistral3ForConditionalGeneration",
        Some("mistral"),
        None,
    );
    assert_eq!(refusal(&config).model_type, "mistral3");
}

#[test]
fn detect_reads_the_config_in_the_directory() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]}"#,
    )
    .unwrap();
    let err = ModelArchitecture::detect(dir.path()).expect_err("declined");
    let text = err.to_string();
    assert!(text.contains("\"qwen3_vl\""), "{text}");
    assert!(text.contains("not supported yet"), "{text}");

    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "deepseek_v3", "index_topk": 2048}"#,
    )
    .unwrap();
    let text = ModelArchitecture::detect(dir.path())
        .expect_err("declined")
        .to_string();
    assert!(text.contains("DeepSeek V3.2"), "{text}");
}
