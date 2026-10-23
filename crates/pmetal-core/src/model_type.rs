//! What a checkpoint's `config.json` says it is, and which families pmetal
//! refuses by name.
//!
//! Every surface that has to decide whether it can run a checkpoint (the
//! loader, the native engines, the sampling presets, the Hub fit estimate)
//! starts from the same two facts: the `model_type`, and the `architectures`
//! class names. They used to be read four different ways, and the loose ones
//! sent a family pmetal does not implement to the nearest one it does: the
//! name contained `qwen`, `mistral` or `deepseek`, so it ran as one, and the
//! weight audit then failed with a count of tensors nothing consumed, or the
//! model loaded and computed something else. This module is the one reading of
//! a config, and the one list of what is declined.
//!
//! The supported side, which maps a `model_type` to an architecture, lives
//! with the architectures in `pmetal-models`. The refused side is plain data
//! and lives here so the crates that cannot depend on the models (the Hub
//! estimate, the data crate) can say so too.

use serde_json::Value;

/// A `model_type` in the one spelling every table compares: lower case, with
/// the `_mtp` suffix removed.
///
/// A `*_mtp` checkpoint is the base architecture plus bundled multi-token
/// prediction layers, which `--mtp` loads separately as a speculative draft.
/// `Qwen3.8-27B-MTP-4bit` states `qwen3_5_mtp` and carries no `architectures`
/// array; it runs as its target model.
pub fn canonical_model_type(model_type: &str) -> String {
    let lower = model_type.trim().to_lowercase();
    match lower.strip_suffix("_mtp") {
        Some(base) => base.to_string(),
        None => lower,
    }
}

/// The `model_type` that names the checkpoint, canonicalised: the one at the
/// top level of the config.
///
/// The top level is authoritative. A multimodal wrapper nests its language
/// tower under `text_config`, with a `model_type` of its own that is the tower's
/// and not the checkpoint's: Kimi-K2.5 says `kimi_k25` outside and `kimi_k2`
/// inside, Mistral Small 3.2 says `mistral3` and `mistral`, a LLaVA says
/// `llava` and `llama`. Reading the inner one first ran each of them as the
/// text model it wraps, minus the vision tower and the projector. The nested
/// type is consulted only when the config states none of its own.
pub fn effective_model_type(config: &Value) -> Option<String> {
    let stated = |value: &Value| {
        value
            .get("model_type")
            .and_then(Value::as_str)
            .map(canonical_model_type)
            .filter(|t| !t.is_empty())
    };
    stated(config).or_else(|| config.get("text_config").and_then(stated))
}

/// The class names in the config's `architectures` array, as written.
pub fn config_architectures(config: &Value) -> Vec<String> {
    config
        .get("architectures")
        .and_then(Value::as_array)
        .map(|names| {
            names
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// The sparse-attention top-k a DeepSeek-V3.2 family config carries, wherever
/// the config states it. Present on `deepseek_v32` and on every checkpoint
/// that reuses the `deepseek_v3` type with the V3.2 indexer, so the type alone
/// does not tell them from V3.
pub fn index_topk(config: &Value) -> Option<u64> {
    config
        .get("index_topk")
        .or_else(|| config.get("text_config")?.get("index_topk"))
        .and_then(Value::as_u64)
}

/// A family pmetal declines, and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsupportedFamily {
    /// The family as people name it.
    pub family: &'static str,
    /// What is missing, as a clause that follows "not supported yet:".
    pub reason: &'static str,
}

/// One row of the refusal table.
struct Refusal {
    family: &'static str,
    reason: &'static str,
    /// Canonical `model_type`s of the family.
    model_types: &'static [&'static str],
    /// Class names of the family, lower case. Consulted when the config names
    /// no `model_type` pmetal knows, because a checkpoint rewrapped under a
    /// custom type still declares its class.
    classes: &'static [&'static str],
}

const VISION_WRAPPER: &str =
    "it is a vision-language model and pmetal does not run its vision tower or projector";
const AUDIO_MODEL: &str = "it is a speech or audio model, not a text decoder";

/// Every family declined by name. Where a refused type is spelled like a
/// supported one (`qwen3_vl` beside `qwen3`, `deepseek_v4` beside
/// `deepseek_v3`) it stays here so a rename of either side cannot quietly
/// route it to the neighbour: the tests hold the two tables disjoint.
const REFUSALS: &[Refusal] = &[
    // Qwen: the rest of the family beside the Qwen 2, 3 and 3.5 text decoders.
    Refusal {
        family: "Qwen3-VL",
        reason: "its vision tower, DeepStack feature injection and interleaved multimodal \
                 rotary embedding are not implemented",
        model_types: &[
            "qwen3_vl",
            "qwen3_vl_text",
            "qwen3_vl_moe",
            "qwen3_vl_moe_text",
        ],
        classes: &[
            "qwen3vlforconditionalgeneration",
            "qwen3vlmoeforconditionalgeneration",
        ],
    },
    Refusal {
        family: "Qwen2-VL / Qwen2.5-VL",
        reason: VISION_WRAPPER,
        model_types: &["qwen2_vl", "qwen2_vl_text", "qwen2_5_vl", "qwen2_5_vl_text"],
        classes: &[
            "qwen2vlforconditionalgeneration",
            "qwen2_5_vlforconditionalgeneration",
        ],
    },
    Refusal {
        family: "Qwen Omni and speech",
        reason: "it is a multimodal speech and audio model; pmetal runs the Qwen text decoders only",
        model_types: &[
            "qwen2_audio",
            "qwen2_5_omni",
            "qwen3_omni_moe",
            "qwen3_asr",
            "qwen3_tts",
            "fish_qwen3_omni",
        ],
        classes: &[
            "qwen3omnimoeforconditionalgeneration",
            "qwen3asrforconditionalgeneration",
            "qwen3ttsforconditionalgeneration",
        ],
    },
    Refusal {
        family: "Qwen2-MoE",
        reason: "the Qwen 1.5 / Qwen2 mixture-of-experts layout (shared expert gate) is not \
                 implemented; pmetal runs Qwen3-MoE",
        model_types: &["qwen2_moe"],
        classes: &["qwen2moeforcausallm"],
    },
    Refusal {
        family: "Qwen (first generation)",
        reason: "the original Qwen architecture (`QWenLMHeadModel`) is not implemented",
        model_types: &["qwen"],
        classes: &["qwenlmheadmodel"],
    },
    Refusal {
        family: "Qwen3.5 with a memory-tuning head",
        reason: "its added memory-tuning layers are not implemented; the plain Qwen3.5 \
                 checkpoint it derives from is supported",
        model_types: &["memt_qwen3_5_moe"],
        classes: &["memtqwen3_5moeforcausallm"],
    },
    // Mistral.
    Refusal {
        family: "Mistral3 / Ministral / Mistral 4",
        reason: "it is the Mistral3 multimodal wrapper (a Pixtral vision tower and a projector \
                 around a Mistral, Ministral or Mistral 4 decoder), whose language weights sit \
                 under a different prefix; none of that is implemented",
        model_types: &["mistral3", "ministral", "ministral3", "mistral4"],
        classes: &[
            "mistral3forconditionalgeneration",
            "mistral3model",
            "ministral3forcausallm",
            "mistral4forcausallm",
            "lightonocrforconditionalgeneration",
        ],
    },
    Refusal {
        family: "Pixtral and LLaVA",
        reason: VISION_WRAPPER,
        model_types: &[
            "pixtral",
            "llava",
            "llava_next",
            "llava_next_video",
            "llava_onevision",
        ],
        classes: &[
            "llavaforconditionalgeneration",
            "llavallamaforcausallm",
            "llavanextforconditionalgeneration",
            "pixtralvisionmodel",
        ],
    },
    Refusal {
        family: "Mixtral",
        reason: "the sparse mixture-of-experts layers (`block_sparse_moe`) are not implemented; \
                 the Mistral decoder is dense",
        model_types: &["mixtral"],
        classes: &["mixtralforcausallm"],
    },
    Refusal {
        family: "Voxtral",
        reason: AUDIO_MODEL,
        model_types: &["voxtral", "voxtral_realtime"],
        classes: &[
            "voxtralforconditionalgeneration",
            "voxtralrealtimeforconditionalgeneration",
        ],
    },
    // DeepSeek.
    Refusal {
        family: "DeepSeek V3.2",
        reason: "its DeepSeek Sparse Attention indexer (`index_topk`) is not implemented; \
                 pmetal runs DeepSeek V2 and V3 (dense MLA)",
        model_types: &["deepseek_v32"],
        classes: &["deepseekv32forcausallm"],
    },
    Refusal {
        family: "DeepSeek V4 / V4.1",
        reason: "its attention and expert routing differ from DeepSeek V3 and are not \
                 implemented",
        model_types: &["deepseek_v4", "deepseek_v41", "memt_deepseek_v4"],
        classes: &[
            "deepseekv4forcausallm",
            "deepseekv41forcausallm",
            "memtdeepseekv4forcausallm",
        ],
    },
    Refusal {
        family: "DeepSeek V1",
        reason: "the first-generation DeepSeek MoE uses plain multi-head attention, not the \
                 multi-head latent attention pmetal implements (DeepSeek V2 and V3)",
        model_types: &["deepseek"],
        classes: &["deepseekforcausallm"],
    },
    Refusal {
        family: "DeepSeek vision-language",
        reason: VISION_WRAPPER,
        model_types: &[
            "deepseek_vl",
            "deepseek_vl_v2",
            "deepseek_vl_hybrid",
            "deepseek_ocr",
            "deepseek_ocr2",
        ],
        classes: &[
            "deepseekvlforconditionalgeneration",
            "deepseekvlv2forcausallm",
            "deepseekocrforcausallm",
        ],
    },
    // GLM.
    Refusal {
        family: "GLM-4",
        reason: "the GLM-4 decoder is not implemented",
        model_types: &["glm", "glm4", "chatglm"],
        classes: &["glmforcausallm", "glm4forcausallm", "chatglmmodel"],
    },
    Refusal {
        family: "GLM-4.5 / GLM-4.6 / GLM-4.7 MoE",
        reason: "the GLM mixture-of-experts decoder is not implemented",
        model_types: &["glm4_moe", "glm4_moe_lite"],
        classes: &["glm4moeforcausallm", "glm4moeliteforcausallm"],
    },
    Refusal {
        family: "GLM-5",
        reason: "its DeepSeek Sparse Attention indexer and mixture-of-experts decoder are not \
                 implemented",
        model_types: &["glm_moe_dsa", "glm5_next"],
        classes: &["glmmoedsaforcausallm", "glm5nextforconditionalgeneration"],
    },
    Refusal {
        family: "GLM vision-language and OCR",
        reason: VISION_WRAPPER,
        model_types: &[
            "glm4v",
            "glm4v_moe",
            "glm4v_text",
            "glm4v_moe_text",
            "glm_ocr",
        ],
        classes: &[
            "glm4vforconditionalgeneration",
            "glm4vmoeforconditionalgeneration",
            "glmocrforconditionalgeneration",
        ],
    },
    // Kimi.
    Refusal {
        family: "Kimi K2",
        reason: "its checkpoints are a vision-language wrapper or a trillion-parameter \
                 mixture-of-experts that pmetal has not validated; it is not wired up",
        model_types: &["kimi_k2", "kimi_k25", "kimi_k3", "kimi_vl"],
        classes: &[
            "kimik25forconditionalgeneration",
            "kimik3forconditionalgeneration",
            "kimivlforconditionalgeneration",
        ],
    },
    Refusal {
        family: "Kimi Linear",
        reason: "its Kimi Delta Attention linear-attention layers are not implemented",
        model_types: &["kimi_linear"],
        classes: &["kimilinearforcausallm"],
    },
    // MiniMax.
    Refusal {
        family: "MiniMax-M2",
        reason: "the MiniMax mixture-of-experts decoder is not implemented",
        model_types: &["minimax_m2", "minimax_m25"],
        classes: &["minimaxm2forcausallm"],
    },
    Refusal {
        family: "MiniMax-M3 and MiniMax-01",
        reason: "the MiniMax hybrid lightning-attention decoder is not implemented",
        model_types: &["minimax", "minimax_m3", "minimax_m3_vl", "minimax_text_01"],
        classes: &[
            "minimaxm3sparseforconditionalgeneration",
            "minimaxtext01forcausallm",
            "minimaxforcausallm",
        ],
    },
    // Gemma beside the Gemma 1-4 text decoders.
    Refusal {
        family: "Gemma 3n",
        reason: "its MatFormer decoder and its audio and vision towers are not implemented",
        model_types: &["gemma3n", "gemma3n_text", "gemma3n_audio", "gemma3n_vision"],
        classes: &["gemma3nforconditionalgeneration", "gemma3nforcausallm"],
    },
    Refusal {
        family: "Gemma 4 assistant (MTP drafter)",
        reason: "it is a drafter for speculative decoding and loads through `--draft-model`, \
                 not as a standalone model",
        model_types: &["gemma4_assistant"],
        classes: &["gemma4assistantforcausallm"],
    },
    Refusal {
        family: "Gemma embedding and PaliGemma",
        reason: "it is an embedding or vision-language variant pmetal does not implement",
        model_types: &[
            "embedding_gemma2",
            "paligemma",
            "t5gemma",
            "t5gemma2",
            "recurrent_gemma",
        ],
        classes: &[
            "embeddinggemma2model",
            "paligemmaforconditionalgeneration",
            "t5gemmaforconditionalgeneration",
            "recurrentgemmaforcausallm",
        ],
    },
    // Phi beside Phi-3 and Phi-4-mini.
    Refusal {
        family: "Phi-1 / Phi-1.5 / Phi-2",
        reason: "its parallel-block architecture (LayerNorm with bias, a single GELU MLP, no \
                 post-attention norm) is not the Phi-3 one pmetal implements",
        model_types: &["phi"],
        classes: &["phiforcausallm"],
    },
    Refusal {
        family: "Phi-3.5-MoE and Phi multimodal",
        reason: "the Phi mixture-of-experts layers and the Phi vision and audio towers are not \
                 implemented",
        model_types: &["phimoe", "phi3small", "phi3_v", "phi4mm", "phi4_multimodal"],
        classes: &[
            "phimoeforcausallm",
            "phi3smallforcausallm",
            "phi3vforcausallm",
            "phi4mmforcausallm",
            "phi4multimodalforcausallm",
        ],
    },
    // Cohere beside Command R and Command R+.
    Refusal {
        family: "Cohere2 MoE and vision",
        reason: "the Cohere mixture-of-experts and vision variants are not implemented",
        model_types: &["cohere2_moe", "cohere2_vision"],
        classes: &[
            "cohere2moeforcausallm",
            "cohere2visionforconditionalgeneration",
        ],
    },
    // Granite beside the four text families.
    Refusal {
        family: "Granite sliding-window",
        reason: "its sliding-window attention layers are not implemented",
        model_types: &["granite_swa", "granitemoe_swa"],
        classes: &["graniteswaforcausallm"],
    },
    Refusal {
        family: "Granite Switch",
        reason: "the granite_switch architecture is not implemented",
        model_types: &["granite_switch"],
        classes: &["graniteswitchforcausallm"],
    },
    Refusal {
        family: "Granite vision",
        reason: "it is a vision-language wrapper; pmetal runs Granite text models only",
        model_types: &["granite4_vision", "granite_vision"],
        classes: &[],
    },
    Refusal {
        family: "Granite speech",
        reason: "it is a speech model; pmetal runs Granite text models only",
        model_types: &["granite_speech", "granite_speech_plus"],
        classes: &["granitespeechforconditionalgeneration"],
    },
    // NVIDIA Nemotron beside Nemotron-H.
    Refusal {
        family: "Nemotron-H Omni and Puzzle",
        reason: "the multimodal Nemotron-H wrapper and the heterogeneous Puzzle variant are not \
                 implemented; plain Nemotron-H is supported",
        model_types: &["nemotron_h_omni", "nemotron_h_puzzle"],
        classes: &["nemotronhpuzzleforcausallm"],
    },
    Refusal {
        family: "Nemotron",
        reason: "the dense Nemotron-4 architecture is not implemented",
        model_types: &["nemotron"],
        classes: &["nemotronforcausallm"],
    },
    // The rest of what the Hub's trending list carries.
    Refusal {
        family: "LFM2",
        reason: "its gated short-convolution hybrid decoder is not implemented",
        model_types: &["lfm2", "lfm2_moe", "lfm2_vl"],
        classes: &[
            "lfm2forcausallm",
            "lfm2moeforcausallm",
            "lfm2vlforconditionalgeneration",
        ],
    },
    Refusal {
        family: "MiMo",
        reason: "the MiMo decoders are not implemented",
        model_types: &["mimo", "mimo_v2"],
        classes: &["mimoforcausallm", "mimov2forcausallm"],
    },
    Refusal {
        family: "Hunyuan",
        reason: "the Hunyuan decoders are not implemented",
        model_types: &["hunyuan_v1_dense", "hunyuan_v1_moe", "hy_v3", "hy_v4"],
        classes: &[
            "hunyuandensev1forcausallm",
            "hunyuanmoev1forcausallm",
            "hyv3forcausallm",
            "hyv4forcausallm",
        ],
    },
    Refusal {
        family: "Ling / Ring (Bailing)",
        reason: "the Bailing mixture-of-experts and hybrid linear-attention decoders are not \
                 implemented",
        model_types: &[
            "bailing_moe",
            "bailing_moe_v2",
            "bailing_hybrid",
            "bailing_moe_v3_audio",
        ],
        classes: &[
            "bailingmoeforcausallm",
            "bailingmoev2forcausallm",
            "bailingmoev3forcausallm",
        ],
    },
    Refusal {
        family: "Step",
        reason: "the Step decoders are not implemented",
        model_types: &["step3p5", "step5_vl"],
        classes: &["step3p5forcausallm", "step5forconditionalgeneration"],
    },
    Refusal {
        family: "OLMo",
        reason: "the OLMo decoders are not implemented",
        model_types: &["olmo", "olmo2", "olmo3", "olmoe"],
        classes: &[
            "olmoforcausallm",
            "olmo2forcausallm",
            "olmo3forcausallm",
            "olmoeforcausallm",
        ],
    },
    Refusal {
        family: "SmolLM3",
        reason: "the SmolLM3 decoder is not implemented",
        model_types: &["smollm3"],
        classes: &["smollm3forcausallm"],
    },
    Refusal {
        family: "EXAONE",
        reason: "the EXAONE decoders are not implemented",
        model_types: &["exaone", "exaone4"],
        classes: &["exaoneforcausallm", "exaone4forcausallm"],
    },
    Refusal {
        family: "Apertus",
        reason: "the Apertus decoders are not implemented",
        model_types: &["apertus", "apertus1p5"],
        classes: &["apertusforcausallm", "apertus1p5forconditionalgeneration"],
    },
    Refusal {
        family: "Falcon",
        reason: "the Falcon and Falcon-H1 decoders are not implemented",
        model_types: &["falcon", "falcon_h1", "falcon_mamba"],
        classes: &["falconforcausallm", "falconh1forcausallm"],
    },
    Refusal {
        family: "ERNIE 4.5",
        reason: "the ERNIE decoders are not implemented",
        model_types: &["ernie4_5", "ernie4_5_moe"],
        classes: &["ernie4_5forcausallm", "ernie4_5_moeforcausallm"],
    },
    Refusal {
        family: "Mamba and Jamba",
        reason: "the pure state-space and Jamba hybrid decoders are not implemented",
        model_types: &["mamba", "mamba2", "jamba"],
        classes: &["mambaforcausallm", "mamba2forcausallm", "jambaforcausallm"],
    },
    Refusal {
        family: "GPT-2 family",
        reason: "the GPT-2, GPT-J, GPT-NeoX, OPT and BLOOM decoders are not implemented",
        model_types: &["gpt2", "gptj", "gpt_neo", "gpt_neox", "opt", "bloom"],
        classes: &[
            "gpt2lmheadmodel",
            "gptjforcausallm",
            "gptneoforcausallm",
            "gptneoxforcausallm",
            "optforcausallm",
            "bloomforcausallm",
        ],
    },
    Refusal {
        family: "Encoder-decoder and speech models",
        reason: "it is an encoder-decoder or speech model, not a causal language model",
        model_types: &[
            "t5",
            "mt5",
            "bart",
            "whisper",
            "wav2vec2",
            "albert",
            "deberta",
            "deberta-v2",
        ],
        classes: &[
            "t5forconditionalgeneration",
            "t5withlmheadmodel",
            "whisperforconditionalgeneration",
            "albertmodel",
            "debertamodel",
            "debertav2model",
        ],
    },
];

/// The family `model_type` or one of `architectures` names, if pmetal declines
/// it. `model_type` is compared canonicalised; class names are compared case
/// insensitively and whole.
///
/// The `model_type` is consulted first, and a class name only when no row
/// carries it: a type pmetal has never heard of may still declare a class.
pub fn unsupported_family(model_type: &str, architectures: &[String]) -> Option<UnsupportedFamily> {
    let model_type = canonical_model_type(model_type);
    let row = REFUSALS
        .iter()
        .find(|row| row.model_types.contains(&model_type.as_str()))
        .or_else(|| {
            architectures.iter().find_map(|class| {
                let class = class.to_lowercase();
                REFUSALS
                    .iter()
                    .find(|row| row.classes.contains(&class.as_str()))
            })
        })
        .or_else(|| {
            // `granite_speech_*` and the like: a family with a numbered or
            // suffixed type is declined by its stem.
            ["granite_speech"]
                .iter()
                .find(|stem| model_type.starts_with(*stem))
                .and_then(|stem| REFUSALS.iter().find(|row| row.model_types.contains(stem)))
        })?;
    Some(UnsupportedFamily {
        family: row.family,
        reason: row.reason,
    })
}

/// The family a config is declined as, for a caller that cannot see which
/// types pmetal supports (the Hub estimate): by `model_type`, by class name only
/// when the config states no type, and a `deepseek_v3` carrying an
/// `index_topk` as the DeepSeek V3.2 it is.
///
/// The loader (`ModelArchitecture::resolve_config` in `pmetal-models`) is the
/// authority; this agrees with it wherever it answers, and says nothing for a
/// type it has never heard of, which the loader may still resolve by class.
pub fn declined_family(config: &Value) -> Option<UnsupportedFamily> {
    let model_type = effective_model_type(config);
    match model_type.as_deref() {
        None => unsupported_family("", &config_architectures(config)),
        Some(t) => unsupported_family(t, &[]).or_else(|| sparse_deepseek_v3(t, config)),
    }
}

/// DeepSeek V3.2 under the `deepseek_v3` type: the sparse-attention indexer
/// (`index_topk`) is what tells it from V3, which the type alone does not.
pub fn sparse_deepseek_v3(model_type: &str, config: &Value) -> Option<UnsupportedFamily> {
    if canonical_model_type(model_type) == "deepseek_v3" && index_topk(config).is_some() {
        unsupported_family("deepseek_v32", &[])
    } else {
        None
    }
}

/// Every `model_type` and class name the refusal table lists, for the tests
/// that hold it against the supported table.
#[doc(hidden)]
pub fn refused_model_types() -> impl Iterator<Item = (&'static str, &'static str)> {
    REFUSALS
        .iter()
        .flat_map(|row| row.model_types.iter().map(|t| (row.family, *t)))
}

/// The refused class names, as [`refused_model_types`] is the refused types.
#[doc(hidden)]
pub fn refused_classes() -> impl Iterator<Item = (&'static str, &'static str)> {
    REFUSALS
        .iter()
        .flat_map(|row| row.classes.iter().map(|c| (row.family, *c)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_top_level_type_decides_not_the_wrapped_tower() {
        let kimi = json!({"model_type": "kimi_k25", "text_config": {"model_type": "kimi_k2"}});
        assert_eq!(effective_model_type(&kimi).as_deref(), Some("kimi_k25"));
        let nested_only = json!({"text_config": {"model_type": "Qwen3_5_Text"}});
        assert_eq!(
            effective_model_type(&nested_only).as_deref(),
            Some("qwen3_5_text")
        );
        assert_eq!(effective_model_type(&json!({"model_type": ""})), None);
    }

    #[test]
    fn a_bundled_mtp_head_is_the_base_type() {
        assert_eq!(canonical_model_type("Qwen3_6_MoE_MTP"), "qwen3_6_moe");
        assert_eq!(canonical_model_type("llama"), "llama");
    }

    #[test]
    fn classes_are_read_as_written() {
        let config = json!({"architectures": ["LlamaForCausalLM", 7, "X"]});
        assert_eq!(config_architectures(&config), ["LlamaForCausalLM", "X"]);
        assert!(config_architectures(&json!({})).is_empty());
    }

    #[test]
    fn index_topk_is_found_on_the_wrapper_or_the_tower() {
        assert_eq!(index_topk(&json!({"index_topk": 2048})), Some(2048));
        assert_eq!(
            index_topk(&json!({"text_config": {"index_topk": 512}})),
            Some(512)
        );
        assert_eq!(index_topk(&json!({"model_type": "deepseek_v3"})), None);
    }

    #[test]
    fn a_refusal_names_its_family_by_type_or_by_class() {
        let by_type = unsupported_family("qwen3_vl", &[]).expect("refused");
        assert_eq!(by_type.family, "Qwen3-VL");
        let by_class = unsupported_family("a_custom_type", &["DeepseekV41ForCausalLM".to_string()])
            .expect("refused");
        assert_eq!(by_class.family, "DeepSeek V4 / V4.1");
        assert!(unsupported_family("llama", &["LlamaForCausalLM".to_string()]).is_none());
        // A refused type wins over a supported-looking class.
        assert_eq!(
            unsupported_family("kimi_k2", &["DeepseekV3ForCausalLM".to_string()])
                .expect("refused")
                .family,
            "Kimi K2"
        );
    }

    #[test]
    fn no_type_or_class_is_listed_twice() {
        let mut seen = std::collections::HashMap::new();
        for (family, model_type) in refused_model_types() {
            if let Some(other) = seen.insert(model_type, family) {
                assert_eq!(other, family, "{model_type} is in two refusal rows");
            }
        }
        let mut classes = std::collections::HashMap::new();
        for (family, class) in refused_classes() {
            assert_eq!(class, class.to_lowercase(), "{class} must be lower case");
            if let Some(other) = classes.insert(class, family) {
                assert_eq!(other, family, "{class} is in two refusal rows");
            }
        }
    }
}
