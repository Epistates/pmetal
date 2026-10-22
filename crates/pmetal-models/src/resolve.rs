//! Which architecture a `config.json` names, exactly, and the error when none.
//!
//! A `model_type` maps to one [`ModelArchitecture`] or to none. There is no
//! "close enough": a name that merely contains `qwen`, `mistral` or `deepseek`
//! used to run as that family, so a Qwen3-VL, a Mistral3 wrapper or a DeepSeek
//! V4 loaded into a decoder that is not theirs, and either failed the weight
//! audit with a count of tensors nothing consumed, or loaded and computed
//! something else. A family pmetal does not implement is now declined by name
//! ([`UnsupportedModel`]), from the table in [`pmetal_core::model_type`].
//!
//! The `architectures` class names are the fallback for a config that states
//! no `model_type` pmetal knows, and they are compared whole, not searched.

use crate::dispatcher::ModelArchitecture;
use pmetal_core::model_type::{
    canonical_model_type, config_architectures, effective_model_type, sparse_deepseek_v3,
    unsupported_family,
};
use pmetal_runtime::compat::Exception;

/// A checkpoint of a family pmetal does not run.
///
/// Names the `model_type` (and the classes, when it states no usable type) so
/// the user can see what was read, says it is not supported yet, and lists the
/// families that are. A family pmetal knows it declines carries its name and
/// the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedModel {
    /// The canonical `model_type` the config states; empty when it states none.
    pub model_type: String,
    /// The `architectures` class names, as written.
    pub architectures: Vec<String>,
    /// The declined family, when pmetal knows it by name.
    pub family: Option<&'static str>,
    /// Why the family is declined, when it is.
    pub reason: Option<&'static str>,
}

impl std::fmt::Display for UnsupportedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let classes = if self.architectures.is_empty() {
            String::new()
        } else {
            format!(" (architectures: {})", self.architectures.join(", "))
        };
        match (self.model_type.is_empty(), self.family, self.reason) {
            (true, None, _) if self.architectures.is_empty() => write!(
                f,
                "the config.json states neither a model_type nor an architectures list, so the \
                 model family cannot be identified."
            )?,
            (_, Some(family), Some(reason)) => write!(
                f,
                "model_type {:?}{classes} is {family}, which is not supported yet: {reason}.",
                self.model_type
            )?,
            _ => write!(
                f,
                "model_type {:?}{classes} is not supported yet: pmetal has no loader for this \
                 architecture.",
                self.model_type
            )?,
        }
        write!(
            f,
            " Supported families: {}.",
            ModelArchitecture::supported_summary()
        )
    }
}

impl std::error::Error for UnsupportedModel {}

impl From<UnsupportedModel> for Exception {
    fn from(e: UnsupportedModel) -> Self {
        Exception::custom(e.to_string())
    }
}

/// Every `model_type` that selects an architecture, canonical spelling. Some
/// are not Hub types but the names other tools write for the same thing (a GGUF
/// `general.architecture`, an older spelling); each maps to the one
/// architecture and is an alias, not a guess.
const MODEL_TYPES: &[(&str, ModelArchitecture)] = &[
    ("llama4", ModelArchitecture::Llama4),
    ("llama4_text", ModelArchitecture::Llama4),
    // Mllama's text sub-config is `mllama_text_model`; plain Llama would drop
    // every cross-attention layer.
    ("mllama", ModelArchitecture::Mllama),
    ("mllama_text_model", ModelArchitecture::Mllama),
    ("llama", ModelArchitecture::Llama),
    ("llama3", ModelArchitecture::Llama),
    ("qwen3_moe", ModelArchitecture::Qwen3MoE),
    ("gpt_oss", ModelArchitecture::GptOss),
    ("gptoss", ModelArchitecture::GptOss),
    ("gpt-oss", ModelArchitecture::GptOss),
    ("qwen3_next", ModelArchitecture::Qwen3Next),
    ("qwen3_5", ModelArchitecture::Qwen3Next),
    ("qwen3.5", ModelArchitecture::Qwen3Next),
    ("qwen3_5_text", ModelArchitecture::Qwen3Next),
    ("qwen3_5_moe", ModelArchitecture::Qwen3Next),
    ("qwen3_5_moe_text", ModelArchitecture::Qwen3Next),
    ("qwen3_6", ModelArchitecture::Qwen3Next),
    ("qwen3.6", ModelArchitecture::Qwen3Next),
    ("qwen3_6_text", ModelArchitecture::Qwen3Next),
    ("qwen3_6_moe", ModelArchitecture::Qwen3Next),
    ("qwen3_6_moe_text", ModelArchitecture::Qwen3Next),
    ("qwen4_exp", ModelArchitecture::Qwen4Exp),
    ("qwen4_exp_text", ModelArchitecture::Qwen4Exp),
    ("qwen3", ModelArchitecture::Qwen3),
    ("qwen2", ModelArchitecture::Qwen2),
    ("qwen2_5", ModelArchitecture::Qwen2),
    ("gemma", ModelArchitecture::Gemma),
    ("gemma2", ModelArchitecture::Gemma),
    ("gemma3", ModelArchitecture::Gemma),
    ("gemma3_text", ModelArchitecture::Gemma),
    // Gemma 4 has its own module (per-layer-type attention, `k_eq_v` on full
    // attention, layer scalar, final logit softcapping). Multimodal wrappers
    // nest the text backbone under `text_config`; the loader unwraps it.
    ("gemma4", ModelArchitecture::Gemma4),
    ("gemma4_text", ModelArchitecture::Gemma4),
    ("gemma4_unified", ModelArchitecture::Gemma4),
    ("gemma4_unified_text", ModelArchitecture::Gemma4),
    ("diffusion_gemma", ModelArchitecture::DiffusionGemma),
    ("diffusion_gemma_text", ModelArchitecture::DiffusionGemma),
    ("mistral", ModelArchitecture::Mistral),
    ("phi4", ModelArchitecture::Phi4),
    // `ModelArchitecture::Phi` is a Phi-3 implementation. `model_type: "phi"`
    // is Phi-1 / Phi-1.5 / Phi-2, which share none of it (parallel block,
    // LayerNorm with bias, a single GELU MLP) and are declined by name.
    ("phi3", ModelArchitecture::Phi),
    ("deepseek_v2", ModelArchitecture::DeepSeek),
    ("deepseek_v3", ModelArchitecture::DeepSeek),
    // The GGUF spelling of V2 and V3.
    ("deepseek2", ModelArchitecture::DeepSeek),
    ("cohere", ModelArchitecture::Cohere),
    ("cohere2", ModelArchitecture::Cohere),
    ("command_r", ModelArchitecture::Cohere),
    ("command-r", ModelArchitecture::Cohere),
    // The four families `GraniteForCausalLM` computes; every other `granite*`
    // type is declined by name.
    ("granite", ModelArchitecture::Granite),
    ("granitemoe", ModelArchitecture::Granite),
    ("granitemoeshared", ModelArchitecture::Granite),
    ("granitemoehybrid", ModelArchitecture::Granite),
    ("nemotron_h", ModelArchitecture::NemotronH),
    ("nemotronh", ModelArchitecture::NemotronH),
    ("nemotron-h", ModelArchitecture::NemotronH),
    ("flux", ModelArchitecture::Flux),
    ("flux-1", ModelArchitecture::Flux),
    ("flux.1", ModelArchitecture::Flux),
    // The encoders `EncoderModel` computes: the BERT family and ModernBERT.
    // ALBERT and DeBERTa are different architectures and are declined.
    ("bert", ModelArchitecture::Bert),
    ("roberta", ModelArchitecture::Bert),
    ("xlm-roberta", ModelArchitecture::Bert),
    ("xlm_roberta", ModelArchitecture::Bert),
    ("camembert", ModelArchitecture::Bert),
    ("distilbert", ModelArchitecture::Bert),
    ("modernbert", ModelArchitecture::Bert),
];

/// Every class name that selects an architecture, lower case, whole.
const CLASSES: &[(&str, ModelArchitecture)] = &[
    ("llamaforcausallm", ModelArchitecture::Llama),
    ("llama4forcausallm", ModelArchitecture::Llama4),
    ("llama4forconditionalgeneration", ModelArchitecture::Llama4),
    ("mllamaforconditionalgeneration", ModelArchitecture::Mllama),
    ("mllamaforcausallm", ModelArchitecture::Mllama),
    ("qwen2forcausallm", ModelArchitecture::Qwen2),
    ("qwen3forcausallm", ModelArchitecture::Qwen3),
    ("qwen3moeforcausallm", ModelArchitecture::Qwen3MoE),
    ("qwen3nextforcausallm", ModelArchitecture::Qwen3Next),
    (
        "qwen3_5forconditionalgeneration",
        ModelArchitecture::Qwen3Next,
    ),
    ("qwen3_5forcausallm", ModelArchitecture::Qwen3Next),
    ("qwen3_5model", ModelArchitecture::Qwen3Next),
    (
        "qwen3_5moeforconditionalgeneration",
        ModelArchitecture::Qwen3Next,
    ),
    ("qwen3_5moeforcausallm", ModelArchitecture::Qwen3Next),
    (
        "qwen3_6forconditionalgeneration",
        ModelArchitecture::Qwen3Next,
    ),
    ("qwen3_6forcausallm", ModelArchitecture::Qwen3Next),
    (
        "qwen3_6moeforconditionalgeneration",
        ModelArchitecture::Qwen3Next,
    ),
    ("qwen3_6moeforcausallm", ModelArchitecture::Qwen3Next),
    (
        "qwen4expforconditionalgeneration",
        ModelArchitecture::Qwen4Exp,
    ),
    ("qwen4expforcausallm", ModelArchitecture::Qwen4Exp),
    ("gemmaforcausallm", ModelArchitecture::Gemma),
    ("gemma2forcausallm", ModelArchitecture::Gemma),
    ("gemma3forcausallm", ModelArchitecture::Gemma),
    ("gemma3forconditionalgeneration", ModelArchitecture::Gemma),
    ("gemma4forcausallm", ModelArchitecture::Gemma4),
    ("gemma4forconditionalgeneration", ModelArchitecture::Gemma4),
    (
        "gemma4unifiedforconditionalgeneration",
        ModelArchitecture::Gemma4,
    ),
    (
        "diffusiongemmaforblockdiffusion",
        ModelArchitecture::DiffusionGemma,
    ),
    ("mistralforcausallm", ModelArchitecture::Mistral),
    ("phi3forcausallm", ModelArchitecture::Phi),
    ("deepseekv2forcausallm", ModelArchitecture::DeepSeek),
    ("deepseekv3forcausallm", ModelArchitecture::DeepSeek),
    ("cohereforcausallm", ModelArchitecture::Cohere),
    ("cohere2forcausallm", ModelArchitecture::Cohere),
    ("graniteforcausallm", ModelArchitecture::Granite),
    ("granitemoeforcausallm", ModelArchitecture::Granite),
    ("granitemoesharedforcausallm", ModelArchitecture::Granite),
    ("granitemoehybridforcausallm", ModelArchitecture::Granite),
    ("gptossforcausallm", ModelArchitecture::GptOss),
    ("nemotronhforcausallm", ModelArchitecture::NemotronH),
    ("fluxtransformer2dmodel", ModelArchitecture::Flux),
    ("fluxpipeline", ModelArchitecture::Flux),
];

/// The encoder class names: a family prefix and a task head, which the Hub
/// spells as `BertForMaskedLM`, `RobertaModel`, `XLMRobertaForTokenClassification`
/// and so on. Both halves are listed, so `AlbertModel` and `DebertaModel`, which
/// end the same way, are not matched.
const ENCODER_PREFIXES: &[&str] = &[
    "bert",
    "roberta",
    "xlmroberta",
    "camembert",
    "distilbert",
    "modernbert",
];
const ENCODER_HEADS: &[&str] = &[
    "model",
    "formaskedlm",
    "forsequenceclassification",
    "fortokenclassification",
    "forquestionanswering",
    "formultiplechoice",
    "forpretraining",
    "fornextsentenceprediction",
    "lmheadmodel",
];

fn is_encoder_class(lower: &str) -> bool {
    ENCODER_PREFIXES.iter().any(|prefix| {
        lower
            .strip_prefix(prefix)
            .is_some_and(|head| ENCODER_HEADS.contains(&head))
    })
}

impl ModelArchitecture {
    /// Every architecture the dispatcher routes to.
    pub const ALL: [Self; 21] = [
        Self::Llama,
        Self::Llama4,
        Self::Qwen2,
        Self::Qwen3,
        Self::Qwen3MoE,
        Self::Gemma,
        Self::Mistral,
        Self::Phi,
        Self::Phi4,
        Self::DeepSeek,
        Self::Cohere,
        Self::Granite,
        Self::NemotronH,
        Self::Qwen3Next,
        Self::GptOss,
        Self::Gemma4,
        Self::Flux,
        Self::Bert,
        Self::DiffusionGemma,
        Self::Mllama,
        Self::Qwen4Exp,
    ];

    /// The architecture a `model_type` names, or `None`. Exact: the type is
    /// lower-cased and a bundled `_mtp` suffix removed, and nothing else.
    ///
    /// Callers that have to tell the user why a checkpoint is refused want
    /// [`resolve`](Self::resolve), which says so by name.
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        let canonical = canonical_model_type(model_type);
        MODEL_TYPES
            .iter()
            .find(|(name, _)| *name == canonical)
            .map(|(_, arch)| *arch)
    }

    /// The architecture the first recognised class name of an `architectures`
    /// array names, or `None`. Class names are compared whole (case
    /// insensitively); a name that merely contains `llama` or `qwen` is not a
    /// match.
    pub fn from_architectures(archs: &[String]) -> Option<Self> {
        archs.iter().find_map(|class| {
            let lower = class.to_lowercase();
            CLASSES
                .iter()
                .find(|(name, _)| *name == lower)
                .map(|(_, arch)| *arch)
                .or_else(|| is_encoder_class(&lower).then_some(Self::Bert))
        })
    }

    /// The architecture a checkpoint's `model_type` and `architectures` name,
    /// or the error that says what it is and that pmetal does not run it.
    ///
    /// The `model_type` decides: a type of a supported family selects it, and a
    /// type of a declined one is declined, whatever the class names say (Kimi K2
    /// declares `DeepseekV3ForCausalLM`). Only a type pmetal has never heard of
    /// falls back to the class names, which are matched whole.
    ///
    /// This sees no `index_topk`, so it cannot tell a DeepSeek-V3.2 that reuses
    /// the `deepseek_v3` type; [`resolve_config`](Self::resolve_config) can.
    pub fn resolve(model_type: &str, architectures: &[String]) -> Result<Self, UnsupportedModel> {
        let canonical = canonical_model_type(model_type);
        if let Some(arch) = Self::from_model_type(&canonical) {
            return Ok(arch);
        }
        let refused = |family: pmetal_core::model_type::UnsupportedFamily| UnsupportedModel {
            model_type: canonical.clone(),
            architectures: architectures.to_vec(),
            family: Some(family.family),
            reason: Some(family.reason),
        };
        if let Some(family) = unsupported_family(&canonical, &[]) {
            return Err(refused(family));
        }
        if let Some(arch) = Self::from_architectures(architectures) {
            return Ok(arch);
        }
        match unsupported_family("", architectures) {
            Some(family) => Err(refused(family)),
            None => Err(UnsupportedModel {
                model_type: canonical,
                architectures: architectures.to_vec(),
                family: None,
                reason: None,
            }),
        }
    }

    /// [`resolve`](Self::resolve) a parsed `config.json`: its effective
    /// `model_type` (the top level's, see
    /// [`effective_model_type`]), its `architectures`, and its `index_topk`.
    ///
    /// A `deepseek_v3` config with an `index_topk` is DeepSeek-V3.2's sparse
    /// attention under the older type, and is declined as that.
    pub fn resolve_config(config: &serde_json::Value) -> Result<Self, UnsupportedModel> {
        let model_type = effective_model_type(config).unwrap_or_default();
        let architectures = config_architectures(config);
        let arch = Self::resolve(&model_type, &architectures)?;
        if arch == Self::DeepSeek
            && let Some(family) = sparse_deepseek_v3(&model_type, config)
        {
            return Err(UnsupportedModel {
                model_type,
                architectures,
                family: Some(family.family),
                reason: Some(family.reason),
            });
        }
        Ok(arch)
    }

    /// The architecture's own `model_type`, for the list of what is supported.
    fn primary_model_type(self) -> &'static str {
        MODEL_TYPES
            .iter()
            .find(|(_, arch)| *arch == self)
            .map_or("", |(name, _)| name)
    }

    /// The supported families as one sentence fragment: `Llama [llama], Llama 4
    /// [llama4], ...`.
    pub fn supported_summary() -> String {
        Self::ALL
            .iter()
            .map(|arch| format!("{arch} [{}]", arch.primary_model_type()))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn archs(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn every_architecture_is_routable_and_listed() {
        for arch in ModelArchitecture::ALL {
            assert!(
                MODEL_TYPES.iter().any(|(_, a)| *a == arch),
                "{arch:?} has no model_type"
            );
        }
        // `ALL` misses a variant if this match stops compiling.
        for arch in ModelArchitecture::ALL {
            match arch {
                ModelArchitecture::Llama
                | ModelArchitecture::Llama4
                | ModelArchitecture::Qwen2
                | ModelArchitecture::Qwen3
                | ModelArchitecture::Qwen3MoE
                | ModelArchitecture::Gemma
                | ModelArchitecture::Mistral
                | ModelArchitecture::Phi
                | ModelArchitecture::Phi4
                | ModelArchitecture::DeepSeek
                | ModelArchitecture::Cohere
                | ModelArchitecture::Granite
                | ModelArchitecture::NemotronH
                | ModelArchitecture::Qwen3Next
                | ModelArchitecture::GptOss
                | ModelArchitecture::Gemma4
                | ModelArchitecture::Flux
                | ModelArchitecture::Bert
                | ModelArchitecture::DiffusionGemma
                | ModelArchitecture::Mllama
                | ModelArchitecture::Qwen4Exp => {}
            }
        }
        let distinct: std::collections::HashSet<_> = ModelArchitecture::ALL
            .iter()
            .map(|arch| arch.identifier())
            .collect();
        assert_eq!(distinct.len(), ModelArchitecture::ALL.len());
    }

    #[test]
    fn no_type_or_class_is_both_supported_and_refused() {
        for (family, model_type) in pmetal_core::model_type::refused_model_types() {
            assert!(
                MODEL_TYPES.iter().all(|(name, _)| *name != model_type),
                "{model_type} ({family}) is refused and routed"
            );
        }
        for (family, class) in pmetal_core::model_type::refused_classes() {
            assert!(
                CLASSES.iter().all(|(name, _)| *name != class) && !is_encoder_class(class),
                "{class} ({family}) is refused and routed"
            );
        }
        // Table entries are in the spelling the lookups compare.
        for (name, _) in MODEL_TYPES {
            assert_eq!(*name, canonical_model_type(name), "{name}");
        }
        for (name, _) in CLASSES {
            assert_eq!(*name, name.to_lowercase(), "{name}");
        }
    }

    /// A type is a lookup, not a search: nothing that merely contains a
    /// supported family's name is routed to it.
    #[test]
    fn a_name_containing_a_family_is_not_that_family() {
        for model_type in [
            "qwen3_vl",
            "qwen2_moe",
            "mistral3",
            "mixtral",
            "deepseek_v4",
            "deepseek_v41",
            "gemma3n",
            "phi3_v",
            "llama_custom",
            "my_qwen3",
            "cohere2_moe",
        ] {
            assert_eq!(
                ModelArchitecture::from_model_type(model_type),
                None,
                "{model_type}"
            );
        }
        for class in [
            "Qwen3VLForConditionalGeneration",
            "Mistral3ForConditionalGeneration",
            "LlavaLlamaForCausalLM",
            "MemTQwen3_5MoeForCausalLM",
            "Phi3VForCausalLM",
            "EmbeddingGemma2Model",
            "Cohere2MoeForCausalLM",
            "AlbertModel",
            "DebertaModel",
            "FluxKontextPipeline",
        ] {
            assert_eq!(
                ModelArchitecture::from_architectures(&archs(&[class])),
                None,
                "{class}"
            );
        }
    }

    #[test]
    fn the_exact_class_names_still_route() {
        for (class, arch) in [
            ("LlamaForCausalLM", ModelArchitecture::Llama),
            (
                "Qwen3_5MoeForConditionalGeneration",
                ModelArchitecture::Qwen3Next,
            ),
            ("Gemma3ForCausalLM", ModelArchitecture::Gemma),
            ("BertForMaskedLM", ModelArchitecture::Bert),
            ("XLMRobertaModel", ModelArchitecture::Bert),
            (
                "DistilBertForSequenceClassification",
                ModelArchitecture::Bert,
            ),
            ("ModernBertModel", ModelArchitecture::Bert),
            ("GraniteMoeHybridForCausalLM", ModelArchitecture::Granite),
        ] {
            assert_eq!(
                ModelArchitecture::from_architectures(&archs(&[class])),
                Some(arch),
                "{class}"
            );
        }
    }

    #[test]
    fn a_refused_family_is_named_with_its_model_type_and_the_supported_list() {
        let err =
            ModelArchitecture::resolve("qwen3_vl", &archs(&["Qwen3VLForConditionalGeneration"]))
                .expect_err("refused");
        let text = err.to_string();
        assert!(text.contains("\"qwen3_vl\""), "{text}");
        assert!(text.contains("Qwen3-VL"), "{text}");
        assert!(text.contains("not supported yet"), "{text}");
        assert!(text.contains("Supported families:"), "{text}");
        assert!(text.contains("Llama [llama]"), "{text}");
        assert!(text.contains("Qwen 3 [qwen3]"), "{text}");
    }

    #[test]
    fn an_unknown_type_is_named_and_not_guessed() {
        let err = ModelArchitecture::resolve("flibbertigibbet", &archs(&["FlibbertForCausalLM"]))
            .expect_err("unknown");
        assert_eq!(err.family, None);
        let text = err.to_string();
        assert!(text.contains("\"flibbertigibbet\""), "{text}");
        assert!(text.contains("FlibbertForCausalLM"), "{text}");
        assert!(text.contains("not supported yet"), "{text}");
        assert!(text.contains("Supported families:"), "{text}");
    }

    #[test]
    fn a_config_with_no_identity_says_so() {
        let err = ModelArchitecture::resolve_config(&json!({"hidden_size": 8}))
            .expect_err("nothing to go on");
        assert!(err.to_string().contains("neither a model_type"), "{err}");
    }

    #[test]
    fn the_model_type_decides_over_the_class_names() {
        // Kimi K2 declares the DeepSeek-V3 class and is not DeepSeek-V3.
        let kimi = json!({"model_type": "kimi_k2", "architectures": ["DeepseekV3ForCausalLM"]});
        let err = ModelArchitecture::resolve_config(&kimi).expect_err("refused");
        assert_eq!(err.family, Some("Kimi K2"));
        // A wrapper is its own type, not the tower inside it.
        let wrapped = json!({
            "model_type": "kimi_k25",
            "architectures": ["KimiK25ForConditionalGeneration"],
            "text_config": {"model_type": "kimi_k2"},
        });
        assert_eq!(
            ModelArchitecture::resolve_config(&wrapped)
                .expect_err("refused")
                .model_type,
            "kimi_k25"
        );
        // An unknown type with a class pmetal knows whole still resolves.
        let custom = json!({"model_type": "my_finetune", "architectures": ["LlamaForCausalLM"]});
        assert_eq!(
            ModelArchitecture::resolve_config(&custom).unwrap(),
            ModelArchitecture::Llama
        );
    }

    #[test]
    fn deepseek_v3_with_a_sparse_indexer_is_v3_2() {
        let dense =
            json!({"model_type": "deepseek_v3", "architectures": ["DeepseekV3ForCausalLM"]});
        assert_eq!(
            ModelArchitecture::resolve_config(&dense).unwrap(),
            ModelArchitecture::DeepSeek
        );
        let sparse = json!({
            "model_type": "deepseek_v3",
            "architectures": ["DeepseekV3ForCausalLM"],
            "index_topk": 2048,
        });
        let err = ModelArchitecture::resolve_config(&sparse).expect_err("refused");
        assert_eq!(err.family, Some("DeepSeek V3.2"));
        assert!(err.to_string().contains("index_topk"), "{err}");
    }

    #[test]
    fn a_bundled_mtp_head_runs_as_its_target() {
        assert_eq!(
            ModelArchitecture::resolve("qwen3_5_mtp", &[]).unwrap(),
            ModelArchitecture::Qwen3Next
        );
    }
}
