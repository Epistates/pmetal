//! Dynamic model dispatch based on config.json model_type.
//!
//! This module provides automatic architecture detection and model loading,
//! eliminating the need for hardcoded model types in application code.

use crate::architectures::*;
use crate::generation::{GenerationConfig, GenerationOutput};
use crate::loader::{
    Exemption, LoadPolicy, ParamSnapshot, Qwen3NextLoadOptions, assign_weights,
    load_encoder_weights, load_generic_weights_with, load_qwen3_next_weights_with_options,
    load_weights,
};
use crate::traits::{CausalLMModel, ModelConfig};
use crate::weight_format::{GgufModelConfig, WeightFormat, WeightLoader};
use pmetal_cache::kv_cache::{
    CacheMode, FusedBatchKVCache, KVCache, KVCacheConfig, MambaCache,
    sanitize_cache_mode_for_config,
};
use pmetal_runtime::compat::{
    Array, Dtype, Exception, Module, ModuleParameters, ModuleParametersExt, nn, ops,
};
use pmetal_runtime::decode::last_token_hidden;
use std::path::Path;

const PARAM_EVAL_BATCH_SIZE: usize = 128;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DynamicModelLoadOptions {
    pub prefer_expert_offload: bool,
}

fn eval_module_parameters_batched(module: &impl ModuleParametersExt) -> Result<(), Exception> {
    let params = module.flatten_params();
    let arrays: Vec<Array> = params.values().cloned().collect();

    for chunk in arrays.chunks(PARAM_EVAL_BATCH_SIZE) {
        pmetal_runtime::compat::transforms::eval(chunk.iter())?;
    }

    Ok(())
}

/// Run an architecture's own weight loader on `model`, then fail unless every
/// parameter was assigned (bar what `policy` exempts).
///
/// The generic loader audits itself; this is for the architectures that fill
/// their parameters by hand. A parameter such a loader forgets keeps the value
/// it was constructed with, which looks like a model and computes noise.
fn load_checked<M: ModuleParametersExt, T>(
    model: &mut M,
    model_dir: &Path,
    policy: &LoadPolicy,
    load: impl FnOnce(&mut M) -> Result<T, Exception>,
) -> Result<T, Exception> {
    let snapshot = ParamSnapshot::of(model);
    let out = load(model)?;
    snapshot
        .require_complete(model, policy, Vec::new())
        .map_err(|e| Exception::custom(e.with_quantization_hint(model_dir).to_string()))?;
    Ok(out)
}

/// Load a checkpoint into a routed-expert model: its experts stacked into the
/// `mlp.switch_mlp` layout the model holds
/// ([`stack_experts`](crate::loader::experts::stack_experts)), then assigned
/// under `policy`, which fails unless checkpoint and model account for each
/// other.
fn load_with_stacked_experts<M: ModuleParametersExt>(
    family: &str,
    model: &mut M,
    model_dir: &Path,
    policy: &LoadPolicy,
) -> Result<(), Exception> {
    let weights = load_weights(model_dir).map_err(|e| Exception::custom(e.to_string()))?;
    let weights = crate::loader::experts::stack_experts(family, weights)?;
    assign_weights(model, weights, policy)
        .map_err(|e| Exception::custom(e.with_quantization_hint(model_dir).to_string()))
}

/// Rewrite the bare `Infinity` / `NaN` literals some released configs contain
/// into `null`.
///
/// Python's `json.dump` emits `Infinity` and `NaN` unquoted, which round-trips
/// through Python but is not valid JSON — `serde_json` rejects it, and
/// `serde_json::Value` could not hold the value anyway (`Number::from_f64`
/// refuses non-finite floats). `nvidia/Nemotron-H-8B-Base-8K` ships
/// `"time_step_limit": [0.0, Infinity]`, which was enough to make every
/// Nemotron-H checkpoint undetectable: `detect` failed on the JSON parse before
/// it ever reached the architecture's own `json5` deserialization, which
/// accepts the literal and keeps the real value.
///
/// Only the `serde_json::Value` view is affected, and that view is read for
/// `model_type`, `architectures`, and `text_config` — never for a numeric
/// field — so flattening these to `null` loses nothing.
fn sanitize_non_finite(src: &str) -> String {
    const TOKENS: [&str; 4] = ["-Infinity", "Infinity", "-NaN", "NaN"];
    let mut out = String::with_capacity(src.len());
    let mut chars = src.char_indices();
    let mut in_string = false;
    let mut escaped = false;
    while let Some((idx, ch)) = chars.next() {
        if in_string {
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
            out.push(ch);
            continue;
        }
        if let Some(token) = TOKENS.into_iter().find(|t| src[idx..].starts_with(t)) {
            out.push_str("null");
            for _ in 1..token.len() {
                chars.next();
            }
            continue;
        }
        out.push(ch);
    }
    out
}

/// Parse a `config.json` into a `serde_json::Value`, tolerating the non-finite
/// literals described on `sanitize_non_finite`.
///
/// Strict parsing is tried first so a well-formed config never pays for the
/// rewrite, and so a genuinely malformed one still reports its original error.
///
/// Public because anything reading a released config needs this tolerance, not
/// just the loader — `serde_json::from_str` on a Nemotron-H config fails, and
/// failing that way looks like a corrupt file rather than a dialect mismatch.
pub fn config_value(config_content: &str) -> Result<serde_json::Value, Exception> {
    match serde_json::from_str(config_content) {
        Ok(value) => Ok(value),
        Err(strict_err) => serde_json::from_str(&sanitize_non_finite(config_content))
            .map_err(|_| Exception::custom(strict_err.to_string())),
    }
}

/// Unwrap a multimodal wrapper config down to its text tower.
///
/// Llama 4, Gemma 3/4, Qwen 3.5 and Gemma 4 all ship released configs where the
/// language-model fields live under `text_config` and the top level carries
/// only the wrapper (vision config, projector, token ids). The guard is
/// `text_config` present *and* no top-level `hidden_size`, so a text-only
/// checkpoint of the same family passes through byte-identical.
///
/// This was copy-pasted into four `load_with_options` arms; it is shared so
/// [`ModelArchitecture::parse_config_json`] cannot disagree with the loader
/// about what a wrapper config means.
fn unwrap_text_config(config_content: &str) -> Result<std::borrow::Cow<'_, str>, Exception> {
    let config_json = config_value(config_content)?;
    if config_json.get("text_config").is_some() && config_json.get("hidden_size").is_none() {
        // `tie_word_embeddings` of a wrapper sits beside `text_config`, and
        // governs the text model's head when the text config says nothing: a
        // model that read only the inner config would build an untied head the
        // checkpoint has no tensor for.
        let mut text = config_json["text_config"].clone();
        if let (Some(obj), Some(tie)) =
            (text.as_object_mut(), config_json.get("tie_word_embeddings"))
        {
            obj.entry("tie_word_embeddings")
                .or_insert_with(|| tie.clone());
        }
        let inner = serde_json::to_string(&text).map_err(|e| Exception::custom(e.to_string()))?;
        Ok(std::borrow::Cow::Owned(inner))
    } else {
        Ok(std::borrow::Cow::Borrowed(config_content))
    }
}

/// Model architecture types supported by PMetal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ModelArchitecture {
    Llama,
    Llama4,
    Qwen2,
    Qwen3,
    Qwen3MoE,
    Gemma,
    Mistral,
    Phi,
    Phi4,
    DeepSeek,
    Cohere,
    Granite,
    NemotronH,
    Qwen3Next,
    GptOss,
    Gemma4,
    Flux,
    /// BERT / RoBERTa / DistilBERT encoder-only model.
    Bert,
    /// DiffusionGemma block-autoregressive discrete-diffusion encoder–decoder LM.
    DiffusionGemma,
    /// Mllama (Llama 3.2 Vision) — tiled vision tower + cross-attending text decoder.
    Mllama,
    /// Qwen4-Exp (Qwen3.8-Flash-Next): hybrid GDN / indexed attention with
    /// hyper-connections and PLE n-gram embeddings. Text tower.
    Qwen4Exp,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Llama => write!(f, "Llama"),
            Self::Llama4 => write!(f, "Llama 4"),
            Self::Qwen2 => write!(f, "Qwen 2"),
            Self::Qwen3 => write!(f, "Qwen 3"),
            Self::Qwen3MoE => write!(f, "Qwen 3 MoE"),
            Self::Gemma => write!(f, "Gemma"),
            Self::Mistral => write!(f, "Mistral"),
            Self::Phi => write!(f, "Phi"),
            Self::Phi4 => write!(f, "Phi 4"),
            Self::DeepSeek => write!(f, "DeepSeek"),
            Self::Cohere => write!(f, "Cohere"),
            Self::Granite => write!(f, "Granite"),
            Self::NemotronH => write!(f, "NemotronH"),
            Self::Qwen3Next => write!(f, "Qwen 3.5 / 3.6"),
            Self::GptOss => write!(f, "GPT-OSS"),
            Self::Gemma4 => write!(f, "Gemma 4"),
            Self::Flux => write!(f, "Flux"),
            Self::Bert => write!(f, "BERT"),
            Self::DiffusionGemma => write!(f, "DiffusionGemma"),
            Self::Mllama => write!(f, "Llama 3.2 Vision (Mllama)"),
            Self::Qwen4Exp => write!(f, "Qwen4-Exp (Qwen3.8-Flash-Next)"),
        }
    }
}

impl ModelArchitecture {
    /// What a load of this architecture may leave unmatched, beyond the buffers
    /// no architecture takes from a checkpoint.
    ///
    /// Every exemption is a claim that the model is right without the tensor
    /// or parameter, so each says why. The default is nothing: a checkpoint
    /// that does not fit the model it names is an error, not a model with a
    /// few random weights.
    pub fn load_policy(self) -> LoadPolicy {
        let policy = LoadPolicy::strict();
        match self {
            Self::NemotronH => policy.allowing_unused_tensors(&[Exemption::new(
                "mtp.*",
                "the multi-token-prediction head some Nemotron 3 releases bundle; the model \
                 does not run it",
            )]),
            _ => policy,
        }
    }

    /// The variant's name as a bare identifier, with no spacing or branding.
    ///
    /// Distinct from `Display`, which is prose for a human ("Qwen 3.5 / 3.6").
    /// Callers compare this against a literal, so a spelling here is part of
    /// the contract rather than a label.
    pub fn identifier(self) -> &'static str {
        match self {
            Self::Llama => "Llama",
            Self::Llama4 => "Llama4",
            Self::Qwen2 => "Qwen2",
            Self::Qwen3 => "Qwen3",
            Self::Qwen3MoE => "Qwen3MoE",
            Self::Gemma => "Gemma",
            Self::Mistral => "Mistral",
            Self::Phi => "Phi",
            Self::Phi4 => "Phi4",
            Self::DeepSeek => "DeepSeek",
            Self::Cohere => "Cohere",
            Self::Granite => "Granite",
            Self::NemotronH => "NemotronH",
            Self::Qwen3Next => "Qwen3Next",
            Self::GptOss => "GptOss",
            Self::Gemma4 => "Gemma4",
            Self::Flux => "Flux",
            Self::Bert => "Bert",
            Self::DiffusionGemma => "DiffusionGemma",
            Self::Mllama => "Mllama",
            Self::Qwen4Exp => "Qwen4Exp",
        }
    }

    pub fn from_model_type(model_type: &str) -> Option<Self> {
        let lower = model_type.to_lowercase();
        // A `*_mtp` checkpoint is the base architecture plus bundled
        // multi-token-prediction layers, which `--mtp` loads separately as a
        // speculative draft. `mlx-community/Qwen3.8-27B-MTP-4bit` states
        // `qwen3_5_mtp` and carries no `architectures` array, so without this
        // the whole checkpoint failed detection rather than running as its
        // own target model.
        let lower = lower.strip_suffix("_mtp").unwrap_or(&lower);
        match lower {
            "llama4" | "llama4_text" => Some(Self::Llama4),
            // Before the generic llama arms: Mllama's text sub-config is
            // `mllama_text_model`, which must not fall through to plain Llama
            // (it would drop every cross-attention layer).
            "mllama" | "mllama_text_model" => Some(Self::Mllama),
            "llama" | "llama3" => Some(Self::Llama),
            "qwen3_moe" => Some(Self::Qwen3MoE),
            "gpt_oss" | "gptoss" | "gpt-oss" => Some(Self::GptOss),
            "qwen3_next" | "qwen3_5" | "qwen3.5" | "qwen3_5_text" | "qwen3_5_moe"
            | "qwen3_5_moe_text" | "qwen3_6" | "qwen3.6" | "qwen3_6_text" | "qwen3_6_moe"
            | "qwen3_6_moe_text" => Some(Self::Qwen3Next),
            "qwen4_exp" | "qwen4_exp_text" => Some(Self::Qwen4Exp),
            "qwen3" => Some(Self::Qwen3),
            "qwen2" | "qwen2_5" => Some(Self::Qwen2),
            "gemma" | "gemma2" | "gemma3" => Some(Self::Gemma),
            // Gemma 4 has its own architecture module (separate attention
            // variants per layer type, k_eq_v for full-attention, layer
            // scalar, final logit softcapping). Multimodal wrappers nest
            // the text backbone under `text_config`; the loader unwraps.
            "gemma4" | "gemma4_text" | "gemma4_unified" => Some(Self::Gemma4),
            // DiffusionGemma must be matched before the generic gemma fallbacks
            // (its model_type is distinct, but keep it explicit).
            "diffusion_gemma" | "diffusion_gemma_text" => Some(Self::DiffusionGemma),
            "mistral" | "mixtral" => Some(Self::Mistral),
            "phi4" => Some(Self::Phi4),
            // `Self::Phi` is a Phi-3 implementation: `self_attn.o_proj`, a
            // fused `mlp.gate_up_proj` SwiGLU, RMSNorm, no biases, and a
            // sequential attention-then-MLP residual.
            //
            // `model_type: "phi"` is Phi-1 / Phi-1.5 / Phi-2, which share none
            // of that. They ship `self_attn.dense`, `mlp.fc1` + `mlp.fc2` with
            // a single GELU, LayerNorm *with bias*, a bias on every projection
            // and on `lm_head`, `model.final_layernorm` rather than
            // `model.norm`, no `post_attention_layernorm` at all, and a
            // parallel block: `x + attn(ln(x)) + mlp(ln(x))`.
            //
            // Routing them here matched almost no weights and returned noise
            // that was finite and correctly shaped. Declining is the honest
            // answer until the architecture exists; map this back the day it
            // does.
            "phi3" => Some(Self::Phi),
            "deepseek" | "deepseek2" | "deepseek_v2" | "deepseek_v3" => Some(Self::DeepSeek),
            "cohere" | "cohere2" | "command_r" | "command-r" => Some(Self::Cohere),
            // The four families `GraniteForCausalLM` computes. Every other
            // `granite*` model_type is refused by name; see `granite_refusal`.
            "granite" | "granitemoe" | "granitemoeshared" | "granitemoehybrid" => {
                Some(Self::Granite)
            }
            "nemotron_h" | "nemotronh" | "nemotron-h" => Some(Self::NemotronH),
            "flux" | "flux-1" | "flux.1" => Some(Self::Flux),
            // The encoders `EncoderModel` computes: the BERT family (BERT,
            // RoBERTa, DistilBERT) and ModernBERT. The BERT family shares the
            // architecture, not the config spelling, the checkpoint prefix or
            // the position ids (RoBERTa counts from its padding index); see
            // `BertConfig` and `loader::bert_param_name`. ALBERT and DeBERTa
            // are different architectures and stay refused.
            "bert" | "roberta" | "xlm-roberta" | "xlm_roberta" | "camembert" | "distilbert"
            | "modernbert" => Some(Self::Bert),
            _ => None,
        }
    }

    pub fn from_architectures(archs: &[String]) -> Option<Self> {
        for arch in archs {
            let lower = arch.to_lowercase();
            // Before the llama checks: `MllamaForConditionalGeneration`
            // contains "llama".
            if lower.contains("mllama") {
                return Some(Self::Mllama);
            }
            if lower.contains("llama4") {
                return Some(Self::Llama4);
            }
            if lower.contains("llama") {
                return Some(Self::Llama);
            }
            // Before the generic "qwen" fallback below, which would take it
            // for Qwen 2.
            if lower.contains("qwen4exp") || lower.contains("qwen4_exp") {
                return Some(Self::Qwen4Exp);
            }
            if lower.contains("qwen3moe") || lower.contains("qwen3_moe") {
                return Some(Self::Qwen3MoE);
            }
            if lower.contains("qwen3next")
                || lower.contains("qwen3_next")
                || lower.contains("qwen35")
                || lower.contains("qwen3_5")
                || lower.contains("qwen3.5")
                || lower.contains("qwen35moe")
                || lower.contains("qwen3_5_moe")
                || lower.contains("qwen36")
                || lower.contains("qwen3_6")
                || lower.contains("qwen3.6")
                || lower.contains("qwen36moe")
                || lower.contains("qwen3_6_moe")
            {
                return Some(Self::Qwen3Next);
            }
            if lower.contains("qwen3") {
                return Some(Self::Qwen3);
            }
            if lower.contains("qwen2") || lower.contains("qwen") {
                return Some(Self::Qwen2);
            }
            // DiffusionGemma before the generic gemma checks: its arch string
            // (`DiffusionGemmaForBlockDiffusion`) contains "gemma".
            if lower.contains("diffusiongemma") || lower.contains("diffusion_gemma") {
                return Some(Self::DiffusionGemma);
            }
            if lower.contains("gemma4assistant") || lower.contains("gemma4_assistant") {
                return None;
            }
            if lower.contains("gemma4") {
                return Some(Self::Gemma4);
            }
            if lower.contains("gemma") {
                return Some(Self::Gemma);
            }
            if lower.contains("mistral") || lower.contains("mixtral") {
                return Some(Self::Mistral);
            }
            if lower.contains("phi4") {
                return Some(Self::Phi4);
            }
            // Only Phi-3. A bare `PhiForCausalLM` is Phi-1 / Phi-1.5 / Phi-2,
            // a different architecture — see the `from_model_type` note.
            if lower.contains("phi3") {
                return Some(Self::Phi);
            }
            if lower.contains("deepseek") {
                return Some(Self::DeepSeek);
            }
            if lower.contains("cohere") || lower.contains("commandr") || lower.contains("command_r")
            {
                return Some(Self::Cohere);
            }
            // Exact class names only. A substring match sent every
            // `Granite*` class here, including the sliding-window, speech and
            // vision variants this does not compute.
            if matches!(
                lower.as_str(),
                "graniteforcausallm"
                    | "granitemoeforcausallm"
                    | "granitemoesharedforcausallm"
                    | "granitemoehybridforcausallm"
            ) {
                return Some(Self::Granite);
            }
            if lower.contains("gptoss") || lower.contains("gpt_oss") || lower.contains("gpt-oss") {
                return Some(Self::GptOss);
            }
            if lower.contains("nemotronhforcausallm") || lower.contains("nemotron_h") {
                return Some(Self::NemotronH);
            }
            if lower.contains("flux") {
                return Some(Self::Flux);
            }
            // The encoders `EncoderModel` computes, by class-name prefix
            // (`BertModel`, `BertForMaskedLM`, `RobertaForSequenceClassification`,
            // `XLMRobertaModel`, `DistilBertForMaskedLM`, `ModernBertModel`,
            // ...). Not a substring match: "albert" and "deberta" contain "bert".
            if [
                "bert",
                "roberta",
                "xlmroberta",
                "camembert",
                "distilbert",
                "modernbert",
            ]
            .iter()
            .any(|prefix| lower.starts_with(prefix))
            {
                return Some(Self::Bert);
            }
        }
        None
    }

    /// Deserialize a `config.json` into this architecture's config struct and
    /// hand back the round-tripped JSON, without reading a single weight.
    ///
    /// [`DynamicModel::load_with_options`] fuses config deserialization into
    /// the weight load, so the only way to discover whether pmetal can parse a
    /// released `config.json` used to be to download the whole checkpoint.
    /// That is how a stock Phi-2 config sat unparseable: nothing cheap ever
    /// tried it. This runs the same deserialization on the config alone, which
    /// lets a caller reject an unsupported checkpoint before paying for it —
    /// and lets `real_config_parity` hold every architecture against its real
    /// released config for the price of a few kilobytes.
    ///
    /// The returned value is `serde_json::to_value` of the parsed struct, so a
    /// caller can compare it field-by-field against the raw config. That
    /// comparison is the point: most of these structs are `#[serde(default)]`,
    /// so a field pmetal spells differently than the checkpoint does not fail
    /// to parse — it silently takes pmetal's default.
    pub fn parse_config_json(self, config_content: &str) -> Result<serde_json::Value, Exception> {
        /// Deserialize into `$ty` exactly as the matching `load` arm does, then
        /// re-serialize. `$content` is the (possibly text-config-unwrapped)
        /// source.
        macro_rules! round_trip {
            ($ty:ty, $content:expr) => {{
                let parsed: $ty =
                    json5::from_str($content).map_err(|e| Exception::custom(e.to_string()))?;
                serde_json::to_value(&parsed).map_err(|e| Exception::custom(e.to_string()))
            }};
        }

        let nested = unwrap_text_config(config_content)?;
        match self {
            Self::Llama => round_trip!(LlamaConfig, config_content),
            Self::Llama4 => round_trip!(Llama4TextConfig, &nested),
            Self::Qwen2 => round_trip!(Qwen2Config, config_content),
            Self::Qwen3 => round_trip!(Qwen3Config, config_content),
            Self::Qwen3MoE => round_trip!(Qwen3MoEConfig, config_content),
            Self::Gemma => round_trip!(GemmaConfig, &nested),
            Self::Mistral => round_trip!(MistralConfig, config_content),
            Self::Phi | Self::Phi4 => round_trip!(PhiConfig, config_content),
            Self::DeepSeek => round_trip!(DeepSeekConfig, config_content),
            Self::Cohere => round_trip!(CohereConfig, config_content),
            Self::Granite => serde_json::to_value(GraniteConfig::from_config_json(config_content)?)
                .map_err(|e| Exception::custom(e.to_string())),
            Self::NemotronH => round_trip!(NemotronHConfig, config_content),
            Self::Qwen3Next => {
                serde_json::to_value(Qwen3NextConfig::from_config_json(config_content)?)
                    .map_err(|e| Exception::custom(e.to_string()))
            }
            Self::GptOss => round_trip!(GptOssConfig, config_content),
            Self::Gemma4 => round_trip!(crate::architectures::gemma4::Gemma4Config, &nested),
            Self::Bert => {
                let model_type = json5::from_str::<serde_json::Value>(config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                if model_type.get("model_type").and_then(|t| t.as_str()) == Some("modernbert") {
                    round_trip!(ModernBertConfig, config_content)
                } else {
                    round_trip!(BertConfig, config_content)
                }
            }
            Self::DiffusionGemma => {
                let parsed = crate::architectures::diffusion_gemma::parse_diffusion_gemma_config(
                    config_content,
                )?;
                serde_json::to_value(&parsed).map_err(|e| Exception::custom(e.to_string()))
            }
            Self::Mllama => round_trip!(MllamaConfig, config_content),
            Self::Qwen4Exp => round_trip!(Qwen4ExpConfig, &nested),
            Self::Flux => Err(Exception::custom(
                "Flux is a diffusion pipeline, not a causal LM; its config is parsed by FluxPipeline.",
            )),
        }
    }

    pub fn detect<P: AsRef<Path>>(model_dir: P) -> Result<Self, Exception> {
        let config_path = model_dir.as_ref().join("config.json");
        if !config_path.exists() {
            return Err(Exception::custom(format!(
                "Config file not found: {:?}",
                config_path
            )));
        }
        let config_content = std::fs::read_to_string(config_path)
            .map_err(|e| Exception::custom(format!("{}", e)))?;
        let config = config_value(&config_content)?;

        let architectures = config["architectures"].as_array().map(|a| {
            a.iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect::<Vec<_>>()
        });
        let model_type = config["model_type"].as_str().unwrap_or("");

        Self::from_model_type(model_type)
            .or_else(|| {
                architectures
                    .as_ref()
                    .and_then(|a| Self::from_architectures(a))
            })
            .ok_or_else(|| {
                Exception::custom(
                    granite_refusal(model_type)
                        .unwrap_or_else(|| format!("Unsupported model type: {}", model_type)),
                )
            })
    }
}

/// Dispatch a method call uniformly across all `DynamicModel` variants.
///
/// Every arm expands to `m.$method($args...)` where `m` is the inner model.
/// Use this only for methods where ALL variants have identical call signatures.
macro_rules! dispatch_uniform {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            Self::Llama(m) => m.$method($($arg),*),
            Self::Llama4(m) => m.$method($($arg),*),
            Self::Qwen2(m) => m.$method($($arg),*),
            Self::Qwen3(m) => m.$method($($arg),*),
            Self::Qwen3MoE(m) => m.$method($($arg),*),
            Self::Gemma(m) => m.$method($($arg),*),
            Self::Mistral(m) => m.$method($($arg),*),
            Self::Phi(m) => m.$method($($arg),*),
            Self::Phi4(m) => m.$method($($arg),*),
            Self::DeepSeek(m) => m.$method($($arg),*),
            Self::Cohere(m) => m.$method($($arg),*),
            Self::Granite(m) => m.$method($($arg),*),
            Self::NemotronH(m) => m.$method($($arg),*),
            Self::Qwen3Next(m) => m.$method($($arg),*),
            Self::GptOss(m) => m.$method($($arg),*),
            Self::Gemma4(m) => m.$method($($arg),*),
            Self::Flux(m) => m.$method($($arg),*),
            Self::Bert(m) => m.$method($($arg),*),
            Self::DiffusionGemma(m) => m.$method($($arg),*),
            Self::Mllama(m) => m.$method($($arg),*),
            Self::Qwen4Exp(m) => m.$method($($arg),*),
        }
    };
}

/// Map each `DynamicModel` variant to its corresponding `ModelArchitecture` constant.
macro_rules! dispatch_architecture {
    ($self:expr) => {
        match $self {
            Self::Llama(_) => ModelArchitecture::Llama,
            Self::Llama4(_) => ModelArchitecture::Llama4,
            Self::Qwen2(_) => ModelArchitecture::Qwen2,
            Self::Qwen3(_) => ModelArchitecture::Qwen3,
            Self::Qwen3MoE(_) => ModelArchitecture::Qwen3MoE,
            Self::Gemma(_) => ModelArchitecture::Gemma,
            Self::Mistral(_) => ModelArchitecture::Mistral,
            Self::Phi(_) => ModelArchitecture::Phi,
            Self::Phi4(_) => ModelArchitecture::Phi4,
            Self::DeepSeek(_) => ModelArchitecture::DeepSeek,
            Self::Cohere(_) => ModelArchitecture::Cohere,
            Self::Granite(_) => ModelArchitecture::Granite,
            Self::NemotronH(_) => ModelArchitecture::NemotronH,
            Self::Qwen3Next(_) => ModelArchitecture::Qwen3Next,
            Self::GptOss(_) => ModelArchitecture::GptOss,
            Self::Gemma4(_) => ModelArchitecture::Gemma4,
            Self::Flux(_) => ModelArchitecture::Flux,
            Self::Bert(_) => ModelArchitecture::Bert,
            Self::DiffusionGemma(_) => ModelArchitecture::DiffusionGemma,
            Self::Mllama(_) => ModelArchitecture::Mllama,
            Self::Qwen4Exp(_) => ModelArchitecture::Qwen4Exp,
        }
    };
}

/// Shared body for the common architecture load path:
///
/// 1. Parse config JSON into the architecture's config type.
/// 2. Construct the model via the provided constructor expression.
/// 3. Run the HuggingFace-style generic weight loader.
/// 4. Batched-eval every `ModuleParameters` so weights materialise on GPU.
/// 5. Wrap in the given `DynamicModel` variant and return.
///
/// Used for architectures that don't need config unwrapping, custom weight
/// remapping, or post-load fast-path initialisation. Replaces ~13 hand-rolled
/// copy-paste match arms in `DynamicModel::load_with_options`.
///
/// `$new` accepts any callable returning `Result<Model, Exception>` — most
/// architectures pass `TypeName::new`; Qwen3 passes `Qwen3ForCausalLM::new_for_loading`.
/// Resolve a `config.json` body to the architecture that should run it.
///
/// `model_type` decides; the `architectures` array is the fallback for
/// checkpoints that omit it. Shared by [`DynamicModel::load_with_options`] and
/// [`DynamicModel::from_config`] so the two cannot disagree about what a
/// checkpoint is.
fn resolve_architecture(config_content: &str) -> Result<ModelArchitecture, Exception> {
    let base_config = config_value(config_content)?;
    let architectures = base_config["architectures"].as_array().map(|a| {
        a.iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect::<Vec<_>>()
    });
    let model_type = base_config["model_type"].as_str().unwrap_or("");
    ModelArchitecture::from_model_type(model_type)
        .or_else(|| {
            architectures
                .as_ref()
                .and_then(|a| ModelArchitecture::from_architectures(a))
        })
        .ok_or_else(|| {
            Exception::custom(
                granite_refusal(model_type)
                    .unwrap_or_else(|| format!("Unsupported architecture: {}", model_type)),
            )
        })
}

/// Parse a Gemma config, deriving the generation flags from `model_type`.
///
/// `is_gemma2` / `is_gemma3` are not fields any checkpoint sets; they are
/// pmetal's own switches for the 4-norm block, the attention and final-logit
/// softcaps, and the local/global window interleave. Deriving them in one place
/// keeps a construction path from silently running the Gemma-v1 math.
///
/// Public because `pmetal-lora`'s dispatcher builds the same config and needs
/// the same flags. Deserializing `GemmaConfig` straight from the checkpoint
/// leaves both `false`, which is a Gemma 1 model wearing a Gemma 3 checkpoint.
pub fn parse_gemma_config(config_content: &str) -> Result<GemmaConfig, Exception> {
    let effective = unwrap_text_config(config_content)?;
    let mut config: GemmaConfig =
        json5::from_str(&effective).map_err(|e| Exception::custom(e.to_string()))?;
    if config.model_type == "gemma3"
        || config.model_type == "gemma4"
        || config.model_type == "gemma4_text"
        || config.model_type == "gemma3_text"
    {
        config.is_gemma3 = true;
    } else if config.model_type == "gemma2" {
        config.is_gemma2 = true;
    }
    Ok(config)
}

/// Construct a model from a parsed config, with no weights to load.
///
/// The counterpart to [`simple_load!`]: same config deserialization, but the
/// architecture's *sized* constructor rather than the placeholder one, because
/// nothing is coming along afterwards to give the parameters their shapes.
macro_rules! simple_new {
    ($config_ty:ty, $new:expr, $content:expr, $variant:ident) => {{
        let config: $config_ty =
            json5::from_str($content).map_err(|e| Exception::custom(e.to_string()))?;
        Ok(Self::$variant(($new)(config)?))
    }};
}

macro_rules! simple_load {
    ($config_ty:ty, $new:expr, $content:expr, $model_dir:expr, $variant:ident) => {{
        let config: $config_ty =
            json5::from_str($content).map_err(|e| Exception::custom(e.to_string()))?;
        let mut model = ($new)(config)?;
        load_generic_weights_with(
            &mut model,
            $model_dir,
            &ModelArchitecture::$variant.load_policy(),
        )
        .map_err(|e| Exception::custom(e.to_string()))?;
        eval_module_parameters_batched(&model)?;
        Ok(Self::$variant(model))
    }};
}

/// An LM head in the form cut cross-entropy computes a loss from: the logits
/// of hidden states `h` are `softcap(logit_scale · (h·Wᵀ + b))`, never
/// materialised. See [`DynamicModel::lm_head`].
#[derive(Debug, Clone)]
pub struct LmHead {
    /// `[vocab, hidden]`, dense, with any adapter on the head folded in.
    pub weight: Array,
    /// `[vocab]`, when the head has a bias.
    pub bias: Option<Array>,
    /// Multiplies the logits: Cohere's `logit_scale`, Granite's
    /// `1 / logits_scaling`, 1 elsewhere.
    pub logit_scale: f32,
    /// `cap · tanh(logits / cap)` after the scale, for Gemma 2 and Gemma 4.
    pub softcap: Option<f32>,
}

/// What a cut cross-entropy loss computes besides the plain mean negative
/// log-likelihood.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CceLoss {
    /// Targets equal to this are not scored.
    pub ignore_index: i32,
    /// Smooth the target distribution toward uniform by this share, in
    /// `[0, 1)`: `(1 - eps)(lse - z_y) + eps(lse - mean z)`, as PyTorch's
    /// `label_smoothing` defines it.
    pub label_smoothing: f32,
    /// Add `weight * lse^2` to each token's loss (the z-loss that keeps the
    /// softmax normaliser near 1), finite and non-negative.
    pub z_loss: f32,
}

impl CceLoss {
    /// The plain loss, skipping `ignore_index`.
    pub const fn new(ignore_index: i32) -> Self {
        Self {
            ignore_index,
            label_smoothing: 0.0,
            z_loss: 0.0,
        }
    }

    /// With label smoothing `epsilon`.
    #[must_use]
    pub const fn with_label_smoothing(mut self, epsilon: f32) -> Self {
        self.label_smoothing = epsilon;
        self
    }

    /// With z-loss `weight`.
    #[must_use]
    pub const fn with_z_loss(mut self, weight: f32) -> Self {
        self.z_loss = weight;
        self
    }
}

impl LmHead {
    /// Mean cross-entropy of `targets` (`[n]`, `loss.ignore_index` skipped)
    /// under the logits of `hidden` (`[n, hidden]`, or any leading shape that
    /// flattens to it), computed a vocabulary chunk at a time, with label
    /// smoothing and z-loss as `loss` asks.
    ///
    /// # Errors
    /// A label smoothing outside `[0, 1)` or a z-loss weight that is negative
    /// or not finite, or the loss kernel failing.
    pub fn cut_cross_entropy(
        &self,
        hidden: &Array,
        targets: &Array,
        loss: CceLoss,
    ) -> Result<Array, Exception> {
        use pmetal_runtime::nn::loss::{CutCrossEntropy, CutCrossEntropyConfig};

        // The runtime's builders assert; a bad value from a config file is an
        // error to report, not a panic.
        if !(0.0..1.0).contains(&loss.label_smoothing) {
            return Err(Exception::custom(format!(
                "label smoothing must be in [0, 1), got {}",
                loss.label_smoothing
            )));
        }
        if !loss.z_loss.is_finite() || loss.z_loss < 0.0 {
            return Err(Exception::custom(format!(
                "z-loss weight must be finite and non-negative, got {}",
                loss.z_loss
            )));
        }

        let width = hidden.dim(-1);
        let flat = hidden.reshape(&[-1, width]);
        let mut config = CutCrossEntropyConfig::new()
            .with_ignore_index(loss.ignore_index)
            .with_logit_scale(self.logit_scale)
            .with_label_smoothing(loss.label_smoothing)
            .with_z_loss(loss.z_loss);
        if let Some(cap) = self.softcap {
            config = config.with_softcap(cap);
        }
        CutCrossEntropy::new(config).forward(&flat, &self.weight, targets, self.bias.as_ref())
    }
}

/// A model whose architecture is dispatched at runtime.
#[expect(
    clippy::large_enum_variant,
    reason = "one per loaded model; the size difference is irrelevant next to its weights"
)]
pub enum DynamicModel {
    Llama(LlamaForCausalLM),
    Llama4(Llama4ForCausalLM),
    Qwen2(Qwen2ForCausalLM),
    Qwen3(Qwen3ForCausalLM),
    Qwen3MoE(Qwen3MoE),
    Gemma(GemmaForCausalLM),
    Mistral(MistralForCausalLM),
    Phi(PhiForCausalLM),
    Phi4(PhiForCausalLM),
    DeepSeek(DeepSeek),
    Cohere(CohereForCausalLM),
    Granite(GraniteForCausalLM),
    NemotronH(NemotronHForCausalLM),
    Qwen3Next(Qwen3NextForCausalLM),
    GptOss(GptOssForCausalLM),
    Gemma4(Gemma4ForCausalLM),
    Flux(FluxDiT),
    Bert(EncoderModel),
    DiffusionGemma(DiffusionGemmaForBlockDiffusion),
    Mllama(MllamaForConditionalGeneration),
    Qwen4Exp(Qwen4ExpForCausalLM),
}

impl std::fmt::Debug for DynamicModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Llama(_) => write!(f, "DynamicModel::Llama"),
            Self::Llama4(_) => write!(f, "DynamicModel::Llama4"),
            Self::Qwen2(_) => write!(f, "DynamicModel::Qwen2"),
            Self::Qwen3(_) => write!(f, "DynamicModel::Qwen3"),
            Self::Qwen3MoE(_) => write!(f, "DynamicModel::Qwen3MoE"),
            Self::Gemma(_) => write!(f, "DynamicModel::Gemma"),
            Self::Mistral(_) => write!(f, "DynamicModel::Mistral"),
            Self::Phi(_) => write!(f, "DynamicModel::Phi"),
            Self::Phi4(_) => write!(f, "DynamicModel::Phi4"),
            Self::DeepSeek(_) => write!(f, "DynamicModel::DeepSeek"),
            Self::Cohere(_) => write!(f, "DynamicModel::Cohere"),
            Self::Granite(_) => write!(f, "DynamicModel::Granite"),
            Self::NemotronH(_) => write!(f, "DynamicModel::NemotronH"),
            Self::Qwen3Next(_) => write!(f, "DynamicModel::Qwen3Next"),
            Self::GptOss(_) => write!(f, "DynamicModel::GptOss"),
            Self::Gemma4(_) => write!(f, "DynamicModel::Gemma4"),
            Self::Flux(_) => write!(f, "DynamicModel::Flux"),
            Self::Bert(_) => write!(f, "DynamicModel::Bert"),
            Self::DiffusionGemma(_) => write!(f, "DynamicModel::DiffusionGemma"),
            Self::Mllama(_) => write!(f, "DynamicModel::Mllama"),
            Self::Qwen4Exp(_) => write!(f, "DynamicModel::Qwen4Exp"),
        }
    }
}

impl DynamicModel {
    fn init_post_load_fast_paths(&mut self) -> Result<(), Exception> {
        match self {
            Self::GptOss(model) => model.init_stacked_moe(),
            _ => Ok(()),
        }
    }

    /// Load a model from a directory, automatically detecting its architecture.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, Exception> {
        Self::load_with_options(model_dir, DynamicModelLoadOptions::default())
    }

    /// Build a randomly initialised model from a `config.json` body, with no
    /// checkpoint.
    ///
    /// [`load`](Self::load) exists to put a checkpoint into a model, and several
    /// architectures exploit that by constructing one-element placeholder
    /// parameters and letting the loader size them (`Qwen3ForCausalLM::new_for_loading`
    /// is the clearest case). That is the right trade when weights are coming,
    /// and useless when they are not: the parameters never acquire a shape and
    /// the first forward pass indexes into an empty dimension.
    ///
    /// This is the other half — every architecture's *sized* constructor, behind
    /// the same `model_type` resolution `load` uses — for pretraining from
    /// scratch and for tests that need a model without a checkpoint.
    ///
    /// Architectures that are not causal language models, or whose construction
    /// is inseparable from their checkpoint layout, return an error naming the
    /// reason rather than a half-built model.
    pub fn from_config(config_content: &str) -> Result<Self, Exception> {
        let arch = resolve_architecture(config_content)?;
        match arch {
            ModelArchitecture::Llama => {
                simple_new!(LlamaConfig, LlamaForCausalLM::new, config_content, Llama)
            }
            ModelArchitecture::Llama4 => {
                let effective = unwrap_text_config(config_content)?;
                simple_new!(Llama4TextConfig, Llama4ForCausalLM::new, &effective, Llama4)
            }
            ModelArchitecture::Qwen2 => {
                simple_new!(Qwen2Config, Qwen2ForCausalLM::new, config_content, Qwen2)
            }
            // `Qwen3ForCausalLM::new`, deliberately, where `load` takes
            // `new_for_loading`.
            ModelArchitecture::Qwen3 => {
                simple_new!(Qwen3Config, Qwen3ForCausalLM::new, config_content, Qwen3)
            }
            ModelArchitecture::Qwen3MoE => {
                simple_new!(Qwen3MoEConfig, Qwen3MoE::new, config_content, Qwen3MoE)
            }
            ModelArchitecture::Gemma => Ok(Self::Gemma(GemmaForCausalLM::new(
                parse_gemma_config(config_content)?,
            )?)),
            ModelArchitecture::Mistral => {
                simple_new!(
                    MistralConfig,
                    MistralForCausalLM::new,
                    config_content,
                    Mistral
                )
            }
            ModelArchitecture::Phi => {
                simple_new!(PhiConfig, PhiForCausalLM::new, config_content, Phi)
            }
            ModelArchitecture::Phi4 => {
                simple_new!(PhiConfig, PhiForCausalLM::new, config_content, Phi4)
            }
            ModelArchitecture::DeepSeek => {
                simple_new!(DeepSeekConfig, DeepSeek::new, config_content, DeepSeek)
            }
            ModelArchitecture::Cohere => {
                simple_new!(CohereConfig, CohereForCausalLM::new, config_content, Cohere)
            }
            ModelArchitecture::Granite => Ok(Self::Granite(GraniteForCausalLM::new(
                GraniteConfig::from_config_json(config_content)?,
            )?)),
            ModelArchitecture::NemotronH => {
                simple_new!(
                    NemotronHConfig,
                    NemotronHForCausalLM::new,
                    config_content,
                    NemotronH
                )
            }
            ModelArchitecture::Qwen3Next => Ok(Self::Qwen3Next(Qwen3NextForCausalLM::new(
                Qwen3NextConfig::from_config_json(config_content)?,
            )?)),
            ModelArchitecture::GptOss => {
                simple_new!(GptOssConfig, GptOssForCausalLM::new, config_content, GptOss)
            }
            ModelArchitecture::Gemma4 => {
                let effective = unwrap_text_config(config_content)?;
                simple_new!(
                    crate::architectures::gemma4::Gemma4Config,
                    crate::architectures::gemma4::Gemma4ForCausalLM::new,
                    &effective,
                    Gemma4
                )
            }
            ModelArchitecture::Bert => {
                Ok(Self::Bert(EncoderModel::from_config_str(config_content)?))
            }
            ModelArchitecture::DiffusionGemma => {
                let config = crate::architectures::diffusion_gemma::parse_diffusion_gemma_config(
                    config_content,
                )?;
                Ok(Self::DiffusionGemma(DiffusionGemmaForBlockDiffusion::new(
                    config,
                )?))
            }
            ModelArchitecture::Mllama => {
                simple_new!(
                    MllamaConfig,
                    MllamaForConditionalGeneration::new,
                    config_content,
                    Mllama
                )
            }
            ModelArchitecture::Qwen4Exp => {
                let config = Qwen4ExpConfig::from_json(&unwrap_text_config(config_content)?)?;
                Ok(Self::Qwen4Exp(Qwen4ExpForCausalLM::new(config)?))
            }
            ModelArchitecture::Flux => Err(Exception::custom(
                "Flux models are diffusion pipelines, not causal language models. Build them via pmetal_models::pipelines::FluxPipeline instead of DynamicModel::from_config.",
            )),
        }
    }

    /// Load a model from a directory with caller-controlled load behavior.
    pub fn load_with_options(
        model_dir: impl AsRef<Path>,
        options: DynamicModelLoadOptions,
    ) -> Result<Self, Exception> {
        let model_dir = model_dir.as_ref();

        // GGUF checkpoints carry their config in metadata, not a config.json.
        // A `.gguf` file path, or a directory whose only weights are GGUF,
        // routes to the GGUF loader. `WeightFormat::detect` prefers safetensors
        // when both are present, so canonical HF dirs fall through unchanged.
        if matches!(WeightFormat::detect(model_dir), Some(WeightFormat::Gguf)) {
            return Self::load_gguf(model_dir, options);
        }

        let config_path = model_dir.join("config.json");
        if !config_path.exists() {
            return Err(Exception::custom(format!(
                "Config file not found: {:?}",
                config_path
            )));
        }
        let config_content = std::fs::read_to_string(&config_path)
            .map_err(|e| Exception::custom(format!("{}", e)))?;
        let arch = resolve_architecture(&config_content)?;

        match arch {
            ModelArchitecture::Llama => simple_load!(
                LlamaConfig,
                LlamaForCausalLM::new,
                &config_content,
                model_dir,
                Llama
            ),
            ModelArchitecture::Llama4 => {
                let effective = unwrap_text_config(&config_content)?;
                let config: Llama4TextConfig =
                    json5::from_str(&effective).map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = Llama4ForCausalLM::new(config)?;
                let weights = crate::loader::load_weights(model_dir)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let weights = crate::architectures::llama4::sanitize_checkpoint(weights);
                assign_weights(
                    &mut model,
                    weights,
                    &ModelArchitecture::Llama4.load_policy(),
                )
                .map_err(|e| Exception::custom(e.with_quantization_hint(model_dir).to_string()))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Llama4(model))
            }
            ModelArchitecture::Qwen2 => simple_load!(
                Qwen2Config,
                Qwen2ForCausalLM::new,
                &config_content,
                model_dir,
                Qwen2
            ),
            ModelArchitecture::Qwen3 => simple_load!(
                Qwen3Config,
                Qwen3ForCausalLM::new_for_loading,
                &config_content,
                model_dir,
                Qwen3
            ),
            // The parameter tree is the checkpoint's layout with each layer's
            // experts stacked: a released checkpoint's expert-per-key tensors
            // and transformers 5's fused `gate_up_proj` are stacked at load,
            // an MLX conversion's `mlp.switch_mlp` is held as it is.
            ModelArchitecture::Qwen3MoE => {
                let config: Qwen3MoEConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = Qwen3MoE::new(config)?;
                load_with_stacked_experts(
                    "qwen3_moe",
                    &mut model,
                    model_dir,
                    &ModelArchitecture::Qwen3MoE.load_policy(),
                )?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Qwen3MoE(model))
            }
            ModelArchitecture::Gemma => {
                let config = parse_gemma_config(&config_content)?;
                let mut model = GemmaForCausalLM::new(config)?;
                let weights = crate::loader::load_weights(model_dir)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                // Gemma 4 stores the language-tower weights under
                // `model.language_model.…` (multimodal wrapper). Strip the
                // infix so the existing loader's `model.…` keys match.
                // Also drop vision / audio tower weights — pmetal only
                // runs the language stack today.
                let needs_lm_strip = weights
                    .keys()
                    .any(|k| k.starts_with("model.language_model."));
                let weights_effective = if needs_lm_strip {
                    let mut remapped: std::collections::HashMap<String, Array> =
                        std::collections::HashMap::with_capacity(weights.len());
                    for (key, value) in &weights {
                        if let Some(rest) = key.strip_prefix("model.language_model.") {
                            remapped.insert(format!("model.{rest}"), value.clone());
                        } else if key.starts_with("model.embed_vision.")
                            || key.starts_with("model.vision_tower.")
                            || key.starts_with("model.audio_tower.")
                            || key.starts_with("model.multi_modal_projector.")
                        {
                            // Skip non-language towers.
                        } else if key == "lm_head.weight" {
                            // Gemma ties, but some checkpoints carry an
                            // explicit head. Keep it under its own key; the
                            // loader will ignore it because Gemma uses
                            // tied embeddings.
                            remapped.insert(key.clone(), value.clone());
                        } else {
                            remapped.insert(key.clone(), value.clone());
                        }
                    }
                    remapped
                } else {
                    weights
                };
                load_checked(&mut model, model_dir, &LoadPolicy::strict(), |model| {
                    crate::loader::load_gemma_weights(model, &weights_effective)
                        .map_err(|e| Exception::custom(e.to_string()))
                })?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Gemma(model))
            }
            ModelArchitecture::Mistral => simple_load!(
                MistralConfig,
                MistralForCausalLM::new,
                &config_content,
                model_dir,
                Mistral
            ),
            // Phi cannot use `simple_load!`: Phi-3 fuses q/k/v into a single
            // `self_attn.qkv_proj`, which the generic name-matching loader has
            // no parameter for and therefore discards without a word.
            ModelArchitecture::Phi => Self::load_phi_variant(&config_content, model_dir, false),
            ModelArchitecture::Phi4 => Self::load_phi_variant(&config_content, model_dir, true),
            // The parameter tree is the checkpoint's layout (router and
            // `e_score_correction_bias` under `mlp.gate`, `mlp.shared_experts`)
            // with each layer's experts stacked, which is how a released
            // expert-per-key checkpoint is held (`stack_experts`) and how an
            // MLX conversion stores them (`mlp.switch_mlp`). The tensors of a
            // multi-token-prediction layer the config does not enable have no
            // parameter and are left unread.
            ModelArchitecture::DeepSeek => {
                let config: DeepSeekConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut policy = LoadPolicy::strict();
                // V3 ships its multi-token-prediction layers after the trunk, as
                // `model.layers.{num_hidden_layers + i}.*`. They are a separate
                // draft head, not part of the model that decodes.
                for i in 0..config.num_nextn_predict_layers.max(0) {
                    policy.unused_tensors.push(Exemption::owned(
                        format!("model.layers.{}.*", config.num_hidden_layers + i),
                        "a multi-token-prediction layer: a draft head, not part of the trunk",
                    ));
                }
                // Only V3's bias-corrected routing selects on the bias; V2
                // checkpoints do not carry one and the router never reads it.
                if config.routing().method != pmetal_runtime::deepseek_native::TopkMethod::NoauxTc {
                    policy.unloaded_params.push(Exemption::new(
                        "*.mlp.gate.e_score_correction_bias",
                        "this routing method selects experts on the scores alone, with no bias",
                    ));
                }
                let mut model = DeepSeek::new(config)?;
                load_with_stacked_experts("deepseek", &mut model, model_dir, &policy)?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::DeepSeek(model))
            }
            ModelArchitecture::Cohere => simple_load!(
                CohereConfig,
                CohereForCausalLM::new,
                &config_content,
                model_dir,
                Cohere
            ),
            // All of the checkpoint or none of it: see `load_granite_weights`.
            ModelArchitecture::Granite => {
                let mut model =
                    GraniteForCausalLM::new(GraniteConfig::from_config_json(&config_content)?)?;
                let report = crate::architectures::granite_hybrid::load_granite_weights(
                    &mut model, model_dir,
                )?;
                tracing::debug!(
                    "Granite weight load: {} loaded, {} ignored by design",
                    report.loaded,
                    report.ignored.len()
                );
                eval_module_parameters_batched(&model)?;
                Ok(Self::Granite(model))
            }
            // The parameter tree is the checkpoint's layout with each MoE
            // layer's experts stacked and the convolutions in the runtime's
            // layout; see `nemotron_h::assign_nemotron_h_weights`.
            ModelArchitecture::NemotronH => {
                let config: NemotronHConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = NemotronHForCausalLM::new(config)?;
                let weights =
                    load_weights(model_dir).map_err(|e| Exception::custom(e.to_string()))?;
                crate::architectures::nemotron_h::assign_nemotron_h_weights(&mut model, weights)
                    .map_err(|e| {
                        Exception::custom(e.with_quantization_hint(model_dir).to_string())
                    })?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::NemotronH(model))
            }
            ModelArchitecture::Qwen3Next => {
                let config = Qwen3NextConfig::from_config_json(&config_content)?;
                let skip_routed_experts = options.prefer_expert_offload && config.num_experts > 0;
                let routed_expert_mode = if skip_routed_experts {
                    Qwen3NextRoutedExpertMode::Placeholder
                } else {
                    Qwen3NextRoutedExpertMode::Resident
                };
                let mut model = Qwen3NextForCausalLM::new_with_routed_expert_mode(
                    config.clone(),
                    routed_expert_mode,
                )?;
                let load_options = if skip_routed_experts {
                    Qwen3NextLoadOptions {
                        skip_routed_experts: true,
                        ..Default::default()
                    }
                } else {
                    Qwen3NextLoadOptions::default()
                };
                load_qwen3_next_weights_with_options(&mut model, model_dir, &config, load_options)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Qwen3Next(model))
            }
            ModelArchitecture::Qwen4Exp => {
                let config = Qwen4ExpConfig::from_json(&unwrap_text_config(&config_content)?)?;
                let skip_routed_experts = options.prefer_expert_offload;
                let mut model = Qwen4ExpForCausalLM::new_for_loading(
                    config,
                    if skip_routed_experts {
                        Qwen3NextRoutedExpertMode::Placeholder
                    } else {
                        Qwen3NextRoutedExpertMode::Resident
                    },
                )?;
                let load_options = Qwen4ExpLoadOptions {
                    skip_routed_experts,
                    ngram_rows_on_disk: model.config.ngram_table_bytes()
                        > Qwen4ExpLoadOptions::RESIDENT_NGRAM_LIMIT_BYTES,
                    // NVFP4 experts stay packed.
                    unpack_quantized_experts: false,
                };
                let report = load_qwen4_exp_weights(&mut model, model_dir, load_options)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                tracing::info!(
                    ngram_rows_on_disk = load_options.ngram_rows_on_disk,
                    "Qwen4-Exp weight load: {} loaded, {} skipped by design (vision tower, MTP)",
                    report.loaded,
                    report.skipped.len()
                );
                eval_module_parameters_batched(&model)?;
                Ok(Self::Qwen4Exp(model))
            }
            ModelArchitecture::Flux => Err(Exception::custom(
                "Flux models are diffusion pipelines, not causal language models. Load them via pmetal_models::pipelines::FluxPipeline instead of DynamicModel::load.",
            )),
            ModelArchitecture::GptOss => {
                let config: GptOssConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = GptOssForCausalLM::new(config)?;
                // The release (MXFP4) and transformers' own layout ship the
                // experts fused; a per-expert `Linear` checkpoint takes the
                // generic loader.
                let policy = ModelArchitecture::GptOss.load_policy();
                let snapshot = ParamSnapshot::of(&model);
                if model.load_fused_checkpoint(model_dir)? {
                    snapshot
                        .require_complete(&model, &policy, Vec::new())
                        .map_err(|e| {
                            Exception::custom(e.with_quantization_hint(model_dir).to_string())
                        })?;
                } else {
                    load_generic_weights_with(&mut model, model_dir, &policy)
                        .map_err(|e| Exception::custom(e.to_string()))?;
                }
                eval_module_parameters_batched(&model)?;
                let mut model = Self::GptOss(model);
                model.init_post_load_fast_paths()?;
                Ok(model)
            }
            ModelArchitecture::Gemma4 => {
                let effective = unwrap_text_config(&config_content)?;
                let config: crate::architectures::gemma4::Gemma4Config =
                    json5::from_str(&effective).map_err(|e| Exception::custom(e.to_string()))?;
                // A layer that shares another's keys and values reads them from
                // its source and has no K/V projections of its own; checkpoints
                // omit them, and the layer never reads them.
                let mut policy = ModelArchitecture::Gemma4.load_policy();
                for layer in config.first_kv_shared_layer_idx()..config.num_hidden_layers as usize {
                    for tensor in ["k_proj.weight", "v_proj.weight", "k_norm.weight"] {
                        policy.unloaded_params.push(Exemption::owned(
                            format!("model.layers.{layer}.self_attn.{tensor}"),
                            "a KV-shared layer reads its source layer's keys and values",
                        ));
                    }
                }
                let mut model = crate::architectures::gemma4::Gemma4ForCausalLM::new(config)?;
                let weights = crate::loader::load_weights(model_dir)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let report = load_checked(&mut model, model_dir, &policy, |model| {
                    crate::architectures::gemma4::load_gemma4_weights(model, &weights)
                        .map_err(|e| Exception::custom(e.to_string()))
                })?;
                if !report.skipped.is_empty() {
                    tracing::info!(
                        "Gemma 4 weight load: {} loaded, {} skipped (first: {:?})",
                        report.loaded,
                        report.skipped.len(),
                        report.skipped.first()
                    );
                }
                eval_module_parameters_batched(&model)?;
                Ok(Self::Gemma4(model))
            }
            ModelArchitecture::Bert => {
                let mut model = EncoderModel::from_config_str(&config_content)?;
                // `transformers` spells the encoder's parameters differently
                // from the model (`encoder.layer.0.attention.self.query` for
                // `layers.0.attention.query`); the loader maps them, and fails
                // unless the checkpoint and the model account for each other.
                let weights =
                    load_weights(model_dir).map_err(|e| Exception::custom(e.to_string()))?;
                load_encoder_weights(&mut model, weights).map_err(|e| {
                    Exception::custom(e.with_quantization_hint(model_dir).to_string())
                })?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Bert(model))
            }
            ModelArchitecture::DiffusionGemma => {
                // DiffusionGemma nests the text tower under `text_config` and
                // carries `canvas_length` at the top level; the parser folds
                // both into the text config. The checkpoint ties the encoder
                // trunk, decoder trunk, and `lm_head` to a single physical
                // copy of each tensor, so a bespoke remapper resolves every
                // trunk slot before the per-tower loaders run.
                let config = crate::architectures::diffusion_gemma::parse_diffusion_gemma_config(
                    &config_content,
                )?;
                let mut model = DiffusionGemmaForBlockDiffusion::new(config)?;
                // Multimodal checkpoints carry a `vision_config`; attach the
                // vision tower + projector so `load_diffusion_gemma_weights`
                // populates `vision_tower.*` / `embed_vision.*`. Text-only
                // checkpoints leave the encoder byte-identical.
                if let Some((vision_config, image_token_id)) =
                    crate::architectures::diffusion_gemma::parse_diffusion_gemma_vision_config(
                        &config_content,
                    )?
                {
                    model.attach_vision(&vision_config, image_token_id)?;
                }
                let weights = crate::loader::load_weights(model_dir)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let report = load_checked(
                    &mut model,
                    model_dir,
                    &ModelArchitecture::DiffusionGemma.load_policy(),
                    |model| {
                        crate::architectures::diffusion_gemma::load_diffusion_gemma_weights(
                            model, &weights,
                        )
                    },
                )?;
                if !report.skipped.is_empty() {
                    tracing::info!(
                        "DiffusionGemma weight load: {} loaded, {} skipped (first: {:?})",
                        report.loaded,
                        report.skipped.len(),
                        report.skipped.first()
                    );
                }
                eval_module_parameters_batched(&model)?;
                Ok(Self::DiffusionGemma(model))
            }
            ModelArchitecture::Mllama => {
                // Mllama's checkpoint layout needs a hand-written loader: the
                // patch-embedding convolution has to be transposed to MLX's
                // NHWC weight order, and the text decoder lives under a
                // `language_model.model.` prefix that the generic loader can't
                // express. `load_mllama_weights` also accepts the newer
                // `model.{vision,language}_model.` layout.
                let config: MllamaConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = MllamaForConditionalGeneration::new(config)?;
                let weights = crate::loader::load_weights(model_dir)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let report = load_checked(
                    &mut model,
                    model_dir,
                    &ModelArchitecture::Mllama.load_policy(),
                    |model| crate::architectures::mllama::load_mllama_weights(model, &weights),
                )?;
                if !report.skipped.is_empty() {
                    tracing::info!(
                        "Mllama weight load: {} loaded, {} skipped (first: {:?})",
                        report.loaded,
                        report.skipped.len(),
                        report.skipped.first()
                    );
                }
                eval_module_parameters_batched(&model)?;
                Ok(Self::Mllama(model))
            }
        }
    }

    /// Load Phi-3 (`is_phi4 = false`) or Phi-4 (`true`), splitting a fused
    /// `self_attn.qkv_proj` first if the checkpoint ships one.
    ///
    /// Both variants are the same `PhiForCausalLM`; they differ only in which
    /// `DynamicModel` arm they land in. Phi-3-mini fuses q/k/v and Phi-4-mini
    /// does not, so the split runs unconditionally and no-ops on the latter.
    fn load_phi_variant(
        config_content: &str,
        model_dir: &Path,
        is_phi4: bool,
    ) -> Result<Self, Exception> {
        let config: PhiConfig =
            json5::from_str(config_content).map_err(|e| Exception::custom(e.to_string()))?;
        let mut model = PhiForCausalLM::new(config.clone())?;
        let mut weights =
            crate::loader::load_weights(model_dir).map_err(|e| Exception::custom(e.to_string()))?;
        crate::loader::split_phi_fused_qkv(&mut weights, &config);
        crate::loader::assign_weights(&mut model, weights, &LoadPolicy::strict())
            .map_err(|e| Exception::custom(e.with_quantization_hint(model_dir).to_string()))?;
        eval_module_parameters_batched(&model)?;
        Ok(if is_phi4 {
            Self::Phi4(model)
        } else {
            Self::Phi(model)
        })
    }

    /// Load a model from a GGUF checkpoint (file path or directory).
    ///
    /// Reads the model config from GGUF metadata, constructs the matching
    /// architecture, then assigns the dequantized (F32), HuggingFace-named
    /// weights produced by [`WeightLoader::load_gguf`].
    ///
    /// Phase B coverage is the dense Llama-family decoders (Llama, Qwen2,
    /// Qwen3, Mistral, Phi) — the architectures whose HF parameter trees map
    /// 1:1 onto GGUF tensor names and that already have metadata→config
    /// converters. Gemma / Gemma 4 (extra norm blocks, embedding scaling,
    /// softcapping, MoE) are handled in later phases and return a typed error
    /// here rather than loading incorrectly.
    pub fn load_gguf(
        path: impl AsRef<Path>,
        _options: DynamicModelLoadOptions,
    ) -> Result<Self, Exception> {
        let path = path.as_ref();
        let gguf_config = GgufModelConfig::from_path(path)
            .map_err(|e| Exception::custom(format!("GGUF config: {e}")))?;
        let arch =
            ModelArchitecture::from_model_type(&gguf_config.architecture).ok_or_else(|| {
                Exception::custom(format!(
                    "GGUF architecture '{}' is not supported for inference loading",
                    gguf_config.architecture
                ))
            })?;

        let mut weights =
            WeightLoader::load_gguf(path).map_err(|e| Exception::custom(format!("GGUF: {e}")))?;

        // A GGUF with no `output.weight` ties its head to the embedding; one
        // with it has an untied head, which the model has to hold. The configs
        // default to tied, so left alone every untied head was dropped and the
        // embedding matrix was used in its place.
        let tied = !weights.contains_key("lm_head.weight");
        let policy = LoadPolicy::strict();
        // The converter's precomputed RoPE frequency factors (Llama 3.x's
        // banded scaling) are the model's rotary scaling: part of the config,
        // not a weight.
        let mut gguf_config = gguf_config;
        if let Some(mut factors) = weights.remove("rope_freqs.weight") {
            let count = factors.size();
            let factors = factors
                .to_f32_vec(count)
                .ok_or_else(|| Exception::custom("GGUF: could not read rope_freqs.weight"))?;
            gguf_config
                .set_rope_frequency_factors(&factors)
                .map_err(|e| Exception::custom(format!("GGUF: {e}")))?;
        }

        match arch {
            ModelArchitecture::Llama => {
                let mut config = gguf_config.to_llama_config();
                config.tie_word_embeddings = tied;
                let num_heads = config.num_attention_heads;
                let num_kv_heads = config.num_key_value_heads.unwrap_or(num_heads);
                // `llama` GGUFs store q/k rows permuted for the GGUF converter's
                // interleaved RoPE; the other families are stored as published.
                crate::weight_format::unpermute_llama_qk(&mut weights, num_heads, num_kv_heads)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = LlamaForCausalLM::new(config)?;
                assign_weights(&mut model, weights, &policy)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                Ok(Self::Llama(model))
            }
            ModelArchitecture::Qwen2 => {
                let mut config = gguf_config.to_qwen2_config();
                config.tie_word_embeddings = tied;
                let mut model = Qwen2ForCausalLM::new(config)?;
                assign_weights(&mut model, weights, &policy)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                Ok(Self::Qwen2(model))
            }
            ModelArchitecture::Qwen3 => {
                let mut config = gguf_config.to_qwen3_config();
                config.tie_word_embeddings = tied;
                let mut model = Qwen3ForCausalLM::new_for_loading(config)?;
                assign_weights(&mut model, weights, &policy)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                Ok(Self::Qwen3(model))
            }
            ModelArchitecture::Mistral => {
                let mut config = gguf_config.to_mistral_config();
                config.tie_word_embeddings = tied;
                let mut model = MistralForCausalLM::new(config)?;
                assign_weights(&mut model, weights, &policy)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                Ok(Self::Mistral(model))
            }
            ModelArchitecture::Phi | ModelArchitecture::Phi4 => {
                // LongRoPE's two factor tables ride beside the weights.
                let mut table = |name: &str| -> Result<Option<Vec<f32>>, Exception> {
                    weights
                        .remove(name)
                        .map(|mut t| {
                            let n = t.size();
                            t.to_f32_vec(n).ok_or_else(|| {
                                Exception::custom(format!("GGUF: could not read {name}"))
                            })
                        })
                        .transpose()
                };
                let short = table("rope_factors_short.weight")?;
                let long = table("rope_factors_long.weight")?;
                gguf_config
                    .set_longrope_factors(short.as_deref(), long.as_deref())
                    .map_err(|e| Exception::custom(format!("GGUF: {e}")))?;
                let mut config = gguf_config.to_phi_config();
                config.tie_word_embeddings = tied;

                crate::weight_format::name_phi3_fused_tensors(&mut weights);
                crate::loader::split_phi_fused_qkv(&mut weights, &config);
                let mut model = PhiForCausalLM::new(config)?;
                assign_weights(&mut model, weights, &policy)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                Ok(if arch == ModelArchitecture::Phi4 {
                    Self::Phi4(model)
                } else {
                    Self::Phi(model)
                })
            }
            other => Err(Exception::custom(format!(
                "GGUF inference loading is not yet implemented for {other} \
                 (supported: dense Llama, Qwen2, Qwen3, Mistral, Phi-3)."
            ))),
        }
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        match self {
            Self::Llama(m) => m.forward(input_ids, mask),
            Self::Llama4(m) => m.forward(input_ids, mask, None),
            Self::Qwen2(m) => m.forward(input_ids, mask),
            Self::Qwen3(m) => m.forward(input_ids, mask),
            Self::Qwen3MoE(m) => m.forward(input_ids, mask, None),
            Self::Gemma(m) => m.forward(input_ids, mask),
            Self::Mistral(m) => m.forward(input_ids, mask),
            Self::Phi(m) => m.forward(input_ids, mask),
            Self::Phi4(m) => m.forward(input_ids, mask),
            Self::DeepSeek(m) => m.forward(input_ids, mask, None),
            Self::Cohere(m) => m.forward(input_ids, mask, None),
            Self::Granite(m) => m.forward(input_ids, mask, None),
            Self::NemotronH(m) => m.forward(input_ids, mask),
            Self::Qwen3Next(m) => m.forward(input_ids, mask),
            Self::Qwen4Exp(m) => m.forward(input_ids, mask),
            Self::GptOss(m) => m.forward(input_ids, mask, None),
            Self::Gemma4(m) => m.forward(input_ids, mask),
            // Text-only: this signature carries no images, so the
            // cross-attention layers are skipped exactly as the reference does
            // for text-only inputs. Image conditioning goes through
            // `as_mllama_mut()`.
            Self::Mllama(m) => m.forward_with_cache(input_ids, mask, None),
            Self::Flux(_) => Err(Exception::custom(
                "Flux is not a CausalLM and does not support standard forward(input_ids, mask)",
            )),
            // BERT encoder: forward returns hidden states [batch, seq, hidden], not logits.
            // Use EmbeddingTrainer::encode() / pmetal_models::pooling::pool() for embeddings.
            Self::Bert(m) => EncoderModel::forward(m, input_ids, mask),
            // DiffusionGemma is a block-autoregressive discrete-diffusion model:
            // it has no single causal next-token forward. Use
            // `as_diffusion_gemma_mut().generate(...)` for sampling, or
            // `forward_hidden` for the encoder trunk representation.
            Self::DiffusionGemma(_) => Err(Exception::custom(
                "DiffusionGemma has no causal forward(input_ids, mask). Use \
                 DynamicModel::as_diffusion_gemma_mut().generate(...) for block-diffusion \
                 sampling, or forward_hidden() for the encoder trunk.",
            )),
        }
    }

    /// Forward pass with one rotary position per token, `[seq_len]`.
    ///
    /// Sequence packing concatenates several training records into one row.
    /// The block-diagonal mask stops one record attending to another, but
    /// nothing stops the positions running straight through the boundary
    /// unless they are given explicitly, which is what this takes.
    /// [`supports_packed_positions`] answers for the architecture *before* a
    /// caller relies on it.
    ///
    /// Plain RoPE rotates by the difference between positions, so the shift a
    /// packed row introduces cancels inside a sealed-off block. What it does
    /// reach is everything reading the absolute position — Llama 4's attention
    /// temperature tuning, Phi-3's LongRoPE table selection — and positions
    /// running past the trained window. `tests/packed_positions.rs` holds
    /// every architecture to the equivalence this is for.
    ///
    /// [`supports_packed_positions`]: Self::supports_packed_positions
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        match self {
            Self::Llama(m) => m.forward_with_positions(input_ids, mask, positions),
            Self::Llama4(m) => m.forward_with_positions(input_ids, mask, positions),
            Self::Qwen2(m) => m.forward_with_positions(input_ids, mask, positions),
            Self::Qwen3(m) => m.forward_with_positions(input_ids, mask, positions),
            Self::Qwen3MoE(m) => m.forward_with_positions(input_ids, mask, positions),
            Self::Qwen3Next(m) => m.forward_with_positions(input_ids, mask, positions),
            Self::Gemma(m) => m.forward_with_positions(input_ids, mask, positions),
            Self::Gemma4(m) => m.forward_with_positions(input_ids, mask, positions),
            Self::Mistral(m) => m.forward_with_positions(input_ids, mask, positions),
            Self::Phi(m) | Self::Phi4(m) => m.forward_with_positions(input_ids, mask, positions),
            Self::DeepSeek(m) => m.forward_with_positions(input_ids, mask, positions),
            Self::Cohere(m) => m.forward_with_positions(input_ids, mask, positions),
            Self::Granite(m) => m.forward_with_positions(input_ids, mask, positions),
            // Its attention carries no positional encoding; the positions tell
            // the Mamba layers where one packed sequence ends and the next
            // begins.
            Self::NemotronH(m) => m.forward_with_positions(input_ids, mask, positions),
            Self::GptOss(m) => m.forward_with_positions(input_ids, mask, positions),
            // Text-only, as for `forward`.
            Self::Mllama(m) => m.forward_with_positions(input_ids, mask, positions),
            // Flux / BERT / DiffusionGemma are not causal LMs.
            other => Err(Exception::custom(format!(
                "forward_with_positions is not implemented for {other:?}; \
                 check supports_packed_positions() before calling"
            ))),
        }
    }

    /// Forward conditioned on **already projected** vision features,
    /// `[batch · images · tiles, patches, hidden]`.
    ///
    /// Raw pixels do not come through here. Mllama's tower needs its
    /// processor's tile geometry — `aspect_ratio_ids` and `aspect_ratio_mask`
    /// — which no uniform batch shape carries;
    /// [`as_mllama_mut`](Self::as_mllama_mut) plus
    /// `prepare_cross_attention` is the path that runs the tower from pixels.
    /// This is the adapter-training shape, where the features are computed
    /// once and reused.
    ///
    /// Errors for an architecture with no cross-attention rather than
    /// quietly running text-only, which is the difference between a VLM run
    /// that failed and one that trained on nothing it was given.
    pub fn forward_with_cross_states(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cross_states: &Array,
    ) -> Result<Array, Exception> {
        match self {
            Self::Mllama(m) => {
                let hidden = m.config.text_config.llama.hidden_size;
                let last = cross_states.shape().last().copied().unwrap_or(0);
                if last != hidden {
                    return Err(Exception::custom(format!(
                        "cross states must already be projected to the text hidden size \
                         ({hidden}), got a last dimension of {last}. Run the vision tower \
                         through as_mllama_mut().prepare_cross_attention(..) first."
                    )));
                }
                let cross = crate::architectures::mllama::CrossAttentionInputs {
                    states: cross_states.clone(),
                    mask: None,
                    full_text_row_mask: None,
                };
                m.forward_full(input_ids, Some(&cross), mask, None)
            }
            other => Err(Exception::custom(format!(
                "{other:?} has no cross-attention layers, so there is nothing for vision \
                 features to condition; check supports_cross_attention() before calling"
            ))),
        }
    }

    /// Whether [`forward_with_cross_states`] conditions on vision features for
    /// this architecture.
    ///
    /// [`forward_with_cross_states`]: Self::forward_with_cross_states
    pub fn supports_cross_attention(&self) -> bool {
        matches!(self, Self::Mllama(_))
    }

    /// Whether [`forward_with_positions`] actually applies the positions for
    /// this architecture, rather than accepting and dropping them.
    ///
    /// [`forward_with_positions`]: Self::forward_with_positions
    pub fn supports_packed_positions(&self) -> bool {
        // A Granite hybrid's attention layers carry no positions (NoPE); its
        // Mamba layers read them to reset where a packed sequence restarts.
        if let Self::Granite(m) = self {
            return m.config.uses_rope() || m.config.has_mamba_layers();
        }
        matches!(
            self,
            Self::Llama(_)
                | Self::Llama4(_)
                | Self::Qwen2(_)
                | Self::Qwen3(_)
                | Self::Qwen3MoE(_)
                | Self::Qwen3Next(_)
                | Self::Gemma(_)
                | Self::Gemma4(_)
                | Self::Mistral(_)
                | Self::Phi(_)
                | Self::Phi4(_)
                | Self::DeepSeek(_)
                | Self::Cohere(_)
                | Self::GptOss(_)
                | Self::Mllama(_)
                // Its Mamba layers reset where the positions restart.
                | Self::NemotronH(_)
        )
    }

    /// Whether this is an encoder-only model, whose `mask` argument is a
    /// `[batch, seq]` padding mask (non-zero for real tokens) rather than the
    /// additive attention mask the causal models take.
    ///
    /// A bidirectional encoder attends over the whole row, so a batch padded to
    /// its longest member embeds each row differently from the same row alone
    /// unless the padding is hidden; a causal model's right-padding is already
    /// out of reach of the real tokens and needs no mask.
    pub fn takes_padding_mask(&self) -> bool {
        matches!(self, Self::Bert(_))
    }

    /// Whether some layer carries recurrent state that two sequences packed
    /// into one row would share. The block-diagonal mask of a packed batch
    /// only reaches attention; a recurrent layer has to reset at the
    /// boundary itself. The Mamba-2 layers of Nemotron-H and Granite do (they
    /// read the position ids, see [`forward_with_positions`]); the gated delta
    /// net of Qwen 3.5 and Qwen4-Exp does not, and those models are trained on
    /// unpacked batches.
    ///
    /// [`forward_with_positions`]: Self::forward_with_positions
    pub fn has_unresettable_recurrent_layers(&self) -> bool {
        self.create_mamba_cache().is_some()
            && !matches!(self, Self::NemotronH(_) | Self::Granite(_))
    }

    /// Forward pass returning last-layer hidden states `[batch, seq, hidden]`
    /// — the pre-lm-head representation used for sentence embeddings and
    /// `/v1/embeddings`-style pooling endpoints.
    ///
    /// Coverage: every dense decoder arch whose `ForCausalLM` wraps a
    /// `pub model: *Model` field routes through that inner trunk. BERT's
    /// canonical `forward` already returns hidden states so it's a
    /// pass-through (its `mask` is a padding mask, see
    /// [`takes_padding_mask`](Self::takes_padding_mask)). Architectures not listed here (Flux, Qwen3MoE, hybrid
    /// attn+mamba variants, etc.) return a typed error — each has a
    /// non-trivial trunk exit point (MoE routing, image conditioning,
    /// dual-cache hybrid state) that the caller needs to opt into
    /// explicitly rather than silently pool over.
    pub fn forward_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        match self {
            Self::Llama(m) => m.model.forward(input_ids, mask),
            // Llama4TextModel takes position_ids in the 3rd slot; None
            // lets the forward derive positions internally. MoD routing
            // in the decoder layers runs fine without a cache.
            Self::Llama4(m) => m.model.forward(input_ids, mask, None),
            Self::Qwen2(m) => m.model.forward(input_ids, mask),
            Self::Qwen3(m) => m.model.forward(input_ids, mask, None),
            Self::Qwen3MoE(m) => m.model.forward(input_ids, mask, None),
            Self::DeepSeek(m) => m.model.forward(input_ids, mask, None),
            // Cohere / Granite inner models take position_ids in the
            // 3rd slot (not cache); `None` is fine for embeddings.
            Self::Cohere(m) => m.model.forward(input_ids, mask, None),
            Self::Granite(m) => m.model.forward(input_ids, mask, None),
            Self::GptOss(m) => m.model.forward(input_ids, mask, None),
            // Hybrid attn+mamba / linear-attn archs — forward runs with
            // caches elided (None).
            Self::NemotronH(m) => m.backbone.forward(input_ids, mask),
            // Qwen3Next linear-attn + Mamba hybrid — cacheless forward
            // wraps the standard dispatch with None/None caches.
            Self::Qwen3Next(m) => m.model.forward(input_ids, mask),
            // The final hyper-connection mix, which is what the LM head reads.
            Self::Qwen4Exp(m) => m.model.forward(input_ids, mask),
            Self::Mistral(m) => m.model.forward(input_ids, mask),
            Self::Gemma(m) => m.model.forward(input_ids, mask),
            // Gemma4 / Phi / Phi4 inner models expose only forward_with_cache;
            // pass None for the cache — embeddings don't need incremental decode.
            Self::Gemma4(m) => m.model.forward_with_cache(input_ids, mask, None),
            Self::Phi(m) => m.model.forward_with_cache(input_ids, mask, None),
            Self::Phi4(m) => m.model.forward_with_cache(input_ids, mask, None),
            Self::Bert(m) => EncoderModel::forward(m, input_ids, mask),
            // DiffusionGemma: the causal encoder trunk is the natural
            // pre-LM-head representation to pool over (the decoder denoises a
            // canvas and has no single hidden-state-per-input-token output).
            Self::DiffusionGemma(m) => m.encode_hidden(input_ids),
            // Text-only trunk, as for `forward`.
            Self::Mllama(m) => m.forward_hidden(input_ids, mask),
            other => Err(Exception::custom(format!(
                "forward_hidden not implemented for {:?} — supported archs: \
                 Llama, Llama4, Qwen2, Qwen3, Qwen3MoE, Mistral, Gemma, \
                 Gemma4, Phi, Phi4, DeepSeek, Cohere, Granite, GptOss, \
                 NemotronH, Qwen3Next, BERT, DiffusionGemma, Mllama",
                other
            ))),
        }
    }

    /// [`forward_hidden`] with one rotary position per token, `[seq_len]`:
    /// the trunk [`forward_with_positions`] runs before its LM head, which is
    /// what cut cross-entropy needs from a packed batch.
    ///
    /// `None` is [`forward_hidden`]. With positions, the architectures are
    /// those of [`supports_packed_positions`], and any other errors as
    /// [`forward_with_positions`] does.
    ///
    /// [`forward_hidden`]: Self::forward_hidden
    /// [`forward_with_positions`]: Self::forward_with_positions
    /// [`supports_packed_positions`]: Self::supports_packed_positions
    pub fn forward_hidden_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        positions: Option<&Array>,
    ) -> Result<Array, Exception> {
        let Some(p) = positions else {
            return self.forward_hidden(input_ids, mask);
        };
        let p = Some(p);
        match self {
            Self::Llama(m) => m.model.forward_with_positions(input_ids, mask, p),
            Self::Llama4(m) => m.model.forward(input_ids, mask, p),
            Self::Qwen2(m) => m.model.forward_with_positions(input_ids, mask, p),
            Self::Qwen3(m) => m.model.forward_with_positions(input_ids, mask, p),
            Self::Qwen3MoE(m) => m.model.forward_with_positions(input_ids, mask, p),
            Self::Qwen3Next(m) => m.model.forward_with_positions(input_ids, mask, p),
            Self::Gemma(m) => m.model.forward_with_positions(input_ids, mask, p),
            Self::Gemma4(m) => m.model.forward_with_positions(input_ids, mask, p),
            Self::Mistral(m) => m.model.forward_with_positions(input_ids, mask, p),
            Self::Phi(m) | Self::Phi4(m) => m.model.forward_with_positions(input_ids, mask, p),
            Self::DeepSeek(m) => m.model.forward_with_positions(input_ids, mask, p),
            Self::Cohere(m) => m.model.forward_with_positions(input_ids, mask, p),
            Self::Granite(m) => m.model.forward_with_positions(input_ids, mask, p),
            Self::NemotronH(m) => m.backbone.forward_with_positions(input_ids, mask, p),
            Self::GptOss(m) => m.model.forward_with_positions(input_ids, mask, p),
            Self::Mllama(m) => m
                .language_model
                .forward_with_positions(input_ids, None, mask, p),
            other => Err(Exception::custom(format!(
                "forward_hidden_with_positions is not implemented for {other:?}; \
                 check supports_packed_positions() before calling"
            ))),
        }
    }

    /// [`forward_with_cache`](Self::forward_with_cache) returning the last
    /// position's logits, `[B, 1, vocab]`, for a caller that samples from them
    /// alone: a prefill, or a decode step.
    ///
    /// The LM head runs on that one position where the architecture has this
    /// path (Llama, Qwen 2/3, Qwen3-MoE, Mistral, Phi, Cohere, Granite); the
    /// others compute every position's logits and slice, which is the same
    /// numbers at the cost of the `[B, T, vocab]` allocation. The cache is
    /// filled as the full forward fills it.
    pub fn forward_with_cache_last(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        match self {
            Self::Llama(m) => m.forward_with_cache_last(input_ids, mask, cache),
            Self::Qwen2(m) => m.forward_with_cache_last(input_ids, mask, cache),
            Self::Qwen3(m) => m.forward_with_cache_last(input_ids, mask, cache),
            Self::Qwen3MoE(m) => m.forward_with_cache_last(input_ids, mask, cache),
            Self::Cohere(m) => m.forward_with_cache_last(input_ids, mask, cache),
            Self::Granite(m) => m.forward_with_cache_last(input_ids, mask, cache),
            Self::Mistral(m) => m.forward_with_cache_last(input_ids, mask, cache),
            Self::Phi(m) | Self::Phi4(m) => m.forward_with_cache_last(input_ids, mask, cache),
            _ => Ok(last_token_hidden(
                &self.forward_with_cache(input_ids, mask, cache)?,
            )),
        }
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, Exception> {
        match self {
            Self::Llama(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Qwen2(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Qwen3(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Qwen3MoE(m) => m.forward(input_ids, mask, cache),
            Self::DeepSeek(m) => m.forward(input_ids, mask, cache),
            Self::Cohere(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Granite(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Llama4(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Gemma(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Mistral(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Phi(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::Phi4(m) => m.forward_with_cache(input_ids, mask, cache),
            Self::GptOss(m) => m.forward(input_ids, mask, cache),
            Self::Gemma4(m) => m.forward_with_cache(input_ids, mask, cache),
            // Text-only cached decode. Image-conditioned decoding needs the
            // vision features re-supplied every step, which this signature
            // cannot express — use
            // `as_mllama_mut().prepare_cross_attention(..)` + `forward_full`.
            Self::Mllama(m) => m.forward_with_cache(input_ids, mask, cache),
            // Hybrid recurrent+attention models require both a KV cache and a
            // Mamba/GDN state cache. Use `forward_with_hybrid_cache` instead.
            Self::NemotronH(_) | Self::Qwen3Next(_) | Self::Qwen4Exp(_) => Err(Exception::custom(
                "Hybrid architecture requires both KV and Mamba caches. \
                 Use DynamicModel::forward_with_hybrid_cache with both \
                 create_cache() and create_mamba_cache().",
            )),
            Self::Flux(_) => Err(Exception::custom(
                "Flux is not a CausalLM and does not support forward_with_cache.",
            )),
            // BERT is encoder-only (no autoregressive cache) — delegate to standard forward.
            Self::Bert(m) => EncoderModel::forward(m, input_ids, mask),
            // DiffusionGemma's KV cache is internal to its encoder–decoder
            // generate loop and is not a standard causal KV cache.
            Self::DiffusionGemma(_) => Err(Exception::custom(
                "DiffusionGemma does not support forward_with_cache. Use \
                 DynamicModel::as_diffusion_gemma_mut().generate(...).",
            )),
        }
    }

    /// Recompute each decoder layer's activations during the backward pass
    /// instead of holding them for it.
    ///
    /// Peak activation memory stops scaling with depth, at the cost of one
    /// extra forward per step. Training only: the flag is ignored on any
    /// forward that carries a KV cache, since generation has no backward pass
    /// for the recompute to pay for.
    ///
    /// Returns `false` when this architecture's trunk does not implement it, so
    /// a caller can report that honestly rather than assume it took effect.
    pub fn set_gradient_checkpointing(&mut self, enabled: bool) -> bool {
        match self.grad_checkpoint_flag_mut() {
            Some(flag) => {
                *flag = enabled;
                true
            }
            None => false,
        }
    }

    /// Whether this architecture's trunk can checkpoint its layers.
    ///
    /// Callers have to ask before enabling, and enabling needs `&mut self`,
    /// which is why this is separate from [`set_gradient_checkpointing`].
    ///
    /// [`set_gradient_checkpointing`]: Self::set_gradient_checkpointing
    pub fn supports_gradient_checkpointing(&self) -> bool {
        self.grad_checkpoint_flag().is_some()
    }

    /// Whether checkpointing is currently on.
    pub fn is_gradient_checkpointing(&self) -> bool {
        self.grad_checkpoint_flag() == Some(true)
    }

    /// The trunk's checkpointing flag, for architectures that have one.
    ///
    /// This and its shared-reference twin are the only places that name the
    /// supported architectures. Adding one means adding an arm to both, which
    /// `the_support_check_and_the_setter_agree` in `pmetal-lora` holds together.
    fn grad_checkpoint_flag_mut(&mut self) -> Option<&mut bool> {
        match self {
            Self::Llama(m) => Some(&mut m.model.grad_checkpoint),
            Self::Llama4(m) => Some(&mut m.model.grad_checkpoint),
            Self::Qwen2(m) => Some(&mut m.model.grad_checkpoint),
            Self::Qwen3(m) => Some(&mut m.model.grad_checkpoint),
            Self::Qwen3MoE(m) => Some(&mut m.model.grad_checkpoint),
            Self::Gemma(m) => Some(&mut m.model.grad_checkpoint),
            Self::Mistral(m) => Some(&mut m.model.grad_checkpoint),
            Self::Phi(m) | Self::Phi4(m) => Some(&mut m.model.grad_checkpoint),
            Self::DeepSeek(m) => Some(&mut m.model.grad_checkpoint),
            Self::Cohere(m) => Some(&mut m.model.grad_checkpoint),
            Self::Granite(m) => Some(&mut m.model.grad_checkpoint),
            Self::NemotronH(m) => Some(&mut m.backbone.grad_checkpoint),
            Self::GptOss(m) => Some(&mut m.model.grad_checkpoint),
            // Gemma 4, Qwen3-Next and Mllama are left out on
            // purpose. Each has a trunk whose layers read or write state
            // belonging to another layer (Gemma 4's shared KV, Qwen 3.5's
            // recurrent caches, Mllama's cross-attention), and a
            // recompute would replay those side effects a second time. They
            // need the state threading through `checkpoint` as an input, not
            // just an extra call site.
            _ => None,
        }
    }

    fn grad_checkpoint_flag(&self) -> Option<bool> {
        match self {
            Self::Llama(m) => Some(m.model.grad_checkpoint),
            Self::Llama4(m) => Some(m.model.grad_checkpoint),
            Self::Qwen2(m) => Some(m.model.grad_checkpoint),
            Self::Qwen3(m) => Some(m.model.grad_checkpoint),
            Self::Qwen3MoE(m) => Some(m.model.grad_checkpoint),
            Self::Gemma(m) => Some(m.model.grad_checkpoint),
            Self::Mistral(m) => Some(m.model.grad_checkpoint),
            Self::Phi(m) | Self::Phi4(m) => Some(m.model.grad_checkpoint),
            Self::DeepSeek(m) => Some(m.model.grad_checkpoint),
            Self::Cohere(m) => Some(m.model.grad_checkpoint),
            Self::Granite(m) => Some(m.model.grad_checkpoint),
            Self::NemotronH(m) => Some(m.backbone.grad_checkpoint),
            Self::GptOss(m) => Some(m.model.grad_checkpoint),
            _ => None,
        }
    }

    /// The token-embedding layer, for architectures that have a single one.
    ///
    /// The input embedding is where NEFTune noise goes, and where a caller that
    /// wants to run the trunk on embeddings it built itself (multimodal, soft
    /// prompts) has to reach. Excludes Flux, which has no token embedding.
    pub fn token_embedding_mut(&mut self) -> Option<&mut nn::Embedding> {
        match self {
            Self::Llama(m) => Some(&mut m.model.embed_tokens),
            Self::Llama4(m) => Some(&mut m.model.embed_tokens),
            Self::Qwen2(m) => Some(&mut m.model.embed_tokens),
            Self::Qwen3(m) => Some(&mut m.model.embed_tokens),
            Self::Qwen3MoE(m) => Some(&mut m.model.embed_tokens),
            Self::Qwen3Next(m) => Some(&mut m.model.embed_tokens),
            Self::Qwen4Exp(m) => Some(&mut m.model.embed_tokens),
            Self::Gemma(m) => Some(&mut m.model.embed_tokens),
            Self::Gemma4(m) => Some(&mut m.model.embed_tokens),
            Self::Mistral(m) => Some(&mut m.model.embed_tokens),
            Self::Phi(m) | Self::Phi4(m) => Some(&mut m.model.embed_tokens),
            Self::DeepSeek(m) => Some(&mut m.model.embed_tokens),
            Self::Cohere(m) => Some(&mut m.model.embed_tokens),
            Self::Granite(m) => Some(&mut m.model.embed_tokens),
            Self::GptOss(m) => Some(&mut m.model.embed_tokens),
            Self::NemotronH(m) => Some(&mut m.backbone.embeddings),
            Self::Mllama(m) => Some(&mut m.language_model.embed_tokens),
            Self::Bert(_) | Self::DiffusionGemma(_) | Self::Flux(_) => None,
        }
    }

    /// The untied LM head, when this architecture has one.
    fn lm_head_layer(&self) -> Option<&nn::Linear> {
        match self {
            Self::Llama(m) => m.lm_head.as_ref(),
            Self::Llama4(m) => m.lm_head.as_ref(),
            Self::Qwen2(m) => m.lm_head.as_ref(),
            Self::Qwen3(m) => m.lm_head.as_ref(),
            Self::Qwen3MoE(m) => m.lm_head.as_ref(),
            Self::Qwen3Next(m) => m.lm_head.as_ref(),
            Self::Qwen4Exp(m) => m.lm_head.as_ref(),
            Self::Mistral(m) => m.lm_head.as_ref(),
            Self::Phi(m) | Self::Phi4(m) => m.lm_head.as_ref(),
            Self::DeepSeek(m) => m.lm_head.as_ref(),
            Self::Cohere(m) => m.lm_head.as_ref(),
            Self::Granite(m) => m.lm_head.as_ref(),
            Self::GptOss(m) => m.lm_head.as_ref(),
            Self::NemotronH(m) => m.lm_head.as_ref(),
            Self::Mllama(m) => Some(&m.lm_head),
            Self::Gemma(_)
            | Self::Gemma4(_)
            | Self::Bert(_)
            | Self::DiffusionGemma(_)
            | Self::Flux(_) => None,
        }
    }

    /// Where the projection onto the vocabulary sits in the walk
    /// [`VisitLinears`](pmetal_runtime::compat::VisitLinears) makes, when it is
    /// a layer of its own; `None` when the model ties it to the token
    /// embedding or has none.
    ///
    /// Found by identity rather than by name, so it holds whatever the layer
    /// is called. This is the layer an empty LoRA `target_modules` leaves
    /// out, as the model's output embedding is in PEFT's `all-linear`.
    pub fn output_projection_path(&mut self) -> Option<String> {
        let head: *const nn::Linear = self.lm_head_layer()?;
        let mut found = None;
        pmetal_runtime::compat::VisitLinears::visit_linears_mut(self, "", &mut |path, linear| {
            if std::ptr::eq(linear, head) {
                found = Some(path.to_string());
            }
        });
        found
    }

    /// The LM head as cut cross-entropy computes the loss from it: the
    /// logits of `hidden`, what [`forward_hidden`](Self::forward_hidden)
    /// returns, are `softcap(logit_scale · (hidden·Wᵀ + b))`.
    ///
    /// The weight is the one [`Linear::forward`](nn::Linear::forward) would
    /// multiply by, adapter folded in, so a loss built on it trains an adapter
    /// on the head as the forward pass would; take it inside the function
    /// being differentiated, where the adapter's parameters are the traced
    /// ones. A model that ties its embeddings uses the embedding matrix.
    ///
    /// `None` when the logits are not of that form, or when no single weight
    /// reproduces the head (an adapter on it applying dropout).
    pub fn lm_head(&self) -> Option<LmHead> {
        let (logit_scale, softcap) = match self {
            // `logits * logit_scale`.
            Self::Cohere(m) => (m.config.logit_scale, None),
            // `logits / logits_scaling`.
            Self::Granite(m) => (m.config.logits_scaling.recip(), None),
            // `cap · tanh(logits / cap)`.
            Self::Gemma(m) => (1.0, m.model.config.final_logit_softcapping),
            Self::Gemma4(m) => (1.0, m.config.final_logit_softcapping),
            Self::Bert(_) | Self::DiffusionGemma(_) | Self::Flux(_) => return None,
            _ => (1.0, None),
        };
        let (weight, bias) = match self.lm_head_layer() {
            Some(linear) => (linear.effective_weight()?, linear.bias.value.clone()),
            None => {
                let params = self.flatten_params();
                let tied = params.get("model.embed_tokens.weight")?;
                let tied = if tied.dtype() == Dtype::Uint8 {
                    ops::from_fp8(tied, Dtype::Bfloat16).ok()?
                } else {
                    tied.clone()
                };
                (tied, None)
            }
        };
        Some(LmHead {
            weight,
            bias,
            logit_scale,
            softcap: softcap.filter(|cap| *cap != 0.0),
        })
    }

    /// The matrix that projects hidden states onto the vocabulary,
    /// `[vocab, hidden]` and dense. See [`lm_head`](Self::lm_head).
    pub fn lm_head_weight(&self) -> Option<Array> {
        self.lm_head().map(|head| head.weight)
    }

    /// Set the NEFTune noise scale on the token embedding, or clear it.
    ///
    /// NEFTune (Jain et al., 2023) perturbs the embedding output during
    /// training. `alpha` is typically 5 to 15; `None` turns it off.
    ///
    /// Returns `false` when this architecture has no single token embedding to
    /// put it on, so a caller can say so rather than assume it took.
    pub fn set_neftune_alpha(&mut self, alpha: Option<f32>) -> bool {
        match self.token_embedding_mut() {
            Some(embedding) => {
                embedding.neftune_alpha = alpha;
                true
            }
            None => false,
        }
    }

    pub fn quantize_fp8(&mut self) -> Result<(), Exception> {
        match self {
            // Flux is a diffusion model whose weight graph is not a flat set of
            // `nn::Linear` layers reachable through a single `ModuleParameters`
            // root.  Callers should use `FluxPipeline` and quantize each
            // component (transformer, VAE, text encoders) individually.
            Self::Flux(_) => Err(Exception::custom(
                "Flux FP8 quantization is not exposed through DynamicModel. \
                 Load the diffusion pipeline via pmetal_models::pipelines::FluxPipeline \
                 and quantize its components explicitly.",
            )),

            // All other causal-LM architectures: traverse the flattened
            // parameter map and quantize every `.weight` tensor generically.
            Self::Llama(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Llama4(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Qwen2(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Qwen3(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Qwen3MoE(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Gemma(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Mistral(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Phi(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Phi4(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::DeepSeek(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Cohere(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Granite(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::NemotronH(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Qwen3Next(m) => crate::fp8_utils::quantize_model_linears(m),
            // The n-gram tables are embeddings, which the generic pass leaves alone.
            Self::Qwen4Exp(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::GptOss(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Gemma4(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Bert(m) => crate::fp8_utils::quantize_model_linears(m),
            // Generic linear-weight quantization; the fused 3-D expert tensors
            // (not `.weight`) stay in their loaded dtype, same as other MoE archs.
            Self::DiffusionGemma(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Mllama(m) => crate::fp8_utils::quantize_model_linears(m),
        }
    }

    /// The dtype the model computes in: that of its floating-point parameters.
    ///
    /// Read from a one-dimensional parameter (a norm weight, which no quantizer
    /// touches), falling back to any floating-point one, never from a packed
    /// weight: a `uint32` payload read as the model dtype is silent noise.
    /// `Float32` for a model with no floating-point parameter at all.
    pub fn compute_dtype(&self) -> Dtype {
        let params = self.flatten_params();
        params
            .iter()
            .filter(|(_, array)| {
                matches!(
                    array.dtype(),
                    Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32
                )
            })
            .min_by_key(|(name, array)| (array.ndim() != 1, name.to_string()))
            .map_or(Dtype::Float32, |(_, array)| array.dtype())
    }

    /// A KV cache for `max_seq_len` tokens, holding keys and values in the
    /// model's [`compute_dtype`](Self::compute_dtype).
    ///
    /// The layout (layers, heads, head size) is the architecture's own. The
    /// dtype used to be `KVCacheConfig`'s default of f32 whatever the model ran
    /// in, which doubled every memory estimate made from the cache's config and
    /// allocated a fused batch cache twice the size of the one the model fills.
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        let layout = self.create_cache_layout(max_seq_len);
        let config = layout.config().clone().with_dtype(self.compute_dtype());
        KVCache::new(config)
    }

    fn create_cache_layout(&self, max_seq_len: usize) -> KVCache {
        match self {
            Self::Llama(m) => m.create_cache(max_seq_len),
            Self::Llama4(m) => KVCache::new(KVCacheConfig::new(
                m.config.num_hidden_layers as usize,
                max_seq_len,
                m.config.num_key_value_heads as usize,
                m.config.head_dim as usize,
            )),
            Self::Qwen2(m) => KVCache::new(KVCacheConfig::new(
                m.config().num_hidden_layers() as usize,
                max_seq_len,
                m.config().num_kv_heads() as usize,
                m.config().head_dim() as usize,
            )),
            Self::Qwen3(m) => KVCache::new(KVCacheConfig::new(
                m.config().num_hidden_layers() as usize,
                max_seq_len,
                m.config().num_kv_heads() as usize,
                m.config().head_dim() as usize,
            )),
            Self::Qwen3MoE(m) => KVCache::new(KVCacheConfig::new(
                m.config.num_hidden_layers as usize,
                max_seq_len,
                m.config.num_kv_heads() as usize,
                (m.config.hidden_size / m.config.num_attention_heads) as usize,
            )),
            Self::Gemma(m) => m.create_cache(max_seq_len),
            Self::Mistral(m) => m.create_cache(max_seq_len),
            Self::Phi(m) => m.create_cache(max_seq_len),
            Self::Phi4(m) => m.create_cache(max_seq_len),
            Self::DeepSeek(m) => m.create_cache(max_seq_len),
            Self::Cohere(m) => KVCache::new(KVCacheConfig::new(
                m.config.num_hidden_layers as usize,
                max_seq_len,
                m.config.num_key_value_heads as usize,
                (m.config.hidden_size / m.config.num_attention_heads) as usize,
            )),
            Self::Granite(m) => m.create_cache(max_seq_len),
            Self::NemotronH(m) => KVCache::new(KVCacheConfig::new(
                m.config().num_hidden_layers() as usize,
                max_seq_len,
                m.config().num_kv_heads() as usize,
                m.config().head_dim() as usize,
            )),
            Self::Qwen3Next(m) => KVCache::new(KVCacheConfig::new(
                m.config().num_hidden_layers() as usize,
                max_seq_len,
                m.config().num_kv_heads() as usize,
                m.config().head_dim() as usize,
            )),
            Self::Qwen4Exp(m) => m.create_cache(max_seq_len),
            Self::GptOss(m) => KVCache::new(KVCacheConfig::new(
                m.config().num_hidden_layers as usize,
                max_seq_len,
                m.config().num_key_value_heads as usize,
                m.config().head_dim as usize,
            )),
            Self::Gemma4(m) => KVCache::new(KVCacheConfig::new(
                m.config.num_hidden_layers as usize,
                max_seq_len,
                m.config.num_key_value_heads as usize,
                m.config.head_dim as usize,
            )),
            Self::Flux(_) => KVCache::new(KVCacheConfig::new(0, 0, 0, 0)),
            // BERT is encoder-only with no autoregressive KV cache.
            Self::Bert(_) => KVCache::new(KVCacheConfig::new(0, 0, 0, 0)),
            // DiffusionGemma manages its encoder KV internally inside generate();
            // there is no external standard causal cache.
            Self::DiffusionGemma(_) => KVCache::new(KVCacheConfig::new(0, 0, 0, 0)),
            // Sized for the text decoder. Cross-attention layers hold no
            // self-attention KV, so their slots are allocated and left unused —
            // that keeps slot indices equal to layer indices.
            Self::Mllama(m) => {
                let tc = &m.config.text_config.llama;
                KVCache::new(KVCacheConfig::new(
                    tc.num_hidden_layers as usize,
                    max_seq_len,
                    tc.num_kv_heads() as usize,
                    tc.get_head_dim() as usize,
                ))
            }
        }
    }

    /// Create a KV cache with a specific cache mode (e.g., quantized).
    ///
    /// This builds the same cache configuration as `create_cache` but applies
    /// the specified mode. Use `CacheMode::Quantized { bits: 8, group_size: 64 }`
    /// for the recommended q8_0 "free lunch" (< 0.4% PPL loss, 12-38% throughput gain).
    pub fn create_cache_with_mode(&self, max_seq_len: usize, mode: CacheMode) -> KVCache {
        let base = self.create_cache(max_seq_len);
        let base_config = base.config();
        let safe_mode = sanitize_cache_mode_for_config(base_config, mode);
        if safe_mode != mode {
            tracing::info!(
                requested = %mode.describe(),
                normalized = %safe_mode.describe(),
                key_head_dim = base_config.head_dim,
                value_head_dim = base_config.value_head_dim,
                "KV cache: normalized requested cache mode"
            );
        }

        let config = base_config.clone().with_mode(safe_mode);
        KVCache::new(config)
    }

    pub fn create_mamba_cache(&self) -> Option<MambaCache> {
        match self {
            Self::NemotronH(m) => Some(MambaCache::new(m.config().num_hidden_layers() as usize)),
            Self::Qwen3Next(m) => Some(MambaCache::new(m.config().num_hidden_layers() as usize)),
            // GDN state plus the indexer and PLE states; see `Qwen4ExpCacheLayout`.
            Self::Qwen4Exp(m) => Some(m.create_mamba_cache()),
            // Only a Granite with Mamba layers; plain and MoE Granite are
            // attention-only.
            Self::Granite(m) => m.create_mamba_cache(),
            _ => None,
        }
    }

    pub fn forward_with_hybrid_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        match self {
            Self::NemotronH(m) => m.forward_with_cache(input_ids, mask, kv_cache, mamba_cache),
            Self::Qwen3Next(m) => m.forward_with_cache(input_ids, mask, kv_cache, mamba_cache),
            Self::Qwen4Exp(m) => m.forward_with_cache(input_ids, mask, kv_cache, mamba_cache),
            Self::Granite(m) => m.forward_with_hybrid_cache(input_ids, mask, kv_cache, mamba_cache),
            _ => self.forward_with_cache(input_ids, mask, kv_cache),
        }
    }

    /// [`forward_with_hybrid_cache`](Self::forward_with_hybrid_cache) returning
    /// the last position's logits, `[B, 1, vocab]`, for a caller that samples
    /// from them alone: a prefill, or a decode step. Qwen 3.5, Granite and the
    /// architectures of [`forward_with_cache_last`](Self::forward_with_cache_last)
    /// run their head on that one position; the others compute every position's
    /// and slice.
    pub fn forward_with_hybrid_cache_last(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, Exception> {
        match self {
            Self::Qwen3Next(m) => m.forward_with_cache_last(input_ids, mask, kv_cache, mamba_cache),
            Self::Granite(m) => {
                m.forward_with_hybrid_cache_last(input_ids, mask, kv_cache, mamba_cache)
            }
            Self::NemotronH(m) => m.forward_with_cache_last(input_ids, mask, kv_cache, mamba_cache),
            Self::Qwen4Exp(m) => m.forward_with_cache_last(input_ids, mask, kv_cache, mamba_cache),
            _ => self.forward_with_cache_last(input_ids, mask, kv_cache),
        }
    }

    /// Generate from `ids` with the model's own caches (the attention KV cache
    /// and, for a hybrid architecture, the recurrent state), as `pmetal infer`
    /// does: the pipelined decode loop, calling `on_token` with each token as
    /// it is read back. `on_token` returns `false` to stop.
    ///
    /// The one call every in-process caller that has a model in hand makes
    /// (the Python bindings, task evaluation), so the cache sizing and the
    /// hybrid wiring are written once.
    pub fn generate_streaming(
        &mut self,
        ids: &[u32],
        config: GenerationConfig,
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<GenerationOutput, Exception> {
        let mut cache = self.create_cache(ids.len() + config.max_new_tokens + 64);
        let mut mamba = self.create_mamba_cache();
        crate::generation::generate_cached_async_streaming(
            |input, cache| {
                self.forward_with_hybrid_cache_last(input, None, Some(cache), mamba.as_mut())
            },
            ids,
            config,
            &mut cache,
            on_token,
        )
    }

    /// Whether this architecture has a fused `[N, 1]` batched-decode
    /// implementation. Continuous batching falls back to the per-slot
    /// serial path when this is `false`. Hybrid attn+recurrent archs
    /// (Mamba/GDN) and bespoke attention variants (MLA, MoD) stay on
    /// the fallback until dedicated fused designs land.
    ///
    /// Per-arch gates reject configs the fused path can't currently
    /// handle (sliding-window attention, softcapping variants that
    /// haven't been plumbed through, etc.). When adding a new arch,
    /// default to `false` and expose `true` only after a parity test.
    pub fn supports_fused_batched(&self) -> bool {
        match self {
            // Every arch below gates on `has_scalar_rope`: `BatchedGqaAttnCfg`
            // is scalar-only by design, carrying one `rope_base` and position
            // scale, so a scaled RoPE (Llama 3 bands, YaRN, LongRoPE, dynamic
            // NTK) takes the serial fallback. The fused path would rotate it
            // as plain RoPE and disagree with the model's own serial decode;
            // `BatchedGqaAttnCfg::for_rotary` refuses if one slips through.
            Self::Llama(m) => m.has_scalar_rope(),
            Self::Mistral(m) => m.config().sliding_window.is_none() && m.has_scalar_rope(),
            Self::Qwen2(m) => !m.config().use_sliding_window && m.has_scalar_rope(),
            Self::Qwen3(m) => !m.config.use_sliding_window && m.has_scalar_rope(),
            // Qwen3-MoE reuses the GQA attention block (with qk-norm); the
            // token-level MoE router already flattens `[N, 1, H]` cleanly
            // via `forward_stacked`.
            Self::Qwen3MoE(m) => m.has_scalar_rope(),
            // GPT-OSS: takes the serial decode path. Its attention has learned
            // per-head *sinks* (an extra softmax-denominator term) and YARN
            // per-dim RoPE frequencies, neither of which the shared scalar
            // `BatchedGqaAttnCfg` + `fused_sdpa` block can express. The serial
            // `GptOssAttention::forward` applies both correctly; enabling the
            // fused path would silently drop the sink term. Re-enable only
            // after threading sinks through `batched_gqa_attn`.
            Self::GptOss(_) => false,
            // Gemma1 uses `batched_prenorm_layer`. Gemma2 uses the 4-norm
            // peri-norm helper plus optional per-layer sliding window and
            // attention logit softcap.
            //
            // Gemma 3 takes the serial fallback. Two things the fused skeleton
            // cannot express: its QK-norm, which the `BatchedGqaAttnCfg` path
            // has no slot for between projection and RoPE, and its two RoPE
            // bases — the cfg carries one scalar `rope_base`, read from
            // `layers[0]`, which for Gemma 3 is a *sliding* layer, so every
            // global layer would get the local base. Correct and serial beats
            // fast and wrong.
            Self::Gemma(m) => !m.config().is_gemma3 && m.has_scalar_rope(),
            // Phi/Phi4: partial RoPE handled by `BatchedGqaAttnCfg::with_rope_dims`.
            // LongRoPE configs take the serial fallback like every scaled RoPE.
            // Sliding-window Phi configs likewise stay on serial until the
            // per-layer overlay is wired into the arch loop.
            Self::Phi(m) | Self::Phi4(m) => {
                m.config().sliding_window.is_none() && m.has_scalar_rope()
            }
            // Cohere: parallel decoder block via `batched_parallel_block`.
            // Sliding-window-with-non-global-layers configs need a per-layer
            // sliding overlay; defer those to a follow-up.
            Self::Cohere(m) => {
                // The fused block is one global, rotated, un-normalized
                // attention shape: Cohere2 layers and Command R+'s query/key
                // norm take the serial path.
                !m.config.has_cohere2_layers() && !m.config.use_qk_norm && m.has_scalar_rope()
            }
            // Granite: plain `granite` configs route through
            // `batched_prenorm_layer`. The MoE and hybrid families (experts,
            // shared MLP, Mamba state) stay on serial.
            //
            // So do configs with a non-unit `residual_multiplier` — which is
            // every released Granite. `batched_prenorm_layer` adds both
            // residual branches unscaled and has no channel for the scale, so
            // taking the fused path would silently drop it. Lifting this means
            // threading the multiplier into that helper.
            Self::Granite(m) => m.supports_fused_batched(),
            _ => false,
        }
    }

    /// Fused batched-decode forward: runs one `[N_active, 1, V]` forward
    /// per tick against a shared [`FusedBatchKVCache`].
    ///
    /// # Arguments
    /// * `input_ids` — `[N_active, 1]` int32 token ids, one per active slot.
    /// * `active_indices` — batch rows in the fused cache corresponding to
    ///   each row of `input_ids`, in the same order.
    /// * `cache` — fused per-layer KV store shared across all slots.
    ///
    /// Returns logits of shape `[N_active, 1, vocab_size]`.
    ///
    /// Architectures that opt in must override via a dedicated match arm
    /// populated by the Tier-1 / Tier-2 rollouts. The default path returns
    /// a typed error so callers gate on [`Self::supports_fused_batched`].
    pub fn forward_batched(
        &mut self,
        input_ids: &Array,
        active_indices: &[usize],
        cache: &mut FusedBatchKVCache,
    ) -> Result<Array, Exception> {
        match self {
            Self::Llama(m) => m.forward_batched_impl(input_ids, active_indices, cache),
            Self::Mistral(m) => m.forward_batched_impl(input_ids, active_indices, cache),
            Self::Qwen2(m) => m.forward_batched_impl(input_ids, active_indices, cache),
            Self::Qwen3(m) => m.forward_batched_impl(input_ids, active_indices, cache),
            Self::Qwen3MoE(m) => m.forward_batched_impl(input_ids, active_indices, cache),
            // GPT-OSS intentionally omitted — `supports_fused_batched` returns
            // false (sinks/YARN can't ride the shared fused block), so this
            // falls through to the typed error below rather than silently
            // dropping the sink term on a mis-gated call.
            Self::Gemma(m) => m.forward_batched_impl(input_ids, active_indices, cache),
            Self::Phi(m) => m.forward_batched_impl(input_ids, active_indices, cache),
            Self::Phi4(m) => m.forward_batched_impl(input_ids, active_indices, cache),
            Self::Cohere(m) => m.forward_batched_impl(input_ids, active_indices, cache),
            Self::Granite(m) => m.forward_batched_impl(input_ids, active_indices, cache),
            other => Err(Exception::custom(format!(
                "forward_batched not implemented for {:?} — check supports_fused_batched first",
                other.architecture()
            ))),
        }
    }

    pub fn architecture(&self) -> ModelArchitecture {
        dispatch_architecture!(self)
    }

    /// Access the underlying Qwen3NextForCausalLM if this is a Qwen3.5 model.
    pub fn as_qwen3_next_mut(&mut self) -> Option<&mut Qwen3NextForCausalLM> {
        match self {
            Self::Qwen3Next(m) => Some(m),
            _ => None,
        }
    }

    /// Access the underlying [`DiffusionGemmaForBlockDiffusion`] for
    /// block-autoregressive discrete-diffusion generation (`generate(...)`),
    /// which is not expressible through the standard causal `forward` path.
    pub fn as_diffusion_gemma_mut(&mut self) -> Option<&mut DiffusionGemmaForBlockDiffusion> {
        match self {
            Self::DiffusionGemma(m) => Some(m),
            _ => None,
        }
    }

    /// Access the underlying [`MllamaForConditionalGeneration`] for
    /// image-conditioned decoding, which needs `pixel_values` /
    /// `aspect_ratio_ids` / `aspect_ratio_mask` that the uniform `forward`
    /// signature cannot carry.
    pub fn as_mllama_mut(&mut self) -> Option<&mut MllamaForConditionalGeneration> {
        match self {
            Self::Mllama(m) => Some(m),
            _ => None,
        }
    }

    pub fn vocab_size(&self) -> i32 {
        match self {
            Self::Llama(m) => m.model.config.vocab_size,
            Self::Llama4(m) => m.config.vocab_size,
            Self::Qwen2(m) => m.config().vocab_size(),
            Self::Qwen3(m) => m.config().vocab_size(),
            Self::Qwen3MoE(m) => m.config.vocab_size,
            Self::Gemma(m) => m.config().vocab_size(),
            Self::Mistral(m) => m.config().vocab_size(),
            Self::Phi(m) => m.config().vocab_size(),
            Self::Phi4(m) => m.config().vocab_size(),
            Self::DeepSeek(m) => m.config.vocab_size,
            Self::Cohere(m) => m.config.vocab_size,
            Self::Granite(m) => m.config.vocab_size,
            Self::NemotronH(m) => m.config().vocab_size(),
            Self::Qwen3Next(m) => m.config().vocab_size(),
            Self::GptOss(m) => m.config().vocab_size,
            Self::Gemma4(m) => m.config.vocab_size,
            Self::Flux(_) => 0,
            Self::Bert(m) => m.vocab_size() as i32,
            Self::DiffusionGemma(m) => m.vocab_size,
            Self::Mllama(m) => m.config.text_config.llama.vocab_size,
            Self::Qwen4Exp(m) => m.config.vocab_size,
        }
    }

    pub fn hidden_size(&self) -> i32 {
        match self {
            Self::Llama(m) => m.model.config.hidden_size,
            Self::Llama4(m) => m.config.hidden_size,
            Self::Qwen2(m) => m.config().hidden_size(),
            Self::Qwen3(m) => m.config().hidden_size(),
            Self::Qwen3MoE(m) => m.config.hidden_size,
            Self::Gemma(m) => m.config().hidden_size(),
            Self::Mistral(m) => m.config().hidden_size(),
            Self::Phi(m) => m.config().hidden_size(),
            Self::Phi4(m) => m.config().hidden_size(),
            Self::DeepSeek(m) => m.config.hidden_size,
            Self::Cohere(m) => m.config.hidden_size,
            Self::Granite(m) => m.config.hidden_size,
            Self::NemotronH(m) => m.config().hidden_size(),
            Self::Qwen3Next(m) => m.config().hidden_size(),
            Self::GptOss(m) => m.config().hidden_size,
            Self::Gemma4(m) => m.config.hidden_size,
            Self::Flux(m) => m.config().hidden_size() as i32,
            Self::Bert(m) => m.hidden_size() as i32,
            Self::DiffusionGemma(m) => m.encoder.config.hidden_size,
            Self::Mllama(m) => m.config.text_config.llama.hidden_size,
            Self::Qwen4Exp(m) => m.config.hidden_size,
        }
    }

    pub fn eval(&self) -> Result<(), Exception> {
        match self {
            Self::Llama(m) => eval_module_parameters_batched(m),
            Self::Llama4(m) => eval_module_parameters_batched(m),
            Self::Qwen2(m) => eval_module_parameters_batched(m),
            Self::Qwen3(m) => eval_module_parameters_batched(m),
            Self::Qwen3MoE(m) => eval_module_parameters_batched(m),
            Self::Gemma(m) => eval_module_parameters_batched(m),
            Self::Mistral(m) => eval_module_parameters_batched(m),
            Self::Phi(m) => eval_module_parameters_batched(m),
            Self::Phi4(m) => eval_module_parameters_batched(m),
            Self::DeepSeek(m) => eval_module_parameters_batched(m),
            Self::Cohere(m) => eval_module_parameters_batched(m),
            Self::Granite(m) => eval_module_parameters_batched(m),
            Self::NemotronH(m) => eval_module_parameters_batched(m),
            Self::Qwen3Next(m) => eval_module_parameters_batched(m),
            Self::GptOss(m) => eval_module_parameters_batched(m),
            Self::Gemma4(m) => eval_module_parameters_batched(m),
            Self::Flux(m) => eval_module_parameters_batched(m),
            Self::Bert(m) => eval_module_parameters_batched(m),
            Self::DiffusionGemma(m) => eval_module_parameters_batched(m),
            Self::Mllama(m) => eval_module_parameters_batched(m),
            Self::Qwen4Exp(m) => eval_module_parameters_batched(m),
        }
    }

    /// Enable SSD-offloaded MoE inference: each sparse layer reads the experts
    /// its tokens route to from `experts_dir`.
    ///
    /// Only supported for architectures built on the Qwen 3.5 MoE block
    /// (Qwen3Next, Qwen4Exp).
    /// The `experts_dir` should contain packed expert files from `pmetal pack-experts`.
    pub fn enable_expert_offloading(&mut self, experts_dir: &Path) -> Result<(), Exception> {
        match self {
            Self::Qwen3Next(m) => m.enable_expert_offloading(experts_dir),
            Self::Qwen4Exp(m) => m.enable_expert_offloading(experts_dir),
            _ => Err(Exception::custom(
                "expert offloading is only supported for the qwen3_next and qwen4_exp architectures",
            )),
        }
    }

    /// What the offloaded experts' files have been asked for since the last
    /// [`reset_expert_read_stats`](Self::reset_expert_read_stats), if expert
    /// offloading is enabled.
    pub fn expert_read_stats(&self) -> Option<crate::expert_offload::ExpertReadStats> {
        match self {
            Self::Qwen3Next(m) => m.offloaded_experts().map(|e| e.stats()),
            Self::Qwen4Exp(m) => m.offloaded_experts().map(|e| e.stats()),
            _ => None,
        }
    }

    /// Zero the expert read counters (if expert offloading is enabled).
    pub fn reset_expert_read_stats(&self) {
        let experts = match self {
            Self::Qwen3Next(m) => m.offloaded_experts(),
            Self::Qwen4Exp(m) => m.offloaded_experts(),
            _ => None,
        };
        if let Some(experts) = experts {
            experts.reset_stats();
        }
    }

    pub fn requires_expert_offloading(&self) -> bool {
        match self {
            Self::Qwen3Next(m) => m.requires_expert_offloading(),
            Self::Qwen4Exp(m) => m.requires_expert_offloading(),
            _ => false,
        }
    }
}

/// Walk every projection in whichever architecture is loaded.
///
/// This is what lets `pmetal-lora` attach adapters to a model it did not build
/// and does not have the concrete type of.
impl pmetal_runtime::compat::VisitLinears for DynamicModel {
    fn visit_linears_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut pmetal_runtime::compat::Linear),
    ) {
        dispatch_uniform!(self, visit_linears_mut, prefix, f)
    }

    fn visit_experts_mut(
        &mut self,
        prefix: &str,
        top_k: usize,
        f: &mut dyn FnMut(&str, usize, &mut pmetal_runtime::nn::moe::Linear),
    ) {
        dispatch_uniform!(self, visit_experts_mut, prefix, top_k, f)
    }
}

/// Remove the router parameters from a parameter tree, in place.
fn drop_routers(tree: &mut pmetal_runtime::compat::module::ModuleParamRef<'_>, prefix: &str) {
    use pmetal_runtime::compat::{NestedValue, child_path};
    tree.retain(|name, value| {
        let path = child_path(prefix, name);
        if crate::moe_routing::is_router_path(&path) {
            return false;
        }
        if let NestedValue::Map(inner) = value {
            drop_routers(inner, &path);
        }
        true
    });
}

impl DynamicModel {
    /// Every parameter a gradient can move, a MoE router's included: what a
    /// full fine-tune trains. Buffers (a router's load-balancing bias) are
    /// absent, as they are from [`trainable_parameters`], which also leaves
    /// out the routers.
    ///
    /// [`trainable_parameters`]: ModuleParameters::trainable_parameters
    pub fn trainable_parameters_with_routers(
        &self,
    ) -> pmetal_runtime::compat::module::ModuleParamRef<'_> {
        dispatch_uniform!(self, trainable_parameters)
    }
}

impl ModuleParameters for DynamicModel {
    fn parameters(&self) -> pmetal_runtime::compat::module::ModuleParamRef<'_> {
        dispatch_uniform!(self, parameters)
    }

    /// Everything an optimizer may move, which is every parameter but a MoE
    /// router's: its argmax picks which computation runs, and a gradient step
    /// (or weight decay) on it reroutes tokens the experts were never tuned
    /// for. Buffers and parameter-free layers are absent from the tree already.
    fn trainable_parameters(&self) -> pmetal_runtime::compat::module::ModuleParamRef<'_> {
        let mut tree = self.trainable_parameters_with_routers();
        drop_routers(&mut tree, "");
        tree
    }

    fn parameters_mut(&mut self) -> pmetal_runtime::compat::module::ModuleParamMut<'_> {
        dispatch_uniform!(self, parameters_mut)
    }

    fn num_parameters(&self) -> usize {
        dispatch_uniform!(self, num_parameters)
    }

    fn freeze_parameters(&mut self, recurse: bool) {
        dispatch_uniform!(self, freeze_parameters, recurse)
    }

    fn unfreeze_parameters(&mut self, recurse: bool) {
        dispatch_uniform!(self, unfreeze_parameters, recurse)
    }

    fn all_frozen(&self) -> Option<bool> {
        dispatch_uniform!(self, all_frozen)
    }

    fn any_frozen(&self) -> Option<bool> {
        dispatch_uniform!(self, any_frozen)
    }
}

impl Module<Array> for DynamicModel {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Array) -> Result<Self::Output, Self::Error> {
        self.forward(&input, None)
    }

    fn training_mode(&mut self, _mode: bool) {
        // No-op for now as most models don't implement Module trait yet
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    /// A bundled-MTP checkpoint is its base architecture plus draft layers.
    /// `mlx-community/Qwen3.8-27B-MTP-4bit` states `qwen3_5_mtp` with no
    /// `architectures` array to fall back on, so before the suffix was
    /// stripped the whole family resolved to `None` and never loaded.
    #[test]
    fn bundled_mtp_checkpoints_resolve_to_their_target_architecture() {
        for mt in ["qwen3_5_mtp", "qwen3_6_mtp", "qwen3_5_moe_mtp"] {
            assert_eq!(
                ModelArchitecture::from_model_type(mt),
                Some(ModelArchitecture::Qwen3Next),
                "{mt} should run as its target architecture"
            );
        }
        // The plain names keep resolving the same way.
        assert_eq!(
            ModelArchitecture::from_model_type("qwen3_5"),
            Some(ModelArchitecture::Qwen3Next)
        );
        // A suffix on something we do not implement still resolves to nothing,
        // rather than to a neighbouring architecture.
        assert_eq!(ModelArchitecture::from_model_type("mamba_mtp"), None);
    }

    /// The literal `nvidia/Nemotron-H-8B-Base-8K` ships, reduced.
    ///
    /// `serde_json` rejects the bare `Infinity`, which made every Nemotron-H
    /// checkpoint undetectable — `detect` failed on the JSON parse long before
    /// the architecture's own `json5` deserialization, which accepts it.
    const NEMOTRON_STYLE_CONFIG: &str = r#"{
        "model_type": "nemotron_h",
        "time_step_limit": [0.0, Infinity],
        "hidden_size": 4096
    }"#;

    #[test]
    fn non_finite_literals_do_not_defeat_config_parsing() {
        assert!(
            serde_json::from_str::<serde_json::Value>(NEMOTRON_STYLE_CONFIG).is_err(),
            "premise: strict JSON must reject this, or the fix is testing nothing"
        );

        let value = config_value(NEMOTRON_STYLE_CONFIG).expect("tolerant parse");
        assert_eq!(value["model_type"], "nemotron_h");
        assert_eq!(value["hidden_size"], 4096);
        // `serde_json::Number` cannot hold a non-finite float at all, so the
        // element is flattened rather than preserved. Nothing reads it from
        // this view — the config struct is deserialized from the original text.
        assert_eq!(value["time_step_limit"][1], serde_json::Value::Null);
    }

    #[test]
    fn sanitizing_leaves_strings_and_well_formed_configs_alone() {
        // `Infinity` inside a string value is data, not a literal.
        let quoted = r#"{"model_type": "llama", "note": "Infinity and NaN"}"#;
        assert_eq!(sanitize_non_finite(quoted), quoted);

        // Non-ASCII must survive the byte-level scan intact.
        let unicode = r#"{"eos_token": "<|end▁of▁sentence|>", "x": NaN}"#;
        let cleaned = sanitize_non_finite(unicode);
        assert!(cleaned.contains("<|end▁of▁sentence|>"));
        assert!(cleaned.contains("\"x\": null"));

        // Negation is consumed with the literal, not left dangling as `-null`.
        let negative = r#"{"lo": -Infinity}"#;
        assert_eq!(sanitize_non_finite(negative), r#"{"lo": null}"#);
    }

    #[test]
    fn the_encoder_families_bert_computes_are_detected_and_lookalikes_are_not() {
        for model_type in [
            "bert",
            "roberta",
            "xlm-roberta",
            "camembert",
            "distilbert",
            "modernbert",
        ] {
            assert_eq!(
                ModelArchitecture::from_model_type(model_type),
                Some(ModelArchitecture::Bert),
                "{model_type}"
            );
        }
        for class in [
            "BertModel",
            "BertForMaskedLM",
            "RobertaForMaskedLM",
            "XLMRobertaModel",
            "CamembertModel",
            "DistilBertForMaskedLM",
            "ModernBertModel",
            "ModernBertForMaskedLM",
        ] {
            assert_eq!(
                ModelArchitecture::from_architectures(&[class.to_string()]),
                Some(ModelArchitecture::Bert),
                "{class}"
            );
        }
        // Different architectures whose names contain "bert".
        for model_type in ["albert", "deberta", "deberta-v2", "megatron-bert"] {
            assert_eq!(
                ModelArchitecture::from_model_type(model_type),
                None,
                "{model_type}"
            );
        }
        for class in ["AlbertModel", "DebertaV2Model", "MegatronBertModel"] {
            assert_eq!(
                ModelArchitecture::from_architectures(&[class.to_string()]),
                None,
                "{class}: the name contains \"bert\" but the architecture differs"
            );
        }
    }

    fn tiny_qwen35_moe_text_config() -> Qwen3NextConfig {
        Qwen3NextConfig {
            model_type: "qwen3_5_moe_text".to_string(),
            vocab_size: 100,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: Some(16),
            max_position_embeddings: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            tie_word_embeddings: true,
            linear_num_value_heads: 2,
            linear_num_key_heads: 1,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            num_experts: 4,
            num_experts_per_tok: 2,
            decoder_sparse_step: 1,
            moe_intermediate_size: 48,
            shared_expert_intermediate_size: 32,
            mlp_only_layers: Vec::new(),
            norm_topk_prob: true,
            partial_rotary_factor: 0.25,
            attention_bias: false,
            rope_scaling: None,
            rope_parameters: None,
            layer_types: Some(vec![
                "linear_attention".to_string(),
                "linear_attention".to_string(),
            ]),
            mtp_num_hidden_layers: None,
            num_nextn_predict_layers: None,
            ..Default::default()
        }
    }

    #[test]
    fn qwen35_moe_model_type_detects_as_qwen3_next() {
        assert_eq!(
            ModelArchitecture::from_model_type("qwen3_5_moe"),
            Some(ModelArchitecture::Qwen3Next)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("qwen3_5_moe_text"),
            Some(ModelArchitecture::Qwen3Next)
        );
    }

    #[test]
    fn qwen36_moe_model_type_detects_as_qwen3_next() {
        assert_eq!(
            ModelArchitecture::from_model_type("qwen3_6_moe"),
            Some(ModelArchitecture::Qwen3Next)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("qwen3_6_moe_text"),
            Some(ModelArchitecture::Qwen3Next)
        );
    }

    #[test]
    fn llama4_text_model_type_detects_as_llama4() {
        assert_eq!(
            ModelArchitecture::from_model_type("llama4_text"),
            Some(ModelArchitecture::Llama4)
        );
    }

    #[test]
    fn qwen35_moe_architecture_string_detects_as_qwen3_next() {
        let architectures = vec!["Qwen3_5_MoeForConditionalGeneration".to_string()];
        assert_eq!(
            ModelArchitecture::from_architectures(&architectures),
            Some(ModelArchitecture::Qwen3Next)
        );
    }

    #[test]
    fn qwen36_moe_architecture_string_detects_as_qwen3_next() {
        let architectures = vec!["Qwen3_6_MoeForConditionalGeneration".to_string()];
        assert_eq!(
            ModelArchitecture::from_architectures(&architectures),
            Some(ModelArchitecture::Qwen3Next)
        );
    }

    /// Qwen3.8-Flash-Next ships `qwen4_exp`. Its class name contains "qwen",
    /// which the generic fallback reads as Qwen 2.
    #[test]
    fn qwen4_exp_detects_from_model_type_and_architecture_string() {
        for mt in ["qwen4_exp", "qwen4_exp_text"] {
            assert_eq!(
                ModelArchitecture::from_model_type(mt),
                Some(ModelArchitecture::Qwen4Exp)
            );
        }
        let architectures = vec!["Qwen4ExpForConditionalGeneration".to_string()];
        assert_eq!(
            ModelArchitecture::from_architectures(&architectures),
            Some(ModelArchitecture::Qwen4Exp)
        );
    }

    #[test]
    fn gemma4_architecture_string_detects_as_gemma4_not_gemma3() {
        let architectures = vec!["Gemma4ForConditionalGeneration".to_string()];
        assert_eq!(
            ModelArchitecture::from_architectures(&architectures),
            Some(ModelArchitecture::Gemma4)
        );
    }

    #[test]
    fn diffusion_gemma_model_type_detects() {
        assert_eq!(
            ModelArchitecture::from_model_type("diffusion_gemma"),
            Some(ModelArchitecture::DiffusionGemma)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("diffusion_gemma_text"),
            Some(ModelArchitecture::DiffusionGemma)
        );
    }

    #[test]
    fn diffusion_gemma_architecture_string_does_not_fall_through_to_gemma() {
        // `DiffusionGemmaForBlockDiffusion` contains "gemma"; the dedicated
        // check must win over the generic Gemma fallback.
        let architectures = vec!["DiffusionGemmaForBlockDiffusion".to_string()];
        assert_eq!(
            ModelArchitecture::from_architectures(&architectures),
            Some(ModelArchitecture::DiffusionGemma)
        );
    }

    #[test]
    #[serial]
    fn qwen35_placeholder_model_requires_expert_offloading() {
        let model = DynamicModel::Qwen3Next(
            Qwen3NextForCausalLM::new_with_routed_expert_mode(
                tiny_qwen35_moe_text_config(),
                Qwen3NextRoutedExpertMode::Placeholder,
            )
            .unwrap(),
        );
        assert!(model.requires_expert_offloading());
    }
}
