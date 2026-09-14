//! Dynamic model dispatch based on config.json model_type.
//!
//! This module provides automatic architecture detection and model loading,
//! eliminating the need for hardcoded model types in application code.

use crate::architectures::*;
use crate::loader::{
    Qwen3NextLoadOptions, assign_weights, load_bert_weights, load_generic_weights,
    load_nemotron_weights, load_qwen3_next_weights_with_options, load_weights,
};
use crate::traits::{CausalLMModel, ModelConfig};
use crate::weight_format::{GgufModelConfig, WeightFormat, WeightLoader};
use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParamMut, ModuleParamRef, ModuleParameters,
    ModuleParametersExt, nn, transforms,
};
use pmetal_mlx::kv_cache::{
    CacheMode, FusedBatchKVCache, KVCache, KVCacheConfig, MambaCache,
    sanitize_cache_mode_for_config,
};
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
        pmetal_bridge::compat::transforms::eval(chunk.iter())?;
    }

    Ok(())
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
/// literals described on [`sanitize_non_finite`].
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
        let inner = serde_json::to_string(&config_json["text_config"])
            .map_err(|e| Exception::custom(e.to_string()))?;
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
        }
    }
}

impl ModelArchitecture {
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        let lower = model_type.to_lowercase();
        match lower.as_str() {
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
            "granite" | "granitehybrid" | "granite_moe" => Some(Self::Granite),
            "nemotron_h" | "nemotronh" | "nemotron-h" => Some(Self::NemotronH),
            "flux" | "flux-1" | "flux.1" => Some(Self::Flux),
            // Only real BERT. `roberta`, `xlm-roberta` and `distilbert` used to
            // map here, and none of the three can actually load:
            // `BertConfig` has no spelling for DistilBERT's `dim` / `n_layers`
            // / `hidden_dim`, and `remap_bert_weight_name` strips a `bert.`
            // prefix, not `roberta.` — so a RoBERTa checkpoint matched zero
            // parameters. RoBERTa additionally offsets its position ids past
            // the padding index, which a prefix fix alone would get silently
            // wrong. Claiming these here only bought a later, more confusing
            // failure.
            "bert" => Some(Self::Bert),
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
            if lower.contains("granite") {
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
            // Check BERT after other checks to avoid false positives. The
            // RoBERTa / DistilBERT exclusion mirrors `from_model_type`: their
            // class names contain "bert" but neither is loadable here.
            if lower.contains("bert") && !lower.contains("distilbert") && !lower.contains("roberta")
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
            Self::Granite => round_trip!(GraniteConfig, config_content),
            Self::NemotronH => round_trip!(NemotronHConfig, config_content),
            Self::Qwen3Next => round_trip!(Qwen3NextConfig, &nested),
            Self::GptOss => round_trip!(GptOssConfig, config_content),
            Self::Gemma4 => round_trip!(crate::architectures::gemma4::Gemma4Config, &nested),
            Self::Bert => round_trip!(BertConfig, config_content),
            Self::DiffusionGemma => {
                let parsed = crate::architectures::diffusion_gemma::parse_diffusion_gemma_config(
                    config_content,
                )?;
                serde_json::to_value(&parsed).map_err(|e| Exception::custom(e.to_string()))
            }
            Self::Mllama => round_trip!(MllamaConfig, config_content),
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
            .ok_or_else(|| Exception::custom(format!("Unsupported model type: {}", model_type)))
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
        .ok_or_else(|| Exception::custom(format!("Unsupported architecture: {}", model_type)))
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
        load_generic_weights(&mut model, $model_dir)
            .map_err(|e| Exception::custom(format!("{:?}", e)))?;
        eval_module_parameters_batched(&model)?;
        Ok(Self::$variant(model))
    }};
}

/// Same as `simple_load!` but additionally calls `init_post_load_fast_paths()`
/// on the wrapped `DynamicModel` before returning — required for MoE
/// architectures (Qwen3MoE, DeepSeek, NemotronH, GptOss) that materialise
/// stacked expert weights after the base load.
macro_rules! simple_load_moe {
    ($config_ty:ty, $new:expr, $content:expr, $model_dir:expr, $variant:ident) => {{
        let config: $config_ty =
            json5::from_str($content).map_err(|e| Exception::custom(e.to_string()))?;
        let mut model = ($new)(config)?;
        load_generic_weights(&mut model, $model_dir)
            .map_err(|e| Exception::custom(format!("{:?}", e)))?;
        eval_module_parameters_batched(&model)?;
        let mut model = Self::$variant(model);
        model.init_post_load_fast_paths()?;
        Ok(model)
    }};
}

/// A model whose architecture is dispatched at runtime.
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
    Bert(BertForEmbedding),
    DiffusionGemma(DiffusionGemmaForBlockDiffusion),
    Mllama(MllamaForConditionalGeneration),
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
        }
    }
}

impl DynamicModel {
    fn init_post_load_fast_paths(&mut self) -> Result<(), Exception> {
        match self {
            Self::Qwen3MoE(model) => model.init_stacked_moe(),
            Self::DeepSeek(model) => model.init_stacked_moe(),
            Self::NemotronH(model) => model.init_stacked_moe(),
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
            ModelArchitecture::Granite => {
                simple_new!(
                    GraniteConfig,
                    GraniteForCausalLM::new,
                    config_content,
                    Granite
                )
            }
            ModelArchitecture::NemotronH => {
                simple_new!(
                    NemotronHConfig,
                    NemotronHForCausalLM::new,
                    config_content,
                    NemotronH
                )
            }
            ModelArchitecture::Qwen3Next => {
                let text_config_str = unwrap_text_config(config_content)?;
                let mut config: Qwen3NextConfig = serde_json::from_str(&text_config_str)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                config.apply_rope_parameters();
                Ok(Self::Qwen3Next(Qwen3NextForCausalLM::new(config)?))
            }
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
                simple_new!(BertConfig, BertForEmbedding::new, config_content, Bert)
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
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                let mut params = model.flatten_params_mut();
                for (key, value) in weights {
                    let remapped = key
                        .strip_prefix("model.language_model.")
                        .map(|rest| format!("model.{rest}"))
                        .unwrap_or(key);
                    if let Some(param) = params.get_mut(&remapped) {
                        **param = value;
                    }
                }
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
            ModelArchitecture::Qwen3MoE => simple_load_moe!(
                Qwen3MoEConfig,
                Qwen3MoE::new,
                &config_content,
                model_dir,
                Qwen3MoE
            ),
            ModelArchitecture::Gemma => {
                let config = parse_gemma_config(&config_content)?;
                let mut model = GemmaForCausalLM::new(config)?;
                let weights = crate::loader::load_weights(model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
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
                crate::loader::load_gemma_weights(&mut model, &weights_effective)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
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
            // DeepSeek cannot use `simple_load_moe!`: its MoE layers name the
            // router, the shared expert and every routed expert differently
            // from the checkpoint, and the generic loader drops what it cannot
            // match by exact name. See `deepseek_param_name`.
            ModelArchitecture::DeepSeek => {
                let config: DeepSeekConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = DeepSeek::new(config)?;
                crate::loader::load_generic_weights_renamed(
                    &mut model,
                    model_dir,
                    crate::loader::deepseek_param_name,
                )
                .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                let mut model = Self::DeepSeek(model);
                model.init_post_load_fast_paths()?;
                Ok(model)
            }
            ModelArchitecture::Cohere => simple_load!(
                CohereConfig,
                CohereForCausalLM::new,
                &config_content,
                model_dir,
                Cohere
            ),
            ModelArchitecture::Granite => simple_load!(
                GraniteConfig,
                GraniteForCausalLM::new,
                &config_content,
                model_dir,
                Granite
            ),
            // NemotronH uses a bespoke weight loader (load_nemotron_weights) so we
            // can't route through simple_load_moe!, but the init_post_load_fast_paths
            // step still applies after weights are materialised.
            ModelArchitecture::NemotronH => {
                let config: NemotronHConfig = json5::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = NemotronHForCausalLM::new(config)?;
                crate::loader::load_nemotron_weights(&mut model, model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                let mut model = Self::NemotronH(model);
                model.init_post_load_fast_paths()?;
                Ok(model)
            }
            ModelArchitecture::Qwen3Next => {
                let text_config_str = unwrap_text_config(&config_content)?;
                let mut config: Qwen3NextConfig = serde_json::from_str(&text_config_str)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                config.apply_rope_parameters();
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
                    }
                } else {
                    Qwen3NextLoadOptions::default()
                };
                load_qwen3_next_weights_with_options(&mut model, model_dir, &config, load_options)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                eval_module_parameters_batched(&model)?;
                Ok(Self::Qwen3Next(model))
            }
            ModelArchitecture::Flux => Err(Exception::custom(
                "Flux models are diffusion pipelines, not causal language models. Load them via pmetal_models::pipelines::FluxPipeline instead of DynamicModel::load.",
            )),
            ModelArchitecture::GptOss => simple_load_moe!(
                GptOssConfig,
                GptOssForCausalLM::new,
                &config_content,
                model_dir,
                GptOss
            ),
            ModelArchitecture::Gemma4 => {
                let effective = unwrap_text_config(&config_content)?;
                let config: crate::architectures::gemma4::Gemma4Config =
                    json5::from_str(&effective).map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = crate::architectures::gemma4::Gemma4ForCausalLM::new(config)?;
                let weights = crate::loader::load_weights(model_dir)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                let report =
                    crate::architectures::gemma4::load_gemma4_weights(&mut model, &weights)
                        .map_err(|e| Exception::custom(format!("{:?}", e)))?;
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
                let config: BertConfig = serde_json::from_str(&config_content)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let mut model = BertForEmbedding::new(config)?;
                // Load weights using the HF→PMetal name remapper.  HuggingFace BERT
                // checkpoints use paths like `bert.encoder.layer.0.attention.self.query.*`
                // which differ from PMetal's `model.layers.0.attention.query.*`.
                let weights =
                    load_weights(model_dir).map_err(|e| Exception::custom(format!("{:?}", e)))?;
                load_bert_weights(&mut model, &weights)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
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
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                let report = crate::architectures::diffusion_gemma::load_diffusion_gemma_weights(
                    &mut model, &weights,
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
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                let report =
                    crate::architectures::mllama::load_mllama_weights(&mut model, &weights)?;
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
        let mut weights = crate::loader::load_weights(model_dir)
            .map_err(|e| Exception::custom(format!("{:?}", e)))?;
        crate::loader::split_phi_fused_qkv(&mut weights, &config);
        crate::loader::assign_weights(&mut model, weights)
            .map_err(|e| Exception::custom(format!("{:?}", e)))?;
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

        let weights =
            WeightLoader::load_gguf(path).map_err(|e| Exception::custom(format!("GGUF: {e}")))?;

        match arch {
            ModelArchitecture::Llama => {
                let mut model = LlamaForCausalLM::new(gguf_config.to_llama_config())?;
                assign_weights(&mut model, weights)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                Ok(Self::Llama(model))
            }
            ModelArchitecture::Qwen2 => {
                let mut model = Qwen2ForCausalLM::new(gguf_config.to_qwen2_config())?;
                assign_weights(&mut model, weights)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                Ok(Self::Qwen2(model))
            }
            ModelArchitecture::Qwen3 => {
                let mut model = Qwen3ForCausalLM::new_for_loading(gguf_config.to_qwen3_config())?;
                assign_weights(&mut model, weights)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                Ok(Self::Qwen3(model))
            }
            ModelArchitecture::Mistral => {
                let mut model = MistralForCausalLM::new(gguf_config.to_mistral_config())?;
                assign_weights(&mut model, weights)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                Ok(Self::Mistral(model))
            }
            ModelArchitecture::Phi => {
                let mut model = PhiForCausalLM::new(gguf_config.to_phi_config())?;
                assign_weights(&mut model, weights)
                    .map_err(|e| Exception::custom(format!("{:?}", e)))?;
                Ok(Self::Phi(model))
            }
            other => Err(Exception::custom(format!(
                "GGUF inference loading is not yet implemented for {other} \
                 (Phase B supports dense Llama, Qwen2, Qwen3, Mistral, Phi)"
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
            Self::NemotronH(m) => m.forward(input_ids, None),
            Self::Qwen3Next(m) => m.forward(input_ids, mask),
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
            Self::Bert(m) => BertForEmbedding::forward(m, input_ids, mask),
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
            Self::GptOss(m) => m.forward_with_positions(input_ids, mask, positions),
            // Text-only, as for `forward`.
            Self::Mllama(m) => m.forward_with_positions(input_ids, mask, positions),
            // NemotronH's attention blocks carry no positional encoding, and a
            // packed row breaks the Mamba recurrence in a way positions cannot
            // repair. Flux / BERT / DiffusionGemma are not causal LMs.
            other => Err(Exception::custom(format!(
                "forward_with_positions is not implemented for {other:?}; \
                 check supports_packed_positions() before calling"
            ))),
        }
    }

    /// Whether [`forward_with_positions`] actually applies the positions for
    /// this architecture, rather than accepting and dropping them.
    ///
    /// [`forward_with_positions`]: Self::forward_with_positions
    pub fn supports_packed_positions(&self) -> bool {
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
                | Self::Granite(_)
                | Self::GptOss(_)
                | Self::Mllama(_)
        )
    }

    /// Forward pass returning last-layer hidden states `[batch, seq, hidden]`
    /// — the pre-lm-head representation used for sentence embeddings and
    /// `/v1/embeddings`-style pooling endpoints.
    ///
    /// Coverage: every dense decoder arch whose `ForCausalLM` wraps a
    /// `pub model: *Model` field routes through that inner trunk. BERT's
    /// canonical `forward` already returns hidden states so it's a
    /// pass-through. Architectures not listed here (Flux, Qwen3MoE, hybrid
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
            Self::Mistral(m) => m.model.forward(input_ids, mask),
            Self::Gemma(m) => m.model.forward(input_ids, mask),
            // Gemma4 / Phi / Phi4 inner models expose only forward_with_cache;
            // pass None for the cache — embeddings don't need incremental decode.
            Self::Gemma4(m) => m.model.forward_with_cache(input_ids, mask, None),
            Self::Phi(m) => m.model.forward_with_cache(input_ids, mask, None),
            Self::Phi4(m) => m.model.forward_with_cache(input_ids, mask, None),
            Self::Bert(m) => BertForEmbedding::forward(m, input_ids, mask),
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
            Self::NemotronH(_) | Self::Qwen3Next(_) => Err(Exception::custom(
                "Hybrid architecture requires both KV and Mamba caches. \
                 Use DynamicModel::forward_with_hybrid_cache with both \
                 create_cache() and create_mamba_cache().",
            )),
            Self::Flux(_) => Err(Exception::custom(
                "Flux is not a CausalLM and does not support forward_with_cache.",
            )),
            // BERT is encoder-only (no autoregressive cache) — delegate to standard forward.
            Self::Bert(m) => BertForEmbedding::forward(m, input_ids, mask),
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
            Self::GptOss(m) => Some(&mut m.model.grad_checkpoint),
            // Gemma 4, Qwen3-Next, Nemotron-H and Mllama are left out on
            // purpose. Each has a trunk whose layers read or write state
            // belonging to another layer (Gemma 4's shared KV, the hybrid
            // models' recurrent caches, Mllama's cross-attention), and a
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
            // NemotronH has a bespoke implementation that operates on concrete
            // `nn::Linear` structs and handles Mamba/attention blocks explicitly.
            Self::NemotronH(model) => model.quantize_fp8_weights(),

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
            Self::Qwen3Next(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::GptOss(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Gemma4(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Bert(m) => crate::fp8_utils::quantize_model_linears(m),
            // Generic linear-weight quantization; the fused 3-D expert tensors
            // (not `.weight`) stay in their loaded dtype, same as other MoE archs.
            Self::DiffusionGemma(m) => crate::fp8_utils::quantize_model_linears(m),
            Self::Mllama(m) => crate::fp8_utils::quantize_model_linears(m),
        }
    }

    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        match self {
            Self::Llama(m) => m.create_cache(max_seq_len),
            Self::Llama4(m) => KVCache::new(KVCacheConfig::new(
                m.config.num_hidden_layers as usize,
                max_seq_len,
                m.config.num_key_value_heads as usize,
                (m.config.hidden_size / m.config.num_attention_heads) as usize,
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
            _ => self.forward_with_cache(input_ids, mask, kv_cache),
        }
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
            // Llama: standard GQA rides the shared fused block. Llama 3
            // (`"rope_type": "llama3"`) takes the serial fallback — it rescales
            // three RoPE frequency bands separately, and `BatchedGqaAttnCfg`
            // is scalar-only by design, carrying a single `rope_base`. Taking
            // the fused path would silently rotate with unscaled RoPE and
            // disagree with this model's own serial decode.
            Self::Llama(m) => !m.has_banded_rope(),
            Self::Mistral(m) => m.config().sliding_window.is_none(),
            Self::Qwen2(m) => !m.config().use_sliding_window,
            Self::Qwen3(m) => !m.config.use_sliding_window,
            // Qwen3-MoE reuses the GQA attention block (with qk-norm); the
            // token-level MoE router already flattens `[N, 1, H]` cleanly
            // via `forward_stacked`.
            Self::Qwen3MoE(_) => true,
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
            Self::Gemma(m) => !m.config().is_gemma3,
            // Phi/Phi4: partial RoPE handled by `BatchedGqaAttnCfg::with_rope_dims`.
            // SuRoPE configs (`rope_scaling = Some(...)`) take the serial fallback
            // because the fused cfg carries a single scalar `rope_base`, not a
            // per-dim freq array. Sliding-window Phi configs likewise stay on
            // serial until the per-layer overlay is wired into the arch loop.
            Self::Phi(m) | Self::Phi4(m) => {
                let c = m.config();
                c.rope_scaling.is_none() && c.sliding_window.is_none()
            }
            // Cohere: parallel decoder block via `batched_parallel_block`.
            // Sliding-window-with-non-global-layers configs need a per-layer
            // sliding overlay; defer those to a follow-up.
            Self::Cohere(m) => !m.config.use_sliding_window,
            // Granite: pure-attention configs route through
            // `batched_prenorm_layer`. Hybrid (Mamba2 + Attention) configs
            // stay on serial until the simplified Mamba2 stub is replaced
            // with a real stateful implementation.
            //
            // So do configs with a non-unit `residual_multiplier` — which is
            // every released Granite. `batched_prenorm_layer` adds both
            // residual branches unscaled and has no channel for the scale, so
            // taking the fused path would silently drop it. Lifting this means
            // threading the multiplier into that helper.
            Self::Granite(m) => !m.config.is_hybrid && m.config.residual_multiplier == 1.0,
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
            Self::Bert(m) => m.config().vocab_size as i32,
            Self::DiffusionGemma(m) => m.vocab_size,
            Self::Mllama(m) => m.config.text_config.llama.vocab_size,
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
            Self::Flux(m) => m.pos_embedder.dim as i32,
            Self::Bert(m) => m.config().hidden_size as i32,
            Self::DiffusionGemma(m) => m.encoder.config.hidden_size,
            Self::Mllama(m) => m.config.text_config.llama.hidden_size,
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
        }
    }

    /// Enable SSD-offloaded MoE inference with expert prefetching.
    ///
    /// Only supported for architectures with MoE (currently Qwen3Next).
    /// The `experts_dir` should contain packed expert files from `pmetal pack-experts`.
    pub fn enable_expert_offloading(&mut self, experts_dir: &Path) -> Result<(), Exception> {
        match self {
            Self::Qwen3Next(m) => m.enable_expert_offloading(experts_dir),
            _ => Err(Exception::custom(
                "expert offloading is only supported for qwen3_next architecture",
            )),
        }
    }

    pub fn requires_expert_offloading(&self) -> bool {
        match self {
            Self::Qwen3Next(m) => m.requires_expert_offloading(),
            _ => false,
        }
    }

    /// Get prefetch hit/miss statistics (if expert offloading is enabled).
    pub fn prefetch_stats(&self) -> Option<crate::expert_prefetch::PrefetchStats> {
        match self {
            Self::Qwen3Next(m) => m.prefetch_stats(),
            _ => None,
        }
    }

    /// Reset prefetch hit/miss statistics (if expert offloading is enabled).
    pub fn reset_prefetch_stats(&self) {
        if let Self::Qwen3Next(m) = self {
            m.reset_prefetch_stats();
        }
    }
}

/// Walk every projection in whichever architecture is loaded.
///
/// This is what lets `pmetal-lora` attach adapters to a model it did not build
/// and does not have the concrete type of.
impl pmetal_bridge::compat::VisitLinears for DynamicModel {
    fn visit_linears_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut pmetal_bridge::compat::Linear),
    ) {
        dispatch_uniform!(self, visit_linears_mut, prefix, f)
    }
}

impl ModuleParameters for DynamicModel {
    fn parameters(&self) -> pmetal_bridge::compat::module::ModuleParamRef<'_> {
        dispatch_uniform!(self, parameters)
    }

    fn trainable_parameters(&self) -> pmetal_bridge::compat::module::ModuleParamRef<'_> {
        dispatch_uniform!(self, trainable_parameters)
    }

    fn parameters_mut(&mut self) -> pmetal_bridge::compat::module::ModuleParamMut<'_> {
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
    fn unsupported_encoder_families_are_rejected_rather_than_routed_to_bert() {
        assert_eq!(
            ModelArchitecture::from_model_type("bert"),
            Some(ModelArchitecture::Bert)
        );
        // Neither can load: DistilBERT's config names `dim` / `n_layers`, and
        // RoBERTa's weights are prefixed `roberta.`, which nothing strips.
        // Rejecting here beats matching zero parameters later.
        assert_eq!(ModelArchitecture::from_model_type("distilbert"), None);
        assert_eq!(ModelArchitecture::from_model_type("roberta"), None);
        assert_eq!(ModelArchitecture::from_model_type("xlm-roberta"), None);
        assert_eq!(
            ModelArchitecture::from_architectures(&["DistilBertForMaskedLM".to_string()]),
            None,
            "the class name contains \"bert\" — the substring check must not claim it"
        );
        assert_eq!(
            ModelArchitecture::from_architectures(&["RobertaForMaskedLM".to_string()]),
            None
        );
        assert_eq!(
            ModelArchitecture::from_architectures(&["BertModel".to_string()]),
            Some(ModelArchitecture::Bert)
        );
    }

    fn tiny_qwen3_moe_config() -> Qwen3MoEConfig {
        Qwen3MoEConfig {
            hidden_size: 32,
            intermediate_size: 64,
            moe_intermediate_size: Some(32),
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: 16,
            vocab_size: 100,
            num_experts: 4,
            num_experts_per_tok: 2,
            decoder_sparse_step: 1,
            tie_word_embeddings: true,
            ..Default::default()
        }
    }

    fn tiny_deepseek_config() -> DeepSeekConfig {
        DeepSeekConfig {
            hidden_size: 16,
            intermediate_size: 32,
            moe_intermediate_size: 24,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(4),
            n_shared_experts: Some(1),
            n_routed_experts: Some(4),
            num_experts_per_tok: 2,
            moe_layer_freq: 1,
            first_k_dense_replace: 0,
            ..DeepSeekConfig::default()
        }
    }

    fn tiny_nemotron_h_config() -> NemotronHConfig {
        NemotronHConfig {
            model_type: "nemotron_h".to_string(),
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 4,
            max_position_embeddings: 512,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            attention_bias: false,
            head_dim: Some(32),
            mamba_num_heads: 4,
            mamba_head_dim: 32,
            mamba_proj_bias: false,
            ssm_state_size: 16,
            conv_kernel: 4,
            n_groups: 2,
            time_step_limit: (0.0, f32::INFINITY),
            time_step_min: None,
            time_step_max: None,
            mlp_bias: false,
            mlp_hidden_act: "relu2".to_string(),
            layer_norm_epsilon: 1e-5,
            use_bias: false,
            use_conv_bias: true,
            tie_word_embeddings: true,
            hybrid_override_pattern: Some("M*-E".to_string()),
            moe_intermediate_size: Some(64),
            moe_shared_expert_intermediate_size: None,
            n_group: None,
            n_routed_experts: Some(2),
            n_shared_experts: None,
            topk_group: None,
            num_experts_per_tok: Some(1),
            norm_topk_prob: None,
            routed_scaling_factor: None,
            rope_theta: 10000.0,
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

    #[test]
    #[serial]
    fn qwen3_moe_post_load_fast_paths_initialize_stacked_experts() {
        let mut model = DynamicModel::Qwen3MoE(Qwen3MoE::new(tiny_qwen3_moe_config()).unwrap());
        model.init_post_load_fast_paths().unwrap();

        let DynamicModel::Qwen3MoE(model) = &model else {
            panic!("expected qwen3-moe model");
        };
        let Qwen3MoEFeedForward::MoE(moe) = &model.model.layers[0].ffn else {
            panic!("expected moe layer");
        };
        assert!(moe.has_stacked_moe());
    }

    #[test]
    #[serial]
    fn deepseek_post_load_fast_paths_initialize_stacked_experts() {
        let mut model = DynamicModel::DeepSeek(DeepSeek::new(tiny_deepseek_config()).unwrap());
        model.init_post_load_fast_paths().unwrap();

        let DynamicModel::DeepSeek(model) = &model else {
            panic!("expected deepseek model");
        };
        let DeepSeekMLPType::MoE(moe) = &model.model.layers[0].mlp else {
            panic!("expected moe layer");
        };
        assert!(moe.has_stacked_moe());
    }

    #[test]
    #[serial]
    fn nemotron_h_post_load_fast_paths_initialize_stacked_experts() {
        let mut model =
            DynamicModel::NemotronH(NemotronHForCausalLM::new(tiny_nemotron_h_config()).unwrap());
        model.init_post_load_fast_paths().unwrap();

        let DynamicModel::NemotronH(model) = &model else {
            panic!("expected nemotron-h model");
        };
        let moe_layer = &model.backbone.layers[3].mixer;
        assert!(moe_layer.stacked_moe_up.is_some());
        assert!(moe_layer.stacked_moe_down.is_some());
    }
}
