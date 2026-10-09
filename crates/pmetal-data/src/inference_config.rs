//! Shared inference configuration utilities.
//!
//! Provides canonical implementations for:
//! - Stop token collection from all available sources
//! - Sampling default loading from `generation_config.json`
//! - Per-model-family sampling presets (from model card best practices)
//!
//! These are used by CLI, GUI, Python bindings, and examples to ensure
//! consistent inference behavior across all consumers.
//!
//! ## Sampling parameter resolution order
//!
//! 1. **CLI/GUI/request explicit value** — always wins
//! 2. **Model-card preset** — the maker's settings for the model's family
//!    ([`ModelFamily`]) and the mode (`--mode`, or thinking / instruct by
//!    whether the model thinks)
//! 3. **`generation_config.json`** — model's declared defaults, greedy
//!    unless it sets `do_sample: true`, as transformers reads it
//! 4. **transformers' defaults** for a field it leaves out —
//!    `SamplingDefaults::default()` (temperature 1.0 when sampling, top_k 50,
//!    top_p 1.0)

use std::path::Path;

use crate::Tokenizer;
use crate::chat_templates::ChatTemplateType;

/// Collect all stop tokens from every available source.
///
/// Merges tokens from:
/// 1. `generation_config.json` — the model's declared `eos_token_id` (single or array)
/// 2. Chat template EOS — the template-specific end token (e.g. `<|im_end|>` for ChatML)
/// 3. Tokenizer's `eos_token_id` — resolved from special_tokens_map / heuristics
/// 4. Well-known special tokens — if they exist in the vocabulary as single tokens,
///    they're likely EOS candidates (e.g. `<|im_end|>`, `<|eot_id|>`, `<|endoftext|>`)
///
/// Returns a deduplicated list. This ensures fine-tuned models stop correctly
/// regardless of whether they produce the base model's EOS or the chat EOS.
pub fn collect_all_stop_tokens(
    model_path: &Path,
    tokenizer: &Tokenizer,
    template_type: Option<ChatTemplateType>,
) -> Vec<u32> {
    let mut tokens = Vec::new();

    // 1. generation_config.json
    let config_path = model_path.join("generation_config.json");
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(eos) = config.get("eos_token_id") {
                    if let Some(arr) = eos.as_array() {
                        for v in arr {
                            if let Some(id) = v.as_u64() {
                                tokens.push(id as u32);
                            }
                        }
                    } else if let Some(id) = eos.as_u64() {
                        tokens.push(id as u32);
                    }
                }
            }
        }
    }

    // 2. Chat template EOS (if template type is known)
    if let Some(tt) = template_type {
        let eos_str = tt.eos_token();
        if let Ok(encoded) = tokenizer.encode(eos_str) {
            if encoded.len() == 1 {
                tokens.push(encoded[0]);
            }
        }
    }

    // 3. Tokenizer's resolved eos_token_id
    if let Some(eos) = tokenizer.eos_token_id() {
        tokens.push(eos);
    }

    // 4. Well-known special tokens — probe the vocabulary for common EOS tokens.
    //    Only add tokens that encode to exactly 1 token (i.e. they're real special tokens,
    //    not subword sequences).
    let candidates = [
        "<|im_end|>",
        "<|eot_id|>",
        "<|eot|>",
        "<|endoftext|>",
        "<|end_of_text|>",
        "<end_of_turn>",
        "<turn|>", // Gemma 4 — distinct from Gemma 2/3's <end_of_turn>.
        "<|end|>",
        "<|return|>",
        "<|END_OF_TURN_TOKEN|>",
        "<｜end▁of▁sentence｜>",
        "</s>",
    ];
    // gpt-oss's harmony format ends each message with `<|end|>`, the
    // reasoning one included, and the turn with `<|return|>`: stopping at
    // `<|end|>` there ends the reply before its answer.
    let harmony = tokenizer.inner().token_to_id("<|channel|>").is_some()
        && tokenizer.inner().token_to_id("<|return|>").is_some();
    for candidate in &candidates {
        if harmony && *candidate == "<|end|>" {
            continue;
        }
        if let Ok(encoded) = tokenizer.encode(candidate) {
            if encoded.len() == 1 {
                tokens.push(encoded[0]);
            }
        }
    }

    // Deduplicate
    tokens.sort_unstable();
    tokens.dedup();

    // Final fallback
    if tokens.is_empty() {
        tokens.push(2);
    }

    tracing::debug!("Collected stop tokens: {:?}", tokens);
    tokens
}

/// transformers' `GenerationConfig` default `top_k`.
const TRANSFORMERS_TOP_K: usize = 50;

/// transformers' `GenerationConfig` default `temperature`, which a model that
/// samples (`do_sample: true`) without naming one generates at.
const TRANSFORMERS_TEMPERATURE: f32 = 1.0;

/// Sampling hyperparameter defaults loaded from model config.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingDefaults {
    /// Sampling temperature (0 = greedy).
    pub temperature: f32,
    /// Top-k sampling (0 = disabled).
    pub top_k: usize,
    /// Top-p nucleus sampling.
    pub top_p: f32,
    /// Min-p dynamic sampling (0 = disabled).
    pub min_p: f32,
    /// Repetition penalty (1.0 = disabled).
    pub repetition_penalty: f32,
    /// Frequency penalty (0.0 = disabled).
    pub frequency_penalty: f32,
    /// Presence penalty (0.0 = disabled).
    pub presence_penalty: f32,
}

/// What transformers' `GenerationConfig` generates with when neither the
/// model nor the caller sets a field (`_get_default_generation_params`):
/// greedy, since `do_sample` is false, with temperature 1.0, top_k 50 and
/// top_p 1.0 for a caller that samples, and no min_p or penalties.
impl Default for SamplingDefaults {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: TRANSFORMERS_TOP_K,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

// =============================================================================
// Per-model-family sampling presets
// =============================================================================

/// Named sampling mode for model-family-specific presets.
///
/// Each model family may define a set of recommended modes with tuned sampling
/// parameters. These are sourced from model card READMEs (not generation_config.json,
/// which often lacks mode-specific values like presence_penalty).
///
/// Use `available_modes()` to list modes for a detected template, and
/// `model_preset()` to resolve a mode to concrete parameters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SamplingMode {
    /// Auto-select based on thinking flag: thinking → ThinkingGeneral, else InstructGeneral.
    #[default]
    Auto,
    /// Thinking mode for general tasks.
    ThinkingGeneral,
    /// Thinking mode for precise coding tasks (e.g., WebDev).
    ThinkingCoding,
    /// Non-thinking (instruct) mode for general tasks.
    InstructGeneral,
    /// Non-thinking mode for reasoning-heavy tasks.
    InstructReasoning,
}

impl std::fmt::Display for SamplingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl SamplingMode {
    /// Return the string representation of this mode.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::ThinkingGeneral => "thinking-general",
            Self::ThinkingCoding => "thinking-coding",
            Self::InstructGeneral => "instruct-general",
            Self::InstructReasoning => "instruct-reasoning",
        }
    }
}

impl std::str::FromStr for SamplingMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "thinking-general" | "general-thinking" | "thinking" => Ok(Self::ThinkingGeneral),
            "thinking-coding" | "coding" => Ok(Self::ThinkingCoding),
            "instruct-general" | "general-instruct" | "instruct" => Ok(Self::InstructGeneral),
            "instruct-reasoning" | "reasoning" => Ok(Self::InstructReasoning),
            _ => Err(format!(
                "unknown mode '{s}': expected auto, thinking-general, thinking-coding, \
                 instruct-general, or instruct-reasoning"
            )),
        }
    }
}

// =============================================================================
// Inference backend selection
// =============================================================================

/// Execution backend used to run a forward/decode step.
///
/// `Auto` asks the runner to pick the fastest backend that still produces
/// correct output for the current device + model. Explicit variants let the
/// user pin a path (for benchmarking, debugging, or when the heuristic is
/// wrong). See `commands/infer.rs::resolve_backend` for the selection rules.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum InferenceBackend {
    /// Heuristic auto-select: best backend for current device + model.
    #[default]
    Auto,
    /// Default streaming MLX path (the current no-flag behavior).
    Standard,
    /// JIT-compiled sampling (mlx.compile).
    Compiled,
    /// Fused Metal sampling kernel.
    MetalSampler,
    /// Apple Neural Engine, same as `--ane` (requires the `ane` feature).
    Ane,
    /// Minimal async generation (debug path).
    Minimal,
}

impl InferenceBackend {
    /// Stable string id used on CLI (`--backend`) and in the TUI dropdown.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Standard => "standard",
            Self::Compiled => "compiled",
            Self::MetalSampler => "metal-sampler",
            Self::Ane => "ane",
            Self::Minimal => "minimal",
        }
    }

    /// Short human-facing label for UI surfaces.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Standard => "Standard",
            Self::Compiled => "Compiled",
            Self::MetalSampler => "Metal",
            Self::Ane => "ANE",
            Self::Minimal => "Minimal",
        }
    }

    /// One-line description shown next to the dropdown in the TUI.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Auto => "pick best for this device + model",
            Self::Standard => "streaming MLX path (token-by-token)",
            Self::Compiled => "JIT-compiled sampling (mlx.compile)",
            Self::MetalSampler => "fused Metal sampling kernel",
            Self::Ane => "Apple Neural Engine (experimental)",
            Self::Minimal => "minimal async loop (debug only)",
        }
    }

    /// All variants in UI cycle order. Used by the TUI dropdown and the
    /// `--backend` clap value-parser.
    pub const ALL: &'static [InferenceBackend] = &[
        Self::Auto,
        Self::Standard,
        Self::Compiled,
        Self::MetalSampler,
        Self::Ane,
        Self::Minimal,
    ];
}

impl std::fmt::Display for InferenceBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for InferenceBackend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "standard" | "default" => Ok(Self::Standard),
            "compiled" | "jit" => Ok(Self::Compiled),
            "metal-sampler" | "metal_sampler" | "metal" => Ok(Self::MetalSampler),
            "ane" => Ok(Self::Ane),
            "minimal" => Ok(Self::Minimal),
            _ => Err(format!(
                "unknown backend '{s}': expected auto, standard, compiled, \
                 metal-sampler, ane or minimal"
            )),
        }
    }
}

// =============================================================================
// Model families with maker-recommended settings
// =============================================================================

/// A model generation whose maker publishes recommended inference settings
/// (sampling per mode, output length). Detected from the checkpoint's
/// `config.json` `model_type` and its name (the Hugging Face repo name in the
/// cache path, or `_name_or_path`), since several generations share one
/// `model_type`: Qwen3.5, 3.6 and 3.8 are all `qwen3_5`.
///
/// Families whose makers publish nothing (Llama, Gemma 2/3, Phi-3/4,
/// Mistral 7B, Mixtral, Cohere, Granite, SmolLM2, Qwen2.5) are absent: their
/// `generation_config.json` and transformers' defaults apply
/// ([`load_sampling_from_generation_config`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// Qwen3 with switchable thinking (Qwen3-0.6B to 235B-A22B, April 2025).
    Qwen3,
    /// Qwen3-2507 Instruct and Qwen3-Next Instruct: non-thinking only.
    Qwen3Instruct2507,
    /// Qwen3-2507 Thinking and Qwen3-Next Thinking: thinking only.
    Qwen3Thinking2507,
    /// Qwen3.5.
    Qwen3_5,
    /// Qwen3.6.
    Qwen3_6,
    /// Qwen3.8 and Qwen3.8-Flash-Next, and a Qwen3.5-family checkpoint the
    /// name does not place (the latest card's settings).
    Qwen3_8,
    /// Gemma 4.
    Gemma4,
    /// gpt-oss.
    GptOss,
    /// Mistral Small 3.x.
    MistralSmall3,
    /// Magistral.
    Magistral,
    /// Phi-4-reasoning and Phi-4-reasoning-plus.
    Phi4Reasoning,
    /// DeepSeek-R1 and its distillations.
    DeepSeekR1,
    /// DeepSeek-V3-0324.
    DeepSeekV3_0324,
    /// NVIDIA Nemotron Nano 2 (9B / 12B v2).
    NemotronNanoV2,
}

impl ModelFamily {
    /// Detect the family of the checkpoint in `model_path`, or `None` when
    /// its maker publishes no settings (or it is not recognised).
    pub fn detect(model_path: &Path) -> Option<Self> {
        let config = read_json(&model_path.join("config.json"));
        let model_type = config
            .as_ref()
            .and_then(|c| {
                c.get("text_config")
                    .and_then(|t| t.get("model_type"))
                    .or_else(|| c.get("model_type"))
            })
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let name_or_path = config
            .as_ref()
            .and_then(|c| c.get("_name_or_path"))
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let name = format!("{} {}", model_name(model_path), name_or_path).to_ascii_lowercase();
        Self::from_model_type_and_name(&model_type, &name)
    }

    /// [`detect`](Self::detect) on an already-read `model_type` and name
    /// (both lower case).
    pub fn from_model_type_and_name(model_type: &str, name: &str) -> Option<Self> {
        let has = |needle: &str| name.contains(needle);
        let model_type = model_type.strip_suffix("_mtp").unwrap_or(model_type);
        if model_type.starts_with("qwen3_5")
            || model_type.starts_with("qwen3_6")
            || model_type.starts_with("qwen4_exp")
            || model_type == "qwen3_next"
        {
            return Some(if has("qwen3-next") || model_type == "qwen3_next" {
                if has("thinking") {
                    Self::Qwen3Thinking2507
                } else {
                    Self::Qwen3Instruct2507
                }
            } else if has("qwen3.5") || has("qwen3_5") {
                Self::Qwen3_5
            } else if has("qwen3.6") || has("qwen3_6") || model_type.starts_with("qwen3_6") {
                Self::Qwen3_6
            } else {
                Self::Qwen3_8
            });
        }
        if model_type == "qwen3" || model_type == "qwen3_moe" {
            return Some(if has("thinking") {
                Self::Qwen3Thinking2507
            } else if has("2507") || has("instruct") {
                Self::Qwen3Instruct2507
            } else {
                Self::Qwen3
            });
        }
        if model_type.starts_with("gemma4") {
            return Some(Self::Gemma4);
        }
        if matches!(model_type, "gpt_oss" | "gptoss" | "gpt-oss") {
            return Some(Self::GptOss);
        }
        if has("magistral") {
            return Some(Self::Magistral);
        }
        if has("mistral-small") {
            return Some(Self::MistralSmall3);
        }
        if has("phi-4-reasoning") {
            return Some(Self::Phi4Reasoning);
        }
        if has("deepseek-r1") {
            return Some(Self::DeepSeekR1);
        }
        if has("deepseek-v3-0324") {
            return Some(Self::DeepSeekV3_0324);
        }
        if model_type.starts_with("nemotron") && (has("nano-9b-v2") || has("nano-12b-v2")) {
            return Some(Self::NemotronNanoV2);
        }
        None
    }

    /// The modes the maker gives settings for.
    pub fn modes(self) -> &'static [SamplingMode] {
        use SamplingMode::*;
        match self {
            Self::Qwen3_5 => &[
                ThinkingGeneral,
                ThinkingCoding,
                InstructGeneral,
                InstructReasoning,
            ],
            Self::Qwen3_6 => &[ThinkingGeneral, ThinkingCoding, InstructGeneral],
            Self::Qwen3 | Self::Qwen3_8 | Self::Gemma4 | Self::NemotronNanoV2 => {
                &[ThinkingGeneral, InstructGeneral]
            }
            Self::Qwen3Instruct2507 | Self::MistralSmall3 | Self::DeepSeekV3_0324 => {
                &[InstructGeneral]
            }
            Self::Qwen3Thinking2507
            | Self::GptOss
            | Self::Magistral
            | Self::Phi4Reasoning
            | Self::DeepSeekR1 => &[ThinkingGeneral],
        }
    }

    /// The maker's sampling settings for `mode`. A mode the maker gives none
    /// for falls back to its general thinking or instruct mode, then to
    /// whichever mode the family has.
    pub fn preset(self, mode: SamplingMode) -> Option<SamplingDefaults> {
        use SamplingMode::*;
        let modes = self.modes();
        let mode = match mode {
            Auto => return None,
            m if modes.contains(&m) => m,
            ThinkingCoding if modes.contains(&ThinkingGeneral) => ThinkingGeneral,
            InstructReasoning if modes.contains(&InstructGeneral) => InstructGeneral,
            _ => modes[0],
        };
        // (temperature, top_p, top_k, presence_penalty); top_k 0 = off.
        let (temperature, top_p, top_k, presence_penalty) = match (self, mode) {
            // Qwen3 card: thinking 0.6/0.95/20, non-thinking 0.7/0.8/20.
            (Self::Qwen3 | Self::Qwen3Thinking2507, _) if mode == ThinkingGeneral => {
                (0.6, 0.95, 20, 0.0)
            }
            (Self::Qwen3 | Self::Qwen3Instruct2507, _) => (0.7, 0.8, 20, 0.0),
            (Self::Qwen3Thinking2507, _) => (0.6, 0.95, 20, 0.0),
            // Qwen3.5 card. Thinking-general's presence penalty is 1.5 on the
            // card; pmetal keeps 0.0, which ended thinking chains early less
            // often and which the Qwen3.6 and 3.8 cards adopted.
            (Self::Qwen3_5, ThinkingGeneral) => (1.0, 0.95, 20, 0.0),
            (Self::Qwen3_5, ThinkingCoding) => (0.6, 0.95, 20, 0.0),
            (Self::Qwen3_5, InstructReasoning) => (1.0, 1.0, 40, 2.0),
            (Self::Qwen3_5, _) => (0.7, 0.8, 20, 1.5),
            // Qwen3.6 card.
            (Self::Qwen3_6, ThinkingGeneral) => (1.0, 0.95, 20, 0.0),
            (Self::Qwen3_6, ThinkingCoding) => (0.6, 0.95, 20, 0.0),
            (Self::Qwen3_6, _) => (0.7, 0.8, 20, 1.5),
            // Qwen3.8 card.
            (Self::Qwen3_8, ThinkingGeneral) => (1.0, 0.95, 20, 0.0),
            (Self::Qwen3_8, _) => (0.7, 0.8, 20, 1.5),
            // Gemma 4 card: one configuration for all use cases.
            (Self::Gemma4, _) => (1.0, 0.95, 64, 0.0),
            // gpt-oss README: temperature 1.0, top_p 1.0.
            (Self::GptOss, _) => (1.0, 1.0, 0, 0.0),
            // Mistral Small 3.x card: a low temperature, such as 0.15.
            (Self::MistralSmall3, _) => (0.15, 1.0, 0, 0.0),
            // Magistral card: top_p 0.95, temperature 0.7.
            (Self::Magistral, _) => (0.7, 0.95, 0, 0.0),
            // Phi-4-reasoning card: temperature 0.8, top_k 50, top_p 0.95.
            (Self::Phi4Reasoning, _) => (0.8, 0.95, 50, 0.0),
            // DeepSeek-R1 card: temperature 0.6 (0.5-0.7), top_p 0.95.
            (Self::DeepSeekR1, _) => (0.6, 0.95, 0, 0.0),
            // DeepSeek-V3-0324 card: temperature 0.3.
            (Self::DeepSeekV3_0324, _) => (0.3, 1.0, 0, 0.0),
            // Nemotron Nano 2 card: 0.6 / 0.95 with reasoning, greedy without.
            (Self::NemotronNanoV2, ThinkingGeneral) => (0.6, 0.95, 0, 0.0),
            (Self::NemotronNanoV2, _) => (0.0, 1.0, 0, 0.0),
        };
        Some(SamplingDefaults {
            temperature,
            top_p,
            top_k,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty,
        })
    }

    /// The output length the maker recommends, when it names one: 32,768
    /// "for most queries" on the Qwen3 to 3.6 cards (Qwen3.8's card gives
    /// only its long agentic budget), 16,384 on the Qwen3-2507 Instruct
    /// cards, 40,960 on Magistral's and 32,768 on Phi-4-reasoning's.
    pub fn recommended_max_tokens(self) -> Option<usize> {
        match self {
            Self::Qwen3
            | Self::Qwen3Thinking2507
            | Self::Qwen3_5
            | Self::Qwen3_6
            | Self::Qwen3_8
            | Self::Phi4Reasoning => Some(THINKING_MAX_TOKENS),
            Self::Qwen3Instruct2507 => Some(16_384),
            Self::Magistral => Some(40_960),
            _ => None,
        }
    }
}

/// The checkpoint's name from its path: the repo name of a Hugging Face
/// cache entry (`models--Qwen--Qwen3.8-27B/snapshots/<hash>`), else the
/// directory's own name.
fn model_name(model_path: &Path) -> String {
    for component in model_path.components().rev() {
        let part = component.as_os_str().to_string_lossy();
        if let Some(repo) = part.strip_prefix("models--") {
            return repo.replacen("--", "/", 1);
        }
    }
    model_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The sampling modes a model's maker gives settings for; empty when none.
pub fn available_modes(family: Option<ModelFamily>) -> &'static [SamplingMode] {
    family.map_or(&[], ModelFamily::modes)
}

/// Resolve a sampling mode to the maker's settings for a model family, or
/// `None` without a family (use `generation_config.json` and transformers'
/// defaults instead). Sources, per family, are on [`ModelFamily::preset`].
pub fn model_preset(family: Option<ModelFamily>, mode: SamplingMode) -> Option<SamplingDefaults> {
    family?.preset(mode)
}

/// Resolve `SamplingMode::Auto` to a concrete mode based on thinking flag.
pub fn resolve_auto_mode(mode: SamplingMode, thinking: bool) -> SamplingMode {
    if mode != SamplingMode::Auto {
        return mode;
    }
    if thinking {
        SamplingMode::ThinkingGeneral
    } else {
        SamplingMode::InstructGeneral
    }
}

/// Read `generation_config.json` into a [`SamplingDefaults`] the way
/// transformers' `generate` reads it: each field the file sets, and
/// transformers' default ([`SamplingDefaults::default`]) for each it leaves
/// out. It samples only with `do_sample: true`, at its `temperature` (1.0
/// when unset); otherwise it is greedy (temperature 0), whatever temperature
/// the file names, and its top_k and top_p apply to a caller that asks for
/// sampling. No file at all is greedy too. This is the raw stage-2 of the
/// loading pipeline (step 1+2 in [`load_sampling_defaults`]) and is exposed
/// so the chat-template audit can verify that HF's declared values are
/// actually picked up, independent of any mode preset that might override
/// them later.
pub fn load_sampling_from_generation_config(model_path: &Path) -> SamplingDefaults {
    let mut defaults = SamplingDefaults::default();
    let Some(config) = read_json(&model_path.join("generation_config.json")) else {
        return defaults;
    };
    if config.get("do_sample").and_then(|v| v.as_bool()) == Some(true) {
        defaults.temperature = config
            .get("temperature")
            .and_then(|v| v.as_f64())
            .map_or(TRANSFORMERS_TEMPERATURE, |v| v as f32);
    }
    if let Some(v) = config.get("top_k").and_then(|v| v.as_u64()) {
        defaults.top_k = v as usize;
    }
    if let Some(v) = config.get("top_p").and_then(|v| v.as_f64()) {
        defaults.top_p = v as f32;
    }
    if let Some(v) = config.get("min_p").and_then(|v| v.as_f64()) {
        defaults.min_p = v as f32;
    }
    if let Some(v) = config.get("repetition_penalty").and_then(|v| v.as_f64()) {
        defaults.repetition_penalty = v as f32;
    }
    if let Some(v) = config.get("frequency_penalty").and_then(|v| v.as_f64()) {
        defaults.frequency_penalty = v as f32;
    }
    if let Some(v) = config.get("presence_penalty").and_then(|v| v.as_f64()) {
        defaults.presence_penalty = v as f32;
    }
    defaults
}

// =============================================================================
// Output length
// =============================================================================

/// Output budget for a thinking model when its `generation_config.json`
/// names none: the length Qwen's cards recommend "for most queries" (Qwen3,
/// Qwen3.5, Qwen3.6, and the Qwen3-2507 Thinking models), which DeepSeek-R1
/// and Phi-4-reasoning also generate with. A thinking model spends most of
/// it reasoning, so a budget of a few hundred tokens ends it mid-thought.
pub const THINKING_MAX_TOKENS: usize = 32_768;

/// Output budget for a model that neither thinks nor names one.
pub const FALLBACK_MAX_TOKENS: usize = 256;

/// Where a default output budget came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxTokensSource {
    /// `generation_config.json` `max_new_tokens`.
    GenerationConfigMaxNewTokens,
    /// `generation_config.json` `max_length` (prompt included, as in transformers).
    GenerationConfigMaxLength,
    /// The model card's recommendation ([`ModelFamily::recommended_max_tokens`]).
    ModelCard,
    /// [`THINKING_MAX_TOKENS`]: the model thinks and names no budget.
    Thinking,
    /// [`FALLBACK_MAX_TOKENS`].
    Fallback,
}

impl std::fmt::Display for MaxTokensSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::GenerationConfigMaxNewTokens => "generation_config.json max_new_tokens",
            Self::GenerationConfigMaxLength => "generation_config.json max_length",
            Self::ModelCard => "model card",
            Self::Thinking => "thinking-model default",
            Self::Fallback => "default",
        })
    }
}

/// A model's default output budget, before the context window bounds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxTokensDefault {
    /// The budget. With [`MaxTokensSource::GenerationConfigMaxLength`] it
    /// counts the prompt too.
    pub tokens: usize,
    /// Where it came from.
    pub source: MaxTokensSource,
}

impl MaxTokensDefault {
    /// New tokens left for a `prompt_len`-token prompt: the budget less the
    /// prompt for a `max_length`, and never past the context window.
    pub fn for_prompt(&self, prompt_len: usize, context_window: Option<usize>) -> usize {
        let budget = match self.source {
            MaxTokensSource::GenerationConfigMaxLength => self.tokens.saturating_sub(prompt_len),
            _ => self.tokens,
        };
        let budget = match context_window {
            Some(context) => budget.min(context.saturating_sub(prompt_len)),
            None => budget,
        };
        budget.max(1)
    }
}

/// The default output budget for a model, used when the caller sets none
/// (`pmetal infer` without `--max-tokens`, a server request without
/// `max_tokens`). In order:
///
/// 1. `generation_config.json` `max_new_tokens`;
/// 2. `generation_config.json` `max_length`, which transformers counts with
///    the prompt;
/// 3. the model card's figure, for the families whose makers give one
///    ([`ModelFamily::recommended_max_tokens`]);
/// 4. [`THINKING_MAX_TOKENS`] when `thinking` (the chat template thinks and
///    thinking is on);
/// 5. [`FALLBACK_MAX_TOKENS`].
///
/// [`MaxTokensDefault::for_prompt`] then keeps it inside the context window.
pub fn default_max_tokens(model_path: &Path, thinking: bool) -> MaxTokensDefault {
    let config = read_json(&model_path.join("generation_config.json"));
    let field = |name: &str| {
        config
            .as_ref()
            .and_then(|c| c.get(name))
            .and_then(|v| v.as_u64())
            .filter(|&v| v > 0)
            .map(|v| v as usize)
    };
    if let Some(tokens) = field("max_new_tokens") {
        return MaxTokensDefault {
            tokens,
            source: MaxTokensSource::GenerationConfigMaxNewTokens,
        };
    }
    if let Some(tokens) = field("max_length") {
        return MaxTokensDefault {
            tokens,
            source: MaxTokensSource::GenerationConfigMaxLength,
        };
    }
    if let Some(tokens) = ModelFamily::detect(model_path).and_then(|f| f.recommended_max_tokens()) {
        return MaxTokensDefault {
            tokens,
            source: MaxTokensSource::ModelCard,
        };
    }
    if thinking {
        MaxTokensDefault {
            tokens: THINKING_MAX_TOKENS,
            source: MaxTokensSource::Thinking,
        }
    } else {
        MaxTokensDefault {
            tokens: FALLBACK_MAX_TOKENS,
            source: MaxTokensSource::Fallback,
        }
    }
}

/// The model's context window: `max_position_embeddings` from
/// `config.json` (its `text_config` first, for multimodal wrappers), or the
/// stretched length a static YaRN `rope_parameters` (or `rope_scaling`)
/// gives, `original_max_position_embeddings * factor`, when longer; else the
/// tokenizer's `model_max_length` when that is a real bound.
pub fn context_window(model_path: &Path) -> Option<usize> {
    let config = read_json(&model_path.join("config.json"));
    let from_config = config.as_ref().and_then(|c| {
        let text = c.get("text_config").filter(|t| t.is_object()).unwrap_or(c);
        let max_pos = text
            .get("max_position_embeddings")
            .or_else(|| c.get("max_position_embeddings"))
            .and_then(|v| v.as_u64());
        let rope = text
            .get("rope_parameters")
            .filter(|r| r.is_object())
            .or_else(|| text.get("rope_scaling").filter(|r| r.is_object()));
        let yarn = rope
            .filter(|r| {
                r.get("rope_type")
                    .or_else(|| r.get("type"))
                    .and_then(|t| t.as_str())
                    == Some("yarn")
            })
            .and_then(|r| {
                let factor = r.get("factor")?.as_f64()?;
                let original = r
                    .get("original_max_position_embeddings")
                    .and_then(|v| v.as_f64())
                    .or(max_pos.map(|m| m as f64))?;
                Some((original * factor) as u64)
            });
        match (max_pos, yarn) {
            (Some(m), Some(y)) => Some(m.max(y)),
            (m, y) => m.or(y),
        }
    });
    let from_tokenizer = || {
        read_json(&model_path.join("tokenizer_config.json"))
            .and_then(|t| t.get("model_max_length").and_then(|v| v.as_u64()))
            // transformers writes a huge sentinel when there is no bound.
            .filter(|&v| v < 1 << 40)
    };
    from_config
        .or_else(from_tokenizer)
        .filter(|&v| v > 0)
        .map(|v| v as usize)
}

fn read_json(path: &Path) -> Option<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

/// Load sampling defaults with the full resolution chain:
///
/// 1. Start with transformers' defaults (`SamplingDefaults::default()`: greedy)
/// 2. Override with `generation_config.json` (if present), as transformers
///    reads it ([`load_sampling_from_generation_config`])
/// 3. Override with mode preset (if mode is set and model family has presets)
///
/// CLI/GUI explicit overrides happen in the caller (inference_runner.rs), not here.
pub fn load_sampling_defaults(
    model_path: &Path,
    mode: SamplingMode,
    thinking: bool,
) -> SamplingDefaults {
    // Steps 1 + 2: baseline + generation_config.json.
    let mut defaults = load_sampling_from_generation_config(model_path);

    // Step 3: model-card preset (overrides generation_config values for
    // mode-specific params like presence_penalty that the JSON usually
    // lacks).
    let resolved_mode = resolve_auto_mode(mode, thinking);
    let family = ModelFamily::detect(model_path);
    if let Some(preset) = model_preset(family, resolved_mode) {
        tracing::info!(
            family = ?family,
            mode = %resolved_mode,
            temp = preset.temperature,
            top_p = preset.top_p,
            top_k = preset.top_k,
            presence_penalty = preset.presence_penalty,
            "Applying model-card sampling preset"
        );
        defaults = preset;
    }

    defaults
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, body) in files {
            std::fs::write(dir.path().join(name), body).unwrap();
        }
        dir
    }

    #[test]
    fn max_tokens_follows_generation_config_then_thinking_then_fallback() {
        let dir = model_dir(&[(
            "generation_config.json",
            r#"{"max_new_tokens": 131072, "max_length": 9}"#,
        )]);
        let d = default_max_tokens(dir.path(), false);
        assert_eq!(d.source, MaxTokensSource::GenerationConfigMaxNewTokens);
        assert_eq!(d.tokens, 131_072);

        let dir = model_dir(&[("generation_config.json", r#"{"max_length": 32768}"#)]);
        let d = default_max_tokens(dir.path(), true);
        assert_eq!(d.source, MaxTokensSource::GenerationConfigMaxLength);
        // transformers counts the prompt inside max_length.
        assert_eq!(d.for_prompt(1000, None), 31_768);

        let dir = model_dir(&[("generation_config.json", r#"{"temperature": 0.6}"#)]);
        assert_eq!(
            default_max_tokens(dir.path(), true),
            MaxTokensDefault {
                tokens: THINKING_MAX_TOKENS,
                source: MaxTokensSource::Thinking
            }
        );
        assert_eq!(
            default_max_tokens(dir.path(), false).tokens,
            FALLBACK_MAX_TOKENS
        );
    }

    #[test]
    fn max_tokens_stays_inside_the_context_window() {
        let dir = model_dir(&[(
            "config.json",
            r#"{"text_config": {"max_position_embeddings": 40960}, "max_position_embeddings": 1}"#,
        )]);
        let context = context_window(dir.path());
        assert_eq!(context, Some(40_960));
        let d = default_max_tokens(dir.path(), true);
        assert_eq!(d.for_prompt(10_000, context), 30_960);
        assert_eq!(d.for_prompt(100, context), THINKING_MAX_TOKENS);
        // A prompt that fills the window still gets one token.
        assert_eq!(d.for_prompt(50_000, context), 1);

        let dir = model_dir(&[(
            "tokenizer_config.json",
            r#"{"model_max_length": 1000000000000000019884624838656}"#,
        )]);
        assert_eq!(context_window(dir.path()), None);

        // The Qwen3.8 card's YaRN block stretches 262,144 to 1,048,576.
        let dir = model_dir(&[(
            "config.json",
            r#"{"text_config": {"max_position_embeddings": 262144, "rope_parameters":
                {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}}}"#,
        )]);
        assert_eq!(context_window(dir.path()), Some(1_048_576));
    }

    #[test]
    fn families_are_told_apart_by_model_type_and_name() {
        use ModelFamily::*;
        let f = ModelFamily::from_model_type_and_name;
        assert_eq!(f("qwen3_5_text", "qwen/qwen3.8-27b"), Some(Qwen3_8));
        assert_eq!(f("qwen3_5_text", "qwen/qwen3.6-27b"), Some(Qwen3_6));
        assert_eq!(f("qwen3_5_text", "qwen/qwen3.5-0.8b"), Some(Qwen3_5));
        assert_eq!(f("qwen3_5_moe_text", "my-finetune"), Some(Qwen3_8));
        assert_eq!(
            f("qwen4_exp_text", "qwen/qwen3.8-flash-next"),
            Some(Qwen3_8)
        );
        assert_eq!(
            f("qwen3_next", "qwen/qwen3-next-80b-a3b-thinking"),
            Some(Qwen3Thinking2507)
        );
        assert_eq!(f("qwen3", "qwen/qwen3-8b"), Some(Qwen3));
        assert_eq!(
            f("qwen3_moe", "qwen/qwen3-30b-a3b-instruct-2507"),
            Some(Qwen3Instruct2507)
        );
        assert_eq!(f("gemma4_text", "google/gemma-4-31b-it"), Some(Gemma4));
        assert_eq!(f("gpt_oss", "openai/gpt-oss-20b"), Some(GptOss));
        assert_eq!(
            f("qwen2", "deepseek-ai/deepseek-r1-distill-qwen-7b"),
            Some(DeepSeekR1)
        );
        assert_eq!(
            f("phi3", "microsoft/phi-4-reasoning-plus"),
            Some(Phi4Reasoning)
        );
        assert_eq!(f("llama", "meta-llama/llama-3.1-8b-instruct"), None);
        assert_eq!(f("qwen2", "qwen/qwen2.5-7b-instruct"), None);
    }

    #[test]
    fn presets_are_the_cards_values() {
        use ModelFamily::*;
        use SamplingMode::*;
        let p = |family: ModelFamily, mode| {
            let d = family.preset(mode).unwrap();
            (d.temperature, d.top_p, d.top_k, d.presence_penalty)
        };
        assert_eq!(p(Qwen3_8, ThinkingGeneral), (1.0, 0.95, 20, 0.0));
        assert_eq!(p(Qwen3_8, InstructGeneral), (0.7, 0.8, 20, 1.5));
        // A mode the card has no row for falls back to its general one.
        assert_eq!(p(Qwen3_8, ThinkingCoding), (1.0, 0.95, 20, 0.0));
        assert_eq!(p(Qwen3_6, ThinkingCoding), (0.6, 0.95, 20, 0.0));
        assert_eq!(p(Qwen3, ThinkingGeneral), (0.6, 0.95, 20, 0.0));
        assert_eq!(p(Qwen3, InstructGeneral), (0.7, 0.8, 20, 0.0));
        assert_eq!(p(Gemma4, ThinkingGeneral), (1.0, 0.95, 64, 0.0));
        assert_eq!(p(GptOss, InstructGeneral), (1.0, 1.0, 0, 0.0));
        assert_eq!(p(NemotronNanoV2, InstructGeneral).0, 0.0);
        assert!(Qwen3_8.preset(Auto).is_none());
        assert_eq!(Qwen3Instruct2507.recommended_max_tokens(), Some(16_384));
        assert_eq!(GptOss.recommended_max_tokens(), None);
    }

    /// Needs a gpt-oss snapshot (its tokenizer and generation config are
    /// enough): `PMETAL_GPT_OSS_DIR=<snapshot> cargo test -p pmetal-data`.
    #[test]
    fn harmony_stops_at_the_end_of_the_turn_not_of_a_message() {
        let Some(dir) = std::env::var_os("PMETAL_GPT_OSS_DIR").map(std::path::PathBuf::from) else {
            eprintln!("PMETAL_GPT_OSS_DIR is not set; skipping");
            return;
        };
        let dir = dir.as_path();
        let tokenizer = Tokenizer::from_model_dir(dir).unwrap();
        let stops = collect_all_stop_tokens(dir, &tokenizer, Some(ChatTemplateType::GptOss));
        let id = |name| tokenizer.inner().token_to_id(name).unwrap();
        assert!(stops.contains(&id("<|return|>")));
        assert!(!stops.contains(&id("<|end|>")));
    }

    #[test]
    fn generation_config_is_read_as_transformers_reads_it() {
        let load = |files: &[(&str, &str)]| {
            let dir = model_dir(files);
            load_sampling_defaults(dir.path(), SamplingMode::Auto, false)
        };
        let greedy = SamplingDefaults {
            temperature: 0.0,
            top_k: 50,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        };
        // No generation_config.json, or one without do_sample: greedy.
        assert_eq!(load(&[]), greedy);
        assert_eq!(
            load(&[("generation_config.json", r#"{"eos_token_id": 2}"#)]),
            greedy
        );
        // A temperature without do_sample is ignored, as transformers does.
        assert_eq!(
            load(&[(
                "generation_config.json",
                r#"{"temperature": 0.6, "top_p": 0.9}"#
            )]),
            SamplingDefaults {
                top_p: 0.9,
                ..greedy.clone()
            }
        );
        // do_sample: every field it sets, transformers' default for the rest.
        assert_eq!(
            load(&[(
                "generation_config.json",
                r#"{"do_sample": true, "temperature": 0.6, "top_p": 0.9,
                    "repetition_penalty": 1.05}"#
            )]),
            SamplingDefaults {
                temperature: 0.6,
                top_p: 0.9,
                repetition_penalty: 1.05,
                ..greedy.clone()
            }
        );
        assert_eq!(
            load(&[("generation_config.json", r#"{"do_sample": true}"#)]),
            SamplingDefaults {
                temperature: 1.0,
                ..greedy.clone()
            }
        );
        // A family with a card keeps its preset over the file.
        let dir = model_dir(&[
            ("config.json", r#"{"model_type": "gemma4_text"}"#),
            ("generation_config.json", r#"{"do_sample": false}"#),
        ]);
        let gemma = load_sampling_defaults(dir.path(), SamplingMode::Auto, true);
        assert_eq!((gemma.temperature, gemma.top_k), (1.0, 64));
    }

    #[test]
    fn hugging_face_cache_paths_name_the_repo() {
        let path = Path::new("/hf/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2");
        assert_eq!(model_name(path), "Qwen/Qwen3.8-27B");
        assert_eq!(model_name(Path::new("/m/gpt-oss-20b")), "gpt-oss-20b");
    }
}
