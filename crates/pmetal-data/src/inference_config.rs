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
//! 1. **CLI/GUI explicit override** — always wins
//! 2. **`--mode` preset** — model-family-specific preset (e.g., `thinking-coding`)
//! 3. **`generation_config.json`** — model's declared defaults
//! 4. **Global fallback** — `SamplingDefaults::default()` (temp=0.7, top_p=0.8)

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
        "<|end|>",
        "<|return|>",
        "<|END_OF_TURN_TOKEN|>",
        "<｜end▁of▁sentence｜>",
        "</s>",
    ];
    for candidate in &candidates {
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

/// Sampling hyperparameter defaults loaded from model config.
#[derive(Debug, Clone)]
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

impl Default for SamplingDefaults {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.8,
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

/// Return the available sampling modes for a model family.
///
/// Models without specific recommendations return an empty slice.
pub fn available_modes(template: Option<ChatTemplateType>) -> &'static [SamplingMode] {
    match template {
        Some(ChatTemplateType::Qwen) => &[
            SamplingMode::ThinkingGeneral,
            SamplingMode::ThinkingCoding,
            SamplingMode::InstructGeneral,
            SamplingMode::InstructReasoning,
        ],
        _ => &[],
    }
}

/// Resolve a sampling mode to concrete parameters for a model family.
///
/// Returns `None` if the model family has no presets (use generation_config.json
/// or global defaults instead).
///
/// Sources:
/// - Qwen3.5 README "Best Practices" section (2026-04)
/// - Qwen3 README "Best Practices" section (2025-04)
/// - DeepSeek-R1 README (params match generation_config.json, no extra presets needed)
pub fn model_preset(
    template: Option<ChatTemplateType>,
    mode: SamplingMode,
) -> Option<SamplingDefaults> {
    let template = template?;

    match template {
        ChatTemplateType::Qwen => qwen_preset(mode),
        _ => None,
    }
}

/// Qwen3 / Qwen3.5 recommended sampling presets.
///
/// From Qwen3.5 model card (applies to all Qwen3.5 sizes):
///   - Thinking general:     temp=1.0, top_p=0.95, top_k=20, presence_penalty=0.0 (card says 1.5, causes early EOS)
///   - Thinking coding:      temp=0.6, top_p=0.95, top_k=20, presence_penalty=0.0
///   - Instruct general:     temp=0.7, top_p=0.8,  top_k=20, presence_penalty=1.5
///   - Instruct reasoning:   temp=1.0, top_p=1.0,  top_k=40, presence_penalty=2.0
///
/// Qwen3 uses the same thinking/non-thinking split with slightly different defaults
/// (temp=0.6 thinking, temp=0.7 non-thinking) which are close enough that the
/// Qwen3.5 presets work well for both.
fn qwen_preset(mode: SamplingMode) -> Option<SamplingDefaults> {
    let preset = match mode {
        SamplingMode::Auto => return None, // caller resolves Auto before calling
        SamplingMode::ThinkingGeneral => SamplingDefaults {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            // Model card says 1.5, but presence penalty during thinking chains
            // penalizes common tokens ("the", "is", etc.) and causes early EOS.
            // ThinkingCoding already uses 0.0; keep thinking modes penalty-free.
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
        },
        SamplingMode::ThinkingCoding => SamplingDefaults {
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
        },
        SamplingMode::InstructGeneral => SamplingDefaults {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            min_p: 0.0,
            presence_penalty: 1.5,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
        },
        SamplingMode::InstructReasoning => SamplingDefaults {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 40,
            min_p: 0.0,
            presence_penalty: 2.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
        },
    };
    Some(preset)
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

/// Load sampling defaults with the full resolution chain:
///
/// 1. Start with global fallback (`SamplingDefaults::default()`)
/// 2. Override with `generation_config.json` (if present)
/// 3. Override with mode preset (if mode is set and model family has presets)
///
/// CLI/GUI explicit overrides happen in the caller (inference_runner.rs), not here.
pub fn load_sampling_defaults(
    model_path: &Path,
    template: Option<ChatTemplateType>,
    mode: SamplingMode,
    thinking: bool,
) -> SamplingDefaults {
    // 1. Global fallback
    let mut defaults = SamplingDefaults::default();

    // 2. generation_config.json (model's declared defaults)
    let config_path = model_path.join("generation_config.json");
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(v) = config.get("temperature").and_then(|v| v.as_f64()) {
                    defaults.temperature = v as f32;
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
            }
        }
    }

    // 3. Mode preset (overrides generation_config.json for mode-specific params)
    let resolved_mode = resolve_auto_mode(mode, thinking);
    if let Some(preset) = model_preset(template, resolved_mode) {
        tracing::info!(
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
