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
        "<turn|>", // Gemma 4 — distinct from Gemma 2/3's <end_of_turn>.
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

/// Read `generation_config.json` into a [`SamplingDefaults`], starting from
/// the global fallback and overriding each field that the file explicitly
/// provides. Missing fields leave the global default in place. This is the
/// raw stage-2 of the loading pipeline (step 1+2 in [`load_sampling_defaults`])
/// and is exposed so the chat-template audit can verify that HF's declared
/// values are actually picked up, independent of any mode preset that might
/// override them later.
pub fn load_sampling_from_generation_config(model_path: &Path) -> SamplingDefaults {
    let mut defaults = SamplingDefaults::default();
    let config_path = model_path.join("generation_config.json");
    if !config_path.exists() {
        return defaults;
    }
    let content = match std::fs::read_to_string(&config_path) {
        Ok(c) => c,
        Err(_) => return defaults,
    };
    let config: serde_json::Value = match serde_json::from_str(&content) {
        Ok(c) => c,
        Err(_) => return defaults,
    };
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
/// 3. [`THINKING_MAX_TOKENS`] when `thinking` (the chat template thinks and
///    thinking is on);
/// 4. [`FALLBACK_MAX_TOKENS`].
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
/// `config.json` (its `text_config` first, for multimodal wrappers), else
/// the tokenizer's `model_max_length` when that is a real bound.
pub fn context_window(model_path: &Path) -> Option<usize> {
    let config = read_json(&model_path.join("config.json"));
    let from_config = config.as_ref().and_then(|c| {
        c.get("text_config")
            .and_then(|t| t.get("max_position_embeddings"))
            .or_else(|| c.get("max_position_embeddings"))
            .and_then(|v| v.as_u64())
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
    // Steps 1 + 2: baseline + generation_config.json.
    let mut defaults = load_sampling_from_generation_config(model_path);

    // Step 3: model-card preset (overrides generation_config values for
    // mode-specific params like presence_penalty that the JSON usually
    // lacks).
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
    }
}
