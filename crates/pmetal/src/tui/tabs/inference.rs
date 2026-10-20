//! Inference tab: chat with a model.
//!
//! Each message runs `pmetal infer --chat` and shows the reply as it is
//! generated. The sidebar edits an [`InferSpec`]: what it starts as, its
//! limits, and the argv it launches are the spec's. A setting that has not been
//! touched is not passed, so the model's own `generation_config.json` rules;
//! stepping a setting starts it from a sensible value, and `d` puts it back to
//! the model's.

use pmetal_core::JobFields;
use pmetal_core::jobs::InferSpec;
use pmetal_data::inference_config::InferenceBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, StatefulWidget,
    Widget,
};

use crate::tui::tabs::infer_stream::{InferStream, split_reasoning};
use crate::tui::theme::THEME;

/// A chat message.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    pub tokens: Option<usize>,
    pub tok_sec: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl std::fmt::Display for ChatRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatRole::System => write!(f, "system"),
            ChatRole::User => write!(f, "user"),
            ChatRole::Assistant => write!(f, "assistant"),
        }
    }
}

/// How the KV cache is quantized: one choice that stands for several spec
/// fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvMode {
    /// The engine decides.
    Auto,
    Q8,
    Q4,
    TurboQuant8,
    TurboQuant4,
    TurboQuant2_5,
    TurboQuant3_5,
    /// No quantization.
    Fp16,
}

impl KvMode {
    const ALL: [KvMode; 8] = [
        KvMode::Auto,
        KvMode::Q8,
        KvMode::Q4,
        KvMode::TurboQuant8,
        KvMode::TurboQuant4,
        KvMode::TurboQuant2_5,
        KvMode::TurboQuant3_5,
        KvMode::Fp16,
    ];

    fn label(self) -> &'static str {
        match self {
            KvMode::Auto => "Auto",
            KvMode::Q8 => "Q8",
            KvMode::Q4 => "Q4",
            KvMode::TurboQuant8 => "TQ8",
            KvMode::TurboQuant4 => "TQ4",
            KvMode::TurboQuant2_5 => "TQ2.5",
            KvMode::TurboQuant3_5 => "TQ3.5",
            KvMode::Fp16 => "FP16",
        }
    }

    /// Set the spec's KV fields to this mode.
    fn apply(self, spec: &mut InferSpec) {
        spec.kv_quant = None;
        spec.kv_turboquant = false;
        spec.kv_quant_preset = None;
        spec.no_kv_quant = false;
        match self {
            KvMode::Auto => {}
            KvMode::Q8 => spec.kv_quant = Some(8),
            KvMode::Q4 => spec.kv_quant = Some(4),
            KvMode::TurboQuant8 => {
                spec.kv_quant = Some(8);
                spec.kv_turboquant = true;
            }
            KvMode::TurboQuant4 => {
                spec.kv_quant = Some(4);
                spec.kv_turboquant = true;
            }
            KvMode::TurboQuant2_5 => spec.kv_quant_preset = Some("q2_5".to_string()),
            KvMode::TurboQuant3_5 => spec.kv_quant_preset = Some("q3_5".to_string()),
            KvMode::Fp16 => spec.no_kv_quant = true,
        }
    }

    /// The mode a spec's KV fields stand for.
    fn of(spec: &InferSpec) -> KvMode {
        Self::ALL
            .into_iter()
            .find(|mode| {
                let mut probe = InferSpec::default();
                mode.apply(&mut probe);
                probe.kv_quant == spec.kv_quant
                    && probe.kv_turboquant == spec.kv_turboquant
                    && probe.kv_quant_preset == spec.kv_quant_preset
                    && probe.no_kv_quant == spec.no_kv_quant
            })
            .unwrap_or(KvMode::Auto)
    }
}

/// `Reasoning` setting values: the chat template's default, then the levels
/// Qwen3.8 (`low`, `medium`, `xhigh`) and gpt-oss (`low`, `medium`, `high`)
/// accept. A level the loaded model refuses is reported by `pmetal infer`.
const REASONING_EFFORTS: &[Option<&str>] = &[
    None,
    Some("low"),
    Some("medium"),
    Some("high"),
    Some("xhigh"),
];

/// What the sidebar can edit, in the order it lists them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Setting {
    Temperature,
    MaxTokens,
    TopK,
    TopP,
    MinP,
    RepetitionPenalty,
    FrequencyPenalty,
    PresencePenalty,
    Seed,
    KvCache,
    Fp8,
    NoThinking,
    Backend,
    Reasoning,
}

const SETTINGS: [Setting; 14] = [
    Setting::Temperature,
    Setting::MaxTokens,
    Setting::TopK,
    Setting::TopP,
    Setting::MinP,
    Setting::RepetitionPenalty,
    Setting::FrequencyPenalty,
    Setting::PresencePenalty,
    Setting::Seed,
    Setting::KvCache,
    Setting::Fp8,
    Setting::NoThinking,
    Setting::Backend,
    Setting::Reasoning,
];

impl Setting {
    fn label(self) -> &'static str {
        match self {
            Setting::Temperature => "Temperature",
            Setting::MaxTokens => "Max Tokens",
            Setting::TopK => "Top-k",
            Setting::TopP => "Top-p",
            Setting::MinP => "Min-p",
            Setting::RepetitionPenalty => "Rep Penalty",
            Setting::FrequencyPenalty => "Freq Penalty",
            Setting::PresencePenalty => "Pres Penalty",
            Setting::Seed => "Seed",
            Setting::KvCache => "KV Cache",
            Setting::Fp8 => "FP8",
            Setting::NoThinking => "No Thinking",
            Setting::Backend => "Backend",
            Setting::Reasoning => "Reasoning",
        }
    }

    /// The spec field behind a numeric setting, for its bounds.
    fn field(self) -> Option<&'static str> {
        Some(match self {
            Setting::Temperature => "temperature",
            Setting::MaxTokens => "max_tokens",
            Setting::TopK => "top_k",
            Setting::TopP => "top_p",
            Setting::MinP => "min_p",
            Setting::RepetitionPenalty => "repetition_penalty",
            Setting::FrequencyPenalty => "frequency_penalty",
            Setting::PresencePenalty => "presence_penalty",
            _ => return None,
        })
    }
}

/// The inclusive bounds the spec gives a numeric field.
fn bounds(field: &str) -> (f64, f64) {
    InferSpec::field_descriptors()
        .iter()
        .find(|d| d.name == field)
        .map_or((f64::MIN, f64::MAX), |d| match d.kind {
            pmetal_core::FieldKind::Number { min, max } => (min, max),
            pmetal_core::FieldKind::Integer { min, max } => (min as f64, max as f64),
            _ => (f64::MIN, f64::MAX),
        })
}

/// Step an optional number: from unset it starts at `start`; it stays within
/// the spec's bounds.
fn step_number(current: Option<f64>, start: f64, step: f64, direction: i32, field: &str) -> f64 {
    let (min, max) = bounds(field);
    let next = match current {
        Some(v) => v + step * f64::from(direction),
        None => start,
    };
    // Round to the step so repeated stepping does not drift (0.7000000001).
    let next = (next / step).round() * step;
    next.clamp(min, max)
}

/// Focus mode for the inference tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceFocus {
    /// Typing in the input box.
    Input,
    /// Navigating settings sidebar.
    Settings,
    /// Browsing messages (sidebar hidden, message selection active).
    Browse,
}

/// What a model's `generation_config.json` says, for display.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenerationDefaults {
    pub temperature: Option<f64>,
    pub top_k: Option<u64>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub max_tokens: Option<u64>,
}

impl GenerationDefaults {
    pub fn read(model_path: &std::path::Path) -> Self {
        let Ok(text) = std::fs::read_to_string(model_path.join("generation_config.json")) else {
            return Self::default();
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
            return Self::default();
        };
        Self {
            temperature: json.get("temperature").and_then(|v| v.as_f64()),
            top_k: json.get("top_k").and_then(|v| v.as_u64()),
            top_p: json.get("top_p").and_then(|v| v.as_f64()),
            min_p: json.get("min_p").and_then(|v| v.as_f64()),
            repetition_penalty: json.get("repetition_penalty").and_then(|v| v.as_f64()),
            max_tokens: json
                .get("max_new_tokens")
                .and_then(|v| v.as_u64())
                .or_else(|| json.get("max_length").and_then(|v| v.as_u64())),
        }
    }
}

/// Inference tab state.
pub struct InferenceTab {
    pub model_id: Option<String>,
    pub messages: Vec<ChatMessage>,
    input_buffer: String,
    cursor_position: usize,
    pub generating: bool,
    /// The settings; a field left at its default is not passed to the command.
    pub spec: InferSpec,
    /// What the loaded model says its defaults are, for the sidebar.
    pub model_defaults: GenerationDefaults,
    pub focus: InferenceFocus,
    pub settings_selected: usize,
    message_scroll: usize,
    /// Currently selected message index in Browse mode.
    pub selected_message: Option<usize>,
    /// Whether the sidebar is visible.
    pub sidebar_visible: bool,
    /// The reader of the reply being generated.
    stream: Option<InferStream>,
    /// The job generating the reply.
    pub job: Option<String>,
}

impl InferenceTab {
    pub fn new() -> Self {
        Self {
            model_id: None,
            messages: Vec::new(),
            input_buffer: String::new(),
            cursor_position: 0,
            generating: false,
            spec: InferSpec {
                chat: true,
                ..InferSpec::default()
            },
            model_defaults: GenerationDefaults::default(),
            focus: InferenceFocus::Input,
            settings_selected: 0,
            message_scroll: 0,
            selected_message: None,
            sidebar_visible: true,
            stream: None,
            job: None,
        }
    }

    pub fn input_focused(&self) -> bool {
        self.focus == InferenceFocus::Input
    }

    pub fn settings_focused(&self) -> bool {
        self.focus == InferenceFocus::Settings
    }

    // ── Input ───────────────────────────────────────────────────────────

    pub fn insert_char(&mut self, c: char) {
        self.input_buffer.insert(self.cursor_position, c);
        self.cursor_position += c.len_utf8();
    }

    pub fn paste(&mut self, text: &str) {
        // A line ending is one newline, whichever way it is written.
        for c in text.replace("\r\n", "\n").replace('\r', "\n").chars() {
            self.insert_char(c);
        }
    }

    pub fn delete_char(&mut self) {
        if self.cursor_position > 0 {
            let prev = self.input_buffer[..self.cursor_position]
                .char_indices()
                .next_back()
                .map_or(0, |(i, _)| i);
            self.input_buffer.remove(prev);
            self.cursor_position = prev;
        }
    }

    pub fn move_cursor_left(&mut self) {
        if self.cursor_position > 0 {
            self.cursor_position = self.input_buffer[..self.cursor_position]
                .char_indices()
                .next_back()
                .map_or(0, |(i, _)| i);
        }
    }

    pub fn move_cursor_right(&mut self) {
        if let Some(c) = self.input_buffer[self.cursor_position..].chars().next() {
            self.cursor_position += c.len_utf8();
        }
    }

    /// Take the current input buffer contents and clear it.
    pub fn take_input(&mut self) -> String {
        let s = std::mem::take(&mut self.input_buffer);
        self.cursor_position = 0;
        s
    }

    /// Set the input buffer (e.g., to restore after an error).
    pub fn set_input(&mut self, text: &str) {
        self.input_buffer = text.to_string();
        self.cursor_position = self.input_buffer.len();
    }

    pub fn input(&self) -> &str {
        &self.input_buffer
    }

    // ── The conversation ────────────────────────────────────────────────

    /// Add a user message to the chat.
    pub fn add_user_message(&mut self, content: &str) {
        self.messages.push(ChatMessage {
            role: ChatRole::User,
            content: content.to_string(),
            tokens: None,
            tok_sec: None,
        });
        self.scroll_to_bottom();
    }

    /// Start an assistant message (placeholder for streaming) read from a new
    /// reply stream.
    pub fn start_generation(&mut self, job: String) {
        self.generating = true;
        self.job = Some(job);
        self.stream = Some(InferStream::new());
        self.messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content: String::new(),
            tokens: None,
            tok_sec: None,
        });
    }

    /// Standard output of the generating job arrived.
    pub fn receive(&mut self, chunk: &str) {
        let Some(stream) = &mut self.stream else {
            return;
        };
        let text = stream.push(chunk).to_string();
        self.set_reply(text);
    }

    fn set_reply(&mut self, text: String) {
        if let Some(msg) = self.messages.last_mut() {
            if msg.role == ChatRole::Assistant {
                msg.content = text;
            }
        }
        self.scroll_to_bottom();
    }

    /// The generation ended. `stdout` is everything the job printed, which is
    /// read afresh: it is authoritative, where the pieces that streamed in may
    /// have missed some. Returns the statistics if the footer was reached.
    pub fn finish_generation(
        &mut self,
        stdout: &str,
    ) -> Option<crate::tui::tabs::infer_stream::GenerationStats> {
        let mut stream = InferStream::new();
        stream.push(stdout);
        let stats = stream.finish();
        let complete = stream.is_complete();
        let text = stream.text().to_string();
        self.generating = false;
        self.stream = None;
        self.job = None;
        if let Some(msg) = self.messages.last_mut() {
            if msg.role == ChatRole::Assistant {
                if !text.is_empty() || msg.content.is_empty() {
                    msg.content = text;
                }
                if complete {
                    msg.tokens = Some(stats.tokens);
                    msg.tok_sec = Some(stats.tokens_per_second);
                }
            }
        }
        self.scroll_to_bottom();
        complete.then_some(stats)
    }

    /// Drop the reply being generated if nothing came of it.
    pub fn discard_empty_reply(&mut self) {
        if self
            .messages
            .last()
            .is_some_and(|m| m.role == ChatRole::Assistant && m.content.is_empty())
        {
            self.messages.pop();
        }
    }

    /// Load generation defaults from a model's `generation_config.json`, to
    /// show beside the settings.
    pub fn load_model_settings(&mut self, model_path: &std::path::Path) {
        self.model_defaults = GenerationDefaults::read(model_path);
    }

    // ── Focus ───────────────────────────────────────────────────────────

    /// Toggle settings sidebar focus (Ctrl+S).
    pub fn toggle_settings_focus(&mut self) {
        match self.focus {
            InferenceFocus::Input => {
                self.focus = InferenceFocus::Settings;
                self.sidebar_visible = true;
            }
            InferenceFocus::Settings => {
                self.focus = InferenceFocus::Input;
            }
            InferenceFocus::Browse => {
                self.focus = InferenceFocus::Settings;
                self.sidebar_visible = true;
                self.selected_message = None;
            }
        }
    }

    /// Toggle sidebar visibility (Ctrl+H).
    pub fn toggle_sidebar(&mut self) {
        self.sidebar_visible = !self.sidebar_visible;
        if !self.sidebar_visible && self.focus == InferenceFocus::Settings {
            self.focus = InferenceFocus::Input;
        }
    }

    /// Enter browse mode (F2): hide sidebar, enable message selection.
    pub fn enter_browse_mode(&mut self) {
        if self.messages.is_empty() {
            return;
        }
        self.focus = InferenceFocus::Browse;
        self.sidebar_visible = false;
        self.selected_message = Some(self.messages.len() - 1);
    }

    /// Exit browse mode back to input.
    pub fn exit_browse_mode(&mut self) {
        self.focus = InferenceFocus::Input;
        self.selected_message = None;
    }

    pub fn next_message(&mut self) {
        if let Some(idx) = self.selected_message {
            if idx + 1 < self.messages.len() {
                self.selected_message = Some(idx + 1);
            }
        }
    }

    pub fn prev_message(&mut self) {
        if let Some(idx) = self.selected_message {
            if idx > 0 {
                self.selected_message = Some(idx - 1);
            }
        }
    }

    /// The text of the selected message to copy: an answer without its
    /// reasoning.
    pub fn yank_selected(&self) -> Option<String> {
        let msg = self.messages.get(self.selected_message?)?;
        Some(if msg.role == ChatRole::Assistant {
            split_reasoning(&msg.content).1.to_string()
        } else {
            msg.content.clone()
        })
    }

    // ── Settings ────────────────────────────────────────────────────────

    pub fn next_setting(&mut self) {
        self.settings_selected = (self.settings_selected + 1) % SETTINGS.len();
    }

    pub fn prev_setting(&mut self) {
        self.settings_selected = (self.settings_selected + SETTINGS.len() - 1) % SETTINGS.len();
    }

    pub fn increment_setting(&mut self) {
        self.step_setting(1);
    }

    pub fn decrement_setting(&mut self) {
        self.step_setting(-1);
    }

    fn step_setting(&mut self, direction: i32) {
        let setting = SETTINGS[self.settings_selected];
        let field = setting.field().unwrap_or("");
        let s = &mut self.spec;
        match setting {
            Setting::Temperature => {
                let v = step_number(s.temperature.map(f64::from), 0.7, 0.1, direction, field);
                s.temperature = Some(v as f32);
            }
            Setting::MaxTokens => {
                let v = step_number(
                    s.max_tokens.map(|v| v as f64),
                    512.0,
                    64.0,
                    direction,
                    field,
                );
                s.max_tokens = Some(v.max(1.0) as usize);
            }
            Setting::TopK => {
                let v = step_number(s.top_k.map(|v| v as f64), 40.0, 10.0, direction, field);
                s.top_k = Some(v as usize);
            }
            Setting::TopP => {
                let v = step_number(s.top_p.map(f64::from), 0.9, 0.05, direction, field);
                s.top_p = Some(v as f32);
            }
            Setting::MinP => {
                let v = step_number(s.min_p.map(f64::from), 0.05, 0.01, direction, field);
                s.min_p = Some(v as f32);
            }
            Setting::RepetitionPenalty => {
                let v = step_number(
                    s.repetition_penalty.map(f64::from),
                    1.1,
                    0.05,
                    direction,
                    field,
                );
                s.repetition_penalty = Some(v as f32);
            }
            Setting::FrequencyPenalty => {
                let v = step_number(
                    s.frequency_penalty.map(f64::from),
                    0.1,
                    0.05,
                    direction,
                    field,
                );
                s.frequency_penalty = Some(v as f32);
            }
            Setting::PresencePenalty => {
                let v = step_number(
                    s.presence_penalty.map(f64::from),
                    0.1,
                    0.05,
                    direction,
                    field,
                );
                s.presence_penalty = Some(v as f32);
            }
            Setting::Seed => {
                s.seed = match (s.seed, direction) {
                    (None, _) => Some(42),
                    (Some(n), d) if d > 0 => Some(n.saturating_add(1)),
                    (Some(0), _) => None,
                    (Some(n), _) => Some(n - 1),
                };
            }
            Setting::KvCache => {
                let modes = KvMode::ALL;
                let at = modes.iter().position(|m| *m == KvMode::of(s)).unwrap_or(0) as i32;
                let next = (at + direction).rem_euclid(modes.len() as i32) as usize;
                modes[next].apply(s);
            }
            Setting::Fp8 => s.fp8 = !s.fp8,
            Setting::NoThinking => s.no_thinking = !s.no_thinking,
            Setting::Backend => {
                let all = InferenceBackend::ALL;
                let current = all
                    .iter()
                    .position(|b| b.as_str() == s.backend)
                    .unwrap_or(0) as i32;
                let next = (current + direction).rem_euclid(all.len() as i32) as usize;
                s.backend = all[next].as_str().to_string();
            }
            Setting::Reasoning => {
                let at = REASONING_EFFORTS
                    .iter()
                    .position(|e| *e == s.reasoning_effort.as_deref())
                    .unwrap_or(0) as i32;
                let next = (at + direction).rem_euclid(REASONING_EFFORTS.len() as i32) as usize;
                s.reasoning_effort = REASONING_EFFORTS[next].map(str::to_string);
            }
        }
    }

    /// Put the selected setting back to the model's own default.
    pub fn reset_setting(&mut self) {
        let defaults = InferSpec::default();
        let s = &mut self.spec;
        match SETTINGS[self.settings_selected] {
            Setting::Temperature => s.temperature = defaults.temperature,
            Setting::MaxTokens => s.max_tokens = defaults.max_tokens,
            Setting::TopK => s.top_k = defaults.top_k,
            Setting::TopP => s.top_p = defaults.top_p,
            Setting::MinP => s.min_p = defaults.min_p,
            Setting::RepetitionPenalty => s.repetition_penalty = defaults.repetition_penalty,
            Setting::FrequencyPenalty => s.frequency_penalty = defaults.frequency_penalty,
            Setting::PresencePenalty => s.presence_penalty = defaults.presence_penalty,
            Setting::Seed => s.seed = defaults.seed,
            Setting::KvCache => KvMode::Auto.apply(s),
            Setting::Fp8 => s.fp8 = defaults.fp8,
            Setting::NoThinking => s.no_thinking = defaults.no_thinking,
            Setting::Backend => s.backend = defaults.backend,
            Setting::Reasoning => s.reasoning_effort = defaults.reasoning_effort,
        }
    }

    /// The spec to run for `prompt`: the settings, the model, an adapter.
    pub fn build_infer_spec(
        &self,
        model: String,
        prompt: String,
        lora: Option<String>,
    ) -> InferSpec {
        InferSpec {
            model,
            prompt,
            lora,
            chat: true,
            ..self.spec.clone()
        }
    }

    // ── Scrolling ───────────────────────────────────────────────────────

    /// Rows the conversation is scrolled from its top.
    #[cfg(test)]
    pub fn scroll_position(&self) -> usize {
        self.message_scroll
    }

    pub fn scroll_up(&mut self) {
        self.message_scroll = self.message_scroll.saturating_sub(3);
    }

    pub fn scroll_down(&mut self) {
        self.message_scroll = self.message_scroll.saturating_add(3);
    }

    fn scroll_to_bottom(&mut self) {
        // Set to a large value; render will clamp it
        self.message_scroll = usize::MAX;
    }

    /// The value the sidebar shows for a setting.
    fn setting_text(&self, setting: Setting) -> String {
        let s = &self.spec;
        let model_default = |value: Option<String>| match value {
            Some(v) => format!("model ({v})"),
            None => "model default".to_string(),
        };
        match setting {
            Setting::Temperature => s.temperature.map_or_else(
                || model_default(self.model_defaults.temperature.map(|v| format!("{v:.1}"))),
                |v| format!("{v:.1}"),
            ),
            Setting::MaxTokens => s.max_tokens.map_or_else(
                || model_default(self.model_defaults.max_tokens.map(|v| v.to_string())),
                |v| v.to_string(),
            ),
            Setting::TopK => s.top_k.map_or_else(
                || model_default(self.model_defaults.top_k.map(|v| v.to_string())),
                |v| v.to_string(),
            ),
            Setting::TopP => s.top_p.map_or_else(
                || model_default(self.model_defaults.top_p.map(|v| format!("{v:.2}"))),
                |v| format!("{v:.2}"),
            ),
            Setting::MinP => s.min_p.map_or_else(
                || model_default(self.model_defaults.min_p.map(|v| format!("{v:.2}"))),
                |v| format!("{v:.2}"),
            ),
            Setting::RepetitionPenalty => s.repetition_penalty.map_or_else(
                || {
                    model_default(
                        self.model_defaults
                            .repetition_penalty
                            .map(|v| format!("{v:.2}")),
                    )
                },
                |v| format!("{v:.2}"),
            ),
            Setting::FrequencyPenalty => s
                .frequency_penalty
                .map_or_else(|| "off".to_string(), |v| format!("{v:.2}")),
            Setting::PresencePenalty => s
                .presence_penalty
                .map_or_else(|| "off".to_string(), |v| format!("{v:.2}")),
            Setting::Seed => s
                .seed
                .map_or_else(|| "Random".to_string(), |v| v.to_string()),
            Setting::KvCache => KvMode::of(s).label().to_string(),
            Setting::Fp8 => on_off(s.fp8),
            Setting::NoThinking => on_off(s.no_thinking),
            Setting::Backend => InferenceBackend::ALL
                .iter()
                .find(|b| b.as_str() == s.backend)
                .map_or_else(|| s.backend.clone(), |b| b.label().to_string()),
            Setting::Reasoning => s
                .reasoning_effort
                .clone()
                .unwrap_or_else(|| "Default".to_string()),
        }
    }
}

fn on_off(on: bool) -> String {
    if on { "On" } else { "Off" }.to_string()
}

impl Default for InferenceTab {
    fn default() -> Self {
        Self::new()
    }
}

// ── Rendering ───────────────────────────────────────────────────────────

impl InferenceTab {
    /// Build the chat lines for rendering, with optional message highlighting.
    fn build_chat_lines(
        messages: &[ChatMessage],
        generating: bool,
        wrap_width: u16,
        selected_message: Option<usize>,
    ) -> Vec<Line<'static>> {
        let mut lines: Vec<Line<'static>> = Vec::new();
        let code_style = THEME.text_dim;
        let code_fence_style = THEME.text_muted;
        let text_width = wrap_width.saturating_sub(4) as usize;

        for (msg_idx, msg) in messages.iter().enumerate() {
            let is_selected = selected_message == Some(msg_idx);
            let role_style = match msg.role {
                ChatRole::System => THEME.text_muted,
                ChatRole::User => THEME.text_success,
                ChatRole::Assistant => THEME.text_bright,
            };
            let role_style = if is_selected {
                role_style.add_modifier(ratatui::style::Modifier::REVERSED)
            } else {
                role_style
            };
            let header = if is_selected {
                format!(" [{}] (y to copy) ", msg.role)
            } else {
                format!("[{}]", msg.role)
            };
            lines.push(Line::from(Span::styled(header, role_style)));

            match msg.role {
                ChatRole::Assistant if msg.content.is_empty() => {
                    if generating {
                        lines.push(Line::from(Span::styled("  ...", THEME.text_muted)));
                    }
                }
                ChatRole::Assistant => {
                    let (thinking, response) = split_reasoning(&msg.content);
                    if let Some(think_text) = thinking {
                        lines.push(Line::from(Span::styled(
                            "  --- thinking ---",
                            THEME.text_muted,
                        )));
                        for line in think_text.lines() {
                            for w in wrap_text(line, text_width) {
                                lines.push(Line::from(Span::styled(
                                    format!("  {w}"),
                                    THEME.text_dim,
                                )));
                            }
                        }
                        lines.push(Line::from(Span::styled(
                            "  --- response ---",
                            THEME.text_muted,
                        )));
                    }
                    render_markdown_content(
                        response,
                        text_width,
                        &mut lines,
                        THEME.text,
                        code_style,
                        code_fence_style,
                    );
                    if response.is_empty() && generating && thinking.is_some() {
                        lines.push(Line::from(Span::styled("  ...", THEME.text_muted)));
                    }
                }
                ChatRole::User | ChatRole::System => {
                    for line in msg.content.lines() {
                        for w in wrap_text(line, text_width) {
                            lines.push(Line::from(Span::styled(format!("  {w}"), THEME.text)));
                        }
                    }
                }
            }

            if let Some(tok_sec) = msg.tok_sec {
                lines.push(Line::from(Span::styled(
                    format!(
                        "  ({} tokens, {:.0} tok/s)",
                        msg.tokens.unwrap_or(0),
                        tok_sec
                    ),
                    THEME.text_muted,
                )));
            }
            lines.push(Line::from(""));
        }
        lines
    }
}

/// Normalize code fences so they always appear on their own line.
///
/// Models sometimes emit text without real newline characters, causing ```
/// to appear inline (e.g. "Here's the code:```pythondef fizzbuzz()...```Done!").
/// This preprocessor inserts `\n` before/after ``` markers so that
/// `render_markdown_content` can detect fences reliably via `.lines()`.
fn normalize_code_fences(text: &str) -> String {
    let mut result = String::with_capacity(text.len() + 128);
    let mut rest = text;

    while let Some(pos) = rest.find("```") {
        let before = &rest[..pos];
        result.push_str(before);
        if !result.is_empty() && !result.ends_with('\n') {
            result.push('\n');
        }
        result.push_str("```");
        rest = &rest[pos + 3..];

        // Consume optional language tag (alphanumeric, _, -, +, .)
        let lang_end = rest
            .find(|c: char| {
                !c.is_ascii_alphanumeric() && c != '_' && c != '-' && c != '+' && c != '.'
            })
            .unwrap_or(rest.len());
        if lang_end > 0 {
            result.push_str(&rest[..lang_end]);
            rest = &rest[lang_end..];
        }
        if !rest.is_empty() && !rest.starts_with('\n') {
            result.push('\n');
        }
    }
    result.push_str(rest);
    result
}

/// Render text content with basic markdown code block detection: fenced
/// blocks get a distinct style, their lines are truncated rather than wrapped,
/// and normal text is word-wrapped.
fn render_markdown_content(
    text: &str,
    wrap_width: usize,
    lines: &mut Vec<Line<'static>>,
    text_style: Style,
    code_style: Style,
    fence_style: Style,
) {
    let normalized = normalize_code_fences(text);
    let mut in_code_block = false;
    let code_inner_width = wrap_width.saturating_sub(4);

    for line in normalized.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            if !in_code_block {
                let lang = trimmed.strip_prefix("```").unwrap_or("").trim();
                let label = if lang.is_empty() {
                    "  ``` ".to_string()
                } else {
                    format!("  ```{lang} ")
                };
                lines.push(Line::from(Span::styled(label, fence_style)));
                in_code_block = true;
            } else {
                lines.push(Line::from(Span::styled("  ```", fence_style)));
                in_code_block = false;
            }
            continue;
        }
        if in_code_block {
            let display: String = line.chars().take(code_inner_width).collect();
            lines.push(Line::from(Span::styled(
                format!("    {display}"),
                code_style,
            )));
        } else if trimmed.is_empty() {
            lines.push(Line::from(Span::styled("  ", text_style)));
        } else {
            for w in wrap_text(line, wrap_width) {
                lines.push(Line::from(Span::styled(format!("  {w}"), text_style)));
            }
        }
    }
}

/// Simple word wrap that breaks at word boundaries. Width is counted in
/// characters; a word longer than the line is broken.
fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 || text.chars().count() <= max_width {
        return vec![text.to_string()];
    }
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;
    for word in text.split_whitespace() {
        let word_width = word.chars().count();
        if !current.is_empty() && current_width + 1 + word_width <= max_width {
            current.push(' ');
            current.push_str(word);
            current_width += 1 + word_width;
            continue;
        }
        if !current.is_empty() {
            lines.push(std::mem::take(&mut current));
            current_width = 0;
        }
        if word_width <= max_width {
            current = word.to_string();
            current_width = word_width;
        } else {
            // Break a long word at character boundaries.
            let chars: Vec<char> = word.chars().collect();
            let mut pieces = chars
                .chunks(max_width)
                .map(|c| c.iter().collect::<String>());
            let last = pieces.next_back();
            lines.extend(pieces);
            if let Some(last) = last {
                current_width = last.chars().count();
                current = last;
            }
        }
    }
    if !current.is_empty() || lines.is_empty() {
        lines.push(current);
    }
    lines
}

impl Widget for &mut InferenceTab {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if self.sidebar_visible && area.width > 50 {
            let sidebar_width = if self.focus == InferenceFocus::Settings {
                34
            } else {
                30
            };
            let [chat_area, sidebar_area] =
                Layout::horizontal([Constraint::Fill(1), Constraint::Length(sidebar_width)])
                    .areas(area);
            let [messages_area, input_area] =
                Layout::vertical([Constraint::Fill(1), Constraint::Length(3)]).areas(chat_area);
            self.render_messages(messages_area, buf);
            self.render_input(input_area, buf);
            self.render_settings(sidebar_area, buf);
        } else {
            let [messages_area, input_area] =
                Layout::vertical([Constraint::Fill(1), Constraint::Length(3)]).areas(area);
            self.render_messages(messages_area, buf);
            self.render_input(input_area, buf);
        }
    }
}

impl InferenceTab {
    fn render_messages(&mut self, area: Rect, buf: &mut Buffer) {
        let browsing = self.focus == InferenceFocus::Browse;
        let title = match (&self.model_id, browsing) {
            (Some(id), true) => format!(" Chat -- {id} [BROWSE] "),
            (Some(id), false) => format!(" Chat -- {id} "),
            (None, _) => " Chat (Ctrl+P to select a model) ".to_string(),
        };
        let block = Block::default()
            .title(title)
            .title_style(if browsing {
                THEME.block_title_focused
            } else {
                THEME.block_title
            })
            .borders(Borders::ALL)
            .border_style(if browsing {
                THEME.block_focused
            } else {
                THEME.block
            });
        let inner = block.inner(area);
        block.render(area, buf);

        if self.messages.is_empty() {
            let help = vec![
                Line::from(""),
                Line::from(Span::styled("  No messages yet.", THEME.text_muted)),
                Line::from(""),
                Line::from(Span::styled(
                    "  1. Select a model with Ctrl+P",
                    THEME.text_dim,
                )),
                Line::from(Span::styled(
                    "  2. Type a message and press Enter",
                    THEME.text_dim,
                )),
                Line::from(""),
                Line::from(Span::styled(
                    "  Ctrl+S: settings  Ctrl+H: toggle sidebar  F2: browse/yank",
                    THEME.text_muted,
                )),
            ];
            Paragraph::new(help).render(inner, buf);
            return;
        }

        let chat_lines = Self::build_chat_lines(
            &self.messages,
            self.generating,
            inner.width,
            self.selected_message,
        );
        let total = chat_lines.len();
        let height = inner.height as usize;
        let max_scroll = total.saturating_sub(height);
        self.message_scroll = self.message_scroll.min(max_scroll);

        // Only the visible rows are handed to the widget: it scrolls by a u16,
        // which a long conversation outgrows.
        let visible: Vec<Line<'static>> = chat_lines
            .into_iter()
            .skip(self.message_scroll)
            .take(height)
            .collect();
        Paragraph::new(visible).render(inner, buf);

        if total > height {
            let mut scrollbar_state = ScrollbarState::new(max_scroll).position(self.message_scroll);
            Scrollbar::new(ScrollbarOrientation::VerticalRight).render(
                inner,
                buf,
                &mut scrollbar_state,
            );
        }
    }

    fn render_input(&self, area: Rect, buf: &mut Buffer) {
        let focused = self.focus == InferenceFocus::Input;
        let title = if self.generating {
            " Generating... (Esc to stop) "
        } else if self.focus == InferenceFocus::Browse {
            " Input (Esc to exit browse) "
        } else {
            " Input "
        };
        let block = Block::default()
            .title(title)
            .title_style(if focused {
                THEME.block_title_focused
            } else {
                THEME.block_title
            })
            .borders(Borders::ALL)
            .border_style(if focused {
                THEME.block_focused
            } else {
                THEME.block
            });

        let (text, style) = if self.input_buffer.is_empty() && !self.generating {
            ("Type a message...".to_string(), THEME.text_muted)
        } else if focused {
            // A cursor, so the person can see where typing goes.
            let mut shown = self.input_buffer.clone();
            shown.insert(self.cursor_position, '▏');
            (shown, THEME.text)
        } else {
            (self.input_buffer.clone(), THEME.text)
        };
        // One line of input: show the end of it, where typing is.
        let width = area.width.saturating_sub(2) as usize;
        let chars: Vec<char> = text.chars().filter(|c| *c != '\n').collect();
        let skip = chars.len().saturating_sub(width);
        let shown: String = chars[skip..].iter().collect();
        Paragraph::new(shown)
            .style(style)
            .block(block)
            .render(area, buf);
    }

    fn render_settings(&self, area: Rect, buf: &mut Buffer) {
        let focused = self.focus == InferenceFocus::Settings;
        let block = Block::default()
            .title(" Settings ")
            .title_style(if focused {
                THEME.block_title_focused
            } else {
                THEME.block_title
            })
            .borders(Borders::ALL)
            .border_style(if focused {
                THEME.block_focused
            } else {
                THEME.block
            });
        let inner = block.inner(area);
        block.render(area, buf);

        let model_label = self.model_id.as_deref().unwrap_or("(none)");
        let max_width = inner.width.saturating_sub(10) as usize;
        let chars: Vec<char> = model_label.chars().collect();
        let model_text = if chars.len() > max_width && max_width > 2 {
            let tail: String = chars[chars.len() - (max_width - 2)..].iter().collect();
            format!("..{tail}")
        } else {
            model_label.to_string()
        };
        let mut lines: Vec<Line<'_>> = vec![
            Line::from(vec![
                Span::styled("  Model: ", THEME.kv_key),
                Span::styled(model_text, THEME.kv_value),
            ]),
            Line::from(""),
        ];

        for (i, setting) in SETTINGS.iter().enumerate() {
            let selected = focused && i == self.settings_selected;
            let value_style = if selected {
                THEME.table_selected
            } else {
                THEME.text
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{:>11}:  ", setting.label()), THEME.kv_key),
                Span::styled(self.setting_text(*setting), value_style),
                Span::styled(if selected { " [-/+, d]" } else { "" }, THEME.text_muted),
            ]));
        }
        let backend = InferenceBackend::ALL
            .iter()
            .find(|b| b.as_str() == self.spec.backend)
            .copied()
            .unwrap_or(InferenceBackend::Auto);
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("  ", THEME.text_muted),
            Span::styled(backend.label(), THEME.kv_value),
            Span::styled(" — ", THEME.text_muted),
            Span::styled(backend.description(), THEME.text_muted),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            if focused {
                " Ctrl+S: back to input"
            } else {
                " Ctrl+S: edit settings"
            },
            THEME.text_muted,
        )));
        lines.push(Line::from(Span::styled(
            " Ctrl+H: hide sidebar",
            THEME.text_muted,
        )));

        Paragraph::new(lines).render(inner, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::cli_check::assert_parses;
    use crate::tui::jobs::LaunchPlan;
    use pmetal_data::stream_format::{ANSWER_LABEL, THINKING_LABEL};

    fn plan_of(tab: &InferenceTab) -> LaunchPlan {
        let spec = tab.build_infer_spec("/models/m".into(), "hello".into(), None);
        LaunchPlan {
            subcommand: "infer".into(),
            argv: pmetal_core::JobFields::to_argv(&spec),
            deltas: true,
            ..LaunchPlan::default()
        }
    }

    fn screen(tab: &mut InferenceTab, w: u16, h: u16) -> String {
        let area = Rect::new(0, 0, w, h);
        let mut buf = Buffer::empty(area);
        tab.render(area, &mut buf);
        (0..h)
            .map(|y| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn a_fresh_chat_passes_nothing_the_model_decides() {
        let tab = InferenceTab::new();
        let plan = plan_of(&tab);
        // Temperature, sampling and the token limit are the model's own until
        // someone sets them.
        for flag in [
            "--temperature",
            "--top-k",
            "--top-p",
            "--min-p",
            "--repetition-penalty",
            "--frequency-penalty",
            "--presence-penalty",
            "--max-tokens",
            "--seed",
            "--reasoning-effort",
            "--kv-quant",
        ] {
            assert!(
                !plan.argv.contains(&flag.to_string()),
                "{flag} in {:?}",
                plan.argv
            );
        }
        assert!(plan.argv.contains(&"--chat".to_string()));
        assert_parses(&plan);
    }

    #[test]
    fn a_launched_chat_is_accepted_by_the_cli_with_every_setting_touched() {
        // The old command line ended in a `--show-thinking` the CLI no longer
        // has, so no message could be sent at all.
        let mut tab = InferenceTab::new();
        for i in 0..SETTINGS.len() {
            tab.settings_selected = i;
            tab.increment_setting();
        }
        tab.spec.reasoning_effort = Some("high".into());
        let plan = plan_of(&tab);
        assert!(!plan.argv.contains(&"--show-thinking".to_string()));
        assert_parses(&plan);
    }

    #[test]
    fn zero_temperature_is_greedy_decoding_and_is_passed() {
        // The old builder turned temperature 0 into "not set".
        let mut tab = InferenceTab::new();
        tab.settings_selected = 0;
        tab.spec.temperature = Some(0.1);
        tab.decrement_setting();
        assert_eq!(tab.spec.temperature, Some(0.0));
        let plan = plan_of(&tab);
        let at = plan.argv.iter().position(|a| a == "--temperature").unwrap();
        assert_eq!(plan.argv[at + 1], "0");
        assert_parses(&plan);
    }

    #[test]
    fn settings_step_within_the_specs_bounds_and_start_from_a_sensible_value() {
        let mut tab = InferenceTab::new();
        tab.settings_selected = 0;
        tab.increment_setting();
        assert_eq!(tab.spec.temperature, Some(0.7));
        for _ in 0..100 {
            tab.increment_setting();
        }
        let (_, max) = bounds("temperature");
        assert_eq!(tab.spec.temperature, Some(max as f32));
        // `d` goes back to the model's own.
        tab.reset_setting();
        assert_eq!(tab.spec.temperature, None);
        // The penalties may be negative where the spec allows it.
        tab.settings_selected = 6; // frequency penalty
        tab.spec.frequency_penalty = Some(0.0);
        tab.decrement_setting();
        assert_eq!(tab.spec.frequency_penalty, Some(-0.05));
        // Steps do not drift: seven 0.1 steps from 0.7 are 1.4.
        tab.settings_selected = 0;
        tab.spec.temperature = Some(0.7);
        for _ in 0..7 {
            tab.increment_setting();
        }
        assert_eq!(tab.spec.temperature, Some(1.4));
    }

    #[test]
    fn the_kv_cache_choice_stands_for_the_specs_fields() {
        let mut tab = InferenceTab::new();
        tab.settings_selected = 9;
        let mut seen = Vec::new();
        for _ in 0..KvMode::ALL.len() {
            tab.increment_setting();
            seen.push(KvMode::of(&tab.spec));
        }
        // A full circle, each mode distinct, ending where it started.
        assert_eq!(seen.last(), Some(&KvMode::Auto));
        let mut distinct = seen.clone();
        distinct.dedup();
        assert_eq!(distinct.len(), KvMode::ALL.len());
        KvMode::Fp16.apply(&mut tab.spec);
        assert!(tab.spec.no_kv_quant && tab.spec.kv_quant.is_none());
        KvMode::TurboQuant4.apply(&mut tab.spec);
        assert!(!tab.spec.no_kv_quant && tab.spec.kv_turboquant && tab.spec.kv_quant == Some(4));
        assert_parses(&plan_of(&tab));
    }

    #[test]
    fn a_reply_streams_in_and_is_settled_from_the_whole_output() {
        let mut tab = InferenceTab::new();
        tab.add_user_message("Capital of France?");
        tab.start_generation("j1".into());
        tab.receive("========\nGenerating...\nThe cap");
        assert_eq!(tab.messages.last().unwrap().content, "The cap");
        // A piece that never arrived is made up from the job's whole output.
        let whole = "========\nGenerating...\nThe capital is Paris.\n\n---\nGenerated 6 tokens in 0.50s\nDecode: 200 tok/s (x)\n";
        let stats = tab
            .finish_generation(whole)
            .expect("the footer was reached");
        assert_eq!(stats.tokens, 6);
        let reply = tab.messages.last().unwrap();
        assert_eq!(reply.content, "The capital is Paris.");
        assert_eq!((reply.tokens, reply.tok_sec), (Some(6), Some(200.0)));
        assert!(!tab.generating && tab.job.is_none());
    }

    #[test]
    fn a_generation_that_produced_nothing_leaves_no_empty_reply() {
        let mut tab = InferenceTab::new();
        tab.add_user_message("hi");
        tab.start_generation("j".into());
        assert!(tab.finish_generation("error: model not found\n").is_none());
        tab.discard_empty_reply();
        assert_eq!(tab.messages.len(), 1);
    }

    #[test]
    fn reasoning_is_shown_apart_from_the_answer_and_copy_takes_only_the_answer() {
        let mut tab = InferenceTab::new();
        tab.messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content: format!("{THINKING_LABEL}count the letters{ANSWER_LABEL}Three"),
            tokens: None,
            tok_sec: None,
        });
        let text = screen(&mut tab, 100, 24);
        assert!(
            text.contains("--- thinking ---") && text.contains("count the letters"),
            "{text}"
        );
        assert!(
            text.contains("--- response ---") && text.contains("Three"),
            "{text}"
        );
        tab.enter_browse_mode();
        assert_eq!(tab.yank_selected().as_deref(), Some("Three"));
    }

    #[test]
    fn the_models_own_defaults_are_shown_until_a_setting_is_touched() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"temperature":0.6,"top_k":20,"top_p":0.95,"max_new_tokens":4096}"#,
        )
        .unwrap();
        let mut tab = InferenceTab::new();
        tab.load_model_settings(dir.path());
        tab.model_id = Some("m".into());
        let text = screen(&mut tab, 120, 30);
        assert!(text.contains("model (0.6)"), "{text}");
        assert!(text.contains("model (4096)"), "{text}");
        // Loading them did not turn them into arguments.
        assert_eq!(tab.spec.temperature, None);
        assert_eq!(tab.spec.max_tokens, None);
    }

    #[test]
    fn the_input_line_edits_multibyte_text() {
        let mut tab = InferenceTab::new();
        for c in "héllo".chars() {
            tab.insert_char(c);
        }
        tab.move_cursor_left();
        tab.move_cursor_left();
        tab.delete_char();
        tab.move_cursor_right();
        assert_eq!(tab.input(), "hélo");
        tab.paste(" wörld\r\nline");
        assert_eq!(tab.input(), "hél wörld\nlineo");
        assert_eq!(tab.take_input().chars().next(), Some('h'));
        assert_eq!(tab.input(), "");
    }

    #[test]
    fn wrapping_breaks_long_words_and_keeps_short_ones_whole() {
        assert_eq!(wrap_text("aaa bbb ccc", 7), ["aaa bbb", "ccc"]);
        assert_eq!(wrap_text("abcdefghij", 4), ["abcd", "efgh", "ij"]);
        assert_eq!(
            wrap_text("x abcdefghij y", 4),
            ["x", "abcd", "efgh", "ij y"]
        );
        assert_eq!(wrap_text("", 5), [""]);
    }

    #[test]
    fn the_chat_draws_at_any_size_and_with_a_long_conversation() {
        let mut tab = InferenceTab::new();
        tab.model_id = Some("org/a-very-long-model-name-for-the-sidebar".into());
        for n in 0..300 {
            tab.add_user_message(&format!("question {n} {}", "word ".repeat(30)));
            tab.messages.push(ChatMessage {
                role: ChatRole::Assistant,
                content: format!(
                    "answer {n}\n```rust\nlet x = {n};\n```\n{}",
                    "text ".repeat(40)
                ),
                tokens: Some(10),
                tok_sec: Some(5.0),
            });
        }
        for &(w, h) in &[(0, 0), (1, 1), (10, 4), (40, 10), (80, 24), (200, 60)] {
            screen(&mut tab, w, h);
            tab.focus = InferenceFocus::Settings;
            screen(&mut tab, w, h);
            tab.focus = InferenceFocus::Input;
        }
        // Scrolling past the start or the end is held at the ends.
        for _ in 0..1000 {
            tab.scroll_up();
        }
        screen(&mut tab, 80, 24);
        for _ in 0..100_000 {
            tab.scroll_down();
        }
        screen(&mut tab, 80, 24);
    }
}
