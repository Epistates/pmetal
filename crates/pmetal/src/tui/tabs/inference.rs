//! Inference tab — interactive chat/completion with real model inference.

use pmetal_core::jobs::InferSpec;
use pmetal_data::inference_config::InferenceBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, StatefulWidget,
    Widget,
};

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

/// Inference settings that can be navigated with arrow keys.
///
/// Order here must match the index arms in `increment_setting` /
/// `decrement_setting` and the column layout in `render_settings`.
/// Backend is last so existing indices stay stable.
const SETTING_NAMES: &[&str] = &[
    "Temperature",
    "Max Tokens",
    "Top-k",
    "Top-p",
    "Min-p",
    "Rep Penalty",
    "Freq Penalty",
    "Pres Penalty",
    "Seed",
    "KV Cache",
    "FP8",
    "No Thinking",
    "Backend",
];

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

/// Inference tab state.
pub struct InferenceTab {
    pub model_id: Option<String>,
    pub messages: Vec<ChatMessage>,
    input_buffer: String,
    cursor_position: usize,
    pub generating: bool,
    // Generation settings
    pub temperature: f32,
    pub max_tokens: usize,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: Option<u64>,
    /// KV cache quantization mode: 0=auto, 8=q8, 4=q4, 255=fp16, 108=tq8, 104=tq4, 125=tq2.5, 135=tq3.5.
    pub kv_quant_mode: u8,
    pub fp8: bool,
    pub no_thinking: bool,
    /// Execution backend (Auto lets `pmetal infer` pick the fastest path).
    pub backend: InferenceBackend,
    pub focus: InferenceFocus,
    pub settings_selected: usize,
    message_scroll: usize,
    /// Currently selected message index in Browse mode.
    pub selected_message: Option<usize>,
    /// Whether the sidebar is visible.
    pub sidebar_visible: bool,
    /// Path to packed expert weights directory for SSD-offloaded MoE inference.
    pub experts_dir: Option<String>,
}

// Keep backwards-compat public fields as computed properties
impl InferenceTab {
    pub fn input_focused(&self) -> bool {
        self.focus == InferenceFocus::Input
    }

    pub fn settings_focused(&self) -> bool {
        self.focus == InferenceFocus::Settings
    }
}

impl InferenceTab {
    pub fn new() -> Self {
        Self {
            model_id: None,
            messages: Vec::new(),
            input_buffer: String::new(),
            cursor_position: 0,
            generating: false,
            temperature: 0.7,
            max_tokens: 2048,
            top_k: 50,
            top_p: 0.9,
            min_p: 0.0,
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
            kv_quant_mode: 0, // auto
            fp8: false,
            no_thinking: false,
            backend: InferenceBackend::Auto,
            focus: InferenceFocus::Input,
            settings_selected: 0,
            message_scroll: 0,
            selected_message: None,
            sidebar_visible: true,
            experts_dir: None,
        }
    }

    pub fn insert_char(&mut self, c: char) {
        self.input_buffer.insert(self.cursor_position, c);
        self.cursor_position += c.len_utf8();
    }

    pub fn delete_char(&mut self) {
        if self.cursor_position > 0 {
            let prev = self.input_buffer[..self.cursor_position]
                .chars()
                .last()
                .map(|c| c.len_utf8())
                .unwrap_or(0);
            self.cursor_position -= prev;
            self.input_buffer.remove(self.cursor_position);
        }
    }

    pub fn move_cursor_left(&mut self) {
        if self.cursor_position > 0 {
            let prev = self.input_buffer[..self.cursor_position]
                .chars()
                .last()
                .map(|c| c.len_utf8())
                .unwrap_or(0);
            self.cursor_position -= prev;
        }
    }

    pub fn move_cursor_right(&mut self) {
        if self.cursor_position < self.input_buffer.len() {
            let next = self.input_buffer[self.cursor_position..]
                .chars()
                .next()
                .map(|c| c.len_utf8())
                .unwrap_or(0);
            self.cursor_position += next;
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

    /// Start an assistant message (placeholder for streaming).
    pub fn start_generation(&mut self) {
        self.generating = true;
        self.messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content: String::new(),
            tokens: None,
            tok_sec: None,
        });
    }

    /// Append a token to the current assistant message (streaming).
    pub fn append_token(&mut self, token: &str) {
        if let Some(msg) = self.messages.last_mut() {
            if msg.role == ChatRole::Assistant {
                msg.content.push_str(token);
                self.scroll_to_bottom();
            }
        }
    }

    /// Mark generation as complete with performance stats.
    pub fn finish_generation(&mut self, tok_sec: f64, total_tokens: usize) {
        self.generating = false;
        if let Some(msg) = self.messages.last_mut() {
            if msg.role == ChatRole::Assistant {
                msg.tokens = Some(total_tokens);
                msg.tok_sec = Some(tok_sec);
            }
        }
    }

    /// Submit the message (legacy — now the app handles this).
    pub fn submit_message(&mut self) {
        // No-op: real submission handled by App::submit_inference
    }

    /// Load generation defaults from a model's `generation_config.json`.
    pub fn load_model_settings(&mut self, model_path: &std::path::Path) {
        let config_path = model_path.join("generation_config.json");
        let Ok(content) = std::fs::read_to_string(&config_path) else {
            return;
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) else {
            return;
        };

        if let Some(v) = json.get("temperature").and_then(|v| v.as_f64()) {
            self.temperature = v as f32;
        }
        if let Some(v) = json.get("top_k").and_then(|v| v.as_u64()) {
            self.top_k = v as usize;
        }
        if let Some(v) = json.get("top_p").and_then(|v| v.as_f64()) {
            self.top_p = v as f32;
        }
        if let Some(v) = json.get("min_p").and_then(|v| v.as_f64()) {
            self.min_p = v as f32;
        }
        if let Some(v) = json.get("repetition_penalty").and_then(|v| v.as_f64()) {
            self.repetition_penalty = v as f32;
        }
        let max_tokens = json
            .get("max_new_tokens")
            .and_then(|v| v.as_u64())
            .or_else(|| json.get("max_length").and_then(|v| v.as_u64()));
        if let Some(v) = max_tokens {
            self.max_tokens = (v as usize).min(8192);
        }
    }

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

    /// Enter browse mode (F2 or Ctrl+B): hide sidebar, enable message selection.
    pub fn enter_browse_mode(&mut self) {
        if self.messages.is_empty() {
            return;
        }
        self.focus = InferenceFocus::Browse;
        self.sidebar_visible = false;
        // Start at the last message
        self.selected_message = Some(self.messages.len() - 1);
    }

    /// Exit browse mode back to input.
    pub fn exit_browse_mode(&mut self) {
        self.focus = InferenceFocus::Input;
        self.selected_message = None;
    }

    /// Select next message (down in browse mode).
    pub fn next_message(&mut self) {
        if let Some(idx) = self.selected_message {
            if idx + 1 < self.messages.len() {
                self.selected_message = Some(idx + 1);
            }
        }
    }

    /// Select previous message (up in browse mode).
    pub fn prev_message(&mut self) {
        if let Some(idx) = self.selected_message {
            if idx > 0 {
                self.selected_message = Some(idx - 1);
            }
        }
    }

    /// Get the content of the currently selected message (for yanking).
    pub fn selected_message_content(&self) -> Option<&str> {
        self.selected_message
            .and_then(|idx| self.messages.get(idx))
            .map(|msg| {
                // For assistant messages, extract just the response (strip thinking)
                msg.content.as_str()
            })
    }

    /// Yank (copy) the selected message content to the system clipboard.
    pub fn yank_selected(&self) -> Option<String> {
        let content = self.selected_message_content()?;

        // For assistant messages, extract the response part (strip thinking blocks)
        let idx = self.selected_message?;
        let msg = &self.messages[idx];
        let text = if msg.role == ChatRole::Assistant {
            let (_thinking, response) = parse_thinking(content);
            response.to_string()
        } else {
            content.to_string()
        };

        Some(text)
    }

    pub fn next_setting(&mut self) {
        self.settings_selected = (self.settings_selected + 1) % SETTING_NAMES.len();
    }

    pub fn prev_setting(&mut self) {
        self.settings_selected =
            (self.settings_selected + SETTING_NAMES.len() - 1) % SETTING_NAMES.len();
    }

    pub fn increment_setting(&mut self) {
        match self.settings_selected {
            0 => self.temperature = (self.temperature + 0.1).min(2.0),
            1 => self.max_tokens = (self.max_tokens + 64).min(8192),
            2 => self.top_k = (self.top_k + 10).min(500),
            3 => self.top_p = (self.top_p + 0.05).min(1.0),
            4 => self.min_p = (self.min_p + 0.01).min(1.0),
            5 => self.repetition_penalty = (self.repetition_penalty + 0.05).min(3.0),
            6 => self.frequency_penalty = (self.frequency_penalty + 0.05).min(2.0),
            7 => self.presence_penalty = (self.presence_penalty + 0.05).min(2.0),
            8 => self.seed = Some(self.seed.map_or(42, |s| s.saturating_add(1))),
            9 => {
                self.kv_quant_mode = match self.kv_quant_mode {
                    0 => 8,
                    8 => 4,
                    4 => 108,   // TQ8
                    108 => 104, // TQ4
                    104 => 125, // TQ2.5
                    125 => 135, // TQ3.5
                    135 => 255, // FP16
                    _ => 0,
                }
            }
            10 => self.fp8 = !self.fp8,
            11 => self.no_thinking = !self.no_thinking,
            12 => self.backend = cycle_backend(self.backend, 1),
            _ => {}
        }
    }

    pub fn decrement_setting(&mut self) {
        match self.settings_selected {
            0 => self.temperature = (self.temperature - 0.1).max(0.0),
            1 => self.max_tokens = self.max_tokens.saturating_sub(64).max(1),
            2 => self.top_k = self.top_k.saturating_sub(10),
            3 => self.top_p = (self.top_p - 0.05).max(0.0),
            4 => self.min_p = (self.min_p - 0.01).max(0.0),
            5 => self.repetition_penalty = (self.repetition_penalty - 0.05).max(1.0),
            6 => self.frequency_penalty = (self.frequency_penalty - 0.05).max(0.0),
            7 => self.presence_penalty = (self.presence_penalty - 0.05).max(0.0),
            8 => {
                self.seed = match self.seed {
                    Some(0) | None => None,
                    Some(s) => Some(s.saturating_sub(1)),
                };
            }
            9 => {
                self.kv_quant_mode = match self.kv_quant_mode {
                    0 => 255,
                    255 => 135,
                    135 => 125,
                    125 => 104,
                    104 => 108,
                    108 => 4,
                    4 => 8,
                    _ => 0,
                }
            }
            10 => self.fp8 = !self.fp8,
            11 => self.no_thinking = !self.no_thinking,
            12 => self.backend = cycle_backend(self.backend, -1),
            _ => {}
        }
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

    /// Build the chat lines for rendering, with optional message highlighting.
    fn build_chat_lines(
        messages: &[ChatMessage],
        generating: bool,
        wrap_width: u16,
        selected_message: Option<usize>,
    ) -> Vec<Line<'static>> {
        let mut lines: Vec<Line<'static>> = Vec::new();
        let code_style = Style::default()
            .fg(Color::Rgb(190, 200, 220))
            .bg(Color::Rgb(30, 41, 59));
        let code_fence_style = Style::default().fg(Color::Rgb(100, 116, 139));

        for (msg_idx, msg) in messages.iter().enumerate() {
            let is_selected = selected_message == Some(msg_idx);

            // Role header
            let role_style = if is_selected {
                Style::default()
                    .fg(match msg.role {
                        ChatRole::System => Color::Rgb(100, 116, 139),
                        ChatRole::User => Color::Rgb(34, 197, 94),
                        ChatRole::Assistant => Color::Rgb(248, 250, 252),
                    })
                    .add_modifier(Modifier::BOLD | Modifier::REVERSED)
            } else {
                match msg.role {
                    ChatRole::System => THEME.text_muted,
                    ChatRole::User => THEME.text_success,
                    ChatRole::Assistant => THEME.text_bright,
                }
            };

            let header = if is_selected {
                format!(" [{}] (y to copy) ", msg.role)
            } else {
                format!("[{}]", msg.role)
            };
            lines.push(Line::from(Span::styled(header, role_style)));

            if msg.content.is_empty() && msg.role == ChatRole::Assistant {
                // Streaming placeholder
                if generating {
                    lines.push(Line::from(Span::styled("  ...", THEME.text_muted)));
                }
            } else if msg.role == ChatRole::Assistant {
                // Parse thinking content from assistant messages
                let (thinking, response) = parse_thinking(&msg.content);

                if let Some(think_text) = thinking {
                    // Render thinking block with distinct style
                    lines.push(Line::from(Span::styled(
                        "  --- thinking ---".to_string(),
                        THEME.text_muted,
                    )));
                    for line in think_text.lines() {
                        let wrapped = wrap_text(line, wrap_width.saturating_sub(4) as usize);
                        for w in wrapped {
                            lines.push(Line::from(Span::styled(format!("  {w}"), THEME.text_dim)));
                        }
                    }
                    lines.push(Line::from(Span::styled(
                        "  --- response ---".to_string(),
                        THEME.text_muted,
                    )));
                }

                // Render response content with code block detection
                render_markdown_content(
                    response,
                    wrap_width.saturating_sub(4) as usize,
                    &mut lines,
                    THEME.text,
                    code_style,
                    code_fence_style,
                );
                // If still generating and response is empty, show cursor
                if response.is_empty() && generating && thinking.is_some() {
                    lines.push(Line::from(Span::styled(
                        "  ...".to_string(),
                        THEME.text_muted,
                    )));
                }
            } else {
                // User/system messages — wrap normally
                for line in msg.content.lines() {
                    let wrapped = wrap_text(line, wrap_width.saturating_sub(4) as usize);
                    for w in wrapped {
                        lines.push(Line::from(Span::styled(format!("  {w}"), THEME.text)));
                    }
                }
            }

            // Performance stats
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

impl InferenceTab {
    /// Build an [`InferSpec`] from the current sidebar settings.
    ///
    /// The caller is responsible for `model`, `prompt`, `lora`, and any
    /// TUI-only flags (`--show-thinking`) that are not part of `InferSpec`.
    ///
    /// # Spec gaps (documented here)
    ///
    /// `InferSpec` has no single `kv_quant_mode: u8` field — it instead has
    /// separate `kv_quant`, `kv_turboquant`, and `kv_quant_preset` fields.
    /// The mapping from the sidebar's combined mode byte is tab-local and
    /// cannot be expressed via a spec round-trip, so we perform it here.
    ///
    /// `InferSpec` has no `--show-thinking` flag (TUI-only); the caller
    /// appends it directly to argv after calling `spec.to_argv()`.
    pub fn build_infer_spec(
        &self,
        model: String,
        prompt: String,
        lora: Option<String>,
    ) -> InferSpec {
        // Resolve optional sampling params (only set when non-default).
        let temperature = if self.temperature > 0.0 {
            Some(self.temperature)
        } else {
            None
        };
        let top_k = if self.top_k > 0 {
            Some(self.top_k)
        } else {
            None
        };
        let top_p = if self.top_p < 1.0 {
            Some(self.top_p)
        } else {
            None
        };
        let min_p = if self.min_p > 0.0 {
            Some(self.min_p)
        } else {
            None
        };
        let repetition_penalty = if self.repetition_penalty > 1.0 {
            Some(self.repetition_penalty)
        } else {
            None
        };
        let frequency_penalty = if self.frequency_penalty > 0.0 {
            Some(self.frequency_penalty)
        } else {
            None
        };
        let presence_penalty = if self.presence_penalty > 0.0 {
            Some(self.presence_penalty)
        } else {
            None
        };

        // KV quant mode — map sidebar's combined byte to spec fields.
        // 0=auto (omit), 255=fp16 (disable), 4/8=Q4/Q8, 104/108=TQ4/TQ8, 125=TQ2.5, 135=TQ3.5
        let (kv_quant, kv_turboquant, kv_quant_preset, no_kv_quant) = match self.kv_quant_mode {
            255 => (None, false, None, true),
            bits @ (4 | 8) => (Some(bits), false, None, false),
            108 => (Some(8), true, None, false),
            104 => (Some(4), true, None, false),
            125 => (None, false, Some("q2_5".to_string()), false),
            135 => (None, false, Some("q3_5".to_string()), false),
            _ => (None, false, None, false), // 0 = auto
        };

        let backend = if self.backend != InferenceBackend::Auto {
            self.backend.as_str().to_string()
        } else {
            "auto".to_string()
        };

        InferSpec {
            model,
            prompt,
            lora,
            max_tokens: self.max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            seed: self.seed,
            chat: true,
            fp8: self.fp8,
            no_thinking: self.no_thinking,
            backend,
            kv_quant,
            kv_turboquant,
            kv_quant_preset,
            no_kv_quant,
            experts_dir: self.experts_dir.clone(),
            ..InferSpec::default()
        }
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
        // Insert \n before ``` if preceded by non-newline content
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
        // Insert \n after fence+lang if not already followed by newline/EOF
        if !rest.is_empty() && !rest.starts_with('\n') {
            result.push('\n');
        }
    }
    result.push_str(rest);
    result
}

/// Render text content with basic markdown code block detection.
///
/// Detects ``` fenced code blocks and renders them with a distinct background style.
/// Code lines are indented and truncated (not wrapped). Normal text is word-wrapped.
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
    let mut code_lang = String::new();
    let code_inner_width = wrap_width.saturating_sub(4); // account for 4-space indent

    for line in normalized.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            if !in_code_block {
                // Opening fence
                code_lang = trimmed.strip_prefix("```").unwrap_or("").trim().to_string();
                let label = if code_lang.is_empty() {
                    "  ``` ".to_string()
                } else {
                    format!("  ```{code_lang} ")
                };
                lines.push(Line::from(Span::styled(label, fence_style)));
                in_code_block = true;
            } else {
                // Closing fence
                lines.push(Line::from(Span::styled("  ```", fence_style)));
                in_code_block = false;
                code_lang.clear();
            }
            continue;
        }

        if in_code_block {
            // Code lines: truncate to fit (no word wrap), indent with 4 spaces
            let char_count = line.chars().count();
            let display: String = if char_count > code_inner_width {
                line.chars().take(code_inner_width).collect()
            } else {
                line.to_string()
            };
            lines.push(Line::from(Span::styled(
                format!("    {display}"),
                code_style,
            )));
        } else {
            // Normal text: word wrap
            if trimmed.is_empty() {
                lines.push(Line::from(Span::styled("  ", text_style)));
            } else {
                let wrapped = wrap_text(line, wrap_width);
                for w in wrapped {
                    lines.push(Line::from(Span::styled(format!("  {w}"), text_style)));
                }
            }
        }
    }

    // If we ended inside a code block (incomplete generation), that's fine
}

/// Parse `<think>...</think>` content from assistant response.
/// Returns (thinking_content, response_content).
fn parse_thinking(text: &str) -> (Option<&str>, &str) {
    // Check for "=== Thinking ===" / "=== Response ===" format (from --show-thinking)
    if let Some(think_start) = text.find("=== Thinking ===") {
        let after_header = &text[think_start + "=== Thinking ===".len()..];
        if let Some(resp_start) = after_header.find("=== Response ===") {
            let thinking = after_header[..resp_start].trim();
            let response = after_header[resp_start + "=== Response ===".len()..].trim();
            return (
                if thinking.is_empty() {
                    None
                } else {
                    Some(thinking)
                },
                response,
            );
        }
    }

    // Check for raw <think>...</think> tags
    if let Some(think_start) = text.find("<think>") {
        let content_start = think_start + "<think>".len();
        if let Some(think_end) = text.find("</think>") {
            let thinking = text[content_start..think_end].trim();
            let response = text[think_end + "</think>".len()..].trim();
            return (
                if thinking.is_empty() {
                    None
                } else {
                    Some(thinking)
                },
                response,
            );
        } else {
            // Incomplete thinking (still generating)
            let thinking = text[content_start..].trim();
            return (
                if thinking.is_empty() {
                    None
                } else {
                    Some(thinking)
                },
                "",
            );
        }
    }

    // No thinking content
    (None, text)
}

/// Simple word wrap that breaks at word boundaries.
/// Uses char-count width (safe for UTF-8).
fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 {
        return vec![text.to_string()];
    }
    // Use char count for width measurement (handles multi-byte UTF-8)
    let text_width: usize = text.chars().count();
    if text_width <= max_width {
        return vec![text.to_string()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_width: usize = 0;

    for word in text.split_whitespace() {
        let word_width = word.chars().count();
        if current.is_empty() {
            if word_width > max_width {
                // Force-break long word at char boundaries
                force_break_word(
                    word,
                    max_width,
                    &mut lines,
                    &mut current,
                    &mut current_width,
                );
            } else {
                current = word.to_string();
                current_width = word_width;
            }
        } else if current_width + 1 + word_width > max_width {
            lines.push(current);
            current = String::new();
            current_width = 0;
            if word_width > max_width {
                force_break_word(
                    word,
                    max_width,
                    &mut lines,
                    &mut current,
                    &mut current_width,
                );
            } else {
                current = word.to_string();
                current_width = word_width;
            }
        } else {
            current.push(' ');
            current.push_str(word);
            current_width += 1 + word_width;
        }
    }

    if !current.is_empty() {
        lines.push(current);
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}

/// Force-break a word that exceeds max_width at char boundaries.
fn force_break_word(
    word: &str,
    max_width: usize,
    lines: &mut Vec<String>,
    current: &mut String,
    current_width: &mut usize,
) {
    let mut chars = word.chars();
    let mut chunk = String::new();
    let mut chunk_len = 0;
    for ch in &mut chars {
        if chunk_len >= max_width {
            lines.push(chunk);
            chunk = String::new();
            chunk_len = 0;
        }
        chunk.push(ch);
        chunk_len += 1;
    }
    *current = chunk;
    *current_width = chunk_len;
}

impl Widget for &mut InferenceTab {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if self.sidebar_visible {
            let sidebar_width = if self.focus == InferenceFocus::Settings {
                28
            } else {
                24
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
            // No sidebar — full width for chat
            let [messages_area, input_area] =
                Layout::vertical([Constraint::Fill(1), Constraint::Length(3)]).areas(area);

            self.render_messages(messages_area, buf);
            self.render_input(input_area, buf);
        }
    }
}

impl InferenceTab {
    fn render_messages(&mut self, area: Rect, buf: &mut Buffer) {
        let title = match (&self.model_id, self.focus) {
            (Some(id), InferenceFocus::Browse) => format!(" Chat -- {id} [BROWSE] "),
            (Some(id), _) => format!(" Chat -- {id} "),
            (None, _) => " Chat (Ctrl+P to select model) ".to_string(),
        };
        let block = Block::default()
            .title(title)
            .title_style(if self.focus == InferenceFocus::Browse {
                THEME.block_title_focused
            } else {
                THEME.block_title
            })
            .borders(Borders::ALL)
            .border_style(if self.focus == InferenceFocus::Browse {
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
        let total_lines = chat_lines.len();
        let visible_height = inner.height as usize;

        // Clamp scroll
        let max_scroll = total_lines.saturating_sub(visible_height);
        if self.message_scroll > max_scroll {
            self.message_scroll = max_scroll;
        }

        Paragraph::new(chat_lines)
            .scroll((self.message_scroll as u16, 0))
            .render(inner, buf);

        // Scrollbar
        if total_lines > visible_height {
            let mut scrollbar_state = ScrollbarState::new(max_scroll).position(self.message_scroll);
            Scrollbar::new(ScrollbarOrientation::VerticalRight).render(
                inner,
                buf,
                &mut scrollbar_state,
            );
        }
    }

    fn render_input(&self, area: Rect, buf: &mut Buffer) {
        let is_focused = self.focus == InferenceFocus::Input;
        let border_style = if is_focused {
            THEME.block_focused
        } else {
            THEME.block
        };
        let title_style = if is_focused {
            THEME.block_title_focused
        } else {
            THEME.block_title
        };

        let title = if self.generating {
            " Generating... (Esc to stop) "
        } else if self.focus == InferenceFocus::Browse {
            " Input (Esc to exit browse) "
        } else {
            " Input "
        };

        let block = Block::default()
            .title(title)
            .title_style(title_style)
            .borders(Borders::ALL)
            .border_style(border_style);

        let display_text = if self.input_buffer.is_empty() && !self.generating {
            "Type a message..."
        } else {
            &self.input_buffer
        };

        let style = if self.input_buffer.is_empty() {
            THEME.text_muted
        } else {
            THEME.text
        };

        Paragraph::new(display_text)
            .style(style)
            .block(block)
            .render(area, buf);
    }

    fn render_settings(&self, area: Rect, buf: &mut Buffer) {
        let is_focused = self.focus == InferenceFocus::Settings;
        let border_style = if is_focused {
            THEME.block_focused
        } else {
            THEME.block
        };
        let block = Block::default()
            .title(" Settings ")
            .title_style(if is_focused {
                THEME.block_title_focused
            } else {
                THEME.block_title
            })
            .borders(Borders::ALL)
            .border_style(border_style);
        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines: Vec<Line> = Vec::new();

        // Model display — truncate to fit sidebar
        let model_label = self.model_id.as_deref().unwrap_or("(none)");
        let max_val_width = inner.width.saturating_sub(10) as usize;
        let truncated_model = if model_label.len() > max_val_width {
            format!(
                "..{}",
                &model_label[model_label.len() - max_val_width + 2..]
            )
        } else {
            model_label.to_string()
        };
        lines.push(Line::from(vec![
            Span::styled("  Model: ", THEME.kv_key),
            Span::styled(truncated_model, THEME.kv_value),
        ]));
        lines.push(Line::from(""));

        // Editable settings
        let kv_label = match self.kv_quant_mode {
            0 => "Auto".to_string(),
            255 => "FP16".to_string(),
            108 => "TQ8".to_string(),
            104 => "TQ4".to_string(),
            125 => "TQ2.5".to_string(),
            135 => "TQ3.5".to_string(),
            bits => format!("Q{bits}"),
        };
        let seed_label = match self.seed {
            Some(s) => format!("{s}"),
            None => "Random".to_string(),
        };
        let setting_values: [String; 13] = [
            format!("{:.1}", self.temperature),
            format!("{}", self.max_tokens),
            format!("{}", self.top_k),
            format!("{:.2}", self.top_p),
            format!("{:.2}", self.min_p),
            format!("{:.2}", self.repetition_penalty),
            format!("{:.2}", self.frequency_penalty),
            format!("{:.2}", self.presence_penalty),
            seed_label,
            kv_label,
            if self.fp8 {
                "On".to_string()
            } else {
                "Off".to_string()
            },
            if self.no_thinking {
                "On".to_string()
            } else {
                "Off".to_string()
            },
            self.backend.label().to_string(),
        ];

        for (i, (name, val)) in SETTING_NAMES.iter().zip(setting_values.iter()).enumerate() {
            let selected = is_focused && i == self.settings_selected;
            let style = if selected {
                THEME.table_selected
            } else {
                THEME.text
            };
            let hint = if selected { " [-/+]" } else { "" };
            lines.push(Line::from(vec![
                Span::styled(format!("{:>10}:  ", name), THEME.kv_key),
                Span::styled(val, style),
                Span::styled(hint, THEME.text_muted),
            ]));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("  ", THEME.text_muted),
            Span::styled(self.backend.label(), THEME.kv_value),
            Span::styled(" — ", THEME.text_muted),
            Span::styled(self.backend.description(), THEME.text_muted),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            if is_focused {
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

/// Cycle the inference backend dropdown. `step=1` advances, `step=-1`
/// goes backward. Wraps at both ends.
fn cycle_backend(current: InferenceBackend, step: i32) -> InferenceBackend {
    let all = InferenceBackend::ALL;
    let idx = all.iter().position(|b| *b == current).unwrap_or(0);
    let len = all.len() as i32;
    let next = ((idx as i32 + step).rem_euclid(len)) as usize;
    all[next]
}
