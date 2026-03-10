//! Inference tab — interactive chat/completion with real model inference.

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
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
const SETTING_NAMES: &[&str] = &["Temperature", "Max Tokens", "Top-k", "Top-p"];

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
    pub input_focused: bool,
    pub settings_focused: bool,
    pub settings_selected: usize,
    message_scroll: usize,
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
            input_focused: true,
            settings_focused: false,
            settings_selected: 0,
            message_scroll: 0,
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
        let s = self.input_buffer.drain(..).collect();
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
        let max_tokens = json
            .get("max_new_tokens")
            .and_then(|v| v.as_u64())
            .or_else(|| json.get("max_length").and_then(|v| v.as_u64()));
        if let Some(v) = max_tokens {
            self.max_tokens = (v as usize).min(8192);
        }
    }

    pub fn toggle_settings_focus(&mut self) {
        self.settings_focused = !self.settings_focused;
        self.input_focused = !self.settings_focused;
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
            _ => {}
        }
    }

    pub fn decrement_setting(&mut self) {
        match self.settings_selected {
            0 => self.temperature = (self.temperature - 0.1).max(0.0),
            1 => self.max_tokens = self.max_tokens.saturating_sub(64).max(1),
            2 => self.top_k = self.top_k.saturating_sub(10),
            3 => self.top_p = (self.top_p - 0.05).max(0.0),
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

    /// Build the chat lines for rendering.
    fn build_chat_lines(
        messages: &[ChatMessage],
        generating: bool,
        wrap_width: u16,
    ) -> Vec<Line<'static>> {
        let mut lines: Vec<Line<'static>> = Vec::new();

        for msg in messages {
            // Role header
            let role_style = match msg.role {
                ChatRole::System => THEME.text_muted,
                ChatRole::User => THEME.text_success,
                ChatRole::Assistant => THEME.text_bright,
            };
            lines.push(Line::from(Span::styled(
                format!("[{}]", msg.role),
                role_style,
            )));

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

                // Render response content
                for line in response.lines() {
                    let wrapped = wrap_text(line, wrap_width.saturating_sub(4) as usize);
                    for w in wrapped {
                        lines.push(Line::from(Span::styled(format!("  {w}"), THEME.text)));
                    }
                }
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
fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 {
        return vec![text.to_string()];
    }
    if text.len() <= max_width {
        return vec![text.to_string()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();

    for word in text.split_whitespace() {
        if current.is_empty() {
            // If a single word exceeds max_width, force-break it
            if word.len() > max_width {
                let mut remaining = word;
                while remaining.len() > max_width {
                    lines.push(remaining[..max_width].to_string());
                    remaining = &remaining[max_width..];
                }
                current = remaining.to_string();
            } else {
                current = word.to_string();
            }
        } else if current.len() + 1 + word.len() > max_width {
            lines.push(current);
            if word.len() > max_width {
                let mut remaining = word;
                while remaining.len() > max_width {
                    lines.push(remaining[..max_width].to_string());
                    remaining = &remaining[max_width..];
                }
                current = remaining.to_string();
            } else {
                current = word.to_string();
            }
        } else {
            current.push(' ');
            current.push_str(word);
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

impl Widget for &mut InferenceTab {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Settings sidebar on the right, chat on the left
        let sidebar_width = if self.settings_focused { 28 } else { 24 };
        let [chat_area, sidebar_area] =
            Layout::horizontal([Constraint::Fill(1), Constraint::Length(sidebar_width)])
                .areas(area);

        let [messages_area, input_area] =
            Layout::vertical([Constraint::Fill(1), Constraint::Length(3)]).areas(chat_area);

        self.render_messages(messages_area, buf);
        self.render_input(input_area, buf);
        self.render_settings(sidebar_area, buf);
    }
}

impl InferenceTab {
    fn render_messages(&mut self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(match &self.model_id {
                Some(id) => format!(" Chat -- {id} "),
                None => " Chat (Ctrl+P to select model) ".to_string(),
            })
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
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
                    "  Ctrl+S: toggle settings  Ctrl+P: pick model",
                    THEME.text_muted,
                )),
                Line::from(Span::styled(
                    "  +/- to adjust settings, Esc to stop generation",
                    THEME.text_muted,
                )),
            ];
            Paragraph::new(help).render(inner, buf);
            return;
        }

        let chat_lines = Self::build_chat_lines(&self.messages, self.generating, inner.width);
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
        let border_style = if self.input_focused {
            THEME.block_focused
        } else {
            THEME.block
        };
        let title_style = if self.input_focused {
            THEME.block_title_focused
        } else {
            THEME.block_title
        };

        let block = Block::default()
            .title(if self.generating {
                " Generating... (Esc to stop) "
            } else {
                " Input "
            })
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
        let border_style = if self.settings_focused {
            THEME.block_focused
        } else {
            THEME.block
        };
        let block = Block::default()
            .title(" Settings ")
            .title_style(if self.settings_focused {
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
        let setting_values: [String; 4] = [
            format!("{:.1}", self.temperature),
            format!("{}", self.max_tokens),
            format!("{}", self.top_k),
            format!("{:.2}", self.top_p),
        ];

        for (i, (name, val)) in SETTING_NAMES.iter().zip(setting_values.iter()).enumerate() {
            let selected = self.settings_focused && i == self.settings_selected;
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
        if self.settings_focused {
            lines.push(Line::from(Span::styled(
                " Ctrl+S: back to input",
                THEME.text_muted,
            )));
        } else {
            lines.push(Line::from(Span::styled(
                " Ctrl+S: edit settings",
                THEME.text_muted,
            )));
        }

        Paragraph::new(lines).render(inner, buf);
    }
}
