//! Inference tab — interactive chat/completion.

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Widget, Wrap};

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
            max_tokens: 256,
            top_k: 50,
            top_p: 0.9,
            input_focused: true,
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

    pub fn submit_message(&mut self) {
        if self.input_buffer.trim().is_empty() || self.generating {
            return;
        }
        let content = self.input_buffer.drain(..).collect::<String>();
        self.cursor_position = 0;
        self.messages.push(ChatMessage {
            role: ChatRole::User,
            content,
            tokens: None,
            tok_sec: None,
        });
        // In a real implementation, this would trigger async generation
        // For now, add a placeholder response
        self.messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content: "(inference not connected — use pmetal infer from CLI)".to_string(),
            tokens: None,
            tok_sec: None,
        });
    }

}

impl Widget for &mut InferenceTab {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let [chat_area, sidebar_area] =
            Layout::horizontal([Constraint::Percentage(70), Constraint::Percentage(30)])
                .areas(area);

        let [messages_area, input_area] =
            Layout::vertical([Constraint::Fill(1), Constraint::Length(3)]).areas(chat_area);

        self.render_messages(messages_area, buf);
        self.render_input(input_area, buf);
        self.render_settings(sidebar_area, buf);
    }
}

impl InferenceTab {
    fn render_messages(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(match &self.model_id {
                Some(id) => format!(" Chat — {id} "),
                None => " Chat ".to_string(),
            })
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        if self.messages.is_empty() {
            let help = vec![
                Line::from(""),
                Line::from(Span::styled(
                    "  No messages yet.",
                    THEME.text_muted,
                )),
                Line::from(""),
                Line::from(Span::styled(
                    "  Type a message below and press Enter.",
                    THEME.text_dim,
                )),
                Line::from(""),
                Line::from(Span::styled(
                    "  Note: Interactive inference requires a loaded model.",
                    THEME.text_muted,
                )),
                Line::from(Span::styled(
                    "  For full inference, use: pmetal infer -m <model> -p <prompt>",
                    THEME.text_muted,
                )),
            ];
            Paragraph::new(help).render(inner, buf);
            return;
        }

        let items: Vec<ListItem> = self
            .messages
            .iter()
            .flat_map(|msg| {
                let role_style = match msg.role {
                    ChatRole::System => THEME.text_muted,
                    ChatRole::User => THEME.text_success,
                    ChatRole::Assistant => THEME.text_bright,
                };

                let mut lines = vec![ListItem::new(Line::from(vec![
                    Span::styled(format!("[{}] ", msg.role), role_style),
                ]))];

                for line in msg.content.lines() {
                    lines.push(ListItem::new(Line::from(Span::styled(
                        format!("  {line}"),
                        THEME.text,
                    ))));
                }

                if let Some(tok_sec) = msg.tok_sec {
                    lines.push(ListItem::new(Line::from(Span::styled(
                        format!(
                            "  ({} tokens, {:.0} tok/s)",
                            msg.tokens.unwrap_or(0),
                            tok_sec
                        ),
                        THEME.text_muted,
                    ))));
                }

                lines.push(ListItem::new(Line::from("")));
                lines
            })
            .collect();

        List::new(items).render(inner, buf);
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
        let block = Block::default()
            .title(" Settings ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let lines = vec![
            Line::from(vec![
                Span::styled("Model:  ", THEME.kv_key),
                Span::styled(
                    self.model_id.as_deref().unwrap_or("(none)"),
                    THEME.kv_value,
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Temp:   ", THEME.kv_key),
                Span::styled(format!("{:.1}", self.temperature), THEME.kv_value),
            ]),
            Line::from(vec![
                Span::styled("Max tok:", THEME.kv_key),
                Span::styled(format!(" {}", self.max_tokens), THEME.kv_value),
            ]),
            Line::from(vec![
                Span::styled("Top-k:  ", THEME.kv_key),
                Span::styled(format!("{}", self.top_k), THEME.kv_value),
            ]),
            Line::from(vec![
                Span::styled("Top-p:  ", THEME.kv_key),
                Span::styled(format!("{:.1}", self.top_p), THEME.kv_value),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "Use pmetal infer for",
                THEME.text_muted,
            )),
            Line::from(Span::styled(
                "full inference with",
                THEME.text_muted,
            )),
            Line::from(Span::styled(
                "loaded models.",
                THEME.text_muted,
            )),
        ];

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}
