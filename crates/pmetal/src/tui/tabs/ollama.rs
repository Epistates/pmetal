//! Modelfile export tab.
//!
//! PMetal keeps this as a file export workflow: generate a Modelfile from a
//! PMetal model or adapter and let the user decide whether to register it with
//! an external runtime later.

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

use crate::tui::theme::THEME;
use crate::tui::widgets::{JobLog, StatusTone, status_line};

/// Runtime status of the `pmetal ollama modelfile` export process.
#[derive(Debug, Clone, Default)]
pub enum OllamaStatus {
    #[default]
    Idle,
    Running,
    Completed,
    Failed(String),
}

/// A minimal editable field for the Ollama tab.
#[derive(Debug, Clone)]
pub struct OllamaField {
    pub label: &'static str,
    pub value: String,
    pub editing: bool,
    pub edit_buf: String,
}

impl OllamaField {
    fn new(label: &'static str, default: &str) -> Self {
        Self {
            label,
            value: default.to_string(),
            editing: false,
            edit_buf: String::new(),
        }
    }
}

/// Ollama tab state.
pub struct OllamaTab {
    pub status: OllamaStatus,
    pub log: JobLog,
    /// Modelfile export fields.
    fields: [OllamaField; 10],
    selected: usize,
}

impl OllamaTab {
    pub fn new() -> Self {
        Self {
            status: OllamaStatus::Idle,
            log: JobLog::with_default_cap(),
            fields: [
                OllamaField::new("Base Model", ""),
                OllamaField::new("LoRA Adapter", ""),
                OllamaField::new("Output Path", "Modelfile"),
                OllamaField::new("Template", "auto"),
                OllamaField::new("System Prompt", ""),
                OllamaField::new("Temperature", ""),
                OllamaField::new("Num Context", ""),
                OllamaField::new("Top K", ""),
                OllamaField::new("Top P", ""),
                OllamaField::new("License", ""),
            ],
            selected: 0,
        }
    }

    // ── Accessors ────────────────────────────────────────────────────────

    pub fn is_editing(&self) -> bool {
        self.fields.iter().any(|f| f.editing)
    }

    pub fn next_param(&mut self) {
        if !self.is_editing() {
            self.selected = (self.selected + 1) % self.fields.len();
        }
    }

    pub fn prev_param(&mut self) {
        if !self.is_editing() {
            self.selected = (self.selected + self.fields.len() - 1) % self.fields.len();
        }
    }

    pub fn handle_enter(&mut self) {
        let f = &mut self.fields[self.selected];
        if f.editing {
            f.value = f.edit_buf.trim().to_string();
            f.editing = false;
        } else {
            f.edit_buf = f.value.clone();
            f.editing = true;
        }
    }

    pub fn cancel_edit(&mut self) {
        for f in &mut self.fields {
            f.editing = false;
        }
    }

    pub fn handle_edit_key(&mut self, k: crossterm::event::KeyEvent) {
        use crossterm::event::KeyCode;
        let f = &mut self.fields[self.selected];
        if !f.editing {
            return;
        }
        match k.code {
            KeyCode::Char(c) => f.edit_buf.push(c),
            KeyCode::Backspace => {
                f.edit_buf.pop();
            }
            _ => {}
        }
    }

    pub fn confirm_edit(&mut self) {
        let f = &mut self.fields[self.selected];
        f.value = f.edit_buf.trim().to_string();
        f.editing = false;
    }

    // ── State transitions ─────────────────────────────────────────────

    pub fn is_running(&self) -> bool {
        matches!(self.status, OllamaStatus::Running)
    }

    pub fn mark_running(&mut self) {
        self.log.clear();
        self.status = OllamaStatus::Running;
    }

    pub fn mark_completed(&mut self) {
        self.status = OllamaStatus::Completed;
    }

    pub fn mark_failed(&mut self, msg: &str) {
        self.status = OllamaStatus::Failed(msg.to_string());
    }

    pub fn append_log(&mut self, line: &str) {
        self.log.push(line);
    }

    // ── Config ───────────────────────────────────────────────────────────

    pub fn validate_config(&self) -> Result<(), String> {
        if self.fields[0].value.trim().is_empty() {
            return Err("Base Model is required.".into());
        }
        if self.fields[2].value.trim().is_empty() {
            return Err("Output Path is required.".into());
        }
        Ok(())
    }

    pub fn config_summary(&self) -> Vec<String> {
        vec![
            format!("Base:     {}", self.fields[0].value),
            format!("Adapter:  {}", display_optional(&self.fields[1].value)),
            format!("Output:   {}", self.fields[2].value),
            format!("Template: {}", self.fields[3].value),
            String::new(),
            "Generate Modelfile?".into(),
        ]
    }

    /// Build CLI args for `pmetal ollama modelfile`.
    pub fn build_cli_args(&self) -> Vec<String> {
        let mut args = vec![
            "ollama".to_string(),
            "modelfile".to_string(),
            "--base".to_string(),
            self.fields[0].value.trim().to_string(),
            "--output".to_string(),
            self.fields[2].value.trim().to_string(),
        ];
        if !self.fields[1].value.trim().is_empty() {
            args.extend([
                "--lora".to_string(),
                self.fields[1].value.trim().to_string(),
            ]);
        }
        let template = self.fields[3].value.trim();
        if !template.is_empty() && template != "auto" {
            args.extend(["--template".to_string(), template.to_string()]);
        }
        if !self.fields[4].value.trim().is_empty() {
            args.extend([
                "--system".to_string(),
                self.fields[4].value.trim().to_string(),
            ]);
        }
        if !self.fields[5].value.trim().is_empty() {
            args.extend([
                "--temperature".to_string(),
                self.fields[5].value.trim().to_string(),
            ]);
        }
        if !self.fields[6].value.trim().is_empty() {
            args.extend([
                "--num-ctx".to_string(),
                self.fields[6].value.trim().to_string(),
            ]);
        }
        if !self.fields[7].value.trim().is_empty() {
            args.extend([
                "--top-k".to_string(),
                self.fields[7].value.trim().to_string(),
            ]);
        }
        if !self.fields[8].value.trim().is_empty() {
            args.extend([
                "--top-p".to_string(),
                self.fields[8].value.trim().to_string(),
            ]);
        }
        if !self.fields[9].value.trim().is_empty() {
            args.extend([
                "--license".to_string(),
                self.fields[9].value.trim().to_string(),
            ]);
        }
        args
    }
}

fn display_optional(value: &str) -> &str {
    if value.trim().is_empty() {
        "(none)"
    } else {
        value
    }
}

// ── Rendering ──────────────────────────────────────────────────────────

impl OllamaTab {
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let [config_area, right_area] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .areas(area);

        self.render_form(config_area, buf);

        let [status_area, log_area] =
            Layout::vertical([Constraint::Length(7), Constraint::Min(0)]).areas(right_area);
        self.render_status(status_area, buf);
        self.log.render(log_area, buf, "Ollama Log");
    }

    fn render_form(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Modelfile Export ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines: Vec<Line<'_>> = Vec::new();
        lines.push(Line::from(""));

        for (i, field) in self.fields.iter().enumerate() {
            let is_sel = i == self.selected;
            let label_style = if is_sel {
                THEME.kv_key
            } else {
                THEME.text_muted
            };
            let val_style = if is_sel {
                THEME.table_selected
            } else {
                THEME.kv_value
            };

            let display_val = if field.editing {
                format!("{}_", field.edit_buf)
            } else {
                field.value.clone()
            };

            lines.push(Line::from(vec![
                Span::styled(format!("  {:>12}:  ", field.label), label_style),
                Span::styled(display_val, val_style),
            ]));
            lines.push(Line::from(""));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  Templates: auto | llama3 | qwen3 | gemma | mistral | phi3 | deep-seek",
            THEME.text_dim,
        )));
        lines.push(Line::from(Span::styled(
            "  Enter to edit  S to export  x to cancel",
            THEME.text_muted,
        )));

        Paragraph::new(lines).render(inner, buf);
    }

    fn render_status(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Export Status ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines: Vec<Line<'_>> = Vec::new();
        match &self.status {
            OllamaStatus::Idle => {
                lines.push(status_line(StatusTone::Idle, "Idle", None));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "  [S] Export  [x] Cancel",
                    THEME.text_muted,
                )));
            }
            OllamaStatus::Running => {
                lines.push(status_line(StatusTone::Running, "Running", None));
            }
            OllamaStatus::Completed => {
                lines.push(status_line(StatusTone::Completed, "Completed", None));
            }
            OllamaStatus::Failed(msg) => {
                lines.push(status_line(StatusTone::Failed, "Failed", Some(msg)));
            }
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}
