//! Tokenize tab — tokenize a text corpus into binary shards for pretraining.
//!
//! Form driven by [`TokenizeSpec`]. Status + log panel on the right.

use pmetal_core::JobFields as _;
use pmetal_core::jobs::TokenizeSpec;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

use crate::tui::theme::THEME;
use crate::tui::widgets::{FormAction, FormTabState, JobLog, StatusTone, status_line};

/// Runtime status of the `pmetal tokenize` process.
#[derive(Debug, Clone, Default)]
pub enum TokenizeStatus {
    #[default]
    Idle,
    Running,
    Completed,
    Failed(String),
}

/// Tokenize tab state.
pub struct TokenizeTab {
    pub form: FormTabState,
    pub status: TokenizeStatus,
    pub log: JobLog,
}

impl TokenizeTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::from_spec_default::<TokenizeSpec>(),
            status: TokenizeStatus::Idle,
            log: JobLog::with_default_cap(),
        }
    }

    // ── Form delegation ─────────────────────────────────────────────────

    pub fn is_editing(&self) -> bool {
        self.form.is_editing()
    }
    pub fn handle_edit_key(&mut self, k: crossterm::event::KeyEvent) {
        self.form.handle_edit_key(k);
    }
    pub fn confirm_edit(&mut self) {
        self.form.confirm_edit();
    }
    pub fn cancel_edit(&mut self) {
        self.form.cancel_edit();
    }
    pub fn next_param(&mut self) {
        self.form.next_param(|_| true);
    }
    pub fn prev_param(&mut self) {
        self.form.prev_param(|_| true);
    }
    pub fn handle_enter(&mut self) -> Option<FormAction> {
        self.form.handle_enter()
    }

    // ── Setters ─────────────────────────────────────────────────────────

    pub fn set_tokenizer(&mut self, model_id: &str) {
        self.form.set_value("Tokenizer Model", model_id);
    }

    // ── State transitions ──────────────────────────────────────────────

    pub fn is_running(&self) -> bool {
        matches!(self.status, TokenizeStatus::Running)
    }

    pub fn mark_running(&mut self) {
        self.log.clear();
        self.status = TokenizeStatus::Running;
    }

    pub fn mark_completed(&mut self) {
        self.status = TokenizeStatus::Completed;
    }

    pub fn mark_failed(&mut self, msg: &str) {
        self.status = TokenizeStatus::Failed(msg.to_string());
    }

    pub fn append_log(&mut self, line: &str) {
        self.log.push(line);
    }

    // ── Config ──────────────────────────────────────────────────────────

    pub fn validate_config(&self) -> Result<(), String> {
        let input = self.form.value("Input JSONL");
        if input.is_empty() {
            return Err("Input JSONL path is required.".into());
        }
        let output = self.form.value("Output Dir");
        if output.is_empty() {
            return Err("Output Dir is required.".into());
        }
        let tokenizer = self.form.value("Tokenizer Model");
        if tokenizer.is_empty() || tokenizer == "(not selected)" {
            return Err("Tokenizer Model is required.".into());
        }
        Ok(())
    }

    pub fn config_summary(&self) -> Vec<String> {
        vec![
            format!("Input:      {}", self.form.value("Input JSONL")),
            format!("Output:     {}", self.form.value("Output Dir")),
            format!("Tokenizer:  {}", self.form.value("Tokenizer Model")),
            format!("Text Col:   {}", self.form.value("Text Column")),
            format!("Shard Size: {}", self.form.value("Docs Per Shard")),
            String::new(),
            "Run tokenize?".into(),
        ]
    }

    pub fn build_cli_args(&self) -> Vec<String> {
        let spec = self.spec_from_form();
        let mut args = vec!["tokenize".to_string()];
        args.extend(spec.to_argv());
        args
    }

    fn spec_from_form(&self) -> TokenizeSpec {
        let mut spec = TokenizeSpec::default();
        spec.input = self.form.value("Input JSONL");
        spec.output = self.form.value("Output Dir");
        spec.tokenizer = self.form.value("Tokenizer Model");
        spec.text_column = self.form.value("Text Column");
        spec.docs_per_shard = self
            .form
            .value("Docs Per Shard")
            .parse()
            .unwrap_or(spec.docs_per_shard);
        spec
    }
}

// ── Rendering ──────────────────────────────────────────────────────────

impl TokenizeTab {
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let [config_area, right_area] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .areas(area);

        self.form
            .render_list(config_area, buf, "Tokenize Configuration", |_| true);

        let [status_area, log_area] =
            Layout::vertical([Constraint::Length(7), Constraint::Min(0)]).areas(right_area);
        self.render_status(status_area, buf);
        self.log.render(log_area, buf, "Tokenize Log");
    }

    fn render_status(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Status ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines: Vec<Line<'_>> = Vec::new();
        match &self.status {
            TokenizeStatus::Idle => {
                lines.push(status_line(StatusTone::Idle, "Idle", None));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "  [S] Start  [x] Cancel",
                    THEME.text_muted,
                )));
            }
            TokenizeStatus::Running => {
                lines.push(status_line(StatusTone::Running, "Running", None));
            }
            TokenizeStatus::Completed => {
                lines.push(status_line(StatusTone::Completed, "Completed", None));
            }
            TokenizeStatus::Failed(msg) => {
                lines.push(status_line(StatusTone::Failed, "Failed", Some(msg)));
            }
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}
