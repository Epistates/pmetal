//! DFlash tab — block-diffusion speculative decoding configuration and control.
//!
//! Mirrors `pmetal dflash`. Requires a target model + draft model + prompt.
//! Fields driven by [`DflashSpec`]. Status + log panel on the right.

use pmetal_core::JobFields as _;
use pmetal_core::jobs::DflashSpec;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

use crate::tui::theme::THEME;
use crate::tui::widgets::{FormAction, FormTabState, JobLog, StatusTone, status_line};

/// Runtime status of the `pmetal dflash` process.
#[derive(Debug, Clone, Default)]
pub enum DflashStatus {
    #[default]
    Idle,
    Running,
    Completed {
        tokens: usize,
    },
    Failed(String),
}

/// DFlash tab state.
pub struct DflashTab {
    pub form: FormTabState,
    pub status: DflashStatus,
    pub log: JobLog,
}

impl DflashTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::from_spec_default::<DflashSpec>(),
            status: DflashStatus::Idle,
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

    pub fn set_target_model(&mut self, model_id: &str) {
        self.form.set_value("Target Model", model_id);
    }

    pub fn set_draft_model(&mut self, model_id: &str) {
        self.form.set_value("Draft Model", model_id);
    }

    // ── State transitions ──────────────────────────────────────────────

    pub fn is_running(&self) -> bool {
        matches!(self.status, DflashStatus::Running)
    }

    pub fn mark_running(&mut self) {
        self.log.clear();
        self.status = DflashStatus::Running;
    }

    pub fn append_log(&mut self, line: &str) {
        // Parse token count from lines like "Generated N tokens".
        if let Some(n) = parse_token_count(line) {
            if let DflashStatus::Running = self.status {
                self.status = DflashStatus::Completed { tokens: n };
            }
        }
        self.log.push(line);
    }

    pub fn mark_completed(&mut self) {
        if matches!(self.status, DflashStatus::Running) {
            self.status = DflashStatus::Completed { tokens: 0 };
        }
    }

    pub fn mark_failed(&mut self, msg: &str) {
        self.status = DflashStatus::Failed(msg.to_string());
    }

    // ── Config ──────────────────────────────────────────────────────────

    pub fn validate_config(&self) -> Result<(), String> {
        let target = self.form.value("Target Model");
        if target.is_empty() || target == "(not selected)" {
            return Err("Target Model is required.".into());
        }
        let draft = self.form.value("Draft Model");
        if draft.is_empty() || draft == "(not selected)" {
            return Err("Draft Model is required.".into());
        }
        if self.form.value("Prompt").is_empty() {
            return Err("Prompt is required.".into());
        }
        Ok(())
    }

    pub fn config_summary(&self) -> Vec<String> {
        vec![
            format!("Target:  {}", self.form.value("Target Model")),
            format!("Draft:   {}", self.form.value("Draft Model")),
            format!("Tokens:  {}", self.form.value("Max New Tokens")),
            format!("Temp:    {}", self.form.value("Temperature")),
            String::new(),
            "Run dflash?".into(),
        ]
    }

    /// Build CLI args from the form via [`DflashSpec::to_argv`].
    pub fn build_cli_args(&self) -> Vec<String> {
        let spec = self.spec_from_form();
        let mut args = vec!["dflash".to_string()];
        args.extend(spec.to_argv());
        args
    }

    fn spec_from_form(&self) -> DflashSpec {
        let mut spec = DflashSpec::default();
        spec.target = self.form.value("Target Model");
        spec.draft = self.form.value("Draft Model");
        spec.prompt = self.form.value("Prompt");
        spec.max_new_tokens = self
            .form
            .value("Max New Tokens")
            .parse()
            .unwrap_or(spec.max_new_tokens);
        spec.temperature = self
            .form
            .value("Temperature")
            .parse()
            .unwrap_or(spec.temperature);
        spec.speculative_tokens = self.form.value("Speculative Tokens").parse().ok();
        spec.draft_fp8 = self.form.value("Draft FP8") == "Enabled";
        spec.json = self.form.value("JSON Output") == "Enabled";
        spec.no_chat = self.form.value("No Chat Template") == "Enabled";
        spec.tree_budget = self
            .form
            .value("Tree Budget")
            .parse()
            .unwrap_or(spec.tree_budget);
        spec.compare_greedy = self.form.value("Compare With Greedy") == "Enabled";
        spec
    }
}

// ── Rendering ──────────────────────────────────────────────────────────

impl DflashTab {
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let [config_area, right_area] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .areas(area);

        self.form
            .render_list(config_area, buf, "DFlash Configuration", |_| true);

        let [status_area, log_area] =
            Layout::vertical([Constraint::Length(7), Constraint::Min(0)]).areas(right_area);
        self.render_status(status_area, buf);
        self.log.render(log_area, buf, "DFlash Log");
    }

    fn render_status(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Status ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let detail_buf = match &self.status {
            DflashStatus::Completed { tokens } if *tokens > 0 => {
                Some(format!("{tokens} tokens generated"))
            }
            DflashStatus::Completed { .. } => Some("Done".to_string()),
            _ => None,
        };

        let mut lines: Vec<Line> = Vec::new();
        match &self.status {
            DflashStatus::Idle => {
                lines.push(status_line(StatusTone::Idle, "Idle", None));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "  [S] Run  [x] Cancel",
                    THEME.text_muted,
                )));
            }
            DflashStatus::Running => {
                lines.push(status_line(StatusTone::Running, "Running", None));
            }
            DflashStatus::Completed { .. } => {
                lines.push(status_line(
                    StatusTone::Completed,
                    "Completed",
                    detail_buf.as_deref(),
                ));
            }
            DflashStatus::Failed(msg) => {
                lines.push(status_line(StatusTone::Failed, "Failed", Some(msg)));
            }
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

fn parse_token_count(line: &str) -> Option<usize> {
    let lower = line.to_lowercase();
    let idx = lower.find("generated")?;
    let rest = &line[idx + "generated".len()..];
    let rest = rest.trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}
