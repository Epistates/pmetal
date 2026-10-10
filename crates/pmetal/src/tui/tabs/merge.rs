//! Merge tab — blend two (or more) models via SLERP / TIES / DARE /
//! linear / task-arithmetic. Mirrors `pmetal merge`.
//!
//! Fields driven by [`MergeSpec`]. Preserves conditional visibility
//! (SLERP/Linear/TIES/DARE sections) and method-specific knobs.

use std::path::PathBuf;

use pmetal_core::JobFields as _;
use pmetal_core::jobs::MergeSpec;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

use crate::tui::theme::THEME;
use crate::tui::widgets::{FormAction, FormField, FormTabState, JobLog, StatusTone, status_line};

/// Runtime status of the `pmetal merge` subprocess.
#[derive(Debug, Clone, Default)]
pub enum MergeStatus {
    #[default]
    Idle,
    Running,
    Completed {
        output: String,
    },
    Failed(String),
}

/// Merge tab state.
pub struct MergeTab {
    pub form: FormTabState,
    pub status: MergeStatus,
    pub log: JobLog,
}

impl MergeTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::from_spec_default::<MergeSpec>(),
            status: MergeStatus::Idle,
            log: JobLog::with_default_cap(),
        }
    }

    /// Hide method-specific fields that don't apply to the current method.
    /// MergeSpec field labels: "SLERP t", "Weight A", "Weight B", "Density",
    /// "Base Model". Groups: "Models", "Method", "Output".
    fn field_visible(method: &str, f: &FormField) -> bool {
        let uses_t = method == "slerp";
        let uses_weights = matches!(method, "linear" | "ties" | "dare_linear" | "dare_ties");
        let uses_density = matches!(
            method,
            "ties" | "dare_ties" | "dare_linear" | "della" | "breadcrumbs"
        );
        let uses_base = matches!(
            method,
            "ties" | "dare_ties" | "dare_linear" | "task_arithmetic" | "della"
        );
        match f.label.as_str() {
            "SLERP t" => uses_t,
            "Weight A" | "Weight B" => uses_weights,
            "Density" => uses_density,
            "Base Model" => uses_base,
            _ => true,
        }
    }

    fn method_snapshot(&self) -> String {
        self.form.value("Method")
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
    pub fn handle_enter(&mut self) -> Option<FormAction> {
        self.form.handle_enter()
    }

    pub fn next_param(&mut self) {
        let method = self.method_snapshot();
        self.form
            .next_param(move |f| Self::field_visible(&method, f));
    }

    pub fn prev_param(&mut self) {
        let method = self.method_snapshot();
        self.form
            .prev_param(move |f| Self::field_visible(&method, f));
    }

    // ── Setters ─────────────────────────────────────────────────────────

    pub fn set_model(&mut self, label: &str, model_id: &str) {
        self.form.set_value(label, model_id);
        // Auto-derive output dir from Model A's short name when it's
        // the first slot populated. MergeSpec uses "Output Dir" field.
        if label == "Model A" {
            let short = super::model_short_name(model_id);
            let current = self.form.value("Output Dir");
            if current.is_empty() || current == "./output/merged" {
                self.form
                    .set_value("Output Dir", format!("./output/{short}--merged"));
            }
        }
    }

    pub fn is_running(&self) -> bool {
        matches!(self.status, MergeStatus::Running)
    }

    pub fn mark_running(&mut self) {
        self.log.clear();
        self.status = MergeStatus::Running;
    }

    pub fn mark_completed(&mut self) {
        self.status = MergeStatus::Completed {
            output: self.form.value("Output Dir"),
        };
    }

    pub fn mark_failed(&mut self, msg: &str) {
        self.status = MergeStatus::Failed(msg.to_string());
    }

    pub fn append_log(&mut self, line: &str) {
        self.log.push(line);
    }

    // ── Config ──────────────────────────────────────────────────────────

    pub fn validate_config(&self) -> Result<(), String> {
        let a = self.form.value("Model A");
        if a.is_empty() || a == "(not selected)" {
            return Err("Model A is required.".into());
        }
        let b = self.form.value("Model B");
        if b.is_empty() || b == "(not selected)" {
            return Err("Model B is required.".into());
        }
        let method = self.form.value("Method");
        let needs_base = matches!(
            method.as_str(),
            "ties" | "dare_ties" | "dare_linear" | "task_arithmetic" | "della"
        );
        let base = self.form.value("Base Model");
        if needs_base && (base.is_empty() || base == "(not selected)") {
            return Err(format!(
                "Method `{method}` requires a base model for task-vector computation."
            ));
        }
        if self.form.value("Output Dir").is_empty() {
            return Err("Output Dir is required.".into());
        }
        Ok(())
    }

    pub fn output_dir(&self) -> PathBuf {
        PathBuf::from(self.form.value("Output Dir"))
    }

    pub fn config_summary(&self) -> Vec<String> {
        let method = self.form.value("Method");
        let mut summary = vec![
            format!("Model A: {}", self.form.value("Model A")),
            format!("Model B: {}", self.form.value("Model B")),
            format!("Method:  {method}"),
            format!("Output:  {}", self.form.value("Output Dir")),
        ];
        let base = self.form.value("Base Model");
        if !base.is_empty() && base != "(not selected)" {
            summary.push(format!("Base:    {base}"));
        }
        summary.push(String::new());
        summary.push("Proceed?".into());
        summary
    }

    /// Build CLI args from the form via [`MergeSpec::to_argv`].
    pub fn build_cli_args(&self) -> Vec<String> {
        let spec = self.spec_from_form();
        let mut args = vec!["merge".to_string()];
        args.extend(spec.to_argv());
        args
    }

    fn spec_from_form(&self) -> MergeSpec {
        let mut spec = MergeSpec::default();
        spec.model_a = self.form.value("Model A");
        spec.model_b = self.form.value("Model B");
        spec.output = self.form.value("Output Dir");
        spec.method = {
            let v = self.form.value("Method");
            if v.is_empty() { spec.method } else { v }
        };
        spec.dtype = {
            let v = self.form.value("Output Dtype");
            if v.is_empty() { spec.dtype } else { v }
        };
        let base = self.form.value("Base Model");
        spec.base = if base.is_empty() || base == "(not selected)" {
            None
        } else {
            Some(base)
        };
        spec.t = self.form.value("SLERP t").parse().unwrap_or(spec.t);
        spec.weight_a = self.form.value("Weight A").parse().unwrap_or(spec.weight_a);
        spec.weight_b = self.form.value("Weight B").parse().unwrap_or(spec.weight_b);
        spec.density = self.form.value("Density").parse().unwrap_or(spec.density);
        spec
    }
}

// ── Rendering ──────────────────────────────────────────────────────────

impl MergeTab {
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let [config_area, right_area] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .areas(area);

        let method = self.method_snapshot();
        self.form
            .render_list(config_area, buf, "Merge Configuration", move |f| {
                Self::field_visible(&method, f)
            });

        let [status_area, log_area] =
            Layout::vertical([Constraint::Length(8), Constraint::Min(0)]).areas(right_area);
        self.render_status(status_area, buf);
        self.log.render(log_area, buf, "Merge Log");
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
            MergeStatus::Idle => {
                lines.push(status_line(StatusTone::Idle, "Idle", None));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "  [S] Start merge  [x] Cancel",
                    THEME.text_muted,
                )));
            }
            MergeStatus::Running => {
                lines.push(status_line(StatusTone::Running, "Merging", None));
            }
            MergeStatus::Completed { output } => {
                lines.push(status_line(StatusTone::Completed, "Completed", None));
                lines.push(Line::from(vec![
                    Span::styled("  Output ", THEME.kv_key),
                    Span::styled(output.clone(), THEME.kv_value),
                ]));
            }
            MergeStatus::Failed(msg) => {
                lines.push(status_line(StatusTone::Failed, "Failed", Some(msg)));
            }
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}
