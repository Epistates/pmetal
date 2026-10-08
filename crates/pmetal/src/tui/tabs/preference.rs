//! Preference tab — DPO, IPO, hinge, SimPO, ORPO and KTO with LoRA.
//!
//! Form driven by [`PreferenceSpec`]. Status + log panel on the right.

use pmetal_core::JobFields as _;
use pmetal_core::jobs::PreferenceSpec;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

use crate::tui::theme::THEME;
use crate::tui::widgets::{FormAction, FormTabState, JobLog, StatusTone, status_line};

/// Runtime status of the `pmetal preference` process.
#[derive(Debug, Clone, Default)]
pub enum PreferenceStatus {
    #[default]
    Idle,
    Running,
    Completed,
    Failed(String),
}

/// Preference tab state.
pub struct PreferenceTab {
    pub form: FormTabState,
    pub status: PreferenceStatus,
    pub log: JobLog,
}

impl PreferenceTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::from_spec_default::<PreferenceSpec>(),
            status: PreferenceStatus::Idle,
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

    pub fn set_model(&mut self, model_id: &str) {
        self.form.set_value("Model", model_id);
    }

    pub fn set_dataset(&mut self, path: &str) {
        self.form.set_value("Dataset", path);
    }

    // ── State transitions ──────────────────────────────────────────────

    pub fn is_running(&self) -> bool {
        matches!(self.status, PreferenceStatus::Running)
    }

    pub fn mark_running(&mut self) {
        self.log.clear();
        self.status = PreferenceStatus::Running;
    }

    pub fn mark_completed(&mut self) {
        self.status = PreferenceStatus::Completed;
    }

    pub fn mark_failed(&mut self, msg: &str) {
        self.status = PreferenceStatus::Failed(msg.to_string());
    }

    pub fn append_log(&mut self, line: &str) {
        self.log.push(line);
    }

    // ── Config ──────────────────────────────────────────────────────────

    pub fn validate_config(&self) -> Result<(), String> {
        for field in ["Model", "Dataset"] {
            let value = self.form.value(field);
            if value.is_empty() || value == "(not selected)" {
                return Err(format!("{field} is required."));
            }
        }
        let mut spec = self.spec_from_form();
        spec.normalize().map_err(|errs| {
            errs.iter()
                .map(|e| e.message.as_str())
                .collect::<Vec<_>>()
                .join("; ")
        })
    }

    pub fn config_summary(&self) -> Vec<String> {
        let spec = self.spec_from_form();
        vec![
            format!("Model:    {}", spec.model),
            format!("Dataset:  {}", spec.dataset),
            format!("Loss:     {} (beta {})", spec.loss, spec.effective_beta()),
            format!("LR:       {:.1e}", spec.learning_rate),
            format!(
                "Batch:    {} x {} accumulation",
                spec.batch_size, spec.gradient_accumulation_steps
            ),
            String::new(),
            "Run preference optimization?".into(),
        ]
    }

    pub fn build_cli_args(&self) -> Vec<String> {
        let spec = self.spec_from_form();
        let mut args = vec!["preference".to_string()];
        args.extend(spec.to_argv());
        args
    }

    fn spec_from_form(&self) -> PreferenceSpec {
        let mut spec = PreferenceSpec::default();
        let value = |label: &str| self.form.value(label);
        spec.model = value("Model");
        spec.dataset = value("Dataset");
        spec.output_dir = value("Output Dir");
        spec.loss = value("Loss");
        spec.beta = value("β").parse().ok();
        spec.simpo_gamma_ratio = value("SimPO γ/β").parse().unwrap_or(spec.simpo_gamma_ratio);
        spec.label_smoothing = value("Label Smoothing")
            .parse()
            .unwrap_or(spec.label_smoothing);
        spec.desirable_weight = value("Desirable Weight")
            .parse()
            .unwrap_or(spec.desirable_weight);
        spec.undesirable_weight = value("Undesirable Weight")
            .parse()
            .unwrap_or(spec.undesirable_weight);
        spec.learning_rate = value("Learning Rate").parse().unwrap_or(spec.learning_rate);
        spec.batch_size = value("Batch Size").parse().unwrap_or(spec.batch_size);
        spec.gradient_accumulation_steps = value("Gradient Accumulation")
            .parse()
            .unwrap_or(spec.gradient_accumulation_steps);
        spec.epochs = value("Epochs").parse().unwrap_or(spec.epochs);
        spec.max_steps = value("Max Steps").parse().ok();
        spec.warmup_ratio = value("Warmup Ratio").parse().unwrap_or(spec.warmup_ratio);
        spec.max_grad_norm = value("Max Grad Norm").parse().unwrap_or(spec.max_grad_norm);
        spec.weight_decay = value("Weight Decay").parse().unwrap_or(spec.weight_decay);
        spec.lora_r = value("LoRA r").parse().unwrap_or(spec.lora_r);
        spec.lora_alpha = value("LoRA α").parse().unwrap_or(spec.lora_alpha);
        spec.max_prompt_length = value("Max Prompt Length")
            .parse()
            .unwrap_or(spec.max_prompt_length);
        spec.max_length = value("Max Length").parse().unwrap_or(spec.max_length);
        spec.seed = value("Seed").parse().unwrap_or(spec.seed);
        let metrics = value("Log Metrics Path");
        spec.log_metrics = (!metrics.is_empty()).then_some(metrics);
        spec
    }
}

// ── Rendering ──────────────────────────────────────────────────────────

impl PreferenceTab {
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let [config_area, right_area] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .areas(area);

        self.form
            .render_list(config_area, buf, "Preference Configuration", |_| true);

        let [status_area, log_area] =
            Layout::vertical([Constraint::Length(7), Constraint::Min(0)]).areas(right_area);
        self.render_status(status_area, buf);
        self.log.render(log_area, buf, "Preference Log");
    }

    fn render_status(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Status ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines: Vec<Line> = Vec::new();
        match &self.status {
            PreferenceStatus::Idle => {
                lines.push(status_line(StatusTone::Idle, "Idle", None));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "  [S] Start  [x] Cancel",
                    THEME.text_muted,
                )));
            }
            PreferenceStatus::Running => {
                lines.push(status_line(StatusTone::Running, "Running", None));
            }
            PreferenceStatus::Completed => {
                lines.push(status_line(StatusTone::Completed, "Completed", None));
            }
            PreferenceStatus::Failed(msg) => {
                lines.push(status_line(StatusTone::Failed, "Failed", Some(msg)));
            }
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The form round-trips the spec's defaults into the same argv the CLI
    /// would build from them.
    #[test]
    fn default_form_builds_the_default_argv() {
        let mut tab = PreferenceTab::new();
        tab.set_model("Qwen/Qwen3-0.6B");
        tab.set_dataset("pairs.jsonl");
        let spec = PreferenceSpec {
            model: "Qwen/Qwen3-0.6B".into(),
            dataset: "pairs.jsonl".into(),
            ..Default::default()
        };
        let mut expected = vec!["preference".to_string()];
        expected.extend(spec.to_argv());
        assert_eq!(tab.build_cli_args(), expected);
        assert!(tab.validate_config().is_ok());
    }
}
