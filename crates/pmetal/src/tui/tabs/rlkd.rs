//! RLKD tab — Reinforcement Learning with Knowledge Distillation.
//!
//! Form driven by [`RlkdSpec`]. Status + log panel on the right.

use pmetal_core::JobFields as _;
use pmetal_core::jobs::RlkdSpec;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

use crate::tui::theme::THEME;
use crate::tui::widgets::{FormAction, FormTabState, JobLog, StatusTone, status_line};

/// Runtime status of the `pmetal rlkd` process.
#[derive(Debug, Clone, Default)]
pub enum RlkdStatus {
    #[default]
    Idle,
    Running,
    Completed,
    Failed(String),
}

/// RLKD tab state.
pub struct RlkdTab {
    pub form: FormTabState,
    pub status: RlkdStatus,
    pub log: JobLog,
}

impl RlkdTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::from_spec_default::<RlkdSpec>(),
            status: RlkdStatus::Idle,
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

    pub fn set_policy_model(&mut self, model_id: &str) {
        self.form.set_value("Policy Model", model_id);
    }

    pub fn set_teacher_model(&mut self, model_id: &str) {
        self.form.set_value("Teacher Model", model_id);
    }

    pub fn set_dataset(&mut self, path: &str) {
        self.form.set_value("Dataset", path);
    }

    // ── State transitions ──────────────────────────────────────────────

    pub fn is_running(&self) -> bool {
        matches!(self.status, RlkdStatus::Running)
    }

    pub fn mark_running(&mut self) {
        self.log.clear();
        self.status = RlkdStatus::Running;
    }

    pub fn mark_completed(&mut self) {
        self.status = RlkdStatus::Completed;
    }

    pub fn mark_failed(&mut self, msg: &str) {
        self.status = RlkdStatus::Failed(msg.to_string());
    }

    pub fn append_log(&mut self, line: &str) {
        self.log.push(line);
    }

    // ── Config ──────────────────────────────────────────────────────────

    pub fn validate_config(&self) -> Result<(), String> {
        let model = self.form.value("Policy Model");
        if model.is_empty() || model == "(not selected)" {
            return Err("Policy Model is required.".into());
        }
        let teacher = self.form.value("Teacher Model");
        if teacher.is_empty() || teacher == "(not selected)" {
            return Err("Teacher Model is required.".into());
        }
        let dataset = self.form.value("Dataset");
        if dataset.is_empty() || dataset == "(not selected)" {
            return Err("Dataset is required.".into());
        }
        Ok(())
    }

    pub fn config_summary(&self) -> Vec<String> {
        vec![
            format!("Policy:   {}", self.form.value("Policy Model")),
            format!("Teacher:  {}", self.form.value("Teacher Model")),
            format!("Dataset:  {}", self.form.value("Dataset")),
            format!("Epochs:   {}", self.form.value("Epochs")),
            format!("Distill α:{}", self.form.value("Distill α (start)")),
            String::new(),
            "Run rlkd?".into(),
        ]
    }

    pub fn build_cli_args(&self) -> Vec<String> {
        let spec = self.spec_from_form();
        let mut args = vec!["rlkd".to_string()];
        args.extend(spec.to_argv());
        args
    }

    fn spec_from_form(&self) -> RlkdSpec {
        let mut spec = RlkdSpec::default();
        spec.model = self.form.value("Policy Model");
        spec.teacher_model = self.form.value("Teacher Model");
        spec.dataset = self.form.value("Dataset");
        spec.output_dir = self.form.value("Output Dir");
        spec.distill_alpha = self
            .form
            .value("Distill α (start)")
            .parse()
            .unwrap_or(spec.distill_alpha);
        spec.final_alpha = self
            .form
            .value("Distill α (final)")
            .parse()
            .unwrap_or(spec.final_alpha);
        spec.anneal_alpha = self.form.value("Anneal α") == "Enabled";
        spec.distill_temperature = self
            .form
            .value("Distill Temperature")
            .parse()
            .unwrap_or(spec.distill_temperature);
        spec.num_generations = self
            .form
            .value("Num Generations")
            .parse()
            .unwrap_or(spec.num_generations);
        spec.beta = self.form.value("KL β").parse().unwrap_or(spec.beta);
        spec.learning_rate = self
            .form
            .value("Learning Rate")
            .parse()
            .unwrap_or(spec.learning_rate);
        spec.epochs = self.form.value("Epochs").parse().unwrap_or(spec.epochs);
        spec.lora_r = self.form.value("LoRA r").parse().unwrap_or(spec.lora_r);
        spec.lora_alpha = self.form.value("LoRA α").parse().unwrap_or(spec.lora_alpha);
        spec.max_seq_len = self
            .form
            .value("Max Seq Len")
            .parse()
            .unwrap_or(spec.max_seq_len);
        spec.max_completion_length = self
            .form
            .value("Max Completion Length")
            .parse()
            .unwrap_or(spec.max_completion_length);
        spec.seed = self.form.value("Seed").parse().unwrap_or(spec.seed);
        spec.reasoning_rewards = self.form.value("Reasoning Rewards") == "Enabled";
        spec
    }
}

// ── Rendering ──────────────────────────────────────────────────────────

impl RlkdTab {
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let [config_area, right_area] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .areas(area);

        self.form
            .render_list(config_area, buf, "RLKD Configuration", |_| true);

        let [status_area, log_area] =
            Layout::vertical([Constraint::Length(7), Constraint::Min(0)]).areas(right_area);
        self.render_status(status_area, buf);
        self.log.render(log_area, buf, "RLKD Log");
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
            RlkdStatus::Idle => {
                lines.push(status_line(StatusTone::Idle, "Idle", None));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "  [S] Start  [x] Cancel",
                    THEME.text_muted,
                )));
            }
            RlkdStatus::Running => {
                lines.push(status_line(StatusTone::Running, "Running", None));
            }
            RlkdStatus::Completed => {
                lines.push(status_line(StatusTone::Completed, "Completed", None));
            }
            RlkdStatus::Failed(msg) => {
                lines.push(status_line(StatusTone::Failed, "Failed", Some(msg)));
            }
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}
