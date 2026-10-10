//! EmbedTrain tab — sentence embedding model training (contrastive losses).
//!
//! Form driven by [`EmbedTrainSpec`]. Status + log panel on the right.

use pmetal_core::JobFields as _;
use pmetal_core::jobs::EmbedTrainSpec;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

use crate::tui::theme::THEME;
use crate::tui::widgets::{FormAction, FormTabState, JobLog, StatusTone, status_line};

/// Runtime status of the `pmetal embed-train` process.
#[derive(Debug, Clone, Default)]
pub enum EmbedTrainStatus {
    #[default]
    Idle,
    Running,
    Completed,
    Failed(String),
}

/// EmbedTrain tab state.
pub struct EmbedTrainTab {
    pub form: FormTabState,
    pub status: EmbedTrainStatus,
    pub log: JobLog,
}

impl EmbedTrainTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::from_spec_default::<EmbedTrainSpec>(),
            status: EmbedTrainStatus::Idle,
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
        matches!(self.status, EmbedTrainStatus::Running)
    }

    pub fn mark_running(&mut self) {
        self.log.clear();
        self.status = EmbedTrainStatus::Running;
    }

    pub fn mark_completed(&mut self) {
        self.status = EmbedTrainStatus::Completed;
    }

    pub fn mark_failed(&mut self, msg: &str) {
        self.status = EmbedTrainStatus::Failed(msg.to_string());
    }

    pub fn append_log(&mut self, line: &str) {
        self.log.push(line);
    }

    // ── Config ──────────────────────────────────────────────────────────

    pub fn validate_config(&self) -> Result<(), String> {
        let model = self.form.value("Model");
        if model.is_empty() || model == "(not selected)" {
            return Err("Model is required.".into());
        }
        let dataset = self.form.value("Dataset");
        if dataset.is_empty() || dataset == "(not selected)" {
            return Err("Dataset is required.".into());
        }
        Ok(())
    }

    pub fn config_summary(&self) -> Vec<String> {
        vec![
            format!("Model:    {}", self.form.value("Model")),
            format!("Dataset:  {}", self.form.value("Dataset")),
            format!("Loss:     {}", self.form.value("Loss")),
            format!("Epochs:   {}", self.form.value("Epochs")),
            String::new(),
            "Run embed-train?".into(),
        ]
    }

    pub fn build_cli_args(&self) -> Vec<String> {
        let spec = self.spec_from_form();
        let mut args = vec!["embed-train".to_string()];
        args.extend(spec.to_argv());
        args
    }

    fn spec_from_form(&self) -> EmbedTrainSpec {
        let mut spec = EmbedTrainSpec::default();
        spec.model = self.form.value("Model");
        spec.dataset = self.form.value("Dataset");
        spec.output_dir = self.form.value("Output Dir");
        spec.loss = self.form.value("Loss");
        spec.pooling = self.form.value("Pooling");
        spec.temperature = self
            .form
            .value("Temperature")
            .parse()
            .unwrap_or(spec.temperature);
        spec.margin = self.form.value("Margin").parse().unwrap_or(spec.margin);
        spec.learning_rate = self
            .form
            .value("Learning Rate")
            .parse()
            .unwrap_or(spec.learning_rate);
        spec.batch_size = self
            .form
            .value("Batch Size")
            .parse()
            .unwrap_or(spec.batch_size);
        spec.epochs = self.form.value("Epochs").parse().unwrap_or(spec.epochs);
        spec.max_seq_len = self
            .form
            .value("Max Seq Len")
            .parse()
            .unwrap_or(spec.max_seq_len);
        spec.weight_decay = self
            .form
            .value("Weight Decay")
            .parse()
            .unwrap_or(spec.weight_decay);
        spec.no_normalize = self.form.value("Disable L2 Norm") == "Enabled";
        spec.log_every = self
            .form
            .value("Log Every")
            .parse()
            .unwrap_or(spec.log_every);
        spec.seed = self.form.value("Seed").parse().unwrap_or(spec.seed);
        spec
    }
}

// ── Rendering ──────────────────────────────────────────────────────────

impl EmbedTrainTab {
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let [config_area, right_area] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .areas(area);

        self.form
            .render_list(config_area, buf, "Embed-Train Configuration", |_| true);

        let [status_area, log_area] =
            Layout::vertical([Constraint::Length(7), Constraint::Min(0)]).areas(right_area);
        self.render_status(status_area, buf);
        self.log.render(log_area, buf, "Embed-Train Log");
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
            EmbedTrainStatus::Idle => {
                lines.push(status_line(StatusTone::Idle, "Idle", None));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "  [S] Start  [x] Cancel",
                    THEME.text_muted,
                )));
            }
            EmbedTrainStatus::Running => {
                lines.push(status_line(StatusTone::Running, "Running", None));
            }
            EmbedTrainStatus::Completed => {
                lines.push(status_line(StatusTone::Completed, "Completed", None));
            }
            EmbedTrainStatus::Failed(msg) => {
                lines.push(status_line(StatusTone::Failed, "Failed", Some(msg)));
            }
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}
