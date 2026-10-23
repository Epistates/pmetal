//! The tabs that train: supervised fine-tuning, pretraining, distillation,
//! GRPO, RLKD, preference optimization and embedding training.
//!
//! They differ in their spec and in a few labels, and are otherwise the same
//! tab: a form of the spec's fields, a job started from it, the run's progress
//! on the right. One struct serves all seven; each constructor names its spec.

use pmetal_core::jobs::{
    DistillSpec, EmbedTrainSpec, GrpoSpec, PreferenceSpec, PretrainSpec, RlkdSpec, Spec, TrainSpec,
};
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget, Wrap};

use crate::tui::jobs::LaunchPlan;
use crate::tui::tabs::dashboard::Telemetry;
use crate::tui::tabs::datasets::DatasetPeek;
use crate::tui::tabs::model_short_name;
use crate::tui::tabs::panel::{FormTab, JobPanel, Pick};
use crate::tui::tabs::status_panel::{render_standard, render_training_status};
use crate::tui::theme::THEME;
use crate::tui::widgets::{FieldKind, FormField, FormIssue};

/// One of the training tabs.
pub struct TrainerTab {
    panel: JobPanel,
    /// The model field whose choice names the output directory.
    primary: &'static str,
    /// What the output directory is called after the model (`--lora`).
    suffix: &'static str,
    plan: fn(&JobPanel) -> Result<LaunchPlan, Vec<FormIssue>>,
    /// The run reports progress the status box can show.
    monitored: bool,
    /// A sample of the dataset picked, for sizing the context.
    peek: Option<(String, DatasetPeek)>,
}

impl TrainerTab {
    fn build<S: Spec>(title: &'static str, primary: &'static str, suffix: &'static str) -> Self {
        Self {
            panel: JobPanel::new::<S>(title),
            primary,
            suffix,
            plan: |panel| panel.plan_for::<S>(),
            monitored: S::REPORTS_EVENTS,
            peek: None,
        }
    }

    pub fn training() -> Self {
        let mut tab = Self::build::<TrainSpec>("Training", "Model", "lora");
        // The trainer reads the model's architecture itself; the form shows what
        // the Models tab knows of the model picked.
        tab.panel.form.insert_after(
            "Model",
            FormField::new(
                "Architecture",
                "(auto-detect)",
                FieldKind::ReadOnly,
                "Model",
            ),
        );
        tab
    }

    pub fn pretrain() -> Self {
        Self::build::<PretrainSpec>("Pretraining", "", "pretrain")
    }

    pub fn distillation() -> Self {
        Self::build::<DistillSpec>("Distillation", "Student", "distill")
    }

    pub fn grpo() -> Self {
        Self::build::<GrpoSpec>("GRPO", "Model", "grpo")
    }

    pub fn rlkd() -> Self {
        Self::build::<RlkdSpec>("RLKD", "Policy Model", "rlkd")
    }

    pub fn preference() -> Self {
        Self::build::<PreferenceSpec>("Preference", "Model", "preference")
    }

    pub fn embed_train() -> Self {
        Self::build::<EmbedTrainSpec>("Embed-Train", "Model", "embed")
    }

    /// The dataset sample and the warnings the tab's context length earns.
    fn dataset_notes(&self) -> Vec<Line<'static>> {
        let Some((path, peek)) = &self.peek else {
            return Vec::new();
        };
        // Only a sample of the dataset the form still names.
        if self.panel.form.value("Dataset") != *path {
            return Vec::new();
        }
        let max_seq_len = self
            .panel
            .form
            .value("Max Seq Len")
            .parse::<usize>()
            .unwrap_or(0);
        let (info, warning) = peek.describe(max_seq_len);
        let mut lines = Vec::new();
        if let Some(info) = info {
            lines.push(Line::from(Span::styled(
                format!("  {info}"),
                THEME.text_dim,
            )));
        }
        if let Some(warning) = warning {
            lines.push(Line::from(Span::styled(
                format!("  {warning}"),
                THEME.text_warning,
            )));
        }
        lines
    }
}

impl FormTab for TrainerTab {
    fn panel(&self) -> &JobPanel {
        &self.panel
    }

    fn panel_mut(&mut self) -> &mut JobPanel {
        &mut self.panel
    }

    fn plan(&self) -> Result<LaunchPlan, Vec<FormIssue>> {
        (self.plan)(&self.panel)
    }

    fn on_pick(&mut self, field: &str, picked: Pick<'_>) {
        self.panel.form.set_value(field, picked.text());
        if let Pick::Model { id, architecture } = picked {
            if field == self.primary {
                self.panel.derive(
                    "Output Dir",
                    format!("./output/{}--{}", model_short_name(id), self.suffix),
                );
                self.panel
                    .form
                    .set_value("Architecture", architecture.unwrap_or("(auto-detect)"));
            }
        }
    }

    fn wants_dataset_peek(&self) -> bool {
        self.panel.form.fields.iter().any(|f| f.label == "Dataset")
    }

    fn set_peek(&mut self, path: &str, peek: DatasetPeek) {
        self.peek = Some((path.to_string(), peek));
    }

    fn render(&mut self, area: Rect, buf: &mut Buffer, telemetry: &Telemetry<'_>) {
        if !self.monitored {
            render_standard(&mut self.panel, area, buf, 9, Vec::new(), "Log");
            return;
        }
        let [config_area, right_area] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .areas(area);

        let notes = self.dataset_notes();
        let [form_area, notes_area] = if notes.is_empty() {
            [config_area, Rect::default()]
        } else {
            Layout::vertical([
                Constraint::Min(0),
                Constraint::Length(notes.len() as u16 * 2),
            ])
            .areas(config_area)
        };
        let title = format!("{} Configuration", self.panel.title);
        self.panel.form.render_list(form_area, buf, &title);
        if !notes.is_empty() {
            Paragraph::new(notes)
                .wrap(Wrap { trim: false })
                .render(notes_area, buf);
        }

        // The log shows under the monitor when the terminal is tall enough.
        let log_height = if right_area.height >= 34 {
            (right_area.height / 4).min(12)
        } else {
            0
        };
        let [status_area, log_area] =
            Layout::vertical([Constraint::Min(0), Constraint::Length(log_height)])
                .areas(right_area);
        render_training_status(&self.panel, telemetry, status_area, buf);
        if log_height > 0 {
            self.panel.log.render(log_area, buf, "Log");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::tabs::panel::RunState;

    fn all() -> Vec<TrainerTab> {
        vec![
            TrainerTab::training(),
            TrainerTab::pretrain(),
            TrainerTab::distillation(),
            TrainerTab::grpo(),
            TrainerTab::rlkd(),
            TrainerTab::preference(),
            TrainerTab::embed_train(),
        ]
    }

    #[test]
    fn picking_the_primary_model_names_the_output_directory_after_it() {
        let mut tab = TrainerTab::training();
        tab.on_pick(
            "Model",
            Pick::Model {
                id: "Qwen/Qwen3-0.6B",
                architecture: Some("qwen3"),
            },
        );
        assert_eq!(tab.panel.form.value("Model"), "Qwen/Qwen3-0.6B");
        assert_eq!(
            tab.panel.form.value("Output Dir"),
            "./output/Qwen3-0.6B--lora"
        );
        assert_eq!(tab.panel.form.value("Architecture"), "qwen3");
        // Picking another model renames it, as long as nobody typed their own.
        tab.on_pick(
            "Model",
            Pick::Model {
                id: "org/Other",
                architecture: None,
            },
        );
        assert_eq!(tab.panel.form.value("Output Dir"), "./output/Other--lora");
        assert_eq!(tab.panel.form.value("Architecture"), "(auto-detect)");
        tab.panel.form.set_value("Output Dir", "/my/place");
        tab.on_pick(
            "Model",
            Pick::Model {
                id: "org/Third",
                architecture: None,
            },
        );
        assert_eq!(tab.panel.form.value("Output Dir"), "/my/place");
    }

    #[test]
    fn a_pick_goes_to_the_field_that_asked_for_it() {
        // The old routing sent every dataset to "Dataset" and every model of
        // GRPO to "Model": the eval dataset and the reward model overwrote them.
        let mut train = TrainerTab::training();
        train.on_pick("Dataset", Pick::Dataset("train.jsonl"));
        train.on_pick("Eval Dataset", Pick::Dataset("eval.jsonl"));
        assert_eq!(train.panel.form.value("Dataset"), "train.jsonl");
        assert_eq!(train.panel.form.value("Eval Dataset"), "eval.jsonl");

        let mut grpo = TrainerTab::grpo();
        grpo.on_pick(
            "Model",
            Pick::Model {
                id: "policy",
                architecture: None,
            },
        );
        grpo.on_pick(
            "Reward Model",
            Pick::Model {
                id: "judge",
                architecture: None,
            },
        );
        assert_eq!(grpo.panel.form.value("Model"), "policy");
        assert_eq!(grpo.panel.form.value("Reward Model"), "judge");
        // Only the primary model names the output.
        assert_eq!(grpo.panel.form.value("Output Dir"), "./output/policy--grpo");

        let mut distill = TrainerTab::distillation();
        distill.on_pick(
            "Teacher",
            Pick::Model {
                id: "big/Teacher",
                architecture: None,
            },
        );
        assert_eq!(
            distill.panel.form.value("Output Dir"),
            DistillSpec::default().output_dir
        );
        distill.on_pick(
            "Student",
            Pick::Model {
                id: "small/Student",
                architecture: None,
            },
        );
        assert_eq!(
            distill.panel.form.value("Output Dir"),
            "./output/Student--distill"
        );
    }

    #[test]
    fn a_form_with_nothing_chosen_says_what_is_missing() {
        for tab in all() {
            let title = tab.panel.title;
            match tab.plan() {
                Ok(_) => {}
                Err(issues) => {
                    assert!(!issues.is_empty(), "{title}");
                    for issue in issues {
                        assert!(
                            !issue.field.is_empty() && !issue.message.is_empty(),
                            "{title}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_training_run_writes_its_events_and_metrics_where_the_monitor_reads_them() {
        let mut tab = TrainerTab::training();
        tab.on_pick(
            "Model",
            Pick::Model {
                id: "Qwen/Qwen3-0.6B",
                architecture: None,
            },
        );
        tab.on_pick("Dataset", Pick::Dataset("data.jsonl"));
        tab.panel.form.set_value("Output Dir", "/out/run");
        let plan = tab.plan().unwrap();
        assert_eq!(plan.subcommand, "train");
        assert_eq!(plan.output, Some(std::path::PathBuf::from("/out/run")));
        assert_eq!(
            plan.events_file,
            Some(std::path::PathBuf::from("/out/run/events.jsonl"))
        );
        let at = plan.argv.iter().position(|a| a == "--log-events").unwrap();
        assert_eq!(plan.argv[at + 1], "/out/run/events.jsonl");
        let at = plan.argv.iter().position(|a| a == "--log-metrics").unwrap();
        assert_eq!(plan.argv[at + 1], "/out/run/metrics.jsonl");
    }

    #[test]
    fn pretraining_has_no_event_stream_to_follow() {
        let mut tab = TrainerTab::pretrain();
        tab.panel.form.set_value("Architecture", "llama");
        let plan = tab.plan();
        let plan = plan.unwrap_or_else(|e| panic!("{e:?}"));
        assert!(plan.events_file.is_none());
        assert!(!plan.argv.iter().any(|a| a == "--log-events"));
    }

    #[test]
    fn the_dataset_notes_follow_the_context_length_being_edited() {
        let mut tab = TrainerTab::training();
        tab.on_pick("Dataset", Pick::Dataset("d.jsonl"));
        tab.set_peek(
            "d.jsonl",
            DatasetPeek {
                columns: vec!["text".into()],
                token_estimates: (1..=100).map(|n| n * 10).collect(),
            },
        );
        tab.panel.form.set_value("Max Seq Len", "100");
        let notes = tab.dataset_notes();
        assert_eq!(notes.len(), 2, "{notes:?}");
        // A sample of some other dataset says nothing about this one.
        tab.on_pick("Dataset", Pick::Dataset("other.jsonl"));
        assert!(tab.dataset_notes().is_empty());
    }

    #[test]
    fn every_trainer_tab_draws_at_any_size() {
        for mut tab in all() {
            tab.panel.run = RunState::Idle;
            for &(w, h) in &[(0, 0), (1, 1), (20, 6), (80, 24), (160, 50)] {
                let area = Rect::new(0, 0, w, h);
                let mut buf = Buffer::empty(area);
                tab.render(area, &mut buf, &Telemetry::none());
            }
        }
    }
}
