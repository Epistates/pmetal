//! GRPO (Group Relative Policy Optimization) configuration and control tab.

use std::path::PathBuf;

use crossterm::event::KeyEvent;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState};

use crate::tui::tabs::dashboard::MetricSample;
use crate::tui::tabs::training::{TrainingStatus, render_status_with_metrics};
use crate::tui::theme::THEME;
use crate::tui::widgets::{FieldKind, FormField};

/// Actions the GRPO tab can request from the app.
#[derive(Debug)]
pub enum GrpoAction {
    OpenModelPicker,
    OpenDatasetPicker,
    StartEdit,
}

/// GRPO tab state.
pub struct GrpoTab {
    pub fields: Vec<FormField>,
    pub list_state: ListState,
    pub status: TrainingStatus,
    field_idx: usize,
}

impl GrpoTab {
    pub fn new() -> Self {
        Self {
            fields: Self::default_fields(),
            list_state: ListState::default().with_selected(Some(1)),
            status: TrainingStatus::Idle,
            field_idx: 0,
        }
    }

    fn default_fields() -> Vec<FormField> {
        vec![
            // Model
            FormField::new("Model", "(not selected)", FieldKind::ModelPicker, "Model"),
            // GRPO
            FormField::new(
                "Num Generations",
                "8",
                FieldKind::Integer { min: 2, max: 64 },
                "GRPO",
            ),
            FormField::new(
                "Beta (KL)",
                "0.001",
                FieldKind::Number { min: 0.0, max: 1.0 },
                "GRPO",
            ),
            FormField::new(
                "Max Completion Len",
                "512",
                FieldKind::Integer { min: 32, max: 8192 },
                "GRPO",
            ),
            FormField::new(
                "GRPO Type",
                "bnpo",
                FieldKind::Enum {
                    options: vec![
                        "bnpo".into(),
                        "dr_grpo".into(),
                        "dapo".into(),
                        "reinforce".into(),
                    ],
                },
                "GRPO",
            ),
            FormField::new("Reasoning Rewards", "Disabled", FieldKind::Toggle, "GRPO"),
            FormField::new("VLM Mode", "Disabled", FieldKind::Toggle, "GRPO"),
            FormField::new(
                "Max Image Size",
                "336",
                FieldKind::Integer { min: 64, max: 2048 },
                "GRPO",
            ),
            // Reward Model
            FormField::new("Reward Model", "", FieldKind::Text, "Reward Model"),
            FormField::new(
                "RM Max Length",
                "2048",
                FieldKind::Integer {
                    min: 128,
                    max: 32768,
                },
                "Reward Model",
            ),
            FormField::new(
                "RM Weight",
                "1.0",
                FieldKind::Number {
                    min: 0.0,
                    max: 10.0,
                },
                "Reward Model",
            ),
            FormField::new("RM Template", "", FieldKind::Text, "Reward Model"),
            FormField::new(
                "Async Rewards",
                "Disabled",
                FieldKind::Toggle,
                "Reward Model",
            ),
            // Training
            FormField::new(
                "Learning Rate",
                "5e-6",
                FieldKind::Number {
                    min: 1e-8,
                    max: 1.0,
                },
                "Training",
            ),
            FormField::new(
                "Batch Size",
                "1",
                FieldKind::Integer { min: 1, max: 128 },
                "Training",
            ),
            FormField::new(
                "Epochs",
                "1",
                FieldKind::Integer { min: 1, max: 100 },
                "Training",
            ),
            FormField::new(
                "Max Seq Len",
                "512",
                FieldKind::Integer {
                    min: 64,
                    max: 131072,
                },
                "Training",
            ),
            FormField::new(
                "LoRA Rank",
                "16",
                FieldKind::Integer { min: 1, max: 256 },
                "LoRA",
            ),
            FormField::new(
                "LoRA Alpha",
                "32",
                FieldKind::Number {
                    min: 1.0,
                    max: 512.0,
                },
                "LoRA",
            ),
            // Data
            FormField::new(
                "Dataset",
                "(not selected)",
                FieldKind::DatasetPicker,
                "Data",
            ),
            // Hardware
            FormField::new("Flash Attention", "Enabled", FieldKind::Toggle, "Hardware"),
            FormField::new(
                "Speculative Decoding",
                "Disabled",
                FieldKind::Toggle,
                "Hardware",
            ),
            FormField::new(
                "Draft Tokens",
                "3",
                FieldKind::Integer { min: 1, max: 16 },
                "Hardware",
            ),
            // RLKD (optional teacher distillation — leave Teacher Model blank for pure GRPO)
            FormField::new("Teacher Model", "", FieldKind::Text, "RLKD"),
            FormField::new(
                "Distill Alpha",
                "0.3",
                FieldKind::Number { min: 0.0, max: 1.0 },
                "RLKD",
            ),
            FormField::new(
                "Distill Temperature",
                "2.0",
                FieldKind::Number {
                    min: 0.5,
                    max: 10.0,
                },
                "RLKD",
            ),
            FormField::new("Anneal Alpha", "Enabled", FieldKind::Toggle, "RLKD"),
            FormField::new(
                "Final Alpha",
                "0.05",
                FieldKind::Number { min: 0.0, max: 1.0 },
                "RLKD",
            ),
            // Output
            FormField::new("Output Dir", "./output/grpo", FieldKind::Text, "Output"),
        ]
    }

    pub fn is_editing(&self) -> bool {
        self.fields.get(self.field_idx).is_some_and(|f| f.editing)
    }

    pub fn handle_edit_key(&mut self, key: KeyEvent) {
        if let Some(field) = self.fields.get_mut(self.field_idx) {
            field.handle_edit_key(key);
        }
    }

    pub fn confirm_edit(&mut self) {
        if let Some(field) = self.fields.get_mut(self.field_idx) {
            field.confirm_edit();
        }
    }

    pub fn cancel_edit(&mut self) {
        if let Some(field) = self.fields.get_mut(self.field_idx) {
            field.cancel_edit();
        }
    }

    pub fn handle_enter(&mut self) -> Option<GrpoAction> {
        let field = self.fields.get_mut(self.field_idx)?;

        if field.is_picker() {
            return match field.label.as_str() {
                "Model" => Some(GrpoAction::OpenModelPicker),
                "Dataset" => Some(GrpoAction::OpenDatasetPicker),
                _ => None,
            };
        }
        if field.is_cycleable() {
            field.cycle();
            return None;
        }
        if field.is_inline_editable() {
            field.start_edit();
            return Some(GrpoAction::StartEdit);
        }
        None
    }

    pub fn next_param(&mut self) {
        let count = self.fields.len();
        if count == 0 {
            return;
        }
        self.field_idx = (self.field_idx + 1) % count;
        self.sync_list_selection();
    }

    pub fn prev_param(&mut self) {
        let count = self.fields.len();
        if count == 0 {
            return;
        }
        self.field_idx = (self.field_idx + count - 1) % count;
        self.sync_list_selection();
    }

    fn sync_list_selection(&mut self) {
        let flat = self.flat_index_for_field(self.field_idx);
        self.list_state.select(Some(flat));
    }

    fn flat_index_for_field(&self, field_idx: usize) -> usize {
        let mut flat = 0;
        let mut current_section: Option<&str> = None;
        for (i, field) in self.fields.iter().enumerate() {
            if current_section != Some(&field.section) {
                current_section = Some(&field.section);
                flat += 1;
            }
            if i == field_idx {
                return flat;
            }
            flat += 1;
        }
        flat
    }

    // --- Setters ---

    pub fn set_model(&mut self, model_id: &str) {
        if let Some(f) = self.fields.iter_mut().find(|f| f.label == "Model") {
            f.value = model_id.to_string();
        }
        // Auto-update output dir with model name
        let short_name = super::model_short_name(model_id);
        if let Some(f) = self.fields.iter_mut().find(|f| f.label == "Output Dir") {
            f.value = format!("./output/{short_name}--grpo");
        }
    }

    pub fn set_dataset(&mut self, path: &str) {
        if let Some(f) = self.fields.iter_mut().find(|f| f.label == "Dataset") {
            f.value = path.to_string();
        }
    }

    // --- Config ---

    pub fn validate_config(&self) -> Result<(), String> {
        if self.field_value("Model") == "(not selected)" {
            return Err("Model is required.".into());
        }
        if self.field_value("Dataset") == "(not selected)" {
            return Err("Dataset is required.".into());
        }
        Ok(())
    }

    pub fn config_summary(&self) -> Vec<String> {
        let teacher = self.field_value("Teacher Model");
        let mode = if teacher.is_empty() { "GRPO" } else { "RLKD" };
        let mut summary = vec![
            format!("Mode:        {}", mode),
            format!("Model:       {}", self.field_value("Model")),
            format!("Dataset:     {}", self.field_value("Dataset")),
            format!("Generations: {}", self.field_value("Num Generations")),
            format!("Beta:        {}", self.field_value("Beta (KL)")),
            format!("GRPO Type:   {}", self.field_value("GRPO Type")),
            format!("LR:          {}", self.field_value("Learning Rate")),
        ];
        if !teacher.is_empty() {
            summary.push(format!("Teacher:     {}", teacher));
            summary.push(format!(
                "Alpha:       {} → {} (anneal={})",
                self.field_value("Distill Alpha"),
                self.field_value("Final Alpha"),
                self.field_value("Anneal Alpha"),
            ));
            summary.push(format!(
                "Temperature: {}",
                self.field_value("Distill Temperature")
            ));
        }
        if self.field_value("VLM Mode") == "Enabled" {
            summary.push(format!(
                "VLM:         enabled (max_img={}px)",
                self.field_value("Max Image Size")
            ));
        }
        let rm_path = self.field_value("Reward Model");
        if !rm_path.is_empty() {
            summary.push(format!(
                "Reward Model: {} (weight={})",
                rm_path,
                self.field_value("RM Weight")
            ));
        }
        summary.push(String::new());
        summary.push("Proceed?".into());
        summary
    }

    pub fn output_dir(&self) -> PathBuf {
        PathBuf::from(self.field_value("Output Dir"))
    }

    pub fn build_cli_args(&self) -> Vec<String> {
        // When a teacher model is provided, switch to the RLKD subcommand which
        // adds distillation on top of the GRPO policy gradient objective.
        let teacher = self.field_value("Teacher Model");
        let use_rlkd = !teacher.is_empty();

        let subcommand = if use_rlkd { "rlkd" } else { "grpo" };
        let mut args = vec![subcommand.to_string()];

        args.extend(["--model".into(), self.field_value("Model")]);
        args.extend(["--dataset".into(), self.field_value("Dataset")]);
        args.extend(["--output".into(), self.field_value("Output Dir")]);
        args.extend([
            "--num-generations".into(),
            self.field_value("Num Generations"),
        ]);
        args.extend(["--beta".into(), self.field_value("Beta (KL)")]);
        args.extend(["--learning-rate".into(), self.field_value("Learning Rate")]);
        args.extend(["--batch-size".into(), self.field_value("Batch Size")]);
        args.extend(["--epochs".into(), self.field_value("Epochs")]);
        args.extend(["--max-seq-len".into(), self.field_value("Max Seq Len")]);
        args.extend(["--lora-r".into(), self.field_value("LoRA Rank")]);
        args.extend(["--lora-alpha".into(), self.field_value("LoRA Alpha")]);
        args.extend([
            "--max-completion-length".into(),
            self.field_value("Max Completion Len"),
        ]);

        let grpo_type = self.field_value("GRPO Type");
        if grpo_type == "dapo" {
            args.push("--dapo".into());
        }

        if self.field_value("Reasoning Rewards") == "Enabled" {
            args.push("--reasoning-rewards".into());
        }
        if self.field_value("Flash Attention") == "Disabled" {
            args.push("--no-flash-attention".into());
        }
        if self.field_value("Speculative Decoding") == "Enabled" {
            args.push("--speculative".into());
            args.extend([
                "--speculative-draft-tokens".into(),
                self.field_value("Draft Tokens"),
            ]);
        }
        if self.field_value("VLM Mode") == "Enabled" {
            args.push("--vlm".into());
            args.extend([
                "--max-image-size".into(),
                self.field_value("Max Image Size"),
            ]);
        }

        let rm_path = self.field_value("Reward Model");
        if !rm_path.is_empty() {
            args.extend(["--reward-model".into(), rm_path]);
            args.extend([
                "--reward-model-max-length".into(),
                self.field_value("RM Max Length"),
            ]);
            args.extend([
                "--reward-model-weight".into(),
                self.field_value("RM Weight"),
            ]);
            let rm_template = self.field_value("RM Template");
            if !rm_template.is_empty() {
                args.extend(["--reward-model-template".into(), rm_template]);
            }
            if self.field_value("Async Rewards") == "Enabled" {
                args.push("--async-rewards".into());
            }
        }

        // RLKD fields — only emitted when Teacher Model is set
        if use_rlkd {
            args.extend(["--teacher-model".into(), teacher]);
            args.extend(["--distill-alpha".into(), self.field_value("Distill Alpha")]);
            args.extend([
                "--distill-temperature".into(),
                self.field_value("Distill Temperature"),
            ]);
            if self.field_value("Anneal Alpha") == "Enabled" {
                args.push("--anneal-alpha".into());
            }
            args.extend(["--final-alpha".into(), self.field_value("Final Alpha")]);
        }

        args
    }

    fn field_value(&self, label: &str) -> String {
        self.fields
            .iter()
            .find(|f| f.label == label)
            .map(|f| f.value.clone())
            .unwrap_or_default()
    }
}

impl GrpoTab {
    /// Render the full GRPO tab with embedded dashboard metrics.
    pub fn render_with_metrics(
        &mut self,
        area: Rect,
        buf: &mut Buffer,
        samples: &[MetricSample],
        throughput: &[u64],
    ) {
        let [config_area, status_area] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .areas(area);

        self.render_config(config_area, buf);
        render_status_with_metrics(&self.status, samples, throughput, status_area, buf);
    }

    fn render_config(&mut self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" GRPO Configuration ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);

        let key_width = self
            .fields
            .iter()
            .map(|f| f.label.len())
            .max()
            .unwrap_or(10);

        let mut current_section: Option<&str> = None;
        let mut items: Vec<ListItem> = Vec::new();

        for (i, field) in self.fields.iter().enumerate() {
            if current_section != Some(&field.section) {
                current_section = Some(&field.section);
                items.push(ListItem::new(Line::from(Span::styled(
                    format!("  --- {} ---", field.section),
                    THEME.text_muted,
                ))));
            }
            let selected = i == self.field_idx;
            items.push(ListItem::new(field.render_line(key_width, selected)));
        }

        let list = List::new(items)
            .block(block)
            .highlight_style(THEME.table_selected);

        ratatui::widgets::StatefulWidget::render(list, area, buf, &mut self.list_state);
    }
}
