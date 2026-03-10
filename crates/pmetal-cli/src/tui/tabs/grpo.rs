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
        vec![
            format!("Model:       {}", self.field_value("Model")),
            format!("Dataset:     {}", self.field_value("Dataset")),
            format!("Generations: {}", self.field_value("Num Generations")),
            format!("Beta:        {}", self.field_value("Beta (KL)")),
            format!("GRPO Type:   {}", self.field_value("GRPO Type")),
            format!("LR:          {}", self.field_value("Learning Rate")),
            String::new(),
            "Proceed?".into(),
        ]
    }

    pub fn output_dir(&self) -> PathBuf {
        PathBuf::from(self.field_value("Output Dir"))
    }

    pub fn build_cli_args(&self) -> Vec<String> {
        let mut args = vec!["grpo".to_string()];

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
