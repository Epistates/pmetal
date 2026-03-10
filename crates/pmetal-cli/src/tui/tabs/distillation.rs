//! Distillation configuration and control tab.
//!
//! Knowledge distillation from a teacher model to a student model.
//! Supports online, offline, and progressive distillation methods.

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

/// Actions the distillation tab can request from the app.
#[derive(Debug)]
pub enum DistillAction {
    OpenTeacherPicker,
    OpenStudentPicker,
    OpenDatasetPicker,
    StartEdit,
}

/// Distillation tab state.
pub struct DistillationTab {
    pub fields: Vec<FormField>,
    pub list_state: ListState,
    pub status: TrainingStatus,
    field_idx: usize,
}

impl DistillationTab {
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
            // Models
            FormField::new(
                "Teacher Model",
                "(not selected)",
                FieldKind::ModelPicker,
                "Models",
            ),
            FormField::new(
                "Student Model",
                "(not selected)",
                FieldKind::ModelPicker,
                "Models",
            ),
            // Distillation
            FormField::new(
                "Method",
                "online",
                FieldKind::Enum {
                    options: vec!["online".into(), "offline".into(), "progressive".into()],
                },
                "Distillation",
            ),
            FormField::new(
                "Loss Type",
                "kl_divergence",
                FieldKind::Enum {
                    options: vec![
                        "kl_divergence".into(),
                        "jensen_shannon".into(),
                        "soft_cross_entropy".into(),
                        "mse_loss".into(),
                    ],
                },
                "Distillation",
            ),
            FormField::new(
                "Temperature",
                "2.0",
                FieldKind::Number {
                    min: 0.1,
                    max: 100.0,
                },
                "Distillation",
            ),
            FormField::new(
                "Alpha",
                "0.5",
                FieldKind::Number { min: 0.0, max: 1.0 },
                "Distillation",
            ),
            FormField::new("Rationale", "Disabled", FieldKind::Toggle, "Distillation"),
            FormField::new(
                "Rationale Weight",
                "1.0",
                FieldKind::Number {
                    min: 0.0,
                    max: 10.0,
                },
                "Distillation",
            ),
            // Training
            FormField::new(
                "Learning Rate",
                "2e-5",
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
                "1024",
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
            // Output
            FormField::new(
                "Output Dir",
                "./output/distilled",
                FieldKind::Text,
                "Output",
            ),
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

    pub fn handle_enter(&mut self) -> Option<DistillAction> {
        let field = self.fields.get_mut(self.field_idx)?;

        if field.is_picker() {
            return match field.label.as_str() {
                "Teacher Model" => Some(DistillAction::OpenTeacherPicker),
                "Student Model" => Some(DistillAction::OpenStudentPicker),
                "Dataset" => Some(DistillAction::OpenDatasetPicker),
                _ => None,
            };
        }
        if field.is_cycleable() {
            field.cycle();
            return None;
        }
        if field.is_inline_editable() {
            field.start_edit();
            return Some(DistillAction::StartEdit);
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

    pub fn set_teacher(&mut self, model_id: &str) {
        if let Some(f) = self.fields.iter_mut().find(|f| f.label == "Teacher Model") {
            f.value = model_id.to_string();
        }
    }

    pub fn set_student(&mut self, model_id: &str) {
        if let Some(f) = self.fields.iter_mut().find(|f| f.label == "Student Model") {
            f.value = model_id.to_string();
        }
        // Auto-update output dir with student model name
        let short_name = super::model_short_name(model_id);
        if let Some(f) = self.fields.iter_mut().find(|f| f.label == "Output Dir") {
            f.value = format!("./output/{short_name}--distill");
        }
    }

    /// Focus a specific field by label.
    pub fn focus_field(&mut self, label: &str) {
        if let Some(idx) = self.fields.iter().position(|f| f.label == label) {
            self.field_idx = idx;
        }
    }

    pub fn set_dataset(&mut self, path: &str) {
        if let Some(f) = self.fields.iter_mut().find(|f| f.label == "Dataset") {
            f.value = path.to_string();
        }
    }

    // --- Config ---

    pub fn validate_config(&self) -> Result<(), String> {
        if self.field_value("Teacher Model") == "(not selected)" {
            return Err("Teacher model is required.".into());
        }
        if self.field_value("Student Model") == "(not selected)" {
            return Err("Student model is required.".into());
        }
        if self.field_value("Dataset") == "(not selected)" {
            return Err("Dataset is required.".into());
        }
        Ok(())
    }

    pub fn config_summary(&self) -> Vec<String> {
        vec![
            format!("Teacher:   {}", self.field_value("Teacher Model")),
            format!("Student:   {}", self.field_value("Student Model")),
            format!("Dataset:   {}", self.field_value("Dataset")),
            format!("Method:    {}", self.field_value("Method")),
            format!("Loss:      {}", self.field_value("Loss Type")),
            format!("Temp:      {}", self.field_value("Temperature")),
            format!("Alpha:     {}", self.field_value("Alpha")),
            String::new(),
            "Proceed?".into(),
        ]
    }

    pub fn output_dir(&self) -> PathBuf {
        PathBuf::from(self.field_value("Output Dir"))
    }

    pub fn build_cli_args(&self) -> Vec<String> {
        let mut args = vec!["distill".to_string()];

        args.extend(["--teacher".into(), self.field_value("Teacher Model")]);
        args.extend(["--student".into(), self.field_value("Student Model")]);
        args.extend(["--dataset".into(), self.field_value("Dataset")]);
        args.extend(["--output".into(), self.field_value("Output Dir")]);
        args.extend(["--method".into(), self.field_value("Method")]);
        args.extend(["--loss-type".into(), self.field_value("Loss Type")]);
        args.extend(["--temperature".into(), self.field_value("Temperature")]);
        args.extend(["--alpha".into(), self.field_value("Alpha")]);
        args.extend(["--learning-rate".into(), self.field_value("Learning Rate")]);
        args.extend(["--batch-size".into(), self.field_value("Batch Size")]);
        args.extend(["--epochs".into(), self.field_value("Epochs")]);
        args.extend(["--max-seq-len".into(), self.field_value("Max Seq Len")]);
        args.extend(["--lora-r".into(), self.field_value("LoRA Rank")]);
        args.extend(["--lora-alpha".into(), self.field_value("LoRA Alpha")]);

        if self.field_value("Rationale") == "Enabled" {
            args.push("--rationale".into());
            args.extend([
                "--rationale-weight".into(),
                self.field_value("Rationale Weight"),
            ]);
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

impl DistillationTab {
    /// Render the full distillation tab with embedded dashboard metrics.
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
            .title(" Distillation Configuration ")
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
