//! Training configuration and control tab.
//!
//! Provides editable form fields for all training parameters. Users can
//! navigate fields, edit values inline, pick models/datasets via modals,
//! and launch training runs.

use std::path::PathBuf;

use crossterm::event::KeyEvent;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Gauge, List, ListItem, ListState, Paragraph, Sparkline, Widget, Wrap,
};

use crate::tui::tabs::dashboard::MetricSample;
use crate::tui::tabs::model_short_name;
use crate::tui::theme::{THEME, palette};
use crate::tui::widgets::{FieldKind, FormField};

/// Actions the training tab can request from the app.
#[derive(Debug)]
pub enum TrainingAction {
    OpenModelPicker,
    OpenDatasetPicker,
    StartEdit,
}

/// Training run status.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum TrainingStatus {
    Idle,
    Running {
        step: usize,
        epoch: usize,
        total_epochs: usize,
        total_steps: usize,
        loss: f64,
    },
    Completed {
        final_loss: f64,
        total_steps: usize,
    },
    Failed(String),
}

/// Training tab state.
pub struct TrainingTab {
    pub fields: Vec<FormField>,
    pub list_state: ListState,
    pub status: TrainingStatus,
    field_idx: usize,
}

impl TrainingTab {
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
            FormField::new("Architecture", "-", FieldKind::ReadOnly, "Model"),
            // Training
            FormField::new(
                "Learning Rate",
                "2e-4",
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
                "2048",
                FieldKind::Integer {
                    min: 0,
                    max: 131072,
                },
                "Training",
            ),
            FormField::new(
                "Grad Accum Steps",
                "4",
                FieldKind::Integer { min: 1, max: 256 },
                "Training",
            ),
            FormField::new(
                "Max Grad Norm",
                "1.0",
                FieldKind::Number {
                    min: 0.0,
                    max: 100.0,
                },
                "Training",
            ),
            FormField::new(
                "Warmup Steps",
                "100",
                FieldKind::Integer {
                    min: 0,
                    max: 100000,
                },
                "Training",
            ),
            FormField::new(
                "Weight Decay",
                "0.01",
                FieldKind::Number { min: 0.0, max: 1.0 },
                "Training",
            ),
            // LoRA
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
            FormField::new(
                "Quantization",
                "None",
                FieldKind::Enum {
                    options: vec!["None".into(), "NF4".into(), "FP4".into(), "INT8".into()],
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
            FormField::new("Eval Dataset", "(none)", FieldKind::Text, "Data"),
            FormField::new("Sequence Packing", "Enabled", FieldKind::Toggle, "Data"),
            // Hardware
            FormField::new("Flash Attention", "Enabled", FieldKind::Toggle, "Hardware"),
            FormField::new("Fused Optimizer", "Enabled", FieldKind::Toggle, "Hardware"),
            FormField::new("JIT Compilation", "Enabled", FieldKind::Toggle, "Hardware"),
            FormField::new(
                "ANE",
                "Auto",
                FieldKind::Enum {
                    options: vec!["Auto".into(), "Enabled".into(), "Disabled".into()],
                },
                "Hardware",
            ),
            // Output
            FormField::new("Output Dir", "./output", FieldKind::Text, "Output"),
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

    /// Handle Enter on the currently selected field. Returns an action if
    /// the app needs to open a modal or perform something external.
    pub fn handle_enter(&mut self) -> Option<TrainingAction> {
        let field = self.fields.get_mut(self.field_idx)?;

        if field.is_picker() {
            return match &field.kind {
                FieldKind::ModelPicker => Some(TrainingAction::OpenModelPicker),
                FieldKind::DatasetPicker => Some(TrainingAction::OpenDatasetPicker),
                _ => None,
            };
        }
        if field.is_cycleable() {
            field.cycle();
            return None;
        }
        if field.is_inline_editable() {
            field.start_edit();
            return Some(TrainingAction::StartEdit);
        }
        None
    }

    pub fn next_param(&mut self) {
        let count = self.fields.len();
        if count == 0 {
            return;
        }
        self.field_idx = (self.field_idx + 1) % count;
        // Skip read-only fields that can't be interacted with
        if matches!(self.fields[self.field_idx].kind, FieldKind::ReadOnly) {
            self.field_idx = (self.field_idx + 1) % count;
        }
        self.sync_list_selection();
    }

    pub fn prev_param(&mut self) {
        let count = self.fields.len();
        if count == 0 {
            return;
        }
        self.field_idx = (self.field_idx + count - 1) % count;
        if matches!(self.fields[self.field_idx].kind, FieldKind::ReadOnly) {
            self.field_idx = (self.field_idx + count - 1) % count;
        }
        self.sync_list_selection();
    }

    fn sync_list_selection(&mut self) {
        // Account for section headers in the flat list
        let flat = self.flat_index_for_field(self.field_idx);
        self.list_state.select(Some(flat));
    }

    fn flat_index_for_field(&self, field_idx: usize) -> usize {
        let mut flat = 0;
        let mut current_section: Option<&str> = None;
        for (i, field) in self.fields.iter().enumerate() {
            if current_section != Some(&field.section) {
                current_section = Some(&field.section);
                flat += 1; // section header
            }
            if i == field_idx {
                return flat;
            }
            flat += 1;
        }
        flat
    }

    // --- Model/Dataset setters ---

    pub fn set_model(&mut self, model_id: &str) {
        if let Some(f) = self.fields.iter_mut().find(|f| f.label == "Model") {
            f.value = model_id.to_string();
        }
        // Auto-detect architecture (will be updated when model is loaded)
        if let Some(f) = self.fields.iter_mut().find(|f| f.label == "Architecture") {
            f.value = "(auto-detect)".to_string();
        }
        // Auto-update output dir with base model name
        let short_name = model_short_name(model_id);
        if let Some(f) = self.fields.iter_mut().find(|f| f.label == "Output Dir") {
            f.value = format!("./output/{short_name}--lora");
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

    // --- Status updates ---

    pub fn set_status_running(
        &mut self,
        step: usize,
        epoch: usize,
        total_epochs: usize,
        total_steps: usize,
        loss: f64,
    ) {
        self.status = TrainingStatus::Running {
            step,
            epoch,
            total_epochs,
            total_steps,
            loss,
        };
    }

    pub fn set_status_completed(&mut self, final_loss: f64, total_steps: usize) {
        self.status = TrainingStatus::Completed {
            final_loss,
            total_steps,
        };
    }

    pub fn set_status_failed(&mut self, msg: &str) {
        self.status = TrainingStatus::Failed(msg.to_string());
    }

    // --- Config validation and CLI arg building ---

    pub fn validate_config(&self) -> Result<(), String> {
        let model = self.field_value("Model");
        if model == "(not selected)" || model.is_empty() {
            return Err("Model is required. Press Enter on the Model field to select one.".into());
        }
        let dataset = self.field_value("Dataset");
        if dataset == "(not selected)" || dataset.is_empty() {
            return Err(
                "Dataset is required. Press Enter on the Dataset field to select one.".into(),
            );
        }
        Ok(())
    }

    pub fn config_summary(&self) -> Vec<String> {
        vec![
            format!("Model:     {}", self.field_value("Model")),
            format!("Dataset:   {}", self.field_value("Dataset")),
            format!("LR:        {}", self.field_value("Learning Rate")),
            format!("Batch:     {}", self.field_value("Batch Size")),
            format!("Epochs:    {}", self.field_value("Epochs")),
            format!("LoRA r:    {}", self.field_value("LoRA Rank")),
            format!("Quant:     {}", self.field_value("Quantization")),
            String::new(),
            "Proceed?".into(),
        ]
    }

    pub fn output_dir(&self) -> PathBuf {
        PathBuf::from(self.field_value("Output Dir"))
    }

    pub fn build_cli_args(&self, subcommand: &str) -> Vec<String> {
        let mut args = vec![subcommand.to_string()];

        args.extend(["--model".into(), self.field_value("Model")]);
        args.extend(["--dataset".into(), self.field_value("Dataset")]);
        args.extend(["--output".into(), self.field_value("Output Dir")]);
        args.extend(["--learning-rate".into(), self.field_value("Learning Rate")]);
        args.extend(["--batch-size".into(), self.field_value("Batch Size")]);
        args.extend(["--epochs".into(), self.field_value("Epochs")]);
        args.extend(["--max-seq-len".into(), self.field_value("Max Seq Len")]);
        args.extend([
            "--gradient-accumulation-steps".into(),
            self.field_value("Grad Accum Steps"),
        ]);
        args.extend(["--max-grad-norm".into(), self.field_value("Max Grad Norm")]);
        args.extend(["--lora-r".into(), self.field_value("LoRA Rank")]);
        args.extend(["--lora-alpha".into(), self.field_value("LoRA Alpha")]);

        let quant = self.field_value("Quantization").to_lowercase();
        if quant != "none" {
            args.extend(["--quantization".into(), quant]);
        }

        let eval = self.field_value("Eval Dataset");
        if eval != "(none)" && !eval.is_empty() {
            args.extend(["--eval-dataset".into(), eval]);
        }

        if self.field_value("Flash Attention") == "Disabled" {
            args.push("--no-flash-attention".into());
        }
        if self.field_value("Fused Optimizer") == "Disabled" {
            args.push("--no-metal-fused-optimizer".into());
        }
        if self.field_value("JIT Compilation") == "Disabled" {
            args.push("--no-jit-compilation".into());
        }
        if self.field_value("Sequence Packing") == "Disabled" {
            args.push("--no-sequence-packing".into());
        }
        if self.field_value("ANE") == "Disabled" {
            args.push("--no-ane".into());
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

impl TrainingTab {
    /// Render the full training tab with embedded dashboard metrics.
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
            .title(" Configuration ")
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
        let mut _flat_to_field: Vec<Option<usize>> = Vec::new();

        for (i, field) in self.fields.iter().enumerate() {
            if current_section != Some(&field.section) {
                current_section = Some(&field.section);
                items.push(ListItem::new(Line::from(Span::styled(
                    format!("  --- {} ---", field.section),
                    THEME.text_muted,
                ))));
                _flat_to_field.push(None);
            }
            let selected = i == self.field_idx;
            items.push(ListItem::new(field.render_line(key_width, selected)));
            _flat_to_field.push(Some(i));
        }

        let list = List::new(items)
            .block(block)
            .highlight_style(THEME.table_selected);

        ratatui::widgets::StatefulWidget::render(list, area, buf, &mut self.list_state);
    }
}

/// Shared status panel renderer with embedded dashboard metrics.
/// Used by Training, Distillation, and GRPO tabs.
pub fn render_status_with_metrics(
    status: &TrainingStatus,
    samples: &[MetricSample],
    _throughput: &[u64],
    area: Rect,
    buf: &mut Buffer,
) {
    let block = Block::default()
        .title(match status {
            TrainingStatus::Running { .. } => " Training Monitor ",
            TrainingStatus::Completed { .. } => " Training Complete ",
            TrainingStatus::Failed(_) => " Training Failed ",
            TrainingStatus::Idle => " Status ",
        })
        .title_style(match status {
            TrainingStatus::Running { .. } => THEME.status_running,
            TrainingStatus::Completed { .. } => THEME.status_success,
            TrainingStatus::Failed(_) => THEME.status_error,
            TrainingStatus::Idle => THEME.block_title,
        })
        .borders(Borders::ALL)
        .border_style(match status {
            TrainingStatus::Running { .. } => THEME.block_focused,
            _ => THEME.block,
        });
    let inner = block.inner(area);
    block.render(area, buf);

    match status {
        TrainingStatus::Idle => {
            let lines = vec![
                Line::from(""),
                Line::from(Span::styled("  Status: Idle", THEME.status_idle)),
                Line::from(""),
                Line::from(Span::styled(
                    "  Navigate with j/k, Edit with Enter",
                    THEME.text_dim,
                )),
                Line::from(Span::styled("  Press S to start", THEME.text_dim)),
                Line::from(""),
                Line::from(Span::styled("  Toggles cycle on Enter", THEME.text_muted)),
                Line::from(Span::styled(
                    "  Pickers open a selection dialog",
                    THEME.text_muted,
                )),
            ];
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .render(inner, buf);
        }
        TrainingStatus::Running {
            step,
            epoch,
            total_epochs,
            total_steps,
            loss: _,
        } => {
            // Split into: stats | sparkline | timing
            let [stats_area, spark_area, timing_area, hint_area] = Layout::vertical([
                Constraint::Length(10),
                Constraint::Length(5),
                Constraint::Fill(1),
                Constraint::Length(1),
            ])
            .areas(inner);

            // --- Stats ---
            let last = samples.last();
            let loss_trend = if samples.len() >= 10 {
                let recent: f64 = samples[samples.len() - 5..]
                    .iter()
                    .map(|s| s.loss)
                    .sum::<f64>()
                    / 5.0;
                let prev: f64 = samples[samples.len() - 10..samples.len() - 5]
                    .iter()
                    .map(|s| s.loss)
                    .sum::<f64>()
                    / 5.0;
                if recent < prev * 0.99 {
                    " (decreasing)"
                } else if recent > prev * 1.01 {
                    " (increasing)"
                } else {
                    " (plateau)"
                }
            } else {
                ""
            };

            // Epoch display
            let epoch_display = if *total_epochs > 0 {
                format!("Epoch {}/{}", epoch + 1, total_epochs)
            } else {
                format!("Epoch {}", epoch + 1)
            };

            let step_display = if *total_steps > 0 {
                let pct = *step as f64 / *total_steps as f64 * 100.0;
                format!("{step}/{total_steps} ({pct:.1}%)")
            } else {
                format!("step {step}")
            };

            let mut stat_lines = vec![
                Line::from(vec![
                    Span::styled("  ", THEME.text),
                    Span::styled(&epoch_display, THEME.status_running),
                    Span::styled("  |  ", THEME.text_dim),
                    Span::styled("Running", THEME.status_running),
                ]),
                Line::from(""),
            ];

            // Progress gauge (only if total_steps known)
            if *total_steps > 0 {
                let gauge_area = Rect {
                    x: stats_area.x + 1,
                    y: stats_area.y + 2,
                    width: stats_area.width.saturating_sub(2),
                    height: 1,
                };
                let ratio = (*step as f64 / *total_steps as f64).clamp(0.0, 1.0);
                Gauge::default()
                    .gauge_style(palette::CHART_2)
                    .ratio(ratio)
                    .label(Span::styled(step_display.clone(), THEME.text))
                    .render(gauge_area, buf);
                stat_lines.push(Line::from("")); // placeholder for gauge row
            } else {
                stat_lines.push(Line::from(vec![
                    Span::styled("  Step:     ", THEME.kv_key),
                    Span::styled(&step_display, THEME.kv_value),
                ]));
            }

            if let Some(s) = last {
                stat_lines.push(Line::from(vec![
                    Span::styled("  Loss:     ", THEME.kv_key),
                    Span::styled(format!("{:.4}", s.loss), THEME.kv_value),
                    Span::styled(loss_trend, THEME.text_dim),
                ]));
                stat_lines.push(Line::from(vec![
                    Span::styled("  LR:       ", THEME.kv_key),
                    Span::styled(format!("{:.2e}", s.lr), THEME.kv_value),
                ]));
                stat_lines.push(Line::from(vec![
                    Span::styled("  Tok/sec:  ", THEME.kv_key),
                    Span::styled(format!("{:.0}", s.tok_sec), THEME.kv_value),
                ]));
                let min_loss = samples
                    .iter()
                    .map(|s| s.loss)
                    .filter(|l| *l > 0.0)
                    .fold(f64::MAX, f64::min);
                if min_loss < f64::MAX {
                    stat_lines.push(Line::from(vec![
                        Span::styled("  Min loss: ", THEME.kv_key),
                        Span::styled(format!("{:.4}", min_loss), THEME.text_success),
                    ]));
                }
            }

            Paragraph::new(stat_lines)
                .wrap(Wrap { trim: false })
                .render(stats_area, buf);

            // --- Loss sparkline ---
            let loss_u64: Vec<u64> = samples
                .iter()
                .rev()
                .take(60)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .map(|s| {
                    // Scale to 0-100 range for sparkline
                    let min = samples.iter().map(|s| s.loss).fold(f64::MAX, f64::min);
                    let max = samples.iter().map(|s| s.loss).fold(f64::MIN, f64::max);
                    let range = (max - min).max(0.001);
                    ((s.loss - min) / range * 100.0) as u64
                })
                .collect();

            if !loss_u64.is_empty() {
                Sparkline::default()
                    .block(
                        Block::default()
                            .title(" Loss Trend ")
                            .title_style(THEME.block_title)
                            .borders(Borders::ALL)
                            .border_style(THEME.block),
                    )
                    .data(&loss_u64)
                    .style(palette::CHART_1)
                    .render(spark_area, buf);
            } else {
                Block::default()
                    .title(" Loss Trend ")
                    .title_style(THEME.block_title)
                    .borders(Borders::ALL)
                    .border_style(THEME.block)
                    .render(spark_area, buf);
            }

            // --- Timing breakdown ---
            if let Some(last) = last {
                let timing_block = Block::default()
                    .title(" Timing ")
                    .title_style(THEME.block_title)
                    .borders(Borders::ALL)
                    .border_style(THEME.block);
                let timing_inner = timing_block.inner(timing_area);
                timing_block.render(timing_area, buf);

                let total = last.total_ms.max(1.0);

                // Detect ANE vs standard training
                let ane_sum = last.ane_fwd_ms
                    + last.ane_bwd_ms
                    + last.rmsnorm_ms
                    + last.cblas_ms
                    + last.adam_ms;
                let is_ane = ane_sum > 0.0;

                if is_ane {
                    // ANE timing breakdown
                    let components = [
                        ("ANE fwd", last.ane_fwd_ms, palette::CHART_1),
                        ("ANE bwd", last.ane_bwd_ms, palette::CHART_2),
                        ("RMSNorm", last.rmsnorm_ms, palette::CHART_3),
                        ("cblas dW", last.cblas_ms, palette::CHART_4),
                        ("Adam", last.adam_ms, palette::CHART_5),
                    ];

                    let constraints: Vec<Constraint> = components
                        .iter()
                        .map(|_| Constraint::Length(1))
                        .chain(std::iter::once(Constraint::Fill(1)))
                        .collect();
                    let rows = Layout::vertical(constraints).split(timing_inner);

                    for (i, (name, ms, color)) in components.iter().enumerate() {
                        if i >= rows.len() - 1 {
                            break;
                        }
                        let ratio = (ms / total).clamp(0.0, 1.0);
                        let label = format!("{:>8}: {:5.1}ms ({:.0}%)", name, ms, ratio * 100.0);
                        Gauge::default()
                            .gauge_style(*color)
                            .ratio(ratio)
                            .label(Span::styled(label, THEME.text))
                            .render(rows[i], buf);
                    }

                    if let Some(&total_row) = rows.last() {
                        // Only show steps/min when total_ms is populated (> 1.0 guards the
                        // max(1.0) floor that prevents divide-by-zero but yields 60000).
                        let steps_per_min_label = if last.total_ms > 1.0 {
                            format!("  ({:.0} steps/min)", 60000.0 / total)
                        } else {
                            String::new()
                        };
                        Line::from(vec![
                            Span::styled("   Total: ", THEME.kv_key),
                            Span::styled(format!("{:.1}ms", total), THEME.kv_value),
                            Span::styled(steps_per_min_label, THEME.text_dim),
                        ])
                        .render(total_row, buf);
                    }
                } else {
                    // Standard MLX training — show step time + throughput
                    let rows = Layout::vertical([
                        Constraint::Length(1),
                        Constraint::Length(1),
                        Constraint::Length(1),
                        Constraint::Fill(1),
                    ])
                    .split(timing_inner);

                    // Step time gauge — only render meaningful values when total_ms is populated.
                    // total = last.total_ms.max(1.0); total_ms == 0 → total == 1.0 (no data).
                    let has_timing = last.total_ms > 1.0;
                    let ratio = if has_timing {
                        (total / 1000.0).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let label = if has_timing {
                        format!("Step time: {:.1}ms", total)
                    } else {
                        "Step time: —".to_string()
                    };
                    Gauge::default()
                        .gauge_style(palette::CHART_2)
                        .ratio(ratio)
                        .label(Span::styled(label, THEME.text))
                        .render(rows[0], buf);

                    let steps_per_min = if has_timing {
                        format!("{:.0}", 60000.0 / total)
                    } else {
                        "—".to_string()
                    };
                    Line::from(vec![
                        Span::styled(" Steps/min: ", THEME.kv_key),
                        Span::styled(steps_per_min, THEME.kv_value),
                    ])
                    .render(rows[1], buf);

                    Line::from(vec![
                        Span::styled(" Tok/sec:   ", THEME.kv_key),
                        Span::styled(format!("{:.0}", last.tok_sec), THEME.kv_value),
                    ])
                    .render(rows[2], buf);
                }
            }

            // Hint
            Line::from(Span::styled("  Press x to stop", THEME.text_dim)).render(hint_area, buf);
        }
        TrainingStatus::Completed {
            final_loss,
            total_steps,
        } => {
            // Show final stats plus the loss sparkline
            let [stats_area, spark_area] =
                Layout::vertical([Constraint::Length(8), Constraint::Fill(1)]).areas(inner);

            let min_loss = samples.iter().map(|s| s.loss).fold(f64::MAX, f64::min);
            let avg_loss = if samples.is_empty() {
                0.0
            } else {
                samples.iter().map(|s| s.loss).sum::<f64>() / samples.len() as f64
            };

            let lines = vec![
                Line::from(Span::styled("  Status: Completed", THEME.status_success)),
                Line::from(""),
                Line::from(vec![
                    Span::styled("  Steps:      ", THEME.kv_key),
                    Span::styled(total_steps.to_string(), THEME.kv_value),
                ]),
                Line::from(vec![
                    Span::styled("  Final Loss: ", THEME.kv_key),
                    Span::styled(format!("{final_loss:.4}"), THEME.text_success),
                ]),
                Line::from(vec![
                    Span::styled("  Min Loss:   ", THEME.kv_key),
                    Span::styled(
                        if min_loss < f64::MAX {
                            format!("{min_loss:.4}")
                        } else {
                            "-".into()
                        },
                        THEME.text_success,
                    ),
                ]),
                Line::from(vec![
                    Span::styled("  Avg Loss:   ", THEME.kv_key),
                    Span::styled(
                        if avg_loss > 0.0 {
                            format!("{avg_loss:.4}")
                        } else {
                            "-".into()
                        },
                        THEME.text_dim,
                    ),
                ]),
            ];
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .render(stats_area, buf);

            // Final loss curve
            let loss_u64: Vec<u64> = samples
                .iter()
                .map(|s| {
                    let min = min_loss;
                    let max = samples.iter().map(|s| s.loss).fold(f64::MIN, f64::max);
                    let range = (max - min).max(0.001);
                    ((s.loss - min) / range * 100.0) as u64
                })
                .collect();
            if !loss_u64.is_empty() {
                Sparkline::default()
                    .block(
                        Block::default()
                            .title(" Loss Curve ")
                            .title_style(THEME.block_title)
                            .borders(Borders::ALL)
                            .border_style(THEME.block),
                    )
                    .data(&loss_u64)
                    .style(palette::CHART_1)
                    .render(spark_area, buf);
            }
        }
        TrainingStatus::Failed(msg) => {
            let mut lines = vec![
                Line::from(""),
                Line::from(Span::styled("  Status: Failed", THEME.status_error)),
                Line::from(""),
            ];
            for line in msg.lines() {
                lines.push(Line::from(Span::styled(
                    format!("  {line}"),
                    THEME.text_error,
                )));
            }
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "  Check Jobs tab for full output",
                THEME.text_muted,
            )));
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .render(inner, buf);
        }
    }
}
