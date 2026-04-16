//! Pretrain configuration and control tab.
//!
//! Full-parameter pretraining from scratch. Form navigation, inline edit,
//! and rendering are delegated to `FormTabState`.

use std::path::PathBuf;

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};

use crate::tui::tabs::dashboard::MetricSample;
use crate::tui::tabs::training::{TrainingStatus, render_status_with_metrics};
use crate::tui::widgets::{FieldKind, FormAction, FormField, FormTabState};

pub struct PretrainTab {
    pub form: FormTabState,
    pub status: TrainingStatus,
}

impl PretrainTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::new(Self::default_fields()),
            status: TrainingStatus::Idle,
        }
    }

    fn default_fields() -> Vec<FormField> {
        vec![
            FormField::new(
                "Architecture",
                "llama",
                FieldKind::Enum {
                    options: vec![
                        "llama".into(),
                        "qwen2".into(),
                        "qwen3".into(),
                        "qwen3.5".into(),
                        "qwen3_moe".into(),
                        "gemma".into(),
                        "mistral".into(),
                        "phi".into(),
                        "gpt-oss".into(),
                    ],
                },
                "Model",
            ),
            FormField::new("Model Config", "", FieldKind::Text, "Model"),
            FormField::new("Shard Files", "", FieldKind::Text, "Data"),
            FormField::new(
                "Seq Length",
                "2048",
                FieldKind::Integer {
                    min: 64,
                    max: 32768,
                },
                "Data",
            ),
            FormField::new(
                "Batch Size",
                "4",
                FieldKind::Integer { min: 1, max: 256 },
                "Data",
            ),
            FormField::new(
                "EOS Token ID",
                "0",
                FieldKind::Integer {
                    min: 0,
                    max: 200000,
                },
                "Data",
            ),
            FormField::new(
                "Steps",
                "10000",
                FieldKind::Integer {
                    min: 1,
                    max: 10_000_000,
                },
                "Training",
            ),
            FormField::new(
                "Learning Rate",
                "3e-4",
                FieldKind::Number {
                    min: 1e-8,
                    max: 1.0,
                },
                "Training",
            ),
            FormField::new(
                "Min LR",
                "1e-5",
                FieldKind::Number { min: 0.0, max: 1.0 },
                "Training",
            ),
            FormField::new(
                "Warmup Steps",
                "1000",
                FieldKind::Integer {
                    min: 0,
                    max: 100000,
                },
                "Training",
            ),
            FormField::new(
                "LR Schedule",
                "cosine",
                FieldKind::Enum {
                    options: vec!["cosine".into(), "linear".into(), "constant".into()],
                },
                "Training",
            ),
            FormField::new(
                "Weight Decay",
                "0.1",
                FieldKind::Number { min: 0.0, max: 1.0 },
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
                "Grad Accum Steps",
                "1",
                FieldKind::Integer { min: 1, max: 256 },
                "Training",
            ),
            FormField::new(
                "Z-Loss",
                "0.0",
                FieldKind::Number { min: 0.0, max: 1.0 },
                "Training",
            ),
            FormField::new(
                "Checkpoint Every",
                "1000",
                FieldKind::Integer {
                    min: 0,
                    max: 1_000_000,
                },
                "Output",
            ),
            FormField::new("Output Dir", "./pretrain-output", FieldKind::Text, "Output"),
            FormField::new(
                "Seed",
                "42",
                FieldKind::Integer {
                    min: 0,
                    max: i64::MAX,
                },
                "Output",
            ),
        ]
    }

    // ── Delegation to FormTabState ──

    pub fn is_editing(&self) -> bool {
        self.form.is_editing()
    }
    pub fn confirm_edit(&mut self) {
        self.form.confirm_edit();
    }
    pub fn cancel_edit(&mut self) {
        self.form.cancel_edit();
    }
    pub fn handle_edit_key(&mut self, key: crossterm::event::KeyEvent) {
        self.form.handle_edit_key(key);
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

    // ── Config ──────────────────────────────────────────────────────────

    pub fn validate_config(&self) -> Result<(), String> {
        if self.form.value("Shard Files").is_empty() {
            return Err("Shard Files path is required.".into());
        }
        Ok(())
    }

    pub fn config_summary(&self) -> Vec<String> {
        vec![
            format!("Architecture: {}", self.form.value("Architecture")),
            format!("Shards:       {}", self.form.value("Shard Files")),
            format!("Seq Length:   {}", self.form.value("Seq Length")),
            format!("Batch Size:   {}", self.form.value("Batch Size")),
            format!("Steps:        {}", self.form.value("Steps")),
            format!("Learning Rate:{}", self.form.value("Learning Rate")),
            format!("Output:       {}", self.form.value("Output Dir")),
            String::new(),
            "Proceed?".into(),
        ]
    }

    pub fn output_dir(&self) -> PathBuf {
        PathBuf::from(self.form.value("Output Dir"))
    }

    /// Build CLI args for `pmetal pretrain`.
    pub fn build_cli_args(&self) -> Vec<String> {
        let mut args = vec!["pretrain".to_string()];

        args.extend(["--arch".into(), self.form.value("Architecture")]);

        let config = self.form.value("Model Config");
        if !config.is_empty() {
            args.extend(["--model-config".into(), config]);
        }

        args.extend(["--shards".into(), self.form.value("Shard Files")]);
        args.extend(["--seq-len".into(), self.form.value("Seq Length")]);
        args.extend(["--batch-size".into(), self.form.value("Batch Size")]);
        args.extend(["--eos-token-id".into(), self.form.value("EOS Token ID")]);
        args.extend(["--steps".into(), self.form.value("Steps")]);
        args.extend(["--learning-rate".into(), self.form.value("Learning Rate")]);
        args.extend(["--min-lr".into(), self.form.value("Min LR")]);
        args.extend(["--warmup-steps".into(), self.form.value("Warmup Steps")]);
        args.extend(["--lr-schedule".into(), self.form.value("LR Schedule")]);
        args.extend(["--weight-decay".into(), self.form.value("Weight Decay")]);
        args.extend(["--max-grad-norm".into(), self.form.value("Max Grad Norm")]);
        args.extend([
            "--gradient-accumulation-steps".into(),
            self.form.value("Grad Accum Steps"),
        ]);
        args.extend(["--z-loss".into(), self.form.value("Z-Loss")]);
        args.extend([
            "--checkpoint-every".into(),
            self.form.value("Checkpoint Every"),
        ]);
        args.extend(["--output".into(), self.form.value("Output Dir")]);
        args.extend(["--seed".into(), self.form.value("Seed")]);

        args
    }
}

impl PretrainTab {
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

        self.form
            .render_list(config_area, buf, "Pretrain Configuration", |_| true);
        render_status_with_metrics(&self.status, samples, throughput, status_area, buf);
    }
}
