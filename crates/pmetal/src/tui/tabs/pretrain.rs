//! Pretrain configuration and control tab.
//!
//! Full-parameter pretraining from scratch. Fields driven by
//! [`PretrainSpec`]. Form navigation, inline edit, and rendering are
//! delegated to `FormTabState`.

use std::path::PathBuf;

use pmetal_core::JobFields as _;
use pmetal_core::jobs::PretrainSpec;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};

use crate::tui::tabs::dashboard::MetricSample;
use crate::tui::tabs::training::{TrainingStatus, render_status_with_metrics};
use crate::tui::widgets::{FormAction, FormTabState};

pub struct PretrainTab {
    pub form: FormTabState,
    pub status: TrainingStatus,
}

impl PretrainTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::from_spec_default::<PretrainSpec>(),
            status: TrainingStatus::Idle,
        }
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
        let arch = self.form.value("Architecture");
        if arch.is_empty() {
            return Err("Architecture is required.".into());
        }
        Ok(())
    }

    pub fn config_summary(&self) -> Vec<String> {
        vec![
            format!("Architecture: {}", self.form.value("Architecture")),
            format!("Shards:       {}", self.form.value("Shards (csv)")),
            format!("Seq Len:      {}", self.form.value("Seq Len")),
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

    /// Build CLI args from the form via [`PretrainSpec::to_argv`].
    pub fn build_cli_args(&self) -> Vec<String> {
        let spec = self.spec_from_form();
        let mut args = vec!["pretrain".to_string()];
        args.extend(spec.to_argv());
        args
    }

    #[allow(clippy::field_reassign_with_default)]
    fn spec_from_form(&self) -> PretrainSpec {
        let mut spec = PretrainSpec::default();
        spec.arch = self.form.value("Architecture");
        let shards = self.form.value("Shards (csv)");
        spec.shards = if shards.is_empty() {
            None
        } else {
            Some(shards)
        };
        spec.seq_len = self.form.value("Seq Len").parse().unwrap_or(spec.seq_len);
        spec.batch_size = self
            .form
            .value("Batch Size")
            .parse()
            .unwrap_or(spec.batch_size);
        spec.steps = self.form.value("Steps").parse().unwrap_or(spec.steps);
        spec.learning_rate = self
            .form
            .value("Learning Rate")
            .parse()
            .unwrap_or(spec.learning_rate);
        spec.min_lr = self.form.value("Min LR").parse().unwrap_or(spec.min_lr);
        spec.warmup_steps = self
            .form
            .value("Warmup Steps")
            .parse()
            .unwrap_or(spec.warmup_steps);
        spec.lr_schedule = {
            let v = self.form.value("LR Schedule");
            if v.is_empty() { spec.lr_schedule } else { v }
        };
        spec.weight_decay = self
            .form
            .value("Weight Decay")
            .parse()
            .unwrap_or(spec.weight_decay);
        spec.max_grad_norm = self
            .form
            .value("Max Grad Norm")
            .parse()
            .unwrap_or(spec.max_grad_norm);
        spec.eos_token_id = self
            .form
            .value("EOS Token ID")
            .parse()
            .unwrap_or(spec.eos_token_id);
        spec.output_dir = {
            let v = self.form.value("Output Dir");
            if v.is_empty() { spec.output_dir } else { v }
        };
        spec.checkpoint_every = self
            .form
            .value("Checkpoint Every")
            .parse()
            .unwrap_or(spec.checkpoint_every);
        spec.gradient_accumulation_steps = self
            .form
            .value("Grad Accum Steps")
            .parse()
            .unwrap_or(spec.gradient_accumulation_steps);
        spec.z_loss = self.form.value("MoE z-loss").parse().unwrap_or(spec.z_loss);
        spec.document_mask = self.form.value("Document Mask") == "Enabled";
        spec.seed = self.form.value("Seed").parse().unwrap_or(spec.seed);
        let model_config = self.form.value("Model Config");
        spec.model_config = if model_config.is_empty() {
            None
        } else {
            Some(model_config)
        };
        spec
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
