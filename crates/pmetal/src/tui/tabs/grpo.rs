//! GRPO (Group Relative Policy Optimization) configuration and control tab.
//!
//! Form navigation, inline edit, and rendering are delegated to
//! `FormTabState`; this module owns only the GRPO-specific CLI wiring
//! and the metric-aware status rendering.
//! Fields are driven by [`GrpoSpec::field_descriptors`].
//!
//! Note: the RLKD upgrade (switching to the `rlkd` subcommand when
//! Teacher Model is set) is preserved via the spec-from-form path.

use std::path::PathBuf;

use pmetal_core::JobFields as _;
use pmetal_core::jobs::GrpoSpec;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};

use crate::tui::tabs::dashboard::MetricSample;
use crate::tui::tabs::training::{TrainingStatus, render_status_with_metrics};
use crate::tui::widgets::{FormAction, FormTabState};

/// GRPO tab state.
pub struct GrpoTab {
    pub form: FormTabState,
    pub status: TrainingStatus,
}

impl GrpoTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::from_spec_default::<GrpoSpec>(),
            status: TrainingStatus::Idle,
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
    pub fn handle_enter(&mut self) -> Option<FormAction> {
        self.form.handle_enter()
    }
    pub fn next_param(&mut self) {
        self.form.next_param(|_| true);
    }
    pub fn prev_param(&mut self) {
        self.form.prev_param(|_| true);
    }

    // ── Setters ─────────────────────────────────────────────────────────

    pub fn set_model(&mut self, model_id: &str) {
        self.form.set_value("Model", model_id);
        let short_name = super::model_short_name(model_id);
        self.form
            .set_value("Output Dir", format!("./output/{short_name}--grpo"));
    }

    pub fn set_dataset(&mut self, path: &str) {
        self.form.set_value("Dataset", path);
    }

    // --- Config ---

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
        let mut summary = vec![
            format!("Model:       {}", self.form.value("Model")),
            format!("Dataset:     {}", self.form.value("Dataset")),
            format!("Generations: {}", self.form.value("Num Generations")),
            format!("KL β:        {}", self.form.value("KL β")),
            format!("LR:          {}", self.form.value("Learning Rate")),
        ];
        summary.push(String::new());
        summary.push("Proceed?".into());
        summary
    }

    pub fn output_dir(&self) -> PathBuf {
        PathBuf::from(self.form.value("Output Dir"))
    }

    /// Build CLI args from the form via [`GrpoSpec::to_argv`].
    /// Subcommand is always "grpo" — RLKD upgrade is a separate tab.
    pub fn build_cli_args(&self) -> Vec<String> {
        let spec = self.spec_from_form();
        let mut args = vec!["grpo".to_string()];
        args.extend(spec.to_argv());
        args
    }

    fn spec_from_form(&self) -> GrpoSpec {
        let mut spec = GrpoSpec::default();
        spec.model = self.form.value("Model");
        spec.dataset = self.form.value("Dataset");
        spec.output_dir = {
            let v = self.form.value("Output Dir");
            if v.is_empty() { spec.output_dir } else { v }
        };
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
        spec.dapo = self.form.value("DAPO") == "Enabled";
        spec.reasoning_rewards = self.form.value("Reasoning Rewards") == "Enabled";
        spec.vlm = self.form.value("VLM Mode") == "Enabled";
        spec.max_image_size = self
            .form
            .value("Max Image Size")
            .parse()
            .unwrap_or(spec.max_image_size);
        spec.async_rewards = self.form.value("Async Rewards") == "Enabled";
        let rm = self.form.value("Reward Model");
        spec.reward_model = if rm.is_empty() { None } else { Some(rm) };
        spec.reward_model_max_length = self
            .form
            .value("Reward Model Max Length")
            .parse()
            .unwrap_or(spec.reward_model_max_length);
        spec.reward_model_weight = self
            .form
            .value("Reward Model Weight")
            .parse()
            .unwrap_or(spec.reward_model_weight);
        let rm_template = self.form.value("Reward Model Template");
        spec.reward_model_template = if rm_template.is_empty() {
            None
        } else {
            Some(rm_template)
        };
        let kv_bits = self.form.value("GRPO KV Bits");
        spec.grpo_kv_bits = kv_bits.parse().ok();
        spec
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

        self.form
            .render_list(config_area, buf, "GRPO Configuration", |_| true);
        render_status_with_metrics(&self.status, samples, throughput, status_area, buf);
    }
}
