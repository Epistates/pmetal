//! `pmetal preference` — offline preference optimization (DPO, IPO, hinge,
//! SimPO, ORPO, KTO) with LoRA.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

/// The objectives `pmetal preference --loss` accepts.
pub const PREFERENCE_LOSSES: &[&str] = &["dpo", "ipo", "hinge", "simpo", "orpo", "kto"];

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Preference", subcommand = "preference")]
#[serde(rename_all = "snake_case")]
pub struct PreferenceSpec {
    #[job(
        label = "Model",
        group = "Model",
        argv = "--model",
        kind = "model_picker",
        required
    )]
    #[serde(default)]
    pub model: String,

    #[job(
        label = "Dataset",
        group = "Data",
        argv = "--dataset",
        kind = "dataset_picker",
        required,
        help = "prompt/chosen/rejected rows (KTO: prompt/completion/label), JSONL, JSON or Parquet"
    )]
    #[serde(default)]
    pub dataset: String,

    #[job(
        label = "Output Dir",
        group = "Output",
        argv = "--output",
        kind = "path",
        default = "./output/preference"
    )]
    #[serde(default = "default_output")]
    pub output_dir: String,

    #[job(label = "Loss", group = "Objective", argv = "--loss", kind = "enum",
          enum_options = ["dpo", "ipo", "hinge", "simpo", "orpo", "kto"], default = "dpo")]
    #[serde(default = "default_loss")]
    pub loss: String,

    #[job(
        label = "β",
        group = "Objective",
        argv = "--beta",
        min = 0.0,
        max = 100.0,
        help = "Default: 2.5 for SimPO, 0.1 otherwise"
    )]
    #[serde(default)]
    pub beta: Option<f32>,

    #[job(
        label = "SimPO γ/β",
        group = "Objective",
        argv = "--simpo-gamma-ratio",
        min = 0.0,
        max = 10.0,
        default_float = 0.5,
        help = "Target reward margin over β (SimPO only)"
    )]
    #[serde(default = "default_simpo_gamma_ratio")]
    pub simpo_gamma_ratio: f32,

    #[job(
        label = "Label Smoothing",
        group = "Objective",
        argv = "--label-smoothing",
        min = 0.0,
        max = 0.49,
        default_float = 0.0,
        help = "Share of labels assumed flipped; enables Robust DPO (DPO only, 0.1 typical)"
    )]
    #[serde(default)]
    pub label_smoothing: f32,

    #[job(
        label = "Desirable Weight",
        group = "Objective",
        argv = "--desirable-weight",
        min = 0.0,
        max = 100.0,
        default_float = 1.0,
        help = "KTO only"
    )]
    #[serde(default = "default_weight")]
    pub desirable_weight: f32,

    #[job(
        label = "Undesirable Weight",
        group = "Objective",
        argv = "--undesirable-weight",
        min = 0.0,
        max = 100.0,
        default_float = 1.0,
        help = "KTO only"
    )]
    #[serde(default = "default_weight")]
    pub undesirable_weight: f32,

    #[job(
        label = "Learning Rate",
        group = "Optimization",
        argv = "--learning-rate",
        min = 1e-9,
        max = 1.0,
        default_float = 0.00001
    )]
    #[serde(default = "default_lr")]
    pub learning_rate: f64,

    #[job(
        label = "Batch Size",
        group = "Optimization",
        argv = "--batch-size",
        min = 1,
        max = 1024,
        default_int = 2
    )]
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    #[job(
        label = "Gradient Accumulation",
        group = "Optimization",
        argv = "--gradient-accumulation-steps",
        min = 1,
        max = 1024,
        default_int = 8
    )]
    #[serde(default = "default_grad_accum")]
    pub gradient_accumulation_steps: usize,

    #[job(
        label = "Epochs",
        group = "Optimization",
        argv = "--epochs",
        min = 1,
        max = 1000,
        default_int = 1
    )]
    #[serde(default = "default_epochs")]
    pub epochs: usize,

    #[job(
        label = "Max Steps",
        group = "Optimization",
        argv = "--max-steps",
        min = 1,
        help = "Stop after this many optimizer steps"
    )]
    #[serde(default)]
    pub max_steps: Option<usize>,

    #[job(
        label = "Warmup Ratio",
        group = "Optimization",
        argv = "--warmup-ratio",
        min = 0.0,
        max = 1.0,
        default_float = 0.1
    )]
    #[serde(default = "default_warmup_ratio")]
    pub warmup_ratio: f64,

    #[job(
        label = "Max Grad Norm",
        group = "Optimization",
        argv = "--max-grad-norm",
        min = 0.0,
        max = 1000.0,
        default_float = 1.0
    )]
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f64,

    #[job(
        label = "Weight Decay",
        group = "Optimization",
        argv = "--weight-decay",
        min = 0.0,
        max = 1.0,
        default_float = 0.0
    )]
    #[serde(default)]
    pub weight_decay: f64,

    #[job(
        label = "Optimizer",
        group = "Optimization",
        argv = "--optimizer",
        kind = "enum",
        enum_options = ["adamw", "sgd", "lion", "adafactor"],
        help = "Lion wants a 3-10x smaller learning rate and 3-10x larger weight decay than AdamW",
        default = "adamw"
    )]
    #[serde(default = "super::default_optimizer")]
    pub optimizer: String,

    #[job(
        label = "LoRA r",
        group = "LoRA",
        argv = "--lora-r",
        min = 1,
        max = 1024,
        default_int = 16
    )]
    #[serde(default = "default_lora_r")]
    pub lora_r: usize,

    #[job(
        label = "LoRA α",
        group = "LoRA",
        argv = "--lora-alpha",
        min = 1.0,
        max = 4096.0,
        default_float = 32.0
    )]
    #[serde(default = "default_lora_alpha")]
    pub lora_alpha: f32,

    #[job(
        label = "Max Prompt Length",
        group = "Data",
        argv = "--max-prompt-length",
        min = 1,
        default_int = 512
    )]
    #[serde(default = "default_max_prompt_length")]
    pub max_prompt_length: usize,

    #[job(
        label = "Max Length",
        group = "Data",
        argv = "--max-length",
        min = 2,
        default_int = 1024,
        help = "Prompt plus completion"
    )]
    #[serde(default = "default_max_length")]
    pub max_length: usize,

    #[job(
        label = "Seed",
        group = "Optimization",
        argv = "--seed",
        default_int = 42
    )]
    #[serde(default = "default_seed")]
    pub seed: u64,

    #[job(
        label = "Log Metrics Path",
        group = "Output",
        argv = "--log-metrics",
        kind = "path"
    )]
    #[serde(default)]
    pub log_metrics: Option<String>,
}

impl Default for PreferenceSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            dataset: String::new(),
            output_dir: default_output(),
            loss: default_loss(),
            beta: None,
            simpo_gamma_ratio: default_simpo_gamma_ratio(),
            label_smoothing: 0.0,
            desirable_weight: default_weight(),
            undesirable_weight: default_weight(),
            learning_rate: default_lr(),
            batch_size: default_batch_size(),
            gradient_accumulation_steps: default_grad_accum(),
            epochs: default_epochs(),
            max_steps: None,
            warmup_ratio: default_warmup_ratio(),
            max_grad_norm: default_max_grad_norm(),
            weight_decay: 0.0,
            optimizer: super::default_optimizer(),
            lora_r: default_lora_r(),
            lora_alpha: default_lora_alpha(),
            max_prompt_length: default_max_prompt_length(),
            max_length: default_max_length(),
            seed: default_seed(),
            log_metrics: None,
        }
    }
}

impl PreferenceSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        self.loss = self.loss.trim().to_ascii_lowercase();
        let mut errs = self.validate_descriptors();
        if !PREFERENCE_LOSSES.contains(&self.loss.as_str()) {
            errs.push(FieldError::new(
                "loss",
                format!(
                    "unknown loss '{}'; expected one of {}",
                    self.loss,
                    PREFERENCE_LOSSES.join(", ")
                ),
            ));
        }
        if self.max_prompt_length >= self.max_length {
            errs.push(FieldError::new(
                "max_prompt_length",
                format!(
                    "--max-prompt-length ({}) must be below --max-length ({}) to leave room for the completion",
                    self.max_prompt_length, self.max_length
                ),
            ));
        }
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }

    /// β for the chosen loss: the explicit value, or the loss's default.
    pub fn effective_beta(&self) -> f32 {
        self.beta
            .unwrap_or(if self.loss == "simpo" { 2.5 } else { 0.1 })
    }
}

fn default_output() -> String {
    crate::defaults::PREFERENCE_OUTPUT_DIR.to_string()
}
fn default_loss() -> String {
    "dpo".to_string()
}
fn default_simpo_gamma_ratio() -> f32 {
    0.5
}
fn default_weight() -> f32 {
    1.0
}
fn default_lr() -> f64 {
    1e-5
}
fn default_batch_size() -> usize {
    2
}
fn default_grad_accum() -> usize {
    8
}
fn default_epochs() -> usize {
    1
}
fn default_warmup_ratio() -> f64 {
    0.1
}
fn default_max_grad_norm() -> f64 {
    1.0
}
fn default_lora_r() -> usize {
    16
}
fn default_lora_alpha() -> f32 {
    32.0
}
fn default_max_prompt_length() -> usize {
    512
}
fn default_max_length() -> usize {
    1024
}
fn default_seed() -> u64 {
    42
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> PreferenceSpec {
        PreferenceSpec {
            model: "m".into(),
            dataset: "d.jsonl".into(),
            ..Default::default()
        }
    }

    #[test]
    fn argv_round_trip() {
        let mut s = spec();
        s.loss = "simpo".into();
        s.beta = Some(2.5);
        let argv = s.to_argv();
        assert!(argv.windows(2).any(|w| w == ["--loss", "simpo"]));
        assert!(argv.windows(2).any(|w| w == ["--beta", "2.5"]));
        assert!(!spec().to_argv().contains(&"--beta".to_string()));
    }

    #[test]
    fn normalize_checks_the_loss_and_lengths() {
        let mut s = spec();
        s.loss = " DPO ".into();
        assert!(s.normalize().is_ok());
        assert_eq!(s.loss, "dpo");

        s.loss = "ppo".into();
        let errs = s.normalize().unwrap_err();
        assert!(errs.iter().any(|e| e.field == "loss"));

        let mut s = spec();
        s.max_prompt_length = 2048;
        assert!(s.normalize().is_err());
    }

    #[test]
    fn beta_defaults_per_loss() {
        let mut s = spec();
        assert_eq!(s.effective_beta(), 0.1);
        s.loss = "simpo".into();
        assert_eq!(s.effective_beta(), 2.5);
        s.beta = Some(0.5);
        assert_eq!(s.effective_beta(), 0.5);
    }
}
