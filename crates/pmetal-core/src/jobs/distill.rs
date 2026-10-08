//! `pmetal distill` — knowledge distillation (online / offline / progressive).

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Distill", subcommand = "distill")]
#[serde(rename_all = "snake_case")]
pub struct DistillSpec {
    #[job(
        label = "Teacher",
        group = "Models",
        argv = "--teacher",
        kind = "model_picker",
        required
    )]
    #[serde(default)]
    pub teacher: String,

    #[job(
        label = "Student",
        group = "Models",
        argv = "--student",
        kind = "model_picker",
        required
    )]
    #[serde(default)]
    pub student: String,

    #[job(
        label = "Dataset",
        group = "Data",
        argv = "--dataset",
        kind = "dataset_picker",
        required
    )]
    #[serde(default)]
    pub dataset: String,

    #[job(
        label = "Output Dir",
        group = "Output",
        argv = "--output",
        kind = "path",
        default = "./output/distilled"
    )]
    #[serde(default = "default_output")]
    pub output_dir: String,

    #[job(label = "Method", group = "Distillation", argv = "--method", kind = "enum",
          enum_options = ["online", "offline", "progressive"], default = "online")]
    #[serde(default = "default_method")]
    pub method: String,

    #[job(
        label = "Offline Shortcut",
        group = "Distillation",
        argv = "--offline",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub offline: bool,

    #[job(
        label = "Offline Cache",
        group = "Distillation",
        argv = "--offline-cache",
        kind = "path"
    )]
    #[serde(default)]
    pub offline_cache: Option<String>,

    #[job(
        label = "Generate Offline Logits",
        group = "Distillation",
        argv = "--offline-generate",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub offline_generate: bool,

    #[job(label = "Offline Compression", group = "Distillation", argv = "--offline-compression", kind = "enum",
          enum_options = ["top_k", "top_p", "fp16", "fp8", "none"], default = "top_k")]
    #[serde(default = "default_compression")]
    pub offline_compression: String,

    #[job(
        label = "Offline Top-k",
        group = "Distillation",
        argv = "--offline-top-k",
        min = 1,
        max = 65536,
        default_int = 128
    )]
    #[serde(default = "default_offline_top_k")]
    pub offline_top_k: usize,

    #[job(label = "Loss Type", group = "Distillation", argv = "--loss-type", kind = "enum",
          enum_options = ["kl_divergence", "jensen_shannon", "soft_cross_entropy", "mse_loss"], default = "kl_divergence")]
    #[serde(default = "default_loss_type")]
    pub loss_type: String,

    #[job(
        label = "Temperature",
        group = "Distillation",
        argv = "--temperature",
        min = 0.1,
        max = 20.0,
        default_float = 2.0
    )]
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    #[job(
        label = "Alpha (hard/soft)",
        group = "Distillation",
        argv = "--alpha",
        min = 0.0,
        max = 1.0,
        default_float = 0.5
    )]
    #[serde(default = "default_alpha")]
    pub alpha: f32,

    #[job(
        label = "Rationale Distillation",
        group = "Distillation",
        argv = "--rationale",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub rationale: bool,

    #[job(
        label = "Rationale Weight",
        group = "Distillation",
        argv = "--rationale-weight",
        min = 0.0,
        max = 100.0,
        default_float = 1.0
    )]
    #[serde(default = "default_rationale_weight")]
    pub rationale_weight: f32,

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
        max = 1024.0,
        default_float = 32.0
    )]
    #[serde(default = "default_lora_alpha")]
    pub lora_alpha: f32,

    #[job(
        label = "Learning Rate",
        group = "Optimization",
        argv = "--learning-rate",
        min = 1e-8,
        max = 1.0,
        default_float = 0.00002
    )]
    #[serde(default = "default_lr")]
    pub learning_rate: f32,

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
        label = "Batch Size",
        group = "Training",
        argv = "--batch-size",
        min = 1,
        max = 1024,
        default_int = 1
    )]
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    #[job(
        label = "Epochs",
        group = "Training",
        argv = "--epochs",
        min = 1,
        max = 1000,
        default_int = 1
    )]
    #[serde(default = "default_epochs")]
    pub epochs: usize,

    #[job(
        label = "Max Seq Len",
        group = "Training",
        argv = "--max-seq-len",
        default_int = 1024
    )]
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    #[job(label = "Seed", group = "Training", argv = "--seed", default_int = 42)]
    #[serde(default = "default_seed")]
    pub seed: u64,

    #[job(label = "Text Column", group = "Data", argv = "--text-column")]
    #[serde(default)]
    pub text_column: Option<String>,

    #[job(label = "Text Columns (csv)", group = "Data", argv = "--text-columns")]
    #[serde(default)]
    pub text_columns: Option<String>,

    #[job(
        label = "Column Separator",
        group = "Data",
        argv = "--column-separator"
    )]
    #[serde(default)]
    pub column_separator: Option<String>,

    #[job(label = "Prompt Column", group = "Data", argv = "--prompt-column")]
    #[serde(default)]
    pub prompt_column: Option<String>,

    #[job(label = "Response Column", group = "Data", argv = "--response-column")]
    #[serde(default)]
    pub response_column: Option<String>,

    #[job(
        label = "Log Metrics Path",
        group = "Output",
        argv = "--log-metrics",
        kind = "path"
    )]
    #[serde(default)]
    pub log_metrics: Option<String>,
}

impl Default for DistillSpec {
    fn default() -> Self {
        Self {
            teacher: String::new(),
            student: String::new(),
            dataset: String::new(),
            output_dir: default_output(),
            method: default_method(),
            offline: false,
            offline_cache: None,
            offline_generate: false,
            offline_compression: default_compression(),
            offline_top_k: default_offline_top_k(),
            loss_type: default_loss_type(),
            temperature: default_temperature(),
            alpha: default_alpha(),
            rationale: false,
            rationale_weight: default_rationale_weight(),
            lora_r: 16,
            lora_alpha: 32.0,
            learning_rate: default_lr(),
            optimizer: super::default_optimizer(),
            batch_size: 1,
            epochs: 1,
            max_seq_len: default_max_seq_len(),
            seed: 42,
            text_column: None,
            text_columns: None,
            column_separator: None,
            prompt_column: None,
            response_column: None,
            log_metrics: None,
        }
    }
}

impl DistillSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_output() -> String {
    crate::defaults::DISTILL_OUTPUT_DIR.to_string()
}
fn default_method() -> String {
    "online".to_string()
}
fn default_compression() -> String {
    "top_k".to_string()
}
fn default_offline_top_k() -> usize {
    128
}
fn default_loss_type() -> String {
    "kl_divergence".to_string()
}
fn default_temperature() -> f32 {
    2.0
}
fn default_alpha() -> f32 {
    0.5
}
fn default_rationale_weight() -> f32 {
    1.0
}
fn default_lr() -> f32 {
    2e-5
}
fn default_max_seq_len() -> usize {
    1024
}
fn default_lora_r() -> usize {
    16
}
fn default_lora_alpha() -> f32 {
    32.0
}
fn default_batch_size() -> usize {
    1
}
fn default_epochs() -> usize {
    1
}
fn default_seed() -> u64 {
    42
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_round_trip() {
        let spec = DistillSpec {
            teacher: "t".into(),
            student: "s".into(),
            dataset: "d".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--teacher".to_string()));
        assert!(argv.contains(&"--student".to_string()));
        assert!(argv.contains(&"--dataset".to_string()));
    }
}
