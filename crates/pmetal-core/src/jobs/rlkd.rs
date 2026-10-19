//! `pmetal rlkd` — Reinforcement Learning with Knowledge Distillation.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Rlkd", subcommand = "rlkd")]
#[serde(rename_all = "snake_case")]
pub struct RlkdSpec {
    #[job(
        label = "Policy Model",
        group = "Models",
        argv = "--model",
        kind = "model_picker",
        required
    )]
    #[serde(default)]
    pub model: String,

    #[job(
        label = "Teacher Model",
        group = "Models",
        argv = "--teacher-model",
        kind = "model_picker",
        required
    )]
    #[serde(default)]
    pub teacher_model: String,

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
        default = "./output/rlkd"
    )]
    #[serde(default = "default_output")]
    pub output_dir: String,

    #[job(
        label = "Distill α (start)",
        group = "Distillation",
        argv = "--distill-alpha",
        min = 0.0,
        max = 1.0,
        default_float = 0.3
    )]
    #[serde(default = "default_distill_alpha")]
    pub distill_alpha: f32,

    #[job(
        label = "Distill α (final)",
        group = "Distillation",
        argv = "--final-alpha",
        min = 0.0,
        max = 1.0,
        default_float = 0.05
    )]
    #[serde(default = "default_final_alpha")]
    pub final_alpha: f32,

    #[job(
        label = "Anneal α",
        group = "Distillation",
        argv = "--anneal-alpha",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub anneal_alpha: bool,

    #[job(
        label = "Distill Temperature",
        group = "Distillation",
        argv = "--distill-temperature",
        min = 0.1,
        max = 20.0,
        default_float = 2.0
    )]
    #[serde(default = "default_temperature")]
    pub distill_temperature: f32,

    #[job(
        label = "Num Generations",
        group = "GRPO",
        argv = "--num-generations",
        min = 1,
        max = 1024,
        default_int = 8
    )]
    #[serde(default = "default_num_generations")]
    pub num_generations: usize,

    #[job(
        label = "KL β",
        group = "GRPO",
        argv = "--beta",
        min = 0.0,
        max = 1.0,
        default_float = 0.0
    )]
    #[serde(default = "default_beta")]
    pub beta: f64,

    #[job(
        label = "Learning Rate",
        group = "Optimization",
        argv = "--learning-rate",
        min = 1e-8,
        max = 1.0,
        default_float = 0.000005
    )]
    #[serde(default = "default_lr")]
    pub learning_rate: f64,

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
        label = "Max Seq Len",
        group = "Training",
        argv = "--max-seq-len",
        default_int = 512
    )]
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    #[job(
        label = "Max Completion Length",
        group = "GRPO",
        argv = "--max-completion-length",
        default_int = 512
    )]
    #[serde(default = "default_max_completion")]
    pub max_completion_length: usize,

    #[job(label = "Seed", group = "Training", argv = "--seed", default_int = 42)]
    #[serde(default = "default_seed")]
    pub seed: u64,

    #[job(
        label = "Reasoning Rewards",
        group = "GRPO",
        argv = "--reasoning-rewards",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub reasoning_rewards: bool,

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

impl Default for RlkdSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            teacher_model: String::new(),
            dataset: String::new(),
            output_dir: default_output(),
            distill_alpha: default_distill_alpha(),
            final_alpha: default_final_alpha(),
            anneal_alpha: false,
            distill_temperature: default_temperature(),
            num_generations: default_num_generations(),
            beta: default_beta(),
            learning_rate: default_lr(),
            epochs: default_epochs(),
            lora_r: default_lora_r(),
            lora_alpha: default_lora_alpha(),
            max_seq_len: default_max_seq_len(),
            max_completion_length: default_max_completion(),
            seed: default_seed(),
            reasoning_rewards: false,
            prompt_column: None,
            response_column: None,
            log_metrics: None,
        }
    }
}

impl RlkdSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_output() -> String {
    crate::defaults::RLKD_OUTPUT_DIR.to_string()
}
fn default_distill_alpha() -> f32 {
    0.3
}
fn default_final_alpha() -> f32 {
    0.05
}
fn default_temperature() -> f32 {
    2.0
}
fn default_num_generations() -> usize {
    8
}
fn default_beta() -> f64 {
    0.0
}
fn default_lr() -> f64 {
    5e-6
}
fn default_epochs() -> usize {
    1
}
fn default_lora_r() -> usize {
    16
}
fn default_lora_alpha() -> f32 {
    32.0
}
fn default_max_seq_len() -> usize {
    512
}
fn default_max_completion() -> usize {
    512
}
fn default_seed() -> u64 {
    42
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_round_trip() {
        let spec = RlkdSpec {
            model: "m".into(),
            teacher_model: "t".into(),
            dataset: "d".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--model".to_string()));
        assert!(argv.contains(&"--teacher-model".to_string()));
        assert!(argv.contains(&"--dataset".to_string()));
    }
}
