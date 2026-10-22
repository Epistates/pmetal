//! `pmetal grpo` — Group Relative Policy Optimization.

use super::{default_lora_alpha, default_lora_r, default_seed};
use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Grpo", subcommand = "grpo", events)]
#[serde(rename_all = "snake_case")]
pub struct GrpoSpec {
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
        required
    )]
    #[serde(default)]
    pub dataset: String,

    #[job(
        label = "Output Dir",
        group = "Output",
        argv = "--output",
        kind = "path",
        default = "./output/grpo"
    )]
    #[serde(default = "default_output")]
    pub output_dir: String,

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
        label = "Optimizer",
        group = "Optimization",
        argv = "--optimizer",
        kind = "enum",
        enum_options = ["adamw", "sgd", "lion", "adafactor", "muon", "adamw_schedule_free", "adamw_8bit"],
        help = "Lion wants a 3-10x smaller learning rate and 3-10x larger weight decay than AdamW",
        default = "adamw"
    )]
    #[serde(default = "super::default_optimizer")]
    pub optimizer: String,

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
        label = "Checkpoint Every",
        group = "Training",
        argv = "--checkpoint-every",
        min = 0,
        max = 1000000,
        default_int = 50,
        help = "Save a LoRA checkpoint every this many policy updates (0 disables interval checkpoints)"
    )]
    #[serde(default = "default_checkpoint_every")]
    pub checkpoint_every: usize,

    #[job(
        label = "DAPO",
        group = "GRPO",
        argv = "--dapo",
        help = "The DAPO recipe: token-level loss, no KL, clip-higher (0.2, 0.28), dynamic sampling, overlong penalty, 16+ completions",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub dapo: bool,

    #[job(
        label = "Loss Type",
        group = "GRPO",
        argv = "--loss-type",
        kind = "enum",
        enum_options = ["dapo", "grpo", "dr_grpo", "gspo"],
        help = "dapo: token-level; grpo: per-sequence mean; dr_grpo: constant normalizer, no std scaling; gspo: sequence-level ratio (needs num_iterations > 1)",
        default = "dapo"
    )]
    #[serde(default = "default_loss_type")]
    pub loss_type: String,

    #[job(
        label = "Updates per Batch",
        group = "GRPO",
        argv = "--num-iterations",
        help = "Optimizer updates per generation batch (μ); above 1 the policy ratio moves off 1 and clipping engages",
        min = 1,
        max = 64,
        default_int = 1
    )]
    #[serde(default = "default_num_iterations")]
    pub num_iterations: usize,

    #[job(
        label = "Reasoning Rewards",
        group = "GRPO",
        argv = "--reasoning-rewards",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub reasoning_rewards: bool,

    #[job(
        label = "VLM Mode",
        group = "GRPO",
        argv = "--vlm",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub vlm: bool,

    #[job(
        label = "Max Image Size",
        group = "GRPO",
        argv = "--max-image-size",
        default_int = 336
    )]
    #[serde(default = "default_max_image_size")]
    pub max_image_size: usize,

    #[job(
        label = "Reward Model",
        group = "Reward",
        argv = "--reward-model",
        kind = "model_picker"
    )]
    #[serde(default)]
    pub reward_model: Option<String>,

    #[job(
        label = "Reward Model Max Length",
        group = "Reward",
        argv = "--reward-model-max-length",
        default_int = 2048
    )]
    #[serde(default = "default_reward_model_max_length")]
    pub reward_model_max_length: usize,

    #[job(
        label = "Reward Model Weight",
        group = "Reward",
        argv = "--reward-model-weight",
        min = 0.0,
        max = 100.0,
        default_float = 1.0
    )]
    #[serde(default = "default_reward_model_weight")]
    pub reward_model_weight: f64,

    #[job(
        label = "Reward Model Template",
        group = "Reward",
        argv = "--reward-model-template"
    )]
    #[serde(default)]
    pub reward_model_template: Option<String>,

    #[job(
        label = "Async Rewards",
        group = "Reward",
        argv = "--async-rewards",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub async_rewards: bool,

    #[job(label = "Prompt Column", group = "Data", argv = "--prompt-column")]
    #[serde(default)]
    pub prompt_column: Option<String>,

    #[job(label = "Response Column", group = "Data", argv = "--response-column")]
    #[serde(default)]
    pub response_column: Option<String>,

    #[job(
        label = "GRPO KV Bits",
        group = "Quantization",
        argv = "--grpo-kv-bits"
    )]
    #[serde(default)]
    pub grpo_kv_bits: Option<u8>,

    #[job(
        label = "Log Metrics Path",
        group = "Output",
        argv = "--log-metrics",
        kind = "path"
    )]
    #[serde(default)]
    pub log_metrics: Option<String>,

    /// Resume LoRA weights from the latest checkpoint in the output directory.
    #[job(
        label = "Resume",
        group = "Training",
        argv = "--resume",
        flag,
        default_bool = false,
        help = "Continue the LoRA weights from the latest checkpoint in the output directory"
    )]
    #[serde(default)]
    pub resume: bool,
}

impl Default for GrpoSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            dataset: String::new(),
            output_dir: default_output(),
            num_generations: default_num_generations(),
            beta: default_beta(),
            learning_rate: default_lr(),
            optimizer: super::default_optimizer(),
            epochs: default_epochs(),
            lora_r: default_lora_r(),
            lora_alpha: default_lora_alpha(),
            max_seq_len: default_max_seq_len(),
            max_completion_length: default_max_completion(),
            seed: default_seed(),
            checkpoint_every: default_checkpoint_every(),
            dapo: false,
            loss_type: default_loss_type(),
            num_iterations: default_num_iterations(),
            reasoning_rewards: false,
            vlm: false,
            max_image_size: default_max_image_size(),
            reward_model: None,
            reward_model_max_length: default_reward_model_max_length(),
            reward_model_weight: default_reward_model_weight(),
            reward_model_template: None,
            async_rewards: false,
            prompt_column: None,
            response_column: None,
            grpo_kv_bits: None,
            log_metrics: None,
            resume: false,
        }
    }
}

impl GrpoSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_output() -> String {
    crate::defaults::GRPO_OUTPUT_DIR.to_string()
}
fn default_loss_type() -> String {
    "dapo".to_string()
}
fn default_num_iterations() -> usize {
    1
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
    crate::defaults::EPOCHS
}
fn default_max_seq_len() -> usize {
    512
}
fn default_max_completion() -> usize {
    512
}
fn default_checkpoint_every() -> usize {
    50
}
fn default_max_image_size() -> usize {
    336
}
fn default_reward_model_max_length() -> usize {
    2048
}
fn default_reward_model_weight() -> f64 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_round_trip() {
        let spec = GrpoSpec {
            model: "m".into(),
            dataset: "d".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--model".to_string()));
        assert!(argv.contains(&"--dataset".to_string()));
    }
}
