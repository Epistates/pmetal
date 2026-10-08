//! `pmetal grpo` — Group Relative Policy Optimization.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Grpo", subcommand = "grpo")]
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
        default_float = 0.001
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
        enum_options = ["adamw", "sgd", "lion", "adafactor"],
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
        label = "DAPO",
        group = "GRPO",
        argv = "--dapo",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub dapo: bool,

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
        label = "Disable Flash Attention",
        group = "Compute",
        argv = "--no-flash-attention",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub no_flash_attention: bool,

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
        label = "Speculative Decoding",
        group = "Compute",
        argv = "--speculative",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub speculative: bool,

    #[job(
        label = "Speculative Draft Tokens",
        group = "Compute",
        argv = "--speculative-draft-tokens",
        min = 1,
        max = 64,
        default_int = 3
    )]
    #[serde(default = "default_speculative_draft")]
    pub speculative_draft_tokens: usize,

    #[job(
        label = "Async Rewards",
        group = "Reward",
        argv = "--async-rewards",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub async_rewards: bool,

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
            dapo: false,
            reasoning_rewards: false,
            no_flash_attention: false,
            vlm: false,
            max_image_size: default_max_image_size(),
            reward_model: None,
            reward_model_max_length: default_reward_model_max_length(),
            reward_model_weight: default_reward_model_weight(),
            reward_model_template: None,
            speculative: false,
            speculative_draft_tokens: default_speculative_draft(),
            async_rewards: false,
            text_column: None,
            text_columns: None,
            column_separator: None,
            prompt_column: None,
            response_column: None,
            grpo_kv_bits: None,
            log_metrics: None,
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
fn default_num_generations() -> usize {
    8
}
fn default_beta() -> f64 {
    0.001
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
fn default_max_image_size() -> usize {
    336
}
fn default_reward_model_max_length() -> usize {
    2048
}
fn default_reward_model_weight() -> f64 {
    1.0
}
fn default_speculative_draft() -> usize {
    3
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
