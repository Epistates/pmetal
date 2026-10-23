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
        label = "Reward Commands",
        group = "Reward",
        argv = "--reward-cmd",
        repeat,
        help = "A command that scores completions: started once, it reads one JSON request per line on stdin ({prompts, completions, plus the dataset's other columns}, one entry per completion) and answers one {\"rewards\": [...]} line on stdout. Repeat for several; each is summed with the others"
    )]
    #[serde(default)]
    pub reward_cmd: Vec<String>,

    #[job(
        label = "Reward Command Weights",
        group = "Reward",
        argv = "--reward-weight",
        repeat,
        help = "The weight of each --reward-cmd, in order (a number; 1 for a command without one)"
    )]
    #[serde(default)]
    pub reward_weight: Vec<String>,

    #[job(
        label = "Reward Command Timeout",
        group = "Reward",
        argv = "--reward-timeout",
        min = 1,
        max = 86400,
        default_int = 120,
        help = "Seconds a reward command has to answer one batch, its first included; past it the run fails"
    )]
    #[serde(default = "default_reward_timeout")]
    pub reward_timeout: u64,

    #[job(
        label = "Reward Regex",
        group = "Reward",
        argv = "--reward-regex",
        help = "A pattern the completion should match anywhere (1 when it does, 0 when not); anchor it with ^ and $ to match the whole completion"
    )]
    #[serde(default)]
    pub reward_regex: Option<String>,

    #[job(
        label = "Reward Regex Weight",
        group = "Reward",
        argv = "--reward-regex-weight",
        min = -100.0,
        max = 100.0,
        default_float = 1.0
    )]
    #[serde(default = "default_reward_model_weight")]
    pub reward_regex_weight: f64,

    #[job(
        label = "Reward JSON",
        group = "Reward",
        argv = "--reward-json",
        flag,
        default_bool = false,
        help = "Reward a completion that is a JSON document, alone or in a fenced code block"
    )]
    #[serde(default)]
    pub reward_json: bool,

    #[job(
        label = "Reward JSON Schema",
        group = "Reward",
        argv = "--reward-json-schema",
        kind = "path",
        help = "A JSON Schema file: reward a completion that is a JSON document valid against it (this includes --reward-json)"
    )]
    #[serde(default)]
    pub reward_json_schema: Option<String>,

    #[job(
        label = "Reward JSON Weight",
        group = "Reward",
        argv = "--reward-json-weight",
        min = -100.0,
        max = 100.0,
        default_float = 1.0
    )]
    #[serde(default = "default_reward_model_weight")]
    pub reward_json_weight: f64,

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
            reward_cmd: Vec::new(),
            reward_weight: Vec::new(),
            reward_timeout: default_reward_timeout(),
            reward_regex: None,
            reward_regex_weight: default_reward_model_weight(),
            reward_json: false,
            reward_json_schema: None,
            reward_json_weight: default_reward_model_weight(),
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
        let mut errs = self.validate_descriptors();
        if let Err(message) = self.reward_cmd_weights() {
            errs.push(FieldError::new("reward_weight", message));
        }
        if self.reward_cmd.iter().any(|cmd| cmd.trim().is_empty()) {
            errs.push(FieldError::new("reward_cmd", "a command is empty"));
        }
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }

    /// The weight of each `reward_cmd`, in order: `reward_weight` as numbers,
    /// and 1 for a command that has none.
    ///
    /// # Errors
    ///
    /// A weight that is not a finite number, or more weights than commands.
    pub fn reward_cmd_weights(&self) -> Result<Vec<f64>, String> {
        if self.reward_weight.len() > self.reward_cmd.len() {
            return Err(format!(
                "{} weights for {} --reward-cmd; give at most one weight per command",
                self.reward_weight.len(),
                self.reward_cmd.len()
            ));
        }
        let mut weights = self
            .reward_weight
            .iter()
            .map(|w| {
                w.trim()
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| format!("`{w}` is not a number"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        weights.resize(self.reward_cmd.len(), 1.0);
        Ok(weights)
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
fn default_reward_timeout() -> u64 {
    120
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

    #[test]
    fn reward_commands_are_repeated_flags_with_their_weights_in_order() {
        let mut spec = GrpoSpec {
            model: "m".into(),
            dataset: "d".into(),
            reward_cmd: vec!["python3 judge.py".into(), "./tests.sh".into()],
            reward_weight: vec!["0.5".into()],
            ..Default::default()
        };
        spec.normalize().unwrap();
        assert_eq!(spec.reward_cmd_weights().unwrap(), [0.5, 1.0]);
        let argv = spec.to_argv();
        let at = |flag: &str| -> Vec<&str> {
            argv.iter()
                .enumerate()
                .filter(|(_, a)| *a == flag)
                .map(|(i, _)| argv[i + 1].as_str())
                .collect()
        };
        assert_eq!(at("--reward-cmd"), ["python3 judge.py", "./tests.sh"]);
        assert_eq!(at("--reward-weight"), ["0.5"]);
    }

    #[test]
    fn reward_weights_must_be_numbers_and_no_more_than_the_commands() {
        let invalid = |cmds: &[&str], weights: &[&str]| {
            let mut spec = GrpoSpec {
                model: "m".into(),
                dataset: "d".into(),
                reward_cmd: cmds.iter().map(|c| c.to_string()).collect(),
                reward_weight: weights.iter().map(|w| w.to_string()).collect(),
                ..Default::default()
            };
            spec.normalize()
                .err()
                .is_some_and(|errs| errs.iter().any(|e| e.field == "reward_weight"))
        };
        assert!(invalid(&["a"], &["heavy"]));
        assert!(invalid(&["a"], &["inf"]));
        assert!(invalid(&["a"], &["1", "2"]));
        assert!(invalid(&[], &["1"]));
        assert!(!invalid(&["a", "b"], &["2", "-0.5"]));
        assert!(!invalid(&["a", "b"], &[]));
    }
}
