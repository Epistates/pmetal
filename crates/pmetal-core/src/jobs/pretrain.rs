//! `pmetal pretrain` — full-parameter pretraining (no LoRA).

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Pretrain", subcommand = "pretrain")]
#[serde(rename_all = "snake_case")]
pub struct PretrainSpec {
    #[job(label = "Architecture", group = "Model", argv = "--arch", required)]
    #[serde(default)]
    pub arch: String,

    #[job(label = "Shards (csv)", group = "Data", argv = "--shards")]
    #[serde(default)]
    pub shards: Option<String>,

    #[job(label = "Eval Shards (csv)", group = "Data", argv = "--eval-shards")]
    #[serde(default)]
    pub eval_shards: Option<String>,

    #[job(
        label = "Seq Len",
        group = "Training",
        argv = "--seq-len",
        min = 1,
        max = 1_048_576,
        default_int = 2048
    )]
    #[serde(default = "default_seq_len")]
    pub seq_len: usize,

    #[job(
        label = "Batch Size",
        group = "Training",
        argv = "--batch-size",
        min = 1,
        max = 4096,
        default_int = 4
    )]
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    #[job(
        label = "Steps",
        group = "Training",
        argv = "--steps",
        min = 1,
        default_int = 10000
    )]
    #[serde(default = "default_steps")]
    pub steps: usize,

    #[job(
        label = "Learning Rate",
        group = "Optimization",
        argv = "--learning-rate",
        min = 1e-8,
        max = 1.0,
        default_float = 0.0003
    )]
    #[serde(default = "default_lr")]
    pub learning_rate: f32,

    #[job(
        label = "Min LR",
        group = "Optimization",
        argv = "--min-lr",
        min = 0.0,
        max = 1.0,
        default_float = 0.00001
    )]
    #[serde(default = "default_min_lr")]
    pub min_lr: f32,

    #[job(
        label = "Warmup Steps",
        group = "Optimization",
        argv = "--warmup-steps",
        default_int = 1000
    )]
    #[serde(default = "default_warmup")]
    pub warmup_steps: usize,

    #[job(label = "LR Schedule", group = "Optimization", argv = "--lr-schedule", kind = "enum",
          enum_options = ["constant", "linear", "cosine"], default = "cosine")]
    #[serde(default = "default_lr_schedule")]
    pub lr_schedule: String,

    #[job(
        label = "Weight Decay",
        group = "Optimization",
        argv = "--weight-decay",
        min = 0.0,
        max = 1.0,
        default_float = 0.1
    )]
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f32,

    #[job(
        label = "Max Grad Norm",
        group = "Optimization",
        argv = "--max-grad-norm",
        min = 0.0,
        max = 100.0,
        default_float = 1.0
    )]
    #[serde(default = "default_grad_norm")]
    pub max_grad_norm: f32,

    #[job(
        label = "EOS Token ID",
        group = "Data",
        argv = "--eos-token-id",
        default_int = 0
    )]
    #[serde(default)]
    pub eos_token_id: u32,

    #[job(
        label = "Output Dir",
        group = "Output",
        argv = "--output",
        kind = "path",
        default = "./pretrain-output"
    )]
    #[serde(default = "default_output")]
    pub output_dir: String,

    #[job(
        label = "Checkpoint Every",
        group = "Output",
        argv = "--checkpoint-every",
        default_int = 1000
    )]
    #[serde(default = "default_checkpoint_every")]
    pub checkpoint_every: usize,

    #[job(
        label = "Resume From",
        group = "Output",
        argv = "--resume",
        kind = "path"
    )]
    #[serde(default)]
    pub resume: Option<String>,

    #[job(
        label = "Model Config",
        group = "Model",
        argv = "--model-config",
        kind = "path"
    )]
    #[serde(default)]
    pub model_config: Option<String>,

    #[job(
        label = "MoE z-loss",
        group = "Loss",
        argv = "--z-loss",
        min = 0.0,
        max = 1.0,
        default_float = 0.0
    )]
    #[serde(default)]
    pub z_loss: f32,

    #[job(
        label = "Document Mask",
        group = "Data",
        argv = "--document-mask",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub document_mask: bool,

    #[job(
        label = "Grad Accum Steps",
        group = "Optimization",
        argv = "--gradient-accumulation-steps",
        min = 1,
        max = 1024,
        default_int = 1
    )]
    #[serde(default = "default_grad_accum")]
    pub gradient_accumulation_steps: usize,

    #[job(
        label = "Log Every",
        group = "Output",
        argv = "--log-every",
        default_int = 10
    )]
    #[serde(default = "default_log_every")]
    pub log_every: usize,

    #[job(
        label = "Eval Every",
        group = "Output",
        argv = "--eval-every",
        default_int = 0
    )]
    #[serde(default)]
    pub eval_every: usize,

    #[job(
        label = "Eval Batches",
        group = "Output",
        argv = "--eval-batches",
        min = 1,
        max = 65536,
        default_int = 10
    )]
    #[serde(default = "default_eval_batches")]
    pub eval_batches: usize,

    #[job(label = "Seed", group = "Training", argv = "--seed", default_int = 42)]
    #[serde(default = "default_seed")]
    pub seed: u64,
}

impl Default for PretrainSpec {
    fn default() -> Self {
        Self {
            arch: String::new(),
            shards: None,
            eval_shards: None,
            seq_len: default_seq_len(),
            batch_size: default_batch_size(),
            steps: default_steps(),
            learning_rate: default_lr(),
            min_lr: default_min_lr(),
            warmup_steps: default_warmup(),
            lr_schedule: default_lr_schedule(),
            weight_decay: default_weight_decay(),
            max_grad_norm: default_grad_norm(),
            eos_token_id: 0,
            output_dir: default_output(),
            checkpoint_every: default_checkpoint_every(),
            resume: None,
            model_config: None,
            z_loss: 0.0,
            document_mask: false,
            gradient_accumulation_steps: default_grad_accum(),
            log_every: default_log_every(),
            eval_every: 0,
            eval_batches: default_eval_batches(),
            seed: default_seed(),
        }
    }
}

impl PretrainSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_seq_len() -> usize {
    2048
}
fn default_batch_size() -> usize {
    4
}
fn default_steps() -> usize {
    10000
}
fn default_lr() -> f32 {
    3e-4
}
fn default_min_lr() -> f32 {
    1e-5
}
fn default_warmup() -> usize {
    1000
}
fn default_lr_schedule() -> String {
    "cosine".to_string()
}
fn default_weight_decay() -> f32 {
    0.1
}
fn default_grad_norm() -> f32 {
    1.0
}
fn default_output() -> String {
    crate::defaults::PRETRAIN_OUTPUT_DIR.to_string()
}
fn default_checkpoint_every() -> usize {
    1000
}
fn default_grad_accum() -> usize {
    1
}
fn default_log_every() -> usize {
    10
}
fn default_eval_batches() -> usize {
    10
}
fn default_seed() -> u64 {
    42
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_round_trip() {
        let spec = PretrainSpec {
            arch: "gpt-oss".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--arch".to_string()));
        assert!(argv.contains(&"--steps".to_string()));
    }
}
