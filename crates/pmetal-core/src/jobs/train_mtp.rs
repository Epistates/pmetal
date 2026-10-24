//! `pmetal train-mtp` — train a Gemma 4 assistant or a Qwen3-Next/Qwen3.6 MTP
//! predictor against a frozen target model.

use super::default_seed;
use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "TrainMtp", subcommand = "train-mtp", events)]
#[serde(rename_all = "snake_case")]
pub struct TrainMtpSpec {
    #[job(
        label = "Target Model",
        group = "Model",
        argv = "--model",
        kind = "model_picker",
        required
    )]
    #[serde(default)]
    pub model: String,

    #[job(
        label = "Family",
        group = "Model",
        argv = "--family",
        kind = "enum",
        enum_options = ["auto", "qwen3-next", "gemma4"],
        help = "auto detects the family from the target's config",
        default = "auto"
    )]
    #[serde(default = "default_family")]
    pub family: String,

    #[job(
        label = "Init MTP Checkpoint",
        group = "Model",
        argv = "--init-mtp",
        kind = "path",
        help = "An existing assistant or MTP checkpoint to continue from"
    )]
    #[serde(default)]
    pub init_mtp: Option<String>,

    #[job(
        label = "Qwen MTP Layers",
        group = "Model",
        argv = "--mtp-layers",
        help = "Predictor layers to create when no checkpoint is given (Qwen)",
        min = 1,
        max = 64,
        default_int = 1
    )]
    #[serde(default = "default_mtp_layers")]
    pub mtp_layers: usize,

    #[job(
        label = "Gemma Assistant Layers",
        group = "Model",
        argv = "--assistant-layers",
        help = "Assistant layers to create when no checkpoint is given (Gemma)",
        min = 1,
        max = 64,
        default_int = 2
    )]
    #[serde(default = "default_assistant_layers")]
    pub assistant_layers: usize,

    #[job(
        label = "Assistant Tokens",
        group = "Model",
        argv = "--num-assistant-tokens",
        help = "Gemma generation_config.json num_assistant_tokens",
        min = 1,
        max = 64,
        default_int = 6
    )]
    #[serde(default = "default_assistant_tokens")]
    pub num_assistant_tokens: usize,

    #[job(
        label = "Shards (csv)",
        group = "Data",
        argv = "--shards",
        csv,
        required,
        help = "Tokenized shard files from `pmetal tokenize`"
    )]
    #[serde(default)]
    pub shards: Vec<String>,

    #[job(
        label = "EOS Token ID",
        group = "Data",
        argv = "--eos-token-id",
        help = "Inserted between documents by the shard reader",
        default_int = 0
    )]
    #[serde(default)]
    pub eos_token_id: u32,

    #[job(
        label = "Output Dir",
        group = "Output",
        argv = "--output",
        kind = "path",
        default = "./mtp-output"
    )]
    #[serde(default = "default_output")]
    pub output: String,

    #[job(
        label = "Seq Len",
        group = "Training",
        argv = "--seq-len",
        min = 1,
        max = 1_048_576,
        default_int = 512
    )]
    #[serde(default = "default_seq_len")]
    pub seq_len: usize,

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
        label = "Steps",
        group = "Training",
        argv = "--steps",
        min = 1,
        default_int = 1000
    )]
    #[serde(default = "default_steps")]
    pub steps: usize,

    #[job(
        label = "Learning Rate",
        group = "Optimization",
        argv = "--learning-rate",
        min = 1e-8,
        max = 1.0,
        default_float = 0.0002
    )]
    #[serde(default = "default_lr")]
    pub learning_rate: f64,

    #[job(
        label = "Min LR",
        group = "Optimization",
        argv = "--min-lr",
        min = 0.0,
        max = 1.0,
        default_float = 0.00001
    )]
    #[serde(default = "default_min_lr")]
    pub min_lr: f64,

    #[job(
        label = "Warmup Steps",
        group = "Optimization",
        argv = "--warmup-steps",
        default_int = 100
    )]
    #[serde(default = "default_warmup")]
    pub warmup_steps: usize,

    #[job(
        label = "LR Schedule",
        group = "Optimization",
        argv = "--lr-schedule",
        kind = "enum",
        enum_options = ["cosine", "linear", "constant", "cosine_with_restarts", "polynomial", "wsd"],
        default = "cosine"
    )]
    #[serde(default = "default_lr_schedule")]
    pub lr_schedule: String,

    #[job(
        label = "Weight Decay",
        group = "Optimization",
        argv = "--weight-decay",
        min = 0.0,
        max = 1.0,
        default_float = 0.01
    )]
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,

    #[job(
        label = "Max Grad Norm",
        group = "Optimization",
        argv = "--max-grad-norm",
        help = "0 disables clipping",
        min = 0.0,
        max = 1000.0,
        default_float = 1.0
    )]
    #[serde(default = "default_grad_norm")]
    pub max_grad_norm: f64,

    #[job(
        label = "Checkpoint Every",
        group = "Output",
        argv = "--checkpoint-every",
        help = "Steps between checkpoints; 0 disables them",
        default_int = 500
    )]
    #[serde(default = "default_checkpoint_every")]
    pub checkpoint_every: usize,

    #[job(
        label = "Log Every",
        group = "Output",
        argv = "--log-every",
        help = "Steps between log lines; 0 disables them",
        default_int = 10
    )]
    #[serde(default = "default_log_every")]
    pub log_every: usize,

    #[job(
        label = "Resume From",
        group = "Output",
        argv = "--resume",
        kind = "path",
        help = "A checkpoint directory, or the output directory of an earlier run (its furthest checkpoint): the weights, the optimizer state and the step are restored and the run continues to Steps"
    )]
    #[serde(default)]
    pub resume: Option<String>,

    #[job(label = "Seed", group = "Training", argv = "--seed", default_int = 42)]
    #[serde(default = "default_seed")]
    pub seed: u64,
}

impl Default for TrainMtpSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            family: default_family(),
            init_mtp: None,
            mtp_layers: default_mtp_layers(),
            assistant_layers: default_assistant_layers(),
            num_assistant_tokens: default_assistant_tokens(),
            shards: Vec::new(),
            eos_token_id: 0,
            output: default_output(),
            seq_len: default_seq_len(),
            batch_size: default_batch_size(),
            steps: default_steps(),
            learning_rate: default_lr(),
            min_lr: default_min_lr(),
            warmup_steps: default_warmup(),
            lr_schedule: default_lr_schedule(),
            weight_decay: default_weight_decay(),
            max_grad_norm: default_grad_norm(),
            checkpoint_every: default_checkpoint_every(),
            log_every: default_log_every(),
            seed: default_seed(),
            resume: None,
        }
    }
}

impl TrainMtpSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let mut errs = self.validate_descriptors();
        if self.resume.is_some() && self.init_mtp.is_some() {
            errs.push(FieldError::new(
                "init_mtp",
                "a resumed run continues from its own checkpoint's weights; give --resume or --init-mtp, not both",
            ));
        }
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_family() -> String {
    "auto".to_string()
}
fn default_mtp_layers() -> usize {
    1
}
fn default_assistant_layers() -> usize {
    2
}
fn default_assistant_tokens() -> usize {
    6
}
fn default_output() -> String {
    "./mtp-output".to_string()
}
fn default_seq_len() -> usize {
    512
}
fn default_batch_size() -> usize {
    1
}
fn default_steps() -> usize {
    1000
}
fn default_lr() -> f64 {
    2e-4
}
fn default_min_lr() -> f64 {
    1e-5
}
fn default_warmup() -> usize {
    100
}
fn default_lr_schedule() -> String {
    "cosine".to_string()
}
fn default_weight_decay() -> f64 {
    0.01
}
fn default_grad_norm() -> f64 {
    1.0
}
fn default_checkpoint_every() -> usize {
    500
}
fn default_log_every() -> usize {
    10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_carries_the_shards_as_one_csv() {
        let spec = TrainMtpSpec {
            model: "Qwen/Qwen3-Next".into(),
            shards: vec!["a.bin".into(), "b.bin".into()],
            ..Default::default()
        };
        let argv = spec.to_argv();
        let at = argv.iter().position(|a| a == "--shards").unwrap();
        assert_eq!(argv[at + 1], "a.bin,b.bin");
        assert_eq!(TrainMtpSpec::subcommand(), "train-mtp");
        assert_eq!(TrainMtpSpec::job_kind(), crate::JobKind::TrainMtp);
    }

    #[test]
    fn a_model_and_shards_are_required() {
        let errs = TrainMtpSpec::default().validate_descriptors();
        assert!(errs.iter().any(|e| e.field == "model"));
        assert!(errs.iter().any(|e| e.field == "shards"));
    }

    #[test]
    fn an_unknown_schedule_is_a_field_error() {
        let mut spec = TrainMtpSpec {
            model: "m".into(),
            shards: vec!["s".into()],
            lr_schedule: "cosin".into(),
            ..Default::default()
        };
        let errs = spec.normalize().unwrap_err();
        assert!(errs.iter().any(|e| e.field == "lr_schedule"));
    }
}
