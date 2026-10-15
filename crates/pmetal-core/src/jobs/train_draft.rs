//! `pmetal train-draft` — train a DFlash block-diffusion draft model against a
//! frozen Qwen3 target.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "TrainDraft", subcommand = "train-draft")]
#[serde(rename_all = "snake_case")]
pub struct TrainDraftSpec {
    #[job(
        label = "Target Model",
        group = "Model",
        argv = "--target",
        kind = "model_picker",
        help = "The frozen Qwen3 target",
        required
    )]
    #[serde(default)]
    pub target: String,

    #[job(
        label = "Draft Checkpoint",
        group = "Model",
        argv = "--draft",
        kind = "path",
        help = "An existing DFlash draft to continue from; exclusive with the draft config"
    )]
    #[serde(default)]
    pub draft: Option<String>,

    #[job(
        label = "Draft Config",
        group = "Model",
        argv = "--draft-config",
        kind = "path",
        help = "DFlash config JSON used when no checkpoint is given"
    )]
    #[serde(default)]
    pub draft_config: Option<String>,

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
        default = "./draft-output"
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

    #[job(label = "Seed", group = "Training", argv = "--seed", default_int = 42)]
    #[serde(default = "default_seed")]
    pub seed: u64,
}

impl Default for TrainDraftSpec {
    fn default() -> Self {
        Self {
            target: String::new(),
            draft: None,
            draft_config: None,
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
        }
    }
}

impl TrainDraftSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let mut errs = self.validate_descriptors();
        if self.draft.is_some() && self.draft_config.is_some() {
            errs.push(FieldError::new(
                "draft_config",
                "give a draft checkpoint or a draft config, not both",
            ));
        }
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_output() -> String {
    "./draft-output".to_string()
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
fn default_seed() -> u64 {
    crate::defaults::SEED
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_target_and_shards_are_required() {
        let errs = TrainDraftSpec::default().validate_descriptors();
        assert!(errs.iter().any(|e| e.field == "target"));
        assert!(errs.iter().any(|e| e.field == "shards"));
    }

    #[test]
    fn a_checkpoint_and_a_config_together_are_refused() {
        let mut spec = TrainDraftSpec {
            target: "t".into(),
            shards: vec!["s".into()],
            draft: Some("d".into()),
            draft_config: Some("c.json".into()),
            ..Default::default()
        };
        let errs = spec.normalize().unwrap_err();
        assert!(errs.iter().any(|e| e.field == "draft_config"));
        spec.draft_config = None;
        spec.normalize().unwrap();
    }

    #[test]
    fn subcommand_and_kind() {
        assert_eq!(TrainDraftSpec::subcommand(), "train-draft");
        assert_eq!(TrainDraftSpec::job_kind(), crate::JobKind::TrainDraft);
    }
}
