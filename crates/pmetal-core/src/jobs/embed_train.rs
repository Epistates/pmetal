//! `pmetal embed-train` — sentence embedding model training (contrastive losses).

use super::{default_lora_alpha, default_lora_r, default_seed};
use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "EmbedTrain", subcommand = "embed-train")]
#[serde(rename_all = "snake_case")]
pub struct EmbedTrainSpec {
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
        default = "./output-embed"
    )]
    #[serde(default = "default_output")]
    pub output_dir: String,

    #[job(label = "Loss", group = "Loss", argv = "--loss", kind = "enum",
          enum_options = ["info_nce", "mnrl", "triplet", "cosent", "cosine_similarity"], default = "info_nce")]
    #[serde(default = "default_loss")]
    pub loss: String,

    #[job(label = "Pooling", group = "Model", argv = "--pooling", kind = "enum",
          enum_options = ["auto", "mean", "cls", "max", "last_token", "weighted_mean"], default = "auto",
          help = "auto reads the pooling the model was trained with (its sentence-transformers configuration), mean without one")]
    #[serde(default = "default_pooling")]
    pub pooling: String,

    #[job(
        label = "Temperature",
        group = "Loss",
        argv = "--temperature",
        min = 0.001,
        max = 10.0,
        default_float = 0.05
    )]
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    #[job(
        label = "Margin",
        group = "Loss",
        argv = "--margin",
        min = 0.0,
        max = 10.0,
        default_float = 0.3
    )]
    #[serde(default = "default_margin")]
    pub margin: f32,

    #[job(
        label = "Learning Rate",
        group = "Optimization",
        argv = "--learning-rate",
        min = 1e-8,
        max = 1.0,
        default_float = 0.00002
    )]
    #[serde(default = "default_lr")]
    pub learning_rate: f64,

    #[job(
        label = "Batch Size",
        group = "Training",
        argv = "--batch-size",
        min = 1,
        max = 4096,
        default_int = 32
    )]
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    #[job(
        label = "Epochs",
        group = "Training",
        argv = "--epochs",
        min = 1,
        max = 1000,
        default_int = 3
    )]
    #[serde(default = "default_epochs")]
    pub epochs: usize,

    #[job(
        label = "Max Seq Len",
        group = "Training",
        argv = "--max-seq-len",
        default_int = 512
    )]
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

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
        label = "Disable L2 Norm",
        group = "Loss",
        argv = "--no-normalize",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub no_normalize: bool,

    #[job(
        label = "Log Every",
        group = "Output",
        argv = "--log-every",
        min = 1,
        max = 65536,
        default_int = 10
    )]
    #[serde(default = "default_log_every")]
    pub log_every: usize,

    #[job(label = "Seed", group = "Training", argv = "--seed", default_int = 42)]
    #[serde(default = "default_seed")]
    pub seed: u64,

    #[job(
        label = "LoRA",
        group = "LoRA",
        argv = "--lora",
        flag,
        default_bool = false,
        help = "Train LoRA adapters on every linear layer and save them as an adapter, instead of fine-tuning the whole encoder"
    )]
    #[serde(default)]
    pub lora: bool,

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
}

impl Default for EmbedTrainSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            dataset: String::new(),
            output_dir: default_output(),
            loss: default_loss(),
            pooling: default_pooling(),
            temperature: default_temperature(),
            margin: default_margin(),
            learning_rate: default_lr(),
            batch_size: default_batch_size(),
            epochs: default_epochs(),
            max_seq_len: default_max_seq_len(),
            weight_decay: default_weight_decay(),
            no_normalize: false,
            log_every: default_log_every(),
            seed: default_seed(),
            lora: false,
            lora_r: default_lora_r(),
            lora_alpha: default_lora_alpha(),
        }
    }
}

impl EmbedTrainSpec {
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

fn default_output() -> String {
    crate::defaults::EMBED_OUTPUT_DIR.to_string()
}
fn default_loss() -> String {
    "info_nce".to_string()
}
fn default_pooling() -> String {
    "auto".to_string()
}
fn default_temperature() -> f32 {
    0.05
}
fn default_margin() -> f32 {
    0.3
}
fn default_lr() -> f64 {
    2e-5
}
fn default_batch_size() -> usize {
    32
}
fn default_epochs() -> usize {
    3
}
fn default_max_seq_len() -> usize {
    512
}
fn default_weight_decay() -> f64 {
    0.01
}
fn default_log_every() -> usize {
    10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_round_trip() {
        let spec = EmbedTrainSpec {
            model: "m".into(),
            dataset: "d".into(),
            loss: "triplet".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--model".to_string()));
        assert!(argv.contains(&"--loss".to_string()));
    }
}
