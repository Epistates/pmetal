//! `pmetal train` — supervised fine-tuning / LoRA / QLoRA / ANE.

use crate::{
    DistributedCompression, DistributedTrainingConfig, FieldError, JobFields, LoraConfig,
    TrainingConfig, defaults,
};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

/// Spec for `pmetal train`.
///
/// Holds the user-facing input fields. Conversion to the trainer's internal
/// `TrainingJobConfig` happens at the consumer boundary.
#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Train", subcommand = "train")]
#[serde(rename_all = "snake_case")]
pub struct TrainSpec {
    // -- Model & data ---------------------------------------------------------
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
        label = "Eval Dataset",
        group = "Data",
        argv = "--eval-dataset",
        kind = "dataset_picker"
    )]
    #[serde(default)]
    pub eval_dataset: Option<String>,

    #[job(
        label = "Output Dir",
        group = "Output",
        argv = "--output",
        kind = "path",
        default = "./output"
    )]
    #[serde(default = "default_train_output_dir")]
    pub output_dir: String,

    // -- Optimization ---------------------------------------------------------
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
        label = "Embedding LR",
        group = "Optimization",
        argv = "--embedding-lr",
        min = 1e-8,
        max = 1.0
    )]
    #[serde(default)]
    pub embedding_lr: Option<f64>,

    #[job(
        label = "Label Smoothing",
        group = "Loss",
        argv = "--label-smoothing",
        min = 0.0,
        max = 0.99,
        default_float = 0.0
    )]
    #[serde(default)]
    pub label_smoothing: f64,

    #[job(
        label = "Z-Loss",
        group = "Loss",
        argv = "--z-loss",
        min = 0.0,
        max = 1.0,
        default_float = 0.0
    )]
    #[serde(default)]
    pub z_loss: f64,

    #[job(
        label = "NEFTune Alpha",
        group = "Loss",
        argv = "--neftune-alpha",
        min = 0.1,
        max = 100.0
    )]
    #[serde(default)]
    pub neftune_alpha: Option<f64>,

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
        help = "0 = auto-detect from model config",
        default_int = 0
    )]
    #[serde(default)]
    pub max_seq_len: usize,

    #[job(
        label = "Grad Accum Steps",
        group = "Training",
        argv = "--gradient-accumulation-steps",
        min = 1,
        max = 1024,
        default_int = 4
    )]
    #[serde(default = "default_grad_accum")]
    pub gradient_accumulation_steps: usize,

    #[job(
        label = "Disable Gradient Checkpointing",
        group = "Optimization",
        argv = "--no-gradient-checkpointing",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub no_gradient_checkpointing: bool,

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
        label = "Warmup Steps",
        group = "Optimization",
        argv = "--warmup-steps",
        default_int = 0
    )]
    #[serde(default)]
    pub warmup_steps: usize,

    #[job(
        label = "Warmup Ratio",
        group = "Optimization",
        argv = "--warmup-ratio",
        min = 0.0,
        max = 1.0,
        help = "Share of the optimizer steps spent warming up; replaces Warmup Steps when set"
    )]
    #[serde(default)]
    pub warmup_ratio: Option<f64>,

    #[job(
        label = "Min LR",
        group = "Optimization",
        argv = "--min-lr",
        min = 0.0,
        max = 1.0,
        help = "Floor the decaying schedule stops at (default 0)"
    )]
    #[serde(default)]
    pub min_lr: Option<f64>,

    #[job(
        label = "WSD Stable Ratio",
        group = "Optimization",
        argv = "--wsd-stable-ratio",
        min = 0.0,
        max = 1.0,
        help = "Share of the steps the wsd schedule holds the peak rate for (default 0.7)"
    )]
    #[serde(default)]
    pub wsd_stable_ratio: Option<f64>,

    #[job(
        label = "WSD Decay Style",
        group = "Optimization",
        argv = "--wsd-decay-style",
        kind = "enum",
        enum_options = ["linear", "cosine", "one_minus_sqrt"],
        help = "Shape of the wsd schedule's final decay (default linear)"
    )]
    #[serde(default)]
    pub wsd_decay_style: Option<String>,

    #[job(
        label = "Cosine Cycles",
        group = "Optimization",
        argv = "--cosine-num-restarts",
        min = 1,
        max = 1000,
        help = "Cycles of the cosine_with_restarts schedule (default 1)"
    )]
    #[serde(default)]
    pub cosine_num_restarts: Option<usize>,

    #[job(
        label = "Polynomial Power",
        group = "Optimization",
        argv = "--polynomial-power",
        min = 0.0,
        max = 100.0,
        help = "Exponent of the polynomial schedule (default 1, linear)"
    )]
    #[serde(default)]
    pub polynomial_power: Option<f64>,

    #[job(
        label = "Max Steps",
        group = "Training",
        argv = "--max-steps",
        min = 1,
        max = 1000000000,
        help = "Stop after this many optimizer steps, whatever the epochs say"
    )]
    #[serde(default)]
    pub max_steps: Option<usize>,

    #[job(
        label = "Logging Steps",
        group = "Training",
        argv = "--logging-steps",
        min = 1,
        max = 1000000,
        default_int = 10,
        help = "Log the loss every this many optimizer steps"
    )]
    #[serde(default = "default_logging_steps")]
    pub logging_steps: usize,

    #[job(
        label = "Eval Steps",
        group = "Training",
        argv = "--eval-steps",
        min = 1,
        max = 1000000,
        help = "Evaluate every this many optimizer steps (needs an eval dataset; default 100)"
    )]
    #[serde(default)]
    pub eval_steps: Option<usize>,

    #[job(
        label = "Save Steps",
        group = "Training",
        argv = "--save-steps",
        min = 1,
        max = 1000000,
        help = "Write a checkpoint every this many optimizer steps (default 500)"
    )]
    #[serde(default)]
    pub save_steps: Option<usize>,

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
        label = "LR Schedule",
        group = "Optimization",
        argv = "--lr-schedule",
        kind = "enum",
        enum_options = ["cosine", "linear", "constant", "cosine_with_restarts", "polynomial", "wsd"],
        default = "cosine"
    )]
    #[serde(default = "default_lr_schedule")]
    pub lr_schedule: String,

    #[job(label = "Seed", group = "Training", argv = "--seed", default_int = 42)]
    #[serde(default = "default_seed")]
    pub seed: u64,

    #[job(
        label = "Loss Scale",
        group = "Training",
        argv = "--loss-scale",
        min = 0.0,
        max = 1e9,
        default_float = 1.0
    )]
    #[serde(default = "default_loss_scale")]
    pub loss_scale: f32,

    // -- LoRA -----------------------------------------------------------------
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
        label = "DoRA",
        group = "LoRA",
        argv = "--dora",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub dora: bool,

    #[job(
        label = "RSLoRA",
        group = "LoRA",
        argv = "--rslora",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub rslora: bool,

    #[job(
        label = "LoRA Dropout",
        group = "LoRA",
        argv = "--lora-dropout",
        min = 0.0,
        max = 1.0,
        default_float = 0.0
    )]
    #[serde(default)]
    pub lora_dropout: f32,

    #[job(
        label = "LoRA Targets (csv)",
        group = "LoRA",
        argv = "--lora-targets",
        csv,
        help = "Projections to adapt, e.g. q_proj,v_proj; empty adapts every linear layer. MoE routers never are"
    )]
    #[serde(default)]
    pub lora_targets: Vec<String>,

    #[job(
        label = "Quantization",
        group = "LoRA",
        argv = "--quantization",
        kind = "enum",
        enum_options = ["none", "nf4", "fp4", "int8"]
    )]
    #[serde(default)]
    pub quantization: Option<String>,

    #[job(
        label = "Quant Block Size",
        group = "LoRA",
        argv = "--quant-block-size",
        min = 1,
        max = 4096,
        default_int = 64
    )]
    #[serde(default = "default_quant_block_size")]
    pub quant_block_size: usize,

    #[job(
        label = "Double Quant",
        group = "LoRA",
        argv = "--double-quant",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub double_quant: bool,

    // -- Dataset columns ------------------------------------------------------
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

    // -- Compute / dispatch flags --------------------------------------------
    #[job(
        label = "Disable Sequence Packing",
        group = "Compute",
        argv = "--no-sequence-packing",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub no_sequence_packing: bool,

    #[job(
        label = "Cut Cross-Entropy",
        group = "Compute",
        argv = "--cut-cross-entropy",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub cut_cross_entropy: bool,

    #[job(
        label = "Disable Adaptive LR",
        group = "Optimization",
        argv = "--no-adaptive-lr",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub no_adaptive_lr: bool,

    #[job(
        label = "Use ANE",
        group = "Compute",
        argv = "--ane",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub ane: bool,

    #[job(
        label = "Pack Max Seq Len",
        group = "Compute",
        argv = "--pack-max-seq-len"
    )]
    #[serde(default)]
    pub pack_max_seq_len: Option<usize>,

    #[job(
        label = "Distributed Peers",
        group = "Distributed",
        argv = "--distributed-peers",
        help = "Comma-separated peer addresses"
    )]
    #[serde(default)]
    pub distributed_peers: Option<String>,

    #[job(
        label = "Distributed Auto",
        group = "Distributed",
        argv = "--distributed-auto",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub distributed_auto: bool,

    #[job(label = "Compression", group = "Distributed", argv = "--compression-strategy", kind = "enum",
          enum_options = ["none", "topk", "fp16", "random"], default = "none")]
    #[serde(default = "default_compression_strategy")]
    pub compression_strategy: String,

    // -- CLI-only -------------------------------------------------------------
    /// YAML config file to load defaults from.
    #[job(skip_descriptor, argv = "--config")]
    #[serde(default)]
    pub config_path: Option<String>,

    /// Resume from latest checkpoint.
    #[job(skip_descriptor, argv = "--resume", flag, default_bool = false)]
    #[serde(default)]
    pub resume: bool,

    /// Write per-step metrics as JSONL to this path.
    #[job(skip_descriptor, argv = "--log-metrics")]
    #[serde(default)]
    pub log_metrics: Option<String>,
}

impl Default for TrainSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            dataset: String::new(),
            eval_dataset: None,
            output_dir: default_train_output_dir(),
            learning_rate: default_lr(),
            embedding_lr: None,
            label_smoothing: 0.0,
            z_loss: 0.0,
            neftune_alpha: None,
            batch_size: default_batch_size(),
            epochs: default_epochs(),
            max_seq_len: 0,
            gradient_accumulation_steps: default_grad_accum(),
            no_gradient_checkpointing: false,
            max_grad_norm: default_max_grad_norm(),
            warmup_steps: 0,
            warmup_ratio: None,
            min_lr: None,
            wsd_stable_ratio: None,
            wsd_decay_style: None,
            cosine_num_restarts: None,
            polynomial_power: None,
            max_steps: None,
            logging_steps: default_logging_steps(),
            eval_steps: None,
            save_steps: None,
            weight_decay: default_weight_decay(),
            optimizer: super::default_optimizer(),
            lr_schedule: default_lr_schedule(),
            seed: default_seed(),
            loss_scale: default_loss_scale(),
            lora_r: default_lora_r(),
            lora_alpha: default_lora_alpha(),
            dora: false,
            rslora: false,
            lora_dropout: 0.0,
            lora_targets: Vec::new(),
            quantization: None,
            quant_block_size: default_quant_block_size(),
            double_quant: false,
            text_column: None,
            text_columns: None,
            column_separator: None,
            prompt_column: None,
            response_column: None,
            no_sequence_packing: false,
            cut_cross_entropy: false,
            no_adaptive_lr: false,
            ane: false,
            pack_max_seq_len: None,
            distributed_peers: None,
            distributed_auto: false,
            compression_strategy: default_compression_strategy(),
            config_path: None,
            resume: false,
            log_metrics: None,
        }
    }
}

impl TrainSpec {
    /// Apply cross-field defaults / sentinels and run descriptor validation.
    ///
    /// With a config file (`--config`) the model and dataset may come from
    /// the file, so their being empty is not an error here; the trainer checks
    /// them once the file's values are merged in.
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let mut errs = self.validate_descriptors();
        if self.config_path.is_some() {
            errs.retain(|e| {
                !(matches!(e.field.as_str(), "model" | "dataset") && e.message == "required")
            });
        }
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }

    /// The adapter this spec asks for. Every surface that runs a spec builds
    /// its `LoraConfig` here, so a field (`dora`, `rslora`, `lora_dropout`)
    /// cannot be honoured by one surface and dropped by another.
    pub fn lora_config(&self) -> LoraConfig {
        LoraConfig {
            r: self.lora_r,
            alpha: self.lora_alpha,
            dropout: self.lora_dropout,
            use_rslora: self.rslora,
            use_dora: self.dora,
            target_modules: self.lora_targets.clone(),
            ..Default::default()
        }
    }

    /// The optimisation settings this spec asks for. An optimizer or schedule
    /// name that is not a known one is an error rather than a default.
    pub fn training_config(&self) -> Result<TrainingConfig, String> {
        Ok(TrainingConfig {
            learning_rate: self.learning_rate,
            embedding_learning_rate: self.embedding_lr,
            label_smoothing: self.label_smoothing,
            z_loss: self.z_loss,
            neftune_noise_alpha: self.neftune_alpha,
            batch_size: self.batch_size,
            gradient_accumulation_steps: self.gradient_accumulation_steps,
            num_epochs: self.epochs,
            max_steps: self.max_steps,
            warmup_steps: self.warmup_steps,
            warmup_ratio: self.warmup_ratio,
            weight_decay: self.weight_decay,
            max_grad_norm: self.max_grad_norm,
            lr_scheduler: super::parse_lr_scheduler(&self.lr_schedule)?,
            min_lr: self.min_lr,
            wsd_stable_ratio: self.wsd_stable_ratio,
            wsd_decay_style: self
                .wsd_decay_style
                .as_deref()
                .map(str::parse)
                .transpose()?,
            cosine_num_restarts: self.cosine_num_restarts,
            polynomial_power: self.polynomial_power,
            logging_steps: self.logging_steps,
            eval_steps: self.eval_steps,
            save_steps: self.save_steps,
            optimizer: super::parse_optimizer(&self.optimizer)?,
            seed: self.seed,
            output_dir: self.output_dir.clone(),
            max_seq_len: self.max_seq_len,
            use_packing: !self.no_sequence_packing,
            ..Default::default()
        })
    }

    /// The multi-Mac settings, or `None` when neither peers nor discovery is
    /// asked for.
    pub fn distributed_config(&self) -> Result<Option<DistributedTrainingConfig>, String> {
        let peers: Vec<String> = self
            .distributed_peers
            .as_deref()
            .map(|csv| {
                csv.split(',')
                    .map(str::trim)
                    .filter(|peer| !peer.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        let compression: DistributedCompression = self.compression_strategy.parse()?;
        if peers.is_empty() && !self.distributed_auto {
            return Ok(None);
        }
        Ok(Some(DistributedTrainingConfig {
            peers,
            auto_discover: self.distributed_auto,
            compression,
            ..Default::default()
        }))
    }
}

// -- defaults shims (for `#[serde(default = "...")]`) -----------------------

fn default_train_output_dir() -> String {
    defaults::TRAIN_OUTPUT_DIR.to_string()
}
fn default_lr() -> f64 {
    defaults::LEARNING_RATE
}
fn default_batch_size() -> usize {
    defaults::CLI_BATCH_SIZE
}
fn default_epochs() -> usize {
    defaults::CLI_EPOCHS
}
fn default_grad_accum() -> usize {
    defaults::GRADIENT_ACCUMULATION_STEPS
}
fn default_max_grad_norm() -> f64 {
    defaults::MAX_GRAD_NORM
}
fn default_weight_decay() -> f64 {
    defaults::WEIGHT_DECAY
}
fn default_lr_schedule() -> String {
    "cosine".to_string()
}
fn default_seed() -> u64 {
    defaults::SEED
}
fn default_loss_scale() -> f32 {
    defaults::LOSS_SCALE as f32
}
fn default_lora_r() -> usize {
    defaults::LORA_R
}
fn default_lora_alpha() -> f32 {
    defaults::LORA_ALPHA
}
fn default_logging_steps() -> usize {
    defaults::LOGGING_STEPS
}
fn default_quant_block_size() -> usize {
    64
}
fn default_compression_strategy() -> String {
    "none".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptors_present() {
        let descs = TrainSpec::field_descriptors();
        assert!(!descs.is_empty());
        assert!(descs.iter().any(|d| d.name == "model"));
        assert!(descs.iter().any(|d| d.name == "dataset"));
        assert!(descs.iter().any(|d| d.name == "learning_rate"));
        // CLI-only fields are excluded from the descriptor list.
        assert!(!descs.iter().any(|d| d.name == "config_path"));
        assert!(!descs.iter().any(|d| d.name == "resume"));
    }

    #[test]
    fn argv_emits_required_flags() {
        let spec = TrainSpec {
            model: "Qwen/Qwen3-0.6B".into(),
            dataset: "data/train.jsonl".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--model".to_string()));
        assert!(argv.contains(&"Qwen/Qwen3-0.6B".to_string()));
        assert!(argv.contains(&"--dataset".to_string()));
        assert!(argv.contains(&"data/train.jsonl".to_string()));
        assert!(argv.contains(&"--output".to_string()));
        assert!(argv.contains(&"--learning-rate".to_string()));
    }

    #[test]
    fn argv_skips_optional_when_none() {
        let spec = TrainSpec {
            model: "m".into(),
            dataset: "d".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(!argv.contains(&"--eval-dataset".to_string()));
        assert!(!argv.contains(&"--quantization".to_string()));
        assert!(!argv.contains(&"--embedding-lr".to_string()));
    }

    #[test]
    fn argv_emits_flag_only_when_true() {
        let spec = TrainSpec {
            model: "m".into(),
            dataset: "d".into(),
            ane: true,
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--ane".to_string()));
        assert!(!argv.contains(&"--cut-cross-entropy".to_string()));
    }

    #[test]
    fn validate_rejects_empty_required() {
        let spec = TrainSpec::default();
        let errs = spec.validate_descriptors();
        assert!(errs.iter().any(|e| e.field == "model"));
        assert!(errs.iter().any(|e| e.field == "dataset"));
    }

    #[test]
    fn validate_passes_for_complete_spec() {
        let spec = TrainSpec {
            model: "m".into(),
            dataset: "d".into(),
            ..Default::default()
        };
        let errs = spec.validate_descriptors();
        assert!(errs.is_empty(), "unexpected validation errors: {errs:?}");
    }

    #[test]
    fn validate_rejects_out_of_range() {
        let spec = TrainSpec {
            model: "m".into(),
            dataset: "d".into(),
            learning_rate: 10.0,
            ..Default::default()
        };
        let errs = spec.validate_descriptors();
        assert!(errs.iter().any(|e| e.field == "learning_rate"));
    }

    #[test]
    fn subcommand_and_kind() {
        assert_eq!(TrainSpec::subcommand(), "train");
        assert_eq!(TrainSpec::job_kind(), crate::JobKind::Train);
    }

    fn enum_options(field: &str) -> Vec<&'static str> {
        let descriptor = TrainSpec::field_descriptors()
            .iter()
            .find(|d| d.name == field)
            .unwrap_or_else(|| panic!("no field {field}"));
        match descriptor.kind {
            crate::FieldKind::Enum { options } => options.to_vec(),
            other => panic!("{field} is not an enum: {other:?}"),
        }
    }

    #[test]
    fn enum_fields_offer_exactly_what_the_config_types_parse() {
        // The option lists live in attributes, so they cannot be read from the
        // enums; this keeps the two from drifting apart.
        let names = |all: Vec<&'static str>| {
            let mut all = all;
            all.sort_unstable();
            all
        };
        assert_eq!(
            names(enum_options("optimizer")),
            names(crate::OptimizerType::ALL.map(|k| k.as_str()).to_vec())
        );
        assert_eq!(
            names(enum_options("lr_schedule")),
            names(crate::LrSchedulerType::ALL.map(|k| k.as_str()).to_vec())
        );
        assert_eq!(
            names(enum_options("compression_strategy")),
            names(DistributedCompression::ALL.map(|k| k.as_str()).to_vec())
        );
    }

    #[test]
    fn the_spec_builds_the_configs_every_surface_runs() {
        let spec = TrainSpec {
            lora_r: 8,
            lora_alpha: 24.0,
            lora_dropout: 0.05,
            dora: true,
            rslora: true,
            learning_rate: 1e-4,
            embedding_lr: Some(5e-5),
            epochs: 2,
            optimizer: "lion".into(),
            lr_schedule: "wsd".into(),
            no_sequence_packing: true,
            distributed_peers: Some("a:1, b:2,".into()),
            compression_strategy: "topk".into(),
            ..Default::default()
        };
        let lora = spec.lora_config();
        assert_eq!((lora.r, lora.alpha, lora.dropout), (8, 24.0, 0.05));
        assert!(lora.use_dora && lora.use_rslora);

        let training = spec.training_config().unwrap();
        assert_eq!(training.learning_rate, 1e-4);
        assert_eq!(training.embedding_learning_rate, Some(5e-5));
        assert_eq!(training.num_epochs, 2);
        assert_eq!(training.optimizer, crate::OptimizerType::Lion);
        assert_eq!(training.lr_scheduler, crate::LrSchedulerType::Wsd);
        assert!(!training.use_packing);

        let distributed = spec.distributed_config().unwrap().unwrap();
        assert_eq!(distributed.peers, ["a:1", "b:2"]);
        assert_eq!(distributed.compression, DistributedCompression::TopK);
        assert!(TrainSpec::default().distributed_config().unwrap().is_none());
    }

    /// The loss options reach the training config and the command line the
    /// spec spells, and a value the trainer would refuse is caught by the
    /// config's own validation.
    #[test]
    fn the_loss_options_reach_the_config_and_the_command_line() {
        let spec = TrainSpec {
            label_smoothing: 0.1,
            z_loss: 1e-4,
            neftune_alpha: Some(5.0),
            ..Default::default()
        };
        let training = spec.training_config().unwrap();
        assert_eq!(training.label_smoothing, 0.1);
        assert_eq!(training.z_loss, 1e-4);
        assert_eq!(training.neftune_noise_alpha, Some(5.0));
        let argv = spec.to_argv().join(" ");
        assert!(argv.contains("--label-smoothing 0.1"), "{argv}");
        assert!(argv.contains("--z-loss 0.0001"), "{argv}");
        assert!(argv.contains("--neftune-alpha 5"), "{argv}");

        let too_smooth = TrainSpec {
            label_smoothing: 1.5,
            ..Default::default()
        };
        assert!(crate::ConfigValidator::validate(&too_smooth.training_config().unwrap()).is_err());
    }

    #[test]
    fn an_unknown_optimizer_or_schedule_is_an_error() {
        let bad_schedule = TrainSpec {
            lr_schedule: "cosin".into(),
            ..Default::default()
        };
        assert!(
            bad_schedule
                .training_config()
                .unwrap_err()
                .contains("cosin")
        );
        let bad_optimizer = TrainSpec {
            optimizer: "adam".into(),
            ..Default::default()
        };
        assert!(bad_optimizer.training_config().is_err());
        let bad_compression = TrainSpec {
            distributed_auto: true,
            compression_strategy: "zstd".into(),
            ..Default::default()
        };
        assert!(bad_compression.distributed_config().is_err());
    }

    #[test]
    fn argv_round_trip_via_serde() {
        let spec = TrainSpec {
            model: "m".into(),
            dataset: "d".into(),
            ..Default::default()
        };
        let yaml = serde_saphyr::to_string(&spec).expect("serialize");
        let back: TrainSpec = serde_saphyr::from_str(&yaml).expect("deserialize");
        assert_eq!(back.model, spec.model);
        assert_eq!(back.dataset, spec.dataset);
        assert_eq!(back.learning_rate, spec.learning_rate);
    }
}
