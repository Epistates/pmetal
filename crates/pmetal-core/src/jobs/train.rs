//! `pmetal train` — supervised fine-tuning / LoRA / QLoRA / ANE.

use crate::{FieldError, JobFields, defaults};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

/// Spec for `pmetal train`.
///
/// Holds the user-facing input fields. Conversion to the trainer's internal
/// `TrainingJobConfig` happens at the consumer boundary (currently inside the
/// CLI's `commands::run_training` after Phase 4).
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

    /// Deprecated, has no effect.
    ///
    /// One decoder layer is the checkpoint unit, as in PyTorch and mlx-lm, so
    /// there is no block size to pick. Kept on the spec so existing job files
    /// and `--gradient-checkpointing-layers` still parse; `min = max = 1` stops
    /// the generated UI offering a control that does nothing.
    #[job(
        label = "Gradient Checkpointing Layers (deprecated, no effect)",
        group = "Optimization",
        argv = "--gradient-checkpointing-layers",
        min = 1,
        max = 1,
        default_int = 1
    )]
    #[serde(default = "default_grad_ckpt_layers")]
    pub gradient_checkpointing_layers: usize,

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
        label = "Disable Flash Attention",
        group = "Compute",
        argv = "--no-flash-attention",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub no_flash_attention: bool,

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
        label = "Disable JIT",
        group = "Compute",
        argv = "--no-jit-compilation",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub no_jit_compilation: bool,

    #[job(
        label = "Disable Fused Step",
        group = "Compute",
        argv = "--no-fused",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub no_fused: bool,

    #[job(
        label = "Disable Fused Optimizer",
        group = "Compute",
        argv = "--no-metal-fused-optimizer",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub no_metal_fused_optimizer: bool,

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
            batch_size: default_batch_size(),
            epochs: default_epochs(),
            max_seq_len: 0,
            gradient_accumulation_steps: default_grad_accum(),
            no_gradient_checkpointing: false,
            gradient_checkpointing_layers: default_grad_ckpt_layers(),
            max_grad_norm: default_max_grad_norm(),
            warmup_steps: 0,
            weight_decay: default_weight_decay(),
            lr_schedule: default_lr_schedule(),
            seed: default_seed(),
            loss_scale: default_loss_scale(),
            lora_r: default_lora_r(),
            lora_alpha: default_lora_alpha(),
            dora: false,
            rslora: false,
            lora_dropout: 0.0,
            quantization: None,
            quant_block_size: default_quant_block_size(),
            double_quant: false,
            text_column: None,
            text_columns: None,
            column_separator: None,
            prompt_column: None,
            response_column: None,
            no_flash_attention: false,
            no_sequence_packing: false,
            no_jit_compilation: false,
            no_fused: false,
            no_metal_fused_optimizer: false,
            cut_cross_entropy: false,
            no_adaptive_lr: false,
            ane: false,
            pack_max_seq_len: None,
            distributed_peers: None,
            distributed_auto: false,
            compression_strategy: default_compression_strategy(),
            config_path: None,
            resume: false,
        }
    }
}

impl TrainSpec {
    /// Apply cross-field defaults / sentinels and run descriptor validation.
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let errs = self.validate_descriptors();
        if errs.is_empty() { Ok(()) } else { Err(errs) }
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
fn default_grad_ckpt_layers() -> usize {
    1
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

    #[test]
    fn argv_round_trip_via_serde() {
        let spec = TrainSpec {
            model: "m".into(),
            dataset: "d".into(),
            ..Default::default()
        };
        let yaml = serde_yaml::to_string(&spec).expect("serialize");
        let back: TrainSpec = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(back.model, spec.model);
        assert_eq!(back.dataset, spec.dataset);
        assert_eq!(back.learning_rate, spec.learning_rate);
    }
}
