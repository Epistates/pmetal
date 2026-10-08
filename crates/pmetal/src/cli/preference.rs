//! Clap argument struct for `pmetal preference`.
//!
//! Flag names and defaults mirror `pmetal_core::jobs::PreferenceSpec`, which
//! is authoritative; `main.rs::argv_roundtrip` checks the two agree.

use clap::Args;
use pmetal_core::jobs::PreferenceSpec;

/// Offline preference optimization with LoRA.
#[derive(Args, Debug)]
pub struct PreferenceArgs {
    /// Model ID or local path.
    #[arg(short, long = "model")]
    pub model: String,

    /// Dataset: a JSONL, JSON or Parquet file, or a Hugging Face dataset ID.
    /// Rows hold prompt/chosen/rejected (KTO also reads prompt/completion/label).
    #[arg(short, long = "dataset")]
    pub dataset: String,

    /// Output directory for the LoRA adapter.
    #[arg(short, long = "output", default_value = "./output/preference")]
    pub output: String,

    /// Objective: dpo, ipo, hinge, simpo, orpo or kto.
    #[arg(long = "loss", default_value = "dpo")]
    pub loss: String,

    /// β (default 2.5 for SimPO, 0.1 for the others).
    #[arg(long = "beta")]
    pub beta: Option<f32>,

    /// SimPO's target reward margin as a share of β (γ/β).
    #[arg(long = "simpo-gamma-ratio", default_value = "0.5")]
    pub simpo_gamma_ratio: f32,

    /// DPO label smoothing: the share of preference labels assumed flipped.
    /// Above 0 it trains Robust DPO (0.1 is typical).
    #[arg(long = "label-smoothing", default_value = "0.0")]
    pub label_smoothing: f32,

    /// KTO weight of desirable examples.
    #[arg(long = "desirable-weight", default_value = "1.0")]
    pub desirable_weight: f32,

    /// KTO weight of undesirable examples.
    #[arg(long = "undesirable-weight", default_value = "1.0")]
    pub undesirable_weight: f32,

    /// Peak learning rate.
    #[arg(long = "learning-rate", default_value = "1e-5")]
    pub learning_rate: f64,

    /// Pairs (or KTO examples) per micro-batch.
    #[arg(long = "batch-size", default_value = "2")]
    pub batch_size: usize,

    /// Micro-batches per optimizer step.
    #[arg(long = "gradient-accumulation-steps", default_value = "8")]
    pub gradient_accumulation_steps: usize,

    /// Passes over the dataset.
    #[arg(long = "epochs", default_value = "1")]
    pub epochs: usize,

    /// Stop after this many optimizer steps.
    #[arg(long = "max-steps")]
    pub max_steps: Option<usize>,

    /// Share of steps spent warming the learning rate up, before cosine decay.
    #[arg(long = "warmup-ratio", default_value = "0.1")]
    pub warmup_ratio: f64,

    /// Clip the gradient's global norm to this (0 disables).
    #[arg(long = "max-grad-norm", default_value = "1.0")]
    pub max_grad_norm: f64,

    /// AdamW weight decay.
    #[arg(long = "weight-decay", default_value = "0.0")]
    pub weight_decay: f64,

    /// LoRA rank.
    #[arg(long = "lora-r", default_value = "16")]
    pub lora_r: usize,

    /// LoRA alpha.
    #[arg(long = "lora-alpha", default_value = "32")]
    pub lora_alpha: f32,

    /// Prompt tokens kept (the most recent ones).
    #[arg(long = "max-prompt-length", default_value = "512")]
    pub max_prompt_length: usize,

    /// Prompt plus completion tokens kept.
    #[arg(long = "max-length", default_value = "1024")]
    pub max_length: usize,

    /// Seed for adapter initialization and shuffling.
    #[arg(long = "seed", default_value = "42")]
    pub seed: u64,

    /// Write per-step metrics as JSONL to this path (relative to --output).
    #[arg(long = "log-metrics")]
    pub log_metrics: Option<String>,
}

impl From<PreferenceArgs> for PreferenceSpec {
    fn from(a: PreferenceArgs) -> Self {
        PreferenceSpec {
            model: a.model,
            dataset: a.dataset,
            output_dir: a.output,
            loss: a.loss,
            beta: a.beta,
            simpo_gamma_ratio: a.simpo_gamma_ratio,
            label_smoothing: a.label_smoothing,
            desirable_weight: a.desirable_weight,
            undesirable_weight: a.undesirable_weight,
            learning_rate: a.learning_rate,
            batch_size: a.batch_size,
            gradient_accumulation_steps: a.gradient_accumulation_steps,
            epochs: a.epochs,
            max_steps: a.max_steps,
            warmup_ratio: a.warmup_ratio,
            max_grad_norm: a.max_grad_norm,
            weight_decay: a.weight_decay,
            lora_r: a.lora_r,
            lora_alpha: a.lora_alpha,
            max_prompt_length: a.max_prompt_length,
            max_length: a.max_length,
            seed: a.seed,
            log_metrics: a.log_metrics,
        }
    }
}
