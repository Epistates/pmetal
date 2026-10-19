//! Clap argument struct for `pmetal train-diffusion`.
//!
//! Block-diffusion LoRA fine-tuning for a DiffusionGemma model. Each dataset
//! example is a `(prompt, response)` pair: the prompt is tokenized into the
//! causal-encoder context and the response into the fixed-length decoder canvas
//! (padded/truncated to the model's `canvas_length`). The trainer corrupts the
//! canvas with the uniform-categorical kernel and optimises the attached LoRA
//! adapters against the decoder denoising cross-entropy.

use clap::Args;
use pmetal_core::jobs::TrainDiffusionSpec;

/// LoRA fine-tune a DiffusionGemma block-diffusion model.
#[derive(Args, Debug)]
pub struct TrainDiffusionArgs {
    /// DiffusionGemma model ID or local path.
    #[arg(long = "model")]
    pub model: String,

    /// JSONL dataset; each line has `prompt`/`response` (aliases
    /// `context`/`target`, `input`/`output`).
    #[arg(long = "dataset")]
    pub dataset: String,

    /// Output directory for the trained LoRA adapters.
    #[arg(short, long = "output", default_value = "./diffusion-lora-output")]
    pub output: String,

    /// LoRA rank.
    #[arg(long = "lora-r", default_value = "16")]
    pub lora_r: usize,

    /// LoRA alpha (scaling = alpha / r).
    #[arg(long = "lora-alpha", default_value = "32")]
    pub lora_alpha: f32,

    /// Comma-separated projections to adapt, by name (`q_proj`, `gate_proj`,
    /// ...), in the encoder and the decoder. Omitted, every linear layer but
    /// the output head is adapted. MoE routers and experts are never adapted.
    #[arg(long = "lora-targets", value_delimiter = ',')]
    pub lora_targets: Vec<String>,

    /// Total optimiser steps (the dataset is cycled to reach this count).
    #[arg(long = "steps", default_value = "1000")]
    pub steps: usize,

    /// AdamW learning rate.
    #[arg(long = "learning-rate", default_value = "2e-4")]
    pub learning_rate: f32,

    /// AdamW weight decay.
    #[arg(long = "weight-decay", default_value = "0.0")]
    pub weight_decay: f32,

    /// Max gradient norm. 0 disables clipping.
    #[arg(long = "max-grad-norm", default_value = "1.0")]
    pub max_grad_norm: f32,

    /// Restrict the denoising loss to corrupted canvas positions (ELBO
    /// reconstruction term). When false, average over the whole canvas.
    #[arg(
        long = "corrupted-only",
        default_value_t = true,
        action = clap::ArgAction::Set
    )]
    pub corrupted_only: bool,

    /// Quantization method for QLoRA (none, nf4, fp4, int8), as for `pmetal
    /// train`. Packs every projection but the routers, and the MoE experts
    /// (4-bit affine for nf4 and fp4, 8-bit for int8), so only the f32 LoRA
    /// adapters train. A large memory reduction for the 26B MoE model.
    #[arg(long = "quantization", value_enum, default_value_t = crate::cli::train::QuantizationMethod::None)]
    pub quantization: crate::cli::train::QuantizationMethod,

    /// Weights per scale for nf4 and int8 (32, 64 or 128; fp4's block is 16)
    #[arg(long = "quant-block-size", default_value = "64")]
    pub quant_block_size: usize,

    /// Pack nf4's absmax values to 8 bits as well (QLoRA double quantization)
    #[arg(long = "double-quant")]
    pub double_quant: bool,

    /// Truncate each tokenized prompt to at most this many context tokens.
    #[arg(long = "max-context-len", default_value = "512")]
    pub max_context_len: usize,

    /// Save an intermediate adapter checkpoint every N steps. 0 disables.
    #[arg(long = "checkpoint-every", default_value = "0")]
    pub checkpoint_every: usize,

    /// Log every N steps. 0 disables step logs.
    #[arg(long = "log-every", default_value = "10")]
    pub log_every: usize,

    /// Random seed.
    #[arg(long = "seed", default_value = "42")]
    pub seed: u64,
}

impl From<TrainDiffusionArgs> for TrainDiffusionSpec {
    fn from(a: TrainDiffusionArgs) -> Self {
        TrainDiffusionSpec {
            model: a.model,
            dataset: a.dataset,
            output: a.output,
            lora_r: a.lora_r,
            lora_alpha: a.lora_alpha,
            lora_targets: a.lora_targets,
            corrupted_only: a.corrupted_only,
            quantization: a.quantization.as_str().to_string(),
            quant_block_size: a.quant_block_size,
            double_quant: a.double_quant,
            steps: a.steps,
            learning_rate: a.learning_rate,
            weight_decay: a.weight_decay,
            max_grad_norm: a.max_grad_norm,
            max_context_len: a.max_context_len,
            checkpoint_every: a.checkpoint_every,
            log_every: a.log_every,
            seed: a.seed,
        }
    }
}
