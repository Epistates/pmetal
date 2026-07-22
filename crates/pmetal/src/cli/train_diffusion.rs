//! Clap argument struct for `pmetal train-diffusion`.
//!
//! Block-diffusion LoRA fine-tuning for a DiffusionGemma model. Each dataset
//! example is a `(prompt, response)` pair: the prompt is tokenized into the
//! causal-encoder context and the response into the fixed-length decoder canvas
//! (padded/truncated to the model's `canvas_length`). The trainer corrupts the
//! canvas with the uniform-categorical kernel and optimises the attached LoRA
//! adapters against the decoder denoising cross-entropy.

use clap::Args;

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

    /// Comma-separated attention projections to adapt.
    #[arg(
        long = "lora-targets",
        value_delimiter = ',',
        default_value = "q_proj,k_proj,v_proj,o_proj"
    )]
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
    #[arg(long = "corrupted-only", default_value = "true")]
    pub corrupted_only: bool,

    /// QLoRA: quantize the frozen base weights (attention + MoE experts) so only
    /// the f32 LoRA adapters are trained. Large memory reduction for big MoE
    /// models.
    #[arg(long = "qlora", default_value = "false")]
    pub qlora: bool,

    /// QLoRA quantization group size (32, 64, or 128).
    #[arg(long = "qlora-group-size", default_value = "64")]
    pub qlora_group_size: i32,

    /// QLoRA quantization bits (one of 2, 3, 4, 5, 6, 8).
    #[arg(long = "qlora-bits", default_value = "4")]
    pub qlora_bits: i32,

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
