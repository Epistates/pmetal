//! Clap argument struct for `pmetal pretrain`.

use clap::Args;

/// Thin clap argument struct for `pmetal pretrain`.
#[derive(Args, Debug)]
pub struct PretrainArgs {
    /// Model architecture (e.g. gpt-oss)
    #[arg(short, long = "arch")]
    pub arch: String,

    /// Glob pattern or comma-separated list of tokenized shard files (.bin)
    #[arg(short, long = "shards", value_delimiter = ',')]
    pub shards: Vec<String>,

    /// Target sequence length for packed training batches
    #[arg(long = "seq-len", default_value = "2048")]
    pub seq_len: usize,

    /// Batch size
    #[arg(long = "batch-size", default_value = "4")]
    pub batch_size: usize,

    /// Number of training steps
    #[arg(long = "steps", default_value = "10000")]
    pub steps: usize,

    /// Peak learning rate
    #[arg(long = "learning-rate", default_value = "3e-4")]
    pub learning_rate: f32,

    /// Minimum learning rate (cosine floor)
    #[arg(long = "min-lr", default_value = "1e-5")]
    pub min_lr: f32,

    /// Linear warmup steps
    #[arg(long = "warmup-steps", default_value = "1000")]
    pub warmup_steps: usize,

    /// LR schedule (constant, linear, cosine)
    #[arg(long = "lr-schedule", default_value = "cosine")]
    pub lr_schedule: String,

    /// AdamW weight decay
    #[arg(long = "weight-decay", default_value = "0.1")]
    pub weight_decay: f32,

    /// Max gradient norm for clipping (0 to disable)
    #[arg(long = "max-grad-norm", default_value = "1.0")]
    pub max_grad_norm: f32,

    /// EOS token ID (inserted between documents in packed sequences)
    #[arg(long = "eos-token-id", default_value = "0")]
    pub eos_token_id: u32,

    /// Output / checkpoint directory
    #[arg(short, long = "output", default_value = "./pretrain-output")]
    pub output: String,

    /// Save checkpoint every N steps (0 to disable)
    #[arg(long = "checkpoint-every", default_value = "1000")]
    pub checkpoint_every: usize,

    /// Resume from checkpoint directory
    #[arg(long = "resume")]
    pub resume: Option<String>,

    /// Model config JSON file (overrides arch defaults)
    #[arg(long = "model-config")]
    pub model_config: Option<String>,

    /// MoE router z-loss coefficient (0 to disable)
    #[arg(long = "z-loss", default_value = "0.0")]
    pub z_loss: f32,

    /// Gradient accumulation steps (effective batch = batch_size * this)
    #[arg(long = "gradient-accumulation-steps", default_value = "1")]
    pub gradient_accumulation_steps: usize,

    /// Log step/loss/LR every N steps (0 to disable)
    #[arg(long = "log-every", default_value = "10")]
    pub log_every: usize,

    /// Evaluate on held-out data every N steps (0 to disable)
    #[arg(long = "eval-every", default_value = "0")]
    pub eval_every: usize,

    /// Number of batches per eval round
    #[arg(long = "eval-batches", default_value = "10")]
    pub eval_batches: usize,

    /// Random seed
    #[arg(long = "seed", default_value = "42")]
    pub seed: u64,
}
