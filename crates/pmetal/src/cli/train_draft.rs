//! Clap argument struct for `pmetal train-draft`.

use clap::Args;

#[derive(Args, Debug)]
pub struct TrainDraftArgs {
    /// Frozen Qwen3 target model ID or local path.
    #[arg(long = "target")]
    pub target: String,

    /// Optional existing DFlash draft checkpoint to continue from.
    #[arg(long = "draft", conflicts_with = "draft_config")]
    pub draft: Option<String>,

    /// DFlash config JSON used when --draft is omitted.
    #[arg(long = "draft-config")]
    pub draft_config: Option<String>,

    /// Tokenized shard files from `pmetal tokenize`.
    #[arg(short, long = "shards", value_delimiter = ',', num_args = 1.., required = true)]
    pub shards: Vec<String>,

    /// Output checkpoint directory.
    #[arg(short, long = "output", default_value = "./draft-output")]
    pub output: String,

    /// Sequence length per training sample.
    #[arg(long = "seq-len", default_value = "512")]
    pub seq_len: usize,

    /// Batch size.
    #[arg(long = "batch-size", default_value = "1")]
    pub batch_size: usize,

    /// Training steps.
    #[arg(long = "steps", default_value = "1000")]
    pub steps: usize,

    /// Peak learning rate.
    #[arg(long = "learning-rate", default_value = "2e-4")]
    pub learning_rate: f32,

    /// Minimum learning rate for cosine schedule.
    #[arg(long = "min-lr", default_value = "1e-5")]
    pub min_lr: f32,

    /// Linear warmup steps.
    #[arg(long = "warmup-steps", default_value = "100")]
    pub warmup_steps: usize,

    /// LR schedule: constant, linear, cosine.
    #[arg(long = "lr-schedule", default_value = "cosine")]
    pub lr_schedule: String,

    /// AdamW weight decay.
    #[arg(long = "weight-decay", default_value = "0.01")]
    pub weight_decay: f32,

    /// Max gradient norm. 0 disables clipping.
    #[arg(long = "max-grad-norm", default_value = "1.0")]
    pub max_grad_norm: f32,

    /// Save checkpoint every N steps. 0 disables periodic checkpoints.
    #[arg(long = "checkpoint-every", default_value = "500")]
    pub checkpoint_every: usize,

    /// Log every N steps. 0 disables step logs.
    #[arg(long = "log-every", default_value = "10")]
    pub log_every: usize,

    /// EOS token ID inserted between documents by the streaming shard reader.
    #[arg(long = "eos-token-id", default_value = "0")]
    pub eos_token_id: u32,

    /// Random seed.
    #[arg(long = "seed", default_value = "42")]
    pub seed: u64,
}
