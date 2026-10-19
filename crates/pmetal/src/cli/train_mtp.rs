//! Clap argument struct for `pmetal train-mtp`.

use clap::{Args, ValueEnum};
use pmetal_core::jobs::TrainMtpSpec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TrainMtpFamily {
    Auto,
    Qwen3Next,
    Gemma4,
}

#[derive(Args, Debug)]
pub struct TrainMtpArgs {
    /// Target model ID or local path.
    #[arg(short, long = "model")]
    pub model: String,

    /// MTP family to train. `auto` detects from the target config.
    #[arg(long = "family", value_enum, default_value = "auto")]
    pub family: TrainMtpFamily,

    /// Tokenized shard files from `pmetal tokenize`.
    #[arg(short, long = "shards", value_delimiter = ',', num_args = 1.., required = true)]
    pub shards: Vec<String>,

    /// Output checkpoint directory.
    #[arg(short, long = "output", default_value = "./mtp-output")]
    pub output: String,

    /// Optional existing assistant/MTP checkpoint to continue from.
    #[arg(long = "init-mtp")]
    pub init_mtp: Option<String>,

    /// Continue an earlier run: a checkpoint directory, or the output
    /// directory of the run (its furthest checkpoint). The weights, the
    /// optimizer state and the step are restored and the run goes on to
    /// --steps; the shards are read from that step on.
    #[arg(long = "resume", conflicts_with = "init_mtp")]
    pub resume: Option<String>,

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
    pub learning_rate: f64,

    /// Minimum learning rate for cosine schedule.
    #[arg(long = "min-lr", default_value = "1e-5")]
    pub min_lr: f64,

    /// Linear warmup steps.
    #[arg(long = "warmup-steps", default_value = "100")]
    pub warmup_steps: usize,

    /// LR schedule: cosine, linear, constant, cosine_with_restarts,
    /// polynomial or wsd.
    #[arg(long = "lr-schedule", default_value = "cosine")]
    pub lr_schedule: pmetal_core::LrSchedulerType,

    /// AdamW weight decay.
    #[arg(long = "weight-decay", default_value = "0.01")]
    pub weight_decay: f64,

    /// Max gradient norm. 0 disables clipping.
    #[arg(long = "max-grad-norm", default_value = "1.0")]
    pub max_grad_norm: f64,

    /// Save checkpoint every N steps. 0 disables periodic checkpoints.
    #[arg(long = "checkpoint-every", default_value = "500")]
    pub checkpoint_every: usize,

    /// Log every N steps. 0 disables step logs.
    #[arg(long = "log-every", default_value = "10")]
    pub log_every: usize,

    /// EOS token ID inserted between documents by the streaming shard reader.
    #[arg(long = "eos-token-id", default_value = "0")]
    pub eos_token_id: u32,

    /// Number of Qwen MTP predictor layers to create when --init-mtp is omitted.
    #[arg(long = "mtp-layers", default_value = "1")]
    pub mtp_layers: usize,

    /// Number of Gemma assistant layers to create when --init-mtp is omitted.
    #[arg(long = "assistant-layers", default_value = "2")]
    pub assistant_layers: usize,

    /// Gemma generation_config.json `num_assistant_tokens`.
    #[arg(long = "num-assistant-tokens", default_value = "6")]
    pub num_assistant_tokens: usize,

    /// Random seed.
    #[arg(long = "seed", default_value = "42")]
    pub seed: u64,
}

impl From<TrainMtpArgs> for TrainMtpSpec {
    fn from(a: TrainMtpArgs) -> Self {
        TrainMtpSpec {
            model: a.model,
            family: match a.family {
                TrainMtpFamily::Auto => "auto",
                TrainMtpFamily::Qwen3Next => "qwen3-next",
                TrainMtpFamily::Gemma4 => "gemma4",
            }
            .to_string(),
            init_mtp: a.init_mtp,
            mtp_layers: a.mtp_layers,
            assistant_layers: a.assistant_layers,
            num_assistant_tokens: a.num_assistant_tokens,
            shards: a.shards,
            eos_token_id: a.eos_token_id,
            output: a.output,
            seq_len: a.seq_len,
            batch_size: a.batch_size,
            steps: a.steps,
            learning_rate: a.learning_rate,
            min_lr: a.min_lr,
            warmup_steps: a.warmup_steps,
            lr_schedule: a.lr_schedule.to_string(),
            weight_decay: a.weight_decay,
            max_grad_norm: a.max_grad_norm,
            checkpoint_every: a.checkpoint_every,
            log_every: a.log_every,
            seed: a.seed,
            resume: a.resume,
        }
    }
}
