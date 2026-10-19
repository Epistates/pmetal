//! Clap argument struct for `pmetal train`.
//!
//! Field `#[arg(long = "...")]` strings MUST match the `argv = "..."` attribute
//! in `TrainSpec` (pmetal-core/src/jobs/train.rs).

use clap::Args;

/// Thin clap argument struct for `pmetal train`.
#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Path to training configuration file (YAML)
    #[arg(short, long)]
    pub config: Option<String>,

    /// Model ID (HuggingFace or local path)
    #[arg(short, long = "model")]
    pub model: Option<String>,

    /// Dataset path (JSONL file)
    #[arg(short, long = "dataset")]
    pub dataset: Option<String>,

    /// Evaluation dataset path (optional JSONL file)
    #[arg(long = "eval-dataset")]
    pub eval_dataset: Option<String>,

    /// Output directory
    #[arg(short, long = "output", default_value = "./output")]
    pub output: String,

    /// LoRA rank
    #[arg(long = "lora-r", default_value = "16")]
    pub lora_r: usize,

    /// LoRA alpha (scaling factor). Recommended: 2x rank.
    #[arg(long = "lora-alpha", default_value = "32")]
    pub lora_alpha: f32,

    /// Train DoRA (Weight-Decomposed LoRA) adapters instead of plain LoRA.
    /// Decomposes each update into magnitude and direction; slower per step,
    /// usually better at low rank.
    #[arg(long = "dora")]
    pub dora: bool,

    /// Use rank-stabilized LoRA scaling (alpha/sqrt(r) instead of alpha/r),
    /// which keeps the update magnitude steady as rank grows.
    #[arg(long = "rslora")]
    pub rslora: bool,

    /// Dropout applied to the LoRA branch during training.
    #[arg(long = "lora-dropout", default_value = "0.0")]
    pub lora_dropout: f32,

    /// Learning rate. Recommended: 2e-4 for most tasks.
    #[arg(long = "learning-rate", default_value = "2e-4")]
    pub learning_rate: f64,

    /// Batch size
    #[arg(long = "batch-size", default_value = "1")]
    pub batch_size: usize,

    /// Number of epochs
    #[arg(long = "epochs", default_value = "1")]
    pub epochs: usize,

    /// Maximum sequence length (0 to auto-detect from model config)
    #[arg(long = "max-seq-len", default_value = "0")]
    pub max_seq_len: usize,

    /// Gradient accumulation steps
    #[arg(long = "gradient-accumulation-steps", default_value = "4")]
    pub gradient_accumulation_steps: usize,

    /// Maximum gradient norm for clipping (0 to disable)
    #[arg(long = "max-grad-norm", default_value = "1.0")]
    pub max_grad_norm: f64,

    /// Resume from checkpoint
    #[arg(long = "resume")]
    pub resume: bool,

    /// Quantization method for QLoRA (none, nf4, fp4, int8)
    #[arg(long = "quantization", value_enum, default_value = "none")]
    pub quantization: crate::QuantizationMethod,

    /// Weights per scale for nf4 and int8 (32, 64 or 128; fp4's block is 16)
    #[arg(long = "quant-block-size", default_value = "64")]
    pub quant_block_size: usize,

    /// Pack nf4's absmax values to 8 bits as well (QLoRA double quantization)
    #[arg(long = "double-quant")]
    pub double_quant: bool,

    /// Disable sequence packing
    #[arg(long = "no-sequence-packing")]
    pub no_sequence_packing: bool,

    /// Override the maximum sequence length for packing.
    #[arg(long = "pack-max-seq-len")]
    pub pack_max_seq_len: Option<usize>,

    /// Disable gradient checkpointing
    #[arg(long = "no-gradient-checkpointing")]
    pub no_gradient_checkpointing: bool,

    /// Deprecated, has no effect. One decoder layer is the checkpoint unit,
    /// as in PyTorch, so there is no block size to pick.
    #[arg(long = "gradient-checkpointing-layers", default_value = "1")]
    pub gradient_checkpointing_layers: usize,

    /// Path to log training metrics as JSONL.
    #[arg(long = "log-metrics")]
    pub log_metrics: Option<String>,

    /// Separate learning rate for embedding layers.
    #[arg(long = "embedding-lr")]
    pub embedding_lr: Option<f32>,

    /// Loss scaling factor for ANE training (default: 1.0).
    #[arg(long = "loss-scale", default_value = "1.0")]
    pub loss_scale: f32,

    /// Number of linear warmup steps.
    #[arg(long = "warmup-steps", default_value = "0")]
    pub warmup_steps: usize,

    /// Learning rate schedule.
    #[arg(long = "lr-schedule", default_value = "cosine")]
    pub lr_schedule: String,

    /// Weight decay coefficient.
    #[arg(long = "weight-decay", default_value = "0.01")]
    pub weight_decay: f64,

    /// Optimizer: adamw, sgd, lion, adafactor, muon, adamw_schedule_free or
    /// adamw_8bit. Lion wants a 3-10x smaller learning rate and 3-10x larger
    /// weight decay than AdamW; schedule-free AdamW wants a constant rate.
    #[arg(long = "optimizer", default_value = "adamw")]
    pub optimizer: pmetal_core::OptimizerType,

    /// Random seed.
    #[arg(long = "seed", default_value = "42")]
    pub seed: u64,

    /// Label smoothing of the training loss, in [0, 1): the target is
    /// 1 - epsilon on the label and epsilon spread over the vocabulary.
    /// Evaluation reports the plain loss.
    #[arg(long = "label-smoothing", default_value = "0.0")]
    pub label_smoothing: f64,

    /// Weight of the z-loss added to each token's loss, `weight * lse^2`
    /// (PaLM; 1e-4 is the paper's value). 0 turns it off.
    #[arg(long = "z-loss", default_value = "0.0")]
    pub z_loss: f64,

    /// NEFTune noise alpha: uniform noise of magnitude
    /// `alpha / sqrt(seq_len * hidden)` on the token embeddings of every
    /// training forward pass (5 for small models, 15 for larger ones).
    #[arg(long = "neftune-alpha")]
    pub neftune_alpha: Option<f64>,

    /// Use cut cross-entropy: the loss from fused Metal kernels that never
    /// hold the logits, with the backward's gradient filtering set to keep
    /// the gradient as precise as the full logits' (the library's
    /// `CutCrossEntropyConfig::filter_eps` trades precision for speed).
    #[arg(long = "cut-cross-entropy")]
    pub cut_cross_entropy: bool,

    /// Disable automatic adaptive LR.
    #[arg(long = "no-adaptive-lr")]
    pub no_adaptive_lr: bool,

    /// Custom JSONL column containing the training text.
    #[arg(long = "text-column")]
    pub text_column: Option<String>,

    /// Multiple JSONL columns to concatenate as the training text.
    #[arg(long = "text-columns", value_delimiter = ',')]
    pub text_columns: Option<Vec<String>>,

    /// Separator inserted between columns when using --text-columns.
    #[arg(long = "column-separator", default_value = "\n\n")]
    pub column_separator: String,

    /// Custom JSONL column containing the prompt portion (loss-masked).
    #[arg(long = "prompt-column")]
    pub prompt_column: Option<String>,

    /// Custom JSONL column containing the response portion.
    #[arg(long = "response-column")]
    pub response_column: Option<String>,

    /// Enable ANE (Apple Neural Engine) for training (experimental).
    #[cfg(feature = "ane")]
    #[arg(long = "ane")]
    pub ane: bool,

    /// Distributed training: comma-separated peer addresses (ip:port).
    #[cfg(feature = "distributed")]
    #[arg(long = "distributed-peers", value_delimiter = ',')]
    pub distributed_peers: Option<Vec<String>>,

    /// Distributed training: enable automatic mDNS peer discovery.
    #[cfg(feature = "distributed")]
    #[arg(long = "distributed-auto")]
    pub distributed_auto: bool,

    /// Gradient compression strategy for distributed training.
    #[cfg(feature = "distributed")]
    #[arg(long = "compression-strategy", default_value = "none")]
    pub compression_strategy: Option<String>,
}
