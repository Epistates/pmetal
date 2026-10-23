//! Clap argument struct for `pmetal train`.
//!
//! Flag names and defaults mirror `pmetal_core::jobs::TrainSpec`, which is
//! authoritative; `cli::parity` checks the two agree, and
//! `cli::argv_roundtrip` that a spec's argv comes back as the same spec.

use clap::{Args, ValueEnum};
use pmetal_core::defaults;
use pmetal_core::jobs::TrainSpec;

/// Fine-tune a model using LoRA/QLoRA.
#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Path to training configuration file (YAML)
    #[arg(short, long)]
    pub config: Option<String>,

    /// Model ID (HuggingFace or local path)
    #[arg(short, long = "model")]
    pub model: Option<String>,

    /// Dataset: a JSONL, JSON, Parquet, Arrow or CSV file, or a HuggingFace dataset ID
    #[arg(short, long = "dataset")]
    pub dataset: Option<String>,

    /// Evaluation dataset (same formats as --dataset)
    #[arg(long = "eval-dataset")]
    pub eval_dataset: Option<String>,

    /// Output directory
    #[arg(short, long = "output", default_value = defaults::TRAIN_OUTPUT_DIR)]
    pub output: String,

    /// LoRA rank
    #[arg(long = "lora-r", default_value_t = defaults::LORA_R)]
    pub lora_r: usize,

    /// LoRA alpha (scaling factor). Recommended: 2x rank.
    #[arg(long = "lora-alpha", default_value_t = defaults::LORA_ALPHA)]
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
    #[arg(long = "lora-dropout", default_value_t = 0.0)]
    pub lora_dropout: f32,

    /// Projections to adapt, comma-separated (q_proj,v_proj,...); the default
    /// adapts every linear layer. MoE routers never are.
    #[arg(long = "lora-targets", value_delimiter = ',')]
    pub lora_targets: Vec<String>,

    /// Learning rate. Recommended: 2e-4 for most tasks.
    #[arg(long = "learning-rate", default_value_t = defaults::LEARNING_RATE)]
    pub learning_rate: f64,

    /// Batch size
    #[arg(long = "batch-size", default_value_t = defaults::CLI_BATCH_SIZE)]
    pub batch_size: usize,

    /// Number of epochs
    #[arg(long = "epochs", default_value_t = defaults::EPOCHS)]
    pub epochs: usize,

    /// Stop after this many optimizer steps, whatever --epochs says
    #[arg(long = "max-steps")]
    pub max_steps: Option<usize>,

    /// Maximum sequence length (0 to auto-detect from model config)
    #[arg(long = "max-seq-len", default_value_t = 0)]
    pub max_seq_len: usize,

    /// Gradient accumulation steps
    #[arg(long = "gradient-accumulation-steps", default_value_t = defaults::GRADIENT_ACCUMULATION_STEPS)]
    pub gradient_accumulation_steps: usize,

    /// Maximum gradient norm for clipping (0 to disable)
    #[arg(long = "max-grad-norm", default_value_t = defaults::MAX_GRAD_NORM)]
    pub max_grad_norm: f64,

    /// Resume from the latest checkpoint in the output directory
    #[arg(long = "resume")]
    pub resume: bool,

    /// Quantization method for QLoRA (none, nf4, fp4, int8)
    #[arg(long = "quantization", value_enum)]
    pub quantization: Option<QuantizationMethod>,

    /// Weights per scale for nf4 and int8 (32, 64 or 128; fp4's block is 16)
    #[arg(long = "quant-block-size", default_value_t = 64)]
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

    /// Write per-step metrics as JSONL to this file (a relative path is relative to --output).
    #[arg(long = "log-metrics")]
    pub log_metrics: Option<String>,

    /// Log the loss every this many optimizer steps.
    #[arg(long = "logging-steps", default_value_t = defaults::LOGGING_STEPS)]
    pub logging_steps: usize,

    /// Evaluate every this many optimizer steps (needs --eval-dataset; default 100).
    #[arg(long = "eval-steps")]
    pub eval_steps: Option<usize>,

    /// Write a checkpoint every this many optimizer steps (default 500).
    #[arg(long = "save-steps")]
    pub save_steps: Option<usize>,

    /// Separate learning rate for embedding layers.
    #[arg(long = "embedding-lr")]
    pub embedding_lr: Option<f64>,

    /// Loss scaling factor for ANE training (default: 1.0).
    #[arg(long = "loss-scale", default_value_t = defaults::LOSS_SCALE as f32)]
    pub loss_scale: f32,

    /// Number of linear warmup steps.
    #[arg(long = "warmup-steps", default_value_t = 0)]
    pub warmup_steps: usize,

    /// Share of the optimizer steps spent warming up, 0 to 1; replaces
    /// --warmup-steps when set.
    #[arg(long = "warmup-ratio")]
    pub warmup_ratio: Option<f64>,

    /// Learning rate schedule.
    #[arg(long = "lr-schedule", default_value_t = pmetal_core::LrSchedulerType::Cosine)]
    pub lr_schedule: pmetal_core::LrSchedulerType,

    /// Floor the decaying schedules stop at (default 0).
    #[arg(long = "min-lr")]
    pub min_lr: Option<f64>,

    /// Share of the steps the wsd schedule holds the peak rate for (default 0.7).
    #[arg(long = "wsd-stable-ratio")]
    pub wsd_stable_ratio: Option<f64>,

    /// Shape of the wsd schedule's final decay: linear, cosine or
    /// one_minus_sqrt (default linear).
    #[arg(long = "wsd-decay-style")]
    pub wsd_decay_style: Option<pmetal_core::WsdDecayStyle>,

    /// Cycles of the cosine_with_restarts schedule (default 1).
    #[arg(long = "cosine-num-restarts")]
    pub cosine_num_restarts: Option<usize>,

    /// Exponent of the polynomial schedule (default 1, linear).
    #[arg(long = "polynomial-power")]
    pub polynomial_power: Option<f64>,

    /// Weight decay coefficient.
    #[arg(long = "weight-decay", default_value_t = defaults::WEIGHT_DECAY)]
    pub weight_decay: f64,

    /// Optimizer: adamw, sgd, lion, adafactor, muon, adamw_schedule_free or
    /// adamw_8bit. Lion wants a 3-10x smaller learning rate and 3-10x larger
    /// weight decay than AdamW; schedule-free AdamW wants a constant rate.
    #[arg(long = "optimizer", default_value_t = pmetal_core::OptimizerType::default())]
    pub optimizer: pmetal_core::OptimizerType,

    /// Random seed.
    #[arg(long = "seed", default_value_t = defaults::SEED)]
    pub seed: u64,

    /// Label smoothing of the training loss, in [0, 1): the target is
    /// 1 - epsilon on the label and epsilon spread over the vocabulary.
    /// Evaluation reports the plain loss.
    #[arg(long = "label-smoothing", default_value_t = 0.0)]
    pub label_smoothing: f64,

    /// Weight of the z-loss added to each token's loss, `weight * lse^2`
    /// (PaLM; 1e-4 is the paper's value). 0 turns it off.
    #[arg(long = "z-loss", default_value_t = 0.0)]
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

    /// Separator inserted between columns when using --text-columns
    /// (default: a blank line).
    #[arg(long = "column-separator")]
    pub column_separator: Option<String>,

    /// Custom JSONL column containing the prompt portion (loss-masked).
    #[arg(long = "prompt-column")]
    pub prompt_column: Option<String>,

    /// Custom JSONL column containing the response portion.
    #[arg(long = "response-column")]
    pub response_column: Option<String>,

    /// Enable ANE (Apple Neural Engine) for training (experimental).
    #[arg(long = "ane")]
    pub ane: bool,

    /// Train every weight instead of LoRA adapters, and save a whole model
    /// that `infer` and `serve` load. Needs several times the model's size in
    /// memory (weights, gradients, optimizer state) and is refused up front
    /// when it will not fit. Use a learning rate of 1e-5 to 5e-5.
    #[arg(long = "full")]
    pub full: bool,

    /// With --full, keep the input token embedding as loaded (a model that
    /// ties its output head to it keeps the head too).
    #[arg(long = "freeze-embeddings", requires = "full")]
    pub freeze_embeddings: bool,

    /// With --full, run without checking the memory estimate against the
    /// device's working set.
    #[arg(long = "no-memory-check", requires = "full")]
    pub no_memory_check: bool,

    /// Distributed training: comma-separated peer addresses (ip:port).
    #[arg(long = "distributed-peers", value_delimiter = ',')]
    pub distributed_peers: Option<Vec<String>>,

    /// Distributed training: enable automatic mDNS peer discovery.
    #[arg(long = "distributed-auto")]
    pub distributed_auto: bool,

    /// Gradient compression strategy for distributed training.
    #[arg(long = "compression-strategy", default_value_t = pmetal_core::DistributedCompression::None)]
    pub compression_strategy: pmetal_core::DistributedCompression,
}

impl From<TrainArgs> for TrainSpec {
    fn from(a: TrainArgs) -> Self {
        let csv = |names: Option<Vec<String>>| names.map(|names| names.join(","));
        TrainSpec {
            model: a.model.unwrap_or_default(),
            dataset: a.dataset.unwrap_or_default(),
            eval_dataset: a.eval_dataset,
            output_dir: a.output,
            learning_rate: a.learning_rate,
            embedding_lr: a.embedding_lr,
            label_smoothing: a.label_smoothing,
            z_loss: a.z_loss,
            neftune_alpha: a.neftune_alpha,
            batch_size: a.batch_size,
            epochs: a.epochs,
            max_seq_len: a.max_seq_len,
            gradient_accumulation_steps: a.gradient_accumulation_steps,
            no_gradient_checkpointing: a.no_gradient_checkpointing,
            max_grad_norm: a.max_grad_norm,
            warmup_steps: a.warmup_steps,
            warmup_ratio: a.warmup_ratio,
            min_lr: a.min_lr,
            wsd_stable_ratio: a.wsd_stable_ratio,
            wsd_decay_style: a.wsd_decay_style.map(|style| style.to_string()),
            cosine_num_restarts: a.cosine_num_restarts,
            polynomial_power: a.polynomial_power,
            max_steps: a.max_steps,
            logging_steps: a.logging_steps,
            eval_steps: a.eval_steps,
            save_steps: a.save_steps,
            weight_decay: a.weight_decay,
            optimizer: a.optimizer.to_string(),
            lr_schedule: a.lr_schedule.to_string(),
            seed: a.seed,
            loss_scale: a.loss_scale,
            lora_r: a.lora_r,
            lora_alpha: a.lora_alpha,
            dora: a.dora,
            rslora: a.rslora,
            lora_dropout: a.lora_dropout,
            lora_targets: a.lora_targets,
            quantization: a.quantization.map(|q| q.as_str().to_string()),
            quant_block_size: a.quant_block_size,
            double_quant: a.double_quant,
            text_column: a.text_column,
            text_columns: csv(a.text_columns),
            column_separator: a.column_separator,
            prompt_column: a.prompt_column,
            response_column: a.response_column,
            no_sequence_packing: a.no_sequence_packing,
            cut_cross_entropy: a.cut_cross_entropy,
            no_adaptive_lr: a.no_adaptive_lr,
            ane: a.ane,
            full_finetune: a.full,
            freeze_embeddings: a.freeze_embeddings,
            no_memory_check: a.no_memory_check,
            pack_max_seq_len: a.pack_max_seq_len,
            distributed_peers: csv(a.distributed_peers),
            distributed_auto: a.distributed_auto,
            compression_strategy: a.compression_strategy.to_string(),
            config_path: a.config,
            resume: a.resume,
            log_metrics: a.log_metrics,
        }
    }
}

/// Quantization method for QLoRA.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum QuantizationMethod {
    /// No quantization (standard LoRA)
    #[default]
    None,
    /// NF4 (4-bit NormalFloat), the QLoRA paper's data type - recommended
    Nf4,
    /// NVFP4: E2M1 values with an E4M3 scale per 16 and an FP32 scale per tensor
    Fp4,
    /// 8-bit affine integers
    Int8,
}

impl QuantizationMethod {
    /// The name the flag and the job specs use.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Nf4 => "nf4",
            Self::Fp4 => "fp4",
            Self::Int8 => "int8",
        }
    }
}
