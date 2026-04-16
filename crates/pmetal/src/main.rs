//! PMetal CLI - LLM fine-tuning for Apple Silicon.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::unnecessary_unwrap)]
#![allow(dead_code)] // Chat template formatters — pending migration to pmetal-data

mod commands;
mod dashboard;
#[cfg(any(feature = "models", feature = "native-only"))]
pub mod native_inference;
mod pack_experts;
#[cfg(feature = "dashboard")]
mod tui;

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
#[cfg(feature = "trainer")]
use pmetal_core::{LoraConfig, TrainingConfig};

/// Quantization method for QLoRA.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum QuantizationMethod {
    /// No quantization (standard LoRA)
    #[default]
    None,
    /// NF4 (Normal Float 4-bit) - recommended, optimal for normally distributed weights
    Nf4,
    /// FP4 (Float Point 4-bit)
    Fp4,
    /// INT8 (8-bit integer)
    Int8,
}

/// GGUF quantization method for the `quantize` subcommand.
///
/// Supports all K-quant types available in the pmetal-gguf crate, plus
/// `dynamic` for importance-matrix-guided mixed-precision quantization.
/// K-quant size suffixes follow the GGUF K-quant naming convention:
///   - `s` = small (lower quality, smaller size)
///   - `m` = medium (balanced, recommended)
///   - `l` = large (higher quality, larger size)
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum QuantizeMethod {
    /// Importance-matrix-guided mixed precision (recommended with --imatrix)
    #[default]
    Dynamic,
    /// 8-bit integer (near-lossless, ~1.06 bpw)
    Q8_0,
    /// 6-bit K-quant (high quality, ~0.80 bpw)
    Q6K,
    /// 5-bit K-quant medium (good quality/size balance)
    Q5KM,
    /// 5-bit K-quant small (slightly smaller than q5-k-m)
    Q5KS,
    /// 4-bit K-quant medium (recommended 4-bit, ~0.58 bpw)
    Q4KM,
    /// 4-bit K-quant small (smallest 4-bit K-quant)
    Q4KS,
    /// 3-bit K-quant medium
    Q3KM,
    /// 3-bit K-quant small
    Q3KS,
    /// 3-bit K-quant large
    Q3KL,
    /// 2-bit K-quant (lowest quality, ~0.37 bpw)
    Q2K,
    /// 16-bit float (lossless, 2.0 bpw)
    F16,
    /// 32-bit float (lossless reference precision, 4.0 bpw)
    F32,
}

impl QuantizeMethod {
    /// Canonical display name matching the GGUF K-quant naming convention.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dynamic => "dynamic",
            Self::Q8_0 => "q8_0",
            Self::Q6K => "q6_k",
            Self::Q5KM => "q5_k_m",
            Self::Q5KS => "q5_k_s",
            Self::Q4KM => "q4_k_m",
            Self::Q4KS => "q4_k_s",
            Self::Q3KM => "q3_k_m",
            Self::Q3KS => "q3_k_s",
            Self::Q3KL => "q3_k_l",
            Self::Q2K => "q2_k",
            Self::F16 => "f16",
            Self::F32 => "f32",
        }
    }

    /// Map to the underlying `GgmlType`. Returns `None` for `Dynamic` (uses
    /// `DynamicQuantizer` instead of a fixed type).
    pub fn to_ggml_type(self) -> Option<pmetal_gguf::GgmlType> {
        use pmetal_gguf::GgmlType;
        match self {
            Self::Dynamic => None,
            Self::Q8_0 => Some(GgmlType::Q8_0),
            Self::Q6K => Some(GgmlType::Q6K),
            // All Q5K size variants share the same K-quant block structure
            Self::Q5KM | Self::Q5KS => Some(GgmlType::Q5K),
            // All Q4K size variants share the same K-quant block structure
            Self::Q4KM | Self::Q4KS => Some(GgmlType::Q4K),
            // All Q3K size variants share the same K-quant block structure
            Self::Q3KM | Self::Q3KS | Self::Q3KL => Some(GgmlType::Q3K),
            Self::Q2K => Some(GgmlType::Q2K),
            Self::F16 => Some(GgmlType::F16),
            Self::F32 => Some(GgmlType::F32),
        }
    }
}

/// Inference-context selection for `bench-workload`.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum WorkloadInferenceContext {
    /// Automatically choose between prompt-only and longer text-prefix continuation context
    #[default]
    Auto,
    /// Use the prompt portion only
    Prompt,
    /// Use a prefix of the full formatted sample text
    TextPrefix,
}

/// GDN projection stage for `bench-gdn`.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum GdnBenchmarkStage {
    /// Benchmark the four GDN input projections (`qkv`, `z`, `b`, `a`)
    #[default]
    #[value(name = "input-proj")]
    InputProj,
    /// Benchmark the GDN output projection after recurrent update + norm
    #[value(name = "out-proj")]
    OutProj,
    /// Benchmark the recurrent GDN prefill update across fixed chunk sizes
    #[value(name = "prefill")]
    Prefill,
}

/// Named workload presets for `bench-workload`.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum WorkloadBenchmarkPreset {
    /// Dense Qwen3 regression workload on the cached reasoning dataset
    #[value(name = "dense-qwen3")]
    DenseQwen3,
    /// Hybrid Qwen3Next regression workload with a conservative training cap
    #[value(name = "hybrid-qwen3next")]
    HybridQwen3Next,
    /// Longer-running steady-state hybrid Qwen3.5 inference regression workload
    #[value(name = "hybrid-qwen35-steady")]
    HybridQwen35Steady,
    /// Nemotron-H sparse/hybrid inference regression workload
    #[value(name = "moe-nemotronh")]
    MoeNemotronH,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum TurboQuantPresetArg {
    #[value(name = "q2_5")]
    Q2_5,
    #[value(name = "q3_5")]
    Q3_5,
}

impl From<TurboQuantPresetArg> for pmetal::inference_runner::TurboQuantPreset {
    fn from(value: TurboQuantPresetArg) -> Self {
        match value {
            TurboQuantPresetArg::Q2_5 => Self::Q2_5,
            TurboQuantPresetArg::Q3_5 => Self::Q3_5,
        }
    }
}

#[derive(Parser)]
#[command(name = "pmetal")]
#[command(author, version, about = "LLM fine-tuning optimized for Apple Silicon", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// Fine-tune a model using LoRA/QLoRA
    #[cfg(feature = "trainer")]
    Train {
        /// Path to training configuration file (YAML)
        #[arg(short, long)]
        config: Option<String>,

        /// Model ID (HuggingFace or local path)
        #[arg(short, long)]
        model: Option<String>,

        /// Dataset path (JSONL file)
        #[arg(short, long)]
        dataset: Option<String>,

        /// Evaluation dataset path (optional JSONL file)
        #[arg(long)]
        eval_dataset: Option<String>,

        /// Output directory
        #[arg(short, long, default_value = "./output")]
        output: String,

        /// LoRA rank
        #[arg(long, default_value = "16")]
        lora_r: usize,

        /// LoRA alpha (scaling factor). Recommended: 2x rank.
        #[arg(long, default_value = "32")]
        lora_alpha: f32,

        /// Learning rate. Recommended: 2e-4 for most tasks.
        #[arg(long, default_value = "2e-4")]
        learning_rate: f64,

        /// Batch size
        #[arg(long, default_value = "1")]
        batch_size: usize,

        /// Number of epochs
        #[arg(long, default_value = "1")]
        epochs: usize,

        /// Maximum sequence length (0 to auto-detect from model config)
        #[arg(long, default_value = "0")]
        max_seq_len: usize,

        /// Gradient accumulation steps
        #[arg(long, default_value = "4")]
        gradient_accumulation_steps: usize,

        /// Disable Metal FlashAttention (enabled by default for O(n) memory)
        #[arg(long)]
        no_flash_attention: bool,

        /// Maximum gradient norm for clipping (0 to disable)
        #[arg(long, default_value = "1.0")]
        max_grad_norm: f64,

        /// Resume from checkpoint
        #[arg(long)]
        resume: bool,

        /// Quantization method for QLoRA (none, nf4, fp4, int8)
        #[arg(long, value_enum, default_value = "none")]
        quantization: QuantizationMethod,

        /// Block size for quantization (default: 64)
        #[arg(long, default_value = "64")]
        quant_block_size: usize,

        /// Enable double quantization for absmax values
        #[arg(long)]
        double_quant: bool,

        /// Disable fused training step (enabled by default when gradient_accumulation_steps=1)
        #[arg(long)]
        no_fused: bool,

        /// Disable Metal fused optimizer (enabled by default for ~40% throughput improvement)
        #[arg(long)]
        no_metal_fused_optimizer: bool,

        /// Disable sequence packing (enabled by default for 2-5x throughput)
        #[arg(long)]
        no_sequence_packing: bool,

        /// Override the maximum sequence length for packing.
        ///
        /// Default: auto-detect from dataset p99 of sample lengths, rounded up to the
        /// next power of two and capped at --max-seq-len.  Use this flag to pin the
        /// packing bucket to a specific length (e.g. 512) when the adaptive heuristic
        /// overshoots due to outlier samples.
        #[arg(long)]
        pack_max_seq_len: Option<usize>,

        /// Disable JIT compilation (enabled by default for up to 50% throughput improvement)
        #[arg(long)]
        no_jit_compilation: bool,

        /// Disable gradient checkpointing (enabled by default for memory efficiency)
        #[arg(long)]
        no_gradient_checkpointing: bool,

        /// Number of layers per checkpoint block (default: 4).
        /// Lower = more memory savings but slower.
        #[arg(long, default_value = "4")]
        gradient_checkpointing_layers: usize,

        /// Path to log training metrics as JSONL (Wandb-compatible).
        /// Metrics can be imported to Wandb: `wandb sync path/to/metrics.jsonl`
        #[arg(long)]
        log_metrics: Option<String>,

        /// Separate learning rate for embedding layers.
        /// Recommended: 5e-5 for embeddings vs 2e-4 for LoRA params.
        /// Improves training stability for large vocabulary models.
        #[arg(long)]
        embedding_lr: Option<f32>,

        /// Loss scaling factor for ANE training (default: 1.0).
        /// Multiplies gradients during backward to prevent fp32 underflow
        /// at >350M params. Automatically unscaled before optimizer step.
        #[arg(long, default_value = "1.0")]
        loss_scale: f32,

        /// Number of linear warmup steps before reaching the target learning rate.
        #[arg(long, default_value = "0")]
        warmup_steps: usize,

        /// Learning rate schedule (constant, linear, cosine, cosine_with_restarts, polynomial, wsd).
        #[arg(long, default_value = "cosine")]
        lr_schedule: String,

        /// AdamW weight decay coefficient.
        #[arg(long, default_value = "0.01")]
        weight_decay: f64,

        /// Random seed for dataset shuffling and initialization.
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Use Cut Cross-Entropy for memory-efficient loss computation.
        ///
        /// Avoids materializing the full [batch, seq, vocab] logits tensor, saving
        /// up to 37x peak memory for large-vocabulary models (e.g. Qwen3 with 150K tokens).
        #[arg(long)]
        cut_cross_entropy: bool,

        /// Disable automatic adaptive LR (spike/plateau/divergence detection).
        /// Control file polling stays active for manual LR control via MCP/TUI.
        /// Use this when you want an external agent (LLM) to fully control the learning rate.
        #[arg(long)]
        no_adaptive_lr: bool,

        /// Custom JSONL column containing the training text.
        ///
        /// When set to anything other than "text", bypasses format auto-detection
        /// and reads training data from the named field. Use with --prompt-column
        /// and --response-column for SFT loss masking.
        #[arg(long)]
        text_column: Option<String>,

        /// Multiple JSONL columns to concatenate as the training text.
        ///
        /// Comma-separated column names, e.g. `--text-columns thinking,solution`.
        /// Takes precedence over --text-column. Columns are joined with the
        /// separator specified by --column-separator (default: two newlines).
        #[arg(long, value_delimiter = ',')]
        text_columns: Option<Vec<String>>,

        /// Separator inserted between columns when using --text-columns.
        /// Default: "\n\n" (two newlines).
        #[arg(long, default_value = "\n\n")]
        column_separator: String,

        /// Custom JSONL column containing the prompt portion (loss-masked).
        ///
        /// Tokens from this field receive label -100 and do not contribute to the
        /// loss. Combine with --response-column to concatenate prompt+response.
        #[arg(long)]
        prompt_column: Option<String>,

        /// Custom JSONL column containing the response portion.
        ///
        /// When provided together with --prompt-column, the full training sequence
        /// is prompt || response with prompt tokens masked from the loss.
        #[arg(long)]
        response_column: Option<String>,

        /// Enable ANE (Apple Neural Engine) for training (experimental).
        #[cfg(feature = "ane")]
        #[arg(long)]
        ane: bool,

        /// Distributed training: comma-separated peer addresses (ip:port).
        /// All nodes in the cluster must specify the same peer list.
        /// Set PMETAL_RANK=N env var to specify this node's rank (0-indexed).
        #[cfg(feature = "distributed")]
        #[arg(long, value_delimiter = ',')]
        distributed_peers: Option<Vec<String>>,

        /// Distributed training: enable automatic mDNS peer discovery.
        /// Finds other pmetal nodes on the local network automatically.
        #[cfg(feature = "distributed")]
        #[arg(long)]
        distributed_auto: bool,

        /// Gradient compression strategy for distributed training (none, topk, fp16, random).
        #[cfg(feature = "distributed")]
        #[arg(long, default_value = "none")]
        compression_strategy: Option<String>,
    },

    /// Pretrain a model from scratch (full-parameter, no LoRA)
    #[cfg(feature = "trainer")]
    Pretrain {
        /// Model architecture (e.g. gpt-oss)
        #[arg(short, long)]
        arch: String,

        /// Glob pattern or comma-separated list of tokenized shard files (.bin)
        #[arg(short, long, value_delimiter = ',')]
        shards: Vec<String>,

        /// Target sequence length for packed training batches
        #[arg(long, default_value = "2048")]
        seq_len: usize,

        /// Batch size
        #[arg(long, default_value = "4")]
        batch_size: usize,

        /// Number of training steps
        #[arg(long, default_value = "10000")]
        steps: usize,

        /// Peak learning rate
        #[arg(long, default_value = "3e-4")]
        learning_rate: f32,

        /// Minimum learning rate (cosine floor)
        #[arg(long, default_value = "1e-5")]
        min_lr: f32,

        /// Linear warmup steps
        #[arg(long, default_value = "1000")]
        warmup_steps: usize,

        /// LR schedule (constant, linear, cosine)
        #[arg(long, default_value = "cosine")]
        lr_schedule: String,

        /// AdamW weight decay
        #[arg(long, default_value = "0.1")]
        weight_decay: f32,

        /// Max gradient norm for clipping (0 to disable)
        #[arg(long, default_value = "1.0")]
        max_grad_norm: f32,

        /// EOS token ID (inserted between documents in packed sequences)
        #[arg(long, default_value = "0")]
        eos_token_id: u32,

        /// Output / checkpoint directory
        #[arg(short, long, default_value = "./pretrain-output")]
        output: String,

        /// Save checkpoint every N steps (0 to disable)
        #[arg(long, default_value = "1000")]
        checkpoint_every: usize,

        /// Resume from checkpoint directory
        #[arg(long)]
        resume: Option<String>,

        /// Model config JSON file (overrides arch defaults)
        #[arg(long)]
        model_config: Option<String>,

        /// MoE router z-loss coefficient (0 to disable)
        #[arg(long, default_value = "0.0")]
        z_loss: f32,

        /// Gradient accumulation steps (effective batch = batch_size * this)
        #[arg(long, default_value = "1")]
        gradient_accumulation_steps: usize,

        /// Log step/loss/LR every N steps (0 to disable)
        #[arg(long, default_value = "10")]
        log_every: usize,

        /// Evaluate on held-out data every N steps (0 to disable)
        #[arg(long, default_value = "0")]
        eval_every: usize,

        /// Number of batches per eval round
        #[arg(long, default_value = "10")]
        eval_batches: usize,

        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Run inference with a model
    Infer {
        /// Model ID or path
        #[arg(short, long)]
        model: String,

        /// LoRA adapter path (optional)
        #[arg(long)]
        lora: Option<String>,

        /// Input prompt
        #[arg(short, long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(long, default_value = "256")]
        max_tokens: usize,

        /// Temperature for sampling (0 = greedy). Defaults to model's generation_config.json
        #[arg(long)]
        temperature: Option<f32>,

        /// Top-k sampling (0 = disabled). Defaults to model's generation_config.json
        #[arg(long)]
        top_k: Option<usize>,

        /// Top-p nucleus sampling (0.0-1.0). Defaults to model's generation_config.json
        #[arg(long)]
        top_p: Option<f32>,

        /// Min-p dynamic sampling (0.0 = disabled). Defaults to model's generation_config.json
        #[arg(long)]
        min_p: Option<f32>,

        /// Repetition penalty applied to prompt + output (1.0 = disabled, 1.0-1.2 typical)
        #[arg(long)]
        repetition_penalty: Option<f32>,

        /// Frequency penalty proportional to token count (0.0 = disabled, 0.0-2.0 typical)
        #[arg(long)]
        frequency_penalty: Option<f32>,

        /// Presence penalty for any appeared token (0.0 = disabled, Qwen3 recommends 0-2)
        #[arg(long)]
        presence_penalty: Option<f32>,

        /// Random seed for reproducible generation
        #[arg(long)]
        seed: Option<u64>,

        /// Apply chat template (auto-detected from tokenizer)
        #[arg(long)]
        chat: bool,

        /// System message for chat mode
        #[arg(long)]
        system: Option<String>,

        /// Disable thinking mode for models that support it (e.g., Qwen3)
        /// By default, the model decides when to use thinking based on query complexity
        #[arg(long)]
        no_thinking: bool,

        /// Sampling mode preset with model-card recommended parameters.
        /// Available modes: auto, thinking-general (thinking), thinking-coding (coding),
        /// instruct-general (instruct), instruct-reasoning (reasoning).
        /// "auto" selects based on --no-thinking flag. CLI sampling params override mode defaults.
        #[arg(long, default_value = "auto")]
        mode: pmetal_data::inference_config::SamplingMode,

        /// Execution backend: auto | standard | compiled | metal-sampler | ane | minimal | dflash.
        /// `auto` picks the best path for the current device + model (default). Explicit
        /// variants pin a backend. `dflash` also requires `--draft-model`.
        /// Takes precedence over the legacy `--compiled` / `--metal-sampler` / `--minimal`
        /// / `--ane` flags when set to anything other than `auto`.
        #[arg(long, default_value = "auto")]
        backend: pmetal_data::inference_config::InferenceBackend,

        /// Draft model for speculative decoding (HF id or local path).
        /// Required when `--backend dflash`; ignored otherwise.
        #[arg(long)]
        draft_model: Option<String>,

        /// Use fused Metal sampling kernel for better battery performance
        /// (bypasses mlx-rs sampling, uses single GPU kernel launch).
        /// Legacy alias — prefer `--backend metal-sampler`.
        #[arg(long)]
        metal_sampler: bool,

        /// Use JIT-compiled sampling for better performance
        /// (matches mlx_lm's @mx.compile approach for kernel fusion).
        /// Legacy alias — prefer `--backend compiled`.
        #[arg(long)]
        compiled: bool,

        /// Use dedicated GPU stream for generation.
        /// NOTE: Currently a no-op placeholder; streaming generation is not yet implemented.
        #[arg(long, hide = true)]
        stream: bool,

        /// Use minimal async generation (for performance debugging)
        #[arg(long)]
        minimal: bool,

        /// Hide thinking trace from output (shown by default for thinking models)
        #[arg(long)]
        hide_thinking: bool,

        /// Path to a JSON file containing tool/function definitions (OpenAI format).
        /// Tools are injected into the system prompt using the model's native format.
        /// Example: [{"type":"function","function":{"name":"get_weather","description":"...","parameters":{...}}}]
        #[arg(long)]
        tools: Option<String>,

        /// Use FP8 quantization for weights (~2x memory reduction).
        /// Quantizes model weights to 8-bit floating point (E4M3 format)
        /// for memory-efficient inference on Apple Silicon.
        #[arg(long)]
        fp8: bool,

        /// Path to packed expert weights directory for SSD-offloaded MoE inference.
        /// Created with `pmetal pack-experts`. Enables expert prefetching for
        /// models that don't fit in memory (e.g., Qwen3.5-397B on 48GB).
        #[arg(long)]
        experts_dir: Option<String>,

        /// Enable ANE (Apple Neural Engine) for inference (experimental).
        #[cfg(feature = "ane")]
        #[arg(long)]
        ane: bool,

        /// Maximum ANE kernel sequence length (power-of-2 bucket cap).
        /// ANE kernels are compiled for a fixed spatial dimension — larger values
        /// allow longer prompts to be processed on ANE but may fail to compile
        /// for models with many attention heads. Default: 1024.
        #[cfg(feature = "ane")]
        #[arg(long, default_value = "1024")]
        ane_max_seq_len: usize,

        /// Use the experimental ANE real-time evaluation path when ANE inference is selected.
        #[cfg(feature = "ane")]
        #[arg(long)]
        ane_real_time: bool,

        /// Run an MLX-LM-compatible benchmark with random prompt ids, one warmup,
        /// EOS disabled, and fixed generation length.
        #[arg(long)]
        benchmark: bool,

        /// Number of measured trials for benchmarking (default: 5)
        #[arg(long, default_value = "5")]
        benchmark_iters: usize,

        /// Synthetic prompt length for --benchmark.
        #[arg(long)]
        benchmark_prompt_tokens: Option<usize>,

        /// Run an opt-in per-layer forward profile for supported hybrid models.
        /// Currently implemented for Qwen 3.5 / qwen3_next standard inference.
        #[arg(long)]
        profile_layers: bool,

        /// Write the layer profile report as pretty JSON.
        #[arg(long)]
        profile_output: Option<String>,

        /// KV cache quantization bits (8=q8_0, 4=q4_0, 0=fp16).
        /// Omit to auto-select the fastest mode that still fits the device budget.
        #[arg(long)]
        kv_quant: Option<u8>,

        /// KV cache key bits (overrides --kv-quant for keys only, for asymmetric K/V).
        #[arg(long)]
        kv_k_bits: Option<u8>,

        /// KV cache value bits (overrides --kv-quant for values only, for asymmetric K/V).
        #[arg(long)]
        kv_v_bits: Option<u8>,

        /// KV cache quantization group size.
        #[arg(long, default_value = "64")]
        kv_group_size: usize,

        /// Use TurboQuant for KV cache compression.
        #[arg(long)]
        kv_turboquant: bool,

        /// Mixed-bit TurboQuant preset (`q2_5` or `q3_5`).
        /// When set, this enables outlier-aware TurboQuant without needing `--kv-turboquant`.
        #[arg(long, value_enum)]
        kv_turboquant_preset: Option<TurboQuantPresetArg>,

        /// TurboQuant v2 mixed-bit affine preset: "q2_5" (2.5-bit) or "q3_5" (3.5-bit).
        /// Enables outlier channel detection + split-bit quantized KV cache (native path only).
        #[arg(long, value_parser = ["q2_5", "q3_5"])]
        kv_quant_preset: Option<String>,

        /// Disable KV cache quantization (use fp16 KV cache).
        #[arg(long)]
        no_kv_quant: bool,

        /// Enable QJL residual correction for Q2-Q3 key quantization.
        ///
        /// Stores 1-bit sign vectors on quantization residuals and applies an
        /// additive correction to attention scores, making key inner products
        /// unbiased. Only active for --kv-quant 2 or 3 (uniform, non-mixed-bit).
        /// Adds ~2-5% latency overhead from one extra D×D matmul per decode step.
        #[arg(long)]
        kv_qjl: bool,

        /// Enable n-gram repetition loop detection.
        /// Force-stops generation when the same 8-token pattern repeats 4 times.
        /// Useful for small models prone to infinite loops in thinking mode.
        #[arg(long)]
        detect_repetition: bool,
    },

    /// Download a model from HuggingFace
    Download {
        /// Model ID
        model: String,

        /// Specific revision
        #[arg(long)]
        revision: Option<String>,
    },

    /// Search HuggingFace Hub for models and show device fit
    Search {
        /// Search query (e.g. "qwen3 0.6B", "llama 8b")
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "15")]
        limit: usize,

        /// Download the first result that fits on your device
        #[arg(long)]
        download: bool,

        /// Show detailed fit analysis (memory breakdown, training estimates)
        #[arg(long)]
        detailed: bool,

        /// Output results as JSON
        #[arg(long)]
        json: bool,
    },

    /// Block-diffusion speculative decoding (DFlash).
    ///
    /// Runs a Qwen3 target alongside a DFlash draft to emit multiple tokens
    /// per forward pass with bit-identical output to greedy baseline.
    Dflash {
        /// Target Qwen3 model (HF id or local path).
        #[arg(long)]
        target: String,

        /// DFlash draft model (HF id or local path, e.g. `z-lab/Qwen3-4B-DFlash-b16`).
        #[arg(long)]
        draft: String,

        /// Input prompt.
        #[arg(short, long)]
        prompt: String,

        /// Maximum tokens to generate after the prompt.
        #[arg(long, default_value = "128")]
        max_new_tokens: usize,

        /// Sampling temperature (0 = greedy, bit-identical to baseline).
        #[arg(long, default_value = "0.0")]
        temperature: f32,

        /// Override the draft block size (defaults to the draft checkpoint's `block_size`).
        #[arg(long)]
        speculative_tokens: Option<usize>,

        /// Quantize the draft's Linear weights to FP8 (E4M3) on load.
        /// Halves draft memory with minimal acceptance-rate impact vs BF16.
        #[arg(long)]
        draft_fp8: bool,

        /// Emit a JSON report of tokens + metrics instead of plain text.
        #[arg(long)]
        json: bool,

        /// Skip the model's chat template and tokenize the prompt verbatim.
        /// By default pmetal dflash mirrors upstream dflash-mlx and applies
        /// the target's chat template (with `enable_thinking=false`) — the
        /// draft was trained against chat-templated target hidden states,
        /// so running raw inputs collapses acceptance to near zero.
        #[arg(long)]
        no_chat: bool,

        /// Tree-verify budget — when set to `N > 0`, the decoder
        /// builds a budget-N candidate tree per round (inspired by
        /// DDTree) and lets the target walk the tree via posterior
        /// argmax. Typical values: 16-64. Set to 0 to force linear
        /// DFlash (the default).
        #[arg(long, default_value = "0")]
        tree_budget: usize,
    },

    /// Show memory usage and available capacity
    Memory {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Benchmark FFI overhead (for performance analysis)
    #[cfg(feature = "trainer")]
    BenchFfi,

    /// Benchmark generation loop timing (detailed profiling)
    #[cfg(feature = "trainer")]
    BenchGen {
        /// Model to benchmark
        #[arg(short, long, default_value = "Qwen/Qwen3-0.6B")]
        model: String,
    },

    /// Benchmark training performance
    #[cfg(feature = "trainer")]
    Bench {
        /// Model to benchmark
        #[arg(short, long, default_value = "meta-llama/Llama-3.2-1B")]
        model: String,

        /// Batch size
        #[arg(short, long, default_value = "1")]
        batch_size: usize,

        /// Sequence length
        #[arg(short, long, default_value = "512")]
        seq_len: usize,
    },

    /// Run a structured kernel benchmark corpus for this device tier
    #[cfg(feature = "trainer")]
    BenchCorpus {
        /// Use a shorter run with fewer iterations and smaller tier-scaled shapes
        #[arg(long)]
        quick: bool,

        /// Output results as JSON
        #[arg(long)]
        json: bool,

        /// Optional output path for the JSON report
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Benchmark a real cached workload for inference and short LoRA training
    #[cfg(feature = "trainer")]
    BenchWorkload {
        /// Named preset that overrides the model/dataset/shape knobs below
        #[arg(long, value_enum)]
        preset: Option<WorkloadBenchmarkPreset>,

        /// Model ID or local path
        #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
        model: String,

        /// Dataset ID or local path
        #[arg(
            long,
            default_value = "TeichAI/gemini-3-pro-preview-high-reasoning-250x"
        )]
        dataset: String,

        /// Packed expert directory for sparse/offloaded inference workloads
        #[arg(long)]
        experts_dir: Option<String>,

        /// Number of prompt samples to benchmark for inference
        #[arg(long, default_value = "8")]
        prompt_samples: usize,

        /// Maximum prompt tokens per inference sample (0 = auto-select from sampled prompts, capped for a quick run)
        #[arg(long, default_value = "0")]
        max_prompt_tokens: usize,

        /// Inference context source for workload benchmarking
        #[arg(long, value_enum, default_value = "auto")]
        inference_context: WorkloadInferenceContext,

        /// Number of decode steps per prompt sample
        #[arg(long, default_value = "32")]
        decode_steps: usize,

        /// Number of untimed warmup passes to run per sampled inference prompt before measurement
        #[arg(long, default_value = "2")]
        inference_warmup_passes: usize,

        /// Number of independent warmed inference sessions to run and aggregate
        #[arg(long, default_value = "1")]
        inference_session_repeats: usize,

        /// Number of timed inference passes per prompt sample
        #[arg(long, default_value = "1")]
        inference_repeats: usize,

        /// Number of raw samples to include in the sampled training subset
        #[arg(long, default_value = "8")]
        train_samples: usize,

        /// Number of LoRA training steps to run
        #[arg(long, default_value = "4")]
        train_steps: usize,

        /// Training batch size
        #[arg(long, default_value = "1")]
        batch_size: usize,

        /// Maximum sequence length for the short training run (0 = auto-select from sampled data, capped for a quick run)
        #[arg(long, default_value = "0")]
        max_seq_len: usize,

        /// Output results as JSON
        #[arg(long)]
        json: bool,

        /// Optional output path for the JSON report
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Benchmark Qwen3.5 GDN backends on the actual model layer shapes
    #[cfg(feature = "trainer")]
    BenchGdn {
        /// Qwen3.5/Qwen3Next model ID or local path
        #[arg(long, default_value = "unsloth/Qwen3.5-0.8B")]
        model: String,

        /// Projection stage to benchmark
        #[arg(long, value_enum, default_value = "input-proj")]
        stage: GdnBenchmarkStage,

        /// Optional transformer layer index to benchmark (defaults to the first GDN layer)
        #[arg(long)]
        layer: Option<usize>,

        /// Batch size for the synthetic benchmark input
        #[arg(long, default_value = "1")]
        batch_size: usize,

        /// Sequence length for the synthetic input (use a longer value for prefill benchmarking)
        #[arg(long, default_value = "1")]
        seq_len: usize,

        /// Warmup iterations per backend
        #[arg(long, default_value = "10")]
        warmup_iterations: usize,

        /// Timed iterations per backend
        #[arg(long, default_value = "50")]
        benchmark_iterations: usize,

        /// Output results as JSON
        #[arg(long)]
        json: bool,

        /// Optional output path for the JSON report
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Generate a sample configuration file
    #[cfg(feature = "trainer")]
    Init {
        /// Output path for the config file
        #[arg(short, long, default_value = "config.yaml")]
        output: String,
    },

    /// Pack expert weights for SSD-offloaded MoE inference
    PackExperts {
        /// Model directory (containing config.json and safetensors)
        #[arg(short, long)]
        model: String,

        /// Output directory for packed expert files
        #[arg(short, long, default_value = "./packed_experts")]
        output: String,

        /// Quantization bit width (4 or 2)
        #[arg(short, long)]
        bits: Option<u8>,
    },

    /// Export trained model for Ollama
    Ollama {
        #[command(subcommand)]
        action: OllamaAction,
    },

    /// Fuse LoRA adapter weights into a base model and save as a complete model
    #[cfg(feature = "lora")]
    Fuse {
        /// Base model ID or path
        #[arg(short, long)]
        model: String,

        /// LoRA adapter path (directory containing lora_weights.safetensors, or the file itself)
        #[arg(short, long)]
        lora: String,

        /// Output directory for the fused model
        #[arg(short, long)]
        output: String,

        /// LoRA scaling alpha (default: auto-detect from adapter)
        #[arg(long)]
        alpha: Option<f32>,

        /// LoRA rank (default: auto-detect from adapter)
        #[arg(long)]
        rank: Option<usize>,

        /// Use f64-accurate LoRA merge (reads adapter_config.json, performs B@A in f64,
        /// writes merged weights in the original storage dtype).
        /// More numerically accurate than the default f32 path.
        #[arg(long, default_value_t = false)]
        accurate: bool,

        /// Use tiled low-memory mode with the --accurate path.
        /// Limits peak f64 allocation to tile_size rows of B at a time.
        /// Only has effect when --accurate is also set.
        #[arg(long, default_value_t = false)]
        low_memory: bool,
    },

    /// Quantize a model to GGUF format (supports Dynamic 2.0 and KL-calibrated)
    Quantize {
        /// Source model path (Safetensors/HF)
        #[arg(short, long)]
        model: String,

        /// Output GGUF file path
        #[arg(short, long)]
        output: String,

        /// Path to Importance Matrix (imatrix.dat) for dynamic quantization
        #[arg(long)]
        imatrix: Option<String>,

        /// Quantization method
        #[arg(long, value_enum, default_value = "dynamic")]
        method: QuantizeMethod,

        /// LoRA adapter to fuse before quantizing (optional)
        #[arg(long)]
        lora: Option<String>,

        /// Use KL-divergence calibration for per-tensor quantization type selection.
        /// Tests multiple quantization types per tensor and picks the one minimizing
        /// quality loss while meeting the threshold and optional BPW budget.
        #[arg(long)]
        kl_calibrate: bool,

        /// Target average bits per weight for KL calibration (e.g. 4.5).
        /// When set, the calibrator will downgrade low-impact tensors until the
        /// budget is satisfied.
        #[arg(long)]
        target_bpw: Option<f32>,

        /// Quality-loss threshold for KL calibration (default: 0.01).
        /// Lower values preserve more quality; higher values allow more compression.
        #[arg(long, default_value = "0.01")]
        kl_threshold: f64,

        /// Output format: "gguf" (default) or "mlx" (MLX safetensors with per-tensor bit
        /// allocation).  When "mlx" is selected the --output argument is treated as an
        /// output directory path rather than a file.
        #[arg(long, default_value = "gguf")]
        format: String,

        /// Default bit-width for MLX-format quantization (3, 4, 5, 6, or 8).
        /// Only used when --format mlx is set.
        #[arg(long, default_value_t = 4)]
        bits: i32,

        /// Group size for MLX-format quantization.
        /// Only used when --format mlx is set.
        #[arg(long, default_value_t = 64)]
        group_size: i32,
    },

    /// Knowledge Distillation from teacher to student
    #[cfg(feature = "trainer")]
    Distill {
        /// Teacher model ID or path
        #[arg(short, long)]
        teacher: String,

        /// Student model ID or path
        #[arg(short, long)]
        student: String,

        /// Dataset path (JSONL file)
        #[arg(short, long)]
        dataset: String,

        /// Output directory for distilled student
        #[arg(short, long, default_value = "./output/distilled")]
        output: String,

        /// Distillation method (online, offline, progressive)
        #[arg(long, default_value = "online")]
        method: String,

        /// Loss type (kl_divergence, jensen_shannon, soft_cross_entropy, mse_loss)
        #[arg(long, default_value = "kl_divergence")]
        loss_type: String,

        /// Softmax temperature
        #[arg(long, default_value = "2.0")]
        temperature: f32,

        /// Alpha for blending hard/soft targets (0.0 to 1.0)
        #[arg(long, default_value = "0.5")]
        alpha: f32,

        /// Use reasoning-aware (rationale) distillation
        #[arg(long)]
        rationale: bool,

        /// Weight for reasoning tokens (if rationale is enabled)
        #[arg(long, default_value = "1.0")]
        rationale_weight: f32,

        /// LoRA rank for student
        #[arg(long, default_value = "16")]
        lora_r: usize,

        /// LoRA alpha scaling factor
        #[arg(long, default_value = "32")]
        lora_alpha: f32,

        /// Learning rate
        #[arg(long, default_value = "2e-5")]
        learning_rate: f32,

        /// Batch size
        #[arg(long, default_value = "1")]
        batch_size: usize,

        /// Number of epochs
        #[arg(long, default_value = "1")]
        epochs: usize,

        /// Maximum sequence length
        #[arg(long, default_value = "1024")]
        max_seq_len: usize,

        /// Random seed for dataset shuffling and initialization.
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Custom text column name in the dataset JSONL.
        #[arg(long)]
        text_column: Option<String>,

        /// Comma-separated list of columns to concatenate as the text field.
        #[arg(long, value_delimiter = ',')]
        text_columns: Option<Vec<String>>,

        /// Separator used when joining multiple text columns.
        #[arg(long, default_value = "\n\n")]
        column_separator: String,

        /// Column name for the prompt portion (enables SFT label masking).
        #[arg(long)]
        prompt_column: Option<String>,

        /// Column name for the response portion (enables SFT label masking).
        #[arg(long)]
        response_column: Option<String>,

        /// Path to write JSONL metrics log (for TUI dashboard)
        #[arg(long)]
        log_metrics: Option<String>,
    },

    /// Group Relative Policy Optimization (GRPO) for reasoning models
    #[cfg(feature = "trainer")]
    Grpo {
        /// Model ID or path
        #[arg(short, long)]
        model: String,

        /// Dataset path (JSONL with prompts)
        #[arg(short, long)]
        dataset: String,

        /// Output directory
        #[arg(short, long, default_value = "./output/grpo")]
        output: String,

        /// Number of generations per prompt (group size)
        #[arg(long, default_value = "8")]
        num_generations: usize,

        /// KL penalty coefficient (beta)
        #[arg(long, default_value = "0.001")]
        beta: f64,

        /// Learning rate
        #[arg(long, default_value = "5e-6")]
        learning_rate: f64,

        /// Number of training epochs
        #[arg(long, default_value = "1")]
        epochs: usize,

        /// LoRA rank for policy model
        #[arg(long, default_value = "16")]
        lora_r: usize,

        /// LoRA alpha scaling factor
        #[arg(long, default_value = "32")]
        lora_alpha: f32,

        /// Maximum sequence length for generations
        #[arg(long, default_value = "512")]
        max_seq_len: usize,

        /// Maximum completion length per generation
        #[arg(long, default_value = "512")]
        max_completion_length: usize,

        /// Random seed for reproducibility
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Enable DAPO (Distribution-Aware Policy Optimization)
        #[arg(long)]
        dapo: bool,

        /// Use reasoning-aware rewards (e.g., length, formatting)
        #[arg(long)]
        reasoning_rewards: bool,

        /// Disable Metal FlashAttention
        #[arg(long)]
        no_flash_attention: bool,

        /// Enable VLM (Vision-Language Model) mode for GRPO with image inputs.
        ///
        /// When set, images are loaded from each dataset sample's `images` field,
        /// passed to reward functions, and used in forward passes via
        /// `forward_with_images`.  Requires a multimodal dataset (JSONL with
        /// `"images": ["/path/to/img.jpg", ...]` per sample).
        #[arg(long)]
        vlm: bool,

        /// Maximum image size (pixels per side) for VLM preprocessing.
        ///
        /// Images are resized to fit within a square of this size while maintaining
        /// aspect ratio.  336 matches CLIP ViT-L/14; 448 and 560 are common for
        /// larger vision encoders.
        #[arg(long, default_value = "336")]
        max_image_size: usize,

        /// Path to a pretrained ML reward model for scoring completions.
        ///
        /// When set, the model is loaded at training start and used alongside any
        /// heuristic reward functions.  Accepts a local model directory or a
        /// HuggingFace model ID (e.g. "RLHFlow/ArmoRM-Llama3-8B-v0.1").
        ///
        /// The reward model runs inference-only — it is not fine-tuned.
        #[arg(long)]
        reward_model: Option<String>,

        /// Maximum input sequence length for the ML reward model (tokens).
        ///
        /// Inputs longer than this are truncated from the right.
        #[arg(long, default_value = "2048")]
        reward_model_max_length: usize,

        /// Weight for the ML reward model in the combined reward.
        ///
        /// The final reward is a weighted sum of all reward functions.  Set to
        /// a value less than 1.0 to blend the ML reward with heuristic rewards.
        #[arg(long, default_value = "1.0")]
        reward_model_weight: f64,

        /// Chat template for the reward model (optional).
        ///
        /// Use `{prompt}` and `{completion}` as placeholders, e.g.:
        /// `"Human: {prompt}\nAssistant: {completion}"`
        ///
        /// When omitted, prompt and completion are concatenated directly.
        #[arg(long)]
        reward_model_template: Option<String>,

        /// Enable speculative decoding for faster rollout generation.
        ///
        /// Uses a draft/verify approach: generate `--speculative-draft-tokens` cheap
        /// draft tokens, then verify them all in a single batched forward pass.
        /// Expected speedup: 2–4x over standard autoregressive generation.
        ///
        /// Automatically disabled for models that do not support KV caching.
        #[arg(long)]
        speculative: bool,

        /// Number of draft tokens per speculative decode step (default: 3).
        ///
        /// Higher values yield more throughput at high acceptance rates but increase
        /// overhead when the draft distribution diverges from the full model.
        /// Typical range: 2–5.  Ignored unless `--speculative` is set.
        #[arg(long, default_value = "3")]
        speculative_draft_tokens: usize,

        /// Enable pipelined (asynchronous) reward scoring.
        ///
        /// When set, reward scoring for step N runs in a background thread
        /// concurrently with GPU training for step N+1.  This is most effective
        /// when using an ML reward model (`--reward-model`) whose inference is
        /// CPU- or ANE-bound while the policy model trains on the GPU.
        ///
        /// For pure heuristic rewards (format, accuracy), the scoring is
        /// so fast that pipelining provides negligible benefit.
        #[arg(long)]
        async_rewards: bool,

        /// Custom text column name in the dataset JSONL.
        #[arg(long)]
        text_column: Option<String>,

        /// Comma-separated list of columns to concatenate as the text field.
        #[arg(long, value_delimiter = ',')]
        text_columns: Option<Vec<String>>,

        /// Separator used when joining multiple text columns.
        #[arg(long, default_value = "\n\n")]
        column_separator: String,

        /// Column name for the prompt portion (enables SFT label masking).
        #[arg(long)]
        prompt_column: Option<String>,

        /// Column name for the response portion (enables SFT label masking).
        #[arg(long)]
        response_column: Option<String>,

        /// KV cache quantization bits for GRPO rollout generation (2, 4, or 8).
        ///
        /// When set, the KV cache used during completion generation is stored at the
        /// requested bit width instead of fp16.  This reduces peak memory during rollout
        /// by 2–8× depending on the bit width, enabling longer completions or larger
        /// generation group sizes on memory-constrained hardware.
        ///
        /// Valid values: 2, 4, 8.  Omit to use the default fp16 cache.
        #[arg(long)]
        grpo_kv_bits: Option<u8>,

        /// Path to write JSONL metrics log (for TUI dashboard)
        #[arg(long)]
        log_metrics: Option<String>,
    },

    /// RLKD: Reinforcement Learning with Knowledge Distillation.
    ///
    /// Combines GRPO policy gradient optimization with knowledge distillation
    /// from a teacher model in a single training loop.
    ///
    /// Loss formula: L = (1 - alpha) * L_grpo + alpha * L_distill
    #[cfg(feature = "trainer")]
    Rlkd {
        /// Policy (student) model ID or local path.
        #[arg(short, long)]
        model: String,

        /// Teacher model ID or local path (frozen, provides soft targets).
        #[arg(long)]
        teacher_model: String,

        /// Dataset path (JSONL with prompts).
        #[arg(short, long)]
        dataset: String,

        /// Output directory for LoRA adapter weights.
        #[arg(short, long, default_value = "./output/rlkd")]
        output: String,

        /// Distillation blend factor: 0.0 = pure RL, 1.0 = pure distillation.
        ///
        /// When `--anneal-alpha` is set this is the starting value; the factor
        /// is linearly annealed toward `--final-alpha` over training.
        #[arg(long, default_value = "0.3")]
        distill_alpha: f32,

        /// Final alpha value when annealing (default: 0.05 = mostly RL by end).
        #[arg(long, default_value = "0.05")]
        final_alpha: f32,

        /// Linearly anneal alpha from `--distill-alpha` toward `--final-alpha`.
        ///
        /// This shifts training from distillation-guided early on to RL-driven
        /// at the end, which typically improves final task performance.
        #[arg(long)]
        anneal_alpha: bool,

        /// Temperature for distillation soft targets (default: 2.0).
        ///
        /// Higher temperatures soften the teacher distribution, transferring
        /// more information about non-dominant token probabilities.
        #[arg(long, default_value = "2.0")]
        distill_temperature: f32,

        /// Number of completions to generate per prompt (GRPO group size).
        #[arg(long, default_value = "8")]
        num_generations: usize,

        /// KL penalty coefficient (beta) for GRPO reference model regularization.
        #[arg(long, default_value = "0.001")]
        beta: f64,

        /// Learning rate.
        #[arg(long, default_value = "5e-6")]
        learning_rate: f64,

        /// Number of training epochs.
        #[arg(long, default_value = "1")]
        epochs: usize,

        /// LoRA rank for the policy model.
        #[arg(long, default_value = "16")]
        lora_r: usize,

        /// LoRA alpha scaling factor.
        #[arg(long, default_value = "32")]
        lora_alpha: f32,

        /// Maximum sequence length (prompt + completion).
        #[arg(long, default_value = "512")]
        max_seq_len: usize,

        /// Maximum completion length per generation.
        #[arg(long, default_value = "512")]
        max_completion_length: usize,

        /// Random seed for reproducibility.
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Use reasoning-aware rewards (format + length signals).
        #[arg(long)]
        reasoning_rewards: bool,

        /// Disable Metal FlashAttention.
        #[arg(long)]
        no_flash_attention: bool,

        /// Custom text column name in the dataset JSONL.
        #[arg(long)]
        text_column: Option<String>,

        /// Comma-separated list of columns to concatenate as the text field.
        #[arg(long, value_delimiter = ',')]
        text_columns: Option<Vec<String>>,

        /// Separator used when joining multiple text columns.
        #[arg(long, default_value = "\n\n")]
        column_separator: String,

        /// Column name for the prompt portion (enables SFT label masking).
        #[arg(long)]
        prompt_column: Option<String>,

        /// Column name for the response portion (enables SFT label masking).
        #[arg(long)]
        response_column: Option<String>,

        /// Path to write JSONL metrics log (for TUI dashboard).
        #[arg(long)]
        log_metrics: Option<String>,
    },

    /// Start MCP server for Claude Desktop integration
    #[cfg(feature = "mcp")]
    Mcp,

    /// Start an OpenAI-compatible inference server
    #[cfg(feature = "serve")]
    Serve {
        /// Model ID or path
        #[arg(short, long)]
        model: String,

        /// LoRA adapter path (optional)
        #[arg(long)]
        lora: Option<String>,

        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Host to bind to
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// Maximum sequence length for KV cache
        #[arg(long, default_value = "4096")]
        max_seq_len: usize,

        /// Path to packed expert weights directory for SSD-offloaded MoE inference.
        /// Created with `pmetal pack-experts`.
        #[arg(long)]
        experts_dir: Option<String>,

        /// Quantize weights to FP8 E4M3 at load time (~2x memory savings).
        #[arg(long)]
        fp8: bool,

        /// KV cache quantization bits: 8 = q8_0, 4 = q4_0, 0 = fp16.
        /// Omit for auto-selection based on model size and device memory budget.
        #[arg(long)]
        kv_quant: Option<u8>,

        /// Disable KV cache quantization (use fp16 KV cache).
        #[arg(long)]
        no_kv_quant: bool,

        /// Quantization group size for KV cache.
        #[arg(long, default_value = "64")]
        kv_group_size: usize,

        /// Enable TurboQuant KV cache compression (provably near-optimal, data-oblivious).
        /// Reduces KV cache memory 4-6x with near-zero quality loss.
        #[arg(long)]
        kv_turboquant: bool,

        /// TurboQuant quality preset: q2_5 (2.5 bits, 6.4x compression) or q3_5 (3.5 bits, 4.6x compression, lossless).
        #[arg(long, value_parser = ["q2_5", "q3_5"])]
        kv_turboquant_preset: Option<String>,

        /// Enable ANE (Apple Neural Engine) for serving (experimental).
        #[cfg(feature = "ane")]
        #[arg(long)]
        ane: bool,

        /// Maximum ANE kernel sequence length (power-of-2 bucket cap).
        /// Higher values allow longer prompts on ANE but may fail to compile on
        /// wider models. Default: 1024.
        #[cfg(feature = "ane")]
        #[arg(long, default_value = "1024")]
        ane_max_seq_len: usize,

        /// Use the experimental ANE real-time evaluation path when ANE serving is selected.
        #[cfg(feature = "ane")]
        #[arg(long)]
        ane_real_time: bool,
    },

    /// Dataset utilities for preparing and analyzing training data
    Dataset {
        #[command(subcommand)]
        action: DatasetAction,
    },

    /// Tokenize a text corpus into binary shards for pretraining
    Tokenize {
        /// Input JSONL file
        #[arg(short, long)]
        input: String,
        /// Output directory for shard files
        #[arg(short, long)]
        output: String,
        /// Tokenizer model ID or path (HuggingFace format)
        #[arg(short, long)]
        tokenizer: String,
        /// JSONL column containing text (default: "text")
        #[arg(long, default_value = "text")]
        text_column: String,
        /// Maximum documents per shard (default: 10000)
        #[arg(long, default_value = "10000")]
        docs_per_shard: usize,
    },

    /// Real-time training dashboard (loss curves, ANE utilization, timing)
    Dashboard {
        /// Path to training metrics JSONL file to visualize
        #[arg(short, long)]
        metrics_file: Option<String>,
    },

    /// Full TUI control center (device, models, datasets, training, inference)
    #[cfg(feature = "dashboard")]
    Tui {
        /// Path to training metrics JSONL file to visualize in the dashboard tab
        #[arg(short, long)]
        metrics_file: Option<String>,
    },

    /// Show device information (GPU architecture, ANE cores, bandwidth, NAX support)
    Info {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Merge two or more models using various merge methods (SLERP, TIES, DARE, linear, etc.)
    #[cfg(feature = "merge")]
    Merge {
        /// First model path or HuggingFace ID
        #[arg(short = 'a', long)]
        model_a: String,

        /// Second model path or HuggingFace ID
        #[arg(short = 'b', long)]
        model_b: String,

        /// Output directory for merged model
        #[arg(short, long)]
        output: String,

        /// Merge method (linear, slerp, ties, dare_ties, dare_linear, task_arithmetic, della, breadcrumbs, model_stock, nearswap, passthrough)
        #[arg(long, default_value = "slerp")]
        method: String,

        /// Base model for task-vector methods (TIES, DARE, task_arithmetic)
        #[arg(long)]
        base: Option<String>,

        /// Interpolation parameter t for SLERP (0.0=model_a, 1.0=model_b)
        #[arg(long, default_value = "0.5")]
        t: f32,

        /// Weight for model_a in linear/ties methods
        #[arg(long, default_value = "0.5")]
        weight_a: f32,

        /// Weight for model_b in linear/ties methods
        #[arg(long, default_value = "0.5")]
        weight_b: f32,

        /// Density for sparsification (TIES/DARE) — fraction of params to keep
        #[arg(long, default_value = "0.5")]
        density: f32,

        /// Output dtype (float32, float16, bfloat16)
        #[arg(long, default_value = "bfloat16")]
        dtype: String,
    },

    /// Evaluate a model's perplexity on a dataset
    #[cfg(feature = "lora")]
    Eval {
        /// Model ID or path
        #[arg(short, long)]
        model: String,

        /// Dataset path (JSONL file)
        #[arg(short, long)]
        dataset: String,

        /// LoRA adapter path (optional)
        #[arg(long)]
        lora: Option<String>,

        /// Maximum sequence length
        #[arg(long, default_value = "1024")]
        max_seq_len: usize,

        /// Number of samples to evaluate (0 = all)
        #[arg(long, default_value = "0")]
        num_samples: usize,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Train a sentence embedding model (BERT / encoder-only) with contrastive losses.
    ///
    /// Supports InfoNCE (default), triplet, CoSENT, and cosine-similarity losses.
    /// Input data must be JSONL with pair or triplet format.
    ///
    /// Pair JSONL:
    ///   {"text_a": "...", "text_b": "...", "label": 0.9}
    ///   {"sentence1": "...", "sentence2": "..."}
    ///   {"query": "...", "positive": "..."}
    ///
    /// Triplet JSONL:
    ///   {"anchor": "...", "positive": "...", "negative": "..."}
    #[command(name = "embed-train")]
    #[cfg(feature = "trainer")]
    EmbedTrain {
        /// Path to the BERT / encoder model directory.
        #[arg(short, long)]
        model: String,

        /// Path to the training dataset (JSONL pairs or triplets).
        #[arg(short, long)]
        dataset: String,

        /// Output directory for trained model weights.
        #[arg(short, long, default_value = "./output-embed")]
        output: String,

        /// Contrastive loss function.
        /// Options: info_nce (default), mnrl, triplet, cosent, cosine_similarity
        #[arg(long, default_value = "info_nce")]
        loss: String,

        /// Pooling strategy for sentence embeddings.
        /// Options: mean (default), cls, max, last_token, weighted_mean
        #[arg(long, default_value = "mean")]
        pooling: String,

        /// Temperature for InfoNCE / CoSENT losses.
        #[arg(long, default_value = "0.05")]
        temperature: f32,

        /// Margin for triplet loss.
        #[arg(long, default_value = "0.3")]
        margin: f32,

        /// Learning rate.
        #[arg(long, default_value = "2e-5")]
        learning_rate: f64,

        /// Training batch size.
        #[arg(long, default_value = "32")]
        batch_size: usize,

        /// Number of training epochs.
        #[arg(long, default_value = "3")]
        epochs: usize,

        /// Maximum input sequence length.
        #[arg(long, default_value = "512")]
        max_seq_len: usize,

        /// AdamW weight decay.
        #[arg(long, default_value = "0.01")]
        weight_decay: f64,

        /// Disable L2 normalisation of embeddings before loss.
        #[arg(long)]
        no_normalize: bool,

        /// Log training progress every N steps.
        #[arg(long, default_value = "10")]
        log_every: usize,

        /// Random seed for dataset shuffling.
        #[arg(long, default_value = "42")]
        seed: u64,
    },
}

/// Dataset subcommands for data preparation.
#[derive(Subcommand)]
enum DatasetAction {
    /// Analyze a dataset and show statistics
    Analyze {
        /// Path to dataset file (JSONL)
        #[arg(short, long)]
        path: String,

        /// Model ID for tokenization (required for accurate token counts)
        #[arg(short, long)]
        model: Option<String>,

        /// Show detailed per-sample statistics
        #[arg(long)]
        detailed: bool,
    },

    /// Download a dataset from HuggingFace Hub
    Download {
        /// Dataset ID (e.g., "tatsu-lab/alpaca", "TeichAI/gemini-3-pro-preview-high-reasoning-1000x")
        dataset_id: String,

        /// Dataset split to download (e.g., "train", "test")
        #[arg(long, default_value = "train")]
        split: String,

        /// Output path for converted JSONL file
        #[arg(short, long)]
        output: Option<String>,

        /// Specific revision/branch
        #[arg(long)]
        revision: Option<String>,
    },

    /// Convert a dataset to pmetal-compatible JSONL format
    Convert {
        /// Input file path (Parquet, JSON, JSONL, CSV)
        #[arg(short, long)]
        input: String,

        /// Output file path (JSONL)
        #[arg(short, long)]
        output: String,

        /// Input format (auto-detected if not specified)
        #[arg(long, value_enum)]
        format: Option<InputFormat>,

        /// Column mapping for non-standard formats (e.g., "text=content,prompt=instruction")
        #[arg(long)]
        columns: Option<String>,

        /// Shuffle the output data
        #[arg(long)]
        shuffle: bool,

        /// Random seed for shuffling
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Validate a dataset for training
    Validate {
        /// Path to dataset file (JSONL)
        #[arg(short, long)]
        path: String,

        /// Model ID for tokenization validation
        #[arg(short, long)]
        model: Option<String>,

        /// Maximum sequence length to check against
        #[arg(long, default_value = "2048")]
        max_seq_len: usize,
    },

    /// Preview first N samples from a HuggingFace dataset
    Preview {
        /// Dataset ID (e.g., "tatsu-lab/alpaca")
        dataset_id: String,

        /// Dataset split (e.g., "train", "test")
        #[arg(long, default_value = "train")]
        split: String,

        /// Number of samples to preview
        #[arg(short, long, default_value = "5")]
        num: usize,
    },

    /// Filter a dataset by various criteria
    Filter {
        /// Input dataset path (JSONL)
        #[arg(short, long)]
        input: String,

        /// Output dataset path (JSONL)
        #[arg(short, long)]
        output: String,

        /// Model ID for token-based filtering
        #[arg(short, long)]
        model: Option<String>,

        /// Minimum token count (requires --model)
        #[arg(long)]
        min_tokens: Option<usize>,

        /// Maximum token count (requires --model)
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Remove duplicate samples (exact match)
        #[arg(long)]
        dedup: bool,

        /// Regex pattern to match (keeps matching samples)
        #[arg(long)]
        pattern: Option<String>,

        /// Invert pattern matching (keeps non-matching samples)
        #[arg(long)]
        invert: bool,

        /// Require all conversation turns (filters incomplete conversations)
        #[arg(long)]
        complete_only: bool,
    },

    /// Split a dataset into train/validation/test sets
    Split {
        /// Input dataset path (JSONL)
        #[arg(short, long)]
        input: String,

        /// Output directory for split files
        #[arg(short, long)]
        output_dir: String,

        /// Validation set ratio (0.0 to 1.0)
        #[arg(long, default_value = "0.1")]
        val_ratio: f64,

        /// Test set ratio (0.0 to 1.0)
        #[arg(long, default_value = "0.0")]
        test_ratio: f64,

        /// Random seed for splitting
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Stratify by a field (e.g., "category", "difficulty")
        #[arg(long)]
        stratify: Option<String>,
    },

    /// Merge multiple datasets into one
    Merge {
        /// Input dataset paths (JSONL)
        #[arg(short, long, num_args = 1..)]
        inputs: Vec<String>,

        /// Output dataset path (JSONL)
        #[arg(short, long)]
        output: String,

        /// Shuffle after merging
        #[arg(long)]
        shuffle: bool,

        /// Random seed for shuffling
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Interleave samples from each dataset (alternating)
        #[arg(long)]
        interleave: bool,

        /// Weights for each input dataset (comma-separated, e.g., "1.0,2.0,0.5")
        #[arg(long)]
        weights: Option<String>,
    },

    /// Take a random sample from a dataset
    Sample {
        /// Input dataset path (JSONL)
        #[arg(short, long)]
        input: String,

        /// Output dataset path (JSONL)
        #[arg(short, long)]
        output: String,

        /// Number of samples to take
        #[arg(short, long)]
        num: usize,

        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Apply a chat template to format conversations
    Template {
        /// Input dataset path (JSONL with conversations)
        #[arg(short, long)]
        input: String,

        /// Output dataset path (JSONL with formatted text)
        #[arg(short, long)]
        output: String,

        /// Chat template to use
        #[arg(long, value_enum, default_value = "chatml")]
        template: ChatTemplate,

        /// Custom system message (overrides default)
        #[arg(long)]
        system: Option<String>,

        /// Model ID for tokenizer-based template (uses tokenizer's chat_template)
        #[arg(short, long)]
        model: Option<String>,

        /// Add generation prompt marker at end
        #[arg(long)]
        add_generation_prompt: bool,

        /// Only mask prompt tokens in labels (for SFT)
        #[arg(long)]
        mask_prompt: bool,
    },

    /// Prepare a dataset for training (full pipeline)
    Prepare {
        /// HuggingFace dataset ID or local path
        dataset: String,

        /// Output directory
        #[arg(short, long)]
        output_dir: String,

        /// Model ID for tokenization
        #[arg(short, long)]
        model: String,

        /// Chat template to apply
        #[arg(long, value_enum, default_value = "chatml")]
        template: ChatTemplate,

        /// Maximum sequence length (filters longer samples)
        #[arg(long, default_value = "2048")]
        max_seq_len: usize,

        /// Validation split ratio
        #[arg(long, default_value = "0.05")]
        val_ratio: f64,

        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Skip deduplication
        #[arg(long)]
        no_dedup: bool,

        /// Column mapping: remap source columns to standard names.
        /// Format: "target=source,target=source" (e.g., "instruction=problem,output=solution")
        #[arg(long)]
        columns: Option<String>,
    },

    /// Show supported formats and templates
    Formats,
}

/// Chat template formats for conversation formatting.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum ChatTemplate {
    /// ChatML format: <|im_start|>role\ncontent<|im_end|>
    Chatml,
    /// Llama 3 format: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
    Llama3,
    /// Llama 2 format: [INST] user [/INST] assistant </s>
    Llama2,
    /// Mistral/Zephyr format: <s>[INST] user [/INST] assistant</s>
    Mistral,
    /// Qwen format: <|im_start|>role\ncontent<|im_end|>
    Qwen,
    /// Phi format: <|user|>\ncontent<|end|>\n<|assistant|>\ncontent<|end|>
    Phi,
    /// Gemma format: <start_of_turn>role\ncontent<end_of_turn>
    Gemma,
    /// Raw text (no template, just concatenate)
    Raw,
    /// Use tokenizer's built-in chat template
    Auto,
}

/// Input format for dataset conversion.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum InputFormat {
    /// Parquet format
    Parquet,
    /// JSON array format
    Json,
    /// JSONL (JSON Lines) format
    Jsonl,
    /// CSV format
    Csv,
    /// ShareGPT conversation format
    ShareGpt,
    /// Alpaca instruction format
    Alpaca,
}

/// Ollama subcommands for model export and registration.
#[derive(Subcommand)]
enum OllamaAction {
    /// Generate a Modelfile for a trained model
    Modelfile {
        /// Base model (GGUF path or Ollama model name)
        #[arg(short, long)]
        base: String,

        /// LoRA adapter path (optional)
        #[arg(long)]
        lora: Option<String>,

        /// Output Modelfile path
        #[arg(short, long, default_value = "Modelfile")]
        output: String,

        /// System prompt
        #[arg(long)]
        system: Option<String>,

        /// Temperature (0.0-2.0)
        #[arg(long)]
        temperature: Option<f32>,

        /// Context window size
        #[arg(long)]
        num_ctx: Option<i32>,

        /// Top-k sampling
        #[arg(long)]
        top_k: Option<i32>,

        /// Top-p nucleus sampling
        #[arg(long)]
        top_p: Option<f32>,

        /// Model template (auto-detected from architecture if not specified)
        #[arg(long, value_enum)]
        template: Option<OllamaTemplate>,

        /// License text for the model
        #[arg(long)]
        license: Option<String>,
    },

    /// Create and register a model with Ollama
    Create {
        /// Model name for Ollama (e.g., "my-finetuned-model")
        #[arg(short, long)]
        name: String,

        /// Base model (GGUF path or Ollama model name)
        #[arg(short, long)]
        base: String,

        /// LoRA adapter path (optional)
        #[arg(long)]
        lora: Option<String>,

        /// System prompt
        #[arg(long)]
        system: Option<String>,

        /// Temperature (0.0-2.0)
        #[arg(long)]
        temperature: Option<f32>,

        /// Context window size
        #[arg(long)]
        num_ctx: Option<i32>,

        /// Model template (auto-detected from architecture if not specified)
        #[arg(long, value_enum)]
        template: Option<OllamaTemplate>,
    },

    /// List available templates
    Templates,
}

/// Ollama template presets.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OllamaTemplate {
    /// Llama 3 chat format
    Llama3,
    /// Qwen3/ChatML format
    Qwen3,
    /// Gemma instruct format
    Gemma,
    /// Mistral instruct format
    Mistral,
    /// Phi-3 instruct format
    Phi3,
    /// DeepSeek chat format
    DeepSeek,
}

/// Gzip-compressed `mlx.metallib` bytes (baked in at compile time by build.rs).
/// ~31MB compressed in the binary, ~102MB decompressed on disk.
#[cfg(target_os = "macos")]
static MLX_METALLIB_GZ: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mlx.metallib.gz"));

/// Extract the embedded `mlx.metallib` to `~/.cache/pmetal/lib/mlx.metallib`,
/// decompressing the gzip'd blob on first run. Skips if a file of non-zero
/// size already exists at the destination.
/// Returns the path if extraction succeeded.
#[cfg(target_os = "macos")]
fn extract_embedded_metallib() -> Option<PathBuf> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    if MLX_METALLIB_GZ.is_empty() {
        return None;
    }
    let home = std::env::var("HOME").ok()?;
    let cache_dir = PathBuf::from(home).join(".cache/pmetal/lib");
    let dest = cache_dir.join("mlx.metallib");

    // Skip if already present with non-zero size
    if dest.is_file() {
        if let Ok(meta) = dest.metadata() {
            if meta.len() > 0 {
                return Some(dest);
            }
        }
    }

    std::fs::create_dir_all(&cache_dir).ok()?;
    let tmp = dest.with_extension("metallib.tmp");

    let mut decoder = GzDecoder::new(MLX_METALLIB_GZ);
    let mut decompressed = Vec::new();
    if decoder.read_to_end(&mut decompressed).is_err() {
        return None;
    }

    std::fs::write(&tmp, &decompressed).ok()?;
    std::fs::rename(&tmp, &dest).ok()?;
    tracing::info!("Extracted embedded mlx.metallib to {}", dest.display());
    Some(dest)
}

/// Discover `mlx.metallib` and set `PMETAL_METALLIB_PATH` so the patched MLX
/// C++ backend can find it regardless of where the binary is installed.
///
/// Search order: colocated → build dir → cache → Homebrew → embedded → download → error.
#[allow(unsafe_code)]
fn ensure_metallib() {
    let metallib_name = "mlx.metallib";
    let cache_dir = std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".cache/pmetal/lib"));
    let mut search_paths: Vec<PathBuf> = Vec::new();

    // 1. Colocated with binary
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            search_paths.push(dir.join(metallib_name));
        }
    }

    // 1b. Search build directories (for local development)
    if let Ok(cwd) = std::env::current_dir() {
        // Search in common build output locations relative to workspace root
        let build_patterns = [
            "target/release/build",
            "target/debug/build",
            "target/aarch64-apple-darwin/release/build",
            "target/aarch64-apple-darwin/debug/build",
        ];

        for pattern in build_patterns {
            let build_dir = cwd.join(pattern);
            if build_dir.exists() {
                // Look for mlx-sys build output specifically
                if let Ok(entries) = std::fs::read_dir(build_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_dir()
                            && path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .is_some_and(|s| s.contains("pmetal-mlx-sys"))
                        {
                            let metallib_candidate = path.join("out/build/lib").join(metallib_name);
                            if metallib_candidate.is_file() {
                                search_paths.push(metallib_candidate);
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. User cache ($HOME/.cache/pmetal/lib/) — matches C++ fallback path
    if let Some(ref cache) = cache_dir {
        search_paths.push(cache.join(metallib_name));
    }
    // 3. Homebrew Apple Silicon
    search_paths.push("/opt/homebrew/lib/mlx.metallib".into());
    // 4. Homebrew Intel / standard
    search_paths.push("/usr/local/lib/mlx.metallib".into());

    if let Some(path) = search_paths.iter().find(|p| p.is_file()) {
        // SAFETY: called before any other threads; env var is process-internal
        unsafe {
            std::env::set_var("PMETAL_METALLIB_PATH", path);
            // Q1 2026 Best Practice: Enable JIT for architecture-specific optimizations (M4+)
            std::env::set_var("MLX_METAL_JIT", "1");
        };
        tracing::debug!(path = %path.display(), "Found mlx.metallib");

        // Auto-cache: if found colocated with binary, copy to cache dir for resilience
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                if path.as_path() == dir.join(metallib_name) {
                    if let Some(ref cache) = cache_dir {
                        let dest = cache.join(metallib_name);
                        if !dest.exists() {
                            let _ = std::fs::create_dir_all(cache);
                            if std::fs::copy(path, &dest).is_ok() {
                                tracing::debug!("Cached metallib to {}", dest.display());
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // Try extracting the embedded metallib (baked in at compile time)
    if let Some(path) = extract_embedded_metallib() {
        unsafe {
            std::env::set_var("PMETAL_METALLIB_PATH", &path);
            std::env::set_var("MLX_METAL_JIT", "1");
        };
        tracing::debug!(path = %path.display(), "Using embedded mlx.metallib");
        return;
    }

    // Not found locally — try to auto-download from GitHub releases
    if let Some(ref cache) = cache_dir {
        let dest = cache.join(metallib_name);
        if download_metallib(&dest) {
            unsafe { std::env::set_var("PMETAL_METALLIB_PATH", &dest) };
            return;
        }
    }

    // Download failed or no HOME — print actionable instructions
    let locations = search_paths
        .iter()
        .map(|p| format!("  - {}", p.display()))
        .collect::<Vec<_>>()
        .join("\n");
    eprintln!(
        "\n\x1b[1;33mwarning:\x1b[0m mlx.metallib not found — Metal GPU acceleration will fail.\n\n\
         Searched:\n{locations}\n\n\
         To fix this, do ONE of the following:\n\
         \x1b[1m  1. Rebuild from source:\x1b[0m  cargo install pmetal  (auto-caches metallib)\n\
         \x1b[1m  2. Download manually:\x1b[0m\n\
             curl -fSL -o ~/.cache/pmetal/lib/mlx.metallib \\\n\
               https://github.com/epistates/pmetal/releases/download/v{version}/mlx.metallib\n\
         \x1b[1m  3. Homebrew:\x1b[0m             brew install epistates/tap/pmetal\n\n\
         The metallib ships with every pmetal release and is built during compilation.\n\
         See: https://github.com/epistates/pmetal#installation\n",
        version = env!("CARGO_PKG_VERSION"),
    );
}

/// SHA-256 of the released `mlx.metallib` artifact.
///
/// Empty string in dev builds.  Set to the hex digest of the artifact for
/// official releases so every download is cryptographically verified before
/// being committed to the cache.
const METALLIB_SHA256: &str = match option_env!("PMETAL_METALLIB_SHA256") {
    Some(s) => s,
    None => "",
};

/// Download `mlx.metallib` from the matching GitHub release.
///
/// Uses `reqwest` (no subprocess) and verifies the SHA-256 of the downloaded
/// bytes when `METALLIB_SHA256` is non-empty.  Writes to a temp file first,
/// then atomically renames into place so a failed download never leaves a
/// corrupted cache entry.
///
/// Returns `true` if the download succeeded and the file is verified.
fn download_metallib(dest: &std::path::Path) -> bool {
    let version = env!("CARGO_PKG_VERSION");
    let url =
        format!("https://github.com/epistates/pmetal/releases/download/v{version}/mlx.metallib");

    eprintln!(
        "\x1b[1;36minfo:\x1b[0m mlx.metallib not found locally. \
         Downloading from GitHub releases (one-time setup)..."
    );

    if let Some(parent) = dest.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // Use a temp file so a partial download never leaves a broken metallib.
    let tmp = dest.with_extension("metallib.tmp");

    // Build a single-threaded tokio runtime for the blocking download
    // (this function is called before the main async runtime is started).
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("\x1b[1;33mwarning:\x1b[0m Failed to create download runtime: {e}");
            return false;
        }
    };

    let bytes: Vec<u8> = match rt.block_on(async {
        let client = reqwest::Client::builder()
            .user_agent(concat!("pmetal/", env!("CARGO_PKG_VERSION")))
            .build()?;
        let response = client
            .get(&url)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?;
        Ok::<_, reqwest::Error>(response.to_vec())
    }) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "\x1b[1;33mwarning:\x1b[0m Download failed (offline or release v{version} \
                 not published yet): {e}\n"
            );
            return false;
        }
    };

    // Verify SHA-256 when a reference digest is compiled in.
    if !METALLIB_SHA256.is_empty() {
        use sha2::Digest as _;
        let digest = sha2::Sha256::digest(&bytes);
        let hex = format!("{digest:x}");
        if hex != METALLIB_SHA256 {
            eprintln!(
                "\x1b[1;31merror:\x1b[0m SHA-256 mismatch for mlx.metallib\n\
                 expected: {METALLIB_SHA256}\n\
                 got:      {hex}\n\
                 The downloaded file has been discarded."
            );
            return false;
        }
    }

    // Write to temp file, then atomically rename into place.
    if let Err(e) = std::fs::write(&tmp, &bytes) {
        eprintln!("\x1b[1;33mwarning:\x1b[0m Failed to write temp metallib: {e}");
        return false;
    }

    if std::fs::rename(&tmp, dest).is_ok() {
        eprintln!(
            "\x1b[1;32minfo:\x1b[0m Cached mlx.metallib to {}\n",
            dest.display()
        );
        true
    } else {
        let _ = std::fs::remove_file(&tmp);
        false
    }
}

/// Synchronous entry point.
///
/// `ensure_metallib` calls `std::env::set_var` which is unsound when other
/// threads are running.  By invoking it here — before the `#[tokio::main]`
/// macro starts the async runtime and its thread pool — we guarantee that
/// the environment mutation happens in a single-threaded context.
fn main() -> anyhow::Result<()> {
    ensure_metallib();
    #[cfg(feature = "metal")]
    let _ = pmetal_metal::context::MetalContext::device_available();
    tokio_main()
}

// ---------------------------------------------------------------------------
// Persistent logging
// ---------------------------------------------------------------------------

/// Initialize logging with file output to `~/.cache/pmetal/logs/{component}.log`.
///
/// - **CLI mode**: logs to both stderr and file
/// - **TUI mode**: logs to file only (stderr would corrupt the terminal), or
///   to `PMETAL_LOG_FILE` if set for backwards compatibility
///
/// The previous log is rotated to `{component}.log.1`.
fn init_logging(component: &str, suppress_stderr: bool) {
    // Determine log file path
    let log_path = if let Ok(custom) = std::env::var("PMETAL_LOG_FILE") {
        std::path::PathBuf::from(custom)
    } else {
        let dir = pmetal_log_dir();
        let path = dir.join(format!("{component}.log"));

        // Rotate: keep one previous log
        let prev = dir.join(format!("{component}.log.1"));
        if path.exists() {
            let _ = std::fs::rename(&path, &prev);
        }
        path
    };

    let filter = tracing_subscriber::EnvFilter::from_default_env()
        .add_directive(tracing::Level::INFO.into());

    if suppress_stderr {
        // TUI: file only (no stderr to avoid corrupting terminal)
        if let Ok(file) = std::fs::File::create(&log_path) {
            tracing_subscriber::fmt()
                .with_writer(std::sync::Mutex::new(file))
                .with_env_filter(filter)
                .with_ansi(false)
                .init();
        }
    } else {
        // CLI: stderr + tee to file
        let file = std::fs::File::create(&log_path).ok();
        let file = file.map(std::sync::Mutex::new).map(std::sync::Arc::new);
        let file_clone = file.clone();

        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(move || TeeWriter {
                stderr: std::io::stderr(),
                file: file_clone.clone(),
            })
            .init();
    }
}

fn pmetal_log_dir() -> std::path::PathBuf {
    let dir = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
        .join("pmetal")
        .join("logs");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Writer that tees output to both stderr and a log file.
struct TeeWriter {
    stderr: std::io::Stderr,
    file: Option<std::sync::Arc<std::sync::Mutex<std::fs::File>>>,
}

impl std::io::Write for TeeWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.stderr.write(buf)?;
        if let Some(ref file) = self.file {
            if let Ok(mut f) = file.lock() {
                let _ = f.write_all(&buf[..n]);
            }
        }
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.stderr.flush()?;
        if let Some(ref file) = self.file {
            if let Ok(mut f) = file.lock() {
                let _ = f.flush();
            }
        }
        Ok(())
    }
}

#[tokio::main]
async fn tokio_main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize logging — suppress stderr output when running the TUI
    // to avoid corrupting the raw terminal display.
    #[cfg(feature = "dashboard")]
    let is_tui = matches!(cli.command, Commands::Tui { .. });
    #[cfg(not(feature = "dashboard"))]
    let is_tui = false;
    init_logging(if is_tui { "tui" } else { "cli" }, is_tui);

    match cli.command {
        #[cfg(feature = "trainer")]
        Commands::Train {
            config,
            model,
            dataset,
            eval_dataset,
            output,
            lora_r,
            lora_alpha,
            learning_rate,
            batch_size,
            epochs,
            max_seq_len,
            gradient_accumulation_steps,
            no_flash_attention,
            max_grad_norm,
            resume,
            quantization,
            quant_block_size,
            double_quant,
            no_fused,
            no_metal_fused_optimizer,
            no_sequence_packing,
            pack_max_seq_len,
            no_jit_compilation,
            no_gradient_checkpointing,
            gradient_checkpointing_layers,
            log_metrics,
            embedding_lr,
            loss_scale,
            warmup_steps,
            lr_schedule,
            weight_decay,
            seed,
            cut_cross_entropy,
            no_adaptive_lr,
            text_column,
            text_columns,
            column_separator,
            prompt_column,
            response_column,
            #[cfg(feature = "ane")]
            ane,
            #[cfg(feature = "distributed")]
            distributed_peers,
            #[cfg(feature = "distributed")]
            distributed_auto,
            #[cfg(feature = "distributed")]
            compression_strategy,
        } => {
            use pmetal_trainer::orchestrator;

            // Parse LR schedule
            let lr_scheduler = match lr_schedule.to_lowercase().as_str() {
                "constant" => pmetal_core::LrSchedulerType::Constant,
                "linear" => pmetal_core::LrSchedulerType::Linear,
                "cosine" => pmetal_core::LrSchedulerType::Cosine,
                "cosine_with_restarts" => pmetal_core::LrSchedulerType::CosineWithRestarts,
                "polynomial" => pmetal_core::LrSchedulerType::Polynomial,
                "wsd" => pmetal_core::LrSchedulerType::Wsd,
                other => {
                    anyhow::bail!(
                        "Unknown lr-schedule '{}'. Valid: constant, linear, cosine, cosine_with_restarts, polynomial, wsd",
                        other
                    );
                }
            };

            // Build QLoRA config if quantization is requested
            let qlora = if !matches!(quantization, QuantizationMethod::None) {
                Some(orchestrator::QLoraOrchConfig {
                    scheme: match quantization {
                        QuantizationMethod::Nf4 => orchestrator::QuantizationScheme::Nf4,
                        QuantizationMethod::Fp4 => orchestrator::QuantizationScheme::Fp4,
                        QuantizationMethod::Int8 => orchestrator::QuantizationScheme::Int8,
                        QuantizationMethod::None => unreachable!(),
                    },
                    block_size: quant_block_size,
                    double_quant,
                })
            } else {
                None
            };

            // Build column config
            let columns = commands::build_column_config(
                text_column,
                text_columns,
                column_separator,
                prompt_column,
                response_column,
            );

            // Build distributed config
            #[cfg(feature = "distributed")]
            let distributed = {
                let has_peers = distributed_peers.as_ref().is_some_and(|p| !p.is_empty());
                if has_peers || distributed_auto {
                    let compression = match compression_strategy.as_deref() {
                        Some("topk") => pmetal_core::DistributedCompression::TopK,
                        Some("fp16") => pmetal_core::DistributedCompression::Fp16,
                        Some("random") => pmetal_core::DistributedCompression::Random,
                        _ => pmetal_core::DistributedCompression::None,
                    };
                    Some(pmetal_core::DistributedTrainingConfig {
                        peers: distributed_peers.unwrap_or_default(),
                        auto_discover: distributed_auto,
                        compression,
                        ..Default::default()
                    })
                } else {
                    None
                }
            };

            let job_config = orchestrator::TrainingJobConfig {
                model_id: model.unwrap_or_default(),
                dataset: dataset.unwrap_or_default(),
                eval_dataset,
                output_dir: output,
                lora: LoraConfig {
                    r: lora_r,
                    alpha: lora_alpha,
                    ..Default::default()
                },
                qlora,
                training: TrainingConfig {
                    learning_rate,
                    batch_size,
                    num_epochs: epochs,
                    max_seq_len,
                    gradient_accumulation_steps,
                    max_grad_norm,
                    warmup_steps,
                    weight_decay,
                    lr_scheduler,
                    embedding_learning_rate: embedding_lr.map(|v| v as f64),
                    ..Default::default()
                },
                columns,
                dispatch: orchestrator::DispatchConfig {
                    flash_attention: !no_flash_attention,
                    sequence_packing: !no_sequence_packing,
                    pack_max_seq_len,
                    jit_compilation: !no_jit_compilation,
                    fused: !no_fused,
                    metal_fused_optimizer: !no_metal_fused_optimizer,
                    gradient_checkpointing: !no_gradient_checkpointing,
                    gradient_checkpointing_layers,
                    cut_cross_entropy,
                    no_adaptive_lr,
                    #[cfg(feature = "ane")]
                    ane,
                    #[cfg(not(feature = "ane"))]
                    ane: false,
                    loss_scale,
                    #[cfg(feature = "distributed")]
                    distributed,
                },
                config_path: config,
                log_metrics,
                resume,
                seed,
                emit_console_output: true,
            };

            orchestrator::run_training(job_config, None, Vec::new()).await?;
        }

        #[cfg(feature = "mcp")]
        Commands::Mcp => {
            pmetal_mcp::run_stdio()
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }

        #[cfg(feature = "serve")]
        Commands::Serve {
            model,
            lora,
            port,
            host,
            max_seq_len,
            experts_dir,
            fp8,
            kv_quant,
            no_kv_quant,
            kv_group_size,
            kv_turboquant,
            kv_turboquant_preset,
            #[cfg(feature = "ane")]
            ane,
            #[cfg(feature = "ane")]
            ane_max_seq_len,
            #[cfg(feature = "ane")]
            ane_real_time,
        } => {
            #[cfg(feature = "ane")]
            let ane_enabled = ane;
            #[cfg(not(feature = "ane"))]
            let ane_enabled = false;

            #[cfg(feature = "ane")]
            let serve_ane_max_seq_len = ane_max_seq_len;
            #[cfg(not(feature = "ane"))]
            let serve_ane_max_seq_len = 1024;
            #[cfg(feature = "ane")]
            let serve_ane_real_time = ane_real_time;
            #[cfg(not(feature = "ane"))]
            let serve_ane_real_time = false;

            commands::serve::run_serve(
                model,
                lora,
                port,
                host,
                max_seq_len,
                experts_dir,
                fp8,
                kv_quant,
                no_kv_quant,
                kv_group_size,
                kv_turboquant,
                kv_turboquant_preset,
                ane_enabled,
                serve_ane_max_seq_len,
                serve_ane_real_time,
            )
            .await?;
        }

        Commands::Infer {
            model,
            lora,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            seed,
            chat,
            system,
            no_thinking,
            mode,
            backend,
            draft_model,
            metal_sampler,
            compiled,
            stream,
            minimal,
            hide_thinking,
            tools,
            fp8,
            experts_dir,
            #[cfg(feature = "ane")]
            ane,
            #[cfg(feature = "ane")]
            ane_max_seq_len,
            #[cfg(feature = "ane")]
            ane_real_time,
            benchmark,
            benchmark_iters,
            benchmark_prompt_tokens,
            profile_layers,
            profile_output,
            kv_quant,
            kv_k_bits,
            kv_v_bits,
            kv_group_size,
            kv_turboquant,
            kv_turboquant_preset,
            kv_quant_preset,
            no_kv_quant,
            kv_qjl,
            detect_repetition,
        } => {
            // Load tool definitions if provided
            let tool_defs: Option<Vec<pmetal_data::chat_templates::ToolDefinition>> =
                if let Some(ref tools_path) = tools {
                    let tools_json = std::fs::read_to_string(tools_path)
                        .map_err(|e| anyhow::anyhow!("Failed to read tools file: {}", e))?;
                    let defs: Vec<pmetal_data::chat_templates::ToolDefinition> =
                        serde_json::from_str(&tools_json)
                            .map_err(|e| anyhow::anyhow!("Failed to parse tools JSON: {}", e))?;
                    tracing::info!("Loaded {} tool definitions from {}", defs.len(), tools_path);
                    Some(defs)
                } else {
                    None
                };

            let validated_profile_output = profile_output
                .as_deref()
                .map(|path| validate_output_path(path, "infer profile output"))
                .transpose()?;

            commands::infer::run_inference(
                &model,
                lora.as_deref(),
                &prompt,
                max_tokens,
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
                frequency_penalty,
                presence_penalty,
                seed,
                chat,
                system.as_deref(),
                no_thinking,
                mode,
                backend,
                draft_model.as_deref(),
                metal_sampler,
                compiled,
                stream,
                minimal,
                hide_thinking,
                fp8,
                tool_defs.as_deref(),
                #[cfg(feature = "ane")]
                ane,
                #[cfg(not(feature = "ane"))]
                false,
                #[cfg(feature = "ane")]
                ane_max_seq_len,
                #[cfg(not(feature = "ane"))]
                1024,
                #[cfg(feature = "ane")]
                ane_real_time,
                #[cfg(not(feature = "ane"))]
                false,
                benchmark,
                benchmark_iters,
                benchmark_prompt_tokens,
                profile_layers,
                validated_profile_output.as_deref(),
                kv_quant,
                kv_k_bits,
                kv_v_bits,
                kv_group_size,
                kv_turboquant,
                kv_turboquant_preset.map(Into::into),
                kv_quant_preset,
                no_kv_quant,
                kv_qjl,
                detect_repetition,
                experts_dir.as_deref(),
            )
            .await?;
        }

        Commands::Download { model, revision } => {
            tracing::info!(model = %model, "Downloading model");
            let path = pmetal_hub::download_model(&model, revision.as_deref(), None).await?;
            println!("Model downloaded to: {}", path.display());
        }

        Commands::PackExperts {
            model,
            output,
            bits,
        } => {
            pack_experts::pack_experts(Path::new(&model), Path::new(&output), bits)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }

        Commands::Search {
            query,
            limit,
            download,
            detailed,
            json,
        } => {
            commands::search::run_search(&query, limit, download, detailed, json).await?;
        }

        Commands::Dflash {
            target,
            draft,
            prompt,
            max_new_tokens,
            temperature,
            speculative_tokens,
            draft_fp8,
            json,
            no_chat,
            tree_budget,
        } => {
            commands::dflash::run_dflash(
                &target,
                &draft,
                &prompt,
                max_new_tokens,
                temperature,
                speculative_tokens,
                draft_fp8,
                json,
                no_chat,
                tree_budget,
            )
            .await?;
        }

        Commands::Memory { json } => {
            let stats = pmetal_mlx::memory::get_memory_stats();
            if json {
                let obj = serde_json::json!({
                    "total_gb": stats.total_gb(),
                    "used_gb": stats.used_gb(),
                    "available_gb": stats.available_gb(),
                    "peak_gb": stats.peak_gb(),
                });
                println!("{}", serde_json::to_string_pretty(&obj)?);
            } else {
                println!("Memory Statistics:");
                println!("  Total:     {:.2} GB", stats.total_gb());
                if stats.total_gb() > 0.0 {
                    let used_pct = (stats.used_gb() / stats.total_gb()) * 100.0;
                    let peak_pct = (stats.peak_gb() / stats.total_gb()) * 100.0;
                    println!("  Used:      {:.2} GB ({:.0}%)", stats.used_gb(), used_pct);
                    println!(
                        "  Available: {:.2} GB ({:.0}%)",
                        stats.available_gb(),
                        100.0 - used_pct
                    );
                    println!("  Peak:      {:.2} GB ({:.0}%)", stats.peak_gb(), peak_pct);
                } else {
                    println!("  Used:      {:.2} GB", stats.used_gb());
                    println!("  Available: {:.2} GB", stats.available_gb());
                    println!("  Peak:      {:.2} GB", stats.peak_gb());
                }
            }
        }

        #[cfg(feature = "trainer")]
        Commands::Bench {
            model,
            batch_size,
            seq_len,
        } => {
            commands::bench::run_benchmark(&model, batch_size, seq_len).await?;
        }

        #[cfg(feature = "trainer")]
        Commands::BenchCorpus {
            quick,
            json,
            output,
        } => {
            let validated_output = output
                .as_deref()
                .map(|path| validate_output_path(path, "benchmark corpus output"))
                .transpose()?;
            commands::bench::run_kernel_benchmark_corpus(quick, validated_output.as_deref(), json)?;
        }

        #[cfg(feature = "trainer")]
        Commands::BenchWorkload {
            preset,
            model,
            dataset,
            experts_dir,
            prompt_samples,
            max_prompt_tokens,
            inference_context,
            decode_steps,
            inference_warmup_passes,
            inference_session_repeats,
            inference_repeats,
            train_samples,
            train_steps,
            batch_size,
            max_seq_len,
            json,
            output,
        } => {
            let validated_output = output
                .as_deref()
                .map(|path| validate_output_path(path, "workload benchmark output"))
                .transpose()?;
            if let Some(preset) = preset {
                commands::bench::run_workload_benchmark_preset(
                    preset,
                    inference_warmup_passes,
                    inference_session_repeats,
                    experts_dir.as_deref(),
                    validated_output.as_deref(),
                    json,
                )
                .await?;
            } else {
                commands::bench::run_workload_benchmark(
                    &model,
                    &dataset,
                    experts_dir.as_deref(),
                    prompt_samples,
                    max_prompt_tokens,
                    inference_context,
                    decode_steps,
                    inference_warmup_passes,
                    inference_session_repeats,
                    inference_repeats,
                    train_samples,
                    train_steps,
                    batch_size,
                    max_seq_len,
                    validated_output.as_deref(),
                    json,
                )
                .await?;
            }
        }

        #[cfg(feature = "trainer")]
        Commands::BenchGdn {
            model,
            stage,
            layer,
            batch_size,
            seq_len,
            warmup_iterations,
            benchmark_iterations,
            json,
            output,
        } => {
            let validated_output = output
                .as_deref()
                .map(|path| validate_output_path(path, "GDN benchmark output"))
                .transpose()?;
            commands::bench::run_gdn_decode_benchmark(
                &model,
                stage,
                layer,
                batch_size,
                seq_len,
                warmup_iterations,
                benchmark_iterations,
                validated_output.as_deref(),
                json,
            )
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::Init { output } => {
            // Validate output path to prevent path traversal attacks
            let validated_output = validate_output_path(&output, "config output")?;
            commands::bench::generate_sample_config(&validated_output.to_string_lossy())?;
        }

        #[cfg(feature = "trainer")]
        Commands::BenchFfi => {
            commands::bench::run_ffi_benchmark()?;
        }

        #[cfg(feature = "trainer")]
        Commands::BenchGen { model } => {
            commands::bench::run_gen_benchmark(&model).await?;
        }

        Commands::Ollama { action } => {
            commands::ollama::run_ollama_command(action).await?;
        }

        #[cfg(feature = "lora")]
        Commands::Fuse {
            model,
            lora,
            output,
            alpha,
            rank,
            accurate,
            low_memory,
        } => {
            if accurate {
                commands::fuse::run_fuse_accurate(&model, &lora, &output, low_memory).await?;
            } else {
                commands::fuse::run_fuse(&model, &lora, &output, alpha, rank).await?;
            }
        }

        Commands::Quantize {
            model,
            output,
            imatrix,
            method,
            lora,
            kl_calibrate,
            target_bpw,
            kl_threshold,
            format,
            bits,
            group_size,
        } => {
            if format == "mlx" {
                // MLX safetensors path — no LoRA fusion support in this path yet.
                if lora.is_some() {
                    anyhow::bail!(
                        "--lora is not yet supported with --format mlx; \
                         fuse the adapter first with `pmetal fuse` then quantize"
                    );
                }
                commands::quantize::run_quantization_mlx(
                    &model, &output, bits, group_size, target_bpw,
                )
                .await?;
            } else {
                #[cfg(feature = "lora")]
                if let Some(lora_path) = &lora {
                    // Fuse LoRA first, then quantize the fused model
                    let fused_dir = format!("{output}.fused_tmp");
                    commands::fuse::run_fuse(&model, lora_path, &fused_dir, None, None).await?;
                    commands::quantize::run_quantization(
                        &fused_dir,
                        &output,
                        imatrix.as_deref(),
                        method,
                        kl_calibrate,
                        target_bpw,
                        kl_threshold,
                    )
                    .await?;
                    // Clean up temp dir
                    let _ = std::fs::remove_dir_all(&fused_dir);
                } else {
                    commands::quantize::run_quantization(
                        &model,
                        &output,
                        imatrix.as_deref(),
                        method,
                        kl_calibrate,
                        target_bpw,
                        kl_threshold,
                    )
                    .await?;
                }

                #[cfg(not(feature = "lora"))]
                {
                    if lora.is_some() {
                        anyhow::bail!(
                            "LoRA fusion during quantization requires the `lora` feature"
                        );
                    }
                    commands::quantize::run_quantization(
                        &model,
                        &output,
                        imatrix.as_deref(),
                        method,
                        kl_calibrate,
                        target_bpw,
                        kl_threshold,
                    )
                    .await?;
                }
            }
        }

        #[cfg(feature = "trainer")]
        Commands::Distill {
            teacher,
            student,
            dataset,
            output,
            method,
            loss_type,
            temperature,
            alpha,
            rationale,
            rationale_weight,
            lora_r,
            lora_alpha,
            learning_rate,
            batch_size,
            epochs,
            max_seq_len,
            seed,
            text_column,
            text_columns,
            column_separator,
            prompt_column,
            response_column,
            log_metrics,
        } => {
            commands::distill::run_distillation_cli(
                &teacher,
                &student,
                &dataset,
                &output,
                &method,
                &loss_type,
                temperature,
                alpha,
                rationale,
                rationale_weight,
                lora_r,
                lora_alpha,
                learning_rate,
                batch_size,
                epochs,
                max_seq_len,
                seed,
                log_metrics,
                true,
                Vec::new(),
                text_column,
                text_columns,
                column_separator,
                prompt_column,
                response_column,
            )
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::Grpo {
            model,
            dataset,
            output,
            num_generations,
            beta,
            learning_rate,
            epochs,
            lora_r,
            lora_alpha,
            max_seq_len,
            max_completion_length,
            seed,
            dapo,
            reasoning_rewards,
            no_flash_attention,
            vlm,
            max_image_size,
            reward_model,
            reward_model_max_length,
            reward_model_weight,
            reward_model_template,
            async_rewards,
            speculative,
            speculative_draft_tokens,
            grpo_kv_bits,
            text_column,
            text_columns,
            column_separator,
            prompt_column,
            response_column,
            log_metrics,
        } => {
            commands::grpo::run_grpo_cli(
                &model,
                &dataset,
                &output,
                num_generations,
                beta,
                learning_rate,
                epochs,
                lora_r,
                lora_alpha,
                max_seq_len,
                max_completion_length,
                seed,
                dapo,
                reasoning_rewards,
                !no_flash_attention,
                vlm,
                max_image_size,
                reward_model,
                reward_model_max_length,
                reward_model_weight,
                reward_model_template,
                async_rewards,
                speculative,
                speculative_draft_tokens,
                grpo_kv_bits,
                log_metrics,
                true,
                Vec::new(),
                text_column,
                text_columns,
                column_separator,
                prompt_column,
                response_column,
            )
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::Rlkd {
            model,
            teacher_model,
            dataset,
            output,
            distill_alpha,
            final_alpha,
            anneal_alpha,
            distill_temperature,
            num_generations,
            beta,
            learning_rate,
            epochs,
            lora_r,
            lora_alpha,
            max_seq_len,
            max_completion_length,
            seed,
            reasoning_rewards,
            no_flash_attention,
            text_column,
            text_columns,
            column_separator,
            prompt_column,
            response_column,
            log_metrics,
        } => {
            commands::rlkd::run_rlkd_cli(
                &model,
                &teacher_model,
                &dataset,
                &output,
                distill_alpha,
                final_alpha,
                anneal_alpha,
                distill_temperature,
                num_generations,
                beta,
                learning_rate,
                epochs,
                lora_r,
                lora_alpha,
                max_seq_len,
                max_completion_length,
                seed,
                reasoning_rewards,
                !no_flash_attention,
                log_metrics,
                true,
                Vec::new(),
                text_column,
                text_columns,
                column_separator,
                prompt_column,
                response_column,
            )
            .await?;
        }

        Commands::Dataset { action } => {
            commands::dataset::run_dataset_command(action).await?;
        }

        Commands::Tokenize {
            input,
            output,
            tokenizer,
            text_column,
            docs_per_shard,
        } => {
            commands::tokenize::run_tokenize(
                &input,
                &output,
                &tokenizer,
                &text_column,
                docs_per_shard,
            )
            .await?;
        }

        Commands::Dashboard { metrics_file } => {
            let path = metrics_file.map(std::path::PathBuf::from);
            dashboard::run_dashboard(path)?;
        }

        #[cfg(feature = "dashboard")]
        Commands::Tui { metrics_file } => {
            let path = metrics_file.map(std::path::PathBuf::from);
            tui::run(path).await?;
        }

        Commands::Info { json } => {
            commands::info::run_info(json).await?;
        }

        #[cfg(feature = "merge")]
        Commands::Merge {
            model_a,
            model_b,
            output,
            method,
            base,
            t,
            weight_a,
            weight_b,
            density,
            dtype,
        } => {
            commands::merge::run_merge_command(
                &model_a,
                &model_b,
                &output,
                &method,
                base.as_deref(),
                t,
                weight_a,
                weight_b,
                density,
                &dtype,
            )
            .await?;
        }

        #[cfg(feature = "lora")]
        Commands::Eval {
            model,
            dataset,
            lora,
            max_seq_len,
            num_samples,
            json,
        } => {
            commands::eval::run_eval(
                &model,
                &dataset,
                lora.as_deref(),
                max_seq_len,
                num_samples,
                json,
            )
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::EmbedTrain {
            model,
            dataset,
            output,
            loss,
            pooling,
            temperature,
            margin,
            learning_rate,
            batch_size,
            epochs,
            max_seq_len,
            weight_decay,
            no_normalize,
            log_every,
            seed,
        } => {
            commands::embed_train::run_embed_train(
                &model,
                &dataset,
                &output,
                &loss,
                &pooling,
                temperature,
                margin,
                learning_rate,
                batch_size,
                epochs,
                max_seq_len,
                weight_decay,
                !no_normalize,
                log_every,
                seed,
            )
            .await?;
        }

        #[cfg(feature = "trainer")]
        Commands::Pretrain {
            arch,
            shards,
            seq_len,
            batch_size,
            steps,
            learning_rate,
            min_lr,
            warmup_steps,
            lr_schedule,
            weight_decay,
            max_grad_norm,
            eos_token_id,
            output,
            checkpoint_every,
            resume,
            model_config: _model_config,
            z_loss,
            gradient_accumulation_steps,
            log_every,
            eval_every,
            eval_batches,
            seed,
        } => {
            use pmetal_bridge::compat::random;
            use pmetal_data::streaming::{StreamConfig, StreamingShardReader};
            use pmetal_trainer::pretrain::{self, PretrainConfig};

            random::seed(seed);
            let _output_dir = validate_output_path(&output, "pretrain output")?;

            // Collect shard paths (each CLI value is a literal path)
            let shard_paths: Vec<std::path::PathBuf> =
                shards.iter().map(std::path::PathBuf::from).collect();
            anyhow::ensure!(!shard_paths.is_empty(), "no shard files specified");
            println!(
                "Pretraining on {} shards, {} steps",
                shard_paths.len(),
                steps
            );

            // Parse LR schedule
            let lr_sched = match lr_schedule.to_lowercase().as_str() {
                "constant" => pmetal_core::LrSchedulerType::Constant,
                "linear" => pmetal_core::LrSchedulerType::Linear,
                "cosine" => pmetal_core::LrSchedulerType::Cosine,
                other => anyhow::bail!("unknown lr-schedule: {other}"),
            };

            // Build model via factory (supports llama, qwen2, qwen3, gemma, mistral, phi, gpt-oss)
            use pmetal_bridge::compat::module::ModuleParameters;
            let config_path = _model_config.as_deref().map(std::path::Path::new);
            let mut model = pretrain::create_model(&arch, config_path)?;
            let n_layers = pretrain::n_layers(&model);
            println!(
                "Architecture: {arch}, layers: {n_layers}, params: {}",
                model.num_parameters()
            );

            let config = PretrainConfig {
                num_steps: steps,
                learning_rate,
                min_lr,
                warmup_steps,
                lr_schedule: lr_sched,
                weight_decay,
                betas: (0.9, 0.95),
                eps: 1e-8,
                max_grad_norm: if max_grad_norm > 0.0 {
                    Some(max_grad_norm)
                } else {
                    None
                },
                ignore_index: None,
                z_loss_coef: if z_loss > 0.0 { Some(z_loss) } else { None },
                n_layers,
                apply_init: resume.is_none(),
                checkpoint_every: if checkpoint_every > 0 {
                    Some(checkpoint_every)
                } else {
                    None
                },
                checkpoint_dir: Some(std::path::PathBuf::from(&output)),
                gradient_accumulation_steps,
                log_every,
                eval_every,
                eval_batches,
            };

            // Resume from checkpoint if requested
            if let Some(ref ckpt_dir) = resume {
                let mut opt = pmetal_bridge::compat::optimizers::AdamWBuilder::new(learning_rate)
                    .weight_decay(weight_decay)
                    .build()?;
                let meta = pretrain::load_checkpoint(
                    std::path::Path::new(ckpt_dir),
                    &mut model,
                    &mut opt,
                )?;
                println!("Resumed from step {}, loss {:.4}", meta.step, meta.loss);
            }

            // Streaming data pipeline
            let stream_config = StreamConfig {
                shard_paths,
                seq_len,
                batch_size,
                eos_token_id,
                resume_from: None,
            };
            let reader = StreamingShardReader::new(stream_config)?;

            // Convert streaming batches to MLX arrays
            let batch_iter = reader.map(move |(batch, _pos)| {
                let flat: Vec<i32> = batch
                    .iter()
                    .flat_map(|seq| seq.iter().map(|&t| t as i32))
                    .collect();
                pmetal_bridge::InlineArray::from_i32_slice_shaped(
                    &flat,
                    &[batch.len() as i32, seq_len as i32],
                )
            });

            // Run
            let losses = pretrain::run_pretrain(&mut model, &config, batch_iter)?;

            // Print summary
            if !losses.is_empty() {
                let last = losses.last().unwrap();
                println!(
                    "Pretraining complete: {} steps, final loss {:.4}",
                    losses.len(),
                    last,
                );
            }
        }
    }

    Ok(())
}

/// Save adapter config alongside LoRA weights.
fn save_adapter_config(
    lora_weights_path: &std::path::Path,
    r: usize,
    alpha: f32,
    target_modules: &[String],
    use_rslora: bool,
    base_model: Option<&str>,
) -> anyhow::Result<()> {
    let mut adapter_config = serde_json::json!({
        "r": r,
        "alpha": alpha,
        "target_modules": target_modules,
        "use_rslora": use_rslora,
    });
    if let Some(bm) = base_model {
        adapter_config["base_model"] = serde_json::Value::String(bm.to_string());
    }
    let config_path = lora_weights_path
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("adapter_config.json");
    std::fs::write(&config_path, serde_json::to_string_pretty(&adapter_config)?)?;
    tracing::info!("Saved adapter config to {:?}", config_path);
    Ok(())
}

fn validate_output_path(path: &str, context: &str) -> anyhow::Result<PathBuf> {
    use std::path::{Component, PathBuf};

    let path = PathBuf::from(path);

    for component in path.components() {
        if matches!(component, Component::ParentDir) {
            anyhow::bail!(
                "Path traversal detected in {}: '{}' contains '..' component. Please use a safe path.",
                context,
                path.display()
            );
        }
    }

    let cwd = std::env::current_dir()?;
    let resolved = if path.is_absolute() {
        path.clone()
    } else {
        cwd.join(&path)
    };

    if let Some(parent) = resolved.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let canonical = if resolved.exists() {
        resolved.canonicalize()?
    } else if let Some(parent) = resolved.parent() {
        let canonical_parent = parent.canonicalize()?;
        if let Some(filename) = resolved.file_name() {
            canonical_parent.join(filename)
        } else {
            canonical_parent
        }
    } else {
        resolved
    };

    let cwd_canonical = cwd.canonicalize()?;
    let home_dir = dirs::home_dir();
    let temp_dir = std::env::temp_dir().canonicalize().ok();

    let is_safe = canonical.starts_with(&cwd_canonical)
        || home_dir
            .as_ref()
            .map(|h| canonical.starts_with(h))
            .unwrap_or(false)
        || temp_dir
            .as_ref()
            .map(|t| canonical.starts_with(t))
            .unwrap_or(false)
        || canonical.starts_with("/tmp")
        || canonical.starts_with("/private/tmp");

    if !is_safe {
        anyhow::bail!(
            "Unsafe output path for {}: '{}' resolves to '{}'",
            context,
            path.display(),
            canonical.display()
        );
    }

    Ok(canonical)
}
