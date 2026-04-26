//! Clap argument struct for `pmetal infer`.

use clap::Args;

/// Thin clap argument struct for `pmetal infer`.
#[derive(Args, Debug)]
pub struct InferArgs {
    /// Model ID or path
    #[arg(short, long = "model")]
    pub model: String,

    /// LoRA adapter path (optional)
    #[arg(long = "lora")]
    pub lora: Option<String>,

    /// Input prompt
    #[arg(short, long = "prompt")]
    pub prompt: String,

    /// Maximum tokens to generate
    #[arg(long = "max-tokens", default_value = "256")]
    pub max_tokens: usize,

    /// Temperature for sampling (0 = greedy). Defaults to model's generation_config.json
    #[arg(long = "temperature")]
    pub temperature: Option<f32>,

    /// Top-k sampling (0 = disabled). Defaults to model's generation_config.json
    #[arg(long = "top-k")]
    pub top_k: Option<usize>,

    /// Top-p nucleus sampling (0.0-1.0). Defaults to model's generation_config.json
    #[arg(long = "top-p")]
    pub top_p: Option<f32>,

    /// Min-p dynamic sampling (0.0 = disabled). Defaults to model's generation_config.json
    #[arg(long = "min-p")]
    pub min_p: Option<f32>,

    /// Repetition penalty applied to prompt + output (1.0 = disabled, 1.0-1.2 typical)
    #[arg(long = "repetition-penalty")]
    pub repetition_penalty: Option<f32>,

    /// Frequency penalty proportional to token count (0.0 = disabled, 0.0-2.0 typical)
    #[arg(long = "frequency-penalty")]
    pub frequency_penalty: Option<f32>,

    /// Presence penalty for any appeared token (0.0 = disabled, Qwen3 recommends 0-2)
    #[arg(long = "presence-penalty")]
    pub presence_penalty: Option<f32>,

    /// Random seed for reproducible generation
    #[arg(long = "seed")]
    pub seed: Option<u64>,

    /// Apply chat template (auto-detected from tokenizer)
    #[arg(long = "chat")]
    pub chat: bool,

    /// System message for chat mode
    #[arg(long = "system")]
    pub system: Option<String>,

    /// Disable thinking mode for models that support it (e.g., Qwen3)
    #[arg(long = "no-thinking")]
    pub no_thinking: bool,

    /// Sampling mode preset with model-card recommended parameters.
    #[arg(long = "mode", default_value = "auto")]
    pub mode: pmetal_data::inference_config::SamplingMode,

    /// Execution backend: auto | standard | compiled | metal-sampler | ane | minimal | dflash.
    #[arg(long = "backend", default_value = "auto")]
    pub backend: pmetal_data::inference_config::InferenceBackend,

    /// Draft model for speculative decoding (HF id or local path).
    #[arg(long = "draft-model")]
    pub draft_model: Option<String>,

    /// Use fused Metal sampling kernel for better battery performance.
    /// Legacy alias — prefer `--backend metal-sampler`.
    #[arg(long = "metal-sampler")]
    pub metal_sampler: bool,

    /// Use JIT-compiled sampling for better performance.
    /// Legacy alias — prefer `--backend compiled`.
    #[arg(long = "compiled")]
    pub compiled: bool,

    /// Use dedicated GPU stream for generation.
    /// NOTE: Currently a no-op placeholder.
    #[arg(long = "stream", hide = true)]
    pub stream: bool,

    /// Use minimal async generation (for performance debugging)
    #[arg(long = "minimal")]
    pub minimal: bool,

    /// Hide thinking trace from output
    #[arg(long = "hide-thinking")]
    pub hide_thinking: bool,

    /// Path to a JSON file containing tool/function definitions (OpenAI format).
    #[arg(long = "tools")]
    pub tools: Option<String>,

    /// Use FP8 quantization for weights (~2x memory reduction).
    #[arg(long = "fp8")]
    pub fp8: bool,

    /// Path to packed expert weights directory for SSD-offloaded MoE inference.
    #[arg(long = "experts-dir")]
    pub experts_dir: Option<String>,

    /// Enable ANE (Apple Neural Engine) for inference (experimental).
    #[cfg(feature = "ane")]
    #[arg(long = "ane")]
    pub ane: bool,

    /// Maximum ANE kernel sequence length (power-of-2 bucket cap).
    #[cfg(feature = "ane")]
    #[arg(long = "ane-max-seq-len", default_value = "1024")]
    pub ane_max_seq_len: usize,

    /// Use the experimental ANE real-time evaluation path when ANE inference is selected.
    #[cfg(feature = "ane")]
    #[arg(long = "ane-real-time")]
    pub ane_real_time: bool,

    /// Run an MLX-LM-compatible benchmark.
    #[arg(long = "benchmark")]
    pub benchmark: bool,

    /// Number of measured trials for benchmarking (default: 5)
    #[arg(long = "benchmark-iters", default_value = "5")]
    pub benchmark_iters: usize,

    /// Synthetic prompt length for --benchmark.
    #[arg(long = "benchmark-prompt-tokens")]
    pub benchmark_prompt_tokens: Option<usize>,

    /// Run an opt-in per-layer forward profile for supported hybrid models.
    #[arg(long = "profile-layers")]
    pub profile_layers: bool,

    /// Write the layer profile report as pretty JSON.
    #[arg(long = "profile-output")]
    pub profile_output: Option<String>,

    /// KV cache quantization bits (8=q8_0, 4=q4_0, 0=fp16).
    #[arg(long = "kv-quant")]
    pub kv_quant: Option<u8>,

    /// KV cache key bits (overrides --kv-quant for keys only, for asymmetric K/V).
    #[arg(long = "kv-k-bits")]
    pub kv_k_bits: Option<u8>,

    /// KV cache value bits (overrides --kv-quant for values only, for asymmetric K/V).
    #[arg(long = "kv-v-bits")]
    pub kv_v_bits: Option<u8>,

    /// KV cache quantization group size.
    #[arg(long = "kv-group-size", default_value = "64")]
    pub kv_group_size: usize,

    /// Use TurboQuant for KV cache compression.
    #[arg(long = "kv-turboquant")]
    pub kv_turboquant: bool,

    /// Mixed-bit TurboQuant preset (`q2_5` or `q3_5`).
    #[arg(long = "kv-turboquant-preset", value_enum)]
    pub kv_turboquant_preset: Option<crate::TurboQuantPresetArg>,

    /// TurboQuant v2 mixed-bit affine preset.
    #[arg(long = "kv-quant-preset", value_parser = ["q2_5", "q3_5"])]
    pub kv_quant_preset: Option<String>,

    /// Disable KV cache quantization (use fp16 KV cache).
    #[arg(long = "no-kv-quant")]
    pub no_kv_quant: bool,

    /// Enable QJL residual correction for Q2-Q3 key quantization.
    #[arg(long = "kv-qjl")]
    pub kv_qjl: bool,

    /// Enable n-gram repetition loop detection.
    #[arg(long = "detect-repetition")]
    pub detect_repetition: bool,
}
