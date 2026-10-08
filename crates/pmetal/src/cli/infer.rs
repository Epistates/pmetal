//! Clap argument struct for `pmetal infer`.

use clap::Args;

/// A frame rate: a finite number above zero.
fn parse_fps(value: &str) -> Result<f64, String> {
    match value.parse::<f64>() {
        Ok(fps) if fps.is_finite() && fps > 0.0 => Ok(fps),
        _ => Err(format!("{value} is not a frame rate above 0")),
    }
}

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

    /// Image file to show the model, before the prompt text (repeatable, in
    /// order). Qwen3.5-family vision models (Qwen3.5 / 3.6 / 3.8).
    #[arg(long = "image")]
    pub image: Vec<String>,

    /// Directory of a video's frames to show the model, after the images
    /// (repeatable, in order). Its image files are the frames, in natural
    /// file-name order; video files are not decoded, so extract the frames
    /// first. Qwen3.5-family vision models.
    #[arg(long = "video", value_name = "FRAMES_DIR")]
    pub video: Vec<String>,

    /// Frame rate of the --video frames, used to sample them and to
    /// time-stamp them in the prompt [default: 24]
    #[arg(long = "video-fps", value_name = "FPS", requires = "video", value_parser = parse_fps)]
    pub video_fps: Option<f64>,

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

    /// Execution backend: auto | standard | compiled | metal-sampler | ane | minimal.
    #[arg(long = "backend", default_value = "auto")]
    pub backend: pmetal_data::inference_config::InferenceBackend,

    /// Draft model for exact speculative decoding (HF id or local path): a
    /// Gemma 4 MTP assistant, or with --ane a DFlash draft model (run on the
    /// GPU, for the model on the ANE).
    #[arg(long = "draft-model")]
    pub draft_model: Option<String>,

    /// Enable bundled Qwen3Next/Qwen3.6 MTP speculative decoding.
    #[arg(long = "mtp")]
    pub mtp: bool,

    /// Optional Qwen MTP checkpoint directory; defaults to bundled mtp.* weights in --model.
    #[arg(long = "mtp-model")]
    pub mtp_model: Option<String>,

    /// Number of Qwen MTP draft tokens to verify per speculative step.
    #[arg(long = "mtp-draft-tokens", default_value = "3")]
    pub mtp_draft_tokens: usize,

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

    /// Largest context (prompt plus output, in tokens) the ANE compiles a
    /// model for. Each request gets the smallest power of two from 512 that
    /// fits it, up to this; a smaller context runs faster.
    #[cfg(feature = "ane")]
    #[arg(long = "ane-max-seq-len", default_value = "4096")]
    pub ane_max_seq_len: usize,

    /// Run a prompt and generation throughput benchmark.
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
