//! Clap argument struct for `pmetal infer`.

use clap::{Args, ValueEnum};
use pmetal_core::jobs::InferSpec;

/// A frame rate: a finite number above zero.
fn parse_fps(value: &str) -> Result<f64, String> {
    match value.parse::<f64>() {
        Ok(fps) if fps.is_finite() && fps > 0.0 => Ok(fps),
        _ => Err(format!("{value} is not a frame rate above 0")),
    }
}

/// Run inference with a model.
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

    /// Frame rate the --video frames were extracted at, used to sample them
    /// to the model's rate (2 per second) and to time-stamp them in the
    /// prompt. Give it: without it 24 is assumed, as the reference does
    /// [default: 24]
    #[arg(long = "video-fps", value_name = "FPS", requires = "video", value_parser = parse_fps)]
    pub video_fps: Option<f64>,

    /// Maximum tokens to generate. Default: the model's generation_config.json
    /// max_new_tokens (or max_length), else 32768 for a thinking model, else
    /// 256; never past the context window.
    #[arg(long = "max-tokens")]
    pub max_tokens: Option<usize>,

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

    /// How long the model thinks, for chat templates with a
    /// `reasoning_effort` control: xhigh | medium | low on Qwen3.8 (default
    /// xhigh), low | medium | high on gpt-oss (default medium).
    #[arg(long = "reasoning-effort")]
    pub reasoning_effort: Option<String>,

    /// Drop the thinking of earlier turns from the prompt, keeping only the
    /// current turn's, for chat templates with a `preserve_thinking` control
    /// (Qwen3.8 keeps it by default).
    #[arg(long = "no-preserve-thinking")]
    pub no_preserve_thinking: bool,

    /// Extra chat template keyword arguments as a JSON object, e.g.
    /// '{"enable_thinking": false}'. --no-thinking, --reasoning-effort and
    /// --no-preserve-thinking win over the same keys here.
    #[arg(long = "chat-template-kwargs")]
    pub chat_template_kwargs: Option<String>,

    /// Sampling mode preset with model-card recommended parameters.
    #[arg(long = "mode", default_value_t = pmetal_data::inference_config::SamplingMode::Auto)]
    pub mode: pmetal_data::inference_config::SamplingMode,

    /// Execution backend: auto | standard | ane.
    ///
    /// `auto` runs on the GPU, or on the ANE with --ane; `standard` pins the
    /// GPU and `ane` the ANE.
    #[arg(long = "backend", default_value_t = pmetal_data::inference_config::InferenceBackend::Auto)]
    pub backend: pmetal_data::inference_config::InferenceBackend,

    /// Draft model for exact speculative decoding (HF id or local path): a
    /// Gemma 4 MTP assistant, or with --ane a DFlash draft model (run on the
    /// GPU, for the model on the ANE).
    #[arg(long = "draft-model")]
    pub draft_model: Option<String>,

    /// Print one JSON object on stdout (the text, token counts, timing, why
    /// generation stopped) instead of streaming the text; the banner and the
    /// statistics go to stderr.
    #[arg(long = "json")]
    pub json: bool,

    /// Enable bundled Qwen3Next/Qwen3.6 MTP speculative decoding.
    #[arg(long = "mtp")]
    pub mtp: bool,

    /// Optional Qwen MTP checkpoint directory; defaults to bundled mtp.* weights in --model.
    #[arg(long = "mtp-model")]
    pub mtp_model: Option<String>,

    /// Number of Qwen MTP draft tokens to verify per speculative step.
    #[arg(long = "mtp-draft-tokens", default_value_t = 3)]
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
    #[arg(long = "ane")]
    pub ane: bool,

    /// Largest context (prompt plus output, in tokens) the ANE compiles a
    /// model for. Each request gets the smallest power of two from 512 that
    /// fits it, up to this; a smaller context runs faster.
    #[arg(long = "ane-max-seq-len", default_value_t = 4096)]
    pub ane_max_seq_len: usize,

    /// Run a prompt and generation throughput benchmark.
    #[arg(long = "benchmark")]
    pub benchmark: bool,

    /// Number of measured trials for benchmarking (default: 5)
    #[arg(long = "benchmark-iters", default_value_t = 5)]
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
    #[arg(long = "kv-group-size", default_value_t = 64)]
    pub kv_group_size: usize,

    /// Use TurboQuant for KV cache compression.
    #[arg(long = "kv-turboquant")]
    pub kv_turboquant: bool,

    /// Mixed-bit TurboQuant preset (`q2_5` or `q3_5`).
    #[arg(long = "kv-turboquant-preset", value_enum)]
    pub kv_turboquant_preset: Option<crate::cli::infer::TurboQuantPresetArg>,

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

impl From<InferArgs> for InferSpec {
    fn from(a: InferArgs) -> Self {
        InferSpec {
            model: a.model,
            lora: a.lora,
            prompt: a.prompt,
            image: a.image,
            video: a.video,
            video_fps: a.video_fps,
            json: a.json,
            max_tokens: a.max_tokens,
            temperature: a.temperature,
            top_k: a.top_k,
            top_p: a.top_p,
            min_p: a.min_p,
            repetition_penalty: a.repetition_penalty,
            frequency_penalty: a.frequency_penalty,
            presence_penalty: a.presence_penalty,
            seed: a.seed,
            chat: a.chat,
            system: a.system,
            no_thinking: a.no_thinking,
            reasoning_effort: a.reasoning_effort,
            no_preserve_thinking: a.no_preserve_thinking,
            chat_template_kwargs: a.chat_template_kwargs,
            hide_thinking: a.hide_thinking,
            mode: a.mode.to_string(),
            backend: a.backend.to_string(),
            draft_model: a.draft_model,
            mtp: a.mtp,
            mtp_model: a.mtp_model,
            mtp_draft_tokens: a.mtp_draft_tokens,
            tools: a.tools,
            fp8: a.fp8,
            experts_dir: a.experts_dir,
            ane: a.ane,
            ane_max_seq_len: a.ane_max_seq_len,
            benchmark: a.benchmark,
            benchmark_iters: a.benchmark_iters,
            benchmark_prompt_tokens: a.benchmark_prompt_tokens,
            profile_layers: a.profile_layers,
            profile_output: a.profile_output,
            kv_quant: a.kv_quant,
            kv_k_bits: a.kv_k_bits,
            kv_v_bits: a.kv_v_bits,
            kv_group_size: a.kv_group_size,
            kv_turboquant: a.kv_turboquant,
            kv_turboquant_preset: a
                .kv_turboquant_preset
                .and_then(|preset| preset.to_possible_value())
                .map(|value| value.get_name().to_string()),
            kv_quant_preset: a.kv_quant_preset,
            no_kv_quant: a.no_kv_quant,
            kv_qjl: a.kv_qjl,
            detect_repetition: a.detect_repetition,
        }
    }
}
