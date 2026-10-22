//! Unified inference pipeline shared by CLI, GUI, and serve.
//!
//! `InferenceRunner` encapsulates the full pre-generation setup (model loading,
//! tokenization, chat template, sampling config, cache creation) so that all
//! consumers get identical behavior from a single code path.

use std::path::{Path, PathBuf};

use pmetal_cache::kv_cache::{
    CacheMode, KVCache, KVCacheConfig, MambaCache, TurboQuantConfig, sanitize_cache_mode_for_config,
};
use pmetal_core::jobs::InferSpec;
use pmetal_data::Tokenizer;
use pmetal_data::chat_templates::{ChatTemplateKwargs, ChatTemplateType, Message, ToolDefinition};
use pmetal_data::qwen_vl_processing::{ProcessedMedia, QwenVlProcessor};
use pmetal_models::architectures::qwen3_5_vision::Qwen3_5MultimodalConfig;
use pmetal_models::architectures::{
    Gemma4AssistantForCausalLM, Qwen3NextConfig, Qwen3NextMtpForCausalLM,
    load_gemma4_assistant_from_dir, load_qwen3_next_mtp_from_dir,
};
pub use pmetal_models::cache_policy::{
    CacheModeRequest, CacheModeSelection, CacheModeSource, DefaultContextLength,
    MIN_DEFAULT_CONTEXT, TurboQuantPreset, default_serve_context_len, estimate_weight_bytes,
    explicit_cache_mode_override, log_cache_selection, select_cache_mode_for_model,
    select_cache_mode_with_working_set,
};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_models::generation::GenerationConfig;
use pmetal_models::{
    Gemma4MtpConfig, GenerationOutput, Qwen3NextMtpConfig, generate_cached_async_streaming,
    generate_gemma4_mtp_streaming, generate_qwen3_native_mtp_streaming,
    generate_qwen3_next_mtp_streaming, validate_gemma4_mtp_pair, validate_qwen3_next_mtp_pair,
};
use pmetal_runtime::compat::{Array, Dtype, Exception, ModuleParameters as _};

#[cfg(feature = "lora")]
use pmetal_lora::{DynamicLoraModel, TrainableModel as _};

/// A video given as a directory of its frames.
#[derive(Clone, Debug)]
pub struct VideoInput {
    /// The directory's image files are the frames, in natural file-name
    /// order ([`pmetal_data::qwen_vl_processing::video_frame_paths`]).
    pub frames_dir: PathBuf,
    /// The frames' rate, used to sample them and time-stamp them in the
    /// prompt; `None` assumes 24 as the reference does.
    pub fps: Option<f64>,
}

/// Configuration for preparing an inference run.
///
/// All sampling fields are `Option` — `None` means "use model's
/// `generation_config.json` default".
#[derive(Clone, Debug)]
pub struct InferenceRunnerConfig {
    // ── Model ────────────────────────────────────────────────────────────
    /// Local path to the model directory (already resolved / downloaded).
    pub model_path: PathBuf,
    /// Optional LoRA adapter path (file or directory).
    pub lora_path: Option<String>,
    /// Optional Gemma 4 MTP assistant checkpoint path.
    pub mtp_assistant_path: Option<PathBuf>,
    /// Optional Qwen MTP checkpoint path; defaults to bundled mtp.* weights in `model_path`.
    pub qwen_mtp_path: Option<PathBuf>,
    /// Enable bundled Qwen3Next/Qwen3.6 MTP weights from the target checkpoint.
    pub qwen_mtp: bool,
    /// Number of Qwen MTP draft tokens to verify per speculative step.
    pub qwen_mtp_draft_tokens: usize,
    /// Optional packed expert weights directory for SSD-offloaded MoE.
    pub experts_dir: Option<String>,
    /// Quantize weights to FP8 E4M3 (~2x memory savings).
    pub fp8: bool,

    // ── Prompt ───────────────────────────────────────────────────────────
    /// User prompt text.
    pub prompt: String,
    /// Optional structured chat history for callers like the GUI.
    ///
    /// When present, the shared runner applies the detected chat template to
    /// these messages directly instead of flattening them into ad-hoc text.
    pub chat_messages: Option<Vec<Message>>,
    /// Optional system message.
    pub system_message: Option<String>,
    /// Apply chat template (auto-detected from model).
    pub chat: bool,
    /// Disable thinking mode for models that support it.
    pub no_thinking: bool,
    /// Keyword arguments for the model's chat template, the `**kwargs` of
    /// `apply_chat_template`: `reasoning_effort`, `preserve_thinking`, or
    /// anything else the template reads. `no_thinking` wins over an
    /// `enable_thinking` here, and `reasoning_effort` / `preserve_thinking`
    /// must be controls the template has (see
    /// [`ChatTemplate::check_thinking_controls`](pmetal_data::chat_templates::ChatTemplate::check_thinking_controls)).
    pub chat_template_kwargs: ChatTemplateKwargs,
    /// Optional tool/function definitions for tool-calling models.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Images to show a Qwen3.5-family vision model, in order. In chat mode
    /// each is an image item of the prompt's message, ahead of its text, for
    /// the chat template to place (as `pmetal serve` does with a request's
    /// image parts); without a template its placeholder precedes the text.
    pub images: Vec<PathBuf>,
    /// Videos to show a Qwen3.5-family vision model, in order, after the
    /// images and before the prompt text.
    pub videos: Vec<VideoInput>,

    // ── Sampling ─────────────────────────────────────────────────────────
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    /// Output budget. `None`: the model's recommendation, see
    /// [`default_max_tokens`](pmetal_data::inference_config::default_max_tokens),
    /// kept inside the context window.
    pub max_tokens: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,

    // ── KV Cache ─────────────────────────────────────────────────────────
    /// Quantization bits for KV cache (8=q8_0, 4=q4_0, 0=fp16).
    ///
    /// `None` means auto-select based on the model size, context window, and
    /// the device working-set budget.
    pub kv_quant: Option<u8>,
    /// Override key bits (for asymmetric K/V quantization).
    pub kv_k_bits: Option<u8>,
    /// Override value bits (for asymmetric K/V quantization).
    pub kv_v_bits: Option<u8>,
    /// Quantization group size.
    pub kv_group_size: usize,
    /// Use TurboQuant instead of MLX affine KV quantization.
    pub kv_turboquant: bool,
    /// Mixed-bit TurboQuant preset.
    pub kv_turboquant_preset: Option<TurboQuantPreset>,
    /// TurboQuant v2 affine mixed-bit preset: "q2_5" or "q3_5".
    /// Enables outlier channel permutation + split-bit KV cache (native path only).
    pub kv_quant_preset: Option<String>,
    /// Disable KV cache quantization entirely.
    pub no_kv_quant: bool,
    /// Sampling mode preset (model-family-specific recommended parameters).
    /// `Auto` selects based on thinking flag. Explicit modes override
    /// generation_config.json defaults. CLI param overrides beat mode presets.
    pub mode: pmetal_data::inference_config::SamplingMode,
    /// Enable n-gram repetition loop detection.
    /// When enabled, force-stops generation when the same 8-token pattern
    /// repeats 4 times (32-token window). Useful for small models prone to
    /// infinite loops in thinking mode.
    pub detect_repetition: bool,
    /// Enable QJL residual correction for Q2-Q3 keys (native path only).
    ///
    /// When true and the cache is configured for uniform Q2 or Q3 quantization,
    /// stores 1-bit sign vectors on key quantization residuals and applies an
    /// additive correction to attention scores. Only active for bits <= 3,
    /// uniform (non-mixed-bit) KV cache.
    pub kv_qjl: bool,
}

impl Default for InferenceRunnerConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            lora_path: None,
            mtp_assistant_path: None,
            qwen_mtp_path: None,
            qwen_mtp: false,
            qwen_mtp_draft_tokens: Qwen3NextMtpConfig::default().num_draft_tokens,
            experts_dir: None,
            fp8: false,
            prompt: String::new(),
            chat_messages: None,
            system_message: None,
            chat: false,
            no_thinking: false,
            chat_template_kwargs: ChatTemplateKwargs::new(),
            tools: None,
            images: Vec::new(),
            videos: Vec::new(),
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            max_tokens: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            kv_quant: None,
            kv_k_bits: None,
            kv_v_bits: None,
            kv_group_size: 64,
            kv_turboquant: false,
            kv_turboquant_preset: None,
            kv_quant_preset: None,
            no_kv_quant: false,
            kv_qjl: false,
            mode: pmetal_data::inference_config::SamplingMode::Auto,
            detect_repetition: false,
        }
    }
}

impl InferenceRunnerConfig {
    /// The runner's configuration for `spec`.
    ///
    /// The one place an `InferSpec` becomes a runner configuration, for the
    /// command line and the Python bindings alike. The spec's own checks
    /// ([`InferSpec::normalize`]) come first; this fails only on what they
    /// cannot see (a malformed chat-template override), so a caller can run
    /// it before fetching anything.
    ///
    /// What is left for the caller: the model paths (`model_path`,
    /// `mtp_assistant_path`, `qwen_mtp_path`), which are resolved on the
    /// caller's own terms (the command line on its async runtime, Python with
    /// the interpreter lock released), and the tool list, a file only the
    /// command line reads.
    pub fn from_spec(spec: &InferSpec) -> Result<Self, String> {
        let chat_template_kwargs = ChatTemplateKwargs::from_controls(
            spec.chat_template_kwargs.as_deref(),
            spec.no_thinking,
            spec.reasoning_effort.as_deref(),
            spec.no_preserve_thinking,
        )?;
        // `chat_template_kwargs='{"enable_thinking": false}'` is `no_thinking`.
        let no_thinking = spec.no_thinking || chat_template_kwargs.enable_thinking() == Some(false);
        let kv_turboquant_preset = spec
            .kv_turboquant_preset
            .as_deref()
            .map(|name| {
                name.parse::<TurboQuantPreset>()
                    .map_err(|e| format!("kv_turboquant_preset: {e}"))
            })
            .transpose()?;
        Ok(Self {
            model_path: PathBuf::new(),
            lora_path: spec.lora.clone(),
            mtp_assistant_path: None,
            qwen_mtp_path: None,
            qwen_mtp: spec.mtp,
            qwen_mtp_draft_tokens: spec.mtp_draft_tokens.max(1),
            experts_dir: spec.experts_dir.clone(),
            fp8: spec.fp8,
            prompt: spec.prompt.clone(),
            chat_messages: None,
            system_message: spec.system.clone(),
            chat: spec.chat,
            no_thinking,
            chat_template_kwargs,
            tools: None,
            images: spec.image.iter().map(PathBuf::from).collect(),
            videos: spec
                .video
                .iter()
                .map(|dir| VideoInput {
                    frames_dir: dir.into(),
                    fps: spec.video_fps,
                })
                .collect(),
            temperature: spec.temperature,
            top_k: spec.top_k,
            top_p: spec.top_p,
            min_p: spec.min_p,
            max_tokens: spec.max_tokens,
            repetition_penalty: spec.repetition_penalty,
            frequency_penalty: spec.frequency_penalty,
            presence_penalty: spec.presence_penalty,
            seed: spec.seed,
            kv_quant: spec.kv_quant,
            kv_k_bits: spec.kv_k_bits,
            kv_v_bits: spec.kv_v_bits,
            kv_group_size: spec.kv_group_size,
            kv_turboquant: spec.kv_turboquant,
            kv_turboquant_preset,
            kv_quant_preset: spec.kv_quant_preset.clone(),
            no_kv_quant: spec.no_kv_quant,
            kv_qjl: spec.kv_qjl,
            mode: spec.mode.parse()?,
            detect_repetition: spec.detect_repetition,
        })
    }
}

/// Loaded model — either a standard model or a LoRA-merged model.
#[allow(clippy::large_enum_variant)]
enum LoadedModel {
    Standard(DynamicModel),
    #[cfg(feature = "lora")]
    Lora(DynamicLoraModel),
    /// Native InlineArray path — no `DynamicModel` needed.
    /// All weights loaded by the pmetal-runtime native engine.
    NativeOnly,
}

/// Prepared inference state, ready for token generation.
///
/// Created by [`InferenceRunner::prepare`]. Owns the model, tokenizer,
/// caches, and generation config so callers only need to provide a
/// token callback.
///
/// `tokenizer` is a separate public field to enable split-borrowing:
/// callers can hold `&runner.tokenizer` while calling
/// `runner.gen.generate_streaming()`.
pub struct InferenceRunner {
    /// Tokenizer for encoding/decoding. Public for split-borrow access
    /// during streaming (borrow this before calling `generate_streaming`).
    pub tokenizer: Tokenizer,
    /// Generation state. Public for split-borrow access.
    pub state: InferenceGenState,
    chat_template_type: Option<ChatTemplateType>,
    is_chat: bool,
    thinks: bool,
}

/// Mutable generation state (model, caches, config).
///
/// Separated from [`InferenceRunner`] to enable split-borrowing the tokenizer
/// during streaming generation.
pub struct InferenceGenState {
    model: LoadedModel,
    gen_config: GenerationConfig,
    input_ids: Vec<u32>,
    cache: KVCache,
    mamba_cache: Option<MambaCache>,
    mtp_assistant: Option<Gemma4AssistantForCausalLM>,
    mtp_config: Option<Gemma4MtpConfig>,
    qwen_mtp: Option<Qwen3NextMtpForCausalLM>,
    qwen_mtp_cache: Option<KVCache>,
    qwen_mtp_config: Option<Qwen3NextMtpConfig>,
    native_turboquant: Option<TurboQuantConfig>,
    /// Zero-overhead affine KV cache quantization config for native path
    native_quant_config: Option<pmetal_runtime::qwen3_native::QuantCacheConfig>,
    /// Model directory path — used for native InlineArray weight loading.
    model_path: PathBuf,
    /// Decode throughput metrics from the last native generation run.
    /// `None` on non-native paths or when fewer than 20 steps were measured.
    pub last_decode_metrics: Option<pmetal_runtime::decode::DecodeMetrics>,
    /// Enable n-gram repetition loop detection (opt-in).
    detect_repetition: bool,
    /// Preprocessed images the prompt's image tokens stand for.
    images: Vec<ProcessedMedia>,
    /// Preprocessed videos the prompt's video tokens stand for.
    videos: Vec<ProcessedMedia>,
}

impl InferenceGenState {
    /// True when using the native bridge path (model loads inside generate).
    pub fn is_native_only(&self) -> bool {
        matches!(self.model, LoadedModel::NativeOnly)
    }
}

impl InferenceRunner {
    /// Prepare everything for generation.
    ///
    /// This consolidates the full pre-generation pipeline:
    /// 1. Load tokenizer
    /// 2. Detect + apply chat template
    /// 3. Tokenize prompt
    /// 4. Load sampling defaults + apply overrides
    /// 5. Collect stop tokens
    /// 6. Build GenerationConfig
    /// 7. Load model (DynamicModel or LoRA merge)
    /// 8. Apply FP8 quantization
    /// 9. Enable expert offloading
    /// 10. Create KV cache (with quantization mode)
    /// 11. Create Mamba cache (for hybrid models)
    pub fn prepare(mut config: InferenceRunnerConfig) -> Result<Self, Exception> {
        let images = std::mem::take(&mut config.images);
        let videos = std::mem::take(&mut config.videos);
        let has_media = !images.is_empty() || !videos.is_empty();
        let model_path = &config.model_path;
        if config.qwen_mtp && config.mtp_assistant_path.is_some() {
            return Err(Exception::custom(
                "Use either Gemma 4 --draft-model MTP or Qwen --mtp, not both",
            ));
        }
        if config.qwen_mtp_path.is_some() && !config.qwen_mtp {
            return Err(Exception::custom("--mtp-model requires Qwen --mtp"));
        }
        let mtp_requested = config.mtp_assistant_path.is_some() || config.qwen_mtp;
        if config.mtp_assistant_path.is_some() && config.lora_path.is_some() {
            return Err(Exception::custom(
                "Gemma 4 MTP is only supported for standard target-model inference, not LoRA-merged inference",
            ));
        }
        if config.lora_path.is_some() && config.experts_dir.is_some() {
            return Err(Exception::custom(
                "LoRA-merged inference does not support --experts-dir; fuse the adapter first or run without expert offload",
            ));
        }

        // 1. Load tokenizer (resolves GGUF tokenizer.ggml.* metadata when the
        //    model is a .gguf without a sibling tokenizer.json)
        let tokenizer = Tokenizer::from_model_path(model_path)
            .map_err(|e| Exception::custom(format!("tokenizer: {e}")))?;

        // 2. Determine chat mode (auto-detect instruction-tuned models)
        let is_instruct = model_looks_instruction_tuned(model_path);
        let use_chat = config.chat || is_instruct || config.tools.is_some();
        let mut template_kwargs = config.chat_template_kwargs.clone();
        template_kwargs
            .check_reserved()
            .map_err(Exception::custom)?;
        if config.no_thinking || (!is_instruct && use_chat) {
            // Base models don't understand <think> tags.
            template_kwargs.set(ChatTemplateKwargs::ENABLE_THINKING, false);
        }

        // 3. Sampling defaults are loaded after template detection (step 6b)
        //    because mode presets depend on the detected model family.
        let native_bridge_info = if config.lora_path.is_none()
            && !config.fp8
            && config.experts_dir.is_none()
            && (!mtp_requested || config.qwen_mtp)
        {
            crate::native_inference::load_native_runtime_info(model_path)
                .map_err(Exception::custom)?
        } else {
            None
        };
        let native_bridge_candidate = native_bridge_info.is_some();
        let multimodal = if !has_media {
            None
        } else {
            if mtp_requested
                || !native_bridge_info
                    .is_some_and(|info| info.arch == crate::native_inference::NativeArch::Qwen3_5)
            {
                return Err(Exception::custom(
                    "images and videos need a Qwen3.5-family vision model on the native engine \
                     (no LoRA, FP8, expert offload or MTP)",
                ));
            }
            Some(Qwen3_5MultimodalConfig::from_model_dir(model_path)?)
        };
        // The media are preprocessed before the prompt is tokenized: a
        // video's placeholder expands to text as well as tokens (each frame's
        // timestamp), so placeholders are expanded in the prompt text.
        let (image_media, video_media) = match &multimodal {
            None => (Vec::new(), Vec::new()),
            Some(multimodal) => preprocess_media(model_path, multimodal, &images, &videos)?,
        };
        let expand_media = |text: String| -> Result<String, Exception> {
            match &multimodal {
                None => Ok(text),
                Some(multimodal) => pmetal_data::qwen_vl_processing::expand_placeholders(
                    &text,
                    &image_media,
                    &video_media,
                    multimodal.vision.spatial_merge_size,
                )
                .map_err(|e| Exception::custom(e.to_string())),
            }
        };

        // 4. Prime the Metal runtime before MLX model construction. The stable
        // benchmark path always initializes Metal first, and doing the same here
        // Prewarm Metal context — but SKIP for native Qwen bridge paths to avoid
        // loading competing Metal pipeline state that degrades MLX performance.
        #[cfg(feature = "metal")]
        {
            if !native_bridge_candidate {
                if let Err(err) = pmetal_metal::context::MetalContext::global() {
                    tracing::warn!("Metal context prewarm failed: {err}");
                }
            }
        }

        // 5. For the standard path, load the model before prompt tokenization so
        // the interactive inference path follows the same load ordering as the
        // stable benchmark flow. LoRA still tokenizes first because its loader
        // needs the final max_seq_len up front.
        let mut preloaded_model = if native_bridge_candidate {
            tracing::info!("Bridge-native path detected — skipping shared model load");
            None
        } else if config.lora_path.is_none() {
            Some(load_standard_model_for_inference(model_path, &config)?)
        } else {
            None
        };

        // 6. Apply chat template + tokenize
        let (input_ids, template_type, thinks) = if use_chat {
            let detected = pmetal_data::chat_templates::detect_chat_template(
                model_path,
                &model_path.to_string_lossy(),
            );

            let mut messages = build_chat_messages(
                config.chat_messages.as_ref(),
                config.system_message.as_deref(),
                &config.prompt,
            );

            detected
                .check_thinking_controls(&template_kwargs)
                .map_err(Exception::custom)?;
            let rendered = if has_media {
                // The media are items of the prompt's message, rendered by
                // the model's own template, as the server renders a
                // request's media parts: the hardcoded fallback formatter
                // would drop them.
                attach_media(&mut messages, &config.prompt, images.len(), videos.len());
                detected.render_inference_jinja_with_kwargs(
                    &messages,
                    config.tools.as_deref(),
                    &template_kwargs,
                )
            } else {
                detected
                    .apply_inference_with_kwargs(
                        &messages,
                        config.tools.as_deref(),
                        &template_kwargs,
                    )
                    .map(|formatted| formatted.text)
            };
            let formatted = expand_media(rendered.map_err(Exception::custom)?)?;

            // The template wrote the model's BOS itself; the tokenizer's
            // post-processor would put a second one in front of it.
            let ids = tokenizer
                .encode(&formatted)
                .map_err(|e| Exception::custom(e.to_string()))?;
            (
                ids,
                Some(detected.template_type),
                detected.thinks_with(&template_kwargs),
            )
        } else {
            // Without a template, the images' and then the videos'
            // placeholders precede the text, as the processor takes them.
            let prompt = format!(
                "{}{}",
                pmetal_data::qwen_vl_processing::media_placeholders(images.len(), videos.len()),
                config.prompt
            );
            let prompt_text = if config.chat_messages.is_some()
                || config
                    .system_message
                    .as_deref()
                    .is_some_and(|system| !system.trim().is_empty())
            {
                build_plain_conversation_prompt(
                    config.chat_messages.as_ref(),
                    config.system_message.as_deref(),
                    &prompt,
                )
            } else {
                prompt
            };
            let ids = encode_plain_prompt(&tokenizer, &expand_media(prompt_text)?)
                .map_err(|e| Exception::custom(e.to_string()))?;
            (ids, None, false)
        };

        tracing::info!(tokens = input_ids.len(), "Prompt tokenized");

        // 6a. Output budget: the caller's, else the model's recommendation
        //     inside its context window.
        let max_tokens = match config.max_tokens {
            Some(max_tokens) => max_tokens,
            None => {
                let default = pmetal_data::inference_config::default_max_tokens(model_path, thinks);
                let max_tokens = default.for_prompt(
                    input_ids.len(),
                    pmetal_data::inference_config::context_window(model_path),
                );
                tracing::info!(max_tokens, source = %default.source, "Default output budget");
                max_tokens
            }
        };

        // 6b. Load sampling defaults: transformers' defaults → generation_config.json → mode preset
        let defaults =
            pmetal_data::inference_config::load_sampling_defaults(model_path, config.mode, thinks);
        let temperature = config.temperature.unwrap_or(defaults.temperature);
        let top_k = config.top_k.unwrap_or(defaults.top_k);
        let top_p = config.top_p.unwrap_or(defaults.top_p);
        let min_p = config.min_p.unwrap_or(defaults.min_p);
        let repetition_penalty = config
            .repetition_penalty
            .unwrap_or(defaults.repetition_penalty);
        let frequency_penalty = config
            .frequency_penalty
            .unwrap_or(defaults.frequency_penalty);
        let presence_penalty = config.presence_penalty.unwrap_or(defaults.presence_penalty);

        // 7. Collect stop tokens from all sources
        let stop_tokens = pmetal_data::inference_config::collect_all_stop_tokens(
            model_path,
            &tokenizer,
            template_type,
        );
        tracing::info!(count = stop_tokens.len(), "Stop tokens collected");

        // 8. Build GenerationConfig
        let gen_config = if temperature < 1e-6 {
            GenerationConfig::greedy(max_tokens).with_stop_tokens(stop_tokens)
        } else {
            let mut gc = GenerationConfig::sampling(max_tokens, temperature)
                .with_top_k(top_k)
                .with_top_p(top_p)
                .with_min_p(min_p)
                .with_repetition_penalty(repetition_penalty)
                .with_frequency_penalty(frequency_penalty)
                .with_presence_penalty(presence_penalty)
                .with_stop_tokens(stop_tokens);
            if let Some(s) = config.seed {
                gc = gc.with_seed(s);
            }
            gc
        };

        // 9. Load model (standard or LoRA-merged)
        let max_seq_len = input_ids.len() + max_tokens + 64;

        let mut cache_request = cache_mode_request_from_config(&config);
        if mtp_requested {
            cache_request.no_kv_quant = true;
        }
        let (model, cache, mamba_cache, native_turboquant, native_quant_config) = if let Some(
            native_info,
        ) =
            native_bridge_info
        {
            let base_cache_config = native_bridge_base_cache_config(native_info, max_seq_len);
            let cache_selection = select_cache_mode_with_working_set(
                &base_cache_config,
                estimate_weight_bytes(model_path, 0, cache_request.fp8),
                cache_request,
                None,
            );

            if native_cache_mode_supported(native_info, cache_selection.mode) {
                log_cache_selection(&cache_selection, max_seq_len);
                let cache = build_native_placeholder_cache(&base_cache_config);
                let mamba_cache = Some(MambaCache::new(native_info.num_layers));

                // TurboQuant v2 mixed-bit preset overrides the standard cache mode.
                // Outlier count is rounded down to the nearest group_size (64) boundary.
                let mixed_bit_override =
                    mixed_bit_config_from_preset(config.kv_quant_preset.as_deref(), native_info);

                // Extract affine quant config from cache mode (or from mixed-bit preset).
                // QJL is only active for uniform (non-mixed-bit) Q2-Q3.
                let qjl_requested = config.kv_qjl;
                let qcfg = if let Some(mb) = mixed_bit_override {
                    // Mixed-bit path: QJL not supported (different architecture)
                    if qjl_requested {
                        tracing::warn!(
                            "--kv-qjl is not supported with mixed-bit presets; ignoring"
                        );
                    }
                    Some(pmetal_runtime::qwen3_native::QuantCacheConfig {
                        bits: mb.regular_bits,
                        group_size: 64,
                        mixed_bit: Some(mb),
                        qjl: false,
                    })
                } else {
                    match cache_selection.mode {
                        CacheMode::Quantized { bits, group_size } => {
                            let qjl_active = qjl_requested && bits <= 3;
                            if qjl_requested && bits > 3 {
                                tracing::warn!(
                                    bits,
                                    "--kv-qjl is only effective for Q2-Q3; ignoring for Q{bits}"
                                );
                            }
                            Some(pmetal_runtime::qwen3_native::QuantCacheConfig {
                                bits,
                                group_size: group_size as i32,
                                mixed_bit: None,
                                qjl: qjl_active,
                            })
                        }
                        _ => None,
                    }
                };
                (
                    LoadedModel::NativeOnly,
                    cache,
                    mamba_cache,
                    turboquant_config_from_mode(native_info, cache_selection.mode),
                    qcfg,
                )
            } else {
                tracing::info!(
                    mode = %cache_selection.mode.describe(),
                    arch = native_info.arch.label(),
                    "Bridge-native path does not support the selected cache mode; using shared model path"
                );
                let m = match preloaded_model.take() {
                    Some(model) => model,
                    None => load_standard_model_for_inference(model_path, &config)?,
                };

                let base_cache_config = m.create_cache(max_seq_len).config().clone();
                tracing::info!(tokens = max_seq_len, "Base KV cache created");
                let cache_selection = select_cache_mode_for_model(
                    &base_cache_config,
                    model_path,
                    m.num_parameters(),
                    cache_request,
                );
                log_cache_selection(&cache_selection, max_seq_len);

                let cache = build_cache_from_base_config(&base_cache_config, cache_selection.mode);
                let mamba_cache = m.create_mamba_cache();
                (LoadedModel::Standard(m), cache, mamba_cache, None, None)
            }
        } else if let Some(ref lora_path) = config.lora_path {
            #[cfg(feature = "lora")]
            {
                let (lora_model, cache, mamba_cache, cache_selection) =
                    load_model_with_lora(model_path, lora_path, max_seq_len, &config)?;
                log_cache_selection(&cache_selection, max_seq_len);
                (
                    LoadedModel::Lora(lora_model),
                    cache,
                    mamba_cache,
                    None,
                    None,
                )
            }
            #[cfg(not(feature = "lora"))]
            {
                let _ = lora_path;
                return Err(Exception::custom(
                    "LoRA adapters require the `lora` feature in this build",
                ));
            }
        } else {
            let m = preloaded_model
                .take()
                .expect("standard path should preload the dynamic model");

            let base_cache_config = m.create_cache(max_seq_len).config().clone();
            tracing::info!(tokens = max_seq_len, "Base KV cache created");
            let cache_selection = select_cache_mode_for_model(
                &base_cache_config,
                model_path,
                m.num_parameters(),
                cache_request,
            );
            log_cache_selection(&cache_selection, max_seq_len);

            let cache = build_cache_from_base_config(&base_cache_config, cache_selection.mode);
            let mamba_cache = m.create_mamba_cache();
            (LoadedModel::Standard(m), cache, mamba_cache, None, None)
        };

        let (mtp_assistant, mtp_config) =
            if let Some(ref assistant_path) = config.mtp_assistant_path {
                match &model {
                    LoadedModel::Standard(DynamicModel::Gemma4(target)) => {
                        let (assistant, assistant_gen_config) =
                            load_gemma4_assistant_from_dir(assistant_path)?;
                        validate_gemma4_mtp_pair(target, &assistant)?;
                        tracing::info!(
                            assistant = %assistant_path.display(),
                            num_assistant_tokens = assistant_gen_config.num_assistant_tokens,
                            "Gemma 4 MTP assistant loaded"
                        );
                        (
                            Some(assistant),
                            Some(Gemma4MtpConfig {
                                num_assistant_tokens: assistant_gen_config.num_assistant_tokens,
                            }),
                        )
                    }
                    LoadedModel::Standard(other) => {
                        return Err(Exception::custom(format!(
                            "Gemma 4 MTP requires a Gemma 4 target model; got {:?}",
                            other.architecture()
                        )));
                    }
                    #[cfg(feature = "lora")]
                    LoadedModel::Lora(_) => {
                        return Err(Exception::custom(
                            "Gemma 4 MTP is not supported with LoRA-merged inference",
                        ));
                    }
                    LoadedModel::NativeOnly => {
                        return Err(Exception::custom(
                            "Gemma 4 MTP requires the standard shared model path",
                        ));
                    }
                }
            } else {
                (None, None)
            };

        let (qwen_mtp, qwen_mtp_cache, qwen_mtp_config) = if config.qwen_mtp {
            let qwen_mtp_path = config.qwen_mtp_path.as_deref().unwrap_or(model_path);
            match &model {
                LoadedModel::Standard(DynamicModel::Qwen3Next(target)) => {
                    let mut mtp = load_qwen3_next_mtp_from_dir(qwen_mtp_path, &target.config)?;
                    validate_qwen3_next_mtp_pair(&target.config, &mtp)?;
                    if config.fp8 {
                        tracing::info!("Quantizing Qwen MTP weights to FP8 E4M3");
                        mtp.quantize_fp8_weights()?;
                    }
                    let mtp_cache = mtp.create_cache(max_seq_len);
                    let mtp_config = Qwen3NextMtpConfig {
                        num_draft_tokens: config.qwen_mtp_draft_tokens.max(1),
                    };
                    tracing::info!(
                        checkpoint = %qwen_mtp_path.display(),
                        draft_tokens = mtp_config.num_draft_tokens,
                        predictor_layers = mtp.config.mtp_num_hidden_layers(),
                        "Qwen MTP loaded"
                    );
                    (Some(mtp), Some(mtp_cache), Some(mtp_config))
                }
                LoadedModel::Standard(other) => {
                    return Err(Exception::custom(format!(
                        "Qwen MTP requires a Qwen3Next/Qwen3.6 target model; got {:?}",
                        other.architecture()
                    )));
                }
                // An adapted model wraps the same `DynamicModel` the standard
                // arm above matches on, so the target is reached through it
                // rather than through a LoRA-specific variant.
                #[cfg(feature = "lora")]
                LoadedModel::Lora(lora) => match lora.model() {
                    DynamicModel::Qwen3Next(target) => {
                        let mut mtp = load_qwen3_next_mtp_from_dir(qwen_mtp_path, &target.config)?;
                        validate_qwen3_next_mtp_pair(&target.config, &mtp)?;
                        if config.fp8 {
                            tracing::info!("Quantizing Qwen MTP weights to FP8 E4M3");
                            mtp.quantize_fp8_weights()?;
                        }
                        let mtp_cache = mtp.create_cache(max_seq_len);
                        let mtp_config = Qwen3NextMtpConfig {
                            num_draft_tokens: config.qwen_mtp_draft_tokens.max(1),
                        };
                        tracing::info!(
                            checkpoint = %qwen_mtp_path.display(),
                            draft_tokens = mtp_config.num_draft_tokens,
                            predictor_layers = mtp.config.mtp_num_hidden_layers(),
                            "Qwen MTP loaded for LoRA-merged target"
                        );
                        (Some(mtp), Some(mtp_cache), Some(mtp_config))
                    }
                    other => {
                        return Err(Exception::custom(format!(
                            "Qwen MTP requires a Qwen3Next/Qwen3.6 target model; got {:?}",
                            other.architecture()
                        )));
                    }
                },
                LoadedModel::NativeOnly => {
                    let target_config = load_qwen3_next_config_for_mtp(model_path)?;
                    let mut mtp = load_qwen3_next_mtp_from_dir(qwen_mtp_path, &target_config)?;
                    validate_qwen3_next_mtp_pair(&target_config, &mtp)?;
                    if config.fp8 {
                        tracing::info!("Quantizing Qwen MTP weights to FP8 E4M3");
                        mtp.quantize_fp8_weights()?;
                    }
                    let mtp_cache = mtp.create_cache(max_seq_len);
                    let mtp_config = Qwen3NextMtpConfig {
                        num_draft_tokens: config.qwen_mtp_draft_tokens.max(1),
                    };
                    tracing::info!(
                        checkpoint = %qwen_mtp_path.display(),
                        draft_tokens = mtp_config.num_draft_tokens,
                        predictor_layers = mtp.config.mtp_num_hidden_layers(),
                        "Qwen MTP loaded for native verifier"
                    );
                    (Some(mtp), Some(mtp_cache), Some(mtp_config))
                }
            }
        } else {
            (None, None, None)
        };

        Ok(Self {
            tokenizer,
            state: InferenceGenState {
                model,
                gen_config,
                input_ids,
                cache,
                mamba_cache,
                mtp_assistant,
                mtp_config,
                qwen_mtp,
                qwen_mtp_cache,
                qwen_mtp_config,
                native_turboquant,
                native_quant_config,
                model_path: config.model_path.clone(),
                last_decode_metrics: None,
                detect_repetition: config.detect_repetition,
                images: image_media,
                videos: video_media,
            },
            chat_template_type: template_type,
            is_chat: use_chat,
            thinks,
        })
    }

    /// Whether chat mode is active.
    pub fn is_chat(&self) -> bool {
        self.is_chat
    }

    /// Whether the prompt asks the model to think before it answers (chat
    /// mode, a thinking template, and thinking on, by flag or by the
    /// template's default).
    pub fn thinks(&self) -> bool {
        self.thinks
    }

    /// The detected chat template type (if chat mode is active).
    pub fn chat_template_type(&self) -> Option<ChatTemplateType> {
        self.chat_template_type
    }

    /// Benchmark the active model with a fixed workload shape:
    /// fixed random prompt ids, one warmup, EOS disabled,
    /// and repeated full prompt+decode runs.
    pub fn benchmark(
        &mut self,
        prompt_tokens: usize,
        generation_tokens: usize,
        num_trials: usize,
        seed: u64,
    ) -> Result<Vec<crate::native_inference::BenchmarkTrialMetrics>, Exception> {
        let vocab_size = self.tokenizer.vocab_size();
        if vocab_size == 0 {
            return Err(Exception::custom("tokenizer vocabulary is empty"));
        }

        let prompt_ids = build_benchmark_prompt(prompt_tokens, vocab_size, seed);
        match &mut self.state.model {
            LoadedModel::NativeOnly => crate::native_inference::benchmark_native(
                &self.state.model_path,
                &prompt_ids,
                generation_tokens,
                self.state.native_turboquant,
                num_trials,
            )
            .map_err(Exception::custom),
            _ => Err(Exception::custom(
                "The inference benchmark is currently only implemented for the native bridge path",
            )),
        }
    }
}

impl InferenceGenState {
    /// Stream tokens via callback. Returns generation output.
    ///
    /// `on_token` receives each generated token ID and returns `true` to
    /// continue or `false` to stop (e.g., on cancellation).
    ///
    /// Split from `InferenceRunner` so callers can hold `&runner.tokenizer`
    /// while calling `runner.gen.generate_streaming(...)`.
    pub fn generate_streaming<F>(&mut self, on_token: F) -> Result<GenerationOutput, Exception>
    where
        F: FnMut(u32) -> bool,
    {
        // Optional repetition-loop detector (opt-in via --detect-repetition).
        //
        // When enabled, detects n-gram repetition loops (8-token n-gram × 4
        // repeats = 32-token window) and force-stops generation. Useful as a
        // safety net for small models prone to infinite loops.
        let mut detector = self
            .detect_repetition
            .then(|| RepetitionDetector::new(8, 4));
        let mut token_count = 0usize;
        let mut on_token = on_token;
        let mut on_token_guarded = |token: u32| -> bool {
            token_count += 1;
            if let Some(ref mut det) = detector {
                if det.push_and_check(token) {
                    tracing::warn!(
                        "Repetition loop detected after {} tokens, stopping",
                        token_count
                    );
                    return false;
                }
            }
            on_token(token)
        };

        if let Some(ref mut assistant) = self.mtp_assistant {
            let mtp_config = self.mtp_config.clone().unwrap_or_default();
            let output = match self.model {
                LoadedModel::Standard(DynamicModel::Gemma4(ref mut target)) => {
                    generate_gemma4_mtp_streaming(
                        target,
                        assistant,
                        &self.input_ids,
                        self.gen_config.clone(),
                        &mut self.cache,
                        mtp_config,
                        on_token_guarded,
                    )
                }
                LoadedModel::Standard(ref model) => Err(Exception::custom(format!(
                    "Gemma 4 MTP requires a Gemma 4 target model; got {:?}",
                    model.architecture()
                ))),
                #[cfg(feature = "lora")]
                LoadedModel::Lora(_) => Err(Exception::custom(
                    "Gemma 4 MTP is not supported with LoRA-merged inference",
                )),
                LoadedModel::NativeOnly => Err(Exception::custom(
                    "Gemma 4 MTP requires the standard shared model path",
                )),
            }?;
            self.last_decode_metrics = output.decode_metrics;
            return Ok(output);
        }

        if let Some(ref mut mtp) = self.qwen_mtp {
            let qwen_mtp_config = self.qwen_mtp_config.clone().unwrap_or_default();
            let mtp_cache = self.qwen_mtp_cache.as_mut().ok_or_else(|| {
                Exception::custom("Qwen MTP internal error: missing MTP KV cache")
            })?;
            let target_mamba_cache = self
                .mamba_cache
                .as_mut()
                .ok_or_else(|| Exception::custom("Qwen MTP requires a target Mamba/GDN cache"))?;
            let output = match self.model {
                LoadedModel::Standard(DynamicModel::Qwen3Next(ref mut target)) => {
                    generate_qwen3_next_mtp_streaming(
                        target,
                        mtp,
                        &self.input_ids,
                        self.gen_config.clone(),
                        &mut self.cache,
                        target_mamba_cache,
                        mtp_cache,
                        qwen_mtp_config,
                        on_token_guarded,
                    )
                }
                LoadedModel::Standard(ref model) => Err(Exception::custom(format!(
                    "Qwen MTP requires a Qwen3Next/Qwen3.6 target model; got {:?}",
                    model.architecture()
                ))),
                #[cfg(feature = "lora")]
                LoadedModel::Lora(ref mut lora) => match lora.model_mut() {
                    DynamicModel::Qwen3Next(target) => generate_qwen3_next_mtp_streaming(
                        target,
                        mtp,
                        &self.input_ids,
                        self.gen_config.clone(),
                        &mut self.cache,
                        target_mamba_cache,
                        mtp_cache,
                        qwen_mtp_config,
                        on_token_guarded,
                    ),
                    other => Err(Exception::custom(format!(
                        "Qwen MTP requires a Qwen3Next/Qwen3.6 target model; got {:?}",
                        other.architecture()
                    ))),
                },
                LoadedModel::NativeOnly => {
                    crate::native_inference::ensure_native_runtime_metal_available()
                        .map_err(Exception::custom)?;
                    generate_qwen3_native_mtp_streaming(
                        &self.model_path,
                        mtp,
                        mtp_cache,
                        &self.input_ids,
                        self.gen_config.clone(),
                        qwen_mtp_config,
                        on_token_guarded,
                    )
                }
            }?;
            self.last_decode_metrics = output.decode_metrics;
            return Ok(output);
        }

        if matches!(self.model, LoadedModel::NativeOnly) {
            let stop_tokens = self.gen_config.stop_tokens.clone();
            // The native engines sample with the filters and penalties the
            // other paths use.
            let sampling_params = self.gen_config.sampling_params();
            let on_native_token =
                |token| forward_native_token(&stop_tokens, &mut on_token_guarded, token);
            let output = if self.images.is_empty() && self.videos.is_empty() {
                crate::native_inference::run_native_inference_ext(
                    &self.model_path,
                    &self.input_ids,
                    self.gen_config.max_new_tokens,
                    sampling_params,
                    self.native_turboquant,
                    self.native_quant_config,
                    on_native_token,
                )
            } else {
                crate::native_inference::run_native_multimodal_inference_ext(
                    &self.model_path,
                    &self.input_ids,
                    &self.images,
                    &self.videos,
                    self.gen_config.max_new_tokens,
                    sampling_params,
                    self.native_turboquant,
                    self.native_quant_config,
                    on_native_token,
                )
            }
            .map_err(Exception::custom)?;

            self.last_decode_metrics = output.decode_metrics;
            return Ok(GenerationOutput {
                token_ids: output.token_ids,
                num_generated: output.num_generated,
                stopped_by_token: output.stopped_by_token,
                stopped_by_length: output.stopped_by_length,
                decode_metrics: output.decode_metrics,
                speculative_metrics: None,
            });
        }

        let output = match self.model {
            LoadedModel::Standard(ref mut model) => {
                let mamba = &mut self.mamba_cache;
                generate_cached_async_streaming(
                    // Sampling reads the last position's logits alone.
                    |input, cache| {
                        model.forward_with_hybrid_cache_last(
                            input,
                            None,
                            Some(cache),
                            mamba.as_mut(),
                        )
                    },
                    &self.input_ids,
                    self.gen_config.clone(),
                    &mut self.cache,
                    on_token_guarded,
                )
            }
            #[cfg(feature = "lora")]
            LoadedModel::Lora(ref mut model) => {
                let mamba = &mut self.mamba_cache;
                generate_cached_async_streaming(
                    |input, cache| {
                        model
                            .forward_with_hybrid_cache_last(
                                input,
                                None,
                                Some(cache),
                                mamba.as_mut(),
                            )
                            .map_err(|e| Exception::custom(e.to_string()))
                    },
                    &self.input_ids,
                    self.gen_config.clone(),
                    &mut self.cache,
                    on_token_guarded,
                )
            }
            LoadedModel::NativeOnly => {
                // native path returns before reaching this fallback code
                unreachable!("NativeOnly model should have returned before the shared fallback")
            }
        }?;
        // Forward the streaming generator's per-step decode metrics into
        // the runner state so the CLI summary print ingests them the same
        // way it does for the bridge-native path.
        self.last_decode_metrics = output.decode_metrics;
        Ok(output)
    }

    /// Run a callback with mutable access to the standard DynamicModel and its caches.
    ///
    /// This is useful for opt-in diagnostics that need model-specific APIs while
    /// preserving the shared inference setup. LoRA-merged models are rejected
    /// because their wrapper does not expose the same architecture-specific
    /// surfaces.
    pub fn run_standard_model_with_state<F, R>(&mut self, f: F) -> Result<R, Exception>
    where
        F: FnOnce(&mut DynamicModel, &mut KVCache, Option<&mut MambaCache>) -> Result<R, Exception>,
    {
        let Self {
            ref mut model,
            ref mut cache,
            ref mut mamba_cache,
            ..
        } = *self;

        match model {
            LoadedModel::Standard(dynamic_model) => f(dynamic_model, cache, mamba_cache.as_mut()),
            #[cfg(feature = "lora")]
            LoadedModel::Lora(_) => Err(Exception::custom(
                "this inference mode is only available for standard models",
            )),
            LoadedModel::NativeOnly => Err(Exception::custom(
                "this inference mode is not available in native mode",
            )),
        }
    }

    /// Forward pass through the model (for non-streaming generation paths).
    ///
    /// Dispatches to the correct model type (standard or LoRA-merged) with
    /// hybrid cache support.
    pub fn forward(&mut self, input: &Array, cache: &mut KVCache) -> Result<Array, Exception> {
        match &mut self.model {
            LoadedModel::Standard(model) => {
                model.forward_with_hybrid_cache(input, None, Some(cache), self.mamba_cache.as_mut())
            }
            #[cfg(feature = "lora")]
            LoadedModel::Lora(model) => model
                .forward_with_hybrid_cache(input, None, Some(cache), self.mamba_cache.as_mut())
                .map_err(|e| Exception::custom(e.to_string())),
            LoadedModel::NativeOnly => {
                Err(Exception::custom("forward is not available in native mode"))
            }
        }
    }

    /// Access the underlying DynamicModel (standard path only).
    pub fn dynamic_model(&self) -> Option<&DynamicModel> {
        match &self.model {
            LoadedModel::Standard(m) => Some(m),
            #[cfg(feature = "lora")]
            LoadedModel::Lora(_) => None,
            LoadedModel::NativeOnly => None,
        }
    }

    /// Mutable access to the DynamicModel.
    pub fn dynamic_model_mut(&mut self) -> Option<&mut DynamicModel> {
        match &mut self.model {
            LoadedModel::Standard(m) => Some(m),
            #[cfg(feature = "lora")]
            LoadedModel::Lora(_) => None,
            LoadedModel::NativeOnly => None,
        }
    }

    /// Access the generation config.
    pub fn gen_config(&self) -> &GenerationConfig {
        &self.gen_config
    }

    /// Access the tokenized input IDs.
    pub fn input_ids(&self) -> &[u32] {
        &self.input_ids
    }

    /// Mutable access to the KV cache.
    pub fn cache_mut(&mut self) -> &mut KVCache {
        &mut self.cache
    }

    /// Mutable access to the Mamba cache.
    pub fn mamba_cache_mut(&mut self) -> Option<&mut MambaCache> {
        self.mamba_cache.as_mut()
    }
}

fn forward_native_token<F>(stop_tokens: &[u32], on_token: &mut F, token: u32) -> bool
where
    F: FnMut(u32) -> bool,
{
    if stop_tokens.contains(&token) {
        return false;
    }
    on_token(token)
}

// ── Repetition loop detection ─────────────────────────────────────────────────

/// Detects infinite n-gram repetition loops in the token stream.
///
/// Maintains a sliding window of recent tokens. After each push, checks whether
/// the last `max_repeats * ngram_size` tokens are the same n-gram repeated
/// `max_repeats` times. Returns `true` (stop signal) when a loop is detected.
///
/// Default parameters (8 tokens × 4 repeats = 32-token window) catch patterns
/// like "in the canton of the canton of the canton of the canton of" without
/// false-positives on legitimate repeated phrases.
pub(crate) struct RepetitionDetector {
    window: Vec<u32>,
    ngram_size: usize,
    max_repeats: usize,
}

impl RepetitionDetector {
    pub(crate) fn new(ngram_size: usize, max_repeats: usize) -> Self {
        Self {
            window: Vec::new(),
            ngram_size,
            max_repeats,
        }
    }

    /// Push `token` and return `true` if a repetition loop is detected.
    ///
    /// A loop is detected when the last `max_repeats * ngram_size` tokens
    /// consist of exactly the same n-gram repeated `max_repeats` times.
    pub(crate) fn push_and_check(&mut self, token: u32) -> bool {
        self.window.push(token);
        let required = self.ngram_size * self.max_repeats;
        if self.window.len() < required {
            return false;
        }
        let tail = &self.window[self.window.len() - required..];
        let ngram = &tail[..self.ngram_size];
        tail.chunks_exact(self.ngram_size)
            .all(|chunk| chunk == ngram)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn cache_mode_request_from_config(config: &InferenceRunnerConfig) -> CacheModeRequest {
    CacheModeRequest {
        kv_quant: config.kv_quant,
        kv_k_bits: config.kv_k_bits,
        kv_v_bits: config.kv_v_bits,
        kv_group_size: config.kv_group_size,
        kv_turboquant: config.kv_turboquant,
        kv_turboquant_preset: config.kv_turboquant_preset,
        no_kv_quant: config.no_kv_quant || config.mtp_assistant_path.is_some() || config.qwen_mtp,
        fp8: config.fp8,
    }
}

fn build_benchmark_prompt(prompt_tokens: usize, vocab_size: usize, seed: u64) -> Vec<u32> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let upper = vocab_size.max(1) as u64;
    let mut out = Vec::with_capacity(prompt_tokens);
    for _ in 0..prompt_tokens {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let sample = state.wrapping_mul(0x2545_F491_4F6C_DD1D) % upper;
        out.push(sample as u32);
    }
    out
}

/// Preprocess a prompt's images and videos with the checkpoint's processor.
fn preprocess_media(
    model_path: &Path,
    multimodal: &Qwen3_5MultimodalConfig,
    images: &[PathBuf],
    videos: &[VideoInput],
) -> Result<(Vec<ProcessedMedia>, Vec<ProcessedMedia>), Exception> {
    let processor = QwenVlProcessor::from_model_dir(model_path)
        .map_err(|e| Exception::custom(e.to_string()))?;
    let merge = multimodal.vision.spatial_merge_size;
    let images = images
        .iter()
        .map(|path| {
            let image = pmetal_data::qwen_vl_processing::load_image_source(&path.to_string_lossy())
                .and_then(|image| processor.preprocess_image(&image))
                .map_err(|e| Exception::custom(format!("{}: {e}", path.display())))?;
            tracing::info!(
                image = %path.display(),
                grid = ?image.grid_thw,
                tokens = image.num_tokens(merge),
                "Image preprocessed"
            );
            Ok(image)
        })
        .collect::<Result<Vec<_>, Exception>>()?;
    let videos = videos
        .iter()
        .map(|video| {
            let frames = pmetal_data::qwen_vl_processing::video_frame_paths(&video.frames_dir)
                .map_err(|e| Exception::custom(e.to_string()))?;
            if video.fps.is_none() {
                // The reference warns here too: frames sampled at a lower
                // rate than the assumed 24 lose most of themselves to the
                // processor's own sampling (a short clip drops to 4 frames)
                // and get timestamps that are off by the same factor.
                tracing::warn!(
                    video = %video.frames_dir.display(),
                    "No frame rate given for the video; assuming {} frames per second, as the \
                     reference does. Give the rate the frames were extracted at (this model \
                     samples {} per second)",
                    pmetal_data::qwen_vl_processing::DEFAULT_VIDEO_FPS,
                    processor.video.fps
                );
            }
            let processed = processor
                .preprocess_video_files(&frames, video.fps)
                .map_err(|e| Exception::custom(e.to_string()))?;
            tracing::info!(
                video = %video.frames_dir.display(),
                frames = frames.len(),
                sampled = processed.frame_indices.len(),
                grid = ?processed.grid_thw,
                tokens = processed.num_tokens(merge),
                "Video preprocessed"
            );
            Ok(processed)
        })
        .collect::<Result<Vec<_>, Exception>>()?;
    Ok((images, videos))
}

fn load_standard_model_for_inference(
    model_path: &Path,
    config: &InferenceRunnerConfig,
) -> Result<DynamicModel, Exception> {
    tracing::info!("Loading dynamic model");
    let mut model = DynamicModel::load_with_options(
        model_path,
        pmetal_models::dispatcher::DynamicModelLoadOptions {
            prefer_expert_offload: config.experts_dir.is_some(),
        },
    )?;
    tracing::info!(arch = %model.architecture(), "Model loaded");

    if config.fp8 {
        tracing::info!("Quantizing weights to FP8 E4M3");
        model.quantize_fp8()?;
    }

    if let Some(ref experts_dir) = config.experts_dir {
        model.enable_expert_offloading(Path::new(experts_dir))?;
    } else if model.requires_expert_offloading() {
        return Err(Exception::custom(
            "this model requires expert offloading; repack routed experts with `pmetal pack-experts` and pass --experts-dir <packed_dir>",
        ));
    }

    Ok(model)
}

fn native_bridge_base_cache_config(
    info: crate::native_inference::NativeRuntimeInfo,
    max_seq_len: usize,
) -> KVCacheConfig {
    KVCacheConfig::new(
        info.num_layers,
        max_seq_len,
        info.num_kv_heads,
        info.head_dim,
    )
    .with_value_head_dim(info.value_head_dim)
    .with_dtype(Dtype::Float16)
}

fn native_cache_mode_supported(
    info: crate::native_inference::NativeRuntimeInfo,
    mode: CacheMode,
) -> bool {
    match mode {
        CacheMode::Standard => true,
        // The native cache quantizes uniform widths only; a mixed-precision
        // preset runs on the shared model path's cache.
        CacheMode::TurboQuant { config } => info.supports_turboquant && config.validate().is_ok(),
        // Zero-overhead quantized KV cache via quantized_matmul.
        // MLX supports 2, 3, 4, 5, 6, 8 bits for affine group quantization.
        CacheMode::Quantized { bits, .. } => matches!(bits, 2 | 3 | 4 | 5 | 6 | 8),
        _ => false,
    }
}

/// Construct a `MixedBitConfig` from a CLI preset string and the model's head_dim.
///
/// Presets:
/// - "q2_5": outlier_bits=3, regular_bits=2, outlier_fraction=0.25
/// - "q3_5": outlier_bits=4, regular_bits=3, outlier_fraction=0.25
///
/// Outlier count is rounded down to the nearest multiple of 64 (the group_size)
/// so that quantization groups align cleanly.
fn mixed_bit_config_from_preset(
    preset: Option<&str>,
    info: crate::native_inference::NativeRuntimeInfo,
) -> Option<pmetal_runtime::qwen3_native::MixedBitConfig> {
    let preset = preset?;
    let (outlier_bits, regular_bits) = match preset {
        "q2_5" => (3u8, 2u8),
        "q3_5" => (4u8, 3u8),
        other => {
            tracing::warn!(preset = other, "Unknown --kv-quant-preset; ignoring");
            return None;
        }
    };
    const OUTLIER_FRACTION: f32 = 0.25;
    const GROUP_SIZE: i32 = 64;
    let head_dim = info.head_dim as i32;
    let raw = (head_dim as f32 * OUTLIER_FRACTION).round() as i32;
    let outlier_count = (raw / GROUP_SIZE) * GROUP_SIZE;
    if outlier_count == 0 || outlier_count >= head_dim {
        tracing::warn!(
            head_dim,
            outlier_count,
            "Mixed-bit outlier_count is invalid for this model's head_dim; ignoring --kv-quant-preset"
        );
        return None;
    }
    tracing::info!(
        preset,
        outlier_count,
        head_dim,
        outlier_bits,
        regular_bits,
        "Enabling TurboQuant v2 mixed-bit KV cache"
    );
    Some(pmetal_runtime::qwen3_native::MixedBitConfig {
        outlier_count,
        outlier_bits,
        regular_bits,
    })
}

fn turboquant_config_from_mode(
    info: crate::native_inference::NativeRuntimeInfo,
    mode: CacheMode,
) -> Option<TurboQuantConfig> {
    if !info.supports_turboquant {
        return None;
    }

    match mode {
        CacheMode::TurboQuant { config } => Some(config),
        _ => None,
    }
}

/// The LoRA hyperparameters an adapter was trained with, from the
/// `adapter_config.json` beside it. `lora_path` may be the adapter directory or
/// its weights file. PEFT's keys and the ones earlier adapters carried are both
/// read. Falls back to the defaults when there is no config, and refuses one
/// that describes something a uniform `LoraConfig` cannot (per-module ranks,
/// extra trained modules; see [`pmetal_lora::peft::lora_config_from_peft`]).
#[cfg(feature = "lora")]
pub fn lora_config_for_adapter(lora_path: &Path) -> Result<pmetal_core::LoraConfig, String> {
    let dir = pmetal_lora::peft::adapter_dir(lora_path);
    let Ok(text) = std::fs::read_to_string(dir.join(pmetal_lora::peft::ADAPTER_CONFIG)) else {
        return Ok(pmetal_core::LoraConfig::default());
    };
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("adapter_config.json: {e}"))?;
    pmetal_lora::peft::lora_config_from_peft(&value).map_err(|e| e.to_string())
}

/// Load a model with LoRA weights merged in.
#[cfg(feature = "lora")]
fn load_model_with_lora(
    model_path: &Path,
    lora_path: &str,
    max_seq_len: usize,
    config: &InferenceRunnerConfig,
) -> Result<
    (
        DynamicLoraModel,
        KVCache,
        Option<MambaCache>,
        CacheModeSelection,
    ),
    Exception,
> {
    let lora_config = lora_config_for_adapter(Path::new(lora_path)).map_err(Exception::custom)?;

    tracing::info!(
        r = lora_config.r,
        alpha = lora_config.alpha,
        "Loading LoRA adapter"
    );

    let mut model = DynamicLoraModel::from_pretrained(model_path, lora_config)
        .map_err(|e| Exception::custom(format!("LoRA load: {e}")))?;
    model
        .load_lora_weights(lora_path)
        .map_err(|e| Exception::custom(format!("LoRA weights: {e}")))?;
    model
        .merge_lora()
        .map_err(|e| Exception::custom(format!("LoRA merge: {e}")))?;
    if config.fp8 {
        tracing::info!("Quantizing merged LoRA model weights to FP8 E4M3");
        model
            .quantize_fp8()
            .map_err(|e| Exception::custom(format!("LoRA FP8 quantization: {e}")))?;
    }
    model
        .eval_all()
        .map_err(|e| Exception::custom(format!("LoRA eval: {e}")))?;

    tracing::info!("LoRA merged into base model");

    let mamba_cache = model.create_mamba_cache();
    let base_cache = model
        .create_inference_kv_cache(max_seq_len)
        .ok_or_else(|| Exception::custom("model does not support KV cache"))?;
    let cache_selection = select_cache_mode_for_model(
        base_cache.config(),
        model_path,
        model.num_parameters(),
        cache_mode_request_from_config(config),
    );
    let cache = build_cache_from_base_config(base_cache.config(), cache_selection.mode);

    Ok((model, cache, mamba_cache, cache_selection))
}

/// Resolve KV cache quantization mode from CLI/GUI parameters.
fn build_cache_from_base_config(base_cache_config: &KVCacheConfig, mode: CacheMode) -> KVCache {
    let safe_mode = sanitize_cache_mode_for_config(base_cache_config, mode);
    KVCache::new(base_cache_config.clone().with_mode(safe_mode))
}

fn build_native_placeholder_cache(base_cache_config: &KVCacheConfig) -> KVCache {
    // Bridge-native inference owns its own KV/TurboQuant cache implementation.
    // Keep the runner's structural cache in the simplest mode so the native
    // path does not instantiate the shared pmetal-cache TurboQuant stack.
    KVCache::new(base_cache_config.clone().with_mode(CacheMode::Standard))
}

fn load_qwen3_next_config_for_mtp(model_path: &Path) -> Result<Qwen3NextConfig, Exception> {
    let config_path = model_path.join("config.json");
    let text = std::fs::read_to_string(&config_path).map_err(|e| {
        Exception::custom(format!(
            "read Qwen target config for native MTP {}: {e}",
            config_path.display()
        ))
    })?;
    // The same reading the target gets: a raw `text_config` deserialize
    // missed `rope_parameters`, so the predictor rotated with the default
    // RoPE base instead of the checkpoint's.
    Qwen3NextConfig::from_config_json(&text).map_err(|e| {
        Exception::custom(format!(
            "decode Qwen3Next config for native MTP {}: {e}",
            config_path.display()
        ))
    })
}

/// Check if a model directory looks instruction-tuned and should default to chat.
///
/// Primary signal is `tokenizer_config.json` containing a `chat_template`,
/// which is authoritative for modern HF checkpoints including Qwen 3.5 VLM
/// wrappers. Name markers are only a fallback for older repos.
fn model_looks_instruction_tuned(model_path: &Path) -> bool {
    let config_path = model_path.join("tokenizer_config.json");
    if let Ok(content) = std::fs::read_to_string(&config_path) {
        if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
            if config.get("chat_template").is_some() {
                return true;
            }
        }
    }

    let name = model_path.to_string_lossy().to_lowercase();
    [
        "instruct", "chat", "-it-", "-it/", "-it", "-sft", "-rlhf", "-dpo", "-grpo", "-rl",
    ]
    .iter()
    .any(|marker| {
        if *marker == "-it" {
            name.ends_with(marker)
        } else {
            name.contains(marker)
        }
    })
}

fn build_chat_messages(
    history: Option<&Vec<Message>>,
    system_message: Option<&str>,
    prompt: &str,
) -> Vec<Message> {
    let mut messages = history.cloned().unwrap_or_default();

    if let Some(system_message) = system_message.filter(|sys| !sys.is_empty()) {
        if matches!(messages.first(), Some(msg) if msg.role == "system") {
            messages[0] = Message::system(system_message);
        } else {
            messages.insert(0, Message::system(system_message));
        }
    }

    let should_append_prompt = !prompt.is_empty()
        && !matches!(messages.last(), Some(msg) if msg.role == "user" && msg.content == prompt);
    if should_append_prompt {
        messages.push(Message::user(prompt));
    }

    messages
}

/// Make `images` images and then `videos` videos items of the prompt's
/// message, ahead of its text: the content `[{"type": "image"}, ...,
/// {"type": "text", ...}]` a client sends `pmetal serve` and the reference
/// processor's chat template takes. The prompt's message is the last one
/// [`build_chat_messages`] built; without a prompt the media are a user
/// message of their own.
fn attach_media(messages: &mut Vec<Message>, prompt: &str, images: usize, videos: usize) {
    use pmetal_data::chat_templates::ContentPart;

    let mut parts: Vec<ContentPart> = std::iter::repeat_n(ContentPart::Image, images)
        .chain(std::iter::repeat_n(ContentPart::Video, videos))
        .collect();
    if prompt.is_empty() {
        messages.push(Message::with_parts("user", parts));
        return;
    }
    parts.push(ContentPart::Text(prompt.to_owned()));
    let last = messages
        .last_mut()
        .expect("build_chat_messages ends with the prompt's message");
    *last = Message::with_parts("user", parts);
}

/// A prompt no chat template rendered, as token ids: the tokenizer's own
/// post-processor adds the BOS (and EOS) the model was trained to start from.
///
/// Text a chat template rendered already carries its BOS and goes through
/// [`Tokenizer::encode`] instead; here nothing has written one, and a base
/// model given a prompt without it answers a different, usually worse,
/// distribution than the one it was trained on. A prompt that opens with the
/// BOS token itself is not given a second one.
fn encode_plain_prompt(tokenizer: &Tokenizer, text: &str) -> pmetal_core::Result<Vec<u32>> {
    tokenizer.encode_with_special_tokens(text)
}

fn build_plain_conversation_prompt(
    history: Option<&Vec<Message>>,
    system_message: Option<&str>,
    prompt: &str,
) -> String {
    let messages = build_chat_messages(history, system_message, prompt);
    let mut lines = Vec::with_capacity(messages.len() + 1);

    for message in messages {
        match message.role.as_str() {
            "system" => lines.push(format!("System: {}", message.content)),
            "user" => lines.push(format!("User: {}", message.content)),
            "assistant" => lines.push(format!("Assistant: {}", message.content)),
            _ => {}
        }
    }

    if !matches!(lines.last(), Some(last) if last.starts_with("Assistant:")) {
        lines.push("Assistant:".to_string());
    }

    lines.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_inference::{NativeArch, NativeRuntimeInfo};
    use tempfile::tempdir;

    fn qwen3_cache_config(max_seq_len: usize) -> KVCacheConfig {
        KVCacheConfig::new(28, max_seq_len, 8, 128)
    }

    fn spec() -> InferSpec {
        InferSpec {
            model: "org/model".into(),
            prompt: "hello".into(),
            ..InferSpec::default()
        }
    }

    #[test]
    fn a_spec_becomes_the_runner_configuration_it_describes() {
        let spec = InferSpec {
            lora: Some("adapter".into()),
            system: Some("be brief".into()),
            chat: true,
            temperature: Some(0.3),
            top_k: Some(40),
            max_tokens: Some(64),
            seed: Some(7),
            mtp: true,
            mtp_draft_tokens: 5,
            kv_quant: Some(4),
            kv_group_size: 32,
            kv_turboquant: true,
            kv_turboquant_preset: Some("q3_5".into()),
            image: vec!["a.png".into(), "b.png".into()],
            video: vec!["frames".into()],
            video_fps: Some(12.0),
            mode: "coding".into(),
            detect_repetition: true,
            ..spec()
        };
        let config = InferenceRunnerConfig::from_spec(&spec).unwrap();

        assert_eq!(config.prompt, "hello");
        assert_eq!(config.lora_path.as_deref(), Some("adapter"));
        assert_eq!(config.system_message.as_deref(), Some("be brief"));
        assert!(config.chat && config.qwen_mtp && config.detect_repetition);
        assert_eq!(config.qwen_mtp_draft_tokens, 5);
        assert_eq!(config.temperature, Some(0.3));
        assert_eq!(
            (config.top_k, config.max_tokens, config.seed),
            (Some(40), Some(64), Some(7))
        );
        assert_eq!((config.kv_quant, config.kv_group_size), (Some(4), 32));
        assert!(config.kv_turboquant);
        assert_eq!(config.kv_turboquant_preset, Some(TurboQuantPreset::Q3_5));
        assert_eq!(
            config.mode,
            pmetal_data::inference_config::SamplingMode::ThinkingCoding
        );
        assert_eq!(
            config.images,
            [PathBuf::from("a.png"), PathBuf::from("b.png")]
        );
        assert_eq!(config.videos.len(), 1);
        assert_eq!(config.videos[0].frames_dir, PathBuf::from("frames"));
        assert_eq!(config.videos[0].fps, Some(12.0));
        // The caller resolves the models and sets the tools.
        assert!(config.model_path.as_os_str().is_empty());
        assert!(config.mtp_assistant_path.is_none() && config.qwen_mtp_path.is_none());
        assert!(config.tools.is_none());
    }

    #[test]
    fn disabling_thinking_by_kwargs_or_by_flag_is_the_same_thing() {
        let by_flag = InferSpec {
            no_thinking: true,
            ..spec()
        };
        let by_kwargs = InferSpec {
            chat_template_kwargs: Some(r#"{"enable_thinking": false}"#.into()),
            ..spec()
        };
        for spec in [by_flag, by_kwargs] {
            let config = InferenceRunnerConfig::from_spec(&spec).unwrap();
            assert!(config.no_thinking);
            assert_eq!(config.chat_template_kwargs.enable_thinking(), Some(false));
        }
        assert!(
            !InferenceRunnerConfig::from_spec(&spec())
                .unwrap()
                .no_thinking
        );
    }

    #[test]
    fn a_draft_budget_of_zero_still_drafts_one_token() {
        let spec = InferSpec {
            mtp_draft_tokens: 0,
            ..spec()
        };
        let config = InferenceRunnerConfig::from_spec(&spec).unwrap();
        assert_eq!(config.qwen_mtp_draft_tokens, 1);
    }

    #[test]
    fn options_the_runner_cannot_use_are_errors_not_defaults() {
        let bad_preset = InferSpec {
            kv_turboquant_preset: Some("q9_9".into()),
            ..spec()
        };
        let err = InferenceRunnerConfig::from_spec(&bad_preset).unwrap_err();
        assert!(
            err.contains("kv_turboquant_preset") && err.contains("q9_9"),
            "{err}"
        );

        let bad_kwargs = InferSpec {
            chat_template_kwargs: Some("[1, 2]".into()),
            ..spec()
        };
        assert!(InferenceRunnerConfig::from_spec(&bad_kwargs).is_err());

        let bad_mode = InferSpec {
            mode: "wild".into(),
            ..spec()
        };
        assert!(InferenceRunnerConfig::from_spec(&bad_mode).is_err());
    }

    /// A base model's raw prompt starts with the BOS its tokenizer adds; a
    /// rendered chat, which carries its own, does not get a second one.
    #[test]
    fn a_plain_prompt_starts_with_the_bos_and_a_rendered_one_does_not_double_it() {
        let tokenizer = pmetal_data::tokenizer::testing::byte_level_bpe(&[], &["<s>"], Some("<s>"));
        let bos = tokenizer
            .inner()
            .token_to_id("<s>")
            .expect("<s> is a token");

        let plain = encode_plain_prompt(&tokenizer, "hi").unwrap();
        assert_eq!(
            plain[0], bos,
            "a plain prompt opens with the BOS: {plain:?}"
        );
        assert_eq!(plain.len(), 3, "BOS and two bytes: {plain:?}");

        let rendered = tokenizer.encode("<s>hi").unwrap();
        assert_eq!(rendered.iter().filter(|&&id| id == bos).count(), 1);
        let plain_with_bos = encode_plain_prompt(&tokenizer, "<s>hi").unwrap();
        assert_eq!(plain_with_bos, rendered, "no second BOS");
    }

    #[test]
    fn native_bridge_base_cache_config_preserves_asymmetric_value_dim() {
        let info = NativeRuntimeInfo {
            arch: NativeArch::DeepSeek,
            num_layers: 61,
            num_kv_heads: 128,
            head_dim: 192,
            value_head_dim: 128,
            supports_turboquant: false,
        };
        let config = native_bridge_base_cache_config(info, 4096);
        assert_eq!(config.num_layers, 61);
        assert_eq!(config.num_kv_heads, 128);
        assert_eq!(config.head_dim, 192);
        assert_eq!(config.value_head_dim, 128);
        assert_eq!(config.dtype, Dtype::Float16);
    }

    #[test]
    fn native_placeholder_cache_stays_standard_even_for_turboquant_mode() {
        let base = qwen3_cache_config(4096).with_mode(CacheMode::TurboQuant {
            config: TurboQuantConfig::uniform(8, 8),
        });
        let cache = build_native_placeholder_cache(&base);
        assert_eq!(cache.config().mode, CacheMode::Standard);
        assert_eq!(cache.config().head_dim, 128);
        assert_eq!(cache.config().value_head_dim, 128);
    }

    #[test]
    fn native_cache_mode_support_respects_turboquant_capability() {
        let qwen = NativeRuntimeInfo {
            arch: NativeArch::Qwen3_5,
            num_layers: 28,
            num_kv_heads: 2,
            head_dim: 128,
            value_head_dim: 128,
            supports_turboquant: true,
        };
        let llama4 = NativeRuntimeInfo {
            arch: NativeArch::Llama4,
            num_layers: 48,
            num_kv_heads: 8,
            head_dim: 128,
            value_head_dim: 128,
            supports_turboquant: false,
        };

        assert!(native_cache_mode_supported(qwen, CacheMode::Standard));
        assert!(native_cache_mode_supported(
            qwen,
            CacheMode::TurboQuant {
                config: TurboQuantConfig::uniform(8, 8),
            }
        ));
        // Mixed-precision presets run on the shared path's cache.
        assert!(!native_cache_mode_supported(
            qwen,
            CacheMode::TurboQuant {
                config: TurboQuantConfig::preset_q3_5(128),
            }
        ));
        assert!(!native_cache_mode_supported(
            llama4,
            CacheMode::TurboQuant {
                config: TurboQuantConfig::uniform(8, 8),
            }
        ));
    }

    #[test]
    fn build_chat_messages_uses_structured_history() {
        let history = vec![
            Message::user("hello"),
            Message::assistant("hi"),
            Message::user("write fizzbuzz"),
        ];

        let messages =
            build_chat_messages(Some(&history), Some("You are a helpful assistant."), "");

        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[2].role, "assistant");
        assert_eq!(messages[3].content, "write fizzbuzz");
    }

    #[test]
    fn build_chat_messages_updates_existing_system_and_no_think() {
        let history = vec![
            Message::system("old"),
            Message::user("hello"),
            Message::assistant("hi"),
        ];

        let messages = build_chat_messages(Some(&history), Some("new"), "");

        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].content, "new");
        assert_eq!(messages.len(), 3);
    }

    #[test]
    fn build_chat_messages_appends_prompt_when_missing() {
        let messages = build_chat_messages(None, None, "write fizzbuzz");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "write fizzbuzz");
    }

    /// The images and videos are items of the prompt's message, so the
    /// model's template places and labels them as transformers'
    /// `apply_chat_template` does: `add_vision_id` numbers each one, which a
    /// placeholder pasted into the text never got.
    #[test]
    fn media_are_items_of_the_prompt_message_for_the_template_to_place() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../pmetal-data/tests/fixtures/chat_template_kwargs_reference.json");
        let reference: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fixture).unwrap()).unwrap();
        let source = &reference["templates"]["qwen3_8"]["source"];
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            serde_json::json!({ "chat_template": source }).to_string(),
        )
        .unwrap();
        let template = pmetal_data::chat_templates::detect_chat_template(dir.path(), "qwen");

        let mut messages = build_chat_messages(None, Some("Be brief."), "Compare them.");
        attach_media(&mut messages, "Compare them.", 2, 1);
        let mut kwargs = ChatTemplateKwargs::new();
        kwargs.set("add_vision_id", true);
        let text = template
            .render_inference_jinja_with_kwargs(&messages, None, &kwargs)
            .unwrap();
        // transformers 5.19, `apply_chat_template(..., add_vision_id=True)`
        // on `[{"type": "image"}, {"type": "image"}, {"type": "video"},
        // {"type": "text", "text": "Compare them."}]`.
        assert_eq!(
            text,
            "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully \
             through the task, validate key assumptions, consider plausible alternatives, and \
             prioritize correctness, consistency, and clarity in the final answer.\n\nBe \
             brief.<|im_end|>\n<|im_start|>user\nPicture 1: \
             <|vision_start|><|image_pad|><|vision_end|>Picture 2: \
             <|vision_start|><|image_pad|><|vision_end|>Video 1: \
             <|vision_start|><|video_pad|><|vision_end|>Compare them.<|im_end|>\n\
             <|im_start|>assistant\n<think>\n"
        );

        // Media without a prompt are a user message of their own.
        let mut messages = build_chat_messages(Some(&vec![Message::user("hi")]), None, "");
        attach_media(&mut messages, "", 1, 0);
        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[1].parts.as_deref(),
            Some(&[pmetal_data::chat_templates::ContentPart::Image][..])
        );
    }

    #[test]
    fn build_plain_conversation_prompt_preserves_history_for_base_models() {
        let history = vec![Message::user("hello"), Message::assistant("hi there")];

        let prompt = build_plain_conversation_prompt(
            Some(&history),
            Some("You are a helpful assistant."),
            "what is the capital of france?",
        );

        assert_eq!(
            prompt,
            "System: You are a helpful assistant.\n\nUser: hello\n\nAssistant: hi there\n\nUser: what is the capital of france?\n\nAssistant:"
        );
    }

    #[test]
    fn forward_native_token_stops_before_callback_on_stop_token() {
        let mut seen = Vec::new();
        let mut callback = |token| {
            seen.push(token);
            true
        };

        assert!(forward_native_token(&[7, 9], &mut callback, 3));
        assert!(!forward_native_token(&[7, 9], &mut callback, 7));
        assert_eq!(seen, vec![3]);
    }

    #[test]
    fn benchmark_prompt_is_deterministic() {
        let a = build_benchmark_prompt(8, 1024, 0);
        let b = build_benchmark_prompt(8, 1024, 0);
        let c = build_benchmark_prompt(8, 1024, 1);

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.iter().all(|&tok| tok < 1024));
    }

    #[test]
    fn instruction_tuned_detection_prefers_chat_template() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template":"{{ messages }}"}"#,
        )
        .unwrap();
        assert!(model_looks_instruction_tuned(dir.path()));
    }

    #[test]
    fn instruction_tuned_detection_falls_back_to_name_markers() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("my-model-instruct");
        std::fs::create_dir(&path).unwrap();
        assert!(model_looks_instruction_tuned(&path));
    }

    #[cfg(unix)]
    #[test]
    fn repetition_detector_no_loop_short_stream() {
        let mut det = RepetitionDetector::new(4, 3);
        for tok in [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10] {
            assert!(!det.push_and_check(tok), "false positive on token {tok}");
        }
    }

    #[test]
    fn repetition_detector_detects_exact_ngram_repeat() {
        let mut det = RepetitionDetector::new(4, 3);
        // Feed 3 × [1,2,3,4] = 12 tokens; loop fires at the 12th push.
        let pattern = [1u32, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
        let last = pattern.len() - 1;
        for (i, &tok) in pattern.iter().enumerate() {
            let fired = det.push_and_check(tok);
            if i == last {
                assert!(fired, "expected loop detection on final token");
            } else {
                assert!(!fired, "false positive at position {i}");
            }
        }
    }

    #[test]
    fn repetition_detector_does_not_fire_on_partial_repeat() {
        let mut det = RepetitionDetector::new(4, 3);
        // Only 2 repetitions of [1,2,3,4] — not enough for max_repeats=3.
        for tok in [1u32, 2, 3, 4, 1, 2, 3, 4] {
            assert!(!det.push_and_check(tok));
        }
    }

    #[test]
    fn repetition_detector_does_not_fire_on_different_tail() {
        let mut det = RepetitionDetector::new(4, 3);
        // Two full repeats then a different final ngram.
        for tok in [1u32, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8] {
            assert!(!det.push_and_check(tok));
        }
    }
}
