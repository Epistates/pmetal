//! Unified inference pipeline shared by CLI, GUI, and serve.
//!
//! `InferenceRunner` encapsulates the full pre-generation setup (model loading,
//! tokenization, chat template, sampling config, cache creation) so that all
//! consumers get identical behavior from a single code path.

use std::{
    collections::HashSet,
    ffi::OsStr,
    path::{Path, PathBuf},
};

use pmetal_bridge::turboquant::{
    TurboQuantConfig as BridgeTurboQuantConfig,
    TurboQuantTensorConfig as BridgeTurboQuantTensorConfig,
};
use pmetal_data::Tokenizer;
use pmetal_data::chat_templates::{ChatTemplateType, Message, ToolDefinition};
use pmetal_mlx::kv_cache::{
    CacheMode, KVCache, KVCacheConfig, MambaCache, TurboQuantConfig, TurboQuantTensorConfig,
    sanitize_cache_mode_for_config,
};
use pmetal_mlx::{Array, Dtype, Exception, ModuleParameters as _};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_models::generation::GenerationConfig;
use pmetal_models::{GenerationOutput, generate_cached_async_streaming};

#[cfg(feature = "lora")]
use pmetal_lora::{DynamicLoraModel, TrainableModel as _};

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
    /// Optional tool/function definitions for tool-calling models.
    pub tools: Option<Vec<ToolDefinition>>,

    // ── Sampling ─────────────────────────────────────────────────────────
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub max_tokens: usize,
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
            experts_dir: None,
            fp8: false,
            prompt: String::new(),
            chat_messages: None,
            system_message: None,
            chat: false,
            no_thinking: false,
            tools: None,
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            max_tokens: 256,
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

/// Loaded model — either a standard model or a LoRA-merged model.
#[allow(clippy::large_enum_variant)]
enum LoadedModel {
    Standard(DynamicModel),
    #[cfg(feature = "lora")]
    Lora(DynamicLoraModel),
    /// Native InlineArray path — no mlx-rs model needed.
    /// All weights loaded through pmetal-bridge's MLX.
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
    /// during streaming (borrow this before calling [`generate_streaming`]).
    pub tokenizer: Tokenizer,
    /// Generation state. Public for split-borrow access.
    pub state: InferenceGenState,
    chat_template_type: Option<ChatTemplateType>,
    is_chat: bool,
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
    native_turboquant: Option<BridgeTurboQuantConfig>,
    /// Zero-overhead affine KV cache quantization config for native path
    native_quant_config: Option<pmetal_bridge::qwen3_native::QuantCacheConfig>,
    /// Model directory path — used for native InlineArray weight loading
    /// (bypasses mlx-rs to avoid dual-MLX-instance 6x slowdown).
    model_path: PathBuf,
    /// Decode throughput metrics from the last native generation run.
    /// `None` on non-native paths or when fewer than 20 steps were measured.
    pub last_decode_metrics: Option<pmetal_bridge::decode::DecodeMetrics>,
    /// Enable n-gram repetition loop detection (opt-in).
    detect_repetition: bool,
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
    pub fn prepare(config: InferenceRunnerConfig) -> Result<Self, Exception> {
        let model_path = &config.model_path;

        // 1. Load tokenizer
        let tokenizer = Tokenizer::from_model_dir(model_path)
            .map_err(|e| Exception::custom(format!("tokenizer: {e}")))?;

        // 2. Determine chat mode (auto-detect instruction-tuned models)
        let is_instruct = model_looks_instruction_tuned(model_path);
        let use_chat = config.chat || is_instruct || config.tools.is_some();
        let no_thinking = if !is_instruct && !config.no_thinking && use_chat {
            true // base models don't understand <think> tags
        } else {
            config.no_thinking
        };

        // 3. Sampling defaults are loaded after template detection (step 6b)
        //    because mode presets depend on the detected model family.
        let native_bridge_info =
            if config.lora_path.is_none() && !config.fp8 && config.experts_dir.is_none() {
                crate::native_inference::load_native_bridge_info(model_path)
                    .map_err(Exception::custom)?
            } else {
                None
            };
        let native_bridge_candidate = native_bridge_info.is_some();

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
        let (input_ids, template_type) = if use_chat {
            let detected = pmetal_data::chat_templates::detect_chat_template(
                model_path,
                &model_path.to_string_lossy(),
            );

            let messages = build_chat_messages(
                config.chat_messages.as_ref(),
                config.system_message.as_deref(),
                &config.prompt,
            );

            let formatted = detected
                .apply_inference(&messages, no_thinking, config.tools.as_deref())
                .text;

            let ids = tokenizer
                .encode_with_special_tokens(&formatted)
                .map_err(|e| Exception::custom(e.to_string()))?;
            (ids, Some(detected.template_type))
        } else {
            let prompt_text = if config.chat_messages.is_some()
                || config
                    .system_message
                    .as_deref()
                    .is_some_and(|system| !system.trim().is_empty())
            {
                build_plain_conversation_prompt(
                    config.chat_messages.as_ref(),
                    config.system_message.as_deref(),
                    &config.prompt,
                )
            } else {
                config.prompt.clone()
            };
            let ids = tokenizer
                .encode(&prompt_text)
                .map_err(|e| Exception::custom(e.to_string()))?;
            (ids, None)
        };

        tracing::info!(tokens = input_ids.len(), "Prompt tokenized");

        // 6b. Load sampling defaults: global fallback → generation_config.json → mode preset
        let defaults = pmetal_data::inference_config::load_sampling_defaults(
            model_path,
            template_type,
            config.mode,
            use_chat && !no_thinking,
        );
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
            GenerationConfig::greedy(config.max_tokens).with_stop_tokens(stop_tokens)
        } else {
            let mut gc = GenerationConfig::sampling(config.max_tokens, temperature)
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
        let max_seq_len = input_ids.len() + config.max_tokens + 64;

        let cache_request = cache_mode_request_from_config(&config);
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
                    Some(pmetal_bridge::qwen3_native::QuantCacheConfig {
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
                            Some(pmetal_bridge::qwen3_native::QuantCacheConfig {
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

        Ok(Self {
            tokenizer,
            state: InferenceGenState {
                model,
                gen_config,
                input_ids,
                cache,
                mamba_cache,
                native_turboquant,
                native_quant_config,
                model_path: config.model_path.clone(),
                last_decode_metrics: None,
                detect_repetition: config.detect_repetition,
            },
            chat_template_type: template_type,
            is_chat: use_chat,
        })
    }

    /// Whether chat mode is active.
    pub fn is_chat(&self) -> bool {
        self.is_chat
    }

    /// The detected chat template type (if chat mode is active).
    pub fn chat_template_type(&self) -> Option<ChatTemplateType> {
        self.chat_template_type
    }

    /// Benchmark the active model using the same workload shape as
    /// `mlx_lm.benchmark`: fixed random prompt ids, one warmup, EOS disabled,
    /// and repeated full prompt+decode runs.
    pub fn benchmark_mlx_lm(
        &mut self,
        prompt_tokens: usize,
        generation_tokens: usize,
        num_trials: usize,
        seed: u64,
    ) -> Result<Vec<crate::native_inference::MlxLmBenchmarkTrial>, Exception> {
        let vocab_size = self.tokenizer.vocab_size();
        if vocab_size == 0 {
            return Err(Exception::custom("tokenizer vocabulary is empty"));
        }

        let prompt_ids = build_mlx_lm_benchmark_prompt(prompt_tokens, vocab_size, seed);
        match &mut self.state.model {
            LoadedModel::NativeOnly => crate::native_inference::benchmark_native_mlx_lm(
                &self.state.model_path,
                &prompt_ids,
                generation_tokens,
                self.state.native_turboquant,
                num_trials,
            )
            .map_err(Exception::custom),
            _ => Err(Exception::custom(
                "MLX-LM parity benchmark is currently only implemented for the native bridge path",
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

        if matches!(self.model, LoadedModel::NativeOnly) {
            let stop_tokens = self.gen_config.stop_tokens.clone();
            // Wire the full filter pipeline through to the bridge.
            // Previously top_k / top_p / min_p were silently dropped on
            // the native path: users passing `--top-k 20 --top-p 0.95`
            // got pure temperature sampling because SamplingParams
            // didn't carry those fields. Now the bridge applies the
            // same filter chain mlx-lm does.
            let sampling_params = pmetal_bridge::decode::SamplingParams {
                temperature: self.gen_config.temperature,
                top_k: self.gen_config.top_k,
                top_p: self.gen_config.top_p,
                min_p: self.gen_config.min_p,
                repetition_penalty: self.gen_config.repetition_penalty,
                frequency_penalty: self.gen_config.frequency_penalty,
                presence_penalty: self.gen_config.presence_penalty,
            };
            let output = crate::native_inference::run_native_inference_ext(
                &self.model_path,
                &self.input_ids,
                self.gen_config.max_new_tokens,
                sampling_params,
                self.native_turboquant,
                self.native_quant_config,
                |token| forward_native_token(&stop_tokens, &mut on_token_guarded, token),
            )
            .map_err(Exception::custom)?;

            self.last_decode_metrics = output.decode_metrics;
            return Ok(GenerationOutput {
                token_ids: output.token_ids,
                num_generated: output.num_generated,
                stopped_by_token: output.stopped_by_token,
                stopped_by_length: output.stopped_by_length,
                decode_metrics: output.decode_metrics,
            });
        }

        let output = match self.model {
            LoadedModel::Standard(ref mut model) => {
                let mamba = &mut self.mamba_cache;
                generate_cached_async_streaming(
                    |input, cache| {
                        model.forward_with_hybrid_cache(input, None, Some(cache), mamba.as_mut())
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
                            .forward_with_hybrid_cache(input, None, Some(cache), mamba.as_mut())
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

    /// Run a generation function with properly split borrows.
    ///
    /// Destructures self to provide a forward closure + cache reference
    /// that don't conflict. The callback receives `(forward_fn, &mut cache)`
    /// where `forward_fn: impl FnMut(&Array, &mut KVCache) -> Result<Array>`.
    pub fn run_with<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(
            &mut dyn FnMut(&Array, &mut KVCache) -> Result<Array, Exception>,
            &mut KVCache,
        ) -> R,
    {
        let Self {
            ref mut model,
            ref mut cache,
            ref mut mamba_cache,
            ..
        } = *self;

        let mut fwd = |input: &Array, kv: &mut KVCache| -> Result<Array, Exception> {
            match model {
                LoadedModel::Standard(m) => {
                    m.forward_with_hybrid_cache(input, None, Some(kv), mamba_cache.as_mut())
                }
                #[cfg(feature = "lora")]
                LoadedModel::Lora(m) => m
                    .forward_with_hybrid_cache(input, None, Some(kv), mamba_cache.as_mut())
                    .map_err(|e| Exception::custom(e.to_string())),
                LoadedModel::NativeOnly => Err(Exception::custom(
                    "run_with is not available in native mode",
                )),
            }
        };

        f(&mut fwd, cache)
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
        no_kv_quant: config.no_kv_quant,
        fp8: config.fp8,
    }
}

pub fn explicit_cache_mode_override(
    base_cache_config: &KVCacheConfig,
    request: CacheModeRequest,
) -> Option<CacheMode> {
    let explicit = request.no_kv_quant
        || request.kv_quant == Some(0)
        || request.kv_turboquant
        || request.kv_turboquant_preset.is_some()
        || request.kv_quant.is_some()
        || request.kv_k_bits.is_some()
        || request.kv_v_bits.is_some();

    if !explicit {
        return None;
    }

    let mode = if request.no_kv_quant || request.kv_quant == Some(0) {
        CacheMode::Standard
    } else {
        resolve_cache_mode(
            base_cache_config.head_dim,
            base_cache_config.value_head_dim,
            request,
        )
    };

    Some(sanitize_cache_mode_for_config(base_cache_config, mode))
}

fn build_mlx_lm_benchmark_prompt(prompt_tokens: usize, vocab_size: usize, seed: u64) -> Vec<u32> {
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
    info: crate::native_inference::NativeBridgeInfo,
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
    info: crate::native_inference::NativeBridgeInfo,
    mode: CacheMode,
) -> bool {
    match mode {
        CacheMode::Standard => true,
        CacheMode::TurboQuant { .. } => info.supports_turboquant,
        // Zero-overhead quantized KV cache via quantized_matmul (matches mlx-lm).
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
    info: crate::native_inference::NativeBridgeInfo,
) -> Option<pmetal_bridge::qwen3_native::MixedBitConfig> {
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
    Some(pmetal_bridge::qwen3_native::MixedBitConfig {
        outlier_count,
        outlier_bits,
        regular_bits,
    })
}

fn turboquant_config_from_mode(
    info: crate::native_inference::NativeBridgeInfo,
    mode: CacheMode,
) -> Option<BridgeTurboQuantConfig> {
    if !info.supports_turboquant {
        return None;
    }

    match mode {
        CacheMode::TurboQuant { config } => Some(BridgeTurboQuantConfig {
            keys: bridge_turboquant_tensor_config(config.keys),
            values: bridge_turboquant_tensor_config(config.values),
        }),
        _ => None,
    }
}

fn bridge_turboquant_tensor_config(config: TurboQuantTensorConfig) -> BridgeTurboQuantTensorConfig {
    match config {
        TurboQuantTensorConfig::Uniform { bits } => BridgeTurboQuantTensorConfig::uniform(bits),
        TurboQuantTensorConfig::Mixed {
            regular_bits,
            outlier_bits,
            outlier_count,
        } => BridgeTurboQuantTensorConfig::mixed(regular_bits, outlier_bits, outlier_count),
    }
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
    let lora_path_buf = Path::new(lora_path);
    let adapter_dir = if lora_path_buf.is_dir() {
        lora_path_buf
    } else {
        lora_path_buf.parent().unwrap_or(Path::new("."))
    };

    // Parse adapter_config.json for LoRA hyperparameters
    let lora_config =
        if let Ok(cfg_str) = std::fs::read_to_string(adapter_dir.join("adapter_config.json")) {
            if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&cfg_str) {
                let r = cfg["r"].as_u64().unwrap_or(16) as usize;
                let alpha = cfg["alpha"]
                    .as_f64()
                    .or_else(|| cfg["lora_alpha"].as_f64())
                    .unwrap_or(32.0) as f32;
                let target_modules: Vec<String> = cfg["target_modules"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let use_rslora = cfg["use_rslora"].as_bool().unwrap_or(false);
                pmetal_core::LoraConfig {
                    r,
                    alpha,
                    target_modules,
                    use_rslora,
                    ..pmetal_core::LoraConfig::default()
                }
            } else {
                pmetal_core::LoraConfig::default()
            }
        } else {
            pmetal_core::LoraConfig::default()
        };

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
    model
        .eval_all()
        .map_err(|e| Exception::custom(format!("LoRA eval: {e}")))?;

    tracing::info!("LoRA merged into base model");

    let base_cache = model
        .create_cache(max_seq_len)
        .ok_or_else(|| Exception::custom("model does not support KV cache"))?;
    let cache_selection = select_cache_mode_for_model(
        base_cache.config(),
        model_path,
        model.num_parameters(),
        cache_mode_request_from_config(config),
    );
    let cache = build_cache_from_base_config(base_cache.config(), cache_selection.mode);
    let mamba_cache = model.create_mamba_cache();

    Ok((model, cache, mamba_cache, cache_selection))
}

/// Resolve KV cache quantization mode from CLI/GUI parameters.
fn resolve_cache_mode(
    key_head_dim: usize,
    value_head_dim: usize,
    request: CacheModeRequest,
) -> CacheMode {
    if request.no_kv_quant || request.kv_quant == Some(0) {
        return CacheMode::Standard;
    }

    if let Some(preset) = request.kv_turboquant_preset {
        if request.kv_quant.is_some() || request.kv_k_bits.is_some() || request.kv_v_bits.is_some()
        {
            tracing::warn!(
                preset = preset.as_str(),
                "TurboQuant preset overrides explicit KV bit flags"
            );
        }
        return CacheMode::TurboQuant {
            config: preset.config(key_head_dim, value_head_dim),
        };
    }

    let kv_quant = request.kv_quant.unwrap_or(8);
    match (request.kv_k_bits, request.kv_v_bits) {
        (Some(k), Some(v)) => {
            if request.kv_turboquant {
                CacheMode::TurboQuant {
                    config: TurboQuantConfig::uniform(k, v),
                }
            } else {
                CacheMode::AsymmetricQuantized {
                    key_bits: k,
                    value_bits: v,
                    group_size: request.kv_group_size,
                }
            }
        }
        (Some(k), None) | (None, Some(k)) => {
            let v = request.kv_v_bits.unwrap_or(kv_quant);
            let k_final = request.kv_k_bits.unwrap_or(kv_quant);
            let _ = k; // suppress unused
            if request.kv_turboquant {
                CacheMode::TurboQuant {
                    config: TurboQuantConfig::uniform(k_final, v),
                }
            } else if k_final == v {
                CacheMode::Quantized {
                    bits: k_final,
                    group_size: request.kv_group_size,
                }
            } else {
                CacheMode::AsymmetricQuantized {
                    key_bits: k_final,
                    value_bits: v,
                    group_size: request.kv_group_size,
                }
            }
        }
        (None, None) => {
            if request.kv_turboquant {
                CacheMode::TurboQuant {
                    config: TurboQuantConfig::uniform(kv_quant, kv_quant),
                }
            } else {
                CacheMode::Quantized {
                    bits: kv_quant,
                    group_size: request.kv_group_size,
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheModeSource {
    ForcedFp16,
    Explicit,
    AutoFp16,
    AutoQ8,
}

impl CacheModeSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ForcedFp16 => "forced-fp16",
            Self::Explicit => "explicit",
            Self::AutoFp16 => "auto-fp16",
            Self::AutoQ8 => "auto-q8",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheModeSelection {
    pub mode: CacheMode,
    pub source: CacheModeSource,
    pub estimated_weight_bytes: u64,
    pub estimated_fp16_kv_bytes: u64,
    pub working_set_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboQuantPreset {
    Q2_5,
    Q3_5,
}

impl TurboQuantPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Q2_5 => "q2_5",
            Self::Q3_5 => "q3_5",
        }
    }

    fn config(self, key_head_dim: usize, value_head_dim: usize) -> TurboQuantConfig {
        let key_config = match self {
            Self::Q2_5 => TurboQuantConfig::preset_q2_5(key_head_dim).keys,
            Self::Q3_5 => TurboQuantConfig::preset_q3_5(key_head_dim).keys,
        };
        let value_config = match self {
            Self::Q2_5 => TurboQuantConfig::preset_q2_5(value_head_dim).values,
            Self::Q3_5 => TurboQuantConfig::preset_q3_5(value_head_dim).values,
        };
        TurboQuantConfig {
            keys: key_config,
            values: value_config,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheModeRequest {
    pub kv_quant: Option<u8>,
    pub kv_k_bits: Option<u8>,
    pub kv_v_bits: Option<u8>,
    pub kv_group_size: usize,
    pub kv_turboquant: bool,
    pub kv_turboquant_preset: Option<TurboQuantPreset>,
    pub no_kv_quant: bool,
    pub fp8: bool,
}

pub fn select_cache_mode_for_model(
    base_cache_config: &KVCacheConfig,
    model_path: &Path,
    param_count: usize,
    request: CacheModeRequest,
) -> CacheModeSelection {
    let working_set_bytes = pmetal_metal::context::MetalContext::global()
        .ok()
        .map(|ctx| ctx.properties().recommended_working_set_size);
    let estimated_weight_bytes = estimate_weight_bytes(model_path, param_count, request.fp8);

    select_cache_mode_with_working_set(
        base_cache_config,
        estimated_weight_bytes,
        request,
        working_set_bytes,
    )
}

fn select_cache_mode_with_working_set(
    base_cache_config: &KVCacheConfig,
    estimated_weight_bytes: u64,
    request: CacheModeRequest,
    working_set_bytes: Option<u64>,
) -> CacheModeSelection {
    let estimated_fp16_kv_bytes = estimate_fp16_kv_cache_bytes(base_cache_config);

    if request.no_kv_quant || request.kv_quant == Some(0) {
        return CacheModeSelection {
            mode: CacheMode::Standard,
            source: CacheModeSource::ForcedFp16,
            estimated_weight_bytes,
            estimated_fp16_kv_bytes,
            working_set_bytes,
        };
    }

    if let Some(mode) = explicit_cache_mode_override(base_cache_config, request) {
        return CacheModeSelection {
            mode,
            source: CacheModeSource::Explicit,
            estimated_weight_bytes,
            estimated_fp16_kv_bytes,
            working_set_bytes,
        };
    }

    let estimated_total_fp16 = estimated_weight_bytes.saturating_add(estimated_fp16_kv_bytes);
    let prefer_q8 = working_set_bytes.is_some_and(|working_set| {
        working_set > 0 && estimated_total_fp16 > ((working_set as f64) * 0.70) as u64
    });

    CacheModeSelection {
        mode: if prefer_q8 {
            CacheMode::Quantized {
                bits: 8,
                group_size: request.kv_group_size,
            }
        } else {
            CacheMode::Standard
        },
        source: if prefer_q8 {
            CacheModeSource::AutoQ8
        } else {
            CacheModeSource::AutoFp16
        },
        estimated_weight_bytes,
        estimated_fp16_kv_bytes,
        working_set_bytes,
    }
}

fn estimate_weight_bytes(model_path: &Path, param_count: usize, fp8: bool) -> u64 {
    let param_estimate = estimate_weight_bytes_from_param_count(param_count, fp8);
    estimate_local_model_weight_bytes(model_path, fp8)
        .map(|bytes| bytes.max(param_estimate))
        .unwrap_or(param_estimate)
}

fn estimate_weight_bytes_from_param_count(param_count: usize, fp8: bool) -> u64 {
    let bytes_per_param = if fp8 { 1.05 } else { 2.0 };
    (param_count as f64 * bytes_per_param) as u64
}

fn estimate_local_model_weight_bytes(model_path: &Path, fp8: bool) -> Option<u64> {
    let on_disk_bytes = sum_model_weight_file_bytes(model_path)?;
    Some(if fp8 {
        ((on_disk_bytes as f64) * (1.05 / 2.0)).ceil() as u64
    } else {
        on_disk_bytes
    })
}

fn sum_model_weight_file_bytes(model_path: &Path) -> Option<u64> {
    let mut total = 0u64;
    let mut visited_dirs = HashSet::new();
    let mut counted_files = HashSet::new();
    accumulate_model_weight_file_bytes(
        model_path,
        &mut visited_dirs,
        &mut counted_files,
        &mut total,
    );
    (total > 0).then_some(total)
}

fn accumulate_model_weight_file_bytes(
    path: &Path,
    visited_dirs: &mut HashSet<PathBuf>,
    counted_files: &mut HashSet<PathBuf>,
    total: &mut u64,
) {
    let Ok(metadata) = std::fs::metadata(path) else {
        return;
    };

    if metadata.is_dir() {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        if !visited_dirs.insert(canonical) {
            return;
        }

        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };

        for entry in entries.flatten() {
            accumulate_model_weight_file_bytes(&entry.path(), visited_dirs, counted_files, total);
        }
        return;
    }

    if !metadata.is_file() || !is_supported_model_weight_file(path) {
        return;
    }

    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    if counted_files.insert(canonical) {
        *total = total.saturating_add(metadata.len());
    }
}

fn is_supported_model_weight_file(path: &Path) -> bool {
    let extension = path
        .extension()
        .and_then(OsStr::to_str)
        .map(|ext| ext.to_ascii_lowercase());
    let file_name = path
        .file_name()
        .and_then(OsStr::to_str)
        .map(|name| name.to_ascii_lowercase())
        .unwrap_or_default();

    match extension.as_deref() {
        Some("safetensors") | Some("gguf") => true,
        Some("bin") | Some("pt") | Some("pth") => {
            file_name.contains("model")
                || file_name.contains("pytorch")
                || file_name.contains("consolidated")
        }
        _ => false,
    }
}

fn estimate_fp16_kv_cache_bytes(base_cache_config: &KVCacheConfig) -> u64 {
    base_cache_config
        .clone()
        .with_dtype(Dtype::Float16)
        .with_mode(CacheMode::Standard)
        .memory_footprint() as u64
}

fn log_cache_selection(selection: &CacheModeSelection, max_seq_len: usize) {
    let estimated_weight_gb = selection.estimated_weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let estimated_fp16_kv_gb =
        selection.estimated_fp16_kv_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let working_set_gb = selection
        .working_set_bytes
        .map(|bytes| bytes as f64 / (1024.0 * 1024.0 * 1024.0));

    tracing::info!(
        mode = %selection.mode.describe(),
        source = selection.source.as_str(),
        tokens = max_seq_len,
        estimated_weight_gb = format!("{estimated_weight_gb:.2}"),
        estimated_fp16_kv_gb = format!("{estimated_fp16_kv_gb:.2}"),
        working_set_gb = working_set_gb.map(|value| format!("{value:.2}")),
        "KV cache"
    );
}

fn build_cache_from_base_config(base_cache_config: &KVCacheConfig, mode: CacheMode) -> KVCache {
    let safe_mode = sanitize_cache_mode_for_config(base_cache_config, mode);
    KVCache::new(base_cache_config.clone().with_mode(safe_mode))
}

fn build_native_placeholder_cache(base_cache_config: &KVCacheConfig) -> KVCache {
    // Bridge-native inference owns its own KV/TurboQuant cache implementation.
    // Keep the runner's structural cache in the simplest mode so the native
    // path does not instantiate the shared pmetal-mlx TurboQuant stack.
    KVCache::new(base_cache_config.clone().with_mode(CacheMode::Standard))
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
    use crate::native_inference::{NativeArch, NativeBridgeInfo};
    use tempfile::tempdir;

    fn qwen3_cache_config(max_seq_len: usize) -> KVCacheConfig {
        KVCacheConfig::new(28, max_seq_len, 8, 128)
    }

    #[test]
    fn native_bridge_base_cache_config_preserves_asymmetric_value_dim() {
        let info = NativeBridgeInfo {
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
        let qwen = NativeBridgeInfo {
            arch: NativeArch::Qwen3_5,
            num_layers: 28,
            num_kv_heads: 2,
            head_dim: 128,
            value_head_dim: 128,
            supports_turboquant: true,
        };
        let llama4 = NativeBridgeInfo {
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
        assert!(!native_cache_mode_supported(
            llama4,
            CacheMode::TurboQuant {
                config: TurboQuantConfig::uniform(8, 8),
            }
        ));
    }

    #[test]
    fn auto_cache_mode_prefers_fp16_when_model_fits_comfortably() {
        let selection = select_cache_mode_with_working_set(
            &qwen3_cache_config(256),
            estimate_weight_bytes_from_param_count(620_000_000, false),
            CacheModeRequest {
                kv_quant: None,
                kv_k_bits: None,
                kv_v_bits: None,
                kv_group_size: 64,
                kv_turboquant: false,
                kv_turboquant_preset: None,
                no_kv_quant: false,
                fp8: false,
            },
            Some(48 * 1024 * 1024 * 1024),
        );

        assert_eq!(selection.mode, CacheMode::Standard);
        assert_eq!(selection.source, CacheModeSource::AutoFp16);
    }

    #[test]
    fn auto_cache_mode_promotes_to_q8_when_budget_is_tight() {
        let selection = select_cache_mode_with_working_set(
            &qwen3_cache_config(4096),
            estimate_weight_bytes_from_param_count(7_000_000_000, false),
            CacheModeRequest {
                kv_quant: None,
                kv_k_bits: None,
                kv_v_bits: None,
                kv_group_size: 64,
                kv_turboquant: false,
                kv_turboquant_preset: None,
                no_kv_quant: false,
                fp8: false,
            },
            Some(12 * 1024 * 1024 * 1024),
        );

        assert_eq!(
            selection.mode,
            CacheMode::Quantized {
                bits: 8,
                group_size: 64
            }
        );
        assert_eq!(selection.source, CacheModeSource::AutoQ8);
    }

    #[test]
    fn explicit_cache_mode_overrides_auto_selection() {
        let selection = select_cache_mode_with_working_set(
            &qwen3_cache_config(4096),
            estimate_weight_bytes_from_param_count(620_000_000, false),
            CacheModeRequest {
                kv_quant: Some(4),
                kv_k_bits: None,
                kv_v_bits: None,
                kv_group_size: 64,
                kv_turboquant: false,
                kv_turboquant_preset: None,
                no_kv_quant: false,
                fp8: false,
            },
            Some(12 * 1024 * 1024 * 1024),
        );

        assert_eq!(
            selection.mode,
            CacheMode::Quantized {
                bits: 4,
                group_size: 64
            }
        );
        assert_eq!(selection.source, CacheModeSource::Explicit);
    }

    #[test]
    fn explicit_turboquant_mode_uses_turboquant_variant() {
        let selection = select_cache_mode_with_working_set(
            &qwen3_cache_config(4096),
            estimate_weight_bytes_from_param_count(620_000_000, false),
            CacheModeRequest {
                kv_quant: Some(4),
                kv_k_bits: None,
                kv_v_bits: Some(3),
                kv_group_size: 64,
                kv_turboquant: true,
                kv_turboquant_preset: None,
                no_kv_quant: false,
                fp8: false,
            },
            Some(12 * 1024 * 1024 * 1024),
        );

        assert_eq!(
            selection.mode,
            CacheMode::TurboQuant {
                config: TurboQuantConfig::uniform(4, 3)
            }
        );
        assert_eq!(selection.source, CacheModeSource::Explicit);
    }

    #[test]
    fn turboquant_flag_without_bits_defaults_to_q8_turboquant() {
        let selection = select_cache_mode_with_working_set(
            &qwen3_cache_config(4096),
            estimate_weight_bytes_from_param_count(620_000_000, false),
            CacheModeRequest {
                kv_quant: None,
                kv_k_bits: None,
                kv_v_bits: None,
                kv_group_size: 64,
                kv_turboquant: true,
                kv_turboquant_preset: None,
                no_kv_quant: false,
                fp8: false,
            },
            Some(48 * 1024 * 1024 * 1024),
        );

        assert_eq!(
            selection.mode,
            CacheMode::TurboQuant {
                config: TurboQuantConfig::uniform(8, 8)
            }
        );
        assert_eq!(selection.source, CacheModeSource::Explicit);
    }

    #[test]
    fn turboquant_preset_uses_mixed_config() {
        let selection = select_cache_mode_with_working_set(
            &qwen3_cache_config(4096),
            estimate_weight_bytes_from_param_count(620_000_000, false),
            CacheModeRequest {
                kv_quant: None,
                kv_k_bits: None,
                kv_v_bits: None,
                kv_group_size: 64,
                kv_turboquant: false,
                kv_turboquant_preset: Some(TurboQuantPreset::Q2_5),
                no_kv_quant: false,
                fp8: false,
            },
            Some(48 * 1024 * 1024 * 1024),
        );

        assert_eq!(
            selection.mode,
            CacheMode::TurboQuant {
                config: TurboQuantConfig::preset_q2_5(128)
            }
        );
        assert_eq!(selection.source, CacheModeSource::Explicit);
    }

    #[test]
    fn explicit_cache_mode_override_uses_real_key_and_value_dims() {
        let base = KVCacheConfig::new(40, 4096, 2, 256)
            .with_value_head_dim(128)
            .with_dtype(Dtype::Float16);
        let mode = explicit_cache_mode_override(
            &base,
            CacheModeRequest {
                kv_quant: None,
                kv_k_bits: None,
                kv_v_bits: None,
                kv_group_size: 64,
                kv_turboquant: false,
                kv_turboquant_preset: Some(TurboQuantPreset::Q2_5),
                no_kv_quant: false,
                fp8: false,
            },
        )
        .expect("explicit turboquant preset should produce a cache override");

        let expected = TurboQuantConfig {
            keys: TurboQuantConfig::preset_q2_5(256).keys,
            values: TurboQuantConfig::preset_q2_5(128).values,
        };
        assert_eq!(mode, CacheMode::TurboQuant { config: expected });
    }

    #[test]
    fn no_kv_quant_forces_fp16_even_when_budget_is_tight() {
        let selection = select_cache_mode_with_working_set(
            &qwen3_cache_config(4096),
            estimate_weight_bytes_from_param_count(7_000_000_000, false),
            CacheModeRequest {
                kv_quant: None,
                kv_k_bits: None,
                kv_v_bits: None,
                kv_group_size: 64,
                kv_turboquant: false,
                kv_turboquant_preset: None,
                no_kv_quant: true,
                fp8: false,
            },
            Some(12 * 1024 * 1024 * 1024),
        );

        assert_eq!(selection.mode, CacheMode::Standard);
        assert_eq!(selection.source, CacheModeSource::ForcedFp16);
    }

    #[test]
    fn estimate_weight_bytes_prefers_real_model_files_when_param_count_is_missing() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("model.safetensors"), vec![0u8; 4096]).unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), vec![0u8; 8192]).unwrap();

        assert_eq!(estimate_weight_bytes(dir.path(), 0, false), 4096);
    }

    #[test]
    fn estimate_weight_bytes_scales_disk_estimate_for_fp8() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("model.safetensors"), vec![0u8; 4000]).unwrap();

        assert_eq!(estimate_weight_bytes(dir.path(), 0, true), 2100);
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
    fn mlx_lm_benchmark_prompt_is_deterministic() {
        let a = build_mlx_lm_benchmark_prompt(8, 1024, 0);
        let b = build_mlx_lm_benchmark_prompt(8, 1024, 0);
        let c = build_mlx_lm_benchmark_prompt(8, 1024, 1);

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
    fn estimate_weight_bytes_dedupes_symlinked_hf_snapshot_files() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().unwrap();
        let blobs = dir.path().join("blobs");
        let snapshot = dir.path().join("snapshots").join("1234");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snapshot).unwrap();

        let blob = blobs.join("weights");
        std::fs::write(&blob, vec![0u8; 3072]).unwrap();
        symlink(&blob, snapshot.join("model-00001-of-00002.safetensors")).unwrap();
        symlink(&blob, snapshot.join("model-duplicate.safetensors")).unwrap();
        std::fs::write(snapshot.join("config.json"), b"{}").unwrap();

        assert_eq!(estimate_weight_bytes(&snapshot, 0, false), 3072);
    }

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
