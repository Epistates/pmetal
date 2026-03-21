//! Unified inference pipeline shared by CLI, GUI, and serve.
//!
//! `InferenceRunner` encapsulates the full pre-generation setup (model loading,
//! tokenization, chat template, sampling config, cache creation) so that all
//! consumers get identical behavior from a single code path.

use std::path::{Path, PathBuf};

use mlx_rs::error::Exception;
use pmetal_data::chat_templates::{ChatTemplateType, Message, ToolDefinition};
use pmetal_data::Tokenizer;
use pmetal_lora::{DynamicLoraModel, TrainableModel as _};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_models::generation::GenerationConfig;
use pmetal_models::{GenerationOutput, generate_cached_async_streaming};
use pmetal_mlx::kv_cache::{CacheMode, KVCache, MambaCache};

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
    /// Quantization bits for KV cache (8=q8_0, 4=q4_0, 0=fp16). Default: 8.
    pub kv_quant: u8,
    /// Override key bits (for asymmetric K/V quantization).
    pub kv_k_bits: Option<u8>,
    /// Override value bits (for asymmetric K/V quantization).
    pub kv_v_bits: Option<u8>,
    /// Quantization group size.
    pub kv_group_size: usize,
    /// Disable KV cache quantization entirely.
    pub no_kv_quant: bool,
}

impl Default for InferenceRunnerConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            lora_path: None,
            experts_dir: None,
            fp8: false,
            prompt: String::new(),
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
            kv_quant: 8,
            kv_k_bits: None,
            kv_v_bits: None,
            kv_group_size: 64,
            no_kv_quant: false,
        }
    }
}

/// Loaded model — either a standard model or a LoRA-merged model.
enum LoadedModel {
    Standard(DynamicModel),
    Lora(DynamicLoraModel),
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
        let is_instruct = is_instruction_tuned(model_path);
        let use_chat = config.chat || is_instruct || config.tools.is_some();
        let no_thinking = if !is_instruct && !config.no_thinking && use_chat {
            true // base models don't understand <think> tags
        } else {
            config.no_thinking
        };

        // 3. Load sampling defaults from model's generation_config.json
        let defaults = pmetal_data::inference_config::load_sampling_defaults(
            model_path,
            use_chat && !no_thinking,
        );
        let temperature = config.temperature.unwrap_or(defaults.temperature);
        let top_k = config.top_k.unwrap_or(defaults.top_k);
        let top_p = config.top_p.unwrap_or(defaults.top_p);
        let min_p = config.min_p.unwrap_or(defaults.min_p);
        let repetition_penalty = config.repetition_penalty.unwrap_or(defaults.repetition_penalty);
        let frequency_penalty = config.frequency_penalty.unwrap_or(defaults.frequency_penalty);
        let presence_penalty = config.presence_penalty.unwrap_or(defaults.presence_penalty);

        // 4. Apply chat template + tokenize
        let (input_ids, template_type) = if use_chat {
            let detected =
                pmetal_data::chat_templates::detect_chat_template(model_path, &model_path.to_string_lossy());

            let formatted = if config.tools.is_some() {
                // Tool-calling path: structured template with tool injection
                let mut msgs = Vec::new();
                let sys_content = match (config.system_message.as_deref(), no_thinking) {
                    (Some(sys), true) => Some(format!("{sys}\n/no_think")),
                    (Some(sys), false) => Some(sys.to_string()),
                    (None, true) => Some("/no_think".to_string()),
                    (None, false) => None,
                };
                if let Some(sys) = sys_content {
                    msgs.push(Message::system(sys));
                }
                msgs.push(Message::user(&config.prompt));
                detected
                    .apply_with_tools(&msgs, config.tools.as_deref())
                    .text
            } else {
                // Standard chat path
                let mut msgs = Vec::new();
                if let Some(ref sys) = config.system_message {
                    if !sys.is_empty() {
                        let sys_text = if no_thinking {
                            format!("{sys}\n/no_think")
                        } else {
                            sys.clone()
                        };
                        msgs.push(Message::system(sys_text));
                    }
                }
                msgs.push(Message::user(&config.prompt));
                detected.apply(&msgs).text
            };

            let ids = tokenizer
                .encode_with_special_tokens(&formatted)
                .map_err(|e| Exception::custom(e.to_string()))?;
            (ids, Some(detected.template_type))
        } else {
            let ids = tokenizer
                .encode(&config.prompt)
                .map_err(|e| Exception::custom(e.to_string()))?;
            (ids, None)
        };

        tracing::info!(tokens = input_ids.len(), "Prompt tokenized");

        // 5. Collect stop tokens from all sources
        let stop_tokens = pmetal_data::inference_config::collect_all_stop_tokens(
            model_path,
            &tokenizer,
            template_type,
        );

        // 6. Build GenerationConfig
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

        // 7. Load model (standard or LoRA-merged)
        let max_seq_len = input_ids.len() + config.max_tokens + 64;
        let cache_mode = resolve_cache_mode(
            config.kv_quant,
            config.kv_k_bits,
            config.kv_v_bits,
            config.kv_group_size,
            config.no_kv_quant,
        );
        tracing::info!(mode = %cache_mode.describe(), tokens = max_seq_len, "KV cache");

        let (model, cache, mamba_cache) = if let Some(ref lora_path) = config.lora_path {
            let (lora_model, cache, mamba_cache) =
                load_model_with_lora(model_path, lora_path, max_seq_len)?;
            (LoadedModel::Lora(lora_model), cache, mamba_cache)
        } else {
            let mut m = DynamicModel::load(model_path)?;
            tracing::info!(arch = %m.architecture(), "Model loaded");

            // 8. FP8 quantization (standard path only — LoRA models are already merged)
            if config.fp8 {
                tracing::info!("Quantizing weights to FP8 E4M3");
                m.quantize_fp8()?;
            }

            // 9. Expert offloading
            if let Some(ref experts_dir) = config.experts_dir {
                m.enable_expert_offloading(Path::new(experts_dir))?;
            }

            let cache = m.create_cache_with_mode(max_seq_len, cache_mode);
            let mamba_cache = m.create_mamba_cache();
            (LoadedModel::Standard(m), cache, mamba_cache)
        };

        Ok(Self {
            tokenizer,
            state: InferenceGenState {
                model,
                gen_config,
                input_ids,
                cache,
                mamba_cache,
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
        match self.model {
            LoadedModel::Standard(ref mut model) => {
                let mamba = &mut self.mamba_cache;
                generate_cached_async_streaming(
                    |input, cache| {
                        model.forward_with_hybrid_cache(input, None, Some(cache), mamba.as_mut())
                    },
                    &self.input_ids,
                    self.gen_config.clone(),
                    &mut self.cache,
                    on_token,
                )
            }
            LoadedModel::Lora(ref mut model) => {
                let mamba = &mut self.mamba_cache;
                generate_cached_async_streaming(
                    |input, cache| {
                        model
                            .forward_with_hybrid_cache(input, None, Some(cache), mamba.as_mut())
                            .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))
                    },
                    &self.input_ids,
                    self.gen_config.clone(),
                    &mut self.cache,
                    on_token,
                )
            }
        }
    }

    /// Run a generation function with properly split borrows.
    ///
    /// Destructures self to provide a forward closure + cache reference
    /// that don't conflict. The callback receives `(forward_fn, &mut cache)`
    /// where `forward_fn: impl FnMut(&Array, &mut KVCache) -> Result<Array>`.
    pub fn run_with<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(
            &mut dyn FnMut(&mlx_rs::Array, &mut KVCache) -> Result<mlx_rs::Array, Exception>,
            &mut KVCache,
        ) -> R,
    {
        let Self {
            ref mut model,
            ref mut cache,
            ref mut mamba_cache,
            ..
        } = *self;

        let mut fwd = |input: &mlx_rs::Array, kv: &mut KVCache| -> Result<mlx_rs::Array, Exception> {
            match model {
                LoadedModel::Standard(m) => {
                    m.forward_with_hybrid_cache(input, None, Some(kv), mamba_cache.as_mut())
                }
                LoadedModel::Lora(m) => m
                    .forward_with_hybrid_cache(input, None, Some(kv), mamba_cache.as_mut())
                    .map_err(|e| Exception::custom(e.to_string())),
            }
        };

        f(&mut fwd, cache)
    }

    /// Forward pass through the model (for non-streaming generation paths).
    ///
    /// Dispatches to the correct model type (standard or LoRA-merged) with
    /// hybrid cache support.
    pub fn forward(
        &mut self,
        input: &mlx_rs::Array,
        cache: &mut KVCache,
    ) -> Result<mlx_rs::Array, Exception> {
        match &mut self.model {
            LoadedModel::Standard(model) => {
                model.forward_with_hybrid_cache(input, None, Some(cache), self.mamba_cache.as_mut())
            }
            LoadedModel::Lora(model) => model
                .forward_with_hybrid_cache(input, None, Some(cache), self.mamba_cache.as_mut())
                .map_err(|e| Exception::custom(e.to_string())),
        }
    }

    /// Access the underlying DynamicModel (standard path only).
    pub fn dynamic_model(&self) -> Option<&DynamicModel> {
        match &self.model {
            LoadedModel::Standard(m) => Some(m),
            LoadedModel::Lora(_) => None,
        }
    }

    /// Mutable access to the DynamicModel.
    pub fn dynamic_model_mut(&mut self) -> Option<&mut DynamicModel> {
        match &mut self.model {
            LoadedModel::Standard(m) => Some(m),
            LoadedModel::Lora(_) => None,
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

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Load a model with LoRA weights merged in.
fn load_model_with_lora(
    model_path: &Path,
    lora_path: &str,
    max_seq_len: usize,
) -> Result<(DynamicLoraModel, KVCache, Option<MambaCache>), Exception> {
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

    tracing::info!(r = lora_config.r, alpha = lora_config.alpha, "Loading LoRA adapter");

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

    let cache = model
        .create_cache(max_seq_len)
        .ok_or_else(|| Exception::custom("model does not support KV cache"))?;
    let mamba_cache = model.create_mamba_cache();

    Ok((model, cache, mamba_cache))
}

/// Resolve KV cache quantization mode from CLI/GUI parameters.
fn resolve_cache_mode(
    kv_quant: u8,
    kv_k_bits: Option<u8>,
    kv_v_bits: Option<u8>,
    kv_group_size: usize,
    no_kv_quant: bool,
) -> CacheMode {
    if no_kv_quant || kv_quant == 0 {
        return CacheMode::Standard;
    }
    match (kv_k_bits, kv_v_bits) {
        (Some(k), Some(v)) => CacheMode::AsymmetricQuantized {
            key_bits: k,
            value_bits: v,
            group_size: kv_group_size,
        },
        (Some(k), None) | (None, Some(k)) => {
            let v = kv_v_bits.unwrap_or(kv_quant);
            let k_final = kv_k_bits.unwrap_or(kv_quant);
            let _ = k; // suppress unused
            if k_final == v {
                CacheMode::Quantized {
                    bits: k_final,
                    group_size: kv_group_size,
                }
            } else {
                CacheMode::AsymmetricQuantized {
                    key_bits: k_final,
                    value_bits: v,
                    group_size: kv_group_size,
                }
            }
        }
        (None, None) => CacheMode::Quantized {
            bits: kv_quant,
            group_size: kv_group_size,
        },
    }
}

/// Check if a model directory looks like an instruction-tuned model.
fn is_instruction_tuned(model_path: &Path) -> bool {
    let name = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Common instruction-tuning suffixes
    let instruct_markers = [
        "instruct",
        "chat",
        "it",
        "-sft",
        "-rlhf",
        "-dpo",
        "-grpo",
        "-rl",
    ];

    instruct_markers.iter().any(|m| name.contains(m))
}
