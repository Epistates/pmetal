//! Core inference engine that wraps model + tokenizer + generation.

use crate::error::{ServeError, ServeResult};
use crate::types::ChatMessage;
use pmetal_data::chat_templates::{ChatTemplate, ChatTemplateType, detect_chat_template};
use pmetal_data::inference_config::collect_all_stop_tokens;
use pmetal_mlx::kv_cache::{CacheMode, KVCache, KVCacheConfig, MambaCache};
use pmetal_mlx::{Array, Dtype, ModuleParameters as _};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_models::generation::{GenerationConfig, Sampler};
use pmetal_models::{
    GenerationOutput, generate_cached_ane_streaming, generate_cached_hybrid_cpu_streaming,
    is_ane_inference_compatible, is_hybrid_cpu_compatible,
};
use std::collections::HashSet;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ────────────────────────────────────────────────────────────────────────────
// Per-request sampling parameters
// ────────────────────────────────────────────────────────────────────────────

/// All sampling parameters for a single generation request.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub extra_stop_token_ids: Vec<u32>,
    pub stop_sequences: Vec<String>,
    /// When `Some(n)`, emit per-token log-probabilities alongside generated
    /// tokens. `n == 0` means chosen-token logprob only; `n > 0` also
    /// includes the top-`n` alternative logprobs. `None` (default) skips
    /// logprob computation entirely to keep the hot path unchanged.
    pub logprobs_top_n: Option<usize>,
}

/// Per-token logprob data returned from [`InferenceEngine::generate`] when
/// [`SamplingParams::logprobs_top_n`] is set.
///
/// The `token` field matches the corresponding entry in the returned tokens
/// vec at the same index. `top_logprobs` is sorted descending by logprob
/// and excludes the chosen token itself (OpenAI's wire convention).
#[derive(Debug, Clone)]
pub struct TokenLogprobEntry {
    pub token: u32,
    pub logprob: f32,
    pub top_logprobs: Vec<(u32, f32)>,
}

// ────────────────────────────────────────────────────────────────────────────
// Per-request metrics
// ────────────────────────────────────────────────────────────────────────────

/// Timing and throughput metrics for a single generation request.
#[derive(Debug, Clone)]
pub struct RequestMetrics {
    /// Time from request start to the first generated token (ms).
    pub first_token_latency_ms: f64,
    /// Total time from start to last token (ms).
    pub total_latency_ms: f64,
    /// Generated tokens per second (completion_tokens / total_latency).
    pub tokens_per_second: f64,
    /// Number of prompt tokens.
    pub prompt_tokens: usize,
    /// Number of completion tokens.
    pub completion_tokens: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Token event (sent through the mpsc channel during streaming)
// ────────────────────────────────────────────────────────────────────────────

/// A single event emitted during token-by-token streaming generation.
pub enum TokenEvent {
    /// A generated token. `logprob` is `Some` only when the request set
    /// `SamplingParams::logprobs_top_n`; otherwise it stays `None` so the
    /// hot path is unchanged for callers that don't care.
    Token {
        id: u32,
        logprob: Option<TokenLogprobEntry>,
    },
    /// Generation is complete — carries finish reason and final metrics.
    Done {
        finish_reason: String,
        metrics: RequestMetrics,
        stripped_tokens: usize,
    },
    /// Generation failed.
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreferredGenerationBackend {
    Gpu,
    Ane,
    CpuHybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServeCacheModeSource {
    AutoFp16,
    AutoQ8,
    Explicit,
}

impl ServeCacheModeSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::AutoFp16 => "auto-fp16",
            Self::AutoQ8 => "auto-q8",
            Self::Explicit => "explicit",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ServeCacheModeSelection {
    mode: CacheMode,
    source: ServeCacheModeSource,
    estimated_weight_bytes: u64,
    estimated_fp16_kv_bytes: u64,
    working_set_bytes: Option<u64>,
}

#[derive(Debug)]
struct BackendState {
    preferred: PreferredGenerationBackend,
}

fn build_generation_config_from_parts(
    stop_token_ids: &[u32],
    max_seq_len: usize,
    ane_real_time: bool,
    params: &SamplingParams,
) -> GenerationConfig {
    let temperature = params.temperature;
    let do_sample = temperature > 0.0;

    let max_tokens = params.max_tokens.min(max_seq_len);

    let mut stop_tokens = stop_token_ids.to_vec();
    stop_tokens.extend_from_slice(&params.extra_stop_token_ids);
    stop_tokens.sort_unstable();
    stop_tokens.dedup();

    let mut config = if do_sample {
        GenerationConfig {
            max_new_tokens: max_tokens,
            temperature,
            do_sample: true,
            stop_tokens,
            seed: params.seed,
            ane_real_time,
            ..GenerationConfig::default()
        }
    } else {
        GenerationConfig::greedy(max_tokens)
            .with_stop_tokens(stop_tokens)
            .with_ane_real_time(ane_real_time)
    };

    if let Some(top_k) = params.top_k {
        config = config.with_top_k(top_k);
    }
    if let Some(top_p) = params.top_p {
        config = config.with_top_p(top_p);
    }
    if let Some(min_p) = params.min_p {
        config = config.with_min_p(min_p);
    }
    if let Some(rp) = params.repetition_penalty {
        config = config.with_repetition_penalty(rp);
    }
    if let Some(fp) = params.frequency_penalty {
        config = config.with_frequency_penalty(fp);
    }
    if let Some(pp) = params.presence_penalty {
        config = config.with_presence_penalty(pp);
    }
    if !do_sample {
        if let Some(seed) = params.seed {
            config = config.with_seed(seed);
        }
    }

    config
}

fn detect_stop_sequence_suffix(
    tokenizer: &pmetal_data::Tokenizer,
    generated: &[u32],
    stop_sequences: &[String],
) -> Option<usize> {
    if generated.is_empty() || stop_sequences.is_empty() {
        return None;
    }

    let decoded = tokenizer.decode(generated).unwrap_or_default();
    let matched = stop_sequences
        .iter()
        .filter(|seq| !seq.is_empty() && decoded.ends_with(seq.as_str()))
        .max_by_key(|seq| seq.len())?;

    for strip_tokens in 1..=generated.len() {
        let suffix = tokenizer
            .decode(&generated[generated.len() - strip_tokens..])
            .unwrap_or_default();
        if suffix == *matched {
            return Some(strip_tokens);
        }
    }

    None
}

fn estimate_parameter_count(config_json: &serde_json::Value) -> Option<u64> {
    let hidden = config_json.get("hidden_size")?.as_u64()?;
    let layers = config_json.get("num_hidden_layers")?.as_u64()?;
    let vocab = config_json.get("vocab_size")?.as_u64()?;

    Some(
        12u64
            .saturating_mul(hidden)
            .saturating_mul(hidden)
            .saturating_mul(layers)
            .saturating_add(hidden.saturating_mul(vocab)),
    )
}

fn select_accelerated_backend(
    config_json: &serde_json::Value,
    ane_enabled: bool,
) -> PreferredGenerationBackend {
    if !ane_enabled {
        return PreferredGenerationBackend::Gpu;
    }

    let prefer_gpu_for_decode = estimate_parameter_count(config_json)
        .map(|params| params < 2_000_000_000)
        .unwrap_or(false);

    if !prefer_gpu_for_decode && is_ane_inference_compatible(config_json).is_ok() {
        return PreferredGenerationBackend::Ane;
    }

    if is_hybrid_cpu_compatible(config_json).is_ok() {
        return PreferredGenerationBackend::CpuHybrid;
    }

    PreferredGenerationBackend::Gpu
}

fn select_serve_cache_mode(
    model_path: &Path,
    param_count: usize,
    base_cache_config: &KVCacheConfig,
) -> ServeCacheModeSelection {
    let working_set_bytes = pmetal_metal::context::MetalContext::global()
        .ok()
        .map(|ctx| ctx.properties().recommended_working_set_size);
    let estimated_weight_bytes = estimate_serve_weight_bytes(
        model_path,
        estimate_weight_bytes_from_param_count(param_count),
    );

    select_serve_cache_mode_with_working_set(
        base_cache_config,
        estimated_weight_bytes,
        working_set_bytes,
    )
}

fn select_serve_cache_mode_with_working_set(
    base_cache_config: &KVCacheConfig,
    estimated_weight_bytes: u64,
    working_set_bytes: Option<u64>,
) -> ServeCacheModeSelection {
    let estimated_fp16_kv_bytes = estimate_fp16_kv_cache_bytes(base_cache_config);
    let estimated_total_fp16 = estimated_weight_bytes.saturating_add(estimated_fp16_kv_bytes);
    let prefer_q8 = working_set_bytes.is_some_and(|working_set| {
        working_set > 0 && estimated_total_fp16 > ((working_set as f64) * 0.70) as u64
    });

    ServeCacheModeSelection {
        mode: if prefer_q8 {
            CacheMode::Quantized {
                bits: 8,
                group_size: 64,
            }
        } else {
            CacheMode::Standard
        },
        source: if prefer_q8 {
            ServeCacheModeSource::AutoQ8
        } else {
            ServeCacheModeSource::AutoFp16
        },
        estimated_weight_bytes,
        estimated_fp16_kv_bytes,
        working_set_bytes,
    }
}

fn log_serve_cache_selection(selection: &ServeCacheModeSelection, max_seq_len: usize) {
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
        "serve KV cache"
    );
}

fn estimate_serve_weight_bytes(model_path: &Path, param_estimate: u64) -> u64 {
    estimate_local_model_weight_bytes(model_path)
        .map(|bytes| bytes.max(param_estimate))
        .unwrap_or(param_estimate)
}

fn estimate_weight_bytes_from_param_count(param_count: usize) -> u64 {
    (param_count as f64 * 2.0) as u64
}

fn estimate_local_model_weight_bytes(model_path: &Path) -> Option<u64> {
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

// ────────────────────────────────────────────────────────────────────────────
// Inference engine
// ────────────────────────────────────────────────────────────────────────────

/// The inference engine encapsulates model, tokenizer, and generation parameters.
pub struct InferenceEngine {
    /// The loaded model (behind a std Mutex — DynamicModel is !Send).
    model: Arc<Mutex<ModelState>>,
    /// The tokenizer.
    tokenizer: Arc<pmetal_data::Tokenizer>,
    /// Detected chat template.
    chat_template: ChatTemplate,
    /// Model name/ID for API responses.
    model_id: String,
    /// Resolved local model directory (used by ANE / CPU-hybrid backends).
    model_path: std::path::PathBuf,
    /// Maximum sequence length for KV cache.
    max_seq_len: usize,
    /// Fixed ANE bucket cap for accelerated backends.
    ane_max_seq_len: usize,
    /// Enable the experimental ANE real-time evaluation path for ANE requests.
    ane_real_time: bool,
    /// Preferred generation backend; falls back to GPU permanently on failure.
    backend: Arc<Mutex<BackendState>>,
    /// Stop token IDs collected from all available sources.
    stop_token_ids: Vec<u32>,
    /// Model creation timestamp.
    created_at: i64,
    /// Explicit cache mode override (bypasses auto-selection when set).
    cache_mode_override: Option<CacheMode>,
}

/// Model + cache state that must be accessed sequentially.
struct ModelState {
    model: DynamicModel,
}

// SAFETY: DynamicModel is !Send because it contains raw pointers from MLX's C FFI.
// We serialize all access through std::sync::Mutex, ensuring no concurrent access.
// The Mutex guard is never held across an await point.
#[allow(unsafe_code)]
unsafe impl Send for ModelState {}

impl InferenceEngine {
    fn create_request_caches(
        model: &DynamicModel,
        model_path: &Path,
        max_seq_len: usize,
        cache_mode_override: Option<CacheMode>,
    ) -> (KVCache, Option<MambaCache>) {
        let base_cache = model.create_cache(max_seq_len);
        let selection = if let Some(mode) = cache_mode_override {
            let estimated_weight_bytes = estimate_serve_weight_bytes(
                model_path,
                estimate_weight_bytes_from_param_count(model.num_parameters()),
            );
            ServeCacheModeSelection {
                mode,
                source: ServeCacheModeSource::Explicit,
                estimated_weight_bytes,
                estimated_fp16_kv_bytes: estimate_fp16_kv_cache_bytes(base_cache.config()),
                working_set_bytes: None,
            }
        } else {
            select_serve_cache_mode(model_path, model.num_parameters(), base_cache.config())
        };
        log_serve_cache_selection(&selection, max_seq_len);
        let cache = model.create_cache_with_mode(max_seq_len, selection.mode);
        let mamba_cache = model.create_mamba_cache();
        (cache, mamba_cache)
    }

    /// Create a new inference engine from a loaded model and tokenizer.
    pub fn new(
        model: DynamicModel,
        tokenizer: pmetal_data::Tokenizer,
        model_id: String,
        model_path: &std::path::Path,
        max_seq_len: usize,
    ) -> ServeResult<Self> {
        Self::new_with_backend(
            model,
            tokenizer,
            model_id,
            model_path,
            max_seq_len,
            true,
            1024,
            false,
        )
    }

    /// Create a new inference engine with explicit backend controls.
    pub fn new_with_backend(
        model: DynamicModel,
        tokenizer: pmetal_data::Tokenizer,
        model_id: String,
        model_path: &std::path::Path,
        max_seq_len: usize,
        ane_enabled: bool,
        ane_max_seq_len: usize,
        ane_real_time: bool,
    ) -> ServeResult<Self> {
        Self::new_with_options(
            model,
            tokenizer,
            model_id,
            model_path,
            max_seq_len,
            ane_enabled,
            ane_max_seq_len,
            ane_real_time,
            None,
        )
    }

    /// Create a new inference engine with explicit backend and cache mode controls.
    pub fn new_with_options(
        model: DynamicModel,
        tokenizer: pmetal_data::Tokenizer,
        model_id: String,
        model_path: &std::path::Path,
        max_seq_len: usize,
        ane_enabled: bool,
        ane_max_seq_len: usize,
        ane_real_time: bool,
        cache_mode_override: Option<CacheMode>,
    ) -> ServeResult<Self> {
        let chat_template = detect_chat_template(model_path, &model_id);

        // Collect stop tokens from all available sources using the canonical
        // `collect_all_stop_tokens` implementation from pmetal-data.
        // This merges generation_config.json EOS, chat-template EOS, tokenizer
        // EOS, and 11 well-known special token probes — deduplicated.
        let template_type: Option<ChatTemplateType> = Some(chat_template.template_type);
        let stop_token_ids = collect_all_stop_tokens(model_path, &tokenizer, template_type);

        tracing::info!(
            "Inference engine ready: model_id={}, stop_tokens={:?}",
            model_id,
            stop_token_ids
        );

        let preferred_backend = match std::fs::read_to_string(model_path.join("config.json")) {
            Ok(config_text) => match serde_json::from_str::<serde_json::Value>(&config_text) {
                Ok(config_json) => select_accelerated_backend(&config_json, ane_enabled),
                Err(err) => {
                    tracing::warn!(
                        model = %model_path.display(),
                        "Failed to parse config.json for backend selection: {}",
                        err
                    );
                    PreferredGenerationBackend::Gpu
                }
            },
            Err(err) => {
                tracing::warn!(
                    model = %model_path.display(),
                    "Failed to read config.json for backend selection: {}",
                    err
                );
                PreferredGenerationBackend::Gpu
            }
        };

        tracing::info!(
            model = %model_path.display(),
            backend = ?preferred_backend,
            ane_enabled,
            ane_max_seq_len,
            ane_real_time,
            "Selected serving generation backend"
        );

        let created_at = chrono::Utc::now().timestamp();

        Ok(Self {
            model: Arc::new(Mutex::new(ModelState { model })),
            tokenizer: Arc::new(tokenizer),
            chat_template,
            model_id,
            model_path: model_path.to_path_buf(),
            max_seq_len,
            ane_max_seq_len,
            ane_real_time,
            backend: Arc::new(Mutex::new(BackendState {
                preferred: preferred_backend,
            })),
            stop_token_ids,
            created_at,
            cache_mode_override,
        })
    }

    /// Model ID for API responses.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Creation timestamp.
    pub fn created_at(&self) -> i64 {
        self.created_at
    }

    /// Shared reference to the tokenizer.
    ///
    /// Returns a cloned `Arc` so that route handlers can hold onto the
    /// tokenizer independently of the engine reference, which is needed
    /// for decoding tokens inside async streaming closures.
    pub fn tokenizer_arc(&self) -> Arc<pmetal_data::Tokenizer> {
        Arc::clone(&self.tokenizer)
    }

    /// Format chat messages using the detected template.
    pub fn format_chat(&self, messages: &[ChatMessage]) -> String {
        self.format_chat_with_tools(messages, None)
    }

    /// Format chat messages with optional tool definitions. The chat template
    /// injects tool definitions into the system prompt using the model-specific
    /// format — Qwen, Llama 3.1+, Mistral v3+, and ChatML support this natively;
    /// other templates fall through to a generic ChatML-style injection.
    pub fn format_chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[pmetal_data::chat_templates::ToolDefinition]>,
    ) -> String {
        let msgs: Vec<pmetal_data::chat_templates::Message> = messages
            .iter()
            .map(|m| pmetal_data::chat_templates::Message {
                role: m.role.clone(),
                content: m.content.clone(),
                tool_calls: m.tool_calls.clone(),
                tool_call_id: None,
            })
            .collect();
        // apply_inference prefers the upstream Jinja template when present, so
        // tool definitions land in the exact shape the model was trained on.
        let formatted = self.chat_template.apply_inference(&msgs, false, tools);
        formatted.text
    }

    /// Tokenize a prompt string.
    pub fn tokenize(&self, text: &str) -> ServeResult<Vec<u32>> {
        self.tokenizer
            .encode(text)
            .map_err(|e| ServeError::Tokenizer(e.to_string()))
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> ServeResult<String> {
        self.tokenizer
            .decode(tokens)
            .map_err(|e| ServeError::Tokenizer(e.to_string()))
    }

    /// Validate sampling parameters, returning an error for any out-of-range value.
    ///
    /// Deliberately does not error on `max_tokens > max_seq_len` — the engine
    /// clamps silently, matching OpenAI behaviour.
    fn validate_params(params: &SamplingParams, _max_seq_len: usize) -> ServeResult<()> {
        if params.max_tokens == 0 {
            return Err(ServeError::BadRequest("max_tokens must be >= 1".into()));
        }
        if params.temperature < 0.0 || !params.temperature.is_finite() {
            return Err(ServeError::BadRequest(
                "temperature must be >= 0.0 and finite".into(),
            ));
        }
        if let Some(top_p) = params.top_p {
            if top_p <= 0.0 || top_p > 1.0 || !top_p.is_finite() {
                return Err(ServeError::BadRequest("top_p must be in (0.0, 1.0]".into()));
            }
        }
        if let Some(min_p) = params.min_p {
            if !(0.0..1.0).contains(&min_p) || !min_p.is_finite() {
                return Err(ServeError::BadRequest("min_p must be in [0.0, 1.0)".into()));
            }
        }
        if let Some(rp) = params.repetition_penalty {
            if rp <= 0.0 || !rp.is_finite() {
                return Err(ServeError::BadRequest(
                    "repetition_penalty must be > 0.0".into(),
                ));
            }
        }
        if let Some(fp) = params.frequency_penalty {
            if !fp.is_finite() {
                return Err(ServeError::BadRequest(
                    "frequency_penalty must be finite".into(),
                ));
            }
        }
        if let Some(pp) = params.presence_penalty {
            if !pp.is_finite() {
                return Err(ServeError::BadRequest(
                    "presence_penalty must be finite".into(),
                ));
            }
        }
        Ok(())
    }

    /// Build a `GenerationConfig` from API request sampling parameters.
    ///
    /// Temperature == 0.0 or unset maps to greedy decoding (`do_sample = false`).
    /// All stop tokens (engine-level + per-request) are merged into the config.
    /// `max_tokens` is silently clamped to `max_seq_len` (matches OpenAI behaviour).
    pub fn build_generation_config(&self, params: &SamplingParams) -> GenerationConfig {
        build_generation_config_from_parts(
            &self.stop_token_ids,
            self.max_seq_len,
            self.ane_real_time,
            params,
        )
    }

    fn backend_or_gpu(backend: &Arc<Mutex<BackendState>>) -> PreferredGenerationBackend {
        backend
            .lock()
            .map(|state| state.preferred)
            .unwrap_or(PreferredGenerationBackend::Gpu)
    }

    fn downgrade_backend(
        backend: &Arc<Mutex<BackendState>>,
        failed_backend: PreferredGenerationBackend,
    ) {
        if let Ok(mut state) = backend.lock() {
            if state.preferred == failed_backend {
                state.preferred = PreferredGenerationBackend::Gpu;
            }
        }
    }

    fn finish_reason(output: &GenerationOutput) -> String {
        if output.stopped_by_token {
            "stop".to_string()
        } else {
            "length".to_string()
        }
    }

    fn build_metrics(
        start: Instant,
        prompt_tokens: usize,
        completion_tokens: usize,
        first_token_time_ms: Option<f64>,
    ) -> RequestMetrics {
        let total_latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let tokens_per_second = if total_latency_ms > 0.0 {
            completion_tokens as f64 / (total_latency_ms / 1000.0)
        } else {
            0.0
        };

        RequestMetrics {
            first_token_latency_ms: first_token_time_ms.unwrap_or(total_latency_ms),
            total_latency_ms,
            tokens_per_second,
            prompt_tokens,
            completion_tokens,
        }
    }

    fn try_accelerated_generate_blocking(
        backend: &Arc<Mutex<BackendState>>,
        model_path: &std::path::Path,
        input_ids: &[u32],
        gen_config: &GenerationConfig,
        ane_max_seq_len: usize,
    ) -> ServeResult<Option<(Vec<u32>, String, RequestMetrics)>> {
        let preferred_backend = Self::backend_or_gpu(backend);
        if preferred_backend == PreferredGenerationBackend::Gpu {
            return Ok(None);
        }

        let prompt_tokens = input_ids.len();
        let start = Instant::now();
        let mut first_token_time_ms = None;

        let output = match preferred_backend {
            PreferredGenerationBackend::Ane => generate_cached_ane_streaming(
                model_path,
                input_ids,
                gen_config,
                ane_max_seq_len,
                |_| {
                    if first_token_time_ms.is_none() {
                        first_token_time_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
                    }
                    true
                },
            ),
            PreferredGenerationBackend::CpuHybrid => {
                generate_cached_hybrid_cpu_streaming(model_path, input_ids, gen_config, |_| {
                    if first_token_time_ms.is_none() {
                        first_token_time_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
                    }
                    true
                })
            }
            PreferredGenerationBackend::Gpu => unreachable!(),
        };

        match output {
            Ok(output) => {
                let generated = output.token_ids[prompt_tokens..].to_vec();
                let metrics =
                    Self::build_metrics(start, prompt_tokens, generated.len(), first_token_time_ms);
                Ok(Some((generated, Self::finish_reason(&output), metrics)))
            }
            Err(err) => {
                tracing::warn!(
                    backend = ?preferred_backend,
                    model = %model_path.display(),
                    "Accelerated serving backend failed ({}), falling back to GPU",
                    err
                );
                Self::downgrade_backend(backend, preferred_backend);
                Ok(None)
            }
        }
    }

    fn try_accelerated_streaming_blocking(
        backend: &Arc<Mutex<BackendState>>,
        model_path: &std::path::Path,
        input_ids: &[u32],
        gen_config: &GenerationConfig,
        ane_max_seq_len: usize,
        tx: &tokio::sync::mpsc::Sender<TokenEvent>,
    ) -> bool {
        let preferred_backend = Self::backend_or_gpu(backend);
        if preferred_backend == PreferredGenerationBackend::Gpu {
            return false;
        }

        let prompt_tokens = input_ids.len();
        let start = Instant::now();
        let mut first_token_time_ms = None;
        let mut completion_tokens = 0usize;
        let mut receiver_dropped = false;

        let output = match preferred_backend {
            PreferredGenerationBackend::Ane => generate_cached_ane_streaming(
                model_path,
                input_ids,
                gen_config,
                ane_max_seq_len,
                |token| {
                    if first_token_time_ms.is_none() {
                        first_token_time_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
                    }
                    completion_tokens += 1;
                    if tx
                        .blocking_send(TokenEvent::Token {
                            id: token,
                            logprob: None,
                        })
                        .is_err()
                    {
                        receiver_dropped = true;
                        return false;
                    }
                    true
                },
            ),
            PreferredGenerationBackend::CpuHybrid => {
                generate_cached_hybrid_cpu_streaming(model_path, input_ids, gen_config, |token| {
                    if first_token_time_ms.is_none() {
                        first_token_time_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
                    }
                    completion_tokens += 1;
                    if tx
                        .blocking_send(TokenEvent::Token {
                            id: token,
                            logprob: None,
                        })
                        .is_err()
                    {
                        receiver_dropped = true;
                        return false;
                    }
                    true
                })
            }
            PreferredGenerationBackend::Gpu => unreachable!(),
        };

        if receiver_dropped {
            return true;
        }

        match output {
            Ok(output) => {
                let metrics = Self::build_metrics(
                    start,
                    prompt_tokens,
                    completion_tokens,
                    first_token_time_ms,
                );
                let _ = tx.blocking_send(TokenEvent::Done {
                    finish_reason: Self::finish_reason(&output),
                    metrics,
                    stripped_tokens: 0,
                });
                true
            }
            Err(err) => {
                tracing::warn!(
                    backend = ?preferred_backend,
                    model = %model_path.display(),
                    "Accelerated serving backend failed ({}), falling back to GPU",
                    err
                );
                Self::downgrade_backend(backend, preferred_backend);
                false
            }
        }
    }

    /// Extract the last-position logits from a model output tensor.
    ///
    /// Model outputs have shape `[1, seq_len, vocab_size]` (after prefill) or
    /// `[1, 1, vocab_size]` (after decode steps). We extract the last position
    /// and flatten to a 1-D array of shape `[vocab_size]` suitable for
    /// `Sampler::sample`.
    fn extract_last_logits(logits: &Array) -> ServeResult<Array> {
        // Shape: [batch=1, seq_len, vocab_size]
        let last_idx = logits.dim(1) - 1;
        let vocab_size = logits.dim(2);
        // take_axis with a 1-element index array extracts position last_idx
        // along axis 1 → [1, 1, vocab_size].  reshape flattens to [vocab_size].
        let idx = Array::from_slice(&[last_idx], &[1]);
        let last = logits.take_axis(&idx, 1);
        Ok(last.reshape(&[vocab_size]))
    }

    /// Generate tokens from input IDs (non-streaming).
    ///
    /// Returns `(generated_tokens, logprobs, finish_reason, metrics)`.
    ///
    /// `logprobs` is `Some(vec_with_one_entry_per_token)` when the caller
    /// set `params.logprobs_top_n` to `Some(n)` — the accelerated ANE/CPU
    /// paths cannot collect logprobs, and they only understand token-ID stop
    /// conditions, so they are bypassed whenever logprobs or raw text stop
    /// sequences are requested.
    pub async fn generate(
        &self,
        input_ids: &[u32],
        params: SamplingParams,
    ) -> ServeResult<(
        Vec<u32>,
        Option<Vec<TokenLogprobEntry>>,
        String,
        RequestMetrics,
    )> {
        // Validate before dispatching to the blocking thread.
        Self::validate_params(&params, self.max_seq_len)?;

        let prompt_tokens = input_ids.len();
        let gen_config = self.build_generation_config(&params);
        let input_ids = input_ids.to_vec();
        let model_arc = Arc::clone(&self.model);
        let model_path = self.model_path.clone();
        let max_seq_len = self.max_seq_len;
        let ane_max_seq_len = self.ane_max_seq_len;
        let backend = Arc::clone(&self.backend);
        let cache_mode_override = self.cache_mode_override;
        let tokenizer = Arc::clone(&self.tokenizer);

        let logprobs_top_n = params.logprobs_top_n;
        let stop_sequences = params.stop_sequences.clone();

        // Generation is synchronous/blocking; run it on a dedicated blocking
        // thread so we don't stall the async executor.
        //
        // DynamicModel is !Send — ModelState wraps it with an unsafe Send impl
        // guarded by the Mutex. The Mutex is cloned (Arc) into the closure.
        let result = tokio::task::spawn_blocking(move || {
            // Accelerated ANE / CPU-hybrid paths can't collect logprobs (they
            // run outside the MLX logits pipeline). Skip them when logprobs
            // are requested so the standard GPU loop handles the request.
            if logprobs_top_n.is_none() && stop_sequences.is_empty() {
                if let Some(result) = Self::try_accelerated_generate_blocking(
                    &backend,
                    &model_path,
                    &input_ids,
                    &gen_config,
                    ane_max_seq_len,
                )? {
                    let (tokens, reason, metrics) = result;
                    return Ok((tokens, None, reason, metrics));
                }
            }

            let max_tokens = gen_config.max_new_tokens;
            let stop_tokens = gen_config.stop_tokens.clone();
            let mut state = model_arc.lock().map_err(|_| ServeError::Busy)?;
            let model = &mut state.model;
            let (mut cache, mut mamba_cache) =
                Self::create_request_caches(model, &model_path, max_seq_len, cache_mode_override);

            // Build input array [1, seq_len] for prefill.
            let i32_ids: Vec<i32> = input_ids.iter().map(|&t| t as i32).collect();
            let seq_len = input_ids.len() as i32;
            let input_arr = Array::from_slice(&i32_ids, &[1, seq_len]);

            let start = Instant::now();

            // Prefill forward pass — produces logits for the first sample step.
            let mut logits = model
                .forward_with_hybrid_cache(&input_arr, None, Some(&mut cache), mamba_cache.as_mut())
                .map_err(ServeError::Model)?;
            // Blocked on mlx_rs::eval_async — would allow GPU pipeline overlap
            // between eval and the next decode step for higher throughput.
            logits.eval();

            // Sampler must be created inside spawn_blocking — it holds
            // MLX Arrays and is !Send.
            let mut sampler = Sampler::new(gen_config);

            let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
            let mut finish_reason = "length".to_string();
            let mut first_token_time: Option<f64> = None;
            // Track all tokens seen (prompt + generated) for repetition penalty.
            let mut all_tokens: Vec<u32> = input_ids.clone();
            // Pre-allocate the logprobs vector when requested so we avoid
            // re-sizing during generation. `None` when the caller didn't opt in.
            let mut logprobs_out: Option<Vec<TokenLogprobEntry>> =
                logprobs_top_n.map(|_| Vec::with_capacity(max_tokens));

            for i in 0..max_tokens {
                // Sample from current logits (prefill logits on i=0, decode logits thereafter).
                let last_logits = Self::extract_last_logits(&logits)?;
                let next_token = sampler
                    .sample(&last_logits, &all_tokens)
                    .map_err(ServeError::Model)?;

                // Record TTFT on first sampled token.
                if first_token_time.is_none() {
                    first_token_time = Some(start.elapsed().as_secs_f64() * 1000.0);
                }

                // Check stop condition before accepting the token.
                if stop_tokens.contains(&next_token) {
                    finish_reason = "stop".to_string();
                    break;
                }

                // Collect logprobs *before* consuming logits for the next
                // decode step. Only fires when caller opted in.
                if let (Some(top_n), Some(out)) = (logprobs_top_n, logprobs_out.as_mut()) {
                    let (logprob, mut top) = pmetal_models::generation::token_logprobs(
                        &last_logits,
                        next_token,
                        top_n + 1, // +1 so we can strip the chosen token if present
                    )
                    .map_err(ServeError::Model)?;
                    // OpenAI convention: top_logprobs excludes the chosen token itself.
                    top.retain(|(tok, _)| *tok != next_token);
                    top.truncate(top_n);
                    out.push(TokenLogprobEntry {
                        token: next_token,
                        logprob,
                        top_logprobs: top,
                    });
                }

                generated.push(next_token);
                all_tokens.push(next_token);

                if let Some(stripped_tokens) =
                    detect_stop_sequence_suffix(tokenizer.as_ref(), &generated, &stop_sequences)
                {
                    let visible_len = generated.len().saturating_sub(stripped_tokens);
                    generated.truncate(visible_len);
                    all_tokens.truncate(input_ids.len() + visible_len);
                    if let Some(out) = logprobs_out.as_mut() {
                        out.truncate(visible_len);
                    }
                    finish_reason = "stop".to_string();
                    break;
                }

                // Only run a decode forward pass when there are more iterations.
                // This avoids the wasted forward pass after the last token.
                if i + 1 < max_tokens {
                    let next_input = Array::from_slice(&[next_token as i32], &[1, 1]);
                    logits = model
                        .forward_with_hybrid_cache(
                            &next_input,
                            None,
                            Some(&mut cache),
                            mamba_cache.as_mut(),
                        )
                        .map_err(ServeError::Model)?;
                    // Sync eval required — lazy graph grows unbounded without it.
                    // Blocked on mlx_rs::eval_async for pipeline overlap.
                    logits.eval();
                }
            }

            let completion_tokens = generated.len();
            let metrics =
                Self::build_metrics(start, prompt_tokens, completion_tokens, first_token_time);

            Ok::<_, ServeError>((generated, logprobs_out, finish_reason, metrics))
        })
        .await
        .map_err(|e| ServeError::Internal(e.to_string()))??;

        Ok(result)
    }

    /// Compute pooled sentence embeddings for a batch of texts.
    ///
    /// Tokenises each input, forwards through the model's pre-lm-head trunk
    /// via [`DynamicModel::forward_hidden`], and applies the requested
    /// pooling strategy. Inputs are padded to the batch max length with a
    /// right-padding attention mask so the pooler ignores padding positions.
    ///
    /// # Errors
    ///
    /// Returns `ServeError::Model` when the architecture doesn't support
    /// pre-lm-head hidden states — see `DynamicModel::forward_hidden` for
    /// the supported set.
    pub async fn embed(
        &self,
        inputs: &[String],
        mode: pmetal_models::pooling::PoolingMode,
    ) -> ServeResult<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenise every input on the async side (pure CPU, no MLX state).
        let tokenized: Vec<Vec<u32>> = inputs
            .iter()
            .map(|s| self.tokenize(s))
            .collect::<ServeResult<_>>()?;
        let batch = tokenized.len();
        let seq_max = tokenized.iter().map(Vec::len).max().unwrap_or(0).max(1);

        // Build padded [batch, seq_max] token + mask arrays up front — the
        // blocking closure receives them by value so !Send DynamicModel never
        // crosses an await point.
        let mut ids_flat: Vec<i32> = vec![0; batch * seq_max];
        let mut mask_flat: Vec<f32> = vec![0.0; batch * seq_max];
        for (b, row) in tokenized.iter().enumerate() {
            for (j, &tok) in row.iter().enumerate() {
                ids_flat[b * seq_max + j] = tok as i32;
                mask_flat[b * seq_max + j] = 1.0;
            }
        }

        let model_arc = Arc::clone(&self.model);
        tokio::task::spawn_blocking(move || -> ServeResult<Vec<Vec<f32>>> {
            let mut state = model_arc.lock().map_err(|_| ServeError::Busy)?;
            let model = &mut state.model;

            let ids = Array::from_slice(&ids_flat, &[batch as i32, seq_max as i32]);
            let mask = Array::from_slice(&mask_flat, &[batch as i32, seq_max as i32]);

            let hidden = model
                .forward_hidden(&ids, None)
                .map_err(ServeError::Model)?;
            let pooled =
                pmetal_models::pooling::pool(&hidden, &mask, mode).map_err(ServeError::Model)?;
            let pooled_eval = pooled;
            pooled_eval.eval();

            let hidden_dim = pooled_eval.dim(1) as usize;
            let flat: Vec<f32> = pooled_eval.as_slice::<f32>().to_vec();
            let out = (0..batch)
                .map(|b| flat[b * hidden_dim..(b + 1) * hidden_dim].to_vec())
                .collect();
            Ok(out)
        })
        .await
        .map_err(|e| ServeError::Model(pmetal_bridge::compat::Exception::custom(e.to_string())))?
    }

    /// Begin token-by-token streaming generation.
    ///
    /// Validates `params` before spawning. If validation fails, sends a single
    /// `TokenEvent::Error` through the channel and returns immediately.
    ///
    /// Spawns a blocking thread that runs the generation loop and sends
    /// `TokenEvent` values through an `mpsc` channel. Returns the receiver
    /// end immediately so the route handler can start consuming events while
    /// generation proceeds in parallel.
    ///
    /// The channel will receive:
    /// - Zero or more `TokenEvent::Token(id)` — one per generated token.
    /// - Exactly one `TokenEvent::Done { .. }` on success.
    /// - Exactly one `TokenEvent::Error(msg)` if generation fails (no [DONE]).
    pub fn generate_streaming(
        &self,
        input_ids: &[u32],
        params: SamplingParams,
    ) -> tokio::sync::mpsc::Receiver<TokenEvent> {
        // Channel capacity: keep a small buffer so the generation thread is
        // never stalled waiting for the HTTP layer to consume events, but
        // don't allocate an unbounded queue.
        let (tx, rx) = tokio::sync::mpsc::channel::<TokenEvent>(64);

        // Validate before spawning — send error through channel if invalid.
        if let Err(e) = Self::validate_params(&params, self.max_seq_len) {
            let _ = tx.try_send(TokenEvent::Error(e.to_string()));
            return rx;
        }

        let prompt_tokens = input_ids.len();
        let gen_config = self.build_generation_config(&params);
        let input_ids = input_ids.to_vec();
        let model_arc = Arc::clone(&self.model);
        let model_path = self.model_path.clone();
        let max_seq_len = self.max_seq_len;
        let ane_max_seq_len = self.ane_max_seq_len;
        let backend = Arc::clone(&self.backend);
        let cache_mode_override = self.cache_mode_override;
        let tokenizer = Arc::clone(&self.tokenizer);
        // Captured per-token logprobs flag for the GPU streaming loop. The
        // accelerated paths cannot collect logprobs, so they always emit
        // TokenEvent::Token { logprob: None } regardless of this value.
        let logprobs_top_n = params.logprobs_top_n;
        let stop_sequences = params.stop_sequences.clone();

        // Spawn generation on a dedicated blocking thread.
        tokio::task::spawn_blocking(move || {
            // Macro-style helper: send an event or bail on channel close.
            macro_rules! send {
                ($event:expr) => {
                    if tx.blocking_send($event).is_err() {
                        // Receiver dropped (client disconnected) — stop generation.
                        return;
                    }
                };
            }

            if stop_sequences.is_empty()
                && Self::try_accelerated_streaming_blocking(
                    &backend,
                    &model_path,
                    &input_ids,
                    &gen_config,
                    ane_max_seq_len,
                    &tx,
                )
            {
                return;
            }

            let max_tokens = gen_config.max_new_tokens;
            let stop_tokens = gen_config.stop_tokens.clone();

            let state_guard = match model_arc.lock() {
                Ok(g) => g,
                Err(_) => {
                    send!(TokenEvent::Error("engine busy".into()));
                    return;
                }
            };
            // Shadow to get mutable access — we need to hold the guard for
            // the entire generation loop.
            let mut state = state_guard;
            let model = &mut state.model;
            let (mut cache, mut mamba_cache) =
                Self::create_request_caches(model, &model_path, max_seq_len, cache_mode_override);

            // Build prefill input array.
            let i32_ids: Vec<i32> = input_ids.iter().map(|&t| t as i32).collect();
            let seq_len = input_ids.len() as i32;
            let input_arr = Array::from_slice(&i32_ids, &[1, seq_len]);

            let start = Instant::now();

            // Prefill forward pass — produces logits for the first sample step.
            let mut logits = match model.forward_with_hybrid_cache(
                &input_arr,
                None,
                Some(&mut cache),
                mamba_cache.as_mut(),
            ) {
                Ok(l) => l,
                Err(e) => {
                    send!(TokenEvent::Error(e.to_string()));
                    return;
                }
            };
            // Sync eval required — lazy graph grows unbounded without it.
            // Blocked on mlx_rs::eval_async for pipeline overlap.
            logits.eval();

            // Sampler created inside spawn_blocking — it holds MLX Arrays.
            let mut sampler = Sampler::new(gen_config);

            let mut completion_tokens = 0usize;
            let mut finish_reason = "length".to_string();
            let mut first_token_time: Option<f64> = None;
            let mut all_tokens: Vec<u32> = input_ids.clone();
            let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
            let mut stripped_tokens = 0usize;

            for i in 0..max_tokens {
                // Sample from current logits (prefill logits on i=0, decode logits thereafter).
                let last_logits = match Self::extract_last_logits(&logits) {
                    Ok(l) => l,
                    Err(e) => {
                        send!(TokenEvent::Error(e.to_string()));
                        return;
                    }
                };

                let next_token = match sampler.sample(&last_logits, &all_tokens) {
                    Ok(t) => t,
                    Err(e) => {
                        send!(TokenEvent::Error(e.to_string()));
                        return;
                    }
                };

                // Record TTFT on first sampled token.
                if first_token_time.is_none() {
                    first_token_time = Some(start.elapsed().as_secs_f64() * 1000.0);
                }

                // Check stop condition before emitting the token.
                if stop_tokens.contains(&next_token) {
                    finish_reason = "stop".to_string();
                    break;
                }

                // Compute logprobs *before* the next forward pass — last_logits
                // is still in scope and we have the chosen token id. Cheap when
                // top_n is small; bypassed entirely when caller didn't opt in.
                let logprob_entry = match logprobs_top_n {
                    Some(top_n) => {
                        match pmetal_models::generation::token_logprobs(
                            &last_logits,
                            next_token,
                            top_n + 1,
                        ) {
                            Ok((lp, mut top)) => {
                                top.retain(|(tok, _)| *tok != next_token);
                                top.truncate(top_n);
                                Some(TokenLogprobEntry {
                                    token: next_token,
                                    logprob: lp,
                                    top_logprobs: top,
                                })
                            }
                            Err(_) => None,
                        }
                    }
                    None => None,
                };

                // Emit token before running the next forward pass so the
                // route handler can begin decoding and sending it to the
                // client while the GPU works on the next token.
                send!(TokenEvent::Token {
                    id: next_token,
                    logprob: logprob_entry,
                });
                completion_tokens += 1;
                all_tokens.push(next_token);
                generated.push(next_token);

                if let Some(matched_tokens) =
                    detect_stop_sequence_suffix(tokenizer.as_ref(), &generated, &stop_sequences)
                {
                    stripped_tokens = matched_tokens;
                    completion_tokens = completion_tokens.saturating_sub(matched_tokens);
                    finish_reason = "stop".to_string();
                    break;
                }

                // Only run a decode forward pass when there are more iterations.
                // This avoids the wasted forward pass after the last token.
                if i + 1 < max_tokens {
                    let next_input = Array::from_slice(&[next_token as i32], &[1, 1]);
                    logits = match model.forward_with_hybrid_cache(
                        &next_input,
                        None,
                        Some(&mut cache),
                        mamba_cache.as_mut(),
                    ) {
                        Ok(l) => l,
                        Err(e) => {
                            send!(TokenEvent::Error(e.to_string()));
                            return;
                        }
                    };
                    // Sync eval required — prevents unbounded lazy graph growth.
                    // Blocked on mlx_rs::eval_async for pipeline overlap.
                    logits.eval();
                }
            }

            let metrics =
                Self::build_metrics(start, prompt_tokens, completion_tokens, first_token_time);

            // Done — send final event (ignore send error, client may be gone).
            let _ = tx.blocking_send(TokenEvent::Done {
                finish_reason,
                metrics,
                stripped_tokens,
            });
        });

        rx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_models::architectures::nemotron_h::{NemotronHConfig, NemotronHForCausalLM};
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    fn dense_config(hidden_size: u64, num_layers: u64, vocab_size: u64) -> serde_json::Value {
        serde_json::json!({
            "model_type": "llama",
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "vocab_size": vocab_size,
            "num_experts": 0,
            "num_local_experts": 0
        })
    }

    fn qwen3_cache_config(max_seq_len: usize) -> KVCacheConfig {
        KVCacheConfig::new(28, max_seq_len, 8, 128)
    }

    fn tiny_nemotron_h_config() -> NemotronHConfig {
        NemotronHConfig {
            model_type: "nemotron_h".to_string(),
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 4,
            max_position_embeddings: 512,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            attention_bias: false,
            head_dim: Some(32),
            mamba_num_heads: 4,
            mamba_head_dim: 32,
            mamba_proj_bias: false,
            ssm_state_size: 16,
            conv_kernel: 4,
            n_groups: 2,
            time_step_limit: (0.0, f32::INFINITY),
            time_step_min: None,
            time_step_max: None,
            mlp_bias: false,
            mlp_hidden_act: "relu2".to_string(),
            layer_norm_epsilon: 1e-5,
            use_bias: false,
            use_conv_bias: true,
            tie_word_embeddings: true,
            hybrid_override_pattern: Some("M*-E".to_string()),
            moe_intermediate_size: Some(64),
            moe_shared_expert_intermediate_size: None,
            n_group: None,
            n_routed_experts: Some(2),
            n_shared_experts: None,
            topk_group: None,
            num_experts_per_tok: Some(1),
            norm_topk_prob: None,
            routed_scaling_factor: None,
            rope_theta: 10000.0,
        }
    }

    #[test]
    fn test_estimate_parameter_count() {
        let config = dense_config(1024, 24, 32768);
        let estimated = estimate_parameter_count(&config).unwrap();
        assert_eq!(estimated, 12 * 1024 * 1024 * 24 + 1024 * 32768);
    }

    #[test]
    fn test_select_accelerated_backend_prefers_gpu_for_small_dense_model() {
        let config = dense_config(1024, 24, 32768);
        assert_eq!(
            select_accelerated_backend(&config, true),
            PreferredGenerationBackend::Gpu
        );
    }

    #[test]
    fn test_select_accelerated_backend_prefers_ane_for_large_dense_model() {
        let config = dense_config(8192, 80, 128_256);
        assert_eq!(
            select_accelerated_backend(&config, true),
            PreferredGenerationBackend::Ane
        );
    }

    #[test]
    fn test_select_accelerated_backend_prefers_cpu_hybrid_for_qwen3_next() {
        let config = serde_json::json!({
            "model_type": "qwen3_next",
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "vocab_size": 151936,
            "num_experts": 0,
            "num_local_experts": 0
        });
        assert_eq!(
            select_accelerated_backend(&config, true),
            PreferredGenerationBackend::CpuHybrid
        );
    }

    #[test]
    fn test_select_accelerated_backend_honors_no_ane() {
        let config = dense_config(8192, 80, 128_256);
        assert_eq!(
            select_accelerated_backend(&config, false),
            PreferredGenerationBackend::Gpu
        );
    }

    #[test]
    fn test_build_generation_config_propagates_ane_real_time() {
        let params = SamplingParams {
            max_tokens: 64,
            temperature: 0.8,
            top_k: None,
            top_p: None,
            min_p: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: Some(7),
            extra_stop_token_ids: vec![99],
            stop_sequences: vec![],
            logprobs_top_n: None,
        };

        let config = build_generation_config_from_parts(&[1, 2], 32, true, &params);

        assert_eq!(config.max_new_tokens, 32);
        assert!(config.do_sample);
        assert!(config.ane_real_time);
        assert_eq!(config.seed, Some(7));
        assert_eq!(config.stop_tokens, vec![1, 2, 99]);
    }

    #[test]
    fn serve_auto_cache_prefers_fp16_when_model_fits_comfortably() {
        let selection = select_serve_cache_mode_with_working_set(
            &qwen3_cache_config(256),
            1_240_000_000,
            Some(48 * 1024 * 1024 * 1024),
        );

        assert_eq!(selection.mode, CacheMode::Standard);
        assert_eq!(selection.source, ServeCacheModeSource::AutoFp16);
    }

    #[test]
    fn serve_auto_cache_prefers_q8_when_budget_is_tight() {
        let selection = select_serve_cache_mode_with_working_set(
            &qwen3_cache_config(8192),
            14_000_000_000,
            Some(18 * 1024 * 1024 * 1024),
        );

        assert_eq!(
            selection.mode,
            CacheMode::Quantized {
                bits: 8,
                group_size: 64,
            }
        );
        assert_eq!(selection.source, ServeCacheModeSource::AutoQ8);
    }

    #[test]
    fn create_request_caches_allocates_mamba_cache_for_hybrid_models() {
        let model =
            DynamicModel::NemotronH(NemotronHForCausalLM::new(tiny_nemotron_h_config()).unwrap());
        let (cache, mamba_cache) = InferenceEngine::create_request_caches(
            &model,
            std::env::temp_dir().as_path(),
            64,
            None,
        );

        assert_eq!(cache.config().max_seq_len, 64);
        assert!(mamba_cache.is_some());
    }

    fn test_tokenizer() -> pmetal_data::Tokenizer {
        let model = WordLevel::builder()
            .vocab(
                [
                    ("<unk>".to_string(), 0),
                    ("alpha".to_string(), 1),
                    ("beta".to_string(), 2),
                    ("gamma".to_string(), 3),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("<unk>".to_string())
            .build()
            .expect("wordlevel");
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        let json = tokenizer.to_string(false).expect("serialize tokenizer");
        pmetal_data::Tokenizer::from_bytes(json.as_bytes()).expect("wrapper tokenizer")
    }

    #[test]
    fn detect_stop_sequence_suffix_matches_multi_token_tail() {
        let tokenizer = test_tokenizer();
        let generated = vec![1, 2, 3];

        let stripped =
            detect_stop_sequence_suffix(&tokenizer, &generated, &["beta gamma".to_string()]);

        assert_eq!(stripped, Some(2));
    }

    #[test]
    fn detect_stop_sequence_suffix_prefers_longest_match() {
        let tokenizer = test_tokenizer();
        let generated = vec![1, 2, 3];

        let stripped = detect_stop_sequence_suffix(
            &tokenizer,
            &generated,
            &["gamma".to_string(), "beta gamma".to_string()],
        );

        assert_eq!(stripped, Some(2));
    }
}
