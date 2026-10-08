//! Core inference engine that wraps model + tokenizer + generation.

use crate::continuous_pump::ContinuousPump;
use crate::error::{ServeError, ServeResult};
use crate::types::ChatMessage;
use pmetal_data::chat_templates::{ChatTemplate, ChatTemplateType, detect_chat_template};
use pmetal_data::image_processing::{
    MllamaImageProcessor, MllamaImageProcessorConfig, mllama_cross_attention_mask,
};
use pmetal_data::inference_config::collect_all_stop_tokens;
use pmetal_data::qwen_vl_processing::{ProcessedMedia, QwenVlProcessor, RgbImage};
use pmetal_mlx::kv_cache::{CacheMode, KVCache, KVCacheConfig, MambaCache};
use pmetal_mlx::{Array, Dtype, ModuleParameters as _};
use pmetal_models::architectures::mllama::{CrossAttentionInputs, MllamaVisionInputs};
use pmetal_models::architectures::qwen3_5_vision::{Qwen3_5MultimodalConfig, Qwen3_5Vision};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_models::generation::{GenerationConfig, Sampler};
use pmetal_models::model_thread::{Background, ModelThread, ModelThreadStartError};
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

/// Per-token signal returned by the decode emit callback.
///
/// `Cancel` is used by the streaming path when the client has dropped the
/// receiver — the loop then returns with a "cancelled" finish reason so the
/// caller can shut down cleanly.
enum StepOutcome {
    Continue,
    Cancel,
}

/// Aggregated result of a single async decode run.
struct DecodeRun {
    /// Generated token IDs (already truncated when a stop-sequence matched).
    generated: Vec<u32>,
    /// Per-token logprobs when the request opted in; `None` otherwise.
    logprobs: Option<Vec<TokenLogprobEntry>>,
    /// Finish reason: `"length"`, `"stop"`, or `"cancelled"`.
    finish_reason: &'static str,
    /// Number of tokens stripped from the tail due to stop-sequence match
    /// (used by the streaming path to tell the client how many to discard).
    stripped_tokens: usize,
    /// Milliseconds to first generated token (TTFT).
    first_token_time_ms: Option<f64>,
    /// Final token count (after stop-sequence truncation).
    completion_tokens: usize,
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
            ..GenerationConfig::default()
        }
    } else {
        GenerationConfig::greedy(max_tokens).with_stop_tokens(stop_tokens)
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

pub(crate) fn detect_stop_sequence_suffix(
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
///
/// The model lives on its own thread (see [`ModelThread`]): it is loaded there,
/// every request that touches it runs there, one at a time, and so does the
/// continuous-batching scheduler when it is enabled. MLX streams belong to the
/// thread that created them, so a model moved between tokio's blocking-pool
/// threads failed with "There is no Stream(gpu, N) in current thread" as soon
/// as a request landed on a thread other than the one that built its arrays.
pub struct InferenceEngine {
    /// The thread that owns the model.
    model: ModelThread<EngineState>,
    /// The tokenizer.
    tokenizer: Arc<pmetal_data::Tokenizer>,
    /// Detected chat template.
    chat_template: ChatTemplate,
    /// How the model reads images and videos, if it can.
    vision: Vision,
    /// Model name/ID for API responses.
    model_id: String,
    /// Resolved local model directory (used by ANE / CPU-hybrid backends).
    model_path: std::path::PathBuf,
    /// Maximum sequence length for KV cache.
    max_seq_len: usize,
    /// Fixed ANE bucket cap for accelerated backends.
    ane_max_seq_len: usize,
    /// DFlash draft model drafting for the ANE engine, if any.
    ane_draft_path: Option<std::path::PathBuf>,
    /// Preferred generation backend; falls back to GPU permanently on failure.
    backend: Arc<Mutex<BackendState>>,
    /// Stop token IDs collected from all available sources.
    stop_token_ids: Vec<u32>,
    /// Model creation timestamp.
    created_at: i64,
    /// Explicit cache mode override (bypasses auto-selection when set).
    cache_mode_override: Option<CacheMode>,
    /// Cross-request KV prefix cache. Consulted at the start of every
    /// request to skip the prefill of any prompt prefix we've already
    /// processed. Empty for hybrid/recurrent models (they can't be
    /// safely snapshot-truncated) and for every request where the
    /// engine is also running an accelerated ANE/CPU-hybrid backend.
    /// Its entries are only touched on the model thread.
    prefix_cache: Arc<Mutex<crate::prefix_cache::ServePrefixCache>>,
    /// The continuous-batching pump while it is enabled (`None` by
    /// default). The model thread drives it; this handle is for
    /// [`continuous_batching_depth`](Self::continuous_batching_depth).
    continuous: Mutex<Option<Arc<Mutex<ContinuousPump>>>>,
}

/// What lives on the model thread.
struct EngineState {
    model: DynamicModel,
    /// The continuous-batching pump, while enabled. The thread runs one of its
    /// steps whenever no request is queued.
    continuous: Option<Arc<Mutex<ContinuousPump>>>,
    /// A Qwen 3.5-family vision tower, loaded by the first request with
    /// media and kept.
    qwen_vision: Option<Qwen3_5Vision>,
}

// ────────────────────────────────────────────────────────────────────────────
// Images and videos
// ────────────────────────────────────────────────────────────────────────────

/// How the engine's model reads images and videos.
enum Vision {
    /// A Qwen 3.5-family checkpoint with a vision tower. Its processor runs
    /// with the request, off the model thread; the tower runs on it.
    Qwen3_5 {
        processor: Box<QwenVlProcessor>,
        config: Box<Qwen3_5MultimodalConfig>,
    },
    /// Llama 3.2 Vision: images only, one `<|image|>` token each, read by
    /// the text model's cross-attention layers. The tower is part of the
    /// loaded model.
    Mllama {
        processor: Box<MllamaImageProcessor>,
        image_token_id: u32,
    },
    /// It can't; the reason goes into the 400.
    Unsupported(String),
}

impl Vision {
    fn detect(model_path: &Path) -> Self {
        let config = match std::fs::read_to_string(model_path.join("config.json")) {
            Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(config) => config,
                Err(e) => return Self::Unsupported(format!("its config.json is unreadable: {e}")),
            },
            Err(_) => return Self::Unsupported("it has no config.json".into()),
        };
        let model_type = config["model_type"].as_str().unwrap_or("unknown");
        if model_type == "qwen3_5" {
            let vision = Qwen3_5MultimodalConfig::from_config_value(&config).and_then(|config| {
                let processor = QwenVlProcessor::from_model_dir(model_path)
                    .map_err(|e| pmetal_mlx::Exception::custom(e.to_string()))?;
                Ok(Self::Qwen3_5 {
                    processor: Box::new(processor),
                    config: Box::new(config),
                })
            });
            return vision.unwrap_or_else(|e| Self::Unsupported(e.to_string()));
        }
        let has_tower = config.get("vision_config").is_some_and(|v| v.is_object());
        if model_type == "mllama" && has_tower {
            // The processor's settings; every field has the released
            // checkpoint's value as its default.
            let processor_config =
                match std::fs::read_to_string(model_path.join("preprocessor_config.json")) {
                    Ok(text) => match serde_json::from_str::<MllamaImageProcessorConfig>(&text) {
                        Ok(config) => config,
                        Err(e) => {
                            return Self::Unsupported(format!(
                                "its preprocessor_config.json is unreadable: {e}"
                            ));
                        }
                    },
                    Err(_) => MllamaImageProcessorConfig::default(),
                };
            return match MllamaImageProcessor::new(processor_config) {
                Ok(processor) => Self::Mllama {
                    processor: Box::new(processor),
                    image_token_id: config["image_token_index"].as_u64().unwrap_or(128_256) as u32,
                },
                Err(e) => Self::Unsupported(e.to_string()),
            };
        }
        if has_tower {
            return Self::Unsupported(format!(
                "pmetal does not run the vision tower of {model_type} checkpoints in generation"
            ));
        }
        Self::Unsupported("it is a text-only model".into())
    }
}

/// A prompt ready to generate from: its token ids and, when it carries images
/// or videos, the preprocessed media they stand for.
///
/// Built by [`InferenceEngine::prepare_chat`]; a text-only prompt is just its
/// ids ([`PreparedPrompt::from_tokens`]).
#[derive(Debug, Clone)]
pub struct PreparedPrompt {
    /// The prompt's ids, each image and video expanded to its token run.
    pub input_ids: Vec<u32>,
    media: Option<Arc<PromptMedia>>,
}

impl PreparedPrompt {
    /// A text-only prompt.
    pub fn from_tokens(input_ids: Vec<u32>) -> Self {
        Self {
            input_ids,
            media: None,
        }
    }

    /// Whether the prompt carries images or videos. Such a prompt runs on the
    /// single-request GPU path: the prefix cache and the continuous-batching
    /// pump key and drive a prompt by its token ids alone, and every image's
    /// tokens are the same placeholder id whatever its pixels.
    pub fn has_media(&self) -> bool {
        self.media.is_some()
    }
}

/// A prompt's media, preprocessed: plain pixels, built off the model thread.
#[derive(Debug)]
enum PromptMedia {
    Qwen3_5 {
        images: Vec<ProcessedMedia>,
        videos: Vec<ProcessedMedia>,
    },
    /// The decoded images (their tiles are cut on the model thread, where
    /// the processor's arrays are made) and which tiles each prompt token
    /// may attend to, `[prompt, images, max_tiles]`.
    Mllama {
        processor: Box<MllamaImageProcessor>,
        images: Vec<RgbImage>,
        cross_attention_mask: Vec<f32>,
    },
}

/// A prompt's media, encoded on the model thread: what its prefill and decode
/// steps feed the model instead of plain token ids.
enum MediaForward {
    /// The prompt's embeddings with the media rows replaced by vision
    /// features, at 3-D positions; decoding continues from `next_position`,
    /// one past the prompt's largest position.
    Qwen3_5 {
        embeddings: Array,
        positions: Array,
        next_position: i32,
    },
    /// The prompt's ids and the projected image features with the prompt's
    /// cross-attention mask; decoding reads the features with the mask's
    /// last row, as the reference extends it for each generated token.
    Mllama {
        input_ids: Array,
        prefill: Box<CrossAttentionInputs>,
        decode: Box<CrossAttentionInputs>,
    },
}

/// The model as Llama 3.2 Vision, for a prompt the Mllama processor prepared.
fn mllama(
    model: &mut DynamicModel,
) -> Result<
    &mut pmetal_models::architectures::mllama::MllamaForConditionalGeneration,
    pmetal_bridge::compat::Exception,
> {
    model
        .as_mllama_mut()
        .ok_or_else(|| pmetal_bridge::compat::Exception::custom("not a Llama 3.2 Vision model"))
}

/// One continuous-batching step, run by the model thread between jobs: a
/// prefill chunk for one slot or a decode step for every decoding slot.
/// Reports idle when no request is pending or in flight; the job that
/// enqueues the next one wakes the thread.
fn continuous_step(state: &mut EngineState) -> Background {
    use crate::continuous_pump::Tick;

    let Some(pump) = state.continuous.as_ref() else {
        return Background::Idle;
    };
    let Ok(mut pump) = pump.lock() else {
        tracing::error!(target: "pmetal_serve::continuous_batch", "pump mutex poisoned");
        return Background::Idle;
    };
    let model = &mut state.model;
    let mut forward = |tokens: &[u32],
                       cache: &mut KVCache,
                       recurrent: Option<&mut MambaCache>|
     -> Result<Array, pmetal_bridge::compat::Exception> {
        let input = Array::from_u32_slice(tokens, &[1, tokens.len() as i32])
            .as_dtype(Dtype::Int32.as_i32());
        model.forward_with_hybrid_cache(&input, None, Some(cache), recurrent)
    };
    match pump.tick(&mut forward) {
        Ok(Tick::Ran) => Background::Busy,
        Ok(Tick::Idle) => Background::Idle,
        // A failed step already failed its requests; anything else left
        // here is the pump's own bookkeeping, which another step won't fix.
        Err(e) => {
            tracing::error!(
                target: "pmetal_serve::continuous_batch",
                "continuous-batching scheduler error: {e:?}"
            );
            Background::Idle
        }
    }
}

// Default prefix-cache budgets. Generous on entries since each one is
// just a token sequence + a KV fork; the byte budget is the hard cap.
const DEFAULT_PREFIX_CACHE_ENTRIES: usize = 16;
const DEFAULT_PREFIX_CACHE_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2 GiB

/// What a GPU request needs on the model thread, besides the model.
#[derive(Clone)]
struct GpuRequestContext {
    model_path: PathBuf,
    max_seq_len: usize,
    cache_mode_override: Option<CacheMode>,
    tokenizer: Arc<pmetal_data::Tokenizer>,
    prefix_cache: Arc<Mutex<crate::prefix_cache::ServePrefixCache>>,
}

/// One GPU generation request, run on the model thread.
struct GpuRequest {
    input_ids: Vec<u32>,
    gen_config: GenerationConfig,
    stop_sequences: Vec<String>,
    logprobs_top_n: Option<usize>,
    /// The images and videos the prompt's media tokens stand for.
    media: Option<Arc<PromptMedia>>,
}

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

    /// Create a new inference engine, loading the model with `load`.
    ///
    /// `load` runs on the engine's model thread, where the model stays: it
    /// has to be built there, since arrays it leaves unevaluated can't be
    /// evaluated on any other thread. Returns once the model is loaded, or
    /// with the error `load` returned.
    pub fn new(
        load: impl FnOnce() -> anyhow::Result<DynamicModel> + Send + 'static,
        tokenizer: pmetal_data::Tokenizer,
        model_id: String,
        model_path: &std::path::Path,
        max_seq_len: usize,
    ) -> anyhow::Result<Self> {
        Self::new_with_backend(
            load,
            tokenizer,
            model_id,
            model_path,
            max_seq_len,
            true,
            4096,
        )
    }

    /// Create a new inference engine with explicit backend controls. See
    /// [`new`](Self::new) for `load`.
    pub fn new_with_backend(
        load: impl FnOnce() -> anyhow::Result<DynamicModel> + Send + 'static,
        tokenizer: pmetal_data::Tokenizer,
        model_id: String,
        model_path: &std::path::Path,
        max_seq_len: usize,
        ane_enabled: bool,
        ane_max_seq_len: usize,
    ) -> anyhow::Result<Self> {
        let chat_template = detect_chat_template(model_path, &model_id);

        // Collect stop tokens from all available sources using the canonical
        // `collect_all_stop_tokens` implementation from pmetal-data.
        // This merges generation_config.json EOS, chat-template EOS, tokenizer
        // EOS, and 11 well-known special token probes — deduplicated.
        let template_type: Option<ChatTemplateType> = Some(chat_template.template_type);
        let stop_token_ids = collect_all_stop_tokens(model_path, &tokenizer, template_type);

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

        let model = ModelThread::spawn_with_background(
            "pmetal-model",
            move || {
                load().map(|model| EngineState {
                    model,
                    continuous: None,
                    qwen_vision: None,
                })
            },
            continuous_step,
        )
        .map_err(|e| match e {
            ModelThreadStartError::Init(e) => e,
            other => anyhow::anyhow!("{other}"),
        })?;

        tracing::info!(
            "Inference engine ready: model_id={}, stop_tokens={:?}",
            model_id,
            stop_token_ids
        );
        tracing::info!(
            model = %model_path.display(),
            backend = ?preferred_backend,
            ane_enabled,
            ane_max_seq_len,
            "Selected serving generation backend"
        );

        Ok(Self {
            model,
            tokenizer: Arc::new(tokenizer),
            chat_template,
            vision: Vision::detect(model_path),
            model_id,
            model_path: model_path.to_path_buf(),
            max_seq_len,
            ane_max_seq_len,
            ane_draft_path: None,
            backend: Arc::new(Mutex::new(BackendState {
                preferred: preferred_backend,
            })),
            stop_token_ids,
            created_at: chrono::Utc::now().timestamp(),
            cache_mode_override: None,
            prefix_cache: Arc::new(Mutex::new(crate::prefix_cache::ServePrefixCache::new(
                DEFAULT_PREFIX_CACHE_ENTRIES,
                DEFAULT_PREFIX_CACHE_BYTES,
            ))),
            continuous: Mutex::new(None),
        })
    }

    /// Draft with the DFlash draft model in `path` when generating on the ANE
    /// (run on the GPU, reading the ANE model's hidden states).
    pub fn with_ane_drafter(mut self, path: std::path::PathBuf) -> Self {
        self.ane_draft_path = Some(path);
        self
    }

    /// Use `mode` for every request's KV cache instead of choosing one from
    /// the model's size and the device's memory.
    pub fn with_cache_mode_override(mut self, mode: CacheMode) -> Self {
        self.cache_mode_override = Some(mode);
        self
    }

    /// Override the default prefix-cache budgets. `max_entries = 0`
    /// disables the cache; `max_bytes = 0` leaves the byte axis
    /// unbounded.
    pub fn set_prefix_cache_limits(&self, max_entries: usize, max_bytes: usize) {
        if let Ok(mut pc) = self.prefix_cache.lock() {
            *pc = crate::prefix_cache::ServePrefixCache::new(max_entries, max_bytes);
        }
    }

    /// Snapshot of prefix-cache stats: `(entries, bytes, hits, misses, hit_rate)`.
    pub fn prefix_cache_stats(&self) -> (usize, usize, u64, u64, f64) {
        match self.prefix_cache.lock() {
            Ok(pc) => (pc.len(), pc.bytes(), pc.hits(), pc.misses(), pc.hit_rate()),
            Err(_) => (0, 0, 0, 0, 0.0),
        }
    }

    /// Enable continuous batching with the given capacity. The model thread
    /// then runs one scheduler step (a prefill chunk or a decode step across
    /// the decoding slots) whenever no other request is queued, and sleeps
    /// when nothing is in flight.
    ///
    /// While enabled, callers dispatch requests through
    /// [`generate_batched`](Self::generate_batched). The single-request
    /// `generate` / `generate_streaming` paths continue to work; each runs
    /// to completion between two scheduler steps.
    ///
    /// Calling this twice is a no-op that returns `Ok` — the first
    /// configuration wins. Use
    /// [`disable_continuous_batching`](Self::disable_continuous_batching)
    /// first if you need to reconfigure.
    ///
    /// `cache_config` must match the model (num_layers, n_kv_heads,
    /// head_dim, max_seq_len). A mismatch will surface as a shape
    /// error on the first forward pass. Blocks until the model thread is
    /// free to set the pump up.
    pub fn enable_continuous_batching(
        &self,
        batcher_config: crate::continuous_batch::BatcherConfig,
        cache_config: KVCacheConfig,
    ) -> ServeResult<()> {
        let mut guard = self.continuous.lock().map_err(|_| ServeError::Busy)?;
        if guard.is_some() {
            return Ok(());
        }
        let tokenizer = Arc::clone(&self.tokenizer);
        let prefix_cache = Arc::clone(&self.prefix_cache);
        let pump = self
            .model
            .call(move |state| {
                let mut pump = ContinuousPump::new_with_prefix_cache(
                    batcher_config,
                    cache_config,
                    Some(tokenizer),
                    Some(prefix_cache),
                );
                // A hybrid model's slots each carry their own recurrent
                // state, and share no prefix cache.
                if let Some(template) = state.model.create_mamba_cache() {
                    pump = pump.with_recurrent_state(&template);
                }
                let pump = Arc::new(Mutex::new(pump));
                state.continuous = Some(Arc::clone(&pump));
                pump
            })
            .map_err(|_| ServeError::ModelNotLoaded)?;
        *guard = Some(pump);
        Ok(())
    }

    /// Stop continuous batching and drop the pump. Requests still in it see
    /// their receivers close.
    pub fn disable_continuous_batching(&self) {
        if let Ok(mut guard) = self.continuous.lock() {
            if guard.take().is_some() {
                let _ = self.model.submit(|state| state.continuous = None);
            }
        }
    }

    /// Dispatch a request through the continuous-batching pump.
    ///
    /// Returns an mpsc receiver that emits one `TokenEvent::Token` per
    /// generated token and exactly one `TokenEvent::Done` /
    /// `TokenEvent::Error` terminator — mirroring the streaming
    /// contract of `generate_streaming`. A request the pump's queue has
    /// no room for runs on the single-request path instead, into the
    /// same receiver.
    ///
    /// Errors if continuous batching is not enabled (call
    /// [`enable_continuous_batching`](Self::enable_continuous_batching)
    /// first) or the model thread has exited.
    pub fn generate_batched(
        &self,
        input_ids: &[u32],
        params: SamplingParams,
    ) -> ServeResult<tokio::sync::mpsc::Receiver<TokenEvent>> {
        Self::validate_params(&params, self.max_seq_len)?;
        if !self.continuous_batching_enabled() {
            return Err(ServeError::Internal(
                "continuous batching not enabled; call enable_continuous_batching first".into(),
            ));
        }

        let gen_config = self.build_generation_config(&params);
        let slot_params = crate::continuous_batch::SlotParams {
            max_new_tokens: gen_config.max_new_tokens,
            stop_tokens: gen_config.stop_tokens.clone(),
            stop_sequences: params.stop_sequences.clone(),
            prefill_step_size: gen_config.prefill_step_size,
            logprobs_top_n: params.logprobs_top_n,
        };
        let request = GpuRequest {
            input_ids: input_ids.to_vec(),
            gen_config,
            stop_sequences: params.stop_sequences,
            logprobs_top_n: params.logprobs_top_n,
            media: None,
        };
        let ctx = self.gpu_request_context();
        // Room for every event the request can produce: the scheduler never
        // waits on one slow reader, and a full channel would drop tokens.
        let (tx, rx) =
            tokio::sync::mpsc::channel::<TokenEvent>(ContinuousPump::events_capacity(&slot_params));

        // Enqueueing touches the prefix cache's KV snapshots, so it runs on
        // the model thread too; queueing the job also wakes the scheduler.
        self.model
            .submit(move |state| {
                let enqueued = match state.continuous.as_ref().map(|pump| pump.lock()) {
                    Some(Ok(mut pump)) => pump.enqueue_with_sender(
                        request.input_ids.clone(),
                        slot_params,
                        request.gen_config.clone(),
                        tx.clone(),
                    ),
                    Some(Err(_)) => {
                        let _ = tx.try_send(TokenEvent::Error(
                            "continuous-batching pump poisoned".into(),
                        ));
                        return;
                    }
                    None => {
                        let _ = tx
                            .try_send(TokenEvent::Error("continuous batching was disabled".into()));
                        return;
                    }
                };
                if let Err(e) = enqueued {
                    tracing::warn!(
                        "continuous-batching enqueue failed ({e}); running the request on the \
                         single-request path"
                    );
                    Self::stream_on_model(state, &ctx, request, &tx);
                }
            })
            .map_err(|_| ServeError::ModelNotLoaded)?;
        Ok(rx)
    }

    /// Whether continuous batching has been enabled on this engine.
    pub fn continuous_batching_enabled(&self) -> bool {
        self.continuous.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    fn create_continuous_cache_config(&self) -> ServeResult<KVCacheConfig> {
        let model_path = self.model_path.clone();
        let max_seq_len = self.max_seq_len;
        let cache_mode_override = self.cache_mode_override;
        self.model
            .call(move |state| {
                let (cache, _) = Self::create_request_caches(
                    &state.model,
                    &model_path,
                    max_seq_len,
                    cache_mode_override,
                );
                cache.config().clone()
            })
            .map_err(|_| ServeError::ModelNotLoaded)
    }

    /// Convenience wrapper that derives the KV-cache config from the
    /// loaded model and calls
    /// [`enable_continuous_batching`](Self::enable_continuous_batching).
    ///
    /// The cache config comes from `DynamicModel::create_cache(max_seq_len)`
    /// so it matches exactly what `create_request_caches` hands out for
    /// single-request generation.
    pub fn enable_continuous_batching_auto(
        &self,
        mut batcher_config: crate::continuous_batch::BatcherConfig,
    ) -> ServeResult<()> {
        let cache_config = self.create_continuous_cache_config()?;
        if batcher_config.max_blocks == 0 {
            let block_size = batcher_config.effective_block_size();
            batcher_config.max_blocks =
                batcher_config.max_slots.max(1) * self.max_seq_len.div_ceil(block_size);
        }
        self.enable_continuous_batching(batcher_config, cache_config)
    }

    /// Inspect pump depth: `(active_slots, pending_depth)`. Returns
    /// `(0, 0)` when continuous batching is not enabled.
    pub fn continuous_batching_depth(&self) -> (usize, usize) {
        let pump = match self.continuous.lock() {
            Ok(guard) => guard.clone(),
            Err(_) => return (0, 0),
        };
        match pump.as_ref().map(|pump| pump.lock()) {
            Some(Ok(pump)) => (pump.active_slots(), pump.pending_depth()),
            _ => (0, 0),
        }
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
        let msgs = Self::template_messages(messages);
        // apply_inference prefers the upstream Jinja template when present, so
        // tool definitions land in the exact shape the model was trained on.
        let formatted = self.chat_template.apply_inference(&msgs, false, tools);
        formatted.text
    }

    /// Format chat messages with tool definitions and chat template kwargs
    /// (`enable_thinking`, `reasoning_effort`, `preserve_thinking`, …), as
    /// `apply_chat_template(messages, tools=tools, add_generation_prompt=True,
    /// **kwargs)` renders them.
    ///
    /// A kwarg the template does not read is ignored, as it is there. A
    /// value the template refuses (its own `raise_exception`), or a
    /// `reasoning_effort` outside the levels a model documents, is a bad
    /// request.
    pub fn format_chat_with_kwargs(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[pmetal_data::chat_templates::ToolDefinition]>,
        kwargs: &pmetal_data::chat_templates::ChatTemplateKwargs,
    ) -> ServeResult<String> {
        self.chat_template
            .check_reasoning_effort_level(kwargs)
            .map_err(ServeError::BadRequest)?;
        let msgs = Self::template_messages(messages);
        self.chat_template
            .apply_inference_with_kwargs(&msgs, tools, kwargs)
            .map(|formatted| formatted.text)
            .map_err(ServeError::BadRequest)
    }

    /// Whether the model thinks under these chat template kwargs: its
    /// template has a thinking control or block and `enable_thinking` is not
    /// `false`.
    pub fn thinks_with(&self, kwargs: &pmetal_data::chat_templates::ChatTemplateKwargs) -> bool {
        self.chat_template.thinks_with(kwargs)
    }

    /// The output budget for a request that sets none: the model's
    /// recommendation ([`pmetal_data::inference_config::default_max_tokens`]:
    /// `generation_config.json`, else 32768 tokens for a thinking model,
    /// else 256), within the context window and this server's
    /// `--max-seq-len` once the prompt's `prompt_len` tokens are in.
    pub fn default_max_tokens(&self, thinking: bool, prompt_len: usize) -> usize {
        let context = pmetal_data::inference_config::context_window(&self.model_path)
            .map_or(self.max_seq_len, |c| c.min(self.max_seq_len));
        pmetal_data::inference_config::default_max_tokens(&self.model_path, thinking)
            .for_prompt(prompt_len, Some(context))
    }

    /// The sampling a request gets for every parameter it leaves out: the
    /// model maker's recommendation for `thinking` or non-thinking mode,
    /// else `generation_config.json`, else the global fallback (see
    /// [`pmetal_data::inference_config::load_sampling_defaults`]).
    pub fn sampling_defaults(
        &self,
        thinking: bool,
    ) -> pmetal_data::inference_config::SamplingDefaults {
        pmetal_data::inference_config::load_sampling_defaults(
            &self.model_path,
            pmetal_data::inference_config::SamplingMode::Auto,
            thinking,
        )
    }

    fn template_messages(messages: &[ChatMessage]) -> Vec<pmetal_data::chat_templates::Message> {
        messages
            .iter()
            .map(|m| crate::media::template_message(m, None))
            .collect()
    }

    /// Whether the model reads images and videos.
    pub fn accepts_media(&self) -> bool {
        !matches!(self.vision, Vision::Unsupported(_))
    }

    /// Turn a chat request into a prompt to generate from.
    ///
    /// A text-only conversation is formatted with the chat template and
    /// tokenized. One whose messages carry images or videos (see
    /// [`crate::media`]) is checked against the model, rendered by the
    /// model's own chat template with each image and video an item of its
    /// message, and preprocessed by the checkpoint's processor, which also
    /// expands each placeholder to the media's token run.
    ///
    /// `kwargs` are the request's chat template kwargs, as
    /// [`format_chat_with_kwargs`](Self::format_chat_with_kwargs) takes them.
    ///
    /// Errors with a 400 when the model reads no images, when a part is
    /// unusable, when the template refuses a kwarg, or when the expanded
    /// prompt does not fit in the context.
    pub async fn prepare_chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[pmetal_data::chat_templates::ToolDefinition]>,
        kwargs: &pmetal_data::chat_templates::ChatTemplateKwargs,
    ) -> ServeResult<PreparedPrompt> {
        if !crate::media::has_media(messages) {
            let prompt = self.format_chat_with_kwargs(messages, tools, kwargs)?;
            return Ok(PreparedPrompt::from_tokens(self.tokenize(&prompt)?));
        }
        self.chat_template
            .check_reasoning_effort_level(kwargs)
            .map_err(ServeError::BadRequest)?;
        if let Vision::Unsupported(why) = &self.vision {
            return Err(ServeError::BadRequest(format!(
                "model '{}' does not accept images or videos: {why}",
                self.model_id
            )));
        }
        let chat = crate::media::split_messages(messages)?;
        let text = self
            .chat_template
            .render_inference_jinja_with_kwargs(&chat.messages, tools, kwargs)
            .map_err(|e| {
                ServeError::BadRequest(format!(
                    "the model's chat template could not place the images and videos: {e}"
                ))
            })?;
        let sources = chat.media;
        let (input_ids, media) = match &self.vision {
            Vision::Qwen3_5 { processor, config } => {
                let processor = processor.clone();
                let merge_size = config.vision.spatial_merge_size;
                let (text, images, videos) = tokio::task::spawn_blocking(move || {
                    let mut images = Vec::new();
                    let mut videos = Vec::new();
                    for media in crate::media::decode_media(&sources)? {
                        match media {
                            crate::media::DecodedMedia::Image(image) => images.push(
                                processor
                                    .preprocess_image(&image)
                                    .map_err(|e| ServeError::BadRequest(format!("image: {e}")))?,
                            ),
                            crate::media::DecodedMedia::Video(video) => videos.push(
                                processor
                                    .preprocess_video(&video)
                                    .map_err(|e| ServeError::BadRequest(format!("video: {e}")))?,
                            ),
                        }
                    }
                    let text = pmetal_data::qwen_vl_processing::expand_placeholders(
                        &text, &images, &videos, merge_size,
                    )
                    .map_err(|e| ServeError::BadRequest(e.to_string()))?;
                    Ok::<_, ServeError>((text, images, videos))
                })
                .await
                .map_err(|e| ServeError::Internal(e.to_string()))??;
                (
                    self.tokenize(&text)?,
                    PromptMedia::Qwen3_5 { images, videos },
                )
            }
            Vision::Mllama {
                processor,
                image_token_id,
            } => {
                if sources
                    .iter()
                    .any(|s| matches!(s, crate::media::MediaSource::Video { .. }))
                {
                    return Err(ServeError::BadRequest(format!(
                        "model '{}' reads images, not videos",
                        self.model_id
                    )));
                }
                let input_ids = self.tokenize(&text)?;
                let processor = processor.clone();
                let image_token_id = *image_token_id;
                tokio::task::spawn_blocking(move || {
                    let images: Vec<RgbImage> = crate::media::decode_media(&sources)?
                        .into_iter()
                        .filter_map(|media| match media {
                            crate::media::DecodedMedia::Image(image) => Some(image),
                            crate::media::DecodedMedia::Video(_) => None,
                        })
                        .collect();
                    let num_tiles = images
                        .iter()
                        .map(|image| processor.num_tiles(image.height(), image.width()))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| ServeError::BadRequest(format!("image: {e}")))?;
                    let cross_attention_mask = mllama_cross_attention_mask(
                        &input_ids,
                        image_token_id,
                        &num_tiles,
                        processor.config().max_image_tiles,
                    )
                    .map_err(|e| ServeError::BadRequest(e.to_string()))?;
                    Ok::<_, ServeError>((
                        input_ids,
                        PromptMedia::Mllama {
                            processor,
                            images,
                            cross_attention_mask,
                        },
                    ))
                })
                .await
                .map_err(|e| ServeError::Internal(e.to_string()))??
            }
            Vision::Unsupported(_) => unreachable!("refused above"),
        };
        if input_ids.len() >= self.max_seq_len {
            return Err(ServeError::BadRequest(format!(
                "the prompt is {} tokens with its images and videos, which leaves no room to \
                 generate in this server's {}-token context (--max-seq-len)",
                input_ids.len(),
                self.max_seq_len
            )));
        }
        Ok(PreparedPrompt {
            input_ids,
            media: Some(Arc::new(media)),
        })
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

    /// Decode token IDs back to text while preserving special tokens.
    pub fn decode_with_special_tokens(&self, tokens: &[u32]) -> ServeResult<String> {
        self.tokenizer
            .decode_with_special_tokens(tokens)
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

    /// Validate sampling parameters against this engine's serving limits.
    pub fn validate_sampling_params(&self, params: &SamplingParams) -> ServeResult<()> {
        Self::validate_params(params, self.max_seq_len)
    }

    /// Build a `GenerationConfig` from API request sampling parameters.
    ///
    /// Temperature == 0.0 or unset maps to greedy decoding (`do_sample = false`).
    /// All stop tokens (engine-level + per-request) are merged into the config.
    /// `max_tokens` is silently clamped to `max_seq_len` (matches OpenAI behaviour).
    pub fn build_generation_config(&self, params: &SamplingParams) -> GenerationConfig {
        build_generation_config_from_parts(&self.stop_token_ids, self.max_seq_len, params)
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
        draft_path: Option<&std::path::Path>,
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
                draft_path,
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
        draft_path: Option<&std::path::Path>,
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
                draft_path,
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

    /// Encode a prompt's media on the model thread: run the vision tower
    /// (loading it on first use) and build what the prefill feeds the model.
    fn media_forward(
        state: &mut EngineState,
        model_path: &Path,
        input_ids: &[u32],
        media: &PromptMedia,
    ) -> ServeResult<MediaForward> {
        match media {
            PromptMedia::Qwen3_5 { images, videos } => {
                if state.qwen_vision.is_none() {
                    let started = Instant::now();
                    state.qwen_vision = Some(Qwen3_5Vision::load(model_path)?);
                    tracing::info!(
                        elapsed_s = format!("{:.1}", started.elapsed().as_secs_f64()),
                        "Vision tower loaded"
                    );
                }
                let EngineState {
                    model, qwen_vision, ..
                } = state;
                let vision = qwen_vision.as_ref().expect("loaded above");
                let encoded = vision.encode(input_ids, images, videos)?;
                let qwen = model.as_qwen3_next_mut().ok_or_else(|| {
                    ServeError::Internal("a Qwen 3.5 vision checkpoint loaded another model".into())
                })?;
                let ids: Vec<i32> = input_ids.iter().map(|&id| id as i32).collect();
                let ids = Array::from_i32_slice_shaped(&ids, &[1, ids.len() as i32]);
                let text =
                    pmetal_bridge::compat::Module::forward(&mut qwen.model.embed_tokens, &ids)?;
                Ok(MediaForward::Qwen3_5 {
                    embeddings: encoded.merge(&text, input_ids, &vision.config)?,
                    positions: encoded.positions.array(),
                    next_position: encoded.positions.next_position,
                })
            }
            PromptMedia::Mllama {
                processor,
                images,
                cross_attention_mask,
            } => {
                use pmetal_bridge::compat::ops::slice_axis;

                let images: Vec<image::DynamicImage> = images
                    .iter()
                    .map(|image| image::DynamicImage::ImageRgb8(image.clone()))
                    .collect();
                let batch = processor.preprocess(&[images])?;
                let mllama = state.model.as_mllama_mut().ok_or_else(|| {
                    ServeError::Internal(
                        "a Llama 3.2 Vision checkpoint loaded another model".into(),
                    )
                })?;
                let (len, count) = (input_ids.len() as i32, batch.num_tiles[0].len() as i32);
                let max_tiles = processor.config().max_image_tiles as i32;
                let mask = Array::from_f32_slice(cross_attention_mask, &[1, len, count, max_tiles]);
                let prefill = mllama.prepare_cross_attention(
                    MllamaVisionInputs {
                        pixel_values: &batch.pixel_values,
                        aspect_ratio_ids: &batch.aspect_ratio_ids,
                        aspect_ratio_mask: &batch.aspect_ratio_mask,
                    },
                    Some(&mask),
                )?;
                // The tower runs once, here; every step reuses its output.
                prefill
                    .states
                    .try_eval()
                    .map_err(|e| ServeError::Internal(format!("vision tower: {e}")))?;
                pmetal_bridge::check_last_error()
                    .map_err(|e| ServeError::Internal(format!("vision tower: {e}")))?;
                let last_row = |array: &Option<Array>, axis: i32| {
                    array.as_ref().map(|a| slice_axis(a, axis, len - 1, len))
                };
                let decode = CrossAttentionInputs {
                    states: prefill.states.clone(),
                    mask: last_row(&prefill.mask, 2),
                    full_text_row_mask: last_row(&prefill.full_text_row_mask, 1),
                };
                let ids: Vec<i32> = input_ids.iter().map(|&id| id as i32).collect();
                Ok(MediaForward::Mllama {
                    input_ids: Array::from_i32_slice_shaped(&ids, &[1, len]),
                    prefill: Box::new(prefill),
                    decode: Box::new(decode),
                })
            }
        }
    }

    /// The prefill of a prompt with media, from its merged embeddings.
    /// Returns the prompt's logits.
    fn prefill_media(
        model: &mut DynamicModel,
        media: &MediaForward,
        cache: &mut KVCache,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, pmetal_bridge::compat::Exception> {
        match media {
            MediaForward::Qwen3_5 {
                embeddings,
                positions,
                ..
            } => {
                let qwen = model.as_qwen3_next_mut().ok_or_else(|| {
                    pmetal_bridge::compat::Exception::custom("not a Qwen 3.5-family model")
                })?;
                let (_hidden, logits) =
                    qwen.forward_embeddings(embeddings, positions, Some(cache), mamba_cache)?;
                Ok(logits)
            }
            MediaForward::Mllama {
                input_ids, prefill, ..
            } => mllama(model)?.forward_full(input_ids, Some(prefill.as_ref()), None, Some(cache)),
        }
    }

    /// Decode step `step` (0 for the first generated token): `input` is that
    /// token, `[1, 1]`. After a prompt with media, positions run from one
    /// past the prompt's largest position rather than from its length.
    fn decode_step(
        model: &mut DynamicModel,
        media: Option<&MediaForward>,
        input: &Array,
        step: usize,
        cache: &mut KVCache,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, pmetal_bridge::compat::Exception> {
        match media {
            None => model.forward_with_hybrid_cache(input, None, Some(cache), mamba_cache),
            Some(MediaForward::Qwen3_5 { next_position, .. }) => {
                let qwen = model.as_qwen3_next_mut().ok_or_else(|| {
                    pmetal_bridge::compat::Exception::custom("not a Qwen 3.5-family model")
                })?;
                let at = Array::from_i32_slice(&[next_position + step as i32]);
                qwen.forward_with_cache_at(input, &at, Some(cache), mamba_cache)
            }
            Some(MediaForward::Mllama { decode, .. }) => {
                mllama(model)?.forward_full(input, Some(decode.as_ref()), None, Some(cache))
            }
        }
    }

    /// Run a single GPU decode request with 1-step look-ahead async
    /// pipelining, chunked prefill, and wired-memory management.
    ///
    /// Shared between [`generate`](Self::generate) and
    /// [`generate_streaming`](Self::generate_streaming). `on_token` is
    /// invoked once per generated token (after the stop-token check, before
    /// the stop-sequence check). For non-streaming the callback is a no-op;
    /// for streaming it pushes a `TokenEvent::Token` into the mpsc channel.
    ///
    /// # Pipelining contract
    ///
    /// - `forward(N+1)` is scheduled under a generation-stream context
    ///   before the host extracts `token(N)`.
    /// - `async_eval` is called outside the stream context so the GPU is
    ///   free to advance while the host samples / extracts.
    /// - The loop is driven by `.item::<u32>()` on the current token
    ///   array — this is the ONLY synchronous host sync per step.
    /// - Wired-memory limit and generation stream are installed once per
    ///   request via RAII guards.
    ///
    /// # Penalty lag
    ///
    /// Because the next forward is scheduled before `token(N)` is
    /// extracted, the repetition / frequency / presence penalty context
    /// at step `N+1` is missing exactly one token (the one that will be
    /// emitted as `token(N)`). This is the usual tradeoff of
    /// async decode — the alternative (sync-per-step) destroys the
    /// pipeline. In practice the lag is inaudible for soft penalties
    /// because the missing token is one of `prompt_len + N` history
    /// tokens feeding the scatter.
    #[allow(clippy::too_many_arguments)]
    fn run_async_decode<E>(
        model: &mut DynamicModel,
        cache: &mut KVCache,
        mamba_cache: &mut Option<MambaCache>,
        sampler: &mut Sampler,
        tokenizer: &pmetal_data::Tokenizer,
        input_ids: &[u32],
        max_tokens: usize,
        stop_tokens: &[u32],
        stop_sequences: &[String],
        logprobs_top_n: Option<usize>,
        prefill_step_size: usize,
        prefix_cache: Option<&Arc<Mutex<crate::prefix_cache::ServePrefixCache>>>,
        media: Option<&MediaForward>,
        start: Instant,
        mut on_token: E,
    ) -> ServeResult<DecodeRun>
    where
        E: FnMut(u32, Option<TokenLogprobEntry>) -> StepOutcome,
    {
        use pmetal_bridge::compat::ops::async_eval;
        use pmetal_models::generation::{
            StreamContext, WiredLimitGuard, clear_generation_caches, create_generation_stream,
            run_cached_prefill_chunks, token_logprobs,
        };

        let _wired_guard = WiredLimitGuard::new();
        let stream = create_generation_stream();

        let mut all_tokens: Vec<u32> = input_ids.to_vec();
        let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut logprobs_out: Option<Vec<TokenLogprobEntry>> =
            logprobs_top_n.map(|_| Vec::with_capacity(max_tokens));
        let mut first_token_time_ms: Option<f64> = None;
        let mut stripped_tokens: usize = 0;
        let mut finish_reason: &'static str = "length";
        let mut cancelled = false;

        // === Prefix-cache lookup (non-hybrid only) ===
        //
        // If the incoming prompt is a strict extension of a cached
        // prefix, restore the KV state and prefill only the suffix.
        // Mamba/GDN/hybrid models can't be snapshot-truncated cleanly,
        // so we skip the cache entirely when `mamba_cache` is populated.
        // A prompt with media skips it too: the cache is keyed by token ids,
        // and an image's tokens are the same placeholder id whatever it shows.
        let prefix_cache = prefix_cache.filter(|_| media.is_none());
        let prefix_hit_len: usize =
            if let (Some(pc), true) = (prefix_cache.as_ref(), mamba_cache.is_none()) {
                let mut guard = match pc.lock() {
                    Ok(g) => g,
                    Err(_) => return Err(ServeError::Busy),
                };
                match guard
                    .find_longest_prefix(input_ids, cache.config().clone())
                    .map_err(ServeError::Model)?
                {
                    Some(hit) => {
                        // Replace the freshly-allocated cache with the
                        // restored one from the snapshot.
                        *cache = hit.restored_cache;
                        tracing::debug!(
                            target: "pmetal_serve::prefix_cache",
                            "prefix cache hit: {}/{} tokens restored",
                            hit.prefix_len,
                            input_ids.len()
                        );
                        hit.prefix_len
                    }
                    None => 0,
                }
            } else {
                0
            };

        let prefill_slice: &[u32] = &input_ids[prefix_hit_len..];

        // === Prefill (chunked, on possibly shortened suffix) ===
        //
        // `run_cached_prefill_chunks` calls `forward` once per chunk; each
        // call wraps the forward in a fresh `StreamContext` so chunked
        // prefill also runs on the generation stream. The final chunk's
        // logits are returned lazily so we can fold them into the async
        // decode pipeline without a host sync.
        //
        // A prompt with media prefills in one forward from its merged
        // embeddings, as `pmetal infer --image` does.
        let prefill_logits = match media {
            None => run_cached_prefill_chunks(prefill_slice, prefill_step_size, |chunk| {
                let _ctx = StreamContext::new(&stream);
                model.forward_with_hybrid_cache(chunk, None, Some(cache), mamba_cache.as_mut())
            }),
            Some(media) => {
                let _ctx = StreamContext::new(&stream);
                Self::prefill_media(model, media, cache, mamba_cache.as_mut())
            }
        }
        .map_err(ServeError::Model)?;

        // === Cache the full-prompt KV state for future hits ===
        //
        // Only insert when we actually did work (prefill_slice non-empty)
        // and the KV cache reflects the full prompt length. Hybrid
        // models skip caching for the same reason they skip lookup.
        if let (Some(pc), true, true) = (
            prefix_cache.as_ref(),
            mamba_cache.is_none(),
            !prefill_slice.is_empty(),
        ) {
            if let Ok(mut guard) = pc.lock() {
                guard.insert(input_ids, cache);
            }
        }

        let mut current_last = Self::extract_last_logits(&prefill_logits)?;
        let (mut current_y, _filtered) = sampler
            .sample_array_with_penalties(&current_last, &all_tokens)
            .map_err(ServeError::Model)?;
        // Schedule the first-token pair async so the GPU starts working
        // before we enter the loop.
        async_eval([&current_y, &current_last]);

        // === Decode loop with 1-step look-ahead ===
        let mut i = 0usize;
        while i < max_tokens {
            // 1. Schedule NEXT forward BEFORE extracting current token.
            let next = if i + 1 < max_tokens {
                let pair = {
                    let _ctx = StreamContext::new(&stream);
                    let next_input = current_y
                        .as_dtype(pmetal_bridge::compat::Dtype::Int32.as_i32())
                        .reshape(&[1, -1]);
                    let next_full = Self::decode_step(
                        model,
                        media,
                        &next_input,
                        i,
                        cache,
                        mamba_cache.as_mut(),
                    )
                    .map_err(ServeError::Model)?;
                    let next_last = Self::extract_last_logits(&next_full)?;
                    let (ny, _) = sampler
                        .sample_array_with_penalties(&next_last, &all_tokens)
                        .map_err(ServeError::Model)?;
                    (ny, next_last)
                };
                async_eval([&pair.0, &pair.1]);
                Some(pair)
            } else {
                None
            };

            // 2. First iteration: force-eval the token so .item() below
            //    gets data (the n==0 special case).
            if i == 0 {
                current_y.try_eval().map_err(|e| {
                    ServeError::Model(pmetal_bridge::compat::Exception::custom(e.to_string()))
                })?;
            }

            // 3. Extract token from GPU — blocks until current_y is ready.
            //    GPU is already computing token(i+1) in parallel.
            let token = current_y.item::<u32>();

            if first_token_time_ms.is_none() {
                first_token_time_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
            }

            // 4. Stop-token check (before any side-effects).
            if stop_tokens.contains(&token) {
                finish_reason = "stop";
                break;
            }

            // 5. Update frequency-penalty history.
            sampler.update_counts(token);

            // 6. Optional logprobs: compute from RAW last-position logits
            //    (the sampler's filtered log-probs aren't suitable for
            //    OpenAI-style reporting). Cheap when top_n is small.
            let logprob_entry = match logprobs_top_n {
                Some(top_n) => match token_logprobs(&current_last, token, top_n + 1) {
                    Ok((lp, mut top)) => {
                        top.retain(|(tok, _)| *tok != token);
                        top.truncate(top_n);
                        Some(TokenLogprobEntry {
                            token,
                            logprob: lp,
                            top_logprobs: top,
                        })
                    }
                    Err(_) => None,
                },
                None => None,
            };

            generated.push(token);
            all_tokens.push(token);

            // 7. Accumulate logprob for non-streaming return value. We
            //    clone so the owned value can still be handed to the
            //    streaming callback below.
            if let (Some(entry), Some(out)) = (logprob_entry.as_ref(), logprobs_out.as_mut()) {
                out.push(entry.clone());
            }

            // 8. Emit to caller (moves logprob entry into TokenEvent for
            //    streaming; ignored by non-streaming).
            match on_token(token, logprob_entry) {
                StepOutcome::Continue => {}
                StepOutcome::Cancel => {
                    cancelled = true;
                    break;
                }
            }

            // 9. Stop-sequence detection on decoded text (multi-token
            //    suffix match via tokenizer).
            if let Some(n_strip) =
                detect_stop_sequence_suffix(tokenizer, &generated, stop_sequences)
            {
                stripped_tokens = n_strip;
                finish_reason = "stop";
                break;
            }

            // 10. Periodic allocation-cache sweep.
            if i > 0 && i % 256 == 0 {
                clear_generation_caches();
            }

            // 11. Swap current ← next.
            if let Some((ny, nl)) = next {
                current_y = ny;
                current_last = nl;
            }

            i += 1;
        }

        // Truncate tail if a stop-sequence was matched. Non-streaming
        // callers see the truncated vec; streaming callers use
        // `stripped_tokens` in the Done event to tell the client how many
        // tokens to drop from the visible stream.
        let visible_len = generated.len().saturating_sub(stripped_tokens);
        if let Some(out) = logprobs_out.as_mut() {
            out.truncate(visible_len);
        }
        generated.truncate(visible_len);

        if cancelled {
            finish_reason = "cancelled";
        }

        Ok(DecodeRun {
            generated,
            logprobs: logprobs_out,
            finish_reason,
            stripped_tokens,
            first_token_time_ms,
            completion_tokens: visible_len,
        })
    }
    /// What a GPU request needs on the model thread, besides the model.
    fn gpu_request_context(&self) -> GpuRequestContext {
        GpuRequestContext {
            model_path: self.model_path.clone(),
            max_seq_len: self.max_seq_len,
            cache_mode_override: self.cache_mode_override,
            tokenizer: Arc::clone(&self.tokenizer),
            prefix_cache: Arc::clone(&self.prefix_cache),
        }
    }

    /// Run one request on the GPU, on the model thread. `on_token` sees each
    /// token as it is generated. Returns the run and when it started.
    fn generate_on_model<E>(
        state: &mut EngineState,
        ctx: &GpuRequestContext,
        request: GpuRequest,
        on_token: E,
    ) -> ServeResult<(DecodeRun, Instant)>
    where
        E: FnMut(u32, Option<TokenLogprobEntry>) -> StepOutcome,
    {
        let start = Instant::now();
        let media = match request.media.as_deref() {
            Some(media) => Some(Self::media_forward(
                state,
                &ctx.model_path,
                &request.input_ids,
                media,
            )?),
            None => None,
        };
        let model = &mut state.model;
        let (mut cache, mut mamba_cache) = Self::create_request_caches(
            model,
            &ctx.model_path,
            ctx.max_seq_len,
            ctx.cache_mode_override,
        );
        let max_tokens = request.gen_config.max_new_tokens;
        let stop_tokens = request.gen_config.stop_tokens.clone();
        let prefill_step_size = request.gen_config.prefill_step_size;
        let mut sampler = Sampler::new(request.gen_config);
        let run = Self::run_async_decode(
            model,
            &mut cache,
            &mut mamba_cache,
            &mut sampler,
            ctx.tokenizer.as_ref(),
            &request.input_ids,
            max_tokens,
            &stop_tokens,
            &request.stop_sequences,
            request.logprobs_top_n,
            prefill_step_size,
            Some(&ctx.prefix_cache),
            media.as_ref(),
            start,
            on_token,
        )?;
        Ok((run, start))
    }

    /// Run one request on the GPU, on the model thread, streaming its tokens
    /// into `tx` and finishing with `Done` or `Error`. Stops early, without
    /// `Done`, when the receiver has gone.
    fn stream_on_model(
        state: &mut EngineState,
        ctx: &GpuRequestContext,
        request: GpuRequest,
        tx: &tokio::sync::mpsc::Sender<TokenEvent>,
    ) {
        let prompt_tokens = request.input_ids.len();
        let run = Self::generate_on_model(state, ctx, request, |token, logprob| {
            if tx
                .blocking_send(TokenEvent::Token { id: token, logprob })
                .is_err()
            {
                StepOutcome::Cancel
            } else {
                StepOutcome::Continue
            }
        });
        let (run, start) = match run {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.blocking_send(TokenEvent::Error(e.to_string()));
                return;
            }
        };
        // If the client dropped mid-stream, nothing is listening for Done.
        if run.finish_reason == "cancelled" {
            return;
        }
        let metrics = Self::build_metrics(
            start,
            prompt_tokens,
            run.completion_tokens,
            run.first_token_time_ms,
        );
        let _ = tx.blocking_send(TokenEvent::Done {
            finish_reason: run.finish_reason.to_string(),
            metrics,
            stripped_tokens: run.stripped_tokens,
        });
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
        self.generate_prompt(&PreparedPrompt::from_tokens(input_ids.to_vec()), params)
            .await
    }

    /// [`generate`](Self::generate) for a prepared prompt, which may carry
    /// images or videos (see [`prepare_chat`](Self::prepare_chat)). A prompt
    /// with media always runs on the GPU.
    pub async fn generate_prompt(
        &self,
        prompt: &PreparedPrompt,
        params: SamplingParams,
    ) -> ServeResult<(
        Vec<u32>,
        Option<Vec<TokenLogprobEntry>>,
        String,
        RequestMetrics,
    )> {
        Self::validate_params(&params, self.max_seq_len)?;

        let prompt_tokens = prompt.input_ids.len();
        let request = GpuRequest {
            input_ids: prompt.input_ids.clone(),
            gen_config: self.build_generation_config(&params),
            stop_sequences: params.stop_sequences,
            logprobs_top_n: params.logprobs_top_n,
            media: prompt.media.clone(),
        };

        // The accelerated ANE / CPU-hybrid engines keep their own models
        // (the ANE ones on their own thread), so they run on the blocking
        // pool. They can't collect logprobs, match raw-text stops or read
        // media, and they hand the request back to the GPU when they fail.
        if request.logprobs_top_n.is_none()
            && request.stop_sequences.is_empty()
            && request.media.is_none()
            && Self::backend_or_gpu(&self.backend) != PreferredGenerationBackend::Gpu
        {
            let backend = Arc::clone(&self.backend);
            let model_path = self.model_path.clone();
            let draft_path = self.ane_draft_path.clone();
            let ane_max_seq_len = self.ane_max_seq_len;
            let input_ids = request.input_ids.clone();
            let gen_config = request.gen_config.clone();
            let accelerated = tokio::task::spawn_blocking(move || {
                Self::try_accelerated_generate_blocking(
                    &backend,
                    &model_path,
                    draft_path.as_deref(),
                    &input_ids,
                    &gen_config,
                    ane_max_seq_len,
                )
            })
            .await
            .map_err(|e| ServeError::Internal(e.to_string()))??;
            if let Some((tokens, reason, metrics)) = accelerated {
                return Ok((tokens, None, reason, metrics));
            }
        }

        let ctx = self.gpu_request_context();
        let (reply, answer) = tokio::sync::oneshot::channel();
        self.model
            .submit(move |state| {
                let result =
                    Self::generate_on_model(state, &ctx, request, |_, _| StepOutcome::Continue)
                        .map(|(run, start)| {
                            let metrics = Self::build_metrics(
                                start,
                                prompt_tokens,
                                run.completion_tokens,
                                run.first_token_time_ms,
                            );
                            (
                                run.generated,
                                run.logprobs,
                                run.finish_reason.to_string(),
                                metrics,
                            )
                        });
                let _ = reply.send(result);
            })
            .map_err(|_| ServeError::ModelNotLoaded)?;
        answer
            .await
            .map_err(|_| ServeError::Internal("generation panicked".into()))?
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

        // Padded [batch, seq_max] token ids and mask, built here as plain
        // vectors; the arrays are made on the model thread.
        let mut ids_flat: Vec<i32> = vec![0; batch * seq_max];
        let mut mask_flat: Vec<f32> = vec![0.0; batch * seq_max];
        for (b, row) in tokenized.iter().enumerate() {
            for (j, &tok) in row.iter().enumerate() {
                ids_flat[b * seq_max + j] = tok as i32;
                mask_flat[b * seq_max + j] = 1.0;
            }
        }

        let (reply, answer) = tokio::sync::oneshot::channel();
        self.model
            .submit(move |state| {
                let mut embed = || -> ServeResult<Vec<Vec<f32>>> {
                    let ids = Array::from_slice(&ids_flat, &[batch as i32, seq_max as i32]);
                    let mask = Array::from_slice(&mask_flat, &[batch as i32, seq_max as i32]);
                    let hidden = state
                        .model
                        .forward_hidden(&ids, None)
                        .map_err(ServeError::Model)?;
                    // CLS and last-token pooling keep the hidden states' dtype.
                    let pooled = pmetal_models::pooling::pool(&hidden, &mask, mode)
                        .map_err(ServeError::Model)?
                        .as_type::<f32>();
                    pooled.try_eval().map_err(|e| {
                        ServeError::Model(pmetal_bridge::compat::Exception::custom(e.to_string()))
                    })?;
                    let hidden_dim = pooled.dim(1) as usize;
                    let flat: Vec<f32> = pooled.as_slice::<f32>().to_vec();
                    Ok((0..batch)
                        .map(|b| flat[b * hidden_dim..(b + 1) * hidden_dim].to_vec())
                        .collect())
                };
                let _ = reply.send(embed());
            })
            .map_err(|_| ServeError::ModelNotLoaded)?;
        answer
            .await
            .map_err(|_| ServeError::Internal("embedding panicked".into()))?
    }

    /// Begin token-by-token streaming generation.
    ///
    /// Validates `params` before dispatching. If validation fails, sends a
    /// single `TokenEvent::Error` through the channel and returns immediately.
    ///
    /// Queues the request on the model thread (or, for an accelerated ANE /
    /// CPU-hybrid backend, runs it on the blocking pool) and returns the
    /// receiver end immediately so the route handler can start consuming
    /// events while generation proceeds in parallel.
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
        self.generate_streaming_prompt(&PreparedPrompt::from_tokens(input_ids.to_vec()), params)
    }

    /// [`generate_streaming`](Self::generate_streaming) for a prepared
    /// prompt, which may carry images or videos (see
    /// [`prepare_chat`](Self::prepare_chat)). A prompt with media always runs
    /// on the GPU, on the single-request path.
    pub fn generate_streaming_prompt(
        &self,
        prompt: &PreparedPrompt,
        params: SamplingParams,
    ) -> tokio::sync::mpsc::Receiver<TokenEvent> {
        // Channel capacity: keep a small buffer so the generation thread is
        // never stalled waiting for the HTTP layer to consume events, but
        // don't allocate an unbounded queue.
        let (tx, rx) = tokio::sync::mpsc::channel::<TokenEvent>(64);

        // Validate before dispatching — send error through channel if invalid.
        if let Err(e) = Self::validate_params(&params, self.max_seq_len) {
            let _ = tx.try_send(TokenEvent::Error(e.to_string()));
            return rx;
        }

        let request = GpuRequest {
            input_ids: prompt.input_ids.clone(),
            gen_config: self.build_generation_config(&params),
            stop_sequences: params.stop_sequences,
            logprobs_top_n: params.logprobs_top_n,
            media: prompt.media.clone(),
        };
        let accelerable = request.stop_sequences.is_empty()
            && request.media.is_none()
            && Self::backend_or_gpu(&self.backend) != PreferredGenerationBackend::Gpu;
        let input_ids = request.input_ids.clone();
        let gen_config = request.gen_config.clone();

        let ctx = self.gpu_request_context();
        let model = self.model.clone();
        let tx_gpu = tx.clone();
        let on_gpu = move |tx: tokio::sync::mpsc::Sender<TokenEvent>| {
            let queued = model.submit(move |state| {
                Self::stream_on_model(state, &ctx, request, &tx_gpu);
            });
            if queued.is_err() {
                let _ = tx.try_send(TokenEvent::Error("the model thread has exited".into()));
            }
        };

        if !accelerable {
            on_gpu(tx);
            return rx;
        }
        // The accelerated ANE / CPU-hybrid engines keep their own models, so
        // they run on the blocking pool; a request they fail goes to the GPU.
        let backend = Arc::clone(&self.backend);
        let model_path = self.model_path.clone();
        let draft_path = self.ane_draft_path.clone();
        let ane_max_seq_len = self.ane_max_seq_len;
        tokio::task::spawn_blocking(move || {
            if !Self::try_accelerated_streaming_blocking(
                &backend,
                &model_path,
                draft_path.as_deref(),
                &input_ids,
                &gen_config,
                ane_max_seq_len,
                &tx,
            ) {
                on_gpu(tx);
            }
        });
        rx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_models::architectures::llama::{LlamaConfig, LlamaForCausalLM};
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

    fn tiny_llama_config() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: Some(8),
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            hidden_act: "silu".to_string(),
            tie_word_embeddings: true,
            ..Default::default()
        }
    }

    /// Which checkpoints read images, and what the 400 says for the rest.
    #[test]
    fn vision_is_detected_from_the_checkpoint() {
        let why = |config: serde_json::Value| {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
            match Vision::detect(dir.path()) {
                Vision::Unsupported(why) => why,
                Vision::Qwen3_5 { .. } | Vision::Mllama { .. } => "reads images".into(),
            }
        };
        assert_eq!(
            why(serde_json::json!({"model_type": "llama"})),
            "it is a text-only model"
        );
        assert_eq!(
            why(serde_json::json!({"model_type": "gemma4", "vision_config": {}})),
            "pmetal does not run the vision tower of gemma4 checkpoints in generation"
        );
        assert_eq!(
            why(serde_json::json!({"model_type": "mllama", "vision_config": {}})),
            "reads images"
        );
        assert!(
            why(serde_json::json!({"model_type": "qwen3_5", "text_config": {}}))
                .contains("text-only model")
        );
        let missing = tempfile::tempdir().unwrap();
        assert!(matches!(
            Vision::detect(missing.path()),
            Vision::Unsupported(why) if why == "it has no config.json"
        ));
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
        let mut config = dense_config(8192, 80, 128_256);
        config["model_type"] = serde_json::json!("qwen3");
        assert_eq!(
            select_accelerated_backend(&config, true),
            PreferredGenerationBackend::Ane
        );
    }

    /// The ANE inference engine is Qwen3-shaped: a Llama served there ran with
    /// q/k norms it doesn't have (#34).
    #[test]
    fn test_select_accelerated_backend_keeps_large_llama_off_the_ane() {
        let config = dense_config(8192, 80, 128_256);
        assert_ne!(
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
    fn test_build_generation_config_merges_request_params() {
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

        let config = build_generation_config_from_parts(&[1, 2], 32, &params);

        assert_eq!(config.max_new_tokens, 32);
        assert!(config.do_sample);
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

    #[test]
    fn continuous_cache_config_honors_cache_mode_override() {
        let override_mode = CacheMode::Quantized {
            bits: 4,
            group_size: 8,
        };
        let engine = InferenceEngine::new_with_backend(
            || {
                Ok(DynamicModel::Llama(LlamaForCausalLM::new(
                    tiny_llama_config(),
                )?))
            },
            test_tokenizer(),
            "tiny-llama".into(),
            std::env::temp_dir().as_path(),
            64,
            false,
            1024,
        )
        .unwrap()
        .with_cache_mode_override(override_mode);

        let cfg = engine.create_continuous_cache_config().unwrap();
        assert_eq!(cfg.mode, override_mode);
    }

    /// Tiny Qwen 3.5 (gated-delta-net + attention): its decode step keeps
    /// state of its own per sequence besides the caller's caches.
    const TINY_QWEN3_NEXT: &str = r#"{
        "model_type": "qwen3_next",
        "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 16, "linear_num_value_heads": 2, "linear_num_key_heads": 1,
        "linear_key_head_dim": 32, "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
        "num_experts": 0, "num_experts_per_tok": 0, "decoder_sparse_step": 1,
        "moe_intermediate_size": 16, "shared_expert_intermediate_size": 32,
        "mlp_only_layers": [], "norm_topk_prob": false, "tie_word_embeddings": true,
        "max_position_embeddings": 256, "rms_norm_eps": 1e-6, "rope_theta": 10000.0
    }"#;

    /// Tiny Granite 4.0-H (Mamba-2 + NoPE attention, routed experts plus a
    /// shared MLP): all of its recurrent state is in the caller's cache.
    const TINY_GRANITE_HYBRID: &str = r#"{
        "model_type": "granitemoehybrid",
        "vocab_size": 64, "hidden_size": 32, "intermediate_size": 16,
        "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
        "layer_types": ["mamba", "attention", "mamba", "mamba"],
        "position_embedding_type": "nope",
        "num_local_experts": 4, "num_experts_per_tok": 2, "shared_intermediate_size": 24,
        "mamba_n_heads": 8, "mamba_d_state": 8, "mamba_chunk_size": 4,
        "embedding_multiplier": 12.0, "residual_multiplier": 0.22,
        "logits_scaling": 6.0, "attention_multiplier": 0.125,
        "tie_word_embeddings": true,
        "max_position_embeddings": 256, "rms_norm_eps": 1e-5
    }"#;

    /// Continuous batching on a hybrid model gives each request exactly what
    /// it gets alone. Every slot carries its own recurrent state, so four
    /// requests through two slots, decoded in alternation and then reusing
    /// the rows, must match their single-request greedy replies token for
    /// token. Without per-slot state the recurrent layers ran stateless, and
    /// the slots shared Qwen 3.5's decode state.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn continuous_batching_on_hybrid_models_matches_single_requests() {
        type Load = fn() -> anyhow::Result<DynamicModel>;
        let models: [(&str, Load); 3] = [
            ("qwen3_next", || {
                Ok(DynamicModel::from_config(TINY_QWEN3_NEXT)?)
            }),
            ("nemotron_h", || {
                Ok(DynamicModel::NemotronH(NemotronHForCausalLM::new(
                    tiny_nemotron_h_config(),
                )?))
            }),
            ("granitemoehybrid", || {
                Ok(DynamicModel::from_config(TINY_GRANITE_HYBRID)?)
            }),
        ];
        let prompts: [&[u32]; 4] = [&[1, 2, 3, 1], &[2, 3], &[3, 1, 2, 2, 1], &[1, 1, 1]];
        for (name, load) in models {
            let engine = InferenceEngine::new_with_backend(
                load,
                test_tokenizer(),
                name.into(),
                std::env::temp_dir().as_path(),
                64,
                false,
                1024,
            )
            .unwrap();
            // Each token's log-probability as well: a tiny random model
            // can repeat one token whatever its state, but not with the same
            // probabilities.
            let params = SamplingParams {
                logprobs_top_n: Some(0),
                ..greedy(6)
            };
            let mut expected = Vec::new();
            for prompt in prompts {
                let (tokens, logprobs, ..) = engine.generate(prompt, params.clone()).await.unwrap();
                let logprobs: Vec<f32> = logprobs.unwrap().iter().map(|e| e.logprob).collect();
                expected.push((tokens, logprobs));
            }
            engine
                .enable_continuous_batching_auto(crate::continuous_batch::BatcherConfig {
                    max_slots: 2,
                    ..Default::default()
                })
                .unwrap();
            for round in 0..2 {
                let streams: Vec<_> = prompts
                    .iter()
                    .map(|prompt| engine.generate_batched(prompt, params.clone()).unwrap())
                    .collect();
                for ((stream, (want, want_logprobs)), prompt) in
                    streams.into_iter().zip(&expected).zip(prompts)
                {
                    let (got, got_logprobs) = collect_stream_with_logprobs(stream).await;
                    assert_eq!(
                        &got, want,
                        "{name}, round {round}: batched reply to {prompt:?}"
                    );
                    for (step, (g, w)) in got_logprobs.iter().zip(want_logprobs).enumerate() {
                        assert!(
                            (g - w).abs() < 1e-4,
                            "{name}, round {round}: reply to {prompt:?}, token {step} has \
                             log-probability {g} batched and {w} alone"
                        );
                    }
                }
            }
        }
    }

    async fn collect_stream_with_logprobs(
        mut rx: tokio::sync::mpsc::Receiver<TokenEvent>,
    ) -> (Vec<u32>, Vec<f32>) {
        let (mut tokens, mut logprobs) = (Vec::new(), Vec::new());
        while let Some(event) = rx.recv().await {
            match event {
                TokenEvent::Token { id, logprob } => {
                    tokens.push(id);
                    logprobs.push(logprob.expect("requested logprobs").logprob);
                }
                TokenEvent::Done { .. } => return (tokens, logprobs),
                TokenEvent::Error(e) => panic!("stream failed: {e}"),
            }
        }
        panic!("stream closed without Done");
    }

    fn greedy(max_tokens: usize) -> SamplingParams {
        SamplingParams {
            max_tokens,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            min_p: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            extra_stop_token_ids: vec![],
            stop_sequences: vec![],
            logprobs_top_n: None,
        }
    }

    async fn collect_stream(mut rx: tokio::sync::mpsc::Receiver<TokenEvent>) -> Vec<u32> {
        let mut tokens = Vec::new();
        while let Some(event) = rx.recv().await {
            match event {
                TokenEvent::Token { id, .. } => tokens.push(id),
                TokenEvent::Done { .. } => return tokens,
                TokenEvent::Error(e) => panic!("stream failed: {e}"),
            }
        }
        panic!("stream closed without Done");
    }

    /// Requests arrive on whatever thread the runtime picks, many at once and
    /// after idle gaps, and every one of them must see the same model. On
    /// tokio's blocking pool the second concurrent request evaluated the
    /// prefix-cache snapshot the first had made on another thread and failed
    /// with "There is no Stream(gpu, N) in current thread".
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn requests_from_every_thread_share_one_model() {
        let engine = Arc::new(
            InferenceEngine::new_with_backend(
                || {
                    Ok(DynamicModel::Llama(LlamaForCausalLM::new(
                        tiny_llama_config(),
                    )?))
                },
                test_tokenizer(),
                "tiny-llama".into(),
                std::env::temp_dir().as_path(),
                64,
                false,
                1024,
            )
            .unwrap(),
        );
        let prompt = [1u32, 2, 3, 1, 2];
        let (expected, ..) = engine.generate(&prompt, greedy(6)).await.unwrap();

        let mut tasks = Vec::new();
        for i in 0..8 {
            let engine = Arc::clone(&engine);
            tasks.push(tokio::spawn(async move {
                if i % 2 == 0 {
                    engine.generate(&prompt, greedy(6)).await.unwrap().0
                } else {
                    collect_stream(engine.generate_streaming(&prompt, greedy(6))).await
                }
            }));
        }
        for task in tasks {
            assert_eq!(task.await.unwrap(), expected);
        }
        let embeddings = engine
            .embed(
                &["alpha beta".to_string()],
                pmetal_models::pooling::PoolingMode::Mean,
            )
            .await
            .unwrap();
        assert_eq!(embeddings[0].len(), 32);
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
