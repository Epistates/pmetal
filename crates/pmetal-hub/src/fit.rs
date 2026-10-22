//! Will this model fit on this Mac, and about how fast will it decode?
//!
//! The estimates are built from what is known about the model:
//!
//! - **Weights** are the measured size of the tensors (safetensors headers,
//!   or the Hub's per-dtype parameter counts) whenever that is known, so a
//!   4-bit checkpoint is counted as 4-bit. Only when it is not known do they
//!   fall back to parameters times a bytes-per-parameter figure for the named
//!   quantization.
//! - **KV cache** follows the attention layout in `config.json`: grouped KV
//!   heads, sliding-window layers that cap at their window, layers with no KV
//!   at all (linear attention, state-space), and compressed latent caches.
//!   It is sized at the stated [`ModelSpec::context_length`], not at the
//!   model's trained maximum.
//! - **The device** is described by its recommended Metal working set, as
//!   reported by the device, and its memory bandwidth.
//! - **Speed** is an estimate, not a measurement: decode reads the active
//!   weights and the KV cache once per token, so tokens per second is the
//!   memory bandwidth, derated by an efficiency, over those bytes.
//!
//! All sizes are decimal gigabytes (10^9 bytes), as the Hub lists them.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// How well a model fits on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FitLevel {
    /// Comfortable headroom (>20% memory free).
    Fits,
    /// Fits but tight (<20% headroom).
    Tight,
    /// Does not fit in available memory.
    TooLarge,
}

impl FitLevel {
    /// Returns a short label for display.
    pub fn label(self) -> &'static str {
        match self {
            FitLevel::Fits => "Fits",
            FitLevel::Tight => "Tight",
            FitLevel::TooLarge => "Too Large",
        }
    }
}

impl std::fmt::Display for FitLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Bytes in a gigabyte, as the Hub counts them.
const GB: f64 = 1e9;

/// The context length estimates are made for when the caller names none.
pub const DEFAULT_FIT_CONTEXT: u32 = 4096;

/// Memory and performance estimation for a model on a device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitEstimate {
    /// Overall fit level for inference.
    pub fit_level: FitLevel,
    /// Weight memory in GB.
    pub weights_gb: f64,
    /// Whether `weights_gb` is the measured size of the tensors (`true`) or
    /// parameters times a bytes-per-parameter figure (`false`).
    pub weights_measured: bool,
    /// KV cache memory in GB, at `context_length` tokens.
    pub kv_cache_gb: f64,
    /// The context length the KV cache and the training estimate are for.
    pub context_length: u32,
    /// Runtime overhead in GB (MLX + Metal context + buffers).
    pub overhead_gb: f64,
    /// Total memory required for inference in GB.
    pub total_required_gb: f64,
    /// Available device memory in GB: the recommended working set.
    pub available_gb: f64,
    /// Memory utilization percentage (0-100).
    pub utilization_pct: f64,
    /// Estimated inference tokens/second (decode). An estimate from memory
    /// bandwidth, not a measurement.
    pub estimated_tps: f64,
    /// Training memory estimate in GB (LoRA, batch size 1, activations
    /// recomputed).
    pub training_memory_gb: f64,
    /// Training fit level.
    pub training_fit: FitLevel,
    /// Recommended max batch size for training.
    pub recommended_batch_size: usize,
    /// Human-readable notes about the estimation.
    pub notes: Vec<String>,
}

/// Device hardware specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DeviceSpec {
    /// Memory the GPU may use, in GB: the device's recommended working set,
    /// not its installed RAM.
    pub memory_gb: f64,
    /// Memory bandwidth in GB/s.
    pub bandwidth_gbps: f64,
    /// Whether memory is unified (always true on Apple Silicon).
    pub unified_memory: bool,
}

impl DeviceSpec {
    /// A device from what the Metal device reports: its recommended working
    /// set in bytes (ask the device; it is not a fixed share of RAM) and its
    /// memory bandwidth in GB/s.
    pub fn from_device(working_set_bytes: u64, bandwidth_gbps: f64, unified_memory: bool) -> Self {
        Self {
            memory_gb: working_set_bytes as f64 / GB,
            bandwidth_gbps,
            unified_memory,
        }
    }
}

/// How a model's KV cache grows with context.
///
/// Per layer, a token adds [`elements_per_token`](Self::elements_per_token)
/// cached values. Full-attention layers keep every token; sliding-window layers
/// keep at most `window` of them; layers of other kinds (linear attention,
/// state-space) keep none that grow with context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct KvShape {
    /// Layers that cache every token.
    pub full_layers: u64,
    /// Layers that cache at most `window` tokens.
    pub window_layers: u64,
    /// The sliding window, in tokens.
    pub window: u32,
    /// Cached values one token adds in one layer (keys plus values).
    pub elements_per_token: u64,
}

impl KvShape {
    /// Every layer caches every token: plain grouped-query attention.
    pub fn full_attention(layers: u64, kv_heads: u64, head_dim: u64) -> Self {
        Self {
            full_layers: layers,
            window_layers: 0,
            window: 0,
            elements_per_token: 2 * kv_heads * head_dim,
        }
    }

    /// Cache size in bytes at `context` tokens, `bytes_per_element` per value.
    pub fn bytes(&self, context: u32, bytes_per_element: f64) -> f64 {
        let per_token = self.elements_per_token as f64 * bytes_per_element;
        let full = self.full_layers as f64 * per_token * f64::from(context);
        let windowed = self.window_layers as f64 * per_token * f64::from(context.min(self.window));
        full + windowed
    }
}

/// Model specification for fit estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelSpec {
    /// Total parameter count in billions.
    pub params_b: f64,
    /// Quantization format string (e.g., "fp16", "Q4_K_M", "mlx-4bit"). Only
    /// used to size the weights when `weight_bytes` is unknown.
    pub quantization: String,
    /// Measured size of the weights in bytes, when known. Beats
    /// `params_b * bytes-per-parameter`.
    pub weight_bytes: Option<u64>,
    /// The context length to estimate for, in tokens.
    pub context_length: u32,
    /// The longest context the model was trained for, when known.
    pub max_context_length: Option<u32>,
    /// KV cache layout, when known; otherwise a typical layout for the size.
    pub kv: Option<KvShape>,
    /// Number of KV heads (for GQA/MQA).
    pub num_kv_heads: Option<u64>,
    /// Head dimension.
    pub head_dim: Option<u64>,
    /// Number of transformer layers.
    pub num_layers: Option<u64>,
    /// Hidden size (sizes the training activations).
    pub hidden_size: Option<u64>,
    /// Vocabulary size (sizes the training logits).
    pub vocab_size: Option<u64>,
    /// Whether the model uses Mixture of Experts.
    pub is_moe: bool,
    /// Total number of experts (MoE).
    pub num_experts: Option<u64>,
    /// Number of active experts per token (MoE).
    pub active_experts: Option<u64>,
    /// Parameters used per token in billions (MoE), when known.
    pub active_params_b: Option<f64>,
    /// KV cache quantization bits (None = fp16, Some(8) = q8, Some(4) = q4).
    pub kv_cache_bits: Option<u8>,
}

impl ModelSpec {
    /// A model of `params_b` billion parameters in `quantization`, estimated at
    /// [`DEFAULT_FIT_CONTEXT`] with nothing else known.
    pub fn new(params_b: f64, quantization: impl Into<String>) -> Self {
        Self {
            params_b,
            quantization: quantization.into(),
            weight_bytes: None,
            context_length: DEFAULT_FIT_CONTEXT,
            max_context_length: None,
            kv: None,
            num_kv_heads: None,
            head_dim: None,
            num_layers: None,
            hidden_size: None,
            vocab_size: None,
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_params_b: None,
            kv_cache_bits: None,
        }
    }

    /// Estimate for `context_length` tokens.
    #[must_use]
    pub fn with_context(mut self, context_length: u32) -> Self {
        self.context_length = context_length;
        self
    }

    /// Use the measured size of the weights.
    #[must_use]
    pub fn with_weight_bytes(mut self, bytes: u64) -> Self {
        self.weight_bytes = Some(bytes);
        self
    }

    /// Quantize the KV cache to `bits` (8, 4 or 2).
    #[must_use]
    pub fn with_kv_cache_bits(mut self, bits: Option<u8>) -> Self {
        self.kv_cache_bits = bits;
        self
    }

    /// Mark the model as a mixture of experts using `active_params_b` billion
    /// parameters per token.
    #[must_use]
    pub fn with_active_params(mut self, active_params_b: f64) -> Self {
        self.is_moe = true;
        self.active_params_b = Some(active_params_b);
        self
    }
}

/// Bytes per parameter for a given quantization format, tuned for Apple Silicon / MLX.
///
/// Thin re-export of [`pmetal_core::constants::bytes_per_param`]; kept here
/// so downstream callers that `use pmetal_hub::fit::bytes_per_param` don't
/// have to change paths.
pub use pmetal_core::constants::bytes_per_param;

/// Fraction of the peak memory bandwidth decode sustains for a dense model.
/// Decode on Apple silicon is memory-bound and reaches most of the roofline;
/// this is deliberately a little below what is typically measured.
const DECODE_EFFICIENCY_DENSE: f64 = 0.75;
/// The same for a mixture of experts, whose expert gathers are less regular.
const DECODE_EFFICIENCY_MOE: f64 = 0.55;
/// MLX runtime, Metal context, compilation cache and intermediate buffers.
const RUNTIME_OVERHEAD_GB: f64 = 0.5;
/// Utilization above which a fit is called tight.
const TIGHT_UTILIZATION: f64 = 0.80;
/// Adapter weights, gradients and Adam state, in bytes per base-model
/// parameter: rank-16 adapters on all linear layers are about 0.5% of the
/// parameters, each held as 2 bytes (weight) + 2 (gradient) + 8 (optimizer state).
const LORA_STATE_BYTES_PER_PARAM: f64 = 0.005 * 12.0;

/// KV bytes per token when the layout is unknown: about 130 KB for an 8B model
/// with grouped KV heads, growing with the square root of the size.
fn assumed_kv_bytes_per_token(params_b: f64) -> f64 {
    130_000.0 * (params_b / 8.0).sqrt().clamp(0.7, 4.0)
}

/// Estimate inference and training fit for a model on a device.
pub fn estimate_fit(model: &ModelSpec, device: &DeviceSpec) -> FitEstimate {
    let mut notes = Vec::new();
    let bpp = bytes_per_param(&model.quantization);

    // --- Weights ---
    let weights_measured = model.weight_bytes.is_some();
    let weight_bytes = model
        .weight_bytes
        .map_or(model.params_b * 1e9 * bpp, |b| b as f64);
    let weights_gb = weight_bytes / GB;

    // --- KV cache, at the stated context ---
    let kv_element_bytes = pmetal_core::constants::kv_bytes_per_element(model.kv_cache_bits);
    let kv_shape = model.kv.or_else(|| {
        Some(KvShape::full_attention(
            model.num_layers?,
            model.num_kv_heads?,
            model.head_dim?,
        ))
    });
    let kv_bytes = |element_bytes: f64| match kv_shape {
        Some(shape) => shape.bytes(model.context_length, element_bytes),
        None => {
            assumed_kv_bytes_per_token(model.params_b)
                * f64::from(model.context_length)
                * (element_bytes / 2.0)
        }
    };
    let kv_cache_gb = kv_bytes(kv_element_bytes) / GB;
    if kv_shape.is_none() {
        notes.push(
            "KV cache assumes a typical layout for this size (no config.json read)".to_string(),
        );
    }
    if model
        .max_context_length
        .is_some_and(|max| model.context_length > max)
    {
        notes.push(format!(
            "Context {} is beyond what the model was trained for ({})",
            model.context_length,
            model.max_context_length.unwrap_or_default()
        ));
    }

    // --- Total inference memory and fit ---
    let total_required_gb = weights_gb + kv_cache_gb + RUNTIME_OVERHEAD_GB;
    let utilization = |required: f64| {
        if device.memory_gb > 0.0 {
            required / device.memory_gb
        } else {
            f64::INFINITY
        }
    };
    let level = |required: f64| {
        if required > device.memory_gb {
            FitLevel::TooLarge
        } else if utilization(required) > TIGHT_UTILIZATION {
            FitLevel::Tight
        } else {
            FitLevel::Fits
        }
    };
    let fit_level = level(total_required_gb);
    let utilization_pct = (utilization(total_required_gb) * 100.0).min(1e6);

    // --- Decode speed ---
    // Per token: read the active weights and the KV cache.
    let active_fraction = if model.is_moe {
        match (model.active_params_b, model.params_b) {
            (Some(active), total) if total > 0.0 => (active / total).clamp(0.0, 1.0),
            _ => match (model.num_experts, model.active_experts) {
                // Experts are roughly 60% of the weights when nothing better is known.
                (Some(total), Some(active)) if total > 0 => {
                    0.4 + 0.6 * (active as f64 / total as f64).min(1.0)
                }
                _ => 1.0,
            },
        }
    } else {
        1.0
    };
    let bytes_per_token = weight_bytes * active_fraction + kv_bytes(kv_element_bytes);
    let efficiency = if model.is_moe {
        DECODE_EFFICIENCY_MOE
    } else {
        DECODE_EFFICIENCY_DENSE
    };
    let estimated_tps = if bytes_per_token > 0.0 && device.bandwidth_gbps > 0.0 {
        device.bandwidth_gbps * GB * efficiency / bytes_per_token
    } else {
        0.0
    };

    // --- Training (LoRA, batch size 1, activations recomputed) ---
    let hidden = model.hidden_size.map(|h| h as f64).unwrap_or_else(|| {
        // hidden ≈ sqrt(params / (12 * layers)) for a dense transformer.
        let layers = model.num_layers.unwrap_or(32).max(1) as f64;
        (model.params_b * 1e9 / (12.0 * layers)).sqrt()
    });
    let layers = model.num_layers.unwrap_or(32) as f64;
    let seq = f64::from(model.context_length);
    // Layer-boundary activations for every layer, plus one layer's full
    // activations while it is recomputed, in 2-byte values.
    let activation_bytes = seq * hidden * (2.0 * layers + 34.0) * 2.0;
    // Logits are materialized in fp32 with their gradient.
    let logits_bytes = model.vocab_size.map_or(0.0, |v| seq * v as f64 * 8.0);
    let lora_bytes = weight_bytes * LORA_STATE_BYTES_PER_PARAM / bpp.max(0.25);
    let activation_gb = (activation_bytes + logits_bytes) / GB;
    let training_memory_gb = weights_gb + activation_gb + lora_bytes / GB + RUNTIME_OVERHEAD_GB;
    let training_fit = level(training_memory_gb);

    let headroom_gb = (device.memory_gb - training_memory_gb).max(0.0);
    let per_batch_gb = activation_gb.max(0.1);
    let recommended_batch_size = if training_fit == FitLevel::TooLarge {
        0
    } else {
        ((headroom_gb / per_batch_gb) as usize + 1).clamp(1, 32)
    };

    // --- Notes ---
    if model.is_moe {
        notes.push(format!(
            "MoE model: {}/{} experts active per token",
            model.active_experts.unwrap_or(0),
            model.num_experts.unwrap_or(0)
        ));
    }
    if fit_level == FitLevel::Tight {
        notes.push("Memory is tight — close other apps for stability".to_string());
    }
    if fit_level == FitLevel::TooLarge {
        match suggest_quantization(model, device) {
            Some(q) => notes.push(format!("Try {q} quantization to fit on this device")),
            None => notes.push("Model too large for this device at any quantization".to_string()),
        }
    }
    if training_fit == FitLevel::TooLarge && fit_level != FitLevel::TooLarge {
        notes.push("Inference OK, but training requires more memory".to_string());
        notes.push("Try reducing max-seq-len or use a smaller model for training".to_string());
    }
    if let Some(kv_bits) = model.kv_cache_bits {
        let saved = (kv_bytes(2.0) / GB) - kv_cache_gb;
        if saved > 0.1 {
            notes.push(format!("KV cache q{kv_bits}: saves {saved:.1} GB vs fp16"));
        }
    } else if fit_level != FitLevel::Fits && kv_cache_gb > 0.2 {
        notes.push("Tip: KV cache q8_0 saves ~50% KV memory at <0.4% quality loss".to_string());
    }

    FitEstimate {
        fit_level,
        weights_gb,
        weights_measured,
        kv_cache_gb,
        context_length: model.context_length,
        overhead_gb: RUNTIME_OVERHEAD_GB,
        total_required_gb,
        available_gb: device.memory_gb,
        utilization_pct,
        estimated_tps,
        training_memory_gb,
        training_fit,
        recommended_batch_size,
        notes,
    }
}

/// The best quantization (highest quality first) that would fit the model on
/// the device with 15% of its memory to spare.
fn suggest_quantization(model: &ModelSpec, device: &DeviceSpec) -> Option<&'static str> {
    const CANDIDATES: [&str; 7] = ["fp8", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"];
    if model.params_b <= 0.0 {
        return None;
    }
    let kv_gb = match model.kv.or_else(|| {
        Some(KvShape::full_attention(
            model.num_layers?,
            model.num_kv_heads?,
            model.head_dim?,
        ))
    }) {
        Some(shape) => {
            shape.bytes(
                model.context_length,
                pmetal_core::constants::kv_bytes_per_element(model.kv_cache_bits),
            ) / GB
        }
        None => assumed_kv_bytes_per_token(model.params_b) * f64::from(model.context_length) / GB,
    };
    CANDIDATES.into_iter().find(|name| {
        let total = model.params_b * bytes_per_param(name) + kv_gb + RUNTIME_OVERHEAD_GB;
        total < device.memory_gb * 0.85
    })
}

// ---------------------------------------------------------------------------
// Building a ModelSpec from a config.json
// ---------------------------------------------------------------------------

/// A `config.json` whose architecture fields may sit at the top level or, for
/// multimodal wrappers, under `text_config`.
struct Config<'a> {
    root: &'a Value,
    text: Option<&'a Value>,
}

impl<'a> Config<'a> {
    fn new(root: &'a Value) -> Self {
        Self {
            root,
            text: root.get("text_config").filter(|t| t.is_object()),
        }
    }

    fn get(&self, key: &str) -> Option<&'a Value> {
        self.text
            .and_then(|t| t.get(key))
            .or_else(|| self.root.get(key))
            .filter(|v| !v.is_null())
    }

    fn u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(Value::as_u64)
    }

    fn str(&self, key: &str) -> Option<&'a str> {
        self.get(key).and_then(Value::as_str)
    }

    fn bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(Value::as_bool)
    }
}

/// Parameter counts, total and used per token, from architecture fields.
///
/// Covers decoder transformers with grouped-query or latent attention, dense
/// or mixture-of-experts feed-forward blocks. It is a formula, not a count:
/// where the real tensors are known, their size is used instead and this only
/// supplies the ratio of active to total.
fn count_params(cfg: &Config<'_>) -> Option<(f64, f64)> {
    let hidden = cfg.u64("hidden_size")? as f64;
    let layers = cfg.u64("num_hidden_layers")? as f64;
    if hidden <= 0.0 || layers <= 0.0 {
        return None;
    }
    let vocab = cfg.u64("vocab_size").unwrap_or(32_000) as f64;
    let heads = cfg.u64("num_attention_heads").unwrap_or(1).max(1) as f64;
    let kv_heads = cfg.u64("num_key_value_heads").map_or(heads, |k| k as f64);
    let head_dim = cfg.u64("head_dim").map_or(hidden / heads, |d| d as f64);

    // Attention per layer.
    let attention = if let Some(kv_rank) = cfg.u64("kv_lora_rank") {
        // Latent attention: low-rank projections for queries and keys/values.
        let rope = cfg.u64("qk_rope_head_dim").unwrap_or(0) as f64;
        let nope = cfg.u64("qk_nope_head_dim").unwrap_or(head_dim as u64) as f64;
        let v_dim = cfg.u64("v_head_dim").unwrap_or(nope as u64) as f64;
        let kv_rank = kv_rank as f64;
        let q = match cfg.u64("q_lora_rank") {
            Some(r) => hidden * r as f64 + r as f64 * heads * (nope + rope),
            None => hidden * heads * (nope + rope),
        };
        q + hidden * (kv_rank + rope) + kv_rank * heads * (nope + v_dim) + heads * v_dim * hidden
    } else {
        hidden * heads * head_dim * 2.0 + hidden * kv_heads * head_dim * 2.0
    };

    // Feed-forward per layer: gated (three matrices) as in current decoders.
    let dense_inter = cfg
        .u64("intermediate_size")
        .map_or(hidden * 4.0, |i| i as f64);
    let dense_mlp = 3.0 * hidden * dense_inter;
    let experts = cfg
        .u64("num_local_experts")
        .or_else(|| cfg.u64("num_experts"))
        .or_else(|| cfg.u64("n_routed_experts"))
        .filter(|e| *e > 1);
    let (moe_layers, expert_mlp, shared_mlp, router, per_token_experts) = match experts {
        Some(e) => {
            let e = e as f64;
            let expert_inter = cfg
                .u64("moe_intermediate_size")
                .map_or(dense_inter, |i| i as f64);
            let shared = cfg
                .u64("n_shared_experts")
                .map_or(0.0, |n| n as f64 * 3.0 * hidden * expert_inter)
                + cfg
                    .u64("shared_expert_intermediate_size")
                    .map_or(0.0, |i| 3.0 * hidden * i as f64);
            let dense_layers = (cfg.u64("first_k_dense_replace").unwrap_or(0) as f64).min(layers);
            let per_token = cfg
                .u64("num_experts_per_tok")
                .or_else(|| cfg.u64("num_activated_experts"))
                .map_or(e, |k| (k as f64).min(e));
            (
                layers - dense_layers,
                e * 3.0 * hidden * expert_inter,
                shared,
                hidden * e,
                per_token * 3.0 * hidden * expert_inter,
            )
        }
        None => (0.0, 0.0, 0.0, 0.0, 0.0),
    };
    let dense_layers = layers - moe_layers;

    let tied = cfg.bool("tie_word_embeddings").unwrap_or(false);
    let embeddings = vocab * hidden * if tied { 1.0 } else { 2.0 };
    let common = embeddings + layers * attention + dense_layers * dense_mlp;
    let total = common + moe_layers * (expert_mlp + shared_mlp + router);
    let active = common + moe_layers * (per_token_experts + shared_mlp + router);
    Some((total, active))
}

/// The KV cache layout declared by a config, or `None` when it has no
/// attention layers to read it from.
fn kv_shape_from_config(cfg: &Config<'_>) -> Option<KvShape> {
    let layers = cfg.u64("num_hidden_layers")?;
    let heads = cfg.u64("num_attention_heads")?.max(1);
    let hidden = cfg.u64("hidden_size")?;
    let head_dim = cfg.u64("head_dim").unwrap_or(hidden / heads);
    let kv_heads = cfg.u64("num_key_value_heads").unwrap_or(heads);
    let elements = match cfg.u64("kv_lora_rank") {
        // Latent attention caches one compressed vector plus the rotary part.
        Some(rank) => rank + cfg.u64("qk_rope_head_dim").unwrap_or(0),
        None => 2 * kv_heads * head_dim,
    };
    let window = cfg
        .u64("sliding_window")
        .filter(|_| cfg.bool("use_sliding_window") != Some(false))
        .map(|w| w.min(u64::from(u32::MAX)) as u32);

    // Which layers cache what, from the most explicit description available.
    let (full, windowed) = if let Some(types) = cfg.get("layer_types").and_then(Value::as_array) {
        let count = |kind: &str| types.iter().filter(|t| t.as_str() == Some(kind)).count() as u64;
        (count("full_attention"), count("sliding_attention"))
    } else if let Some(pattern) = cfg.str("hybrid_override_pattern") {
        // One character per layer; `*` is attention, the rest are state-space
        // or feed-forward layers.
        (pattern.chars().filter(|c| *c == '*').count() as u64, 0)
    } else if let Some(interval) = cfg.u64("full_attention_interval").filter(|i| *i > 0) {
        // Every `interval`-th layer is full attention; the rest are linear.
        (layers / interval, 0)
    } else if let (Some(pattern), Some(_)) =
        (cfg.u64("sliding_window_pattern").filter(|p| *p > 0), window)
    {
        // Every `pattern`-th layer is global; the rest slide.
        let global = layers / pattern;
        (global, layers - global)
    } else if window.is_some()
        && cfg
            .str("model_type")
            .is_some_and(|t| t.starts_with("gemma2"))
    {
        // Alternating local and global layers.
        (layers / 2, layers - layers / 2)
    } else if window.is_some() {
        (0, layers)
    } else {
        (layers, 0)
    };
    // A sliding layer without a stated window caches everything.
    let (full, windowed, window) = match window {
        Some(w) => (full, windowed, w),
        None => (full + windowed, 0, 0),
    };
    Some(KvShape {
        full_layers: full,
        window_layers: windowed,
        window,
        elements_per_token: elements,
    })
}

/// Build a `ModelSpec` from a parsed config.json and the measured size of the
/// weight files, when known (see [`crate::verify_model_dir`]).
///
/// The context length defaults to [`DEFAULT_FIT_CONTEXT`] (or the model's
/// maximum when that is shorter); the trained maximum is kept in
/// [`ModelSpec::max_context_length`]. Change it with [`ModelSpec::with_context`].
pub fn model_spec_from_config(config: &Value, weight_bytes: Option<u64>) -> ModelSpec {
    let cfg = Config::new(config);
    let counted = count_params(&cfg);
    let quantization = detect_quantization(config);

    // Parameters: the formula where it applies; failing that, the weights'
    // size at the stated precision.
    let params_b = counted.map(|(total, _)| total / 1e9).unwrap_or_else(|| {
        weight_bytes
            .map(|b| b as f64 / bytes_per_param(&quantization) / 1e9)
            .unwrap_or(0.0)
    });
    let experts = cfg
        .u64("num_local_experts")
        .or_else(|| cfg.u64("num_experts"))
        .or_else(|| cfg.u64("n_routed_experts"));
    let active_experts = cfg
        .u64("num_experts_per_tok")
        .or_else(|| cfg.u64("num_activated_experts"))
        .or_else(|| cfg.u64("num_active_experts"));
    let is_moe = experts.is_some_and(|e| e > 1);

    let max_context = cfg
        .u64("max_position_embeddings")
        .or_else(|| cfg.u64("max_sequence_length"))
        .or_else(|| cfg.u64("seq_length"))
        .map(|c| c.min(1_048_576) as u32);
    let context_length = max_context.map_or(DEFAULT_FIT_CONTEXT, |m| m.min(DEFAULT_FIT_CONTEXT));

    let layers = cfg.u64("num_hidden_layers").filter(|l| *l > 0);
    let heads = cfg.u64("num_attention_heads");
    let hidden = cfg.u64("hidden_size");
    ModelSpec {
        params_b,
        quantization,
        weight_bytes,
        context_length,
        max_context_length: max_context,
        kv: kv_shape_from_config(&cfg),
        num_kv_heads: cfg.u64("num_key_value_heads").or(heads),
        head_dim: cfg
            .u64("head_dim")
            .or_else(|| Some(hidden? / heads?.max(1))),
        num_layers: layers,
        hidden_size: hidden,
        vocab_size: cfg.u64("vocab_size"),
        is_moe,
        num_experts: experts,
        active_experts,
        active_params_b: counted.filter(|_| is_moe).map(|(_, active)| active / 1e9),
        kv_cache_bits: None, // Set by caller based on inference config
    }
}

/// Build a `ModelSpec` for a model directory: the architecture from its
/// `config.json` and the measured size of the weight files it holds. `None`
/// when there is no readable `config.json` (a LoRA adapter directory, say).
pub fn model_spec_from_dir(dir: &std::path::Path) -> Option<ModelSpec> {
    let config: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
    let report = crate::verify_model_dir(dir);
    Some(model_spec_from_config(
        &config,
        (report.weight_bytes > 0).then_some(report.weight_bytes),
    ))
}

/// Detect quantization format from config.json.
fn detect_quantization(config: &Value) -> String {
    let cfg = Config::new(config);
    if let Some(qc) = cfg.get("quantization_config") {
        let bits = qc.get("bits").and_then(Value::as_u64);
        match qc.get("quant_method").and_then(Value::as_str) {
            Some("bitsandbytes") => {
                if qc.get("load_in_4bit").and_then(Value::as_bool) == Some(true) {
                    return "nf4".to_string();
                }
                if qc.get("load_in_8bit").and_then(Value::as_bool) == Some(true) {
                    return "int8".to_string();
                }
            }
            Some(method @ ("gptq" | "awq")) => return method.to_string(),
            Some("fp8") => return "fp8".to_string(),
            Some(method) => return bits_label(bits).unwrap_or(method).to_string(),
            None => {}
        }
    }
    // MLX-style `quantization: {bits, group_size}`.
    if let Some(bits) = cfg
        .get("quantization")
        .and_then(|q| q.get("bits"))
        .and_then(Value::as_u64)
    {
        if let Some(label) = bits_label(Some(bits)) {
            return label.to_string();
        }
    }
    "fp16".to_string()
}

/// A bytes-per-parameter label for a quantization bit width.
fn bits_label(bits: Option<u64>) -> Option<&'static str> {
    Some(match bits? {
        2 => "2bit",
        3 => "Q3_K_M",
        4 => "mlx-4bit",
        5 => "Q5_K_M",
        6 => "Q6_K",
        8 => "mlx-8bit",
        _ => return None,
    })
}

/// Detect quantization from a model ID string (heuristic).
///
/// Thin re-export of [`pmetal_core::constants::detect_quantization_from_id`];
/// kept here so downstream callers that `use pmetal_hub::fit::detect_quantization_from_id`
/// (search + TUI + MCP + CLI) don't have to change paths.
pub use pmetal_core::constants::detect_quantization_from_id;

/// Format a parameter count for display.
pub fn format_params(params_b: f64) -> String {
    if params_b >= 1.0 {
        format!("{:.1}B", params_b)
    } else if params_b >= 0.001 {
        format!("{:.0}M", params_b * 1000.0)
    } else {
        "?".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn device(memory_gb: f64) -> DeviceSpec {
        DeviceSpec {
            memory_gb,
            bandwidth_gbps: 400.0,
            unified_memory: true,
        }
    }

    fn billions(config: Value) -> (f64, f64) {
        let (total, active) = count_params(&Config::new(&config)).unwrap();
        (total / 1e9, active / 1e9)
    }

    fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{what}: {actual:.3} vs {expected:.3}"
        );
    }

    #[test]
    fn parameter_counts_match_published_sizes() {
        // A dense model with grouped KV heads and tied embeddings: 1.24B.
        let (total, active) = billions(json!({
            "hidden_size": 2048, "num_hidden_layers": 16, "intermediate_size": 8192,
            "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 64,
            "vocab_size": 128256, "tie_word_embeddings": true,
        }));
        close(total, 1.236, 0.01, "dense total");
        assert_eq!(total, active);

        // A sparse MoE with 128 experts, 8 active: 30.5B total, 3.3B active.
        let (total, active) = billions(json!({
            "hidden_size": 2048, "num_hidden_layers": 48, "intermediate_size": 6144,
            "moe_intermediate_size": 768, "num_experts": 128, "num_experts_per_tok": 8,
            "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
            "vocab_size": 151936, "tie_word_embeddings": false,
        }));
        close(total, 30.53, 0.1, "moe total");
        close(active, 3.35, 0.1, "moe active");

        // 8 experts, top 2, expert width = intermediate size: 46.7B, 12.9B.
        let (total, active) = billions(json!({
            "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336,
            "num_local_experts": 8, "num_experts_per_tok": 2,
            "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 32000,
        }));
        close(total, 46.7, 0.2, "mixtral total");
        close(active, 12.9, 0.2, "mixtral active");
    }

    #[test]
    fn kv_cache_follows_the_attention_layout() {
        let spec = |config: Value, ctx: u32| {
            let s = model_spec_from_config(&config, None).with_context(ctx);
            s.kv.expect("kv layout").bytes(ctx, 2.0)
        };
        // Grouped-query attention: 2 * layers * kv_heads * head_dim * ctx * 2 B.
        let gqa = json!({"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
                         "num_key_value_heads": 8});
        assert_eq!(
            spec(gqa, 131_072),
            2.0 * 32.0 * 8.0 * 128.0 * 131_072.0 * 2.0
        );

        // Sliding layers cap at their window; one in six is global.
        let sliding = json!({"hidden_size": 2560, "num_hidden_layers": 36, "num_attention_heads": 8,
                             "num_key_value_heads": 4, "head_dim": 256, "sliding_window": 1024,
                             "sliding_window_pattern": 6});
        let bytes = spec(sliding, 32_768);
        let per_token = 2.0 * 4.0 * 256.0 * 2.0;
        assert_eq!(bytes, per_token * (6.0 * 32_768.0 + 30.0 * 1024.0));

        // Layers listed by type: linear-attention layers cache nothing that grows.
        let hybrid = json!({"hidden_size": 2048, "num_hidden_layers": 4, "num_attention_heads": 16,
                            "num_key_value_heads": 2, "head_dim": 256,
                            "layer_types": ["linear_attention", "linear_attention",
                                            "linear_attention", "full_attention"]});
        assert_eq!(
            spec(hybrid, 10_000),
            1.0 * (2.0 * 2.0 * 256.0) * 2.0 * 10_000.0
        );

        // Latent attention caches one compressed vector per token and layer.
        let latent = json!({"hidden_size": 7168, "num_hidden_layers": 61, "num_attention_heads": 128,
                            "kv_lora_rank": 512, "qk_rope_head_dim": 64});
        assert_eq!(spec(latent, 1000), 61.0 * 576.0 * 2.0 * 1000.0);

        // A window the config switches off is not a window.
        let off = json!({"hidden_size": 1024, "num_hidden_layers": 2, "num_attention_heads": 8,
                         "sliding_window": 128, "use_sliding_window": false});
        assert_eq!(spec(off, 4096), 2.0 * (2.0 * 8.0 * 128.0) * 2.0 * 4096.0);
    }

    #[test]
    fn a_vision_wrapper_is_read_through_its_text_config() {
        let config = json!({"model_type": "wrapper", "text_config": {
            "hidden_size": 2048, "num_hidden_layers": 16, "intermediate_size": 8192,
            "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256}});
        let spec = model_spec_from_config(&config, None);
        close(spec.params_b, 1.5, 0.3, "params through text_config");
        assert!(spec.kv.is_some());
    }

    #[test]
    fn measured_weights_beat_the_parameter_formula() {
        let model = ModelSpec::new(8.0, "fp16").with_weight_bytes(4_500_000_000);
        let fit = estimate_fit(&model, &device(64.0));
        close(fit.weights_gb, 4.5, 1e-9, "measured weights");
        assert!(fit.weights_measured);
        let guess = estimate_fit(&ModelSpec::new(8.0, "fp16"), &device(64.0));
        close(guess.weights_gb, 16.0, 1e-9, "estimated weights");
        assert!(!guess.weights_measured);
    }

    #[test]
    fn a_packed_checkpoint_is_counted_at_its_real_size() {
        // A 4-bit checkpoint of an 8B model on disk is ~4.5 GB. The old
        // reading (parameters from the size at fp16) and the new agree on the
        // bytes; what must not happen is counting 8B * 2 bytes = 16 GB.
        let config = json!({"hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336,
                            "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256,
                            "quantization": {"bits": 4, "group_size": 64}});
        let spec = model_spec_from_config(&config, Some(4_500_000_000));
        assert_eq!(spec.quantization, "mlx-4bit");
        let fit = estimate_fit(&spec, &device(24.0));
        close(fit.weights_gb, 4.5, 1e-9, "weights");
        assert_eq!(fit.fit_level, FitLevel::Fits);
    }

    #[test]
    fn the_stated_context_sizes_the_cache_not_the_trained_maximum() {
        let config = json!({"hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336,
                            "num_attention_heads": 32, "num_key_value_heads": 8,
                            "max_position_embeddings": 131_072});
        let spec = model_spec_from_config(&config, Some(16_000_000_000));
        assert_eq!(spec.context_length, DEFAULT_FIT_CONTEXT);
        assert_eq!(spec.max_context_length, Some(131_072));
        let short = estimate_fit(&spec, &device(64.0));
        let long = estimate_fit(&spec.clone().with_context(131_072), &device(64.0));
        close(short.kv_cache_gb, 0.537, 0.01, "4k cache");
        close(long.kv_cache_gb, 17.18, 0.05, "128k cache");
        assert!(long.estimated_tps < short.estimated_tps);
    }

    #[test]
    fn decode_speed_is_bandwidth_over_bytes_per_token() {
        // 400 GB/s over a 4 GB model, no KV cache to speak of.
        let mut model = ModelSpec::new(0.0, "fp16").with_weight_bytes(4_000_000_000);
        model.kv = Some(KvShape::full_attention(0, 0, 0));
        let fit = estimate_fit(&model, &device(64.0));
        close(
            fit.estimated_tps,
            400.0 / 4.0 * DECODE_EFFICIENCY_DENSE,
            1e-6,
            "dense tps",
        );

        // A mixture of experts reads only the active share.
        let moe = ModelSpec::new(30.0, "fp16")
            .with_weight_bytes(60_000_000_000)
            .with_active_params(3.0);
        let fit = estimate_fit(&moe, &device(128.0));
        let expected = 400.0 * DECODE_EFFICIENCY_MOE / (60.0 * 0.1 + fit.kv_cache_gb);
        close(fit.estimated_tps, expected, 1e-6, "moe tps");
    }

    #[test]
    fn fit_levels_follow_the_device_memory() {
        let model = ModelSpec::new(0.0, "fp16").with_weight_bytes(10_000_000_000);
        assert_eq!(
            estimate_fit(&model, &device(64.0)).fit_level,
            FitLevel::Fits
        );
        assert_eq!(
            estimate_fit(&model, &device(12.0)).fit_level,
            FitLevel::Tight
        );
        let too_large = estimate_fit(&model, &device(8.0));
        assert_eq!(too_large.fit_level, FitLevel::TooLarge);
        assert!(
            too_large.notes.iter().any(|n| n.contains("quantization")),
            "{:?}",
            too_large.notes
        );
    }

    #[test]
    fn the_device_is_described_in_decimal_gigabytes_from_its_working_set() {
        // A 48 GiB working set is 51.5 GB; counting it as 48 would shrink the
        // device by 7% against the Hub's decimal model sizes.
        let spec = DeviceSpec::from_device(48 * 1024 * 1024 * 1024, 546.0, true);
        close(spec.memory_gb, 51.539_607_552, 1e-9, "working set");
    }

    #[test]
    fn training_estimate_ignores_the_kv_cache_and_counts_logits() {
        let mut spec = ModelSpec::new(8.0, "fp16").with_weight_bytes(16_000_000_000);
        spec.hidden_size = Some(4096);
        spec.num_layers = Some(32);
        spec.vocab_size = Some(128_256);
        let with_logits = estimate_fit(&spec, &device(128.0)).training_memory_gb;
        spec.vocab_size = None;
        let without = estimate_fit(&spec, &device(128.0)).training_memory_gb;
        // 4096 tokens * 128256 * 8 bytes = 4.2 GB.
        close(with_logits - without, 4.2, 0.05, "logits");
        // Growing the context grows activations, not just the KV cache.
        let longer =
            estimate_fit(&spec.clone().with_context(16_384), &device(128.0)).training_memory_gb;
        assert!(longer > without + 1.0);
    }

    #[test]
    fn quantization_is_read_from_both_config_styles() {
        for (config, expected) in [
            (json!({}), "fp16"),
            (
                json!({"quantization": {"bits": 4, "group_size": 64}}),
                "mlx-4bit",
            ),
            (json!({"quantization": {"bits": 8}}), "mlx-8bit"),
            (
                json!({"quantization_config": {"quant_method": "gptq", "bits": 4}}),
                "gptq",
            ),
            (
                json!({"quantization_config": {"quant_method": "awq"}}),
                "awq",
            ),
            (
                json!({"quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": true}}),
                "nf4",
            ),
            (
                json!({"quantization_config": {"quant_method": "fp8"}}),
                "fp8",
            ),
            (
                json!({"quantization_config": {"quant_method": "compressed-tensors", "bits": 8}}),
                "mlx-8bit",
            ),
        ] {
            assert_eq!(detect_quantization(&config), expected, "{config}");
        }
    }

    #[test]
    fn parameter_counts_are_formatted_in_billions_and_millions() {
        assert_eq!(format_params(7.6), "7.6B");
        assert_eq!(format_params(0.6), "600M");
    }
}
