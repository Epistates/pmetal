//! Model memory fit estimation for Apple Silicon.
//!
//! Given a model specification and device specification, estimates whether the model
//! fits in memory for inference and training, and provides estimated throughput.

use serde::{Deserialize, Serialize};

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

/// Memory and performance estimation for a model on a device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitEstimate {
    /// Overall fit level for inference.
    pub fit_level: FitLevel,
    /// Weight memory in GB.
    pub weights_gb: f64,
    /// KV cache memory in GB (at full context length).
    pub kv_cache_gb: f64,
    /// Runtime overhead in GB (MLX + Metal context + buffers).
    pub overhead_gb: f64,
    /// Total memory required for inference in GB.
    pub total_required_gb: f64,
    /// Available device memory in GB.
    pub available_gb: f64,
    /// Memory utilization percentage (0-100).
    pub utilization_pct: f64,
    /// Estimated inference tokens/second (decode).
    pub estimated_tps: f64,
    /// Training memory estimate in GB (LoRA, batch_size=1).
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
pub struct DeviceSpec {
    /// Total unified memory in GB.
    pub memory_gb: f64,
    /// Memory bandwidth in GB/s.
    pub bandwidth_gbps: f64,
    /// Whether memory is unified (always true on Apple Silicon).
    pub unified_memory: bool,
}

/// Model specification for fit estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Total parameter count in billions.
    pub params_b: f64,
    /// Quantization format string (e.g., "fp16", "Q4_K_M", "mlx-4bit").
    pub quantization: String,
    /// Maximum context/sequence length in tokens.
    pub context_length: u32,
    /// Number of KV heads (for GQA/MQA).
    pub num_kv_heads: Option<u64>,
    /// Head dimension.
    pub head_dim: Option<u64>,
    /// Number of transformer layers.
    pub num_layers: Option<u64>,
    /// Whether the model uses Mixture of Experts.
    pub is_moe: bool,
    /// Total number of experts (MoE).
    pub num_experts: Option<u64>,
    /// Number of active experts per token (MoE).
    pub active_experts: Option<u64>,
}

/// Bytes per parameter for a given quantization format.
///
/// Based on llmfit's `quant_bytes_per_param` table, tuned for Apple Silicon / MLX.
pub fn bytes_per_param(quantization: &str) -> f64 {
    match quantization.to_lowercase().as_str() {
        "fp32" | "f32" | "float32" => 4.0,
        "fp16" | "f16" | "bf16" | "bfloat16" | "float16" | "" => 2.0,
        "fp8" | "f8" | "e4m3" | "e5m2" => 1.05,
        "q8_0" | "int8" | "8bit" | "mlx-8bit" | "w8a16" => 1.05,
        "q6_k" => 0.80,
        "q5_k_m" | "q5_k_s" | "q5_0" | "q5_1" => 0.68,
        "q4_k_m" | "q4_k_s" | "q4_0" | "q4_1" | "4bit" | "mlx-4bit" | "nf4" | "awq" | "gptq"
        | "w4a16" => 0.58,
        "q3_k_m" | "q3_k_s" | "q3_k_l" => 0.48,
        "q2_k" | "q2_k_s" | "2bit" => 0.37,
        _ => 2.0, // default to fp16
    }
}

/// Estimate inference and training fit for a model on a device.
pub fn estimate_fit(model: &ModelSpec, device: &DeviceSpec) -> FitEstimate {
    let bpp = bytes_per_param(&model.quantization);
    let mut notes = Vec::new();

    // --- Weight memory ---
    let weights_gb = model.params_b * bpp;

    // --- KV cache memory ---
    // Exact formula when we have architectural details:
    //   kv_cache = 2 * num_layers * 2 * num_kv_heads * head_dim * context_length * 2_bytes / 1e9
    // Approximate formula (from llmfit) when we don't:
    //   kv_cache = 0.000008 * params_b * context_length
    let kv_cache_gb = match (model.num_layers, model.num_kv_heads, model.head_dim) {
        (Some(layers), Some(kv_heads), Some(hd)) => {
            // 2 for K and V, 2 bytes for fp16
            let kv_bytes = 2.0
                * layers as f64
                * kv_heads as f64
                * hd as f64
                * model.context_length as f64
                * 2.0;
            kv_bytes / 1e9
        }
        _ => 0.000008 * model.params_b * model.context_length as f64,
    };

    // --- Overhead ---
    // MLX runtime, Metal context, compilation cache, intermediate buffers
    let overhead_gb = 0.5;

    // --- Total inference memory ---
    let total_required_gb = weights_gb + kv_cache_gb + overhead_gb;

    // --- Fit level ---
    let utilization_pct = if device.memory_gb > 0.0 {
        (total_required_gb / device.memory_gb) * 100.0
    } else {
        100.0
    };

    let fit_level = if total_required_gb > device.memory_gb {
        FitLevel::TooLarge
    } else if utilization_pct > 80.0 {
        FitLevel::Tight
    } else {
        FitLevel::Fits
    };

    // --- Estimated tok/s ---
    // Bandwidth-bound decode: tok/s ≈ (bandwidth / model_size_per_token) * efficiency
    // For MoE: only active expert parameters matter for decode bandwidth
    let active_params_b = if model.is_moe {
        if let (Some(total), Some(active)) = (model.num_experts, model.active_experts) {
            if total > 0 {
                // Rough: non-MoE params + (active/total) * MoE params
                // Approximate MoE params as 60% of total (FFN layers)
                let moe_fraction = 0.6;
                let non_moe = model.params_b * (1.0 - moe_fraction);
                let moe_active = model.params_b * moe_fraction * (active as f64 / total as f64);
                non_moe + moe_active
            } else {
                model.params_b
            }
        } else {
            model.params_b
        }
    } else {
        model.params_b
    };

    let active_size_gb = active_params_b * bpp;
    let estimated_tps = if active_size_gb > 0.0 {
        // 0.55 efficiency factor (kernel overhead, KV cache reads, memory controller)
        (device.bandwidth_gbps / active_size_gb) * 0.55
    } else {
        0.0
    };

    // --- Training memory estimate ---
    // LoRA training memory ≈ weights + gradients(LoRA only) + optimizer states + activations + KV cache
    // Rough estimate: ~1.5x-2x inference for LoRA (much less than full fine-tuning which is 4-6x)
    // Activation memory for batch_size=1, single sequence:
    let activation_gb = if let Some(layers) = model.num_layers {
        // ~hidden_size * seq_len * num_layers * 2 bytes * 2 (forward+backward)
        // Approximate hidden_size from params: hidden ≈ sqrt(params_b * 1e9 / (12 * layers))
        let hidden_est = ((model.params_b * 1e9) / (12.0 * layers as f64)).sqrt();
        let act_bytes = hidden_est * model.context_length as f64 * layers as f64 * 4.0;
        act_bytes / 1e9
    } else {
        // Rough: ~0.5 * weights
        weights_gb * 0.5
    };

    // LoRA optimizer states are small (only rank * layers * 2 * 2 tensors)
    let lora_overhead_gb = 0.2; // Conservative estimate for rank=16

    let training_memory_gb =
        weights_gb + kv_cache_gb + activation_gb + lora_overhead_gb + overhead_gb;

    let training_utilization = if device.memory_gb > 0.0 {
        (training_memory_gb / device.memory_gb) * 100.0
    } else {
        100.0
    };

    let training_fit = if training_memory_gb > device.memory_gb {
        FitLevel::TooLarge
    } else if training_utilization > 80.0 {
        FitLevel::Tight
    } else {
        FitLevel::Fits
    };

    // --- Recommended batch size ---
    let headroom_gb = (device.memory_gb - training_memory_gb).max(0.0);
    // Each additional batch element costs roughly activation_gb more
    let per_batch_gb = activation_gb.max(0.5);
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
        let min_quant =
            suggest_quantization(model.params_b, device.memory_gb, model.context_length);
        if let Some(q) = min_quant {
            notes.push(format!("Try {q} quantization to fit on this device"));
        } else {
            notes.push("Model too large for this device at any quantization".to_string());
        }
    }
    if training_fit == FitLevel::TooLarge && fit_level != FitLevel::TooLarge {
        notes.push("Inference OK, but training requires more memory".to_string());
        notes.push("Try reducing max-seq-len or use a smaller model for training".to_string());
    }

    FitEstimate {
        fit_level,
        weights_gb,
        kv_cache_gb,
        overhead_gb,
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

/// Suggest the best quantization level that would fit the model on the device.
fn suggest_quantization(
    params_b: f64,
    available_gb: f64,
    context_length: u32,
) -> Option<&'static str> {
    let quants = [
        ("fp8", 1.05),
        ("Q8_0", 1.05),
        ("Q6_K", 0.80),
        ("Q5_K_M", 0.68),
        ("Q4_K_M", 0.58),
        ("Q3_K_M", 0.48),
        ("Q2_K", 0.37),
    ];

    let kv_cache_gb = 0.000008 * params_b * context_length as f64;
    let overhead_gb = 0.5;

    for (name, bpp) in quants {
        let total = params_b * bpp + kv_cache_gb + overhead_gb;
        if total < available_gb * 0.85 {
            return Some(name);
        }
    }
    None
}

/// Build a `ModelSpec` from a parsed config.json and optional safetensors size.
pub fn model_spec_from_config(
    config: &serde_json::Value,
    safetensors_bytes: Option<u64>,
) -> ModelSpec {
    let hidden = config["hidden_size"].as_u64().unwrap_or(0);
    let layers = config["num_hidden_layers"].as_u64().unwrap_or(0);
    let vocab = config["vocab_size"].as_u64().unwrap_or(32000);
    let intermediate = config["intermediate_size"]
        .as_u64()
        .unwrap_or(hidden.saturating_mul(4));
    let num_attention_heads = config["num_attention_heads"].as_u64();
    let num_kv_heads = config["num_key_value_heads"]
        .as_u64()
        .or(num_attention_heads);
    let head_dim = config["head_dim"].as_u64().or_else(|| {
        let nh = num_attention_heads?;
        if nh > 0 { Some(hidden / nh) } else { None }
    });

    // MoE detection
    let num_experts = config["num_local_experts"]
        .as_u64()
        .or_else(|| config["num_experts"].as_u64());
    let active_experts = config["num_experts_per_tok"]
        .as_u64()
        .or_else(|| config["num_active_experts"].as_u64());
    let is_moe = num_experts.is_some() && num_experts.unwrap_or(0) > 1;

    // Parameter estimation
    let params_b = if let Some(bytes) = safetensors_bytes {
        // For fp16 models, params ≈ bytes / 2
        // Detect quantization from config
        let has_quant = config.get("quantization_config").is_some();
        if has_quant {
            // Can't reliably infer params from quantized size, use formula
            estimate_params_from_config(
                hidden,
                layers,
                vocab,
                intermediate,
                num_experts,
                active_experts,
            )
        } else {
            (bytes as f64) / 2.0 / 1e9
        }
    } else {
        estimate_params_from_config(
            hidden,
            layers,
            vocab,
            intermediate,
            num_experts,
            active_experts,
        )
    };

    // Context length (cap at 1M tokens — no real model exceeds this)
    let context_length = config["max_position_embeddings"]
        .as_u64()
        .or_else(|| config["max_sequence_length"].as_u64())
        .or_else(|| config["seq_length"].as_u64())
        .unwrap_or(2048)
        .min(1_048_576) as u32;

    // Detect quantization
    let quantization = detect_quantization(config);

    ModelSpec {
        params_b,
        quantization,
        context_length,
        num_kv_heads,
        head_dim,
        num_layers: if layers > 0 { Some(layers) } else { None },
        is_moe,
        num_experts,
        active_experts,
    }
}

/// Estimate parameter count from config values.
fn estimate_params_from_config(
    hidden: u64,
    layers: u64,
    vocab: u64,
    intermediate: u64,
    num_experts: Option<u64>,
    _active_experts: Option<u64>,
) -> f64 {
    if hidden == 0 || layers == 0 {
        return 0.0;
    }

    let embed = vocab.saturating_mul(hidden);
    let attn = 4u64.saturating_mul(hidden).saturating_mul(hidden);
    let mlp = 3u64.saturating_mul(hidden).saturating_mul(intermediate);

    let total = if let Some(experts) = num_experts {
        if experts > 1 {
            let router = hidden.saturating_mul(experts);
            let shared_mlp = mlp;
            embed.saturating_add(
                layers.saturating_mul(
                    attn.saturating_add(experts.saturating_mul(mlp))
                        .saturating_add(shared_mlp)
                        .saturating_add(router),
                ),
            )
        } else {
            embed.saturating_add(layers.saturating_mul(attn.saturating_add(mlp)))
        }
    } else {
        embed.saturating_add(layers.saturating_mul(attn.saturating_add(mlp)))
    };

    total as f64 / 1e9
}

/// Detect quantization format from config.json.
fn detect_quantization(config: &serde_json::Value) -> String {
    if let Some(qc) = config.get("quantization_config") {
        if let Some(method) = qc.get("quant_method").and_then(|v| v.as_str()) {
            match method {
                "gptq" => return "gptq".to_string(),
                "awq" => return "awq".to_string(),
                "bitsandbytes" => {
                    if qc
                        .get("load_in_4bit")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                    {
                        return "nf4".to_string();
                    }
                    if qc
                        .get("load_in_8bit")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                    {
                        return "int8".to_string();
                    }
                }
                _ => return method.to_string(),
            }
        }
    }
    "fp16".to_string()
}

/// Detect quantization from a model ID string (heuristic).
pub fn detect_quantization_from_id(model_id: &str) -> String {
    let lower = model_id.to_lowercase();
    if lower.contains("q2_k") {
        "Q2_K".to_string()
    } else if lower.contains("q3_k") {
        "Q3_K_M".to_string()
    } else if lower.contains("q4_k") || lower.contains("q4_0") {
        "Q4_K_M".to_string()
    } else if lower.contains("q5_k") {
        "Q5_K_M".to_string()
    } else if lower.contains("q6_k") {
        "Q6_K".to_string()
    } else if lower.contains("q8_0") || lower.contains("int8") || lower.contains("8bit") {
        "Q8_0".to_string()
    } else if lower.contains("fp8") || lower.contains("f8") {
        "fp8".to_string()
    } else if lower.contains("4bit") || lower.contains("mlx-4bit") || lower.contains("w4a16") {
        "mlx-4bit".to_string()
    } else if lower.contains("gptq") {
        "gptq".to_string()
    } else if lower.contains("awq") {
        "awq".to_string()
    } else if lower.contains("gguf") {
        "Q4_K_M".to_string() // GGUF default assumption
    } else {
        "fp16".to_string()
    }
}

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

/// Format bytes as a human-readable size.
pub fn format_bytes(bytes: u64) -> String {
    let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    if gb >= 1.0 {
        format!("{:.1} GB", gb)
    } else {
        let mb = bytes as f64 / (1024.0 * 1024.0);
        format!("{:.0} MB", mb)
    }
}
