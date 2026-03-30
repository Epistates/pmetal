//! Standalone GPT-OSS inference engine — zero dependency on mlx-rs or pmetal-models.
//!
//! GPT-OSS is OpenAI's first Apache-2.0 open-weight model (Aug 2025).  Available
//! in 20B and 120B variants, it uses:
//!   - Mixture of Experts (MoE) with top-k sigmoid routing and per-expert bias
//!   - Alternating sliding window (128 tok) and full-context attention patterns
//!   - GPT-OSS SwiGLU: `x_glu * sigmoid(α * x_glu) * (x_linear + 1)` with clamping
//!   - Grouped Multi-Query Attention (GQA), bias on q/k/v/o projections
//!   - Standard full-head RoPE (head_dim = 64)
//!   - No Q/K norm (unlike Qwen3.5)
//!
//! Every op on the hot path uses [`InlineArray`] (stack-allocated `mlx::core::array`,
//! direct C++ bridge). This eliminates ALL per-op heap allocation, matching
//! Python/nanobind's direct C++ binding performance.
//!
//! The entire stack — config, weights, caches, forward pass, generation loop —
//! lives in this single module.

use serde::Deserialize;

use crate::InlineArray;
use crate::inline_array as bridge;

// ============================================================================
// Config
// ============================================================================

fn default_model_type() -> String {
    "gpt_oss".to_string()
}
fn default_vocab_size() -> i32 {
    201088
}
fn default_hidden_size() -> i32 {
    2880
}
fn default_intermediate_size() -> i32 {
    2880
}
fn default_num_hidden_layers() -> i32 {
    24
}
fn default_num_attention_heads() -> i32 {
    64
}
fn default_num_key_value_heads() -> i32 {
    8
}
fn default_head_dim() -> i32 {
    64
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    150000.0
}
fn default_num_local_experts() -> i32 {
    32
}
fn default_experts_per_token() -> i32 {
    4
}
fn default_sliding_window() -> i32 {
    128
}
fn default_true() -> bool {
    true
}
fn default_swiglu_alpha() -> f32 {
    1.702
}
fn default_swiglu_limit() -> f32 {
    7.0
}

/// Attention layer type for GPT-OSS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AttentionLayerType {
    SlidingAttention,
    #[default]
    FullAttention,
}

/// RoPE scaling configuration (YaRN).
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScalingConfig {
    pub rope_type: String,
    pub factor: f32,
    #[serde(default)]
    pub original_max_position_embeddings: i32,
}

/// Minimal, serde-deserializable GPT-OSS config.
///
/// Only the fields required for inference are included; unknown keys are
/// silently ignored by serde.
#[derive(Debug, Clone, Deserialize)]
pub struct GptOssConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
    #[serde(default = "default_true")]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_num_local_experts")]
    pub num_local_experts: i32,
    /// Primary field for experts-per-token.
    #[serde(default = "default_experts_per_token")]
    pub experts_per_token: i32,
    /// Alternate field name used in some checkpoints.
    #[serde(default)]
    pub num_experts_per_tok: Option<i32>,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    /// Explicit per-layer type list; if empty, alternates sliding/full.
    #[serde(default)]
    pub layer_types: Vec<AttentionLayerType>,
    /// SwiGLU alpha (scaling factor, default 1.702).
    #[serde(default = "default_swiglu_alpha")]
    pub swiglu_alpha: f32,
    /// SwiGLU clamp limit (default 7.0).
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f32,
}

impl GptOssConfig {
    /// Effective experts per token.
    pub fn experts_per_tok(&self) -> i32 {
        self.num_experts_per_tok.unwrap_or(self.experts_per_token)
    }

    /// Attention type at layer index `i`.
    pub fn layer_type(&self, i: usize) -> AttentionLayerType {
        if !self.layer_types.is_empty() && i < self.layer_types.len() {
            self.layer_types[i]
        } else {
            // Default: even indices are sliding, odd are full.
            if i % 2 == 0 {
                AttentionLayerType::SlidingAttention
            } else {
                AttentionLayerType::FullAttention
            }
        }
    }

    /// Index of the first full-attention layer (used for causal-mask build).
    pub fn first_full_attn_layer(&self) -> usize {
        for i in 0..self.num_hidden_layers as usize {
            if self.layer_type(i) == AttentionLayerType::FullAttention {
                return i;
            }
        }
        0
    }

    /// Index of the first sliding-attention layer.
    pub fn first_sliding_attn_layer(&self) -> usize {
        for i in 0..self.num_hidden_layers as usize {
            if self.layer_type(i) == AttentionLayerType::SlidingAttention {
                return i;
            }
        }
        0
    }
}

/// Parse `config.json` from a model directory.
pub fn load_config(model_dir: &std::path::Path) -> Result<GptOssConfig, String> {
    let path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    // Some checkpoints nest config under "text_config"
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("failed to parse config.json: {e}"))?;
    let config_str = if json.get("text_config").is_some() {
        serde_json::to_string(&json["text_config"]).map_err(|e| e.to_string())?
    } else {
        text
    };
    serde_json::from_str(&config_str).map_err(|e| format!("failed to parse config: {e}"))
}

// ============================================================================
// Per-layer weights
// ============================================================================

/// GPT-OSS layer weights — attention + MoE.
struct LayerWeights {
    // Layer norms
    input_ln_w: InlineArray,
    input_ln_eps: f32,
    post_ln_w: InlineArray,
    post_ln_eps: f32,

    // Attention projections (pre-transposed [in, out] for direct matmul)
    attn_q_w: InlineArray,         // [hidden, n_heads * head_dim]
    attn_q_b: Option<InlineArray>, // [n_heads * head_dim]
    attn_k_w: InlineArray,         // [hidden, n_kv_heads * head_dim]
    attn_k_b: Option<InlineArray>, // [n_kv_heads * head_dim]
    attn_v_w: InlineArray,         // [hidden, n_kv_heads * head_dim]
    attn_v_b: Option<InlineArray>, // [n_kv_heads * head_dim]
    attn_o_w: InlineArray,         // [n_heads * head_dim, hidden]
    attn_o_b: Option<InlineArray>, // [hidden]

    // Attention dims
    attn_n_heads: i32,
    attn_n_kv_heads: i32,
    attn_head_dim: i32,
    attn_scale: f32,
    attn_rope_base: f32,
    attn_is_sliding: bool,
    attn_sliding_window: i32,

    // MoE: router + stacked expert projections
    // Router: [hidden, num_experts] (NO transpose — direct matmul hidden @ router_w)
    moe_router_w: InlineArray,
    // Stacked expert weights — shape [num_experts, hidden_size, intermediate_size]
    // pre-transposed to [num_experts, intermediate_size, hidden_size] for batched gather_mm
    // but actually stored as [num_experts, hidden, intermediate] with matmul handling transpose
    moe_gate_w: InlineArray, // [num_experts, hidden, intermediate]
    moe_gate_b: InlineArray, // [num_experts, intermediate]
    moe_up_w: InlineArray,   // [num_experts, hidden, intermediate]
    moe_up_b: InlineArray,   // [num_experts, intermediate]
    moe_down_w: InlineArray, // [num_experts, intermediate, hidden]
    moe_down_b: InlineArray, // [num_experts, hidden]

    moe_num_experts: i32,
    moe_top_k: i32,

    // SwiGLU parameters
    swiglu_alpha: f32,
    swiglu_limit: f32,
}

// ============================================================================
// Full model weights
// ============================================================================

/// All GPT-OSS model weights as InlineArray. Zero dependency on mlx-rs.
pub struct NativeWeights {
    pub embed_w: InlineArray,
    pub final_norm_w: InlineArray,
    pub final_norm_eps: f32,
    /// None when `tie_word_embeddings = true`.
    pub lm_head_w: Option<InlineArray>,
    pub tie_word_embeddings: bool,
    /// Per-layer weights — opaque to callers.
    layers: Vec<LayerWeights>,
    /// Model activation dtype (e.g., 11 = bfloat16).
    pub model_dtype: i32,
}

impl std::fmt::Debug for NativeWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeWeights")
            .field("layers", &self.layers.len())
            .field("tie_word_embeddings", &self.tie_word_embeddings)
            .field("model_dtype", &self.model_dtype)
            .finish()
    }
}

// ============================================================================
// Caches
// ============================================================================

/// KV cache for one attention layer.
///
/// GPT-OSS uses two kinds:
///   - Full attention: unbounded growth (256-token chunk reallocation strategy)
///   - Sliding attention: rotating window of `sliding_window` tokens
pub struct KvLayerCache {
    pub keys: Option<InlineArray>, // [B, H, MAX_T, D] (or [B, H, window, D] for sliding)
    pub values: Option<InlineArray>, // [B, H, MAX_T, D]
    pub offset: i32,               // total tokens written
    pub is_sliding: bool,
    pub window: i32, // sliding window size (ignored when is_sliding=false)
}

/// Full model cache — one KV entry per layer.
pub struct NativeCache {
    pub kv_caches: Vec<KvLayerCache>,
    pub rope_offset: i32,
}

impl NativeCache {
    /// Evaluate and detach all cache state arrays in one GPU submission.
    ///
    /// Must be called after the prefill forward pass and before decode.
    /// Equivalent to Python's `mx.eval([c.state for c in prompt_cache])`.
    pub fn eval_and_detach_states(&mut self) {
        let mut to_eval: Vec<&mut InlineArray> = Vec::new();
        for c in &mut self.kv_caches {
            if let Some(k) = c.keys.take() {
                let trimmed = if c.offset > 0 && c.offset < k.dim(2) {
                    k.slice(&[0, 0, 0, 0], &[k.dim(0), k.dim(1), c.offset, k.dim(3)])
                } else {
                    k
                };
                c.keys = Some(trimmed);
            }
            if let Some(v) = c.values.take() {
                let trimmed = if c.offset > 0 && c.offset < v.dim(2) {
                    v.slice(&[0, 0, 0, 0], &[v.dim(0), v.dim(1), c.offset, v.dim(3)])
                } else {
                    v
                };
                c.values = Some(trimmed);
            }
            if let Some(ref mut k) = c.keys {
                to_eval.push(k);
            }
            if let Some(ref mut v) = c.values {
                to_eval.push(v);
            }
        }
        bridge::eval_and_detach_many(&mut to_eval);
    }

    /// Create a fresh, empty cache for the given weight set.
    pub fn new_empty(weights: &NativeWeights) -> Self {
        let kv_caches = weights
            .layers
            .iter()
            .map(|lw| KvLayerCache {
                keys: None,
                values: None,
                offset: 0,
                is_sliding: lw.attn_is_sliding,
                window: lw.attn_sliding_window,
            })
            .collect();

        NativeCache {
            kv_caches,
            rope_offset: 0,
        }
    }
}

impl std::fmt::Debug for NativeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeCache")
            .field("layers", &self.kv_caches.len())
            .field("rope_offset", &self.rope_offset)
            .finish()
    }
}

// ============================================================================
// Weight loading
// ============================================================================

/// Load GPT-OSS model weights from a directory containing safetensors shards.
///
/// ## Sanitization applied
///
/// The checkpoint stores MoE expert weights in a `gate_up_proj` / `down_proj`
/// format (two projections fused) with optional MXFP4 quantization suffixes
/// (`_blocks`, `_scales`).  We split and rename:
///
///   `*.gate_up_proj_blocks`  → gate and up weight blocks (split along expert dim)
///   `*.gate_up_proj_bias`    → split into `gate_proj.bias` + `up_proj.bias`
///   `*.down_proj_blocks`     → down weight blocks
///   `*.down_proj_bias`       → `down_proj.bias`
///
/// After sanitization all expert weights are stacked into tensors of shape
/// `[num_experts, ...]` for efficient batched gather_mm during inference.
pub fn load_model(
    model_dir: &std::path::Path,
    config: &GptOssConfig,
) -> Result<NativeWeights, String> {
    // ── Step 1: Shard discovery ─────────────────────────────────────────────
    let single_path = model_dir.join("model.safetensors");
    let index_path = model_dir.join("model.safetensors.index.json");

    let shard_paths: Vec<std::path::PathBuf> = if single_path.exists() {
        vec![single_path]
    } else if index_path.exists() {
        let content = std::fs::read_to_string(&index_path)
            .map_err(|e| format!("failed to read index JSON: {e}"))?;
        let index: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("failed to parse index JSON: {e}"))?;
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "index JSON missing weight_map".to_string())?;
        let mut seen = std::collections::HashSet::new();
        let mut paths = Vec::new();
        for shard_file in weight_map.values() {
            let name = shard_file
                .as_str()
                .ok_or_else(|| "shard filename is not a string".to_string())?;
            if seen.insert(name.to_string()) {
                if name.contains("..") || name.starts_with('/') {
                    return Err(format!("shard filename contains path traversal: {name}"));
                }
                paths.push(model_dir.join(name));
            }
        }
        paths
    } else {
        return Err(format!(
            "no model.safetensors or model.safetensors.index.json in {}",
            model_dir.display()
        ));
    };

    // ── Step 2: Load all tensors from all shards ────────────────────────────
    let mut raw: std::collections::HashMap<String, InlineArray> = std::collections::HashMap::new();

    for shard_path in &shard_paths {
        let path_str = shard_path
            .to_str()
            .ok_or_else(|| format!("non-UTF-8 shard path: {:?}", shard_path))?;
        let entries = bridge::load_safetensors_shard(path_str)
            .ok_or_else(|| format!("failed to load shard: {path_str}"))?;
        for (key, arr) in entries {
            raw.insert(key, arr);
        }
    }

    if raw.is_empty() {
        return Err(format!("no weights loaded from {}", model_dir.display()));
    }

    // ── Step 3: Sanitization — mirror Python's Model.sanitize() ────────────
    //
    // The checkpoint may store weights in one of two states:
    //   (a) Already-sanitized: gate_proj.weight / up_proj.weight present.
    //   (b) Fused gate_up_proj: must be split; may have _blocks/_scales quantization
    //       suffixes for MXFP4 weights.
    //
    // We detect state by checking for any "gate_proj.weight" key; if absent we
    // perform the full sanitization pass.

    let already_sanitized = raw.keys().any(|k| k.ends_with("gate_proj.weight"));

    if !already_sanitized {
        let original_keys: Vec<String> = raw.keys().cloned().collect();
        let mut new_entries: Vec<(String, InlineArray)> = Vec::new();

        for k in &original_keys {
            if k.contains("gate_up_proj") && !k.contains("bias") {
                // Could be: gate_up_proj_blocks, gate_up_proj_scales, gate_up_proj.weight
                // Handle quantized (_blocks, _scales) and dense (.weight) variants.

                if let Some(v) = raw.remove(k) {
                    let (base_key, array, suffix) = if k.contains("_blocks") {
                        // MXFP4 blocks — flatten the last two dims: view as uint32, flatten(-2)
                        // Python: v.view(mx.uint32).flatten(-2)
                        // For now we pass through as-is; quantized_matmul handles _blocks/_scales.
                        let flat = flatten_mxfp4_blocks(&v);
                        let bk = k.replace("_blocks", ".weight");
                        (bk, flat, ".weight")
                    } else if k.contains("_scales") {
                        let scaled = v.clone();
                        let bk = k.replace("_scales", ".scales");
                        (bk, scaled, ".scales")
                    } else {
                        (k.clone(), v, ".weight")
                    };

                    // Split the fused gate_up tensor along the expert/output dim at index 1:
                    //   gate: even rows (::2), up: odd rows (1::2) along the second-to-last axis.
                    // For stacked expert weights shape [num_experts, out*2, hidden] or flat [out*2, hidden].
                    // We split the gate_up into two halves.
                    let (gate_arr, up_arr) = split_gate_up(&array, suffix);

                    // Produce two keys by replacing gate_up_proj with gate_proj / up_proj
                    let gate_key = base_key.replace("gate_up_proj", "gate_proj");
                    let up_key = base_key.replace("gate_up_proj", "up_proj");

                    new_entries.push((gate_key, gate_arr));
                    new_entries.push((up_key, up_arr));
                }
            } else if k.contains("gate_up_proj_bias") {
                if let Some(v) = raw.remove(k) {
                    let (gate_b, up_b) = split_gate_up_bias(&v);
                    new_entries.push((k.replace("gate_up_proj_bias", "gate_proj.bias"), gate_b));
                    new_entries.push((k.replace("gate_up_proj_bias", "up_proj.bias"), up_b));
                }
            } else if k.contains("down_proj") && !k.contains("bias") {
                if let Some(v) = raw.remove(k) {
                    let (final_key, final_v) = if k.contains("_blocks") {
                        let flat = flatten_mxfp4_blocks(&v);
                        (k.replace("_blocks", ".weight"), flat)
                    } else if k.contains("_scales") {
                        (k.replace("_scales", ".scales"), v)
                    } else {
                        (k.clone(), v)
                    };
                    new_entries.push((final_key, final_v));
                }
            } else if k.contains("down_proj_bias") {
                if let Some(v) = raw.remove(k) {
                    new_entries.push((k.replace("down_proj_bias", "down_proj.bias"), v));
                }
            }
        }

        for (k, v) in new_entries {
            raw.insert(k, v);
        }
    }

    // Drop lm_head when embeddings are tied.
    if config.tie_word_embeddings {
        raw.remove("lm_head.weight");
    }

    // ── Step 4: Build per-layer weight structs ──────────────────────────────

    let get = |key: &str| -> Result<InlineArray, String> {
        raw.get(key).cloned().ok_or_else(|| {
            let parts: Vec<&str> = key.rsplitn(2, '.').collect();
            let suffix = parts[0];
            let close: Vec<&String> = raw.keys().filter(|k| k.ends_with(suffix)).take(5).collect();
            format!("missing weight key: {key} (close matches: {close:?})")
        })
    };
    let get_opt = |key: &str| -> Option<InlineArray> { raw.get(key).cloned() };

    let embed_w = get("model.embed_tokens.weight")?;
    let final_norm_w = get("model.norm.weight")?;
    let final_norm_eps = config.rms_norm_eps;
    let lm_head_w = if config.tie_word_embeddings {
        None
    } else {
        // lm_head.weight stored as [vocab, hidden]; pre-transpose to [hidden, vocab]
        Some(get("lm_head.weight")?.t())
    };

    let model_dtype = embed_w.dtype_raw();

    let n_heads = config.num_attention_heads;
    let n_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let attn_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let rope_base = config.rope_theta;
    let n_experts = config.num_local_experts;
    let top_k = config.experts_per_tok();
    let use_bias = config.attention_bias;

    let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);

    for li in 0..config.num_hidden_layers as usize {
        let p = format!("model.layers.{li}");
        let sa = format!("{p}.self_attn");
        let mlp = format!("{p}.mlp");
        let layer_type = config.layer_type(li);
        let is_sliding = layer_type == AttentionLayerType::SlidingAttention;

        // Layer norms
        let input_ln_w = get(&format!("{p}.input_layernorm.weight"))?;
        let post_ln_w = get(&format!("{p}.post_attention_layernorm.weight"))?;

        // Attention projections — stored as [out, in], pre-transpose to [in, out]
        let attn_q_w = get(&format!("{sa}.q_proj.weight"))?.t();
        let attn_k_w = get(&format!("{sa}.k_proj.weight"))?.t();
        let attn_v_w = get(&format!("{sa}.v_proj.weight"))?.t();
        let attn_o_w = get(&format!("{sa}.o_proj.weight"))?.t();

        // Optional attention biases
        let attn_q_b = if use_bias {
            get_opt(&format!("{sa}.q_proj.bias"))
        } else {
            None
        };
        let attn_k_b = if use_bias {
            get_opt(&format!("{sa}.k_proj.bias"))
        } else {
            None
        };
        let attn_v_b = if use_bias {
            get_opt(&format!("{sa}.v_proj.bias"))
        } else {
            None
        };
        let attn_o_b = if use_bias {
            get_opt(&format!("{sa}.o_proj.bias"))
        } else {
            None
        };

        // MoE router — stored as [num_experts, hidden]; pre-transpose to [hidden, num_experts]
        let moe_router_w = get(&format!("{mlp}.router.weight"))?.t();

        // Expert projections.  These are stacked tensors of shape
        //   gate_proj / up_proj:  [num_experts, intermediate, hidden] (stored [num_experts, out, in])
        //   down_proj:            [num_experts, hidden, intermediate]
        // We pre-transpose gate/up to [num_experts, hidden, intermediate] so that
        // batched gather_mm(hidden[tok], stacked_w[expert_idx]) works correctly.
        // NOTE: The stacked weights arrive from the checkpoint as
        //   [num_experts, out_features, in_features] (PyTorch convention).
        // InlineArray.t() on 3-D does a full transpose of the last two dims — this
        // matches what we want: [num_experts, in, out] after transposing.
        let moe_gate_w = get(&format!("{mlp}.experts.gate_proj.weight"))?.t();
        let moe_up_w = get(&format!("{mlp}.experts.up_proj.weight"))?.t();
        // down_proj: [num_experts, hidden, intermediate] — transpose to [num_experts, intermediate, hidden]
        // to match gather_mm(clamped, down_w) → clamped: [T, inter], down_w: [inter, hidden]
        let moe_down_w = get(&format!("{mlp}.experts.down_proj.weight"))?.t();

        // Expert biases — [num_experts, out_features]; accessed by index during routing
        let moe_gate_b = get(&format!("{mlp}.experts.gate_proj.bias"))?;
        let moe_up_b = get(&format!("{mlp}.experts.up_proj.bias"))?;
        let moe_down_b = get(&format!("{mlp}.experts.down_proj.bias"))?;

        layers.push(LayerWeights {
            input_ln_w,
            input_ln_eps: config.rms_norm_eps,
            post_ln_w,
            post_ln_eps: config.rms_norm_eps,

            attn_q_w,
            attn_q_b,
            attn_k_w,
            attn_k_b,
            attn_v_w,
            attn_v_b,
            attn_o_w,
            attn_o_b,

            attn_n_heads: n_heads,
            attn_n_kv_heads: n_kv_heads,
            attn_head_dim: head_dim,
            attn_scale,
            attn_rope_base: rope_base,
            attn_is_sliding: is_sliding,
            attn_sliding_window: config.sliding_window,

            moe_router_w,
            moe_gate_w,
            moe_gate_b,
            moe_up_w,
            moe_up_b,
            moe_down_w,
            moe_down_b,
            moe_num_experts: n_experts,
            moe_top_k: top_k,

            swiglu_alpha: config.swiglu_alpha,
            swiglu_limit: config.swiglu_limit,
        });

        if li == 0 {
            eprintln!(
                "[GPT-OSS] layer 0: type={:?} n_heads={n_heads} n_kv={n_kv_heads} \
                 head_dim={head_dim} experts={n_experts} top_k={top_k}",
                layer_type,
            );
        }
    }

    // ── Step 5: copy_fresh — force all weights into fresh Metal buffers ─────
    let zero = InlineArray::from_f32(0.0).as_dtype(model_dtype);
    let copy_fresh = |w: &InlineArray| -> InlineArray {
        let mut fresh = w.add(&zero);
        fresh.eval();
        fresh.detach();
        fresh
    };
    let copy_fresh_opt =
        |w: Option<InlineArray>| -> Option<InlineArray> { w.map(|w| copy_fresh(&w)) };

    let embed_w = copy_fresh(&embed_w);
    let final_norm_w = copy_fresh(&final_norm_w);
    let lm_head_w = lm_head_w.map(|w| copy_fresh(&w));

    for lw in &mut layers {
        lw.input_ln_w = copy_fresh(&lw.input_ln_w);
        lw.post_ln_w = copy_fresh(&lw.post_ln_w);
        lw.attn_q_w = copy_fresh(&lw.attn_q_w);
        lw.attn_k_w = copy_fresh(&lw.attn_k_w);
        lw.attn_v_w = copy_fresh(&lw.attn_v_w);
        lw.attn_o_w = copy_fresh(&lw.attn_o_w);
        lw.attn_q_b = copy_fresh_opt(lw.attn_q_b.take());
        lw.attn_k_b = copy_fresh_opt(lw.attn_k_b.take());
        lw.attn_v_b = copy_fresh_opt(lw.attn_v_b.take());
        lw.attn_o_b = copy_fresh_opt(lw.attn_o_b.take());
        lw.moe_router_w = copy_fresh(&lw.moe_router_w);
        lw.moe_gate_w = copy_fresh(&lw.moe_gate_w);
        lw.moe_gate_b = copy_fresh(&lw.moe_gate_b);
        lw.moe_up_w = copy_fresh(&lw.moe_up_w);
        lw.moe_up_b = copy_fresh(&lw.moe_up_b);
        lw.moe_down_w = copy_fresh(&lw.moe_down_w);
        lw.moe_down_b = copy_fresh(&lw.moe_down_b);
    }

    eprintln!("[GPT-OSS] load_model: all weights force-copied into fresh Metal buffers");

    Ok(NativeWeights {
        embed_w,
        final_norm_w,
        final_norm_eps,
        lm_head_w,
        tie_word_embeddings: config.tie_word_embeddings,
        layers,
        model_dtype,
    })
}

// ── MXFP4 / weight sanitization helpers ──────────────────────────────────────

/// Flatten the last two dims of an MXFP4 `_blocks` tensor, as Python does:
///   `v.view(mx.uint32).flatten(-2)`
///
/// In the InlineArray bridge there is no direct view-as-uint32 op, so we
/// reinterpret by reshaping the last dimension (which packs bytes into uint32
/// words).  The bridge's `reshape` accepts -1 to infer a dimension.
///
/// If the shape cannot be inferred (unexpected layout), return the tensor
/// unchanged and let the downstream matmul fail with a useful shape error.
fn flatten_mxfp4_blocks(v: &InlineArray) -> InlineArray {
    // Shape is typically [num_experts, out_features//2, in_features, 2] for MXFP4
    // or [out_features//2, in_features, 2] for dense expert.
    // After flattening last two dims: [num_experts, out_features//2, in_features*2]
    // or [out_features//2, in_features*2].
    // Use reshape with -1 on the last dim to let MLX compute the product.
    let ndim = v.ndim();
    if ndim < 2 {
        return v.clone();
    }
    let mut new_shape: Vec<i32> = (0..ndim - 2).map(|i| v.dim(i)).collect();
    new_shape.push(-1); // flatten last two dims
    v.reshape(&new_shape)
}

/// Split a fused gate_up weight along the second-to-last dim.
///
/// For dense weights: shape `[num_experts, out*2, hidden]` →
///   gate: `[num_experts, out, hidden]` (first half)
///   up:   `[num_experts, out, hidden]` (second half)
///
/// For `.scales` or `.weight` the same slice indices apply to whatever dim
/// encodes the fused projection.
///
/// Returns (gate, up).
fn split_gate_up(v: &InlineArray, suffix: &str) -> (InlineArray, InlineArray) {
    let ndim = v.ndim();
    // The fused dim is the second-to-last (-2 from end, i.e. ndim-2 in 0-indexed).
    // For 2-D: dim 0. For 3-D expert tensor: dim 1.
    if ndim < 2 {
        // Scalar or 1-D — shouldn't happen; return copies.
        return (v.clone(), v.clone());
    }

    let split_dim = (ndim - 2) as i32; // second-to-last axis
    let fused_size = v.dim(split_dim);
    let half = fused_size / 2;

    // Build start/stop index arrays for slice on split_dim.
    // All other dims are taken in full: start=0, stop=dim(i).
    let start_gate = vec![0i32; ndim as usize];
    let mut stop_gate = (0..ndim).map(|i| v.dim(i)).collect::<Vec<_>>();
    let mut start_up = vec![0i32; ndim as usize];
    let stop_up = stop_gate.clone();

    // gate: [0..half] along split_dim
    stop_gate[split_dim as usize] = half;
    // up: [half..fused_size] along split_dim
    start_up[split_dim as usize] = half;

    if suffix == ".scales" {
        // For scales the fused axis may be the last dim — use the same logic
        // but fall back to a split on axis -1 when ndim == 1.
        let axis = if ndim == 1 { 0i32 } else { split_dim };
        let sg = vec![0i32; ndim as usize];
        let mut eg = (0..ndim).map(|i| v.dim(i)).collect::<Vec<_>>();
        let mut su = sg.clone();
        let eu = eg.clone();
        let half_s = eg[axis as usize] / 2;
        eg[axis as usize] = half_s;
        su[axis as usize] = half_s;
        let gate = v.slice(&sg, &eg);
        let up = v.slice(&su, &eu);
        return (gate, up);
    }

    let gate = v.slice(&start_gate, &stop_gate);
    let up = v.slice(&start_up, &stop_up);
    (gate, up)
}

/// Split a fused gate_up BIAS along the last axis.
///
/// Shape `[num_experts, out*2]` → gate `[num_experts, out]`, up `[num_experts, out]`.
/// Shape `[out*2]` → gate `[out]`, up `[out]`.
fn split_gate_up_bias(v: &InlineArray) -> (InlineArray, InlineArray) {
    let ndim = v.ndim();
    let last = (ndim - 1) as i32;
    let total = v.dim(last);
    let half = total / 2;

    let sg = vec![0i32; ndim as usize];
    let mut eg = (0..ndim).map(|i| v.dim(i)).collect::<Vec<_>>();
    let mut su = sg.clone();
    let eu = eg.clone();

    eg[last as usize] = half;
    su[last as usize] = half;

    let gate = v.slice(&sg, &eg);
    let up = v.slice(&su, &eu);
    (gate, up)
}

// ============================================================================
// Forward step
// ============================================================================

/// Run one forward step — works for both T=1 decode and T=N prefill.
///
/// `token_ids` must be shape `[B, T]` int32.  Returns logits `[B, T, vocab]`.
///
/// Architecture:
///   For each layer:
///     1. input_layernorm (RMSNorm)
///     2. Attention (sliding or full causal) + residual
///     3. post_attention_layernorm (RMSNorm)
///     4. MoE (sigmoid top-k routing + SwiGLU experts + per-expert bias) + residual
///   Final norm → lm_head
pub fn forward_step(
    weights: &NativeWeights,
    token_ids: &InlineArray, // [B, T]
    cache: &mut NativeCache,
) -> InlineArray {
    let b = token_ids.dim(0);
    let s = token_ids.dim(1);
    let dtype = weights.model_dtype;

    // Embedding lookup: [B, T, hidden]
    let mut hidden = weights.embed_w.take_axis(token_ids, 0);

    for (li, lw) in weights.layers.iter().enumerate() {
        // Input LayerNorm
        let normed = hidden.rms_norm(Some(&lw.input_ln_w), lw.input_ln_eps);

        // Attention
        let attn_out = attn_forward(
            lw,
            &normed,
            b,
            s,
            &mut cache.kv_caches[li],
            cache.rope_offset,
            dtype,
        );

        // Residual
        let h = hidden.add(&attn_out);

        // Post-attention LayerNorm
        let mlp_in = h.rms_norm(Some(&lw.post_ln_w), lw.post_ln_eps);

        // MoE
        let moe_out = moe_forward(lw, &mlp_in, b, s);

        // Residual
        hidden = h.add(&moe_out);
    }

    // Advance position counter
    cache.rope_offset += s;

    // Final norm + LM head
    let hidden = hidden.rms_norm(Some(&weights.final_norm_w), weights.final_norm_eps);
    if weights.tie_word_embeddings {
        hidden.matmul(&weights.embed_w.t())
    } else {
        hidden.matmul(weights.lm_head_w.as_ref().unwrap())
    }
}

// ============================================================================
// Attention layer forward
// ============================================================================

fn attn_forward(
    lw: &LayerWeights,
    normed: &InlineArray,
    b: i32,
    s: i32,
    cache: &mut KvLayerCache,
    rope_offset: i32,
    dtype: i32,
) -> InlineArray {
    let n_heads = lw.attn_n_heads;
    let n_kv_heads = lw.attn_n_kv_heads;
    let head_dim = lw.attn_head_dim;
    let scale = lw.attn_scale;

    // Q, K, V projections — [B, S, n_heads*head_dim]
    let mut q = normed.matmul(&lw.attn_q_w);
    let mut k = normed.matmul(&lw.attn_k_w);
    let mut v = normed.matmul(&lw.attn_v_w);

    // Add attention biases if present
    if let Some(ref qb) = lw.attn_q_b {
        q = q.add(qb);
    }
    if let Some(ref kb) = lw.attn_k_b {
        k = k.add(kb);
    }
    if let Some(ref vb) = lw.attn_v_b {
        v = v.add(vb);
    }

    // Reshape to [B, S, H, D] then transpose to [B, H, S, D]
    let q = q
        .reshape(&[b, s, n_heads, head_dim])
        .transpose_axes(&[0, 2, 1, 3]);
    let k = k
        .reshape(&[b, s, n_kv_heads, head_dim])
        .transpose_axes(&[0, 2, 1, 3]);
    let v = v
        .reshape(&[b, s, n_kv_heads, head_dim])
        .transpose_axes(&[0, 2, 1, 3]);

    // Full RoPE (head_dim = 64, no partial rotation)
    let q = q.rope(head_dim, false, lw.attn_rope_base, 1.0, rope_offset);
    let k = k.rope(head_dim, false, lw.attn_rope_base, 1.0, rope_offset);

    // KV cache update
    let prev = cache.offset;
    let num_new = k.dim(2); // S
    let next = prev + num_new;

    if lw.attn_is_sliding {
        // Rotating / sliding window cache: only keep `window` most recent tokens.
        // On the first call, allocate the window buffer; thereafter we rotate:
        //   if next <= window: write into [prev..next]
        //   else:             rotate buffer left by (next - window), write last `num_new`
        let window = lw.attn_sliding_window;

        if cache.keys.is_none() {
            cache.keys = Some(InlineArray::zeros(
                &[b, n_kv_heads, window, head_dim],
                dtype,
            ));
            cache.values = Some(InlineArray::zeros(
                &[b, n_kv_heads, window, head_dim],
                dtype,
            ));
        }

        if next <= window {
            // Simple write: fits in window without rotation
            let start = [0, 0, prev, 0];
            let stop = [b, n_kv_heads, next, head_dim];
            let k_buf = cache.keys.take().unwrap();
            let v_buf = cache.values.take().unwrap();
            cache.keys = Some(k_buf.slice_set(&k, &start, &stop));
            cache.values = Some(v_buf.slice_set(&v, &start, &stop));
        } else {
            // Rotate: drop oldest tokens to make room for `num_new`.
            // shift = how many positions to rotate left
            let shift = (next - window).min(window);
            let remain = window - shift; // tokens kept from previous
            let k_buf = cache.keys.take().unwrap();
            let v_buf = cache.values.take().unwrap();

            // Copy the tail [shift..window] → [0..remain]
            let k_old = k_buf.slice(&[0, 0, shift, 0], &[b, n_kv_heads, window, head_dim]);
            let v_old = v_buf.slice(&[0, 0, shift, 0], &[b, n_kv_heads, window, head_dim]);

            let new_k_buf = InlineArray::zeros(&[b, n_kv_heads, window, head_dim], dtype);
            let new_v_buf = InlineArray::zeros(&[b, n_kv_heads, window, head_dim], dtype);

            // Write old tail to front
            let k_rotated =
                new_k_buf.slice_set(&k_old, &[0, 0, 0, 0], &[b, n_kv_heads, remain, head_dim]);
            let v_rotated =
                new_v_buf.slice_set(&v_old, &[0, 0, 0, 0], &[b, n_kv_heads, remain, head_dim]);

            // Append new tokens after old tail
            let write_start = remain.min(window - num_new);
            let write_end = (write_start + num_new).min(window);
            let actual_new = write_end - write_start;

            let k_slice = k.slice(
                &[0, 0, num_new - actual_new, 0],
                &[b, n_kv_heads, num_new, head_dim],
            );
            let v_slice = v.slice(
                &[0, 0, num_new - actual_new, 0],
                &[b, n_kv_heads, num_new, head_dim],
            );

            let k_final = k_rotated.slice_set(
                &k_slice,
                &[0, 0, write_start, 0],
                &[b, n_kv_heads, write_end, head_dim],
            );
            let v_final = v_rotated.slice_set(
                &v_slice,
                &[0, 0, write_start, 0],
                &[b, n_kv_heads, write_end, head_dim],
            );

            cache.keys = Some(k_final);
            cache.values = Some(v_final);
        }
        cache.offset = next;

        // For SDPA we use the full window buffer (up to `min(next, window)` valid tokens)
        let valid = next.min(window);
        let valid_keys = cache
            .keys
            .as_ref()
            .unwrap()
            .slice(&[0, 0, 0, 0], &[b, n_kv_heads, valid, head_dim]);
        let valid_values = cache
            .values
            .as_ref()
            .unwrap()
            .slice(&[0, 0, 0, 0], &[b, n_kv_heads, valid, head_dim]);
        let output =
            crate::decode::sdpa_causal_like_mlx(&q, &valid_keys, &valid_values, scale, s);
        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[b, s, n_heads * head_dim]);

        // Output projection + bias
        let mut proj = output.matmul(&lw.attn_o_w);
        if let Some(ref ob) = lw.attn_o_b {
            proj = proj.add(ob);
        }
        proj
    } else {
        // Full attention: growing pre-allocated buffer (256-token chunk strategy)
        if cache.keys.is_none() {
            let alloc = 256i32;
            cache.keys = Some(InlineArray::zeros(&[b, n_kv_heads, alloc, head_dim], dtype));
            cache.values = Some(InlineArray::zeros(&[b, n_kv_heads, alloc, head_dim], dtype));
        } else {
            let allocated = cache.keys.as_ref().unwrap().dim(2);
            if next > allocated {
                let old_k = cache.keys.take().unwrap();
                let old_v = cache.values.take().unwrap();
                let ext_k = InlineArray::zeros(&[b, n_kv_heads, 256, head_dim], dtype);
                let ext_v = InlineArray::zeros(&[b, n_kv_heads, 256, head_dim], dtype);
                cache.keys = Some(old_k.kv_cache_append(&ext_k, 2));
                cache.values = Some(old_v.kv_cache_append(&ext_v, 2));
            }
        }

        // In-place update: cache[..., prev:next, :] = new_kv
        let start = [0, 0, prev, 0];
        let stop = [b, n_kv_heads, next, head_dim];
        let k_buf = cache.keys.take().unwrap();
        let v_buf = cache.values.take().unwrap();
        cache.keys = Some(k_buf.slice_set(&k, &start, &stop));
        cache.values = Some(v_buf.slice_set(&v, &start, &stop));
        cache.offset = next;

        // SDPA on the valid portion
        let valid_keys = cache
            .keys
            .as_ref()
            .unwrap()
            .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, head_dim]);
        let valid_values = cache
            .values
            .as_ref()
            .unwrap()
            .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, head_dim]);
        let output =
            crate::decode::sdpa_causal_like_mlx(&q, &valid_keys, &valid_values, scale, s);
        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[b, s, n_heads * head_dim]);

        // Output projection + bias
        let mut proj = output.matmul(&lw.attn_o_w);
        if let Some(ref ob) = lw.attn_o_b {
            proj = proj.add(ob);
        }
        proj
    }
}

// ============================================================================
// MoE forward
// ============================================================================

/// GPT-OSS MoE forward pass with sigmoid routing and per-expert bias.
///
/// Routing:
///   1. `router_logits = hidden @ router_w`           [B*T, num_experts]
///   2. `scores = sigmoid(router_logits)`             [B*T, num_experts]
///   3. Top-k via argpartition(-k) → take_along_axis  [B*T, top_k]
///   4. Normalize: `weights = scores_topk / sum(scores_topk)`  (safe sum with 1e-8 floor)
///
/// Expert computation per slot s in [0..top_k):
///   1. Gather gate/up/down weight rows for expert_ids[:, s]  via take_axis
///   2. gate_out = batched_matmul(hidden_flat, gate_w[slot]) + gate_b[slot]
///   3. up_out   = batched_matmul(hidden_flat, up_w[slot])   + up_b[slot]
///   4. act      = gpt_oss_swiglu(gate_out, up_out)            (clamped + alpha-scaled)
///   5. slot_out = batched_matmul(act, down_w[slot])          + down_b[slot]
///   6. output  += slot_out * weights[:, s:s+1]
fn moe_forward(lw: &LayerWeights, normed: &InlineArray, b: i32, s: i32) -> InlineArray {
    // Flatten to [B*T, hidden]
    let hidden_size = normed.dim(2);
    let bt = b * s;
    let hidden_flat = normed.reshape(&[bt, hidden_size]);

    // Router logits: [B*T, num_experts]
    let router_logits = hidden_flat.matmul(&lw.moe_router_w);

    // sigmoid scores
    let scores = router_logits.sigmoid();

    // Top-k: argpartition at kth = -top_k gives top-k indices in the last k slots
    let neg_k = -lw.moe_top_k;
    let partitioned = scores.argpartition(neg_k, -1); // [B*T, num_experts]
    let top_k_indices = partitioned.slice(
        &[0, lw.moe_num_experts - lw.moe_top_k],
        &[bt, lw.moe_num_experts],
    );
    // Re-cast to int32 for gather ops (argpartition returns int32 already, but ensure)
    let top_k_scores = scores.take_along_axis(&top_k_indices, -1); // [B*T, top_k]

    // Normalize: weights = scores / max(sum, 1e-8)
    let sum_scores = top_k_scores.sum_axis(-1, true); // [B*T, 1]
    let eps = InlineArray::from_f32(1e-8).as_dtype(sum_scores.dtype_raw());
    let safe_sum = sum_scores.maximum(&eps);
    let expert_weights = top_k_scores.divide(&safe_sum); // [B*T, top_k]

    // Accumulate expert outputs
    let mut output = InlineArray::zeros(&[bt, hidden_size], hidden_flat.dtype_raw());

    for slot in 0..lw.moe_top_k {
        // Expert indices for this slot: [B*T]
        let slot_experts = top_k_indices
            .slice(&[0, slot], &[bt, slot + 1])
            .reshape(&[bt]);
        // Expert weights for this slot: [B*T, 1]
        let slot_weights = expert_weights.slice(&[0, slot], &[bt, slot + 1]);

        // Gather per-token expert weights from stacked tensors.
        // stacked shape: [num_experts, hidden, intermediate] (gate/up) or [num_experts, intermediate, hidden] (down)
        // take_axis(slot_experts, 0) → [B*T, hidden, intermediate]
        let gate_w = lw.moe_gate_w.take_axis(&slot_experts, 0); // [B*T, hidden, inter]
        let up_w = lw.moe_up_w.take_axis(&slot_experts, 0); // [B*T, hidden, inter]
        let down_w = lw.moe_down_w.take_axis(&slot_experts, 0); // [B*T, inter, hidden]
        let gate_b = lw.moe_gate_b.take_axis(&slot_experts, 0); // [B*T, inter]
        let up_b = lw.moe_up_b.take_axis(&slot_experts, 0); // [B*T, inter]
        let down_b = lw.moe_down_b.take_axis(&slot_experts, 0); // [B*T, hidden]

        // Batched matmul: [B*T, 1, hidden] @ [B*T, hidden, inter] → [B*T, 1, inter] → [B*T, inter]
        let h_exp = hidden_flat.reshape(&[bt, 1, hidden_size]);
        let gate_out = h_exp.matmul(&gate_w).reshape(&[bt, -1]).add(&gate_b);
        let up_out = h_exp.matmul(&up_w).reshape(&[bt, -1]).add(&up_b);

        // GPT-OSS SwiGLU activation with clamping
        let act = gpt_oss_swiglu(&gate_out, &up_out, lw.swiglu_alpha, lw.swiglu_limit);

        // Down projection: [B*T, 1, inter] @ [B*T, inter, hidden] → [B*T, hidden]
        let act_exp = act.reshape(&[bt, 1, -1]);
        let slot_out = act_exp
            .matmul(&down_w)
            .reshape(&[bt, hidden_size])
            .add(&down_b);

        // Weighted accumulation
        output = output.add(&slot_out.multiply(&slot_weights));
    }

    // Restore [B, S, hidden]
    output.reshape(&[b, s, hidden_size])
}

// ============================================================================
// GPT-OSS SwiGLU activation
// ============================================================================

/// GPT-OSS custom SwiGLU activation (from Python `swiglu` compiled function):
///
///   x_glu  = clip(x_glu,    a_max=limit)
///   x_lin  = clip(x_linear, a_min=-limit, a_max=limit)
///   out    = x_glu * sigmoid(alpha * x_glu) * (x_linear + 1)
///
/// This differs from standard SwiGLU (`silu(gate) * up`) in two ways:
///   1. Clamping is applied to prevent FP16 overflow at large values.
///   2. The linear branch gets a bias of +1 before the gate multiply.
///   3. The gate uses a parametric alpha (1.702) instead of 1.0.
///
/// `InlineArray` exposes `maximum` but not `minimum`.  Upper-clamp is
/// implemented as `−maximum(−x, −limit)` (de Morgan's min/max identity).
#[inline]
fn gpt_oss_swiglu(
    x_linear: &InlineArray,
    x_glu: &InlineArray,
    alpha: f32,
    limit: f32,
) -> InlineArray {
    let neg_limit_arr = InlineArray::from_f32(-limit).as_dtype(x_glu.dtype_raw());
    let alpha_arr = InlineArray::from_f32(alpha).as_dtype(x_glu.dtype_raw());
    let one_arr = InlineArray::from_f32(1.0).as_dtype(x_linear.dtype_raw());

    // clip(x_glu, a_max=limit) = -maximum(-x_glu, -limit)
    let x_glu_clamped = x_glu.negative().maximum(&neg_limit_arr).negative();
    // clip(x_linear, a_min=-limit, a_max=limit):
    //   lower: maximum(x_linear, -limit)
    //   upper: -maximum(-result, -limit)
    let x_lin_lo = x_linear.maximum(&neg_limit_arr);
    let x_lin_clamped = x_lin_lo.negative().maximum(&neg_limit_arr).negative();

    // sigmoid(alpha * x_glu)
    let glu_scaled = x_glu_clamped.multiply(&alpha_arr);
    let sig = glu_scaled.sigmoid();

    // out_glu = x_glu * sigmoid(alpha * x_glu)
    let out_glu = x_glu_clamped.multiply(&sig);

    // (x_linear + 1)
    let lin_biased = x_lin_clamped.add(&one_arr);

    // out = out_glu * (x_linear + 1)
    out_glu.multiply(&lin_biased)
}

// ============================================================================
// Sampling
// ============================================================================

/// Sample one token from `logits_2d` of shape `[B, vocab]`.
///
/// `temperature <= 0.0` → greedy argmax; otherwise categorical sampling.
pub fn sample_token(logits_2d: &InlineArray, temperature: f32) -> InlineArray {
    crate::decode::sample_token(logits_2d, temperature)
}

// ============================================================================
// Generation loop
// ============================================================================

/// Run the full GPT-OSS generation loop with async GPU pipelining.
///
/// `first_token` is the last token of the already-prefilled prompt.
/// `on_token(token_id)` is called after each decoded token; return `false` to
/// stop early (e.g. on EOS).
///
/// Returns all generated token IDs (not including `first_token`).
pub fn generate(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    max_tokens: usize,
    temperature: f32,
    mut on_token: impl FnMut(u32) -> bool,
) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(max_tokens);

    crate::decode::begin_generation_session("GPT-OSS", weights.model_dtype);

    // Evaluate and detach all prefill cache states before decode.
    cache.eval_and_detach_states();
    bridge::clear_cache();

    // First decode step
    let input_token = InlineArray::from_i32(first_token as i32).reshape(&[1, 1]);
    let logits = forward_step(weights, &input_token, cache);
    // Squeeze sequence dim: [B, 1, vocab] → [B, vocab]
    let logits_2d = logits.squeeze(1);
    let mut current_y = sample_token(&logits_2d, temperature);
    // Start async GPU eval concurrently with CPU.
    current_y.async_eval_ref();

    let mut step_times: Vec<f64> = Vec::new();

    for step in 0..max_tokens {
        let next_y = if step + 1 < max_tokens {
            let t_step = std::time::Instant::now();
            let next_input = current_y.reshape(&[1, 1]);
            let next_logits = forward_step(weights, &next_input, cache);
            let next_logits_2d = next_logits.squeeze(1);
            let next_y = sample_token(&next_logits_2d, temperature);
            next_y.async_eval_ref();
            step_times.push(t_step.elapsed().as_secs_f64() * 1000.0);
            Some(next_y)
        } else {
            None
        };

        if step == 0 {
            current_y.eval();
        }
        let token_val = current_y.item_u32();

        tokens.push(token_val);
        if !on_token(token_val) {
            break;
        }
        let Some(next_y) = next_y else {
            break;
        };
        current_y = next_y;

        // Periodically flush the buffer cache to prevent memory accumulation.
        if step % 256 == 255 {
            bridge::clear_cache();
        }
    }

    if step_times.len() > 20 {
        step_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let skip = 10;
        let avg = step_times[skip..].iter().sum::<f64>() / (step_times.len() - skip) as f64;
        let p50 = step_times[step_times.len() / 2];
        eprintln!(
            "[GPT-OSS] per-step: avg={avg:.2}ms p50={p50:.2}ms = {:.0} tok/s",
            1000.0 / avg,
        );
    }

    bridge::synchronize();
    tokens
}
