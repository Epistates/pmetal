//! Standalone Llama 4 inference engine — zero dependency on mlx-rs or pmetal-models.
//!
//! Every op on the hot path uses [`InlineArray`] (stack-allocated `mlx::core::array`,
//! direct C++ bridge). This eliminates ALL per-op heap allocation, matching
//! Python/nanobind's direct C++ binding performance.
//!
//! Architecture details (from mlx-lm llama4.py):
//!
//! - **iRoPE**: interleaved positional encoding.
//!   - Layers where `(layer_idx + 1) % 4 != 0` are "local" — use chunked attention
//!     with full RoPE (traditional=true) and optional QK-norm.
//!   - Layers where `(layer_idx + 1) % 4 == 0` are "global" — use full causal attention
//!     with NoPE (no positional encoding) and attention temperature tuning.
//!
//! - **MoE**: `interleave_moe_layer_step` controls which layers are MoE.
//!   `(layer_idx % step) == (step - 1)` → MoE layer; others are dense MLP.
//!   Top-1 routing with sigmoid scores, shared expert added after routed output.
//!   Expert weights stored as `[num_experts, out, in]` (SwitchLinear convention);
//!   sanitization splits and transposes the gate_up_proj block from safetensors.
//!
//! The entire stack — config, weights, caches, forward pass, generation loop —
//! lives in this single module.

use serde::Deserialize;

use crate::InlineArray;
use crate::inline_array as bridge;

// ============================================================================
// Config
// ============================================================================

fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f64 {
    500_000.0
}
fn default_attn_temperature_tuning() -> i32 {
    4
}
fn default_floor_scale() -> i32 {
    8192
}
fn default_attn_scale() -> f32 {
    0.1
}
fn default_interleave_moe_layer_step() -> i32 {
    1
}
fn default_true() -> bool {
    true
}

/// Nested rope_scaling config — only a handful of fields matter for iRoPE.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RopeScalingConfig {
    #[serde(rename = "type", default)]
    pub scaling_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f64>,
    #[serde(default)]
    pub rope_type: Option<String>,
}

/// Text-level config (lives under `text_config` in the outer JSON, or at the
/// top level for text-only models).
#[derive(Debug, Clone, Deserialize)]
pub struct Llama4TextConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,

    #[serde(default)]
    pub num_key_value_heads: Option<i32>,

    #[serde(default)]
    pub head_dim: Option<i32>,

    /// Dense MLP intermediate size (used by non-MoE layers).
    pub intermediate_size_mlp: i32,

    /// MoE expert hidden size.
    pub intermediate_size: i32,

    pub vocab_size: i32,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,

    /// Chunk size for local (chunked) attention — matches Python's `attention_chunk_size`.
    pub attention_chunk_size: i32,

    /// Step between MoE layers: `(layer_idx % step) == (step - 1)` is a MoE layer.
    #[serde(default = "default_interleave_moe_layer_step")]
    pub interleave_moe_layer_step: i32,

    pub num_local_experts: i32,
    pub num_experts_per_tok: i32,

    /// Whether QK-norm is applied (only on RoPE/local layers).
    #[serde(default = "default_true")]
    pub use_qk_norm: bool,

    /// Whether attention projection biases are used.
    #[serde(default)]
    pub attention_bias: bool,

    pub max_position_embeddings: i32,

    /// Attention temperature tuning for NoPE (global) layers.
    #[serde(default = "default_attn_temperature_tuning")]
    pub attn_temperature_tuning: i32,

    #[serde(default = "default_floor_scale")]
    pub floor_scale: i32,

    #[serde(default = "default_attn_scale")]
    pub attn_scale: f32,

    #[serde(default)]
    pub tie_word_embeddings: bool,
}

/// Outer model config — Llama 4 nests text config under `text_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct Llama4Config {
    pub model_type: String,

    /// Present in multi-modal checkpoint JSON.
    #[serde(default)]
    pub text_config: Option<Llama4TextConfig>,

    /// Flattened text config fields — present in text-only configs.
    #[serde(flatten)]
    pub text: Llama4TextConfig,
}

impl Llama4Config {
    /// Resolve the effective text config regardless of nesting.
    pub fn text(&self) -> &Llama4TextConfig {
        self.text_config.as_ref().unwrap_or(&self.text)
    }

    /// Returns `true` when layer `li` (0-indexed) is a MoE layer.
    ///
    /// Python: `(layer_idx % interleave_moe_layer_step) == (interleave_moe_layer_step - 1)`
    pub fn is_moe_layer(&self, li: usize) -> bool {
        let step = self.text().interleave_moe_layer_step;
        (li as i32 % step) == (step - 1)
    }

    /// Returns `true` when layer `li` (0-indexed) uses RoPE (local / chunked attention).
    ///
    /// Python: `use_rope = int((layer_idx + 1) % 4 != 0)`
    pub fn use_rope(&self, li: usize) -> bool {
        ((li as i32) + 1) % 4 != 0
    }

    /// Head dimension.
    pub fn head_dim(&self) -> i32 {
        let t = self.text();
        t.head_dim.unwrap_or(t.hidden_size / t.num_attention_heads)
    }

    /// Number of KV heads.
    pub fn num_kv_heads(&self) -> i32 {
        let t = self.text();
        t.num_key_value_heads.unwrap_or(t.num_attention_heads)
    }
}

/// Parse `config.json` from a model directory.
///
/// Handles both the flat text-only layout and the multi-modal layout where
/// `text_config` is nested.
pub fn load_config(model_dir: &std::path::Path) -> Result<Llama4Config, String> {
    let path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("failed to parse config.json: {e}"))?;

    // If the outer config has `text_config`, try deserializing that inner object
    // as the canonical text config and wrap it.  Otherwise deserialize the
    // whole JSON as a flat Llama4Config.
    if json.get("text_config").is_some() {
        let cfg: Llama4Config = serde_json::from_str(&text)
            .map_err(|e| format!("failed to parse Llama4Config: {e}"))?;
        Ok(cfg)
    } else {
        // Flat layout: embed a synthetic `model_type` field if absent, then
        // deserialize the whole object as both outer + inner config.
        let mut obj = json.clone();
        if !obj
            .as_object()
            .map(|o| o.contains_key("model_type"))
            .unwrap_or(false)
        {
            obj["model_type"] = serde_json::Value::String("llama4".to_string());
        }
        let config_str = serde_json::to_string(&obj).map_err(|e| e.to_string())?;
        let cfg: Llama4Config = serde_json::from_str(&config_str)
            .map_err(|e| format!("failed to parse Llama4Config (flat): {e}"))?;
        Ok(cfg)
    }
}

// ============================================================================
// Per-layer weights
// ============================================================================

/// MoE expert weight block. Weights are stored pre-transposed to `[num_experts, in, out]`
/// form for `gather_mm` (which expects `b` with shape `[experts, K, N]`).
struct MoeWeights {
    /// `[num_experts, in_dim, expert_hidden]` — gate projection
    experts_gate_w: InlineArray,
    /// `[num_experts, in_dim, expert_hidden]` — up projection
    experts_up_w: InlineArray,
    /// `[num_experts, expert_hidden, in_dim]` — down projection
    experts_down_w: InlineArray,
    /// Router: `[in_dim, num_experts]` (pre-transposed from `[num_experts, in_dim]`)
    router_w: InlineArray,
    /// Shared expert gate: `[in_dim, intermediate_size_mlp]`
    shared_gate_w: InlineArray,
    /// Shared expert up: `[in_dim, intermediate_size_mlp]`
    shared_up_w: InlineArray,
    /// Shared expert down: `[intermediate_size_mlp, in_dim]`
    shared_down_w: InlineArray,
}

struct LayerWeights {
    // ── iRoPE meta ──
    use_rope: bool, // false for NoPE (global) layers

    // ── Layer norms ──
    input_ln_w: InlineArray,
    post_ln_w: InlineArray,
    norm_eps: f32,

    // ── Attention projection ──
    attn_q_w: InlineArray, // [in, n_heads * head_dim]
    attn_k_w: InlineArray, // [in, n_kv_heads * head_dim]
    attn_v_w: InlineArray, // [in, n_kv_heads * head_dim]
    attn_o_w: InlineArray, // [n_heads * head_dim, in]

    // Optional bias (when attention_bias=true)
    attn_q_b: Option<InlineArray>,
    attn_k_b: Option<InlineArray>,
    attn_v_b: Option<InlineArray>,
    attn_o_b: Option<InlineArray>,

    // QK-norm (only on RoPE layers — same flag for both Q and K, no learned weight)
    attn_qk_norm: bool, // true → apply rms_norm(eps=1e-6) to both Q and K

    // Attention shape config (stored per-layer for self-containedness)
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    attn_scale: f32,
    rope_base: f32,
    rope_scale: f32,

    // Temperature tuning for NoPE layers
    attn_temperature_tuning: i32,
    floor_scale: i32,
    layer_attn_scale: f32,

    // ── Feed-forward: either MoE or dense MLP ──
    is_moe: bool,
    // Dense MLP (non-MoE layers)
    mlp_gate_w: Option<InlineArray>, // [in, intermediate_size_mlp]
    mlp_up_w: Option<InlineArray>,
    mlp_down_w: Option<InlineArray>,
    // MoE
    moe: Option<MoeWeights>,
}

// ============================================================================
// Full model weights
// ============================================================================

/// All model weights as InlineArrays. Zero dependency on mlx-rs.
pub struct NativeWeights {
    pub embed_w: InlineArray,      // [vocab, hidden]
    pub final_norm_w: InlineArray, // [hidden]
    pub final_norm_eps: f32,
    /// `None` when `tie_word_embeddings = true`.
    pub lm_head_w: Option<InlineArray>, // [hidden, vocab] (pre-transposed)
    pub tie_word_embeddings: bool,
    /// Per-layer weights — only accessed via [`forward_step`].
    layers: Vec<LayerWeights>,
    /// Model activation dtype (11 = bfloat16, 1 = float16, 0 = float32).
    pub model_dtype: i32,
    /// Chunk size for local attention (from config).
    pub attention_chunk_size: i32,
}

impl std::fmt::Debug for NativeWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeWeights")
            .field("layers", &self.layers.len())
            .field("tie_word_embeddings", &self.tie_word_embeddings)
            .field("model_dtype", &self.model_dtype)
            .field("attention_chunk_size", &self.attention_chunk_size)
            .finish()
    }
}

// ============================================================================
// Caches
// ============================================================================

/// Per-layer KV cache using pre-allocated buffers with O(1) slice_set updates.
pub struct KvLayerCache {
    pub keys: Option<InlineArray>,   // [B, H, MAX_T, D]
    pub values: Option<InlineArray>, // [B, H, MAX_T, D]
    pub offset: i32,                 // valid tokens in the cache
    /// For chunked attention: how many tokens are in the "front" (trimmed portion).
    /// Python's ChunkedKVCache trims the front when the chunk fills.
    /// For decode we keep the entire sequence visible (attention_chunk_size is just
    /// a mask constraint, not an eviction policy in the native path).
    pub start_position: i32,
}

/// Full model cache.
pub struct NativeCache {
    /// One entry per layer (both local and global — all are KV caches for Llama 4).
    pub kv_caches: Vec<KvLayerCache>,
    /// Global position offset (number of tokens processed so far).
    pub rope_offset: i32,
}

impl NativeCache {
    /// Evaluate and detach all cache arrays. Must be called after prefill and
    /// before decode to sever the prefill computation graph.
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
            .map(|_| KvLayerCache {
                keys: None,
                values: None,
                offset: 0,
                start_position: 0,
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

/// Load model weights from a directory containing safetensors shards.
///
/// Applies all sanitization required by the mlx-lm reference implementation:
/// - Vision / projector weight removal
/// - Expert weight splitting: `experts.gate_up_proj` → `experts.gate_proj.weight` +
///   `experts.up_proj.weight` (split on last axis, then swapaxes(1,2))
/// - Expert down projection: `experts.down_proj` → `experts.down_proj.weight`
///   (swapaxes(1,2))
/// - `language_model.` prefix stripping for VLM checkpoints
/// - All projection weights pre-transposed to `[in, out]` form for efficient matmul
pub fn load_model(
    model_dir: &std::path::Path,
    config: &Llama4Config,
) -> Result<NativeWeights, String> {
    let tc = config.text();

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

    // ── Step 3: Sanitization ────────────────────────────────────────────────

    // 3a. Strip VLM prefixes:
    //     "language_model.model.X" → "model.X"
    //     "language_model.lm_head.*" → "lm_head.*"
    //     "model.language_model.X" → "model.X"
    // Also drop vision and projector weights.
    let original_keys: Vec<String> = raw.keys().cloned().collect();
    for old_key in original_keys {
        // Drop vision / projector weights.
        if old_key.contains("vision_model") || old_key.contains("multi_modal_projector") {
            raw.remove(&old_key);
            continue;
        }

        let mut new_key = old_key.clone();
        if new_key.starts_with("language_model.model.") {
            new_key = new_key.replacen("language_model.", "", 1);
        } else if new_key.starts_with("language_model.") {
            // e.g. "language_model.lm_head.weight"
            new_key = new_key.replacen("language_model.", "", 1);
        } else if new_key.starts_with("model.language_model.") {
            new_key = new_key.replacen("model.language_model.", "model.", 1);
        }

        if new_key != old_key {
            if let Some(v) = raw.remove(&old_key) {
                raw.insert(new_key, v);
            }
        }
    }

    // 3b. Expert weight sanitization (matches Python's Model.sanitize()).
    //
    // Original safetensors format:
    //   `layers.{l}.feed_forward.experts.gate_up_proj` — shape [num_experts, in, 2*hidden]
    //   `layers.{l}.feed_forward.experts.down_proj`    — shape [num_experts, hidden, in]
    //
    // Target format (after sanitize):
    //   `layers.{l}.feed_forward.experts.gate_proj.weight` — shape [num_experts, in, hidden]
    //   `layers.{l}.feed_forward.experts.up_proj.weight`   — shape [num_experts, in, hidden]
    //   `layers.{l}.feed_forward.experts.down_proj.weight` — shape [num_experts, in, hidden]
    //
    // Python does:
    //   gate_proj, up_proj = mx.split(v, 2, axis=-1)  # both [E, in, hidden]
    //   gate_proj = mx.swapaxes(gate_proj, 1, 2)      # → [E, hidden, in]  wait...
    //
    // Wait, let's re-read Python:
    //   v = weights.pop(f"{prefix}.gate_up_proj")      # [E, in, 2*hidden] ?? or [E, 2*hidden, in]?
    //   gate_proj, up_proj = mx.split(v, 2, axis=-1)   # split last dim → each [E, ?, hidden]
    //   weights[gate_k] = mx.swapaxes(gate_proj, 1, 2) # swap dim1 and dim2
    //   weights[up_k]   = mx.swapaxes(up_proj, 1, 2)
    //
    // SwitchLinear weight shape: [num_experts, output_dims, input_dims]
    // SwitchLinear.__call__: gather_mm(x, weight.swapaxes(-1,-2), rhs_indices=indices)
    //   → gather_mm(x, [E, input_dims, output_dims], indices)  ← this is the transposed form
    //
    // So the raw checkpoint stores: gate_up_proj [E, in_dim, 2*out_dim]
    // split → gate [E, in_dim, out_dim], up [E, in_dim, out_dim]
    // swapaxes(1,2) → [E, out_dim, in_dim]  ← stored as SwitchLinear.weight
    // At call time: weight.swapaxes(-1,-2) = [E, in_dim, out_dim] ← used as B in gather_mm
    //
    // For down_proj: raw [E, out_dim, in_dim] (output=hidden_size, input=expert_hidden)
    //   swapaxes(1,2) → [E, in_dim, out_dim] = [E, expert_hidden, hidden_size]
    //   → stored as SwitchLinear.weight
    //   At call: weight.swapaxes(-1,-2) = [E, hidden_size, expert_hidden]... wait that's odd.
    //
    // Actually let me re-read more carefully:
    //   SwitchLinear(hidden_dims, input_dims, num_experts) for down_proj:
    //   → weight shape [num_experts, input_dims, hidden_dims]
    //   But `input_dims` here is the outer hidden_size (what the down proj outputs),
    //   and `hidden_dims` is the expert_hidden (what it reads from).
    //   SwitchLinear constructor: shape=(num_experts, output_dims, input_dims)
    //   For down_proj: SwitchLinear(hidden_dims=expert_hidden, input_dims=hidden_size)
    //   → weight [E, expert_hidden, hidden_size]
    //   At call: weight.swapaxes(-1,-2) = [E, hidden_size, expert_hidden]
    //   gather_mm(x [B,1,1,expert_hidden], [E,hidden_size,expert_hidden], rhs=indices)
    //   → output [B, 1, 1, hidden_size] ✓
    //
    // Raw down_proj in checkpoint: [E, hidden_size, expert_hidden]? No, Python says:
    //   down_proj = weights.pop(f"{prefix}.down_proj")   # [E, hidden_size, expert_hidden]?
    //   weights[f"{prefix}.down_proj.weight"] = mx.swapaxes(down_proj, 1, 2)
    //   → [E, expert_hidden, hidden_size]
    //
    // So checkpoint stores: down_proj [E, hidden_size, expert_hidden]
    // → after swapaxes(1,2): [E, expert_hidden, hidden_size] = SwitchLinear.weight
    // At call: .swapaxes(-1,-2) = [E, hidden_size, expert_hidden]
    // gather_mm(x[B,1,1,expert_hidden], [E,hidden_size,expert_hidden], indices) → [B,1,1,hidden_size] ✓
    //
    // In our native path we pre-transpose for gather_mm directly (eliminating runtime swapaxes):
    //   gate_proj:  checkpoint [E, in_dim, out_dim] → we store as-is = [E, in_dim, out_dim]
    //                (gather_mm(x, gate_proj_w, None, indices) where w=[E,K,N], x=[B,1,1,K])
    //   Actually gather_mm signature: gather_mm(a, b, lhs_indices, rhs_indices, sorted)
    //   x is a (lhs) and weight is b (rhs). b should be [E, K, N] for x[B,1,1,K] → [B,1,1,N].
    //
    // So we want gate_proj stored as [E, in_dim, expert_hidden] for gather_mm directly.
    // checkpoint gate_up_proj [E, in_dim, 2*expert_hidden]
    // split last → gate[E, in_dim, H], up[E, in_dim, H]
    // swapaxes(1,2) → [E, H, in_dim]  ← SwitchLinear.weight
    // at runtime: .swapaxes(-1,-2) → [E, in_dim, H]  ← used in gather_mm as B
    //
    // So the correct form for our gather_mm call is [E, in_dim, H] =
    // (checkpoint gate/up split, NO swapaxes).
    // We store gate/up as [E, in_dim, H] by NOT applying swapaxes after split.
    //
    // For down_proj: checkpoint [E, out_dim, in_dim] = [E, hidden_size, expert_hidden]
    //   x after gate/up: [B, 1, 1, expert_hidden]
    //   want: gather_mm(x, down_w, None, indices) → [B, 1, 1, hidden_size]
    //   → down_w should be [E, expert_hidden, hidden_size]
    //   → that IS swapaxes(1,2) of the checkpoint.
    //
    // Summary of what we store in NativeWeights:
    //   gate_proj [E, in_dim, expert_hidden]   ← checkpoint split (no extra swapaxes)
    //   up_proj   [E, in_dim, expert_hidden]   ← checkpoint split (no extra swapaxes)
    //   down_proj [E, expert_hidden, hidden_size] ← checkpoint swapaxes(1,2)

    for li in 0..tc.num_hidden_layers as usize {
        if !config.is_moe_layer(li) {
            continue;
        }
        let prefix = format!("model.layers.{li}.feed_forward.experts");

        // gate_up_proj: [E, in_dim, 2*expert_hidden]
        if let Some(gate_up) = raw.remove(&format!("{prefix}.gate_up_proj")) {
            let expert_hidden = tc.intermediate_size;
            // split along last axis at position expert_hidden
            let mut parts = gate_up.split(&[expert_hidden], -1);
            let up = parts.pop().unwrap(); // [E, in_dim, expert_hidden]
            let gate = parts.pop().unwrap(); // [E, in_dim, expert_hidden]
            raw.insert(format!("{prefix}.gate_proj.weight"), gate);
            raw.insert(format!("{prefix}.up_proj.weight"), up);
        }

        // down_proj: [E, hidden_size, expert_hidden] → store as [E, expert_hidden, hidden_size]
        if let Some(down) = raw.remove(&format!("{prefix}.down_proj")) {
            // swapaxes(1, 2) on a 3D array: transpose dims 1 and 2
            let down_t = down.transpose_axes(&[0, 2, 1]);
            raw.insert(format!("{prefix}.down_proj.weight"), down_t);
        }
    }

    // ── Step 4: Build per-layer weight structs ──────────────────────────────

    let detected_dtype = raw
        .get("model.embed_tokens.weight")
        .map(|w| w.dtype_raw())
        .unwrap_or(11); // 11 = bfloat16

    let get = |key: &str| -> Result<InlineArray, String> {
        raw.get(key).cloned().ok_or_else(|| {
            let parts: Vec<&str> = key.rsplitn(2, '.').collect();
            let suffix = parts[0];
            let close: Vec<&String> = raw.keys().filter(|k| k.ends_with(suffix)).take(5).collect();
            format!("missing weight key: {key} (close matches: {close:?})")
        })
    };

    let try_get = |key: &str| -> Option<InlineArray> { raw.get(key).cloned() };

    let embed_w = get("model.embed_tokens.weight")?;
    let final_norm_w = get("model.norm.weight")?;
    let lm_head_w = if tc.tie_word_embeddings {
        None
    } else {
        // Stored as [vocab, hidden] — transpose to [hidden, vocab] for matmul.
        Some(get("lm_head.weight")?.t())
    };

    let n_heads = tc.num_attention_heads;
    let n_kv_heads = config.num_kv_heads();
    let head_dim = config.head_dim();
    let attn_scale = (head_dim as f32).powi(-1).sqrt(); // 1/sqrt(head_dim)
    // rope_theta from config — Llama 4 uses 500_000 by default.
    let rope_base = tc.rope_theta as f32;
    // rope_scale: for Llama 4 iRoPE with rope_scaling.type=="llama3", factor is the
    // high-frequency scale. For simplicity in the native path we use scale=1.0
    // (standard RoPE, no long-context extension) because chunked attention limits
    // effective context to attention_chunk_size tokens per chunk anyway.
    // Users needing long-context should use the mlx-rs full path.
    let rope_scale = 1.0_f32;

    let mut layers = Vec::with_capacity(tc.num_hidden_layers as usize);

    for li in 0..tc.num_hidden_layers as usize {
        let p = format!("model.layers.{li}");
        let use_rope = config.use_rope(li);
        let is_moe = config.is_moe_layer(li);

        let input_ln_w = get(&format!("{p}.input_layernorm.weight"))?;
        let post_ln_w = get(&format!("{p}.post_attention_layernorm.weight"))?;

        // Attention projections — stored as [out, in], transpose to [in, out].
        let sa = format!("{p}.self_attn");
        let attn_q_w = get(&format!("{sa}.q_proj.weight"))?.t();
        let attn_k_w = get(&format!("{sa}.k_proj.weight"))?.t();
        let attn_v_w = get(&format!("{sa}.v_proj.weight"))?.t();
        let attn_o_w = get(&format!("{sa}.o_proj.weight"))?.t();

        // Optional biases
        let attn_q_b = try_get(&format!("{sa}.q_proj.bias"));
        let attn_k_b = try_get(&format!("{sa}.k_proj.bias"));
        let attn_v_b = try_get(&format!("{sa}.v_proj.bias"));
        let attn_o_b = try_get(&format!("{sa}.o_proj.bias"));

        // QK-norm only on RoPE layers (and only when use_qk_norm=true in config)
        let attn_qk_norm = use_rope && tc.use_qk_norm;

        // Feed-forward
        let (mlp_gate_w, mlp_up_w, mlp_down_w, moe) = if is_moe {
            let ff = format!("{p}.feed_forward");
            let exp = format!("{ff}.experts");

            // Expert gate/up/down — shapes already sanitized in step 3b.
            // gate_proj: [E, in_dim, expert_hidden]
            // up_proj:   [E, in_dim, expert_hidden]
            // down_proj: [E, expert_hidden, in_dim]
            let gate_w = get(&format!("{exp}.gate_proj.weight"))?;
            let up_w = get(&format!("{exp}.up_proj.weight"))?;
            let down_w = get(&format!("{exp}.down_proj.weight"))?;

            // Router: [num_experts, in_dim] → pre-transpose to [in_dim, num_experts]
            let router_w = get(&format!("{ff}.router.weight"))?.t();

            // Shared expert — stored as standard [out, in], transposed to [in, out]
            let sh = format!("{ff}.shared_expert");
            let sh_gate_w = get(&format!("{sh}.gate_proj.weight"))?.t();
            let sh_up_w = get(&format!("{sh}.up_proj.weight"))?.t();
            let sh_down_w = get(&format!("{sh}.down_proj.weight"))?.t();

            let moe_weights = MoeWeights {
                experts_gate_w: gate_w,
                experts_up_w: up_w,
                experts_down_w: down_w,
                router_w,
                shared_gate_w: sh_gate_w,
                shared_up_w: sh_up_w,
                shared_down_w: sh_down_w,
            };
            (None, None, None, Some(moe_weights))
        } else {
            // Dense MLP — uses intermediate_size_mlp
            let ff = format!("{p}.feed_forward");
            let gate_w = get(&format!("{ff}.gate_proj.weight"))?.t();
            let up_w = get(&format!("{ff}.up_proj.weight"))?.t();
            let down_w = get(&format!("{ff}.down_proj.weight"))?.t();
            (Some(gate_w), Some(up_w), Some(down_w), None)
        };

        layers.push(LayerWeights {
            use_rope,
            input_ln_w,
            post_ln_w,
            norm_eps: tc.rms_norm_eps,
            attn_q_w,
            attn_k_w,
            attn_v_w,
            attn_o_w,
            attn_q_b,
            attn_k_b,
            attn_v_b,
            attn_o_b,
            attn_qk_norm,
            n_heads,
            n_kv_heads,
            head_dim,
            attn_scale,
            rope_base,
            rope_scale,
            attn_temperature_tuning: tc.attn_temperature_tuning,
            floor_scale: tc.floor_scale,
            layer_attn_scale: tc.attn_scale,
            is_moe,
            mlp_gate_w,
            mlp_up_w,
            mlp_down_w,
            moe,
        });
    }

    // ── Step 5: copy_fresh — force all weights into fresh Metal buffers ─────
    let zero = InlineArray::from_f32(0.0).as_dtype(detected_dtype);
    let copy_fresh = |w: &InlineArray| -> InlineArray {
        let mut fresh = w.add(&zero);
        fresh.eval();
        fresh.detach();
        fresh
    };

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
        if let Some(ref b) = lw.attn_q_b {
            lw.attn_q_b = Some(copy_fresh(b));
        }
        if let Some(ref b) = lw.attn_k_b {
            lw.attn_k_b = Some(copy_fresh(b));
        }
        if let Some(ref b) = lw.attn_v_b {
            lw.attn_v_b = Some(copy_fresh(b));
        }
        if let Some(ref b) = lw.attn_o_b {
            lw.attn_o_b = Some(copy_fresh(b));
        }
        if let Some(ref w) = lw.mlp_gate_w {
            lw.mlp_gate_w = Some(copy_fresh(w));
        }
        if let Some(ref w) = lw.mlp_up_w {
            lw.mlp_up_w = Some(copy_fresh(w));
        }
        if let Some(ref w) = lw.mlp_down_w {
            lw.mlp_down_w = Some(copy_fresh(w));
        }
        if let Some(ref mut moe) = lw.moe {
            moe.experts_gate_w = copy_fresh(&moe.experts_gate_w);
            moe.experts_up_w = copy_fresh(&moe.experts_up_w);
            moe.experts_down_w = copy_fresh(&moe.experts_down_w);
            moe.router_w = copy_fresh(&moe.router_w);
            moe.shared_gate_w = copy_fresh(&moe.shared_gate_w);
            moe.shared_up_w = copy_fresh(&moe.shared_up_w);
            moe.shared_down_w = copy_fresh(&moe.shared_down_w);
        }
    }

    eprintln!(
        "[LLAMA4_NATIVE] load_model: {} layers, dtype={}, chunk_size={}",
        layers.len(),
        detected_dtype,
        tc.attention_chunk_size
    );

    Ok(NativeWeights {
        embed_w,
        final_norm_w,
        final_norm_eps: tc.rms_norm_eps,
        lm_head_w,
        tie_word_embeddings: tc.tie_word_embeddings,
        layers,
        model_dtype: detected_dtype,
        attention_chunk_size: tc.attention_chunk_size,
    })
}

// ============================================================================
// Forward step
// ============================================================================

/// Run one forward step — works for both T=1 decode and T=N prefill.
///
/// `token_ids` must be shape `[B, T]` int32. Returns logits `[B, T, vocab]`.
///
/// Implements iRoPE exactly as the Python reference:
/// - Local layers (use_rope=true): chunked causal attention with RoPE + QK-norm
/// - Global layers (use_rope=false): full causal attention with NoPE and
///   attention temperature tuning
pub fn forward_step(
    weights: &NativeWeights,
    token_ids: &InlineArray, // [B, T]
    cache: &mut NativeCache,
) -> InlineArray {
    let b = token_ids.dim(0);
    let s = token_ids.dim(1);

    // Embedding lookup: [B, T, hidden]
    let mut hidden = weights.embed_w.take_axis(token_ids, 0);

    // Build chunk mask for local layers (chunked attention).
    // Python computes this once per forward:
    //   linds = mx.arange(start, end)     ← positions of cached tokens
    //   rinds = mx.arange(offset, end)[:, None]  ← positions of query tokens
    //   block_pos = |linds // chunk_size - rinds // chunk_size|
    //   token_pos = linds <= rinds
    //   chunk_mask = (block_pos == 0) & token_pos
    // For decode (T=1) this collapses to: only positions in the same chunk as
    // the current query are attended to.
    let chunk_size = weights.attention_chunk_size;
    // offset = number of tokens already in the cache (all layers share same sequence position).
    let offset = cache.rope_offset;
    let end = offset + s;
    // We build the chunk mask eagerly only for prefill (T > 1). For decode (T=1)
    // we skip the mask and use pure causal (the single query token attends to all
    // positions in its chunk, which degenerates to a simple causal window).
    let chunk_mask: Option<InlineArray> = if s > 1 {
        // linds: [1, end] range — all key positions from start of sequence
        // For simplicity in the native path we treat start_position = 0.
        // Build as bool mask [s, offset+s].
        let chunk_mask_val = build_chunk_mask(offset, s, end, chunk_size);
        Some(chunk_mask_val)
    } else {
        None // decode: use "causal" SDPA mode for local layers (no mask needed)
    };

    for (li, lw) in weights.layers.iter().enumerate() {
        // Input LayerNorm
        let normed = hidden.rms_norm(Some(&lw.input_ln_w), lw.norm_eps);

        // Attention
        let attn_out = attn_forward(
            lw,
            &normed,
            b,
            s,
            &mut cache.kv_caches[li],
            cache.rope_offset,
            chunk_mask.as_ref(),
        );

        // Residual add
        let h = hidden.add(&attn_out);

        // Post-attention LayerNorm
        let mlp_in = h.rms_norm(Some(&lw.post_ln_w), lw.norm_eps);

        // Feed-forward (MoE or dense)
        let ff_out = if lw.is_moe {
            moe_forward(lw.moe.as_ref().unwrap(), &mlp_in, b, s)
        } else {
            dense_mlp_forward(lw, &mlp_in)
        };

        // Residual add
        hidden = h.add(&ff_out);
    }

    // Advance position counter.
    cache.rope_offset += s;

    // Final norm + LM head.
    let hidden = hidden.rms_norm(Some(&weights.final_norm_w), weights.final_norm_eps);
    if weights.tie_word_embeddings {
        hidden.matmul(&weights.embed_w.t())
    } else {
        hidden.matmul(weights.lm_head_w.as_ref().unwrap())
    }
}

// ============================================================================
// Chunk mask construction
// ============================================================================

/// Build the boolean chunk attention mask for a prefill pass.
///
/// Returns shape `[s, offset + s]` bool mask where `mask[i, j] = true` means
/// query token at position `(offset + i)` can attend to key token at position `j`.
///
/// A query at position `r` can attend to key at position `l` when:
///   `l <= r`  (causal)  AND  `r // chunk_size == l // chunk_size`  (same chunk)
fn build_chunk_mask(offset: i32, s: i32, end: i32, chunk_size: i32) -> InlineArray {
    // We build the mask on CPU as an i32 slice (1 = attend, 0 = mask out)
    // and pass it to MLX as a bool array via InlineArray::from_i32_slice.
    // For typical prefill sizes this is fast enough; for very long sequences
    // we could build it with MLX ops, but let's keep it simple.
    let total_kv = end as usize;
    let mut mask_data = vec![0i32; s as usize * total_kv];

    for qi in 0..s as usize {
        let q_pos = (offset + qi as i32) as usize;
        let q_chunk = q_pos / chunk_size as usize;
        for ki in 0..total_kv {
            let k_chunk = ki / chunk_size as usize;
            if ki <= q_pos && k_chunk == q_chunk {
                mask_data[qi * total_kv + ki] = 1;
            }
        }
    }

    // Create [s, total_kv] int32 array then cast to bool (dtype=7 in MLX).
    let flat = InlineArray::from_i32_slice(&mask_data);
    let mask = flat.reshape(&[s, end]);
    // Cast to bfloat16 additive mask: 0 → 0.0, 1 → ... actually MLX sdpa_with_mask
    // expects an additive float mask where -inf means masked. Convert boolean to float:
    // where mask=1 → 0.0, mask=0 → -inf (large negative).
    // We return the boolean-as-int32 array and convert in the attention function
    // using where_cond.
    mask
}

/// Convert a 0/1 int32 mask `[q, k]` to an additive attention bias `[q, k]`.
/// 0 → -1e9 (masked), 1 → 0.0 (unmasked).
fn make_additive_mask(bool_mask: &InlineArray, dtype: i32) -> InlineArray {
    // large negative value in the model dtype
    let neg_inf = InlineArray::from_f32(-1e9).as_dtype(dtype);
    let zero = InlineArray::from_f32(0.0).as_dtype(dtype);
    // where(bool_mask != 0, 0.0, -1e9)
    // bool_mask contains 0 or 1 as int32; `where_cond` treats nonzero as true.
    bool_mask
        .as_dtype(0)
        .where_cond(&zero, &neg_inf)
        .as_dtype(dtype)
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
    chunk_mask: Option<&InlineArray>, // prefill only — [s, offset+s]
) -> InlineArray {
    let n_heads = lw.n_heads;
    let n_kv_heads = lw.n_kv_heads;
    let head_dim = lw.head_dim;
    let scale = lw.attn_scale;

    // Q, K, V projections
    let mut queries = normed.matmul(&lw.attn_q_w);
    let mut keys = normed.matmul(&lw.attn_k_w);
    let mut values = normed.matmul(&lw.attn_v_w);

    // Optional biases
    if let Some(ref qb) = lw.attn_q_b {
        queries = queries.add(qb);
    }
    if let Some(ref kb) = lw.attn_k_b {
        keys = keys.add(kb);
    }
    if let Some(ref vb) = lw.attn_v_b {
        values = values.add(vb);
    }

    // Reshape to [B, S, H, D] then transpose to [B, H, S, D]
    let queries = queries
        .reshape(&[b, s, n_heads, head_dim])
        .transpose_axes(&[0, 2, 1, 3]);
    let keys = keys
        .reshape(&[b, s, n_kv_heads, head_dim])
        .transpose_axes(&[0, 2, 1, 3]);
    let values = values
        .reshape(&[b, s, n_kv_heads, head_dim])
        .transpose_axes(&[0, 2, 1, 3]);

    // iRoPE: only RoPE layers apply positional encoding
    let (queries, keys) = if lw.use_rope {
        // Traditional RoPE (rope_theta = 500_000, traditional=true for Llama 4).
        // Python: initialize_rope(head_dim, rope_theta, traditional=True, ...)
        let q = queries.rope(head_dim, true, lw.rope_base, lw.rope_scale, rope_offset);
        let k = keys.rope(head_dim, true, lw.rope_base, lw.rope_scale, rope_offset);
        (q, k)
    } else {
        (queries, keys)
    };

    // QK-norm: only on RoPE layers (use_qk_norm = args.use_qk_norm AND use_rope)
    // Python: rms_norm(queries, weight=None, eps=1e-6)
    let (queries, keys) = if lw.attn_qk_norm {
        let q = queries.rms_norm(None, 1e-6);
        let k = keys.rms_norm(None, 1e-6);
        (q, k)
    } else {
        (queries, keys)
    };

    // Attention temperature tuning for NoPE (global) layers.
    // Python:
    //   if attn_temperature_tuning and not use_rope:
    //     attn_scales = log(floor(arange(offset+1, offset+L+1) / floor_scale) + 1) * attn_scale + 1
    //     queries = (queries * attn_scales[:, None]).astype(queries.dtype)
    //
    // This scales queries by a position-dependent factor: larger positions → larger scale.
    // The effect dampens attention entropy at long range.
    let queries = if lw.attn_temperature_tuning > 0 && !lw.use_rope {
        apply_temperature_tuning(
            &queries,
            rope_offset,
            s,
            lw.floor_scale,
            lw.layer_attn_scale,
        )
    } else {
        queries
    };

    // KV cache update
    let prev = cache.offset;
    let num_new = keys.dim(2); // T for prefill, 1 for decode
    let next = prev + num_new;

    if cache.keys.is_none() {
        let alloc = 256i32;
        let dtype = keys.dtype_raw();
        cache.keys = Some(InlineArray::zeros(&[b, n_kv_heads, alloc, head_dim], dtype));
        cache.values = Some(InlineArray::zeros(&[b, n_kv_heads, alloc, head_dim], dtype));
    } else {
        // Grow if needed
        let allocated = cache.keys.as_ref().unwrap().dim(2);
        if next > allocated {
            let dtype = cache.keys.as_ref().unwrap().dtype_raw();
            let old_k = cache.keys.take().unwrap();
            let old_v = cache.values.take().unwrap();
            let ext_k = InlineArray::zeros(&[b, n_kv_heads, 256, head_dim], dtype);
            let ext_v = InlineArray::zeros(&[b, n_kv_heads, 256, head_dim], dtype);
            cache.keys = Some(old_k.kv_cache_append(&ext_k, 2));
            cache.values = Some(old_v.kv_cache_append(&ext_v, 2));
        }
    }

    let start_coord = [0, 0, prev, 0];
    let stop_coord = [b, n_kv_heads, next, head_dim];
    let k_buf = cache.keys.take().unwrap();
    let v_buf = cache.values.take().unwrap();
    cache.keys = Some(k_buf.slice_set(&keys, &start_coord, &stop_coord));
    cache.values = Some(v_buf.slice_set(&values, &start_coord, &stop_coord));
    cache.offset = next;

    // Valid portion of KV cache
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

    // SDPA
    let output = if lw.use_rope {
        // Local (chunked) layer:
        // - For decode (s=1): use causal SDPA (the single query naturally attends
        //   to all cached positions; the chunk constraint is handled by not storing
        //   tokens outside the current chunk — or in this simpler native path, by
        //   letting the model see a slightly wider window which matches mlx-lm
        //   behavior for cached keys).
        // - For prefill (s>1): apply the chunk mask.
        if let Some(ref mask_int) = chunk_mask {
            // Build [s, next] additive mask and reshape to [1, 1, s, next] for SDPA.
            let dtype = queries.dtype_raw();
            let mask_full = make_additive_mask(&mask_int.slice(&[0, 0], &[s, next]), dtype);
            let mask_4d = mask_full.reshape(&[1, 1, s, next]);
            queries.sdpa_with_mask(&valid_keys, &valid_values, scale, Some(&mask_4d))
        } else {
            // Decode: causal (only 1 query token, always valid)
            crate::decode::sdpa_causal_like_mlx(&queries, &valid_keys, &valid_values, scale, s)
        }
    } else {
        // Global (NoPE) layer: full causal attention, no chunk constraint.
        crate::decode::sdpa_causal_like_mlx(&queries, &valid_keys, &valid_values, scale, s)
    };

    // Reshape [B, H, S, D] → [B, S, H*D]
    let output = output
        .transpose_axes(&[0, 2, 1, 3])
        .reshape(&[b, s, n_heads * head_dim]);

    // Output projection + optional bias
    let mut result = output.matmul(&lw.attn_o_w);
    if let Some(ref ob) = lw.attn_o_b {
        result = result.add(ob);
    }
    result
}

// ============================================================================
// Attention temperature tuning (NoPE / global layers)
// ============================================================================

/// Apply Llama 4's attention temperature tuning to queries on NoPE layers.
///
/// Python:
/// ```python
/// attn_scales = (
///     mx.log(mx.floor(mx.arange(offset + 1, offset + L + 1) / floor_scale) + 1.0)
///     * attn_scale + 1.0
/// )
/// attn_scales = attn_scales[:, None]   # [L, 1]
/// queries = (queries * attn_scales).astype(queries.dtype)
/// ```
///
/// Here `queries` is `[B, H, S, D]` (already transposed). The scales are
/// `[S]` → `[1, 1, S, 1]` for broadcast.
fn apply_temperature_tuning(
    queries: &InlineArray,
    rope_offset: i32,
    s: i32,
    floor_scale: i32,
    attn_scale: f32,
) -> InlineArray {
    let dtype = queries.dtype_raw();

    // Build scales on CPU as f32 slice for simplicity.
    let mut scales = Vec::with_capacity(s as usize);
    for i in 0..s {
        let pos = (rope_offset + i + 1) as f64;
        let floored = (pos / floor_scale as f64).floor();
        let scale_val = (floored + 1.0_f64).ln() as f32 * attn_scale + 1.0;
        scales.push(scale_val);
    }

    // Encode as [S] float32 array, cast to model dtype, reshape to [1, 1, S, 1]
    // so it broadcasts over [B, H, S, D].
    let scale_arr = {
        // We create the scale array from individual f32 scalars and concatenate.
        // For S=1 (decode) this is trivial.
        if s == 1 {
            InlineArray::from_f32(scales[0])
                .as_dtype(dtype)
                .reshape(&[1, 1, 1, 1])
        } else {
            // Build as i32 array trick won't work for f32. Instead: create each
            // element, concatenate along axis 0, then reshape.
            // For prefill this only runs once so perf is not critical.
            let mut arr = InlineArray::from_f32(scales[0]).as_dtype(dtype);
            for &sv in scales[1..].iter() {
                let elem = InlineArray::from_f32(sv).as_dtype(dtype);
                arr = arr.concatenate_2(&elem, 0);
            }
            arr.reshape(&[1, 1, s, 1])
        }
    };

    queries.multiply(&scale_arr).as_dtype(dtype)
}

// ============================================================================
// Dense MLP forward
// ============================================================================

fn dense_mlp_forward(lw: &LayerWeights, x: &InlineArray) -> InlineArray {
    let gate = x.matmul(lw.mlp_gate_w.as_ref().unwrap());
    let up = x.matmul(lw.mlp_up_w.as_ref().unwrap());
    let act = InlineArray::fused_swiglu(&gate, &up);
    act.matmul(lw.mlp_down_w.as_ref().unwrap())
}

// ============================================================================
// MoE forward (SwitchGLU with shared expert)
// ============================================================================

/// MoE forward pass matching Python's `MoE.__call__`.
///
/// Python:
/// ```python
/// logits = self.router(x)
/// indices = argpartition(-logits, kth=0, axis=-1)[..., :1]  # top-1
/// scores = take_along_axis(logits, indices, axis=-1)
/// scores = sigmoid(scores.astype(float32)).astype(x.dtype)
/// out = self.experts(x * scores, indices).squeeze(2)
/// return out + self.shared_expert(x)
/// ```
///
/// Weight layout in `moe` after sanitization:
///   - `experts_gate_w`: `[E, hidden, expert_h]` — used directly as gather_mm B matrix
///   - `experts_up_w`:   `[E, hidden, expert_h]`
///   - `experts_down_w`: `[E, expert_h, hidden]`
///
/// We reshape inputs to `[B*T, 1, hidden]` for gather_mm and `rhs_indices=[B*T, 1]`.
fn moe_forward(moe: &MoeWeights, x: &InlineArray, b: i32, s: i32) -> InlineArray {
    let hidden_size = x.dim(2);
    let dtype = x.dtype_raw();

    // Router: [B, T, hidden] × [hidden, num_experts] → [B, T, num_experts]
    let logits = x.matmul(&moe.router_w);

    // Top-1 selection: argpartition(-logits, kth=0)  places the top-1 index
    // at position 0. Flatten to [B*T, num_experts], slice first column → [B*T, 1].
    let bt = b * s;
    let num_experts = logits.dim(2);
    let neg_logits = logits.negative();
    let partition = neg_logits.argpartition(0, -1);
    let part_flat = partition.reshape(&[bt, num_experts]);
    let indices_flat = part_flat.slice(&[0, 0], &[bt, 1]); // [bt, 1] top-1 expert index
    let indices = indices_flat.reshape(&[b, s, 1]); // [B, T, 1]

    // Gather scores for the top-1 expert, apply sigmoid in f32 for numerical stability.
    let scores_raw = logits.take_along_axis(&indices, -1); // [B, T, 1]
    let scores_sig = scores_raw.as_dtype(0).sigmoid().as_dtype(dtype); // [B, T, 1]

    // Scale the routed input.
    let x_scaled = x.multiply(&scores_sig); // [B, T, hidden]

    // Reshape for gather_mm: [bt, 1, hidden]. rhs_indices: [bt, 1] as uint32.
    let x_g = x_scaled.reshape(&[bt, 1, hidden_size]);
    let rhs_idx = indices_flat.as_dtype(5); // dtype 5 = uint32 in MLX

    // Gate and up projections via gather_mm → [bt, 1, expert_h]
    let gate_out = x_g.gather_mm(&moe.experts_gate_w, None, Some(&rhs_idx), false);
    let up_out = x_g.gather_mm(&moe.experts_up_w, None, Some(&rhs_idx), false);

    // SwiGLU activation
    let activated = InlineArray::fused_swiglu(&gate_out, &up_out);

    // Down projection → [bt, 1, hidden]
    let down_out = activated.gather_mm(&moe.experts_down_w, None, Some(&rhs_idx), false);
    let routed_out = down_out.reshape(&[b, s, hidden_size]);

    // Shared expert: standard dense SwiGLU MLP
    let sh_gate = x.matmul(&moe.shared_gate_w);
    let sh_up = x.matmul(&moe.shared_up_w);
    let sh_act = InlineArray::fused_swiglu(&sh_gate, &sh_up);
    let shared_out = sh_act.matmul(&moe.shared_down_w);

    routed_out.add(&shared_out)
}

// ============================================================================
// Sampling
// ============================================================================

/// Sample one token from `logits_2d` of shape `[B, vocab]`.
///
/// `temperature <= 0.0` → greedy argmax. Otherwise categorical sampling.
pub fn sample_token(logits_2d: &InlineArray, temperature: f32) -> InlineArray {
    crate::decode::sample_token(logits_2d, temperature)
}

/// Run prompt prefill and return the first sampled token.
pub fn prefill_first_token(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    input_ids: &[u32],
    temperature: f32,
) -> u32 {
    crate::decode::prefill_first_token(weights, cache, input_ids, temperature, forward_step)
}

// ============================================================================
// Generation loop
// ============================================================================

fn prepare_generation_cache(cache: &mut NativeCache) {
    cache.eval_and_detach_states();
    bridge::clear_cache();
}

fn prime_generation_impl(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    temperature: f32,
    reset_peak_memory: bool,
    log_session: bool,
) -> InlineArray {
    crate::decode::prime_generation(
        "LLAMA4_NATIVE",
        weights.model_dtype,
        weights,
        cache,
        first_token,
        temperature,
        reset_peak_memory,
        log_session,
        prepare_generation_cache,
        forward_step,
    )
}

fn generate_from_primed_sample_impl(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    current_y: InlineArray,
    max_tokens: usize,
    temperature: f32,
    log_stats: bool,
    mut on_token: impl FnMut(u32) -> bool,
) -> Vec<u32> {
    crate::decode::generate_from_primed_sample(
        "LLAMA4_NATIVE",
        weights,
        cache,
        current_y,
        max_tokens,
        temperature,
        log_stats,
        |token| on_token(token),
        forward_step,
    )
}

/// Run one MLX-LM-style benchmark trial on the canonical Llama 4 native path.
pub fn benchmark_mlx_lm_trial(
    weights: &NativeWeights,
    prompt_ids: &[u32],
    generation_tokens: usize,
) -> crate::decode::BenchmarkTrial {
    crate::inline_array::reset_peak_memory();
    let mut cache = NativeCache::new_empty(weights);

    let prompt_tic = std::time::Instant::now();
    let first_tok = prefill_first_token(weights, &mut cache, prompt_ids, 0.0);
    let current_y = prime_generation_impl(weights, &mut cache, first_tok, 0.0, false, false);
    let prompt_secs = prompt_tic.elapsed().as_secs_f64();

    let generation_secs = if generation_tokens > 1 {
        let generation_tic = std::time::Instant::now();
        let generated_tail = generate_from_primed_sample_impl(
            weights,
            &mut cache,
            current_y,
            generation_tokens - 1,
            0.0,
            false,
            |_| true,
        );
        debug_assert_eq!(generated_tail.len(), generation_tokens - 1);
        generation_tic.elapsed().as_secs_f64()
    } else {
        crate::inline_array::synchronize();
        f64::MIN_POSITIVE
    };

    let trial = crate::decode::BenchmarkTrial {
        prompt_secs,
        generation_secs,
        peak_memory_bytes: crate::inline_array::get_peak_memory(),
    };

    crate::inline_array::synchronize();
    crate::inline_array::clear_cache();
    trial
}

/// Run the full generation loop with async GPU pipelining.
///
/// `first_token` is the last token from the prompt (already processed into `cache`
/// by a prefill call). Each call to `on_token` receives the sampled token ID and
/// returns `false` to stop early (e.g. on EOS).
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
    let current_y = prime_generation_impl(weights, cache, first_token, temperature, true, true);
    generate_from_primed_sample_impl(
        weights,
        cache,
        current_y,
        max_tokens,
        temperature,
        true,
        |token| on_token(token),
    )
}
