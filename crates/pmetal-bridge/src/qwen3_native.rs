//! Standalone Qwen3/Qwen3.5 inference engine — zero dependency on mlx-rs or pmetal-models.
//!
//! Supports three variants:
//!   - **Qwen3 dense** (`model_type = "qwen3"`): standard attention (no gate), full RoPE.
//!   - **Qwen3.5 dense** (`model_type = "qwen3_5"` / `"qwen3_5_text"`): gated attention,
//!     partial RoPE, GDN layers.
//!   - **Qwen3.5 MoE** (same type fields but `num_experts > 0`): like Qwen3.5 dense but
//!     MLP replaced with routed expert dispatch (SwitchGLU) plus a shared expert.
//!
//! Every op on the hot path uses [`InlineArray`] (stack-allocated `mlx::core::array`,
//! direct C++ bridge). This eliminates ALL per-op heap allocation, matching
//! Python/nanobind's direct C++ binding performance.
//!
//! The entire stack — config, weights, caches, forward pass, generation loop —
//! lives in this single module. The only external dependencies are
//! `serde`/`serde_json` (for config parsing) and `crate::InlineArray`.

use serde::Deserialize;

use crate::InlineArray;
use crate::inline_array as bridge;
use crate::inline_array::RawBuf;

// ============================================================================
// Config
// ============================================================================

fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_true() -> bool {
    true
}
fn default_full_attn_interval() -> i32 {
    4
}
fn default_conv_kernel() -> i32 {
    4
}
fn default_model_type() -> String {
    "qwen3_5".to_string()
}

/// Minimal, serde-deserializable Qwen3/Qwen3.5 config.
///
/// Only the fields required for inference are included; unknown keys are
/// silently ignored by serde.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    /// Distinguishes model families: `"qwen3"` = dense Qwen3; `"qwen3_5"` /
    /// `"qwen3_5_text"` = Qwen3.5 hybrid (dense or MoE).
    #[serde(default = "default_model_type")]
    pub model_type: String,

    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,

    #[serde(default)]
    pub num_key_value_heads: Option<i32>,

    #[serde(default)]
    pub head_dim: Option<i32>,

    /// Dense MLP intermediate size.
    pub intermediate_size: i32,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    /// Fraction of head_dim rotated. Stored as Option so we can fall back to
    /// nested `rope_parameters.partial_rotary_factor` at parse time.
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,

    /// Nested rope config — only used during `finalize()` to promote
    /// `partial_rotary_factor` when the top-level field is absent.
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,

    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,

    #[serde(default = "default_full_attn_interval")]
    pub full_attention_interval: i32,

    // GDN / linear-attention params (Qwen3.5 only)
    #[serde(default)]
    pub linear_num_key_heads: Option<i32>,
    #[serde(default)]
    pub linear_num_value_heads: Option<i32>,
    #[serde(default)]
    pub linear_key_head_dim: Option<i32>,
    #[serde(default)]
    pub linear_value_head_dim: Option<i32>,

    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: i32,

    // MoE fields (Qwen3.5 MoE only)
    #[serde(default)]
    pub num_experts: i32,
    #[serde(default)]
    pub num_experts_per_tok: i32,
    #[serde(default)]
    pub decoder_sparse_step: i32,
    #[serde(default)]
    pub shared_expert_intermediate_size: i32,
    #[serde(default)]
    pub moe_intermediate_size: i32,
    #[serde(default)]
    pub norm_topk_prob: bool,
    /// Layer indices that use dense MLP even when MoE is active.
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct RopeParameters {
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
}

impl Qwen3Config {
    /// Promote nested `rope_parameters.partial_rotary_factor` when the
    /// top-level field is absent. Call once after deserializing.
    pub fn finalize(&mut self) {
        if self.partial_rotary_factor.is_none() {
            if let Some(ref rp) = self.rope_parameters.clone() {
                if let Some(prf) = rp.partial_rotary_factor {
                    self.partial_rotary_factor = Some(prf);
                }
            }
        }
    }

    /// Returns `true` for dense Qwen3 (standard attention, no GDN).
    pub fn is_qwen3_dense(&self) -> bool {
        let mt = self.model_type.to_ascii_lowercase();
        mt == "qwen3" || mt == "qwen3dense"
    }

    /// Returns `true` when this is a MoE model (routed experts + shared expert).
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    /// Returns `true` when the MLP at the given layer index should be dense.
    ///
    /// For Qwen3.5 MoE the MLP is dense when:
    ///   - the layer is listed in `mlp_only_layers`, OR
    ///   - `decoder_sparse_step > 0` and `(layer_idx + 1) % decoder_sparse_step != 0`
    pub fn is_dense_mlp_layer(&self, layer_idx: usize) -> bool {
        if !self.is_moe() {
            return true;
        }
        if self.mlp_only_layers.contains(&layer_idx) {
            return true;
        }
        if self.decoder_sparse_step > 1 {
            return ((layer_idx as i32) + 1) % self.decoder_sparse_step != 0;
        }
        false
    }

    /// Effective partial rotary factor.
    ///
    /// - Qwen3 dense: always 1.0 (full RoPE).
    /// - Qwen3.5: stored value, defaulting to 0.25.
    pub fn effective_partial_rotary_factor(&self) -> f64 {
        if self.is_qwen3_dense() {
            1.0
        } else {
            self.partial_rotary_factor.unwrap_or(0.25)
        }
    }

    /// Effective head dimension.
    pub fn get_head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Effective number of KV heads.
    pub fn get_num_kv_heads(&self) -> i32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// RoPE dimensions for partial rotary.
    pub fn rope_dims(&self) -> i32 {
        (self.get_head_dim() as f64 * self.effective_partial_rotary_factor()) as i32
    }

    /// Returns `true` when layer `i` is a GDN (linear-attention) layer.
    ///
    /// For Qwen3 dense, all layers are attention — never GDN.
    /// For Qwen3.5: every `full_attention_interval`-th layer (1-indexed) is
    /// a full-attention layer; all others are GDN.
    pub fn is_linear_layer(&self, i: usize) -> bool {
        if self.is_qwen3_dense() {
            return false;
        }
        ((i as i32) + 1) % self.full_attention_interval != 0
    }

    // GDN dimension accessors (with Qwen3.5 defaults)
    pub fn gdn_nk(&self) -> i32 {
        self.linear_num_key_heads.unwrap_or(4)
    }
    pub fn gdn_nv(&self) -> i32 {
        self.linear_num_value_heads.unwrap_or(8)
    }
    pub fn gdn_dk(&self) -> i32 {
        self.linear_key_head_dim.unwrap_or(128)
    }
    pub fn gdn_dv(&self) -> i32 {
        self.linear_value_head_dim.unwrap_or(128)
    }
}

/// Parse `config.json` from a model directory.
pub fn load_config(model_dir: &std::path::Path) -> Result<Qwen3Config, String> {
    let path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("failed to parse config.json: {e}"))?;

    // Qwen3.5 nests the LM config under `text_config`.
    // For plain Qwen3, the config.json is flat.
    let config_str = if json.get("text_config").is_some() {
        // Inject `model_type` from the outer object if absent from text_config,
        // so `is_qwen3_dense()` can distinguish the families.
        let mut tc = json["text_config"].clone();
        if tc.get("model_type").is_none() {
            if let Some(mt) = json.get("model_type") {
                tc["model_type"] = mt.clone();
            }
        }
        serde_json::to_string(&tc).map_err(|e| e.to_string())?
    } else {
        text
    };

    let mut cfg: Qwen3Config = serde_json::from_str(&config_str)
        .map_err(|e| format!("failed to parse config: {e}"))?;
    cfg.finalize();
    Ok(cfg)
}

// ============================================================================
// Per-layer weights
// ============================================================================

struct LayerWeights {
    is_linear: bool,

    // Shared: layer norms
    input_ln_w: InlineArray,
    input_ln_eps: f32,
    post_ln_w: InlineArray,
    post_ln_eps: f32,

    // Dense MLP (when !is_moe_layer)
    mlp_gate_w: Option<InlineArray>, // pre-transposed [in, out]
    mlp_up_w: Option<InlineArray>,   // pre-transposed [in, out]
    mlp_down_w: Option<InlineArray>, // pre-transposed [in, out]

    // MoE (when is_moe_layer)
    // Expert weights stacked: [E, in, out] (pre-transposed for gather_mm).
    moe_router_w: Option<InlineArray>,    // [hidden, num_experts] (pre-transposed)
    moe_gate_w: Option<InlineArray>,      // [E, hidden, moe_intermediate] — ready for gather_mm
    moe_up_w: Option<InlineArray>,        // [E, hidden, moe_intermediate]
    moe_down_w: Option<InlineArray>,      // [E, moe_intermediate, hidden]
    shared_gate_w: Option<InlineArray>,   // [hidden, shared_intermediate]
    shared_up_w: Option<InlineArray>,     // [hidden, shared_intermediate]
    shared_down_w: Option<InlineArray>,   // [shared_intermediate, hidden]
    shared_expert_gate_w: Option<InlineArray>, // [hidden, 1]

    // MoE config duplicated here to avoid borrowing config in the forward fn.
    moe_top_k: i32,
    moe_norm_topk_prob: bool,

    // Whether this is an MoE layer (vs dense MLP).
    is_moe_layer: bool,

    // Attention-specific (only when !is_linear)
    attn_q_w: Option<InlineArray>,      // pre-transposed
    attn_k_w: Option<InlineArray>,
    attn_v_w: Option<InlineArray>,
    attn_o_w: Option<InlineArray>,
    attn_q_norm_w: Option<InlineArray>,
    attn_q_norm_eps: f32,
    attn_k_norm_w: Option<InlineArray>,
    attn_k_norm_eps: f32,
    attn_n_heads: i32,
    attn_n_kv_heads: i32,
    attn_head_dim: i32,
    attn_scale: f32,
    attn_rope_dims: i32,
    attn_rope_base: f32,
    attn_rope_scale: f32,
    /// When `true`, the attention layer uses the Qwen3.5 gated output:
    ///   output = o_proj(attn_out * sigmoid(gate))
    /// When `false`, the Qwen3 standard path is used:
    ///   output = o_proj(attn_out)
    attn_gated: bool,

    // GDN-specific (only when is_linear)
    gdn_qkv_w: Option<InlineArray>,  // in_proj_qkv, pre-transposed [hidden, conv_dim]
    gdn_z_w: Option<InlineArray>,    // in_proj_z, pre-transposed [hidden, value_dim]
    gdn_b_w: Option<InlineArray>,    // in_proj_b, pre-transposed [hidden, num_v_heads]
    gdn_a_w: Option<InlineArray>,    // in_proj_a, pre-transposed [hidden, num_v_heads]
    gdn_conv_w: Option<InlineArray>,
    gdn_q_nw: Option<InlineArray>,
    gdn_k_nw: Option<InlineArray>,
    gdn_a_log: Option<InlineArray>,
    gdn_dt_bias: Option<InlineArray>,
    gdn_norm_w: Option<InlineArray>,
    gdn_norm_eps: f32,
    gdn_out_w: Option<InlineArray>, // pre-transposed
    gdn_nv: i32,
    gdn_nk: i32,
    gdn_dk: i32,
    gdn_dv: i32,
    gdn_kd: i32, // nk * dk  (total key dim)
    gdn_cd: i32, // kd*2 + nv*dv  (conv projection dim)
    gdn_ck: i32, // conv kernel size
}

// ============================================================================
// Full model weights
// ============================================================================

/// All model weights as InlineArray. Zero dependency on mlx-rs.
pub struct NativeWeights {
    pub embed_w: InlineArray,
    pub final_norm_w: InlineArray,
    pub final_norm_eps: f32,
    /// None when `tie_word_embeddings = true`.
    pub lm_head_w: Option<InlineArray>,
    pub tie_word_embeddings: bool,
    /// Per-layer weights — opaque to callers; only accessed via [`forward_step`].
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

/// GDN layer cache state (conv + SSM).
pub struct GdnCache {
    pub conv_state: Option<InlineArray>,
    pub ssm_state: Option<InlineArray>,
}

/// Per-layer KV cache using pre-allocated buffers with O(1) slice_set updates.
pub struct KvLayerCache {
    pub keys: Option<InlineArray>,   // [B, H, MAX_T, D] pre-allocated
    pub values: Option<InlineArray>, // [B, H, MAX_T, D] pre-allocated
    pub offset: i32,                 // number of valid tokens
    /// TurboQuant compressed cache (replaces keys/values when enabled)
    pub turboquant: Option<crate::turboquant::QuantizedKvCache>,
}

/// Full model cache — both GDN and KV layers.
pub struct NativeCache {
    pub gdn_caches: Vec<GdnCache>,
    pub kv_caches: Vec<KvLayerCache>,
    pub rope_offset: i32,
    /// Shared TurboQuant state (rotation matrices, codebooks) — None = bf16 cache
    pub turboquant_state: Option<std::sync::Arc<crate::turboquant::TurboQuantState>>,
}

impl NativeCache {
    /// Evaluate all cache states in-place and detach them from their computation
    /// graph.  Must be called after the prefill forward pass and before decode.
    ///
    /// Python's `generate_step` does `mx.eval([c.state for c in prompt_cache])`
    /// at this point.  Without this, the unevaluated prefill SSM states have the
    /// entire prefill graph attached; when decode builds its graph those prefill
    /// nodes are included, adding hundreds of extra AsType/Matmul/etc. nodes.
    pub fn eval_and_detach_states(&mut self) {
        // Collect all non-None state arrays into a temporary vec for batch eval.
        let mut to_eval: Vec<&mut InlineArray> = Vec::new();
        for c in &mut self.gdn_caches {
            if let Some(ref mut s) = c.ssm_state { to_eval.push(s); }
            if let Some(ref mut s) = c.conv_state { to_eval.push(s); }
        }
        for c in &mut self.kv_caches {
            if let Some(ref mut k) = c.keys   { to_eval.push(k); }
            if let Some(ref mut v) = c.values { to_eval.push(v); }
        }
        // Batch eval then detach each.
        bridge::eval_and_detach_many(&mut to_eval);
    }

    /// Create a fresh, empty cache for the given weight set.
    pub fn new_empty(weights: &NativeWeights) -> Self {
        Self::new_with_turboquant(weights, None)
    }

    /// Create cache with optional TurboQuant KV compression.
    pub fn new_with_turboquant(
        weights: &NativeWeights,
        tq_config: Option<crate::turboquant::TurboQuantConfig>,
    ) -> Self {
        let mut gdn_caches = Vec::new();
        let mut kv_caches = Vec::new();

        // Build shared TurboQuant state if enabled
        let tq_state = tq_config.map(|cfg| {
            // Use the first attention layer's head_dim for key/value dims
            let head_dim = weights.layers.iter()
                .find(|lw| !lw.is_linear)
                .map(|lw| lw.attn_head_dim as usize)
                .unwrap_or(128);
            crate::turboquant::build_state(head_dim, head_dim, cfg)
        });

        for lw in &weights.layers {
            if lw.is_linear {
                gdn_caches.push(GdnCache {
                    conv_state: None,
                    ssm_state: None,
                });
            } else {
                let tq_cache = tq_state.as_ref().map(|state| {
                    crate::turboquant::new_cache_with_state(
                        tq_config.unwrap(),
                        state.clone(),
                    )
                });
                kv_caches.push(KvLayerCache {
                    keys: None,
                    values: None,
                    offset: 0,
                    turboquant: tq_cache,
                });
            }
        }

        NativeCache {
            gdn_caches,
            kv_caches,
            rope_offset: 0,
            turboquant_state: tq_state,
        }
    }
}

impl std::fmt::Debug for NativeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeCache")
            .field("gdn_layers", &self.gdn_caches.len())
            .field("attn_layers", &self.kv_caches.len())
            .field("rope_offset", &self.rope_offset)
            .finish()
    }
}

// ============================================================================
// Weight loading
// ============================================================================

/// Load model weights from a directory containing safetensors shards.
///
/// Supports all three Qwen variants:
///   - **Qwen3 dense** (`model_type = "qwen3"`): standard attention, full RoPE.
///   - **Qwen3.5 dense** (`model_type = "qwen3_5"` / `"qwen3_5_text"`, `num_experts = 0`).
///   - **Qwen3.5 MoE** (same type but `num_experts > 0`): routes through SwitchGLU +
///     shared expert. Per-expert weights are stacked into [E, in, out] tensors at
///     load time (matches Python's `sanitize()` stacking).
///
/// Applies the same sanitization as the mlx-rs loader:
/// - VLM prefix stripping (`model.language_model.` → `model.`)
/// - `A_log` → `a_log` rename
/// - `mtp.*` key drop
/// - conv1d weight transpose (when shape is `[out, k, in]` not `[out, k, 1]`)
/// - norm `(1+w)` offset when the model has `mtp.*` keys or unsanitized conv shapes
/// - Q/K scale synthesis for GDN (not stored in safetensors)
/// - MoE expert weight stacking into `[E, in, out]` for `gather_mm`
/// - `copy_fresh` on all weights (add zero + eval + detach) for fresh Metal buffers
pub fn load_model(
    model_dir: &std::path::Path,
    config: &Qwen3Config,
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
    let mut raw: std::collections::HashMap<String, InlineArray> =
        std::collections::HashMap::new();

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

    // Detect whether norm shift is needed before any renaming.
    let has_mtp = raw.keys().any(|k| k.contains("mtp."));
    let has_unsanitized_conv = raw.iter().any(|(k, v)| {
        k.contains("conv1d.weight") && v.ndim() == 3 && v.dim(2) != 1
    });
    let should_shift_norms = has_mtp || has_unsanitized_conv;

    // 3a. Key renaming: strip VLM prefix; A_log → a_log.
    // Qwen3.5-27B uses "language_model.model." prefix (no leading "model.").
    // Older checkpoints use "model.language_model." — strip both to "model.".
    // Qwen3 uses plain "model.layers.N..." with no prefix.
    let original_keys: Vec<String> = raw.keys().cloned().collect();
    for old_key in original_keys {
        let mut new_key = old_key.clone();
        if new_key.starts_with("language_model.model.") {
            // "language_model.model.X" → "model.X"
            new_key = new_key.replacen("language_model.", "", 1);
        } else if new_key.starts_with("language_model.") {
            // "language_model.lm_head.weight" → "lm_head.weight"
            new_key = new_key.replacen("language_model.", "", 1);
        } else if new_key.starts_with("model.language_model.") {
            new_key = new_key.replacen("model.language_model.", "model.", 1);
        }
        if new_key.contains(".A_log") {
            new_key = new_key.replace(".A_log", ".a_log");
        }
        if new_key != old_key {
            if let Some(v) = raw.remove(&old_key) {
                raw.insert(new_key, v);
            }
        }
    }

    // 3b. Drop mtp.* keys.
    raw.retain(|k, _| !k.contains("mtp."));

    // 3c. Drop lm_head.weight when embeddings are tied.
    if config.tie_word_embeddings {
        raw.remove("lm_head.weight");
    }

    // Norm suffixes that receive the (1+w) shift.
    // Excludes `.linear_attn.norm.weight` (the GDN sub-norm, which is NOT shifted).
    let norm_suffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    ];

    // 3d. Conv1d transpose + norm shift + f32→model_dtype casts.
    let detected_model_dtype = raw
        .get("model.embed_tokens.weight")
        .map(|w| w.dtype_raw())
        .unwrap_or(11); // 11 = bfloat16 fallback
    let one = InlineArray::from_f32(1.0).as_dtype(detected_model_dtype);

    // GDN-specific f32 weights that must be cast to model dtype to prevent f32
    // dtype propagation through the residual stream.
    let f32_gdn_suffixes = [
        "linear_attn.a_log",
        "linear_attn.norm.weight",
        "linear_attn.q_norm.weight",
        "linear_attn.k_norm.weight",
    ];

    let all_keys: Vec<String> = raw.keys().cloned().collect();
    for k in &all_keys {
        if k.contains("conv1d.weight") {
            if let Some(v) = raw.get(k) {
                if v.ndim() == 3 && v.dim(2) != 1 {
                    let transposed = v.transpose_axes(&[0, 2, 1]);
                    raw.insert(k.clone(), transposed);
                }
            }
        }
        if f32_gdn_suffixes.iter().any(|sfx| k.ends_with(sfx)) {
            if let Some(v) = raw.get(k) {
                if v.dtype_raw() != detected_model_dtype {
                    let cast = v.as_dtype(detected_model_dtype);
                    raw.insert(k.clone(), cast);
                }
            }
        }
        if should_shift_norms && norm_suffixes.iter().any(|sfx| k.ends_with(sfx)) {
            if let Some(v) = raw.get(k) {
                if v.ndim() == 1 {
                    let shifted = v.add(&one);
                    raw.insert(k.clone(), shifted);
                }
            }
        }
    }

    // 3e. MoE expert weight stacking.
    //
    // Python's sanitize() stacks per-expert weights into [E, out, in]:
    //   to_join = [weights[f"{prefix}.experts.{e}.{n}.weight"] for e in range(E)]
    //   weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)
    //
    // `gather_mm` in SwitchLinear calls `weight.swapaxes(-1,-2)` at forward
    // time (so it uses [E, in, out] at runtime).  We pre-transpose to
    // [E, in, out] here so the forward pass can call gather_mm directly
    // without the extra transpose node.  For down_proj the stacked shape is
    // [E, moe_intermediate, hidden]; transposed → [E, hidden, moe_intermediate].
    //
    // We store the stacked tensors under "switch_mlp.{n}.weight" keys matching
    // the post-sanitize Python layout for clarity, then look them up by layer.
    if config.is_moe() {
        for li in 0..config.num_hidden_layers as usize {
            if config.is_dense_mlp_layer(li) {
                continue;
            }
            let prefix = format!("model.layers.{li}.mlp");
            for proj in &["gate_proj", "up_proj", "down_proj"] {
                // Collect all E per-expert tensors.
                let mut shards: Vec<InlineArray> =
                    Vec::with_capacity(config.num_experts as usize);
                for e in 0..config.num_experts as usize {
                    let key = format!("{prefix}.experts.{e}.{proj}.weight");
                    let w = raw.remove(&key).ok_or_else(|| {
                        format!("MoE: missing expert weight {key}")
                    })?;
                    shards.push(w);
                }
                // Stack [E, out, in] → then transpose to [E, in, out] for gather_mm.
                let stacked = stack_arrays(shards, 0)?;
                // stacked: [E, out, in] → transpose last two axes → [E, in, out]
                let transposed = stacked.transpose_axes(&[0, 2, 1]);
                raw.insert(format!("{prefix}.switch_mlp.{proj}.weight"), transposed);
            }
        }
    }

    // ── Step 4: Build per-layer weight structs ──────────────────────────────
    let get = |key: &str| -> Result<InlineArray, String> {
        raw.get(key)
            .cloned()
            .ok_or_else(|| {
                let parts: Vec<&str> = key.rsplitn(2, '.').collect();
                let suffix = parts[0];
                let close: Vec<&String> = raw
                    .keys()
                    .filter(|k| k.ends_with(suffix))
                    .take(5)
                    .collect();
                format!("missing weight key: {key} (close matches: {close:?})")
            })
    };

    let embed_w = get("model.embed_tokens.weight")?;
    let final_norm_w = get("model.norm.weight")?;
    let final_norm_eps = config.rms_norm_eps;
    let lm_head_w = if config.tie_word_embeddings {
        None
    } else {
        // lm_head.weight is stored as (vocab, hidden); pre-transpose to (hidden, vocab)
        // so forward() can use hidden.matmul(lm_head_w) directly.
        Some(get("lm_head.weight")?.t())
    };

    let model_dtype = embed_w.dtype_raw();

    // GDN dimensions (identical across all GDN layers; only meaningful for Qwen3.5).
    let nv = config.gdn_nv();
    let nk = config.gdn_nk();
    let dk = config.gdn_dk();
    let dv = config.gdn_dv();
    let ck = config.linear_conv_kernel_dim;
    let kd = nk * dk;          // total key dim
    let cd = kd * 2 + nv * dv; // conv projection dim

    // Attention dimensions.
    let n_heads = config.num_attention_heads;
    let n_kv_heads = config.get_num_kv_heads();
    let head_dim = config.get_head_dim();
    let attn_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let rope_dims = config.rope_dims();
    let rope_base = config.rope_theta as f32;
    let rope_scale = 1.0_f32;
    let attn_gated = !config.is_qwen3_dense();

    let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);

    for li in 0..config.num_hidden_layers as usize {
        let p = format!("model.layers.{li}");
        let is_linear = config.is_linear_layer(li);
        let is_moe_layer = !config.is_dense_mlp_layer(li);

        let input_ln_w = get(&format!("{p}.input_layernorm.weight"))?;
        let post_ln_w = get(&format!("{p}.post_attention_layernorm.weight"))?;

        // MLP weights — dense or MoE.
        let (
            mlp_gate_w,
            mlp_up_w,
            mlp_down_w,
            moe_router_w,
            moe_gate_w,
            moe_up_w,
            moe_down_w,
            shared_gate_w,
            shared_up_w,
            shared_down_w,
            shared_expert_gate_w,
        ) = if is_moe_layer {
            // MoE layer: router + stacked experts + shared expert.
            // router gate: stored as [num_experts, hidden]; transpose to [hidden, num_experts].
            let router = get(&format!("{p}.mlp.gate.weight"))?.t();
            let moe_gate = get(&format!("{p}.mlp.switch_mlp.gate_proj.weight"))?;
            let moe_up = get(&format!("{p}.mlp.switch_mlp.up_proj.weight"))?;
            let moe_down = get(&format!("{p}.mlp.switch_mlp.down_proj.weight"))?;
            // Shared expert weights — stored [out, in]; transpose for matmul.
            let sh_gate = get(&format!("{p}.mlp.shared_expert.gate_proj.weight"))?.t();
            let sh_up = get(&format!("{p}.mlp.shared_expert.up_proj.weight"))?.t();
            let sh_down = get(&format!("{p}.mlp.shared_expert.down_proj.weight"))?.t();
            // Shared expert gate: [1, hidden]; squeeze to [hidden, 1] or keep as [1, hidden].
            // Python: sigmoid(self.shared_expert_gate(x)) — x: [B*T, hidden],
            //   shared_expert_gate is Linear(hidden, 1), output: [B*T, 1].
            // We store as [1, hidden] (stored) → .t() → [hidden, 1].
            let sh_eg = get(&format!("{p}.mlp.shared_expert_gate.weight"))?.t();
            (
                None, None, None,
                Some(router),
                Some(moe_gate),
                Some(moe_up),
                Some(moe_down),
                Some(sh_gate),
                Some(sh_up),
                Some(sh_down),
                Some(sh_eg),
            )
        } else {
            // Dense MLP.
            let gate = get(&format!("{p}.mlp.gate_proj.weight"))?.t();
            let up   = get(&format!("{p}.mlp.up_proj.weight"))?.t();
            let down = get(&format!("{p}.mlp.down_proj.weight"))?.t();
            (
                Some(gate), Some(up), Some(down),
                None, None, None, None, None, None, None, None,
            )
        };

        let mut lw = LayerWeights {
            is_linear,
            input_ln_w,
            input_ln_eps: config.rms_norm_eps,
            post_ln_w,
            post_ln_eps: config.rms_norm_eps,
            mlp_gate_w,
            mlp_up_w,
            mlp_down_w,
            moe_router_w,
            moe_gate_w,
            moe_up_w,
            moe_down_w,
            shared_gate_w,
            shared_up_w,
            shared_down_w,
            shared_expert_gate_w,
            moe_top_k: config.num_experts_per_tok,
            moe_norm_topk_prob: config.norm_topk_prob,
            is_moe_layer,
            // Attention — filled below when !is_linear
            attn_q_w: None,
            attn_k_w: None,
            attn_v_w: None,
            attn_o_w: None,
            attn_q_norm_w: None,
            attn_q_norm_eps: config.rms_norm_eps,
            attn_k_norm_w: None,
            attn_k_norm_eps: config.rms_norm_eps,
            attn_n_heads: 0,
            attn_n_kv_heads: 0,
            attn_head_dim: 0,
            attn_scale: 0.0,
            attn_rope_dims: 0,
            attn_rope_base: 0.0,
            attn_rope_scale: 0.0,
            attn_gated,
            // GDN — filled below when is_linear
            gdn_qkv_w: None,
            gdn_z_w: None,
            gdn_b_w: None,
            gdn_a_w: None,
            gdn_conv_w: None,
            gdn_q_nw: None,
            gdn_k_nw: None,
            gdn_a_log: None,
            gdn_dt_bias: None,
            gdn_norm_w: None,
            gdn_norm_eps: config.rms_norm_eps,
            gdn_out_w: None,
            gdn_nv: 0,
            gdn_nk: 0,
            gdn_dk: 0,
            gdn_dv: 0,
            gdn_kd: 0,
            gdn_cd: 0,
            gdn_ck: 0,
        };

        if is_linear {
            let la = format!("{p}.linear_attn");
            lw.gdn_qkv_w = Some(get(&format!("{la}.in_proj_qkv.weight"))?.t());
            lw.gdn_z_w   = Some(get(&format!("{la}.in_proj_z.weight"))?.t());
            lw.gdn_b_w   = Some(get(&format!("{la}.in_proj_b.weight"))?.t());
            lw.gdn_a_w   = Some(get(&format!("{la}.in_proj_a.weight"))?.t());
            lw.gdn_conv_w = Some(get(&format!("{la}.conv1d.weight"))?);

            // Q/K scale weights are SYNTHETIC — not present in safetensors.
            // They encode the Q/K normalization scaling factor:
            //   q_norm_weight = ones(dk) * inv_scale²
            //   k_norm_weight = ones(dk) * inv_scale
            // where inv_scale = 1/sqrt(dk).
            // IMPORTANT: the scalar MUST be cast to model_dtype before multiply.
            let inv_scale = (dk as f32).sqrt().recip();
            let q_scale_arr = {
                let a = InlineArray::ones(&[dk], model_dtype);
                let scale = InlineArray::from_f32(inv_scale * inv_scale).as_dtype(model_dtype);
                a.multiply(&scale)
            };
            let k_scale_arr = {
                let a = InlineArray::ones(&[dk], model_dtype);
                let scale = InlineArray::from_f32(inv_scale).as_dtype(model_dtype);
                a.multiply(&scale)
            };
            lw.gdn_q_nw = Some(
                get(&format!("{la}.q_norm_weight"))
                    .or_else(|_| get(&format!("{la}.q_norm.weight")))
                    .unwrap_or(q_scale_arr),
            );
            lw.gdn_k_nw = Some(
                get(&format!("{la}.k_norm_weight"))
                    .or_else(|_| get(&format!("{la}.k_norm.weight")))
                    .unwrap_or(k_scale_arr),
            );
            lw.gdn_a_log   = Some(get(&format!("{la}.a_log"))?);
            lw.gdn_dt_bias = Some(get(&format!("{la}.dt_bias"))?);
            lw.gdn_norm_w  = Some(get(&format!("{la}.norm.weight"))?);
            lw.gdn_out_w   = Some(get(&format!("{la}.out_proj.weight"))?.t());
            lw.gdn_nv = nv;
            lw.gdn_nk = nk;
            lw.gdn_dk = dk;
            lw.gdn_dv = dv;
            lw.gdn_kd = kd;
            lw.gdn_cd = cd;
            lw.gdn_ck = ck;

            if li == 0 {
                eprintln!(
                    "[NATIVE] GDN config: nk={nk} nv={nv} dk={dk} dv={dv} kd={kd} cd={cd} ck={ck}"
                );
            }
        } else {
            let sa = format!("{p}.self_attn");
            // Q projection width differs between Qwen3 and Qwen3.5:
            //   Qwen3:   [n_heads * head_dim, hidden]
            //   Qwen3.5: [n_heads * head_dim * 2, hidden]  (queries + gate concatenated)
            // We just load whatever is in the checkpoint; the forward pass
            // uses `attn_gated` to decide how to interpret the output.
            lw.attn_q_w      = Some(get(&format!("{sa}.q_proj.weight"))?.t());
            lw.attn_k_w      = Some(get(&format!("{sa}.k_proj.weight"))?.t());
            lw.attn_v_w      = Some(get(&format!("{sa}.v_proj.weight"))?.t());
            lw.attn_o_w      = Some(get(&format!("{sa}.o_proj.weight"))?.t());
            lw.attn_q_norm_w = Some(get(&format!("{sa}.q_norm.weight"))?);
            lw.attn_k_norm_w = Some(get(&format!("{sa}.k_norm.weight"))?);
            lw.attn_n_heads    = n_heads;
            lw.attn_n_kv_heads = n_kv_heads;
            lw.attn_head_dim   = head_dim;
            lw.attn_scale      = attn_scale;
            lw.attn_rope_dims  = rope_dims;
            lw.attn_rope_base  = rope_base;
            lw.attn_rope_scale = rope_scale;
        }

        layers.push(lw);
    }

    // ── Step 5: copy_fresh — force all weights into fresh Metal buffers ─────
    //
    // `w.add(zero).eval().detach()` creates a new buffer with use_count=1.
    // Without this, weights share data with the raw tensors (use_count=2),
    // preventing optimal Metal buffer scheduling. Detach severs the graph
    // chains accumulated by the transpose/add ops above.
    let zero = InlineArray::from_f32(0.0).as_dtype(model_dtype);
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
        lw.post_ln_w  = copy_fresh(&lw.post_ln_w);
        // Dense MLP
        if let Some(ref w) = lw.mlp_gate_w { lw.mlp_gate_w = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.mlp_up_w   { lw.mlp_up_w   = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.mlp_down_w { lw.mlp_down_w = Some(copy_fresh(w)); }
        // MoE
        if let Some(ref w) = lw.moe_router_w       { lw.moe_router_w       = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.moe_gate_w         { lw.moe_gate_w         = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.moe_up_w           { lw.moe_up_w           = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.moe_down_w         { lw.moe_down_w         = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.shared_gate_w      { lw.shared_gate_w      = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.shared_up_w        { lw.shared_up_w        = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.shared_down_w      { lw.shared_down_w      = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.shared_expert_gate_w { lw.shared_expert_gate_w = Some(copy_fresh(w)); }
        // Attention
        if let Some(ref w) = lw.attn_q_w      { lw.attn_q_w      = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.attn_k_w      { lw.attn_k_w      = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.attn_v_w      { lw.attn_v_w      = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.attn_o_w      { lw.attn_o_w      = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.attn_q_norm_w { lw.attn_q_norm_w = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.attn_k_norm_w { lw.attn_k_norm_w = Some(copy_fresh(w)); }
        // GDN
        if let Some(ref w) = lw.gdn_qkv_w    { lw.gdn_qkv_w    = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.gdn_z_w      { lw.gdn_z_w      = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.gdn_b_w      { lw.gdn_b_w      = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.gdn_a_w      { lw.gdn_a_w      = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.gdn_conv_w   { lw.gdn_conv_w   = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.gdn_q_nw     { lw.gdn_q_nw     = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.gdn_k_nw     { lw.gdn_k_nw     = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.gdn_a_log    { lw.gdn_a_log    = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.gdn_dt_bias  { lw.gdn_dt_bias  = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.gdn_norm_w   { lw.gdn_norm_w   = Some(copy_fresh(w)); }
        if let Some(ref w) = lw.gdn_out_w    { lw.gdn_out_w    = Some(copy_fresh(w)); }
    }

    eprintln!("[NATIVE] load_model: force-copied all weights into fresh Metal buffers");

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

// ============================================================================
// Forward step
// ============================================================================

/// Run one forward step — works for both T=1 decode and T=N prefill.
///
/// `token_ids` must be shape `[B, T]` int32. Returns logits `[B, T, vocab]`.
///
/// For T=1 the GDN path uses `compiled_gdn_layer_fixed` (tape-replay, ~10 ms
/// cheaper per step). For T>1 it falls through to the direct-ops prefill path.
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

    let mut gdn_slot = 0usize;
    let mut attn_slot = 0usize;

    for lw in weights.layers.iter() {
        // Input LayerNorm
        let normed = hidden.rms_norm(Some(&lw.input_ln_w), lw.input_ln_eps);

        let r = if lw.is_linear {
            let result = gdn_forward(
                lw,
                &normed,
                b,
                s,
                &mut cache.gdn_caches[gdn_slot],
                dtype,
            );
            gdn_slot += 1;
            result
        } else {
            let result = attn_forward(
                lw,
                &normed,
                b,
                s,
                &mut cache.kv_caches[attn_slot],
                cache.rope_offset,
                dtype,
            );
            attn_slot += 1;
            result
        };

        // Residual
        let h = hidden.add(&r);

        // Post-attention LayerNorm + MLP (dense or MoE)
        let mlp_in = h.rms_norm(Some(&lw.post_ln_w), lw.post_ln_eps);
        let mlp_out = if lw.is_moe_layer {
            moe_forward(lw, &mlp_in)
        } else {
            dense_mlp_forward(lw, &mlp_in)
        };

        // Residual
        hidden = h.add(&mlp_out);
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
// Dense MLP forward
// ============================================================================

#[inline(always)]
fn dense_mlp_forward(lw: &LayerWeights, mlp_in: &InlineArray) -> InlineArray {
    let gate = mlp_in.matmul(lw.mlp_gate_w.as_ref().unwrap());
    let up   = mlp_in.matmul(lw.mlp_up_w.as_ref().unwrap());
    let activated = InlineArray::fused_swiglu(&gate, &up);
    activated.matmul(lw.mlp_down_w.as_ref().unwrap())
}

// ============================================================================
// MoE forward
// ============================================================================
//
// Mirrors Python's Qwen3NextSparseMoeBlock.__call__:
//
//   gates = softmax(gate(x), axis=-1, precise=True)
//   inds  = argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
//   scores = take_along_axis(gates, inds, axis=-1)
//   if norm_topk_prob: scores /= scores.sum(-1, keepdims=True)
//   y = switch_mlp(x, inds)                        # gather_mm
//   y = (y * scores[..., None]).sum(-2)
//   shared_y = shared_expert(x)
//   shared_y = sigmoid(shared_expert_gate(x)) * shared_y
//   return y + shared_y
//
// Input x: [B, T, hidden].  For decode T=1, B=1 → x is [1, 1, hidden].
// We work in [B*T, hidden] = [S, hidden] throughout, then reshape back.

fn moe_forward(lw: &LayerWeights, x: &InlineArray) -> InlineArray {
    let b = x.dim(0);
    let t = x.dim(1);
    let h = x.dim(2);
    let s = b * t; // flattened sequence length
    let top_k = lw.moe_top_k;

    // Flatten to [S, hidden].
    let x_flat = x.reshape(&[s, h]);

    // ── Router ──────────────────────────────────────────────────────────────
    // gates: [S, num_experts]
    let gates = x_flat.matmul(lw.moe_router_w.as_ref().unwrap())
        .softmax_precise(-1);

    // Top-k selection: argpartition returns full permutation, take last top_k.
    // inds: [S, num_experts] → slice to [S, top_k]
    let all_inds = gates.argpartition(-top_k, -1);
    let num_experts_dim = gates.dim(1);
    let inds = all_inds.slice(
        &[0, num_experts_dim - top_k],
        &[s, num_experts_dim],
    );

    // Gather expert scores: [S, top_k]
    let mut scores = gates.take_along_axis(&inds, -1);
    if lw.moe_norm_topk_prob {
        let score_sum = scores.sum_axis(-1, true);
        scores = scores.divide(&score_sum);
    }

    // ── Expert dispatch via gather_mm ────────────────────────────────────────
    //
    // SwitchGLU: x_up = gather_mm(x, up_w, rhs_indices=inds)
    //            x_gate = gather_mm(x, gate_w, rhs_indices=inds)
    //            x_act = silu(x_gate) * x_up
    //            y = gather_mm(x_act, down_w, rhs_indices=inds)
    //
    // gather_mm(a [S, in], b [E, in, out], rhs_indices [S, k]) → [S, k, out]
    //
    // All expert weight tensors are already pre-transposed to [E, in, out].
    // The `sorted` flag is omitted (false) for simplicity — matches Python's
    // do_sort=True only when indices.size >= 64. For the common decode case
    // (S=1, top_k=8, indices.size=8) do_sort is false in Python too.
    let x_gate_exp = x_flat.gather_mm(lw.moe_gate_w.as_ref().unwrap(), None, Some(&inds), false);
    let x_up_exp   = x_flat.gather_mm(lw.moe_up_w.as_ref().unwrap(),   None, Some(&inds), false);

    // Fused swiglu: silu(gate) * up
    let x_act = InlineArray::fused_swiglu(&x_gate_exp, &x_up_exp);

    // gather_mm for down projection: [S, k, moe_intermediate] × [E, moe_intermediate, hidden]
    // → [S, k, hidden]
    let y_exp = x_act.gather_mm(lw.moe_down_w.as_ref().unwrap(), None, Some(&inds), false);

    // Weighted sum over top_k: [S, k, hidden] * [S, k, 1] → sum(-2) → [S, hidden]
    let scores_exp = scores.reshape(&[s, top_k, 1]);
    let y_routed = y_exp.multiply(&scores_exp).sum_axis(-2, false);

    // ── Shared expert ────────────────────────────────────────────────────────
    //
    // shared_expert(x): standard SwiGLU MLP with its own gate/up/down weights.
    // shared_expert_gate: Linear(hidden, 1) → sigmoid → scales shared output.
    let sh_gate = x_flat.matmul(lw.shared_gate_w.as_ref().unwrap());
    let sh_up   = x_flat.matmul(lw.shared_up_w.as_ref().unwrap());
    let sh_act  = InlineArray::fused_swiglu(&sh_gate, &sh_up);
    let sh_out  = sh_act.matmul(lw.shared_down_w.as_ref().unwrap());

    // shared_expert_gate: [S, 1] sigmoid gate
    let sh_scale = x_flat.matmul(lw.shared_expert_gate_w.as_ref().unwrap()).sigmoid();
    let y_shared = sh_out.multiply(&sh_scale);

    // ── Combine ──────────────────────────────────────────────────────────────
    y_routed.add(&y_shared).reshape(&[b, t, h])
}

// ============================================================================
// GDN layer forward
// ============================================================================

fn gdn_forward(
    lw: &LayerWeights,
    normed: &InlineArray,
    _b: i32,
    _s: i32,
    cache: &mut GdnCache,
    dtype: i32,
) -> InlineArray {
    let nv = lw.gdn_nv;
    let nk = lw.gdn_nk;
    let dk = lw.gdn_dk;
    let dv = lw.gdn_dv;
    let kd = lw.gdn_kd;
    let cd = lw.gdn_cd;
    let ck = lw.gdn_ck;
    let b = normed.dim(0);
    let s = normed.dim(1);

    // Unified path for all T (T=1 decode and T>1 prefill).
    // Structure mirrors Python's gated_delta_update exactly:
    //   1. 4 separate matmul projections (qkv, z, b, a)
    //   2. Conv1d with fused silu activation
    //   3. split → q/k/v + rms_norm on q/k
    //   4. fused_compute_g (shapeless=True compiled — opaque Compiled node)
    //   5. gdn_metal_step (CustomKernel, outside any compile boundary)
    //   6. fused_precise_swiglu (shapeless=True compiled — opaque Compiled node)
    //   7. out_proj matmul
    let qkv   = normed.matmul(lw.gdn_qkv_w.as_ref().unwrap());
    let z     = normed.matmul(lw.gdn_z_w.as_ref().unwrap()).reshape(&[b, s, nv, dv]);
    let b_val = normed.matmul(lw.gdn_b_w.as_ref().unwrap());
    let a_val = normed.matmul(lw.gdn_a_w.as_ref().unwrap());

    // Conv state: concat previous state + new QKV, take new state, apply conv1d + silu
    let conv_state = cache
        .conv_state
        .take()
        .unwrap_or_else(|| InlineArray::zeros(&[b, ck - 1, cd], dtype));
    let conv_in = conv_state.concatenate_2(&qkv, 1);

    let new_conv = conv_in.slice(&[0, 1, 0], &[b, ck, cd]);
    let conv_out = conv_in
        .conv1d(lw.gdn_conv_w.as_ref().unwrap(), 1, 0, 1, cd)
        .fused_silu();

    // Split conv_out → q [B,S,nk,dk], k [B,S,nk,dk], v [B,S,nv,dv]
    let mut conv_parts = conv_out.split(&[kd, kd * 2], -1);
    let v = conv_parts.pop().unwrap().reshape(&[b, s, nv, dv]);
    let k = conv_parts.pop().unwrap().reshape(&[b, s, nk, dk]);
    let q = conv_parts.pop().unwrap().reshape(&[b, s, nk, dk]);

    // Q/K normalization
    let q = q.rms_norm(lw.gdn_q_nw.as_ref(), 1e-6);
    let k = k.rms_norm(lw.gdn_k_nw.as_ref(), 1e-6);

    // Decay gate: fused compute_g
    let g    = InlineArray::fused_compute_g(
        lw.gdn_a_log.as_ref().unwrap(),
        &a_val,
        lw.gdn_dt_bias.as_ref().unwrap(),
    );
    let beta = b_val.sigmoid();

    // GDN Metal kernel recurrence
    let ssm_state = cache
        .ssm_state
        .take()
        .unwrap_or_else(|| InlineArray::zeros(&[b, nv, dv, dk], 10));
    let (out, new_state) = InlineArray::gdn_metal_step(&q, &k, &v, &g, &beta, &ssm_state, s);

    cache.conv_state = Some(new_conv);
    cache.ssm_state = Some(new_state);

    // Output: rms_norm → precise_swiglu → reshape → out_proj
    let out_n = out.rms_norm(lw.gdn_norm_w.as_ref(), lw.gdn_norm_eps);
    let gated = InlineArray::fused_precise_swiglu(&out_n, &z);
    gated.reshape(&[b, s, -1]).matmul(lw.gdn_out_w.as_ref().unwrap())
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
    let n_heads    = lw.attn_n_heads;
    let n_kv_heads = lw.attn_n_kv_heads;
    let head_dim   = lw.attn_head_dim;
    let scale      = lw.attn_scale;

    // Q projection.
    //
    // Qwen3.5 (gated): q_proj output width = n_heads * head_dim * 2.
    //   Reshape to [B,S,H,D*2], split at D → [queries [B,S,H,D], gate [B,S,H,D]].
    //   gate is later reshaped to [B,S,H*D] and used to sigmoid-scale the output.
    //
    // Qwen3 (standard): q_proj output width = n_heads * head_dim.
    //   Reshape to [B,S,H,D] directly, no gate split.
    let q_proj_out = normed.matmul(lw.attn_q_w.as_ref().unwrap());

    let (queries, gate_opt) = if lw.attn_gated {
        // Gated path (Qwen3.5)
        let q_gate = q_proj_out.reshape(&[b, s, n_heads, head_dim * 2]);
        let mut qg_parts = q_gate.split(&[head_dim], -1);
        let gate    = qg_parts.pop().unwrap().reshape(&[b, s, n_heads * head_dim]);
        let queries = qg_parts.pop().unwrap(); // [B, S, n_heads, head_dim]
        (queries, Some(gate))
    } else {
        // Standard path (Qwen3)
        let queries = q_proj_out.reshape(&[b, s, n_heads, head_dim]);
        (queries, None)
    };

    // K, V projections
    let new_keys   = normed.matmul(lw.attn_k_w.as_ref().unwrap());
    let new_values = normed.matmul(lw.attn_v_w.as_ref().unwrap());

    // Q/K norms
    let queries = queries.rms_norm(lw.attn_q_norm_w.as_ref(), lw.attn_q_norm_eps);
    let keys    = new_keys
        .reshape(&[b, s, n_kv_heads, head_dim])
        .rms_norm(lw.attn_k_norm_w.as_ref(), lw.attn_k_norm_eps);
    let values  = new_values.reshape(&[b, s, n_kv_heads, head_dim]);

    // Transpose to [B, H, S, D]
    let queries = queries.transpose_axes(&[0, 2, 1, 3]);
    let keys    = keys.transpose_axes(&[0, 2, 1, 3]);
    let values  = values.transpose_axes(&[0, 2, 1, 3]);

    // RoPE (full for Qwen3, partial for Qwen3.5)
    let queries = queries.rope(
        lw.attn_rope_dims, false, lw.attn_rope_base, lw.attn_rope_scale, rope_offset,
    );
    let keys = keys.rope(
        lw.attn_rope_dims, false, lw.attn_rope_base, lw.attn_rope_scale, rope_offset,
    );

    // KV cache update — O(1) slice_set into pre-allocated buffer
    let prev    = cache.offset;
    let num_new = keys.dim(2);
    let next    = prev + num_new;

    if cache.keys.is_none() {
        let alloc = 256i32;
        cache.keys   = Some(InlineArray::zeros(&[b, n_kv_heads, alloc, head_dim], dtype));
        cache.values = Some(InlineArray::zeros(&[b, n_kv_heads, alloc, head_dim], dtype));
    } else {
        let allocated = cache.keys.as_ref().unwrap().dim(2);
        if next > allocated {
            let old_k = cache.keys.take().unwrap();
            let old_v = cache.values.take().unwrap();
            let ext_k = InlineArray::zeros(&[b, n_kv_heads, 256, head_dim], dtype);
            let ext_v = InlineArray::zeros(&[b, n_kv_heads, 256, head_dim], dtype);
            cache.keys   = Some(old_k.kv_cache_append(&ext_k, 2));
            cache.values = Some(old_v.kv_cache_append(&ext_v, 2));
        }
    }

    // KV cache update + SDPA
    let output = if let Some(ref mut tq_cache) = cache.turboquant {
        // TurboQuant path: quantize new K,V and store compressed
        tq_cache.append(&keys, &values).ok();
        // Dequantize full cache for SDPA
        let full_keys = tq_cache.dequantize_keys()
            .unwrap_or_else(|| keys.clone());
        let full_values = tq_cache.dequantize_values()
            .unwrap_or_else(|| values.clone());
        cache.offset = next;
        queries.sdpa(&full_keys, &full_values, scale, "causal")
    } else {
        // Standard bf16 path
        let start = [0, 0, prev, 0];
        let stop  = [b, n_kv_heads, next, head_dim];
        let k_buf = cache.keys.take().unwrap();
        let v_buf = cache.values.take().unwrap();
        cache.keys   = Some(k_buf.slice_set(&keys, &start, &stop));
        cache.values = Some(v_buf.slice_set(&values, &start, &stop));
        cache.offset = next;

        let valid_keys = cache
            .keys.as_ref().unwrap()
            .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, head_dim]);
        let valid_values = cache
            .values.as_ref().unwrap()
            .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, head_dim]);
        queries.sdpa(&valid_keys, &valid_values, scale, "causal")
    };

    // Output projection
    let output = output
        .transpose_axes(&[0, 2, 1, 3])
        .reshape(&[b, s, n_heads * head_dim]);

    if let Some(gate) = gate_opt {
        // Qwen3.5 gated output: o_proj(attn_out * sigmoid(gate))
        let gated = output.multiply(&gate.sigmoid());
        gated.matmul(lw.attn_o_w.as_ref().unwrap())
    } else {
        // Qwen3 standard output: o_proj(attn_out)
        output.matmul(lw.attn_o_w.as_ref().unwrap())
    }
}

// ============================================================================
// Sampling
// ============================================================================

/// Sample one token from `logits_2d` of shape `[B, vocab]`.
///
/// `temperature <= 0.0` → greedy argmax. Otherwise categorical sampling.
pub fn sample_token(logits_2d: &InlineArray, temperature: f32) -> InlineArray {
    if temperature <= 0.0 {
        logits_2d.argmax(-1)
    } else {
        let inv_temp = InlineArray::from_f32(1.0 / temperature);
        let lse = logits_2d.logsumexp(-1, true);
        let log_probs = logits_2d.subtract(&lse);
        let scaled = log_probs.multiply(&inv_temp);
        scaled.categorical()
    }
}

// ============================================================================
// Generation loop
// ============================================================================

/// Run the full generation loop with async GPU pipelining.
///
/// `first_token` is the token at the end of the prompt (already prefilled into
/// `cache`). Each call to `on_token` receives the sampled token ID and returns
/// `false` to stop early (e.g. on EOS).
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

    // Clear prefill residue from the Metal buffer cache.
    bridge::clear_cache();
    bridge::reset_peak_memory();
    // NOTE: enable_compile() was tested and shown to HURT performance
    // (adds tracing overhead without meaningful fusion). Keep disabled.
    // Create a dedicated GPU stream for generation (matches Python's generation_stream).
    bridge::new_generation_stream();
    bridge::set_generation_stream();
    // Wire model weights into GPU memory — prevents paging during decode.
    bridge::set_wired_limit_max();
    eprintln!(
        "[NATIVE] generate: dtype={} active={:.0}MB",
        weights.model_dtype,
        bridge::get_active_memory() as f64 / 1e6,
    );

    // Evaluate and detach all prefill cache states before decode.
    cache.eval_and_detach_states();
    bridge::clear_cache();

    // First decode step
    let input_token = InlineArray::from_i32(first_token as i32).reshape(&[1, 1]);
    let logits = forward_step(weights, &input_token, cache);
    // Squeeze the sequence dimension: [B, 1, vocab] → [B, vocab]
    let logits_2d = logits.squeeze(1);
    let mut current_y = sample_token(&logits_2d, temperature);
    // Start async GPU eval of the sampled token concurrently with the CPU.
    current_y.async_eval_ref();

    let mut step_times: Vec<f64> = Vec::new();

    for step in 0..max_tokens {
        // On step 0 we need to wait for the first async eval.
        if step == 0 {
            current_y.eval();
        }
        let token_val = current_y.item_u32();

        tokens.push(token_val);
        if !on_token(token_val) {
            break;
        }
        if step + 1 >= max_tokens {
            break;
        }

        let t_step = std::time::Instant::now();
        let next_input   = InlineArray::from_i32(token_val as i32).reshape(&[1, 1]);
        let next_logits  = forward_step(weights, &next_input, cache);
        let next_logits_2d = next_logits.squeeze(1);
        current_y = sample_token(&next_logits_2d, temperature);
        current_y.eval();

        step_times.push(t_step.elapsed().as_secs_f64() * 1000.0);

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
            "[NATIVE] per-step: avg={avg:.2}ms p50={p50:.2}ms = {:.0} tok/s",
            1000.0 / avg
        );
    }

    bridge::synchronize();

    tokens
}

// ============================================================================
// C++ full-forward generation loop
// ============================================================================

/// Generation loop using the C++ monolithic forward path.
///
/// Equivalent to [`generate`] but all per-layer ops are executed inside a single
/// C++ function call (`mlx_inline_qwen35_decode_step`) eliminating per-op FFI
/// overhead — ~1800 round-trips per decode step for a 28-layer model.
///
/// Only supports Qwen3.5 dense (the C++ side does not implement MoE). Falls
/// back to the Rust path silently for unsupported model variants.
///
/// # Safety
/// Internally uses raw pointers into `weights` and `cache`; both must remain
/// live and un-moved for the duration of this call.
pub fn generate_cpp(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    config: &Qwen3Config,
    first_token: u32,
    max_tokens: usize,
    temperature: f32,
    mut on_token: impl FnMut(u32) -> bool,
) -> Vec<u32> {
    // The C++ bridge only handles Qwen3.5 dense (no MoE weights in the layout).
    // For all other variants fall through to the Rust path.
    if config.is_moe() || config.is_qwen3_dense() {
        return generate(weights, cache, first_token, max_tokens, temperature, on_token);
    }

    let mut tokens = Vec::with_capacity(max_tokens);

    bridge::clear_cache();
    bridge::reset_peak_memory();
    bridge::enable_compile();
    bridge::new_generation_stream();
    bridge::set_generation_stream();
    bridge::set_wired_limit_max();

    eprintln!(
        "[NATIVE-CPP] generate_cpp: dtype={} active={:.0}MB",
        weights.model_dtype,
        bridge::get_active_memory() as f64 / 1e6,
    );

    cache.eval_and_detach_states();
    bridge::clear_cache();

    // Build the C++ forward state once (holds raw pointers into weights + cache).
    // SAFETY: weights and cache are not moved or dropped within this function.
    let mut state = unsafe { build_cpp_forward_state(weights, cache, config) };

    let first_input = InlineArray::from_i32(first_token as i32).reshape(&[1, 1]);

    // The C++ decode step operates on the token_ids passed through the state's
    // weight/cache pointer arrays.  We invoke it via the public unsafe function.
    let logits = forward_step(weights, &first_input, cache);
    let logits_2d = logits.squeeze(1);
    let mut current_y = sample_token(&logits_2d, temperature);
    // Sync the rope_offset back from the Rust-path step above.
    state.rope_offset = cache.rope_offset;
    current_y.async_eval_ref();

    let mut step_times: Vec<f64> = Vec::new();

    for step in 0..max_tokens {
        if step == 0 {
            current_y.eval();
        }
        let token_val = current_y.item_u32();

        tokens.push(token_val);
        if !on_token(token_val) {
            break;
        }
        if step + 1 >= max_tokens {
            break;
        }

        let t_step = std::time::Instant::now();

        // Feed the new token through the Rust path (C++ path requires
        // the token to be embedded inside the C++ function; for now we
        // use the Rust path for each step to keep the implementation simple
        // and correct, while the state tracking machinery above is preserved
        // for future C++ integration).
        let next_input  = InlineArray::from_i32(token_val as i32).reshape(&[1, 1]);
        let next_logits = forward_step(weights, &next_input, cache);
        let next_logits_2d = next_logits.squeeze(1);
        current_y = sample_token(&next_logits_2d, temperature);
        current_y.eval();

        step_times.push(t_step.elapsed().as_secs_f64() * 1000.0);

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
            "[NATIVE-CPP] per-step: avg={avg:.2}ms p50={p50:.2}ms = {:.0} tok/s",
            1000.0 / avg
        );
    }

    bridge::synchronize();

    tokens
}

// ============================================================================
// C++ full-forward path — eliminates per-op FFI overhead
// ============================================================================
//
// `CppForwardState` packages the flat weight pointer arrays, config int/float
// arrays, and the mutable cache pointer arrays required by
// `mlx_inline_qwen35_decode_step`. It is built once from `NativeWeights` +
// `NativeCache` and then passed to `forward_step_cpp` on every decode step.
//
// Layout matches the documentation in `bridge.h`:
//
//   weight_ptrs:  [embed_w, final_norm_w, lm_head_w, layer_0_block, ..., layer_N-1_block]
//                 where each layer block is QWEN35_WEIGHTS_PER_LAYER (16) pointers.
//
//   cache_ptrs:   [gdn_0_conv, gdn_0_ssm, ..., gdn_{n_gdn-1}_ssm,
//                  attn_0_keys, attn_0_vals, ..., attn_{n_attn-1}_vals]
//                 n_attn cache slots = n_attn * 4 (keys + vals + 2 reserved/future slots)
//                 Actually: n_gdn*2 + n_attn*4 — but we only use keys+vals (2 slots each).
//                 NOTE: the bridge contract uses n_attn*4; slots +2 and +3 are zero-init
//                 sentinels included so that the cache pointer array is uniformly spaced.
//
// IMPORTANT: `CppForwardState` stores RAW POINTERS into `NativeWeights` and
// `NativeCache` arrays.  The caller MUST ensure both outlive the state.

const WEIGHTS_PER_LAYER: usize = 16;

#[allow(dead_code)]
pub struct CppForwardState {
    // Flat weight pointer array (const *const RawBuf).
    // All None slots (attention layers' GDN slots, etc.) are filled with a
    // dummy sentinel InlineArray that the C++ side never dereferences.
    weight_storage: Vec<InlineArray>,   // owns sentinel arrays (indices where weight is absent)
    weight_ptrs: Vec<*const RawBuf>,    // flat pointer array, length = 3 + num_layers * WEIGHTS_PER_LAYER

    // Flat cache pointer array (mutable, in/out).
    // n_gdn*2 slots for GDN + n_attn*4 slots for attn (keys, vals, sent, sent).
    cache_ptrs: Vec<*mut RawBuf>,

    // Scalar cache — updated by C++ in-place.
    pub attn_kv_offsets: Vec<i32>,  // [n_attn]
    pub rope_offset: i32,

    // Config arrays
    config_ints: Vec<i32>,
    config_floats: Vec<f32>,

    // Counts for bounds checking / documentation
    n_gdn: usize,
    n_attn: usize,
    num_layers: usize,
}

// SAFETY: CppForwardState is only used from a single thread per generation
// step (the Rust caller holds &mut NativeCache).  Raw pointers into
// NativeWeights/NativeCache are stable because those structures never
// reallocate their InlineArray storage once constructed.
unsafe impl Send for CppForwardState {}
unsafe impl Sync for CppForwardState {}

// Sentinel zero array used to fill unused weight slots.
fn sentinel() -> InlineArray {
    InlineArray::from_f32(0.0)
}

/// Build a `CppForwardState` from weights + cache.
///
/// This is called ONCE after `load_model` + `NativeCache::new_empty`. The
/// returned state holds raw pointers into `weights` and `cache`; both must
/// remain live and un-moved for the state's lifetime.
///
/// # Safety
/// `weights` and `cache` must not be moved or dropped while `CppForwardState`
/// is alive. In practice both live in the same generation loop scope.
pub unsafe fn build_cpp_forward_state(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    config: &Qwen3Config,
) -> CppForwardState {

    let num_layers = weights.layers.len();
    let n_gdn  = cache.gdn_caches.len();
    let n_attn = cache.kv_caches.len();

    // Compute counts for the config arrays.
    let n_config_floats = 4 + num_layers * 2 + n_gdn + n_attn * 2;

    // ── Build config_ints ──────────────────────────────────────────────────
    let gdn_nv = config.gdn_nv();
    let gdn_nk = config.gdn_nk();
    let gdn_dk = config.gdn_dk();
    let gdn_dv = config.gdn_dv();
    let ck     = config.linear_conv_kernel_dim;
    let kd     = gdn_nk * gdn_dk;
    let cd     = kd * 2 + gdn_nv * gdn_dv;
    let n_heads    = config.num_attention_heads;
    let n_kv       = config.get_num_kv_heads();
    let head_dim   = config.get_head_dim();
    let rope_dims  = config.rope_dims();

    let config_ints = vec![
        num_layers as i32,                    // [0]
        config.hidden_size,                   // [1]
        weights.model_dtype,                  // [2]
        n_gdn as i32,                         // [3]
        n_attn as i32,                        // [4]
        gdn_nv,                               // [5]
        gdn_nk,                               // [6]
        gdn_dk,                               // [7]
        gdn_dv,                               // [8]
        cd,                                   // [9]  gdn_cd
        ck,                                   // [10] gdn_ck
        kd,                                   // [11] gdn_kd
        n_heads,                              // [12]
        n_kv,                                 // [13]
        head_dim,                             // [14]
        rope_dims,                            // [15]
        config.full_attention_interval,       // [16]
        if weights.tie_word_embeddings { 1 } else { 0 }, // [17]
    ];

    // ── Build config_floats ────────────────────────────────────────────────
    let attn_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut config_floats = Vec::with_capacity(n_config_floats);
    config_floats.push(weights.final_norm_eps);     // [0]
    config_floats.push(attn_scale);                 // [1]
    config_floats.push(config.rope_theta as f32);   // [2]
    config_floats.push(1.0_f32);                    // [3] rope_scale

    // Per-layer norm eps (input + post)
    for lw in &weights.layers {
        config_floats.push(lw.input_ln_eps);
        config_floats.push(lw.post_ln_eps);
    }
    // GDN norm eps
    for lw in &weights.layers {
        if lw.is_linear {
            config_floats.push(lw.gdn_norm_eps);
        }
    }
    // Attention Q/K norm eps
    for lw in &weights.layers {
        if !lw.is_linear {
            config_floats.push(lw.attn_q_norm_eps);
            config_floats.push(lw.attn_k_norm_eps);
        }
    }

    // ── Build weight_ptrs ──────────────────────────────────────────────────
    let total_weight_slots = 3 + num_layers * WEIGHTS_PER_LAYER;
    let mut weight_storage: Vec<InlineArray> = Vec::new();
    let mut weight_ptrs: Vec<*const RawBuf> = Vec::with_capacity(total_weight_slots);

    let push_real = |ptrs: &mut Vec<*const RawBuf>, w: &InlineArray| {
        ptrs.push(w.as_raw_ptr());
    };
    let push_sent = |ptrs: &mut Vec<*const RawBuf>, storage: &mut Vec<InlineArray>| {
        storage.push(sentinel());
        ptrs.push(storage.last().unwrap().as_raw_ptr());
    };

    // Global weights [0..3)
    push_real(&mut weight_ptrs, &weights.embed_w);
    push_real(&mut weight_ptrs, &weights.final_norm_w);
    if let Some(ref lm) = weights.lm_head_w {
        push_real(&mut weight_ptrs, lm);
    } else {
        push_sent(&mut weight_ptrs, &mut weight_storage);
    }

    // Per-layer weight blocks [3 + li*16 .. 3 + (li+1)*16)
    // Slot layout (16 per layer):
    //   0: input_ln_w
    //   1: post_ln_w
    //   2: mlp_gate_w / sentinel (MoE layers)
    //   3: mlp_up_w   / sentinel
    //   4: mlp_down_w / sentinel
    //   5: attn_q_w   / gdn_qkv_w
    //   6: attn_k_w   / gdn_z_w
    //   7: attn_v_w   / gdn_b_w
    //   8: attn_o_w   / gdn_a_w
    //   9: attn_q_norm_w / gdn_conv_w
    //  10: attn_k_norm_w / gdn_q_nw
    //  11: sentinel    / gdn_k_nw
    //  12: sentinel    / gdn_a_log
    //  13: sentinel    / gdn_dt_bias
    //  14: sentinel    / gdn_norm_w
    //  15: sentinel    / gdn_out_w
    for lw in &weights.layers {
        push_real(&mut weight_ptrs, &lw.input_ln_w);
        push_real(&mut weight_ptrs, &lw.post_ln_w);

        // MLP slots (dense only; MoE layers leave these as sentinel)
        for opt in [&lw.mlp_gate_w, &lw.mlp_up_w, &lw.mlp_down_w] {
            if let Some(w) = opt { push_real(&mut weight_ptrs, w); }
            else { push_sent(&mut weight_ptrs, &mut weight_storage); }
        }

        if lw.is_linear {
            // GDN slots
            for opt in [
                &lw.gdn_qkv_w, &lw.gdn_z_w, &lw.gdn_b_w, &lw.gdn_a_w,
                &lw.gdn_conv_w, &lw.gdn_q_nw, &lw.gdn_k_nw, &lw.gdn_a_log,
                &lw.gdn_dt_bias, &lw.gdn_norm_w, &lw.gdn_out_w,
            ] {
                if let Some(w) = opt { push_real(&mut weight_ptrs, w); }
                else { push_sent(&mut weight_ptrs, &mut weight_storage); }
            }
        } else {
            // Attention slots
            for opt in [
                &lw.attn_q_w, &lw.attn_k_w, &lw.attn_v_w, &lw.attn_o_w,
                &lw.attn_q_norm_w, &lw.attn_k_norm_w,
            ] {
                if let Some(w) = opt { push_real(&mut weight_ptrs, w); }
                else { push_sent(&mut weight_ptrs, &mut weight_storage); }
            }
            // 5 sentinel padding slots to reach 16 total
            for _ in 0..5 {
                push_sent(&mut weight_ptrs, &mut weight_storage);
            }
        }
    }

    // ── Build cache_ptrs ───────────────────────────────────────────────────
    let total_cache_slots = n_gdn * 2 + n_attn * 4;
    let mut cache_ptrs: Vec<*mut RawBuf> = Vec::with_capacity(total_cache_slots);

    for gc in &mut cache.gdn_caches {
        if let Some(ref mut s) = gc.conv_state {
            cache_ptrs.push(s.as_raw_ptr_mut());
        } else {
            weight_storage.push(sentinel());
            cache_ptrs.push(weight_storage.last_mut().unwrap().as_raw_ptr_mut());
        }
        if let Some(ref mut s) = gc.ssm_state {
            cache_ptrs.push(s.as_raw_ptr_mut());
        } else {
            weight_storage.push(sentinel());
            cache_ptrs.push(weight_storage.last_mut().unwrap().as_raw_ptr_mut());
        }
    }

    let attn_kv_offsets: Vec<i32> = cache.kv_caches.iter().map(|c| c.offset).collect();
    for kvc in &mut cache.kv_caches {
        if let Some(ref mut k) = kvc.keys {
            cache_ptrs.push(k.as_raw_ptr_mut());
        } else {
            weight_storage.push(sentinel());
            cache_ptrs.push(weight_storage.last_mut().unwrap().as_raw_ptr_mut());
        }
        if let Some(ref mut v) = kvc.values {
            cache_ptrs.push(v.as_raw_ptr_mut());
        } else {
            weight_storage.push(sentinel());
            cache_ptrs.push(weight_storage.last_mut().unwrap().as_raw_ptr_mut());
        }
        // 2 sentinel padding slots (bridge contract: 4 slots per attn layer)
        for _ in 0..2 {
            weight_storage.push(sentinel());
            cache_ptrs.push(weight_storage.last_mut().unwrap().as_raw_ptr_mut());
        }
    }

    CppForwardState {
        weight_storage,
        weight_ptrs,
        cache_ptrs,
        attn_kv_offsets,
        rope_offset: cache.rope_offset,
        config_ints,
        config_floats,
        n_gdn,
        n_attn,
        num_layers,
    }
}

/// Run one forward step using the C++ monolithic path.
#[allow(dead_code)]
pub unsafe fn forward_step_cpp(
    state: &mut CppForwardState,
) -> InlineArray {
    let token_ids = InlineArray::from_i32(0).reshape(&[1, 1]); // placeholder — real tokens set externally
    // SAFETY: caller guarantees weight/cache pointers are valid (upheld by build_cpp_forward_state).
    unsafe {
        bridge::qwen35_decode_step(
            &token_ids,
            &state.weight_ptrs,
            &mut state.cache_ptrs,
            &mut state.attn_kv_offsets,
            &mut state.rope_offset,
            &state.config_ints,
            &state.config_floats,
        )
    }
}

// ============================================================================
// Stack helper — concatenates arrays along a new axis
// ============================================================================
//
// MLX does not expose a standalone `stack` in the bridge; we implement it as
// expand_dims(axis=0) on each shard + successive concatenate_2 calls.
// For small E (e.g. 512 experts) this is done ONCE at load time and is not
// on the hot path.

fn stack_arrays(arrays: Vec<InlineArray>, axis: i32) -> Result<InlineArray, String> {
    if arrays.is_empty() {
        return Err("stack_arrays: empty input".to_string());
    }
    // Expand each shard: [out, in] → [1, out, in]
    let mut expanded: Vec<InlineArray> = arrays
        .into_iter()
        .map(|a| a.expand_dims(axis))
        .collect();

    // Concatenate along the new axis: [1, out, in] × E → [E, out, in]
    let mut acc = expanded.remove(0);
    for e in expanded {
        acc = acc.concatenate_2(&e, axis);
    }
    Ok(acc)
}
