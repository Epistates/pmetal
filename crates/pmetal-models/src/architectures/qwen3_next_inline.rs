//! InlineArray-based Qwen3.5 decode forward — zero mlx-c handle overhead.
//!
//! Every op on the hot path uses `InlineArray` (stack-allocated mlx::core::array,
//! direct C++ bridge). This eliminates the 6.8x build overhead from mlx-c handle
//! management (mlx_array_new/free/set per op), matching Python's nanobind path.
//!
//! Weights are converted from `pmetal_bridge::compat::Array` → `InlineArray` once at first decode
//! call (cold path). All subsequent decode calls use InlineArray exclusively.

use pmetal_bridge::InlineArray;
use pmetal_bridge::compat::{Array, Dtype, Exception};

use super::qwen3_next::Qwen3NextForCausalLM;
use pmetal_mlx::kv_cache::{KVCache, MambaCache, MambaCacheEntry};

// Interop helpers — COLD PATH ONLY (weight init, cache bootstrap).
// These go through the raw void* pointer (shared_ptr copy, ~10ns).
// The hot-path decode uses ONLY InlineArray — zero mlx-rs.

/// Convert a bridge Array (= InlineArray) to InlineArray — identity since they're the same type.
pub fn ia_from_array(arr: &Array) -> InlineArray {
    arr.clone()
}

/// Convert an InlineArray to a bridge Array — identity since they're the same type.
pub fn ia_to_array(ia: &InlineArray) -> Array {
    ia.clone()
}

// ============================================================================
// Cached InlineArray weights for one decoder layer
// ============================================================================

pub struct InlineLayerWeights {
    is_linear: bool,

    // Shared: layer norms + MLP
    input_ln_w: InlineArray,
    input_ln_eps: f32,
    post_ln_w: InlineArray,
    post_ln_eps: f32,
    mlp_gate_w: InlineArray, // pre-transposed
    mlp_up_w: InlineArray,   // pre-transposed
    mlp_down_w: InlineArray, // pre-transposed

    // Attention-specific (only if !is_linear)
    attn_q_w: Option<InlineArray>, // pre-transposed
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

    // GDN-specific (only if is_linear)
    gdn_qkv_w: Option<InlineArray>, // in_proj_qkv, pre-transposed [hidden, conv_dim]
    gdn_z_w: Option<InlineArray>,   // in_proj_z, pre-transposed [hidden, value_dim]
    gdn_b_w: Option<InlineArray>,   // in_proj_b, pre-transposed [hidden, num_v_heads]
    gdn_a_w: Option<InlineArray>,   // in_proj_a, pre-transposed [hidden, num_v_heads]
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
    gdn_kd: i32,
    gdn_cd: i32,
    gdn_ck: i32,
}

// ============================================================================
// Cached model weights
// ============================================================================

pub struct InlineModelWeights {
    pub embed_w: InlineArray,
    pub final_norm_w: InlineArray,
    pub final_norm_eps: f32,
    pub lm_head_w: Option<InlineArray>, // None if tie_word_embeddings
    pub tie_word_embeddings: bool,
    pub layers: Vec<InlineLayerWeights>,
    /// Model activation dtype (e.g., 11=bfloat16) — used for KV cache and conv state
    /// so they match the model's compute precision instead of wasting 2x memory on float32.
    pub model_dtype: i32,
}

impl std::fmt::Debug for InlineModelWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InlineModelWeights")
            .field("layers", &self.layers.len())
            .field("tie_word_embeddings", &self.tie_word_embeddings)
            .finish()
    }
}

// ============================================================================
// InlineArray-native cache — zero mlx-rs on hot path
// ============================================================================

/// GDN layer cache state (conv + SSM) stored as InlineArray.
pub struct InlineGdnCache {
    pub conv_state: Option<InlineArray>,
    pub ssm_state: Option<InlineArray>,
}

/// Per-layer KV cache stored as InlineArray.
/// Uses pre-allocated buffers with slice_set for O(1) per-step updates
/// (matching Python's in-place slice assignment pattern).
pub struct InlineKvLayerCache {
    pub keys: Option<InlineArray>, // [B, H, MAX_T, D] pre-allocated buffer
    pub values: Option<InlineArray>, // [B, H, MAX_T, D] pre-allocated buffer
    pub offset: i32,               // number of valid tokens in cache
}

/// Full cache for the InlineArray decode path.
pub struct InlineCache {
    pub gdn_caches: Vec<InlineGdnCache>, // indexed by layer position in gdn_layers
    pub kv_caches: Vec<InlineKvLayerCache>, // indexed by layer position in attn_layers
    pub gdn_layer_indices: Vec<usize>,   // which layers are GDN
    pub attn_layer_indices: Vec<usize>,  // which layers are attention
    pub rope_offset: i32,                // current sequence position
}

impl InlineCache {
    /// Bootstrap from existing mlx-rs caches (called once after prefill).
    pub fn from_caches(
        kv_cache: &KVCache,
        mamba_cache: &MambaCache,
        layers: &[InlineLayerWeights],
    ) -> Self {
        let mut gdn_caches = Vec::new();
        let mut kv_caches = Vec::new();
        let mut gdn_layer_indices = Vec::new();
        let mut attn_layer_indices = Vec::new();

        for (i, lw) in layers.iter().enumerate() {
            if lw.is_linear {
                gdn_layer_indices.push(i);
                let entry = mamba_cache.get(i);
                gdn_caches.push(InlineGdnCache {
                    conv_state: entry
                        .and_then(|e| e.conv_state.as_ref())
                        .map(|a| ia_from_array(a)),
                    ssm_state: entry
                        .and_then(|e| e.ssm_state.as_ref())
                        .map(|a| ia_from_array(a)),
                });
            } else {
                attn_layer_indices.push(i);
                let (keys, values) = kv_cache
                    .fetch_for_compiled_decode(i)
                    .map(|(k, v)| (Some(ia_from_array(&k)), Some(ia_from_array(&v))))
                    .unwrap_or((None, None));
                let offset = keys.as_ref().map(|k| k.dim(2)).unwrap_or(0);
                kv_caches.push(InlineKvLayerCache {
                    keys,
                    values,
                    offset,
                });
            }
        }

        InlineCache {
            gdn_caches,
            kv_caches,
            gdn_layer_indices,
            attn_layer_indices,
            rope_offset: kv_cache.rope_offset(),
        }
    }

    /// Write back to mlx-rs caches (for compatibility with generation loop).
    pub fn write_back(&self, kv_cache: &mut KVCache, mamba_cache: &mut MambaCache) {
        for (slot, &layer_idx) in self.gdn_layer_indices.iter().enumerate() {
            if let Some(entry) = mamba_cache.get_mut(layer_idx) {
                entry.conv_state = self.gdn_caches[slot]
                    .conv_state
                    .as_ref()
                    .map(|a| ia_to_array(a));
                entry.ssm_state = self.gdn_caches[slot]
                    .ssm_state
                    .as_ref()
                    .map(|a| ia_to_array(a));
            }
        }
        for (slot, &layer_idx) in self.attn_layer_indices.iter().enumerate() {
            if let (Some(k), Some(v)) = (&self.kv_caches[slot].keys, &self.kv_caches[slot].values) {
                let _ = kv_cache.update_from_compiled_decode(
                    layer_idx,
                    &ia_to_array(k),
                    &ia_to_array(v),
                );
            }
        }
    }

    /// Create a fresh, empty cache from a set of layer weights.
    ///
    /// All GDN conv/SSM states start as `None` (initialised on first decode step).
    /// All KV cache buffers start as `None` with `offset = 0`.
    /// `rope_offset` is set to 0.
    ///
    /// Use this when loading weights via `InlineModelWeights::from_safetensors`
    /// without running a prefill through the mlx-rs path first.
    pub fn new_empty(layers: &[InlineLayerWeights]) -> Self {
        let mut gdn_caches = Vec::new();
        let mut kv_caches = Vec::new();
        let mut gdn_layer_indices = Vec::new();
        let mut attn_layer_indices = Vec::new();

        for (i, lw) in layers.iter().enumerate() {
            if lw.is_linear {
                gdn_layer_indices.push(i);
                gdn_caches.push(InlineGdnCache {
                    conv_state: None,
                    ssm_state: None,
                });
            } else {
                attn_layer_indices.push(i);
                kv_caches.push(InlineKvLayerCache {
                    keys: None,
                    values: None,
                    offset: 0,
                });
            }
        }

        InlineCache {
            gdn_caches,
            kv_caches,
            gdn_layer_indices,
            attn_layer_indices,
            rope_offset: 0,
        }
    }
}

impl std::fmt::Debug for InlineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InlineCache")
            .field("gdn_layers", &self.gdn_caches.len())
            .field("attn_layers", &self.kv_caches.len())
            .field("rope_offset", &self.rope_offset)
            .finish()
    }
}

impl InlineModelWeights {
    /// Convert all model weights from Array to InlineArray. Called once.
    pub fn from_model(model: &mut Qwen3NextForCausalLM) -> Result<Self, Exception> {
        let config = &model.config;
        let mut embed_w = ia_from_array(model.model.embed_tokens.weight.as_ref());
        let mut final_norm_w = ia_from_array(model.model.norm.weight.as_ref());
        let final_norm_eps = model.model.norm.eps;
        let mut lm_head_w = model
            .lm_head
            .as_ref()
            .map(|l| ia_from_array(l.weight.as_ref()));

        let mut layers = Vec::with_capacity(model.model.layers.len());
        for (li, layer) in model.model.layers.iter_mut().enumerate() {
            let mut lw = InlineLayerWeights {
                is_linear: layer.is_linear,
                input_ln_w: ia_from_array(layer.input_layernorm.weight.as_ref()),
                input_ln_eps: layer.input_layernorm.eps,
                post_ln_w: ia_from_array(layer.post_attention_layernorm.weight.as_ref()),
                post_ln_eps: layer.post_attention_layernorm.eps,
                // MLP weights (pre-transposed for matmul)
                mlp_gate_w: InlineArray::from_f32(0.0),
                mlp_up_w: InlineArray::from_f32(0.0),
                mlp_down_w: InlineArray::from_f32(0.0),
                // Attention
                attn_q_w: None,
                attn_k_w: None,
                attn_v_w: None,
                attn_o_w: None,
                attn_q_norm_w: None,
                attn_q_norm_eps: 1e-6,
                attn_k_norm_w: None,
                attn_k_norm_eps: 1e-6,
                attn_n_heads: 0,
                attn_n_kv_heads: 0,
                attn_head_dim: 0,
                attn_scale: 0.0,
                attn_rope_dims: 0,
                attn_rope_base: 0.0,
                attn_rope_scale: 0.0,
                // GDN
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
                gdn_norm_eps: 1e-6,
                gdn_out_w: None,
                gdn_nv: 0,
                gdn_nk: 0,
                gdn_dk: 0,
                gdn_dv: 0,
                gdn_kd: 0,
                gdn_cd: 0,
                gdn_ck: 0,
            };

            // MLP
            match &layer.mlp {
                super::qwen3_next::Qwen3NextFeedForward::Dense(mlp) => {
                    lw.mlp_gate_w = ia_from_array(mlp.gate_proj.weight.as_ref()).t();
                    lw.mlp_up_w = ia_from_array(mlp.up_proj.weight.as_ref()).t();
                    lw.mlp_down_w = ia_from_array(mlp.down_proj.weight.as_ref()).t();
                }
                super::qwen3_next::Qwen3NextFeedForward::MoE(_) => {
                    // MoE models use the standard Array path for now
                    return Err(Exception::custom(
                        "InlineArray decode not supported for MoE models yet",
                    ));
                }
            }

            if layer.is_linear {
                let gdn = layer.linear_attn.as_mut().unwrap();
                // Separate projection weights (pre-transposed), matching Python's 4 Linear layers
                lw.gdn_qkv_w = Some(ia_from_array(gdn.in_proj_qkv.weight.as_ref()).t());
                lw.gdn_z_w = Some(ia_from_array(gdn.in_proj_z.weight.as_ref()).t());
                lw.gdn_b_w = Some(ia_from_array(gdn.in_proj_b.weight.as_ref()).t());
                lw.gdn_a_w = Some(ia_from_array(gdn.in_proj_a.weight.as_ref()).t());
                lw.gdn_conv_w = Some(ia_from_array(gdn.conv1d.weight.as_ref()));
                lw.gdn_q_nw = Some(ia_from_array(&gdn.q_norm_weight));
                lw.gdn_k_nw = Some(ia_from_array(&gdn.k_norm_weight));
                lw.gdn_a_log = Some(ia_from_array(gdn.a_log.as_ref()));
                lw.gdn_dt_bias = Some(ia_from_array(gdn.dt_bias.as_ref()));
                lw.gdn_norm_w = Some(ia_from_array(gdn.norm.weight.as_ref()));
                lw.gdn_norm_eps = gdn.norm.eps;
                lw.gdn_out_w = Some(ia_from_array(gdn.out_proj.weight.as_ref()).t());
                lw.gdn_nv = gdn.num_v_heads;
                lw.gdn_nk = gdn.num_k_heads;
                lw.gdn_dk = gdn.head_k_dim;
                lw.gdn_dv = gdn.head_v_dim;
                lw.gdn_kd = gdn.key_dim;
                lw.gdn_cd = gdn.conv_dim;
                lw.gdn_ck = gdn.conv_kernel_size;
                if li == 0 {
                    eprintln!(
                        "[INLINE-GEN] GDN config: nk={} nv={} dk={} dv={} kd={} cd={} ck={}",
                        gdn.num_k_heads,
                        gdn.num_v_heads,
                        gdn.head_k_dim,
                        gdn.head_v_dim,
                        gdn.key_dim,
                        gdn.conv_dim,
                        gdn.conv_kernel_size
                    );
                }
            } else {
                let attn = layer.self_attn.as_ref().unwrap();
                lw.attn_q_w = Some(ia_from_array(attn.q_proj.weight.as_ref()).t());
                lw.attn_k_w = Some(ia_from_array(attn.k_proj.weight.as_ref()).t());
                lw.attn_v_w = Some(ia_from_array(attn.v_proj.weight.as_ref()).t());
                lw.attn_o_w = Some(ia_from_array(attn.o_proj.weight.as_ref()).t());
                lw.attn_q_norm_w = Some(ia_from_array(attn.q_norm.weight.as_ref()));
                lw.attn_q_norm_eps = attn.q_norm.eps;
                lw.attn_k_norm_w = Some(ia_from_array(attn.k_norm.weight.as_ref()));
                lw.attn_k_norm_eps = attn.k_norm.eps;
                lw.attn_n_heads = attn.n_heads;
                lw.attn_n_kv_heads = attn.n_kv_heads;
                lw.attn_head_dim = attn.head_dim;
                lw.attn_scale = attn.scale;
                lw.attn_rope_dims = attn.rope_dims;
                lw.attn_rope_base = attn.effective_base;
                lw.attn_rope_scale = attn.rope_scale;
            }

            layers.push(lw);
        }

        let model_dtype = embed_w.dtype_raw();

        // CRITICAL: Detach ALL weight arrays from their graph chains.
        // Weight arrays come from ia_from_array(weight).t() which creates:
        //   Transpose(Copy(Concatenate(individual_weights...)))
        // Even though all nodes are evaluated, the graph chain keeps
        // hundreds of ArrayDesc objects alive with references.
        // Detaching severs these chains, potentially reducing eval overhead.
        // Eval ALL weights (forces lazy transposes to materialize) then detach
        // to sever graph chains. Without this, each weight carries:
        //   Transpose → Copy → Concatenate → individual weights
        // These chains add ~200+ graph nodes that the eval engine traverses.
        // Force-copy ALL weights into fresh Metal buffers (data.use_count=1).
        // Model weights share data with mlx-rs (use_count=2), which prevents
        // optimal Metal buffer scheduling during eval. Copying breaks the sharing.
        let zero = InlineArray::from_f32(0.0).as_dtype(model_dtype);
        let copy_fresh = |w: &InlineArray| -> InlineArray {
            let mut fresh = w.add(&zero);
            fresh.eval();
            fresh.detach();
            fresh
        };
        embed_w = copy_fresh(&embed_w);
        final_norm_w = copy_fresh(&final_norm_w);
        lm_head_w = lm_head_w.map(|w| copy_fresh(&w));
        for lw in &mut layers {
            lw.input_ln_w = copy_fresh(&lw.input_ln_w);
            lw.post_ln_w = copy_fresh(&lw.post_ln_w);
            lw.mlp_gate_w = copy_fresh(&lw.mlp_gate_w);
            lw.mlp_up_w = copy_fresh(&lw.mlp_up_w);
            lw.mlp_down_w = copy_fresh(&lw.mlp_down_w);
            if let Some(ref w) = lw.attn_q_w {
                lw.attn_q_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.attn_k_w {
                lw.attn_k_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.attn_v_w {
                lw.attn_v_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.attn_o_w {
                lw.attn_o_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.attn_q_norm_w {
                lw.attn_q_norm_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.attn_k_norm_w {
                lw.attn_k_norm_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_qkv_w {
                lw.gdn_qkv_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_z_w {
                lw.gdn_z_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_b_w {
                lw.gdn_b_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_a_w {
                lw.gdn_a_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_conv_w {
                lw.gdn_conv_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_q_nw {
                lw.gdn_q_nw = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_k_nw {
                lw.gdn_k_nw = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_a_log {
                lw.gdn_a_log = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_dt_bias {
                lw.gdn_dt_bias = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_norm_w {
                lw.gdn_norm_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_out_w {
                lw.gdn_out_w = Some(copy_fresh(w));
            }
        }
        eprintln!("[INLINE-GEN] Force-copied all weights into fresh Metal buffers");

        Ok(Self {
            embed_w,
            final_norm_w,
            final_norm_eps,
            lm_head_w,
            tie_word_embeddings: config.tie_word_embeddings,
            layers,
            model_dtype,
        })
    }

    /// Load weights directly from safetensors files through pmetal-bridge's MLX instance.
    ///
    /// This bypasses mlx-rs entirely, avoiding the dual-MLX-instance 6x performance
    /// penalty. Shard discovery, key renaming, conv1d transpose, and norm +1.0 offset
    /// are all applied here before building the weight struct.
    ///
    /// Returns an error for MoE models — those use a separate code path.
    pub fn from_safetensors(
        model_dir: &std::path::Path,
        config: &super::qwen3_next::Qwen3NextConfig,
    ) -> Result<Self, String> {
        if config.num_experts > 0 {
            return Err(
                "InlineModelWeights::from_safetensors: MoE models are not supported; \
                 use the mlx-rs loader path instead"
                    .to_string(),
            );
        }

        // ── Step 1: Shard discovery ─────────────────────────────────────────
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
            // Deduplicate shard filenames while preserving deterministic order.
            let mut seen = std::collections::HashSet::new();
            let mut paths = Vec::new();
            for shard_file in weight_map.values() {
                let name = shard_file
                    .as_str()
                    .ok_or_else(|| "shard filename is not a string".to_string())?;
                if seen.insert(name.to_string()) {
                    // Path traversal guard (mirrors loader.rs validate_shard_path).
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

        // ── Step 2: Load all weights ────────────────────────────────────────
        // Use load_safetensors_shard (single parse per file) — avoids per-key
        // file-open overhead vs the key-by-key InlineArray::load_safetensors path.
        let mut raw: std::collections::HashMap<String, InlineArray> =
            std::collections::HashMap::new();
        for shard_path in &shard_paths {
            let path_str = shard_path
                .to_str()
                .ok_or_else(|| format!("non-UTF-8 shard path: {:?}", shard_path))?;
            let entries = pmetal_bridge::inline_array::load_safetensors_shard(path_str)
                .ok_or_else(|| format!("failed to load shard: {path_str}"))?;
            for (key, arr) in entries {
                raw.insert(key, arr);
            }
        }

        if raw.is_empty() {
            return Err(format!("no weights loaded from {}", model_dir.display()));
        }

        // ── Step 3: Sanitization ────────────────────────────────────────────
        // Detect shift condition before any renaming (mirrors sanitize_weights).
        let has_mtp = raw.keys().any(|k| k.contains("mtp."));
        let has_unsanitized_conv = raw
            .iter()
            .any(|(k, v)| k.contains("conv1d.weight") && v.ndim() == 3 && v.dim(2) != 1);
        let should_shift_norms = has_mtp || has_unsanitized_conv;

        // 3a. Key renaming: strip VLM prefix and rename A_log → a_log.
        let original_keys: Vec<String> = raw.keys().cloned().collect();
        for old_key in original_keys {
            let mut new_key = old_key.clone();
            if new_key.starts_with("model.language_model.") {
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

        // Norm suffixes that get the (1+w) shift — excludes .linear_attn.norm.weight.
        let norm_suffixes = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ];
        let detected_model_dtype = raw
            .get("model.embed_tokens.weight")
            .map(|w| w.dtype_raw())
            .unwrap_or(11); // 11 = bfloat16 fallback
        let one = InlineArray::from_f32(1.0).as_dtype(detected_model_dtype);
        let f32_gdn_suffixes = [
            "linear_attn.a_log",
            "linear_attn.norm.weight",
            "linear_attn.q_norm.weight",
            "linear_attn.k_norm.weight",
        ];

        // 3d. Conv1d transpose + norm shift + f32→model_dtype casts.
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

        // ── Step 4: Build InlineModelWeights ───────────────────────────────
        let get = |key: &str| -> Result<InlineArray, String> {
            raw.get(key).cloned().ok_or_else(|| {
                // Debug: find closest matching keys
                let parts: Vec<&str> = key.rsplitn(2, '.').collect();
                let suffix = parts[0];
                let close: Vec<&String> = raw
                    .keys()
                    .filter(|k| k.ends_with(suffix) || k.contains("q_norm"))
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
            Some(get("lm_head.weight")?)
        };

        // Derive model dtype from embedding weights.
        let model_dtype = embed_w.dtype_raw();

        // GDN derived dimensions (same across all GDN layers).
        let nv = config.linear_num_value_heads;
        let nk = config.linear_num_key_heads;
        let dk = config.linear_key_head_dim;
        let dv = config.linear_value_head_dim;
        let ck = config.linear_conv_kernel_dim;
        let kd = nk * dk; // total key dimension
        let cd = kd * 2 + nv * dv; // conv projection dim

        // Attention derived dimensions.
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.get_num_kv_heads();
        let head_dim = config.get_head_dim();
        let attn_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let rope_dims = config.rope_dims();
        let rope_base = config.rope_theta;
        let rope_scale = 1.0_f32;

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for li in 0..config.num_hidden_layers as usize {
            let p = format!("model.layers.{li}");
            let is_linear = config.is_linear_layer(li);

            let input_ln_w = get(&format!("{p}.input_layernorm.weight"))?;
            let post_ln_w = get(&format!("{p}.post_attention_layernorm.weight"))?;
            // MLP weights — pre-transposed for matmul (stored as [out, in], transpose → [in, out]).
            let mlp_gate_w = get(&format!("{p}.mlp.gate_proj.weight"))?.t();
            let mlp_up_w = get(&format!("{p}.mlp.up_proj.weight"))?.t();
            let mlp_down_w = get(&format!("{p}.mlp.down_proj.weight"))?.t();

            let mut lw = InlineLayerWeights {
                is_linear,
                input_ln_w,
                input_ln_eps: config.rms_norm_eps,
                post_ln_w,
                post_ln_eps: config.rms_norm_eps,
                mlp_gate_w,
                mlp_up_w,
                mlp_down_w,
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
                lw.gdn_z_w = Some(get(&format!("{la}.in_proj_z.weight"))?.t());
                lw.gdn_b_w = Some(get(&format!("{la}.in_proj_b.weight"))?.t());
                lw.gdn_a_w = Some(get(&format!("{la}.in_proj_a.weight"))?.t());
                lw.gdn_conv_w = Some(get(&format!("{la}.conv1d.weight"))?);
                // q_norm_weight / k_norm_weight are SYNTHETIC — they are not stored in
                // safetensors. They are computed in the model constructor as:
                //   q_norm_weight = ones * inv_scale^2   (inv_scale = 1/sqrt(dk))
                //   k_norm_weight = ones * inv_scale
                // Falling back to plain ones() would apply no Q/K scaling at all,
                // corrupting every GDN layer's output and producing garbage tokens.
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
                lw.gdn_a_log = Some(get(&format!("{la}.a_log"))?);
                lw.gdn_dt_bias = Some(get(&format!("{la}.dt_bias"))?);
                lw.gdn_norm_w = Some(get(&format!("{la}.norm.weight"))?);
                lw.gdn_out_w = Some(get(&format!("{la}.out_proj.weight"))?.t());
                lw.gdn_nv = nv;
                lw.gdn_nk = nk;
                lw.gdn_dk = dk;
                lw.gdn_dv = dv;
                lw.gdn_kd = kd;
                lw.gdn_cd = cd;
                lw.gdn_ck = ck;
                if li == 0 {
                    eprintln!(
                        "[INLINE-GEN] GDN config (from_safetensors): nk={nk} nv={nv} \
                         dk={dk} dv={dv} kd={kd} cd={cd} ck={ck}"
                    );
                }
            } else {
                let sa = format!("{p}.self_attn");
                lw.attn_q_w = Some(get(&format!("{sa}.q_proj.weight"))?.t());
                lw.attn_k_w = Some(get(&format!("{sa}.k_proj.weight"))?.t());
                lw.attn_v_w = Some(get(&format!("{sa}.v_proj.weight"))?.t());
                lw.attn_o_w = Some(get(&format!("{sa}.o_proj.weight"))?.t());
                lw.attn_q_norm_w = Some(get(&format!("{sa}.q_norm.weight"))?);
                lw.attn_k_norm_w = Some(get(&format!("{sa}.k_norm.weight"))?);
                lw.attn_n_heads = n_heads;
                lw.attn_n_kv_heads = n_kv_heads;
                lw.attn_head_dim = head_dim;
                lw.attn_scale = attn_scale;
                lw.attn_rope_dims = rope_dims;
                lw.attn_rope_base = rope_base;
                lw.attn_rope_scale = rope_scale;
            }

            layers.push(lw);
        }

        // ── Step 5: Eval + detach all weights ──────────────────────────────
        // Force-copy every weight into a fresh Metal buffer (use_count=1) so
        // MLX can schedule ops without buffer aliasing overhead.  This is the
        // same `copy_fresh` pattern used by from_model() and is the key step
        // that achieves 369 tok/s.
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
            lw.post_ln_w = copy_fresh(&lw.post_ln_w);
            lw.mlp_gate_w = copy_fresh(&lw.mlp_gate_w);
            lw.mlp_up_w = copy_fresh(&lw.mlp_up_w);
            lw.mlp_down_w = copy_fresh(&lw.mlp_down_w);
            if let Some(ref w) = lw.attn_q_w {
                lw.attn_q_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.attn_k_w {
                lw.attn_k_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.attn_v_w {
                lw.attn_v_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.attn_o_w {
                lw.attn_o_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.attn_q_norm_w {
                lw.attn_q_norm_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.attn_k_norm_w {
                lw.attn_k_norm_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_qkv_w {
                lw.gdn_qkv_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_z_w {
                lw.gdn_z_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_b_w {
                lw.gdn_b_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_a_w {
                lw.gdn_a_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_conv_w {
                lw.gdn_conv_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_q_nw {
                lw.gdn_q_nw = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_k_nw {
                lw.gdn_k_nw = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_a_log {
                lw.gdn_a_log = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_dt_bias {
                lw.gdn_dt_bias = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_norm_w {
                lw.gdn_norm_w = Some(copy_fresh(w));
            }
            if let Some(ref w) = lw.gdn_out_w {
                lw.gdn_out_w = Some(copy_fresh(w));
            }
        }
        eprintln!(
            "[INLINE-GEN] from_safetensors: force-copied all weights into fresh Metal buffers"
        );

        Ok(Self {
            embed_w,
            final_norm_w,
            final_norm_eps,
            lm_head_w,
            tie_word_embeddings: config.tie_word_embeddings,
            layers,
            model_dtype,
        })
    }
}

// ============================================================================
// Diagnostic: weight comparison between from_safetensors and from_model
// ============================================================================

/// Compare each weight array between two `InlineModelWeights` sets.
///
/// For every (name, native, model) triple, checks shape, dtype, and the first
/// scalar value.  Prints a `[CMP]` line per weight — OK when everything matches,
/// MISMATCH when any field differs.  Call this during debugging to find which
/// weight is responsible for garbage output.
///
/// Typical usage:
/// ```ignore
/// let native    = InlineModelWeights::from_safetensors(&model_dir, &config);
/// let from_mdl  = InlineModelWeights::from_model(&mut qwen3_next_model);
/// InlineModelWeights::compare_weights(&native, &from_mdl);
/// ```
pub fn compare_weights(native: &InlineModelWeights, from_model: &InlineModelWeights) {
    // Extract the first f32 element from an InlineArray.
    // Eval is required before reading scalar data.
    fn first_val(arr: &InlineArray) -> f32 {
        // Flatten to 1-D and take index [0].
        let flat = arr.reshape(&[-1]);
        let idx = InlineArray::from_i32(0).reshape(&[1]);
        let elem = flat.take_axis(&idx, 0);
        elem.eval();
        elem.item_f32()
    }

    let check = |name: &str, a: &InlineArray, b: &InlineArray| {
        let mut ok = true;
        let mut msg = String::new();

        // Shape check
        let a_ndim = a.ndim();
        let b_ndim = b.ndim();
        if a_ndim != b_ndim {
            msg.push_str(&format!(" ndim native={a_ndim} model={b_ndim} MISMATCH"));
            ok = false;
        } else {
            let mut shape_ok = true;
            for d in 0..a_ndim as i32 {
                if a.dim(d) != b.dim(d) {
                    shape_ok = false;
                    msg.push_str(&format!(
                        " dim[{d}] native={} model={} MISMATCH",
                        a.dim(d),
                        b.dim(d)
                    ));
                }
            }
            if shape_ok {
                msg.push_str(" shape OK");
            } else {
                ok = false;
            }
        }

        // Dtype check
        let a_dt = a.dtype_raw();
        let b_dt = b.dtype_raw();
        if a_dt == b_dt {
            msg.push_str(" dtype OK");
        } else {
            msg.push_str(&format!(" dtype native={a_dt} model={b_dt} MISMATCH"));
            ok = false;
        }

        // First element check
        let a_v = first_val(a);
        let b_v = first_val(b);
        let tol = 1e-4_f32;
        if (a_v - b_v).abs() <= tol {
            msg.push_str(&format!(" val[0]={a_v:.6} OK"));
        } else {
            msg.push_str(&format!(" val[0] native={a_v:.6} model={b_v:.6} MISMATCH"));
            ok = false;
        }

        let status = if ok { "OK" } else { "MISMATCH!" };
        eprintln!("[CMP] {name}:{msg} [{status}]");
    };

    // Top-level weights
    check("embed_w", &native.embed_w, &from_model.embed_w);
    check(
        "final_norm_w",
        &native.final_norm_w,
        &from_model.final_norm_w,
    );
    match (&native.lm_head_w, &from_model.lm_head_w) {
        (Some(a), Some(b)) => check("lm_head_w", a, b),
        (None, None) => eprintln!("[CMP] lm_head_w: both tied OK"),
        _ => eprintln!("[CMP] lm_head_w: tie mismatch MISMATCH!"),
    }

    // Per-layer weights
    let n_layers = native.layers.len().min(from_model.layers.len());
    if native.layers.len() != from_model.layers.len() {
        eprintln!(
            "[CMP] layer count: native={} model={} MISMATCH!",
            native.layers.len(),
            from_model.layers.len()
        );
    }

    for li in 0..n_layers {
        let a = &native.layers[li];
        let b = &from_model.layers[li];

        let prefix = format!("layers[{li}]");

        if a.is_linear != b.is_linear {
            eprintln!(
                "[CMP] {prefix}.is_linear: native={} model={} MISMATCH!",
                a.is_linear, b.is_linear
            );
        }

        check(
            &format!("{prefix}.input_ln_w"),
            &a.input_ln_w,
            &b.input_ln_w,
        );
        check(&format!("{prefix}.post_ln_w"), &a.post_ln_w, &b.post_ln_w);
        check(
            &format!("{prefix}.mlp_gate_w"),
            &a.mlp_gate_w,
            &b.mlp_gate_w,
        );
        check(&format!("{prefix}.mlp_up_w"), &a.mlp_up_w, &b.mlp_up_w);
        check(
            &format!("{prefix}.mlp_down_w"),
            &a.mlp_down_w,
            &b.mlp_down_w,
        );

        if a.is_linear {
            macro_rules! cmp_opt {
                ($field:ident, $name:literal) => {
                    match (&a.$field, &b.$field) {
                        (Some(x), Some(y)) => check(&format!("{prefix}.{}", $name), x, y),
                        (None, None) => {}
                        _ => eprintln!("[CMP] {prefix}.{}: presence mismatch MISMATCH!", $name),
                    }
                };
            }
            cmp_opt!(gdn_qkv_w, "gdn_qkv_w");
            cmp_opt!(gdn_z_w, "gdn_z_w");
            cmp_opt!(gdn_b_w, "gdn_b_w");
            cmp_opt!(gdn_a_w, "gdn_a_w");
            cmp_opt!(gdn_conv_w, "gdn_conv_w");
            cmp_opt!(gdn_q_nw, "gdn_q_nw");
            cmp_opt!(gdn_k_nw, "gdn_k_nw");
            cmp_opt!(gdn_a_log, "gdn_a_log");
            cmp_opt!(gdn_dt_bias, "gdn_dt_bias");
            cmp_opt!(gdn_norm_w, "gdn_norm_w");
            cmp_opt!(gdn_out_w, "gdn_out_w");
        } else {
            macro_rules! cmp_opt {
                ($field:ident, $name:literal) => {
                    match (&a.$field, &b.$field) {
                        (Some(x), Some(y)) => check(&format!("{prefix}.{}", $name), x, y),
                        (None, None) => {}
                        _ => eprintln!("[CMP] {prefix}.{}: presence mismatch MISMATCH!", $name),
                    }
                };
            }
            cmp_opt!(attn_q_w, "attn_q_w");
            cmp_opt!(attn_k_w, "attn_k_w");
            cmp_opt!(attn_v_w, "attn_v_w");
            cmp_opt!(attn_o_w, "attn_o_w");
            cmp_opt!(attn_q_norm_w, "attn_q_norm_w");
            cmp_opt!(attn_k_norm_w, "attn_k_norm_w");
        }
    }

    eprintln!("[CMP] comparison complete ({n_layers} layers)");
}

// ============================================================================
// InlineArray decode forward
// ============================================================================

/// Run one decode step (T=1) using InlineArray exclusively.
///
/// ZERO mlx-rs on the hot path. Returns logits as InlineArray.
/// The caller converts to Array once for sampling.
pub fn inline_decode_step_pure(
    weights: &InlineModelWeights,
    token_id: &InlineArray, // [1, 1] int32
    cache: &mut InlineCache,
) -> InlineArray {
    let b = token_id.dim(0);
    let s = token_id.dim(1); // T=1 for decode, T=seq_len for prefill
    let dtype = weights.model_dtype;

    // Embedding: take(embed_w, token_id, axis=0)
    let mut hidden = weights.embed_w.take_axis(token_id, 0);

    let mut gdn_slot = 0usize;
    let mut attn_slot = 0usize;

    for (_layer_idx, lw) in weights.layers.iter().enumerate() {
        // Input LayerNorm
        let normed = hidden.rms_norm(Some(&lw.input_ln_w), lw.input_ln_eps);

        // Attention or GDN
        let r = if lw.is_linear {
            let result =
                inline_gdn_forward_pure(lw, &normed, b, s, &mut cache.gdn_caches[gdn_slot], dtype);
            gdn_slot += 1;
            result
        } else {
            let result = inline_attn_forward_pure(
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

        // Post-attention LayerNorm + MLP
        let mlp_in = h.rms_norm(Some(&lw.post_ln_w), lw.post_ln_eps);

        // SwiGLU MLP: down(fused_swiglu(gate(x), up(x)))
        let gate = mlp_in.matmul(&lw.mlp_gate_w);
        let up = mlp_in.matmul(&lw.mlp_up_w);
        let activated = InlineArray::fused_swiglu(&gate, &up);
        let mlp_out = activated.matmul(&lw.mlp_down_w);

        // Residual
        hidden = h.add(&mlp_out);
    }

    // Advance position for next step (s=1 for decode, s=seq_len for prefill)
    cache.rope_offset += s;

    // Final norm + LM head
    let hidden = hidden.rms_norm(Some(&weights.final_norm_w), weights.final_norm_eps);
    if weights.tie_word_embeddings {
        hidden.matmul(&weights.embed_w.t())
    } else {
        hidden.matmul(weights.lm_head_w.as_ref().unwrap())
    }
}

/// Backward-compatible wrapper that uses KVCache/MambaCache.
/// Bootstraps InlineCache on first call, writes back after.
pub fn inline_decode_step(
    weights: &InlineModelWeights,
    input_id: &Array,
    kv_cache: &mut KVCache,
    mamba_cache: &mut MambaCache,
) -> Result<Array, Exception> {
    // Bootstrap InlineCache from mlx-rs caches
    let mut cache = InlineCache::from_caches(kv_cache, mamba_cache, &weights.layers);

    // Pure InlineArray forward — ZERO mlx-rs
    let token = ia_from_array(input_id);
    let logits = inline_decode_step_pure(weights, &token, &mut cache);

    // Write back updated cache state
    cache.write_back(kv_cache, mamba_cache);

    Ok(ia_to_array(&logits))
}

// ============================================================================
// InlineArray generation loop — zero mlx-rs on entire hot path
// ============================================================================

/// Run inference with ZERO mlx-rs on the hot path.
/// The entire decode+sample loop uses only InlineArray via pmetal-bridge.
/// Returns generated token IDs.
pub fn inline_generate(
    weights: &InlineModelWeights,
    cache: &mut InlineCache,
    first_token: u32,
    max_tokens: usize,
    temperature: f32,
    mut on_token: impl FnMut(u32) -> bool,
) -> Vec<u32> {
    use pmetal_bridge::inline_array as bridge;

    let mut tokens = Vec::with_capacity(max_tokens);

    // Clear prefill residue from Metal buffer cache.
    bridge::clear_cache();
    bridge::reset_peak_memory();

    eprintln!(
        "[INLINE-GEN] dtype={} active={:.0}MB",
        weights.model_dtype,
        bridge::get_active_memory() as f64 / 1e6
    );

    // Clear Metal buffer cache before decode starts
    bridge::clear_cache();

    // Build first step
    let input_token = InlineArray::from_i32(first_token as i32).reshape(&[1, 1]);
    let logits = inline_decode_step_pure(weights, &input_token, cache);
    let logits_2d = logits.squeeze(1);
    let mut current_y = sample_token(&logits_2d, temperature);
    current_y.async_eval_ref();

    let mut step_times: Vec<f64> = Vec::new();

    for step in 0..max_tokens {
        let t0 = std::time::Instant::now();

        if step == 0 {
            current_y.eval();
        }
        let token_val = current_y.item_u32();

        // DISABLED for profiling — cache detach adds sync overhead

        tokens.push(token_val);
        if !on_token(token_val) {
            break;
        }
        if step + 1 >= max_tokens {
            break;
        }

        let next_input = InlineArray::from_i32(token_val as i32).reshape(&[1, 1]);
        let next_logits = inline_decode_step_pure(weights, &next_input, cache);
        let next_logits_2d = next_logits.squeeze(1);
        current_y = sample_token(&next_logits_2d, temperature);
        current_y.async_eval_ref();

        step_times.push(t0.elapsed().as_secs_f64() * 1000.0);

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
            "[INLINE-GEN] per-step: avg={avg:.2}ms p50={p50:.2}ms = {:.0} tok/s",
            1000.0 / avg
        );
    }

    tokens
}

/// Sample one token from logits (greedy or temperature).
fn sample_token(logits_2d: &InlineArray, temperature: f32) -> InlineArray {
    pmetal_bridge::decode::sample_token(logits_2d, temperature)
}

// ============================================================================
// GDN layer forward (InlineArray)
// ============================================================================

/// Pure InlineArray GDN forward — 4 separate projections matching Python.
fn inline_gdn_forward_pure(
    lw: &InlineLayerWeights,
    normed: &InlineArray,
    _b: i32,
    _s: i32,
    cache: &mut InlineGdnCache,
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

    // For T=1 decode: use fixed-shape compiled version (shapeless=false).
    // This replays a pre-recorded tape instead of building+traversing a graph,
    // eliminating ~10ms of per-step dispatch overhead.
    if s == 1 {
        let conv_state = cache
            .conv_state
            .take()
            .unwrap_or_else(|| InlineArray::zeros(&[b, ck - 1, cd], dtype));
        let ssm_state = cache
            .ssm_state
            .take()
            .unwrap_or_else(|| InlineArray::zeros(&[b, nv, dv, dk], 10));

        let (output, new_conv, new_state) = InlineArray::compiled_gdn_layer_fixed(
            normed,
            lw.gdn_qkv_w.as_ref().unwrap(),
            lw.gdn_z_w.as_ref().unwrap(),
            lw.gdn_b_w.as_ref().unwrap(),
            lw.gdn_a_w.as_ref().unwrap(),
            lw.gdn_conv_w.as_ref().unwrap(),
            lw.gdn_q_nw.as_ref().unwrap(),
            lw.gdn_k_nw.as_ref().unwrap(),
            lw.gdn_a_log.as_ref().unwrap(),
            lw.gdn_dt_bias.as_ref().unwrap(),
            lw.gdn_norm_w.as_ref().unwrap(),
            lw.gdn_out_w.as_ref().unwrap(),
            &conv_state,
            &ssm_state,
            nv,
            nk,
            dk,
            dv,
            cd,
            ck,
            kd,
            lw.gdn_norm_eps,
        );

        cache.conv_state = Some(new_conv);
        cache.ssm_state = Some(new_state);
        return output;
    }

    // For T>1 (prefill): use direct ops (shapes vary per prompt length)
    // 4 separate projections — matches Python's in_proj_qkv/z/b/a exactly
    let qkv = normed.matmul(lw.gdn_qkv_w.as_ref().unwrap());
    let z = normed
        .matmul(lw.gdn_z_w.as_ref().unwrap())
        .reshape(&[b, s, nv, dv]);
    let b_val = normed.matmul(lw.gdn_b_w.as_ref().unwrap());
    let a_val = normed.matmul(lw.gdn_a_w.as_ref().unwrap());

    // Conv state + conv1d + fused silu
    let conv_state = cache
        .conv_state
        .take()
        .unwrap_or_else(|| InlineArray::zeros(&[b, ck - 1, cd], dtype));
    let conv_in = conv_state.concatenate_2(&qkv, 1);
    let new_conv = conv_in.slice(&[0, 1, 0], &[b, ck, cd]);
    let conv_out = conv_in
        .conv1d(lw.gdn_conv_w.as_ref().unwrap(), 1, 0, 1, cd)
        .fused_silu();

    // Split conv_out → q, k, v via slices
    let q = conv_out
        .slice(&[0, 0, 0], &[b, s, kd])
        .reshape(&[b, s, nk, dk]);
    let k = conv_out
        .slice(&[0, 0, kd], &[b, s, kd * 2])
        .reshape(&[b, s, nk, dk]);
    let v = conv_out
        .slice(&[0, 0, kd * 2], &[b, s, cd])
        .reshape(&[b, s, nv, dv]);

    // Q/K normalization
    let q = q.rms_norm(lw.gdn_q_nw.as_ref(), 1e-6);
    let k = k.rms_norm(lw.gdn_k_nw.as_ref(), 1e-6);

    // Gating
    let g = InlineArray::fused_compute_g(
        lw.gdn_a_log.as_ref().unwrap(),
        &a_val,
        lw.gdn_dt_bias.as_ref().unwrap(),
    );
    let beta = b_val.sigmoid();

    // GDN Metal kernel
    let ssm_state = cache
        .ssm_state
        .take()
        .unwrap_or_else(|| InlineArray::zeros(&[b, nv, dv, dk], 10));
    let (out, new_state) = InlineArray::gdn_metal_step(&q, &k, &v, &g, &beta, &ssm_state, s);

    cache.conv_state = Some(new_conv);
    cache.ssm_state = Some(new_state);

    // Output: rms_norm → precise_swiglu → reshape → matmul
    let out_n = out.rms_norm(lw.gdn_norm_w.as_ref(), lw.gdn_norm_eps);
    let gated = InlineArray::fused_precise_swiglu(&out_n, &z);
    gated
        .reshape(&[b, s, -1])
        .matmul(lw.gdn_out_w.as_ref().unwrap())
}

/// Pure InlineArray attention forward — zero mlx-rs.
fn inline_attn_forward_pure(
    lw: &InlineLayerWeights,
    normed: &InlineArray,
    b: i32,
    s: i32,
    cache: &mut InlineKvLayerCache,
    rope_offset: i32,
    dtype: i32,
) -> InlineArray {
    let n_heads = lw.attn_n_heads;
    let n_kv_heads = lw.attn_n_kv_heads;
    let head_dim = lw.attn_head_dim;
    let scale = lw.attn_scale;

    let q_proj_out = normed.matmul(lw.attn_q_w.as_ref().unwrap());
    let q_gate = q_proj_out.reshape(&[b, s, n_heads, head_dim * 2]);
    // split → [queries, gate] (1 Split op, not 2 Slice ops)
    let mut qg_parts = q_gate.split(&[head_dim], -1);
    let gate = qg_parts.pop().unwrap().reshape(&[b, s, n_heads * head_dim]);
    let queries = qg_parts.pop().unwrap();

    let new_keys = normed.matmul(lw.attn_k_w.as_ref().unwrap());
    let new_values = normed.matmul(lw.attn_v_w.as_ref().unwrap());

    let queries = queries.rms_norm(lw.attn_q_norm_w.as_ref(), lw.attn_q_norm_eps);
    let keys = new_keys
        .reshape(&[b, s, n_kv_heads, head_dim])
        .rms_norm(lw.attn_k_norm_w.as_ref(), lw.attn_k_norm_eps);
    let values = new_values.reshape(&[b, s, n_kv_heads, head_dim]);

    let queries = queries.transpose_axes(&[0, 2, 1, 3]);
    let keys = keys.transpose_axes(&[0, 2, 1, 3]);
    let values = values.transpose_axes(&[0, 2, 1, 3]);

    // RoPE — pure InlineArray
    let queries = queries.rope(
        lw.attn_rope_dims,
        false,
        lw.attn_rope_base,
        lw.attn_rope_scale,
        rope_offset,
    );
    let keys = keys.rope(
        lw.attn_rope_dims,
        false,
        lw.attn_rope_base,
        lw.attn_rope_scale,
        rope_offset,
    );

    // KV cache update — O(1) slice_set into pre-allocated buffer (matching Python)
    let prev = cache.offset;
    let num_new = keys.dim(2); // T=1 for decode, T=seq_len for prefill
    let next = prev + num_new;
    let b = queries.dim(0);

    if cache.keys.is_none() {
        // First call: allocate buffer with 256-step chunks.
        // Uses model dtype (bf16) — NOT float32. Float32 wastes 2x memory and
        // bandwidth in SDPA which is memory-bandwidth-bound for decode.
        let alloc = 256i32;
        cache.keys = Some(InlineArray::zeros(&[b, n_kv_heads, alloc, head_dim], dtype));
        cache.values = Some(InlineArray::zeros(&[b, n_kv_heads, alloc, head_dim], dtype));
    } else {
        // Check if we need to grow the buffer
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

    // O(1) in-place update: cache[..., prev:next, :] = new_kv
    let start = [0, 0, prev, 0];
    let stop = [b, n_kv_heads, next, head_dim];
    let k_buf = cache.keys.take().unwrap();
    let v_buf = cache.values.take().unwrap();
    cache.keys = Some(k_buf.slice_set(&keys, &start, &stop));
    cache.values = Some(v_buf.slice_set(&values, &start, &stop));
    cache.offset = next;

    // SDPA on the valid portion of the buffer
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
    let output = queries.sdpa(&valid_keys, &valid_values, scale, "causal");

    let output = output
        .transpose_axes(&[0, 2, 1, 3])
        .reshape(&[b, s, n_heads * head_dim]);
    let gated = output.multiply(&gate.sigmoid());
    gated.matmul(lw.attn_o_w.as_ref().unwrap())
}

// Keep legacy wrappers for backward compatibility
#[allow(dead_code)]
fn inline_gdn_forward(
    lw: &InlineLayerWeights,
    normed: &InlineArray,
    b: i32,
    s: i32,
    mamba_cache: &mut MambaCache,
    layer_idx: usize,
) -> Result<InlineArray, Exception> {
    let nv = lw.gdn_nv;
    let nk = lw.gdn_nk;
    let dk = lw.gdn_dk;
    let dv = lw.gdn_dv;
    let kd = lw.gdn_kd;
    let cd = lw.gdn_cd;
    let ck = lw.gdn_ck;
    let dtype = normed.dtype_raw();

    // Separate input projections (4 matmuls — matches Python exactly)
    let qkv = normed.matmul(lw.gdn_qkv_w.as_ref().unwrap());
    let z = normed
        .matmul(lw.gdn_z_w.as_ref().unwrap())
        .reshape(&[b, s, nv, dv]);
    let b_val = normed.matmul(lw.gdn_b_w.as_ref().unwrap());
    let a = normed.matmul(lw.gdn_a_w.as_ref().unwrap());

    // Conv state management
    let entry = mamba_cache
        .get_mut(layer_idx)
        .ok_or_else(|| Exception::custom(format!("missing mamba cache for layer {layer_idx}")))?;

    let conv_state = entry
        .conv_state
        .as_ref()
        .map(|s| ia_from_array(s))
        .unwrap_or_else(|| InlineArray::zeros(&[b, ck - 1, cd], dtype));

    // Conv: concat state + qkv, extract new state, apply conv1d + silu
    let conv_in = conv_state.concatenate_2(&qkv, 1);
    let new_conv = conv_in.slice(
        &[0, 1, 0], // skip first row (keep last ck-1 rows)
        &[b, ck, cd],
    );

    let conv_out = conv_in
        .conv1d(lw.gdn_conv_w.as_ref().unwrap(), 1, 0, 1, cd)
        .fused_silu();

    // Split conv output → q, k, v
    let q = conv_out
        .slice(&[0, 0, 0], &[b, s, kd])
        .reshape(&[b, s, nk, dk]);
    let k = conv_out
        .slice(&[0, 0, kd], &[b, s, kd * 2])
        .reshape(&[b, s, nk, dk]);
    let v = conv_out
        .slice(&[0, 0, kd * 2], &[b, s, cd])
        .reshape(&[b, s, nv, dv]);

    // Q/K normalization
    let q = q.rms_norm(lw.gdn_q_nw.as_ref(), 1e-6);
    let k = k.rms_norm(lw.gdn_k_nw.as_ref(), 1e-6);

    // GDN recurrence — compute g/beta with FUSED ops (1 dispatch each),
    // then dispatch Metal kernel with pre-computed g/beta.
    let a_log = lw.gdn_a_log.as_ref().unwrap();
    let dt_bias = lw.gdn_dt_bias.as_ref().unwrap();

    // Fused compute_g: 6 ops → 1 compiled dispatch
    let g = InlineArray::fused_compute_g(a_log, &a, dt_bias);
    let beta = b_val.sigmoid();

    let ssm_state = entry
        .ssm_state
        .as_ref()
        .map(|s| ia_from_array(s))
        .unwrap_or_else(|| InlineArray::zeros(&[b, nv, dv, dk], 10));

    // Direct Metal kernel dispatch with pre-computed g/beta
    let (out, new_state) = InlineArray::gdn_metal_step(&q, &k, &v, &g, &beta, &ssm_state, s);

    // Update cache
    entry.conv_state = Some(ia_to_array(&new_conv));
    entry.ssm_state = Some(ia_to_array(&new_state));

    // Gated norm — fused precise_swiglu: (silu(gate.f32()) * norm.f32()).as(dtype)
    let out_n = out.rms_norm(lw.gdn_norm_w.as_ref(), lw.gdn_norm_eps);
    let gated = InlineArray::fused_precise_swiglu(&out_n, &z);

    // Output projection
    Ok(gated
        .reshape(&[b, s, -1])
        .matmul(lw.gdn_out_w.as_ref().unwrap()))
}

// ============================================================================
// Attention layer forward (InlineArray)
// ============================================================================

#[allow(dead_code)] // Infrastructure for inline (zero-alloc) attention forward path; not yet wired into dispatch
fn inline_attn_forward(
    lw: &InlineLayerWeights,
    normed: &InlineArray,
    b: i32,
    s: i32,
    kv_cache: &mut KVCache,
    layer_idx: usize,
) -> Result<InlineArray, Exception> {
    let n_heads = lw.attn_n_heads;
    let n_kv_heads = lw.attn_n_kv_heads;
    let head_dim = lw.attn_head_dim;
    let scale = lw.attn_scale;

    // Q (with gate), K, V projections
    let q_proj_out = normed.matmul(lw.attn_q_w.as_ref().unwrap());
    let q_gate = q_proj_out.reshape(&[b, s, n_heads, head_dim * 2]);
    let queries = q_gate.slice(&[0, 0, 0, 0], &[b, s, n_heads, head_dim]);
    let gate = q_gate
        .slice(&[0, 0, 0, head_dim], &[b, s, n_heads, head_dim * 2])
        .reshape(&[b, s, n_heads * head_dim]);

    let new_keys = normed.matmul(lw.attn_k_w.as_ref().unwrap());
    let new_values = normed.matmul(lw.attn_v_w.as_ref().unwrap());

    // Q/K norms
    let queries = queries.rms_norm(lw.attn_q_norm_w.as_ref(), lw.attn_q_norm_eps);
    let keys = new_keys
        .reshape(&[b, s, n_kv_heads, head_dim])
        .rms_norm(lw.attn_k_norm_w.as_ref(), lw.attn_k_norm_eps);
    let values = new_values.reshape(&[b, s, n_kv_heads, head_dim]);

    // Transpose to [B, H, S, D]
    let queries = queries.transpose_axes(&[0, 2, 1, 3]);
    let keys = keys.transpose_axes(&[0, 2, 1, 3]);
    let values = values.transpose_axes(&[0, 2, 1, 3]);

    // RoPE — pure InlineArray, no Array conversion
    let offset = kv_cache.rope_offset();
    let queries = queries.rope(
        lw.attn_rope_dims,
        false,
        lw.attn_rope_base,
        lw.attn_rope_scale,
        offset,
    );
    let keys = keys.rope(
        lw.attn_rope_dims,
        false,
        lw.attn_rope_base,
        lw.attn_rope_scale,
        offset,
    );

    // KV cache update — use Array path for now (manages pre-allocated buffer)
    // TODO: port to InlineArray slice/concatenate for zero-copy
    let (cached_keys, cached_values) =
        kv_cache.update_and_fetch(layer_idx, &ia_to_array(&keys), &ia_to_array(&values))?;

    // SDPA — pure InlineArray
    let k_cached = ia_from_array(&cached_keys);
    let v_cached = ia_from_array(&cached_values);
    let output = queries.sdpa(&k_cached, &v_cached, scale, "causal");

    // Reshape + gate
    let output = output
        .transpose_axes(&[0, 2, 1, 3])
        .reshape(&[b, s, n_heads * head_dim]);
    let gated = output.multiply(&gate.sigmoid());

    // O projection
    Ok(gated.matmul(lw.attn_o_w.as_ref().unwrap()))
}
