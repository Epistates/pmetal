//! Per-layer + full-model weight bundles + safetensors loader.
//!
//! Sanitization mirrors Python's `Model.sanitize()`:
//! * Vision / projector weight removal
//! * VLM prefix stripping (`language_model.model.X` → `model.X`)
//! * Expert weight split: `experts.gate_up_proj` → `experts.gate_proj.weight`
//!   + `experts.up_proj.weight` (split last axis)
//! * Expert down projection: `experts.down_proj` → `experts.down_proj.weight`
//!   (swapaxes(1,2))
//! * All projection weights pre-transposed to `[in, out]` form for efficient matmul

use crate::InlineArray;

use super::Llama4Config;

/// MoE expert weight block. Weights are stored pre-transposed to `[num_experts, in, out]`
/// form for `gather_mm` (which expects `b` with shape `[experts, K, N]`).
pub(super) struct MoeWeights {
    /// `[num_experts, in_dim, expert_hidden]` — gate projection
    pub(super) experts_gate_w: InlineArray,
    /// `[num_experts, in_dim, expert_hidden]` — up projection
    pub(super) experts_up_w: InlineArray,
    /// `[num_experts, expert_hidden, in_dim]` — down projection
    pub(super) experts_down_w: InlineArray,
    /// Router: `[in_dim, num_experts]` (pre-transposed from `[num_experts, in_dim]`)
    pub(super) router_w: InlineArray,
    /// Shared expert gate: `[in_dim, intermediate_size_mlp]`
    pub(super) shared_gate_w: InlineArray,
    /// Shared expert up: `[in_dim, intermediate_size_mlp]`
    pub(super) shared_up_w: InlineArray,
    /// Shared expert down: `[intermediate_size_mlp, in_dim]`
    pub(super) shared_down_w: InlineArray,
}

pub(super) struct LayerWeights {
    // ── iRoPE meta ──
    pub(super) use_rope: bool, // false for NoPE (global) layers

    // ── Layer norms ──
    pub(super) input_ln_w: InlineArray,
    pub(super) post_ln_w: InlineArray,
    pub(super) norm_eps: f32,

    // ── Attention projection ──
    pub(super) attn_q_w: InlineArray, // [in, n_heads * head_dim]
    pub(super) attn_k_w: InlineArray, // [in, n_kv_heads * head_dim]
    pub(super) attn_v_w: InlineArray, // [in, n_kv_heads * head_dim]
    pub(super) attn_o_w: InlineArray, // [n_heads * head_dim, in]

    // Optional bias (when attention_bias=true)
    pub(super) attn_q_b: Option<InlineArray>,
    pub(super) attn_k_b: Option<InlineArray>,
    pub(super) attn_v_b: Option<InlineArray>,
    pub(super) attn_o_b: Option<InlineArray>,

    // QK-norm (only on RoPE layers — same flag for both Q and K, no learned weight)
    pub(super) attn_qk_norm: bool, // true → apply rms_norm(eps=1e-6) to both Q and K

    // Attention shape config (stored per-layer for self-containedness)
    pub(super) n_heads: i32,
    pub(super) n_kv_heads: i32,
    pub(super) head_dim: i32,
    pub(super) attn_scale: f32,
    pub(super) rope_base: f32,
    pub(super) rope_scale: f32,

    // Temperature tuning for NoPE layers
    pub(super) attn_temperature_tuning: i32,
    pub(super) floor_scale: i32,
    pub(super) layer_attn_scale: f32,

    // ── Feed-forward: either MoE or dense MLP ──
    pub(super) is_moe: bool,
    // Dense MLP (non-MoE layers)
    pub(super) mlp_gate_w: Option<InlineArray>, // [in, intermediate_size_mlp]
    pub(super) mlp_up_w: Option<InlineArray>,
    pub(super) mlp_down_w: Option<InlineArray>,
    // MoE
    pub(super) moe: Option<MoeWeights>,
}

/// All model weights as InlineArrays. Zero dependency on mlx-rs.
pub struct NativeWeights {
    pub embed_w: InlineArray,      // [vocab, hidden]
    pub final_norm_w: InlineArray, // [hidden]
    pub final_norm_eps: f32,
    /// `None` when `tie_word_embeddings = true`.
    pub lm_head_w: Option<InlineArray>, // [hidden, vocab] (pre-transposed)
    pub tie_word_embeddings: bool,
    /// Per-layer weights — only accessed via [`super::forward_step`].
    pub(super) layers: Vec<LayerWeights>,
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

    // ── Step 1+2: Shard discovery and bulk-load ─────────────────────────────
    let shard_paths = crate::native_loader::discover_safetensors_shards(model_dir)?;
    let mut raw = crate::native_loader::load_shards_into_map(&shard_paths, model_dir)?;

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
    // Raw safetensors format:
    //   `layers.{l}.feed_forward.experts.gate_up_proj` — shape [E, in_dim, 2*expert_hidden]
    //   `layers.{l}.feed_forward.experts.down_proj`    — shape [E, hidden_size, expert_hidden]
    //
    // Target format (after sanitize + our pre-transpose for direct gather_mm):
    //   `experts.gate_proj.weight` : [E, in_dim, expert_hidden] (split only, no swapaxes)
    //   `experts.up_proj.weight`   : [E, in_dim, expert_hidden]
    //   `experts.down_proj.weight` : [E, expert_hidden, hidden_size] (swapaxes(1,2))
    //
    // At runtime, gather_mm(x, b, None, indices) expects b in shape [E, K, N].
    // - gate/up: x=[B,1,1,in_dim], b=[E,in_dim,expert_hidden] → [B,1,1,expert_hidden] ✓
    // - down:    x=[B,1,1,expert_hidden], b=[E,expert_hidden,hidden_size] → [B,1,1,hidden_size] ✓

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
    let zero = InlineArray::scalar_with_dtype(0.0, detected_dtype);
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
