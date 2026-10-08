//! Per-layer + full-model weight bundles and safetensors loading.
//!
//! Sanitization mirrors Python's `Model.sanitize()`:
//!   * fused `gate_up_proj` (and its `_blocks` / `_scales` MXFP4 variants)
//!     → split into `gate_proj` / `up_proj`
//!   * fused `gate_up_proj_bias` → split into `gate_proj.bias` + `up_proj.bias`
//!   * `down_proj_blocks` → `down_proj.weight` (flattened)
//!   * `down_proj_bias`   → `down_proj.bias`

use crate::InlineArray;

use super::{AttentionLayerType, GptOssConfig};

/// GPT-OSS layer weights — attention + MoE.
pub(super) struct LayerWeights {
    // Layer norms
    pub(super) input_ln_w: InlineArray,
    pub(super) input_ln_eps: f32,
    pub(super) post_ln_w: InlineArray,
    pub(super) post_ln_eps: f32,

    // Attention projections (pre-transposed [in, out] for direct matmul)
    pub(super) attn_q_w: InlineArray, // [hidden, n_heads * head_dim]
    pub(super) attn_q_b: Option<InlineArray>, // [n_heads * head_dim]
    pub(super) attn_k_w: InlineArray, // [hidden, n_kv_heads * head_dim]
    pub(super) attn_k_b: Option<InlineArray>, // [n_kv_heads * head_dim]
    pub(super) attn_v_w: InlineArray, // [hidden, n_kv_heads * head_dim]
    pub(super) attn_v_b: Option<InlineArray>, // [n_kv_heads * head_dim]
    pub(super) attn_o_w: InlineArray, // [n_heads * head_dim, hidden]
    pub(super) attn_o_b: Option<InlineArray>, // [hidden]

    // Attention dims
    pub(super) attn_n_heads: i32,
    pub(super) attn_n_kv_heads: i32,
    pub(super) attn_head_dim: i32,
    pub(super) attn_scale: f32,
    /// The rotary embedding (YaRN on every release).
    pub(super) attn_rotary: crate::rope::RotaryEmbedding,
    pub(super) attn_is_sliding: bool,
    pub(super) attn_sliding_window: i32,

    // MoE: router + stacked expert projections
    // Router: [hidden, num_experts] (NO transpose — direct matmul hidden @ router_w)
    pub(super) moe_router_w: InlineArray,
    // Stacked expert weights — shape [num_experts, hidden_size, intermediate_size]
    // pre-transposed to [num_experts, intermediate_size, hidden_size] for batched gather_mm
    // but actually stored as [num_experts, hidden, intermediate] with matmul handling transpose
    pub(super) moe_gate_w: InlineArray, // [num_experts, hidden, intermediate]
    pub(super) moe_gate_b: InlineArray, // [num_experts, intermediate]
    pub(super) moe_up_w: InlineArray,   // [num_experts, hidden, intermediate]
    pub(super) moe_up_b: InlineArray,   // [num_experts, intermediate]
    pub(super) moe_down_w: InlineArray, // [num_experts, intermediate, hidden]
    pub(super) moe_down_b: InlineArray, // [num_experts, hidden]

    pub(super) moe_num_experts: i32,
    pub(super) moe_top_k: i32,

    // SwiGLU parameters
    pub(super) swiglu_alpha: f32,
    pub(super) swiglu_limit: f32,
}

/// All GPT-OSS model weights as InlineArray. Zero dependency on mlx-rs.
pub struct NativeWeights {
    pub embed_w: InlineArray,
    pub final_norm_w: InlineArray,
    pub final_norm_eps: f32,
    /// None when `tie_word_embeddings = true`.
    pub lm_head_w: Option<InlineArray>,
    pub tie_word_embeddings: bool,
    /// Per-layer weights — opaque to callers.
    pub(super) layers: Vec<LayerWeights>,
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
    // ── Step 1+2: Shard discovery and bulk-load ─────────────────────────────
    let shard_paths = crate::native_loader::discover_safetensors_shards(model_dir)?;
    let mut raw = crate::native_loader::load_shards_into_map(&shard_paths, model_dir)?;

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
    let rotary = config.rotary()?;
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
            attn_rotary: rotary.clone(),
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
    let zero = InlineArray::scalar_with_dtype(0.0, model_dtype);
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

    let split_dim = ndim - 2; // second-to-last axis
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
    let last = ndim - 1;
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
