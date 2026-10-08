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

fn ia_from_weight(arr: &Array) -> InlineArray {
    if arr.dtype() == Dtype::Uint8 {
        arr.from_fp8(Dtype::Bfloat16.as_i32())
    } else {
        arr.clone()
    }
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
    attn_scaled_rope: Option<pmetal_bridge::qwen3_native::mrope::ScaledRope>,

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

impl InlineLayerWeights {
    /// Every array this layer holds; the ones its kind does not use are absent.
    fn arrays_mut(&mut self) -> impl Iterator<Item = &mut InlineArray> {
        [
            &mut self.input_ln_w,
            &mut self.post_ln_w,
            &mut self.mlp_gate_w,
            &mut self.mlp_up_w,
            &mut self.mlp_down_w,
        ]
        .into_iter()
        .chain(
            [
                &mut self.attn_q_w,
                &mut self.attn_k_w,
                &mut self.attn_v_w,
                &mut self.attn_o_w,
                &mut self.attn_q_norm_w,
                &mut self.attn_k_norm_w,
                &mut self.gdn_qkv_w,
                &mut self.gdn_z_w,
                &mut self.gdn_b_w,
                &mut self.gdn_a_w,
                &mut self.gdn_conv_w,
                &mut self.gdn_q_nw,
                &mut self.gdn_k_nw,
                &mut self.gdn_a_log,
                &mut self.gdn_dt_bias,
                &mut self.gdn_norm_w,
                &mut self.gdn_out_w,
            ]
            .into_iter()
            .flatten(),
        )
    }
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
                    conv_state: entry.and_then(|e| e.conv_state.as_ref()).map(ia_from_array),
                    ssm_state: entry.and_then(|e| e.ssm_state.as_ref()).map(ia_from_array),
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

        // Snapshotted before any layer runs, so every attention layer is at
        // the same offset and the first one answers for all of them.
        let rope_offset = attn_layer_indices
            .first()
            .map_or(0, |&layer| kv_cache.rope_offset_for(layer));

        InlineCache {
            gdn_caches,
            kv_caches,
            gdn_layer_indices,
            attn_layer_indices,
            rope_offset,
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
        // This decode path hard-wires the SiLU output gate. A sigmoid-gated
        // checkpoint declines it and decodes on the standard path instead.
        if config.gdn_gate()? != super::qwen3_next::GateActivation::Silu {
            return Err(Exception::custom(
                "the InlineArray decode path implements only the SiLU gated-delta-net gate",
            ));
        }
        let mut embed_w = ia_from_weight(model.model.embed_tokens.weight.as_ref());
        let mut final_norm_w = ia_from_weight(model.model.norm.weight.as_ref());
        let final_norm_eps = model.model.norm.eps;
        // Pre-transposed like every other projection here: the decode step
        // computes `hidden @ lm_head_w`. Kept as the `[vocab, hidden]` weight,
        // that matmul threw on any untied model and silently used the wrong
        // matrix when vocab == hidden.
        let mut lm_head_w = model
            .lm_head
            .as_ref()
            .map(|l| ia_from_weight(l.weight.as_ref()).t());

        let mut layers = Vec::with_capacity(model.model.layers.len());
        for (li, layer) in model.model.layers.iter_mut().enumerate() {
            let mut lw = InlineLayerWeights {
                is_linear: layer.is_linear,
                input_ln_w: ia_from_weight(layer.input_layernorm.weight.as_ref()),
                input_ln_eps: layer.input_layernorm.eps,
                post_ln_w: ia_from_weight(layer.post_attention_layernorm.weight.as_ref()),
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
                attn_scaled_rope: None,
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
                    lw.mlp_gate_w = ia_from_weight(mlp.gate_proj.weight.as_ref()).t();
                    lw.mlp_up_w = ia_from_weight(mlp.up_proj.weight.as_ref()).t();
                    lw.mlp_down_w = ia_from_weight(mlp.down_proj.weight.as_ref()).t();
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
                lw.gdn_qkv_w = Some(ia_from_weight(gdn.in_proj_qkv.weight.as_ref()).t());
                lw.gdn_z_w = Some(ia_from_weight(gdn.in_proj_z.weight.as_ref()).t());
                lw.gdn_b_w = Some(ia_from_weight(gdn.in_proj_b.weight.as_ref()).t());
                lw.gdn_a_w = Some(ia_from_weight(gdn.in_proj_a.weight.as_ref()).t());
                lw.gdn_conv_w = Some(ia_from_weight(gdn.conv1d.weight.as_ref()));
                lw.gdn_q_nw = Some(ia_from_weight(&gdn.q_norm_weight));
                lw.gdn_k_nw = Some(ia_from_weight(&gdn.k_norm_weight));
                lw.gdn_a_log = Some(ia_from_weight(gdn.a_log.as_ref()));
                lw.gdn_dt_bias = Some(ia_from_weight(gdn.dt_bias.as_ref()));
                lw.gdn_norm_w = Some(ia_from_weight(gdn.norm.weight.as_ref()));
                lw.gdn_norm_eps = gdn.norm.eps;
                lw.gdn_out_w = Some(ia_from_weight(gdn.out_proj.weight.as_ref()).t());
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
                lw.attn_q_w = Some(ia_from_weight(attn.q_proj.weight.as_ref()).t());
                lw.attn_k_w = Some(ia_from_weight(attn.k_proj.weight.as_ref()).t());
                lw.attn_v_w = Some(ia_from_weight(attn.v_proj.weight.as_ref()).t());
                lw.attn_o_w = Some(ia_from_weight(attn.o_proj.weight.as_ref()).t());
                lw.attn_q_norm_w = Some(ia_from_weight(attn.q_norm.weight.as_ref()));
                lw.attn_q_norm_eps = attn.q_norm.eps;
                lw.attn_k_norm_w = Some(ia_from_weight(attn.k_norm.weight.as_ref()));
                lw.attn_k_norm_eps = attn.k_norm.eps;
                lw.attn_n_heads = attn.n_heads;
                lw.attn_n_kv_heads = attn.n_kv_heads;
                lw.attn_head_dim = attn.head_dim;
                lw.attn_scale = attn.scale;
                lw.attn_rope_dims = attn.rope_dims;
                lw.attn_rope_base = attn.effective_base;
                lw.attn_rope_scale = attn.rope_scale;
                lw.attn_scaled_rope = attn.scaled_rope.clone();
            }

            layers.push(lw);
        }

        let model_dtype = embed_w.dtype_raw();

        // Every entry above shares its buffer with the model's own weight; the
        // projections are transposed views of it, not copies. Evaluating them
        // here settles the views (and dequantizes any FP8 weight once, the
        // only case that needs a new buffer), and detaching drops the
        // graph nodes that produced them, so each decode step's graph starts
        // at plain arrays.
        let settle = |w: &mut InlineArray| {
            w.eval();
            w.detach();
        };
        settle(&mut embed_w);
        settle(&mut final_norm_w);
        lm_head_w.iter_mut().for_each(settle);
        for lw in &mut layers {
            lw.arrays_mut().for_each(settle);
        }

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

    for lw in weights.layers.iter() {
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
            pmetal_bridge::qwen3_native::family::gdn_qk_rms_norm_eps(dk),
            // `from_model` admits only SiLU-gated checkpoints onto this path.
            false,
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
    // The last `ck - 1` inputs (`[1, ck)` is that only for s == 1).
    let new_conv = conv_in.slice(&[0, s, 0], &[b, s + ck - 1, cd]);
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
    let qk_eps = pmetal_bridge::qwen3_native::family::gdn_qk_rms_norm_eps(lw.gdn_dk);
    let q = q.rms_norm(lw.gdn_q_nw.as_ref(), qk_eps);
    let k = k.rms_norm(lw.gdn_k_nw.as_ref(), qk_eps);

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
    let (queries, keys) = if let Some(scaled) = &lw.attn_scaled_rope {
        (
            scaled.apply(&queries, rope_offset),
            scaled.apply(&keys, rope_offset),
        )
    } else {
        (
            queries.rope(
                lw.attn_rope_dims,
                false,
                lw.attn_rope_base,
                lw.attn_rope_scale,
                rope_offset,
            ),
            keys.rope(
                lw.attn_rope_dims,
                false,
                lw.attn_rope_base,
                lw.attn_rope_scale,
                rope_offset,
            ),
        )
    };

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
