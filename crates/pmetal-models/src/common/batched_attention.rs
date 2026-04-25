//! Shared fused `[N_active, 1, H]` GQA attention block.
//!
//! Every architecture that reports `supports_fused_batched = true` and
//! follows the standard GQA pattern (Llama / Qwen / Mistral / Gemma /
//! Phi / Cohere / Granite / Qwen3MoE / GptOss) routes its per-layer
//! attention through [`batched_gqa_attn`]. Arch-specific flags
//! (qk-norm, logit softcap, partial RoPE) live on
//! [`BatchedGqaAttnCfg`]; the block itself is one implementation.
//!
//! # Shapes
//!
//! Input:
//! - `x: [N_active, 1, hidden]` — per-slot last-token hidden state.
//!
//! Output:
//! - `[N_active, 1, hidden]` — attention output, fed straight into the
//!   arch's residual add + post-norm + MLP.
//!
//! # Contract with [`FusedBatchKVCache`]
//!
//! The cache's `update_and_fetch_batched` reads each slot's *pre-update*
//! offset when writing the new K/V at position `offset[slot]`. This
//! block reads the same pre-update offset to build per-slot RoPE
//! positions, so Q and the newly-written K/V rotate with identical
//! positions — preserving the invariant the serial `forward_with_cache`
//! path relies on (`apply_rope(Q, ..., offset)` matched by
//! `apply_rope(K, ..., offset)` before the cache write).

use pmetal_bridge::compat::{Array, Dtype, Exception, Module, nn, ops};
use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::apply_rope_with_per_batch_positions,
};
use pmetal_mlx::kv_cache::FusedBatchKVCache;

use crate::decoder_layer::{MlpModule, NormModule};

/// Arch-agnostic configuration for a single GQA attention layer.
///
/// Scalar-only by design so a layer loop can stack-allocate one of
/// these per layer without borrowing the model. The `&mut` references
/// to `nn::Linear` / `nn::RmsNorm` are passed as separate function
/// arguments.
#[derive(Debug, Clone, Copy)]
pub struct BatchedGqaAttnCfg {
    /// Number of query heads.
    pub n_heads: i32,
    /// Number of key/value heads (GQA/MQA).
    pub n_kv_heads: i32,
    /// Head dimension (applies to Q/K/V; DeepSeek MLA is out of scope).
    pub head_dim: i32,
    /// Softmax scale (typically `1/sqrt(head_dim)`).
    pub scale: f32,
    /// Effective RoPE base after scaling.
    pub rope_base: f32,
    /// RoPE position scale (Linear / Dynamic / YaRN factor).
    pub rope_scale: f32,
    /// Dimensions RoPE rotates (usually `head_dim`; Phi uses partial).
    pub rope_dims: i32,
    /// Traditional (interleaved) vs. split-half RoPE.
    pub rope_traditional: bool,
    /// Optional Gemma2-style logit softcapping.
    pub logit_softcap: Option<f32>,
    /// Sliding-window width. When `Some(w)`, each slot only attends to
    /// the most recent `w` tokens (GPT-OSS/Cohere interleaved pattern).
    /// The lower-bound mask is composed on top of the cache's padding
    /// mask inside [`batched_gqa_attn`].
    pub sliding_window: Option<i32>,
}

impl BatchedGqaAttnCfg {
    /// Build a config with sane defaults: non-traditional RoPE, full-dim
    /// rotation, standard `1/sqrt(head_dim)` scale, no softcap.
    pub fn new(
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        rope_base: f32,
        rope_scale: f32,
    ) -> Self {
        Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope_base,
            rope_scale,
            rope_dims: head_dim,
            rope_traditional: false,
            logit_softcap: None,
            sliding_window: None,
        }
    }

    /// Set a custom scale (overrides the default `1/sqrt(head_dim)`).
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Rotate only the first `rope_dims` of each head (Phi style).
    pub fn with_rope_dims(mut self, rope_dims: i32) -> Self {
        self.rope_dims = rope_dims;
        self
    }

    /// Use traditional (interleaved) RoPE instead of split-half.
    pub fn with_rope_traditional(mut self, traditional: bool) -> Self {
        self.rope_traditional = traditional;
        self
    }

    /// Enable Gemma2-style logit softcapping.
    pub fn with_logit_softcap(mut self, cap: f32) -> Self {
        self.logit_softcap = Some(cap);
        self
    }

    /// Restrict attention to the last `window` tokens (per slot).
    ///
    /// At decode time slot `n` has already written through position
    /// `q_pos_n`, so the valid range becomes
    /// `[max(0, q_pos_n - window + 1), q_pos_n]`. The upper bound is
    /// already enforced by the cache's padding mask; this method
    /// introduces the lower bound.
    pub fn with_sliding_window(mut self, window: i32) -> Self {
        self.sliding_window = Some(window);
        self
    }

    /// Clear any sliding-window setting (full attention).
    pub fn without_sliding_window(mut self) -> Self {
        self.sliding_window = None;
        self
    }
}

/// Fused single-layer GQA attention for continuous-batching decode.
///
/// Runs Q/K/V projections, per-slot RoPE, fused SDPA against the shared
/// [`FusedBatchKVCache`], and the output projection — all in batched
/// `[N_active, 1, H]` space.
///
/// # Parameters
/// - `x` — input `[N_active, 1, hidden]`.
/// - `q_proj` / `k_proj` / `v_proj` / `o_proj` — arch-owned linears.
/// - `q_norm` / `k_norm` — optional Qwen3-style post-projection RMS norm
///   applied on the `[N_active, heads, 1, head_dim]` tensor.
/// - `cfg` — scalar config.
/// - `cache` — fused KV cache shared across all slots.
/// - `active_indices` — batch rows in the cache, one per row of `x`.
/// - `layer_idx` — 0-based layer index into the cache's per-layer buffers.
#[allow(clippy::too_many_arguments)]
pub fn batched_gqa_attn(
    x: &Array,
    q_proj: &mut nn::Linear,
    k_proj: &mut nn::Linear,
    v_proj: &mut nn::Linear,
    o_proj: &mut nn::Linear,
    q_norm: Option<&mut nn::RmsNorm>,
    k_norm: Option<&mut nn::RmsNorm>,
    cfg: &BatchedGqaAttnCfg,
    cache: &mut FusedBatchKVCache,
    active_indices: &[usize],
    layer_idx: usize,
) -> Result<Array, Exception> {
    let shape = x.shape();
    if shape.len() != 3 || shape[1] != 1 {
        return Err(Exception::custom(format!(
            "batched_gqa_attn: expected [N_active, 1, hidden], got {shape:?}"
        )));
    }
    let n_active = shape[0];
    if n_active as usize != active_indices.len() {
        return Err(Exception::custom(format!(
            "batched_gqa_attn: active_indices len {} != input batch {}",
            active_indices.len(),
            n_active
        )));
    }

    // Project to Q/K/V.
    let queries = Module::forward(q_proj, x)?;
    let keys = Module::forward(k_proj, x)?;
    let values = Module::forward(v_proj, x)?;

    // Reshape to [N_active, heads, 1, head_dim] layout used by RoPE / SDPA.
    let queries = queries
        .reshape(&[n_active, 1, cfg.n_heads, cfg.head_dim])
        .transpose_axes(&[0, 2, 1, 3]);
    let keys = keys
        .reshape(&[n_active, 1, cfg.n_kv_heads, cfg.head_dim])
        .transpose_axes(&[0, 2, 1, 3]);
    let values = values
        .reshape(&[n_active, 1, cfg.n_kv_heads, cfg.head_dim])
        .transpose_axes(&[0, 2, 1, 3]);

    // Optional Qwen3-style qk-norm (applied per head on the last dim).
    let queries = if let Some(norm) = q_norm {
        Module::forward(norm, &queries)?
    } else {
        queries
    };
    let keys = if let Some(norm) = k_norm {
        Module::forward(norm, &keys)?
    } else {
        keys
    };

    // Build per-slot position ids from the cache's *pre-update* offsets.
    // position_ids[n, 0] = cache.offset(active_indices[n]).
    let pre_offsets: Vec<i32> = active_indices
        .iter()
        .map(|&idx| cache.offset(idx) as i32)
        .collect();
    let position_ids = Array::from_i32_slice_shaped(&pre_offsets, &[n_active, 1]);

    let queries = apply_rope_with_per_batch_positions(
        &queries,
        &position_ids,
        cfg.rope_dims,
        cfg.rope_traditional,
        cfg.rope_base,
        cfg.rope_scale,
    )?;
    let keys = apply_rope_with_per_batch_positions(
        &keys,
        &position_ids,
        cfg.rope_dims,
        cfg.rope_traditional,
        cfg.rope_base,
        cfg.rope_scale,
    )?;

    // Write new K/V, fetch T_max-sliced K/V + per-slot padding mask.
    let (k_all, v_all, mask) =
        cache.update_and_fetch_batched(layer_idx, active_indices, &keys, &values)?;

    // Optionally overlay sliding-window lower-bound mask. The cache's
    // padding mask already blocks `t >= post_offset`; sliding window
    // additionally blocks `t < q_pos - window + 1`.
    //
    // Critical: `q_pos` here is the *current layer's* pre-update offset,
    // not layer-0's. The serial `AttentionMaskType::SlidingWindow` derives
    // q_pos from `K_seq - 1` (the last cached position in *this* layer's
    // KV buffer), and the fused path must match that to stay parity-clean
    // with the per-layer-offset quirk preserved by the cache.
    let mask = if let Some(window) = cfg.sliding_window {
        let mask_shape = mask.shape();
        debug_assert_eq!(
            mask_shape.len(),
            4,
            "cache padding mask must be 4-D, got {mask_shape:?}"
        );
        let t_max_i32 = mask_shape[3];
        // After `update_and_fetch_batched`, `offset_for(layer_idx, idx)` is
        // the post-update offset. q_pos = post_offset - 1, so the lower
        // bound (`q_pos - window + 1`) simplifies to `post_offset - window`.
        let lower_bounds: Vec<i32> = active_indices
            .iter()
            .map(|&idx| cache.offset_for(layer_idx, idx) as i32 - window)
            .collect();
        let lower =
            Array::from_i32_slice_shaped(&lower_bounds, &[n_active, 1, 1, 1])
                .as_dtype(Dtype::Float32.as_i32());
        let t_range = ops::arange_range(0, t_max_i32).reshape(&[1, 1, 1, t_max_i32]);
        let invalid = ops::less(&t_range, &lower);
        let zero = Array::from_f32(0.0);
        let neg_inf = Array::from_f32(f32::NEG_INFINITY);
        let sliding = ops::r#where(&invalid, &neg_inf, &zero);
        mask.add(&sliding)
    } else {
        mask
    };

    // Fused SDPA with explicit per-slot additive mask (no causal flag —
    // the mask already encodes left-padding up to each slot's offset).
    let mut attn_cfg = FusedAttentionConfig::new(cfg.n_heads, cfg.n_kv_heads, cfg.head_dim)
        .with_scale(cfg.scale)
        .with_mask_type(AttentionMaskType::None);
    if let Some(cap) = cfg.logit_softcap {
        attn_cfg = attn_cfg.with_logit_softcapping(cap);
    }
    let output = fused_sdpa(&queries, &k_all, &v_all, &attn_cfg, Some(&mask))?;

    // [N_active, heads, 1, head_dim] → [N_active, 1, n_heads*head_dim].
    let output = output
        .transpose_axes(&[0, 2, 1, 3])
        .reshape(&[n_active, 1, -1]);

    Module::forward(o_proj, &output)
}

/// Run one **Gemma2-style 4-norm peri-norm** decoder layer through the
/// fused path.
///
/// Pattern:
/// ```text
///   normed     = input_layernorm(x)
///   attn_out   = post_attention_layernorm(attn(normed))
///   h          = x + attn_out
///   normed     = pre_feedforward_layernorm(h)
///   mlp_out    = post_feedforward_layernorm(mlp(normed))
///   return       h + mlp_out
/// ```
/// Used by Gemma2 and Gemma3 (which share the layer body; per-layer
/// sliding-window selection rides on `attn_cfg.sliding_window`, set per
/// call from the arch's layer loop).
#[allow(clippy::too_many_arguments)]
pub fn batched_perinorm_layer<N: NormModule, M: MlpModule>(
    hidden: &Array,
    input_layernorm: &mut N,
    q_proj: &mut nn::Linear,
    k_proj: &mut nn::Linear,
    v_proj: &mut nn::Linear,
    o_proj: &mut nn::Linear,
    q_norm: Option<&mut nn::RmsNorm>,
    k_norm: Option<&mut nn::RmsNorm>,
    post_attention_layernorm: &mut N,
    pre_feedforward_layernorm: &mut N,
    post_feedforward_layernorm: &mut N,
    mlp: &mut M,
    attn_cfg: &BatchedGqaAttnCfg,
    cache: &mut FusedBatchKVCache,
    active_indices: &[usize],
    layer_idx: usize,
) -> Result<Array, Exception> {
    let normed = input_layernorm.forward(hidden)?;
    let attn_out = batched_gqa_attn(
        &normed,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        attn_cfg,
        cache,
        active_indices,
        layer_idx,
    )?;
    let attn_out = post_attention_layernorm.forward(&attn_out)?;
    let h = hidden.add(&attn_out);
    let normed = pre_feedforward_layernorm.forward(&h)?;
    let mlp_out = mlp.forward(&normed)?;
    let mlp_out = post_feedforward_layernorm.forward(&mlp_out)?;
    Ok(h.add(&mlp_out))
}

/// Run one **parallel** decoder block through the fused path.
///
/// Pattern: `x + attn(norm(x)) + ffn(norm(x))` — both branches consume the
/// same normed input and there is no post-attention norm. Used by Cohere
/// (Command R/R+/A) and any future parallel-decoder model.
///
/// Distinct from [`batched_prenorm_layer`] which runs the standard
/// pre-norm sequence (`norm → attn → +residual → norm → mlp → +residual`).
#[allow(clippy::too_many_arguments)]
pub fn batched_parallel_block<N: NormModule, M: MlpModule>(
    hidden: &Array,
    input_layernorm: &mut N,
    q_proj: &mut nn::Linear,
    k_proj: &mut nn::Linear,
    v_proj: &mut nn::Linear,
    o_proj: &mut nn::Linear,
    q_norm: Option<&mut nn::RmsNorm>,
    k_norm: Option<&mut nn::RmsNorm>,
    mlp: &mut M,
    attn_cfg: &BatchedGqaAttnCfg,
    cache: &mut FusedBatchKVCache,
    active_indices: &[usize],
    layer_idx: usize,
) -> Result<Array, Exception> {
    let normed = input_layernorm.forward(hidden)?;
    let attn_out = batched_gqa_attn(
        &normed,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        attn_cfg,
        cache,
        active_indices,
        layer_idx,
    )?;
    let mlp_out = mlp.forward(&normed)?;
    Ok(hidden.add(&attn_out).add(&mlp_out))
}

/// Run one standard pre-norm decoder layer through the fused path.
///
/// Pattern: `x → input_norm → batched_gqa_attn → +residual → post_norm → mlp → +residual`.
/// Matches [`crate::decoder_layer::std_pre_norm_forward`] but with the
/// batched attention call and `FusedBatchKVCache` instead of the serial
/// pair. Each Tier-1 arch calls this once per layer inside
/// `forward_batched_impl`, so the arch-specific file stays tiny.
#[allow(clippy::too_many_arguments)]
pub fn batched_prenorm_layer<N: NormModule, M: MlpModule>(
    hidden: &Array,
    input_layernorm: &mut N,
    q_proj: &mut nn::Linear,
    k_proj: &mut nn::Linear,
    v_proj: &mut nn::Linear,
    o_proj: &mut nn::Linear,
    q_norm: Option<&mut nn::RmsNorm>,
    k_norm: Option<&mut nn::RmsNorm>,
    post_attention_layernorm: &mut N,
    mlp: &mut M,
    attn_cfg: &BatchedGqaAttnCfg,
    cache: &mut FusedBatchKVCache,
    active_indices: &[usize],
    layer_idx: usize,
) -> Result<Array, Exception> {
    let normed = input_layernorm.forward(hidden)?;
    let attn_out = batched_gqa_attn(
        &normed,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        attn_cfg,
        cache,
        active_indices,
        layer_idx,
    )?;
    let after_attn = hidden.add(&attn_out);
    let normed2 = post_attention_layernorm.forward(&after_attn)?;
    let mlp_out = mlp.forward(&normed2)?;
    Ok(after_attn.add(&mlp_out))
}
