//! Attention forward: per-position RoPE (used for tree-verify),
//! the `TreeVerifyInputs` context struct, the main `attn_forward`, and the
//! tree-context variant that accepts an arbitrary position vector.

use crate::InlineArray;

use super::cache::{KvLayerCache, QuantizedTuple};
use super::mrope::MropeTables;
use super::weights::{LayerWeight, LayerWeights};

// ============================================================================
// Mixed-bit quantization helpers
// ============================================================================

/// Quantize one half of a `[B, H_kv, S, channels]` K/V slice through MLX's
/// `quantize_weights` and reshape the (packed, scales, biases) triple back
/// to 4-D.
///
/// Folds the flatten → quantize → reshape ritual that previously repeated
/// once per (key/value × hi/lo) pair in the mixed-bit cache append path.
/// `channels` is the slice's last-dim size (`outlier_count` for the hi
/// half, `head_dim - outlier_count` for the lo half).
fn quantize_kv_slice(
    src: &InlineArray,
    b: i32,
    n_kv_heads: i32,
    s: i32,
    channels: i32,
    group_size: i32,
    bits: i32,
) -> (InlineArray, InlineArray, InlineArray) {
    let flat = src.reshape(&[b * n_kv_heads * s, channels]);
    let (packed, scales, biases) = flat.quantize_weights(group_size, bits);
    let packed_dim = (channels * bits + 31) / 32;
    let scales_dim = channels / group_size;
    (
        packed.reshape(&[b, n_kv_heads, s, packed_dim]),
        scales.reshape(&[b, n_kv_heads, s, scales_dim]),
        biases.reshape(&[b, n_kv_heads, s, scales_dim]),
    )
}

// ============================================================================
// Attention layer forward
// ============================================================================

#[allow(clippy::too_many_arguments)]
/// Optional inputs for tree-verify attention: per-token absolute
/// position ids and a custom additive attention mask.
///
/// Routes the per-token RoPE through MLX's fused `fast::rope` (the
/// SAME kernel linear DFlash uses) via a reshape trick: the input
/// `[B, H, T, D]` tensor is transposed to `[T, H, 1, D]` so the
/// `T`-token sequence becomes the rope's batch axis. MLX's array-
/// offset overload accepts one offset per batch element, so passing
/// `pos_ids` of shape `[T]` rotates each token at its own absolute
/// position with the EXACT same numerics linear DFlash gets. No
/// hand-rolled cos/sin → no bf16 ULP drift compounding across
/// 100+ decode rounds.
#[derive(Clone, Copy)]
pub(crate) struct TreeVerifyInputs<'a> {
    /// `[T]` int32 — per-token absolute positions (round offset +
    /// tree depth). Used as the array-offset for MLX rope.
    pub pos_ids: &'a InlineArray,
    /// `[1, 1, seq_len, past_length + seq_len]` additive mask in the
    /// current dtype. `0` at visible positions, `-inf` elsewhere.
    pub attention_mask: &'a InlineArray,
}

/// Apply per-token RoPE to a `[1, H, T, D]` tensor by routing through
/// MLX's fused `fast::rope` array-offset overload.
///
/// The trick: MLX's `rope(x, array offset)` accepts one offset per
/// BATCH element. We transpose `[1, H, T, D]` → `[T, H, 1, D]` so the
/// `T`-token sequence becomes the rope's batch axis with one token
/// per batch slot. The `pos_ids` array (shape `[T]`) then provides
/// the per-token absolute positions. After the rope call, transpose
/// back to `[1, H, T, D]`.
///
/// This guarantees numerical equivalence with linear DFlash's RoPE
/// path because both use the SAME MLX kernel — no hand-rolled
/// cos/sin, no separate ops with intermediate fp32 stores. Each
/// position rotation is computed in fused fp32 registers exactly as
/// MLX does for sequential positions.
///
/// `rotated_dims` must equal the head's rotary width (== head_dim
/// for Qwen3 full RoPE).
fn apply_per_position_rope(
    x: &InlineArray,
    pos_ids: &InlineArray,
    rotated_dims: i32,
    base: f32,
    scale: f32,
) -> InlineArray {
    // x shape: [B=1, H, T, D]. Transpose to [T, H, 1, D] so the rope
    // batch axis carries one token per row. The fast::rope kernel
    // sees a "batch" of T elements, each with H heads × 1 token × D
    // dim, and applies offset[batch_idx] + 0 = pos_ids[batch_idx] to
    // each one. Bit-exact with the linear path because both go
    // through the SAME MLX kernel.
    let xt = x.transpose_axes(&[2, 1, 0, 3]);
    let rotated = xt.rope_with_pos_ids(rotated_dims, false, base, scale, pos_ids);
    rotated.transpose_axes(&[2, 1, 0, 3])
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attn_forward(
    lw: &LayerWeights,
    normed: &InlineArray,
    b: i32,
    s: i32,
    cache: &mut KvLayerCache,
    rope_offset: i32,
    dtype: i32,
    qjl_matrix: Option<&InlineArray>,
) -> InlineArray {
    attn_forward_with_tree_ctx(
        lw,
        normed,
        b,
        s,
        cache,
        rope_offset,
        dtype,
        qjl_matrix,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attn_forward_with_tree_ctx(
    lw: &LayerWeights,
    normed: &InlineArray,
    b: i32,
    s: i32,
    cache: &mut KvLayerCache,
    rope_offset: i32,
    dtype: i32,
    qjl_matrix: Option<&InlineArray>,
    tree_ctx: Option<TreeVerifyInputs>,
    mrope: Option<&MropeTables>,
) -> InlineArray {
    let n_heads = lw.attn_n_heads;
    let n_kv_heads = lw.attn_n_kv_heads;
    let head_dim = lw.attn_head_dim;
    let scale = lw.attn_scale;
    let prev = cache.offset;
    let next = prev + s;
    crate::native_common::kv_cache::alloc_or_grow_kv(
        crate::native_common::kv_cache::GrowthPolicy::AmortizedChunked,
        &mut cache.keys,
        &mut cache.values,
        b,
        n_kv_heads,
        next,
        head_dim,
        dtype,
    );

    // The compiled layer rotates by `(base, scale)`, which a scaled rotary
    // embedding (YaRN) has no equivalent for.
    if s == 1
        && mrope.is_none()
        && lw.attn_scaled_rope.is_none()
        && cache.turboquant.is_none()
        && cache.quant_config.is_none()
    {
        if let (
            Some(LayerWeight::Dense(q_w)),
            Some(LayerWeight::Dense(k_w)),
            Some(LayerWeight::Dense(v_w)),
            Some(LayerWeight::Dense(o_w)),
        ) = (&lw.attn_q_w, &lw.attn_k_w, &lw.attn_v_w, &lw.attn_o_w)
        {
            let cache_keys = cache.keys.take().unwrap();
            let cache_vals = cache.values.take().unwrap();
            let (output, new_cache_keys, new_cache_vals) = InlineArray::compiled_attn_layer_fixed(
                normed,
                q_w,
                k_w,
                v_w,
                o_w,
                lw.attn_q_norm_w.as_ref().unwrap(),
                lw.attn_k_norm_w.as_ref().unwrap(),
                &cache_keys,
                &cache_vals,
                prev,
                rope_offset,
                n_heads,
                n_kv_heads,
                head_dim,
                scale,
                lw.attn_rope_dims,
                lw.attn_rope_base,
                lw.attn_rope_scale,
                lw.attn_q_norm_eps,
                lw.attn_k_norm_eps,
                lw.attn_gated,
            );
            cache.keys = Some(new_cache_keys);
            cache.values = Some(new_cache_vals);
            cache.offset = next;
            return output;
        }
    }

    // Q projection.
    //
    // Qwen3.5 (gated): q_proj output width = n_heads * head_dim * 2.
    //   Reshape to [B,S,H,D*2], split at D → [queries [B,S,H,D], gate [B,S,H,D]].
    //   gate is later reshaped to [B,S,H*D] and used to sigmoid-scale the output.
    //
    // Qwen3 (standard): q_proj output width = n_heads * head_dim.
    //   Reshape to [B,S,H,D] directly, no gate split.
    let q_proj_out = lw.attn_q_w.as_ref().unwrap().matmul_from(normed);

    let (queries, gate_opt) = if lw.attn_gated {
        // Gated path (Qwen3.5)
        let q_gate = q_proj_out.reshape(&[b, s, n_heads, head_dim * 2]);
        let mut qg_parts = q_gate.split(&[head_dim], -1);
        let gate = qg_parts.pop().unwrap().reshape(&[b, s, n_heads * head_dim]);
        let queries = qg_parts.pop().unwrap(); // [B, S, n_heads, head_dim]
        (queries, Some(gate))
    } else {
        // Standard path (Qwen3)
        let queries = q_proj_out.reshape(&[b, s, n_heads, head_dim]);
        (queries, None)
    };

    // K, V projections
    let new_keys = lw.attn_k_w.as_ref().unwrap().matmul_from(normed);
    let new_values = lw.attn_v_w.as_ref().unwrap().matmul_from(normed);

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

    // RoPE: default path uses `fast::rope` with a scalar offset.
    // Tree-verify mode threads each token's absolute position
    // through MLX's array-offset overload via the [1,H,T,D]→[T,H,1,D]
    // transpose trick. Both paths use the SAME MLX kernel — no
    // hand-rolled rope, no bf16 ULP drift between the two modes, so
    // tree DFlash output stays bit-exact with linear DFlash
    // (and therefore with greedy decode at temperature=0).
    let (queries, keys) = if let Some(tables) = mrope {
        // A prompt with media: three positions per token (see `mrope`).
        (tables.apply(&queries), tables.apply(&keys))
    } else if let Some(scaled) = &lw.attn_scaled_rope {
        // Scaled RoPE (YaRN): its frequencies and attention factor.
        match tree_ctx {
            Some(ctx) => (
                scaled.apply_at(&queries, ctx.pos_ids),
                scaled.apply_at(&keys, ctx.pos_ids),
            ),
            None => (
                scaled.apply(&queries, rope_offset),
                scaled.apply(&keys, rope_offset),
            ),
        }
    } else if let Some(ctx) = tree_ctx {
        (
            apply_per_position_rope(
                &queries,
                ctx.pos_ids,
                lw.attn_rope_dims,
                lw.attn_rope_base,
                lw.attn_rope_scale,
            ),
            apply_per_position_rope(
                &keys,
                ctx.pos_ids,
                lw.attn_rope_dims,
                lw.attn_rope_base,
                lw.attn_rope_scale,
            ),
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

    // KV cache update + SDPA
    let output = if let Some(ref mut tq_cache) = cache.turboquant {
        let out = crate::turboquant_dispatch::turboquant_attention_step(
            tq_cache, &queries, &keys, &values, scale, prev, "QWEN",
        )
        .expect("Qwen TurboQuant attention step failed");
        cache.offset = next;
        out
    } else if let Some(qcfg) = cache.quant_config {
        let group_size = qcfg.group_size;

        if let Some(mb) = qcfg.mixed_bit {
            // ---- MIXED-BIT PATH (TurboQuant v2: Q2.5 / Q3.5) ----
            // After outlier permutation, the first `oc` dims of each head are
            // outliers (quantized at higher bits); the remaining `rc` are regular
            // (quantized at lower bits).
            let oc = mb.outlier_count; // outlier channel count per head
            let rc = head_dim - oc; // regular channel count per head
            let bits_hi = mb.outlier_bits as i32;
            let bits_lo = mb.regular_bits as i32;

            // MLX packed-uint32 dims for each half
            let packed_dim_hi = (oc * bits_hi + 31) / 32;
            let packed_dim_lo = (rc * bits_lo + 31) / 32;
            let scales_dim_hi = oc / group_size;
            let scales_dim_lo = rc / group_size;

            // Split K/V along the head-dim axis: [B, Hkv, S, oc] and [B, Hkv, S, rc]
            let k_hi = keys.slice(&[0, 0, 0, 0], &[b, n_kv_heads, s, oc]);
            let k_lo = keys.slice(&[0, 0, 0, oc], &[b, n_kv_heads, s, head_dim]);
            let v_hi = values.slice(&[0, 0, 0, 0], &[b, n_kv_heads, s, oc]);
            let v_lo = values.slice(&[0, 0, 0, oc], &[b, n_kv_heads, s, head_dim]);

            // Quantize each half → (packed, scales, biases). The helper folds
            // the flatten/quantize/reshape ritual into one call per leg.
            let (kp_hi, ks_hi, kb_hi) =
                quantize_kv_slice(&k_hi, b, n_kv_heads, s, oc, group_size, bits_hi);
            let (kp_lo, ks_lo, kb_lo) =
                quantize_kv_slice(&k_lo, b, n_kv_heads, s, rc, group_size, bits_lo);
            let (vp_hi, vs_hi, vb_hi) =
                quantize_kv_slice(&v_hi, b, n_kv_heads, s, oc, group_size, bits_hi);
            let (vp_lo, vs_lo, vb_lo) =
                quantize_kv_slice(&v_lo, b, n_kv_heads, s, rc, group_size, bits_lo);

            // ---- Cache management: allocate or grow 4 quantized buffers ----
            let uint32_dt = crate::compat::Dtype::Uint32.as_i32();
            QuantizedTuple::ensure_capacity(
                &mut cache.quantized_keys_hi,
                crate::native_common::kv_cache::GrowthPolicy::AmortizedChunked,
                b,
                n_kv_heads,
                next,
                packed_dim_hi,
                scales_dim_hi,
                uint32_dt,
                dtype,
            );
            QuantizedTuple::ensure_capacity(
                &mut cache.quantized_keys,
                crate::native_common::kv_cache::GrowthPolicy::AmortizedChunked,
                b,
                n_kv_heads,
                next,
                packed_dim_lo,
                scales_dim_lo,
                uint32_dt,
                dtype,
            );
            QuantizedTuple::ensure_capacity(
                &mut cache.quantized_values_hi,
                crate::native_common::kv_cache::GrowthPolicy::AmortizedChunked,
                b,
                n_kv_heads,
                next,
                packed_dim_hi,
                scales_dim_hi,
                uint32_dt,
                dtype,
            );
            QuantizedTuple::ensure_capacity(
                &mut cache.quantized_values,
                crate::native_common::kv_cache::GrowthPolicy::AmortizedChunked,
                b,
                n_kv_heads,
                next,
                packed_dim_lo,
                scales_dim_lo,
                uint32_dt,
                dtype,
            );

            // slice_set new tokens into all four cache buffers
            let start_q = [0, 0, prev, 0];

            let qkh_ref = cache.quantized_keys_hi.as_mut().unwrap();
            qkh_ref.packed =
                qkh_ref
                    .packed
                    .slice_set(&kp_hi, &start_q, &[b, n_kv_heads, next, packed_dim_hi]);
            qkh_ref.scales =
                qkh_ref
                    .scales
                    .slice_set(&ks_hi, &start_q, &[b, n_kv_heads, next, scales_dim_hi]);
            qkh_ref.biases =
                qkh_ref
                    .biases
                    .slice_set(&kb_hi, &start_q, &[b, n_kv_heads, next, scales_dim_hi]);

            let qkl_ref = cache.quantized_keys.as_mut().unwrap();
            qkl_ref.packed =
                qkl_ref
                    .packed
                    .slice_set(&kp_lo, &start_q, &[b, n_kv_heads, next, packed_dim_lo]);
            qkl_ref.scales =
                qkl_ref
                    .scales
                    .slice_set(&ks_lo, &start_q, &[b, n_kv_heads, next, scales_dim_lo]);
            qkl_ref.biases =
                qkl_ref
                    .biases
                    .slice_set(&kb_lo, &start_q, &[b, n_kv_heads, next, scales_dim_lo]);

            let qvh_ref = cache.quantized_values_hi.as_mut().unwrap();
            qvh_ref.packed =
                qvh_ref
                    .packed
                    .slice_set(&vp_hi, &start_q, &[b, n_kv_heads, next, packed_dim_hi]);
            qvh_ref.scales =
                qvh_ref
                    .scales
                    .slice_set(&vs_hi, &start_q, &[b, n_kv_heads, next, scales_dim_hi]);
            qvh_ref.biases =
                qvh_ref
                    .biases
                    .slice_set(&vb_hi, &start_q, &[b, n_kv_heads, next, scales_dim_hi]);

            let qvl_ref = cache.quantized_values.as_mut().unwrap();
            qvl_ref.packed =
                qvl_ref
                    .packed
                    .slice_set(&vp_lo, &start_q, &[b, n_kv_heads, next, packed_dim_lo]);
            qvl_ref.scales =
                qvl_ref
                    .scales
                    .slice_set(&vs_lo, &start_q, &[b, n_kv_heads, next, scales_dim_lo]);
            qvl_ref.biases =
                qvl_ref
                    .biases
                    .slice_set(&vb_lo, &start_q, &[b, n_kv_heads, next, scales_dim_lo]);

            cache.offset = next;

            // Slice valid portions from all four cache buffers
            let qkh = cache.quantized_keys_hi.as_ref().unwrap();
            let cached_kp_hi = qkh
                .packed
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, packed_dim_hi]);
            let cached_ks_hi = qkh
                .scales
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim_hi]);
            let cached_kb_hi = qkh
                .biases
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim_hi]);

            let qkl = cache.quantized_keys.as_ref().unwrap();
            let cached_kp_lo = qkl
                .packed
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, packed_dim_lo]);
            let cached_ks_lo = qkl
                .scales
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim_lo]);
            let cached_kb_lo = qkl
                .biases
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim_lo]);

            let qvh = cache.quantized_values_hi.as_ref().unwrap();
            let cached_vp_hi = qvh
                .packed
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, packed_dim_hi]);
            let cached_vs_hi = qvh
                .scales
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim_hi]);
            let cached_vb_hi = qvh
                .biases
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim_hi]);

            let qvl = cache.quantized_values.as_ref().unwrap();
            let cached_vp_lo = qvl
                .packed
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, packed_dim_lo]);
            let cached_vs_lo = qvl
                .scales
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim_lo]);
            let cached_vb_lo = qvl
                .biases
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim_lo]);

            // Mixed-bit SDPA: two quantized_matmul calls per score/value aggregation
            crate::decode::quantized_sdpa_mixed(
                &queries,
                (&cached_kp_lo, &cached_ks_lo, &cached_kb_lo),
                (&cached_vp_lo, &cached_vs_lo, &cached_vb_lo),
                (&cached_kp_hi, &cached_ks_hi, &cached_kb_hi),
                (&cached_vp_hi, &cached_vs_hi, &cached_vb_hi),
                scale,
                s,
                n_heads,
                n_kv_heads,
                oc,
                group_size,
                bits_lo,
                bits_hi,
            )
        } else {
            // ---- UNIFORM-BIT PATH (unchanged) ----
            // Zero-overhead quantized KV cache path using quantized_matmul.
            // Same layout as the reference QuantizedKVCache: quantize K/V immediately after RoPE,
            // store as (packed_uint32, scales, biases), pass to quantized_matmul
            // which dequantizes inside the Metal kernel. No separate dequant pass.
            let bits = qcfg.bits as i32;

            // MLX packs quantized values into uint32: packed_dim = ceil(head_dim * bits / 32).
            // This is NOT head_dim / (32/bits) which fails for non-power-of-2 bit widths (Q3, Q5, Q6).
            let packed_dim = (head_dim * bits + 31) / 32;
            let scales_dim = head_dim / group_size;

            // Quantize new K/V → (packed, scales, biases)
            let keys_2d = keys.reshape(&[b * n_kv_heads * s, head_dim]);
            let (kp, ks, kb) = keys_2d.quantize_weights(group_size, bits);
            let kp = kp.reshape(&[b, n_kv_heads, s, packed_dim]);
            let ks = ks.reshape(&[b, n_kv_heads, s, scales_dim]);
            let kb = kb.reshape(&[b, n_kv_heads, s, scales_dim]);

            // QJL residual computation (keys only, Q2-Q3, uniform path).
            //
            // After quantizing keys, reconstruct the approximate key, compute the
            // residual (original - reconstructed), and store:
            //   qjl_signs      = sign(S · residual)  [B, Hkv, s, D] dtype ±1.0
            //   residual_norms = ||residual||₂        [B, Hkv, s, 1] f32
            //
            // These are later used in quantized_sdpa_with_qjl to add an unbiased
            // correction to attention scores: E[⟨q, k̃⟩] = ⟨q, k⟩.
            let qjl_active = qcfg.qjl && bits <= 3 && qjl_matrix.is_some();
            let (new_qjl_signs, new_qjl_norms) = if qjl_active {
                let s_mat = qjl_matrix.unwrap();
                // Dequantize to get the affine reconstruction.
                // kp/ks/kb are [B,Hkv,s,*] — reshape back to 2D for dequantize.
                let kp_flat = kp.reshape(&[b * n_kv_heads * s, packed_dim]);
                let ks_flat = ks.reshape(&[b * n_kv_heads * s, scales_dim]);
                let kb_flat = kb.reshape(&[b * n_kv_heads * s, scales_dim]);
                let k_recon_2d = kp_flat.dequantize(&ks_flat, &kb_flat, group_size, bits);
                // Residual: original keys (2D) minus affine reconstruction.
                let residual = keys_2d.subtract(&k_recon_2d); // [N, D]
                // Per-row L2 norm: [N, 1]
                let norms_2d = residual.square().sum_axis(-1, true).sqrt(); // [N, 1]
                // Project residual through S: [N, D] @ [D, D]^T = [N, D]
                // S is [D, D], so S^T = S.transpose_axes([1, 0])
                let s_t = s_mat.transpose_axes(&[1, 0]);
                let projected = residual.matmul(&s_t); // [N, D]
                // sign: positive → 1.0, negative → -1.0, zero → 0.0
                let signs_2d = projected.sign(); // [N, D] dtype (same as keys)
                // Reshape back to [B, Hkv, s, D] and [B, Hkv, s, 1]
                let signs = signs_2d.reshape(&[b, n_kv_heads, s, head_dim]);
                let norms = norms_2d.reshape(&[b, n_kv_heads, s, 1]);
                // Cast norms to f32 for numerical stability in correction.
                let norms_f32 = norms.as_dtype(crate::compat::Dtype::Float32.as_i32());
                (Some(signs), Some(norms_f32))
            } else {
                (None, None)
            };

            let values_2d = values.reshape(&[b * n_kv_heads * s, head_dim]);
            let (vp, vs, vb) = values_2d.quantize_weights(group_size, bits);
            let vp = vp.reshape(&[b, n_kv_heads, s, packed_dim]);
            let vs = vs.reshape(&[b, n_kv_heads, s, scales_dim]);
            let vb = vb.reshape(&[b, n_kv_heads, s, scales_dim]);

            // Cache management: allocate or grow quantized + QJL buffers
            let uint32_dt = crate::compat::Dtype::Uint32.as_i32();
            let f32_dt = crate::compat::Dtype::Float32.as_i32();
            QuantizedTuple::ensure_capacity(
                &mut cache.quantized_keys,
                crate::native_common::kv_cache::GrowthPolicy::AmortizedChunked,
                b,
                n_kv_heads,
                next,
                packed_dim,
                scales_dim,
                uint32_dt,
                dtype,
            );
            QuantizedTuple::ensure_capacity(
                &mut cache.quantized_values,
                crate::native_common::kv_cache::GrowthPolicy::AmortizedChunked,
                b,
                n_kv_heads,
                next,
                packed_dim,
                scales_dim,
                uint32_dt,
                dtype,
            );
            if qjl_active {
                crate::native_common::kv_cache::alloc_or_grow_buffer(
                    crate::native_common::kv_cache::GrowthPolicy::AmortizedChunked,
                    &mut cache.qjl_signs,
                    next,
                    2,
                    dtype,
                    |cap| [b, n_kv_heads, cap, head_dim],
                );
                crate::native_common::kv_cache::alloc_or_grow_buffer(
                    crate::native_common::kv_cache::GrowthPolicy::AmortizedChunked,
                    &mut cache.qjl_residual_norms,
                    next,
                    2,
                    f32_dt,
                    |cap| [b, n_kv_heads, cap, 1],
                );
            }

            // slice_set quantized data into cache
            let start_q = [0, 0, prev, 0];
            let qk_ref = cache.quantized_keys.as_mut().unwrap();
            let stop_kp = [b, n_kv_heads, next, packed_dim];
            let stop_ks = [b, n_kv_heads, next, scales_dim];
            qk_ref.packed = qk_ref.packed.slice_set(&kp, &start_q, &stop_kp);
            qk_ref.scales = qk_ref.scales.slice_set(&ks, &start_q, &stop_ks);
            qk_ref.biases = qk_ref.biases.slice_set(&kb, &start_q, &stop_ks);

            let qv_ref = cache.quantized_values.as_mut().unwrap();
            qv_ref.packed = qv_ref.packed.slice_set(&vp, &start_q, &stop_kp);
            qv_ref.scales = qv_ref.scales.slice_set(&vs, &start_q, &stop_ks);
            qv_ref.biases = qv_ref.biases.slice_set(&vb, &start_q, &stop_ks);

            // slice_set QJL data into cache
            if let (Some(signs), Some(norms)) = (new_qjl_signs, new_qjl_norms) {
                let stop_signs = [b, n_kv_heads, next, head_dim];
                let stop_norms = [b, n_kv_heads, next, 1];
                if let Some(ref mut qs) = cache.qjl_signs {
                    *qs = qs.slice_set(&signs, &start_q, &stop_signs);
                }
                if let Some(ref mut qn) = cache.qjl_residual_norms {
                    *qn = qn.slice_set(&norms, &start_q, &stop_norms);
                }
            }
            cache.offset = next;

            // Slice valid portion
            let qk = cache.quantized_keys.as_ref().unwrap();
            let cached_kp = qk
                .packed
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, packed_dim]);
            let cached_ks = qk
                .scales
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim]);
            let cached_kb = qk
                .biases
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim]);
            let qv = cache.quantized_values.as_ref().unwrap();
            let cached_vp = qv
                .packed
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, packed_dim]);
            let cached_vs = qv
                .scales
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim]);
            let cached_vb = qv
                .biases
                .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, scales_dim]);

            // SDPA — with optional QJL correction when enabled
            if qjl_active {
                // Slice valid QJL data and project queries through S^T for correction.
                let cached_signs = cache
                    .qjl_signs
                    .as_ref()
                    .unwrap()
                    .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, head_dim]);
                let cached_norms = cache
                    .qjl_residual_norms
                    .as_ref()
                    .unwrap()
                    .slice(&[0, 0, 0, 0], &[b, n_kv_heads, next, 1]);
                // Project queries through S^T: [B, Hq, L, D] @ [D, D] = [B, Hq, L, D]
                let s_mat = qjl_matrix.unwrap();
                crate::decode::quantized_sdpa_with_qjl(
                    &queries,
                    (&cached_kp, &cached_ks, &cached_kb),
                    (&cached_vp, &cached_vs, &cached_vb),
                    &cached_signs,
                    &cached_norms,
                    s_mat,
                    scale,
                    s,
                    n_heads,
                    n_kv_heads,
                    group_size,
                    bits,
                )
            } else {
                // Quantized SDPA — zero overhead, dequant fused into Metal kernel
                crate::decode::quantized_sdpa(
                    &queries,
                    (&cached_kp, &cached_ks, &cached_kb),
                    (&cached_vp, &cached_vs, &cached_vb),
                    scale,
                    s,
                    n_heads,
                    n_kv_heads,
                    group_size,
                    bits,
                )
            }
        }
    } else {
        // Standard bf16 path
        let start = [0, 0, prev, 0];
        let stop = [b, n_kv_heads, next, head_dim];
        let k_buf = cache.keys.take().unwrap();
        let v_buf = cache.values.take().unwrap();
        cache.keys = Some(k_buf.slice_set(&keys, &start, &stop));
        cache.values = Some(v_buf.slice_set(&values, &start, &stop));
        cache.offset = next;

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
        if let Some(ctx) = tree_ctx {
            // Tree verify: use the caller-supplied additive mask that
            // encodes tree visibility instead of "causal" semantics.
            queries.sdpa_with_mask(&valid_keys, &valid_values, scale, Some(ctx.attention_mask))
        } else {
            crate::decode::sdpa_causal_like_mlx(&queries, &valid_keys, &valid_values, scale, s)
        }
    };

    // Output projection
    let output = output
        .transpose_axes(&[0, 2, 1, 3])
        .reshape(&[b, s, n_heads * head_dim]);

    let o_proj = lw.attn_o_w.as_ref().unwrap();
    if let Some(gate) = gate_opt {
        // Qwen3.5 gated output: o_proj(attn_out * sigmoid(gate))
        let gated = output.multiply(&gate.sigmoid());
        o_proj.matmul_from(&gated)
    } else {
        // Qwen3 standard output: o_proj(attn_out)
        o_proj.matmul_from(&output)
    }
}
