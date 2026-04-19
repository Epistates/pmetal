//! Full-model forward passes: the canonical `forward_step`, the
//! activation-capturing variant for DFlash, and the tree-verify variant for
//! EAGLE-style speculative decoding.

use crate::InlineArray;

use super::attention::{TreeVerifyInputs, attn_forward, attn_forward_with_tree_ctx};
use super::cache::NativeCache;
use super::mlp_moe::{dense_mlp_forward, gdn_forward, moe_forward};
use super::weights::NativeWeights;

pub fn forward_step(
    weights: &NativeWeights,
    token_ids: &InlineArray, // [B, T]
    cache: &mut NativeCache,
) -> InlineArray {
    let b = token_ids.dim(0);
    let s = token_ids.dim(1);
    let dtype = weights.model_dtype;

    // Debug removed
    // Embedding lookup: [B, T, hidden]
    // For quantized models: index into weight/scales/biases rows, then dequantize.
    // Matches Python's QuantizedEmbedding: dequantize(weight[x], scales[x], biases[x])
    let mut hidden =
        if let (Some(scales), Some(biases)) = (&weights.embed_scales, &weights.embed_biases) {
            let qcfg = weights.quantization_config.as_ref();
            let gs = qcfg.map(|q| q.group_size).unwrap_or(64);
            let bits = qcfg.map(|q| q.bits).unwrap_or(4);
            let w_rows = weights.embed_w.take_axis(token_ids, 0); // [B, T, hidden/pack]
            let s_rows = scales.take_axis(token_ids, 0); // [B, T, hidden/group_size]
            let b_rows = biases.take_axis(token_ids, 0); // [B, T, hidden/group_size]
            w_rows.dequantize(&s_rows, &b_rows, gs, bits) // [B, T, hidden] bf16
        } else {
            weights.embed_w.take_axis(token_ids, 0)
        };
    let trace_qwen35 = std::env::var_os("PMETAL_TRACE_QWEN35").is_some();

    let mut gdn_slot = 0usize;
    let mut attn_slot = 0usize;

    for (layer_idx, lw) in weights.layers.iter().enumerate() {
        if trace_qwen35 {
            eprintln!(
                "[QWEN35 TRACE] layer={layer_idx} start linear={} moe={} rope_offset={} seq={s}",
                lw.is_linear, lw.is_moe_layer, cache.rope_offset
            );
        }
        // Input LayerNorm
        let normed = hidden.rms_norm(Some(&lw.input_ln_w), lw.input_ln_eps);

        let r = if lw.is_linear {
            let result = gdn_forward(lw, &normed, b, s, &mut cache.gdn_caches[gdn_slot], dtype);
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
                weights.qjl_matrix.as_ref(),
            );
            attn_slot += 1;
            result
        };
        if trace_qwen35 {
            eprintln!("[QWEN35 TRACE] layer={layer_idx} after_attention");
        }

        // Residual
        let h = hidden.add(&r);

        // Post-attention LayerNorm + MLP (dense or MoE)
        let mlp_in = h.rms_norm(Some(&lw.post_ln_w), lw.post_ln_eps);
        let mlp_out = if lw.is_moe_layer {
            moe_forward(lw, &mlp_in)
        } else {
            dense_mlp_forward(lw, &mlp_in)
        };
        if trace_qwen35 {
            eprintln!("[QWEN35 TRACE] layer={layer_idx} after_mlp");
        }

        // Residual
        hidden = h.add(&mlp_out);
    }

    // Advance position counter
    cache.rope_offset += s;

    // Final norm + LM head
    let hidden = hidden.rms_norm(Some(&weights.final_norm_w), weights.final_norm_eps);
    if weights.tie_word_embeddings {
        // For quantized models: use quantized_matmul with the packed embedding weight
        if let (Some(scales), Some(biases)) = (&weights.embed_scales, &weights.embed_biases) {
            let qcfg = weights.quantization_config.as_ref();
            let gs = qcfg.map(|q| q.group_size).unwrap_or(64);
            let bits = qcfg.map(|q| q.bits).unwrap_or(4);
            hidden.quantized_matmul(&weights.embed_w, scales, Some(biases), true, gs, bits)
        } else {
            hidden.matmul(&weights.embed_w.t())
        }
    } else {
        weights.lm_head_w.as_ref().unwrap().matmul_from(&hidden)
    }
}

/// Variant of [`forward_step`] that tees post-layer hidden states at the
/// indices listed in `tap_layers` into `captured`, in ascending-layer order.
///
/// The extra per-layer check is just a `contains` on a tiny vec and a
/// refcount bump (`InlineArray` is a thin handle around an `mx::array`), so
/// this path is within a few percent of the non-capture `forward_step`. It
/// is intentionally a separate entry point rather than a generic hook so the
/// hot path stays unbranched for non-DFlash callers.
///
/// `captured` is populated in the SAME ORDER as `tap_layers`. Callers that
/// want the concatenated target-hidden tensor for a DFlash draft should
/// concatenate along the last axis after this returns:
///
/// ```ignore
/// let refs: Vec<&InlineArray> = captured.iter().collect();
/// let target_hidden = ops::concatenate_axis(&refs, -1);
/// ```
pub fn forward_step_with_capture(
    weights: &NativeWeights,
    token_ids: &InlineArray,
    cache: &mut NativeCache,
    tap_layers: &[usize],
    captured: &mut Vec<InlineArray>,
) -> InlineArray {
    captured.clear();
    let b = token_ids.dim(0);
    let s = token_ids.dim(1);
    let dtype = weights.model_dtype;

    let mut hidden =
        if let (Some(scales), Some(biases)) = (&weights.embed_scales, &weights.embed_biases) {
            let qcfg = weights.quantization_config.as_ref();
            let gs = qcfg.map(|q| q.group_size).unwrap_or(64);
            let bits = qcfg.map(|q| q.bits).unwrap_or(4);
            let w_rows = weights.embed_w.take_axis(token_ids, 0);
            let s_rows = scales.take_axis(token_ids, 0);
            let b_rows = biases.take_axis(token_ids, 0);
            w_rows.dequantize(&s_rows, &b_rows, gs, bits)
        } else {
            weights.embed_w.take_axis(token_ids, 0)
        };

    let mut gdn_slot = 0usize;
    let mut attn_slot = 0usize;

    for (layer_idx, lw) in weights.layers.iter().enumerate() {
        let normed = hidden.rms_norm(Some(&lw.input_ln_w), lw.input_ln_eps);
        let r = if lw.is_linear {
            let result = gdn_forward(lw, &normed, b, s, &mut cache.gdn_caches[gdn_slot], dtype);
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
                weights.qjl_matrix.as_ref(),
            );
            attn_slot += 1;
            result
        };
        let h = hidden.add(&r);
        let mlp_in = h.rms_norm(Some(&lw.post_ln_w), lw.post_ln_eps);
        let mlp_out = if lw.is_moe_layer {
            moe_forward(lw, &mlp_in)
        } else {
            dense_mlp_forward(lw, &mlp_in)
        };
        hidden = h.add(&mlp_out);

        // Tap hidden state POST-residual, matching mlx-lm's
        // `hidden_states = layer(...)` + `if idx in target_layer_ids:
        // selected_hidden_states.append(hidden_states)` pattern in
        // `dflash_mlx/adapters.py`.
        if tap_layers.contains(&layer_idx) {
            captured.push(hidden.clone());
        }
    }

    cache.rope_offset += s;

    let hidden = hidden.rms_norm(Some(&weights.final_norm_w), weights.final_norm_eps);
    if weights.tie_word_embeddings {
        if let (Some(scales), Some(biases)) = (&weights.embed_scales, &weights.embed_biases) {
            let qcfg = weights.quantization_config.as_ref();
            let gs = qcfg.map(|q| q.group_size).unwrap_or(64);
            let bits = qcfg.map(|q| q.bits).unwrap_or(4);
            hidden.quantized_matmul(&weights.embed_w, scales, Some(biases), true, gs, bits)
        } else {
            hidden.matmul(&weights.embed_w.t())
        }
    } else {
        weights.lm_head_w.as_ref().unwrap().matmul_from(&hidden)
    }
}

/// Tree-verify variant of [`forward_step_with_capture`]. Each token
/// carries its own position id (from the DDTree tree compile) and the
/// target's attention is gated by a custom additive mask encoding the
/// tree visibility. All tapped layers are captured, same as the
/// linear variant, so the DFlash draft can condition its next round
/// on the accepted path's hidden states.
///
/// Only the plain-bf16 KV cache path is wired today. TurboQuant /
/// quant-config modes silently fall back to their existing causal
/// paths — the tree context is ignored for those (and the caller gets
/// a warning via the normal `cache.turboquant.is_some()` routing).
pub fn forward_step_tree_verify(
    weights: &NativeWeights,
    token_ids: &InlineArray,
    cache: &mut NativeCache,
    position_ids: &InlineArray,
    attention_mask: &InlineArray,
    tap_layers: &[usize],
    captured: &mut Vec<InlineArray>,
) -> InlineArray {
    captured.clear();
    let b = token_ids.dim(0);
    let s = token_ids.dim(1);
    let dtype = weights.model_dtype;
    // Compile_tree builds `position_ids` as [1, S] for consistency
    // with `token_ids`. Squeeze the batch axis to 1D — MLX's rope
    // array-offset API expects offset shape `[batch]` where batch is
    // the rope kernel's batch dim. After our [1,H,T,D]→[T,H,1,D]
    // transpose trick, the rope batch dim has T elements.
    let pos_ids_1d = if position_ids.ndim() == 2 && position_ids.dim(0) == 1 {
        position_ids.reshape(&[position_ids.dim(1)])
    } else {
        position_ids.clone()
    };
    let tree_ctx = TreeVerifyInputs {
        pos_ids: &pos_ids_1d,
        attention_mask,
    };

    let mut hidden =
        if let (Some(scales), Some(biases)) = (&weights.embed_scales, &weights.embed_biases) {
            let qcfg = weights.quantization_config.as_ref();
            let gs = qcfg.map(|q| q.group_size).unwrap_or(64);
            let bits = qcfg.map(|q| q.bits).unwrap_or(4);
            let w_rows = weights.embed_w.take_axis(token_ids, 0);
            let s_rows = scales.take_axis(token_ids, 0);
            let b_rows = biases.take_axis(token_ids, 0);
            w_rows.dequantize(&s_rows, &b_rows, gs, bits)
        } else {
            weights.embed_w.take_axis(token_ids, 0)
        };

    let mut gdn_slot = 0usize;
    let mut attn_slot = 0usize;

    for (layer_idx, lw) in weights.layers.iter().enumerate() {
        let normed = hidden.rms_norm(Some(&lw.input_ln_w), lw.input_ln_eps);
        let r = if lw.is_linear {
            // Linear-attn layers in Qwen3.5 do not yet support tree
            // verify; they process the sequence recurrently with no
            // meaningful "mask" analogue. Fall back to scalar offset.
            let result = gdn_forward(lw, &normed, b, s, &mut cache.gdn_caches[gdn_slot], dtype);
            gdn_slot += 1;
            result
        } else {
            let result = attn_forward_with_tree_ctx(
                lw,
                &normed,
                b,
                s,
                &mut cache.kv_caches[attn_slot],
                cache.rope_offset,
                dtype,
                weights.qjl_matrix.as_ref(),
                Some(tree_ctx),
            );
            attn_slot += 1;
            result
        };
        let h = hidden.add(&r);
        let mlp_in = h.rms_norm(Some(&lw.post_ln_w), lw.post_ln_eps);
        let mlp_out = if lw.is_moe_layer {
            moe_forward(lw, &mlp_in)
        } else {
            dense_mlp_forward(lw, &mlp_in)
        };
        hidden = h.add(&mlp_out);

        if tap_layers.contains(&layer_idx) {
            captured.push(hidden.clone());
        }
    }

    // Advance the global rope offset by the full tree length so that
    // `rollback_cache` (which subtracts from BOTH `rope_offset` and
    // each `kv.offset`) leaves the cache in a consistent state when
    // the caller wants to discard the tree write entirely (the
    // tree-pick + linear-commit path). Compact_tree_cache, used by
    // the tree-only commit path, also still works because it
    // unconditionally OVERWRITES `cache.rope_offset` to
    // `past_length + accepted_count`.
    cache.rope_offset += s;

    let hidden = hidden.rms_norm(Some(&weights.final_norm_w), weights.final_norm_eps);
    if weights.tie_word_embeddings {
        if let (Some(scales), Some(biases)) = (&weights.embed_scales, &weights.embed_biases) {
            let qcfg = weights.quantization_config.as_ref();
            let gs = qcfg.map(|q| q.group_size).unwrap_or(64);
            let bits = qcfg.map(|q| q.bits).unwrap_or(4);
            hidden.quantized_matmul(&weights.embed_w, scales, Some(biases), true, gs, bits)
        } else {
            hidden.matmul(&weights.embed_w.t())
        }
    } else {
        weights.lm_head_w.as_ref().unwrap().matmul_from(&hidden)
    }
}

/// Compact the target KV cache after a tree-verify round: the verify
/// forward wrote `tree_length` (full tree) keys/values into the cache
/// starting at `past_length`; this function gathers the accepted
/// indices (relative to the tree's local index space 0..tree_length)
/// and rewrites them contiguously at the start of the appended
/// window. Advances `cache.offset` by `accepted_indices.len()` on
/// each layer and sets `cache.rope_offset` accordingly.
///
/// Matches DDTree's `compact_dynamic_cache` but operates on the
/// native `KvLayerCache` buffers directly: we use `take_axis` for the
/// row gather and `slice_set` to write the compacted rows back.
pub fn compact_tree_cache(
    cache: &mut NativeCache,
    past_length: i32,
    tree_length: i32,
    accepted_indices: &[usize],
) {
    if accepted_indices.is_empty() {
        return;
    }
    let keep_len = accepted_indices.len() as i32;
    let kept_indices: Vec<i32> = accepted_indices
        .iter()
        .map(|&i| past_length + i as i32)
        .collect();
    let idx_arr = InlineArray::from_i32_slice(&kept_indices);

    for kv in cache.kv_caches.iter_mut() {
        let Some(k_buf) = kv.keys.take() else {
            continue;
        };
        let Some(v_buf) = kv.values.take() else {
            continue;
        };
        let shape = k_buf.shape().to_vec();
        let n_kv = shape[1];
        let head_dim = shape[3];

        // Gather the accepted rows from along the seq axis using
        // `take_axis(axis=2)`. Result is [B, Hkv, keep_len, D].
        let kept_k = k_buf.take_axis(&idx_arr, 2);
        let kept_v = v_buf.take_axis(&idx_arr, 2);

        // Write the gathered rows back starting at `past_length` via
        // `slice_set`. This leaves any rows past `past_length +
        // keep_len` in the buffer untouched — the rejected
        // verify-block keys/values. They'll get overwritten by the
        // next round's writes since `offset` is now lower.
        let start = [0, 0, past_length, 0];
        let stop = [1, n_kv, past_length + keep_len, head_dim];
        let new_k = k_buf.slice_set(&kept_k, &start, &stop);
        let new_v = v_buf.slice_set(&kept_v, &start, &stop);

        kv.keys = Some(new_k);
        kv.values = Some(new_v);
        kv.offset = past_length + keep_len;
    }
    cache.rope_offset = past_length + keep_len;
    // `tree_length` is the total number of positions the verify forward
    // wrote beyond `past_length`. `accepted_indices.len()` is how many
    // we committed. The unused slots (tree_length - keep_len) stay
    // allocated but are now beyond `offset` — effectively garbage
    // that the next write will clobber.
    let _ = tree_length;
}

/// Rollback the target KV state by `n` positions. Mirrors
/// [`pmetal_mlx::kv_cache::KVCache::rollback`] but operates on the native
/// cache — decrements `rope_offset` on the cache and on each KV layer,
/// leaving the underlying buffer intact. Subsequent writes refill the
/// rejected slots.
///
/// Caller is responsible for ensuring the cache is in a quiescent state
/// (no pending ops referencing the rolled-back slots).
pub fn rollback_cache(cache: &mut NativeCache, n: i32) {
    if n <= 0 {
        return;
    }
    cache.rope_offset = cache.rope_offset.saturating_sub(n);
    for kv in cache.kv_caches.iter_mut() {
        kv.offset = kv.offset.saturating_sub(n);
    }
}
