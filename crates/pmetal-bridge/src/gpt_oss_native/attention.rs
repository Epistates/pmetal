//! Attention forward step: full RoPE (head_dim = 64), three cache paths —
//! sliding rotating window, full bf16, full zero-overhead-quantized.

use crate::InlineArray;

use super::cache::KvLayerCache;
use super::weights::LayerWeights;

pub(super) fn attn_forward(
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

    // Fused-decode fast path: T=1 on a full-attention bf16 layer with all
    // four biases present. Mirrors the qwen3_native pattern — alloc/grow
    // first so the compiled kernel sees a writable cache buffer, then
    // call `compiled_gptoss_attn_layer_fixed` which fuses Q/K/V proj +
    // bias adds + RoPE (YaRN's period table and attention factor) + cache
    // write + SDPA + o_proj + bias into one mx.compile graph. Sliding-window
    // layers, turboquant cache, and zero-overhead-quantized cache stay on
    // the per-op paths below.
    let rope_kernel = lw
        .attn_rotary
        .fixed_kernel(rope_offset as i64 + s as i64)
        .filter(|_| crate::decode::compiled_decode_enabled());
    if s == 1
        && !lw.attn_is_sliding
        && cache.turboquant.is_none()
        && cache.quant_config.is_none()
        && let Some(rope_kernel) = rope_kernel
    {
        if let (Some(qb), Some(kb), Some(vb), Some(ob)) = (
            lw.attn_q_b.as_ref(),
            lw.attn_k_b.as_ref(),
            lw.attn_v_b.as_ref(),
            lw.attn_o_b.as_ref(),
        ) {
            let next = cache.offset + s;
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
            let cache_keys = cache.keys.take().unwrap();
            let cache_vals = cache.values.take().unwrap();
            let (output, new_cache_keys, new_cache_vals) =
                InlineArray::compiled_gptoss_attn_layer_fixed(
                    normed,
                    &lw.attn_q_w,
                    &lw.attn_k_w,
                    &lw.attn_v_w,
                    &lw.attn_o_w,
                    qb,
                    kb,
                    vb,
                    ob,
                    &cache_keys,
                    &cache_vals,
                    cache.offset,
                    rope_offset,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    scale,
                    rope_kernel,
                );
            cache.keys = Some(new_cache_keys);
            cache.values = Some(new_cache_vals);
            cache.offset = next;
            return output;
        }
    }

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

    // Full RoPE (head_dim = 64, no partial rotation): YaRN's frequencies and
    // attention factor, split-half.
    let q = lw.attn_rotary.apply(&q, rope_offset);
    let k = lw.attn_rotary.apply(&k, rope_offset);

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
        let output = crate::decode::sdpa_causal_like_mlx(&q, &valid_keys, &valid_values, scale, s);
        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[b, s, n_heads * head_dim]);

        // Output projection + bias
        let mut proj = output.matmul(&lw.attn_o_w);
        if let Some(ref ob) = lw.attn_o_b {
            proj = proj.add(ob);
        }
        proj
    } else if let Some(ref mut tq_cache) = cache.turboquant {
        // ── Full attention, TurboQuant compressed KV cache path ────────────
        // Sliding layers can't take this branch (turboquant is None there).
        let out = crate::turboquant_dispatch::turboquant_attention_step(
            tq_cache, &q, &k, &v, scale, prev, "GPT_OSS",
        )
        .expect("GPT-OSS TurboQuant attention step failed");
        cache.offset = next;
        let output = out
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[b, s, n_heads * head_dim]);
        let mut proj = output.matmul(&lw.attn_o_w);
        if let Some(ref ob) = lw.attn_o_b {
            proj = proj.add(ob);
        }
        proj
    } else if let Some(qcfg) = cache.quant_config {
        // ── Full attention, zero-overhead quantized KV cache path ──────────
        // quant_config is None on sliding layers (set in new_with_quant), so
        // this branch is only reached for full-attention layers.
        let bits = qcfg.bits as i32;
        let group_size = qcfg.group_size;
        let packed_dim = (head_dim * bits + 31) / 32;
        let scales_dim = head_dim / group_size;
        let uint32_dt = crate::compat::Dtype::Uint32.as_i32();

        // Quantize new K/V
        let k_2d = k.reshape(&[b * n_kv_heads * num_new, head_dim]);
        let (kp, ks, kb) = k_2d.quantize_weights(group_size, bits);
        let kp = kp.reshape(&[b, n_kv_heads, num_new, packed_dim]);
        let ks = ks.reshape(&[b, n_kv_heads, num_new, scales_dim]);
        let kb = kb.reshape(&[b, n_kv_heads, num_new, scales_dim]);

        let v_2d = v.reshape(&[b * n_kv_heads * num_new, head_dim]);
        let (vp, vs, vb) = v_2d.quantize_weights(group_size, bits);
        let vp = vp.reshape(&[b, n_kv_heads, num_new, packed_dim]);
        let vs = vs.reshape(&[b, n_kv_heads, num_new, scales_dim]);
        let vb = vb.reshape(&[b, n_kv_heads, num_new, scales_dim]);

        crate::qwen3_native::QuantizedTuple::ensure_capacity(
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
        crate::qwen3_native::QuantizedTuple::ensure_capacity(
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

        // slice_set new tokens
        let start_q = [0, 0, prev, 0];
        let stop_kp = [b, n_kv_heads, next, packed_dim];
        let stop_ks = [b, n_kv_heads, next, scales_dim];
        let qk_ref = cache.quantized_keys.as_mut().unwrap();
        qk_ref.packed = qk_ref.packed.slice_set(&kp, &start_q, &stop_kp);
        qk_ref.scales = qk_ref.scales.slice_set(&ks, &start_q, &stop_ks);
        qk_ref.biases = qk_ref.biases.slice_set(&kb, &start_q, &stop_ks);

        let qv_ref = cache.quantized_values.as_mut().unwrap();
        qv_ref.packed = qv_ref.packed.slice_set(&vp, &start_q, &stop_kp);
        qv_ref.scales = qv_ref.scales.slice_set(&vs, &start_q, &stop_ks);
        qv_ref.biases = qv_ref.biases.slice_set(&vb, &start_q, &stop_ks);

        cache.offset = next;

        // Slice valid portions
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

        let output = crate::decode::quantized_sdpa(
            &q,
            (&cached_kp, &cached_ks, &cached_kb),
            (&cached_vp, &cached_vs, &cached_vb),
            scale,
            num_new,
            n_heads,
            n_kv_heads,
            group_size,
            bits,
        );
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
        // ── Full attention: standard bf16 path ────────────────────────────
        //
        // Single-token decode on a full-attention layer with all four biases
        // present already returned at the top via
        // `compiled_gptoss_attn_layer_fixed`. This block handles the leftover
        // cases: prefill (S>1), bias-less layers, and the cold path before
        // the compiled-trace cache warms up. Sliding-window layers always
        // reach this file via the earlier branch; their cache rotation
        // would need a different layout to express in a compiled graph.
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
        let output = crate::decode::sdpa_causal_like_mlx(&q, &valid_keys, &valid_values, scale, s);
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
