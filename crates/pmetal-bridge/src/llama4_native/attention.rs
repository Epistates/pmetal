//! iRoPE attention: chunk-mask construction, temperature tuning for NoPE
//! layers, and the attention forward step (bf16 + zero-overhead-quantized
//! cache paths).

use crate::InlineArray;

use super::cache::KvLayerCache;
use super::weights::LayerWeights;

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
pub(super) fn build_chunk_mask(offset: i32, s: i32, end: i32, chunk_size: i32) -> InlineArray {
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

    // Cast to bfloat16 additive mask: 0 → 0.0, 1 → ... actually MLX sdpa_with_mask
    // expects an additive float mask where -inf means masked. Convert boolean to float:
    // where mask=1 → 0.0, mask=0 → -inf (large negative).
    // We return the boolean-as-int32 array and convert in the attention function
    // using where_cond.
    flat.reshape(&[s, end])
}

/// Convert a 0/1 int32 mask `[q, k]` to an additive attention bias `[q, k]`.
/// 0 → -1e9 (masked), 1 → 0.0 (unmasked).
fn make_additive_mask(bool_mask: &InlineArray, dtype: i32) -> InlineArray {
    // large negative value in the model dtype
    let neg_inf = InlineArray::scalar_with_dtype(-1e9, dtype);
    let zero = InlineArray::scalar_with_dtype(0.0, dtype);
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

#[allow(clippy::too_many_arguments)]
pub(super) fn attn_forward(
    lw: &LayerWeights,
    normed: &InlineArray,
    b: i32,
    s: i32,
    cache: &mut KvLayerCache,
    rope_offset: i32,
    chunk_mask: Option<&InlineArray>, // prefill only — [s, offset+s]
    chunk_size: i32,
) -> InlineArray {
    let n_heads = lw.n_heads;
    let n_kv_heads = lw.n_kv_heads;
    let head_dim = lw.head_dim;
    let scale = lw.attn_scale;

    // A single decoded token on a chunked (RoPE) layer attends to the keys of
    // its own chunk only, `[p - p % chunk, p]`, as transformers'
    // `create_chunked_causal_mask` and the reference implementation's
    // `block_pos == 0` mask define it. The cache keeps every position, so
    // the chunk is where the keys read start.
    let kv_start = if lw.use_rope && s == 1 && chunk_size > 0 {
        (rope_offset / chunk_size) * chunk_size
    } else {
        0
    };

    // Fused-decode fast path: T=1 on the bf16 path. One compiled kernel
    // covers both layer flavours (RoPE/NoPE) via static flags captured in
    // the closure — `use_rope`, `use_qk_norm`, `has_biases`, `temp_tuning`.
    // Each flag combo gets its own compile trace. Quantized/turboquant
    // caches and prefill (S>1) stay on the per-op paths below.
    let dtype_for_dummy = normed.dtype_raw();
    // The compiled graph rotates as the per-op path does, Scout's Llama 3
    // bands included. NoPE layers never rotate, so any rotation will do
    // there; the plain one keeps their trace free of a period table.
    let rope_kernel = if lw.use_rope {
        lw.rotary.fixed_kernel(rope_offset as i64 + s as i64)
    } else {
        Some(crate::rope::RopeKernel {
            base: lw.rotary.rotary().theta as f32,
            scale: 1.0,
            periods: None,
            gain: None,
        })
    }
    .filter(|_| crate::decode::compiled_decode_enabled());
    if s == 1
        && chunk_mask.is_none()
        && cache.turboquant.is_none()
        && cache.quant_config.is_none()
        && let Some(rope_kernel) = rope_kernel
    {
        // Biases are all-or-none in real Llama 4 configs (audited at load
        // time). Disallow the mixed case to keep the compiled graph simple
        // — the per-op path handles oddballs correctly.
        let has_biases = lw.attn_q_b.is_some()
            && lw.attn_k_b.is_some()
            && lw.attn_v_b.is_some()
            && lw.attn_o_b.is_some();
        let bias_consistent = has_biases
            || (lw.attn_q_b.is_none()
                && lw.attn_k_b.is_none()
                && lw.attn_v_b.is_none()
                && lw.attn_o_b.is_none());
        if bias_consistent {
            // When biases are absent we still pass four placeholder arrays
            // so the FFI shape stays stable — the compiled closure ignores
            // them via the `has_biases` flag.
            let dummy = InlineArray::scalar_with_dtype(0.0, dtype_for_dummy);
            let qb = lw.attn_q_b.as_ref().unwrap_or(&dummy);
            let kb = lw.attn_k_b.as_ref().unwrap_or(&dummy);
            let vb = lw.attn_v_b.as_ref().unwrap_or(&dummy);
            let ob = lw.attn_o_b.as_ref().unwrap_or(&dummy);

            let next = cache.offset + s;
            crate::native_common::kv_cache::alloc_or_grow_kv(
                crate::native_common::kv_cache::GrowthPolicy::AmortizedChunked,
                &mut cache.keys,
                &mut cache.values,
                b,
                n_kv_heads,
                next,
                head_dim,
                dtype_for_dummy,
            );
            let cache_keys = cache.keys.take().unwrap();
            let cache_vals = cache.values.take().unwrap();
            let temp_tuning_enabled = lw.attn_temperature_tuning > 0 && !lw.use_rope;
            let (output, new_cache_keys, new_cache_vals) =
                InlineArray::compiled_llama4_attn_layer_fixed(
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
                    lw.use_rope,
                    lw.attn_qk_norm,
                    has_biases,
                    temp_tuning_enabled,
                    lw.floor_scale,
                    lw.layer_attn_scale,
                    lw.norm_eps,
                    kv_start,
                );
            cache.keys = Some(new_cache_keys);
            cache.values = Some(new_cache_vals);
            cache.offset = next;
            return output;
        }
    }

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
        // Traditional (interleaved) RoPE at rope_theta (500_000), with the
        // Llama 3 bands when `rope_scaling` sets them.
        let q = lw.rotary.apply(&queries, rope_offset);
        let k = lw.rotary.apply(&keys, rope_offset);
        (q, k)
    } else {
        (queries, keys)
    };

    // QK-norm: only on RoPE layers (use_qk_norm = args.use_qk_norm AND use_rope),
    // weightless, at the model's `rms_norm_eps` (transformers'
    // `Llama4TextL2Norm(config.rms_norm_eps)`).
    let (queries, keys) = if lw.attn_qk_norm {
        let q = queries.rms_norm(None, lw.norm_eps);
        let k = keys.rms_norm(None, lw.norm_eps);
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

    let output = if let Some(ref mut tq_cache) = cache.turboquant {
        // ── TurboQuant compressed KV cache path ────────────────────────────
        // Decode + NoPE prefill: shared dispatch helper handles
        // append_and_compute_attention (with hot/cold split) and surfaces
        // failures before the layer offset is advanced.
        //
        // Local-layer prefill with chunk_mask: TurboQuant's direct-attention
        // path doesn't accept a custom additive mask, so we mirror the affine
        // prefill — append, dequantize the full cache, run sdpa_with_mask.
        if let (true, Some(mask_int)) = (lw.use_rope, chunk_mask) {
            let dtype = queries.dtype_raw();
            tq_cache
                .append(&keys, &values)
                .expect("Llama 4 TurboQuant masked prefill append failed");
            let valid_keys = tq_cache
                .dequantize_keys()
                .expect("Llama 4 TurboQuant masked prefill failed to dequantize keys");
            let valid_values = tq_cache
                .dequantize_values()
                .expect("Llama 4 TurboQuant masked prefill failed to dequantize values");
            cache.offset = next;
            let mask_full = make_additive_mask(&mask_int.slice(&[0, 0], &[s, next]), dtype);
            let mask_4d = mask_full.reshape(&[1, 1, s, next]);
            let output = queries.sdpa_with_mask(&valid_keys, &valid_values, scale, Some(&mask_4d));
            crate::error::check_last_error()
                .expect("Llama 4 TurboQuant masked prefill SDPA failed");
            output
        } else if kv_start > 0 {
            // A decode past the first chunk on a chunked layer reads only its
            // chunk, which the fused kernels can't restrict to: dequantize and
            // attend over that slice.
            tq_cache
                .append(&keys, &values)
                .expect("Llama 4 TurboQuant chunked decode append failed");
            let all_keys = tq_cache
                .dequantize_keys()
                .expect("Llama 4 TurboQuant chunked decode failed to dequantize keys");
            let all_values = tq_cache
                .dequantize_values()
                .expect("Llama 4 TurboQuant chunked decode failed to dequantize values");
            cache.offset = next;
            let chunk =
                |a: &InlineArray| a.slice(&[0, 0, kv_start, 0], &[b, n_kv_heads, next, head_dim]);
            let output =
                queries.sdpa_with_mask(&chunk(&all_keys), &chunk(&all_values), scale, None);
            crate::error::check_last_error()
                .expect("Llama 4 TurboQuant chunked decode SDPA failed");
            output
        } else {
            let out = crate::turboquant_dispatch::turboquant_attention_step(
                tq_cache, &queries, &keys, &values, scale, prev, "LLAMA4",
            )
            .expect("Llama 4 TurboQuant attention step failed");
            cache.offset = next;
            out
        }
    } else if let Some(qcfg) = cache.quant_config {
        // ── Zero-overhead affine-quantized KV cache path ──────────────────
        // Same layout as the reference QuantizedKVCache: quantize K/V immediately after
        // RoPE/QK-norm, store as (packed_uint32, scales, biases), pass to
        // quantized_matmul which dequantizes inside the Metal kernel.
        //
        // For prefill with a chunk_mask we must dequantize for the masked SDPA
        // call (quantized_sdpa does not accept an additive mask). Memory savings
        // still apply for the stored tokens.
        let bits = qcfg.bits as i32;
        let group_size = qcfg.group_size;
        let packed_dim = (head_dim * bits + 31) / 32;
        let scales_dim = head_dim / group_size;
        let uint32_dt = crate::compat::Dtype::Uint32.as_i32();
        let dtype = keys.dtype_raw();

        // Quantize new K/V
        let keys_2d = keys.reshape(&[b * n_kv_heads * num_new, head_dim]);
        let (kp, ks, kb) = keys_2d.quantize_weights(group_size, bits);
        let kp = kp.reshape(&[b, n_kv_heads, num_new, packed_dim]);
        let ks = ks.reshape(&[b, n_kv_heads, num_new, scales_dim]);
        let kb = kb.reshape(&[b, n_kv_heads, num_new, scales_dim]);

        let values_2d = values.reshape(&[b * n_kv_heads * num_new, head_dim]);
        let (vp, vs, vb) = values_2d.quantize_weights(group_size, bits);
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

        // slice_set new tokens into cache buffers
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
            .slice(&[0, 0, kv_start, 0], &[b, n_kv_heads, next, packed_dim]);
        let cached_ks = qk
            .scales
            .slice(&[0, 0, kv_start, 0], &[b, n_kv_heads, next, scales_dim]);
        let cached_kb = qk
            .biases
            .slice(&[0, 0, kv_start, 0], &[b, n_kv_heads, next, scales_dim]);
        let qv = cache.quantized_values.as_ref().unwrap();
        let cached_vp = qv
            .packed
            .slice(&[0, 0, kv_start, 0], &[b, n_kv_heads, next, packed_dim]);
        let cached_vs = qv
            .scales
            .slice(&[0, 0, kv_start, 0], &[b, n_kv_heads, next, scales_dim]);
        let cached_vb = qv
            .biases
            .slice(&[0, 0, kv_start, 0], &[b, n_kv_heads, next, scales_dim]);

        if lw.use_rope {
            if let Some(mask_int) = chunk_mask {
                // Prefill with chunk mask: dequantize for masked SDPA.
                // Memory savings still apply; compute overhead only at prefill.
                let kp_flat = cached_kp.reshape(&[b * n_kv_heads * next, packed_dim]);
                let ks_flat = cached_ks.reshape(&[b * n_kv_heads * next, scales_dim]);
                let kb_flat = cached_kb.reshape(&[b * n_kv_heads * next, scales_dim]);
                let valid_keys = kp_flat
                    .dequantize(&ks_flat, &kb_flat, group_size, bits)
                    .reshape(&[b, n_kv_heads, next, head_dim]);

                let vp_flat = cached_vp.reshape(&[b * n_kv_heads * next, packed_dim]);
                let vs_flat = cached_vs.reshape(&[b * n_kv_heads * next, scales_dim]);
                let vb_flat = cached_vb.reshape(&[b * n_kv_heads * next, scales_dim]);
                let valid_values = vp_flat
                    .dequantize(&vs_flat, &vb_flat, group_size, bits)
                    .reshape(&[b, n_kv_heads, next, head_dim]);

                let mask_full = make_additive_mask(&mask_int.slice(&[0, 0], &[s, next]), dtype);
                let mask_4d = mask_full.reshape(&[1, 1, s, next]);
                queries.sdpa_with_mask(&valid_keys, &valid_values, scale, Some(&mask_4d))
            } else {
                // Decode on a local layer — pure zero-overhead quantized path
                crate::decode::quantized_sdpa(
                    &queries,
                    (&cached_kp, &cached_ks, &cached_kb),
                    (&cached_vp, &cached_vs, &cached_vb),
                    scale,
                    num_new,
                    n_heads,
                    n_kv_heads,
                    group_size,
                    bits,
                    None,
                )
            }
        } else {
            // Global (NoPE) layer: quantized_sdpa handles causal mask internally
            crate::decode::quantized_sdpa(
                &queries,
                (&cached_kp, &cached_ks, &cached_kb),
                (&cached_vp, &cached_vs, &cached_vb),
                scale,
                num_new,
                n_heads,
                n_kv_heads,
                group_size,
                bits,
                None,
            )
        }
    } else {
        // ── Standard bf16 path ────────────────────────────────────────────
        //
        // Single-token decode without a chunk_mask already returned at the
        // top via `compiled_llama4_attn_layer_fixed`, which uses static
        // flags (use_rope, use_qk_norm, has_biases, temp_tuning) to fuse
        // both RoPE and NoPE layer flavours into one parameterised graph.
        // This block handles the leftover cases: prefill (S>1), prefill
        // with chunk_mask on local layers, and the bias-mixed config that
        // the fused path explicitly rejects.
        let dtype = cache
            .keys
            .as_ref()
            .map(|k| k.dtype_raw())
            .unwrap_or_else(|| keys.dtype_raw());
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
            .slice(&[0, 0, kv_start, 0], &[b, n_kv_heads, next, head_dim]);
        let valid_values = cache
            .values
            .as_ref()
            .unwrap()
            .slice(&[0, 0, kv_start, 0], &[b, n_kv_heads, next, head_dim]);

        // SDPA
        if lw.use_rope {
            // Local (chunked) layer:
            // - For decode (s=1): the keys read start at the token's chunk
            //   (`kv_start`), so every one of them is visible.
            // - For prefill (s>1): apply the chunk mask.
            if let Some(mask_int) = chunk_mask {
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
        }
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

    // [1, 1, S, 1] so it broadcasts over [B, H, S, D]; the product is taken
    // in f32 and cast back, as transformers does.
    let scale_arr = InlineArray::from_f32_slice(&scales, &[1, 1, s, 1]);
    queries.multiply(&scale_arr).as_dtype(dtype)
}
