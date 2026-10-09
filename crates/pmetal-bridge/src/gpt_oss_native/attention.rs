//! Attention forward step: full split-half RoPE (YaRN), per-head attention
//! sinks, and three cache paths — the sliding-window ring, full attention in
//! the model dtype, and full attention zero-overhead-quantized.
//!
//! The sinks and the banded window follow the gpt-oss reference
//! implementation and transformers' `GptOssAttention`: each head's sink logit
//! joins its softmax row and attends to nothing, and a sliding layer's query
//! at position `q` sees keys `k` with `q - window < k <= q`.

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
    if let Some(output) = compiled_decode(lw, normed, b, s, cache, rope_offset, dtype) {
        return output;
    }

    let n_heads = lw.attn_n_heads;
    let n_kv_heads = lw.attn_n_kv_heads;
    let head_dim = lw.attn_head_dim;

    // Q, K, V projections — [B, S, n_heads*head_dim], with their biases.
    let project = |w: &InlineArray, bias: &Option<InlineArray>| {
        let y = normed.matmul(w);
        match bias {
            Some(bias) => y.add(bias),
            None => y,
        }
    };
    let heads = |x: InlineArray, n: i32| {
        x.reshape(&[b, s, n, head_dim])
            .transpose_axes(&[0, 2, 1, 3])
    };
    let q = heads(project(&lw.attn_q_w, &lw.attn_q_b), n_heads);
    let k = heads(project(&lw.attn_k_w, &lw.attn_k_b), n_kv_heads);
    let v = heads(project(&lw.attn_v_w, &lw.attn_v_b), n_kv_heads);

    // Full RoPE (no partial rotation): YaRN's frequencies and attention
    // factor, split-half.
    let q = lw.attn_rotary.apply(&q, rope_offset);
    let k = lw.attn_rotary.apply(&k, rope_offset);

    let output = if lw.attn_is_sliding {
        sliding_attention(lw, &q, &k, &v, cache, dtype)
    } else if cache.quant_config.is_some() {
        quantized_attention(lw, &q, &k, &v, cache, dtype)
    } else {
        full_attention(lw, &q, &k, &v, cache, dtype)
    };

    let output = output
        .transpose_axes(&[0, 2, 1, 3])
        .reshape(&[b, s, n_heads * head_dim]);
    let proj = output.matmul(&lw.attn_o_w);
    match &lw.attn_o_b {
        Some(ob) => proj.add(ob),
        None => proj,
    }
}

/// The single-token step as one compiled graph: Q/K/V projections and biases,
/// RoPE (YaRN's period table and attention factor), the cache write, SDPA
/// with the sinks, and the output projection. Full-attention layers write
/// slot `offset` of their growing buffer; sliding layers slot
/// `offset % window` of their ring. `None` when the step has to take the
/// per-op path (prefill, a quantized cache, a missing bias, or the graphs
/// switched off).
fn compiled_decode(
    lw: &LayerWeights,
    normed: &InlineArray,
    b: i32,
    s: i32,
    cache: &mut KvLayerCache,
    rope_offset: i32,
    dtype: i32,
) -> Option<InlineArray> {
    if s != 1 || cache.quant_config.is_some() || !crate::decode::compiled_decode_enabled() {
        return None;
    }
    let rope = lw.attn_rotary.fixed_kernel(rope_offset as i64 + 1)?;
    let (Some(qb), Some(kb), Some(vb), Some(ob)) = (
        lw.attn_q_b.as_ref(),
        lw.attn_k_b.as_ref(),
        lw.attn_v_b.as_ref(),
        lw.attn_o_b.as_ref(),
    ) else {
        return None;
    };
    let (n_kv_heads, head_dim) = (lw.attn_n_kv_heads, lw.attn_head_dim);
    let prev = cache.offset;
    let (write, valid) = if lw.attn_is_sliding {
        let window = lw.attn_sliding_window;
        ensure_ring(cache, b, n_kv_heads, window, head_dim, dtype);
        (prev % window, (prev + 1).min(window))
    } else {
        crate::native_common::kv_cache::alloc_or_grow_kv(
            crate::native_common::kv_cache::GrowthPolicy::AmortizedChunked,
            &mut cache.keys,
            &mut cache.values,
            b,
            n_kv_heads,
            prev + 1,
            head_dim,
            dtype,
        );
        (prev, prev + 1)
    };
    let cache_keys = cache.keys.take().unwrap();
    let cache_vals = cache.values.take().unwrap();
    let (output, keys, values) = InlineArray::compiled_gptoss_attn_layer_fixed(
        normed,
        &lw.attn_q_w,
        &lw.attn_k_w,
        &lw.attn_v_w,
        &lw.attn_o_w,
        qb,
        kb,
        vb,
        ob,
        &lw.attn_sinks,
        &cache_keys,
        &cache_vals,
        write,
        valid,
        rope_offset,
        lw.attn_n_heads,
        n_kv_heads,
        head_dim,
        lw.attn_scale,
        rope,
    );
    cache.keys = Some(keys);
    cache.values = Some(values);
    cache.offset = prev + 1;
    Some(output)
}

/// Allocate a sliding layer's ring of `window` slots on first use.
fn ensure_ring(cache: &mut KvLayerCache, b: i32, n_kv: i32, window: i32, hd: i32, dtype: i32) {
    if cache.keys.is_none() {
        cache.keys = Some(InlineArray::zeros(&[b, n_kv, window, hd], dtype));
        cache.values = Some(InlineArray::zeros(&[b, n_kv, window, hd], dtype));
    }
}

/// `[B, H, L, D]` sliced to `start..end` on the sequence axis.
fn seq(x: &InlineArray, start: i32, end: i32) -> InlineArray {
    let shape = x.shape();
    x.slice(&[0, 0, start, 0], &[shape[0], shape[1], end, shape[3]])
}

/// The ring's filled slots, oldest position first. Position `p` lives in
/// slot `p % window`, so once the ring has wrapped the oldest is at
/// `offset % window`.
fn ring_in_order(ring: &InlineArray, offset: i32, window: i32) -> InlineArray {
    if offset <= window {
        return seq(ring, 0, offset);
    }
    let split = offset % window;
    if split == 0 {
        return ring.clone();
    }
    seq(ring, split, window).concatenate_2(&seq(ring, 0, split), 2)
}

/// The last `window` of `ordered` (positions `end - len .. end`, oldest
/// first), laid out as a ring: position `p` in slot `p % window`.
fn as_ring(ordered: &InlineArray, end: i32, window: i32) -> InlineArray {
    let len = ordered.dim(2);
    if len < window {
        // Positions 0..end with end < window: slot p holds position p.
        let shape = ordered.shape();
        let ring = InlineArray::zeros(&[shape[0], shape[1], window, shape[3]], ordered.dtype_raw());
        return ring.slice_set(ordered, &[0, 0, 0, 0], &[shape[0], shape[1], len, shape[3]]);
    }
    let last = seq(ordered, len - window, len);
    // last[i] is position end - window + i, which belongs in slot (end + i) % window.
    let first_slot = end % window;
    if first_slot == 0 {
        return last;
    }
    let wrap = window - first_slot;
    seq(&last, wrap, window).concatenate_2(&seq(&last, 0, wrap), 2)
}

/// Sliding-window attention over the ring: the banded window of the
/// reference implementation, `q - window < k <= q`.
fn sliding_attention(
    lw: &LayerWeights,
    q: &InlineArray,
    k: &InlineArray,
    v: &InlineArray,
    cache: &mut KvLayerCache,
    dtype: i32,
) -> InlineArray {
    let (b, n_kv_heads, s, head_dim) = (k.dim(0), k.dim(1), k.dim(2), k.dim(3));
    let window = lw.attn_sliding_window;
    ensure_ring(cache, b, n_kv_heads, window, head_dim, dtype);
    let prev = cache.offset;
    let next = prev + s;
    let ring_k = cache.keys.take().unwrap();
    let ring_v = cache.values.take().unwrap();

    let output = if s == 1 {
        // Every filled slot after this write is inside the window.
        let slot = prev % window;
        let (start, stop) = ([0, 0, slot, 0], [b, n_kv_heads, slot + 1, head_dim]);
        let ring_k = ring_k.slice_set(k, &start, &stop);
        let ring_v = ring_v.slice_set(v, &start, &stop);
        let valid = next.min(window);
        let output = q.sdpa_with_sinks(
            &seq(&ring_k, 0, valid),
            &seq(&ring_v, 0, valid),
            lw.attn_scale,
            "",
            None,
            &lw.attn_sinks,
        );
        cache.keys = Some(ring_k);
        cache.values = Some(ring_v);
        output
    } else {
        // The cached positions, oldest first, then the new ones.
        let keys = ring_in_order(&ring_k, prev, window).concatenate_2(k, 2);
        let values = ring_in_order(&ring_v, prev, window).concatenate_2(v, 2);
        let output = if next <= window {
            // Every key from position 0 is inside every query's window.
            q.sdpa_with_sinks(
                &keys,
                &values,
                lw.attn_scale,
                "causal",
                None,
                &lw.attn_sinks,
            )
        } else {
            let mask = band_mask(prev, s, keys.dim(2), window);
            q.sdpa_with_sinks(
                &keys,
                &values,
                lw.attn_scale,
                "",
                Some(&mask),
                &lw.attn_sinks,
            )
        };
        cache.keys = Some(as_ring(&keys, next, window));
        cache.values = Some(as_ring(&values, next, window));
        output
    };
    cache.offset = next;
    output
}

/// `[s, kv_len]` boolean mask for queries at `prev..prev + s` over keys at
/// `prev + s - kv_len..prev + s`: `true` where `q - window < k <= q`.
fn band_mask(prev: i32, s: i32, kv_len: i32, window: i32) -> InlineArray {
    let first_key = prev + s - kv_len;
    let mut allowed = Vec::with_capacity((s * kv_len) as usize);
    for qi in 0..s {
        let qp = prev + qi;
        for kj in 0..kv_len {
            let kp = first_key + kj;
            allowed.push(i32::from(kp <= qp && qp - kp < window));
        }
    }
    InlineArray::from_i32_slice(&allowed)
        .reshape(&[s, kv_len])
        .as_dtype(crate::compat::Dtype::Bool.as_i32())
}

/// Full attention over a growing buffer in the model dtype.
fn full_attention(
    lw: &LayerWeights,
    q: &InlineArray,
    k: &InlineArray,
    v: &InlineArray,
    cache: &mut KvLayerCache,
    dtype: i32,
) -> InlineArray {
    let (b, n_kv_heads, s, head_dim) = (k.dim(0), k.dim(1), k.dim(2), k.dim(3));
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
    let (start, stop) = ([0, 0, prev, 0], [b, n_kv_heads, next, head_dim]);
    let k_buf = cache.keys.take().unwrap();
    let v_buf = cache.values.take().unwrap();
    cache.keys = Some(k_buf.slice_set(k, &start, &stop));
    cache.values = Some(v_buf.slice_set(v, &start, &stop));
    cache.offset = next;

    let keys = seq(cache.keys.as_ref().unwrap(), 0, next);
    let values = seq(cache.values.as_ref().unwrap(), 0, next);
    let mode = if s == 1 { "" } else { "causal" };
    q.sdpa_with_sinks(&keys, &values, lw.attn_scale, mode, None, &lw.attn_sinks)
}

/// Full attention over the zero-overhead affine-quantized cache.
fn quantized_attention(
    lw: &LayerWeights,
    q: &InlineArray,
    k: &InlineArray,
    v: &InlineArray,
    cache: &mut KvLayerCache,
    dtype: i32,
) -> InlineArray {
    let qcfg = cache.quant_config.expect("quantized cache");
    let (b, n_kv_heads, num_new, head_dim) = (k.dim(0), k.dim(1), k.dim(2), k.dim(3));
    let prev = cache.offset;
    let next = prev + num_new;
    let bits = qcfg.bits as i32;
    let group_size = qcfg.group_size;
    let packed_dim = (head_dim * bits + 31) / 32;
    let scales_dim = head_dim / group_size;
    let uint32_dt = crate::compat::Dtype::Uint32.as_i32();

    let quantize = |x: &InlineArray| {
        let (p, sc, bi) = x
            .reshape(&[b * n_kv_heads * num_new, head_dim])
            .quantize_weights(group_size, bits);
        (
            p.reshape(&[b, n_kv_heads, num_new, packed_dim]),
            sc.reshape(&[b, n_kv_heads, num_new, scales_dim]),
            bi.reshape(&[b, n_kv_heads, num_new, scales_dim]),
        )
    };
    let (kp, ks, kb) = quantize(k);
    let (vp, vs, vb) = quantize(v);

    for tuple in [&mut cache.quantized_keys, &mut cache.quantized_values] {
        crate::qwen3_native::QuantizedTuple::ensure_capacity(
            tuple,
            crate::native_common::kv_cache::GrowthPolicy::AmortizedChunked,
            b,
            n_kv_heads,
            next,
            packed_dim,
            scales_dim,
            uint32_dt,
            dtype,
        );
    }

    let start = [0, 0, prev, 0];
    let stop_p = [b, n_kv_heads, next, packed_dim];
    let stop_s = [b, n_kv_heads, next, scales_dim];
    let qk = cache.quantized_keys.as_mut().unwrap();
    qk.packed = qk.packed.slice_set(&kp, &start, &stop_p);
    qk.scales = qk.scales.slice_set(&ks, &start, &stop_s);
    qk.biases = qk.biases.slice_set(&kb, &start, &stop_s);
    let qv = cache.quantized_values.as_mut().unwrap();
    qv.packed = qv.packed.slice_set(&vp, &start, &stop_p);
    qv.scales = qv.scales.slice_set(&vs, &start, &stop_s);
    qv.biases = qv.biases.slice_set(&vb, &start, &stop_s);
    cache.offset = next;

    let qk = cache.quantized_keys.as_ref().unwrap();
    let qv = cache.quantized_values.as_ref().unwrap();
    crate::decode::quantized_sdpa(
        q,
        (
            &seq(&qk.packed, 0, next),
            &seq(&qk.scales, 0, next),
            &seq(&qk.biases, 0, next),
        ),
        (
            &seq(&qv.packed, 0, next),
            &seq(&qv.scales, 0, next),
            &seq(&qv.biases, 0, next),
        ),
        lw.attn_scale,
        num_new,
        lw.attn_n_heads,
        n_kv_heads,
        group_size,
        bits,
        Some(&lw.attn_sinks),
    )
}
