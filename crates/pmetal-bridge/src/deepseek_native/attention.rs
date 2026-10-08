//! Multi-head Latent Attention forward pass — the defining innovation of
//! DeepSeek V3. Decode absorbs `embed_q` into Q and runs SDPA directly in
//! latent space; prefill expands K/V from the latent and runs standard SDPA.

use crate::InlineArray;

use super::cache::MlaLayerCache;
use super::weights::LayerWeights;

/// Multi-head Latent Attention forward pass.
///
/// Mirrors `DeepseekV3Attention.__call__` exactly:
///
/// 1. Compute Q via low-rank projection (or direct) → split into q_nope + q_pe
/// 2. Compress KV: x → [c_kv || k_pe_raw], apply RMSNorm to c_kv → kv_latent
/// 3. Apply RoPE to q_pe and k_pe
/// 4. Cache kv_latent + k_pe (MLA stores latent, not full K/V)
/// 5. Compute PE scores: `pe_scores = (q_pe * scale) @ k_pe.T`
/// 6. Decode (T=1): absorb embed_q into q_nope, then SDPA with latent as K=V
///    Prefill (T>1): expand K = embed_q(latent), V = unembed_out(latent), then SDPA
/// 7. Decode (T=1): project output via unembed_out
/// 8. Output projection o_proj
pub(super) fn mla_forward(
    lw: &LayerWeights,
    x: &InlineArray,
    b: i32,
    s: i32,
    cache: &mut MlaLayerCache,
    rope_offset: i32,
) -> InlineArray {
    let n_heads = lw.n_heads;
    let q_head_dim = lw.q_head_dim;
    let nope_dim = lw.qk_nope_head_dim;
    let rope_dim = lw.qk_rope_head_dim;
    let v_dim = lw.v_head_dim;
    let lora_rank = lw.kv_lora_rank;
    let scale = lw.scale;

    // ── Q projection ─────────────────────────────────────────────────────
    // Low-rank: x → q_a_proj → rms_norm → q_b_proj → [B, S, H, q_head_dim]
    // Direct:   x → q_proj → [B, S, H, q_head_dim]
    let q_raw = if let Some(ref q_a_w) = lw.q_a_w {
        let q_a = x.matmul(q_a_w);
        let q_a_norm = q_a.rms_norm(lw.q_a_norm_w.as_ref(), 1e-6);
        q_a_norm.matmul(lw.q_b_w.as_ref().unwrap())
    } else {
        x.matmul(lw.q_w.as_ref().unwrap())
    };
    // Reshape to [B, S, H, q_head_dim] then transpose to [B, H, S, q_head_dim]
    let q = q_raw
        .reshape(&[b, s, n_heads, q_head_dim])
        .transpose_axes(&[0, 2, 1, 3]);
    // Split q into [q_nope, q_pe] along last axis at nope_dim.
    let mut q_parts = q.split(&[nope_dim], -1);
    let q_pe = q_parts.pop().unwrap(); // [B, H, S, rope_dim]
    let q_nope = q_parts.pop().unwrap(); // [B, H, S, nope_dim]

    // ── KV compression ───────────────────────────────────────────────────
    // x → kv_a_proj_with_mqa → [B, S, kv_lora_rank + qk_rope_head_dim]
    let compressed_kv = x.matmul(&lw.kv_a_proj_w);
    // Split: [compressed_kv (lora_rank), k_pe_raw (rope_dim)]
    let mut kv_parts = compressed_kv.split(&[lora_rank], -1);
    let k_pe_raw = kv_parts.pop().unwrap(); // [B, S, rope_dim]
    let compressed = kv_parts.pop().unwrap(); // [B, S, lora_rank]

    // RMS-norm the latent.
    let kv_latent_tok = compressed.rms_norm(Some(&lw.kv_a_norm_w), 1e-6);

    // Reshape k_pe_raw → [B, 1, S, rope_dim] and transpose → [B, 1, S, rope_dim]
    // (Python: reshape(B,L,1,rope_dim).transpose(0,2,1,3) → [B,1,S,rope_dim])
    let k_pe_raw_4d = k_pe_raw
        .reshape(&[b, s, 1, rope_dim])
        .transpose_axes(&[0, 2, 1, 3]); // [B, 1, S, rope_dim]

    // Apply RoPE to q_pe [B, H, S, rope_dim] and k_pe [B, 1, S, rope_dim]:
    // interleaved pairs, with YaRN's frequencies and attention factor when
    // `rope_scaling` sets them (DeepSeek-V3/R1 ship factor 40).
    let q_pe = lw.rotary.apply(&q_pe, rope_offset);
    let k_pe = lw.rotary.apply(&k_pe_raw_4d, rope_offset);

    // Expand kv_latent: [B, S, lora] → [B, 1, S, lora] (Python: expand_dims(axis=1))
    let kv_latent_4d = kv_latent_tok.expand_dims(1); // [B, 1, S, lora_rank]

    // ── KV cache update ───────────────────────────────────────────────────
    // Cache stores (kv_latent, k_pe) — NOT full K,V tensors.
    // When quant_config is set, the latent and k_pe are stored quantized to
    // reduce memory; they are dequantized before each SDPA call.
    let prev = cache.offset;
    let next = prev + s;

    let (all_kv_latent, all_k_pe) = if let Some(qcfg) = cache.quant_config {
        // ── Quantized latent cache path ───────────────────────────────────
        let bits = qcfg.bits as i32;
        let group_size = qcfg.group_size;
        let uint32_dt = crate::compat::Dtype::Uint32.as_i32();
        // Use bf16 (dtype=11) for scales/biases — same convention as the
        // main quantized KV cache in qwen3_native.
        let dtype = 11i32; // bf16

        // ---- kv_latent: [B, 1, S, lora_rank] ----
        let packed_lat = (lora_rank * bits + 31) / 32;
        let scales_lat = lora_rank / group_size;
        let lat_2d = kv_latent_4d.reshape(&[b * s, lora_rank]);
        let (lp, ls, lb) = lat_2d.quantize_weights(group_size, bits);
        let lp = lp.reshape(&[b, 1, s, packed_lat]);
        let ls = ls.reshape(&[b, 1, s, scales_lat]);
        let lb = lb.reshape(&[b, 1, s, scales_lat]);

        // ---- k_pe: [B, 1, S, rope_dim] ----
        let packed_pe = (rope_dim * bits + 31) / 32;
        let scales_pe = (rope_dim / group_size).max(1);
        let pe_2d = k_pe.reshape(&[b * s, rope_dim]);
        let (pp, ps, pb) = pe_2d.quantize_weights(group_size, bits);
        let pp = pp.reshape(&[b, 1, s, packed_pe]);
        let ps = ps.reshape(&[b, 1, s, scales_pe]);
        let pb = pb.reshape(&[b, 1, s, scales_pe]);

        crate::qwen3_native::QuantizedTuple::ensure_capacity(
            &mut cache.quantized_latent,
            crate::native_common::kv_cache::GrowthPolicy::AmortizedChunked,
            b,
            1,
            next,
            packed_lat,
            scales_lat,
            uint32_dt,
            dtype,
        );
        crate::qwen3_native::QuantizedTuple::ensure_capacity(
            &mut cache.quantized_k_pe,
            crate::native_common::kv_cache::GrowthPolicy::AmortizedChunked,
            b,
            1,
            next,
            packed_pe,
            scales_pe,
            uint32_dt,
            dtype,
        );

        // slice_set new quantized tokens into cache
        let start_q = [0i32, 0, prev, 0];
        let stop_lp = [b, 1, next, packed_lat];
        let stop_ls = [b, 1, next, scales_lat];
        let ql_ref = cache.quantized_latent.as_mut().unwrap();
        ql_ref.packed = ql_ref.packed.slice_set(&lp, &start_q, &stop_lp);
        ql_ref.scales = ql_ref.scales.slice_set(&ls, &start_q, &stop_ls);
        ql_ref.biases = ql_ref.biases.slice_set(&lb, &start_q, &stop_ls);

        let stop_pp = [b, 1, next, packed_pe];
        let stop_ps = [b, 1, next, scales_pe];
        let qp_ref = cache.quantized_k_pe.as_mut().unwrap();
        qp_ref.packed = qp_ref.packed.slice_set(&pp, &start_q, &stop_pp);
        qp_ref.scales = qp_ref.scales.slice_set(&ps, &start_q, &stop_ps);
        qp_ref.biases = qp_ref.biases.slice_set(&pb, &start_q, &stop_ps);

        cache.offset = next;

        // Dequantize valid portions for use in SDPA
        let ql = cache.quantized_latent.as_ref().unwrap();
        let qp = cache.quantized_k_pe.as_ref().unwrap();

        let lat_packed = ql.packed.slice(&[0, 0, 0, 0], &[b, 1, next, packed_lat]);
        let lat_scales = ql.scales.slice(&[0, 0, 0, 0], &[b, 1, next, scales_lat]);
        let lat_biases = ql.biases.slice(&[0, 0, 0, 0], &[b, 1, next, scales_lat]);
        let all_kv_latent = lat_packed
            .reshape(&[b * next, packed_lat])
            .dequantize(
                &lat_scales.reshape(&[b * next, scales_lat]),
                &lat_biases.reshape(&[b * next, scales_lat]),
                group_size,
                bits,
            )
            .reshape(&[b, 1, next, lora_rank]); // [B, 1, T_total, lora]

        let pe_packed = qp.packed.slice(&[0, 0, 0, 0], &[b, 1, next, packed_pe]);
        let pe_scales = qp.scales.slice(&[0, 0, 0, 0], &[b, 1, next, scales_pe]);
        let pe_biases = qp.biases.slice(&[0, 0, 0, 0], &[b, 1, next, scales_pe]);
        let all_k_pe = pe_packed
            .reshape(&[b * next, packed_pe])
            .dequantize(
                &pe_scales.reshape(&[b * next, scales_pe]),
                &pe_biases.reshape(&[b * next, scales_pe]),
                group_size,
                bits,
            )
            .reshape(&[b, 1, next, rope_dim]); // [B, 1, T_total, rope_dim]

        (all_kv_latent, all_k_pe)
    } else {
        // ── Standard bf16 path ────────────────────────────────────────────
        // MLA stores `kv_latent` (B,1,T,lora_rank) and `k_pe` (B,1,T,rope_dim)
        // — same time axis, different last dims — so each leg uses
        // alloc_or_grow_buffer with a per-leg shape closure. dtype 11 is
        // bfloat16 (matches the reference MLA cache layout).
        use crate::native_common::kv_cache::{GrowthPolicy, alloc_or_grow_buffer};
        const BF16: i32 = 11;
        alloc_or_grow_buffer(
            GrowthPolicy::AmortizedChunked,
            &mut cache.kv_latent,
            next,
            2,
            BF16,
            |cap| [b, 1, cap, lora_rank],
        );
        alloc_or_grow_buffer(
            GrowthPolicy::AmortizedChunked,
            &mut cache.k_pe,
            next,
            2,
            BF16,
            |cap| [b, 1, cap, rope_dim],
        );

        let start_kv = [0i32, 0, prev, 0];
        let stop_kv = [b, 1, next, lora_rank];
        let start_kp = [0i32, 0, prev, 0];
        let stop_kp = [b, 1, next, rope_dim];

        let kv_buf = cache.kv_latent.take().unwrap();
        let kp_buf = cache.k_pe.take().unwrap();

        cache.kv_latent = Some(kv_buf.slice_set(&kv_latent_4d, &start_kv, &stop_kv));
        cache.k_pe = Some(kp_buf.slice_set(&k_pe, &start_kp, &stop_kp));
        cache.offset = next;

        // Valid portions of the cache.
        let all_kv_latent = cache
            .kv_latent
            .as_ref()
            .unwrap()
            .slice(&[0, 0, 0, 0], &[b, 1, next, lora_rank]); // [B, 1, T_total, lora]
        let all_k_pe = cache
            .k_pe
            .as_ref()
            .unwrap()
            .slice(&[0, 0, 0, 0], &[b, 1, next, rope_dim]); // [B, 1, T_total, rope_dim]

        (all_kv_latent, all_k_pe)
    };

    // ── PE attention scores ───────────────────────────────────────────────
    // pe_scores = (q_pe * scale) @ k_pe.swapaxes(-1,-2)
    // q_pe:    [B, H, S, rope_dim]
    // k_pe:    [B, 1, T, rope_dim]  (broadcast over H)
    // Swap last two axes of k_pe: [B, 1, T, rope_dim] → [B, 1, rope_dim, T]
    let k_pe_t = all_k_pe.transpose_axes(&[0, 1, 3, 2]); // [B, 1, rope_dim, T]
    let scale_arr = crate::decode::scalar_f32_like(scale, &q_pe);
    let q_pe_scaled = q_pe.multiply(&scale_arr);
    // [B, H, S, rope_dim] @ [B, 1, rope_dim, T] → [B, H, S, T]  (H broadcasts 1→H)
    let pe_scores = q_pe_scaled.matmul(&k_pe_t); // [B, H, S, T]

    // ── Decode (T=1) vs Prefill (T>1) ───────────────────────────────────
    // DeepSeek MLA uses a clever "absorbed" representation at decode time:
    //
    // Decode (L=1):
    //   Transform q_nope into the latent space via embed_q:
    //     q_nope_latent = q_nope @ embed_q.weight.swapaxes(-1,-2)
    //   where embed_q.weight = [H, lora_rank, nope_dim] (stored as wk after sanitize)
    //   so embed_q.weight.swapaxes(-1,-2) = [H, nope_dim, lora_rank]
    //   q_nope [B, H, 1, nope_dim] @ [H, nope_dim, lora_rank] → [B, H, 1, lora_rank]
    //   Then use kv_latent as BOTH K and V in SDPA (latent-space attention):
    //     scores_nope = q_nope_latent @ kv_latent.T (already in latent space)
    //     output = softmax(scores_nope + pe_scores) @ kv_latent
    //   Post-project output via unembed_out:
    //     [B, H, 1, lora_rank] @ unembed_out.weight → [B, H, 1, v_dim]
    //
    // Prefill (L>1):
    //   Expand K from latent: k_nope = embed_q(kv_latent, transpose=False)
    //     = kv_latent [B, 1, T, lora] @ embed_q.weight [H, lora, nope] → [B, H, T, nope]
    //   Expand V: v = unembed_out(kv_latent)
    //     = kv_latent [B, 1, T, lora] @ unembed_out.weight.swapaxes(-1,-2) [H, v, lora].T → [B, H, T, v]
    //   Standard SDPA in expanded space with pe_scores bias.
    //
    // This asymmetry is the key MLA insight: at decode time we NEVER materialize
    // full K/V — we operate entirely in the compressed latent space.

    let output = if s == 1 {
        // ── Decode path ───────────────────────────────────────────────────
        // q_nope: [B, H, 1, nope_dim]
        // embed_q_w (stored as [H, lora_rank, nope_dim]) → swapaxes(-1,-2) = [H, nope_dim, lora_rank]
        // The transpose is done by multiplying: q_nope @ embed_q_w.transpose_axes([0,2,1])
        let embed_q_t = lw.embed_q_w.transpose_axes(&[0, 2, 1]); // [H, nope_dim, lora_rank]
        let q_nope_latent = q_nope.matmul(&embed_q_t); // [B, H, 1, lora_rank]

        // k = v = kv_latent (all_kv_latent): [B, 1, T, lora_rank]
        // SDPA: q_nope_latent [B, H, 1, lora_rank] vs k,v [B, 1, T, lora_rank]
        // The K head (1) broadcasts to H. pe_scores [B, H, 1, T] is the additive bias.
        // Use sdpa_with_mask where mask = pe_scores (additive, not boolean).
        let out_latent =
            q_nope_latent.sdpa_with_mask(&all_kv_latent, &all_kv_latent, scale, Some(&pe_scores)); // [B, H, 1, lora_rank]

        // Project through unembed_out:
        // unembed_out_w: [H, v_dim, lora_rank]
        // out_latent [B, H, 1, lora_rank] @ unembed_out_w.transpose_axes([0,2,1]) [H, lora_rank, v_dim]
        // Wait — Python: unembed_out(output) with default transpose=True:
        //   output @ unembed_out.weight.swapaxes(-1,-2)
        //   where unembed_out.weight = [H, v_dim, lora_rank] (from sanitize: wv)
        //   swapaxes(-1,-2) = [H, lora_rank, v_dim]
        // So: [B, H, 1, lora] @ [H, lora, v_dim] → [B, H, 1, v_dim]
        let unembed_t = lw.unembed_out_w.transpose_axes(&[0, 2, 1]); // [H, lora, v_dim]
        out_latent.matmul(&unembed_t) // [B, H, 1, v_dim]
    } else {
        // ── Prefill path ──────────────────────────────────────────────────
        // Expand K (nope component): kv_latent @ embed_q_w (transpose=False)
        //   all_kv_latent: [B, 1, T, lora_rank]
        //   embed_q_w:     [H, lora_rank, nope_dim]
        //   matmul: [B, 1, T, lora] @ [H, lora, nope] → [B, H, T, nope]
        let k_nope = all_kv_latent.matmul(&lw.embed_q_w); // [B, H, T, nope_dim] via broadcast

        // Expand V: kv_latent @ unembed_out_w.swapaxes(-1,-2)
        //   unembed_out_w: [H, v_dim, lora_rank]
        //   swapaxes(-1,-2) = [H, lora_rank, v_dim]
        //   matmul: [B, 1, T, lora] @ [H, lora, v_dim] → [B, H, T, v_dim]
        let unembed_t = lw.unembed_out_w.transpose_axes(&[0, 2, 1]); // [H, lora, v_dim]
        let v = all_kv_latent.matmul(&unembed_t); // [B, H, T, v_dim]

        // SDPA: q_nope [B, H, S, nope_dim] vs k_nope [B, H, T, nope_dim] vs v [B, H, T, v_dim]
        // with pe_scores [B, H, S, T] as additive bias.
        q_nope.sdpa_with_mask(&k_nope, &v, scale, Some(&pe_scores)) // [B, H, S, v_dim]
    };

    // ── Output projection ─────────────────────────────────────────────────
    // Transpose [B, H, S, v_dim] → [B, S, H, v_dim] and reshape to [B, S, H*v_dim]
    let output_flat = output
        .transpose_axes(&[0, 2, 1, 3])
        .reshape(&[b, s, n_heads * v_dim]);

    output_flat.matmul(&lw.o_proj_w)
}
