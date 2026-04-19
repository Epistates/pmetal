//! Full-model forward step + prefill/prime/generate wrappers.

use crate::InlineArray;
use crate::inline_array as bridge;

use super::attention::{attn_forward, build_chunk_mask};
use super::cache::NativeCache;
use super::moe::{dense_mlp_forward, moe_forward};
use super::weights::NativeWeights;

/// Run one forward step — works for both T=1 decode and T=N prefill.
///
/// `token_ids` must be shape `[B, T]` int32. Returns logits `[B, T, vocab]`.
///
/// Implements iRoPE exactly as the Python reference:
/// - Local layers (use_rope=true): chunked causal attention with RoPE + QK-norm
/// - Global layers (use_rope=false): full causal attention with NoPE and
///   attention temperature tuning
pub fn forward_step(
    weights: &NativeWeights,
    token_ids: &InlineArray, // [B, T]
    cache: &mut NativeCache,
) -> InlineArray {
    let b = token_ids.dim(0);
    let s = token_ids.dim(1);

    // Embedding lookup: [B, T, hidden]
    let mut hidden = weights.embed_w.take_axis(token_ids, 0);

    // Build chunk mask for local layers (chunked attention).
    // Python computes this once per forward:
    //   linds = mx.arange(start, end)     ← positions of cached tokens
    //   rinds = mx.arange(offset, end)[:, None]  ← positions of query tokens
    //   block_pos = |linds // chunk_size - rinds // chunk_size|
    //   token_pos = linds <= rinds
    //   chunk_mask = (block_pos == 0) & token_pos
    // For decode (T=1) this collapses to: only positions in the same chunk as
    // the current query are attended to.
    let chunk_size = weights.attention_chunk_size;
    // offset = number of tokens already in the cache (all layers share same sequence position).
    let offset = cache.rope_offset;
    let end = offset + s;
    // We build the chunk mask eagerly only for prefill (T > 1). For decode (T=1)
    // we skip the mask and use pure causal (the single query token attends to all
    // positions in its chunk, which degenerates to a simple causal window).
    let chunk_mask: Option<InlineArray> = if s > 1 {
        // linds: [1, end] range — all key positions from start of sequence
        // For simplicity in the native path we treat start_position = 0.
        // Build as bool mask [s, offset+s].
        let chunk_mask_val = build_chunk_mask(offset, s, end, chunk_size);
        Some(chunk_mask_val)
    } else {
        None // decode: use "causal" SDPA mode for local layers (no mask needed)
    };

    for (li, lw) in weights.layers.iter().enumerate() {
        // Input LayerNorm
        let normed = hidden.rms_norm(Some(&lw.input_ln_w), lw.norm_eps);

        // Attention
        let attn_out = attn_forward(
            lw,
            &normed,
            b,
            s,
            &mut cache.kv_caches[li],
            cache.rope_offset,
            chunk_mask.as_ref(),
        );

        // Residual add
        let h = hidden.add(&attn_out);

        // Post-attention LayerNorm
        let mlp_in = h.rms_norm(Some(&lw.post_ln_w), lw.norm_eps);

        // Feed-forward (MoE or dense)
        let ff_out = if lw.is_moe {
            moe_forward(lw.moe.as_ref().unwrap(), &mlp_in, b, s)
        } else {
            dense_mlp_forward(lw, &mlp_in)
        };

        // Residual add
        hidden = h.add(&ff_out);
    }

    // Advance position counter.
    cache.rope_offset += s;

    // Final norm + LM head.
    let hidden = hidden.rms_norm(Some(&weights.final_norm_w), weights.final_norm_eps);
    if weights.tie_word_embeddings {
        hidden.matmul(&weights.embed_w.t())
    } else {
        hidden.matmul(weights.lm_head_w.as_ref().unwrap())
    }
}

// ============================================================================
// Generation loop
// ============================================================================

/// Run prompt prefill and return the first sampled token.
pub fn prefill_first_token(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    input_ids: &[u32],
    temperature: f32,
) -> u32 {
    crate::decode::prefill_first_token(weights, cache, input_ids, temperature, forward_step)
}

fn prepare_generation_cache(cache: &mut NativeCache) {
    cache.eval_and_detach_states();
    bridge::clear_cache();
}

fn prime_generation_impl(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    temperature: f32,
    reset_peak_memory: bool,
    log_session: bool,
) -> InlineArray {
    crate::decode::prime_generation(
        "LLAMA4_NATIVE",
        weights.model_dtype,
        weights,
        cache,
        first_token,
        temperature,
        reset_peak_memory,
        log_session,
        prepare_generation_cache,
        forward_step,
    )
}

fn generate_from_primed_sample_impl(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    current_y: InlineArray,
    max_tokens: usize,
    params: crate::decode::SamplingParams,
    log_stats: bool,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    crate::decode::generate_from_primed_sample_with_params(
        "LLAMA4_NATIVE",
        weights,
        cache,
        current_y,
        max_tokens,
        params,
        log_stats,
        on_token,
        forward_step,
    )
}

/// Run one MLX-LM-style benchmark trial on the canonical Llama 4 native path.
pub fn benchmark_mlx_lm_trial(
    weights: &NativeWeights,
    prompt_ids: &[u32],
    generation_tokens: usize,
) -> crate::decode::BenchmarkTrial {
    crate::inline_array::reset_peak_memory();
    let mut cache = NativeCache::new_empty(weights);

    let prompt_tic = std::time::Instant::now();
    let first_tok = prefill_first_token(weights, &mut cache, prompt_ids, 0.0);
    let current_y = prime_generation_impl(weights, &mut cache, first_tok, 0.0, false, false);
    let prompt_secs = prompt_tic.elapsed().as_secs_f64();

    let generation_secs = if generation_tokens > 1 {
        let generation_tic = std::time::Instant::now();
        let (generated_tail, _) = generate_from_primed_sample_impl(
            weights,
            &mut cache,
            current_y,
            generation_tokens - 1,
            crate::decode::SamplingParams::new(0.0),
            false,
            |_| true,
        );
        debug_assert_eq!(generated_tail.len(), generation_tokens - 1);
        generation_tic.elapsed().as_secs_f64()
    } else {
        crate::inline_array::synchronize();
        f64::MIN_POSITIVE
    };

    let trial = crate::decode::BenchmarkTrial {
        prompt_secs,
        generation_secs,
        peak_memory_bytes: crate::inline_array::get_peak_memory(),
    };

    crate::inline_array::synchronize();
    crate::inline_array::clear_cache();
    trial
}

/// Run the full generation loop with async GPU pipelining.
///
/// `first_token` is the last token from the prompt (already processed into `cache`
/// by a prefill call). Each call to `on_token` receives the sampled token ID and
/// returns `false` to stop early (e.g. on EOS).
///
/// Returns all generated token IDs (not including `first_token`).
pub fn generate(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    max_tokens: usize,
    params: crate::decode::SamplingParams,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    let current_y =
        prime_generation_impl(weights, cache, first_token, params.temperature, true, true);
    generate_from_primed_sample_impl(
        weights, cache, current_y, max_tokens, params, true, on_token,
    )
}
