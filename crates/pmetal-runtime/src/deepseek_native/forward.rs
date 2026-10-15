//! Full-model forward step + prefill/prime/generate wrappers.

use crate::InlineArray;
use crate::inline_array;

use super::attention::mla_forward;
use super::cache::NativeCache;
use super::moe::{mlp_forward, moe_forward};
use super::weights::Ffn;
use super::weights::NativeWeights;

/// The decoder layers over `token_ids` (`[B, T]` int32), advancing `cache`:
/// the final hidden states `[B, T, hidden]`, before the final norm.
fn decoder_trunk(
    weights: &NativeWeights,
    token_ids: &InlineArray,
    cache: &mut NativeCache,
) -> InlineArray {
    let b = token_ids.dim(0);
    let s = token_ids.dim(1);

    // Embedding lookup: [B, T, hidden]
    let mut hidden = weights.embed_w.lookup(token_ids);

    for (li, lw) in weights.layers.iter().enumerate() {
        let cache_slot = &mut cache.mla_caches[li];

        // Input LayerNorm
        let normed = hidden.rms_norm(Some(&lw.input_ln_w), lw.norm_eps);

        // MLA Attention
        let attn_out = mla_forward(lw, &normed, b, s, cache_slot, cache.rope_offset);

        // Residual
        let h = hidden.add(&attn_out);

        // Post-attention LayerNorm
        let mlp_in = h.rms_norm(Some(&lw.post_ln_w), lw.norm_eps);

        // MLP / MoE
        let mlp_out = match &lw.ffn {
            Ffn::Moe(moe) => moe_forward(moe, &mlp_in, b, s),
            Ffn::Dense(mlp) => mlp_forward(mlp, &mlp_in),
        };

        // Residual
        hidden = h.add(&mlp_out);
    }

    // Advance position counter
    cache.rope_offset += s;
    hidden
}

/// Final norm and LM head over `hidden` (`[B, T, hidden]`).
fn lm_head(weights: &NativeWeights, hidden: &InlineArray) -> InlineArray {
    let normed = hidden.rms_norm(Some(&weights.final_norm_w), weights.final_norm_eps);
    match &weights.lm_head_w {
        Some(head) => head.matmul_from(&normed),
        None => weights.embed_w.as_linear(&normed),
    }
}

/// Run one forward step. `token_ids` is `[B, T]` int32. Returns `[B, T, vocab]`.
pub fn forward_step(
    weights: &NativeWeights,
    token_ids: &InlineArray,
    cache: &mut NativeCache,
) -> InlineArray {
    lm_head(weights, &decoder_trunk(weights, token_ids, cache))
}

/// [`forward_step`]'s logits for the last position alone, `[B, vocab]`.
///
/// The LM head runs over one row instead of `T`: over DeepSeek's 102k-token
/// (V2) to 129k-token (V3) vocabulary the logits of a long prompt are
/// gigabytes, and all but the last row are discarded by whoever samples.
pub fn forward_step_last(
    weights: &NativeWeights,
    token_ids: &InlineArray,
    cache: &mut NativeCache,
) -> InlineArray {
    let hidden = decoder_trunk(weights, token_ids, cache);
    lm_head(weights, &crate::decode::last_token_hidden(&hidden)).squeeze(1)
}

// ============================================================================
// Generation loop
// ============================================================================

/// Run prompt prefill and return the first sampled token and the
/// generation's sampler.
///
/// The prompt goes through in [`PREFILL_CHUNK`](crate::decode::PREFILL_CHUNK)
/// passes and only the last position's logits are computed.
pub fn prefill_first_token(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    input_ids: &[u32],
    params: crate::decode::SamplingParams,
) -> (u32, crate::decode::DecodeSampler) {
    crate::decode::prefill_first_token_chunked(
        weights,
        cache,
        input_ids,
        params,
        crate::decode::PREFILL_CHUNK,
        forward_step_last,
    )
}

fn prepare_generation_cache(cache: &mut NativeCache) {
    cache.eval_and_detach_states();
    inline_array::clear_cache();
}

fn prime_generation_impl(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    sampler: &mut crate::decode::DecodeSampler,
    reset_peak_memory: bool,
    log_session: bool,
) -> InlineArray {
    crate::decode::prime_generation(
        "DEEPSEEK",
        weights.model_dtype,
        weights,
        cache,
        first_token,
        sampler,
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
    sampler: crate::decode::DecodeSampler,
    log_stats: bool,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    crate::decode::generate_from_primed_sample(
        "DEEPSEEK",
        weights,
        cache,
        current_y,
        max_tokens,
        sampler,
        log_stats,
        on_token,
        forward_step,
    )
}

/// Run one benchmark trial on the canonical DeepSeek native path.
pub fn benchmark_trial(
    weights: &NativeWeights,
    prompt_ids: &[u32],
    generation_tokens: usize,
) -> crate::decode::BenchmarkTrial {
    crate::inline_array::reset_peak_memory();
    let mut cache = NativeCache::new_empty(weights);

    let prompt_tic = std::time::Instant::now();
    let (first_tok, mut sampler) = prefill_first_token(
        weights,
        &mut cache,
        prompt_ids,
        crate::decode::SamplingParams::new(0.0),
    );
    let current_y =
        prime_generation_impl(weights, &mut cache, first_tok, &mut sampler, false, false);
    let prompt_secs = prompt_tic.elapsed().as_secs_f64();

    let generation_secs = if generation_tokens > 1 {
        let generation_tic = std::time::Instant::now();
        let (generated_tail, _) = generate_from_primed_sample_impl(
            weights,
            &mut cache,
            current_y,
            generation_tokens - 1,
            sampler,
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
/// `first_token` is the last token of the prompt (prefill already committed
/// to `cache`). Each call to `on_token` receives the sampled token ID and
/// returns `false` to stop early (EOS or other condition).
///
/// Returns all generated token IDs (not including `first_token`).
pub fn generate(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    max_tokens: usize,
    mut sampler: crate::decode::DecodeSampler,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    let current_y = prime_generation_impl(weights, cache, first_token, &mut sampler, true, true);
    generate_from_primed_sample_impl(
        weights, cache, current_y, max_tokens, sampler, true, on_token,
    )
}
