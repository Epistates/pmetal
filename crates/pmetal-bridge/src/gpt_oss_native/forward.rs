//! Full-model forward step + prefill/prime/generate wrappers.

use crate::InlineArray;
use crate::inline_array as bridge;

use super::attention::attn_forward;
use super::cache::NativeCache;
use super::moe::moe_forward;
use super::weights::NativeWeights;

/// Run one forward step — works for both T=1 decode and T=N prefill.
///
/// `token_ids` must be shape `[B, T]` int32.  Returns logits `[B, T, vocab]`.
///
/// Architecture:
///   For each layer:
///     1. input_layernorm (RMSNorm)
///     2. Attention (sliding or full causal) + residual
///     3. post_attention_layernorm (RMSNorm)
///     4. MoE (sigmoid top-k routing + SwiGLU experts + per-expert bias) + residual
///   Final norm → lm_head
pub fn forward_step(
    weights: &NativeWeights,
    token_ids: &InlineArray, // [B, T]
    cache: &mut NativeCache,
) -> InlineArray {
    let b = token_ids.dim(0);
    let s = token_ids.dim(1);
    let dtype = weights.model_dtype;

    // Embedding lookup: [B, T, hidden]
    // MLX >= 0.32: indices must be stop_gradient'd inside a grad trace
    let mut hidden = weights.embed_w.take_axis(&token_ids.stop_gradient(), 0);

    for (li, lw) in weights.layers.iter().enumerate() {
        // Input LayerNorm
        let normed = hidden.rms_norm(Some(&lw.input_ln_w), lw.input_ln_eps);

        // Attention
        let attn_out = attn_forward(
            lw,
            &normed,
            b,
            s,
            &mut cache.kv_caches[li],
            cache.rope_offset,
            dtype,
        );

        // Residual
        let h = hidden.add(&attn_out);

        // Post-attention LayerNorm
        let mlp_in = h.rms_norm(Some(&lw.post_ln_w), lw.post_ln_eps);

        // MoE
        let moe_out = moe_forward(lw, &mlp_in, b, s);

        // Residual
        hidden = h.add(&moe_out);
    }

    // Advance position counter
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
        "GPT-OSS",
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
        "GPT-OSS",
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

/// Run one MLX-LM-style benchmark trial on the canonical GPT-OSS native path.
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

/// Run the full GPT-OSS generation loop with async GPU pipelining.
///
/// `first_token` is the last token of the already-prefilled prompt.
/// `on_token(token_id)` is called after each decoded token; return `false` to
/// stop early (e.g. on EOS).
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
