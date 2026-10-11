//! Prefill / prime / generate loops and benchmark trials.

use crate::InlineArray;
use crate::decode::{DecodeSampler, SamplingParams};
use crate::inline_array as bridge;

use super::cache::NativeCache;
use super::forward::forward_step;
use super::weights::NativeWeights;

/// Run prompt prefill and return the first sampled token and the
/// generation's sampler, which [`generate`] continues.
pub fn prefill_first_token(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    input_ids: &[u32],
    params: SamplingParams,
) -> (u32, DecodeSampler) {
    crate::decode::prefill_first_token(weights, cache, input_ids, params, forward_step)
}

// ============================================================================
// Generation loop
// ============================================================================

fn prepare_generation_cache(cache: &mut NativeCache, reserve_decode_inputs: i32, model_dtype: i32) {
    let trace_qwen35 = std::env::var_os("PMETAL_TRACE_QWEN35").is_some();
    if trace_qwen35 {
        eprintln!("[QWEN35 TRACE] begin_generation_session before_eval_and_detach");
    }
    cache.eval_and_detach_states();
    if trace_qwen35 {
        eprintln!("[QWEN35 TRACE] begin_generation_session after_eval_and_detach");
    }
    cache.reserve_decode_inputs(reserve_decode_inputs, model_dtype);
    if trace_qwen35 {
        eprintln!(
            "[QWEN35 TRACE] begin_generation_session after_reserve decode_inputs={reserve_decode_inputs}"
        );
    }
    if std::env::var_os("PMETAL_SKIP_CLEAR_CACHE").is_none() {
        bridge::clear_cache();
        if trace_qwen35 {
            eprintln!("[QWEN35 TRACE] begin_generation_session after_clear_cache");
        }
    } else if trace_qwen35 {
        eprintln!("[QWEN35 TRACE] begin_generation_session skipped_clear_cache");
    }
}

fn prime_generation_impl(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    reserve_decode_inputs: usize,
    sampler: &mut DecodeSampler,
    reset_peak_memory: bool,
    log_session: bool,
) -> InlineArray {
    let reserve_decode_inputs = reserve_decode_inputs.min(i32::MAX as usize) as i32;
    crate::decode::prime_generation(
        "NATIVE",
        weights.model_dtype,
        weights,
        cache,
        first_token,
        sampler,
        reset_peak_memory,
        log_session,
        |cache| prepare_generation_cache(cache, reserve_decode_inputs, weights.model_dtype),
        forward_step,
    )
}

fn generate_from_primed_sample_impl(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    current_y: InlineArray,
    max_tokens: usize,
    sampler: DecodeSampler,
    log_stats: bool,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    crate::decode::generate_from_primed_sample(
        "NATIVE",
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

/// Run one benchmark trial on the canonical Qwen native path.
///
/// The timing split: prompt timing includes prefill,
/// first-token sampling, and priming the next decode step; generation timing
/// covers only the remaining decode loop.
pub fn benchmark_trial(
    weights: &NativeWeights,
    prompt_ids: &[u32],
    generation_tokens: usize,
    turboquant: Option<crate::turboquant::TurboQuantConfig>,
) -> crate::decode::BenchmarkTrial {
    crate::inline_array::reset_peak_memory();
    let mut cache = NativeCache::new_with_turboquant(weights, turboquant);

    let prompt_tic = std::time::Instant::now();
    let (first_tok, mut sampler) =
        prefill_first_token(weights, &mut cache, prompt_ids, SamplingParams::new(0.0));
    let current_y = prime_generation_impl(
        weights,
        &mut cache,
        first_tok,
        generation_tokens.saturating_sub(1),
        &mut sampler,
        false,
        false,
    );
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
/// `first_token` is the token [`prefill_first_token`] sampled (the prompt is
/// already prefilled into `cache`) and `sampler` the one it returned. Each
/// call to `on_token` receives the sampled token ID and returns `false` to
/// stop early (e.g. on EOS).
///
/// Returns all generated token IDs (not including `first_token`).
pub fn generate(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    max_tokens: usize,
    mut sampler: DecodeSampler,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    let current_y = prime_generation_impl(
        weights,
        cache,
        first_token,
        max_tokens,
        &mut sampler,
        true,
        true,
    );
    generate_from_primed_sample_impl(
        weights, cache, current_y, max_tokens, sampler, true, on_token,
    )
}
