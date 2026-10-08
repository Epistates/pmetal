//! Prefill / prime / generate loops and benchmark trials.

use crate::InlineArray;
use crate::inline_array as bridge;

use super::Qwen3Config;
use super::cache::NativeCache;
use super::forward::forward_step;
use super::weights::NativeWeights;

pub fn prefill_first_token(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    input_ids: &[u32],
    temperature: f32,
) -> u32 {
    crate::decode::prefill_first_token(weights, cache, input_ids, temperature, forward_step)
}

// ============================================================================
// Generation loop
// ============================================================================

/// Run the full generation loop with async GPU pipelining.
///
/// `first_token` is the token at the end of the prompt (already prefilled into
/// `cache`). Each call to `on_token` receives the sampled token ID and returns
/// `false` to stop early (e.g. on EOS).
///
/// Returns all generated token IDs (not including `first_token`).
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
    temperature: f32,
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
        temperature,
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
    params: crate::decode::SamplingParams,
    log_stats: bool,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    crate::decode::generate_from_primed_sample_with_params(
        "NATIVE",
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

/// Prime the canonical decode loop without resetting peak memory.
///
/// This is used by `infer --benchmark` so the timing path shares the
/// same bridge decode implementation as live inference.
pub fn prime_generation_preserve_peak(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    reserve_decode_inputs: usize,
    temperature: f32,
) -> InlineArray {
    prime_generation_impl(
        weights,
        cache,
        first_token,
        reserve_decode_inputs,
        temperature,
        false,
        true,
    )
}

pub fn prime_generation_preserve_peak_silent(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    reserve_decode_inputs: usize,
    temperature: f32,
) -> InlineArray {
    prime_generation_impl(
        weights,
        cache,
        first_token,
        reserve_decode_inputs,
        temperature,
        false,
        false,
    )
}

/// Continue generation from an already-primed async sample.
///
/// `current_y` must come from [`prime_generation_preserve_peak`] or the
/// equivalent internal priming path.
pub fn generate_from_primed_sample(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    current_y: InlineArray,
    max_tokens: usize,
    temperature: f32,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    generate_from_primed_sample_impl(
        weights,
        cache,
        current_y,
        max_tokens,
        crate::decode::SamplingParams::new(temperature),
        true,
        on_token,
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
    let first_tok = prefill_first_token(weights, &mut cache, prompt_ids, 0.0);
    let current_y = prime_generation_preserve_peak_silent(
        weights,
        &mut cache,
        first_tok,
        generation_tokens.saturating_sub(1),
        0.0,
    );
    let prompt_secs = prompt_tic.elapsed().as_secs_f64();

    let generation_secs = if generation_tokens > 1 {
        let generation_tic = std::time::Instant::now();
        let generated_tail = generate_from_primed_sample_silent(
            weights,
            &mut cache,
            current_y,
            generation_tokens - 1,
            0.0,
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

pub fn generate_from_primed_sample_silent(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    current_y: InlineArray,
    max_tokens: usize,
    temperature: f32,
    on_token: impl FnMut(u32) -> bool,
) -> Vec<u32> {
    generate_from_primed_sample_impl(
        weights,
        cache,
        current_y,
        max_tokens,
        crate::decode::SamplingParams::new(temperature),
        false,
        on_token,
    )
    .0
}

pub fn generate(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    max_tokens: usize,
    params: crate::decode::SamplingParams,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    let current_y = prime_generation_impl(
        weights,
        cache,
        first_token,
        max_tokens,
        params.temperature,
        true,
        true,
    );
    generate_from_primed_sample_impl(
        weights, cache, current_y, max_tokens, params, true, on_token,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenDecodeBackend {
    RustBridge,
}

pub fn canonical_decode_backend(
    _config: &Qwen3Config,
    _turboquant: Option<crate::turboquant::TurboQuantConfig>,
) -> QwenDecodeBackend {
    QwenDecodeBackend::RustBridge
}

#[allow(clippy::too_many_arguments)]
pub fn generate_canonical(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    config: &Qwen3Config,
    first_token: u32,
    max_tokens: usize,
    params: crate::decode::SamplingParams,
    turboquant: Option<crate::turboquant::TurboQuantConfig>,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    match canonical_decode_backend(config, turboquant) {
        QwenDecodeBackend::RustBridge => {
            generate(weights, cache, first_token, max_tokens, params, on_token)
        }
    }
}

pub fn benchmark_trial_canonical(
    weights: &NativeWeights,
    config: &Qwen3Config,
    prompt_ids: &[u32],
    generation_tokens: usize,
    turboquant: Option<crate::turboquant::TurboQuantConfig>,
) -> crate::decode::BenchmarkTrial {
    match canonical_decode_backend(config, turboquant) {
        QwenDecodeBackend::RustBridge => {
            benchmark_trial(weights, prompt_ids, generation_tokens, turboquant)
        }
    }
}

pub fn generate_preserve_peak(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    max_tokens: usize,
    temperature: f32,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    let current_y = prime_generation_impl(
        weights,
        cache,
        first_token,
        max_tokens,
        temperature,
        false,
        true,
    );
    generate_from_primed_sample_impl(
        weights,
        cache,
        current_y,
        max_tokens,
        crate::decode::SamplingParams::new(temperature),
        true,
        on_token,
    )
}
