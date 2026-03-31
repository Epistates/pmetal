use std::sync::OnceLock;

use crate::InlineArray;

fn trace_decode_graph_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PMETAL_TRACE_DECODE_GRAPH").is_some())
}

fn trace_decode_graph(tag: &str, step: usize, logits: &InlineArray, sampled: &InlineArray) {
    let should_log = step < 2 || (step + 1) % 16 == 0;
    if !trace_decode_graph_enabled() || !should_log {
        return;
    }

    eprintln!(
        "[{tag}] decode_graph step={step} logits_descs={} sampled_descs={} logits_shape={:?}",
        crate::inline_array::graph_desc_count(logits),
        crate::inline_array::graph_desc_count(sampled),
        logits.shape()
    );
}

#[derive(Debug, Clone, Copy)]
pub struct BenchmarkTrial {
    pub prompt_secs: f64,
    pub generation_secs: f64,
    pub peak_memory_bytes: usize,
}

/// Convert token ids to the native `[1, T]` int32 prompt input expected by
/// bridge-backed forward passes.
pub fn prompt_tokens_to_input(input_ids: &[u32]) -> InlineArray {
    let ids_i32: Vec<i32> = input_ids.iter().map(|&t| t as i32).collect();
    InlineArray::from_i32_slice_shaped(&ids_i32, &[1, ids_i32.len() as i32])
}

/// Extract the last sequence-position logits from a `[B, T, vocab]` tensor.
pub fn last_token_logits(logits: &InlineArray) -> InlineArray {
    let b = logits.dim(0);
    let t = logits.dim(1);
    let vocab = logits.dim(2);
    logits
        .slice(&[0, t - 1, 0], &[b, t, vocab])
        .reshape(&[b, vocab])
}

/// Create an f32 scalar cast to a specific MLX dtype.
///
/// Any scalar introduced into a decode graph must match the surrounding tensor
/// dtype unless the reference MLX path intentionally keeps it in float32.
#[inline(always)]
pub fn scalar_f32_dtype(value: f32, dtype: i32) -> InlineArray {
    InlineArray::from_f32(value).as_dtype(dtype)
}

/// Create an f32 scalar that matches the dtype of an existing tensor.
#[inline(always)]
pub fn scalar_f32_like(value: f32, like: &InlineArray) -> InlineArray {
    scalar_f32_dtype(value, like.dtype_raw())
}

/// Shared temperature sampling helper for bridge-backed decode paths.
///
/// The inverse-temperature scalar is cast to the logits dtype before the
/// multiply so bf16/f16 decode graphs do not get silently promoted to f32.
pub fn sample_token(logits_2d: &InlineArray, temperature: f32) -> InlineArray {
    if temperature <= 0.0 {
        logits_2d.argmax(-1)
    } else {
        let inv_temp = scalar_f32_like(1.0 / temperature, logits_2d);
        let lse = logits_2d.logsumexp(-1, true);
        let log_probs = logits_2d.subtract(&lse);
        let scaled = log_probs.multiply(&inv_temp);
        scaled.categorical()
    }
}

/// Sample a single token id from `[B, vocab]` logits.
pub fn sample_token_id(logits_2d: &InlineArray, temperature: f32) -> u32 {
    let tok_arr = sample_token(logits_2d, temperature);
    tok_arr.eval();
    tok_arr.item_u32()
}

/// Run prompt prefill and return the first sampled token.
pub fn prefill_first_token<Weights, Cache>(
    weights: &Weights,
    cache: &mut Cache,
    input_ids: &[u32],
    temperature: f32,
    mut forward_step: impl FnMut(&Weights, &InlineArray, &mut Cache) -> InlineArray,
) -> u32 {
    let prompt = prompt_tokens_to_input(input_ids);
    let logits = forward_step(weights, &prompt, cache);
    let last_logits = last_token_logits(&logits);
    sample_token_id(&last_logits, temperature)
}

/// Match MLX-LM's cache-aware causal-attention behavior.
///
/// Upstream uses `mask=None` for single-token decode (`N == 1`) and `"causal"`
/// for multi-token prefill. Keeping decode on the unmasked fast path matters
/// for apples-to-apples performance against `mlx-lm`.
#[inline(always)]
pub fn sdpa_causal_like_mlx(
    queries: &InlineArray,
    keys: &InlineArray,
    values: &InlineArray,
    scale: f32,
    query_len: i32,
) -> InlineArray {
    if query_len == 1 {
        queries.sdpa_with_mask(keys, values, scale, None)
    } else {
        queries.sdpa(keys, values, scale, "causal")
    }
}

/// Shared generation-session setup for bridge-native decode loops.
///
/// `mlx::core::enable_compile()` was benchmarked and shown to regress decode
/// throughput on the active native paths, so the canonical bridge path keeps
/// it disabled here.
fn begin_generation_session_impl(
    tag: &str,
    model_dtype: i32,
    reset_peak_memory: bool,
    log_session: bool,
) {
    crate::inline_array::clear_cache();
    if reset_peak_memory {
        crate::inline_array::reset_peak_memory();
    }
    static GENERATION_STREAM_INIT: std::sync::Once = std::sync::Once::new();
    GENERATION_STREAM_INIT.call_once(crate::inline_array::new_generation_stream);
    crate::inline_array::set_generation_stream();
    crate::inline_array::set_wired_limit_max();

    if log_session {
        eprintln!(
            "[{tag}] generate: dtype={model_dtype} active={:.0}MB",
            crate::inline_array::get_active_memory() as f64 / 1e6,
        );
    }
}

pub fn begin_generation_session(tag: &str, model_dtype: i32) {
    begin_generation_session_impl(tag, model_dtype, true, true);
}

pub fn begin_generation_session_preserve_peak(tag: &str, model_dtype: i32) {
    begin_generation_session_impl(tag, model_dtype, false, true);
}

pub fn begin_generation_session_preserve_peak_silent(tag: &str, model_dtype: i32) {
    begin_generation_session_impl(tag, model_dtype, false, false);
}

/// Prime a decode loop by preparing the cache, running one forward step, and
/// asynchronously sampling the first decode token.
pub fn prime_generation<Weights, Cache>(
    tag: &str,
    model_dtype: i32,
    weights: &Weights,
    cache: &mut Cache,
    first_token: u32,
    temperature: f32,
    reset_peak_memory: bool,
    log_session: bool,
    mut prepare_cache: impl FnMut(&mut Cache),
    mut forward_step: impl FnMut(&Weights, &InlineArray, &mut Cache) -> InlineArray,
) -> InlineArray {
    begin_generation_session_impl(tag, model_dtype, reset_peak_memory, log_session);
    prepare_cache(cache);

    let input_token = InlineArray::from_i32(first_token as i32).reshape(&[1, 1]);
    let logits = forward_step(weights, &input_token, cache);
    let logits_2d = logits.squeeze(1);
    let current_y = sample_token(&logits_2d, temperature);
    current_y.async_eval_ref();
    current_y
}

/// Continue generation from an already-primed async sample.
pub fn generate_from_primed_sample<Weights, Cache>(
    tag: &str,
    weights: &Weights,
    cache: &mut Cache,
    mut current_y: InlineArray,
    max_tokens: usize,
    temperature: f32,
    log_stats: bool,
    mut on_token: impl FnMut(u32) -> bool,
    mut forward_step: impl FnMut(&Weights, &InlineArray, &mut Cache) -> InlineArray,
) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(max_tokens);
    let mut step_times: Vec<f64> = Vec::new();

    for step in 0..max_tokens {
        let next_y = if step + 1 < max_tokens {
            let t_step = std::time::Instant::now();
            let next_input = current_y.reshape(&[1, 1]);
            let next_logits = forward_step(weights, &next_input, cache);
            let next_logits_2d = next_logits.squeeze(1);
            let next_y = sample_token(&next_logits_2d, temperature);
            trace_decode_graph(tag, step, &next_logits_2d, &next_y);
            next_y.async_eval_ref();
            step_times.push(t_step.elapsed().as_secs_f64() * 1000.0);
            Some(next_y)
        } else {
            None
        };

        if step == 0 {
            current_y.eval();
        }
        let token_val = current_y.item_u32();

        tokens.push(token_val);
        if !on_token(token_val) {
            break;
        }
        let Some(next_y) = next_y else {
            break;
        };
        current_y = next_y;

        if step % 256 == 255 {
            crate::inline_array::clear_cache();
        }
    }

    if log_stats && step_times.len() > 20 {
        step_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let skip = 10;
        let avg = step_times[skip..].iter().sum::<f64>() / (step_times.len() - skip) as f64;
        let p50 = step_times[step_times.len() / 2];
        eprintln!(
            "[{tag}] per-step: avg={avg:.2}ms p50={p50:.2}ms = {:.0} tok/s",
            1000.0 / avg
        );
    }

    crate::inline_array::synchronize();
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_like_matches_reference_dtype() {
        let like = InlineArray::from_f32(1.0).as_dtype(crate::compat::Dtype::Bfloat16.as_i32());
        let scalar = scalar_f32_like(0.5, &like);
        assert_eq!(scalar.dtype_raw(), like.dtype_raw());
    }

    #[test]
    fn prompt_tokens_to_input_uses_single_batch_i32_layout() {
        let prompt = prompt_tokens_to_input(&[11, 22, 33]);
        assert_eq!(prompt.shape(), &[1, 3]);
        assert_eq!(prompt.dtype_raw(), crate::compat::Dtype::Int32.as_i32());
    }

    #[test]
    fn last_token_logits_selects_final_sequence_position() {
        let logits = InlineArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
        let last = last_token_logits(&logits);
        assert_eq!(last.shape(), &[1, 3]);
        let first = last.slice(&[0, 0], &[1, 1]).item_f32();
        let third = last.slice(&[0, 2], &[1, 3]).reshape(&[1]).item_f32();
        assert_eq!(first, 4.0);
        assert_eq!(third, 6.0);
    }
}
