use std::collections::HashMap;
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

/// Per-token decode performance metrics returned from generation loops.
#[derive(Debug, Clone, Copy)]
pub struct DecodeMetrics {
    /// Tokens per second (excludes warmup steps).
    pub tok_per_sec: f64,
    /// Average milliseconds per decode step.
    pub avg_step_ms: f64,
    /// Median (p50) milliseconds per decode step.
    pub p50_step_ms: f64,
    /// Number of decode steps measured (excluding warmup).
    pub measured_steps: usize,
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

/// Sampling parameters for the bridge decode loop.
///
/// Includes the full mlx-lm-style filter pipeline: temperature, top-k,
/// top-p (nucleus), and min-p, plus repetition / frequency / presence
/// penalties. Defaults are no-ops for every filter so callers that only
/// care about temperature can use [`SamplingParams::new`] and ignore the
/// rest.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    /// Top-K filter. `0` disables. Keeps only the top `k` highest-logit
    /// tokens before sampling.
    pub top_k: usize,
    /// Top-P (nucleus) filter. `1.0` disables. Keeps the smallest set of
    /// tokens whose cumulative probability exceeds `1 - p`.
    pub top_p: f32,
    /// Min-P filter. `0.0` disables. Drops tokens with probability less
    /// than `min_p * top_token_probability`.
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

impl SamplingParams {
    pub fn new(temperature: f32) -> Self {
        Self {
            temperature,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }

    /// Whether any penalty is active (requires token counting).
    pub fn has_penalties(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
    }

    /// Whether any logit filter (top-k, top-p, min-p) is active.
    pub fn has_filters(&self) -> bool {
        self.top_k > 0 || (self.top_p < 1.0 && self.top_p > 0.0) || self.min_p > 0.0
    }
}

/// Apply the top-k filter: mask all logits except the `k` highest to
/// negative infinity. Uses `argpartition_axis` (O(n)) instead of a full
/// sort. Mirrors `pmetal-models::sampling::compiled_sampler::apply_top_k_2d`,
/// kept here so the bridge can do native-path filtered sampling without
/// taking a dependency on `pmetal-models`.
fn apply_top_k(logits_2d: &InlineArray, k: usize, neg_inf: &InlineArray) -> InlineArray {
    use crate::compat::ops::{argpartition_axis, put_along_axis, slice_last_from};
    let vocab = logits_2d.dim(-1) as usize;
    let k = k.min(vocab).max(1);
    // argpartition on -logits → first k indices are the top-k positions.
    // We want the indices of the *bottom* (vocab - k) positions to mask
    // out, which is everything past index k-1 along the partitioned axis.
    let neg = logits_2d.negative();
    let part = argpartition_axis(&neg, (k - 1) as i32, -1);
    let mask_idx = slice_last_from(&part, k as i32);
    put_along_axis(logits_2d, &mask_idx, neg_inf, -1)
}

/// Apply the top-p (nucleus) filter: keep the smallest set of tokens
/// whose cumulative probability exceeds `p`. Mirrors
/// `compiled_sampler::apply_top_p_2d`.
fn apply_top_p(logits_2d: &InlineArray, p: f32, neg_inf: &InlineArray) -> InlineArray {
    use crate::compat::ops::{
        argsort_axis, cumsum, exp, put_along_axis, take_along_axis, which, zeros_like,
    };
    let vocab = logits_2d.dim(-1) as usize;
    let probs = exp(logits_2d);
    let sorted_indices = argsort_axis(logits_2d, -1);
    let sorted_probs = take_along_axis(&probs, &sorted_indices, -1);
    let cumulative = cumsum(&sorted_probs, -1);
    // Restore original ordering of the cumulative probabilities.
    let vocab_range = InlineArray::from_i32_slice_shaped(
        &(0..vocab as i32).collect::<Vec<_>>(),
        &[1, vocab as i32],
    );
    let inverse_indices = put_along_axis(
        &zeros_like(&sorted_indices),
        &sorted_indices,
        &vocab_range,
        -1,
    );
    let cumulative = take_along_axis(&cumulative, &inverse_indices, -1);
    let threshold = scalar_f32_like(1.0 - p, logits_2d);
    let keep = cumulative.greater(&threshold);
    which(&keep, logits_2d, neg_inf)
}

/// Apply the min-p filter: drop tokens whose probability is less than
/// `min_p * top_token_probability`. Mirrors
/// `compiled_sampler::apply_min_p_2d`.
fn apply_min_p(logits_2d: &InlineArray, min_p: f32, neg_inf: &InlineArray) -> InlineArray {
    use crate::compat::ops::{
        argsort_axis, put_along_axis, slice_axis, take_along_axis, which, zeros_like,
    };
    let vocab = logits_2d.dim(-1) as usize;
    let neg = logits_2d.negative();
    let sorted_indices = argsort_axis(&neg, -1);
    let sorted_logits = take_along_axis(logits_2d, &sorted_indices, -1);
    let top_logits = slice_axis(&sorted_logits, -1, 0, 1);
    let log_min_p = scalar_f32_like(min_p.ln(), logits_2d);
    let scaled = top_logits.add(&log_min_p);
    let drop_mask = sorted_logits.less(&scaled);
    let selected = which(&drop_mask, neg_inf, &sorted_logits);
    let vocab_range = InlineArray::from_i32_slice_shaped(
        &(0..vocab as i32).collect::<Vec<_>>(),
        &[1, vocab as i32],
    );
    let inverse_indices = put_along_axis(
        &zeros_like(&sorted_indices),
        &sorted_indices,
        &vocab_range,
        -1,
    );
    take_along_axis(&selected, &inverse_indices, -1)
}

/// Reusable buffer for penalty application.
///
/// Pre-allocates a vocab-sized vec once and tracks dirty indices so each step
/// only zeros the positions that were previously set, instead of re-allocating
/// and zeroing the entire ~1 MB buffer every decode step.
struct PenaltyBuffer {
    vec: Vec<f32>,
    dirty: Vec<usize>,
}

impl PenaltyBuffer {
    fn new(vocab_size: usize) -> Self {
        Self {
            vec: vec![0.0f32; vocab_size],
            dirty: Vec::with_capacity(256),
        }
    }

    /// Apply penalties and return penalized logits. Reuses the internal buffer.
    fn apply(
        &mut self,
        logits_2d: &InlineArray,
        token_counts: &HashMap<u32, usize>,
        params: &SamplingParams,
    ) -> InlineArray {
        // Clear only previously-dirty positions (O(unique_tokens) not O(vocab))
        for &idx in &self.dirty {
            self.vec[idx] = 0.0;
        }
        self.dirty.clear();

        let vocab_size = self.vec.len();
        for (&token, &count) in token_counts {
            let idx = token as usize;
            if idx >= vocab_size {
                continue;
            }
            // Frequency + presence (subtractive): logit -= freq*count + pres*(count>0)
            self.vec[idx] = params.frequency_penalty * count as f32 + params.presence_penalty;
            self.dirty.push(idx);
        }

        let pen_arr = InlineArray::from_slice(&self.vec, &[1, vocab_size as i32]);
        logits_2d.subtract(&pen_arr)
    }
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

/// Sample a token from `[B, vocab]` logits using the full filter
/// pipeline in `params`: temperature → top-k → top-p → min-p →
/// categorical. At `temperature == 0` the call collapses to argmax and
/// every filter is a no-op. Otherwise the filter chain matches mlx-lm's
/// per-step sampling pipeline. Use [`sample_token`] when only
/// temperature matters.
///
/// IMPORTANT: penalty (`repetition_penalty` / `frequency_penalty` /
/// `presence_penalty`) handling lives in [`PenaltyBuffer::apply`] which
/// must run BEFORE this call — penalties operate on raw logits, not
/// log-probs, and depend on per-token counts the sampler does not track.
pub fn sample_token_with_params(
    logits_2d: &InlineArray,
    params: &SamplingParams,
) -> InlineArray {
    if params.temperature <= 0.0 {
        return logits_2d.argmax(-1);
    }
    let log_probs = {
        let inv_temp = scalar_f32_like(1.0 / params.temperature, logits_2d);
        let lse = logits_2d.logsumexp(-1, true);
        let scaled = logits_2d.subtract(&lse).multiply(&inv_temp);
        scaled
    };
    if !params.has_filters() {
        return log_probs.categorical();
    }
    // Pre-construct the negative-infinity sentinel once per call. Filters
    // splat it into masked positions; sample_token is called per step so
    // the alloc is small but unavoidable. mlx-lm caches it on a sampler
    // struct — same opportunity exists if/when we factor out a Sampler.
    let neg_inf = scalar_f32_like(f32::NEG_INFINITY, &log_probs);
    let mut filtered = log_probs;
    if params.top_k > 0 {
        filtered = apply_top_k(&filtered, params.top_k, &neg_inf);
    }
    if params.top_p < 1.0 && params.top_p > 0.0 {
        filtered = apply_top_p(&filtered, params.top_p, &neg_inf);
    }
    if params.min_p > 0.0 && params.min_p < 1.0 {
        filtered = apply_min_p(&filtered, params.min_p, &neg_inf);
    }
    filtered.categorical()
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

/// Quantized scaled-dot-product attention using MLX's fused `quantized_matmul`.
///
/// Matches mlx-lm's `quantized_scaled_dot_product_attention` (base.py:64-105).
/// K/V are stored as `(packed_uint32, scales, biases)` tuples and never fully
/// dequantized — `quantized_matmul` dequantizes inside the Metal kernel during
/// the matmul, yielding zero overhead vs standard SDPA.
///
/// - `queries`: `[B, n_q_heads, L, D]`
/// - `q_keys` / `q_values`: `(packed, scales, biases)` where packed is uint32
/// - For GQA (`n_q_heads > n_kv_heads`): queries are reshaped and quantized
///   tuples are broadcast, matching upstream behavior exactly.
#[allow(clippy::too_many_arguments)]
pub fn quantized_sdpa(
    queries: &InlineArray,
    q_keys: (&InlineArray, &InlineArray, &InlineArray),
    q_values: (&InlineArray, &InlineArray, &InlineArray),
    scale: f32,
    query_len: i32,
    n_q_heads: i32,
    n_kv_heads: i32,
    group_size: i32,
    bits: i32,
) -> InlineArray {
    let b = queries.dim(0);
    let l = queries.dim(2);
    let d = queries.dim(3);
    let n_repeats = n_q_heads / n_kv_heads;

    // Scale queries (matches: queries *= scale)
    let scale_arr = InlineArray::from_f32(scale).as_dtype(queries.dtype_raw());
    let queries_scaled = queries.multiply(&scale_arr);

    // GQA expansion: reshape queries [B, n_kv, n_rep, L, D], expand quantized tuples
    let (queries_work, k_packed, k_scales, k_biases, v_packed, v_scales, v_biases) =
        if n_repeats > 1 {
            let q = queries_scaled.reshape(&[b, n_kv_heads, n_repeats, l, d]);
            // expand_dims(-3) on each quantized component
            let kp = q_keys.0.expand_dims(-3);
            let ks = q_keys.1.expand_dims(-3);
            let kb = q_keys.2.expand_dims(-3);
            let vp = q_values.0.expand_dims(-3);
            let vs = q_values.1.expand_dims(-3);
            let vb = q_values.2.expand_dims(-3);
            (q, kp, ks, kb, vp, vs, vb)
        } else {
            (
                queries_scaled,
                q_keys.0.clone(),
                q_keys.1.clone(),
                q_keys.2.clone(),
                q_values.0.clone(),
                q_values.1.clone(),
                q_values.2.clone(),
            )
        };

    // Score: Q @ K^T via quantized_matmul (fused dequant inside Metal kernel)
    let scores = queries_work.quantized_matmul(
        &k_packed,
        &k_scales,
        Some(&k_biases),
        true, // transpose=true for Q @ K^T
        group_size,
        bits,
    );

    // Apply causal mask for prefill, no mask for decode
    let scores = if query_len > 1 {
        // Build causal mask: q_indices[:, None] >= k_indices[None]
        let kl = scores.dim(-1); // total KV length
        let ql = scores.dim(-2); // query length
        let dtype_i32 = crate::compat::Dtype::Int32.as_i32();
        let k_indices = InlineArray::arange(kl, dtype_i32);
        // q_indices = arange(ql) + (kl - ql)
        let offset = InlineArray::from_i32(kl - ql);
        let q_indices = InlineArray::arange(ql, dtype_i32).add(&offset);
        let q_col = q_indices.reshape(&[ql, 1]);
        let k_row = k_indices.reshape(&[1, kl]);
        let mask = q_col.greater_equal(&k_row);
        let neg_inf = InlineArray::from_f32(f32::NEG_INFINITY).as_dtype(scores.dtype_raw());
        mask.where_cond(&scores, &neg_inf)
    } else {
        scores
    };

    // Softmax with precise=true
    let weights = scores.softmax_precise(-1);

    // Value aggregation: weights @ V via quantized_matmul (fused dequant)
    let out = weights.quantized_matmul(
        &v_packed,
        &v_scales,
        Some(&v_biases),
        false, // transpose=false for weights @ V
        group_size,
        bits,
    );

    // Reshape back for GQA
    if n_repeats > 1 {
        out.reshape(&[b, n_q_heads, l, d])
    } else {
        out
    }
}

/// Mixed-bit quantized SDPA for TurboQuant v2 presets (Q2.5 / Q3.5).
///
/// Splits queries into outlier and regular channels, computes attention scores
/// via two `quantized_matmul` calls (one per group), sums them, then applies
/// softmax. Values are also split and aggregated separately, then concatenated
/// back to reconstruct the full `[B, H, L, D]` output.
///
/// Channel layout (post-permutation): `[outlier_count | head_dim - outlier_count]`
///
/// - `queries`: `[B, n_q_heads, L, D]`
/// - `q_keys_lo` / `q_values_lo`: regular-channel (lower-bit) `(packed, scales, biases)`
/// - `q_keys_hi` / `q_values_hi`: outlier-channel (higher-bit) `(packed, scales, biases)`
#[allow(clippy::too_many_arguments)]
pub fn quantized_sdpa_mixed(
    queries: &InlineArray,
    q_keys_lo: (&InlineArray, &InlineArray, &InlineArray),
    q_values_lo: (&InlineArray, &InlineArray, &InlineArray),
    q_keys_hi: (&InlineArray, &InlineArray, &InlineArray),
    q_values_hi: (&InlineArray, &InlineArray, &InlineArray),
    scale: f32,
    query_len: i32,
    n_q_heads: i32,
    n_kv_heads: i32,
    outlier_count: i32,
    group_size: i32,
    bits_lo: i32,
    bits_hi: i32,
) -> InlineArray {
    let b = queries.dim(0);
    let l = queries.dim(2);
    let d = queries.dim(3);
    let n_repeats = n_q_heads / n_kv_heads;
    let rc = d - outlier_count; // regular channel count

    // Scale queries
    let scale_arr = InlineArray::from_f32(scale).as_dtype(queries.dtype_raw());
    let queries_scaled = queries.multiply(&scale_arr);

    // Split scaled queries along last dim: outlier (hi) and regular (lo) halves
    let q_hi_raw = queries_scaled.slice(&[0, 0, 0, 0], &[b, n_q_heads, l, outlier_count]);
    let q_lo_raw = queries_scaled.slice(&[0, 0, 0, outlier_count], &[b, n_q_heads, l, d]);

    // GQA expansion
    let (
        q_hi,
        q_lo,
        kp_hi,
        ks_hi,
        kb_hi,
        vp_hi,
        vs_hi,
        vb_hi,
        kp_lo,
        ks_lo,
        kb_lo,
        vp_lo,
        vs_lo,
        vb_lo,
    ) = if n_repeats > 1 {
        let q_h = q_hi_raw.reshape(&[b, n_kv_heads, n_repeats, l, outlier_count]);
        let q_l = q_lo_raw.reshape(&[b, n_kv_heads, n_repeats, l, rc]);
        (
            q_h,
            q_l,
            q_keys_hi.0.expand_dims(-3),
            q_keys_hi.1.expand_dims(-3),
            q_keys_hi.2.expand_dims(-3),
            q_values_hi.0.expand_dims(-3),
            q_values_hi.1.expand_dims(-3),
            q_values_hi.2.expand_dims(-3),
            q_keys_lo.0.expand_dims(-3),
            q_keys_lo.1.expand_dims(-3),
            q_keys_lo.2.expand_dims(-3),
            q_values_lo.0.expand_dims(-3),
            q_values_lo.1.expand_dims(-3),
            q_values_lo.2.expand_dims(-3),
        )
    } else {
        (
            q_hi_raw,
            q_lo_raw,
            q_keys_hi.0.clone(),
            q_keys_hi.1.clone(),
            q_keys_hi.2.clone(),
            q_values_hi.0.clone(),
            q_values_hi.1.clone(),
            q_values_hi.2.clone(),
            q_keys_lo.0.clone(),
            q_keys_lo.1.clone(),
            q_keys_lo.2.clone(),
            q_values_lo.0.clone(),
            q_values_lo.1.clone(),
            q_values_lo.2.clone(),
        )
    };

    // Scores: Q_hi @ K_hi^T + Q_lo @ K_lo^T (both fused-dequant inside Metal kernel)
    let scores_hi = q_hi.quantized_matmul(&kp_hi, &ks_hi, Some(&kb_hi), true, group_size, bits_hi);
    let scores_lo = q_lo.quantized_matmul(&kp_lo, &ks_lo, Some(&kb_lo), true, group_size, bits_lo);
    let scores = scores_hi.add(&scores_lo);

    // Causal mask for prefill
    let scores = if query_len > 1 {
        let kl = scores.dim(-1);
        let ql = scores.dim(-2);
        let dtype_i32 = crate::compat::Dtype::Int32.as_i32();
        let k_indices = InlineArray::arange(kl, dtype_i32);
        let offset = InlineArray::from_i32(kl - ql);
        let q_indices = InlineArray::arange(ql, dtype_i32).add(&offset);
        let q_col = q_indices.reshape(&[ql, 1]);
        let k_row = k_indices.reshape(&[1, kl]);
        let mask = q_col.greater_equal(&k_row);
        let neg_inf = InlineArray::from_f32(f32::NEG_INFINITY).as_dtype(scores.dtype_raw());
        mask.where_cond(&scores, &neg_inf)
    } else {
        scores
    };

    let weights = scores.softmax_precise(-1);

    // Value aggregation: split into outlier and regular, each via fused-dequant matmul
    let out_hi = weights.quantized_matmul(&vp_hi, &vs_hi, Some(&vb_hi), false, group_size, bits_hi);
    let out_lo = weights.quantized_matmul(&vp_lo, &vs_lo, Some(&vb_lo), false, group_size, bits_lo);

    // Concatenate outlier and regular value outputs along last dim
    let out = out_hi.kv_cache_append(&out_lo, -1);

    // Reshape back for GQA
    if n_repeats > 1 {
        out.reshape(&[b, n_q_heads, l, d])
    } else {
        out
    }
}

/// Quantized SDPA with QJL residual correction for unbiased key inner products.
///
/// Adds an additive correction term to attention scores computed from affine-
/// quantized keys. The correction makes `E[⟨q, k̃⟩] = ⟨q, k⟩` (unbiased),
/// using the 1-bit QJL sign vectors stored alongside the quantized cache.
///
/// # Correction formula
///
/// From TurboQuant Algorithm 2:
///
/// ```text
/// q_proj = queries @ S            [B, Hq, L, D]
/// correction_raw = q_proj @ qjl_signs^T  [B, Hq, L, KV_T]
/// correction = residual_norms^T * sqrt(π/2) / D * correction_raw
/// scores_corrected = scores_affine + correction
/// ```
///
/// # Arguments
///
/// - `queries`: `[B, n_q_heads, L, D]`
/// - `q_keys` / `q_values`: `(packed, scales, biases)` quantized KV tuples
/// - `qjl_signs`: `[B, n_kv_heads, KV_T, D]` — sign(S · residual), ±1.0
/// - `qjl_residual_norms`: `[B, n_kv_heads, KV_T, 1]` f32 — L2 norm of residual
/// - `qjl_s`: `[D, D]` Gaussian projection matrix S (model dtype)
#[allow(clippy::too_many_arguments)]
pub fn quantized_sdpa_with_qjl(
    queries: &InlineArray,
    q_keys: (&InlineArray, &InlineArray, &InlineArray),
    q_values: (&InlineArray, &InlineArray, &InlineArray),
    qjl_signs: &InlineArray,          // [B, Hkv, KV_T, D] ±1.0 model_dtype
    qjl_residual_norms: &InlineArray, // [B, Hkv, KV_T, 1] f32
    qjl_s: &InlineArray,              // [D, D] model_dtype
    scale: f32,
    query_len: i32,
    n_q_heads: i32,
    n_kv_heads: i32,
    group_size: i32,
    bits: i32,
) -> InlineArray {
    let b = queries.dim(0);
    let l = queries.dim(2);
    let d = queries.dim(3);
    let n_repeats = n_q_heads / n_kv_heads;

    // Scale queries (matches: queries *= scale)
    let scale_arr = InlineArray::from_f32(scale).as_dtype(queries.dtype_raw());
    let queries_scaled = queries.multiply(&scale_arr);

    // GQA expansion
    let (
        queries_work,
        k_packed,
        k_scales,
        k_biases,
        v_packed,
        v_scales,
        v_biases,
        signs_work,
        norms_work,
    ) = if n_repeats > 1 {
        let q = queries_scaled.reshape(&[b, n_kv_heads, n_repeats, l, d]);
        (
            q,
            q_keys.0.expand_dims(-3),
            q_keys.1.expand_dims(-3),
            q_keys.2.expand_dims(-3),
            q_values.0.expand_dims(-3),
            q_values.1.expand_dims(-3),
            q_values.2.expand_dims(-3),
            qjl_signs.expand_dims(-3),
            qjl_residual_norms.expand_dims(-3),
        )
    } else {
        (
            queries_scaled,
            q_keys.0.clone(),
            q_keys.1.clone(),
            q_keys.2.clone(),
            q_values.0.clone(),
            q_values.1.clone(),
            q_values.2.clone(),
            qjl_signs.clone(),
            qjl_residual_norms.clone(),
        )
    };

    // Affine attention scores: Q_scaled @ K^T via quantized_matmul
    let scores_affine = queries_work.quantized_matmul(
        &k_packed,
        &k_scales,
        Some(&k_biases),
        true,
        group_size,
        bits,
    );

    // QJL correction:
    //   q_proj = queries_work @ S      [B, Hq/Hkv, (n_rep,) L, D]
    //   correction_raw = q_proj @ signs^T  [B, Hq/Hkv, (n_rep,) L, KV_T]
    //
    // queries_work is [B, Hkv, n_rep, L, D] (GQA) or [B, Hq, L, D] (uniform).
    // We apply S as a linear map on the last dimension, then matmul with signs^T.
    //
    // signs_work is [B, Hkv, (n_rep,) KV_T, D] — transpose last two dims to get
    // [B, Hkv, (n_rep,) D, KV_T] for the matmul.
    // Project queries through S^T: q_proj = queries @ S^T
    // Signs were computed as sign(S · r), so the inner product estimate is:
    //   E[q^T s] ≈ sqrt(π/2) * q^T S^T sign(S r) / ||sign(S r)||
    // giving unbiased E[q · r] recovery after scaling by ||r||.
    let qjl_s_t = qjl_s.transpose_axes(&[1, 0]); // [D, D]
    let q_proj = queries_work.matmul(&qjl_s_t); // [*, L, D] @ [D, D] = [*, L, D]
    // signs^T: swap last two axes of signs_work
    let ndim_signs = signs_work.ndim();
    let signs_t = signs_work.transpose_axes(&{
        let mut axes: Vec<i32> = (0..ndim_signs).collect();
        let last = (ndim_signs - 1) as usize;
        let second_last = (ndim_signs - 2) as usize;
        axes.swap(last, second_last);
        axes
    }); // [*, D, KV_T]
    let correction_raw = q_proj.matmul(&signs_t); // [*, L, KV_T]

    // Scale factor: sqrt(π/2) / D (cast to query dtype for graph coherence)
    let qjl_factor = ((std::f32::consts::PI / 2.0).sqrt()) / (d as f32);
    let qjl_factor_arr = InlineArray::from_f32(qjl_factor).as_dtype(queries.dtype_raw());

    // norms_work is [B, Hkv, (n_rep,) KV_T, 1] f32.
    // We need [B, Hkv, (n_rep,) 1, KV_T] to broadcast against [*, L, KV_T].
    let ndim_norms = norms_work.ndim();
    let norms_t = norms_work.transpose_axes(&{
        let mut axes: Vec<i32> = (0..ndim_norms).collect();
        let last = (ndim_norms - 1) as usize;
        let second_last = (ndim_norms - 2) as usize;
        axes.swap(last, second_last);
        axes
    }); // [*, 1, KV_T] f32
    // Cast norms to query dtype for multiply
    let norms_t_cast = norms_t.as_dtype(queries.dtype_raw());
    let correction = correction_raw
        .multiply(&norms_t_cast)
        .multiply(&qjl_factor_arr);

    let scores = scores_affine.add(&correction);

    // Apply causal mask for prefill
    let scores = if query_len > 1 {
        let kl = scores.dim(-1);
        let ql = scores.dim(-2);
        let dtype_i32 = crate::compat::Dtype::Int32.as_i32();
        let k_indices = InlineArray::arange(kl, dtype_i32);
        let offset = InlineArray::from_i32(kl - ql);
        let q_indices = InlineArray::arange(ql, dtype_i32).add(&offset);
        let q_col = q_indices.reshape(&[ql, 1]);
        let k_row = k_indices.reshape(&[1, kl]);
        let mask = q_col.greater_equal(&k_row);
        let neg_inf = InlineArray::from_f32(f32::NEG_INFINITY).as_dtype(scores.dtype_raw());
        mask.where_cond(&scores, &neg_inf)
    } else {
        scores
    };

    let weights = scores.softmax_precise(-1);

    // Value aggregation
    let out = weights.quantized_matmul(
        &v_packed,
        &v_scales,
        Some(&v_biases),
        false,
        group_size,
        bits,
    );

    // Reshape back for GQA
    if n_repeats > 1 {
        out.reshape(&[b, n_q_heads, l, d])
    } else {
        out
    }
}

/// Shared generation-session setup for bridge-native decode loops.
///
/// `mlx::core::enable_compile()` was benchmarked and shown to regress decode
/// throughput on the active native paths, so the canonical bridge path keeps
/// it disabled here.
fn begin_generation_session_impl(
    tag: &str,
    _model_dtype: i32,
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

    let _ = (tag, log_session);
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
#[allow(clippy::too_many_arguments)]
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
#[allow(clippy::too_many_arguments)]
pub fn generate_from_primed_sample<Weights, Cache>(
    tag: &str,
    weights: &Weights,
    cache: &mut Cache,
    current_y: InlineArray,
    max_tokens: usize,
    temperature: f32,
    log_stats: bool,
    on_token: impl FnMut(u32) -> bool,
    forward_step: impl FnMut(&Weights, &InlineArray, &mut Cache) -> InlineArray,
) -> (Vec<u32>, Option<DecodeMetrics>) {
    generate_from_primed_sample_with_params(
        tag,
        weights,
        cache,
        current_y,
        max_tokens,
        SamplingParams::new(temperature),
        log_stats,
        on_token,
        forward_step,
    )
}

/// Continue generation with full sampling parameter control.
///
/// Like [`generate_from_primed_sample`] but accepts [`SamplingParams`] for
/// repetition, frequency, and presence penalties.
#[allow(clippy::too_many_arguments)]
pub fn generate_from_primed_sample_with_params<Weights, Cache>(
    tag: &str,
    weights: &Weights,
    cache: &mut Cache,
    mut current_y: InlineArray,
    max_tokens: usize,
    params: SamplingParams,
    log_stats: bool,
    mut on_token: impl FnMut(u32) -> bool,
    mut forward_step: impl FnMut(&Weights, &InlineArray, &mut Cache) -> InlineArray,
) -> (Vec<u32>, Option<DecodeMetrics>) {
    let mut tokens = Vec::with_capacity(max_tokens);
    let mut step_times: Vec<f64> = Vec::new();
    let use_penalties = params.has_penalties();
    let mut token_counts: HashMap<u32, usize> = if use_penalties {
        HashMap::new()
    } else {
        HashMap::with_capacity(0)
    };
    // Lazily initialised on first step that needs it (once vocab_size is known).
    let mut penalty_buf: Option<PenaltyBuffer> = None;

    for step in 0..max_tokens {
        let next_y = if step + 1 < max_tokens {
            let t_step = std::time::Instant::now();
            let next_input = current_y.reshape(&[1, 1]);
            let next_logits = forward_step(weights, &next_input, cache);
            let next_logits_2d = next_logits.squeeze(1);

            // Apply penalties before sampling (zero-cost when disabled or no
            // tokens seen yet; reuses pre-allocated buffer otherwise).
            // Route through the full filter pipeline only when any
            // filter is actually active — params.has_filters() is
            // false on the common greedy / temperature-only path so
            // we use the leaner sample_token and skip the
            // scalar_f32_like neg-inf construction inside
            // sample_token_with_params.
            let needs_filters = params.has_filters();
            let next_y = if use_penalties && !token_counts.is_empty() {
                let buf = penalty_buf
                    .get_or_insert_with(|| PenaltyBuffer::new(next_logits_2d.dim(-1) as usize));
                let penalized = buf.apply(&next_logits_2d, &token_counts, &params);
                if needs_filters {
                    sample_token_with_params(&penalized, &params)
                } else {
                    sample_token(&penalized, params.temperature)
                }
            } else if needs_filters {
                sample_token_with_params(&next_logits_2d, &params)
            } else {
                sample_token(&next_logits_2d, params.temperature)
            };

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
        if use_penalties {
            *token_counts.entry(token_val).or_insert(0) += 1;
        }
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

    let metrics = if log_stats && !step_times.is_empty() {
        step_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Skip warmup steps: 10 for long runs, proportionally fewer for short ones.
        let skip = step_times.len().min(10);
        let measured = &step_times[skip..];
        if measured.is_empty() {
            // Fewer steps than the skip window — use all of them.
            let avg = step_times.iter().sum::<f64>() / step_times.len() as f64;
            let p50 = step_times[step_times.len() / 2];
            Some(DecodeMetrics {
                tok_per_sec: 1000.0 / avg,
                avg_step_ms: avg,
                p50_step_ms: p50,
                measured_steps: step_times.len(),
            })
        } else {
            let avg = measured.iter().sum::<f64>() / measured.len() as f64;
            let p50 = step_times[step_times.len() / 2];
            Some(DecodeMetrics {
                tok_per_sec: 1000.0 / avg,
                avg_step_ms: avg,
                p50_step_ms: p50,
                measured_steps: measured.len(),
            })
        }
    } else {
        None
    };

    crate::inline_array::synchronize();
    // Restore the default stream before returning. InlineArray Drops happen
    // when the caller's weights/cache go out of scope — they must execute on
    // the main stream, not the generation stream, to avoid SIGSEGV from
    // Metal teardown racing with stream cleanup.
    crate::inline_array::reset_default_stream();
    (tokens, metrics)
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
