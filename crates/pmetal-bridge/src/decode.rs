use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::InlineArray;

static COMPILED_DECODE: OnceLock<AtomicBool> = OnceLock::new();

fn compiled_decode_flag() -> &'static AtomicBool {
    COMPILED_DECODE.get_or_init(|| {
        AtomicBool::new(std::env::var_os("PMETAL_DISABLE_COMPILED_DECODE").is_none())
    })
}

/// Whether the native engines take their compiled single-token attention
/// graphs: on unless `PMETAL_DISABLE_COMPILED_DECODE` is set or
/// [`set_compiled_decode`] turned them off. Off, every step runs the per-op
/// path the graph replaces, which is how the parity tests and decode
/// benchmarks compare the two.
pub fn compiled_decode_enabled() -> bool {
    compiled_decode_flag().load(Ordering::Relaxed)
}

/// Turn the compiled single-token attention graphs on or off, process-wide.
/// See [`compiled_decode_enabled`].
pub fn set_compiled_decode(enabled: bool) {
    compiled_decode_flag().store(enabled, Ordering::Relaxed);
}

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

/// Convert token ids (a prompt, or tokens to feed or verify) to the `[1, T]`
/// int32 input bridge-backed forward passes take.
pub fn prompt_tokens_to_input(input_ids: &[u32]) -> InlineArray {
    let ids_i32: Vec<i32> = input_ids.iter().map(|&t| t as i32).collect();
    InlineArray::from_i32_slice_shaped(&ids_i32, &[1, ids_i32.len() as i32])
}

/// Extract the last sequence-position logits from a `[B, T, vocab]` tensor:
/// `[B, vocab]`.
pub fn last_token_logits(logits: &InlineArray) -> InlineArray {
    let b = logits.dim(0);
    let t = logits.dim(1);
    let vocab = logits.dim(2);
    logits
        .slice(&[0, t - 1, 0], &[b, t, vocab])
        .reshape(&[b, vocab])
}

/// The last sequence position of a `[B, T, D]` tensor (hidden states),
/// keeping the sequence axis: `[B, 1, D]`.
pub fn last_token_hidden(values: &InlineArray) -> InlineArray {
    let b = values.dim(0);
    let t = values.dim(1);
    let d = values.dim(2);
    values.slice(&[0, t - 1, 0], &[b, t, d])
}

/// The greedy token of `[1, vocab]` (or `[vocab]`) logits, read back to the
/// host.
pub fn greedy_token(logits: &InlineArray) -> u32 {
    logits.argmax(-1).item::<u32>()
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
/// Includes the full filter pipeline: temperature, top-k,
/// top-p (nucleus), and min-p, plus repetition / frequency / presence
/// penalties. Defaults are no-ops for every filter so callers that only
/// care about temperature can use [`SamplingParams::new`] and ignore the
/// rest. [`filtered_log_probs`] applies the filters, after the temperature.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    /// Temperature; at or below 0, greedy decoding.
    pub temperature: f32,
    /// Top-K filter. `0` disables. Keeps the tokens at least as likely as
    /// the `k`-th most likely.
    pub top_k: usize,
    /// Top-P (nucleus) filter. `1.0` (or `0.0`) disables. Keeps the most
    /// likely tokens whose probabilities sum to at least `p`.
    pub top_p: f32,
    /// Min-P filter. `0.0` disables. Drops tokens with probability less
    /// than `min_p * top_token_probability`.
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    /// Seed of the sampled draws: with one, the `n`-th sampled token is
    /// drawn with [`draw_key`]`(seed, n)`, so the same seed, prompt and
    /// model give the same tokens. `None` draws from the thread's MLX key.
    pub seed: Option<u64>,
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
            seed: None,
        }
    }

    /// The key for the `draw`-th sampled token, if these params are seeded.
    pub fn draw_key(&self, draw: u64) -> Option<InlineArray> {
        self.seed.map(|seed| draw_key(seed, draw))
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

/// The log-probabilities a sampled token is drawn from: `[B, vocab]` logits
/// in f32, scaled by the temperature, as log-probabilities, then filtered by
/// top-k, top-p and min-p, in that order. That is the order and the
/// semantics of transformers' sampling warpers (`TemperatureLogitsWarper`,
/// `TopKLogitsWarper`, `TopPLogitsWarper`, `MinPLogitsWarper`, each keeping
/// at least one token), so each filter reads the distribution the ones
/// before it left:
///
/// - top-k keeps the tokens at least as likely as the `k`-th most likely
///   (all of them on a tie);
/// - top-p keeps the most likely tokens whose probabilities, renormalized
///   after top-k, sum to at least `p`: it drops the least likely tokens
///   while their total stays at most `1 - p`, and on a tie at the cut keeps
///   every token as likely as the least likely one it keeps;
/// - min-p keeps the tokens at least `min_p` times as likely as the most
///   likely one.
///
/// A dropped token's log-probability is `-inf`; the rest are the tempered
/// distribution's, not renormalized over what is kept, which a categorical
/// draw doesn't need (a caller wanting probabilities subtracts their
/// `logsumexp`). A `temperature` at or below 0 is taken as 1; greedy
/// decoding doesn't come here ([`sample_token`]).
///
/// Every sampler in the crate graph draws from this: [`sample_token`] for
/// the native engines, and `pmetal_models`' `Sampler` and
/// `sampling_log_probs`, which speculative decoding verifies against.
pub fn filtered_log_probs(logits_2d: &InlineArray, params: &SamplingParams) -> InlineArray {
    let mut scores = logits_2d.as_dtype(crate::compat::Dtype::Float32.as_i32());
    if params.temperature > 0.0 && params.temperature != 1.0 {
        scores = scores.multiply(&InlineArray::from_f32(1.0 / params.temperature));
    }
    let mut log_probs = scores.subtract(&scores.logsumexp(-1, true));
    if !params.has_filters() {
        return log_probs;
    }
    let neg_inf = InlineArray::from_f32(f32::NEG_INFINITY);
    let vocab = log_probs.dim(-1);
    if params.top_k > 0 && (params.top_k as i32) < vocab {
        // The k-th largest value: the smallest of the top k.
        let kth = log_probs.topk(params.top_k as i32, -1).min_axis(-1, true);
        log_probs = log_probs.less(&kth).where_cond(&neg_inf, &log_probs);
    }
    if params.top_p > 0.0 && params.top_p < 1.0 {
        // Ascending, the dropped tokens are a prefix: those whose running
        // total of probability (renormalized over what top-k kept) is at
        // most 1 - p, short of the last, most likely, token.
        let order = log_probs.argsort(-1);
        let sorted = log_probs.take_along_axis(&order, -1);
        let renormalized = sorted.subtract(&sorted.logsumexp(-1, true));
        let below = renormalized
            .exp()
            .cumsum(-1)
            .less_equal(&InlineArray::from_f32(1.0 - params.top_p));
        let dropped = below
            .as_dtype(crate::compat::Dtype::Int32.as_i32())
            .sum_axes(&[-1], true)
            .minimum(&InlineArray::from_i32(vocab - 1));
        let cut = sorted.take_along_axis(&dropped, -1);
        log_probs = log_probs.less(&cut).where_cond(&neg_inf, &log_probs);
    }
    if params.min_p > 0.0 && params.min_p <= 1.0 {
        // p < min_p * p_max, in log space, where renormalizing cancels.
        let floor = log_probs
            .max_axis(-1, true)
            .add(&InlineArray::from_f32(params.min_p.ln()));
        log_probs = log_probs.less(&floor).where_cond(&neg_inf, &log_probs);
    }
    log_probs
}

/// The repetition penalty applied to the logits of tokens already in the
/// sequence: a positive logit is divided by `penalty`, a negative one
/// multiplied by it, so a penalty above 1 makes every repeat less likely
/// (CTRL, Keskar et al., arXiv 1909.05858; transformers'
/// `RepetitionPenaltyLogitsProcessor`).
pub fn repetition_penalized(logits: &InlineArray, penalty: f32) -> InlineArray {
    let penalty = scalar_f32_like(penalty, logits);
    let zero = scalar_f32_like(0.0, logits);
    logits
        .less(&zero)
        .where_cond(&logits.multiply(&penalty), &logits.divide(&penalty))
}

/// The repetition, frequency and presence penalties of one generation, with
/// the token history they read kept on the GPU.
///
/// The repetition penalty covers every token in the sequence, prompt
/// included, as transformers' `RepetitionPenaltyLogitsProcessor` does for a
/// decoder-only model; the frequency and presence penalties (OpenAI's
/// `logit -= frequency * count + presence * (count > 0)`) count the
/// generated tokens only. Repetition is applied first, as the samplers of
/// `pmetal-models` do.
///
/// [`record`](Self::record) folds a sampled token in without reading it
/// back, so a pipelined decode loop keeps its next step queued behind the
/// sample and every step still sees every token before it.
///
/// A clone shares the history's arrays and records on its own, so a
/// speculative decoder can follow its drafts on a clone and keep only the
/// tokens it accepts in the original.
#[derive(Clone)]
pub struct DecodePenalties {
    repetition: f32,
    frequency: f32,
    presence: f32,
    /// The prompt, until the first logits give the vocabulary size.
    prompt: Vec<u32>,
    /// Tokens recorded before the first logits.
    pending: Vec<InlineArray>,
    history: Option<PenaltyHistory>,
}

/// The token history the penalties read, as `[1, vocab]` masks.
#[derive(Clone)]
struct PenaltyHistory {
    /// `0..vocab`, int32.
    ids: InlineArray,
    /// Tokens in the prompt or generated: the repetition penalty's set.
    seen: InlineArray,
    /// How often each token was generated, f32.
    counts: InlineArray,
}

impl DecodePenalties {
    /// The penalties of a generation from `prompt`, or `None` when none is
    /// active (a repetition penalty of 1, frequency and presence 0).
    pub fn new(repetition: f32, frequency: f32, presence: f32, prompt: &[u32]) -> Option<Self> {
        (repetition != 1.0 || frequency != 0.0 || presence != 0.0).then(|| Self {
            repetition,
            frequency,
            presence,
            prompt: prompt.to_vec(),
            pending: Vec::new(),
            history: None,
        })
    }

    fn history(&mut self, vocab: i32) -> &mut PenaltyHistory {
        if self.history.is_none() {
            // Tokens past the logits' vocabulary have no logit to penalize.
            let mut prompt_mask = vec![0.0f32; vocab.max(0) as usize];
            for &token in &self.prompt {
                if let Some(slot) = prompt_mask.get_mut(token as usize) {
                    *slot = 1.0;
                }
            }
            self.prompt = Vec::new();
            let zero = InlineArray::from_f32(0.0);
            self.history = Some(PenaltyHistory {
                ids: InlineArray::arange(vocab, crate::compat::Dtype::Int32.as_i32())
                    .reshape(&[1, vocab]),
                seen: InlineArray::from_f32_slice(&prompt_mask, &[1, vocab]).greater(&zero),
                counts: InlineArray::zeros(&[1, vocab], crate::compat::Dtype::Float32.as_i32()),
            });
            for token in std::mem::take(&mut self.pending) {
                self.history.as_mut().unwrap().record(&token);
            }
        }
        self.history.as_mut().unwrap()
    }

    /// `[1, vocab]` logits with the penalties applied, in f32.
    pub fn apply(&mut self, logits_2d: &InlineArray) -> InlineArray {
        let (repetition, frequency, presence) = (self.repetition, self.frequency, self.presence);
        let logits = logits_2d.as_dtype(crate::compat::Dtype::Float32.as_i32());
        let history = self.history(logits.dim(-1));
        let mut out = logits;
        if repetition != 1.0 {
            out = history
                .seen
                .where_cond(&repetition_penalized(&out, repetition), &out);
        }
        if frequency != 0.0 || presence != 0.0 {
            let generated = history.counts.greater(&InlineArray::from_f32(0.0));
            let penalty = history
                .counts
                .multiply(&InlineArray::from_f32(frequency))
                .add(
                    &generated
                        .as_dtype(crate::compat::Dtype::Float32.as_i32())
                        .multiply(&InlineArray::from_f32(presence)),
                );
            out = out.subtract(&penalty);
        }
        out
    }

    /// Add a generated token (a one-element array, read lazily) to the
    /// history.
    pub fn record(&mut self, token: &InlineArray) {
        match self.history.as_mut() {
            Some(history) => history.record(token),
            None => self.pending.push(token.clone()),
        }
    }

    /// [`record`](Self::record) a token the host already holds.
    pub fn record_id(&mut self, token: u32) {
        self.record(&InlineArray::from_slice(&[token], &[1]));
    }
}

impl PenaltyHistory {
    fn record(&mut self, token: &InlineArray) {
        let token = token
            .reshape(&[1, 1])
            .as_dtype(crate::compat::Dtype::Int32.as_i32());
        let hit = self.ids.equal(&token);
        self.seen = self.seen.logical_or(&hit);
        self.counts = self
            .counts
            .add(&hit.as_dtype(crate::compat::Dtype::Float32.as_i32()));
    }
}

/// The sampler of one native generation: its [`SamplingParams`], its
/// penalties over the prompt and every token drawn so far, and the index of
/// its next draw.
///
/// Every token of the generation comes from [`sample`](Self::sample): the
/// prompt's last position (draw 0), the primed step (draw 1), and each step
/// of the decode loop after them, so each is penalized for all the tokens
/// before it.
pub struct DecodeSampler {
    params: SamplingParams,
    penalties: Option<DecodePenalties>,
    draws: u64,
}

impl DecodeSampler {
    /// The sampler of a generation continuing `prompt`.
    pub fn new(params: SamplingParams, prompt: &[u32]) -> Self {
        let penalties = DecodePenalties::new(
            params.repetition_penalty,
            params.frequency_penalty,
            params.presence_penalty,
            prompt,
        );
        Self {
            params,
            penalties,
            draws: 0,
        }
    }

    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Sample the next token from `[1, vocab]` logits ([`sample_token`]
    /// after the penalties), without reading it back.
    pub fn sample(&mut self, logits_2d: &InlineArray) -> InlineArray {
        let draw = self.draws;
        self.draws += 1;
        match self.penalties.as_mut() {
            Some(penalties) => {
                let token = sample_token(&penalties.apply(logits_2d), &self.params, draw);
                penalties.record(&token);
                token
            }
            None => sample_token(logits_2d, &self.params, draw),
        }
    }
}

/// The MLX PRNG key for draw `draw` of a generation seeded with `seed`.
///
/// It is the key for the `draw`-th output of a SplitMix64 sequence started
/// at `seed` (Steele, Lea and Flood, *Fast Splittable Pseudorandom Number
/// Generators*, OOPSLA 2014), so every sampled token has its own key and no
/// key state is threaded through the decode loop: a seeded generation picks
/// the same tokens whichever thread runs it and whatever else draws random
/// numbers meanwhile, which the thread's shared MLX key cannot promise.
pub fn draw_key(seed: u64, draw: u64) -> InlineArray {
    InlineArray::random_key(draw_bits(seed, draw))
}

/// The `draw`-th output of the SplitMix64 sequence started at `seed`: the
/// bits [`draw_key`] makes its key from, for a draw taken on the host.
pub fn draw_bits(seed: u64, draw: u64) -> u64 {
    let mut z = seed.wrapping_add(draw.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Sample a token from `[B, vocab]` logits under `params`: a categorical
/// draw from [`filtered_log_probs`] (temperature, then top-k, top-p and
/// min-p), with the key for `draw` (the token's index in the generated text)
/// when `params` has a seed. At `temperature == 0` it is argmax.
///
/// The penalties (`repetition_penalty`, `frequency_penalty`,
/// `presence_penalty`) are not applied here: they act on the raw logits and
/// read the token history, which [`DecodeSampler`] keeps.
pub fn sample_token(logits_2d: &InlineArray, params: &SamplingParams, draw: u64) -> InlineArray {
    if params.temperature <= 0.0 {
        return logits_2d.argmax(-1);
    }
    filtered_log_probs(logits_2d, params).categorical(-1, params.draw_key(draw).as_ref())
}

/// Run prompt prefill and return the first sampled token (draw 0) with the
/// generation's sampler, which [`prime_generation`] and
/// [`generate_from_primed_sample`] continue.
pub fn prefill_first_token<Weights, Cache>(
    weights: &Weights,
    cache: &mut Cache,
    input_ids: &[u32],
    params: SamplingParams,
    mut forward_step: impl FnMut(&Weights, &InlineArray, &mut Cache) -> InlineArray,
) -> (u32, DecodeSampler) {
    let mut sampler = DecodeSampler::new(params, input_ids);
    let prompt = prompt_tokens_to_input(input_ids);
    let logits = forward_step(weights, &prompt, cache);
    let last_logits = last_token_logits(&logits);
    let token = sampler.sample(&last_logits);
    token.eval();
    (token.item_u32(), sampler)
}

/// Cache-aware causal attention.
///
/// Uses `mask=None` for single-token decode (`N == 1`) and `"causal"`
/// for multi-token prefill. Keeping decode on the unmasked fast path matters
/// for decode throughput.
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

/// Checked variant of [`sdpa_causal_like_mlx`] for paths that must not accept
/// the bridge's scalar-zero sentinel after a C++ exception.
#[inline(always)]
pub fn try_sdpa_causal_like_mlx(
    queries: &InlineArray,
    keys: &InlineArray,
    values: &InlineArray,
    scale: f32,
    query_len: i32,
) -> crate::error::BridgeResult<InlineArray> {
    let output = sdpa_causal_like_mlx(queries, keys, values, scale, query_len);
    crate::error::check_last_error()?;
    Ok(output)
}

/// Quantized scaled-dot-product attention using MLX's fused `quantized_matmul`.
///
/// Same algorithm as the reference `quantized_scaled_dot_product_attention`.
/// K/V are stored as `(packed_uint32, scales, biases)` tuples and never fully
/// dequantized — `quantized_matmul` dequantizes inside the Metal kernel during
/// the matmul, yielding zero overhead vs standard SDPA.
///
/// - `queries`: `[B, n_q_heads, L, D]`
/// - `q_keys` / `q_values`: `(packed, scales, biases)` where packed is uint32
/// - For GQA (`n_q_heads > n_kv_heads`): queries are reshaped and quantized
///   tuples are broadcast, matching upstream behavior exactly.
/// - `sinks`: per-head attention sink logits `[n_q_heads]` (gpt-oss), or
///   `None`.
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
    sinks: Option<&InlineArray>,
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

    // Softmax with precise=true. Attention sinks (`[n_q_heads]`) join each
    // row as one more logit and are dropped again, as MLX's fused SDPA
    // computes them.
    let weights = match sinks {
        Some(sinks) => {
            let mut sink_shape = scores.shape().to_vec();
            let n = sink_shape.len();
            sink_shape[n - 1] = 1;
            let head_shape: Vec<i32> = if n_repeats > 1 {
                vec![1, n_kv_heads, n_repeats, 1, 1]
            } else {
                vec![1, n_q_heads, 1, 1]
            };
            let sink_col = sinks
                .as_dtype(scores.dtype_raw())
                .reshape(&head_shape)
                .broadcast_to(&sink_shape);
            let kv_len = scores.dim(-1);
            crate::compat::ops::slice_axis(
                &scores.concatenate_2(&sink_col, -1).softmax_precise(-1),
                -1,
                0,
                kv_len,
            )
        }
        None => scores.softmax_precise(-1),
    };

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
    sampler: &mut DecodeSampler,
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
    let current_y = sampler.sample(&logits_2d);
    current_y.async_eval_ref();
    current_y
}

/// Continue generation from an already-primed async sample, drawing every
/// token with `sampler` (the one that drew `current_y`).
#[allow(clippy::too_many_arguments)]
pub fn generate_from_primed_sample<Weights, Cache>(
    tag: &str,
    weights: &Weights,
    cache: &mut Cache,
    mut current_y: InlineArray,
    max_tokens: usize,
    mut sampler: DecodeSampler,
    log_stats: bool,
    mut on_token: impl FnMut(u32) -> bool,
    mut forward_step: impl FnMut(&Weights, &InlineArray, &mut Cache) -> InlineArray,
) -> (Vec<u32>, Option<DecodeMetrics>) {
    let mut tokens = Vec::with_capacity(max_tokens);
    let mut step_times: Vec<f64> = Vec::new();

    for step in 0..max_tokens {
        let next_y = if step + 1 < max_tokens {
            let t_step = std::time::Instant::now();
            let next_input = current_y.reshape(&[1, 1]);
            let next_logits = forward_step(weights, &next_input, cache);
            let next_logits_2d = next_logits.squeeze(1);
            // Queued behind `current_y` without reading it back: the
            // sampler's penalties take it in on the GPU.
            let next_y = sampler.sample(&next_logits_2d);

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

    #[test]
    fn last_token_hidden_keeps_the_sequence_axis() {
        let hidden = InlineArray::from_f32_slice(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[2, 3, 2],
        );
        let last = last_token_hidden(&hidden);
        assert_eq!(last.shape(), &[2, 1, 2]);
        let mut last = last.reshape(&[4]);
        assert_eq!(last.to_f32_vec(4).unwrap(), [5.0, 6.0, 11.0, 12.0]);
        crate::check_unobserved_error().expect("no bridge op failed");
    }

    /// The slices are the gathers they replace, element for element.
    #[test]
    fn last_position_slices_equal_the_gather_of_the_last_index() {
        use crate::compat::ops::select_axis;
        let x = crate::compat::random::normal_keyed(
            &[1, 5, 7],
            crate::compat::Dtype::Float32,
            Some(&draw_key(3, 0)),
        );
        let values = |a: InlineArray| {
            let mut a = a.reshape(&[-1]);
            let n = a.size();
            a.to_f32_vec(n).unwrap()
        };
        assert_eq!(
            values(last_token_logits(&x)),
            values(select_axis(&x, 4, 1)),
            "last_token_logits"
        );
        assert_eq!(last_token_hidden(&x).shape(), &[1, 1, 7]);
        assert_eq!(
            values(last_token_hidden(&x)),
            values(select_axis(&x, 4, 1).reshape(&[1, 1, -1])),
            "last_token_hidden"
        );
        crate::check_unobserved_error().expect("no bridge op failed");
    }

    #[test]
    fn greedy_token_is_the_argmax_of_the_last_axis() {
        let logits = InlineArray::from_f32_slice(&[0.5, -1.0, 3.0, 2.0], &[1, 4]);
        assert_eq!(greedy_token(&logits), 2);
        let flat = InlineArray::from_f32_slice(&[9.0, -1.0, 3.0], &[3]);
        assert_eq!(greedy_token(&flat), 0);
        crate::check_unobserved_error().expect("no bridge op failed");
    }

    fn values(array: &InlineArray) -> Vec<f32> {
        let mut array = array.reshape(&[-1]);
        let n = array.size();
        array.to_f32_vec(n).unwrap()
    }

    fn token(id: u32) -> InlineArray {
        InlineArray::from_slice(&[id], &[1])
    }

    /// Repetition over the prompt and the generated tokens, then frequency
    /// and presence over the generated ones, worked out by hand:
    ///
    /// | token | logit | in prompt | generated | penalized                  |
    /// |-------|-------|-----------|-----------|----------------------------|
    /// | 0     |  2.0  |           |           |  2.0                       |
    /// | 1     | -1.0  | yes       |           | -1.0 * 2 = -2.0            |
    /// | 2     |  0.5  |           | twice     |  0.5 / 2 - (2 * 0.5 + 0.25) = -1.0 |
    /// | 3     | -3.0  |           |           | -3.0                       |
    /// | 4     |  4.0  | twice     | once      |  4.0 / 2 - (0.5 + 0.25) = 1.25 |
    /// | 5     |  1.0  |           |           |  1.0                       |
    ///
    /// Token 99 of the prompt is past the vocabulary and penalizes nothing.
    #[test]
    fn penalties_match_hand_computed_logits() {
        let logits = InlineArray::from_f32_slice(&[2.0, -1.0, 0.5, -3.0, 4.0, 1.0], &[1, 6]);
        let mut penalties = DecodePenalties::new(2.0, 0.5, 0.25, &[1, 4, 4, 99]).unwrap();
        // Recorded before the first logits fix the vocabulary.
        penalties.record(&token(2));
        assert_eq!(
            values(&penalties.apply(&logits)),
            [2.0, -2.0, 0.25 - 0.75, -3.0, 2.0, 1.0]
        );
        penalties.record(&token(4));
        penalties.record(&token(2));
        let penalized = penalties.apply(&logits);
        assert_eq!(
            penalized.dtype_raw(),
            crate::compat::Dtype::Float32.as_i32()
        );
        assert_eq!(values(&penalized), [2.0, -2.0, -1.0, -3.0, 1.25, 1.0]);

        // bf16 logits come out penalized in f32.
        let bf16 = logits.as_dtype(crate::compat::Dtype::Bfloat16.as_i32());
        assert_eq!(
            values(&penalties.apply(&bf16)),
            [2.0, -2.0, -1.0, -3.0, 1.25, 1.0]
        );
        crate::check_unobserved_error().expect("no bridge op failed");
    }

    /// A clone records on its own: what a speculative decoder records on a
    /// clone for its drafts leaves the original's history as it was.
    #[test]
    fn a_clone_keeps_its_own_history() {
        let logits = InlineArray::from_f32_slice(&[1.0, 1.0, 1.0], &[1, 3]);
        let mut committed = DecodePenalties::new(1.0, 1.0, 0.0, &[]).unwrap();
        committed.record_id(0);
        assert_eq!(values(&committed.apply(&logits)), [0.0, 1.0, 1.0]);

        let mut drafting = committed.clone();
        drafting.record_id(1);
        drafting.record_id(0);
        assert_eq!(values(&drafting.apply(&logits)), [-1.0, 0.0, 1.0]);
        assert_eq!(values(&committed.apply(&logits)), [0.0, 1.0, 1.0]);
        crate::check_unobserved_error().expect("no bridge op failed");
    }

    #[test]
    fn no_penalty_means_no_penalties() {
        assert!(DecodePenalties::new(1.0, 0.0, 0.0, &[1, 2, 3]).is_none());
        assert!(DecodePenalties::new(1.0, 0.0, 0.1, &[]).is_some());
    }

    /// A greedy sampler takes every token it draws into the penalties
    /// before the next draw, without the host reading the token first.
    #[test]
    fn the_sampler_penalizes_the_tokens_it_drew() {
        let logits = InlineArray::from_f32_slice(&[3.0, 2.9, 0.0], &[1, 3]);
        let params = SamplingParams {
            repetition_penalty: 1.1,
            ..SamplingParams::new(0.0)
        };
        let mut sampler = DecodeSampler::new(params, &[0]);
        // The prompt's token 0: 3.0 / 1.1 < 2.9.
        let first = sampler.sample(&logits);
        // Both seen: 3.0 / 1.1 > 2.9 / 1.1.
        let second = sampler.sample(&logits);
        assert_eq!((first.item_u32(), second.item_u32()), (1, 0));

        let mut plain = DecodeSampler::new(SamplingParams::new(0.0), &[0]);
        assert_eq!(plain.sample(&logits).item_u32(), 0);
        crate::check_unobserved_error().expect("no bridge op failed");
    }

    /// Which tokens of the distribution `probs` [`filtered_log_probs`]
    /// keeps under `params`, and their probabilities renormalized.
    fn kept(probs: &[f32], params: &SamplingParams) -> Vec<Option<f32>> {
        let logits: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
        let lp = filtered_log_probs(
            &InlineArray::from_f32_slice(&logits, &[1, probs.len() as i32]),
            params,
        );
        let lp = lp.subtract(&lp.logsumexp(-1, true));
        lp.eval();
        crate::check_unobserved_error().expect("no bridge op failed");
        lp.as_slice::<f32>()
            .iter()
            .map(|&v| (v > f32::NEG_INFINITY).then(|| v.exp()))
            .collect()
    }

    fn sampling(temperature: f32) -> SamplingParams {
        SamplingParams::new(temperature)
    }

    fn kept_ids(probs: &[f32], params: &SamplingParams) -> Vec<usize> {
        kept(probs, params)
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.map(|_| i))
            .collect()
    }

    /// Temperature comes first, as transformers applies its warpers, so
    /// top-p reads the tempered distribution: at 0.5, `[0.5, 0.3, 0.2]`
    /// becomes `[0.66, 0.24, 0.11]`, whose top 0.6 is the first token alone.
    #[test]
    fn top_p_reads_the_tempered_distribution() {
        let probs = [0.5, 0.3, 0.2];
        let top_p = |t| SamplingParams {
            top_p: 0.6,
            ..sampling(t)
        };
        assert_eq!(kept_ids(&probs, &top_p(1.0)), [0, 1]);
        assert_eq!(kept_ids(&probs, &top_p(0.5)), [0]);
        let only = kept(&probs, &top_p(0.5));
        assert!((only[0].unwrap() - 1.0).abs() < 1e-6);
    }

    /// Top-k keeps every token tied with the k-th, and top-p then reads
    /// what top-k kept, renormalized: `[0.4, 0.3]` is `[0.57, 0.43]`, which
    /// top 0.6 keeps whole.
    #[test]
    fn top_k_keeps_ties_and_top_p_renormalizes_after_it() {
        let tied = [0.3, 0.3, 0.3, 0.1];
        let top_k = |k| SamplingParams {
            top_k: k,
            ..sampling(1.0)
        };
        assert_eq!(kept_ids(&tied, &top_k(2)), [0, 1, 2]);
        assert_eq!(kept_ids(&tied, &top_k(4)), [0, 1, 2, 3]);

        let probs = [0.4, 0.3, 0.2, 0.1];
        let both = SamplingParams {
            top_k: 2,
            top_p: 0.6,
            ..sampling(1.0)
        };
        assert_eq!(kept_ids(&probs, &both), [0, 1]);
        let renormalized = kept(&probs, &both);
        assert!((renormalized[0].unwrap() - 4.0 / 7.0).abs() < 1e-6);
    }

    /// Top-p drops the least likely tokens while their total is at most
    /// `1 - p`, always keeping the most likely one; on a tie at the cut it
    /// keeps every token as likely as the last one kept.
    #[test]
    fn top_p_keeps_the_most_likely_token_and_ties_at_the_cut() {
        let top_p = |p| SamplingParams {
            top_p: p,
            ..sampling(1.0)
        };
        // Ascending totals 0.1, 0.3, 0.6: at p = 0.65 the first two are at
        // most 0.35.
        assert_eq!(kept_ids(&[0.4, 0.3, 0.2, 0.1], &top_p(0.65)), [0, 1]);
        assert_eq!(kept_ids(&[0.9, 0.1], &top_p(1e-6)), [0]);
        assert_eq!(kept_ids(&[0.25; 4], &top_p(0.5)), [0, 1, 2, 3]);
    }

    /// Min-p keeps the tokens at least `min_p` times as likely as the most
    /// likely, after the temperature: `[0.5, 0.3, 0.15, 0.05]` at 0.4 keeps
    /// two, and at temperature 2 (ratios 1, 0.77, 0.55, 0.32) three.
    #[test]
    fn min_p_scales_with_the_most_likely_token_after_the_temperature() {
        let min_p = |t| SamplingParams {
            min_p: 0.4,
            ..sampling(t)
        };
        let probs = [0.5, 0.3, 0.15, 0.05];
        assert_eq!(kept_ids(&probs, &min_p(1.0)), [0, 1]);
        assert_eq!(kept_ids(&probs, &min_p(2.0)), [0, 1, 2]);
    }

    /// The filters run on `[B, vocab]` rows independently and in f32,
    /// whatever the logits' dtype.
    #[test]
    fn the_filters_run_per_row_in_f32() {
        let logits = InlineArray::from_f32_slice(&[0.0, 1.0, 2.0, 2.0, 1.0, 0.0], &[2, 3])
            .as_dtype(crate::compat::Dtype::Bfloat16.as_i32());
        let params = SamplingParams {
            top_k: 1,
            ..sampling(0.7)
        };
        let lp = filtered_log_probs(&logits, &params);
        assert_eq!(lp.dtype_raw(), crate::compat::Dtype::Float32.as_i32());
        lp.eval();
        let finite: Vec<bool> = lp.as_slice::<f32>().iter().map(|v| v.is_finite()).collect();
        assert_eq!(finite, [false, false, true, true, false, false]);
        crate::check_unobserved_error().expect("no bridge op failed");
    }
}
