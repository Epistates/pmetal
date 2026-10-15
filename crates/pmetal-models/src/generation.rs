//! Text generation for language models.
//!
//! The generators here drive any model that maps tokens (and a KV cache) to
//! logits. Sampling is the runtime's one sampler,
//! [`DecodeSampler`]: temperature, top-k, top-p, min-p (in transformers'
//! order), the repetition, frequency and presence penalties, the logit bias,
//! and any [`LogitsProcessor`](pmetal_runtime::decode::LogitsProcessor) a
//! constraint adds. [`GenerationConfig`] is the request-side description of
//! it, and [`GenerationConfig::sampler`] builds one for a generation.
//!
//! Sampling stays on the GPU, the penalties' token history included, and the
//! pipelined generators queue each step's forward before the previous token
//! is read back, so the host never stalls the GPU.
use pmetal_cache::kv_cache::KVCache;
use pmetal_runtime::compat::ops::{async_eval, logsumexp_axis_keepdims, select_axis};
use pmetal_runtime::compat::{Array, Dtype, Exception};
use pmetal_runtime::decode::{
    DecodeSampler, SamplingParams, last_token_logits, prompt_tokens_to_input,
};
use pmetal_runtime::inline_array::{GenerationStream, WiredLimit};

use crate::speculative::SpeculativeDecodeMetrics;

/// The KV cache an ANE request needs: prompt and output, rounded up to a
/// power of two (so nearby requests share a compiled model), 512 to `cap`.
#[cfg(feature = "ane")]
fn ane_context(prompt_len: usize, max_new: usize, cap: usize) -> usize {
    (prompt_len + max_new)
        .next_power_of_two()
        .clamp(512, cap.max(512))
}

/// Most attention one prefill chunk computes: its length times the depth it
/// attends over (the tokens already cached plus its own). A chunk's attention
/// runs as one Metal dispatch whose time grows with that product, and macOS
/// kills a command buffer that runs past a few seconds
/// (`kIOGPUCommandBufferCallbackErrorTimeout`), so a fixed chunk that is safe
/// near the start of a long prompt is not safe near its end. 1.6e8 is the
/// budget measured to hold a 245k-token prompt under the watchdog on an
/// M5 Pro; a 2048-token chunk stays whole to a depth of about 76k tokens.
pub const PREFILL_ATTENTION_BUDGET: usize = 160_000_000;

/// Shortest chunk a deep prefill shrinks to.
pub const MIN_PREFILL_CHUNK: usize = 128;

/// Length of the prefill chunk that starts `depth` tokens into the cache:
/// `step_size`, or, once `step_size * (depth + step_size)` passes
/// [`PREFILL_ATTENTION_BUDGET`], the largest power of two inside the budget,
/// never under [`MIN_PREFILL_CHUNK`] (nor over `step_size`).
pub fn prefill_chunk_len(step_size: usize, depth: usize) -> usize {
    let step_size = step_size.max(1);
    let fits = PREFILL_ATTENTION_BUDGET / (depth + step_size);
    let pow2 = match fits {
        0 => 0,
        n => 1usize << (usize::BITS - 1 - n.leading_zeros()),
    };
    step_size.min(pow2.max(MIN_PREFILL_CHUNK))
}

/// Run cached prompt prefill, optionally chunking long prompts to bound peak memory.
///
/// Used by the async-pipelined generators (and by `pmetal-serve`) to keep
/// prefill latency bounded on long prompts: each chunk is `eval`ed and the
/// MLX allocation cache is cleared between chunks so peak memory matches a
/// single-chunk forward. The final chunk's logits are returned lazily (no
/// `eval`) so the caller can fold them into the async decode pipeline.
///
/// Chunks shrink as the prompt gets deeper ([`prefill_chunk_len`]) so no
/// chunk's attention runs long enough for the GPU watchdog to kill it.
pub fn run_cached_prefill_chunks<F>(
    input_ids: &[u32],
    prefill_step_size: usize,
    forward: F,
) -> Result<Array, Exception>
where
    F: FnMut(&Array) -> Result<Array, Exception>,
{
    run_cached_prefill_chunks_at(input_ids, 0, prefill_step_size, forward)
}

/// [`run_cached_prefill_chunks`] into a cache that already holds `depth`
/// tokens (a restored prefix), which the chunk sizing counts.
/// `prefill_step_size == 0` prefills in one forward, unbounded.
pub fn run_cached_prefill_chunks_at<F>(
    input_ids: &[u32],
    depth: usize,
    prefill_step_size: usize,
    mut forward: F,
) -> Result<Array, Exception>
where
    F: FnMut(&Array) -> Result<Array, Exception>,
{
    if input_ids.is_empty() || prefill_step_size == 0 {
        return forward(&prompt_tokens_to_input(input_ids));
    }

    let mut last_logits = None;
    let mut start = 0;
    while start < input_ids.len() {
        let len = prefill_chunk_len(prefill_step_size, depth + start);
        let end = (start + len).min(input_ids.len());
        let chunk_input = prompt_tokens_to_input(&input_ids[start..end]);
        let logits = forward(&chunk_input)?;
        if end < input_ids.len() {
            logits.eval();
            pmetal_runtime::inline_array::clear_cache();
        } else {
            last_logits = Some(logits);
        }
        start = end;
    }

    last_logits.ok_or_else(|| Exception::custom("chunked prefill produced no output"))
}

/// Configuration for text generation.
///
/// Default values are tuned for balanced quality and diversity.
/// For specific models like Qwen3, use the preset methods.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Temperature for sampling (1.0 = no change, < 1.0 = more deterministic, > 1.0 = more random).
    /// Recommended range: 0.1 - 1.5
    pub temperature: f32,
    /// Top-k sampling parameter (0 = disabled).
    /// Keeps only the k highest probability tokens.
    /// Recommended: 20-100 for most models.
    pub top_k: usize,
    /// Top-p (nucleus) sampling parameter (1.0 = disabled).
    /// Keeps smallest set of tokens with cumulative probability >= top_p.
    /// Recommended: 0.8-0.95
    pub top_p: f32,
    /// Min-p sampling parameter (0.0 = disabled).
    /// Dynamic threshold: keeps tokens with prob >= min_p * top_token_prob.
    /// Recommended: 0.05-0.1 for high-temperature creativity with coherence.
    pub min_p: f32,
    /// Repetition penalty applied to prompt + generated tokens (1.0 = disabled).
    /// Values > 1.0 discourage repetition, < 1.0 encourage it.
    /// Recommended: 1.0-1.2
    pub repetition_penalty: f32,
    /// Frequency penalty proportional to token appearance count (0.0 = disabled).
    /// Applied as: logit -= frequency_penalty * count
    /// Recommended: 0.0-2.0
    pub frequency_penalty: f32,
    /// Presence penalty for any token that has appeared (0.0 = disabled).
    /// Applied as flat penalty regardless of count.
    /// Recommended: 0.0-2.0 (Qwen3 uses 1.5 for non-thinking mode)
    pub presence_penalty: f32,
    /// Token IDs that trigger end of generation.
    pub stop_tokens: Vec<u32>,
    /// Random seed for reproducible generation.
    pub seed: Option<u64>,
    /// `(token, bias)` pairs added to the logits before sampling, after the
    /// penalties (OpenAI's `logit_bias`: -100 all but bans a token, 100 all
    /// but forces it).
    pub logit_bias: Vec<(u32, f32)>,
    /// Whether to use greedy decoding (ignores temperature, top_k, top_p, min_p).
    pub do_sample: bool,
    /// Maximum prompt tokens to process per cached-prefill chunk (0 = disable chunking).
    pub prefill_step_size: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_tokens: vec![],
            seed: None,
            logit_bias: Vec::new(),
            do_sample: true,
            prefill_step_size: 2048,
        }
    }
}

impl GenerationConfig {
    /// Create a greedy decoding config.
    pub fn greedy(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            do_sample: false,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            ..Default::default()
        }
    }

    /// Create a sampling config with temperature.
    pub fn sampling(max_new_tokens: usize, temperature: f32) -> Self {
        Self {
            max_new_tokens,
            temperature,
            do_sample: true,
            ..Default::default()
        }
    }

    /// Create optimal config for Qwen3 thinking mode.
    ///
    /// Uses official Qwen3 recommendations: temp=0.6, top_p=0.95, top_k=20.
    /// Presence penalty=1.5 to prevent endless repetitions.
    pub fn qwen3_thinking(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            temperature: 0.6,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 1.5, // Prevents endless repetitions
            stop_tokens: vec![],
            seed: None,
            logit_bias: Vec::new(),
            do_sample: true,
            prefill_step_size: 2048,
        }
    }

    /// Create optimal config for Qwen3 non-thinking mode.
    ///
    /// Uses official Qwen3 recommendations: temp=0.7, top_p=0.8, top_k=20, presence_penalty=1.5.
    pub fn qwen3_non_thinking(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            temperature: 0.7,
            top_k: 20,
            top_p: 0.8,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 1.5,
            stop_tokens: vec![],
            seed: None,
            logit_bias: Vec::new(),
            do_sample: true,
            prefill_step_size: 2048,
        }
    }

    /// Create config optimized for creative writing at high temperature.
    ///
    /// Uses min-p sampling to maintain coherence at high temperatures.
    pub fn creative(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            temperature: 1.2,
            top_k: 0,   // Disabled, relying on min-p
            top_p: 1.0, // Disabled, relying on min-p
            min_p: 0.1,
            repetition_penalty: 1.15,
            frequency_penalty: 0.5,
            presence_penalty: 0.5,
            stop_tokens: vec![],
            seed: None,
            logit_bias: Vec::new(),
            do_sample: true,
            prefill_step_size: 2048,
        }
    }

    /// Create config for precise, factual responses.
    pub fn precise(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            temperature: 0.3,
            top_k: 10,
            top_p: 0.9,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_tokens: vec![],
            seed: None,
            logit_bias: Vec::new(),
            do_sample: true,
            prefill_step_size: 2048,
        }
    }

    /// Load model-default generation parameters from `generation_config.json`.
    ///
    /// Reads the file if present and applies any fields found as defaults.
    /// Fields not present in the file keep their current values.
    /// User-specified CLI overrides should be applied after calling this.
    pub fn from_model_dir(model_dir: &std::path::Path) -> Self {
        let path = model_dir.join("generation_config.json");
        let mut config = Self::default();

        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => return config,
        };

        let json: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("Failed to parse generation_config.json: {e}");
                return config;
            }
        };

        if let Some(t) = json.get("temperature").and_then(|v| v.as_f64()) {
            config.temperature = t as f32;
        }
        if let Some(k) = json.get("top_k").and_then(|v| v.as_u64()) {
            config.top_k = k as usize;
        }
        if let Some(p) = json.get("top_p").and_then(|v| v.as_f64()) {
            config.top_p = p as f32;
        }
        if let Some(p) = json.get("min_p").and_then(|v| v.as_f64()) {
            config.min_p = p as f32;
        }
        if let Some(rp) = json.get("repetition_penalty").and_then(|v| v.as_f64()) {
            config.repetition_penalty = rp as f32;
        }
        if let Some(ds) = json.get("do_sample").and_then(|v| v.as_bool()) {
            config.do_sample = ds;
        }
        if let Some(ml) = json.get("max_new_tokens").and_then(|v| v.as_u64()) {
            config.max_new_tokens = ml as usize;
        } else if let Some(ml) = json.get("max_length").and_then(|v| v.as_u64()) {
            // max_length is the total context length; use as fallback if max_new_tokens absent
            config.max_new_tokens = ml as usize;
        }

        // Stop token IDs — can be a single int or array
        if let Some(eos) = json.get("eos_token_id") {
            match eos {
                serde_json::Value::Number(n) => {
                    if let Some(id) = n.as_u64() {
                        config.stop_tokens.push(id as u32);
                    }
                }
                serde_json::Value::Array(arr) => {
                    for v in arr {
                        if let Some(id) = v.as_u64() {
                            config.stop_tokens.push(id as u32);
                        }
                    }
                }
                _ => {}
            }
        }

        tracing::debug!(
            "Loaded generation defaults: temp={}, top_k={}, top_p={}, do_sample={}, stop_tokens={:?}",
            config.temperature,
            config.top_k,
            config.top_p,
            config.do_sample,
            config.stop_tokens,
        );

        config
    }

    /// Set top-k sampling.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set top-p (nucleus) sampling.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set min-p sampling threshold.
    ///
    /// Min-p provides dynamic truncation based on top token probability.
    /// Tokens are kept if their probability >= min_p * top_token_probability.
    /// Recommended values: 0.05-0.1 for balanced creativity and coherence.
    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    /// Set repetition penalty (applied to prompt + output).
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Set frequency penalty (proportional to appearance count).
    ///
    /// Applied as: logit -= frequency_penalty * count
    pub fn with_frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = penalty;
        self
    }

    /// Set presence penalty (flat penalty for appeared tokens).
    ///
    /// Applied as: logit -= presence_penalty (if token has appeared)
    pub fn with_presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = penalty;
        self
    }

    /// The sampler of a generation under this config continuing `prompt`:
    /// the runtime's [`DecodeSampler`], whose penalties read the prompt and
    /// every token the generation records into it, in order.
    pub fn sampler(&self, prompt: &[u32]) -> DecodeSampler {
        DecodeSampler::new(self.sampling_params(), prompt)
    }

    /// Whether `token` ends generation.
    pub fn is_stop_token(&self, token: u32) -> bool {
        self.stop_tokens.contains(&token)
    }

    /// This config's sampling as the runtime's [`SamplingParams`], which
    /// [`DecodeSampler`] and the native engines read: greedy
    /// (temperature 0) without `do_sample`, and a sampling temperature at
    /// or below 0 taken as 1.
    pub fn sampling_params(&self) -> SamplingParams {
        let temperature = match (self.do_sample, self.temperature) {
            (false, _) => 0.0,
            (true, t) if t > 0.0 => t,
            (true, _) => 1.0,
        };
        SamplingParams {
            temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            seed: self.seed,
            logit_bias: self.logit_bias.clone(),
        }
    }

    /// Add `bias` to `token`'s logit before sampling.
    pub fn with_logit_bias(mut self, token: u32, bias: f32) -> Self {
        self.logit_bias.push((token, bias));
        self
    }

    /// Set stop tokens.
    pub fn with_stop_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.stop_tokens = tokens;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the cached-prefill chunk size for long prompts.
    ///
    /// `0` disables chunking and processes the entire prompt in one forward pass.
    pub fn with_prefill_step_size(mut self, step_size: usize) -> Self {
        self.prefill_step_size = step_size;
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
}

/// Output from text generation.
#[derive(Debug, Clone, Default)]
pub struct GenerationOutput {
    /// The prompt, then the generated token IDs. A stop token ends the
    /// generation and is not part of the output.
    pub token_ids: Vec<u32>,
    /// Number of tokens generated (excluding prompt).
    pub num_generated: usize,
    /// Whether generation stopped due to stop token.
    pub stopped_by_token: bool,
    /// Whether generation stopped due to max length.
    pub stopped_by_length: bool,
    /// Per-step decode timing, populated by `generate_cached_async_streaming`.
    /// The native path on `inference_runner` also produces the same struct
    /// via `pmetal_runtime::decode::DecodeMetrics`, which is what the CLI
    /// summary print ingests — keeping the shape identical keeps the stats
    /// display DRY across both paths.
    pub decode_metrics: Option<pmetal_runtime::decode::DecodeMetrics>,
    /// Speculative decoding acceptance/accounting metrics.
    pub speculative_metrics: Option<SpeculativeDecodeMetrics>,
}

/// Why a generation ended.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ended {
    /// The model produced a stop token.
    Stop,
    /// `max_new_tokens` tokens were generated.
    Length,
    /// The caller's token callback asked to stop.
    Caller,
}

impl GenerationOutput {
    fn ended(
        token_ids: Vec<u32>,
        prompt_len: usize,
        ended: Ended,
        decode_metrics: Option<pmetal_runtime::decode::DecodeMetrics>,
    ) -> Self {
        Self {
            num_generated: token_ids.len() - prompt_len,
            token_ids,
            stopped_by_token: ended == Ended::Stop,
            stopped_by_length: ended == Ended::Length,
            decode_metrics,
            speculative_metrics: None,
        }
    }
}

/// One generation's [`DecodeSampler`], deciding tokens from logits an engine
/// hands over on the host (the ANE engine), so they are sampled and
/// penalized as [`generate_cached`] samples and penalizes them on the GPU.
///
/// Each call wraps the row as `[1, vocab]` f32 logits and samples it through
/// the sampler, which takes the token into its penalties, which is what
/// [`TokenSampler`](pmetal_metal::ane::lm::TokenSampler) asks of it: the
/// engines ask for the generation's tokens in order, a speculative one never
/// past a guess it rejects, so the history holds exactly the tokens before
/// each position, and the `n`-th token takes the `n`-th seeded draw.
///
/// Greedy decoding with nothing to apply and draws nothing, so it takes the
/// argmax on the host, which is the GPU's argmax (the first maximum),
/// without the round trip: about 0.5 ms a token on a 151,936-token
/// vocabulary, several times a pass when guesses are verified.
#[cfg(feature = "ane")]
pub struct HostLogitsSampler {
    sampler: DecodeSampler,
}

#[cfg(feature = "ane")]
impl HostLogitsSampler {
    /// The sampler of a generation under `config` continuing `prompt`.
    pub fn new(config: &GenerationConfig, prompt: &[u32]) -> Self {
        Self {
            sampler: config.sampler(prompt),
        }
    }
}

#[cfg(feature = "ane")]
impl pmetal_metal::ane::lm::TokenSampler for HostLogitsSampler {
    fn sample(&mut self, logits: &[f32]) -> pmetal_metal::error::Result<u32> {
        use pmetal_metal::error::MetalError;
        if self.sampler.is_plain_argmax() {
            return pmetal_metal::ane::lm::Greedy.sample(logits);
        }
        let row = Array::from_f32_slice(logits, &[1, logits.len() as i32]);
        let token = self.sampler.sample(&row).item_u32();
        pmetal_runtime::check_unobserved_error()
            .map_err(|e| MetalError::Internal(format!("sampling: {e}")))?;
        Ok(token)
    }
}

/// Convert logits to log probabilities (log-softmax), in f32.
/// logprobs = logits - logsumexp(logits, keepdims=True)
///
/// Computed in f32 whatever the model's dtype, as transformers' `generate`
/// upcasts next-token logits before scoring them. In bf16 the result rounds
/// to the logits' own precision, steps of 1/8 for logits between 16 and 32,
/// so reported logprobs came out coarse and the sampling filters cut on
/// rounded values.
fn logits_to_log_probs(logits: &Array) -> Array {
    let owned;
    let logits = if logits.dtype() == Dtype::Float32 {
        logits
    } else {
        owned = logits.cast(Dtype::Float32);
        &owned
    };
    let lse = logsumexp_axis_keepdims(logits, -1, true);
    logits.subtract(&lse)
}

/// Compute the log-probability of a specific token under the given logits
/// distribution, plus the top-N highest-logprob alternatives.
///
/// Used by the `/v1/chat/completions` `logprobs` feature — the sampler
/// returns only the chosen token ID, but OpenAI clients can request the
/// model's confidence in that token and the next-best alternatives.
///
/// * `logits` — raw next-token logits, shape `[..., vocab]`. Any batch/seq
///   dimensions are squeezed to the last axis internally.
/// * `token` — the token ID whose log-prob is requested.
/// * `top_n` — when > 0, also returns the `top_n` highest-logprob tokens
///   (id + logprob). When 0, the returned alternative list is empty.
///
/// Returns `(chosen_logprob, top_alternatives)` — both ready for wire
/// serialisation. `chosen_logprob` is log-softmax at `token`; the
/// alternative list is sorted descending by logprob.
pub fn token_logprobs(
    logits: &Array,
    token: u32,
    top_n: usize,
) -> Result<(f32, Vec<(u32, f32)>), Exception> {
    use pmetal_runtime::compat::ops::{argpartition_axis, negative, slice_axis, take_along_axis};

    // Flatten every leading dim so single-row and batched callers both land
    // on a [1, vocab] view. The sampler path only ever has one active
    // prediction per step so this is safe.
    let vocab = *logits
        .shape()
        .last()
        .ok_or_else(|| Exception::custom("token_logprobs: empty logits shape"))?;
    if token as i64 >= i64::from(vocab) {
        return Err(Exception::custom(format!(
            "token_logprobs: token {token} out of range"
        )));
    }
    let log_probs = logits_to_log_probs(logits).reshape(&[1, vocab]);

    // The chosen token's log-probability, and the `top_n` largest ones,
    // gathered on the device: only a handful of values come back, not the
    // vocabulary (600 KB per token over a 152k vocabulary).
    let chosen = take_along_axis(&log_probs, &Array::from_slice(&[token as i32], &[1, 1]), -1);
    let k = top_n.min(vocab as usize);
    let top = (k > 0).then(|| {
        // The first k columns of a partition of the negated log-probs at
        // k - 1 are the k largest, in no order.
        let part = argpartition_axis(&negative(&log_probs), k as i32 - 1, -1);
        let ids = slice_axis(&part, 1, 0, k as i32);
        (take_along_axis(&log_probs, &ids, -1), ids)
    });
    chosen.eval();
    if let Some((values, ids)) = &top {
        values.eval();
        ids.eval();
    }
    pmetal_runtime::check_unobserved_error()
        .map_err(|e| Exception::custom(format!("token_logprobs: {e}")))?;
    let chosen = chosen.as_slice::<f32>()[0];

    let mut alternatives = Vec::new();
    if let Some((values, ids)) = &top {
        let (values, ids) = (values.as_slice::<f32>(), ids.as_slice::<u32>());
        alternatives = ids.iter().copied().zip(values.iter().copied()).collect();
        // Descending by log-probability, ties by token id.
        alternatives.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    }
    Ok((chosen, alternatives))
}

/// Simple generation function that works with any model that has a `forward` method.
///
/// It runs the model over the whole sequence for every token (no cache), which
/// is what makes it the reference the cached generators are checked against.
///
/// # Arguments
/// * `forward_fn` - Function that takes input_ids and returns logits
/// * `input_ids` - Initial token IDs [1, seq_len]
/// * `config` - Generation configuration
///
/// # Returns
/// Generation output containing all tokens and metadata
pub fn generate<F>(
    mut forward_fn: F,
    input_ids: &[u32],
    config: GenerationConfig,
) -> Result<GenerationOutput, Exception>
where
    F: FnMut(&Array) -> Result<Array, Exception>,
{
    let mut all_tokens: Vec<u32> = input_ids.to_vec();
    let mut sampler = config.sampler(input_ids);
    let prompt_len = input_ids.len();

    for _ in 0..config.max_new_tokens {
        let logits = forward_fn(&prompt_tokens_to_input(&all_tokens))?;
        let next_token = sampler.sample(&last_token_logits(&logits)).item_u32();
        if config.is_stop_token(next_token) {
            return Ok(GenerationOutput::ended(
                all_tokens,
                prompt_len,
                Ended::Stop,
                None,
            ));
        }
        all_tokens.push(next_token);
    }
    Ok(GenerationOutput::ended(
        all_tokens,
        prompt_len,
        Ended::Length,
        None,
    ))
}

/// KV-cached generation for efficient autoregressive decoding.
///
/// This function uses a KV cache to avoid recomputing attention for
/// previous tokens, providing 10-50x speedup for longer generations. It reads
/// every token back before running the next step, so it is the simple
/// reference the pipelined [`generate_cached_async_streaming`] is held to.
///
/// # Arguments
/// * `forward_fn` - Function that takes (input_ids, cache) and returns logits
/// * `input_ids` - Initial token IDs (prompt)
/// * `config` - Generation configuration
/// * `cache` - Pre-allocated KV cache
///
/// # Returns
/// Generation output containing all tokens and metadata
///
/// # Performance
/// - Prefill: O(n) - processes full prompt once
/// - Decode: O(1) per token - only processes new token
/// - Total: O(n + k) instead of O(n*k) where n=prompt_len, k=new_tokens
pub fn generate_cached<F>(
    mut forward_fn: F,
    input_ids: &[u32],
    config: GenerationConfig,
    cache: &mut KVCache,
) -> Result<GenerationOutput, Exception>
where
    F: FnMut(&Array, &mut KVCache) -> Result<Array, Exception>,
{
    let _wired = WiredLimit::max();

    let mut all_tokens: Vec<u32> = input_ids.to_vec();
    let mut sampler = config.sampler(input_ids);
    let prompt_len = input_ids.len();

    // Prefill: process long prompts in chunks to bound peak allocator pressure.
    let logits = run_cached_prefill_chunks(input_ids, config.prefill_step_size, |chunk_input| {
        forward_fn(chunk_input, cache)
    })?;
    let mut next_token = sampler.sample(&last_token_logits(&logits)).item_u32();

    for _ in 0..config.max_new_tokens {
        if config.is_stop_token(next_token) {
            return Ok(GenerationOutput::ended(
                all_tokens,
                prompt_len,
                Ended::Stop,
                None,
            ));
        }
        all_tokens.push(next_token);
        if all_tokens.len() - prompt_len == config.max_new_tokens {
            break;
        }

        // Forward with cache: only the new token runs.
        let token_input = Array::from_slice(&[next_token as i32], &[1, 1]);
        let logits = forward_fn(&token_input, cache)?;
        next_token = sampler.sample(&select_axis(&logits, 0, 1)).item_u32();
    }

    Ok(GenerationOutput::ended(
        all_tokens,
        prompt_len,
        Ended::Length,
        None,
    ))
}

/// Generation with async pipelining.
///
/// [`generate_cached_async_streaming`] without a token callback.
pub fn generate_cached_async<F>(
    forward_fn: F,
    input_ids: &[u32],
    config: GenerationConfig,
    cache: &mut KVCache,
) -> Result<GenerationOutput, Exception>
where
    F: FnMut(&Array, &mut KVCache) -> Result<Array, Exception>,
{
    generate_cached_async_streaming(forward_fn, input_ids, config, cache, |_| true)
}

/// The pipelined decode loop every cached generator runs: [`generate_cached_async_streaming`],
/// and the serving engine's per-request decode.
///
/// Step `n + 1`'s forward is queued on the GPU before step `n`'s token is
/// read back, and the token stays an array until then (the next step takes it
/// as its input, and the sampler's penalties take it in on the GPU), so the
/// host's work overlaps the GPU's:
///
/// ```text
/// y = sample(first logits)              // an array, still lazy
/// async_eval(y)
/// loop:
///     next_y = sample(step(y))          // y is an array, passed directly
///     async_eval(next_y)                // queued BEFORE y is read
///     on_token(y.item())                // blocks while next_y computes
///     y = next_y
/// ```
///
/// * `first_logits` `[1, vocab]` is where the first token is sampled from
///   (the prompt's last position).
/// * `step(input, i)` runs the model on the `[1, 1]` int32 token array
///   `input` (the token the `i`-th step samples from) and returns the
///   `[1, vocab]` logits of the next one.
/// * `on_token(token, logits)` sees each token with the `[1, vocab]` logits
///   it was sampled from (before the sampler's constraints, for log-probability
///   reporting) and returns [`ControlFlow::Break`](std::ops::ControlFlow::Break) to end the decode, which it
///   does without the token being sampled past. One forward past the last
///   token read has been queued by then, so the cache holds a token the caller
///   has not seen; a caller that reuses the cache must account for it.
///
/// At most `max_tokens` tokens are produced. Returns each step's latency in
/// nanoseconds, measured from the step's queueing to its token's readback and
/// closed before `on_token` runs, so a slow consumer does not contaminate it
/// (the first includes the wait for the prefill).
///
/// Run it with a [`GenerationStream`] and a [`WiredLimit`] held, as
/// [`generate_cached_async_streaming`] does.
pub fn decode_pipelined<S, C>(
    first_logits: &Array,
    sampler: &mut DecodeSampler,
    max_tokens: usize,
    mut step: S,
    mut on_token: C,
) -> Result<Vec<u128>, Exception>
where
    S: FnMut(&Array, usize) -> Result<Array, Exception>,
    C: FnMut(u32, &Array) -> std::ops::ControlFlow<()>,
{
    let mut step_ns = Vec::with_capacity(max_tokens);
    if max_tokens == 0 {
        return Ok(step_ns);
    }
    let mut logits = first_logits.clone();
    let mut current = sampler.sample(&logits);
    async_eval([&current]);

    for i in 0..max_tokens {
        let started = std::time::Instant::now();
        let next = if i + 1 < max_tokens {
            let input = current.as_dtype(Dtype::Int32.as_i32()).reshape(&[1, -1]);
            let next_logits = step(&input, i)?;
            let next = sampler.sample(&next_logits);
            async_eval([&next]);
            Some((next, next_logits))
        } else {
            None
        };

        if i == 0 {
            // Surfaces an error in the prefill or the first step here, as an
            // error, instead of in the read below.
            current
                .try_eval()
                .map_err(|e| Exception::custom(e.to_string()))?;
        }
        // Blocks while the GPU computes the step queued above.
        let token = current.item_u32();
        step_ns.push(started.elapsed().as_nanos());

        if on_token(token, &logits).is_break() {
            break;
        }
        if i % 256 == 255 {
            pmetal_runtime::inline_array::clear_cache();
        }
        let Some((next, next_logits)) = next else {
            break;
        };
        current = next;
        logits = next_logits;
    }
    Ok(step_ns)
}

/// KV-cached generation with async pipelining ([`decode_pipelined`]), calling
/// `on_token` with each generated token ID as it's produced, enabling
/// real-time display. The callback returns `true` to continue generation or
/// `false` to stop early.
///
/// Everything runs on the thread's generation stream under a wired-memory
/// limit at the device's recommended maximum, both restored on return. A
/// stop token ends the generation: it is neither emitted nor in the output.
/// One forward past it has been queued by then, so the cache holds a token
/// the output does not; a caller that reuses the cache must not.
///
/// # Arguments
/// * `forward_fn` - Function that takes (input_ids, cache) and returns logits
/// * `input_ids` - Initial token IDs (prompt)
/// * `config` - Generation configuration
/// * `cache` - Pre-allocated KV cache
pub fn generate_cached_async_streaming<F, C>(
    mut forward_fn: F,
    input_ids: &[u32],
    config: GenerationConfig,
    cache: &mut KVCache,
    mut on_token: C,
) -> Result<GenerationOutput, Exception>
where
    F: FnMut(&Array, &mut KVCache) -> Result<Array, Exception>,
    C: FnMut(u32) -> bool,
{
    let _wired = WiredLimit::max();
    let _stream = GenerationStream::enter();

    let mut all_tokens: Vec<u32> = input_ids.to_vec();
    let mut sampler = config.sampler(input_ids);
    let prompt_len = input_ids.len();

    let logits = run_cached_prefill_chunks(input_ids, config.prefill_step_size, |chunk_input| {
        forward_fn(chunk_input, cache)
    })?;
    let mut ended = Ended::Length;
    let step_ns = decode_pipelined(
        &last_token_logits(&logits),
        &mut sampler,
        config.max_new_tokens,
        |input, _| Ok(select_axis(&forward_fn(input, cache)?, 0, 1)),
        |token, _| {
            if config.is_stop_token(token) {
                ended = Ended::Stop;
                return std::ops::ControlFlow::Break(());
            }
            all_tokens.push(token);
            if !on_token(token) {
                ended = Ended::Caller;
                return std::ops::ControlFlow::Break(());
            }
            std::ops::ControlFlow::Continue(())
        },
    )?;
    Ok(GenerationOutput::ended(
        all_tokens,
        prompt_len,
        ended,
        build_decode_metrics(&step_ns),
    ))
}

/// Compute a [`DecodeMetrics`] summary from raw per-step latencies (ns).
///
/// Drops the first step as warmup (it covers prefill fall-out / JIT caches)
/// and returns `None` if fewer than 2 usable samples are available. The
/// shape is identical to `pmetal_runtime::decode::DecodeMetrics` so the CLI
/// summary print path works for both the native and dynamic-model routes.
fn build_decode_metrics(step_ns: &[u128]) -> Option<pmetal_runtime::decode::DecodeMetrics> {
    // n=0 is warmup (includes prefill fall-out / JIT caches).
    let body = step_ns.get(1..)?;
    if body.is_empty() {
        return None;
    }
    let measured_steps = body.len();
    let total_ns: u128 = body.iter().sum();
    let avg_step_ms = (total_ns as f64 / measured_steps as f64) / 1_000_000.0;
    let mut sorted: Vec<u128> = body.to_vec();
    sorted.sort_unstable();
    let p50_ns = sorted[measured_steps / 2];
    let p50_step_ms = (p50_ns as f64) / 1_000_000.0;
    let tok_per_sec = if total_ns > 0 {
        measured_steps as f64 / (total_ns as f64 / 1_000_000_000.0)
    } else {
        0.0
    };
    Some(pmetal_runtime::decode::DecodeMetrics {
        tok_per_sec,
        avg_step_ms,
        p50_step_ms,
        measured_steps,
    })
}

// ============================================================================
// ANE generation (Apple Neural Engine)
// ============================================================================

#[cfg(feature = "ane")]
fn build_cached_generation_output(
    input_len: usize,
    token_ids: Vec<u32>,
    gen_config: &GenerationConfig,
    cancelled: bool,
    saw_stop_token: bool,
) -> GenerationOutput {
    let num_generated = token_ids.len().saturating_sub(input_len);
    let stopped_by_token =
        !cancelled && (saw_stop_token || num_generated < gen_config.max_new_tokens);

    GenerationOutput {
        token_ids,
        num_generated,
        stopped_by_token,
        stopped_by_length: !cancelled
            && !stopped_by_token
            && num_generated >= gen_config.max_new_tokens,
        decode_metrics: None,
        speculative_metrics: None,
    }
}

/// Generate on the ANE: the model's layers as multi-layer ANE programs over a
/// KV cache sized to the request (at most `ane_max_context` tokens), with
/// guesses at the following tokens verified alongside each one, which changes
/// the speed but not the output. The guesses come from the DFlash draft model
/// in `draft_path` (run on the GPU) when there is one, and from prompt lookup
/// otherwise. Each token is sampled from the ANE's logits as
/// [`generate_cached`] samples it ([`HostLogitsSampler`]): temperature,
/// top-k, top-p, min-p, the seed, and the repetition, frequency and presence
/// penalties.
#[cfg(feature = "ane")]
pub fn generate_cached_ane_streaming<F>(
    model_path: &std::path::Path,
    draft_path: Option<&std::path::Path>,
    input_ids: &[u32],
    gen_config: &GenerationConfig,
    ane_max_context: usize,
    mut on_token: F,
) -> std::result::Result<GenerationOutput, pmetal_metal::error::MetalError>
where
    F: FnMut(u32) -> bool,
{
    use pmetal_metal::ane::lm::GenerateOptions;

    let prompt_len = input_ids.len();
    let context = ane_context(prompt_len, gen_config.max_new_tokens, ane_max_context);
    if prompt_len >= context {
        return Err(pmetal_metal::error::MetalError::InvalidConfig(format!(
            "the prompt is {prompt_len} tokens; the ANE context is {context} \
             (raise it with --ane-max-seq-len)"
        )));
    }
    let opts = GenerateOptions {
        max_new: gen_config.max_new_tokens,
        stop: gen_config.stop_tokens.clone(),
    };
    let request = crate::ane_worker::AneRequest {
        model_path: model_path.to_path_buf(),
        draft_path: draft_path.map(std::path::Path::to_path_buf),
        context,
        input_ids: input_ids.to_vec(),
        opts,
        sampling: gen_config.clone(),
    };
    let mut cancelled = false;
    let (generated, stats) = crate::ane_worker::generate(request, |token| {
        cancelled = !on_token(token);
        !cancelled
    })?;
    tracing::debug!(
        passes = stats.passes + 1,
        guesses_kept = format!("{}/{}", stats.accepted_tokens, stats.drafted_tokens),
        draft_ms = format!("{:.0}", stats.draft_secs * 1e3),
        "ANE decode"
    );

    // Stop tokens end generation without being part of the output.
    let saw_stop_token = generated
        .last()
        .is_some_and(|t| gen_config.stop_tokens.contains(t));
    let mut token_ids = input_ids.to_vec();
    token_ids.extend(&generated[..generated.len() - usize::from(saw_stop_token)]);
    Ok(build_cached_generation_output(
        prompt_len,
        token_ids,
        gen_config,
        cancelled,
        saw_stop_token,
    ))
}

/// Check if a model config is compatible with ANE inference.
#[cfg(feature = "ane")]
pub fn is_ane_inference_compatible(
    config_json: &serde_json::Value,
) -> std::result::Result<(), String> {
    pmetal_metal::ane::lm::check_supported(config_json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::token_probability_from_log_probs;
    use pmetal_runtime::decode::greedy_token;

    /// A stand-in model whose logits at each position depend only on the
    /// token there, a fixed pseudo-random table over a 32-token vocabulary:
    /// cheap, deterministic, and varied enough that sampling, penalties and
    /// a bias all change what is picked.
    fn table_logits(tokens: &[i32]) -> Array {
        const VOCAB: usize = 32;
        let mut data = Vec::with_capacity(tokens.len() * VOCAB);
        for &token in tokens {
            for j in 0..VOCAB {
                data.push(((token as usize * 7 + j * 13 + token as usize * j) % 17) as f32 * 0.35);
            }
        }
        Array::from_slice(&data, &[1, tokens.len() as i32, VOCAB as i32])
    }

    fn table_cached(input: &Array, _cache: &mut KVCache) -> Result<Array, Exception> {
        input.eval();
        Ok(table_logits(input.as_slice::<i32>()))
    }

    fn table_uncached(input: &Array) -> Result<Array, Exception> {
        input.eval();
        Ok(table_logits(input.as_slice::<i32>()))
    }

    fn tiny_cache() -> KVCache {
        KVCache::new(pmetal_cache::kv_cache::KVCacheConfig::new(1, 64, 1, 4))
    }

    const PROMPT: [u32; 4] = [3, 9, 1, 20];

    /// All three generators on `config`: the uncached reference, the
    /// cached one and the pipelined one (streaming into the returned vec).
    fn run_all(config: &GenerationConfig) -> [GenerationOutput; 3] {
        let plain = generate(table_uncached, &PROMPT, config.clone()).unwrap();
        let cached =
            generate_cached(table_cached, &PROMPT, config.clone(), &mut tiny_cache()).unwrap();
        let mut streamed = Vec::new();
        let pipelined = generate_cached_async_streaming(
            table_cached,
            &PROMPT,
            config.clone(),
            &mut tiny_cache(),
            |t| {
                streamed.push(t);
                true
            },
        )
        .unwrap();
        pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
        assert_eq!(streamed, pipelined.token_ids[PROMPT.len()..], "streamed");
        [plain, cached, pipelined]
    }

    /// The uncached, the cached and the pipelined generators are one decode:
    /// the same tokens, greedy or drawn from a seed (the `n`-th token takes
    /// the `n`-th draw in every one), under the penalties and a bias, and a
    /// stop token ends all of them without being emitted.
    #[test]
    fn the_generators_agree_on_every_token() {
        let configs = [
            GenerationConfig::greedy(14),
            GenerationConfig::greedy(14)
                .with_repetition_penalty(1.4)
                .with_frequency_penalty(0.3)
                .with_logit_bias(5, -100.0)
                .with_logit_bias(11, 2.5),
            GenerationConfig::sampling(14, 0.9)
                .with_top_k(8)
                .with_top_p(0.9)
                .with_presence_penalty(0.4)
                .with_seed(17),
        ];
        for (i, config) in configs.iter().enumerate() {
            let [plain, cached, pipelined] = run_all(config);
            assert_eq!(plain.num_generated, 14, "config {i}");
            assert_eq!(cached.token_ids, plain.token_ids, "cached, config {i}");
            assert_eq!(
                pipelined.token_ids, plain.token_ids,
                "pipelined, config {i}"
            );
            for out in [&plain, &cached, &pipelined] {
                assert!(out.stopped_by_length && !out.stopped_by_token, "config {i}");
            }
            let made = &plain.token_ids[PROMPT.len()..];
            if i == 1 {
                assert!(!made.contains(&5), "a ban of -100 holds: {made:?}");
            }

            // The first repeat-free pick is made a stop token: every
            // generator ends before it and leaves it out.
            let stop = made[3];
            let at = made.iter().position(|&t| t == stop).unwrap();
            let stopped = config.clone().with_stop_tokens(vec![stop]);
            let [plain, cached, pipelined] = run_all(&stopped);
            for out in [&plain, &cached, &pipelined] {
                assert_eq!(out.num_generated, at, "config {i}");
                assert!(out.stopped_by_token && !out.stopped_by_length, "config {i}");
                assert!(!out.token_ids[PROMPT.len()..].contains(&stop), "config {i}");
            }
            assert_eq!(cached.token_ids, plain.token_ids, "cached stop, config {i}");
            assert_eq!(
                pipelined.token_ids, plain.token_ids,
                "pipelined stop, config {i}"
            );
        }
    }

    /// The length limit is exact, a limit of zero generates nothing, and a
    /// callback that declines ends the generation with the token it saw.
    #[test]
    fn the_generators_respect_the_limits_and_the_callback() {
        for n in [0usize, 1, 2] {
            let [plain, cached, pipelined] = run_all(&GenerationConfig::greedy(n));
            for out in [&plain, &cached, &pipelined] {
                assert_eq!(out.num_generated, n);
                assert_eq!(out.token_ids.len(), PROMPT.len() + n);
                assert!(out.stopped_by_length);
            }
            assert_eq!(cached.token_ids, plain.token_ids);
            assert_eq!(pipelined.token_ids, plain.token_ids);
        }

        let mut seen = Vec::new();
        let out = generate_cached_async_streaming(
            table_cached,
            &PROMPT,
            GenerationConfig::greedy(10),
            &mut tiny_cache(),
            |t| {
                seen.push(t);
                seen.len() < 3
            },
        )
        .unwrap();
        assert_eq!(out.num_generated, 3);
        assert_eq!(seen, out.token_ids[PROMPT.len()..]);
        assert!(!out.stopped_by_token && !out.stopped_by_length);
    }

    /// The wired limit a generation takes is put back when it returns.
    #[test]
    #[serial_test::serial(wired_limit)]
    fn a_generation_leaves_the_wired_limit_as_it_found_it() {
        let before = pmetal_runtime::inline_array::set_wired_limit(1 << 30);
        let out = generate_cached_async(
            table_cached,
            &PROMPT,
            GenerationConfig::greedy(3),
            &mut tiny_cache(),
        )
        .unwrap();
        assert_eq!(out.num_generated, 3);
        let after = pmetal_runtime::inline_array::set_wired_limit(before);
        assert_eq!(after, 1 << 30, "the wired limit is back");
    }

    /// The sampler draws from exactly the distribution it reports (which
    /// speculative decoding verifies against), under every filter, the
    /// temperature and the penalties together: unseeded draws land on the
    /// kept tokens about as often as that distribution says, and a seed fixes
    /// the draw.
    #[test]
    fn the_sampler_draws_from_its_reported_distribution() {
        let config = GenerationConfig::sampling(1, 0.7)
            .with_top_k(6)
            .with_top_p(0.9)
            .with_min_p(0.05)
            .with_repetition_penalty(1.3)
            .with_frequency_penalty(0.2);
        let logits = Array::from_slice(
            &[2.0f32, 1.8, 1.5, 1.4, 1.0, 0.6, 0.5, -0.3, -1.0, -2.0],
            &[1, 10],
        );
        let prompt = [0u32, 3];
        let fresh = |config: &GenerationConfig| {
            let mut sampler = config.sampler(&prompt);
            sampler.record_id(1);
            sampler
        };
        let want = fresh(&config).log_probs(&logits);
        want.eval();
        let want: Vec<f32> = want.as_slice::<f32>().iter().map(|v| v.exp()).collect();
        assert!(want.iter().filter(|&&p| p > 0.0).count() > 2, "{want:?}");
        assert!(want.contains(&0.0), "{want:?}");

        let draws = 4000;
        let mut counts = [0usize; 10];
        for _ in 0..draws {
            counts[fresh(&config).sample(&logits).item_u32() as usize] += 1;
        }
        for (i, (&count, &p)) in counts.iter().zip(&want).enumerate() {
            let freq = count as f32 / draws as f32;
            assert!(
                (freq - p).abs() < 0.03,
                "token {i}: drawn {freq:.3} of the time, probability {p:.3}"
            );
        }

        // A seed makes the first draw the same token every time.
        let seeded = config.clone().with_seed(11);
        let first = fresh(&seeded).sample(&logits).item_u32();
        assert!(want[first as usize] > 0.0);
        for _ in 0..20 {
            assert_eq!(fresh(&seeded).sample(&logits).item_u32(), first);
        }
        pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
    }

    /// A model's `generation_config.json` sets the defaults the sampler
    /// uses: the filters a model card recommends, its stop tokens (one or
    /// several), and a length.
    #[test]
    fn a_model_directory_sets_the_sampling_defaults() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.02,
                "repetition_penalty": 1.05, "do_sample": true,
                "max_new_tokens": 99, "eos_token_id": [7, 8]}"#,
        )
        .unwrap();
        let config = GenerationConfig::from_model_dir(dir.path());
        assert_eq!(
            (config.temperature, config.top_k, config.top_p, config.min_p),
            (0.6, 20, 0.95, 0.02)
        );
        assert_eq!(config.repetition_penalty, 1.05);
        assert_eq!(config.max_new_tokens, 99);
        assert_eq!(config.stop_tokens, vec![7, 8]);
        // No file, no change.
        let empty = tempfile::tempdir().unwrap();
        assert_eq!(
            GenerationConfig::from_model_dir(empty.path()).top_k,
            GenerationConfig::default().top_k
        );
    }

    #[test]
    fn stop_tokens_are_the_configs() {
        let config = GenerationConfig::default().with_stop_tokens(vec![2, 50256]);
        assert!(config.is_stop_token(2));
        assert!(config.is_stop_token(50256));
        assert!(!config.is_stop_token(100));
    }

    /// Reading the sampler's distribution takes nothing into its penalties;
    /// drawing a token does, and the next draw is penalized for it.
    #[test]
    fn reading_the_distribution_records_nothing_and_drawing_does() {
        let config = GenerationConfig::greedy(4)
            .with_frequency_penalty(1.0)
            .with_presence_penalty(0.5);
        let logits = Array::from_f32_slice(&[5.0, 5.0, 5.0, 5.0, 5.0], &[1, 5]);
        let mut sampler = config.sampler(&[]);
        for token in [0, 2, 2, 2] {
            sampler.record_id(token);
        }

        // Token 0: 5 - 1 - 0.5 = 3.5; token 2: 5 - 3 - 0.5 = 1.5, so the
        // greedy point mass is on token 1, the first of the untouched ones.
        for _ in 0..2 {
            let log_probs = sampler.log_probs(&logits);
            assert_eq!(
                token_probability_from_log_probs(&log_probs, 1).unwrap(),
                1.0
            );
        }
        // Tokens 1, 3 and 4 tie; each pick is penalized for the next.
        let picks: Vec<u32> = (0..3).map(|_| sampler.sample(&logits).item_u32()).collect();
        assert_eq!(picks, [1, 3, 4]);
        pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
    }

    #[test]
    fn token_logprobs_chosen_matches_log_softmax() {
        // [1, 5] logits — uniform vocabulary size 5.
        let logits = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0], &[1, 5]);
        let (chosen, top) = token_logprobs(&logits, 3, 0).unwrap();

        // Manual log-softmax at index 3
        let max = 4.0f32;
        let sum_exp: f32 = [0.0, 1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|x| (x - max).exp())
            .sum();
        let expected = 3.0 - max - sum_exp.ln();
        assert!(
            (chosen - expected).abs() < 1e-5,
            "chosen {chosen} vs expected {expected}"
        );
        assert!(top.is_empty(), "top_n = 0 should return empty list");
    }

    #[test]
    fn token_logprobs_top_n_is_sorted_descending() {
        // Known-ordering logits: [0, 1, 2, 3, 4] → biggest logprob is idx 4.
        let logits = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0], &[1, 5]);
        let (_, top) = token_logprobs(&logits, 0, 3).unwrap();
        assert_eq!(top.len(), 3);
        // Highest-logprob first: tokens 4, 3, 2.
        assert_eq!(top[0].0, 4);
        assert_eq!(top[1].0, 3);
        assert_eq!(top[2].0, 2);
        // Descending logprobs.
        assert!(top[0].1 >= top[1].1);
        assert!(top[1].1 >= top[2].1);
    }

    #[test]
    fn token_logprobs_are_computed_in_f32_from_bf16_logits() {
        // Two logits one bf16 step apart: log-softmax is -ln(1 + e^0.125)
        // = -0.75764, which bf16 arithmetic rounds to -0.75.
        let logits = Array::from_slice(&[16.0f32, 16.125], &[2]).cast(Dtype::Bfloat16);
        let (chosen, _) = token_logprobs(&logits, 0, 0).unwrap();
        let expected = -(1.0f32 + 0.125f32.exp()).ln();
        assert!((chosen - expected).abs() < 1e-5, "{chosen} vs {expected}");
    }

    /// Over a vocabulary far larger than `top_n`, with many exact ties, the
    /// device-side top-n is what a full sort of the log-softmax gives.
    #[test]
    fn token_logprobs_top_n_matches_a_full_sort_over_a_large_vocabulary() {
        let vocab = 500usize;
        let row: Vec<f32> = (0..vocab)
            .map(|t| ((t * 41 + 3) % 19) as f32 * 0.3 + f32::from(t == 123) * 4.0)
            .collect();
        let logits = Array::from_slice(&row, &[1, 1, vocab as i32]);
        let lse = row.iter().map(|&x| f64::from(x).exp()).sum::<f64>().ln() as f32;
        let mut want: Vec<(u32, f32)> = row
            .iter()
            .enumerate()
            .map(|(t, &x)| (t as u32, x - lse))
            .collect();
        want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let (chosen, top) = token_logprobs(&logits, 77, 12).unwrap();
        assert!((chosen - (row[77] - lse)).abs() < 1e-4);
        assert_eq!(top.len(), 12);
        for (rank, ((id, lp), (_, want_lp))) in top.iter().zip(&want).enumerate() {
            assert!((lp - want_lp).abs() < 1e-4, "rank {rank}");
            assert!((lp - (row[*id as usize] - lse)).abs() < 1e-4, "rank {rank}");
        }
        assert_eq!(top[0].0, 123);
        pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
    }

    #[test]
    fn token_logprobs_rejects_out_of_range_token() {
        let logits = Array::from_slice(&[0.0f32, 1.0, 2.0], &[1, 3]);
        assert!(token_logprobs(&logits, 3, 0).is_err());
    }

    #[test]
    fn token_logprobs_handles_top_n_larger_than_vocab() {
        let logits = Array::from_slice(&[0.0f32, 1.0, 2.0], &[1, 3]);
        let (_, top) = token_logprobs(&logits, 0, 10).unwrap();
        assert_eq!(top.len(), 3, "top_n is capped by vocab size");
    }

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 256);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_k, 40);
        assert_eq!(config.top_p, 0.95);
        assert_eq!(config.min_p, 0.05);
        assert_eq!(config.repetition_penalty, 1.1);
        assert_eq!(config.frequency_penalty, 0.0);
        assert_eq!(config.presence_penalty, 0.0);
        assert!(config.do_sample);
        assert_eq!(config.prefill_step_size, 2048);
    }

    #[test]
    fn test_generation_config_greedy() {
        let config = GenerationConfig::greedy(100);
        assert_eq!(config.max_new_tokens, 100);
        assert!(!config.do_sample);
    }

    #[test]
    fn test_generation_config_prefill_step_builder() {
        let config = GenerationConfig::greedy(16).with_prefill_step_size(512);
        assert_eq!(config.prefill_step_size, 512);
    }

    #[test]
    fn test_chunked_prefill_splits_prompt_and_returns_last_chunk() {
        let input_ids = vec![11, 22, 33, 44, 55];
        let mut seen_chunks = Vec::new();

        let logits = run_cached_prefill_chunks(&input_ids, 2, |chunk_input| {
            let seq_len = chunk_input.dim(1);
            seen_chunks.push(seq_len);

            let last_token =
                select_axis(&select_axis(chunk_input, 0, 0), seq_len - 1, 0).item::<i32>() as usize;
            let vocab = 64usize;
            let mut data = vec![-1000.0f32; seq_len as usize * vocab];
            data[((seq_len as usize - 1) * vocab) + (last_token % vocab)] = 10.0;
            Ok(Array::from_slice(&data, &[1, seq_len, vocab as i32]))
        })
        .unwrap();

        logits.eval().unwrap();
        assert_eq!(seen_chunks, vec![2, 2, 1]);
        assert_eq!(logits.dim(1), 1);

        let last_logits = select_axis(&logits, 0, 0);
        let token = greedy_token(&last_logits);
        assert_eq!(token as usize, 55);
    }

    #[test]
    fn test_chunked_prefill_zero_step_uses_single_pass() {
        let input_ids = vec![1, 2, 3, 4];
        let mut seen_chunks = Vec::new();

        let _ = run_cached_prefill_chunks(&input_ids, 0, |chunk_input| {
            seen_chunks.push(chunk_input.dim(1));
            Ok(Array::zeros_f32(&[1, chunk_input.dim(1), 8]))
        })
        .unwrap();

        assert_eq!(seen_chunks, vec![4]);
    }

    #[test]
    fn prefill_chunks_shrink_with_depth_to_stay_inside_the_attention_budget() {
        // Whole chunks until the budget binds, at about 76k tokens deep.
        assert_eq!(prefill_chunk_len(2048, 0), 2048);
        assert_eq!(prefill_chunk_len(2048, 70_000), 2048);
        assert_eq!(prefill_chunk_len(2048, 100_000), 1024);
        // Near the end of a 262,144-token window.
        assert_eq!(prefill_chunk_len(2048, 255_000), 512);
        // A million-token YaRN window bottoms out at the floor.
        assert_eq!(prefill_chunk_len(2048, 1_048_576), MIN_PREFILL_CHUNK);
        // A step under the floor is never stretched.
        assert_eq!(prefill_chunk_len(64, 10_000_000), 64);
        for depth in [0, 50_000, 80_000, 150_000, 262_144, 600_000] {
            let len = prefill_chunk_len(2048, depth);
            assert!(len.is_power_of_two());
            assert!(
                len == MIN_PREFILL_CHUNK || len * (depth + 2048) <= PREFILL_ATTENTION_BUDGET,
                "{len} tokens at depth {depth}"
            );
        }
    }

    #[test]
    fn deep_prefill_counts_the_restored_prefix_and_covers_every_token() {
        let input_ids: Vec<u32> = (0..3000).collect();
        let mut seen = Vec::new();
        let _ = run_cached_prefill_chunks_at(&input_ids, 255_000, 2048, |chunk| {
            seen.push(chunk.dim(1) as usize);
            Ok(Array::zeros_f32(&[1, chunk.dim(1), 8]))
        })
        .unwrap();
        assert!(seen[..seen.len() - 1].iter().all(|&len| len == 512));
        assert_eq!(seen.iter().sum::<usize>(), 3000);
    }

    #[test]
    fn test_greedy_token() {
        // Create logits where token 5 has highest value
        let mut logits_vec = vec![-10.0f32; 100];
        logits_vec[5] = 10.0;
        let logits = Array::from_slice(&logits_vec, &[100]);

        let token = greedy_token(&logits);
        assert_eq!(token, 5);
    }

    #[test]
    fn test_qwen3_thinking_preset() {
        let config = GenerationConfig::qwen3_thinking(512);
        assert_eq!(config.max_new_tokens, 512);
        assert_eq!(config.temperature, 0.6);
        assert_eq!(config.top_k, 20);
        assert_eq!(config.top_p, 0.95);
        assert_eq!(config.min_p, 0.0);
        assert_eq!(config.presence_penalty, 1.5); // Prevents repetitions
        assert!(config.do_sample);
    }

    #[test]
    fn test_qwen3_non_thinking_preset() {
        let config = GenerationConfig::qwen3_non_thinking(512);
        assert_eq!(config.max_new_tokens, 512);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_k, 20);
        assert_eq!(config.top_p, 0.8);
        assert_eq!(config.min_p, 0.0);
        assert_eq!(config.presence_penalty, 1.5);
        assert!(config.do_sample);
    }

    #[cfg(feature = "ane")]
    #[test]
    fn test_cached_generation_output_marks_early_stop_as_token_stop() {
        let gen_config = GenerationConfig::greedy(4);
        let output =
            build_cached_generation_output(3, vec![10, 11, 12, 13], &gen_config, false, false);

        assert_eq!(output.num_generated, 1);
        assert!(output.stopped_by_token);
        assert!(!output.stopped_by_length);
    }

    #[cfg(feature = "ane")]
    #[test]
    fn test_cached_generation_output_preserves_cancellation_state() {
        let gen_config = GenerationConfig::greedy(4);
        let output =
            build_cached_generation_output(3, vec![10, 11, 12, 13], &gen_config, true, false);

        assert_eq!(output.num_generated, 1);
        assert!(!output.stopped_by_token);
        assert!(!output.stopped_by_length);
    }

    #[cfg(feature = "ane")]
    #[test]
    fn test_is_ane_inference_compatible_rejects_hybrid_architectures() {
        let dense: serde_json::Value = serde_json::json!({
            "model_type": "qwen3",
            "num_experts": 0,
            "num_local_experts": 0
        });
        assert!(is_ane_inference_compatible(&dense).is_ok());

        // The engine applies per-head q/k norm, which Llama doesn't have (#34).
        let llama: serde_json::Value = serde_json::json!({"model_type": "llama"});
        assert!(is_ane_inference_compatible(&llama).is_err());

        let hybrid: serde_json::Value = serde_json::json!({
            "model_type": "qwen3_next",
            "num_experts": 0,
            "num_local_experts": 0
        });
        assert!(is_ane_inference_compatible(&hybrid).is_err());
    }

    #[cfg(feature = "ane")]
    #[test]
    fn ane_context_fits_the_request_in_a_power_of_two() {
        assert_eq!(ane_context(10, 20, 4096), 512);
        assert_eq!(ane_context(600, 100, 4096), 1024);
        assert_eq!(ane_context(3000, 2000, 4096), 4096);
        // A cap under the floor still gets the floor.
        assert_eq!(ane_context(10, 20, 64), 512);
    }
}
