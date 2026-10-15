//! Text generation for language models.
//!
//! This module provides SOTA sampling techniques:
//! - Greedy decoding
//! - Temperature sampling
//! - Top-k sampling
//! - Top-p (nucleus) sampling
//! - Min-p sampling (dynamic threshold based on top token probability)
//! - Repetition penalty (applied to prompt + output)
//! - Frequency penalty (proportional to appearance count)
//! - Presence penalty (flat penalty for appeared tokens)
//! - Stop token handling
//! - KV-cached generation for fast inference
//!
//! Performance optimizations:
//! - GPU-native sampling filters (top_k, top_p, min_p) - no CPU round-trips
//! - Uses pmetal_runtime::compat::random::categorical for GPU-native categorical sampling
//! - Dedicated generation stream for parallel execution
//! - All tensor operations stay on GPU until final token extraction
use pmetal_mlx::kv_cache::KVCache;
use pmetal_runtime::compat::ops::{
    argmax_axis, async_eval, logsumexp_axis_keepdims, select_axis, take_along_axis,
};
use pmetal_runtime::compat::random::categorical;
use pmetal_runtime::compat::{Array, Dtype, Exception};
use pmetal_runtime::decode::{
    DecodePenalties, SamplingParams, filtered_log_probs, last_token_logits, prompt_tokens_to_input,
};

/// The KV cache an ANE request needs: prompt and output, rounded up to a
/// power of two (so nearby requests share a compiled model), 512 to `cap`.
#[cfg(feature = "ane")]
fn ane_context(prompt_len: usize, max_new: usize, cap: usize) -> usize {
    (prompt_len + max_new)
        .next_power_of_two()
        .clamp(512, cap.max(512))
}

/// Set the wired memory limit for Metal.
///
/// This matches Python's `mx.set_wired_limit()` which prevents page faults
/// and ensures GPU memory stays resident.
///
/// # Returns
/// The previous wired limit.
#[inline]
fn set_wired_limit(limit: usize) -> usize {
    pmetal_runtime::inline_array::set_wired_limit(limit)
}

/// Get the Metal device info to determine optimal wired limit.
///
/// Returns the max recommended working set size.
fn get_max_recommended_wired_limit() -> usize {
    pmetal_runtime::inline_array::get_max_recommended_size()
}

/// RAII guard for managing wired memory limit during generation.
///
/// This matches Python's `wired_limit` context manager that sets
/// the memory limit based on the Metal device's recommended working set size.
pub struct WiredLimitGuard {
    previous_limit: usize,
}

impl WiredLimitGuard {
    /// Create a new wired limit guard, setting the limit to the device's
    /// max recommended working set size.
    pub fn new() -> Self {
        let max_limit = get_max_recommended_wired_limit();
        let previous_limit = set_wired_limit(max_limit);
        WiredLimitGuard { previous_limit }
    }
}

impl Default for WiredLimitGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for WiredLimitGuard {
    fn drop(&mut self) {
        // Restore the previous wired limit
        set_wired_limit(self.previous_limit);
    }
}

/// Make this thread's generation stream the default stream for the MLX ops
/// that follow, creating it on the thread's first call.
///
/// The bridge keeps one generation stream per thread (an MLX stream works
/// only on the thread that made it), so there is no stream value to hold.
/// The stream stays the default after the call returns, async evaluations
/// included; [`pmetal_runtime::inline_array::reset_default_stream`] restores
/// MLX's own.
#[inline]
pub fn use_generation_stream() {
    pmetal_runtime::inline_array::new_generation_stream();
    pmetal_runtime::inline_array::set_generation_stream();
}

/// Release the Metal buffers MLX's allocator holds for reuse.
#[inline]
pub fn clear_generation_caches() {
    pmetal_runtime::inline_array::clear_cache();
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

/// Check if logits are Float32 (avoids clone in hot path).
#[inline]
fn is_f32(logits: &Array) -> bool {
    logits.dtype() == Dtype::Float32
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

    /// The repetition, frequency and presence penalties of a generation
    /// from `prompt`, kept on the GPU, or `None` when this config has none.
    /// Every token the generation emits goes into them, in order.
    pub fn decode_penalties(&self, prompt: &[u32]) -> Option<DecodePenalties> {
        DecodePenalties::new(
            self.repetition_penalty,
            self.frequency_penalty,
            self.presence_penalty,
            prompt,
        )
    }

    /// This config's sampling as the bridge's [`SamplingParams`], which
    /// [`filtered_log_probs`] and the native engines read: greedy
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
        }
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
    /// Generated token IDs (including prompt).
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

/// Acceptance and accounting counters for speculative decoding.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeculativeDecodeMetrics {
    /// Number of draft/assistant tokens proposed to the target verifier.
    pub drafted_tokens: usize,
    /// Number of drafted tokens accepted by the target verifier.
    pub accepted_draft_tokens: usize,
    /// Number of accepted draft tokens actually emitted to the caller.
    pub emitted_draft_tokens: usize,
    /// Target bonus tokens emitted after fully accepted draft blocks.
    pub bonus_tokens: usize,
    /// Target correction tokens emitted after partial or zero-token accept.
    pub correction_tokens: usize,
    /// Number of target verification passes.
    pub verify_steps: usize,
    /// Verification passes where every drafted token was accepted.
    pub full_accept_steps: usize,
    /// Verification passes where some but not all drafted tokens were accepted.
    pub partial_accept_steps: usize,
    /// Verification passes where no drafted token was accepted.
    pub zero_accept_steps: usize,
    /// Largest draft block attempted in one speculative step.
    pub max_draft_tokens: usize,
}

impl SpeculativeDecodeMetrics {
    pub fn acceptance_rate(&self) -> f32 {
        if self.drafted_tokens == 0 {
            0.0
        } else {
            self.accepted_draft_tokens as f32 / self.drafted_tokens as f32
        }
    }

    pub fn avg_accepted_per_verify(&self) -> f32 {
        if self.verify_steps == 0 {
            0.0
        } else {
            self.accepted_draft_tokens as f32 / self.verify_steps as f32
        }
    }

    pub fn record_verify_step(&mut self, drafted: usize, accepted: usize) {
        if drafted == 0 {
            return;
        }
        let accepted = accepted.min(drafted);
        self.drafted_tokens += drafted;
        self.accepted_draft_tokens += accepted;
        self.verify_steps += 1;
        self.max_draft_tokens = self.max_draft_tokens.max(drafted);
        if accepted == drafted {
            self.full_accept_steps += 1;
        } else if accepted == 0 {
            self.zero_accept_steps += 1;
        } else {
            self.partial_accept_steps += 1;
        }
    }
}

/// Sampler for token generation.
///
/// Implements SOTA sampling techniques with GPU-native operations.
/// All sampling runs on GPU without CPU round-trips for maximum performance.
///
/// The repetition, frequency and presence penalties read a generation's
/// token history, which the caller keeps beside the sampler for the
/// generation's lifetime ([`decode_penalties`](Self::decode_penalties)).
pub struct Sampler {
    config: GenerationConfig,
    /// `config` as the bridge reads it ([`GenerationConfig::sampling_params`]).
    params: SamplingParams,
    /// How many tokens this sampler has drawn: with a seed, draw `n` uses
    /// [`pmetal_runtime::decode::draw_key`]`(seed, n)`.
    draws: std::cell::Cell<u64>,
}

impl Sampler {
    /// Create a new sampler. A seeded config draws every token with its own
    /// key derived from the seed, so the sampler's tokens don't depend on
    /// the thread it runs on or on any other random draw.
    pub fn new(config: GenerationConfig) -> Self {
        Self {
            params: config.sampling_params(),
            config,
            draws: std::cell::Cell::new(0),
        }
    }

    /// The key for the next sampled draw (`None` when unseeded), counting it.
    fn next_key(&self) -> Option<Array> {
        let draw = self.draws.get();
        self.draws.set(draw + 1);
        self.config
            .seed
            .map(|seed| pmetal_runtime::decode::draw_key(seed, draw))
    }

    /// Sample the next token from `[1, vocab]` logits and read it back:
    /// [`sample_array_penalized`](Self::sample_array_penalized), then a host
    /// sync, for loops that need each token before running the next step.
    ///
    /// Sampling order:
    /// 1. Apply `penalties` (repetition, frequency, presence) on raw logits,
    ///    and take the sampled token into them
    /// 2. Scale by the temperature and filter by top-k, top-p and min-p
    ///    ([`filtered_log_probs`])
    /// 3. Draw from what is left (argmax when greedy)
    ///
    /// `penalties` are the generation's, from
    /// [`decode_penalties`](Self::decode_penalties) over its prompt, passed
    /// for every token it samples.
    pub fn sample(
        &self,
        logits: &Array,
        penalties: Option<&mut DecodePenalties>,
    ) -> Result<u32, Exception> {
        let (token, _log_probs) = self.sample_array_penalized(logits, penalties)?;
        Ok(token.item::<u32>())
    }

    /// Sample the next token, returning Arrays for async pipelining.
    ///
    /// Unlike `sample()`, this method does NOT call `.item()` - the token stays
    /// as an Array on the GPU. This is critical for SOTA performance:
    ///
    /// ```text
    /// Correct pattern:
    ///   sampled = sampler(logprobs)        # Returns Array, stays on GPU
    ///   next_y, next_logprobs = _step(y)   # y is Array, passed directly
    ///   mx.async_eval(next_y, next_logprobs)  # BOTH scheduled async
    ///   yield y.item(), logprobs           # .item() only AFTER async_eval
    /// ```
    ///
    /// The caller should:
    /// 1. Use the returned token Array directly for the next forward pass
    /// 2. Schedule async_eval on the next iteration's outputs
    /// 3. Only call `.item()` on the token AFTER scheduling the next computation
    ///
    /// NOTE: Penalties (repetition, frequency, presence) are NOT applied in this
    /// method; [`sample_array_penalized`](Self::sample_array_penalized)
    /// applies them without a GPU→CPU sync.
    ///
    /// # Returns
    /// `(token_array, log_probs)` - both stay as Arrays for lazy evaluation
    pub fn sample_array(&self, logits: &Array) -> Result<(Array, Array), Exception> {
        // Greedy decoding - GPU-native argmax directly on logits (fastest path)
        // Skips log_softmax since argmax(logits) == argmax(log_softmax(logits)) due to monotonicity
        if !self.config.do_sample {
            let token = greedy_sample_array(logits)?;
            // Return minimal view for log_probs since greedy doesn't need them
            let empty_logprobs = logits.clone(); // Placeholder, not used by greedy callers
            return Ok((token, empty_logprobs));
        }

        let log_probs = filtered_log_probs(logits, &self.params);
        let key = self.next_key();
        let token = categorical(&log_probs, -1, key.as_ref());
        Ok((token, log_probs))
    }

    /// The penalties of a generation from `prompt`, kept on the GPU, or
    /// `None` when this sampler's config has none
    /// ([`GenerationConfig::decode_penalties`]). Pass them to
    /// [`sample_array_penalized`](Self::sample_array_penalized) or
    /// [`sample`](Self::sample) for every token of the generation.
    pub fn decode_penalties(&self, prompt: &[u32]) -> Option<DecodePenalties> {
        self.config.decode_penalties(prompt)
    }

    /// [`sample_array`](Self::sample_array) after `penalties`, which take
    /// the sampled token in without reading it back, so the next step,
    /// queued before this token reaches the host, is penalized for it.
    pub fn sample_array_penalized(
        &self,
        logits: &Array,
        penalties: Option<&mut DecodePenalties>,
    ) -> Result<(Array, Array), Exception> {
        let Some(penalties) = penalties else {
            return self.sample_array(logits);
        };
        let (token, log_probs) = self.sample_array(&penalties.apply(logits))?;
        penalties.record(&token);
        Ok((token, log_probs))
    }

    /// Check if token is a stop token.
    pub fn is_stop_token(&self, token: u32) -> bool {
        self.config.stop_tokens.contains(&token)
    }

    /// Get the generation config.
    pub fn config(&self) -> &GenerationConfig {
        &self.config
    }

    /// Get the random seed used (useful for reproducibility logging).
    pub fn seed(&self) -> u64 {
        self.config.seed.unwrap_or(0)
    }
}

/// One generation's [`Sampler`] and [`DecodePenalties`], deciding tokens from
/// logits an engine hands over on the host (the ANE engine), so they are
/// sampled and penalized as [`generate_cached`] samples and
/// penalizes them on the GPU.
///
/// Each call wraps the row as `[1, vocab]` f32 logits, samples it through
/// [`Sampler::sample`] and takes the token into the penalties, which is what
/// [`TokenSampler`](pmetal_metal::ane::lm::TokenSampler) asks of it: the
/// engines ask for the generation's tokens in order, a speculative one never
/// past a guess it rejects, so the history holds exactly the tokens before
/// each position, and the `n`-th token takes the `n`-th seeded draw.
///
/// Greedy decoding with no penalty has nothing to apply and draws nothing,
/// so it takes the argmax on the host, which is the GPU's argmax (the first
/// maximum), without the round trip: about 0.5 ms a token on a
/// 151,936-token vocabulary, several times a pass when guesses are verified.
#[cfg(feature = "ane")]
pub struct HostLogitsSampler {
    sampler: Sampler,
    penalties: Option<DecodePenalties>,
}

#[cfg(feature = "ane")]
impl HostLogitsSampler {
    /// The sampler of a generation under `config` continuing `prompt`.
    pub fn new(config: &GenerationConfig, prompt: &[u32]) -> Self {
        Self {
            penalties: config.decode_penalties(prompt),
            sampler: Sampler::new(config.clone()),
        }
    }
}

#[cfg(feature = "ane")]
impl pmetal_metal::ane::lm::TokenSampler for HostLogitsSampler {
    fn sample(&mut self, logits: &[f32]) -> pmetal_metal::error::Result<u32> {
        use pmetal_metal::error::MetalError;
        if !self.sampler.config().do_sample && self.penalties.is_none() {
            return pmetal_metal::ane::lm::Greedy.sample(logits);
        }
        let row = Array::from_f32_slice(logits, &[1, logits.len() as i32]);
        let token = self
            .sampler
            .sample(&row, self.penalties.as_mut())
            .map_err(|e| MetalError::Internal(format!("sampling: {e}")))?;
        pmetal_runtime::check_unobserved_error()
            .map_err(|e| MetalError::Internal(format!("sampling: {e}")))?;
        Ok(token)
    }
}

/// Greedy sampling returning Array for async pipelining.
/// No GPU→CPU sync - stays lazy for maximum performance.
/// Greedy sampling - return argmax along last axis to match Python's pattern.
/// Returns shape [batch] for input shape [batch, vocab].
fn greedy_sample_array(logits: &Array) -> Result<Array, Exception> {
    Ok(argmax_axis(logits, -1)) // axis=-1 like Python's mx.argmax(x, axis=-1)
}

/// Convert logits to log probabilities (log-softmax), in f32.
/// logprobs = logits - logsumexp(logits, keepdims=True)
///
/// Computed in f32 whatever the model's dtype, as transformers' `generate`
/// upcasts next-token logits before scoring them. In bf16 the result rounds
/// to the logits' own precision, steps of 1/8 for logits between 16 and 32,
/// so reported logprobs came out coarse and the sampling filters cut on
/// rounded values.
fn logits_to_log_probs(logits: &Array) -> Result<Array, Exception> {
    let owned;
    let logits = if is_f32(logits) {
        logits
    } else {
        owned = logits.cast(Dtype::Float32);
        &owned
    };
    let lse = logsumexp_axis_keepdims(logits, -1, true);
    Ok(logits.subtract(&lse))
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
    let log_probs = logits_to_log_probs(logits)?;
    // Flatten every leading dim so single-row and batched callers both land
    // on a 1-D [vocab] view. The sampler path only ever has one active
    // prediction per step so this is safe.
    let vocab = *log_probs
        .shape()
        .last()
        .ok_or_else(|| Exception::custom("token_logprobs: empty logits shape"))?;
    let flat = log_probs.reshape(&[vocab]);
    flat.eval();
    let values: Vec<f32> = flat.as_slice().to_vec();

    let idx = token as usize;
    let chosen = values
        .get(idx)
        .copied()
        .ok_or_else(|| Exception::custom(format!("token_logprobs: token {token} out of range")))?;

    let mut top = Vec::new();
    if top_n > 0 {
        // O(V log top_n) partial sort via a min-heap of size top_n. Cheaper
        // than a full sort for realistic vocab/top_n ratios (128k / 5).
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;
        #[derive(PartialEq)]
        struct Entry(f32, u32);
        impl Eq for Entry {}
        impl PartialOrd for Entry {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Entry {
            fn cmp(&self, other: &Self) -> Ordering {
                // Min-heap on logprob — reverse the natural order.
                other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
            }
        }
        let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(top_n + 1);
        for (i, &v) in values.iter().enumerate() {
            if heap.len() < top_n {
                heap.push(Entry(v, i as u32));
            } else if let Some(min) = heap.peek() {
                if v > min.0 {
                    heap.pop();
                    heap.push(Entry(v, i as u32));
                }
            }
        }
        let mut collected: Vec<(u32, f32)> = heap.into_iter().map(|e| (e.1, e.0)).collect();
        // Sort descending by logprob.
        collected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        top = collected;
    }

    Ok((chosen, top))
}

/// The exact sampling distribution [`Sampler`] draws from, for `[1, vocab]`
/// logits and the generation's `penalties` (which this reads but does not
/// record into).
///
/// The returned array is normalized log-probabilities after repetition,
/// frequency/presence penalties, temperature, and top-k/top-p/min-p filtering
/// ([`filtered_log_probs`], renormalized over the tokens it keeps). This is
/// useful for speculative sampling verification, where the verifier needs
/// probabilities for both target and draft distributions rather than only the
/// sampled token.
pub fn sampling_log_probs(
    logits: &Array,
    penalties: Option<&mut DecodePenalties>,
    config: &GenerationConfig,
) -> Result<Array, Exception> {
    let filtered = match penalties {
        Some(penalties) => filtered_log_probs(&penalties.apply(logits), &config.sampling_params()),
        None => filtered_log_probs(logits, &config.sampling_params()),
    };
    Ok(filtered.subtract(&logsumexp_axis_keepdims(&filtered, -1, true)))
}

/// The random draws of one speculative generation: the tokens it samples
/// (drafts, corrections, bonus tokens) and the uniforms its acceptance tests
/// compare against.
///
/// Seeded, the `n`-th draw of either kind comes from
/// [`pmetal_runtime::decode::draw_key`]`(seed, n)` (or its bits, for a
/// uniform), as the plain sampler's tokens do: no random state is shared
/// with the thread or with another generation, so a seeded speculative
/// decode picks the same tokens whichever thread runs it and whatever else
/// draws meanwhile. Unseeded, tokens take the thread's MLX key and uniforms
/// a SplitMix64 stream started from the clock.
#[derive(Debug)]
pub struct SpeculativeDraws {
    seed: Option<u64>,
    /// The seed, or the clock when unseeded: where the uniforms come from.
    uniform_seed: u64,
    next: u64,
}

impl SpeculativeDraws {
    /// The draws of a generation with this seed (`None`: unseeded).
    pub fn new(seed: Option<u64>) -> Self {
        let uniform_seed = seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0x1234_5678_9abc_def0, |d| d.as_nanos() as u64)
        });
        Self {
            seed,
            uniform_seed,
            next: 0,
        }
    }

    fn advance(&mut self) -> u64 {
        let draw = self.next;
        self.next += 1;
        draw
    }

    /// Sample a token from normalized log-probabilities.
    pub fn sample(&mut self, log_probs: &Array) -> Result<u32, Exception> {
        let draw = self.advance();
        let key = self
            .seed
            .map(|seed| pmetal_runtime::decode::draw_key(seed, draw));
        let token = categorical(log_probs, -1, key.as_ref());
        token.eval();
        pmetal_runtime::check_unobserved_error()
            .map_err(|e| Exception::custom(format!("speculative sample: {e}")))?;
        Ok(token.item::<u32>())
    }

    /// A uniform in `[0, 1)`, with 24 random bits.
    pub fn uniform(&mut self) -> f32 {
        let draw = self.advance();
        let bits = pmetal_runtime::decode::draw_bits(self.uniform_seed, draw);
        (bits >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// The log-probabilities a speculative decoder samples its correction token
/// from after rejecting a draft token: the residual `norm(max(0, p − q))` of
/// the target's probabilities `p` over the draft's `q` (Leviathan et al.,
/// arXiv 2211.17192, section 2.3; Chen et al., arXiv 2302.01318). With the
/// acceptance test `min(1, p/q)` before it, every token comes out distributed
/// as the target alone would sample it. A residual with no mass (the draft
/// agreeing with the target) leaves the target's distribution.
pub fn correction_log_probs(
    target_log_probs: &Array,
    draft_log_probs: &Array,
) -> Result<Array, Exception> {
    let residual = target_log_probs
        .exp()
        .subtract(&draft_log_probs.exp())
        .maximum(&Array::from_f32(0.0));
    let total = residual.sum_axis(-1, true);
    let total_value = total.item::<f32>();
    pmetal_runtime::check_unobserved_error()
        .map_err(|e| Exception::custom(format!("speculative correction distribution: {e}")))?;
    if !total_value.is_finite() || total_value <= 1e-20 {
        return Ok(target_log_probs.clone());
    }
    Ok(residual.divide(&total).log())
}

/// Return `exp(log_probs[token])`, validating the token against the vocabulary.
pub fn token_probability_from_log_probs(log_probs: &Array, token: u32) -> Result<f32, Exception> {
    let vocab_size = log_probs.dim(-1);
    if token as i32 >= vocab_size {
        return Err(Exception::custom(format!(
            "token_probability_from_log_probs: token {token} out of range for vocab {vocab_size}"
        )));
    }
    let row = if log_probs.ndim() == 1 {
        log_probs.reshape(&[1, vocab_size])
    } else {
        log_probs.reshape(&[-1, vocab_size])
    };
    if row.dim(0) != 1 {
        return Err(Exception::custom(format!(
            "token_probability_from_log_probs expects one distribution row, got {}",
            row.dim(0)
        )));
    }
    let token_index = Array::from_slice(&[token as i32], &[1, 1]);
    let log_prob = take_along_axis(&row, &token_index, -1);
    Ok(log_prob.item::<f32>().exp())
}

/// Simple generation function that works with any model that has a `forward` method.
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
    let sampler = Sampler::new(config.clone());
    let mut penalties = sampler.decode_penalties(input_ids);
    let prompt_len = input_ids.len();

    for _ in 0..config.max_new_tokens {
        let input = prompt_tokens_to_input(&all_tokens);

        // Get logits from model
        let logits = forward_fn(&input)?;
        logits.eval();

        // Logits for the last position, [1, vocab]
        let last_logits = last_token_logits(&logits);

        // Sample next token
        let next_token = sampler.sample(&last_logits, penalties.as_mut())?;

        // Check for stop token
        if sampler.is_stop_token(next_token) {
            let num_generated = all_tokens.len() - prompt_len;
            return Ok(GenerationOutput {
                token_ids: all_tokens,
                num_generated,
                stopped_by_token: true,
                stopped_by_length: false,
                decode_metrics: None,
                speculative_metrics: None,
            });
        }

        all_tokens.push(next_token);
    }

    let num_generated = all_tokens.len() - prompt_len;
    Ok(GenerationOutput {
        token_ids: all_tokens,
        num_generated,
        stopped_by_token: false,
        stopped_by_length: true,
        decode_metrics: None,
        speculative_metrics: None,
    })
}

/// KV-cached generation for efficient autoregressive decoding.
///
/// This function uses a KV cache to avoid recomputing attention for
/// previous tokens, providing 10-50x speedup for longer generations.
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
    // Set wired memory limit for optimal GPU memory management
    let _wired_guard = WiredLimitGuard::new();

    let mut all_tokens: Vec<u32> = input_ids.to_vec();
    let sampler = Sampler::new(config.clone());
    let mut penalties = sampler.decode_penalties(input_ids);
    let prompt_len = input_ids.len();

    // Prefill: process long prompts in chunks to bound peak allocator pressure.
    let logits = run_cached_prefill_chunks(input_ids, config.prefill_step_size, |chunk_input| {
        forward_fn(chunk_input, cache)
    })?;
    logits.eval();

    // Get logits for the last position and sample first new token
    let last_logits = last_token_logits(&logits);

    let mut next_token = sampler.sample(&last_logits, penalties.as_mut())?;

    // Check for stop token
    if sampler.is_stop_token(next_token) {
        let num_generated = 0;
        return Ok(GenerationOutput {
            token_ids: all_tokens,
            num_generated,
            stopped_by_token: true,
            stopped_by_length: false,
            decode_metrics: None,
            speculative_metrics: None,
        });
    }

    all_tokens.push(next_token);

    // Decode: Process one token at a time using cache
    for _ in 1..config.max_new_tokens {
        // Create input for just the new token
        let token_input = Array::from_slice(&[next_token as i32], &[1, 1]);

        // Forward with cache - only processes the new token
        let logits = forward_fn(&token_input, cache)?;
        logits.eval();

        // Get logits for the (only) position
        let last_logits = select_axis(&logits, 0, 1);
        // next squeeze removed - select_axis already reduces dim

        // Sample next token
        next_token = sampler.sample(&last_logits, penalties.as_mut())?;

        // Check for stop token
        if sampler.is_stop_token(next_token) {
            let num_generated = all_tokens.len() - prompt_len;
            return Ok(GenerationOutput {
                token_ids: all_tokens,
                num_generated,
                stopped_by_token: true,
                stopped_by_length: false,
                decode_metrics: None,
                speculative_metrics: None,
            });
        }

        all_tokens.push(next_token);
    }

    let num_generated = all_tokens.len() - prompt_len;
    Ok(GenerationOutput {
        token_ids: all_tokens,
        num_generated,
        stopped_by_token: false,
        stopped_by_length: true,
        decode_metrics: None,
        speculative_metrics: None,
    })
}

/// Generation with async pipelining.
///
/// This function uses true async evaluation pipelining where:
/// - Token N's logits are computed while token N-1 is being extracted
/// - Tokens stay as Arrays on GPU until after next computation is scheduled
/// - This eliminates GPU→CPU sync bottleneck that was limiting performance
///
/// The pattern:
/// ```text
/// y, logprobs = _step(prompt)
/// async_eval(y, logprobs)  // Schedule first token
/// while n < max_tokens:
///     next_y, next_logprobs = _step(y)  // y is Array, passed directly!
///     async_eval(next_y, next_logprobs)  // Schedule BEFORE .item()
///     yield y.item()  // NOW extract (next is already computing)
///     y = next_y  // Swap
/// ```
///
/// The key insight: `.item()` forces GPU→CPU sync. By calling it AFTER
/// scheduling the next forward pass, we overlap computation with extraction.
///
/// ## Memory Management
///
/// Uses wired memory limit (like Python's `wired_limit` context manager) to
/// prevent page faults and ensure GPU memory stays resident. This is critical
/// for consistent high-throughput inference.
///
/// ## Backpressure
///
/// Implements explicit backpressure by waiting for the current token before
/// scheduling the next one. This mimics Python's async_eval behavior where
/// `MAX_ACTIVE_TASKS=10` causes blocking, ensuring perfect GPU pipelining.
///
/// The repetition, frequency and presence penalties are applied on the GPU
/// ([`DecodePenalties`]), so they cost the pipeline no sync.
///
/// # Arguments
/// * `forward_fn` - Function that takes (input_ids, cache) and returns logits
/// * `input_ids` - Initial token IDs (prompt)
/// * `config` - Generation configuration
/// * `cache` - Pre-allocated KV cache
///
/// # Returns
/// Generation output containing all tokens and metadata
pub fn generate_cached_async<F>(
    mut forward_fn: F,
    input_ids: &[u32],
    config: GenerationConfig,
    cache: &mut KVCache,
) -> Result<GenerationOutput, Exception>
where
    F: FnMut(&Array, &mut KVCache) -> Result<Array, Exception>,
{
    // Set wired memory limit for optimal GPU memory management
    // This prevents page faults and matches Python's wired_limit context manager
    let _wired_guard = WiredLimitGuard::new();
    // This is what enables true async pipelining.

    let mut all_tokens: Vec<u32> = input_ids.to_vec();
    let sampler = Sampler::new(config.clone());
    let mut penalties = sampler.decode_penalties(input_ids);
    let prompt_len = input_ids.len();

    // The prefill's last-position logits, [1, vocab].
    let (mut current_y, mut current_logprobs) = {
        let logits =
            run_cached_prefill_chunks(input_ids, config.prefill_step_size, |chunk_input| {
                use_generation_stream();
                forward_fn(chunk_input, cache)
            })?;
        let current_logits = last_token_logits(&logits);
        sampler.sample_array_penalized(&current_logits, penalties.as_mut())?
    };

    // async_eval OUTSIDE stream context (critical for pipelining!)
    async_eval([&current_y, &current_logprobs]);

    // Decode loop with TRUE async pipelining
    //
    // Python pattern: _step() wraps forward in stream context, async_eval outside
    // This enables the GPU to compute next token while CPU extracts current token.
    //
    let mut n = 0;

    loop {
        // 1. Schedule NEXT forward pass if not at max
        let next_pair = if n + 1 < config.max_new_tokens {
            // Forward pass INSIDE stream context
            let (y, lp) = {
                use_generation_stream();
                // Convert Uint32 token to Int32 for model input (argmax returns Uint32)
                let next_input = current_y
                    .as_dtype(pmetal_runtime::compat::Dtype::Int32.as_i32())
                    .reshape(&[1, -1]);
                let next_output = forward_fn(&next_input, cache)?;
                let next_logits = select_axis(&next_output, 0, 1);
                sampler.sample_array_penalized(&next_logits, penalties.as_mut())?
            };
            // async_eval OUTSIDE stream context (enables pipelining)
            async_eval([&y, &lp]);
            Some((y, lp))
        } else {
            None
        };

        // 2. First iteration: force eval on first token (like Python's mx.eval(y) on n==0)
        if n == 0 {
            current_y.eval();
        }

        // 3. Check max tokens BEFORE extraction (like Python)
        if n >= config.max_new_tokens {
            break;
        }

        // 4. Extract current token - blocks naturally while GPU computes next
        let token = current_y.item::<u32>();

        // 5. Check stop token
        if sampler.is_stop_token(token) {
            let num_generated = all_tokens.len() - prompt_len;
            return Ok(GenerationOutput {
                token_ids: all_tokens,
                num_generated,
                stopped_by_token: true,
                stopped_by_length: false,
                decode_metrics: None,
                speculative_metrics: None,
            });
        }

        all_tokens.push(token);

        // Clear cache every 256 tokens (matches Python)
        if n % 256 == 0 && n > 0 {
            clear_generation_caches();
        }

        // 6. Swap if we have next
        if let Some((y, lp)) = next_pair {
            current_y = y;
            current_logprobs = lp;
        }

        n += 1;
    }

    // Keep compiler happy - logprobs used only for async scheduling
    let _ = current_logprobs;

    let num_generated = all_tokens.len() - prompt_len;
    Ok(GenerationOutput {
        token_ids: all_tokens,
        num_generated,
        stopped_by_token: false,
        stopped_by_length: true,
        decode_metrics: None,
        speculative_metrics: None,
    })
}

/// Streaming variant of [`generate_cached_async`] that calls `on_token` with each
/// generated token ID as it's produced, enabling real-time display.
///
/// The callback returns `true` to continue generation or `false` to stop early.
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
    let _wired_guard = WiredLimitGuard::new();

    let mut all_tokens: Vec<u32> = input_ids.to_vec();
    let sampler = Sampler::new(config.clone());
    let mut penalties = sampler.decode_penalties(input_ids);
    let prompt_len = input_ids.len();

    // Per-step decode latencies in nanoseconds. The first step (n=0) is
    // dropped as warmup — it includes prefill fallout that skews the
    // tokens-per-second number. See `build_decode_metrics` below.
    let mut step_ns: Vec<u128> = Vec::with_capacity(config.max_new_tokens);

    let (mut current_y, mut current_logprobs) = {
        let logits =
            run_cached_prefill_chunks(input_ids, config.prefill_step_size, |chunk_input| {
                use_generation_stream();
                forward_fn(chunk_input, cache)
            })?;
        let current_logits = last_token_logits(&logits);
        sampler.sample_array_penalized(&current_logits, penalties.as_mut())?
    };
    async_eval([&current_y, &current_logprobs]);

    let mut n = 0;

    loop {
        let step_start = std::time::Instant::now();
        let next_pair = if n + 1 < config.max_new_tokens {
            let (y, lp) = {
                use_generation_stream();
                let next_input = current_y
                    .as_dtype(pmetal_runtime::compat::Dtype::Int32.as_i32())
                    .reshape(&[1, -1]);
                let next_output = forward_fn(&next_input, cache)?;
                let next_logits = select_axis(&next_output, 0, 1);
                sampler.sample_array_penalized(&next_logits, penalties.as_mut())?
            };
            async_eval([&y, &lp]);
            Some((y, lp))
        } else {
            None
        };

        if n == 0 {
            current_y.eval();
        }

        if n >= config.max_new_tokens {
            break;
        }

        let token = current_y.item::<u32>();
        // End of this decode step (we've materialised the token). Record
        // the timing before the callback fires so stream-print latency
        // doesn't contaminate the metric.
        step_ns.push(step_start.elapsed().as_nanos());

        if sampler.is_stop_token(token) {
            let num_generated = all_tokens.len() - prompt_len;
            return Ok(GenerationOutput {
                token_ids: all_tokens,
                num_generated,
                stopped_by_token: true,
                stopped_by_length: false,
                decode_metrics: build_decode_metrics(&step_ns),
                speculative_metrics: None,
            });
        }

        all_tokens.push(token);

        // Stream token to caller
        if !on_token(token) {
            let num_generated = all_tokens.len() - prompt_len;
            return Ok(GenerationOutput {
                token_ids: all_tokens,
                num_generated,
                stopped_by_token: false,
                stopped_by_length: false,
                decode_metrics: build_decode_metrics(&step_ns),
                speculative_metrics: None,
            });
        }

        if n % 256 == 0 && n > 0 {
            clear_generation_caches();
        }

        if let Some((y, lp)) = next_pair {
            current_y = y;
            current_logprobs = lp;
        }

        n += 1;
    }

    let _ = current_logprobs;

    let num_generated = all_tokens.len() - prompt_len;
    Ok(GenerationOutput {
        token_ids: all_tokens,
        num_generated,
        stopped_by_token: false,
        stopped_by_length: true,
        decode_metrics: build_decode_metrics(&step_ns),
        speculative_metrics: None,
    })
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
    use pmetal_runtime::decode::greedy_token;

    /// The correction distribution is the clipped, renormalized residual of
    /// the target over the draft, and the target when there is none.
    #[test]
    fn the_correction_samples_the_residual_of_target_over_draft() {
        let log = |p: &[f32]| {
            Array::from_slice(
                &p.iter().map(|p| p.ln()).collect::<Vec<_>>(),
                &[1, p.len() as i32],
            )
        };
        let probs = |a: Array| {
            let a = a.exp();
            a.eval();
            a.as_slice::<f32>().to_vec()
        };
        let target = [0.5f32, 0.3, 0.2];
        // Residual (0.3, 0, 0.1) over 0.4.
        let got = probs(correction_log_probs(&log(&target), &log(&[0.2, 0.6, 0.1])).unwrap());
        for (g, w) in got.iter().zip([0.75f32, 0.0, 0.25]) {
            assert!((g - w).abs() < 1e-6, "{got:?}");
        }
        // A draft equal to the target leaves no residual: the target.
        let got = probs(correction_log_probs(&log(&target), &log(&target)).unwrap());
        for (g, w) in got.iter().zip(target) {
            assert!((g - w).abs() < 1e-6, "{got:?}");
        }
        pmetal_runtime::check_unobserved_error().expect("no bridge op failed");
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

    /// The Sampler draws from exactly the distribution `sampling_log_probs`
    /// reports (which speculative decoding verifies against), under every
    /// filter, the temperature and the penalties together, and its seeded
    /// draws land on the kept tokens about as often as that distribution
    /// says.
    #[test]
    fn the_sampler_draws_from_sampling_log_probs() {
        let config = GenerationConfig::sampling(1, 0.7)
            .with_top_k(6)
            .with_top_p(0.9)
            .with_min_p(0.05)
            .with_repetition_penalty(1.3)
            .with_frequency_penalty(0.2)
            .with_seed(11);
        let logits = Array::from_slice(
            &[2.0f32, 1.8, 1.5, 1.4, 1.0, 0.6, 0.5, -0.3, -1.0, -2.0],
            &[1, 10],
        );
        let prompt = [0u32, 3];
        let mut reported = config.decode_penalties(&prompt);
        reported.as_mut().unwrap().record_id(1);
        let want = sampling_log_probs(&logits, reported.as_mut(), &config).unwrap();
        let want: Vec<f32> = want.as_slice::<f32>().iter().map(|v| v.exp()).collect();

        let sampler = Sampler::new(config.clone());
        let mut counts = [0usize; 10];
        let draws = 4000;
        for _ in 0..draws {
            let mut penalties = config.decode_penalties(&prompt);
            penalties.as_mut().unwrap().record_id(1);
            let (token, drawn_from) = sampler
                .sample_array_penalized(&logits, penalties.as_mut())
                .unwrap();
            let drawn_from = drawn_from.subtract(&logsumexp_axis_keepdims(&drawn_from, -1, true));
            let got: Vec<f32> = drawn_from
                .as_slice::<f32>()
                .iter()
                .map(|v| v.exp())
                .collect();
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-6, "{got:?} against {want:?}");
            }
            counts[token.item::<u32>() as usize] += 1;
        }
        pmetal_runtime::check_unobserved_error().expect("no bridge op failed");
        assert!(want.iter().filter(|&&p| p > 0.0).count() > 2, "{want:?}");
        assert!(want.contains(&0.0), "{want:?}");
        for (i, (&count, &p)) in counts.iter().zip(&want).enumerate() {
            let freq = count as f32 / draws as f32;
            assert!(
                (freq - p).abs() < 0.03,
                "token {i}: drawn {freq:.3} of the time, probability {p:.3}"
            );
        }
    }

    #[test]
    fn test_sampling_log_probs_applies_filters() {
        let config = GenerationConfig::sampling(1, 1.0)
            .with_top_k(1)
            .with_top_p(1.0)
            .with_min_p(0.0)
            .with_repetition_penalty(1.0);
        let logits = Array::from_slice(&[0.0f32, 1.0, 3.0], &[3]);
        let log_probs = sampling_log_probs(&logits, None, &config).unwrap();

        let top_prob = token_probability_from_log_probs(&log_probs, 2).unwrap();
        let masked_prob = token_probability_from_log_probs(&log_probs, 1).unwrap();
        assert!((top_prob - 1.0).abs() < 1e-6);
        assert_eq!(masked_prob, 0.0);
    }

    #[test]
    fn test_speculative_decode_metrics_accumulate_verify_steps() {
        let mut metrics = SpeculativeDecodeMetrics::default();

        metrics.record_verify_step(3, 3);
        metrics.record_verify_step(3, 1);
        metrics.record_verify_step(2, 0);
        metrics.record_verify_step(0, 0);
        metrics.record_verify_step(2, 5);

        assert_eq!(metrics.drafted_tokens, 10);
        assert_eq!(metrics.accepted_draft_tokens, 6);
        assert_eq!(metrics.verify_steps, 4);
        assert_eq!(metrics.full_accept_steps, 2);
        assert_eq!(metrics.partial_accept_steps, 1);
        assert_eq!(metrics.zero_accept_steps, 1);
        assert_eq!(metrics.max_draft_tokens, 3);
        assert!((metrics.acceptance_rate() - 0.6).abs() < f32::EPSILON);
        assert!((metrics.avg_accepted_per_verify() - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sampler_greedy() {
        let config = GenerationConfig::greedy(10);
        let sampler = Sampler::new(config);

        let mut logits_vec = vec![-10.0f32; 100];
        logits_vec[42] = 10.0;
        let logits = Array::from_slice(&logits_vec, &[100]);

        let token = sampler.sample(&logits, None).unwrap();
        assert_eq!(token, 42);
    }

    #[test]
    fn test_stop_token_detection() {
        let config = GenerationConfig::default().with_stop_tokens(vec![2, 50256]);
        let sampler = Sampler::new(config);

        assert!(sampler.is_stop_token(2));
        assert!(sampler.is_stop_token(50256));
        assert!(!sampler.is_stop_token(100));
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

    /// `sampling_log_probs` reads the generation's penalties without taking
    /// anything into them; the `Sampler` takes in every token it samples.
    #[test]
    fn sampling_reads_the_penalties_and_the_sampler_records_into_them() {
        let config = GenerationConfig::greedy(4)
            .with_frequency_penalty(1.0)
            .with_presence_penalty(0.5);
        let logits = Array::from_f32_slice(&[5.0, 5.0, 5.0, 5.0, 5.0], &[1, 5]);
        let mut penalties = config.decode_penalties(&[]);
        for token in [0, 2, 2, 2] {
            penalties.as_mut().unwrap().record_id(token);
        }

        // Token 0: 5 - 1 - 0.5 = 3.5; token 2: 5 - 3 - 0.5 = 1.5.
        for _ in 0..2 {
            let log_probs = sampling_log_probs(&logits, penalties.as_mut(), &config).unwrap();
            let p = |token| token_probability_from_log_probs(&log_probs, token).unwrap();
            assert!((p(1) / p(0) - 1.5f32.exp()).abs() < 1e-3);
            assert!((p(1) / p(2) - 3.5f32.exp()).abs() < 1e-2);
            assert!((p(1) - p(3)).abs() < 1e-6);
        }

        // Tokens 1, 3 and 4 tie; each pick is penalized for the next.
        let sampler = Sampler::new(config);
        let picks: Vec<u32> = (0..3)
            .map(|_| sampler.sample(&logits, penalties.as_mut()).unwrap())
            .collect();
        assert_eq!(picks, [1, 3, 4]);
        pmetal_runtime::check_unobserved_error().expect("no bridge op failed");
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
