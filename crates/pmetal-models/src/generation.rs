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
//! Performance optimizations (matching mlx_lm exactly):
//! - GPU-native sampling filters (top_k, top_p, min_p) - no CPU round-trips
//! - Uses mlx_rs::random::categorical for GPU-native categorical sampling
//! - Dedicated generation stream for parallel execution
//! - All tensor operations stay on GPU until final token extraction

use mlx_rs::{
    error::Exception,
    ops::{
        argpartition_axis, argsort_axis, exp, expand_dims_axes,
        indexing::{argmax, argmax_axis, put_along_axis, take_along_axis, IndexOp},
        logsumexp_axis, squeeze_axes, which, zeros_like,
    },
    random::{categorical, seed as mlx_seed},
    transforms::async_eval,
    Array, Device, Dtype, Stream,
};
use pmetal_mlx::kv_cache::KVCache;
use std::collections::HashMap;

/// Wait for a specific array to be ready (computed).
///
/// This is the key to matching Python's async_eval behavior. Python's async_eval
/// internally blocks when there's memory pressure (MAX_ACTIVE_TASKS=10), waiting
/// for previous work to complete. This creates perfect pipelining where:
///
/// - Python: blocking happens in async_eval, GPU is productive
/// - Rust without fix: blocking happens in item(), GPU may be idle
/// - Rust with fix: we explicitly wait before scheduling next work
///
/// By calling array_wait(current_y) BEFORE scheduling the next computation,
/// we ensure the GPU has finished the current token and can immediately
/// start the next one. The subsequent item() call is instant.
#[inline]
fn array_wait(arr: &Array) {
    // SAFETY:
    // 1. arr.as_ptr() returns the mlx_array pointer for this Array
    // 2. _mlx_array_wait is an internal MLX C API function that blocks until
    //    the array's computation is complete
    // 3. The array remains valid (we have a reference to it)
    // 4. This is a read-only operation that doesn't modify the array
    unsafe {
        mlx_sys::_mlx_array_wait(arr.as_ptr());
    }
}

/// Check if an array is available (computed) without blocking.
///
/// Returns true if the array's data is ready on CPU.
#[inline]
#[allow(dead_code)]
fn array_is_available(arr: &Array) -> bool {
    let mut result: bool = false;
    // SAFETY:
    // 1. arr.as_ptr() returns the mlx_array pointer for this Array
    // 2. _mlx_array_is_available is an internal MLX C API function that checks
    //    if the array's data is available without blocking
    // 3. We pass a valid mutable pointer to result for the output
    // 4. This is a read-only query that doesn't modify the array
    unsafe {
        mlx_sys::_mlx_array_is_available(&mut result, arr.as_ptr());
    }
    result
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
    let mut result: usize = 0;
    // SAFETY:
    // 1. mlx_set_wired_limit is a public MLX C API function
    // 2. We pass a valid mutable pointer to result for the output
    // 3. limit is a valid usize value representing bytes
    // 4. This configures Metal's memory allocation behavior globally
    unsafe {
        mlx_sys::mlx_set_wired_limit(&mut result, limit);
    }
    result
}

/// Get the Metal device info to determine optimal wired limit.
///
/// Returns the max recommended working set size.
fn get_max_recommended_wired_limit() -> usize {
    // SAFETY:
    // 1. mlx_metal_device_info is a public MLX C API function
    // 2. It returns a struct containing Metal device information
    // 3. The function queries the default Metal device (GPU)
    // 4. The returned struct is copied by value, no lifetime concerns
    let info = unsafe { mlx_sys::mlx_metal_device_info() };
    info.max_recommended_working_set_size
}

/// RAII guard for managing wired memory limit during generation.
///
/// This matches Python's `wired_limit` context manager that sets
/// the memory limit based on the Metal device's recommended working set size.
struct WiredLimitGuard {
    previous_limit: usize,
}

impl WiredLimitGuard {
    /// Create a new wired limit guard, setting the limit to the device's
    /// max recommended working set size.
    fn new() -> Self {
        let max_limit = get_max_recommended_wired_limit();
        let previous_limit = set_wired_limit(max_limit);
        WiredLimitGuard { previous_limit }
    }
}

impl Drop for WiredLimitGuard {
    fn drop(&mut self) {
        // Restore the previous wired limit
        set_wired_limit(self.previous_limit);
    }
}

/// A RAII guard that sets a stream as the default for the duration of its lifetime.
///
/// This mirrors Python's `with mx.stream(stream):` context manager, which actually
/// calls the C++ `set_default_stream()` function. The mlx-rs `with_new_default_stream`
/// only sets a Rust thread-local and doesn't affect the C++ scheduler.
struct StreamContext {
    previous_stream: Stream,
}

impl StreamContext {
    /// Create a new stream context, setting the given stream as the default.
    fn new(stream: &Stream) -> Self {
        // Get the current default stream to restore later
        let previous_stream = Stream::gpu();

        // SAFETY:
        // 1. mlx_set_default_stream is a public MLX C API function
        // 2. stream.as_ptr() returns a valid mlx_stream pointer
        // 3. This sets the thread-local default stream for MLX operations
        // 4. The stream must remain valid while it's the default
        unsafe {
            mlx_sys::mlx_set_default_stream(stream.as_ptr());
        }

        StreamContext { previous_stream }
    }
}

impl Drop for StreamContext {
    fn drop(&mut self) {
        // SAFETY:
        // 1. mlx_set_default_stream is a public MLX C API function
        // 2. previous_stream was obtained from Stream::gpu() which returns a valid stream
        // 3. We're restoring the previous state, which is always valid
        unsafe {
            mlx_sys::mlx_set_default_stream(self.previous_stream.as_ptr());
        }
    }
}

/// Create a new generation stream.
///
/// Note: We'd like to cache this like Python's module-level `generation_stream`,
/// but mlx-rs Stream contains a raw pointer that doesn't implement Sync.
/// Stream creation is cheap (~0.001ms) so this is acceptable.
#[inline]
fn create_generation_stream() -> Stream {
    Stream::new_with_device(&Device::gpu())
}

/// Ensure logits are in Float32 format for sampling operations.
/// Converts BFloat16/Float16 to Float32 if needed.
/// Note: Does NOT call eval() - keeps operations lazy for GPU pipelining.
fn ensure_f32(logits: &Array) -> Result<Array, Exception> {
    match logits.dtype() {
        Dtype::Float32 => Ok(logits.clone()),
        Dtype::Bfloat16 | Dtype::Float16 => logits.as_type::<f32>(),
        _ => logits.as_type::<f32>(),
    }
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
        }
    }
}

impl GenerationConfig {
    /// Create a greedy decoding config.
    pub fn greedy(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            do_sample: false,
            temperature: 1.0,
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
        }
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

    /// Set temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
}

/// Output from text generation.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    /// Generated token IDs (including prompt).
    pub token_ids: Vec<u32>,
    /// Number of tokens generated (excluding prompt).
    pub num_generated: usize,
    /// Whether generation stopped due to stop token.
    pub stopped_by_token: bool,
    /// Whether generation stopped due to max length.
    pub stopped_by_length: bool,
}

/// Sampler for token generation.
///
/// Implements SOTA sampling techniques with GPU-native operations.
/// All sampling runs on GPU without CPU round-trips for maximum performance.
pub struct Sampler {
    config: GenerationConfig,
    /// Token frequency counts for frequency penalty
    token_counts: HashMap<u32, usize>,
    /// Tracks whether MLX RNG was seeded (stored for debugging/introspection).
    /// The actual seeding happens as a side-effect during construction.
    #[allow(dead_code)]
    seeded: bool,
    /// Cached -inf scalar for filter operations (avoids allocation per token)
    neg_inf: Array,
    /// Cached inverse temperature for categorical sampling (1.0 / temp)
    /// None if temp == 1.0 (no scaling needed)
    inv_temp: Option<Array>,
    /// Cached vocab_range array for filter operations [0, 1, 2, ..., vocab_size-1]
    /// Lazily initialized on first filter use. Avoids ~600KB allocation per filter call.
    cached_vocab_range: Option<Array>,
    /// Cached vocab size to detect when vocab_range needs regeneration
    cached_vocab_size: usize,
}

impl Sampler {
    /// Create a new sampler.
    pub fn new(config: GenerationConfig) -> Self {
        // Seed MLX random state if a seed was provided
        let seeded = if let Some(seed) = config.seed {
            let _ = mlx_seed(seed);
            true
        } else {
            false
        };

        // Pre-allocate cached scalar arrays (avoids allocation per token)
        let neg_inf = Array::from_f32(f32::NEG_INFINITY);

        // Cache inverse temperature for categorical sampling
        // If temp == 1.0, no scaling needed so we store None
        let inv_temp = if config.do_sample && config.temperature != 1.0 && config.temperature > 0.0
        {
            Some(Array::from_f32(1.0 / config.temperature))
        } else {
            None
        };

        Self {
            config,
            token_counts: HashMap::new(),
            seeded,
            neg_inf,
            inv_temp,
            cached_vocab_range: None,
            cached_vocab_size: 0,
        }
    }

    /// Get cached vocab_range array, creating it if needed.
    /// This avoids allocating ~600KB per filter call for large vocabularies.
    #[inline]
    fn get_vocab_range(&mut self, vocab_size: usize) -> &Array {
        if self.cached_vocab_size != vocab_size || self.cached_vocab_range.is_none() {
            self.cached_vocab_range = Some(Array::from_iter(
                0..vocab_size as i32,
                &[1, vocab_size as i32],
            ));
            self.cached_vocab_size = vocab_size;
        }
        self.cached_vocab_range.as_ref().unwrap()
    }

    /// Sample the next token from logits.
    ///
    /// Matches mlx_lm's sampling approach exactly:
    /// 1. Apply penalties (repetition, frequency, presence) on raw logits
    /// 2. Convert logits to log probabilities: logprobs = logits - logsumexp(logits)
    /// 3. Apply filters on log probabilities (top_k, top_p, min_p)
    /// 4. Categorical sampling: categorical(logprobs * (1/temp))
    ///
    /// All operations are GPU-native with lazy evaluation.
    /// Only the final token extraction requires CPU sync.
    ///
    /// # Arguments
    /// * `logits` - Logits for the next token [vocab_size]
    /// * `generated_tokens` - Previously generated tokens (for repetition penalty)
    ///
    /// # Returns
    /// The sampled token ID
    pub fn sample(&mut self, logits: &Array, generated_tokens: &[u32]) -> Result<u32, Exception> {
        // Ensure logits are in Float32 format (handles BFloat16 models)
        // Note: No eval() - keeps operations lazy
        let mut logits = ensure_f32(logits)?;

        // Apply repetition penalty on raw logits (before log-softmax)
        if self.config.repetition_penalty != 1.0 && !generated_tokens.is_empty() {
            logits = apply_repetition_penalty(
                &logits,
                generated_tokens,
                self.config.repetition_penalty,
            )?;
        }

        // Apply frequency and presence penalties on raw logits
        if (self.config.frequency_penalty != 0.0 || self.config.presence_penalty != 0.0)
            && !self.token_counts.is_empty()
        {
            logits = apply_frequency_presence_penalty(
                &logits,
                &self.token_counts,
                self.config.frequency_penalty,
                self.config.presence_penalty,
            )?;
        }

        // Greedy decoding - GPU-native argmax (no log-softmax needed)
        if !self.config.do_sample {
            let token = greedy_sample(&logits)?;
            self.update_counts(token);
            return Ok(token);
        }

        // Convert logits to log probabilities (exactly like mlx_lm):
        // logprobs = logits - logsumexp(logits, keepdims=True)
        let log_probs = logits_to_log_probs(&logits)?;

        // Apply fused GPU-native filters (cached arrays, single reshape)
        let log_probs = self.apply_filters_fused(&log_probs)?;

        // Sample from the distribution using GPU-native categorical
        // Matches mlx_lm: categorical(logprobs * (1/temp))
        let token = gpu_categorical_sample(&log_probs, self.config.temperature)?;
        self.update_counts(token);
        Ok(token)
    }

    /// Sample the next token, returning Arrays for async pipelining.
    ///
    /// Unlike `sample()`, this method does NOT call `.item()` - the token stays
    /// as an Array on the GPU. This is critical for SOTA performance:
    ///
    /// ```text
    /// mlx_lm pattern (correct):
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
    /// method to avoid GPU→CPU sync for token history tracking. Use `sample()`
    /// for penalty support, or apply penalties on logits before calling this.
    ///
    /// # Returns
    /// `(token_array, log_probs)` - both stay as Arrays for lazy evaluation
    pub fn sample_array(&self, logits: &Array) -> Result<(Array, Array), Exception> {
        // Greedy decoding - GPU-native argmax directly on logits (fastest path)
        // Skips log_softmax since argmax(logits) == argmax(log_softmax(logits)) due to monotonicity
        if !self.config.do_sample {
            let token = greedy_sample_array(logits)?;
            // Return minimal view for log_probs since greedy doesn't need them
            let empty_logprobs = logits.index((.., ..1)); // Minimal view, not used
            return Ok((token, empty_logprobs));
        }

        // For sampling, ensure Float32 - only convert if not already f32
        // This avoids clone for the common Float32 case
        let owned_logits;
        let logits_f32: &Array = if is_f32(logits) {
            logits // Borrow, no clone needed
        } else {
            owned_logits = logits.as_type::<f32>()?;
            &owned_logits
        };

        // Convert logits to log probabilities (exactly like mlx_lm):
        // logprobs = logits - logsumexp(logits, keepdims=True)
        let log_probs = logits_to_log_probs(logits_f32)?;

        // Apply fused GPU-native filters (cached arrays, single reshape)
        let log_probs = self.apply_filters_fused(&log_probs)?;

        // Sample from the distribution using GPU-native categorical
        // Use cached inv_temp to avoid allocation per token
        let token = if let Some(ref inv_temp) = self.inv_temp {
            // Scale by cached inverse temperature
            let scaled = log_probs.multiply(inv_temp)?;
            categorical(&scaled, None, None, None)?
        } else {
            // No temperature scaling needed (temp == 1.0)
            categorical(&log_probs, None, None, None)?
        };
        Ok((token, log_probs))
    }

    /// Update token frequency counts for frequency penalty.
    fn update_counts(&mut self, token: u32) {
        *self.token_counts.entry(token).or_insert(0) += 1;
    }

    /// Reset token counts (call between generations).
    pub fn reset_counts(&mut self) {
        self.token_counts.clear();
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

    /// Apply all configured filters in a single fused pass.
    ///
    /// Optimizations over separate filter calls:
    /// - Ensures 2D only once at start (not per-filter)
    /// - Reuses cached neg_inf Array (no allocation per token)
    /// - Squeezes back only once at end (not per-filter)
    fn apply_filters_fused(&self, log_probs: &Array) -> Result<Array, Exception> {
        let needs_top_k = self.config.top_k > 0;
        let needs_top_p = self.config.top_p < 1.0 && self.config.top_p > 0.0;
        let needs_min_p = self.config.min_p > 0.0 && self.config.min_p < 1.0;

        // Fast path: no filtering needed - return reference (no clone!)
        if !needs_top_k && !needs_top_p && !needs_min_p {
            // MLX operations create views, so returning as-is is safe
            // The caller's logits_to_log_probs already created a new array
            return Ok(log_probs.clone()); // NOTE: This clone is unavoidable due to return type
        }

        let vocab_size = log_probs.dim(-1) as usize;
        let was_1d = log_probs.ndim() == 1;

        // Ensure 2D once at start - NO clone needed!
        // Filter functions (put_along_axis, which, etc.) create new arrays,
        // so we just need a 2D view to pass to them.
        let input_2d;
        let input_2d_ref: &Array = if was_1d {
            input_2d = log_probs.reshape(&[1, vocab_size as i32])?;
            &input_2d
        } else {
            log_probs // Already 2D, just borrow - no clone!
        };

        // First filter creates a new array, subsequent filters modify that
        let mut result = input_2d_ref.clone(); // Single clone here

        // Apply top-k filter (uses cached neg_inf)
        if needs_top_k {
            result = self.top_k_filter_internal(&result, vocab_size)?;
        }

        // Apply top-p filter (uses cached neg_inf)
        if needs_top_p {
            result = self.top_p_filter_internal(&result, vocab_size)?;
        }

        // Apply min-p filter (uses cached neg_inf)
        if needs_min_p {
            result = self.min_p_filter_internal(&result, vocab_size)?;
        }

        // Squeeze back once at end
        if was_1d {
            result.squeeze()
        } else {
            Ok(result)
        }
    }

    /// Internal top-k filter using cached arrays. Input must be 2D [1, vocab_size].
    fn top_k_filter_internal(
        &self,
        logits_2d: &Array,
        vocab_size: usize,
    ) -> Result<Array, Exception> {
        let k = (self.config.top_k as usize).min(vocab_size);

        // argpartition on -logits gives indices that partition around k-th largest
        let neg_logits = logits_2d.negative()?;
        let mask_idx = argpartition_axis(&neg_logits, (k - 1) as i32, -1)?;
        let mask_idx = mask_idx.index((.., k as i32..));

        // Use cached neg_inf
        put_along_axis(logits_2d, &mask_idx, &self.neg_inf, -1)
    }

    /// Internal top-p filter using cached arrays. Input must be 2D [1, vocab_size].
    fn top_p_filter_internal(
        &self,
        logits_2d: &Array,
        vocab_size: usize,
    ) -> Result<Array, Exception> {
        // Convert logits to probabilities
        let probs = exp(logits_2d)?;

        // Sort indices in ascending order
        let sorted_indices = argsort_axis(logits_2d, -1)?;
        let sorted_probs = take_along_axis(&probs, &sorted_indices, -1)?;
        let cumulative_probs = sorted_probs.cumsum(-1, None, None)?;

        // Create inverse indices to map back
        let vocab_range = Array::from_iter(0..vocab_size as i32, &[1, vocab_size as i32]);
        let inverse_indices = put_along_axis(
            &zeros_like(&sorted_indices)?,
            &sorted_indices,
            &vocab_range,
            -1,
        )?;
        let cumulative_probs = take_along_axis(&cumulative_probs, &inverse_indices, -1)?;

        // Keep tokens where cumulative probability > (1 - top_p)
        let threshold = Array::from_f32(1.0 - self.config.top_p);
        let mask = cumulative_probs.gt(&threshold)?;

        // Use cached neg_inf
        which(&mask, logits_2d, &self.neg_inf)
    }

    /// Internal min-p filter using cached arrays. Input must be 2D [1, vocab_size].
    fn min_p_filter_internal(
        &self,
        logits_2d: &Array,
        vocab_size: usize,
    ) -> Result<Array, Exception> {
        // Sort indices in descending order
        let neg_logits = logits_2d.negative()?;
        let sorted_indices = argsort_axis(&neg_logits, -1)?;
        let sorted_logits = take_along_axis(logits_2d, &sorted_indices, -1)?;

        // Get top logprob
        let top_logits = sorted_logits.index((.., 0..1));
        let log_min_p = Array::from_f32(self.config.min_p.ln());
        let scaled_min_p = top_logits.add(&log_min_p)?;

        // Mask tokens below threshold
        let tokens_to_remove = sorted_logits.lt(&scaled_min_p)?;
        let selected_logits = which(&tokens_to_remove, &self.neg_inf, &sorted_logits)?;

        // Map back to original order
        let vocab_range = Array::from_iter(0..vocab_size as i32, &[1, vocab_size as i32]);
        let inverse_indices = put_along_axis(
            &zeros_like(&sorted_indices)?,
            &sorted_indices,
            &vocab_range,
            -1,
        )?;

        take_along_axis(&selected_logits, &inverse_indices, -1)
    }
}

/// Greedy sampling - returns the token with highest probability.
/// Note: item() internally calls eval(), so no explicit eval() needed.
fn greedy_sample(logits: &Array) -> Result<u32, Exception> {
    let token_id = argmax(logits, None)?;
    Ok(token_id.item::<u32>())
}

/// Greedy sampling returning Array for async pipelining.
/// No GPU→CPU sync - stays lazy for maximum performance.
/// Greedy sampling - return argmax along last axis to match Python's pattern.
/// Returns shape [batch] for input shape [batch, vocab].
fn greedy_sample_array(logits: &Array) -> Result<Array, Exception> {
    argmax_axis(logits, -1, None) // axis=-1 like Python's mx.argmax(x, axis=-1)
}

/// Convert logits to log probabilities (log-softmax).
/// Matches mlx_lm: logprobs = logits - logsumexp(logits, keepdims=True)
fn logits_to_log_probs(logits: &Array) -> Result<Array, Exception> {
    let lse = logsumexp_axis(logits, -1, true)?;
    logits.subtract(&lse)
}

/// GPU-native repetition penalty matching mlx_lm.
///
/// For positive logits: divide by penalty (reduces probability)
/// For negative logits: multiply by penalty (reduces probability)
///
/// All operations stay on GPU - no CPU round-trip.
fn apply_repetition_penalty(
    logits: &Array,
    generated_tokens: &[u32],
    penalty: f32,
) -> Result<Array, Exception> {
    if generated_tokens.is_empty() || penalty == 1.0 {
        return Ok(logits.clone());
    }

    // Ensure logits is 2D [1, vocab_size] for indexing
    let vocab_size = logits.dim(-1) as usize;
    let logits_2d = if logits.ndim() == 1 {
        logits.reshape(&[1, vocab_size as i32])?
    } else {
        logits.clone()
    };

    // Create token indices array (only unique tokens, limit to reasonable context)
    let context_size = 20.min(generated_tokens.len());
    let recent_tokens: Vec<i32> = generated_tokens
        .iter()
        .rev()
        .take(context_size)
        .map(|&t| t as i32)
        .collect();

    if recent_tokens.is_empty() {
        return Ok(logits.clone());
    }

    let token_indices = Array::from_slice(&recent_tokens, &[1, recent_tokens.len() as i32]);

    // Get logits at token positions: logits[:, tokens]
    let selected_logits = take_along_axis(&logits_2d, &token_indices, -1)?;

    // Apply penalty: divide positive, multiply negative
    let zero = Array::from_f32(0.0);
    let penalty_arr = Array::from_f32(penalty);
    let inv_penalty = Array::from_f32(1.0 / penalty);

    // selected < 0 ? selected * penalty : selected / penalty
    let is_negative = selected_logits.lt(&zero)?;
    let penalized_positive = selected_logits.multiply(&inv_penalty)?;
    let penalized_negative = selected_logits.multiply(&penalty_arr)?;
    let penalized = which(&is_negative, &penalized_negative, &penalized_positive)?;

    // Put the penalized values back: logits[:, tokens] = penalized
    let result = put_along_axis(&logits_2d, &token_indices, &penalized, -1)?;

    // Squeeze back to original shape if needed
    if logits.ndim() == 1 {
        result.squeeze()
    } else {
        Ok(result)
    }
}

/// GPU-native top-k filtering - keeps only the k tokens with highest probability.
///
/// Matches mlx_lm's apply_top_k exactly:
/// - Uses argpartition to find the k-th largest element efficiently
/// - Uses put_along_axis to mask out tokens below top-k
/// - All operations stay on GPU - no CPU round-trip
fn top_k_filter(logits: &Array, k: usize) -> Result<Array, Exception> {
    let vocab_size = logits.dim(-1) as usize;
    let k = k.min(vocab_size);

    // Ensure logits is 2D for consistent indexing [1, vocab_size]
    let logits_2d = if logits.ndim() == 1 {
        logits.reshape(&[1, vocab_size as i32])?
    } else {
        logits.clone()
    };

    // argpartition on -logits gives indices that partition around k-th largest
    // Elements before kth are >= kth, elements after are <= kth
    let neg_logits = logits_2d.negative()?;
    let mask_idx = argpartition_axis(&neg_logits, (k - 1) as i32, -1)?;

    // Get indices of tokens to mask (everything after top-k)
    let mask_idx = mask_idx.index((.., k as i32..));

    // Create -inf value for masking
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);

    // Put -inf at the mask indices
    let masked = put_along_axis(&logits_2d, &mask_idx, &neg_inf, -1)?;

    // Squeeze back to original shape if needed
    if logits.ndim() == 1 {
        masked.squeeze()
    } else {
        Ok(masked)
    }
}

/// GPU-native top-p (nucleus) filtering.
///
/// Matches mlx_lm's apply_top_p exactly:
/// - Sorts probs in ascending order
/// - Computes cumulative sum
/// - Keeps tokens with cumsum > (1 - top_p) threshold
/// - All operations stay on GPU - no CPU round-trip
fn top_p_filter(logits: &Array, p: f32) -> Result<Array, Exception> {
    let vocab_size = logits.dim(-1) as usize;

    // Ensure logits is 2D for consistent indexing [1, vocab_size]
    let logits_2d = if logits.ndim() == 1 {
        logits.reshape(&[1, vocab_size as i32])?
    } else {
        logits.clone()
    };

    // Convert logits to probabilities
    let probs = exp(&logits_2d)?;

    // Sort indices in ascending order (by logits, which preserves prob ordering)
    let sorted_indices = argsort_axis(&logits_2d, -1)?;

    // Gather sorted probs
    let sorted_probs = take_along_axis(&probs, &sorted_indices, -1)?;

    // Compute cumulative sum
    let cumulative_probs = sorted_probs.cumsum(-1, None, None)?;

    // Create inverse indices to map back to original order
    let vocab_range = Array::from_iter(0..vocab_size as i32, &[1, vocab_size as i32]);
    let inverse_indices = put_along_axis(
        &zeros_like(&sorted_indices)?,
        &sorted_indices,
        &vocab_range,
        -1,
    )?;

    // Rearrange cumulative probs back to original order
    let cumulative_probs = take_along_axis(&cumulative_probs, &inverse_indices, -1)?;

    // Keep tokens where cumulative probability > (1 - top_p)
    // This matches mlx_lm's logic: select tokens with cumsum > 1 - top_p
    let threshold = Array::from_f32(1.0 - p);
    let mask = cumulative_probs.gt(&threshold)?;

    // Apply mask: keep original logits where mask is true, else -inf
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let result = which(&mask, &logits_2d, &neg_inf)?;

    // Squeeze back to original shape if needed
    if logits.ndim() == 1 {
        result.squeeze()
    } else {
        Ok(result)
    }
}

/// GPU-native min-p filtering - dynamic threshold based on top token probability.
///
/// Matches mlx_lm's apply_min_p exactly:
/// - Works in log-probability space for numerical stability
/// - Computes scaled_min_p = top_logprob + log(min_p)
/// - Masks tokens with logprob < scaled_min_p
/// - All operations stay on GPU - no CPU round-trip
///
/// Unlike top-p which uses a fixed cumulative probability threshold,
/// min-p scales the threshold based on the model's confidence (top token probability).
/// This helps maintain coherence at high temperatures while allowing creativity.
/// Recommended values: 0.05-0.1
fn min_p_filter(logits: &Array, min_p: f32) -> Result<Array, Exception> {
    let vocab_size = logits.dim(-1) as usize;

    // Ensure logits is 2D for consistent indexing [1, vocab_size]
    let logits_2d = if logits.ndim() == 1 {
        logits.reshape(&[1, vocab_size as i32])?
    } else {
        logits.clone()
    };

    // Sort indices in descending order
    let neg_logits = logits_2d.negative()?;
    let sorted_indices = argsort_axis(&neg_logits, -1)?;
    let sorted_logits = take_along_axis(&logits_2d, &sorted_indices, -1)?;

    // Get top logprob (first element after descending sort)
    let top_logits = sorted_logits.index((.., 0..1));

    // Calculate the min_p threshold in log space: scaled_min_p = top_logprob + log(min_p)
    let log_min_p = Array::from_f32(min_p.ln());
    let scaled_min_p = top_logits.add(&log_min_p)?;

    // Find tokens to remove: those with logprob < scaled_min_p
    let tokens_to_remove = sorted_logits.lt(&scaled_min_p)?;

    // Apply mask: -inf where tokens should be removed
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let selected_logits = which(&tokens_to_remove, &neg_inf, &sorted_logits)?;

    // Create inverse indices to map back to original order
    let vocab_range = Array::from_iter(0..vocab_size as i32, &[1, vocab_size as i32]);
    let inverse_indices = put_along_axis(
        &zeros_like(&sorted_indices)?,
        &sorted_indices,
        &vocab_range,
        -1,
    )?;

    // Rearrange back to original order
    let result = take_along_axis(&selected_logits, &inverse_indices, -1)?;

    // Squeeze back to original shape if needed
    if logits.ndim() == 1 {
        result.squeeze()
    } else {
        Ok(result)
    }
}

/// GPU-native frequency and presence penalties.
///
/// Frequency penalty: logit -= frequency_penalty * count
/// Presence penalty: logit -= presence_penalty (if token appeared at all)
///
/// Uses scatter operations to stay on GPU.
fn apply_frequency_presence_penalty(
    logits: &Array,
    token_counts: &HashMap<u32, usize>,
    frequency_penalty: f32,
    presence_penalty: f32,
) -> Result<Array, Exception> {
    if token_counts.is_empty() {
        return Ok(logits.clone());
    }

    // Ensure logits is 2D [1, vocab_size] for indexing
    let vocab_size = logits.dim(-1) as usize;
    let logits_2d = if logits.ndim() == 1 {
        logits.reshape(&[1, vocab_size as i32])?
    } else {
        logits.clone()
    };

    // Collect token indices and their penalty values
    let (indices, penalties): (Vec<i32>, Vec<f32>) = token_counts
        .iter()
        .map(|(&token, &count)| {
            let penalty = frequency_penalty * count as f32 + presence_penalty;
            (token as i32, penalty)
        })
        .unzip();

    if indices.is_empty() {
        return Ok(logits.clone());
    }

    let token_indices = Array::from_slice(&indices, &[1, indices.len() as i32]);
    let penalty_values = Array::from_slice(&penalties, &[1, penalties.len() as i32]);

    // Get current logits at token positions
    let selected_logits = take_along_axis(&logits_2d, &token_indices, -1)?;

    // Apply penalties: logit -= penalty
    let penalized = selected_logits.subtract(&penalty_values)?;

    // Put the penalized values back
    let result = put_along_axis(&logits_2d, &token_indices, &penalized, -1)?;

    // Squeeze back to original shape if needed
    if logits.ndim() == 1 {
        result.squeeze()
    } else {
        Ok(result)
    }
}

/// GPU-native categorical sampling from log probabilities.
///
/// Matches mlx_lm exactly: categorical(logprobs * (1/temp))
/// Uses mlx_rs::random::categorical for efficient GPU sampling.
/// This is ~10x faster than CPU sampling for large vocabularies.
/// Note: item() internally calls eval(), so no explicit eval() needed.
fn gpu_categorical_sample(log_probs: &Array, temperature: f32) -> Result<u32, Exception> {
    // Exactly like mlx_lm: multiply log_probs by inverse temperature
    // categorical_sampling(logits, temp) -> mx.random.categorical(logits * (1 / temp))
    let scaled = if temperature != 1.0 && temperature > 0.0 {
        let inv_temp = Array::from_f32(1.0 / temperature);
        log_probs.multiply(&inv_temp)?
    } else {
        log_probs.clone()
    };

    // Sample using GPU-native categorical (axis=-1 by default)
    let sampled = categorical(&scaled, None, None, None)?;

    // Extract the scalar token ID (item() calls eval() internally)
    Ok(sampled.item::<u32>())
}

/// GPU-native categorical sampling returning Array for async pipelining.
///
/// No GPU→CPU sync - stays lazy for maximum performance.
/// Matches mlx_lm: sampler returns mx.array, not scalar.
fn gpu_categorical_sample_array(log_probs: &Array, temperature: f32) -> Result<Array, Exception> {
    let scaled = if temperature != 1.0 && temperature > 0.0 {
        let inv_temp = Array::from_f32(1.0 / temperature);
        log_probs.multiply(&inv_temp)?
    } else {
        log_probs.clone()
    };

    // Returns Array - no .item() call, stays on GPU
    categorical(&scaled, None, None, None)
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
    let mut sampler = Sampler::new(config.clone());
    let prompt_len = input_ids.len();

    for _ in 0..config.max_new_tokens {
        // Create input array
        let input = Array::from_slice(
            &all_tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
            &[1, all_tokens.len() as i32],
        );

        // Get logits from model
        let logits = forward_fn(&input)?;
        logits.eval()?;

        // Extract logits for the last position [vocab_size]
        let last_idx = logits.dim(1) - 1;
        let last_logits = logits.index((.., last_idx, ..));
        let last_logits = last_logits.squeeze()?;

        // Sample next token
        let next_token = sampler.sample(&last_logits, &all_tokens)?;

        // Check for stop token
        if sampler.is_stop_token(next_token) {
            let num_generated = all_tokens.len() - prompt_len;
            return Ok(GenerationOutput {
                token_ids: all_tokens,
                num_generated,
                stopped_by_token: true,
                stopped_by_length: false,
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
    let mut sampler = Sampler::new(config.clone());
    let prompt_len = input_ids.len();

    // Prefill: Process the entire prompt at once
    let prompt_input = Array::from_slice(
        &input_ids.iter().map(|&t| t as i32).collect::<Vec<_>>(),
        &[1, prompt_len as i32],
    );
    let logits = forward_fn(&prompt_input, cache)?;
    logits.eval()?;

    // Get logits for the last position and sample first new token
    let last_idx = logits.dim(1) - 1;
    let last_logits = logits.index((.., last_idx, ..));
    let last_logits = last_logits.squeeze()?;

    let mut next_token = sampler.sample(&last_logits, &all_tokens)?;

    // Check for stop token
    if sampler.is_stop_token(next_token) {
        let num_generated = 0;
        return Ok(GenerationOutput {
            token_ids: all_tokens,
            num_generated,
            stopped_by_token: true,
            stopped_by_length: false,
        });
    }

    all_tokens.push(next_token);

    // Decode: Process one token at a time using cache
    for _ in 1..config.max_new_tokens {
        // Create input for just the new token
        let token_input = Array::from_slice(&[next_token as i32], &[1, 1]);

        // Forward with cache - only processes the new token
        let logits = forward_fn(&token_input, cache)?;
        logits.eval()?;

        // Get logits for the (only) position
        let last_logits = logits.index((.., 0, ..));
        let last_logits = last_logits.squeeze()?;

        // Sample next token
        next_token = sampler.sample(&last_logits, &all_tokens)?;

        // Check for stop token
        if sampler.is_stop_token(next_token) {
            let num_generated = all_tokens.len() - prompt_len;
            return Ok(GenerationOutput {
                token_ids: all_tokens,
                num_generated,
                stopped_by_token: true,
                stopped_by_length: false,
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
    })
}

/// SOTA generation with async pipelining matching mlx_lm exactly.
///
/// This function uses true async evaluation pipelining where:
/// - Token N's logits are computed while token N-1 is being extracted
/// - Tokens stay as Arrays on GPU until after next computation is scheduled
/// - This eliminates GPU→CPU sync bottleneck that was limiting performance
///
/// The mlx_lm pattern:
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
/// NOTE: Penalties are disabled in async mode to avoid token history tracking
/// which would require sync. Use `generate_cached()` if penalties are needed.
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

    // Create stream once (like Python's module-level generation_stream)
    // CRITICAL: Stream context must wrap ONLY forward passes, NOT async_eval!
    // This is what enables true async pipelining.
    let generation_stream = create_generation_stream();

    let mut all_tokens: Vec<u32> = input_ids.to_vec();
    let sampler = Sampler::new(config.clone());
    let prompt_len = input_ids.len();

    // Helper to get last logits from output - keeps 2D [1, vocab] format
    // Avoiding squeeze reduces reshape operations in sampling pipeline
    let extract_logits = |logits: &Array| -> Array {
        let last_idx = logits.dim(1) - 1;
        logits.index((.., last_idx, ..)) // Returns [1, vocab]
    };

    // Prefill: Process the entire prompt at once (INSIDE stream context)
    let prompt_input = Array::from_slice(
        &input_ids.iter().map(|&t| t as i32).collect::<Vec<_>>(),
        &[1, prompt_len as i32],
    );

    let (mut current_y, mut current_logprobs) = {
        let _stream_ctx = StreamContext::new(&generation_stream);
        let logits = forward_fn(&prompt_input, cache)?;
        let current_logits = extract_logits(&logits);
        sampler.sample_array(&current_logits)?
    };

    // async_eval OUTSIDE stream context (critical for pipelining!)
    async_eval([&current_y, &current_logprobs])?;

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
                let _stream_ctx = StreamContext::new(&generation_stream);
                // Convert Uint32 token to Int32 for model input (argmax returns Uint32)
                let next_input = current_y
                    .as_dtype(mlx_rs::Dtype::Int32)?
                    .reshape(&[1, -1])?;
                let next_output = forward_fn(&next_input, cache)?;
                let next_logits = next_output.index((.., 0, ..));
                sampler.sample_array(&next_logits)?
            };
            // async_eval OUTSIDE stream context (enables pipelining)
            async_eval([&y, &lp])?;
            Some((y, lp))
        } else {
            None
        };

        // 2. First iteration: force eval on first token (like Python's mx.eval(y) on n==0)
        if n == 0 {
            current_y.eval()?;
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
            });
        }

        all_tokens.push(token);

        // Clear cache every 256 tokens (matches Python)
        if n % 256 == 0 && n > 0 {
            mlx_rs::transforms::compile::clear_cache();
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
    })
}

/// High-performance generation using fused Metal sampling kernel.
///
/// This function uses a custom Metal kernel that fuses all sampling operations
/// into a single GPU kernel launch, providing significant speedups especially
/// on battery power where CPU throttling impacts the standard mlx-rs path.
///
/// # Performance Benefits
///
/// - **Single kernel launch** vs 10+ separate launches with mlx-rs
/// - **Minimal CPU overhead** - critical for battery mode
/// - **Zero-copy** from MLX arrays via unified memory
///
/// # Features
///
/// - **Repetition penalty** - Discourages repeating tokens from prompt/output
/// - **Frequency penalty** - Proportional to token appearance count
/// - **Presence penalty** - Flat penalty for any repeated token
///
/// # Arguments
/// * `forward_fn` - Function that takes (input_ids, cache) and returns logits
/// * `input_ids` - Initial token IDs (prompt)
/// * `config` - Generation configuration
/// * `cache` - Pre-allocated KV cache
///
/// # Returns
/// Generation output containing all tokens and metadata
#[cfg(target_os = "macos")]
pub fn generate_cached_metal<F>(
    mut forward_fn: F,
    input_ids: &[u32],
    config: GenerationConfig,
    cache: &mut KVCache,
) -> Result<GenerationOutput, Exception>
where
    F: FnMut(&Array, &mut KVCache) -> Result<Array, Exception>,
{
    use crate::sampling::MetalSampler;
    use std::collections::HashMap;

    // Set wired memory limit for optimal GPU memory management
    let _wired_guard = WiredLimitGuard::new();

    // Create stream once (like Python's module-level generation_stream)
    // CRITICAL: Stream context must wrap ONLY forward passes, NOT async operations!
    let generation_stream = create_generation_stream();

    let mut all_tokens: Vec<u32> = input_ids.to_vec();
    let prompt_len = input_ids.len();

    // Track token counts for frequency/presence penalties
    let mut token_counts: HashMap<u32, usize> = HashMap::new();

    // Get vocab size from first forward pass (INSIDE stream context)
    let prompt_input = Array::from_slice(
        &input_ids.iter().map(|&t| t as i32).collect::<Vec<_>>(),
        &[1, prompt_len as i32],
    );

    let logits = {
        let _stream_ctx = StreamContext::new(&generation_stream);
        forward_fn(&prompt_input, cache)?
    };
    let vocab_size = logits.dim(-1) as usize;

    // Create Metal sampler with optional seed for reproducibility
    let mut metal_sampler = if let Some(seed) = config.seed {
        MetalSampler::with_seed(vocab_size, seed)
            .map_err(|e| Exception::from(e.to_string().as_str()))?
    } else {
        MetalSampler::new(vocab_size).map_err(|e| Exception::from(e.to_string().as_str()))?
    };

    // Check if penalties are enabled
    let use_repetition = config.repetition_penalty != 1.0;
    let use_freq_presence = config.frequency_penalty != 0.0 || config.presence_penalty != 0.0;

    // Helper to get last logits from output
    let extract_logits = |logits: &Array| -> Result<Array, Exception> {
        let last_idx = logits.dim(1) - 1;
        let last_logits = logits.index((.., last_idx, ..));
        let squeezed = last_logits.squeeze()?;
        // Ensure f32 for Metal kernel
        ensure_f32(&squeezed)
    };

    // Helper to apply penalties to logits
    let apply_penalties = |logits: &Array,
                           generated: &[u32],
                           counts: &HashMap<u32, usize>,
                           config: &GenerationConfig|
     -> Result<Array, Exception> {
        let mut penalized = logits.clone();

        // Apply repetition penalty
        if config.repetition_penalty != 1.0 && !generated.is_empty() {
            penalized = apply_repetition_penalty(&penalized, generated, config.repetition_penalty)?;
        }

        // Apply frequency and presence penalties
        if (config.frequency_penalty != 0.0 || config.presence_penalty != 0.0) && !counts.is_empty()
        {
            penalized = apply_frequency_presence_penalty(
                &penalized,
                counts,
                config.frequency_penalty,
                config.presence_penalty,
            )?;
        }

        Ok(penalized)
    };

    // Sample first token
    let current_logits = extract_logits(&logits)?;
    let penalized_logits = if use_repetition || use_freq_presence {
        apply_penalties(&current_logits, &[], &token_counts, &config)?
    } else {
        current_logits
    };
    penalized_logits.eval()?; // Ensure data is available

    let temperature = if config.do_sample {
        config.temperature
    } else {
        0.0
    };

    // Dispatch first sampling (OUTSIDE stream context for async pipelining)
    metal_sampler
        .sample_async(
            &penalized_logits,
            temperature,
            config.top_k as i32,
            config.top_p,
            config.min_p,
        )
        .map_err(|e| Exception::from(e.to_string().as_str()))?;

    // Decode loop
    for n in 0..config.max_new_tokens {
        // Get sampled token (waits for kernel if still running)
        let token = metal_sampler
            .get_token()
            .map_err(|e| Exception::from(e.to_string().as_str()))?;

        // Check stop token
        if config.stop_tokens.contains(&token) {
            let num_generated = all_tokens.len() - prompt_len;
            return Ok(GenerationOutput {
                token_ids: all_tokens,
                num_generated,
                stopped_by_token: true,
                stopped_by_length: false,
            });
        }

        all_tokens.push(token);

        // Update token counts for frequency/presence penalties
        if use_freq_presence {
            *token_counts.entry(token).or_insert(0) += 1;
        }

        // Check if we're done
        if n + 1 >= config.max_new_tokens {
            break;
        }

        // Forward pass for next token (INSIDE stream context)
        let token_input = Array::from_slice(&[token as i32], &[1, 1]);
        let next_output = {
            let _stream_ctx = StreamContext::new(&generation_stream);
            forward_fn(&token_input, cache)?
        };
        let next_logits = extract_logits(&next_output)?;

        // Apply penalties to logits before sampling
        let generated = &all_tokens[prompt_len..]; // Only tokens we generated
        let penalized_logits = if use_repetition || use_freq_presence {
            apply_penalties(&next_logits, generated, &token_counts, &config)?
        } else {
            next_logits
        };
        penalized_logits.eval()?;

        // Dispatch next sampling (OUTSIDE stream context - async pipelining)
        metal_sampler
            .sample_async(
                &penalized_logits,
                temperature,
                config.top_k as i32,
                config.top_p,
                config.min_p,
            )
            .map_err(|e| Exception::from(e.to_string().as_str()))?;

        // Clear cache every 256 tokens (matches Python)
        if n % 256 == 0 && n > 0 {
            mlx_rs::transforms::compile::clear_cache();
        }
    }

    let num_generated = all_tokens.len() - prompt_len;
    Ok(GenerationOutput {
        token_ids: all_tokens,
        num_generated,
        stopped_by_token: false,
        stopped_by_length: true,
    })
}

/// Minimal async generation - exactly matches Python pattern.
///
/// Stream context is INSIDE forward pass, async_eval is OUTSIDE.
pub fn generate_minimal_async<F>(
    mut forward_fn: F,
    input_ids: &[u32],
    config: GenerationConfig,
    cache: &mut KVCache,
) -> Result<GenerationOutput, Exception>
where
    F: FnMut(&Array, &mut KVCache) -> Result<Array, Exception>,
{
    use mlx_rs::ops::indexing::argmax_axis;

    let _wired_guard = WiredLimitGuard::new();

    // Create stream once (like Python's module-level generation_stream)
    let generation_stream = create_generation_stream();

    // Helper to do forward pass INSIDE stream context (like Python's _step)
    let mut step = |input: &Array, cache: &mut KVCache| -> Result<Array, Exception> {
        // Stream context ONLY around forward pass (like Python's `with mx.stream(generation_stream):`)
        let _stream_ctx = StreamContext::new(&generation_stream);
        let out = forward_fn(input, cache)?;
        let logits = out.index((.., -1, ..));
        argmax_axis(&logits, -1, None)
    };

    let mut all_tokens: Vec<u32> = input_ids.to_vec();
    let prompt_len = input_ids.len();
    let stop_tokens = &config.stop_tokens;

    // Prefill (inside stream context)
    let prompt_input = Array::from_slice(
        &input_ids.iter().map(|&t| t as i32).collect::<Vec<_>>(),
        &[1, prompt_len as i32],
    );
    let mut y = step(&prompt_input, cache)?;

    // async_eval is OUTSIDE stream context (like Python)
    async_eval([&y])?;

    let max_tokens = config.max_new_tokens;
    let mut n = 0;

    loop {
        // Schedule NEXT (if not at max)
        let next_y = if n < max_tokens - 1 {
            // Convert Uint32 token to Int32 for model input (argmax returns Uint32)
            let input = y.as_dtype(mlx_rs::Dtype::Int32)?.reshape(&[1, 1])?;
            // step() wraps forward in stream context
            let next = step(&input, cache)?;
            // async_eval OUTSIDE stream context
            async_eval([&next])?;
            Some(next)
        } else {
            None
        };

        // Wait for first token
        if n == 0 {
            y.eval()?;
        }

        // Check max
        if n >= max_tokens {
            break;
        }

        // Extract current (blocks naturally)
        let token = y.item::<u32>();

        // Check stop
        if stop_tokens.contains(&token) {
            let num_generated = all_tokens.len() - prompt_len;
            return Ok(GenerationOutput {
                token_ids: all_tokens,
                num_generated,
                stopped_by_token: true,
                stopped_by_length: false,
            });
        }

        all_tokens.push(token);
        n += 1;

        // Clear cache periodically
        if n % 256 == 0 {
            mlx_rs::transforms::compile::clear_cache();
        }

        // Swap
        if let Some(next) = next_y {
            y = next;
        }
    }

    let num_generated = all_tokens.len() - prompt_len;
    Ok(GenerationOutput {
        token_ids: all_tokens,
        num_generated,
        stopped_by_token: false,
        stopped_by_length: true,
    })
}

/// High-performance generation using JIT-compiled sampling.
///
/// This function uses MLX's JIT compilation (like Python's `@mx.compile`)
/// to fuse sampling operations into optimized Metal kernels.
///
/// # Performance Benefits
///
/// - **JIT-compiled sampling** - operations fused into single kernel like mlx_lm
/// - **Async pipelining** - next forward pass scheduled while extracting current token
/// - **Matches mlx_lm approach** - uses the same compilation strategy
///
/// # Limitations
///
/// - Penalties (repetition, frequency, presence) are NOT applied
/// - Use `generate_cached()` if penalties are needed
///
/// # Arguments
/// * `forward_fn` - Function that takes (input_ids, cache) and returns logits
/// * `input_ids` - Initial token IDs (prompt)
/// * `config` - Generation configuration
/// * `cache` - Pre-allocated KV cache
///
/// # Returns
/// Generation output containing all tokens and metadata
pub fn generate_cached_compiled<F>(
    mut forward_fn: F,
    input_ids: &[u32],
    config: GenerationConfig,
    cache: &mut KVCache,
) -> Result<GenerationOutput, Exception>
where
    F: FnMut(&Array, &mut KVCache) -> Result<Array, Exception>,
{
    use crate::sampling::CompiledSampler;

    // Set wired memory limit for optimal GPU memory management
    // This prevents page faults and matches Python's wired_limit context manager
    let _wired_guard = WiredLimitGuard::new();

    // Create stream once (like Python's module-level generation_stream)
    // CRITICAL: Stream context must wrap ONLY forward passes, NOT async_eval!
    let generation_stream = create_generation_stream();

    let mut all_tokens: Vec<u32> = input_ids.to_vec();
    let prompt_len = input_ids.len();

    // Create compiled sampler with config parameters
    let mut compiled_sampler =
        CompiledSampler::new(config.temperature, config.top_k, config.top_p, config.min_p)?;

    // Create a standard sampler for stop token checking
    let sampler = Sampler::new(config.clone());

    // Prefill: Process the entire prompt at once (INSIDE stream context)
    let prompt_input = Array::from_slice(
        &input_ids.iter().map(|&t| t as i32).collect::<Vec<_>>(),
        &[1, prompt_len as i32],
    );

    let mut current_y = {
        let _stream_ctx = StreamContext::new(&generation_stream);
        let logits = forward_fn(&prompt_input, cache)?;
        let last_idx = logits.dim(1) - 1;
        let current_logits = logits
            .index((.., last_idx..last_idx + 1, ..))
            .squeeze_axes(&[1])?;
        compiled_sampler.sample(&current_logits)?
    };

    // async_eval OUTSIDE stream context (enables pipelining!)
    async_eval([&current_y])?;

    // Decode loop with async pipelining
    let mut n = 0;
    loop {
        // 1. Schedule NEXT forward pass (stream context inside, async_eval outside)
        let next_y = if n + 1 < config.max_new_tokens {
            let y = {
                let _stream_ctx = StreamContext::new(&generation_stream);
                // Convert Uint32 token to Int32 for model input (argmax returns Uint32)
                let next_input = current_y.as_dtype(mlx_rs::Dtype::Int32)?.reshape(&[1, 1])?;
                let next_output = forward_fn(&next_input, cache)?;
                let next_logits = next_output.index((.., 0..1, ..)).squeeze_axes(&[1])?;
                compiled_sampler.sample(&next_logits)?
            };
            // async_eval OUTSIDE stream context
            async_eval([&y])?;
            Some(y)
        } else {
            None
        };

        // 2. For first token only, sync to ensure prompt processing completes
        if n == 0 {
            current_y.eval()?;
        }

        // 3. Extract current token - blocks naturally while GPU computes next
        let token = current_y.item::<u32>();

        // 4. Check stop token
        if sampler.is_stop_token(token) {
            let num_generated = all_tokens.len() - prompt_len;
            return Ok(GenerationOutput {
                token_ids: all_tokens,
                num_generated,
                stopped_by_token: true,
                stopped_by_length: false,
            });
        }

        all_tokens.push(token);
        n += 1;

        // Clear cache every 256 tokens (matches Python mlx-lm)
        if n % 256 == 0 {
            mlx_rs::transforms::compile::clear_cache();
        }

        if n >= config.max_new_tokens {
            break;
        }

        current_y = next_y.expect("next_y should exist when not at max tokens");
    }

    let num_generated = all_tokens.len() - prompt_len;
    Ok(GenerationOutput {
        token_ids: all_tokens,
        num_generated,
        stopped_by_token: false,
        stopped_by_length: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    #[test]
    fn test_generation_config_greedy() {
        let config = GenerationConfig::greedy(100);
        assert_eq!(config.max_new_tokens, 100);
        assert!(!config.do_sample);
    }

    #[test]
    fn test_greedy_sample() {
        // Create logits where token 5 has highest value
        let mut logits_vec = vec![-10.0f32; 100];
        logits_vec[5] = 10.0;
        let logits = Array::from_slice(&logits_vec, &[100]);

        let token = greedy_sample(&logits).unwrap();
        assert_eq!(token, 5);
    }

    #[test]
    fn test_top_k_filter() {
        // Create logits where first 3 tokens have high values
        let mut logits_vec = vec![-100.0f32; 10];
        logits_vec[0] = 10.0;
        logits_vec[1] = 9.0;
        logits_vec[2] = 8.0;
        let logits = Array::from_slice(&logits_vec, &[10]);

        let filtered = top_k_filter(&logits, 3).unwrap();
        filtered.eval().unwrap();

        // Get filtered values
        let mut filtered_vec: Vec<f32> = Vec::new();
        for i in 0..10 {
            let val = filtered.index(i);
            val.eval().unwrap();
            filtered_vec.push(val.item::<f32>());
        }

        // Check that tokens 0, 1, 2 are kept
        assert!(filtered_vec[0] > f32::NEG_INFINITY);
        assert!(filtered_vec[1] > f32::NEG_INFINITY);
        assert!(filtered_vec[2] > f32::NEG_INFINITY);

        // Check that other tokens are filtered
        for val in filtered_vec.iter().skip(3).take(7) {
            assert!(val.is_infinite());
        }
    }

    #[test]
    fn test_sampler_greedy() {
        let config = GenerationConfig::greedy(10);
        let mut sampler = Sampler::new(config);

        let mut logits_vec = vec![-10.0f32; 100];
        logits_vec[42] = 10.0;
        let logits = Array::from_slice(&logits_vec, &[100]);

        let token = sampler.sample(&logits, &[]).unwrap();
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
    fn test_top_p_filter() {
        // Create logits with uneven distribution
        let logits = Array::from_slice(&[10.0f32, 5.0, 1.0, 0.0, -10.0], &[5]);

        let filtered = top_p_filter(&logits, 0.9).unwrap();
        filtered.eval().unwrap();

        // The first token has very high probability, should be kept
        let val0 = filtered.index(0);
        val0.eval().unwrap();
        assert!(val0.item::<f32>() > f32::NEG_INFINITY);
    }

    #[test]
    fn test_categorical_sample() {
        // Create log probabilities where one token is dominant (log-softmax'd)
        // Token 0 has log-prob ~0, others are -inf
        let log_probs = Array::from_slice(&[0.0f32, -100.0, -100.0, -100.0, -100.0], &[5]);

        // Seed mlx random state for reproducibility
        let _ = mlx_seed(42);

        // Temperature = 1.0 (no scaling)
        let token = gpu_categorical_sample(&log_probs, 1.0).unwrap();

        // Should almost always sample token 0 due to dominant probability
        assert_eq!(token, 0);
    }

    #[test]
    fn test_min_p_filter() {
        // Create logits with descending probabilities
        // After softmax: ~0.88, ~0.12, ~0.004, ... (very small for rest)
        let logits = Array::from_slice(&[10.0f32, 8.0, 5.0, 1.0, -5.0], &[5]);

        // With min_p = 0.1, threshold = 0.1 * 0.88 = 0.088
        // Should keep tokens with prob >= 0.088 (token 0 and 1)
        let filtered = min_p_filter(&logits, 0.1).unwrap();
        filtered.eval().unwrap();

        // Token 0 and 1 should be kept
        let val0 = filtered.index(0);
        val0.eval().unwrap();
        assert!(val0.item::<f32>() > f32::NEG_INFINITY);

        let val1 = filtered.index(1);
        val1.eval().unwrap();
        assert!(val1.item::<f32>() > f32::NEG_INFINITY);
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

    #[test]
    fn test_frequency_presence_penalty() {
        let logits = Array::from_slice(&[5.0f32, 5.0, 5.0, 5.0, 5.0], &[5]);

        // Token 0 appeared once, token 2 appeared 3 times
        let mut counts = HashMap::new();
        counts.insert(0, 1);
        counts.insert(2, 3);

        // Frequency penalty = 1.0, presence penalty = 0.5
        let filtered = apply_frequency_presence_penalty(&logits, &counts, 1.0, 0.5).unwrap();
        filtered.eval().unwrap();

        let logits_vec: Vec<f32> = filtered.as_slice::<f32>().to_vec();

        // Token 0: 5.0 - 1.0*1 - 0.5 = 3.5
        assert!((logits_vec[0] - 3.5).abs() < 0.01);

        // Token 1: 5.0 (no penalty)
        assert!((logits_vec[1] - 5.0).abs() < 0.01);

        // Token 2: 5.0 - 1.0*3 - 0.5 = 1.5
        assert!((logits_vec[2] - 1.5).abs() < 0.01);

        // Token 3: 5.0 (no penalty)
        assert!((logits_vec[3] - 5.0).abs() < 0.01);
    }
}
