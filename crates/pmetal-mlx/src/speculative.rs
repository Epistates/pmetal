//! Speculative decoding for accelerated inference.
//!
//! Speculative decoding uses a smaller "draft" model to propose multiple tokens,
//! which are then verified by the larger "target" model in parallel. This can
//! provide 2-3x speedups for autoregressive generation.
//!
//! ## How It Works
//!
//! 1. **Draft phase**: Small model generates K tokens autoregressively
//! 2. **Verify phase**: Large model scores all K tokens in one forward pass
//! 3. **Accept/Reject**: Tokens are accepted probabilistically based on ratio
//! 4. **Correction**: First rejected position gets sampled from adjusted distribution
//!
//! ## Theoretical Guarantee
//!
//! Speculative decoding produces the **exact same distribution** as standard
//! sampling from the target model, just faster. This is achieved through
//! rejection sampling with careful probability adjustments.
//!
//! ## When to Use
//!
//! - Draft model should be 4-10x smaller than target
//! - Works best when draft model has high acceptance rate (>70%)
//! - Memory overhead: need to load both models
//! - Most effective for greedy/low-temperature sampling

use mlx_rs::{
    error::Exception,
    ops::indexing::{argmax, IndexOp},
    Array,
};

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens to draft per iteration (K).
    pub num_draft_tokens: usize,
    /// Temperature for sampling (0 = greedy).
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold.
    pub top_p: f32,
    /// Minimum acceptance rate before falling back to standard decoding.
    pub min_acceptance_rate: f32,
    /// Enable adaptive K (adjust num_draft_tokens based on acceptance).
    pub adaptive_k: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_draft_tokens: 5,
            temperature: 0.0, // Greedy by default
            top_p: 1.0,
            min_acceptance_rate: 0.5,
            adaptive_k: true,
        }
    }
}

impl SpeculativeConfig {
    /// Create config with specified draft tokens.
    pub fn new(num_draft_tokens: usize) -> Self {
        Self {
            num_draft_tokens,
            ..Default::default()
        }
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp.max(0.0);
        self
    }

    /// Set top-p sampling.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p.clamp(0.0, 1.0);
        self
    }

    /// Enable/disable adaptive K.
    pub fn with_adaptive_k(mut self, adaptive: bool) -> Self {
        self.adaptive_k = adaptive;
        self
    }
}

/// Result of a speculative decoding step.
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    /// Accepted token IDs.
    pub accepted_tokens: Vec<i32>,
    /// Number of tokens accepted from draft.
    pub num_accepted: usize,
    /// Total tokens generated (accepted + 1 correction if rejected).
    pub num_generated: usize,
    /// Acceptance rate for this iteration.
    pub acceptance_rate: f32,
}

/// Statistics for speculative decoding session.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total tokens generated.
    pub total_tokens: usize,
    /// Total draft tokens proposed.
    pub total_draft_tokens: usize,
    /// Total accepted draft tokens.
    pub total_accepted: usize,
    /// Number of speculative iterations.
    pub num_iterations: usize,
    /// Current adaptive K value.
    pub current_k: usize,
}

impl SpeculativeStats {
    /// Get overall acceptance rate.
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_draft_tokens == 0 {
            0.0
        } else {
            self.total_accepted as f32 / self.total_draft_tokens as f32
        }
    }

    /// Get tokens per iteration (effective batch size).
    pub fn tokens_per_iteration(&self) -> f32 {
        if self.num_iterations == 0 {
            0.0
        } else {
            self.total_tokens as f32 / self.num_iterations as f32
        }
    }

    /// Estimate speedup over standard decoding.
    pub fn estimated_speedup(&self) -> f32 {
        // Speedup = tokens_per_iter / (1 + draft_overhead)
        // Assuming draft is ~5x faster, overhead is ~0.2 per draft token
        let tpi = self.tokens_per_iteration();
        let k = if self.num_iterations > 0 {
            self.total_draft_tokens as f32 / self.num_iterations as f32
        } else {
            1.0
        };
        let overhead = 1.0 + k * 0.2; // Rough estimate
        tpi / overhead
    }
}

/// Speculative decoder for accelerated inference.
pub struct SpeculativeDecoder {
    config: SpeculativeConfig,
    stats: SpeculativeStats,
}

impl SpeculativeDecoder {
    /// Create a new speculative decoder.
    pub fn new(config: SpeculativeConfig) -> Self {
        let current_k = config.num_draft_tokens;
        Self {
            config,
            stats: SpeculativeStats {
                current_k,
                ..Default::default()
            },
        }
    }

    /// Get current configuration.
    pub fn config(&self) -> &SpeculativeConfig {
        &self.config
    }

    /// Get accumulated statistics.
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats {
            current_k: self.config.num_draft_tokens,
            ..Default::default()
        };
    }

    /// Get current K (may be adaptive).
    pub fn current_k(&self) -> usize {
        if self.config.adaptive_k {
            self.stats.current_k
        } else {
            self.config.num_draft_tokens
        }
    }

    /// Verify draft tokens against target model probabilities.
    ///
    /// # Arguments
    /// * `draft_tokens` - Tokens proposed by draft model
    /// * `draft_probs` - Probabilities from draft model for each token
    /// * `target_probs` - Probabilities from target model for each token
    ///
    /// # Returns
    /// SpeculativeResult with accepted tokens and statistics
    pub fn verify(
        &mut self,
        draft_tokens: &[i32],
        draft_probs: &[f32],
        target_probs: &[f32],
    ) -> Result<SpeculativeResult, Exception> {
        let k = draft_tokens.len();
        let mut accepted_tokens = Vec::new();
        let mut num_accepted = 0;

        // Rejection sampling for each position
        for i in 0..k {
            let p_draft = draft_probs[i];
            let p_target = target_probs[i];

            // Accept with probability min(1, p_target / p_draft)
            let accept_prob = if p_draft > 0.0 {
                (p_target / p_draft).min(1.0)
            } else {
                0.0
            };

            // Sample acceptance
            let r = rand_uniform();
            if r < accept_prob {
                accepted_tokens.push(draft_tokens[i]);
                num_accepted += 1;
            } else {
                // Reject - need to sample from adjusted distribution
                break;
            }
        }

        let acceptance_rate = if k > 0 {
            num_accepted as f32 / k as f32
        } else {
            0.0
        };

        // Update statistics
        self.stats.total_draft_tokens += k;
        self.stats.total_accepted += num_accepted;
        self.stats.total_tokens += accepted_tokens.len();
        self.stats.num_iterations += 1;

        // Adaptive K adjustment
        if self.config.adaptive_k {
            self.adjust_k(acceptance_rate);
        }

        Ok(SpeculativeResult {
            accepted_tokens,
            num_accepted,
            num_generated: num_accepted, // Will add correction token separately
            acceptance_rate,
        })
    }

    /// Verify with full probability distributions (MLX arrays).
    ///
    /// # Arguments
    /// * `draft_tokens` - Token IDs from draft model [K]
    /// * `draft_logits` - Logits from draft model [K, vocab_size]
    /// * `target_logits` - Logits from target model [K, vocab_size]
    ///
    /// # Returns
    /// (accepted_tokens, correction_logits) where correction_logits can be
    /// used to sample the corrected token if needed.
    pub fn verify_logits(
        &mut self,
        draft_tokens: &Array,
        draft_logits: &Array,
        target_logits: &Array,
    ) -> Result<(Vec<i32>, Option<Array>), Exception> {
        let k = draft_tokens.dim(0) as usize;

        // Apply temperature
        let temp = self.config.temperature.max(1e-6);
        let draft_scaled = draft_logits.divide(Array::from_f32(temp))?;
        let target_scaled = target_logits.divide(Array::from_f32(temp))?;

        // Compute probabilities
        let draft_probs = mlx_rs::ops::softmax_axis(&draft_scaled, -1, None)?;
        let target_probs = mlx_rs::ops::softmax_axis(&target_scaled, -1, None)?;

        // Evaluate for CPU access
        draft_tokens.eval()?;
        draft_probs.eval()?;
        target_probs.eval()?;

        let tokens: Vec<i32> = draft_tokens.as_slice().to_vec();
        let mut accepted_tokens = Vec::new();
        let mut rejection_idx = None;

        // Rejection sampling
        for i in 0..k {
            let token = tokens[i];

            // Get probability of this token under both models
            let p_draft = draft_probs.index((i as i32, token));
            let p_target = target_probs.index((i as i32, token));
            p_draft.eval()?;
            p_target.eval()?;

            let p_d = p_draft.item::<f32>();
            let p_t = p_target.item::<f32>();

            let accept_prob = if p_d > 1e-10 {
                (p_t / p_d).min(1.0)
            } else {
                0.0
            };

            let r = rand_uniform();
            if r < accept_prob {
                accepted_tokens.push(token);
            } else {
                rejection_idx = Some(i);
                break;
            }
        }

        // Compute correction distribution if rejected
        let correction_logits = if let Some(idx) = rejection_idx {
            // Correction distribution: max(0, p_target - p_draft) normalized
            let p_draft_row = draft_probs.index(idx as i32);
            let p_target_row = target_probs.index(idx as i32);

            let diff = p_target_row.subtract(&p_draft_row)?;
            let zero = Array::zeros::<f32>(&[diff.dim(0)])?;
            let clipped = mlx_rs::ops::maximum(&diff, &zero)?;

            // Convert back to logits (log of normalized diff)
            let sum = clipped.sum(None)?;
            let normalized = clipped.divide(&sum)?;
            let eps = Array::from_f32(1e-10);
            let safe_probs = normalized.add(&eps)?;
            Some(safe_probs.log()?)
        } else {
            None
        };

        // Update stats
        self.stats.total_draft_tokens += k;
        self.stats.total_accepted += accepted_tokens.len();
        self.stats.total_tokens += accepted_tokens.len();
        self.stats.num_iterations += 1;

        let acceptance_rate = if k > 0 {
            accepted_tokens.len() as f32 / k as f32
        } else {
            0.0
        };

        if self.config.adaptive_k {
            self.adjust_k(acceptance_rate);
        }

        Ok((accepted_tokens, correction_logits))
    }

    /// Adjust K based on acceptance rate.
    fn adjust_k(&mut self, acceptance_rate: f32) {
        let current = self.stats.current_k;
        let max_k = self.config.num_draft_tokens * 2;
        let min_k = 1;

        if acceptance_rate > 0.8 && current < max_k {
            // High acceptance - try more tokens
            self.stats.current_k = (current + 1).min(max_k);
        } else if acceptance_rate < 0.4 && current > min_k {
            // Low acceptance - try fewer tokens
            self.stats.current_k = (current - 1).max(min_k);
        }
    }

    /// Sample from a probability distribution.
    pub fn sample(&self, logits: &Array) -> Result<i32, Exception> {
        logits.eval()?;

        if self.config.temperature == 0.0 {
            // Greedy
            let max_idx = argmax(logits, None)?;
            max_idx.eval()?;
            Ok(max_idx.item::<u32>() as i32)
        } else {
            // Temperature sampling
            let scaled = logits.divide(Array::from_f32(self.config.temperature))?;
            let probs = mlx_rs::ops::softmax_axis(&scaled, -1, None)?;

            // Apply top-p if specified
            let probs = if self.config.top_p < 1.0 {
                self.apply_top_p(&probs)?
            } else {
                probs
            };

            // Sample from distribution
            self.categorical_sample(&probs)
        }
    }

    /// Apply nucleus (top-p) sampling.
    fn apply_top_p(&self, probs: &Array) -> Result<Array, Exception> {
        probs.eval()?;
        let probs_vec: Vec<f32> = probs.as_slice().to_vec();

        // Sort indices by probability (descending)
        let mut indexed: Vec<(usize, f32)> = probs_vec.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Find cutoff
        let mut cumsum = 0.0;
        let mut cutoff_idx = indexed.len();
        for (i, (_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum > self.config.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out probabilities below cutoff
        let mut filtered = vec![0.0f32; probs_vec.len()];
        for (orig_idx, _) in indexed.iter().take(cutoff_idx) {
            filtered[*orig_idx] = probs_vec[*orig_idx];
        }

        // Renormalize
        let sum: f32 = filtered.iter().sum();
        if sum > 0.0 {
            for p in &mut filtered {
                *p /= sum;
            }
        }

        Ok(Array::from_slice(&filtered, probs.shape()))
    }

    /// Sample from categorical distribution.
    fn categorical_sample(&self, probs: &Array) -> Result<i32, Exception> {
        probs.eval()?;
        let probs_vec: Vec<f32> = probs.as_slice().to_vec();

        let r = rand_uniform();
        let mut cumsum = 0.0;
        for (i, &p) in probs_vec.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return Ok(i as i32);
            }
        }

        // Fallback to last index
        Ok((probs_vec.len() - 1) as i32)
    }
}

/// Generate a uniform random number in [0, 1).
///
/// Uses a thread-local xorshift64 PRNG seeded from system time on first use.
/// This avoids the correlation problem of re-seeding from nanosecond time
/// on every call.
fn rand_uniform() -> f32 {
    use std::cell::Cell;
    use std::time::{SystemTime, UNIX_EPOCH};

    thread_local! {
        static STATE: Cell<u64> = Cell::new({
            let seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            // Ensure non-zero initial state (xorshift requirement)
            if seed == 0 { 0xdeadbeefcafe1234 } else { seed }
        });
    }

    STATE.with(|s| {
        let mut x = s.get();
        // xorshift64
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        // Map to [0, 1) — use upper 23 bits for f32 mantissa precision
        (x >> 40) as f32 / (1u64 << 24) as f32
    })
}

/// Estimate optimal K based on model size ratio.
///
/// # Arguments
/// * `draft_params` - Number of parameters in draft model
/// * `target_params` - Number of parameters in target model
///
/// # Returns
/// Recommended K value.
pub fn estimate_optimal_k(draft_params: u64, target_params: u64) -> usize {
    // Rule of thumb: K ≈ sqrt(target_params / draft_params)
    let ratio = (target_params as f64) / (draft_params as f64);
    let k = ratio.sqrt().round() as usize;
    k.clamp(2, 10)
}

/// Check if speculative decoding is beneficial.
///
/// # Arguments
/// * `draft_latency_ms` - Latency for draft model forward pass
/// * `target_latency_ms` - Latency for target model forward pass
/// * `expected_acceptance` - Expected acceptance rate (0-1)
/// * `k` - Number of draft tokens
///
/// # Returns
/// (is_beneficial, estimated_speedup)
pub fn is_speculative_beneficial(
    draft_latency_ms: f32,
    target_latency_ms: f32,
    expected_acceptance: f32,
    k: usize,
) -> (bool, f32) {
    // Standard decoding: 1 target forward per token
    let standard_latency = target_latency_ms;

    // Speculative: K draft + 1 target, generates (1 + K * acceptance) tokens on average
    let speculative_latency = (k as f32) * draft_latency_ms + target_latency_ms;
    let expected_tokens = 1.0 + (k as f32) * expected_acceptance;
    let speculative_per_token = speculative_latency / expected_tokens;

    let speedup = standard_latency / speculative_per_token;
    (speedup > 1.0, speedup)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_config_default() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.num_draft_tokens, 5);
        assert_eq!(config.temperature, 0.0);
        assert!(config.adaptive_k);
    }

    #[test]
    fn test_speculative_config_builder() {
        let config = SpeculativeConfig::new(8)
            .with_temperature(0.7)
            .with_top_p(0.9)
            .with_adaptive_k(false);

        assert_eq!(config.num_draft_tokens, 8);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_p, 0.9);
        assert!(!config.adaptive_k);
    }

    #[test]
    fn test_verify_basic() {
        let config = SpeculativeConfig::new(3);
        let mut decoder = SpeculativeDecoder::new(config);

        // High acceptance scenario: draft probs similar to target
        let draft_tokens = vec![1, 2, 3];
        let draft_probs = vec![0.8, 0.7, 0.9];
        let target_probs = vec![0.9, 0.8, 0.85]; // Similar to draft

        let result = decoder
            .verify(&draft_tokens, &draft_probs, &target_probs)
            .unwrap();

        // With similar probs, most should be accepted
        assert!(result.num_accepted >= 2);
    }

    #[test]
    fn test_verify_rejection() {
        let config = SpeculativeConfig::new(3);
        let mut decoder = SpeculativeDecoder::new(config);

        // Low acceptance scenario: target prefers different tokens
        let draft_tokens = vec![1, 2, 3];
        let draft_probs = vec![0.9, 0.9, 0.9]; // Draft is confident
        let target_probs = vec![0.1, 0.1, 0.1]; // Target disagrees

        let result = decoder
            .verify(&draft_tokens, &draft_probs, &target_probs)
            .unwrap();

        // Should reject most/all
        assert!(result.acceptance_rate < 0.5);
    }

    #[test]
    fn test_stats_tracking() {
        let config = SpeculativeConfig::new(4);
        let mut decoder = SpeculativeDecoder::new(config);

        // Run a few iterations
        let tokens = vec![1, 2, 3, 4];
        let probs = vec![0.8, 0.8, 0.8, 0.8];

        for _ in 0..3 {
            decoder.verify(&tokens, &probs, &probs).unwrap();
        }

        assert_eq!(decoder.stats().num_iterations, 3);
        assert_eq!(decoder.stats().total_draft_tokens, 12); // 4 * 3
    }

    #[test]
    fn test_adaptive_k() {
        let config = SpeculativeConfig::new(5).with_adaptive_k(true);
        let mut decoder = SpeculativeDecoder::new(config);

        // High acceptance - should increase K
        let tokens = vec![1, 2, 3, 4, 5];
        let probs = vec![0.9; 5];

        for _ in 0..5 {
            decoder.verify(&tokens, &probs, &probs).unwrap();
        }

        // K should have increased
        assert!(decoder.current_k() >= 5);
    }

    #[test]
    fn test_estimate_optimal_k() {
        // 7B target, 350M draft (20x ratio) -> K ≈ sqrt(20) ≈ 4-5
        let k = estimate_optimal_k(350_000_000, 7_000_000_000);
        assert!(k >= 3 && k <= 6);

        // 70B target, 1B draft (70x ratio) -> K ≈ sqrt(70) ≈ 8
        let k = estimate_optimal_k(1_000_000_000, 70_000_000_000);
        assert!(k >= 6 && k <= 10);
    }

    #[test]
    fn test_is_speculative_beneficial() {
        // Scenario: draft 10ms, target 100ms, 70% acceptance, K=5
        let (beneficial, speedup) = is_speculative_beneficial(10.0, 100.0, 0.7, 5);

        // 5 * 10 + 100 = 150ms for ~4.5 tokens = 33ms/token
        // vs 100ms/token for standard
        // Speedup should be ~3x
        assert!(beneficial);
        assert!(speedup > 2.0);
    }

    #[test]
    fn test_greedy_sample() {
        let config = SpeculativeConfig::new(5).with_temperature(0.0);
        let decoder = SpeculativeDecoder::new(config);

        // Logits with clear maximum
        let logits = Array::from_slice(&[-10.0f32, -5.0, 10.0, -3.0, 0.0], &[5]);
        let token = decoder.sample(&logits).unwrap();

        assert_eq!(token, 2); // Index of max
    }

    #[test]
    fn test_stats_acceptance_rate() {
        let mut stats = SpeculativeStats::default();
        stats.total_draft_tokens = 100;
        stats.total_accepted = 75;

        assert!((stats.acceptance_rate() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_stats_tokens_per_iteration() {
        let mut stats = SpeculativeStats::default();
        stats.total_tokens = 30;
        stats.num_iterations = 10;

        assert!((stats.tokens_per_iteration() - 3.0).abs() < 0.01);
    }
}
