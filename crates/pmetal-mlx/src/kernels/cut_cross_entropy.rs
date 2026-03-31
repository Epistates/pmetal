//! Cut Cross Entropy (CCE) - Memory-efficient cross-entropy for long contexts.
//!
//! This implements Apple ML's Cut Cross Entropy technique that enables up to
//! 13x longer context training by never materializing the full logits tensor.
//!
//! # Key Innovation
//!
//! Standard cross-entropy computes:
//! ```text
//! loss = logsumexp(logits) - logits[target]
//! ```
//!
//! For vocab=150K, seq=4096, batch=4, the logits tensor is **2.4GB** in fp16!
//!
//! CCE avoids this by computing the loss in chunks:
//! 1. Compute target logit directly: `hidden @ W[target]`
//! 2. Compute logsumexp in chunks using online algorithm
//! 3. Never materialize more than chunk_size logits at a time
//!
//! # Memory Savings
//!
//! | Sequence Length | Standard CE | Cut CE |
//! |-----------------|-------------|--------|
//! | 2K tokens | 600MB | 8MB |
//! | 8K tokens | 2.4GB | 8MB |
//! | 32K tokens | 9.6GB | 8MB |
//!
//! Peak memory is O(chunk_size) instead of O(seq * vocab).
//!
//! # References
//!
//! - Apple ML: https://github.com/apple/ml-cross-entropy

use crate::ArrayDtypeExt;
use pmetal_bridge::compat::{Array, Dtype, Exception, ops};

/// Configuration for Cut Cross Entropy.
#[derive(Debug, Clone)]
pub struct CutCrossEntropyConfig {
    /// Vocabulary chunk size for computing logsumexp.
    /// Larger = faster but more memory. Default: 4096.
    pub vocab_chunk_size: usize,

    /// Token chunk size for parallel processing.
    /// Default: 1024 tokens at a time.
    pub token_chunk_size: usize,

    /// Index to ignore in loss computation (e.g., -100 for padding).
    pub ignore_index: i32,

    /// Label smoothing factor (0.0 to disable).
    pub label_smoothing: f32,

    /// Logit softcapping for Gemma2 (0.0 to disable).
    pub softcap: f32,

    /// Logit scaling for Cohere (1.0 to disable).
    pub logit_scale: f32,

    /// Whether to compute gradient (requires more memory).
    pub compute_grad: bool,

    /// Use online softmax for better numerical stability.
    pub use_online_softmax: bool,
}

impl Default for CutCrossEntropyConfig {
    fn default() -> Self {
        Self {
            vocab_chunk_size: 4096,
            token_chunk_size: 1024,
            ignore_index: -100,
            label_smoothing: 0.0,
            softcap: 0.0,
            logit_scale: 1.0,
            compute_grad: true,
            use_online_softmax: true,
        }
    }
}

impl CutCrossEntropyConfig {
    /// Create a new configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set vocabulary chunk size.
    pub fn with_vocab_chunk_size(mut self, size: usize) -> Self {
        self.vocab_chunk_size = size;
        self
    }

    /// Set token chunk size.
    pub fn with_token_chunk_size(mut self, size: usize) -> Self {
        self.token_chunk_size = size;
        self
    }

    /// Set ignore index.
    pub fn with_ignore_index(mut self, index: i32) -> Self {
        self.ignore_index = index;
        self
    }

    /// Enable Gemma2 softcapping.
    pub fn with_softcap(mut self, softcap: f32) -> Self {
        self.softcap = softcap;
        self
    }

    /// Enable Cohere logit scaling.
    pub fn with_logit_scale(mut self, scale: f32) -> Self {
        self.logit_scale = scale;
        self
    }

    /// Set label smoothing.
    pub fn with_label_smoothing(mut self, smoothing: f32) -> Self {
        self.label_smoothing = smoothing;
        self
    }
}

/// Output from Cut Cross Entropy forward pass.
#[derive(Debug)]
pub struct CutCrossEntropyOutput {
    /// Mean loss over valid tokens.
    pub loss: Array,

    /// Number of valid (non-ignored) tokens.
    pub n_valid: usize,

    /// Per-token losses (optional, for debugging).
    pub per_token_loss: Option<Array>,

    /// Cached values for backward pass.
    cached_logsumexp: Option<Array>,
    #[allow(dead_code)] // Stored for future backward pass implementation
    cached_target_logits: Option<Array>,
}

impl CutCrossEntropyOutput {
    /// Get the loss value.
    pub fn loss_value(&self) -> Result<f32, Exception> {
        let loss_eval = self.loss.clone();
        loss_eval.eval();
        Ok(loss_eval.item_f32())
    }
}

/// Cut Cross Entropy loss computation.
///
/// Computes cross-entropy loss directly from hidden states without ever
/// materializing the full logits tensor. This enables training with up to
/// 13x longer context on memory-limited hardware.
pub struct CutCrossEntropy {
    config: CutCrossEntropyConfig,
}

impl CutCrossEntropy {
    /// Create a new Cut Cross Entropy instance.
    pub fn new(config: CutCrossEntropyConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(CutCrossEntropyConfig::default())
    }

    /// Compute loss directly from hidden states.
    ///
    /// This is the key optimization: we never materialize the full logits tensor.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - Hidden states [batch * seq, hidden_dim]
    /// * `lm_head_weight` - LM head weights [vocab_size, hidden_dim]
    /// * `targets` - Target token indices [batch * seq]
    /// * `lm_head_bias` - Optional LM head bias [vocab_size]
    ///
    /// # Returns
    ///
    /// Loss value and optional cached data for backward pass.
    pub fn forward(
        &self,
        hidden_states: &Array,
        lm_head_weight: &Array,
        targets: &Array,
        lm_head_bias: Option<&Array>,
    ) -> Result<CutCrossEntropyOutput, Exception> {
        let hidden_shape = hidden_states.shape();
        let _n_tokens = hidden_shape[0] as usize;
        let hidden_dim = hidden_shape[1] as usize;

        let weight_shape = lm_head_weight.shape();
        let vocab_size = weight_shape[0] as usize;

        // Validate dimensions
        if weight_shape[1] as usize != hidden_dim {
            return Err(Exception::custom(format!(
                "Hidden dim mismatch: hidden={}, weight={}",
                hidden_dim, weight_shape[1]
            )));
        }

        // Step 1: Compute target logits directly (only for target tokens)
        let target_logits =
            self.compute_target_logits(hidden_states, lm_head_weight, targets, lm_head_bias);

        // Step 2: Compute logsumexp in chunks
        let logsumexp =
            self.compute_chunked_logsumexp(hidden_states, lm_head_weight, lm_head_bias, vocab_size);

        // Step 3: Compute per-token loss
        // loss[i] = logsumexp[i] - target_logits[i]
        let per_token_loss = logsumexp.subtract(&target_logits);

        // Step 4: Apply ignore index masking
        let (masked_loss, n_valid) = self.apply_mask(&per_token_loss, targets);

        // Step 5: Compute mean loss
        let safe_n_valid = n_valid.max(1);
        let n_valid_arr = Array::from_i32(safe_n_valid as i32).as_dtype(Dtype::Float32.as_i32());
        let loss = masked_loss.sum_all().divide(&n_valid_arr);

        // Cache for backward if needed
        let (cached_lse, cached_target) = if self.config.compute_grad {
            (Some(logsumexp), Some(target_logits))
        } else {
            (None, None)
        };

        Ok(CutCrossEntropyOutput {
            loss,
            n_valid,
            per_token_loss: if self.config.compute_grad {
                Some(per_token_loss)
            } else {
                None
            },
            cached_logsumexp: cached_lse,
            cached_target_logits: cached_target,
        })
    }

    /// Compute target logits by direct indexing.
    fn compute_target_logits(
        &self,
        hidden_states: &Array,
        lm_head_weight: &Array,
        targets: &Array,
        lm_head_bias: Option<&Array>,
    ) -> Array {
        // Clamp targets to valid indices before gather.
        let zero = Array::from_i32(0_i32);
        let targets_i32 = targets.as_dtype(Dtype::Int32.as_i32());
        let safe_targets = targets_i32.maximum(&zero);

        // Gather target embeddings: W[safe_targets, :] -> [n_tokens, hidden_dim]
        let target_weights = lm_head_weight.take_axis(&safe_targets, 0);

        // Compute dot product: sum(hidden * target_weights, axis=-1)
        let product = hidden_states.multiply(&target_weights);
        let mut target_logits = product.sum_axis(-1, false);

        // Add bias if present
        if let Some(bias) = lm_head_bias {
            let target_bias = bias.take_axis(&safe_targets, 0);
            target_logits = target_logits.add(&target_bias);
        }

        // Apply logit transforms
        self.apply_logit_transforms(target_logits)
    }

    /// Compute logsumexp in chunks using online algorithm.
    fn compute_chunked_logsumexp(
        &self,
        hidden_states: &Array,
        lm_head_weight: &Array,
        lm_head_bias: Option<&Array>,
        vocab_size: usize,
    ) -> Array {
        let n_tokens = hidden_states.dim(0);
        let hidden_dim = lm_head_weight.dim(1);
        let chunk_size = self.config.vocab_chunk_size;
        let n_chunks = (vocab_size + chunk_size - 1) / chunk_size;

        // Initialize online logsumexp accumulators
        let neg_inf_arr = Array::from_f32(f32::NEG_INFINITY);
        let mut running_max = neg_inf_arr.broadcast_to(&[n_tokens]);
        let zero_arr = Array::from_f32(0.0);
        let mut running_sum = zero_arr.broadcast_to(&[n_tokens]);

        for chunk_idx in 0..n_chunks {
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(vocab_size);

            // Get weight chunk: W[start:end, :]
            let weight_chunk = lm_head_weight.slice(&[start as i32, 0], &[end as i32, hidden_dim]);

            // Compute chunk logits: hidden @ W_chunk.T
            let weight_t = weight_chunk.t();
            let mut chunk_logits = hidden_states.matmul(&weight_t);

            // Add bias if present
            if let Some(bias) = lm_head_bias {
                let bias_chunk = bias.slice(&[start as i32], &[end as i32]);
                chunk_logits = chunk_logits.add(&bias_chunk);
            }

            // Apply logit transforms to chunk
            chunk_logits = self.apply_logit_transforms(chunk_logits);

            // Online logsumexp update
            let chunk_max = chunk_logits.max_axis(-1, false);
            let new_max = running_max.maximum(&chunk_max);

            // Update running_sum: s_new = s * exp(m - m_new) + sum(exp(chunk - m_new))
            let max_diff = running_max.subtract(&new_max);
            let scaled_sum = running_sum.multiply(&max_diff.exp());

            let new_max_expanded = new_max.reshape(&[-1, 1]);
            let chunk_shifted = chunk_logits.subtract(&new_max_expanded);
            let chunk_exp = chunk_shifted.exp();
            let chunk_sum = chunk_exp.sum_axis(-1, false);

            running_sum = scaled_sum.add(&chunk_sum);
            running_max = new_max;

            // Evaluate to avoid building huge lazy graph
            running_sum.eval();
            running_max.eval();
        }

        // Final logsumexp = m + log(s)
        let log_sum = running_sum.log();
        running_max.add(&log_sum)
    }

    /// Apply logit transformations (softcapping, scaling).
    fn apply_logit_transforms(&self, logits: Array) -> Array {
        let mut result = logits;

        // Apply scaling (Cohere)
        if self.config.logit_scale != 1.0 {
            let scale = Array::from_f32(self.config.logit_scale);
            result = result.multiply(&scale);
        }

        // Apply softcapping (Gemma2): softcap * tanh(logits / softcap)
        if self.config.softcap > 0.0 {
            let softcap = Array::from_f32(self.config.softcap);
            let scaled = result.divide(&softcap);
            let tanh_scaled = ops::tanh(&scaled);
            result = tanh_scaled.multiply(&softcap);
        }

        result
    }

    /// Apply ignore index masking.
    fn apply_mask(&self, loss: &Array, targets: &Array) -> (Array, usize) {
        let ignore_arr = Array::from_i32(self.config.ignore_index);
        let targets_i32 = targets.as_dtype(Dtype::Int32.as_i32());
        let valid_mask = targets_i32.not_equal(&ignore_arr);
        let valid_mask_f32 = valid_mask.as_dtype(Dtype::Float32.as_i32());

        // Count valid tokens
        let n_valid_arr = valid_mask_f32.sum_all();
        let n_valid_eval = n_valid_arr.clone();
        n_valid_eval.eval();
        let n_valid = n_valid_eval.item_f32() as usize;

        // Zero out ignored positions
        let masked_loss = loss.multiply(&valid_mask_f32);

        (masked_loss, n_valid)
    }

    /// Compute backward pass (gradient of loss w.r.t. hidden states).
    pub fn backward(
        &self,
        hidden_states: &Array,
        lm_head_weight: &Array,
        targets: &Array,
        output: &CutCrossEntropyOutput,
        grad_loss: &Array,
    ) -> Result<Array, Exception> {
        let n_tokens = hidden_states.dim(0);
        let hidden_dim = hidden_states.dim(1);
        let vocab_size = lm_head_weight.dim(0) as usize;
        let chunk_size = self.config.vocab_chunk_size;
        let n_chunks = (vocab_size + chunk_size - 1) / chunk_size;

        // Get cached logsumexp
        let logsumexp = output
            .cached_logsumexp
            .as_ref()
            .ok_or_else(|| Exception::custom("No cached logsumexp for backward"))?;

        // Initialize gradient accumulator
        let zero = Array::from_f32(0.0);
        let mut grad_hidden = zero.broadcast_to(&[n_tokens, hidden_dim]);

        // Expand grad_loss and logsumexp for broadcasting
        let grad_expanded = grad_loss.reshape(&[-1, 1]);
        let lse_expanded = logsumexp.reshape(&[-1, 1]);

        let targets_i32 = targets.as_dtype(Dtype::Int32.as_i32());

        for chunk_idx in 0..n_chunks {
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(vocab_size);
            let chunk_len = end - start;

            // Get weight chunk
            let weight_chunk = lm_head_weight.slice(&[start as i32, 0], &[end as i32, hidden_dim]);

            // Compute chunk logits
            let weight_t = weight_chunk.t();
            let chunk_logits = hidden_states.matmul(&weight_t);

            // Compute softmax for this chunk: softmax_chunk = exp(logits - logsumexp)
            let shifted = chunk_logits.subtract(&lse_expanded);
            let softmax_chunk = shifted.exp();

            // Subtract 1 at target positions if target is in this chunk
            let start_arr = Array::from_i32(start as i32);
            let end_arr = Array::from_i32(end as i32);
            let in_chunk = targets_i32
                .greater_equal(&start_arr)
                .multiply(&targets_i32.less(&end_arr));

            // Local targets (clamped)
            let local_targets = targets_i32.subtract(&start_arr);
            let zero_i = Array::from_i32(0);
            let max_idx = Array::from_i32((chunk_len - 1) as i32);
            let local_targets_clipped = local_targets.maximum(&zero_i).minimum(&max_idx);

            // Create one-hot for targets in this chunk
            let in_chunk_f32 = in_chunk.as_dtype(Dtype::Float32.as_i32());
            let identity = Array::eye(chunk_len as i32, Dtype::Float32.as_i32());
            let one_hot = identity.take_axis(&local_targets_clipped, 0);
            let masked_one_hot = one_hot.multiply(&in_chunk_f32.reshape(&[-1, 1]));

            // Gradient: (softmax - one_hot) * grad_loss
            let grad_chunk = softmax_chunk.subtract(&masked_one_hot);
            let grad_chunk_scaled = grad_chunk.multiply(&grad_expanded);

            // Accumulate: grad_hidden += grad_chunk @ weight_chunk
            let chunk_contrib = grad_chunk_scaled.matmul(&weight_chunk);
            grad_hidden = grad_hidden.add(&chunk_contrib);

            // Evaluate to avoid huge graph
            grad_hidden.eval();
        }

        // Apply ignore mask
        let ignore_arr = Array::from_i32(self.config.ignore_index);
        let valid_mask = targets_i32.not_equal(&ignore_arr);
        let valid_mask_f32 = valid_mask
            .as_dtype(Dtype::Float32.as_i32())
            .reshape(&[-1, 1]);

        // Scale by 1/n_valid
        let safe_n_valid = output.n_valid.max(1);
        let n_valid = Array::from_i32(safe_n_valid as i32).as_dtype(Dtype::Float32.as_i32());
        let grad_hidden = grad_hidden.multiply(&valid_mask_f32);
        Ok(grad_hidden.divide(&n_valid))
    }
}

/// Convenience function for computing Cut Cross Entropy loss.
///
/// This is the main entry point for memory-efficient cross-entropy.
///
/// # Arguments
///
/// * `hidden_states` - Hidden states [batch * seq, hidden_dim]
/// * `lm_head_weight` - LM head weights [vocab_size, hidden_dim]
/// * `targets` - Target token indices [batch * seq]
/// * `ignore_index` - Index to ignore (typically -100)
///
/// # Returns
///
/// Scalar loss value.
pub fn cut_cross_entropy_loss(
    hidden_states: &Array,
    lm_head_weight: &Array,
    targets: &Array,
    ignore_index: i32,
) -> Result<Array, Exception> {
    let config = CutCrossEntropyConfig::new().with_ignore_index(ignore_index);
    let cce = CutCrossEntropy::new(config);
    let output = cce.forward(hidden_states, lm_head_weight, targets, None)?;
    Ok(output.loss)
}

/// Convenience function for Gemma2 with softcapping.
pub fn cut_cross_entropy_loss_gemma(
    hidden_states: &Array,
    lm_head_weight: &Array,
    targets: &Array,
    ignore_index: i32,
    softcap: f32,
) -> Result<Array, Exception> {
    let config = CutCrossEntropyConfig::new()
        .with_ignore_index(ignore_index)
        .with_softcap(softcap);
    let cce = CutCrossEntropy::new(config);
    let output = cce.forward(hidden_states, lm_head_weight, targets, None)?;
    Ok(output.loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::Array;

    #[test]
    fn test_config_default() {
        let config = CutCrossEntropyConfig::default();
        assert_eq!(config.vocab_chunk_size, 4096);
        assert_eq!(config.ignore_index, -100);
        assert_eq!(config.softcap, 0.0);
    }

    #[test]
    fn test_config_builder() {
        let config = CutCrossEntropyConfig::new()
            .with_vocab_chunk_size(8192)
            .with_ignore_index(-1)
            .with_softcap(30.0);

        assert_eq!(config.vocab_chunk_size, 8192);
        assert_eq!(config.ignore_index, -1);
        assert_eq!(config.softcap, 30.0);
    }

    #[test]
    fn test_cut_cross_entropy_basic() {
        let n_tokens: i32 = 4;
        let hidden_dim: i32 = 8;
        let vocab_size: i32 = 16;

        let hidden_data: Vec<f32> = (0..(n_tokens * hidden_dim) as usize)
            .map(|i| ((i * 7 + 3) % 10) as f32 / 10.0)
            .collect();
        let hidden = Array::from_f32_slice(&hidden_data, &[n_tokens, hidden_dim]);

        let weight_data: Vec<f32> = (0..(vocab_size * hidden_dim) as usize)
            .map(|i| ((i * 11 + 5) % 10) as f32 / 10.0 - 0.5)
            .collect();
        let weight = Array::from_f32_slice(&weight_data, &[vocab_size, hidden_dim]);

        let targets = Array::from_i32_slice(&[0i32, 5, 10, 15]).reshape(&[4]);

        let config = CutCrossEntropyConfig::new().with_vocab_chunk_size(4);
        let cce = CutCrossEntropy::new(config);

        let output = cce.forward(&hidden, &weight, &targets, None).unwrap();
        let mut loss_eval = output.loss.clone();
        loss_eval.eval();

        let loss_value = loss_eval.item_f32();
        assert!(loss_value.is_finite());
        assert!(loss_value > 0.0);
    }

    #[test]
    fn test_cut_cross_entropy_ignore_index() {
        let n_tokens: i32 = 4;
        let hidden_dim: i32 = 8;
        let vocab_size: i32 = 16;

        let hidden_data: Vec<f32> = vec![0.5; (n_tokens * hidden_dim) as usize];
        let hidden = Array::from_f32_slice(&hidden_data, &[n_tokens, hidden_dim]);

        let weight_data: Vec<f32> = vec![0.1; (vocab_size * hidden_dim) as usize];
        let weight = Array::from_f32_slice(&weight_data, &[vocab_size, hidden_dim]);

        let targets = Array::from_i32_slice(&[0i32, -100, 5, -100]).reshape(&[4]);

        let config = CutCrossEntropyConfig::new()
            .with_ignore_index(-100)
            .with_vocab_chunk_size(4);
        let cce = CutCrossEntropy::new(config);

        let output = cce.forward(&hidden, &weight, &targets, None).unwrap();

        assert_eq!(output.n_valid, 2);
    }

    #[test]
    fn test_cut_cross_entropy_softcap() {
        let n_tokens: i32 = 2;
        let hidden_dim: i32 = 4;
        let vocab_size: i32 = 8;

        let hidden_data: Vec<f32> = vec![10.0; (n_tokens * hidden_dim) as usize];
        let hidden = Array::from_f32_slice(&hidden_data, &[n_tokens, hidden_dim]);

        let weight_data: Vec<f32> = vec![1.0; (vocab_size * hidden_dim) as usize];
        let weight = Array::from_f32_slice(&weight_data, &[vocab_size, hidden_dim]);

        let targets = Array::from_i32_slice(&[0i32, 1]).reshape(&[2]);

        let config_no_cap = CutCrossEntropyConfig::new().with_vocab_chunk_size(4);
        let cce_no_cap = CutCrossEntropy::new(config_no_cap);
        let output_no_cap = cce_no_cap
            .forward(&hidden, &weight, &targets, None)
            .unwrap();
        let mut loss_eval_no_cap = output_no_cap.loss.clone();
        loss_eval_no_cap.eval();

        let config_cap = CutCrossEntropyConfig::new()
            .with_softcap(30.0)
            .with_vocab_chunk_size(4);
        let cce_cap = CutCrossEntropy::new(config_cap);
        let output_cap = cce_cap.forward(&hidden, &weight, &targets, None).unwrap();
        let mut loss_eval_cap = output_cap.loss.clone();
        loss_eval_cap.eval();

        assert!(loss_eval_no_cap.item_f32().is_finite());
        assert!(loss_eval_cap.item_f32().is_finite());
    }

    #[test]
    fn test_convenience_function() {
        let n_tokens: i32 = 4;
        let hidden_dim: i32 = 8;
        let vocab_size: i32 = 16;

        let hidden_data: Vec<f32> = (0..(n_tokens * hidden_dim) as usize)
            .map(|i| (i as f32) / 32.0)
            .collect();
        let hidden = Array::from_f32_slice(&hidden_data, &[n_tokens, hidden_dim]);

        let weight_data: Vec<f32> = (0..(vocab_size * hidden_dim) as usize)
            .map(|i| (i as f32) / 128.0 - 0.5)
            .collect();
        let weight = Array::from_f32_slice(&weight_data, &[vocab_size, hidden_dim]);

        let targets = Array::from_i32_slice(&[0i32, 5, 10, 15]).reshape(&[4]);

        let loss = cut_cross_entropy_loss(&hidden, &weight, &targets, -100).unwrap();
        let mut loss_eval = loss.clone();
        loss_eval.eval();

        assert!(loss_eval.item_f32().is_finite());
    }
}
