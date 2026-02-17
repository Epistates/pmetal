//! 8-bit Adam Optimizer for memory-efficient training.
//!
//! This module implements an 8-bit Adam optimizer inspired by bitsandbytes,
//! which stores optimizer states (first and second moments) in 8-bit precision
//! with block-wise dynamic quantization.
//!
//! # Memory Savings
//!
//! Traditional Adam stores:
//! - Parameters: 4 bytes/param (fp32)
//! - First moment (m): 4 bytes/param
//! - Second moment (v): 4 bytes/param
//! - Total: 12 bytes/param
//!
//! 8-bit Adam stores:
//! - Parameters: 4 bytes/param (fp32)
//! - First moment (m): 1 byte/param + scaling factors
//! - Second moment (v): 1 byte/param + scaling factors
//! - Total: ~6 bytes/param
//!
//! This results in ~50% memory reduction for optimizer states (from 8 to ~2 bytes).
//!
//! # Block-wise Quantization
//!
//! Instead of a single scale factor for the entire tensor, we use block-wise
//! scaling (default block size: 2048) for better precision retention.
//!
//! # Example
//!
//! ```ignore
//! use pmetal_trainer::{Adam8bit, Adam8bitBuilder};
//!
//! let optimizer = Adam8bitBuilder::new(2e-4)
//!     .with_weight_decay(0.01)
//!     .with_block_size(2048)
//!     .build()?;
//! ```

use std::collections::HashMap;
use std::rc::Rc;

use mlx_rs::{
    error::{Exception, Result},
    ops::indexing::IndexOp,
    Array, Dtype,
};

/// Error type for 8-bit Adam operations.
#[derive(Debug, thiserror::Error)]
pub enum Adam8bitError {
    /// MLX error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),
}

/// 8-bit Adam optimizer state for a single parameter.
#[derive(Debug, Clone)]
pub struct Adam8bitState {
    /// First moment (m) in int8 format.
    pub m_quantized: Vec<i8>,
    /// Scale factors for m (one per block).
    pub m_scales: Vec<f32>,
    /// Second moment (v) in uint8 format (always positive).
    pub v_quantized: Vec<u8>,
    /// Scale factors for v (one per block).
    pub v_scales: Vec<f32>,
    /// Original shape of the parameter.
    pub shape: Vec<i32>,
    /// Block size used for quantization.
    pub block_size: usize,
}

/// 8-bit Adam optimizer configuration.
#[derive(Debug, Clone)]
pub struct Adam8bitConfig {
    /// Learning rate.
    pub lr: f32,
    /// First moment decay (beta1).
    pub beta1: f32,
    /// Second moment decay (beta2).
    pub beta2: f32,
    /// Epsilon for numerical stability.
    pub eps: f32,
    /// Weight decay coefficient (decoupled, AdamW-style).
    pub weight_decay: f32,
    /// Block size for quantization (default: 2048).
    ///
    /// Smaller blocks preserve more precision for mixed-magnitude values
    /// at the cost of more scale factors. Must be >= 1.
    pub block_size: usize,
    /// Minimum absolute value to quantize (values below become 0).
    ///
    /// When the maximum absolute value in a block is below this threshold,
    /// the entire block is quantized to zeros. Default: 1e-12.
    pub eps_quantize: f32,
}

impl Default for Adam8bitConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            block_size: 2048,
            eps_quantize: 1e-12,
        }
    }
}

/// 8-bit Adam optimizer.
///
/// Memory-efficient Adam that stores first and second moments in 8-bit format
/// with block-wise dynamic quantization. Achieves ~50% memory reduction for
/// optimizer states compared to standard Adam.
///
/// # Quantization
///
/// Moments are stored per-block (default 2048 elements) with independent scaling:
/// - First moment (m): signed int8 with ~0.4% typical precision loss
/// - Second moment (v): unsigned uint8 with ~0.4% typical precision loss
///
/// Values much smaller than the block maximum may be quantized to zero.
/// This is expected and matches the bitsandbytes behavior.
///
/// # Thread Safety
///
/// This optimizer is **not thread-safe**. It uses `Rc<str>` for parameter keys
/// (matching the crate-wide convention) and `&mut self` for all update methods.
/// Do not share across threads.
///
/// # Requirements
///
/// All gradient and parameter arrays must be `Float32` dtype.
#[derive(Debug)]
pub struct Adam8bit {
    /// Optimizer configuration.
    pub config: Adam8bitConfig,
    /// Per-parameter state (8-bit m and v).
    pub state: HashMap<Rc<str>, Adam8bitState>,
    /// Training step counter.
    pub step: u64,
}

impl Adam8bit {
    /// Create a new 8-bit Adam optimizer.
    pub fn new(config: Adam8bitConfig) -> Self {
        Self {
            config,
            state: HashMap::new(),
            step: 0,
        }
    }

    /// Create with a simple learning rate.
    pub fn with_lr(lr: f32) -> Self {
        Self::new(Adam8bitConfig {
            lr,
            ..Default::default()
        })
    }

    /// Quantize a tensor to int8 with block-wise scaling.
    fn quantize_signed(&self, data: &[f32]) -> (Vec<i8>, Vec<f32>) {
        let size = data.len();
        let block_size = self.config.block_size.max(1); // Prevent division by zero

        // Use checked arithmetic to prevent overflow
        let num_blocks = size.saturating_add(block_size - 1) / block_size;

        let mut quantized = Vec::with_capacity(size);
        let mut scales = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(size);
            let block_data = &data[start..end];

            // Find max absolute value in block
            let max_abs = block_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

            // Scale factor: max_abs / 127 (for int8 range)
            let scale = if max_abs < self.config.eps_quantize {
                0.0
            } else {
                max_abs / 127.0
            };
            scales.push(scale);

            // Quantize block values
            for &val in block_data {
                let q = if scale > 0.0 {
                    (val / scale).round().clamp(-127.0, 127.0) as i8
                } else {
                    0i8
                };
                quantized.push(q);
            }
        }

        (quantized, scales)
    }

    /// Quantize a tensor to uint8 (for second moment, always positive).
    fn quantize_unsigned(&self, data: &[f32]) -> (Vec<u8>, Vec<f32>) {
        let size = data.len();
        let block_size = self.config.block_size.max(1); // Prevent division by zero

        // Use checked arithmetic to prevent overflow
        let num_blocks = size.saturating_add(block_size - 1) / block_size;

        let mut quantized = Vec::with_capacity(size);
        let mut scales = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(size);
            let block_data = &data[start..end];

            // Find max value in block (v is always positive due to squared gradients)
            let max_val = block_data.iter().fold(0.0f32, |a, &b| f32::max(a, b));

            // Scale factor: max_val / 255 (for uint8 range)
            let scale = if max_val < self.config.eps_quantize {
                0.0
            } else {
                max_val / 255.0
            };
            scales.push(scale);

            // Quantize block values
            for &val in block_data {
                let q = if scale > 0.0 {
                    (val / scale).round().clamp(0.0, 255.0) as u8
                } else {
                    0u8
                };
                quantized.push(q);
            }
        }

        (quantized, scales)
    }

    /// Dequantize a signed int8 tensor.
    fn dequantize_signed(&self, quantized: &[i8], scales: &[f32]) -> Vec<f32> {
        let size = quantized.len();
        let block_size = self.config.block_size;

        let mut result = Vec::with_capacity(size);
        for (i, &q) in quantized.iter().enumerate() {
            let block_idx = i / block_size;
            let scale = scales.get(block_idx).copied().unwrap_or(0.0);
            result.push((q as f32) * scale);
        }

        result
    }

    /// Dequantize an unsigned uint8 tensor.
    fn dequantize_unsigned(&self, quantized: &[u8], scales: &[f32]) -> Vec<f32> {
        let size = quantized.len();
        let block_size = self.config.block_size;

        let mut result = Vec::with_capacity(size);
        for (i, &q) in quantized.iter().enumerate() {
            let block_idx = i / block_size;
            let scale = scales.get(block_idx).copied().unwrap_or(0.0);
            result.push((q as f32) * scale);
        }

        result
    }

    /// Get memory usage statistics.
    pub fn memory_usage(&self) -> Adam8bitMemoryStats {
        let mut total_param_elements = 0usize;
        let mut total_state_bytes = 0usize;

        for state in self.state.values() {
            let param_size = state.m_quantized.len();
            total_param_elements += param_size;

            // 8-bit quantized m and v
            total_state_bytes += param_size; // m in int8
            total_state_bytes += param_size; // v in uint8

            // Scale factors (4 bytes each)
            total_state_bytes += state.m_scales.len() * 4; // m scales
            total_state_bytes += state.v_scales.len() * 4; // v scales
        }

        let fp32_state_bytes = total_param_elements * 8; // m + v in fp32

        Adam8bitMemoryStats {
            total_params: total_param_elements,
            state_bytes_8bit: total_state_bytes,
            state_bytes_fp32: fp32_state_bytes,
            memory_saved: if fp32_state_bytes > 0 {
                1.0 - (total_state_bytes as f64 / fp32_state_bytes as f64)
            } else {
                0.0
            },
        }
    }

    /// Get the learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.config.lr
    }

    /// Set the learning rate.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.lr = lr;
    }

    /// Apply a single parameter update (vectorized), incrementing the step counter.
    ///
    /// This is the public entry point for standalone single-parameter updates.
    /// For batch updates, use [`update`] which increments the step counter once
    /// for all parameters.
    ///
    /// Adam update rule:
    /// ```text
    /// m = beta1 * m + (1 - beta1) * g
    /// v = beta2 * v + (1 - beta2) * g^2
    /// m_hat = m / (1 - beta1^t)
    /// v_hat = v / (1 - beta2^t)
    /// param = param - lr * m_hat / (sqrt(v_hat) + eps) - lr * weight_decay * param
    /// ```
    pub fn update_single(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<()> {
        self.step += 1;
        self.apply_update(key, gradient, parameter)
    }

    /// Internal: apply Adam update for one parameter using the current step counter.
    ///
    /// Does NOT increment `self.step` — caller is responsible for that.
    fn apply_update(
        &mut self,
        key: &Rc<str>,
        gradient: &Array,
        parameter: &mut Array,
    ) -> Result<()> {
        debug_assert!(
            self.step > 0,
            "apply_update called with step=0; call update_single() or update() instead"
        );

        gradient.eval()?;
        parameter.eval()?;

        // Validate dtypes — quantization assumes Float32
        if gradient.dtype() != Dtype::Float32 {
            return Err(Exception::custom(format!(
                "Adam8bit requires Float32 gradients, got {:?}",
                gradient.dtype()
            )));
        }
        if parameter.dtype() != Dtype::Float32 {
            return Err(Exception::custom(format!(
                "Adam8bit requires Float32 parameters, got {:?}",
                parameter.dtype()
            )));
        }

        let shape: Vec<i32> = parameter.shape().to_vec();
        let size = parameter.size();

        // Validate size to prevent integer overflow
        if size > 2_500_000_000 {
            return Err(Exception::custom(format!(
                "Parameter size {} exceeds maximum allowed (2.5B elements)",
                size
            )));
        }

        // Flatten arrays for vectorized operations
        let flat_grad = gradient.flatten(None, None)?;
        let flat_param = parameter.flatten(None, None)?;

        // Bias correction terms (step is already >= 1 by the time we get here).
        // Clamp to i32::MAX to prevent truncation on very long runs (>2.1B steps).
        // Beyond ~1000 steps, bias correction is negligible anyway (beta^1000 ≈ 0).
        let step_i32 = self.step.min(i32::MAX as u64) as i32;
        let beta1_t = self.config.beta1.powi(step_i32);
        let beta2_t = self.config.beta2.powi(step_i32);
        let bias_correction1 = 1.0 - beta1_t;
        let bias_correction2 = 1.0 - beta2_t;

        // Get or initialize state as Arrays for vectorized ops
        let (m_array, v_array) = if let Some(state) = self.state.get(key) {
            // Dequantize to arrays
            let m_data = self.dequantize_signed(&state.m_quantized, &state.m_scales);
            let v_data = self.dequantize_unsigned(&state.v_quantized, &state.v_scales);
            (
                Array::from_slice(&m_data, &[size as i32]),
                Array::from_slice(&v_data, &[size as i32]),
            )
        } else {
            // Initialize to zeros
            (
                Array::from_slice(&vec![0.0f32; size], &[size as i32]),
                Array::from_slice(&vec![0.0f32; size], &[size as i32]),
            )
        };

        // Vectorized moment updates: m = beta1 * m + (1 - beta1) * g
        let beta1_arr = Array::from_f32(self.config.beta1);
        let one_minus_beta1 = Array::from_f32(1.0 - self.config.beta1);
        let m_new = m_array
            .multiply(&beta1_arr)?
            .add(&flat_grad.multiply(&one_minus_beta1)?)?;

        // v = beta2 * v + (1 - beta2) * g^2
        let beta2_arr = Array::from_f32(self.config.beta2);
        let one_minus_beta2 = Array::from_f32(1.0 - self.config.beta2);
        let grad_sq = flat_grad.multiply(&flat_grad)?;
        let v_new = v_array
            .multiply(&beta2_arr)?
            .add(&grad_sq.multiply(&one_minus_beta2)?)?;

        // Bias-corrected estimates
        let bc1 = Array::from_f32(bias_correction1);
        let bc2 = Array::from_f32(bias_correction2);
        let m_hat = m_new.divide(&bc1)?;
        let v_hat = v_new.divide(&bc2)?;

        // Compute update: lr * m_hat / (sqrt(v_hat) + eps)
        let lr_arr = Array::from_f32(self.config.lr);
        let eps_arr = Array::from_f32(self.config.eps);
        let v_sqrt = v_hat.sqrt()?;
        let denom = v_sqrt.add(&eps_arr)?;
        let update = m_hat.divide(&denom)?.multiply(&lr_arr)?;

        // Apply weight decay (AdamW-style) and update parameter
        let param_new = if self.config.weight_decay > 0.0 {
            let wd_factor = Array::from_f32(1.0 - self.config.lr * self.config.weight_decay);
            flat_param.multiply(&wd_factor)?.subtract(&update)?
        } else {
            flat_param.subtract(&update)?
        };

        // Evaluate to get data for quantization
        m_new.eval()?;
        v_new.eval()?;
        param_new.eval()?;

        // Extract data for quantization (this is still needed for 8-bit storage)
        let m_data = self.array_to_vec(&m_new)?;
        let v_data = self.array_to_vec(&v_new)?;

        // Quantize updated state
        let (m_q, m_s) = self.quantize_signed(&m_data);
        let (v_q, v_s) = self.quantize_unsigned(&v_data);

        // Store state
        self.state.insert(
            key.clone(),
            Adam8bitState {
                m_quantized: m_q,
                m_scales: m_s,
                v_quantized: v_q,
                v_scales: v_s,
                shape: shape.clone(),
                block_size: self.config.block_size,
            },
        );

        // Update parameter
        *parameter = param_new.reshape(&shape)?;

        Ok(())
    }

    /// Convert Array to Vec<f32> efficiently using as_slice.
    ///
    /// # Panics
    /// Debug-asserts that the array is Float32. In release builds, a dtype
    /// mismatch would produce garbage data.
    fn array_to_vec(&self, arr: &Array) -> Result<Vec<f32>> {
        arr.eval()?;
        let flat = arr.flatten(None, None)?;
        flat.eval()?;
        debug_assert_eq!(
            flat.dtype(),
            Dtype::Float32,
            "array_to_vec called with {:?}, expected Float32",
            flat.dtype()
        );
        Ok(flat.as_slice::<f32>().to_vec())
    }

    /// Apply updates to multiple parameters (one optimizer step).
    ///
    /// Increments the step counter once, then applies the Adam update to each
    /// parameter that has a corresponding gradient.
    pub fn update(
        &mut self,
        gradients: &HashMap<Rc<str>, Array>,
        parameters: &mut HashMap<Rc<str>, Array>,
    ) -> Result<()> {
        self.step += 1;

        for (key, grad) in gradients {
            if let Some(param) = parameters.get_mut(key) {
                self.apply_update(key, grad, param)?;
            }
        }
        Ok(())
    }
}

/// Memory usage statistics for 8-bit Adam.
#[derive(Debug, Clone)]
pub struct Adam8bitMemoryStats {
    /// Total number of parameters tracked.
    pub total_params: usize,
    /// Bytes used by 8-bit state.
    pub state_bytes_8bit: usize,
    /// Bytes that would be used by fp32 state.
    pub state_bytes_fp32: usize,
    /// Fraction of memory saved (0.0 - 1.0).
    pub memory_saved: f64,
}

impl Adam8bitMemoryStats {
    /// Get state size in megabytes (8-bit).
    pub fn state_mb_8bit(&self) -> f64 {
        self.state_bytes_8bit as f64 / 1_000_000.0
    }

    /// Get state size in megabytes (fp32 equivalent).
    pub fn state_mb_fp32(&self) -> f64 {
        self.state_bytes_fp32 as f64 / 1_000_000.0
    }
}

/// Builder for 8-bit Adam optimizer.
#[derive(Debug, Clone)]
pub struct Adam8bitBuilder {
    config: Adam8bitConfig,
}

impl Adam8bitBuilder {
    /// Create a new builder with the given learning rate.
    pub fn new(lr: f32) -> Self {
        Self {
            config: Adam8bitConfig {
                lr,
                ..Default::default()
            },
        }
    }

    /// Set beta1 (first moment decay).
    pub fn with_beta1(mut self, beta1: f32) -> Self {
        self.config.beta1 = beta1;
        self
    }

    /// Set beta2 (second moment decay).
    pub fn with_beta2(mut self, beta2: f32) -> Self {
        self.config.beta2 = beta2;
        self
    }

    /// Set betas (beta1, beta2).
    pub fn with_betas(mut self, betas: (f32, f32)) -> Self {
        self.config.beta1 = betas.0;
        self.config.beta2 = betas.1;
        self
    }

    /// Set epsilon for numerical stability.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.config.eps = eps;
        self
    }

    /// Set weight decay (AdamW-style decoupled).
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.config.weight_decay = wd;
        self
    }

    /// Set block size for quantization.
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.config.block_size = block_size;
        self
    }

    /// Build the optimizer.
    pub fn build(self) -> Adam8bit {
        Adam8bit::new(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam8bit_creation() {
        let optimizer = Adam8bitBuilder::new(2e-4).with_weight_decay(0.01).build();

        assert!((optimizer.learning_rate() - 2e-4).abs() < 1e-8);
        assert_eq!(optimizer.config.block_size, 2048);
    }

    #[test]
    fn test_adam8bit_config() {
        let config = Adam8bitConfig::default();
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.999);
        assert_eq!(config.block_size, 2048);
    }

    #[test]
    fn test_adam8bit_update() {
        let mut optimizer = Adam8bitBuilder::new(0.1).build();

        // Create test parameter and gradient
        let mut param = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let grad = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[4]);

        let key: Rc<str> = Rc::from("test.weight");
        optimizer.update_single(&key, &grad, &mut param).unwrap();

        // Parameter should have changed
        param.eval().unwrap();
        let p0 = param.index(0);
        p0.eval().unwrap();
        // After one step, param should be updated
        assert!((p0.item::<f32>() - 1.0).abs() > 1e-6);

        // State should exist
        assert!(optimizer.state.contains_key(&key));
    }

    #[test]
    fn test_adam8bit_memory_stats() {
        let mut optimizer = Adam8bitBuilder::new(0.1).with_block_size(64).build();

        // Add some parameters
        let mut param1 = Array::from_slice(&vec![1.0f32; 1024], &[1024]);
        let grad1 = Array::from_slice(&vec![0.01f32; 1024], &[1024]);

        let key: Rc<str> = Rc::from("layer1.weight");
        optimizer.update_single(&key, &grad1, &mut param1).unwrap();

        let stats = optimizer.memory_usage();
        assert_eq!(stats.total_params, 1024);
        // 8-bit state should be significantly smaller than fp32
        assert!(stats.state_bytes_8bit < stats.state_bytes_fp32);
        assert!(stats.memory_saved > 0.0);
    }

    #[test]
    fn test_adam8bit_quantization() {
        let optimizer = Adam8bitBuilder::new(0.1).with_block_size(4).build();

        // Test signed quantization
        let data = vec![-1.0f32, 0.5, -0.5, 1.0];
        let (q, s) = optimizer.quantize_signed(&data);

        // Should have 1 block (4 elements, block size 4)
        assert_eq!(s.len(), 1);

        // Dequantize and check it's close to original
        let dq = optimizer.dequantize_signed(&q, &s);

        // Values should be approximately recovered
        for i in 0..4 {
            assert!((data[i] - dq[i]).abs() < 0.02);
        }
    }

    #[test]
    fn test_adam8bit_learning_rate() {
        let mut optimizer = Adam8bitBuilder::new(1e-4).build();
        assert!((optimizer.learning_rate() - 1e-4).abs() < 1e-10);

        optimizer.set_learning_rate(5e-5);
        assert!((optimizer.learning_rate() - 5e-5).abs() < 1e-10);
    }

    #[test]
    fn test_adam8bit_builder() {
        let optimizer = Adam8bitBuilder::new(2e-4)
            .with_betas((0.85, 0.95))
            .with_eps(1e-7)
            .with_weight_decay(0.1)
            .with_block_size(1024)
            .build();

        assert!((optimizer.config.beta1 - 0.85).abs() < 1e-6);
        assert!((optimizer.config.beta2 - 0.95).abs() < 1e-6);
        assert!((optimizer.config.eps - 1e-7).abs() < 1e-10);
        assert!((optimizer.config.weight_decay - 0.1).abs() < 1e-6);
        assert_eq!(optimizer.config.block_size, 1024);
    }

    #[test]
    fn test_multiple_updates() {
        let mut optimizer = Adam8bitBuilder::new(0.01).build();

        let mut param = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let key: Rc<str> = Rc::from("test.weight");

        // Multiple updates should accumulate moments
        for i in 0..10 {
            let grad = Array::from_slice(&[0.1f32 * (i as f32 + 1.0); 4], &[4]);
            optimizer.update_single(&key, &grad, &mut param).unwrap();
        }

        // Check step count
        assert_eq!(optimizer.step, 10);

        // Parameter should have moved (with small lr=0.01 and 10 steps)
        param.eval().unwrap();
        let p0 = param.index(0);
        p0.eval().unwrap();
        // Check that param moved from original 1.0
        assert!((p0.item::<f32>() - 1.0).abs() > 0.001);
    }
}
