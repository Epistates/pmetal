//! Soft Cross-Entropy loss for knowledge distillation.
//!
//! GPU-first implementation using Metal kernels for optimal performance on Apple Silicon.
//! Falls back to MLX operations when Metal is unavailable.
//!
//! Uses teacher's softmax outputs as soft targets instead of one-hot labels.
//! CE(teacher_soft, student_logits) = -sum(teacher_soft * log_softmax(student))
//!
//! # Zero-Copy Optimization
//!
//! On Apple Silicon, MLX and Metal share unified memory. This implementation uses
//! zero-copy bridging to pass MLX array data directly to Metal kernels without
//! copying, providing significant performance improvements for large tensors.

use std::ops::Neg;

use super::{softmax, DistillLoss};
use crate::Result;
use mlx_rs::Array;

#[cfg(feature = "metal")]
use std::sync::Arc;

#[cfg(feature = "metal")]
use pmetal_metal::{
    bridge::metal_buffer_from_ptr,
    context::MetalContext,
    kernels::{DistillLossType as MetalDistillLossType, FusedDistill, FusedDistillConfig},
};

/// Soft Cross-Entropy loss for knowledge distillation.
///
/// Computes cross-entropy between teacher's soft targets and student's predictions.
/// This is equivalent to optimizing KL divergence up to a constant.
///
/// # GPU Acceleration
///
/// When the `metal` feature is enabled (default), this implementation uses
/// custom Metal kernels with online softmax for optimal memory efficiency.
///
/// # Zero-Copy Optimization
///
/// This implementation uses zero-copy bridging to pass MLX array data directly
/// to Metal kernels without copying. This is possible because MLX and Metal share
/// unified memory on Apple Silicon.
pub struct SoftCrossEntropyLoss {
    /// Cached Metal context for GPU acceleration.
    #[cfg(feature = "metal")]
    ctx: Option<Arc<MetalContext>>,

    #[cfg(not(feature = "metal"))]
    _phantom: (),
}

impl SoftCrossEntropyLoss {
    /// Create a new soft cross-entropy loss.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "metal")]
            ctx: MetalContext::global().ok(),
            #[cfg(not(feature = "metal"))]
            _phantom: (),
        }
    }

    /// Check if GPU acceleration is available.
    #[cfg(feature = "metal")]
    pub fn is_gpu_available(&self) -> bool {
        self.ctx.is_some()
    }

    #[cfg(not(feature = "metal"))]
    pub fn is_gpu_available(&self) -> bool {
        false
    }

    /// GPU-accelerated forward pass using Metal kernels with zero-copy bridging.
    ///
    /// Uses zero-copy bridging to pass MLX array data directly to Metal kernels
    /// without copying. This is possible because MLX and Metal share unified
    /// memory on Apple Silicon.
    #[cfg(feature = "metal")]
    fn compute_gpu(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
    ) -> Result<Array> {
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| crate::DistillError::Metal("Metal context not available".to_string()))?;

        let shape = teacher_logits.shape();
        if shape.len() < 2 {
            return Err(crate::DistillError::Other(
                "Logits must have at least 2 dimensions".to_string(),
            ));
        }

        // Get dimensions - handle both [batch, seq, vocab] and [tokens, vocab]
        let vocab_size = shape[shape.len() - 1] as usize;
        let num_tokens: usize = shape[..shape.len() - 1]
            .iter()
            .map(|&d| d as usize)
            .product();
        let total_elements = num_tokens * vocab_size;

        // Flatten to [num_tokens, vocab] for Metal kernel
        let teacher_flat = teacher_logits.reshape(&[-1, vocab_size as i32])?;
        let student_flat = student_logits.reshape(&[-1, vocab_size as i32])?;

        // Evaluate the arrays to ensure data is computed and available
        teacher_flat.eval()?;
        student_flat.eval()?;

        // Get raw data pointers using mlx-rs safe API (zero-copy on Apple Silicon unified memory)
        // Using as_slice() is safe - it returns a slice backed by the array's data
        let teacher_slice = teacher_flat.as_slice::<f32>();
        let student_slice = student_flat.as_slice::<f32>();
        let teacher_ptr = teacher_slice.as_ptr() as *mut f32;
        let student_ptr = student_slice.as_ptr() as *mut f32;

        // Create zero-copy Metal buffer views from the MLX array pointers
        // SAFETY:
        // 1. Pointers are from valid slices (as_slice ensures array is evaluated)
        // 2. Arrays remain in scope - slices borrow from them
        // 3. Apple Silicon unified memory allows GPU access to CPU memory
        // 4. total_elements correctly represents the array size
        let teacher_view = unsafe {
            metal_buffer_from_ptr(ctx, teacher_ptr, total_elements)
                .map_err(|e| crate::DistillError::Metal(format!("Buffer view error: {}", e)))?
        };
        let student_view = unsafe {
            metal_buffer_from_ptr(ctx, student_ptr, total_elements)
                .map_err(|e| crate::DistillError::Metal(format!("Buffer view error: {}", e)))?
        };

        // Configure kernel with automatic SIMD selection for large vocabularies
        let config = FusedDistillConfig::new(num_tokens, vocab_size).with_temperature(temperature);

        let kernel = FusedDistill::new(ctx.clone(), config)
            .map_err(|e| crate::DistillError::Metal(format!("Kernel error: {}", e)))?;

        // Execute kernel with zero-copy buffer views
        let output = kernel
            .forward(
                &teacher_view,
                &student_view,
                MetalDistillLossType::SoftCrossEntropy,
            )
            .map_err(|e| crate::DistillError::Metal(format!("Execution error: {}", e)))?;

        // Return mean loss
        let mean_loss = output.mean_loss();

        Ok(Array::from_f32(mean_loss))
    }

    /// MLX fallback implementation.
    fn compute_mlx(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
    ) -> Result<Array> {
        // Scale logits by temperature
        let temp = Array::from_f32(temperature);
        let teacher_scaled = teacher_logits.divide(&temp)?;
        let student_scaled = student_logits.divide(&temp)?;

        // Teacher soft targets
        let teacher_probs = softmax(&teacher_scaled, -1)?;

        // Student log probabilities (log_softmax for numerical stability)
        let student_log_probs = log_softmax(&student_scaled, -1)?;

        // Cross-entropy: -sum(p * log(q))
        let neg_ce = teacher_probs.multiply(&student_log_probs)?;

        // Sum over vocabulary dimension
        let ce_per_token = neg_ce.sum_axes(&[-1], Some(false))?;

        // Negate and mean over batch and sequence
        let loss = ce_per_token.neg().mean(None)?;

        Ok(loss)
    }
}

impl Default for SoftCrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillLoss for SoftCrossEntropyLoss {
    fn compute(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
    ) -> Result<Array> {
        // GPU-first: try Metal, fall back to MLX
        #[cfg(feature = "metal")]
        {
            if self.ctx.is_some() {
                return self.compute_gpu(teacher_logits, student_logits, temperature);
            }
        }

        self.compute_mlx(teacher_logits, student_logits, temperature)
    }

    fn name(&self) -> &'static str {
        "soft_cross_entropy"
    }
}

/// Log-softmax for numerical stability.
fn log_softmax(x: &Array, axis: i32) -> Result<Array> {
    let max_x = x.max_axes(&[axis], Some(true))?;
    let shifted = x.subtract(&max_x)?;
    let exp_shifted = shifted.exp()?;
    let sum_exp = exp_shifted.sum_axes(&[axis], Some(true))?;
    let log_sum_exp = sum_exp.log()?;
    Ok(shifted.subtract(&log_sum_exp)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_soft_ce_identical_distributions() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let loss = SoftCrossEntropyLoss::new();
        let result = loss.compute(&logits, &logits, 1.0).unwrap();
        let value: f32 = result.item();

        // Soft CE of a distribution with itself equals its entropy
        // This should be positive and bounded
        assert!(value > 0.0, "Soft CE should be positive, got {}", value);
        assert!(value < 10.0, "Soft CE should be reasonable, got {}", value);
    }

    #[test]
    fn test_soft_ce_different_distributions() {
        let teacher = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = SoftCrossEntropyLoss::new();

        // CE with itself (entropy)
        let self_ce = loss.compute(&teacher, &teacher, 1.0).unwrap();
        // CE with different distribution
        let cross_ce = loss.compute(&teacher, &student, 1.0).unwrap();

        let self_val: f32 = self_ce.item();
        let cross_val: f32 = cross_ce.item();

        // Cross-entropy should be >= entropy (Gibbs' inequality)
        assert!(
            cross_val >= self_val - 1e-4,
            "CE(P, Q) should be >= H(P): CE={}, H={}",
            cross_val,
            self_val
        );
    }

    #[test]
    fn test_soft_ce_temperature_effect() {
        let teacher = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = SoftCrossEntropyLoss::new();

        // Higher temperature makes distributions more uniform
        let ce_t1 = loss.compute(&teacher, &student, 1.0).unwrap();
        let ce_t4 = loss.compute(&teacher, &student, 4.0).unwrap();

        let v1: f32 = ce_t1.item();
        let v4: f32 = ce_t4.item();

        // At higher temperature, distributions are more similar
        // so cross-entropy approaches entropy
        assert!(
            v4 < v1,
            "Higher temp should reduce soft CE: T=1: {}, T=4: {}",
            v1,
            v4
        );
    }

    #[test]
    fn test_soft_ce_batch_processing() {
        // Test with batch of sequences
        let teacher = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0], &[2, 1, 4]);
        let student = Array::from_slice(&[4.0_f32, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0], &[2, 1, 4]);

        let loss = SoftCrossEntropyLoss::new();
        let result = loss.compute(&teacher, &student, 1.0).unwrap();

        // Result should be a scalar
        assert!(result.shape().is_empty());
        let value: f32 = result.item();
        assert!(value > 0.0);
    }

    #[cfg(feature = "metal")]
    #[test]
    #[serial]
    fn test_gpu_acceleration_available() {
        let loss = SoftCrossEntropyLoss::new();
        println!("GPU available: {}", loss.is_gpu_available());
    }

    #[test]
    #[serial]
    fn test_larger_batch() {
        // Test with larger tensors to exercise GPU path
        let batch_size = 4;
        let seq_len = 8;
        let vocab_size = 1024;

        let teacher_data: Vec<f32> = (0..(batch_size * seq_len * vocab_size))
            .map(|i| ((i % 100) as f32 - 50.0) / 10.0)
            .collect();
        let student_data: Vec<f32> = (0..(batch_size * seq_len * vocab_size))
            .map(|i| ((i * 7 % 100) as f32 - 50.0) / 10.0)
            .collect();

        let teacher = Array::from_slice(
            &teacher_data,
            &[batch_size as i32, seq_len as i32, vocab_size as i32],
        );
        let student = Array::from_slice(
            &student_data,
            &[batch_size as i32, seq_len as i32, vocab_size as i32],
        );

        let loss = SoftCrossEntropyLoss::new();
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();

        // Should be positive and finite
        assert!(value > 0.0, "Soft CE should be positive");
        assert!(value.is_finite(), "Soft CE should be finite");
    }
}
