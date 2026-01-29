//! Jensen-Shannon Divergence loss for knowledge distillation.
//!
//! GPU-first implementation using Metal kernels for optimal performance on Apple Silicon.
//! Falls back to MLX operations when Metal is unavailable.
//!
//! JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
//! where M = 0.5 * (P + Q)
//!
//! Jensen-Shannon is symmetric and bounded [0, log(2)], making it
//! more stable than KL divergence for distillation.
//!
//! # Zero-Copy Optimization
//!
//! On Apple Silicon, MLX and Metal share unified memory. This implementation uses
//! zero-copy bridging to pass MLX array data directly to Metal kernels without
//! copying, providing significant performance improvements for large tensors.

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

/// Jensen-Shannon Divergence loss for knowledge distillation.
///
/// A symmetric, bounded alternative to KL divergence.
/// JS(P || Q) = JS(Q || P), unlike KL divergence.
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
pub struct JensenShannonLoss {
    /// Cached Metal context for GPU acceleration.
    #[cfg(feature = "metal")]
    ctx: Option<Arc<MetalContext>>,

    #[cfg(not(feature = "metal"))]
    _phantom: (),
}

impl JensenShannonLoss {
    /// Create a new Jensen-Shannon divergence loss.
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
                MetalDistillLossType::JensenShannon,
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

        // Compute softmax probabilities
        let teacher_probs = softmax(&teacher_scaled, -1)?;
        let student_probs = softmax(&student_scaled, -1)?;

        // Compute mixture distribution M = 0.5 * (P + Q)
        let half = Array::from_f32(0.5);
        let mixture = teacher_probs.add(&student_probs)?.multiply(&half)?;

        // Add epsilon for numerical stability
        let eps = Array::from_f32(1e-10);
        let teacher_safe = teacher_probs.add(&eps)?;
        let student_safe = student_probs.add(&eps)?;
        let mixture_safe = mixture.add(&eps)?;

        // Compute KL(teacher || mixture)
        let log_teacher = teacher_safe.log()?;
        let log_mixture = mixture_safe.log()?;
        let kl_teacher_m = teacher_safe.multiply(&log_teacher.subtract(&log_mixture)?)?;

        // Compute KL(student || mixture)
        let log_student = student_safe.log()?;
        let kl_student_m = student_safe.multiply(&log_student.subtract(&log_mixture)?)?;

        // JS = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        let js = kl_teacher_m.add(&kl_student_m)?.multiply(&half)?;

        // Sum over vocabulary, mean over batch and sequence
        let js_sum = js.sum_axes(&[-1], Some(false))?;
        let loss = js_sum.mean(None)?;

        Ok(loss)
    }
}

impl Default for JensenShannonLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillLoss for JensenShannonLoss {
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
        "jensen_shannon"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_js_identical_distributions() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let loss = JensenShannonLoss::new();
        let result = loss.compute(&logits, &logits, 1.0).unwrap();
        let value: f32 = result.item();

        // JS of identical distributions should be 0
        assert!(
            value.abs() < 1e-4,
            "JS of identical distributions should be ~0, got {}",
            value
        );
    }

    #[test]
    fn test_js_symmetry() {
        let p = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let q = Array::from_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = JensenShannonLoss::new();
        let js_pq = loss.compute(&p, &q, 1.0).unwrap();
        let js_qp = loss.compute(&q, &p, 1.0).unwrap();

        let v_pq: f32 = js_pq.item();
        let v_qp: f32 = js_qp.item();

        // JS should be symmetric
        assert!(
            (v_pq - v_qp).abs() < 1e-4,
            "JS should be symmetric: JS(P||Q)={}, JS(Q||P)={}",
            v_pq,
            v_qp
        );
    }

    #[test]
    fn test_js_bounded() {
        // Even for very different distributions, JS should be bounded by log(2)
        let p = Array::from_slice(&[10.0_f32, 0.0, 0.0, 0.0], &[1, 1, 4]);
        let q = Array::from_slice(&[0.0_f32, 0.0, 0.0, 10.0], &[1, 1, 4]);

        let loss = JensenShannonLoss::new();
        let result = loss.compute(&p, &q, 1.0).unwrap();
        let value: f32 = result.item();

        let ln2 = 2.0_f32.ln();
        assert!(
            value <= ln2 + 1e-4,
            "JS should be bounded by ln(2)={}, got {}",
            ln2,
            value
        );
        assert!(value >= 0.0, "JS should be non-negative, got {}", value);
    }

    #[test]
    fn test_js_temperature_effect() {
        let teacher = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = JensenShannonLoss::new();

        // Higher temperature should reduce JS (softer distributions)
        let js_t1 = loss.compute(&teacher, &student, 1.0).unwrap();
        let js_t2 = loss.compute(&teacher, &student, 2.0).unwrap();

        let v1: f32 = js_t1.item();
        let v2: f32 = js_t2.item();

        assert!(
            v2 < v1,
            "Higher temp should reduce JS: T=1: {}, T=2: {}",
            v1,
            v2
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    #[serial]
    fn test_gpu_acceleration_available() {
        let loss = JensenShannonLoss::new();
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

        let loss = JensenShannonLoss::new();
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();

        // Should be positive and finite
        assert!(value >= 0.0, "JS should be non-negative");
        assert!(value.is_finite(), "JS should be finite");
    }
}
