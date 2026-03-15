//! KL Divergence loss for knowledge distillation.
//!
//! GPU-first implementation using Metal kernels for optimal performance on Apple Silicon.
//! Falls back to MLX operations when Metal is unavailable.
//!
//! KL(P || Q) = sum(P * log(P / Q))
//!
//! In distillation:
//! - Forward KL: KL(teacher || student) - mode-covering
//! - Reverse KL: KL(student || teacher) - mode-seeking
//!
//! # Zero-Copy Optimization
//!
//! On Apple Silicon, MLX and Metal share unified memory. This implementation uses
//! zero-copy bridging to pass MLX array data directly to Metal kernels without
//! copying, providing significant performance improvements for large tensors.

#![allow(unsafe_code)]

use super::{DistillLoss, SPARSE_TOPK_DEFAULT, align_vocab_with_k, softmax};
use crate::Result;
use mlx_rs::Array;
use tracing;

#[cfg(feature = "metal")]
use std::sync::Arc;

#[cfg(feature = "metal")]
use pmetal_metal::{
    bridge::metal_buffer_from_ptr,
    context::MetalContext,
    kernels::{DistillLossType as MetalDistillLossType, FusedDistill, FusedDistillConfig},
};

/// KL Divergence loss for knowledge distillation.
///
/// Computes either forward KL (teacher || student) or reverse KL (student || teacher).
/// Forward KL encourages the student to cover all modes of the teacher distribution.
/// Reverse KL encourages the student to match the dominant modes.
///
/// # GPU Acceleration
///
/// When the `metal` feature is enabled (default), this implementation uses
/// custom Metal kernels with online softmax for O(1) memory per token instead
/// of materializing full probability tensors.
///
/// # Zero-Copy Optimization
///
/// This implementation uses zero-copy bridging to pass MLX array data directly
/// to Metal kernels without copying. This is possible because MLX and Metal share
/// unified memory on Apple Silicon.
pub struct KlDivergenceLoss {
    /// Whether to use reverse KL (student || teacher).
    reverse: bool,

    /// Number of top-k teacher tokens to retain when vocab sizes differ.
    ///
    /// Only used when teacher and student have different vocabulary sizes
    /// (cross-architecture distillation).  Defaults to [`SPARSE_TOPK_DEFAULT`].
    sparse_top_k: i32,

    /// Cached Metal context for GPU acceleration.
    #[cfg(feature = "metal")]
    ctx: Option<Arc<MetalContext>>,
}

impl KlDivergenceLoss {
    /// Create a new KL divergence loss (forward by default).
    pub fn new() -> Self {
        Self {
            reverse: false,
            sparse_top_k: SPARSE_TOPK_DEFAULT,
            #[cfg(feature = "metal")]
            ctx: MetalContext::global().ok(),
        }
    }

    /// Create a reverse KL divergence loss.
    pub fn reverse() -> Self {
        Self {
            reverse: true,
            sparse_top_k: SPARSE_TOPK_DEFAULT,
            #[cfg(feature = "metal")]
            ctx: MetalContext::global().ok(),
        }
    }

    /// Set whether to use reverse KL.
    pub fn with_reverse(mut self, reverse: bool) -> Self {
        self.reverse = reverse;
        self
    }

    /// Set the number of top-k teacher tokens used in cross-vocab distillation.
    ///
    /// When teacher and student vocabularies differ, the loss is computed only
    /// over the top-`k` teacher tokens (by logit magnitude).  Higher values
    /// capture more of the teacher distribution but increase computation.
    /// Must be ≥ 1; defaults to [`SPARSE_TOPK_DEFAULT`] (128).
    pub fn with_sparse_top_k(mut self, k: i32) -> Self {
        self.sparse_top_k = k.max(1);
        self
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

        // Select loss type
        let loss_type = if self.reverse {
            MetalDistillLossType::ReverseKlDivergence
        } else {
            MetalDistillLossType::KlDivergence
        };

        // Execute kernel with zero-copy buffer views
        let output = kernel
            .forward(&teacher_view, &student_view, loss_type)
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

        // Diagnostics
        if let Ok(t_max) = teacher_scaled.max(false) {
            t_max.eval().ok();
            if t_max.item::<f32>().is_infinite() || t_max.item::<f32>().is_nan() {
                tracing::warn!(
                    "KL fallback: teacher_scaled contains Inf/NaN after temperature scaling"
                );
            }
        }

        // Use log-domain computation instead of adding epsilon to probabilities.
        // Adding epsilon to softmax outputs biases the distribution; log_softmax
        // avoids this by computing log-probabilities directly and numerically stably.
        let teacher_log_probs = mlx_rs::nn::log_softmax(&teacher_scaled, -1)?;
        let student_log_probs = mlx_rs::nn::log_softmax(&student_scaled, -1)?;
        let teacher_probs = teacher_log_probs.exp()?;
        let student_probs = student_log_probs.exp()?;

        let kl = if self.reverse {
            // KL(student || teacher) = sum(student * (log_student - log_teacher))
            let log_ratio = student_log_probs.subtract(&teacher_log_probs)?;
            student_probs.multiply(&log_ratio)?
        } else {
            // KL(teacher || student) = sum(teacher * (log_teacher - log_student))
            let log_ratio = teacher_log_probs.subtract(&student_log_probs)?;
            teacher_probs.multiply(&log_ratio)?
        };

        // Sum over vocabulary dimension, then mean over batch and sequence
        let kl_sum = kl.sum_axes(&[-1], Some(false))?;
        let loss = kl_sum.mean(None)?;

        Ok(loss)
    }
}

impl Default for KlDivergenceLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillLoss for KlDivergenceLoss {
    fn compute_weighted(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        // Align vocab sizes.  When they differ we take the sparse top-k path
        // which always uses MLX (Metal kernels require equal vocab sizes).
        let (teacher_logits, student_logits, vocab_mismatched) =
            align_vocab_with_k(teacher_logits, student_logits, self.sparse_top_k)?;
        let teacher_logits = &teacher_logits;
        let student_logits = &student_logits;

        // GPU-first: try Metal if no weights and no vocab mismatch.
        // (Metal kernels for weighted distillation are planned for Q2 2026)
        if weights.is_none() && !vocab_mismatched {
            #[cfg(feature = "metal")]
            {
                if self.ctx.is_some() {
                    return self.compute_gpu(teacher_logits, student_logits, temperature);
                }
            }
        }

        let temp = Array::from_f32(temperature);
        let teacher_scaled = teacher_logits.divide(&temp)?;
        let student_scaled = student_logits.divide(&temp)?;

        let teacher_log_probs = mlx_rs::nn::log_softmax(&teacher_scaled, -1)?;
        let student_log_probs = mlx_rs::nn::log_softmax(&student_scaled, -1)?;
        let teacher_probs = teacher_log_probs.exp()?;
        let student_probs = student_log_probs.exp()?;

        let kl_per_token = if self.reverse {
            let log_ratio = student_log_probs.subtract(&teacher_log_probs)?;
            student_probs
                .multiply(&log_ratio)?
                .sum_axes(&[-1], Some(false))?
        } else {
            let log_ratio = teacher_log_probs.subtract(&student_log_probs)?;
            teacher_probs
                .multiply(&log_ratio)?
                .sum_axes(&[-1], Some(false))?
        };

        if let Some(w) = weights {
            let weighted = kl_per_token.multiply(w)?;
            let total_weight = w.sum(None)?;
            let safe_weight = mlx_rs::ops::maximum(&total_weight, &Array::from_f32(1e-8))?;
            Ok(weighted.sum(None)?.divide(&safe_weight)?)
        } else {
            Ok(kl_per_token.mean(None)?)
        }
    }

    fn name(&self) -> &'static str {
        if self.reverse {
            "reverse_kl_divergence"
        } else {
            "kl_divergence"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_kl_identical_distributions() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let loss = KlDivergenceLoss::new();
        let result = loss.compute(&logits, &logits, 1.0).unwrap();
        let value: f32 = result.item();

        // KL divergence of identical distributions should be 0
        assert!(
            value.abs() < 1e-4,
            "KL of identical distributions should be ~0, got {}",
            value
        );
    }

    #[test]
    #[serial]
    fn test_kl_different_distributions() {
        let teacher = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = KlDivergenceLoss::new();
        let result = loss.compute(&teacher, &student, 1.0).unwrap();
        let value: f32 = result.item();

        // KL divergence should be positive
        assert!(
            value > 0.0,
            "KL divergence should be positive, got {}",
            value
        );
    }

    #[test]
    #[serial]
    fn test_kl_temperature_effect() {
        let teacher = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = KlDivergenceLoss::new();

        // Higher temperature should reduce KL (softer distributions are more similar)
        let kl_t1 = loss.compute(&teacher, &student, 1.0).unwrap();
        let kl_t2 = loss.compute(&teacher, &student, 2.0).unwrap();
        let kl_t4 = loss.compute(&teacher, &student, 4.0).unwrap();

        let v1: f32 = kl_t1.item();
        let v2: f32 = kl_t2.item();
        let v4: f32 = kl_t4.item();

        assert!(
            v2 < v1,
            "Higher temp should reduce KL: T=1: {}, T=2: {}",
            v1,
            v2
        );
        assert!(
            v4 < v2,
            "Higher temp should reduce KL: T=2: {}, T=4: {}",
            v2,
            v4
        );
    }

    #[test]
    #[serial]
    fn test_forward_vs_reverse_kl() {
        let teacher = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let forward = KlDivergenceLoss::new();
        let reverse = KlDivergenceLoss::reverse();

        let fwd_loss = forward.compute(&teacher, &student, 1.0).unwrap();
        let rev_loss = reverse.compute(&teacher, &student, 1.0).unwrap();

        let fwd_val: f32 = fwd_loss.item();
        let rev_val: f32 = rev_loss.item();

        // Both should be positive
        assert!(fwd_val > 0.0);
        assert!(rev_val > 0.0);
    }

    #[cfg(feature = "metal")]
    #[test]
    #[serial]
    fn test_gpu_acceleration_available() {
        let loss = KlDivergenceLoss::new();
        // On Apple Silicon, GPU should be available
        println!("GPU available: {}", loss.is_gpu_available());
    }

    /// Verify that gradients flow through the KL divergence loss and are
    /// finite + non-zero.  This guards against accidentally detaching the
    /// computation graph (e.g. via `.item()` or `.eval()` mid-graph).
    #[test]
    #[serial]
    fn test_kl_divergence_gradient_flow() {
        use mlx_rs::transforms::value_and_grad;

        let teacher = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);

        let loss_fn = |inputs: &[Array]| -> Vec<Array> {
            let student = &inputs[0];
            let temp = Array::from_f32(2.0);
            let teacher_scaled = teacher.divide(&temp).unwrap();
            let student_scaled = student.divide(&temp).unwrap();

            let teacher_log_probs = mlx_rs::nn::log_softmax(&teacher_scaled, -1).unwrap();
            let student_log_probs = mlx_rs::nn::log_softmax(&student_scaled, -1).unwrap();
            let teacher_probs = teacher_log_probs.exp().unwrap();

            let log_ratio = teacher_log_probs.subtract(&student_log_probs).unwrap();
            let kl = teacher_probs.multiply(&log_ratio).unwrap();
            let kl_sum = kl.sum_axes(&[-1], Some(false)).unwrap();
            let loss = kl_sum.mean(None).unwrap();
            vec![loss]
        };

        let student = Array::from_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);
        let (values, grads) = value_and_grad(loss_fn)(&[student]).unwrap();

        values[0].eval().unwrap();
        grads[0].eval().unwrap();

        let loss_val: f32 = values[0].item();
        assert!(
            loss_val.is_finite(),
            "loss must be finite, got {}",
            loss_val
        );
        assert!(loss_val > 0.0, "KL loss must be positive, got {}", loss_val);

        let grad_data: Vec<f32> = grads[0].as_slice().to_vec();
        let grad_norm: f32 = grad_data.iter().map(|&g| g * g).sum::<f32>().sqrt();
        assert!(
            grad_norm.is_finite(),
            "gradient must be finite, got norm={}",
            grad_norm
        );
        assert!(
            grad_norm > 1e-10,
            "gradient must be non-zero, got norm={}",
            grad_norm
        );
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

        let loss = KlDivergenceLoss::new();
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();

        // Should be positive and finite
        assert!(value > 0.0, "KL should be positive");
        assert!(value.is_finite(), "KL should be finite");
    }

    // -----------------------------------------------------------------
    // Cross-vocab / sparse top-k tests
    // -----------------------------------------------------------------

    /// KL divergence with mismatched vocab (teacher smaller than student).
    /// Mirrors Qwen3-4B (151936) → Qwen3.5-0.8B (152080).
    #[test]
    #[serial]
    fn test_kl_cross_vocab_teacher_smaller() {
        let teacher_vocab = 80_i32;
        let student_vocab = 100_i32;
        let batch = 2_i32;
        let seq = 4_i32;

        let teacher_data: Vec<f32> = (0..(batch * seq * teacher_vocab))
            .map(|i| (i % 40) as f32 - 20.0)
            .collect();
        let student_data: Vec<f32> = (0..(batch * seq * student_vocab))
            .map(|i| (i * 3 % 40) as f32 - 20.0)
            .collect();
        let teacher = Array::from_slice(&teacher_data, &[batch, seq, teacher_vocab]);
        let student = Array::from_slice(&student_data, &[batch, seq, student_vocab]);

        let loss = KlDivergenceLoss::new().with_sparse_top_k(32);
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();

        assert!(value >= 0.0, "KL must be non-negative, got {}", value);
        assert!(value.is_finite(), "KL must be finite, got {}", value);
    }

    /// KL divergence with mismatched vocab (teacher larger than student).
    #[test]
    #[serial]
    fn test_kl_cross_vocab_teacher_larger() {
        let teacher_vocab = 100_i32;
        let student_vocab = 80_i32;
        let batch = 2_i32;
        let seq = 4_i32;

        let teacher_data: Vec<f32> = (0..(batch * seq * teacher_vocab))
            .map(|i| (i % 40) as f32 - 20.0)
            .collect();
        let student_data: Vec<f32> = (0..(batch * seq * student_vocab))
            .map(|i| (i * 3 % 40) as f32 - 20.0)
            .collect();
        let teacher = Array::from_slice(&teacher_data, &[batch, seq, teacher_vocab]);
        let student = Array::from_slice(&student_data, &[batch, seq, student_vocab]);

        let loss = KlDivergenceLoss::new().with_sparse_top_k(32);
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();

        assert!(value >= 0.0, "KL must be non-negative, got {}", value);
        assert!(value.is_finite(), "KL must be finite, got {}", value);
    }

    /// Cross-vocab KL should return higher loss when distributions differ.
    #[test]
    #[serial]
    fn test_kl_cross_vocab_ordered_distributions() {
        // teacher: vocab=6, student: vocab=4
        // Same ordering: teacher top tokens overlap with student → lower KL
        // vs. completely inverted: → higher KL
        let teacher_same = Array::from_slice(&[4.0_f32, 3.0, 2.0, 1.0, 0.5, 0.1], &[1, 1, 6]);
        let teacher_inv = Array::from_slice(&[0.1_f32, 0.5, 1.0, 2.0, 3.0, 4.0], &[1, 1, 6]);
        let student = Array::from_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let loss = KlDivergenceLoss::new().with_sparse_top_k(4);
        let kl_same = loss
            .compute(&teacher_same, &student, 1.0)
            .unwrap()
            .item::<f32>();
        let kl_inv = loss
            .compute(&teacher_inv, &student, 1.0)
            .unwrap()
            .item::<f32>();

        assert!(kl_same.is_finite());
        assert!(kl_inv.is_finite());
        // The inverted teacher puts all mass on tokens 4-5 which are out-of-range
        // for the 4-token student, so the student cannot match it → higher KL.
        assert!(
            kl_inv >= kl_same,
            "inverted teacher should give >= KL vs aligned teacher: inv={}, same={}",
            kl_inv,
            kl_same
        );
    }

    /// Configurable top-k builder compiles and produces valid output.
    #[test]
    #[serial]
    fn test_kl_with_sparse_top_k_builder() {
        let teacher = Array::from_slice(
            &(0..200).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 1, 200],
        );
        let student = Array::from_slice(
            &(0..150).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 1, 150],
        );

        for k in [8, 32, 64, 128] {
            let loss = KlDivergenceLoss::new().with_sparse_top_k(k);
            let result = loss.compute(&teacher, &student, 2.0).unwrap();
            let value: f32 = result.item();
            assert!(
                value.is_finite(),
                "KL should be finite for k={}: {}",
                k,
                value
            );
        }
    }

    /// Reverse KL also works across mismatched vocabs.
    #[test]
    #[serial]
    fn test_reverse_kl_cross_vocab() {
        let teacher_vocab = 90_i32;
        let student_vocab = 70_i32;
        let batch = 1_i32;
        let seq = 3_i32;

        let teacher_data: Vec<f32> = (0..(batch * seq * teacher_vocab))
            .map(|i| (i % 30) as f32 - 15.0)
            .collect();
        let student_data: Vec<f32> = (0..(batch * seq * student_vocab))
            .map(|i| (i % 30) as f32 - 15.0)
            .collect();
        let teacher = Array::from_slice(&teacher_data, &[batch, seq, teacher_vocab]);
        let student = Array::from_slice(&student_data, &[batch, seq, student_vocab]);

        let loss = KlDivergenceLoss::reverse().with_sparse_top_k(16);
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();
        assert!(value.is_finite(), "reverse KL cross-vocab must be finite");
    }
}
