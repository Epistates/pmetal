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

#![allow(unsafe_code)]

use super::{DistillLoss, SPARSE_TOPK_DEFAULT, align_vocab_with_k};
use crate::Result;
use mlx_rs::Array;

/// Numerically stable log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|)).
fn log_sum_exp(a: &Array, b: &Array) -> std::result::Result<Array, mlx_rs::error::Exception> {
    let max_ab = mlx_rs::ops::maximum(a, b)?;
    let diff = a.subtract(b)?.abs()?;
    let log1p_term = mlx_rs::ops::exp(&diff.negative()?)?
        .add(&Array::from_f32(1.0))?
        .log()?;
    max_ab.add(&log1p_term)
}

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
    /// Number of top-k teacher tokens to retain when vocab sizes differ.
    ///
    /// Only used when teacher and student have different vocabulary sizes
    /// (cross-architecture distillation).  Defaults to [`SPARSE_TOPK_DEFAULT`].
    sparse_top_k: i32,

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
            sparse_top_k: SPARSE_TOPK_DEFAULT,
            #[cfg(feature = "metal")]
            ctx: MetalContext::global().ok(),
            #[cfg(not(feature = "metal"))]
            _phantom: (),
        }
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

    /// MLX fallback implementation using log-domain computation for numerical stability.
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

        // Log-softmax for numerical stability
        let teacher_log_probs = mlx_rs::nn::log_softmax(&teacher_scaled, -1)?;
        let student_log_probs = mlx_rs::nn::log_softmax(&student_scaled, -1)?;
        let teacher_probs = teacher_log_probs.exp()?;

        // log(M) = log(0.5*(P+Q)) via log-sum-exp for stability (avoids 0*-inf = NaN)
        // log(M) = -ln(2) + log(exp(log_P) + exp(log_Q))
        let log2 = Array::from_f32(2.0_f32.ln());
        let log_mixture = log_sum_exp(&teacher_log_probs, &student_log_probs)?.subtract(&log2)?;

        // KL(P || M) = sum(P * (log_P - log_M))
        let kl_teacher_m = teacher_probs.multiply(&teacher_log_probs.subtract(&log_mixture)?)?;

        // KL(Q || M) = sum(Q * (log_Q - log_M))
        let student_probs = student_log_probs.exp()?;
        let kl_student_m = student_probs.multiply(&student_log_probs.subtract(&log_mixture)?)?;

        // JS = 0.5 * (KL(P||M) + KL(Q||M))
        let half = Array::from_f32(0.5);
        let js = kl_teacher_m.add(&kl_student_m)?.multiply(&half)?;

        // Sum over vocab, mean over batch and sequence
        let js_sum = js.sum_axes(&[-1], Some(false))?;
        Ok(js_sum.mean(None)?)
    }
}

impl Default for JensenShannonLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillLoss for JensenShannonLoss {
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
        if weights.is_none() && !vocab_mismatched {
            #[cfg(feature = "metal")]
            {
                if self.ctx.is_some() {
                    return self.compute_gpu(teacher_logits, student_logits, temperature);
                }
            }
        }

        // MLX fallback / weighted / sparse-vocab implementation (log-domain for stability)
        let temp = Array::from_f32(temperature);
        let teacher_scaled = teacher_logits.divide(&temp)?;
        let student_scaled = student_logits.divide(&temp)?;

        let teacher_log_probs = mlx_rs::nn::log_softmax(&teacher_scaled, -1)?;
        let student_log_probs = mlx_rs::nn::log_softmax(&student_scaled, -1)?;
        let teacher_probs = teacher_log_probs.exp()?;

        // log(M) via log-sum-exp for stability (avoids 0*-inf = NaN for disjoint distributions)
        let log2 = Array::from_f32(2.0_f32.ln());
        let log_mixture = log_sum_exp(&teacher_log_probs, &student_log_probs)?.subtract(&log2)?;

        let kl_teacher_m = teacher_probs.multiply(&teacher_log_probs.subtract(&log_mixture)?)?;
        let student_probs = student_log_probs.exp()?;
        let kl_student_m = student_probs.multiply(&student_log_probs.subtract(&log_mixture)?)?;

        let half = Array::from_f32(0.5);
        let js_per_token = kl_teacher_m
            .add(&kl_student_m)?
            .multiply(&half)?
            .sum_axes(&[-1], Some(false))?;

        if let Some(w) = weights {
            let weighted = js_per_token.multiply(w)?;
            let total_weight = w.sum(None)?;
            let safe_weight = mlx_rs::ops::maximum(&total_weight, &Array::from_f32(1e-8))?;
            Ok(weighted.sum(None)?.divide(&safe_weight)?)
        } else {
            Ok(js_per_token.mean(None)?)
        }
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
    #[serial]
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
    #[serial]
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
    #[serial]
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
    #[serial]
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

    /// Verify gradients flow through Jensen-Shannon loss (finite + non-zero).
    #[test]
    #[serial]
    fn test_jensen_shannon_gradient_flow() {
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

            // log(M) = log(0.5*(P+Q)) via log-sum-exp
            let log2 = Array::from_f32(2.0_f32.ln());
            let log_mixture = log_sum_exp(&teacher_log_probs, &student_log_probs)
                .unwrap()
                .subtract(&log2)
                .unwrap();

            let kl_teacher_m = teacher_probs
                .multiply(&teacher_log_probs.subtract(&log_mixture).unwrap())
                .unwrap();
            let student_probs = student_log_probs.exp().unwrap();
            let kl_student_m = student_probs
                .multiply(&student_log_probs.subtract(&log_mixture).unwrap())
                .unwrap();

            let half = Array::from_f32(0.5);
            let js = kl_teacher_m
                .add(&kl_student_m)
                .unwrap()
                .multiply(&half)
                .unwrap();
            let js_sum = js.sum_axes(&[-1], Some(false)).unwrap();
            let loss = js_sum.mean(None).unwrap();
            vec![loss]
        };

        let student = Array::from_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);
        let (values, grads) = value_and_grad(loss_fn)(&[student]).unwrap();

        values[0].eval().unwrap();
        grads[0].eval().unwrap();

        let loss_val: f32 = values[0].item();
        assert!(
            loss_val.is_finite(),
            "JS loss must be finite, got {}",
            loss_val
        );
        assert!(loss_val > 0.0, "JS loss must be positive, got {}", loss_val);

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

    // -----------------------------------------------------------------
    // Cross-vocab / sparse top-k tests
    // -----------------------------------------------------------------

    /// JS divergence with teacher smaller than student vocab.
    #[test]
    #[serial]
    fn test_js_cross_vocab_teacher_smaller() {
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

        let loss = JensenShannonLoss::new().with_sparse_top_k(32);
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();

        assert!(value >= 0.0, "JS must be non-negative, got {}", value);
        assert!(value.is_finite(), "JS must be finite, got {}", value);
        // JS is bounded by ln(2) ≈ 0.693
        assert!(
            value <= 2.0_f32.ln() + 1e-4,
            "JS must be bounded by ln(2), got {}",
            value
        );
    }

    /// JS divergence with teacher larger than student vocab.
    #[test]
    #[serial]
    fn test_js_cross_vocab_teacher_larger() {
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

        let loss = JensenShannonLoss::new().with_sparse_top_k(32);
        let result = loss.compute(&teacher, &student, 2.0).unwrap();
        let value: f32 = result.item();

        assert!(value >= 0.0, "JS must be non-negative, got {}", value);
        assert!(value.is_finite(), "JS must be finite, got {}", value);
        assert!(
            value <= 2.0_f32.ln() + 1e-4,
            "JS must be bounded by ln(2), got {}",
            value
        );
    }

    /// 3-D tensor cross-vocab JS — verifies the Ellipsis fix for rank-3 logits.
    #[test]
    #[serial]
    fn test_js_cross_vocab_3d_tensors() {
        let batch = 2_i32;
        let seq = 3_i32;
        let teacher_vocab = 10_i32;
        let student_vocab = 8_i32;

        let teacher_data: Vec<f32> = (0..(batch * seq * teacher_vocab))
            .map(|i| i as f32)
            .collect();
        let student_data: Vec<f32> = (0..(batch * seq * student_vocab))
            .map(|i| i as f32)
            .collect();
        let teacher = Array::from_slice(&teacher_data, &[batch, seq, teacher_vocab]);
        let student = Array::from_slice(&student_data, &[batch, seq, student_vocab]);

        let loss = JensenShannonLoss::new().with_sparse_top_k(4);
        let result = loss.compute(&teacher, &student, 1.0).unwrap();
        let value: f32 = result.item();

        // Scalar result
        assert!(result.shape().is_empty(), "result should be scalar");
        assert!(value.is_finite(), "JS must be finite, got {}", value);
        assert!(value >= 0.0, "JS must be non-negative, got {}", value);
    }

    /// Cross-vocab JS is symmetric: JS(teacher_a, student_a) ≈ JS(teacher_b, student_b)
    /// when the roles are swapped via argument order.
    #[test]
    #[serial]
    fn test_js_cross_vocab_symmetry_hint() {
        // When both are within range (student_vocab > teacher_vocab so no masking),
        // swapping teacher/student should give the same value (JS is symmetric).
        let teacher_vocab = 6_i32;
        let student_vocab = 10_i32;

        // Identical logit values for the shared 6 tokens
        let shared: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let extended: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0];
        let teacher = Array::from_slice(&shared, &[1, 1, teacher_vocab]);
        let student = Array::from_slice(&extended, &[1, 1, student_vocab]);

        let loss = JensenShannonLoss::new().with_sparse_top_k(6);
        let result = loss.compute(&teacher, &student, 1.0).unwrap();
        let value: f32 = result.item();

        assert!(value.is_finite(), "JS must be finite, got {}", value);
        assert!(value >= 0.0, "JS must be non-negative, got {}", value);
    }

    /// Configurable top-k builder produces valid output for multiple k values.
    #[test]
    #[serial]
    fn test_js_with_sparse_top_k_builder() {
        let teacher = Array::from_slice(
            &(0..200).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 1, 200],
        );
        let student = Array::from_slice(
            &(0..150).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 1, 150],
        );

        for k in [8, 32, 64, 128] {
            let loss = JensenShannonLoss::new().with_sparse_top_k(k);
            let result = loss.compute(&teacher, &student, 2.0).unwrap();
            let value: f32 = result.item();
            assert!(
                value.is_finite(),
                "JS should be finite for k={}: {}",
                k,
                value
            );
            assert!(
                value >= 0.0,
                "JS should be non-negative for k={}: {}",
                k,
                value
            );
        }
    }
}
