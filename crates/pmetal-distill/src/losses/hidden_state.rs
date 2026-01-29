//! Hidden state alignment losses for knowledge distillation.
//!
//! GPU-first implementation using Metal kernels for optimal performance on Apple Silicon.
//! Falls back to MLX operations when Metal is unavailable.
//!
//! These losses enable distilling knowledge from intermediate layers,
//! not just the final logits. This can improve student model quality
//! by ensuring internal representations align with the teacher.
//!
//! # Zero-Copy Optimization
//!
//! On Apple Silicon, MLX and Metal share unified memory. This implementation uses
//! zero-copy bridging to pass MLX array data directly to Metal kernels without
//! copying, providing significant performance improvements for large tensors.

use crate::{HiddenStateLossType, Result};
use mlx_rs::Array;

#[cfg(feature = "metal")]
use std::sync::Arc;

#[cfg(feature = "metal")]
use pmetal_metal::{
    bridge::metal_buffer_from_ptr,
    context::MetalContext,
    kernels::{
        FusedHiddenAlign, HiddenAlignConfig, HiddenAlignLossType as MetalHiddenAlignLossType,
    },
};

/// Hidden state alignment loss.
///
/// Aligns hidden states between teacher and student layers.
/// Supports MSE, cosine similarity, and L1 losses.
///
/// # GPU Acceleration
///
/// When the `metal` feature is enabled (default), this implementation uses
/// custom Metal kernels for MSE and cosine similarity losses.
pub struct HiddenStateLoss {
    /// Type of loss to use.
    loss_type: HiddenStateLossType,
    /// Optional projection matrix for dimension mismatch.
    projection: Option<Array>,

    /// Cached Metal context for GPU acceleration.
    #[cfg(feature = "metal")]
    ctx: Option<Arc<MetalContext>>,
}

impl HiddenStateLoss {
    /// Create a new hidden state loss.
    pub fn new(loss_type: HiddenStateLossType) -> Self {
        Self {
            loss_type,
            projection: None,
            #[cfg(feature = "metal")]
            ctx: MetalContext::global().ok(),
        }
    }

    /// Create MSE hidden state loss.
    pub fn mse() -> Self {
        Self::new(HiddenStateLossType::Mse)
    }

    /// Create cosine similarity hidden state loss.
    pub fn cosine() -> Self {
        Self::new(HiddenStateLossType::Cosine)
    }

    /// Create L1 hidden state loss.
    pub fn l1() -> Self {
        Self::new(HiddenStateLossType::L1)
    }

    /// Set projection matrix for dimension mismatch.
    ///
    /// If teacher and student have different hidden dimensions,
    /// use a learned projection to align them.
    pub fn with_projection(mut self, projection: Array) -> Self {
        self.projection = Some(projection);
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

    /// Compute alignment loss between teacher and student hidden states.
    ///
    /// # Arguments
    /// * `teacher_hidden` - Hidden states from teacher [batch, seq, hidden_teacher]
    /// * `student_hidden` - Hidden states from student [batch, seq, hidden_student]
    pub fn compute(&self, teacher_hidden: &Array, student_hidden: &Array) -> Result<Array> {
        // Apply projection if dimensions don't match
        let student_aligned = if let Some(proj) = &self.projection {
            // student_hidden @ projection -> [batch, seq, hidden_teacher]
            student_hidden.matmul(proj)?
        } else {
            student_hidden.clone()
        };

        // GPU-first: try Metal for supported loss types
        #[cfg(feature = "metal")]
        {
            if self.ctx.is_some() {
                match self.loss_type {
                    HiddenStateLossType::Mse => {
                        return self.compute_gpu(
                            teacher_hidden,
                            &student_aligned,
                            MetalHiddenAlignLossType::Mse,
                        );
                    }
                    HiddenStateLossType::Cosine => {
                        return self.compute_gpu(
                            teacher_hidden,
                            &student_aligned,
                            MetalHiddenAlignLossType::Cosine,
                        );
                    }
                    HiddenStateLossType::L1 => {
                        // L1 not implemented in Metal, fall through to MLX
                    }
                }
            }
        }

        // MLX fallback
        match self.loss_type {
            HiddenStateLossType::Mse => self.mse_loss_mlx(teacher_hidden, &student_aligned),
            HiddenStateLossType::Cosine => self.cosine_loss_mlx(teacher_hidden, &student_aligned),
            HiddenStateLossType::L1 => self.l1_loss_mlx(teacher_hidden, &student_aligned),
        }
    }

    /// GPU-accelerated forward pass using Metal kernels with zero-copy bridging.
    ///
    /// Uses zero-copy bridging to pass MLX array data directly to Metal kernels
    /// without copying. This is possible because MLX and Metal share unified
    /// memory on Apple Silicon.
    #[cfg(feature = "metal")]
    fn compute_gpu(
        &self,
        teacher_hidden: &Array,
        student_hidden: &Array,
        loss_type: MetalHiddenAlignLossType,
    ) -> Result<Array> {
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| crate::DistillError::Metal("Metal context not available".to_string()))?;

        let t_shape = teacher_hidden.shape();
        let s_shape = student_hidden.shape();

        if t_shape.len() < 2 || s_shape.len() < 2 {
            return Err(crate::DistillError::Other(
                "Hidden states must have at least 2 dimensions".to_string(),
            ));
        }

        // Get dimensions
        let teacher_dim = t_shape[t_shape.len() - 1] as usize;
        let student_dim = s_shape[s_shape.len() - 1] as usize;
        let num_tokens: usize = t_shape[..t_shape.len() - 1]
            .iter()
            .map(|&d| d as usize)
            .product();
        let teacher_elements = num_tokens * teacher_dim;
        let student_elements = num_tokens * student_dim;

        // Flatten to [num_tokens, hidden] for Metal kernel
        let teacher_flat = teacher_hidden.reshape(&[-1, teacher_dim as i32])?;
        let student_flat = student_hidden.reshape(&[-1, student_dim as i32])?;

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
        // 4. teacher_elements/student_elements correctly represent the array sizes
        let teacher_view = unsafe {
            metal_buffer_from_ptr(ctx, teacher_ptr, teacher_elements)
                .map_err(|e| crate::DistillError::Metal(format!("Buffer view error: {}", e)))?
        };
        let student_view = unsafe {
            metal_buffer_from_ptr(ctx, student_ptr, student_elements)
                .map_err(|e| crate::DistillError::Metal(format!("Buffer view error: {}", e)))?
        };

        // Configure kernel
        let config = HiddenAlignConfig::new(num_tokens, teacher_dim, student_dim);

        let kernel = FusedHiddenAlign::new(ctx.clone(), config)
            .map_err(|e| crate::DistillError::Metal(format!("Kernel error: {}", e)))?;

        // Execute kernel with zero-copy buffer views
        let losses = kernel
            .forward(&teacher_view, &student_view, loss_type)
            .map_err(|e| crate::DistillError::Metal(format!("Execution error: {}", e)))?;

        // Compute mean loss
        let loss_data = losses.as_slice();
        let mean_loss: f32 = loss_data.iter().sum::<f32>() / loss_data.len() as f32;

        Ok(Array::from_f32(mean_loss))
    }

    /// MSE loss on hidden states (MLX fallback).
    fn mse_loss_mlx(&self, teacher: &Array, student: &Array) -> Result<Array> {
        let diff = student.subtract(teacher)?;
        let squared = diff.multiply(&diff)?;
        Ok(squared.mean(None)?)
    }

    /// Cosine similarity loss (1 - cosine_similarity) (MLX fallback).
    fn cosine_loss_mlx(&self, teacher: &Array, student: &Array) -> Result<Array> {
        // Cosine similarity along last dimension
        // cos(t, s) = (t · s) / (||t|| * ||s||)

        // Dot product
        let dot = teacher.multiply(student)?.sum_axes(&[-1], Some(true))?;

        // Norms
        let teacher_norm = teacher
            .multiply(teacher)?
            .sum_axes(&[-1], Some(true))?
            .sqrt()?;
        let student_norm = student
            .multiply(student)?
            .sum_axes(&[-1], Some(true))?
            .sqrt()?;

        // Cosine similarity with epsilon for stability
        let eps = Array::from_f32(1e-8);
        let norms = teacher_norm.multiply(&student_norm)?.add(&eps)?;
        let cosine_sim = dot.divide(&norms)?;

        // Loss = 1 - cosine_similarity
        let one = Array::from_f32(1.0);
        let loss = one.subtract(&cosine_sim)?;

        Ok(loss.mean(None)?)
    }

    /// L1 loss on hidden states (MLX only - no Metal kernel).
    fn l1_loss_mlx(&self, teacher: &Array, student: &Array) -> Result<Array> {
        let diff = student.subtract(teacher)?;
        let abs_diff = diff.abs()?;
        Ok(abs_diff.mean(None)?)
    }

    /// Get the name of this loss.
    pub fn name(&self) -> &'static str {
        match self.loss_type {
            HiddenStateLossType::Mse => "hidden_mse",
            HiddenStateLossType::Cosine => "hidden_cosine",
            HiddenStateLossType::L1 => "hidden_l1",
        }
    }
}

/// Configuration for multi-layer hidden state distillation.
pub struct LayerDistillation {
    /// Mapping from teacher layer to student layer.
    layer_mapping: Vec<(usize, usize)>,
    /// Loss function for each layer pair.
    loss: HiddenStateLoss,
    /// Weight for the total hidden state loss.
    weight: f32,
}

impl LayerDistillation {
    /// Create a new layer distillation config.
    pub fn new(layer_mapping: Vec<(usize, usize)>, loss: HiddenStateLoss, weight: f32) -> Self {
        Self {
            layer_mapping,
            loss,
            weight,
        }
    }

    /// Create a linear layer mapping (evenly distributed).
    ///
    /// Maps student layers to corresponding teacher layers proportionally.
    pub fn linear_mapping(teacher_layers: usize, student_layers: usize) -> Vec<(usize, usize)> {
        (0..student_layers)
            .map(|s| {
                let t = (s * teacher_layers) / student_layers;
                (t, s)
            })
            .collect()
    }

    /// Create a skip layer mapping (every Nth teacher layer).
    pub fn skip_mapping(
        teacher_layers: usize,
        student_layers: usize,
        skip: usize,
    ) -> Vec<(usize, usize)> {
        (0..student_layers)
            .filter_map(|s| {
                let t = s * skip;
                if t < teacher_layers {
                    Some((t, s))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Compute total hidden state loss across all layer pairs.
    pub fn compute(&self, teacher_hiddens: &[Array], student_hiddens: &[Array]) -> Result<Array> {
        if self.layer_mapping.is_empty() {
            return Ok(Array::from_f32(0.0));
        }

        let mut total_loss = Array::from_f32(0.0);
        let mut count = 0;

        for (teacher_idx, student_idx) in &self.layer_mapping {
            if *teacher_idx < teacher_hiddens.len() && *student_idx < student_hiddens.len() {
                let loss = self.loss.compute(
                    &teacher_hiddens[*teacher_idx],
                    &student_hiddens[*student_idx],
                )?;
                total_loss = total_loss.add(&loss)?;
                count += 1;
            }
        }

        if count > 0 {
            let avg_loss = total_loss.divide(&Array::from_f32(count as f32))?;
            Ok(avg_loss.multiply(&Array::from_f32(self.weight))?)
        } else {
            Ok(Array::from_f32(0.0))
        }
    }

    /// Get the layer mapping.
    pub fn mapping(&self) -> &[(usize, usize)] {
        &self.layer_mapping
    }

    /// Get the weight.
    pub fn weight(&self) -> f32 {
        self.weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_mse_hidden_loss() {
        let teacher = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let student = Array::from_slice(&[2.0_f32, 3.0, 4.0, 5.0], &[1, 2, 2]);

        let loss = HiddenStateLoss::mse();
        let result = loss.compute(&teacher, &student).unwrap();
        let value: f32 = result.item();

        // Each element differs by 1, MSE = 1
        assert!(
            (value - 1.0).abs() < 1e-4,
            "MSE should be 1.0, got {}",
            value
        );
    }

    #[test]
    fn test_cosine_identical_vectors() {
        let hidden = Array::from_slice(&[1.0_f32, 0.0, 0.0, 1.0], &[1, 2, 2]);

        let loss = HiddenStateLoss::cosine();
        let result = loss.compute(&hidden, &hidden).unwrap();
        let value: f32 = result.item();

        // Cosine loss of identical vectors should be 0 (cosine sim = 1)
        assert!(
            value.abs() < 1e-4,
            "Cosine loss of identical vectors should be ~0, got {}",
            value
        );
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        // Orthogonal vectors: [1, 0] and [0, 1]
        let teacher = Array::from_slice(&[1.0_f32, 0.0], &[1, 1, 2]);
        let student = Array::from_slice(&[0.0_f32, 1.0], &[1, 1, 2]);

        let loss = HiddenStateLoss::cosine();
        let result = loss.compute(&teacher, &student).unwrap();
        let value: f32 = result.item();

        // Cosine loss of orthogonal vectors should be 1 (cosine sim = 0)
        assert!(
            (value - 1.0).abs() < 1e-4,
            "Cosine loss of orthogonal vectors should be ~1, got {}",
            value
        );
    }

    #[test]
    fn test_l1_hidden_loss() {
        let teacher = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let student = Array::from_slice(&[2.0_f32, 3.0, 4.0, 5.0], &[1, 2, 2]);

        let loss = HiddenStateLoss::l1();
        let result = loss.compute(&teacher, &student).unwrap();
        let value: f32 = result.item();

        // Each element differs by 1, L1 = 1
        assert!(
            (value - 1.0).abs() < 1e-4,
            "L1 should be 1.0, got {}",
            value
        );
    }

    #[test]
    fn test_linear_mapping() {
        // 12 teacher layers -> 4 student layers
        let mapping = LayerDistillation::linear_mapping(12, 4);
        assert_eq!(mapping.len(), 4);
        // Should map: (0, 0), (3, 1), (6, 2), (9, 3)
        assert_eq!(mapping[0], (0, 0));
        assert_eq!(mapping[1], (3, 1));
        assert_eq!(mapping[2], (6, 2));
        assert_eq!(mapping[3], (9, 3));
    }

    #[test]
    fn test_layer_distillation_compute() {
        let teacher_hiddens: Vec<Array> = (0..4)
            .map(|_| Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 2, 2]))
            .collect();
        let student_hiddens: Vec<Array> = (0..2)
            .map(|_| Array::from_slice(&[2.0_f32, 3.0, 4.0, 5.0], &[1, 2, 2]))
            .collect();

        let mapping = vec![(0, 0), (2, 1)]; // Map teacher 0 -> student 0, teacher 2 -> student 1
        let distill = LayerDistillation::new(mapping, HiddenStateLoss::mse(), 0.5);

        let result = distill.compute(&teacher_hiddens, &student_hiddens).unwrap();
        let value: f32 = result.item();

        // MSE = 1.0 for each pair, weight = 0.5
        assert!(
            (value - 0.5).abs() < 1e-4,
            "Weighted hidden loss should be 0.5, got {}",
            value
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    #[serial]
    fn test_gpu_acceleration_available() {
        let loss = HiddenStateLoss::mse();
        println!("GPU available: {}", loss.is_gpu_available());
    }

    #[test]
    #[serial]
    fn test_larger_batch() {
        // Test with larger tensors to exercise GPU path
        let batch_size = 4;
        let seq_len = 8;
        let hidden_dim = 256;

        let teacher_data: Vec<f32> = (0..(batch_size * seq_len * hidden_dim))
            .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
            .collect();
        let student_data: Vec<f32> = (0..(batch_size * seq_len * hidden_dim))
            .map(|i| ((i * 7 % 100) as f32 - 50.0) / 100.0)
            .collect();

        let teacher = Array::from_slice(
            &teacher_data,
            &[batch_size as i32, seq_len as i32, hidden_dim as i32],
        );
        let student = Array::from_slice(
            &student_data,
            &[batch_size as i32, seq_len as i32, hidden_dim as i32],
        );

        let loss = HiddenStateLoss::mse();
        let result = loss.compute(&teacher, &student).unwrap();
        let value: f32 = result.item();

        // Should be positive and finite
        assert!(value >= 0.0, "MSE should be non-negative");
        assert!(value.is_finite(), "MSE should be finite");
    }
}
