//! Hidden state alignment losses for knowledge distillation.
//!
//! These losses enable distilling knowledge from intermediate layers,
//! not just the final logits. This can improve student model quality
//! by ensuring internal representations align with the teacher.

use crate::{HiddenStateLossType, Result};
use pmetal_bridge::compat::Array;

/// Hidden state alignment loss.
///
/// Aligns hidden states between teacher and student layers.
/// Supports MSE, cosine similarity, and L1 losses.
pub struct HiddenStateLoss {
    /// Type of loss to use.
    loss_type: HiddenStateLossType,
    /// Optional projection matrix for dimension mismatch.
    projection: Option<Array>,
}

impl HiddenStateLoss {
    /// Create a new hidden state loss.
    pub fn new(loss_type: HiddenStateLossType) -> Self {
        Self {
            loss_type,
            projection: None,
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

    /// Compute alignment loss between teacher and student hidden states.
    ///
    /// # Arguments
    /// * `teacher_hidden` - Hidden states from teacher [batch, seq, hidden_teacher]
    /// * `student_hidden` - Hidden states from student [batch, seq, hidden_student]
    pub fn compute(&self, teacher_hidden: &Array, student_hidden: &Array) -> Result<Array> {
        // Apply projection if dimensions don't match
        let student_aligned = if let Some(proj) = &self.projection {
            // Validate the matmul shape contract before reaching MLX so the
            // failure surfaces with a precise message instead of an opaque
            // backend error.
            //
            //   student_hidden  : [..., hidden_student]
            //   projection      : [hidden_student, hidden_teacher]
            //   result          : [..., hidden_teacher]
            let s_last = student_hidden.dim(-1);
            let p_first = proj.dim(-2);
            if s_last != p_first {
                return Err(crate::DistillError::InvalidConfig(format!(
                    "hidden-state projection shape mismatch: student.dim(-1) = {} but \
                     projection.dim(-2) = {}; projection must be [hidden_student, hidden_teacher]",
                    s_last, p_first
                )));
            }
            student_hidden.matmul(proj)
        } else {
            // Without a projection the shapes must already align on the
            // hidden axis; otherwise the loss would fail with a
            // shape-broadcast error.
            let t_last = teacher_hidden.dim(-1);
            let s_last = student_hidden.dim(-1);
            if t_last != s_last {
                return Err(crate::DistillError::InvalidConfig(format!(
                    "hidden-state alignment requires matching last dim or a projection: \
                     teacher.dim(-1) = {}, student.dim(-1) = {}",
                    t_last, s_last
                )));
            }
            student_hidden.clone()
        };

        match self.loss_type {
            HiddenStateLossType::Mse => self.mse_loss_mlx(teacher_hidden, &student_aligned),
            HiddenStateLossType::Cosine => self.cosine_loss_mlx(teacher_hidden, &student_aligned),
            HiddenStateLossType::L1 => self.l1_loss_mlx(teacher_hidden, &student_aligned),
        }
    }

    /// MSE loss on hidden states.
    fn mse_loss_mlx(&self, teacher: &Array, student: &Array) -> Result<Array> {
        let diff = student.subtract(teacher);
        let squared = diff.multiply(&diff);
        Ok(squared.mean_all())
    }

    /// Cosine similarity loss (1 - cosine_similarity).
    fn cosine_loss_mlx(&self, teacher: &Array, student: &Array) -> Result<Array> {
        // Cosine similarity along last dimension
        // cos(t, s) = (t · s) / (||t|| * ||s||)

        // Dot product
        let dot = teacher.multiply(student).sum_axes(&[-1], true);

        // Norms
        let teacher_norm = teacher.multiply(teacher).sum_axes(&[-1], true).sqrt();
        let student_norm = student.multiply(student).sum_axes(&[-1], true).sqrt();

        // Cosine similarity with epsilon for stability
        let eps = Array::from_f32(1e-8);
        let norms = teacher_norm.multiply(&student_norm).add(&eps);
        let cosine_sim = dot.divide(&norms);

        // Loss = 1 - cosine_similarity
        let one = Array::from_f32(1.0);
        let loss = one.subtract(&cosine_sim);

        Ok(loss.mean_all())
    }

    /// L1 loss on hidden states.
    fn l1_loss_mlx(&self, teacher: &Array, student: &Array) -> Result<Array> {
        let diff = student.subtract(teacher);
        let abs_diff = diff.abs_val();
        Ok(abs_diff.mean_all())
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
    /// Relative per-layer weights (must be the same length as `layer_mapping`).
    /// These are normalized internally: the final loss is a weighted average over
    /// in-bounds pairs multiplied by `global_weight`.
    weights: Vec<f32>,
    /// Global scalar applied after the weighted average.
    global_weight: f32,
}

impl LayerDistillation {
    /// Create a new layer distillation config with a uniform weight for all pairs.
    ///
    /// The `weight` argument is a global multiplier applied to the mean loss
    /// across all layer pairs (backwards-compatible with the original API).
    pub fn new(layer_mapping: Vec<(usize, usize)>, loss: HiddenStateLoss, weight: f32) -> Self {
        let n = layer_mapping.len();
        Self {
            weights: vec![1.0; n],
            layer_mapping,
            loss,
            global_weight: weight,
        }
    }

    /// Create a layer distillation config with per-layer relative weights.
    ///
    /// Each pair's contribution is `weight[i] * loss[i]` divided by the sum of
    /// the in-bounds weights. The result is further scaled by `global_weight`.
    /// `weights` must have the same length as `layer_mapping`.
    pub fn new_with_weights(
        layer_mapping: Vec<(usize, usize)>,
        loss: HiddenStateLoss,
        weights: Vec<f32>,
        global_weight: f32,
    ) -> Self {
        assert_eq!(
            layer_mapping.len(),
            weights.len(),
            "layer_mapping and weights must have the same length"
        );
        Self {
            layer_mapping,
            loss,
            weights,
            global_weight,
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
    ///
    /// Each pair is weighted by its corresponding entry in `self.weights`.
    /// The result is the sum of weighted losses divided by the total weight of
    /// in-bounds pairs (so out-of-bounds pairs do not dilute the signal).
    pub fn compute(&self, teacher_hiddens: &[Array], student_hiddens: &[Array]) -> Result<Array> {
        if self.layer_mapping.is_empty() {
            return Ok(Array::from_f32(0.0));
        }

        let mut total_loss = Array::from_f32(0.0);
        let mut weight_total = 0.0_f32;

        for ((teacher_idx, student_idx), &w) in self.layer_mapping.iter().zip(self.weights.iter()) {
            if *teacher_idx < teacher_hiddens.len() && *student_idx < student_hiddens.len() {
                let loss = self.loss.compute(
                    &teacher_hiddens[*teacher_idx],
                    &student_hiddens[*student_idx],
                )?;
                total_loss = total_loss.add(&loss.multiply(&Array::from_f32(w)));
                weight_total += w;
            } else {
                tracing::warn!(
                    teacher_idx,
                    student_idx,
                    teacher_layers = teacher_hiddens.len(),
                    student_layers = student_hiddens.len(),
                    "layer pair ({}, {}) is out of bounds and will be skipped",
                    teacher_idx,
                    student_idx,
                );
            }
        }

        if weight_total > 0.0 {
            let avg = total_loss.divide(&Array::from_f32(weight_total));
            Ok(avg.multiply(&Array::from_f32(self.global_weight)))
        } else {
            Ok(Array::from_f32(0.0))
        }
    }

    /// Get the layer mapping.
    pub fn mapping(&self) -> &[(usize, usize)] {
        &self.layer_mapping
    }

    /// Get the per-layer relative weights.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Get the global weight multiplier.
    pub fn global_weight(&self) -> f32 {
        self.global_weight
    }
}

// SAFETY: `HiddenStateLoss` is constructed once and its `projection` Array (if any)
// is thereafter immutable. MLX arrays on Apple Silicon use an internal reference-count
// that is thread-safe; we never mutate the array from multiple threads simultaneously.
#[allow(unsafe_code)]
unsafe impl Send for HiddenStateLoss {}
#[allow(unsafe_code)]
unsafe impl Sync for HiddenStateLoss {}

impl super::DistillLoss for HiddenStateLoss {
    fn compute_weighted(
        &self,
        teacher: &Array,
        student: &Array,
        _temperature: f32,
        _weights: Option<&Array>,
    ) -> Result<Array> {
        self.compute(teacher, student)
    }

    fn name(&self) -> &'static str {
        HiddenStateLoss::name(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_mse_hidden_loss() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let student = Array::from_f32_slice(&[2.0_f32, 3.0, 4.0, 5.0], &[1, 2, 2]);

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
    #[serial]
    fn test_cosine_identical_vectors() {
        let hidden = Array::from_f32_slice(&[1.0_f32, 0.0, 0.0, 1.0], &[1, 2, 2]);

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
    #[serial]
    fn test_cosine_orthogonal_vectors() {
        // Orthogonal vectors: [1, 0] and [0, 1]
        let teacher = Array::from_f32_slice(&[1.0_f32, 0.0], &[1, 1, 2]);
        let student = Array::from_f32_slice(&[0.0_f32, 1.0], &[1, 1, 2]);

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
    #[serial]
    fn test_l1_hidden_loss() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let student = Array::from_f32_slice(&[2.0_f32, 3.0, 4.0, 5.0], &[1, 2, 2]);

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
    #[serial]
    fn test_layer_distillation_compute() {
        let teacher_hiddens: Vec<Array> = (0..4)
            .map(|_| Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 2, 2]))
            .collect();
        let student_hiddens: Vec<Array> = (0..2)
            .map(|_| Array::from_f32_slice(&[2.0_f32, 3.0, 4.0, 5.0], &[1, 2, 2]))
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

    /// Mismatched hidden dimensions without a projection must surface a
    /// descriptive `InvalidConfig` error before the matmul kernel runs.
    #[test]
    #[serial]
    fn unprojected_hidden_dim_mismatch_errors_clearly() {
        let teacher = Array::from_f32_slice(&[1.0_f32; 8], &[1, 2, 4]);
        let student = Array::from_f32_slice(&[1.0_f32; 4], &[1, 2, 2]);

        let loss = HiddenStateLoss::mse();
        let err = loss
            .compute(&teacher, &student)
            .expect_err("must reject mismatched dims with no projection");
        let msg = err.to_string();
        assert!(
            msg.contains("matching last dim or a projection"),
            "got: {msg}"
        );
        assert!(msg.contains("teacher.dim(-1) = 4"), "got: {msg}");
        assert!(msg.contains("student.dim(-1) = 2"), "got: {msg}");
    }

    /// A wrong-shape projection must error with shape numbers in the message
    /// rather than escalating to an opaque MLX matmul error.
    #[test]
    #[serial]
    fn projection_shape_mismatch_errors_clearly() {
        let teacher = Array::from_f32_slice(&[1.0_f32; 8], &[1, 2, 4]);
        let student = Array::from_f32_slice(&[1.0_f32; 4], &[1, 2, 2]);
        // student is dim 2 but projection's leading dim is 3 — a real bug shape.
        let bad_proj = Array::from_f32_slice(&[1.0_f32; 12], &[3, 4]);

        let loss = HiddenStateLoss::mse().with_projection(bad_proj);
        let err = loss
            .compute(&teacher, &student)
            .expect_err("must reject mismatched projection");
        let msg = err.to_string();
        assert!(msg.contains("projection shape mismatch"), "got: {msg}");
        assert!(msg.contains("student.dim(-1) = 2"), "got: {msg}");
        assert!(msg.contains("projection.dim(-2) = 3"), "got: {msg}");
    }

    /// A correctly-shaped projection must let the loss run and return a
    /// finite, non-negative value.
    #[test]
    #[serial]
    fn correct_projection_runs_loss_pathway() {
        let teacher = Array::from_f32_slice(&[1.0_f32; 8], &[1, 2, 4]);
        let student = Array::from_f32_slice(&[0.5_f32; 4], &[1, 2, 2]);
        // [hidden_student=2, hidden_teacher=4]; identity-style.
        let proj = Array::from_f32_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 4]);

        let loss = HiddenStateLoss::mse().with_projection(proj);
        let result = loss.compute(&teacher, &student).unwrap();
        let value: f32 = result.item();
        assert!(value.is_finite() && value >= 0.0);
    }

    #[test]
    #[serial]
    fn test_larger_batch() {
        // Realistic hidden width
        let batch_size = 4;
        let seq_len = 8;
        let hidden_dim = 256;

        let teacher_data: Vec<f32> = (0..(batch_size * seq_len * hidden_dim))
            .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
            .collect();
        let student_data: Vec<f32> = (0..(batch_size * seq_len * hidden_dim))
            .map(|i| ((i * 7 % 100) as f32 - 50.0) / 100.0)
            .collect();

        let teacher = Array::from_f32_slice(
            &teacher_data,
            &[batch_size as i32, seq_len as i32, hidden_dim as i32],
        );
        let student = Array::from_f32_slice(
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
