//! TAID: Temporally Adaptive Interpolated Distillation.
//!
//! Implements the TAID algorithm from "TAID: Temporally Adaptive Interpolated
//! Distillation for Efficient Knowledge Transfer" (ICLR 2025 Spotlight).
//!
//! Key innovation: Instead of distilling directly from teacher to student,
//! TAID creates an adaptive intermediate distribution that:
//!
//! 1. **Starts closer to teacher**: Early in training when student is weak
//! 2. **Gradually shifts toward student**: As student improves
//! 3. **Adapts per-sample**: Harder samples use more teacher guidance
//!
//! This prevents mode collapse and allows more stable knowledge transfer.
//!
//! # Algorithm
//!
//! Given teacher distribution P_T and student distribution P_S:
//! 1. Compute interpolated target: P_I = α * P_T + (1 - α) * P_S
//! 2. α adapts based on: (a) training progress, (b) sample difficulty
//! 3. Student learns to match P_I instead of P_T directly
//!
//! # Usage
//!
//! ```rust,ignore
//! use pmetal_distill::{TaidConfig, TaidDistiller};
//!
//! let config = TaidConfig::default()
//!     .with_initial_alpha(0.9)
//!     .with_final_alpha(0.5)
//!     .with_difficulty_scaling(true);
//!
//! let distiller = TaidDistiller::new(config);
//! let loss = distiller.compute_loss(&teacher_logits, &student_logits, step, total_steps)?;
//! ```

use crate::losses::softmax;
use crate::Result;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{error::Exception, Array};

/// Log-softmax along specified axis.
fn log_softmax(x: &Array, axis: i32) -> Result<Array> {
    // log_softmax(x) = x - log(sum(exp(x)))
    // For numerical stability: x - max(x) - log(sum(exp(x - max(x))))
    let max_x = x.max_axes(&[axis], Some(true))?;
    let shifted = x.subtract(&max_x)?;
    let exp_shifted = shifted.exp()?;
    let sum_exp = exp_shifted.sum_axes(&[axis], Some(true))?;
    let log_sum_exp = sum_exp.log()?;
    Ok(shifted.subtract(&log_sum_exp)?)
}

/// Error type for TAID operations.
#[derive(Debug, thiserror::Error)]
pub enum TaidError {
    /// MLX computation error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),
    /// Distillation error.
    #[error("Distillation error: {0}")]
    Distill(#[from] crate::DistillError),
}

/// Result type for TAID operations.
pub type TaidResult<T> = std::result::Result<T, TaidError>;

/// TAID configuration.
#[derive(Debug, Clone)]
pub struct TaidConfig {
    /// Initial interpolation factor (α at step 0).
    /// Higher values = more teacher guidance early on.
    /// Default: 0.9
    pub initial_alpha: f64,

    /// Final interpolation factor (α at final step).
    /// Lower values = student-driven learning late in training.
    /// Default: 0.5
    pub final_alpha: f64,

    /// Interpolation schedule type.
    /// Default: Cosine
    pub schedule: TaidSchedule,

    /// Temperature for softmax over teacher logits.
    /// Higher temperature = softer targets.
    /// Default: 4.0
    pub temperature: f64,

    /// Whether to use per-sample difficulty-aware alpha.
    /// Default: true
    pub difficulty_aware: bool,

    /// Scaling factor for difficulty-based alpha adjustment.
    /// Default: 0.2
    pub difficulty_scale: f64,

    /// Minimum alpha value (clamp for numerical stability).
    /// Default: 0.1
    pub min_alpha: f64,

    /// Maximum alpha value (clamp for numerical stability).
    /// Default: 1.0
    pub max_alpha: f64,

    /// Whether to use KL divergence or cross-entropy for loss.
    /// Default: KL
    pub loss_type: TaidLossType,

    /// Whether to use label smoothing.
    /// Default: false
    pub label_smoothing: bool,

    /// Label smoothing factor.
    /// Default: 0.1
    pub label_smoothing_factor: f64,

    /// Hard target weight (optional ground truth loss).
    /// Default: 0.0 (pure distillation)
    pub hard_target_weight: f64,
}

/// Schedule for alpha interpolation over training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TaidSchedule {
    /// Linear decay from initial to final alpha.
    Linear,
    /// Cosine annealing (smoother transition).
    #[default]
    Cosine,
    /// Exponential decay.
    Exponential,
    /// Step-wise schedule (drops at specific milestones).
    Step,
    /// Constant alpha (no temporal adaptation).
    Constant,
}

/// Loss type for TAID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TaidLossType {
    /// KL divergence: D_KL(P_I || P_S)
    #[default]
    KlDivergence,
    /// Cross-entropy: -P_I * log(P_S)
    CrossEntropy,
    /// Jensen-Shannon divergence: symmetric KL
    JensenShannon,
    /// Reverse KL: D_KL(P_S || P_I)
    ReverseKl,
}

impl Default for TaidConfig {
    fn default() -> Self {
        Self {
            initial_alpha: 0.9,
            final_alpha: 0.5,
            schedule: TaidSchedule::Cosine,
            temperature: 4.0,
            difficulty_aware: true,
            difficulty_scale: 0.2,
            min_alpha: 0.1,
            max_alpha: 1.0,
            loss_type: TaidLossType::KlDivergence,
            label_smoothing: false,
            label_smoothing_factor: 0.1,
            hard_target_weight: 0.0,
        }
    }
}

impl TaidConfig {
    /// Create a new TAID config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set initial alpha.
    pub fn with_initial_alpha(mut self, alpha: f64) -> Self {
        self.initial_alpha = alpha;
        self
    }

    /// Set final alpha.
    pub fn with_final_alpha(mut self, alpha: f64) -> Self {
        self.final_alpha = alpha;
        self
    }

    /// Set schedule type.
    pub fn with_schedule(mut self, schedule: TaidSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Enable/disable difficulty-aware alpha.
    pub fn with_difficulty_scaling(mut self, enabled: bool) -> Self {
        self.difficulty_aware = enabled;
        self
    }

    /// Set loss type.
    pub fn with_loss_type(mut self, loss_type: TaidLossType) -> Self {
        self.loss_type = loss_type;
        self
    }

    /// Set hard target weight.
    pub fn with_hard_target_weight(mut self, weight: f64) -> Self {
        self.hard_target_weight = weight;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> TaidResult<()> {
        if self.initial_alpha < 0.0 || self.initial_alpha > 1.0 {
            return Err(TaidError::Config("initial_alpha must be in [0, 1]".into()));
        }
        if self.final_alpha < 0.0 || self.final_alpha > 1.0 {
            return Err(TaidError::Config("final_alpha must be in [0, 1]".into()));
        }
        if self.temperature <= 0.0 {
            return Err(TaidError::Config("temperature must be positive".into()));
        }
        if self.min_alpha > self.max_alpha {
            return Err(TaidError::Config("min_alpha must be <= max_alpha".into()));
        }
        Ok(())
    }
}

/// TAID Distiller.
///
/// Implements temporally adaptive interpolated distillation.
pub struct TaidDistiller {
    /// Configuration.
    config: TaidConfig,
}

impl TaidDistiller {
    /// Create a new TAID distiller.
    pub fn new(config: TaidConfig) -> TaidResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Compute the interpolation factor alpha based on training progress.
    ///
    /// # Arguments
    /// * `step` - Current training step
    /// * `total_steps` - Total training steps
    ///
    /// # Returns
    /// Base alpha value (before per-sample difficulty adjustment)
    pub fn compute_base_alpha(&self, step: usize, total_steps: usize) -> f64 {
        if total_steps == 0 {
            return self.config.initial_alpha;
        }

        let progress = (step as f64) / (total_steps as f64);
        let alpha_range = self.config.initial_alpha - self.config.final_alpha;

        let alpha = match self.config.schedule {
            TaidSchedule::Linear => self.config.initial_alpha - alpha_range * progress,
            TaidSchedule::Cosine => {
                // Cosine annealing: starts slow, speeds up, then slows down
                let cosine_factor = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                self.config.final_alpha + alpha_range * cosine_factor
            }
            TaidSchedule::Exponential => {
                // Exponential decay
                let decay_rate = (self.config.final_alpha / self.config.initial_alpha).ln();
                self.config.initial_alpha * (decay_rate * progress).exp()
            }
            TaidSchedule::Step => {
                // Step-wise: 4 stages
                if progress < 0.25 {
                    self.config.initial_alpha
                } else if progress < 0.5 {
                    self.config.initial_alpha - alpha_range * 0.33
                } else if progress < 0.75 {
                    self.config.initial_alpha - alpha_range * 0.66
                } else {
                    self.config.final_alpha
                }
            }
            TaidSchedule::Constant => self.config.initial_alpha,
        };

        alpha.clamp(self.config.min_alpha, self.config.max_alpha)
    }

    /// Compute per-sample difficulty and adjust alpha.
    ///
    /// Difficulty is measured by KL divergence between teacher and student.
    /// Higher difficulty = higher alpha (more teacher guidance).
    ///
    /// # Arguments
    /// * `teacher_probs` - Teacher probability distribution [batch, seq, vocab]
    /// * `student_probs` - Student probability distribution [batch, seq, vocab]
    /// * `base_alpha` - Base alpha from temporal schedule
    ///
    /// # Returns
    /// Per-sample alpha values [batch]
    pub fn compute_difficulty_alpha(
        &self,
        teacher_probs: &Array,
        student_probs: &Array,
        base_alpha: f64,
    ) -> TaidResult<Array> {
        if !self.config.difficulty_aware {
            // Return constant alpha for all samples
            let batch_size = teacher_probs.dim(0);
            return Ok(Array::from_slice(
                &vec![base_alpha as f32; batch_size as usize],
                &[batch_size],
            ));
        }

        // Compute KL divergence per sample as difficulty measure
        // KL(P_T || P_S) = sum(P_T * log(P_T / P_S))
        let eps = Array::from_f32(1e-10);
        let student_safe = student_probs.add(&eps)?;
        let log_ratio = teacher_probs.divide(&student_safe)?.log()?;
        let kl_per_token = teacher_probs.multiply(&log_ratio)?;

        // Sum over vocab and seq dimensions to get per-sample KL
        let kl_per_sample = kl_per_token.sum_axis(-1, None)?.sum_axis(-1, None)?;
        kl_per_sample.eval()?;

        // Normalize difficulty to [0, 1] range using sigmoid
        // Higher KL = harder sample = higher alpha
        let scale = Array::from_f32(self.config.difficulty_scale as f32);
        let difficulty = kl_per_sample.multiply(&scale)?;
        let sigmoid_diff = difficulty.negative()?.exp()?.add(&Array::from_f32(1.0))?;
        let normalized_diff = Array::from_f32(1.0).divide(&sigmoid_diff)?;

        // Adjust alpha: alpha = base_alpha + difficulty_adjustment
        // Where difficulty_adjustment scales from 0 to (max_alpha - base_alpha)
        let alpha_adjustment_range = (self.config.max_alpha - base_alpha) as f32;
        let adjustment = normalized_diff.multiply(&Array::from_f32(alpha_adjustment_range))?;
        let alpha = adjustment.add(&Array::from_f32(base_alpha as f32))?;

        // Clamp to valid range
        let min_alpha = Array::from_f32(self.config.min_alpha as f32);
        let max_alpha = Array::from_f32(self.config.max_alpha as f32);
        let clamped = mlx_rs::ops::maximum(&alpha, &min_alpha)?;
        let clamped = mlx_rs::ops::minimum(&clamped, &max_alpha)?;

        Ok(clamped)
    }

    /// Compute the interpolated target distribution.
    ///
    /// P_I = α * P_T + (1 - α) * P_S
    ///
    /// # Arguments
    /// * `teacher_probs` - Teacher softmax probs [batch, seq, vocab]
    /// * `student_probs` - Student softmax probs [batch, seq, vocab]
    /// * `alpha` - Interpolation factor [batch] or scalar
    ///
    /// # Returns
    /// Interpolated distribution [batch, seq, vocab]
    pub fn interpolate_distributions(
        &self,
        teacher_probs: &Array,
        student_probs: &Array,
        alpha: &Array,
    ) -> TaidResult<Array> {
        // Expand alpha for broadcasting: [batch] -> [batch, 1, 1]
        let alpha_expanded = alpha.reshape(&[alpha.dim(0), 1, 1])?;
        let one_minus_alpha = Array::from_f32(1.0).subtract(&alpha_expanded)?;

        // P_I = α * P_T + (1 - α) * P_S
        let teacher_contrib = teacher_probs.multiply(&alpha_expanded)?;
        let student_contrib = student_probs.multiply(&one_minus_alpha)?;
        let interpolated = teacher_contrib.add(&student_contrib)?;

        Ok(interpolated)
    }

    /// Apply temperature scaling to logits.
    pub fn apply_temperature(&self, logits: &Array) -> TaidResult<Array> {
        let temp = Array::from_f32(self.config.temperature as f32);
        Ok(logits.divide(&temp)?)
    }

    /// Compute TAID loss.
    ///
    /// # Arguments
    /// * `teacher_logits` - Teacher model logits [batch, seq, vocab]
    /// * `student_logits` - Student model logits [batch, seq, vocab]
    /// * `step` - Current training step
    /// * `total_steps` - Total training steps
    /// * `labels` - Optional ground truth labels for hard target loss
    ///
    /// # Returns
    /// TaidLossOutput containing total loss and components
    pub fn compute_loss(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        step: usize,
        total_steps: usize,
        labels: Option<&Array>,
    ) -> TaidResult<TaidLossOutput> {
        // Apply temperature scaling
        let teacher_scaled = self.apply_temperature(teacher_logits)?;
        let student_scaled = self.apply_temperature(student_logits)?;

        // Compute softmax probabilities
        let teacher_probs = softmax(&teacher_scaled, -1)?;
        let student_probs = softmax(&student_scaled, -1)?;
        let student_log_probs = log_softmax(&student_scaled, -1)?;

        // Compute base alpha from temporal schedule
        let base_alpha = self.compute_base_alpha(step, total_steps);

        // Compute per-sample alpha (difficulty-aware)
        let alpha = self.compute_difficulty_alpha(&teacher_probs, &student_probs, base_alpha)?;

        // Compute interpolated target distribution
        let target_probs =
            self.interpolate_distributions(&teacher_probs, &student_probs, &alpha)?;

        // Compute distillation loss
        let distill_loss = match self.config.loss_type {
            TaidLossType::KlDivergence => {
                // KL(P_I || P_S) = sum(P_I * log(P_I / P_S))
                let eps = Array::from_f32(1e-10);
                let target_safe = target_probs.add(&eps)?;
                let student_safe = student_probs.add(&eps)?;
                let log_ratio = target_safe.divide(&student_safe)?.log()?;
                let kl = target_probs.multiply(&log_ratio)?;
                kl.sum_axis(-1, None)?.mean(None)?
            }
            TaidLossType::CrossEntropy => {
                // -P_I * log(P_S)
                let ce = target_probs.multiply(&student_log_probs)?.negative()?;
                ce.sum_axis(-1, None)?.mean(None)?
            }
            TaidLossType::JensenShannon => {
                // JS = 0.5 * KL(P_I || M) + 0.5 * KL(P_S || M)
                // where M = 0.5 * (P_I + P_S)
                let eps = Array::from_f32(1e-10);
                let m = target_probs
                    .add(&student_probs)?
                    .multiply(&Array::from_f32(0.5))?;
                let m_safe = m.add(&eps)?;

                let log_ratio_target = target_probs.add(&eps)?.divide(&m_safe)?.log()?;
                let kl_target = target_probs.multiply(&log_ratio_target)?;

                let log_ratio_student = student_probs.add(&eps)?.divide(&m_safe)?.log()?;
                let kl_student = student_probs.multiply(&log_ratio_student)?;

                let js = kl_target
                    .add(&kl_student)?
                    .multiply(&Array::from_f32(0.5))?;
                js.sum_axis(-1, None)?.mean(None)?
            }
            TaidLossType::ReverseKl => {
                // KL(P_S || P_I) = sum(P_S * log(P_S / P_I))
                let eps = Array::from_f32(1e-10);
                let target_safe = target_probs.add(&eps)?;
                let student_safe = student_probs.add(&eps)?;
                let log_ratio = student_safe.divide(&target_safe)?.log()?;
                let kl = student_probs.multiply(&log_ratio)?;
                kl.sum_axis(-1, None)?.mean(None)?
            }
        };

        // Scale distillation loss by temperature^2 (standard practice)
        let temp_sq = (self.config.temperature * self.config.temperature) as f32;
        let scaled_distill_loss = distill_loss.multiply(&Array::from_f32(temp_sq))?;

        // Optional hard target loss
        let hard_loss = if self.config.hard_target_weight > 0.0 {
            if let Some(lbl) = labels {
                // Cross-entropy with ground truth
                let student_log_probs_unscaled = log_softmax(student_logits, -1)?;
                // Gather log probs at label positions
                // This is a simplified version - full implementation would handle padding
                let batch_size = student_logits.dim(0);
                let seq_len = student_logits.dim(1);
                let _vocab_size = student_logits.dim(2);

                // Create mask for valid labels (not -100)
                let valid_mask = lbl.ne(&Array::from_int(-100))?;
                let valid_mask_f32 = valid_mask.as_dtype(mlx_rs::Dtype::Float32)?;

                // Replace -100 with 0 for gathering
                let safe_labels = mlx_rs::ops::maximum(lbl, &Array::from_int(0))?;

                // Manual gather - for each position, get the log prob at the label index
                let mut ce_values = Vec::new();
                student_log_probs_unscaled.eval()?;
                safe_labels.eval()?;

                for b in 0..batch_size {
                    for s in 0..seq_len {
                        let label_idx = safe_labels.index((b as i32, s as i32));
                        label_idx.eval()?;
                        let idx = label_idx.item::<i32>();
                        let log_prob = student_log_probs_unscaled.index((b as i32, s as i32, idx));
                        log_prob.eval()?;
                        ce_values.push(-log_prob.item::<f32>());
                    }
                }

                let ce_array = Array::from_slice(&ce_values, &[batch_size, seq_len]);
                let masked_ce = ce_array.multiply(&valid_mask_f32)?;
                let total_valid = valid_mask_f32.sum(None)?;
                Some(masked_ce.sum(None)?.divide(&total_valid)?)
            } else {
                None
            }
        } else {
            None
        };

        // Combine losses
        let total_loss = if let Some(hl) = &hard_loss {
            let hard_weight = Array::from_f32(self.config.hard_target_weight as f32);
            let soft_weight = Array::from_f32(1.0 - self.config.hard_target_weight as f32);
            scaled_distill_loss
                .multiply(&soft_weight)?
                .add(&hl.multiply(&hard_weight)?)?
        } else {
            scaled_distill_loss.clone()
        };

        // Compute mean alpha for logging
        alpha.eval()?;
        let mean_alpha = alpha.mean(None)?;
        mean_alpha.eval()?;

        Ok(TaidLossOutput {
            total: total_loss,
            distillation: scaled_distill_loss,
            hard_target: hard_loss,
            mean_alpha: mean_alpha.item::<f32>(),
            base_alpha: base_alpha as f32,
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &TaidConfig {
        &self.config
    }
}

/// Output from TAID loss computation.
#[derive(Debug)]
pub struct TaidLossOutput {
    /// Total combined loss.
    pub total: Array,
    /// Distillation loss component.
    pub distillation: Array,
    /// Hard target loss component (if used).
    pub hard_target: Option<Array>,
    /// Mean alpha used in this batch.
    pub mean_alpha: f32,
    /// Base alpha from temporal schedule.
    pub base_alpha: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_taid_config_default() {
        let config = TaidConfig::default();
        assert!((config.initial_alpha - 0.9).abs() < 1e-10);
        assert!((config.final_alpha - 0.5).abs() < 1e-10);
        assert_eq!(config.schedule, TaidSchedule::Cosine);
        assert!(config.difficulty_aware);
    }

    #[test]
    fn test_taid_config_validation() {
        let config = TaidConfig::default();
        assert!(config.validate().is_ok());

        let invalid = TaidConfig {
            initial_alpha: 1.5,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        let invalid_temp = TaidConfig {
            temperature: 0.0,
            ..Default::default()
        };
        assert!(invalid_temp.validate().is_err());
    }

    #[test]
    fn test_compute_base_alpha_linear() {
        let config = TaidConfig {
            initial_alpha: 1.0,
            final_alpha: 0.0,
            schedule: TaidSchedule::Linear,
            ..Default::default()
        };
        let distiller = TaidDistiller::new(config).unwrap();

        // Start
        let alpha_0 = distiller.compute_base_alpha(0, 100);
        assert!((alpha_0 - 1.0).abs() < 0.01);

        // Middle
        let alpha_50 = distiller.compute_base_alpha(50, 100);
        assert!((alpha_50 - 0.5).abs() < 0.01);

        // End
        let alpha_100 = distiller.compute_base_alpha(100, 100);
        assert!((alpha_100 - 0.1).abs() < 0.01); // Clamped to min_alpha
    }

    #[test]
    fn test_compute_base_alpha_cosine() {
        let config = TaidConfig {
            initial_alpha: 0.9,
            final_alpha: 0.5,
            schedule: TaidSchedule::Cosine,
            ..Default::default()
        };
        let distiller = TaidDistiller::new(config).unwrap();

        let alpha_0 = distiller.compute_base_alpha(0, 100);
        let alpha_50 = distiller.compute_base_alpha(50, 100);
        let alpha_100 = distiller.compute_base_alpha(100, 100);

        // Cosine: starts at initial, ends at final
        assert!((alpha_0 - 0.9).abs() < 0.01);
        assert!((alpha_100 - 0.5).abs() < 0.01);
        // Middle should be between but closer to mean due to cosine shape
        assert!(alpha_50 > 0.5 && alpha_50 < 0.9);
    }

    #[test]
    fn test_interpolate_distributions() {
        let config = TaidConfig::default();
        let distiller = TaidDistiller::new(config).unwrap();

        // Simple 2x1x3 distributions
        let teacher = Array::from_slice(&[0.7f32, 0.2, 0.1, 0.1, 0.8, 0.1], &[2, 1, 3]);
        let student = Array::from_slice(&[0.3f32, 0.4, 0.3, 0.5, 0.3, 0.2], &[2, 1, 3]);
        let alpha = Array::from_slice(&[0.5f32, 0.8], &[2]);

        let interpolated = distiller
            .interpolate_distributions(&teacher, &student, &alpha)
            .unwrap();

        interpolated.eval().unwrap();
        assert_eq!(interpolated.shape(), &[2, 1, 3]);

        // For first sample: 0.5 * teacher + 0.5 * student
        // = 0.5 * [0.7, 0.2, 0.1] + 0.5 * [0.3, 0.4, 0.3]
        // = [0.5, 0.3, 0.2]
        let vals: Vec<f32> = interpolated.as_slice::<f32>().to_vec();
        assert!((vals[0] - 0.5).abs() < 0.01);
        assert!((vals[1] - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_taid_loss_computation() {
        let config = TaidConfig {
            difficulty_aware: false, // Simpler test
            temperature: 1.0,
            ..Default::default()
        };
        let distiller = TaidDistiller::new(config).unwrap();

        // Create simple logits
        let teacher_logits = Array::from_slice(&[2.0f32, 1.0, 0.0, 0.0, 1.0, 2.0], &[2, 1, 3]);
        let student_logits = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 1, 3]);

        let output = distiller
            .compute_loss(&teacher_logits, &student_logits, 0, 100, None)
            .unwrap();

        output.total.eval().unwrap();
        output.distillation.eval().unwrap();

        assert!(output.total.item::<f32>().is_finite());
        assert!(output.distillation.item::<f32>() >= 0.0);
        assert!(output.hard_target.is_none());
        assert!(output.base_alpha > 0.0);
    }

    #[test]
    fn test_taid_with_hard_targets() {
        let config = TaidConfig {
            difficulty_aware: false,
            hard_target_weight: 0.5,
            ..Default::default()
        };
        let distiller = TaidDistiller::new(config).unwrap();

        let teacher_logits = Array::from_slice(&[2.0f32, 1.0, 0.0], &[1, 1, 3]);
        let student_logits = Array::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3]);
        let labels = Array::from_slice(&[0i32], &[1, 1]);

        let output = distiller
            .compute_loss(&teacher_logits, &student_logits, 50, 100, Some(&labels))
            .unwrap();

        output.total.eval().unwrap();
        assert!(output.total.item::<f32>().is_finite());
        assert!(output.hard_target.is_some());
    }

    #[test]
    fn test_taid_schedule_step() {
        let config = TaidConfig {
            initial_alpha: 1.0,
            final_alpha: 0.2,
            schedule: TaidSchedule::Step,
            min_alpha: 0.1,
            ..Default::default()
        };
        let distiller = TaidDistiller::new(config).unwrap();

        let alpha_10 = distiller.compute_base_alpha(10, 100);
        let alpha_30 = distiller.compute_base_alpha(30, 100);
        let alpha_60 = distiller.compute_base_alpha(60, 100);
        let alpha_80 = distiller.compute_base_alpha(80, 100);

        // Step-wise decreases
        assert!(alpha_10 > alpha_30);
        assert!(alpha_30 > alpha_60);
        assert!(alpha_60 > alpha_80);
    }

    #[test]
    fn test_difficulty_alpha() {
        let config = TaidConfig {
            difficulty_aware: true,
            difficulty_scale: 1.0, // High scaling for visible effect
            ..Default::default()
        };
        let distiller = TaidDistiller::new(config).unwrap();

        // Easy sample: teacher and student agree
        let teacher_easy = Array::from_slice(&[0.8f32, 0.1, 0.1], &[1, 1, 3]);
        let student_easy = Array::from_slice(&[0.7f32, 0.15, 0.15], &[1, 1, 3]);

        // Hard sample: teacher and student disagree
        let teacher_hard = Array::from_slice(&[0.9f32, 0.05, 0.05], &[1, 1, 3]);
        let student_hard = Array::from_slice(&[0.1f32, 0.45, 0.45], &[1, 1, 3]);

        let alpha_easy = distiller
            .compute_difficulty_alpha(&teacher_easy, &student_easy, 0.5)
            .unwrap();
        let alpha_hard = distiller
            .compute_difficulty_alpha(&teacher_hard, &student_hard, 0.5)
            .unwrap();

        alpha_easy.eval().unwrap();
        alpha_hard.eval().unwrap();

        // Hard samples should have higher alpha (more teacher guidance)
        assert!(alpha_hard.item::<f32>() > alpha_easy.item::<f32>());
    }
}
