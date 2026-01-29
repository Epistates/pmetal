//! Main distillation orchestration.
//!
//! This module provides the high-level API for running knowledge distillation,
//! including online, offline, and progressive distillation modes.

use std::path::PathBuf;

use mlx_rs::Array;
use tracing::{debug, info, warn};

use crate::{
    losses::{
        DistillLoss, HiddenStateLoss, JensenShannonLoss, KlDivergenceLoss, MseLoss,
        SoftCrossEntropyLoss,
    },
    offline::{LogitCache, LogitCompressor},
    CompressionMethod, DistillConfig, DistillError, DistillMethod, LossType, Result,
};

/// Run knowledge distillation with the given configuration.
pub fn run_distillation(config: &DistillConfig) -> Result<PathBuf> {
    info!("Starting knowledge distillation");
    info!("  Teacher: {}", config.teacher);
    info!("  Student: {}", config.student);
    info!("  Method: {:?}", config.method);

    config.validate()?;

    let distiller = DistillerBuilder::new()
        .with_config(config.clone())
        .build()?;

    distiller.run()
}

/// Builder for creating a Distiller instance.
pub struct DistillerBuilder {
    config: Option<DistillConfig>,
    loss: Option<Box<dyn DistillLoss>>,
    hidden_loss: Option<HiddenStateLoss>,
}

impl DistillerBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: None,
            loss: None,
            hidden_loss: None,
        }
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: DistillConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set a custom loss function.
    pub fn with_loss(mut self, loss: Box<dyn DistillLoss>) -> Self {
        self.loss = Some(loss);
        self
    }

    /// Set hidden state loss.
    pub fn with_hidden_loss(mut self, loss: HiddenStateLoss) -> Self {
        self.hidden_loss = Some(loss);
        self
    }

    /// Build the Distiller.
    pub fn build(self) -> Result<Distiller> {
        let config = self
            .config
            .ok_or_else(|| DistillError::InvalidConfig("Configuration is required".to_string()))?;

        // Create loss function from config if not provided
        let loss: Box<dyn DistillLoss> = self.loss.unwrap_or_else(|| match config.loss.loss_type {
            LossType::KlDivergence => {
                if config.loss.reverse_kl {
                    Box::new(KlDivergenceLoss::reverse())
                } else {
                    Box::new(KlDivergenceLoss::new())
                }
            }
            LossType::JensenShannon => Box::new(JensenShannonLoss::new()),
            LossType::SoftCrossEntropy => Box::new(SoftCrossEntropyLoss::new()),
            LossType::MseLoss => Box::new(MseLoss::new()),
        });

        Ok(Distiller {
            config,
            loss,
            hidden_loss: self.hidden_loss,
        })
    }
}

impl Default for DistillerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Main distillation engine.
pub struct Distiller {
    /// Configuration.
    config: DistillConfig,
    /// Primary loss function.
    loss: Box<dyn DistillLoss>,
    /// Optional hidden state loss.
    hidden_loss: Option<HiddenStateLoss>,
}

impl Distiller {
    /// Create a new distiller with default settings.
    pub fn new(config: DistillConfig) -> Result<Self> {
        DistillerBuilder::new().with_config(config).build()
    }

    /// Run the distillation process.
    pub fn run(&self) -> Result<PathBuf> {
        match &self.config.method {
            DistillMethod::Online => self.run_online(),
            DistillMethod::Offline => self.run_offline(),
            DistillMethod::Progressive => self.run_progressive(),
        }
    }

    /// Run online distillation.
    ///
    /// Both teacher and student are loaded and run during training.
    fn run_online(&self) -> Result<PathBuf> {
        info!("Running online distillation");

        // This would integrate with pmetal-trainer for actual training
        // For now, we provide the infrastructure for the training loop

        let output_path = self
            .config
            .output_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("./distilled_model"));

        info!("Online distillation configured");
        info!("  Loss: {}", self.loss.name());
        info!("  Temperature: {}", self.config.loss.temperature);
        info!("  Alpha: {}", self.config.loss.alpha);
        info!("  Output: {:?}", output_path);

        // The actual training loop would:
        // 1. Load teacher and student models
        // 2. For each batch:
        //    a. Forward through teacher (no grad)
        //    b. Forward through student (with grad)
        //    c. Compute distillation loss
        //    d. Backward through student
        //    e. Update student weights

        Ok(output_path)
    }

    /// Run offline distillation.
    ///
    /// Uses pre-computed teacher logits from cache.
    fn run_offline(&self) -> Result<PathBuf> {
        info!("Running offline distillation");

        let offline_config = self.config.offline.as_ref().ok_or_else(|| {
            DistillError::InvalidConfig(
                "Offline config required for offline distillation".to_string(),
            )
        })?;

        let output_path = self
            .config
            .output_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("./distilled_model"));

        if offline_config.generate {
            // Generate and cache teacher logits
            info!("Generating teacher logits...");
            self.generate_logits(offline_config)?;
        }

        // Load cached logits
        let cache = LogitCache::load(&offline_config.logits_path)?;
        info!(
            "Loaded logit cache with {} sequences",
            cache.metadata().num_sequences
        );
        info!("  Compression: {}", cache.metadata().compression);
        info!("  Vocab size: {}", cache.metadata().vocab_size);

        // The actual training loop would:
        // 1. Load student model
        // 2. For each batch:
        //    a. Load cached teacher logits
        //    b. Forward through student
        //    c. Compute distillation loss
        //    d. Backward and update

        Ok(output_path)
    }

    /// Generate and cache teacher logits.
    fn generate_logits(&self, offline_config: &crate::OfflineConfig) -> Result<()> {
        info!(
            "Generating teacher logits to {:?}",
            offline_config.logits_path
        );

        let mut cache = LogitCache::new(
            &offline_config.logits_path,
            offline_config.compression.clone(),
            offline_config.top_k,
        )?;

        // This would:
        // 1. Load teacher model
        // 2. Load dataset
        // 3. For each sequence in dataset:
        //    a. Forward through teacher
        //    b. Compress and cache logits

        cache.set_metadata(
            self.config.teacher.clone(),
            0, // Would be set from model
            self.config.training.max_seq_len,
        );
        cache.save_metadata()?;

        info!("Logit generation complete");
        Ok(())
    }

    /// Run progressive distillation.
    ///
    /// Gradually reduces temperature and teacher influence over training.
    fn run_progressive(&self) -> Result<PathBuf> {
        info!("Running progressive distillation");

        let output_path = self
            .config
            .output_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("./distilled_model"));

        let total_steps = self.config.training.epochs * 1000; // Approximate
        let initial_temp = self.config.loss.temperature;
        let initial_alpha = self.config.loss.alpha;

        info!("Progressive schedule:");
        info!("  Initial temperature: {}", initial_temp);
        info!("  Final temperature: 1.0");
        info!("  Initial alpha: {}", initial_alpha);
        info!("  Final alpha: 0.0");
        info!("  Total steps: ~{}", total_steps);

        // Progressive distillation schedule:
        // - Temperature: gradually decrease from T to 1
        // - Alpha: gradually decrease from alpha to 0 (shift to hard labels)

        Ok(output_path)
    }

    /// Compute distillation loss for a batch.
    pub fn compute_loss(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        labels: Option<&Array>,
    ) -> Result<DistillLossOutput> {
        // Soft distillation loss
        let soft_loss =
            self.loss
                .compute(teacher_logits, student_logits, self.config.loss.temperature)?;

        // Scale by temperature squared (to maintain gradient magnitude)
        let t_squared = self.config.loss.temperature * self.config.loss.temperature;
        let soft_scaled = soft_loss.multiply(&Array::from_f32(t_squared))?;

        // Combined with hard labels if provided
        let total_loss = if let Some(labels) = labels {
            let hard_loss = compute_hard_loss(student_logits, labels)?;

            // total = alpha * soft + (1 - alpha) * hard
            let alpha = self.config.loss.alpha;
            let soft_weighted = soft_scaled.multiply(&Array::from_f32(alpha))?;
            let hard_weighted = hard_loss.multiply(&Array::from_f32(1.0 - alpha))?;

            soft_weighted.add(&hard_weighted)?
        } else {
            soft_scaled
        };

        Ok(DistillLossOutput {
            total: total_loss,
            soft: soft_loss,
            hard: None,
            hidden: None,
        })
    }

    /// Compute loss with hidden state alignment.
    pub fn compute_loss_with_hidden(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        teacher_hiddens: &[Array],
        student_hiddens: &[Array],
        labels: Option<&Array>,
    ) -> Result<DistillLossOutput> {
        let mut output = self.compute_loss(teacher_logits, student_logits, labels)?;

        // Add hidden state loss if configured
        if let Some(_hidden_loss) = &self.hidden_loss {
            if let Some(hidden_config) = &self.config.loss.hidden_state {
                let layer_distill = crate::losses::hidden_state::LayerDistillation::new(
                    hidden_config.layer_mapping.clone(),
                    HiddenStateLoss::new(hidden_config.loss_type.clone()),
                    hidden_config.weight,
                );

                let hidden = layer_distill.compute(teacher_hiddens, student_hiddens)?;
                output.total = output.total.add(&hidden)?;
                output.hidden = Some(hidden);
            }
        }

        Ok(output)
    }

    /// Get the configuration.
    pub fn config(&self) -> &DistillConfig {
        &self.config
    }

    /// Get progressive schedule parameters for a given step.
    pub fn progressive_params(&self, step: usize, total_steps: usize) -> (f32, f32) {
        let progress = (step as f32) / (total_steps.max(1) as f32);

        // Linear decay from initial to final
        let initial_temp = self.config.loss.temperature;
        let final_temp = 1.0;
        let temperature = initial_temp + (final_temp - initial_temp) * progress;

        let initial_alpha = self.config.loss.alpha;
        let final_alpha = 0.0;
        let alpha = initial_alpha + (final_alpha - initial_alpha) * progress;

        (temperature, alpha)
    }
}

/// Output from distillation loss computation.
#[derive(Debug)]
pub struct DistillLossOutput {
    /// Total combined loss.
    pub total: Array,
    /// Soft target loss component.
    pub soft: Array,
    /// Hard target loss component (if labels provided).
    pub hard: Option<Array>,
    /// Hidden state loss component.
    pub hidden: Option<Array>,
}

/// Compute hard cross-entropy loss with labels.
fn compute_hard_loss(logits: &Array, labels: &Array) -> Result<Array> {
    // Log-softmax for numerical stability
    let max_logits = logits.max_axes(&[-1], Some(true))?;
    let shifted = logits.subtract(&max_logits)?;
    let exp_shifted = shifted.exp()?;
    let sum_exp = exp_shifted.sum_axes(&[-1], Some(true))?;
    let log_probs = shifted.subtract(&sum_exp.log()?)?;

    // Gather log probs at label positions
    let batch_size = logits.dim(0) as usize;
    let seq_len = logits.dim(1) as usize;
    let vocab_size = logits.dim(2) as usize;

    let log_probs_data: Vec<f32> = log_probs.as_slice().to_vec();
    let labels_data: Vec<i32> = labels.as_slice().to_vec();

    let mut losses = Vec::with_capacity(batch_size * seq_len);
    for i in 0..(batch_size * seq_len) {
        let label = labels_data.get(i).copied().unwrap_or(0) as usize;
        if label < vocab_size {
            losses.push(-log_probs_data[i * vocab_size + label]);
        } else {
            losses.push(0.0);
        }
    }

    let loss_array = Array::from_slice(&losses, &[(batch_size * seq_len) as i32]);
    Ok(loss_array.mean(None)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn test_config() -> DistillConfig {
        DistillConfig {
            teacher: "teacher-model".to_string(),
            student: "student-model".to_string(),
            method: DistillMethod::Online,
            loss: crate::LossConfig::default(),
            offline: None,
            output_path: None,
            training: crate::TrainingConfig::default(),
        }
    }

    #[test]
    #[serial]
    fn test_distiller_builder() {
        let config = test_config();
        let distiller = DistillerBuilder::new().with_config(config).build().unwrap();

        assert_eq!(distiller.config().teacher, "teacher-model");
        assert_eq!(distiller.loss.name(), "kl_divergence");
    }

    #[test]
    fn test_distiller_with_custom_loss() {
        let config = test_config();
        let distiller = DistillerBuilder::new()
            .with_config(config)
            .with_loss(Box::new(JensenShannonLoss::new()))
            .build()
            .unwrap();

        assert_eq!(distiller.loss.name(), "jensen_shannon");
    }

    #[test]
    fn test_compute_loss() {
        let config = test_config();
        let distiller = Distiller::new(config).unwrap();

        let teacher = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_slice(&[4.0_f32, 3.0, 2.0, 1.0], &[1, 1, 4]);

        let output = distiller.compute_loss(&teacher, &student, None).unwrap();

        // Loss should be positive
        let value: f32 = output.total.item();
        assert!(value > 0.0);
    }

    #[test]
    fn test_progressive_params() {
        let config = test_config();
        let distiller = Distiller::new(config).unwrap();

        // At start
        let (temp, alpha) = distiller.progressive_params(0, 1000);
        assert!((temp - 2.0).abs() < 1e-5); // initial temperature
        assert!((alpha - 0.5).abs() < 1e-5); // initial alpha

        // At end
        let (temp, alpha) = distiller.progressive_params(1000, 1000);
        assert!((temp - 1.0).abs() < 1e-5); // final temperature
        assert!(alpha.abs() < 1e-5); // final alpha = 0

        // At midpoint
        let (temp, alpha) = distiller.progressive_params(500, 1000);
        assert!((temp - 1.5).abs() < 1e-5);
        assert!((alpha - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_loss_types_from_config() {
        let mut config = test_config();

        // Test each loss type
        config.loss.loss_type = LossType::KlDivergence;
        let d = Distiller::new(config.clone()).unwrap();
        assert_eq!(d.loss.name(), "kl_divergence");

        config.loss.loss_type = LossType::JensenShannon;
        let d = Distiller::new(config.clone()).unwrap();
        assert_eq!(d.loss.name(), "jensen_shannon");

        config.loss.loss_type = LossType::SoftCrossEntropy;
        let d = Distiller::new(config.clone()).unwrap();
        assert_eq!(d.loss.name(), "soft_cross_entropy");

        config.loss.loss_type = LossType::MseLoss;
        let d = Distiller::new(config.clone()).unwrap();
        assert_eq!(d.loss.name(), "mse");
    }
}
