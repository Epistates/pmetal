//! NEFTune: Noisy Embeddings for Fine-Tuning.
//!
//! Implementation of NEFTune from "NEFTune: Noisy Embeddings Improve Instruction Finetuning"
//! (Jain et al., 2023). This technique adds uniform noise to embeddings during training,
//! which acts as a regularizer and has been shown to consistently improve model quality.
//!
//! ## How It Works
//!
//! During training, uniform noise is added to the embedding output:
//!
//! ```text
//! noisy_embeds = embeds + noise * (alpha / sqrt(seq_len * hidden_dim))
//! ```
//!
//! Where:
//! - `alpha` is the noise scale (typically 5-15)
//! - `noise` is uniform random in [-1, 1]
//! - The denominator normalizes by sequence length and hidden dimension
//!
//! ## Benefits
//!
//! - Simple to implement (single line change to embedding forward)
//! - Consistent improvements across models and tasks
//! - No additional hyperparameters to tune (alpha=5 works well generally)
//! - Zero inference overhead (noise only added during training)
//!
//! ## Recommended Values
//!
//! | Model Size | Recommended Alpha |
//! |------------|-------------------|
//! | 7B         | 5                 |
//! | 13B        | 5                 |
//! | 70B        | 10-15             |
//!
//! ## Usage
//!
//! ```ignore
//! let config = NEFTuneConfig::default(); // alpha=5
//! let noisy_embeds = apply_neftune(&embeds, &config)?;
//! ```

use mlx_rs::{error::Exception, Array};

/// Configuration for NEFTune.
#[derive(Debug, Clone)]
pub struct NEFTuneConfig {
    /// Noise scaling factor (alpha).
    /// Higher values = more noise. Typical values: 5-15.
    pub alpha: f32,
    /// Whether NEFTune is enabled.
    pub enabled: bool,
}

impl Default for NEFTuneConfig {
    fn default() -> Self {
        Self {
            alpha: 5.0,
            enabled: true,
        }
    }
}

impl NEFTuneConfig {
    /// Create a new NEFTune config with the specified alpha.
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            enabled: true,
        }
    }

    /// Create a disabled NEFTune config.
    pub fn disabled() -> Self {
        Self {
            alpha: 0.0,
            enabled: false,
        }
    }

    /// Set the alpha value.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Enable or disable NEFTune.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

/// Apply NEFTune noise to embeddings.
///
/// Adds uniform noise scaled by `alpha / sqrt(seq_len * hidden_dim)` to the embeddings.
/// This is only applied during training; for inference, use the original embeddings.
///
/// # Arguments
/// * `embeddings` - Input embeddings [batch, seq_len, hidden_dim]
/// * `config` - NEFTune configuration
///
/// # Returns
/// Embeddings with noise added (or original if disabled).
pub fn apply_neftune(embeddings: &Array, config: &NEFTuneConfig) -> Result<Array, Exception> {
    if !config.enabled || config.alpha == 0.0 {
        return Ok(embeddings.clone());
    }

    let shape = embeddings.shape();
    if shape.len() < 2 {
        return Err(Exception::custom(
            "NEFTune requires at least 2D embeddings [seq, hidden] or [batch, seq, hidden]",
        ));
    }

    // Get seq_len and hidden_dim
    let (seq_len, hidden_dim) = if shape.len() == 2 {
        (shape[0] as f32, shape[1] as f32)
    } else {
        (shape[shape.len() - 2] as f32, shape[shape.len() - 1] as f32)
    };

    // Compute noise scale
    let magnitude = config.alpha / (seq_len * hidden_dim).sqrt();

    // Generate uniform noise in [-1, 1]
    let noise = mlx_rs::random::uniform::<_, f32>(-1.0f32, 1.0f32, shape, None)?;

    // Scale noise
    let scaled_noise = noise.multiply(Array::from_f32(magnitude))?;

    // Add to embeddings
    embeddings.add(&scaled_noise)
}

/// NEFTune-aware embedding wrapper.
///
/// Wraps an embedding lookup with optional NEFTune noise injection.
/// Automatically handles training vs inference mode.
#[derive(Debug)]
pub struct NEFTuneEmbedding {
    /// NEFTune configuration.
    config: NEFTuneConfig,
    /// Whether currently in training mode.
    training: bool,
}

impl NEFTuneEmbedding {
    /// Create a new NEFTune embedding wrapper.
    pub fn new(config: NEFTuneConfig) -> Self {
        Self {
            config,
            training: true,
        }
    }

    /// Set training mode.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set evaluation mode.
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Check if in training mode.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Get the configuration.
    pub fn config(&self) -> &NEFTuneConfig {
        &self.config
    }

    /// Apply NEFTune to embeddings if in training mode.
    ///
    /// # Arguments
    /// * `embeddings` - Embedding output [batch, seq, hidden]
    ///
    /// # Returns
    /// Embeddings with noise if training, original otherwise.
    pub fn forward(&self, embeddings: &Array) -> Result<Array, Exception> {
        if self.training && self.config.enabled {
            apply_neftune(embeddings, &self.config)
        } else {
            Ok(embeddings.clone())
        }
    }
}

impl Default for NEFTuneEmbedding {
    fn default() -> Self {
        Self::new(NEFTuneConfig::default())
    }
}

/// Compute recommended alpha based on model size.
///
/// Based on empirical results from the NEFTune paper:
/// - 7B models: alpha = 5
/// - 13B models: alpha = 5
/// - 70B+ models: alpha = 10-15
pub fn recommended_alpha(num_parameters: u64) -> f32 {
    if num_parameters < 10_000_000_000 {
        // < 10B
        5.0
    } else if num_parameters < 50_000_000_000 {
        // 10B - 50B
        10.0
    } else {
        // 50B+
        15.0
    }
}

/// Estimate number of parameters from hidden dim and num layers.
///
/// Rough estimate: params ≈ 12 * L * d² for transformer
/// where L = num_layers, d = hidden_dim
pub fn estimate_params(num_layers: usize, hidden_dim: usize) -> u64 {
    12 * num_layers as u64 * hidden_dim as u64 * hidden_dim as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neftune_config_default() {
        let config = NEFTuneConfig::default();
        assert_eq!(config.alpha, 5.0);
        assert!(config.enabled);
    }

    #[test]
    fn test_neftune_config_disabled() {
        let config = NEFTuneConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_neftune_config_builder() {
        let config = NEFTuneConfig::new(10.0).with_enabled(true);
        assert_eq!(config.alpha, 10.0);
        assert!(config.enabled);
    }

    #[test]
    fn test_apply_neftune_shape() {
        let config = NEFTuneConfig::default();
        let embeds = Array::zeros::<f32>(&[2, 10, 768]).unwrap();

        let noisy = apply_neftune(&embeds, &config).unwrap();

        assert_eq!(noisy.shape(), embeds.shape());
    }

    #[test]
    fn test_apply_neftune_adds_noise() {
        let config = NEFTuneConfig::new(5.0);
        let embeds = Array::zeros::<f32>(&[2, 10, 768]).unwrap();

        let noisy = apply_neftune(&embeds, &config).unwrap();
        noisy.eval().unwrap();

        // Sum should not be zero (noise was added)
        let sum = noisy.abs().unwrap().sum(None).unwrap();
        sum.eval().unwrap();
        let sum_val = sum.item::<f32>();
        assert!(sum_val > 0.0);
    }

    #[test]
    fn test_apply_neftune_disabled() {
        let config = NEFTuneConfig::disabled();
        let embeds = Array::ones::<f32>(&[2, 10, 768]).unwrap();

        let result = apply_neftune(&embeds, &config).unwrap();
        result.eval().unwrap();
        embeds.eval().unwrap();

        // Should be exactly the same
        let diff = result.subtract(&embeds).unwrap();
        let sum = diff.abs().unwrap().sum(None).unwrap();
        sum.eval().unwrap();
        assert!(sum.item::<f32>() < 1e-6);
    }

    #[test]
    fn test_neftune_noise_magnitude() {
        // The noise magnitude should be alpha / sqrt(seq_len * hidden_dim)
        let alpha = 5.0;
        let config = NEFTuneConfig::new(alpha);

        let seq_len = 100;
        let hidden_dim = 1000;
        let embeds = Array::zeros::<f32>(&[1, seq_len, hidden_dim]).unwrap();

        let noisy = apply_neftune(&embeds, &config).unwrap();
        noisy.eval().unwrap();

        // Expected magnitude: alpha / sqrt(seq_len * hidden_dim)
        let expected_magnitude = alpha / ((seq_len * hidden_dim) as f32).sqrt();

        // Max value should be around the expected magnitude
        let max_val = noisy.abs().unwrap().max(None).unwrap();
        max_val.eval().unwrap();
        let max_f32 = max_val.item::<f32>();

        // Should be close to expected magnitude (uniform in [-mag, mag])
        assert!(max_f32 < expected_magnitude * 1.5);
        assert!(max_f32 > expected_magnitude * 0.5);
    }

    #[test]
    fn test_neftune_embedding_train_eval() {
        let mut wrapper = NEFTuneEmbedding::new(NEFTuneConfig::default());

        assert!(wrapper.is_training());

        wrapper.eval();
        assert!(!wrapper.is_training());

        wrapper.train();
        assert!(wrapper.is_training());
    }

    #[test]
    fn test_neftune_embedding_forward_training() {
        let wrapper = NEFTuneEmbedding::new(NEFTuneConfig::default());
        let embeds = Array::zeros::<f32>(&[2, 10, 768]).unwrap();

        let output = wrapper.forward(&embeds).unwrap();
        output.eval().unwrap();

        // Should have noise added
        let sum = output.abs().unwrap().sum(None).unwrap();
        sum.eval().unwrap();
        assert!(sum.item::<f32>() > 0.0);
    }

    #[test]
    fn test_neftune_embedding_forward_eval() {
        let mut wrapper = NEFTuneEmbedding::new(NEFTuneConfig::default());
        wrapper.eval();

        let embeds = Array::ones::<f32>(&[2, 10, 768]).unwrap();
        let output = wrapper.forward(&embeds).unwrap();

        output.eval().unwrap();
        embeds.eval().unwrap();

        // Should be unchanged
        let diff = output.subtract(&embeds).unwrap();
        let sum = diff.abs().unwrap().sum(None).unwrap();
        sum.eval().unwrap();
        assert!(sum.item::<f32>() < 1e-6);
    }

    #[test]
    fn test_recommended_alpha() {
        // 7B model
        assert_eq!(recommended_alpha(7_000_000_000), 5.0);
        // 13B model
        assert_eq!(recommended_alpha(13_000_000_000), 10.0);
        // 70B model
        assert_eq!(recommended_alpha(70_000_000_000), 15.0);
    }

    #[test]
    fn test_estimate_params() {
        // Roughly check estimate is in right ballpark
        // Llama 7B: ~32 layers, 4096 hidden
        let params = estimate_params(32, 4096);
        assert!(params > 5_000_000_000); // > 5B
        assert!(params < 10_000_000_000); // < 10B
    }

    #[test]
    fn test_neftune_2d_input() {
        // Should work with 2D input [seq, hidden]
        let config = NEFTuneConfig::default();
        let embeds = Array::zeros::<f32>(&[10, 768]).unwrap();

        let noisy = apply_neftune(&embeds, &config).unwrap();
        assert_eq!(noisy.shape(), embeds.shape());
    }

    #[test]
    fn test_neftune_invalid_input() {
        // Should fail with 1D input
        let config = NEFTuneConfig::default();
        let embeds = Array::zeros::<f32>(&[768]).unwrap();

        let result = apply_neftune(&embeds, &config);
        assert!(result.is_err());
    }
}
