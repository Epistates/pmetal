//! Configuration for Manifold-Constrained Hyper-Connections (mHC).
//!
//! This module provides configuration structures for mHC layers,
//! including expansion rate, Sinkhorn iterations, and initialization parameters.

use serde::{Deserialize, Serialize};

/// Configuration for mHC (Manifold-Constrained Hyper-Connections).
///
/// Based on the DeepSeek paper arXiv:2512.24880.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MhcConfig {
    /// Expansion rate (number of parallel streams).
    ///
    /// The residual stream is expanded from C to n×C dimensions.
    /// Paper uses n=4 for all experiments.
    pub expansion_rate: usize,

    /// Number of Sinkhorn-Knopp iterations for doubly stochastic projection.
    ///
    /// Higher values give better approximation but cost more compute.
    /// Paper uses t_max=20.
    pub sinkhorn_iterations: usize,

    /// Gating factor initialization value (α).
    ///
    /// The dynamic mapping coefficients are scaled by learnable α values
    /// initialized to this small value. Paper uses 0.01.
    pub alpha_init: f32,

    /// Hidden dimension (C) of the model.
    ///
    /// This is the per-stream dimension, not the total expanded dimension.
    pub hidden_dim: usize,

    /// Whether to use input-dependent (dynamic) mappings.
    ///
    /// When true, mappings depend on the input via learned projections.
    /// When false, only static biases are used.
    pub use_dynamic_mappings: bool,

    /// Recomputation block size (L_r) for activation checkpointing.
    ///
    /// If None, automatically computed as sqrt(n*L/(n+2)).
    /// If Some, uses the specified value.
    pub recompute_block_size: Option<usize>,

    /// Epsilon for numerical stability in RMSNorm and Sinkhorn.
    pub epsilon: f32,

    /// Whether to use mixed precision (BF16 activations, FP32 mappings).
    pub use_mixed_precision: bool,

    /// Whether to fuse kernels for efficiency.
    pub fuse_kernels: bool,
}

impl Default for MhcConfig {
    fn default() -> Self {
        Self {
            expansion_rate: 4,
            sinkhorn_iterations: 20,
            alpha_init: 0.01,
            hidden_dim: 2560, // 27B model default
            use_dynamic_mappings: true,
            recompute_block_size: None,
            epsilon: 1e-6,
            use_mixed_precision: true,
            fuse_kernels: true,
        }
    }
}

impl MhcConfig {
    /// Create a config for a 3B parameter model.
    pub fn config_3b() -> Self {
        Self {
            hidden_dim: 1280,
            ..Default::default()
        }
    }

    /// Create a config for a 9B parameter model.
    pub fn config_9b() -> Self {
        Self {
            hidden_dim: 1920,
            ..Default::default()
        }
    }

    /// Create a config for a 27B parameter model.
    pub fn config_27b() -> Self {
        Self {
            hidden_dim: 2560,
            ..Default::default()
        }
    }

    /// Compute the optimal recomputation block size.
    ///
    /// Based on Eq. 20 from the paper:
    /// L*_r = sqrt(n*L/(n+2))
    pub fn optimal_recompute_block_size(&self, num_layers: usize) -> usize {
        let n = self.expansion_rate as f64;
        let l = num_layers as f64;
        let optimal = ((n * l) / (n + 2.0)).sqrt();
        optimal.ceil() as usize
    }

    /// Get the total expanded dimension (n × C).
    pub fn expanded_dim(&self) -> usize {
        self.expansion_rate * self.hidden_dim
    }

    /// Get the number of parameters in H^res (n × n).
    pub fn res_mapping_size(&self) -> usize {
        self.expansion_rate * self.expansion_rate
    }

    /// Get the number of parameters in H^pre or H^post (1 × n).
    pub fn pre_post_mapping_size(&self) -> usize {
        self.expansion_rate
    }

    /// Create configuration from a preset.
    pub fn from_preset(preset: MhcPreset) -> Self {
        preset.to_config()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), MhcConfigError> {
        if self.expansion_rate == 0 {
            return Err(MhcConfigError::InvalidExpansionRate(self.expansion_rate));
        }
        if self.expansion_rate > 16 {
            return Err(MhcConfigError::ExpansionRateTooLarge(self.expansion_rate));
        }
        if self.sinkhorn_iterations == 0 {
            return Err(MhcConfigError::InvalidSinkhornIterations(
                self.sinkhorn_iterations,
            ));
        }
        if self.hidden_dim == 0 {
            return Err(MhcConfigError::InvalidHiddenDim(self.hidden_dim));
        }
        if self.alpha_init <= 0.0 || self.alpha_init >= 1.0 {
            return Err(MhcConfigError::InvalidAlphaInit(self.alpha_init));
        }
        if self.epsilon <= 0.0 {
            return Err(MhcConfigError::InvalidEpsilon(self.epsilon));
        }
        Ok(())
    }
}

/// Configuration errors for mHC.
#[derive(Debug, Clone, thiserror::Error)]
pub enum MhcConfigError {
    #[error("Invalid expansion rate: {0} (must be > 0)")]
    InvalidExpansionRate(usize),

    #[error("Expansion rate too large: {0} (max 16)")]
    ExpansionRateTooLarge(usize),

    #[error("Invalid Sinkhorn iterations: {0} (must be > 0)")]
    InvalidSinkhornIterations(usize),

    #[error("Invalid hidden dimension: {0} (must be > 0)")]
    InvalidHiddenDim(usize),

    #[error("Invalid alpha init: {0} (must be in (0, 1))")]
    InvalidAlphaInit(f32),

    #[error("Invalid epsilon: {0} (must be > 0)")]
    InvalidEpsilon(f32),
}

/// Presets for different model scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MhcPreset {
    /// 3B parameter model (1280 hidden dim)
    Small,
    /// 9B parameter model (1920 hidden dim)
    Medium,
    /// 27B parameter model (2560 hidden dim)
    Large,
    /// Custom configuration
    Custom,
}

impl MhcPreset {
    /// Convert preset to configuration.
    pub fn to_config(self) -> MhcConfig {
        match self {
            Self::Small => MhcConfig::config_3b(),
            Self::Medium => MhcConfig::config_9b(),
            Self::Large => MhcConfig::config_27b(),
            Self::Custom => MhcConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MhcConfig::default();
        assert_eq!(config.expansion_rate, 4);
        assert_eq!(config.sinkhorn_iterations, 20);
        assert!((config.alpha_init - 0.01).abs() < 1e-6);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_expanded_dim() {
        let config = MhcConfig {
            expansion_rate: 4,
            hidden_dim: 2560,
            ..Default::default()
        };
        assert_eq!(config.expanded_dim(), 10240);
    }

    #[test]
    fn test_optimal_recompute_block_size() {
        let config = MhcConfig::default();
        // For n=4, L=30: sqrt(4*30/6) = sqrt(20) ≈ 4.47 → 5
        let block_size = config.optimal_recompute_block_size(30);
        assert!(block_size >= 4 && block_size <= 6);
    }

    #[test]
    fn test_validation_errors() {
        let mut config = MhcConfig::default();

        config.expansion_rate = 0;
        assert!(matches!(
            config.validate(),
            Err(MhcConfigError::InvalidExpansionRate(0))
        ));

        config.expansion_rate = 4;
        config.sinkhorn_iterations = 0;
        assert!(matches!(
            config.validate(),
            Err(MhcConfigError::InvalidSinkhornIterations(0))
        ));
    }

    #[test]
    fn test_presets() {
        let small = MhcPreset::Small.to_config();
        assert_eq!(small.hidden_dim, 1280);

        let medium = MhcPreset::Medium.to_config();
        assert_eq!(medium.hidden_dim, 1920);

        let large = MhcPreset::Large.to_config();
        assert_eq!(large.hidden_dim, 2560);
    }
}
