//! Gradient checkpointing configuration.
//!
//! The mechanism lives in
//! [`pmetal_bridge::compat::checkpointed`](pmetal_bridge::compat::checkpointed),
//! which runs a module's forward under `mlx::core::checkpoint`.
//!
//! This module used to carry an implementation built on a different idea: call
//! `eval()` at layer boundaries and let MLX recompute what it needs on the way
//! back. That does not checkpoint anything. `eval()` materialises a tensor, but
//! the graph behind it is still held for the backward pass, so nothing is freed
//! and nothing is recomputed. Nothing ever called it, which is the other half
//! of why it went unnoticed.
//!
//! `layers_per_block` has no counterpart in the real mechanism. Both PyTorch
//! and mlx-lm checkpoint one decoder layer at a time; the knob only ever chose
//! which trace lines got printed.

/// Configuration for gradient checkpointing.
///
/// Retained for the per-architecture LoRA models, which carry it as an inert
/// flag and are being replaced by `AdaptedModel`. New code should not reach for
/// it.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Enable gradient checkpointing.
    pub enabled: bool,
    /// Number of layers per checkpoint block. Inert: see the module docs.
    pub layers_per_block: usize,
    /// Force evaluation at checkpoint boundaries. Inert: see the module docs.
    pub eval_at_boundaries: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            layers_per_block: 2,
            eval_at_boundaries: true,
        }
    }
}

impl CheckpointConfig {
    /// Create a new checkpoint config with checkpointing enabled.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpointing_is_off_unless_asked_for() {
        assert!(!CheckpointConfig::default().enabled);
        assert!(CheckpointConfig::enabled().enabled);
    }
}
