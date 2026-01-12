//! Gradient checkpointing utilities.

use pmetal_core::CheckpointStrategy;

/// Gradient checkpointer for memory-efficient training.
pub struct GradientCheckpointer {
    strategy: CheckpointStrategy,
}

impl GradientCheckpointer {
    /// Create a new gradient checkpointer.
    pub fn new(strategy: CheckpointStrategy) -> Self {
        Self { strategy }
    }

    /// Check if a layer should be checkpointed.
    pub fn should_checkpoint(&self, layer_idx: usize, total_layers: usize) -> bool {
        match &self.strategy {
            CheckpointStrategy::None => false,
            CheckpointStrategy::EveryN(n) => layer_idx % n == 0,
            CheckpointStrategy::Smart => {
                // Checkpoint first and last quarters, skip middle
                let quarter = total_layers / 4;
                layer_idx < quarter || layer_idx >= total_layers - quarter
            }
            CheckpointStrategy::SelectiveAttention => {
                // Always checkpoint (attention layers only)
                true
            }
        }
    }
}
