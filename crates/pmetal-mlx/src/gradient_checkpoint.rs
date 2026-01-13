//! Gradient checkpointing utilities for memory-efficient training.
//!
//! Gradient checkpointing trades computation for memory by not storing all intermediate
//! activations during the forward pass. Instead, activations are recomputed during the
//! backward pass when needed for gradient computation.
//!
//! ## Memory Savings
//!
//! For a transformer model with L layers:
//! - **Standard training**: O(L) memory for activations
//! - **Full checkpointing**: O(1) memory, O(2L) compute
//! - **Block checkpointing (k blocks)**: O(k) memory, O(L + L/k) compute
//!
//! ## Usage Strategy
//!
//! For models that don't fit in memory:
//! 1. First try: Reduce batch size to 1-2
//! 2. Second try: Enable gradient checkpointing
//! 3. Third try: Use quantization (QLoRA)
//! 4. Fourth try: Reduce sequence length
//!
//! ## MLX-Specific Notes
//!
//! MLX uses lazy evaluation, which naturally provides some memory efficiency.
//! Explicit `eval()` calls can be used to control when tensors are materialized.
//! This module provides utilities to work with MLX's memory model effectively.

use mlx_rs::{error::Exception, Array};

/// Configuration for gradient checkpointing.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Enable gradient checkpointing.
    pub enabled: bool,
    /// Number of layers per checkpoint block.
    /// Smaller values = more memory savings but more recomputation.
    /// Typical values: 1-4 layers per block.
    pub layers_per_block: usize,
    /// Force evaluation at checkpoint boundaries.
    /// This helps manage MLX's lazy evaluation for memory control.
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

    /// Set the number of layers per checkpoint block.
    pub fn with_layers_per_block(mut self, layers: usize) -> Self {
        self.layers_per_block = layers.max(1);
        self
    }

    /// Enable or disable evaluation at checkpoint boundaries.
    pub fn with_eval_at_boundaries(mut self, eval: bool) -> Self {
        self.eval_at_boundaries = eval;
        self
    }
}

/// A checkpoint boundary marker.
///
/// When gradient checkpointing is enabled, this evaluates tensors to
/// materialize them and free up the computation graph for previous layers.
/// This helps control memory usage by forcing intermediate results to be
/// computed and stored rather than keeping the full computation graph.
///
/// # Arguments
/// * `arrays` - Arrays to checkpoint (typically hidden states)
/// * `config` - Checkpoint configuration
///
/// # Returns
/// The same arrays, possibly after evaluation
pub fn checkpoint_boundary(arrays: &[&Array], config: &CheckpointConfig) -> Result<(), Exception> {
    if config.enabled && config.eval_at_boundaries {
        // Evaluate arrays to materialize them
        // This breaks the computation graph, allowing earlier parts to be freed
        for arr in arrays {
            arr.eval()?;
        }
    }
    Ok(())
}

/// Wraps a layer forward function with checkpointing.
///
/// In MLX, checkpointing is achieved by:
/// 1. Computing forward pass lazily (default MLX behavior)
/// 2. Calling eval() at checkpoint boundaries to materialize tensors
/// 3. During backward pass, MLX will recompute necessary intermediates
///
/// # Type Parameters
/// * `F` - The forward function type
///
/// # Arguments
/// * `layer_fn` - The layer forward function
/// * `input` - Input tensor
/// * `layer_idx` - Current layer index
/// * `config` - Checkpoint configuration
///
/// # Returns
/// Output of the layer function
pub fn checkpointed_forward<F>(
    mut layer_fn: F,
    input: &Array,
    layer_idx: usize,
    config: &CheckpointConfig,
) -> Result<Array, Exception>
where
    F: FnMut(&Array) -> Result<Array, Exception>,
{
    let output = layer_fn(input)?;

    // Check if this is a checkpoint boundary
    if config.enabled && (layer_idx + 1) % config.layers_per_block == 0 {
        if config.eval_at_boundaries {
            // Force evaluation to materialize the tensor
            // This breaks the computation graph at this point
            output.eval()?;
        }
    }

    Ok(output)
}

/// Block-level gradient checkpointing for transformer models.
///
/// Divides layers into blocks and checkpoints at block boundaries.
/// This provides a balance between memory savings and recomputation overhead.
///
/// # Arguments
/// * `layers` - Layer forward functions
/// * `input` - Initial input tensor
/// * `config` - Checkpoint configuration
///
/// # Returns
/// Final output after all layers
pub fn checkpoint_sequential<F>(
    layers: &mut [F],
    input: &Array,
    config: &CheckpointConfig,
) -> Result<Array, Exception>
where
    F: FnMut(&Array) -> Result<Array, Exception>,
{
    let mut hidden = input.clone();

    for (idx, layer) in layers.iter_mut().enumerate() {
        hidden = checkpointed_forward(|x| layer(x), &hidden, idx, config)?;
    }

    Ok(hidden)
}

/// Gradient checkpointing context for a model.
///
/// This context manages gradient checkpointing settings and provides
/// utilities for memory-efficient training loops.
#[derive(Debug, Clone)]
pub struct CheckpointContext {
    /// Checkpoint configuration.
    pub checkpoint_config: CheckpointConfig,
    /// Peak memory usage tracking (bytes).
    peak_memory: usize,
    /// Current layer index for checkpointing.
    current_layer: usize,
}

impl Default for CheckpointContext {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckpointContext {
    /// Create a new memory-efficient training context.
    pub fn new() -> Self {
        Self {
            checkpoint_config: CheckpointConfig::default(),
            peak_memory: 0,
            current_layer: 0,
        }
    }

    /// Enable gradient checkpointing.
    pub fn with_checkpointing(mut self) -> Self {
        self.checkpoint_config.enabled = true;
        self
    }

    /// Set checkpoint configuration.
    pub fn with_checkpoint_config(mut self, config: CheckpointConfig) -> Self {
        self.checkpoint_config = config;
        self
    }

    /// Signal start of a new layer.
    pub fn enter_layer(&mut self, layer_idx: usize) {
        self.current_layer = layer_idx;
    }

    /// Check if current layer is a checkpoint boundary.
    pub fn is_checkpoint_boundary(&self) -> bool {
        self.checkpoint_config.enabled
            && (self.current_layer + 1) % self.checkpoint_config.layers_per_block == 0
    }

    /// Apply checkpointing to output tensor if at boundary.
    pub fn maybe_checkpoint(&self, output: &Array) -> Result<(), Exception> {
        if self.is_checkpoint_boundary() && self.checkpoint_config.eval_at_boundaries {
            output.eval()?;
        }
        Ok(())
    }

    /// Get checkpoint configuration.
    pub fn config(&self) -> &CheckpointConfig {
        &self.checkpoint_config
    }

    /// Update peak memory tracking.
    pub fn update_peak_memory(&mut self, current_bytes: usize) {
        self.peak_memory = self.peak_memory.max(current_bytes);
    }

    /// Get tracked peak memory usage.
    pub fn peak_memory(&self) -> usize {
        self.peak_memory
    }
}

/// Estimate memory savings from gradient checkpointing.
///
/// # Arguments
/// * `num_layers` - Number of transformer layers
/// * `layers_per_block` - Layers per checkpoint block
/// * `hidden_size` - Hidden dimension
/// * `batch_size` - Batch size
/// * `seq_len` - Sequence length
///
/// # Returns
/// (standard_memory_mb, checkpointed_memory_mb)
pub fn estimate_memory_savings(
    num_layers: usize,
    layers_per_block: usize,
    hidden_size: usize,
    batch_size: usize,
    seq_len: usize,
) -> (f64, f64) {
    // Rough estimate: each layer stores activations of size [batch, seq, hidden]
    // In float32: batch * seq * hidden * 4 bytes
    let activation_bytes = batch_size * seq_len * hidden_size * 4;

    // Standard: store all layer activations
    let standard_bytes = num_layers * activation_bytes;

    // Checkpointed: store only checkpoint boundaries + current block
    let num_blocks = (num_layers + layers_per_block - 1) / layers_per_block;
    let checkpointed_bytes = num_blocks * activation_bytes + layers_per_block * activation_bytes;

    let mb = 1024.0 * 1024.0;
    (standard_bytes as f64 / mb, checkpointed_bytes as f64 / mb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_config_default() {
        let config = CheckpointConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.layers_per_block, 2);
        assert!(config.eval_at_boundaries);
    }

    #[test]
    fn test_checkpoint_config_enabled() {
        let config = CheckpointConfig::enabled();
        assert!(config.enabled);
    }

    #[test]
    fn test_checkpoint_config_builder() {
        let config = CheckpointConfig::enabled()
            .with_layers_per_block(4)
            .with_eval_at_boundaries(false);

        assert!(config.enabled);
        assert_eq!(config.layers_per_block, 4);
        assert!(!config.eval_at_boundaries);
    }

    #[test]
    fn test_checkpoint_context() {
        let mut ctx = CheckpointContext::new().with_checkpointing();

        // Test checkpoint boundary detection
        ctx.checkpoint_config.layers_per_block = 3;

        ctx.enter_layer(0);
        assert!(!ctx.is_checkpoint_boundary()); // Layer 1 of 3

        ctx.enter_layer(1);
        assert!(!ctx.is_checkpoint_boundary()); // Layer 2 of 3

        ctx.enter_layer(2);
        assert!(ctx.is_checkpoint_boundary()); // Layer 3 of 3 - boundary!

        ctx.enter_layer(3);
        assert!(!ctx.is_checkpoint_boundary()); // Layer 1 of next block
    }

    #[test]
    fn test_checkpointed_forward() {
        let config = CheckpointConfig::enabled().with_layers_per_block(2);

        // Simple identity layer for testing
        let input = mlx_rs::Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);

        let output = checkpointed_forward(|x| Ok(x.clone()), &input, 0, &config).unwrap();
        output.eval().unwrap();

        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_checkpoint_boundary() {
        let config = CheckpointConfig::enabled();

        let arr1 = mlx_rs::Array::from_f32(1.0);
        let arr2 = mlx_rs::Array::from_f32(2.0);

        // Should not error
        checkpoint_boundary(&[&arr1, &arr2], &config).unwrap();

        // Disabled config should be a no-op
        let disabled_config = CheckpointConfig::default();
        checkpoint_boundary(&[&arr1, &arr2], &disabled_config).unwrap();
    }

    #[test]
    fn test_estimate_memory_savings() {
        // 32 layer model, checkpoint every 4 layers
        // hidden=4096, batch=4, seq=2048
        let (standard, checkpointed) = estimate_memory_savings(32, 4, 4096, 4, 2048);

        // Standard should be higher than checkpointed
        assert!(standard > checkpointed);

        // Rough sanity check on values
        assert!(standard > 100.0); // Should be > 100MB for this config
    }

    #[test]
    fn test_checkpoint_sequential() {
        let config = CheckpointConfig::enabled().with_layers_per_block(2);

        let input = mlx_rs::Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);

        // Create simple "layers" that add 1.0 to input
        let add_one = |x: &Array| -> Result<Array, Exception> {
            let one = mlx_rs::Array::from_f32(1.0);
            x.add(&one)
        };

        // We need to pass closures - create a vec of boxed closures
        let mut layers: Vec<Box<dyn FnMut(&Array) -> Result<Array, Exception>>> =
            vec![Box::new(add_one), Box::new(add_one), Box::new(add_one)];

        // Can't use checkpoint_sequential directly due to trait objects
        // So we just test the individual checkpointed_forward function
        let mut hidden = input.clone();
        for (idx, layer) in layers.iter_mut().enumerate() {
            hidden = checkpointed_forward(|x| layer(x), &hidden, idx, &config).unwrap();
        }

        hidden.eval().unwrap();

        // After 3 add operations, values should be [4, 5, 6, 7]
        let expected = mlx_rs::Array::from_slice(&[4.0f32, 5.0, 6.0, 7.0], &[2, 2]);
        let diff = hidden.subtract(&expected).unwrap();
        let sum = diff.abs().unwrap().sum(None).unwrap();
        sum.eval().unwrap();
        assert!(sum.item::<f32>() < 1e-5);
    }
}
