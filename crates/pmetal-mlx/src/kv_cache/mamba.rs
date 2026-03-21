//! Mamba SSM state cache for hybrid architectures.

use mlx_rs::{Array, error::Exception, ops::concatenate_axis, ops::indexing::IndexOp};

/// Cache for Mamba-2 SSM state during autoregressive generation.
///
/// Mamba layers require two types of state for incremental generation:
/// 1. **Conv state**: Last (kernel_size - 1) conv1d inputs for causal convolution
/// 2. **SSM state**: The hidden state matrix from the state space model
///
/// Without this cache, each generated token is processed without context from
/// previous tokens through Mamba layers, producing incoherent output.
#[derive(Debug, Clone)]
pub struct MambaCache {
    /// Per-layer cache entries.
    /// Each entry is (conv_state, ssm_state) where both may be None initially.
    layers: Vec<MambaCacheEntry>,
}

/// Cache entry for a single Mamba layer.
#[derive(Debug, Clone, Default)]
pub struct MambaCacheEntry {
    /// Convolutional state - last (kernel_size - 1) inputs.
    /// Shape: [batch, kernel_size - 1, conv_dim]
    pub conv_state: Option<Array>,
    /// SSM hidden state.
    /// Shape: [batch, num_heads, head_dim, state_dim]
    pub ssm_state: Option<Array>,
}

impl MambaCache {
    /// Create a new Mamba cache with the specified number of layers.
    pub fn new(num_layers: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| MambaCacheEntry::default())
            .collect();
        Self { layers }
    }

    /// Get a mutable reference to a layer's cache entry.
    pub fn get_mut(&mut self, layer_idx: usize) -> Option<&mut MambaCacheEntry> {
        self.layers.get_mut(layer_idx)
    }

    /// Get an immutable reference to a layer's cache entry.
    pub fn get(&self, layer_idx: usize) -> Option<&MambaCacheEntry> {
        self.layers.get(layer_idx)
    }

    /// Reset all cache entries to None.
    pub fn reset(&mut self) {
        for entry in &mut self.layers {
            entry.conv_state = None;
            entry.ssm_state = None;
        }
    }

    /// Check if the cache is empty (no state stored).
    pub fn is_empty(&self) -> bool {
        self.layers
            .iter()
            .all(|e| e.conv_state.is_none() && e.ssm_state.is_none())
    }
}

impl MambaCacheEntry {
    /// Update the conv state with new input, returning the padded input for conv1d.
    ///
    /// This implements causal convolution by:
    /// 1. Concatenating stored state with new input
    /// 2. Storing the last (kernel_size - 1) values for next call
    /// 3. Returning the padded input for conv1d processing
    ///
    /// # Arguments
    /// * `input` - New input tensor [batch, seq_len, conv_dim]
    /// * `kernel_size` - Conv1d kernel size
    ///
    /// # Returns
    /// Padded input [batch, seq_len + kernel_size - 1, conv_dim]
    pub fn update_conv_state(
        &mut self,
        input: &Array,
        kernel_size: i32,
    ) -> Result<Array, Exception> {
        let pad_len = (kernel_size - 1) as usize;
        let shape = input.shape();
        let batch = shape[0] as i32;
        let conv_dim = shape[2] as i32;

        // Get or initialize conv state with matching dtype
        let conv_state = if let Some(ref state) = self.conv_state {
            state.clone()
        } else {
            // Initialize to zeros with shape [batch, pad_len, conv_dim]
            // Use same dtype as input to avoid dtype mismatch issues
            Array::zeros::<f32>(&[batch, pad_len as i32, conv_dim])?.as_dtype(input.dtype())?
        };

        // Concatenate state with new input along sequence dimension
        let padded = concatenate_axis(&[&conv_state, input], 1)?;

        // Store last (kernel_size - 1) values for next call
        let seq_len = padded.dim(1);
        let start_idx = seq_len - pad_len as i32;
        self.conv_state = Some(padded.index((.., start_idx.., ..)));

        Ok(padded)
    }

    /// Get the current SSM state, returning None if not initialized.
    pub fn get_ssm_state(&self) -> Option<&Array> {
        self.ssm_state.as_ref()
    }

    /// Update the SSM state.
    pub fn set_ssm_state(&mut self, state: Array) {
        self.ssm_state = Some(state);
    }
}
