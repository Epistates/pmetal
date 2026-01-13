//! Trait for trainable LoRA/QLoRA models.
//!
//! This trait provides a common interface for models that can be trained
//! with gradient descent, supporting both standard LoRA and QLoRA.

use std::collections::HashMap;
use std::rc::Rc;

use mlx_rs::module::ModuleParameters;
use mlx_rs::Array;
use pmetal_mlx::kv_cache::KVCache;

use crate::LoraError;

/// Trait for models that can be trained with LoRA/QLoRA.
///
/// This trait combines the requirements for:
/// - Forward pass (computing logits from input IDs)
/// - Parameter access (for autodiff and checkpointing)
/// - LoRA adapter management
///
/// Both `LlamaLoraForCausalLM` and `LlamaQloraForCausalLM` implement this trait.
pub trait TrainableModel: ModuleParameters {
    /// Perform forward pass through the model.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// Logits [batch, seq_len, vocab_size]
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError>;

    /// Perform forward pass with explicit position IDs.
    ///
    /// This is used for packed sequence training where position IDs
    /// need to reset at sequence boundaries.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `mask` - Optional attention mask
    /// * `position_ids` - Position indices [seq_len]
    ///
    /// # Returns
    /// Logits [batch, seq_len, vocab_size]
    fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Result<Array, LoraError> {
        // Default implementation: ignore position_ids and use standard forward
        // Models that support packed sequences should override this
        let _ = position_ids;
        self.forward(input_ids, mask)
    }

    /// Perform forward pass for Vision-Language Models with image inputs.
    ///
    /// This is used for VLM training (e.g., Llama 3.2 Vision) where the model
    /// receives both text tokens and image pixel values.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `mask` - Optional attention mask
    /// * `pixel_values` - Optional image pixel values [batch, channels, height, width]
    ///
    /// # Returns
    /// Logits [batch, seq_len, vocab_size]
    fn forward_with_images(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        pixel_values: Option<&Array>,
    ) -> Result<Array, LoraError> {
        // Default implementation: ignore pixel_values for text-only models
        let _ = pixel_values;
        self.forward(input_ids, mask)
    }

    /// Check if this model supports multimodal (image+text) inputs.
    fn is_multimodal(&self) -> bool {
        false
    }

    /// Perform forward pass with KV cache for efficient inference.
    ///
    /// KV caching stores previously computed key and value tensors,
    /// avoiding redundant computation for past tokens during generation.
    /// This provides O(n) complexity instead of O(n²) per token.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional mutable reference to KV cache
    ///
    /// # Returns
    /// Logits [batch, seq_len, vocab_size]
    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        // Default implementation: ignore cache and use standard forward
        // Models that support KV cache should override this
        let _ = cache;
        self.forward(input_ids, mask)
    }

    /// Create a KV cache for this model.
    ///
    /// Creates an appropriately sized cache based on the model's configuration.
    ///
    /// # Arguments
    /// * `max_seq_len` - Maximum sequence length to cache
    ///
    /// # Returns
    /// A new KVCache instance, or None if the model doesn't support caching
    fn create_cache(&self, _max_seq_len: usize) -> Option<KVCache> {
        // Default: no cache support
        None
    }

    /// Check if this model supports KV caching for efficient inference.
    fn supports_kv_cache(&self) -> bool {
        false
    }

    /// Get the number of trainable parameters.
    fn num_trainable_params(&self) -> usize;

    /// Get all LoRA parameters as a flat HashMap.
    ///
    /// This is used for checkpointing and saving adapters.
    fn lora_parameters(&self) -> HashMap<Rc<str>, Array>;

    /// Set LoRA parameters from a HashMap.
    ///
    /// This is used for restoring from checkpoints.
    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>);

    /// Save LoRA weights to a safetensors file.
    fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError>;

    /// Load LoRA weights from a safetensors file.
    fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError>;

    /// Enable gradient checkpointing for memory-efficient training.
    ///
    /// Gradient checkpointing trades compute for memory by periodically
    /// evaluating intermediate tensors to break the computation graph.
    /// This allows ~2x larger batch sizes with ~30% slowdown.
    ///
    /// # Arguments
    /// * `layers_per_block` - Number of layers between checkpoints.
    ///   Lower = more memory savings but slower. Recommended: 4.
    ///
    /// Default implementation does nothing. Models that support checkpointing
    /// should override this method.
    fn enable_gradient_checkpointing(&mut self, _layers_per_block: usize) {
        // Default: no-op. Models that support checkpointing override this.
    }

    /// Disable gradient checkpointing.
    fn disable_gradient_checkpointing(&mut self) {
        // Default: no-op.
    }

    /// Check if gradient checkpointing is supported by this model.
    fn supports_gradient_checkpointing(&self) -> bool {
        false
    }
}
