//! Core traits for causal language models.
//!
//! This module defines the common interfaces that all model architectures implement,
//! enabling extensible model support with static dispatch.

use pmetal_runtime::compat::{Array, Exception};

/// Configuration common to all causal LM architectures.
///
/// Each architecture (Llama, Qwen2, Gemma, etc.) implements this trait
/// for its specific config type, enabling generic code to work with
/// any model configuration.
pub trait ModelConfig: Clone {
    /// Get the model type identifier (e.g., "llama", "qwen2", "mistral").
    fn model_type(&self) -> &str;

    /// Vocabulary size.
    fn vocab_size(&self) -> i32;

    /// Hidden dimension.
    fn hidden_size(&self) -> i32;

    /// Number of transformer layers.
    fn num_hidden_layers(&self) -> i32;

    /// Number of attention heads.
    fn num_attention_heads(&self) -> i32;

    /// Number of KV heads (for GQA/MQA).
    fn num_kv_heads(&self) -> i32;

    /// Head dimension.
    fn head_dim(&self) -> i32;

    /// Intermediate/FFN size.
    fn intermediate_size(&self) -> i32;

    /// Maximum sequence length.
    fn max_position_embeddings(&self) -> i32;

    /// RMS/Layer norm epsilon.
    fn norm_eps(&self) -> f32;

    /// RoPE theta base frequency.
    fn rope_theta(&self) -> f32;

    /// Whether to tie input/output embeddings.
    fn tie_word_embeddings(&self) -> bool;
}

/// Core trait for all causal language model architectures.
///
/// This trait provides a unified interface for inference across all
/// supported model architectures (Llama, Qwen2, Gemma, Mistral, Phi, etc.).
///
/// # Design
///
/// Uses associated types rather than trait objects for:
/// - Compile-time optimization (monomorphization)
/// - No dynamic dispatch overhead
/// - Type-safe configuration handling
///
/// Note: This trait intentionally does not require `Send + Sync` because
/// MLX arrays contain raw pointers that are not thread-safe. Use the
/// `DynamicModel` enum for runtime polymorphism instead of trait objects.
pub trait CausalLMModel {
    /// The configuration type for this model.
    type Config: ModelConfig;

    /// Create a new model from configuration.
    ///
    /// Initializes the model with random weights; [`DynamicModel::load`](crate::DynamicModel::load)
    /// builds one from a checkpoint, auditing that every tensor is accounted for.
    fn new(config: Self::Config) -> Result<Self, Exception>
    where
        Self: Sized;

    /// Forward pass producing logits.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape `[batch, seq_len]`
    /// * `mask` - Optional attention mask (additive, -inf for masked positions)
    ///
    /// # Returns
    /// Logits of shape `[batch, seq_len, vocab_size]`
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, Exception>;

    /// Get model configuration.
    fn config(&self) -> &Self::Config;

    /// Evaluate all parameters to materialize them on device.
    fn eval(&self) -> Result<(), Exception>;
}
