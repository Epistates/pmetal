//! Core traits for causal language models.
//!
//! This module defines the common interfaces that all model architectures implement,
//! enabling extensible model support with static dispatch.

use std::collections::HashMap;

use mlx_rs::{error::Exception, Array};

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
    /// Initializes the model with random weights. Use `load_weights`
    /// to load pretrained weights after construction.
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

    /// Load weights from a HashMap of tensors.
    ///
    /// The keys should match the model's parameter names in HuggingFace format:
    /// - `model.embed_tokens.weight`
    /// - `model.layers.{i}.self_attn.q_proj.weight`
    /// - etc.
    fn load_weights(&mut self, weights: &HashMap<String, Array>) -> Result<(), Exception>;

    /// Evaluate all parameters to materialize them on device.
    fn eval(&self) -> Result<(), Exception>;
}

/// Extension trait for models that support LoRA fine-tuning.
pub trait LoraCapable: CausalLMModel {
    /// The LoRA-enabled version of this model.
    type LoraModel;

    /// Convert to LoRA-enabled model with the given configuration.
    fn into_lora(
        self,
        lora_config: &pmetal_core::LoraConfig,
    ) -> Result<Self::LoraModel, Exception>;
}

/// Trait for models that can be quantized.
pub trait Quantizable: CausalLMModel {
    /// Quantize the model to the specified type.
    fn quantize(&mut self, quant_type: QuantizationType) -> Result<(), Exception>;

    /// Check if the model is quantized.
    fn is_quantized(&self) -> bool;
}

/// Quantization type options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizationType {
    /// 4-bit quantization.
    Q4,
    /// 8-bit quantization.
    Q8,
    /// No quantization (full precision).
    None,
}

impl std::fmt::Display for QuantizationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Q4 => write!(f, "Q4"),
            Self::Q8 => write!(f, "Q8"),
            Self::None => write!(f, "None"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test that QuantizationType display works
    #[test]
    fn test_quant_type_display() {
        assert_eq!(QuantizationType::Q4.to_string(), "Q4");
        assert_eq!(QuantizationType::Q8.to_string(), "Q8");
        assert_eq!(QuantizationType::None.to_string(), "None");
    }
}
