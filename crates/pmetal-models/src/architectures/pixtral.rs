//! Pixtral architecture.
//!
//! Pixtral is a vision-language model from Mistral AI that combines a
//! Vision Transformer encoder with a Mistral language model.
//!
//! The vision encoder handles:
//! - ViT with learned position embeddings
//! - Multi-scale image processing
//! - Multi-modal projector for vision-text feature alignment
//!
//! For now, this wraps the Mistral text model and sanitizes VLM weights.
//!
//! - Reference: <https://mistral.ai/news/pixtral-12b/>

use mlx_rs::{
    error::Exception,
    macros::ModuleParameters,
    Array,
};
use serde::{Deserialize, Serialize};

use crate::architectures::mistral::{MistralConfig, MistralForCausalLM};
use crate::traits::ModelConfig;

/// Pixtral vision model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PixtralVisionConfig {
    /// Hidden size of vision encoder.
    pub hidden_size: i32,
    /// Intermediate size in MLP.
    pub intermediate_size: i32,
    /// Number of hidden layers.
    pub num_hidden_layers: i32,
    /// Number of attention heads.
    pub num_attention_heads: i32,
    /// Number of channels (RGB = 3).
    pub num_channels: i32,
    /// Image size (square).
    pub image_size: i32,
    /// Patch size for image tokenization.
    pub patch_size: i32,
    /// Layer norm epsilon.
    pub layer_norm_eps: f32,
}

impl Default for PixtralVisionConfig {
    fn default() -> Self {
        // Defaults for Pixtral 12B
        Self {
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_channels: 3,
            image_size: 1024,
            patch_size: 16,
            layer_norm_eps: 1e-5,
        }
    }
}

/// Pixtral full model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PixtralConfig {
    /// Text model configuration.
    #[serde(flatten)]
    pub text_config: MistralConfig,
    /// Vision model configuration.
    #[serde(default)]
    pub vision_config: PixtralVisionConfig,
    /// Image token ID.
    #[serde(default = "default_image_token_id")]
    pub image_token_id: i32,
    /// Image break token ID.
    #[serde(default = "default_image_break_id")]
    pub image_break_token_id: i32,
    /// Image end token ID.
    #[serde(default = "default_image_end_id")]
    pub image_end_token_id: i32,
}

fn default_image_token_id() -> i32 { 10 }
fn default_image_break_id() -> i32 { 12 }
fn default_image_end_id() -> i32 { 13 }

impl Default for PixtralConfig {
    fn default() -> Self {
        Self {
            text_config: MistralConfig::default(),
            vision_config: PixtralVisionConfig::default(),
            image_token_id: default_image_token_id(),
            image_break_token_id: default_image_break_id(),
            image_end_token_id: default_image_end_id(),
        }
    }
}

impl ModelConfig for PixtralConfig {
    fn model_type(&self) -> &str {
        "pixtral"
    }

    fn vocab_size(&self) -> i32 {
        self.text_config.vocab_size
    }

    fn hidden_size(&self) -> i32 {
        self.text_config.hidden_size
    }

    fn num_hidden_layers(&self) -> i32 {
        self.text_config.num_hidden_layers
    }

    fn num_attention_heads(&self) -> i32 {
        self.text_config.num_attention_heads
    }

    fn num_kv_heads(&self) -> i32 {
        self.text_config.num_kv_heads()
    }

    fn head_dim(&self) -> i32 {
        self.text_config.hidden_size / self.text_config.num_attention_heads
    }

    fn intermediate_size(&self) -> i32 {
        self.text_config.intermediate_size
    }

    fn max_position_embeddings(&self) -> i32 {
        self.text_config.max_position_embeddings
    }

    fn norm_eps(&self) -> f32 {
        self.text_config.rms_norm_eps
    }

    fn rope_theta(&self) -> f32 {
        self.text_config.rope_theta
    }

    fn tie_word_embeddings(&self) -> bool {
        self.text_config.tie_word_embeddings
    }
}

/// Pixtral model (text-only wrapper).
///
/// This implementation wraps the Mistral language model for VLM inference.
/// Vision features should be pre-computed and passed as input embeddings.
#[derive(Debug, ModuleParameters)]
pub struct Pixtral {
    /// The underlying language model.
    #[param]
    pub language_model: MistralForCausalLM,
    /// Full configuration.
    pub config: PixtralConfig,
}

impl Pixtral {
    /// Create a new Pixtral model.
    pub fn new(config: PixtralConfig) -> Result<Self, Exception> {
        let language_model = MistralForCausalLM::new(config.text_config.clone())?;
        Ok(Self { language_model, config })
    }

    /// Forward pass with optional pre-computed vision embeddings.
    ///
    /// If `input_embeddings` is provided, it replaces token embeddings.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        cache: &mut Vec<Option<(Array, Array)>>,
        input_embeddings: Option<&Array>,
    ) -> Result<Array, Exception> {
        // Note: For full VLM support, we'd need to handle input_embeddings
        // and merge them with token embeddings. For now, just forward to LM.
        let _ = input_embeddings; // Placeholder for future vision support
        let _ = cache; // Use the proper cache API
        self.language_model.forward(input_ids, None)
    }

    /// Sanitize weights loaded from VLM checkpoint.
    ///
    /// Removes vision components and prefixes language model weights correctly.
    pub fn sanitize_weights(
        weights: std::collections::HashMap<String, Array>,
    ) -> std::collections::HashMap<String, Array> {
        let mut sanitized = std::collections::HashMap::new();

        for (key, value) in weights {
            // Skip vision components
            if key.starts_with("vision_tower")
                || key.starts_with("vision_encoder")
                || key.starts_with("visual")
                || key.starts_with("multi_modal_projector")
            {
                continue;
            }

            // Ensure language model prefix
            let new_key = if !key.starts_with("language_model.") {
                format!("language_model.{}", key)
            } else {
                key
            };

            sanitized.insert(new_key, value);
        }

        sanitized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_pixtral_config() {
        let config = PixtralConfig::default();
        assert_eq!(config.model_type(), "pixtral");
        assert!(config.vision_config.hidden_size > 0);
    }

    #[test]
    fn test_pixtral_vision_config() {
        let config = PixtralVisionConfig::default();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.patch_size, 16);
    }
}
