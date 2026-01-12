//! Qwen2-VL architecture.
//!
//! Qwen2-VL is a vision-language model that combines a Vision Transformer encoder
//! with a Qwen2 language model. This implementation provides the text component
//! with placeholder hooks for vision feature injection.
//!
//! The vision encoder handles:
//! - Naive Dynamic Resolution (NaiveDyRes) for varying image sizes
//! - 2D/3D Rotary Position Embeddings for vision-text alignment
//! - Multi-modal projector for vision-text feature fusion
//!
//! For now, this wraps the Qwen2 text model and sanitizes VLM weights.
//!
//! - Paper: <https://arxiv.org/abs/2409.12191>

use mlx_rs::{
    error::Exception,
    macros::ModuleParameters,
    Array,
};
use serde::{Deserialize, Serialize};

use crate::architectures::qwen2::{Qwen2Config, Qwen2ForCausalLM};
use crate::traits::ModelConfig;

/// Qwen2-VL vision model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen2VLVisionConfig {
    /// Hidden size of vision encoder.
    pub hidden_size: i32,
    /// Intermediate size in MLP.
    pub intermediate_size: i32,
    /// Number of hidden layers.
    pub num_hidden_layers: i32,
    /// Number of attention heads.
    pub num_attention_heads: i32,
    /// Number of channels (RGB = 3).
    pub in_channels: i32,
    /// Patch size for image tokenization.
    pub patch_size: i32,
    /// Spatial merge size for pooling.
    pub spatial_merge_size: i32,
    /// Temporal patch size for video.
    pub temporal_patch_size: i32,
    /// Layer norm epsilon.
    pub layer_norm_eps: f32,
}

impl Default for Qwen2VLVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            in_channels: 3,
            patch_size: 14,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            layer_norm_eps: 1e-6,
        }
    }
}

/// Qwen2-VL full model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen2VLConfig {
    /// Text model configuration.
    #[serde(flatten)]
    pub text_config: Qwen2Config,
    /// Vision model configuration.
    #[serde(default)]
    pub vision_config: Qwen2VLVisionConfig,
    /// Image token ID for marking image positions.
    #[serde(default = "default_image_token_id")]
    pub image_token_id: i32,
    /// Video token ID for marking video positions.
    #[serde(default = "default_video_token_id")]
    pub video_token_id: i32,
}

fn default_image_token_id() -> i32 { 151655 }
fn default_video_token_id() -> i32 { 151656 }

impl Default for Qwen2VLConfig {
    fn default() -> Self {
        Self {
            text_config: Qwen2Config::default(),
            vision_config: Qwen2VLVisionConfig::default(),
            image_token_id: default_image_token_id(),
            video_token_id: default_video_token_id(),
        }
    }
}

impl ModelConfig for Qwen2VLConfig {
    fn model_type(&self) -> &str {
        "qwen2_vl"
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

/// Qwen2-VL model (text-only wrapper).
///
/// This implementation wraps the Qwen2 language model for VLM inference.
/// Vision features should be pre-computed and passed as input embeddings.
#[derive(Debug, ModuleParameters)]
pub struct Qwen2VL {
    /// The underlying language model.
    #[param]
    pub language_model: Qwen2ForCausalLM,
    /// Full configuration.
    pub config: Qwen2VLConfig,
}

impl Qwen2VL {
    /// Create a new Qwen2-VL model.
    pub fn new(config: Qwen2VLConfig) -> Result<Self, Exception> {
        let language_model = Qwen2ForCausalLM::new(config.text_config.clone())?;
        Ok(Self { language_model, config })
    }

    /// Forward pass with optional pre-computed vision embeddings.
    ///
    /// If `input_embeddings` is provided, it replaces token embeddings.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        _cache: &mut Vec<Option<(Array, Array)>>,
        _input_embeddings: Option<&Array>,
    ) -> Result<Array, Exception> {
        // Note: For full VLM support, we'd need to handle input_embeddings
        // and merge them with token embeddings. For now, just forward to LM.
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
            if key.starts_with("visual") || key.starts_with("vision_tower") {
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
    fn test_qwen2_vl_config() {
        let config = Qwen2VLConfig::default();
        assert_eq!(config.model_type(), "qwen2_vl");
        assert!(config.vision_config.hidden_size > 0);
    }

    #[test]
    fn test_qwen2_vl_vision_config() {
        let config = Qwen2VLVisionConfig::default();
        assert_eq!(config.hidden_size, 1280);
        assert_eq!(config.patch_size, 14);
    }
}
