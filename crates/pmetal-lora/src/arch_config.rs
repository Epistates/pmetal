//! Architecture configuration trait for generic LoRA implementations.
//!
//! This module provides the `LoraArchitectureConfig` trait that abstracts
//! common model configuration fields needed by LoRA layers. By implementing
//! this trait, model-specific configs can be used with generic LoRA implementations.

/// Trait for model configurations that can be used with generic LoRA layers.
///
/// This trait extracts the common configuration values needed by LoRA attention
/// and MLP layers, enabling code reuse across architectures.
pub trait LoraArchitectureConfig {
    /// Hidden size (embedding dimension).
    fn hidden_size(&self) -> i32;

    /// Number of attention heads.
    fn num_attention_heads(&self) -> i32;

    /// Number of key-value heads (for GQA/MQA).
    fn num_kv_heads(&self) -> i32;

    /// Head dimension (hidden_size / num_attention_heads or explicit).
    fn head_dim(&self) -> i32;

    /// Intermediate size for MLP.
    fn intermediate_size(&self) -> i32;

    /// RoPE theta base frequency.
    fn rope_theta(&self) -> f32;

    /// Sliding window size (None for full attention).
    fn sliding_window(&self) -> Option<i32> {
        None
    }

    /// RMS norm epsilon.
    fn rms_norm_eps(&self) -> f32 {
        1e-5
    }

    /// Hidden activation function name.
    fn hidden_act(&self) -> &str {
        "silu"
    }

    /// Whether this architecture uses gated MLP (SiLU/SwiGLU).
    fn use_gated_mlp(&self) -> bool {
        true
    }

    /// Number of hidden layers.
    fn num_hidden_layers(&self) -> i32;

    /// Vocabulary size.
    fn vocab_size(&self) -> i32;

    /// Whether to tie embeddings with output.
    fn tie_word_embeddings(&self) -> bool {
        false
    }
}

// ============================================================================
// Implementations for existing model configs
// ============================================================================

use pmetal_models::architectures::llama::LlamaConfig;
use pmetal_models::architectures::mistral::MistralConfig;
use pmetal_models::architectures::qwen3::Qwen3Config;

impl LoraArchitectureConfig for LlamaConfig {
    fn hidden_size(&self) -> i32 {
        self.hidden_size
    }

    fn num_attention_heads(&self) -> i32 {
        self.num_attention_heads
    }

    fn num_kv_heads(&self) -> i32 {
        self.num_kv_heads()
    }

    fn head_dim(&self) -> i32 {
        self.get_head_dim()
    }

    fn intermediate_size(&self) -> i32 {
        self.intermediate_size
    }

    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }

    fn hidden_act(&self) -> &str {
        &self.hidden_act
    }

    fn num_hidden_layers(&self) -> i32 {
        self.num_hidden_layers
    }

    fn vocab_size(&self) -> i32 {
        self.vocab_size
    }

    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
}

impl LoraArchitectureConfig for MistralConfig {
    fn hidden_size(&self) -> i32 {
        self.hidden_size
    }

    fn num_attention_heads(&self) -> i32 {
        self.num_attention_heads
    }

    fn num_kv_heads(&self) -> i32 {
        self.num_kv_heads()
    }

    fn head_dim(&self) -> i32 {
        self.get_head_dim()
    }

    fn intermediate_size(&self) -> i32 {
        self.intermediate_size
    }

    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    fn sliding_window(&self) -> Option<i32> {
        self.sliding_window
    }

    fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }

    fn hidden_act(&self) -> &str {
        &self.hidden_act
    }

    fn num_hidden_layers(&self) -> i32 {
        self.num_hidden_layers
    }

    fn vocab_size(&self) -> i32 {
        self.vocab_size
    }

    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
}

impl LoraArchitectureConfig for Qwen3Config {
    fn hidden_size(&self) -> i32 {
        self.hidden_size
    }

    fn num_attention_heads(&self) -> i32 {
        self.num_attention_heads
    }

    fn num_kv_heads(&self) -> i32 {
        self.num_kv_heads()
    }

    fn head_dim(&self) -> i32 {
        self.get_head_dim()
    }

    fn intermediate_size(&self) -> i32 {
        self.intermediate_size
    }

    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }

    fn hidden_act(&self) -> &str {
        &self.hidden_act
    }

    fn num_hidden_layers(&self) -> i32 {
        self.num_hidden_layers
    }

    fn vocab_size(&self) -> i32 {
        self.vocab_size
    }

    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
}
