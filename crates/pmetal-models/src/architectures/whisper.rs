//! OpenAI Whisper speech-to-text model implementation.
//!
//! Whisper is an encoder-decoder model for automatic speech recognition (ASR).
//! The encoder processes mel-spectrogram audio features, and the decoder
//! generates text tokens autoregressively.
//!
//! # Architecture
//!
//! ```text
//! Audio (mel spectrogram)
//!     │
//!     ▼
//! ┌─────────────────┐
//! │  AudioEncoder   │
//! │  - Conv1d x2    │
//! │  - Positional   │
//! │  - Transformer  │
//! └────────┬────────┘
//!          │ (cross-attention)
//!          ▼
//! ┌─────────────────┐
//! │  TextDecoder    │
//! │  - Embedding    │
//! │  - Positional   │
//! │  - Transformer  │
//! │  (w/ cross-attn)│
//! └────────┬────────┘
//!          │
//!          ▼
//!     Token Logits
//! ```
//!
//! # Reference
//!
//! - Paper: <https://arxiv.org/abs/2212.04356>
//! - Implementation based on: candle-transformers/src/models/whisper

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{indexing::IndexOp, softmax_axis},
    Array,
};
use serde::{Deserialize, Serialize};

/// Whisper model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    /// Model hidden dimension.
    pub d_model: i32,
    /// Number of attention heads in encoder.
    pub encoder_attention_heads: i32,
    /// Number of encoder layers.
    pub encoder_layers: i32,
    /// Number of attention heads in decoder.
    pub decoder_attention_heads: i32,
    /// Number of decoder layers.
    pub decoder_layers: i32,
    /// MLP intermediate dimension.
    pub encoder_ffn_dim: i32,
    /// Decoder MLP dimension.
    pub decoder_ffn_dim: i32,
    /// Vocabulary size.
    pub vocab_size: i32,
    /// Number of mel bins in audio input.
    pub num_mel_bins: i32,
    /// Maximum source positions (encoder context length).
    pub max_source_positions: i32,
    /// Maximum target positions (decoder context length).
    pub max_target_positions: i32,
    /// Pad token ID.
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: i32,
    /// BOS token ID.
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: i32,
    /// EOS token ID.
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: i32,
    /// Layer norm epsilon.
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
}

fn default_pad_token_id() -> i32 {
    50257
}
fn default_bos_token_id() -> i32 {
    50258
}
fn default_eos_token_id() -> i32 {
    50257
}
fn default_layer_norm_eps() -> f32 {
    1e-5
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self::tiny()
    }
}

impl WhisperConfig {
    /// Whisper tiny configuration.
    pub fn tiny() -> Self {
        Self {
            d_model: 384,
            encoder_attention_heads: 6,
            encoder_layers: 4,
            decoder_attention_heads: 6,
            decoder_layers: 4,
            encoder_ffn_dim: 1536,
            decoder_ffn_dim: 1536,
            vocab_size: 51865,
            num_mel_bins: 80,
            max_source_positions: 1500,
            max_target_positions: 448,
            pad_token_id: 50257,
            bos_token_id: 50258,
            eos_token_id: 50257,
            layer_norm_eps: 1e-5,
        }
    }

    /// Whisper base configuration.
    pub fn base() -> Self {
        Self {
            d_model: 512,
            encoder_attention_heads: 8,
            encoder_layers: 6,
            decoder_attention_heads: 8,
            decoder_layers: 6,
            encoder_ffn_dim: 2048,
            decoder_ffn_dim: 2048,
            vocab_size: 51865,
            num_mel_bins: 80,
            max_source_positions: 1500,
            max_target_positions: 448,
            pad_token_id: 50257,
            bos_token_id: 50258,
            eos_token_id: 50257,
            layer_norm_eps: 1e-5,
        }
    }

    /// Whisper small configuration.
    pub fn small() -> Self {
        Self {
            d_model: 768,
            encoder_attention_heads: 12,
            encoder_layers: 12,
            decoder_attention_heads: 12,
            decoder_layers: 12,
            encoder_ffn_dim: 3072,
            decoder_ffn_dim: 3072,
            vocab_size: 51865,
            num_mel_bins: 80,
            max_source_positions: 1500,
            max_target_positions: 448,
            pad_token_id: 50257,
            bos_token_id: 50258,
            eos_token_id: 50257,
            layer_norm_eps: 1e-5,
        }
    }

    /// Whisper medium configuration.
    pub fn medium() -> Self {
        Self {
            d_model: 1024,
            encoder_attention_heads: 16,
            encoder_layers: 24,
            decoder_attention_heads: 16,
            decoder_layers: 24,
            encoder_ffn_dim: 4096,
            decoder_ffn_dim: 4096,
            vocab_size: 51865,
            num_mel_bins: 80,
            max_source_positions: 1500,
            max_target_positions: 448,
            pad_token_id: 50257,
            bos_token_id: 50258,
            eos_token_id: 50257,
            layer_norm_eps: 1e-5,
        }
    }

    /// Whisper large configuration.
    pub fn large() -> Self {
        Self {
            d_model: 1280,
            encoder_attention_heads: 20,
            encoder_layers: 32,
            decoder_attention_heads: 20,
            decoder_layers: 32,
            encoder_ffn_dim: 5120,
            decoder_ffn_dim: 5120,
            vocab_size: 51865,
            num_mel_bins: 80,
            max_source_positions: 1500,
            max_target_positions: 448,
            pad_token_id: 50257,
            bos_token_id: 50258,
            eos_token_id: 50257,
            layer_norm_eps: 1e-5,
        }
    }

    /// Head dimension.
    pub fn head_dim(&self) -> i32 {
        self.d_model / self.encoder_attention_heads
    }
}

/// Whisper multi-head attention layer.
#[derive(Debug, ModuleParameters)]
pub struct WhisperAttention {
    /// Number of attention heads.
    pub n_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale factor.
    pub scale: f32,

    /// Query projection.
    #[param]
    pub q_proj: nn::Linear,
    /// Key projection.
    #[param]
    pub k_proj: nn::Linear,
    /// Value projection.
    #[param]
    pub v_proj: nn::Linear,
    /// Output projection.
    #[param]
    pub out_proj: nn::Linear,
}

impl WhisperAttention {
    /// Create a new attention layer.
    pub fn new(d_model: i32, n_heads: i32, with_bias: bool) -> Result<Self, Exception> {
        let head_dim = d_model / n_heads;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = nn::LinearBuilder::new(d_model, d_model)
            .bias(with_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(d_model, d_model)
            .bias(false) // Key has no bias in Whisper
            .build()?;
        let v_proj = nn::LinearBuilder::new(d_model, d_model)
            .bias(with_bias)
            .build()?;
        let out_proj = nn::LinearBuilder::new(d_model, d_model)
            .bias(with_bias)
            .build()?;

        Ok(Self {
            n_heads,
            head_dim,
            scale,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
    }

    /// Forward pass.
    pub fn forward(
        &mut self,
        x: &Array,
        xa: Option<&Array>,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Project to Q, K, V
        let q = Module::forward(&mut self.q_proj, x)?;
        let (k, v) = match xa {
            Some(encoder_out) => {
                let k = Module::forward(&mut self.k_proj, encoder_out)?;
                let v = Module::forward(&mut self.v_proj, encoder_out)?;
                (k, v)
            }
            None => {
                let k = Module::forward(&mut self.k_proj, x)?;
                let v = Module::forward(&mut self.v_proj, x)?;
                (k, v)
            }
        };

        // Reshape to multi-head: [B, T, H, D]
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        let k = k.reshape(&[batch, -1, self.n_heads, self.head_dim])?;
        let v = v.reshape(&[batch, -1, self.n_heads, self.head_dim])?;

        // Transpose to [B, H, T, D]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Scaled dot-product attention
        let scale = Array::from_f32(self.scale);
        let q_scaled = mlx_rs::ops::multiply(&q, &scale)?;
        let k_t = k.transpose_axes(&[0, 1, 3, 2])?;
        let mut attn = mlx_rs::ops::matmul(&q_scaled, &k_t)?;

        // Apply mask if provided
        if let Some(m) = mask {
            attn = mlx_rs::ops::add(&attn, m)?;
        }

        // Softmax
        let attn = softmax_axis(&attn, -1, None)?;

        // Attention output
        let out = mlx_rs::ops::matmul(&attn, &v)?;

        // Transpose back to [B, T, H, D] and reshape to [B, T, D_model]
        let out = out.transpose_axes(&[0, 2, 1, 3])?;
        let d_model = self.n_heads * self.head_dim;
        let out = out.reshape(&[batch, seq_len, d_model])?;

        // Output projection
        Module::forward(&mut self.out_proj, &out)
    }
}

/// Whisper encoder attention block.
#[derive(Debug, ModuleParameters)]
pub struct WhisperEncoderBlock {
    /// Self-attention.
    #[param]
    pub self_attn: WhisperAttention,
    /// Self-attention layer norm.
    #[param]
    pub self_attn_layer_norm: nn::LayerNorm,
    /// First MLP layer.
    #[param]
    pub fc1: nn::Linear,
    /// Second MLP layer.
    #[param]
    pub fc2: nn::Linear,
    /// Final layer norm.
    #[param]
    pub final_layer_norm: nn::LayerNorm,
}

impl WhisperEncoderBlock {
    /// Create a new encoder block.
    pub fn new(config: &WhisperConfig) -> Result<Self, Exception> {
        let self_attn =
            WhisperAttention::new(config.d_model, config.encoder_attention_heads, true)?;
        let self_attn_layer_norm = nn::LayerNormBuilder::new(config.d_model)
            .eps(config.layer_norm_eps)
            .build()?;
        let fc1 = nn::LinearBuilder::new(config.d_model, config.encoder_ffn_dim)
            .bias(true)
            .build()?;
        let fc2 = nn::LinearBuilder::new(config.encoder_ffn_dim, config.d_model)
            .bias(true)
            .build()?;
        let final_layer_norm = nn::LayerNormBuilder::new(config.d_model)
            .eps(config.layer_norm_eps)
            .build()?;

        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
        })
    }

    /// Forward pass.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // Self-attention with residual
        let x_norm = Module::forward(&mut self.self_attn_layer_norm, x)?;
        let attn_out = self.self_attn.forward(&x_norm, None, None)?;
        let x = mlx_rs::ops::add(x, &attn_out)?;

        // MLP with residual
        let x_norm = Module::forward(&mut self.final_layer_norm, &x)?;
        let mlp_out = Module::forward(&mut self.fc1, &x_norm)?;
        let mlp_out = mlx_rs::nn::gelu(&mlp_out)?;
        let mlp_out = Module::forward(&mut self.fc2, &mlp_out)?;

        mlx_rs::ops::add(&x, &mlp_out)
    }
}

/// Whisper decoder attention block.
#[derive(Debug, ModuleParameters)]
pub struct WhisperDecoderBlock {
    /// Self-attention.
    #[param]
    pub self_attn: WhisperAttention,
    /// Self-attention layer norm.
    #[param]
    pub self_attn_layer_norm: nn::LayerNorm,
    /// Cross-attention.
    #[param]
    pub encoder_attn: WhisperAttention,
    /// Cross-attention layer norm.
    #[param]
    pub encoder_attn_layer_norm: nn::LayerNorm,
    /// First MLP layer.
    #[param]
    pub fc1: nn::Linear,
    /// Second MLP layer.
    #[param]
    pub fc2: nn::Linear,
    /// Final layer norm.
    #[param]
    pub final_layer_norm: nn::LayerNorm,
}

impl WhisperDecoderBlock {
    /// Create a new decoder block.
    pub fn new(config: &WhisperConfig) -> Result<Self, Exception> {
        let self_attn =
            WhisperAttention::new(config.d_model, config.decoder_attention_heads, true)?;
        let self_attn_layer_norm = nn::LayerNormBuilder::new(config.d_model)
            .eps(config.layer_norm_eps)
            .build()?;
        let encoder_attn =
            WhisperAttention::new(config.d_model, config.decoder_attention_heads, true)?;
        let encoder_attn_layer_norm = nn::LayerNormBuilder::new(config.d_model)
            .eps(config.layer_norm_eps)
            .build()?;
        let fc1 = nn::LinearBuilder::new(config.d_model, config.decoder_ffn_dim)
            .bias(true)
            .build()?;
        let fc2 = nn::LinearBuilder::new(config.decoder_ffn_dim, config.d_model)
            .bias(true)
            .build()?;
        let final_layer_norm = nn::LayerNormBuilder::new(config.d_model)
            .eps(config.layer_norm_eps)
            .build()?;

        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            encoder_attn,
            encoder_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
        })
    }

    /// Forward pass.
    pub fn forward(
        &mut self,
        x: &Array,
        encoder_out: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        // Self-attention with residual
        let x_norm = Module::forward(&mut self.self_attn_layer_norm, x)?;
        let attn_out = self.self_attn.forward(&x_norm, None, mask)?;
        let x = mlx_rs::ops::add(x, &attn_out)?;

        // Cross-attention with residual
        let x_norm = Module::forward(&mut self.encoder_attn_layer_norm, &x)?;
        let cross_out = self
            .encoder_attn
            .forward(&x_norm, Some(encoder_out), None)?;
        let x = mlx_rs::ops::add(&x, &cross_out)?;

        // MLP with residual
        let x_norm = Module::forward(&mut self.final_layer_norm, &x)?;
        let mlp_out = Module::forward(&mut self.fc1, &x_norm)?;
        let mlp_out = mlx_rs::nn::gelu(&mlp_out)?;
        let mlp_out = Module::forward(&mut self.fc2, &mlp_out)?;

        mlx_rs::ops::add(&x, &mlp_out)
    }
}

/// Whisper audio encoder.
#[derive(Debug, ModuleParameters)]
pub struct WhisperEncoder {
    /// First conv layer.
    #[param]
    pub conv1: nn::Conv1d,
    /// Second conv layer.
    #[param]
    pub conv2: nn::Conv1d,
    /// Positional embedding.
    #[param]
    pub positional_embedding: Param<Array>,
    /// Encoder blocks.
    #[param]
    pub layers: Vec<WhisperEncoderBlock>,
    /// Post layer norm.
    #[param]
    pub layer_norm: nn::LayerNorm,
}

impl WhisperEncoder {
    /// Create a new encoder.
    pub fn new(config: &WhisperConfig) -> Result<Self, Exception> {
        let conv1 = nn::Conv1dBuilder::new(config.num_mel_bins, config.d_model, 3)
            .padding(1)
            .stride(1)
            .build()?;
        let conv2 = nn::Conv1dBuilder::new(config.d_model, config.d_model, 3)
            .padding(1)
            .stride(2)
            .build()?;

        // Sinusoidal positional embeddings
        let positional_embedding = Param::new(sinusoids(
            config.max_source_positions as usize,
            config.d_model as usize,
        )?);

        let layers = (0..config.encoder_layers)
            .map(|_| WhisperEncoderBlock::new(config))
            .collect::<Result<Vec<_>, _>>()?;

        let layer_norm = nn::LayerNormBuilder::new(config.d_model)
            .eps(config.layer_norm_eps)
            .build()?;

        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            layers,
            layer_norm,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Mel spectrogram [batch, n_mel, time]
    ///
    /// # Returns
    /// Encoder output [batch, time/2, d_model]
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // Conv layers with GELU
        let x = Module::forward(&mut self.conv1, x)?;
        let x = mlx_rs::nn::gelu(&x)?;
        let x = Module::forward(&mut self.conv2, &x)?;
        let x = mlx_rs::nn::gelu(&x)?;

        // Transpose: [B, C, T] -> [B, T, C]
        let x = x.transpose_axes(&[0, 2, 1])?;

        // Add positional embedding
        let seq_len = x.shape()[1] as i32;
        let pos_embed = self.positional_embedding.value.index((0..seq_len, ..));
        let mut x = mlx_rs::ops::add(&x, &pos_embed)?;

        // Transformer blocks
        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }

        Module::forward(&mut self.layer_norm, &x)
    }
}

/// Whisper text decoder.
#[derive(Debug, ModuleParameters)]
pub struct WhisperDecoder {
    /// Token embedding.
    #[param]
    pub embed_tokens: nn::Embedding,
    /// Positional embedding.
    #[param]
    pub embed_positions: Param<Array>,
    /// Decoder blocks.
    #[param]
    pub layers: Vec<WhisperDecoderBlock>,
    /// Final layer norm.
    #[param]
    pub layer_norm: nn::LayerNorm,
}

impl WhisperDecoder {
    /// Create a new decoder.
    pub fn new(config: &WhisperConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.d_model)?;

        // Learned positional embeddings (initialized to zeros, will be loaded from checkpoint)
        let embed_positions = Param::new(Array::zeros::<f32>(&[
            config.max_target_positions,
            config.d_model,
        ])?);

        let layers = (0..config.decoder_layers)
            .map(|_| WhisperDecoderBlock::new(config))
            .collect::<Result<Vec<_>, _>>()?;

        let layer_norm = nn::LayerNormBuilder::new(config.d_model)
            .eps(config.layer_norm_eps)
            .build()?;

        Ok(Self {
            embed_tokens,
            embed_positions,
            layers,
            layer_norm,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `tokens` - Token IDs [batch, seq_len]
    /// * `encoder_out` - Encoder output
    /// * `mask` - Causal attention mask
    ///
    /// # Returns
    /// Decoder output [batch, seq_len, d_model]
    pub fn forward(
        &mut self,
        tokens: &Array,
        encoder_out: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let seq_len = tokens.shape()[1] as i32;

        // Token embeddings
        let x = Module::forward(&mut self.embed_tokens, tokens)?;

        // Add positional embeddings
        let pos_embed = self.embed_positions.value.index((0..seq_len, ..));
        let mut x = mlx_rs::ops::add(&x, &pos_embed)?;

        // Transformer blocks
        for layer in &mut self.layers {
            x = layer.forward(&x, encoder_out, mask)?;
        }

        Module::forward(&mut self.layer_norm, &x)
    }

    /// Compute logits from hidden states (tied weights with embedding).
    pub fn logits(&self, hidden: &Array) -> Result<Array, Exception> {
        self.embed_tokens.as_linear(hidden)
    }
}

/// Complete Whisper model.
#[derive(Debug, ModuleParameters)]
pub struct Whisper {
    /// Configuration.
    pub config: WhisperConfig,
    /// Audio encoder.
    #[param]
    pub encoder: WhisperEncoder,
    /// Text decoder.
    #[param]
    pub decoder: WhisperDecoder,
}

impl Whisper {
    /// Create a new Whisper model.
    pub fn new(config: WhisperConfig) -> Result<Self, Exception> {
        let encoder = WhisperEncoder::new(&config)?;
        let decoder = WhisperDecoder::new(&config)?;

        Ok(Self {
            config,
            encoder,
            decoder,
        })
    }

    /// Encode audio to hidden states.
    pub fn encode(&mut self, mel: &Array) -> Result<Array, Exception> {
        self.encoder.forward(mel)
    }

    /// Decode tokens given encoder output.
    pub fn decode(
        &mut self,
        tokens: &Array,
        encoder_out: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let hidden = self.decoder.forward(tokens, encoder_out, mask)?;
        self.decoder.logits(&hidden)
    }

    /// Full forward pass.
    pub fn forward(&mut self, mel: &Array, tokens: &Array) -> Result<Array, Exception> {
        let encoder_out = self.encode(mel)?;

        // Create causal mask
        let seq_len = tokens.shape()[1] as usize;
        let mask = create_causal_mask(seq_len)?;

        self.decode(tokens, &encoder_out, Some(&mask))
    }
}

/// Generate sinusoidal positional embeddings.
fn sinusoids(length: usize, channels: usize) -> Result<Array, Exception> {
    let max_timescale = 10000.0f32;
    let log_timescale_increment = max_timescale.ln() / (channels / 2 - 1) as f32;

    let mut data = vec![0.0f32; length * channels];

    for pos in 0..length {
        for i in 0..channels / 2 {
            let inv_timescale = (-log_timescale_increment * i as f32).exp();
            let angle = pos as f32 * inv_timescale;
            data[pos * channels + i] = angle.sin();
            data[pos * channels + channels / 2 + i] = angle.cos();
        }
    }

    Ok(Array::from_slice(&data, &[length as i32, channels as i32]))
}

/// Create causal attention mask.
fn create_causal_mask(size: usize) -> Result<Array, Exception> {
    let mut data = vec![0.0f32; size * size];

    for i in 0..size {
        for j in 0..size {
            if j > i {
                data[i * size + j] = f32::NEG_INFINITY;
            }
        }
    }

    Ok(Array::from_slice(&data, &[size as i32, size as i32]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_whisper_config_sizes() {
        let tiny = WhisperConfig::tiny();
        assert_eq!(tiny.d_model, 384);
        assert_eq!(tiny.encoder_layers, 4);

        let base = WhisperConfig::base();
        assert_eq!(base.d_model, 512);
        assert_eq!(base.encoder_layers, 6);

        let small = WhisperConfig::small();
        assert_eq!(small.d_model, 768);
        assert_eq!(small.encoder_layers, 12);

        let medium = WhisperConfig::medium();
        assert_eq!(medium.d_model, 1024);
        assert_eq!(medium.encoder_layers, 24);

        let large = WhisperConfig::large();
        assert_eq!(large.d_model, 1280);
        assert_eq!(large.encoder_layers, 32);
    }

    #[test]
    fn test_head_dim() {
        let config = WhisperConfig::base();
        assert_eq!(config.head_dim(), 64); // 512 / 8 = 64
    }
}
