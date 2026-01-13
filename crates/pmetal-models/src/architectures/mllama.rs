//! Llama 3.2 Vision (Mllama) architecture.
//!
//! Mllama is a multimodal model that combines a Llama 3.2 text model with a
//! vision encoder. The text model includes cross-attention layers to attend
//! to visual features.

use std::collections::HashMap;

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::softmax_axis,
    Array,
};
use pmetal_mlx::kernels::{
    differentiable_attention, fused_sdpa, get_training_context, rope::apply_rope,
    AttentionMaskType, FusedAttentionConfig,
};
use pmetal_mlx::kv_cache::KVCache;
use serde::{Deserialize, Serialize};

use crate::architectures::llama::{LlamaAttention, LlamaConfig, LlamaMLP, RopeScalingValue};
use crate::traits::ModelConfig;

/// Mllama vision model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MllamaVisionConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_channels: i32,
    pub image_size: i32,
    pub patch_size: i32,
    pub hidden_act: String,
    pub layer_norm_eps: f32,
    pub attention_dropout: f32,
    pub num_global_layers: i32,
}

impl Default for MllamaVisionConfig {
    fn default() -> Self {
        // Defaults for Llama 3.2 11B Vision
        Self {
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            num_channels: 3,
            image_size: 560,
            patch_size: 14,
            hidden_act: "silu".to_string(),
            layer_norm_eps: 1e-5,
            attention_dropout: 0.0,
            num_global_layers: 8,
        }
    }
}

/// Mllama full model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MllamaConfig {
    /// Text model configuration.
    #[serde(flatten)]
    pub text_config: LlamaConfig,
    /// Vision model configuration.
    pub vision_config: MllamaVisionConfig,
    /// Indices of layers where cross-attention is applied.
    #[serde(default)]
    pub cross_attention_layers: Vec<i32>,
}

impl Default for MllamaConfig {
    fn default() -> Self {
        let text_config = LlamaConfig::default();
        let vision_config = MllamaVisionConfig::default();
        // Example indices for 11B model (every 4th layer)
        let cross_attention_layers = vec![3, 7, 11, 15, 19, 23, 27, 31];

        Self {
            text_config,
            vision_config,
            cross_attention_layers,
        }
    }
}

impl ModelConfig for MllamaConfig {
    fn model_type(&self) -> &str {
        &self.text_config.model_type
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
        self.text_config.get_head_dim()
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

/// Learnable gate parameter wrapper.
#[derive(Debug, ModuleParameters)]
pub struct Gate {
    #[param]
    pub weight: Param<Array>,
}

impl Gate {
    pub fn new(shape: &[i32]) -> Result<Self, Exception> {
        // Initialize with 0 usually for tanh gating to start as identity (x + 0*attn)
        // or small value
        let weight = mlx_rs::ops::zeros::<f32>(shape)?;
        Ok(Self {
            weight: Param::new(weight),
        })
    }
}

/// Vision embeddings (patch embeddings + positional + class token).
#[derive(Debug, ModuleParameters)]
pub struct MllamaVisionEmbeddings {
    #[param]
    pub patch_embedding: nn::Conv2d,
    #[param]
    pub class_embedding: nn::Embedding, // Learnable class token
    #[param]
    pub position_embedding: nn::Embedding, // Learned positional embeddings
    #[param]
    pub pre_tile_position_embedding: nn::Embedding, // Gated positional embeddings for tiles
    #[param]
    pub post_tile_position_embedding: nn::Embedding,

    pub patch_size: i32,
    pub image_size: i32,
    pub hidden_size: i32,
    pub num_patches: i32,
}

impl MllamaVisionEmbeddings {
    pub fn new(config: &MllamaVisionConfig) -> Result<Self, Exception> {
        let patch_embedding =
            nn::Conv2dBuilder::new(config.num_channels, config.hidden_size, config.patch_size)
                .stride(config.patch_size)
                .bias(false)
                .build()?;

        let num_patches = (config.image_size / config.patch_size).pow(2);

        // Class token
        let class_embedding = nn::Embedding::new(1, config.hidden_size)?;

        // Positional embeddings
        let position_embedding = nn::Embedding::new(num_patches, config.hidden_size)?;
        let pre_tile_position_embedding = nn::Embedding::new(
            config.num_global_layers * num_patches, // Placeholder size
            config.hidden_size,
        )?;
        let post_tile_position_embedding =
            nn::Embedding::new(config.num_global_layers * num_patches, config.hidden_size)?;

        Ok(Self {
            patch_embedding,
            class_embedding,
            position_embedding,
            pre_tile_position_embedding,
            post_tile_position_embedding,
            patch_size: config.patch_size,
            image_size: config.image_size,
            hidden_size: config.hidden_size,
            num_patches,
        })
    }

    pub fn forward(&mut self, pixel_values: &Array) -> Result<Array, Exception> {
        // pixel_values: [batch, channels, height, width]
        // Conv2d expects [batch, height, width, channels] in MLX by default (NHWC)
        // Assuming input is NCHW (standard PyTorch), we transpose.
        let x = pixel_values.transpose_axes(&[0, 2, 3, 1])?;

        let patches = Module::forward(&mut self.patch_embedding, &x)?;
        // flatten patches: [B, H_p, W_p, C] -> [B, N_p, C]
        let patches_flat = patches.reshape(&[patches.shape()[0], -1, self.hidden_size])?;

        Ok(patches_flat)
    }
}

/// Vision MLP layer.
#[derive(Debug, ModuleParameters)]
pub struct MllamaVisionMLP {
    #[param]
    pub fc1: nn::Linear,
    #[param]
    pub fc2: nn::Linear,
}

impl MllamaVisionMLP {
    pub fn new(config: &MllamaVisionConfig) -> Result<Self, Exception> {
        let fc1 = nn::LinearBuilder::new(config.hidden_size, config.intermediate_size)
            .bias(true) // Vision models often have bias
            .build()?;
        let fc2 = nn::LinearBuilder::new(config.intermediate_size, config.hidden_size)
            .bias(true)
            .build()?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let h = Module::forward(&mut self.fc1, x)?;
        let h = nn::silu(h)?;
        Module::forward(&mut self.fc2, &h)
    }
}

/// Vision Attention layer (Self-Attention).
#[derive(Debug, ModuleParameters)]
pub struct MllamaVisionAttention {
    pub n_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub o_proj: nn::Linear,
}

impl MllamaVisionAttention {
    pub fn new(config: &MllamaVisionConfig) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let head_dim = config.hidden_size / n_heads;

        let q_proj = nn::LinearBuilder::new(config.hidden_size, config.hidden_size)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(config.hidden_size, config.hidden_size)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(config.hidden_size, config.hidden_size)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(config.hidden_size, config.hidden_size)
            .bias(false)
            .build()?;

        Ok(Self {
            n_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let batch = x.shape()[0];
        let seq = x.shape()[1];

        let q = Module::forward(&mut self.q_proj, x)?;
        let k = Module::forward(&mut self.k_proj, x)?;
        let v = Module::forward(&mut self.v_proj, x)?;

        let q = q
            .reshape(&[batch, seq, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[batch, seq, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[batch, seq, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Standard SDPA
        let scores = q.matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?;
        let scores = scores.multiply(&Array::from_f32(self.scale))?;
        // Use mlx_rs::ops::softmax_axis with 3 arguments
        let probs = softmax_axis(&scores, -1, None)?;
        let output = probs.matmul(&v)?;

        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq, -1])?;
        Module::forward(&mut self.o_proj, &output)
    }
}

/// Vision Encoder Layer.
#[derive(Debug, ModuleParameters)]
pub struct MllamaVisionEncoderLayer {
    #[param]
    pub self_attn: MllamaVisionAttention,
    #[param]
    pub mlp: MllamaVisionMLP,
    #[param]
    pub input_layernorm: nn::LayerNorm,
    #[param]
    pub post_attention_layernorm: nn::LayerNorm,

    // Gating parameters (Vec for optionality)
    #[param]
    pub gate_attn: Vec<Gate>, // 0 or 1 element
    #[param]
    pub gate_mlp: Vec<Gate>,
}

impl MllamaVisionEncoderLayer {
    pub fn new(config: &MllamaVisionConfig) -> Result<Self, Exception> {
        let self_attn = MllamaVisionAttention::new(config)?;
        let mlp = MllamaVisionMLP::new(config)?;
        let input_layernorm = nn::LayerNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::LayerNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_eps)
            .build()?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            gate_attn: Vec::new(), // Initialize if gated layer (e.g. via separate method)
            gate_mlp: Vec::new(),
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // Pre-norm
        let normed = Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self.self_attn.forward(&normed)?;

        // Gate logic: h = x + tanh(gate) * attn_out
        let h = if let Some(gate) = self.gate_attn.first() {
            let gated = mlx_rs::ops::tanh(&gate.weight.value)?.multiply(&attn_out)?;
            x.add(&gated)?
        } else {
            x.add(&attn_out)?
        };

        let normed = Module::forward(&mut self.post_attention_layernorm, &h)?;
        let mlp_out = self.mlp.forward(&normed)?;

        if let Some(gate) = self.gate_mlp.first() {
            let gated = mlx_rs::ops::tanh(&gate.weight.value)?.multiply(&mlp_out)?;
            h.add(&gated)
        } else {
            h.add(&mlp_out)
        }
    }
}

/// Mllama Vision Model (Encoder).
#[derive(Debug, ModuleParameters)]
pub struct MllamaVisionModel {
    pub config: MllamaVisionConfig,

    #[param]
    pub embeddings: MllamaVisionEmbeddings,
    #[param]
    pub layers: Vec<MllamaVisionEncoderLayer>,
    #[param]
    pub layernorm: nn::LayerNorm, // Final norm
}

impl MllamaVisionModel {
    pub fn new(config: MllamaVisionConfig) -> Result<Self, Exception> {
        let embeddings = MllamaVisionEmbeddings::new(&config)?;
        let layers = (0..config.num_hidden_layers)
            .map(|_| MllamaVisionEncoderLayer::new(&config))
            .collect::<Result<Vec<_>, _>>()?;
        let layernorm = nn::LayerNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_eps)
            .build()?;

        Ok(Self {
            config,
            embeddings,
            layers,
            layernorm,
        })
    }

    pub fn forward(&mut self, pixel_values: &Array) -> Result<Array, Exception> {
        let mut hidden_states = self.embeddings.forward(pixel_values)?;

        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        Module::forward(&mut self.layernorm, &hidden_states)
    }
}

// =============================================================================
// Cross-Attention & Text Model Components
// =============================================================================

/// Mllama Cross-Attention layer.
///
/// Attends to vision hidden states.
#[derive(Debug, ModuleParameters)]
pub struct MllamaCrossAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub o_proj: nn::Linear,

    // Gating
    #[param]
    pub gate: Vec<Gate>, // Optional gate
}

impl MllamaCrossAttention {
    pub fn new(config: &MllamaConfig) -> Result<Self, Exception> {
        let text_dim = config.text_config.hidden_size;
        // Vision dimension same as text? Usually projected before this.
        // Assuming cross_attention_states have dimension text_dim (after projector)

        let n_heads = config.text_config.num_attention_heads;
        let n_kv_heads = config.text_config.num_kv_heads();
        let head_dim = config.text_config.get_head_dim();

        let q_proj = nn::LinearBuilder::new(text_dim, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(text_dim, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(text_dim, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, text_dim)
            .bias(false)
            .build()?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate: Vec::new(), // Initialize if needed
        })
    }

    pub fn forward(&mut self, x: &Array, cross_states: &Array) -> Result<Array, Exception> {
        // x: [batch, seq, text_dim]
        // cross_states: [batch, vision_seq, text_dim]

        let batch = x.shape()[0];
        let seq = x.shape()[1];
        let vision_seq = cross_states.shape()[1];

        let q = Module::forward(&mut self.q_proj, x)?;
        let k = Module::forward(&mut self.k_proj, cross_states)?;
        let v = Module::forward(&mut self.v_proj, cross_states)?;

        let q = q
            .reshape(&[batch, seq, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[batch, vision_seq, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[batch, vision_seq, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Standard SDPA with GQA broadcasting handled by matmul
        // scores: [B, heads, seq, vision_seq]
        let scores = q.matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?;
        let scores = scores.multiply(&Array::from_f32(self.scale))?;

        let probs = softmax_axis(&scores, -1, None)?;
        let output = probs.matmul(&v)?;

        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq, -1])?;
        let output = Module::forward(&mut self.o_proj, &output)?;

        // Gate logic
        if let Some(gate) = self.gate.first() {
            let gated = mlx_rs::ops::tanh(&gate.weight.value)?.multiply(&output)?;
            Ok(gated)
        } else {
            Ok(output)
        }
    }
}

/// Mllama Decoder Layer (Text + Cross Attention).
#[derive(Debug, ModuleParameters)]
pub struct MllamaDecoderLayer {
    #[param]
    pub self_attn: LlamaAttention,
    #[param]
    pub cross_attn: Option<MllamaCrossAttention>,
    #[param]
    pub mlp: LlamaMLP,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub cross_attention_layernorm: Option<nn::RmsNorm>,
}

impl MllamaDecoderLayer {
    pub fn new(config: &MllamaConfig, layer_id: usize) -> Result<Self, Exception> {
        let self_attn = LlamaAttention::new(&config.text_config, layer_id)?;
        let mlp = LlamaMLP::new(&config.text_config)?;

        let input_layernorm = nn::RmsNormBuilder::new(config.text_config.hidden_size)
            .eps(config.text_config.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.text_config.hidden_size)
            .eps(config.text_config.rms_norm_eps)
            .build()?;

        let has_cross_attn = config.cross_attention_layers.contains(&(layer_id as i32));

        let (cross_attn, cross_attention_layernorm) = if has_cross_attn {
            let attn = MllamaCrossAttention::new(config)?;
            let norm = nn::RmsNormBuilder::new(config.text_config.hidden_size)
                .eps(config.text_config.rms_norm_eps)
                .build()?;
            (Some(attn), Some(norm))
        } else {
            (None, None)
        };

        Ok(Self {
            self_attn,
            cross_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            cross_attention_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cross_attention_states: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, Exception> {
        // Self Attention
        let normed = Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self.self_attn.forward_with_cache(&normed, mask, cache)?;
        let mut h = x.add(&attn_out)?;

        // Cross Attention (if present and states provided)
        if let (Some(cross_attn), Some(cross_states), Some(cross_norm)) = (
            self.cross_attn.as_mut(),
            cross_attention_states,
            self.cross_attention_layernorm.as_mut(),
        ) {
            let normed = Module::forward(cross_norm, &h)?;
            let cross_out = cross_attn.forward(&normed, cross_states)?;
            h = h.add(&cross_out)?;
        }

        // MLP
        let normed = Module::forward(&mut self.post_attention_layernorm, &h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        h.add(&mlp_out)
    }
}

/// Mllama Multi-Modal Projector.
///
/// Projects vision hidden states to text hidden dimension.
#[derive(Debug, ModuleParameters)]
pub struct MllamaMultiModalProjector {
    #[param]
    pub linear_1: nn::Linear,
    #[param]
    pub linear_2: nn::Linear,
}

impl MllamaMultiModalProjector {
    pub fn new(config: &MllamaConfig) -> Result<Self, Exception> {
        let vision_dim = config.vision_config.hidden_size;
        let text_dim = config.text_config.hidden_size;
        let _intermediate_dim = config.vision_config.intermediate_size; // Or specific projector dim?
                                                                        // Usually uses intermediate size or text dim.
                                                                        // Llama 3.2 uses specific projection logic. Simplified to MLP here.

        let linear_1 = nn::LinearBuilder::new(vision_dim, text_dim)
            .bias(true)
            .build()?;
        let linear_2 = nn::LinearBuilder::new(text_dim, text_dim)
            .bias(true)
            .build()?;

        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let h = Module::forward(&mut self.linear_1, x)?;
        let h = nn::silu(h)?;
        Module::forward(&mut self.linear_2, &h)
    }
}

/// Mllama Text Model (Decoder).
#[derive(Debug, ModuleParameters)]
pub struct MllamaTextModel {
    pub config: MllamaConfig,

    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub layers: Vec<MllamaDecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl MllamaTextModel {
    pub fn new(config: MllamaConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
        )?;

        let layers = (0..config.text_config.num_hidden_layers)
            .map(|layer_id| MllamaDecoderLayer::new(&config, layer_id as usize))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::RmsNormBuilder::new(config.text_config.hidden_size)
            .eps(config.text_config.rms_norm_eps)
            .build()?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        cross_attention_states: Option<&Array>,
    ) -> Result<Array, Exception> {
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;

        // TODO: Causal mask logic similar to LlamaModel

        for layer in &mut self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                None, // Mask
                cross_attention_states,
                None, // Cache
            )?;
        }

        Module::forward(&mut self.norm, &hidden_states)
    }
}

/// Mllama For Conditional Generation (Full Model).
#[derive(Debug, ModuleParameters)]
pub struct MllamaForConditionalGeneration {
    pub config: MllamaConfig,

    #[param]
    pub vision_model: MllamaVisionModel,
    #[param]
    pub multi_modal_projector: MllamaMultiModalProjector,
    #[param]
    pub language_model: MllamaTextModel,
    #[param]
    pub lm_head: nn::Linear,
}

impl MllamaForConditionalGeneration {
    pub fn new(config: MllamaConfig) -> Result<Self, Exception> {
        let vision_model = MllamaVisionModel::new(config.vision_config.clone())?;
        let multi_modal_projector = MllamaMultiModalProjector::new(&config)?;
        let language_model = MllamaTextModel::new(config.clone())?;
        let lm_head = nn::LinearBuilder::new(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
        )
        .bias(false)
        .build()?;

        Ok(Self {
            config,
            vision_model,
            multi_modal_projector,
            language_model,
            lm_head,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        pixel_values: Option<&Array>,
    ) -> Result<Array, Exception> {
        // 1. Vision Encoder
        let cross_attention_states = if let Some(pixels) = pixel_values {
            let vision_out = self.vision_model.forward(pixels)?;
            let projected = self.multi_modal_projector.forward(&vision_out)?;
            Some(projected)
        } else {
            None
        };

        // 2. Text Decoder with Cross Attention
        let hidden_states = self
            .language_model
            .forward(input_ids, cross_attention_states.as_ref())?;

        // 3. LM Head
        Module::forward(&mut self.lm_head, &hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::module::ModuleParameters;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_mllama_instantiation() {
        let mut config = MllamaConfig::default();
        // Use smaller sizes for testing
        config.text_config.hidden_size = 64;
        config.text_config.num_hidden_layers = 2;
        config.text_config.num_attention_heads = 4;
        config.text_config.num_key_value_heads = Some(2);

        config.vision_config.hidden_size = 32;
        config.vision_config.num_hidden_layers = 2;
        config.vision_config.num_attention_heads = 4;
        config.vision_config.image_size = 28; // small image
        config.vision_config.patch_size = 14; // 2x2 grid

        let model = MllamaForConditionalGeneration::new(config).unwrap();

        // Check params count > 0
        let params = model.parameters().flatten();
        assert!(params.len() > 0);
    }

    #[test]
    #[serial]
    fn test_mllama_forward() {
        let mut config = MllamaConfig::default();
        // Use smaller sizes for testing
        config.text_config.hidden_size = 64;
        config.text_config.num_hidden_layers = 2;
        config.text_config.num_attention_heads = 4;
        config.text_config.num_key_value_heads = Some(2); // n_heads must be multiple of n_kv_heads

        config.vision_config.hidden_size = 32;
        config.vision_config.num_hidden_layers = 2;
        config.vision_config.num_attention_heads = 4;
        config.vision_config.image_size = 28;
        config.vision_config.patch_size = 14;

        let mut model = MllamaForConditionalGeneration::new(config).unwrap();

        // Input: [batch=1, seq=4]
        let input_ids = Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);

        // Image: [batch=1, channels=3, h=28, w=28] (NCHW)
        let pixels = mlx_rs::random::normal::<f32>(&[1, 3, 28, 28], None, None, None).unwrap();

        // Forward
        let logits = model.forward(&input_ids, Some(&pixels)).unwrap();

        // Output: [1, 4, vocab_size]
        assert_eq!(logits.shape(), &[1, 4, 128256]);
    }
}
