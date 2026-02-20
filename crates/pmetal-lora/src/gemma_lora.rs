//! LoRA-enabled Gemma model architecture.
//!
//! Implements Gemma and Gemma2 with LoRA adapters for efficient fine-tuning.
//! Key differences from Llama:
//! - GemmaRMSNorm: output = x * (1 + weight) instead of x * weight
//! - GeGLU instead of SwiGLU (uses GELU instead of SiLU)
//! - Embedding scaling by sqrt(hidden_size)
//!
//! # Performance Optimizations (SOTA)
//!
//! This implementation uses several state-of-the-art optimizations:
//! - **Compiled GELU**: Uses `mlx_rs::nn::gelu_approximate()` with kernel fusion
//! - **Fast RMS Norm**: Uses `mlx_rs::fast::rms_norm()` for optimized normalization
//! - **Unsloth-style Gemma norm**: Efficient +1 weight offset handling

use std::collections::HashMap;
use std::rc::Rc;

use mlx_rs::{
    builder::Builder,
    error::Exception,
    fast,
    module::{Module, ModuleParamMut, ModuleParamRef, ModuleParameters, Param},
    nested::NestedValue,
    nn, Array,
};

use pmetal_core::LoraConfig;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_mlx::kernels::{fused_sdpa, rope::apply_rope, AttentionMaskType, FusedAttentionConfig};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};
use pmetal_models::architectures::gemma::GemmaConfig;

use crate::{LoraError, LoraLinear};

/// Gemma-style RMSNorm with +1 offset.
///
/// Uses `mlx_rs::fast::rms_norm()` for optimized fused kernel execution,
/// following the Unsloth pattern for Gemma models where the output is:
/// `output = rms_norm(x) * (1 + weight)`
#[derive(Debug)]
pub struct GemmaRmsNorm {
    /// Weight parameter (stored as the raw weight, +1 applied during forward).
    pub weight: Param<Array>,
    /// Pre-computed weight + 1 for fast::rms_norm (cached for performance).
    weight_plus_one: Option<Array>,
    /// Epsilon for numerical stability.
    pub eps: f32,
}

impl GemmaRmsNorm {
    /// Create a new GemmaRmsNorm layer.
    pub fn new(hidden_size: i32, eps: f32) -> Result<Self, Exception> {
        let weight = mlx_rs::ops::zeros::<f32>(&[hidden_size])?;
        Ok(Self {
            weight: Param::new(weight),
            weight_plus_one: None,
            eps,
        })
    }

    /// Forward pass using optimized fast::rms_norm.
    ///
    /// This implementation:
    /// 1. Pre-computes `weight + 1` on first call (Gemma-specific offset)
    /// 2. Uses `mlx_rs::fast::rms_norm()` which is a fused kernel
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        // Compute weight + 1 for Gemma's +1 offset
        // Note: We compute this each time because the weight could be updated during training
        // For inference, consider using update_weight_cache() for better performance
        let weight_with_offset = self.weight.as_ref().add(&Array::from_f32(1.0))?;

        // Use the optimized fused RMS norm kernel
        fast::rms_norm(x, weight_with_offset, self.eps)
    }

    /// Update the cached weight+1 array after weight changes.
    ///
    /// Call this after loading weights or at the start of inference
    /// to avoid recomputing weight+1 on every forward pass.
    pub fn update_weight_cache(&mut self) -> Result<(), Exception> {
        let weight_with_offset = self.weight.as_ref().add(&Array::from_f32(1.0))?;
        weight_with_offset.eval()?;
        self.weight_plus_one = Some(weight_with_offset);
        Ok(())
    }

    /// Forward pass using cached weight (for inference).
    ///
    /// Requires `update_weight_cache()` to be called first.
    pub fn forward_cached(&self, x: &Array) -> Result<Array, Exception> {
        if let Some(ref cached_weight) = self.weight_plus_one {
            fast::rms_norm(x, cached_weight, self.eps)
        } else {
            // Fall back to regular forward if cache not initialized
            self.forward(x)
        }
    }
}

/// LoRA-enabled attention layer for Gemma.
#[derive(Debug)]
pub struct GemmaLoraAttention {
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads.
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale factor.
    pub scale: f32,
    /// Attention logit softcapping (Gemma2).
    pub logit_softcapping: Option<f32>,

    /// Query projection with LoRA.
    pub q_proj: LoraLinear,
    /// Key projection with LoRA.
    pub k_proj: LoraLinear,
    /// Value projection with LoRA.
    pub v_proj: LoraLinear,
    /// Output projection with LoRA.
    pub o_proj: LoraLinear,
    /// RoPE layer.
    pub rope: nn::Rope,
}

impl GemmaLoraAttention {
    /// Create a new LoRA attention layer.
    pub fn new(config: &GemmaConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_kv_heads();
        let head_dim = config.get_head_dim();
        let scale = config.attention_scale();

        let rank = lora_config.r as i32;
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;

        let q_proj = LoraLinear::new(
            config.hidden_size,
            n_heads * head_dim,
            rank,
            alpha,
            use_rslora,
            false,
        )?;
        let k_proj = LoraLinear::new(
            config.hidden_size,
            n_kv_heads * head_dim,
            rank,
            alpha,
            use_rslora,
            false,
        )?;
        let v_proj = LoraLinear::new(
            config.hidden_size,
            n_kv_heads * head_dim,
            rank,
            alpha,
            use_rslora,
            false,
        )?;
        let o_proj = LoraLinear::new(
            n_heads * head_dim,
            config.hidden_size,
            rank,
            alpha,
            use_rslora,
            false,
        )?;

        let rope = nn::RopeBuilder::new(head_dim)
            .base(config.rope_theta)
            .traditional(false)
            .build()
            .unwrap();

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            logit_softcapping: config.attn_logit_softcapping,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
        })
    }

    /// Forward pass through attention.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let queries = queries
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let keys = keys
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = values
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE
        let queries = Module::forward(&mut self.rope, &queries)?;
        let keys = Module::forward(&mut self.rope, &keys)?;

        // Expand KV heads for GQA
        let keys = if self.n_kv_heads < self.n_heads {
            let repeats = self.n_heads / self.n_kv_heads;
            expand_kv_heads(&keys, repeats)?
        } else {
            keys
        };
        let values = if self.n_kv_heads < self.n_heads {
            let repeats = self.n_heads / self.n_kv_heads;
            expand_kv_heads(&values, repeats)?
        } else {
            values
        };

        // Scaled dot-product attention
        let mut scores = queries.matmul(&keys.transpose_axes(&[0, 1, 3, 2])?)?;
        scores = scores.multiply(Array::from_f32(self.scale))?;

        // Apply logit softcapping (Gemma2)
        if let Some(cap) = self.logit_softcapping {
            let cap_val = Array::from_f32(cap);
            scores = scores.divide(&cap_val)?;
            scores = mlx_rs::ops::tanh(&scores)?;
            scores = scores.multiply(&cap_val)?;
        }

        let scores = if let Some(m) = mask {
            scores.add(m)?
        } else {
            scores
        };

        let weights = mlx_rs::ops::softmax_axis(&scores, -1, None)?;
        let output = weights.matmul(&values)?;

        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, -1])?;

        self.o_proj.forward(&output).map_err(LoraError::from)
    }

    /// Forward pass with KV cache for efficient inference.
    ///
    /// This method is optimized for autoregressive generation with O(n)
    /// complexity per token instead of O(n²).
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, hidden_size]
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional (KVCache, layer_idx) tuple for cached generation
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Project to Q, K, V using LoRA layers
        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape for multi-head attention: [B, L, heads, head_dim]
        let queries = queries.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        let keys = keys.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;
        let values = values.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;

        // Transpose for attention: [B, heads, L, head_dim]
        let queries = queries.transpose_axes(&[0, 2, 1, 3])?;
        let keys = keys.transpose_axes(&[0, 2, 1, 3])?;
        let values = values.transpose_axes(&[0, 2, 1, 3])?;

        // Get RoPE offset and apply RoPE
        let (queries, keys, values) = if let Some((ref cache_ref, _layer_idx)) = cache {
            let offset = cache_ref.rope_offset();
            let queries = apply_rope(&queries, self.head_dim, false, self.rope.base, 1.0, offset)?;
            let keys = apply_rope(&keys, self.head_dim, false, self.rope.base, 1.0, offset)?;
            (queries, keys, values)
        } else {
            let queries = Module::forward(&mut self.rope, &queries)?;
            let keys = Module::forward(&mut self.rope, &keys)?;
            (queries, keys, values)
        };

        // Handle KV cache update - keys/values are already in [B, heads, seq, head_dim] format
        let (keys, values) = if let Some((cache, layer_idx)) = cache {
            cache
                .update_and_fetch(layer_idx, &keys, &values)
                .map_err(|e| LoraError::Mlx(e))?
        } else {
            (keys, values)
        };

        // Use fused attention kernel for inference
        let mut attn_config =
            FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
                .with_scale(self.scale)
                .with_mask_type(AttentionMaskType::Causal);

        // Apply logit softcapping if configured (Gemma2)
        if let Some(cap) = self.logit_softcapping {
            attn_config = attn_config.with_logit_softcapping(cap);
        }

        let output = fused_sdpa(&queries, &keys, &values, &attn_config, mask)
            .map_err(|e| LoraError::Mlx(e))?;

        // Reshape back: [B, heads, L, head_dim] -> [B, L, hidden]
        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, -1])?;

        // Output projection
        self.o_proj.forward(&output).map_err(LoraError::from)
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self.v_proj.num_trainable_params()
            + self.o_proj.num_trainable_params()
    }
}

fn expand_kv_heads(x: &Array, repeats: i32) -> Result<Array, Exception> {
    let shape = x.shape();
    let batch = shape[0];
    let n_kv_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    let x = x.reshape(&[batch, n_kv_heads, 1, seq_len, head_dim])?;
    let x = mlx_rs::ops::broadcast_to(&x, &[batch, n_kv_heads, repeats, seq_len, head_dim])?;
    x.reshape(&[batch, n_kv_heads * repeats, seq_len, head_dim])
}

/// LoRA-enabled MLP layer for Gemma (GeGLU).
#[derive(Debug)]
pub struct GemmaLoraMLP {
    /// Gate projection with LoRA.
    pub gate_proj: LoraLinear,
    /// Up projection with LoRA.
    pub up_proj: LoraLinear,
    /// Down projection with LoRA.
    pub down_proj: LoraLinear,
}

impl GemmaLoraMLP {
    /// Create a new LoRA MLP layer.
    pub fn new(config: &GemmaConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let rank = lora_config.r as i32;
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;

        let gate_proj = LoraLinear::new(
            config.hidden_size,
            config.intermediate_size,
            rank,
            alpha,
            use_rslora,
            false,
        )?;
        let up_proj = LoraLinear::new(
            config.hidden_size,
            config.intermediate_size,
            rank,
            alpha,
            use_rslora,
            false,
        )?;
        let down_proj = LoraLinear::new(
            config.intermediate_size,
            config.hidden_size,
            rank,
            alpha,
            use_rslora,
            false,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass (GeGLU activation).
    ///
    /// Uses `nn::gelu_approximate()` which is a compiled/fused kernel for
    /// better performance than the manual tanh approximation.
    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let gate = self.gate_proj.forward(x)?;
        // Use the optimized compiled GELU approximation (equivalent to tanh approximation)
        let gate = nn::gelu_approximate(&gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = gate.multiply(&up)?;
        self.down_proj.forward(&hidden)
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.gate_proj.num_trainable_params()
            + self.up_proj.num_trainable_params()
            + self.down_proj.num_trainable_params()
    }
}

/// LoRA-enabled Gemma decoder layer.
#[derive(Debug)]
pub struct GemmaLoraDecoderLayer {
    /// Self-attention layer with LoRA.
    pub self_attn: GemmaLoraAttention,
    /// MLP layer with LoRA.
    pub mlp: GemmaLoraMLP,
    /// Input layer norm.
    pub input_layernorm: GemmaRmsNorm,
    /// Post-attention layer norm.
    pub post_attention_layernorm: GemmaRmsNorm,
}

impl GemmaLoraDecoderLayer {
    /// Create a new decoder layer with LoRA.
    pub fn new(config: &GemmaConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let self_attn = GemmaLoraAttention::new(config, lora_config)?;
        let mlp = GemmaLoraMLP::new(config, lora_config)?;

        let input_layernorm = GemmaRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = GemmaRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, mask)?;
        let h = x.add(&attn_out)?;

        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok(h.add(&mlp_out)?)
    }

    /// Forward pass with KV cache for efficient inference.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional (KVCache, layer_idx) tuple for cached generation
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        // Pre-norm + attention + residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward_with_cache(&normed, mask, cache)?;
        let h = x.add(&attn_out)?;

        // Pre-norm + MLP + residual
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok(h.add(&mlp_out)?)
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.self_attn.num_trainable_params() + self.mlp.num_trainable_params()
    }
}

/// LoRA-enabled Gemma2 decoder layer with extra normalization.
#[derive(Debug)]
pub struct Gemma2LoraDecoderLayer {
    /// Self-attention layer with LoRA.
    pub self_attn: GemmaLoraAttention,
    /// MLP layer with LoRA.
    pub mlp: GemmaLoraMLP,
    /// Input layer norm.
    pub input_layernorm: GemmaRmsNorm,
    /// Post-attention layer norm.
    pub post_attention_layernorm: GemmaRmsNorm,
    /// Pre-feedforward layer norm (Gemma2 specific).
    pub pre_feedforward_layernorm: GemmaRmsNorm,
    /// Post-feedforward layer norm (Gemma2 specific).
    pub post_feedforward_layernorm: GemmaRmsNorm,
}

impl Gemma2LoraDecoderLayer {
    /// Create a new Gemma2 decoder layer with LoRA.
    pub fn new(config: &GemmaConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let self_attn = GemmaLoraAttention::new(config, lora_config)?;
        let mlp = GemmaLoraMLP::new(config, lora_config)?;

        let input_layernorm = GemmaRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = GemmaRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let pre_feedforward_layernorm = GemmaRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_feedforward_layernorm =
            GemmaRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    /// Forward pass with extra normalization (Gemma2 style).
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        // Pre-norm + attention
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, mask)?;
        // Post-attention norm before residual (Gemma2 specific)
        let attn_out = self.post_attention_layernorm.forward(&attn_out)?;
        let h = x.add(&attn_out)?;

        // Pre-feedforward norm + MLP
        let normed = self.pre_feedforward_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        // Post-feedforward norm before residual (Gemma2 specific)
        let mlp_out = self.post_feedforward_layernorm.forward(&mlp_out)?;
        Ok(h.add(&mlp_out)?)
    }

    /// Forward pass with KV cache for efficient inference.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional (KVCache, layer_idx) tuple for cached generation
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        // Pre-norm + attention
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward_with_cache(&normed, mask, cache)?;
        // Post-attention norm before residual (Gemma2 specific)
        let attn_out = self.post_attention_layernorm.forward(&attn_out)?;
        let h = x.add(&attn_out)?;

        // Pre-feedforward norm + MLP
        let normed = self.pre_feedforward_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        // Post-feedforward norm before residual (Gemma2 specific)
        let mlp_out = self.post_feedforward_layernorm.forward(&mlp_out)?;
        Ok(h.add(&mlp_out)?)
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.self_attn.num_trainable_params() + self.mlp.num_trainable_params()
    }
}

/// Container for Gemma LoRA layers (supports both Gemma and Gemma2).
#[derive(Debug)]
pub enum GemmaLoraLayers {
    /// Gemma v1 layers.
    Gemma1(Vec<GemmaLoraDecoderLayer>),
    /// Gemma v2 layers with extra normalization.
    Gemma2(Vec<Gemma2LoraDecoderLayer>),
}

impl GemmaLoraLayers {
    /// Get the number of layers.
    pub fn len(&self) -> usize {
        match self {
            Self::Gemma1(layers) => layers.len(),
            Self::Gemma2(layers) => layers.len(),
        }
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        match self {
            Self::Gemma1(layers) => layers.iter().map(|l| l.num_trainable_params()).sum(),
            Self::Gemma2(layers) => layers.iter().map(|l| l.num_trainable_params()).sum(),
        }
    }
}

/// LoRA-enabled Gemma model (without LM head).
#[derive(Debug)]
pub struct GemmaLoraModel {
    /// Configuration.
    pub config: GemmaConfig,
    /// LoRA configuration.
    pub lora_config: LoraConfig,
    /// Token embeddings (frozen).
    pub embed_tokens: nn::Embedding,
    /// Transformer layers with LoRA (supports both Gemma and Gemma2).
    pub layers: GemmaLoraLayers,
    /// Final layer norm (frozen).
    pub norm: GemmaRmsNorm,
    /// Embedding scale factor.
    pub embedding_scale: f32,
}

impl GemmaLoraModel {
    /// Create a new LoRA Gemma model.
    pub fn new(config: GemmaConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;
        let embedding_scale = config.embedding_scale();

        // Create appropriate layer type based on config
        let layers = if config.is_gemma2 {
            GemmaLoraLayers::Gemma2(
                (0..config.num_hidden_layers)
                    .map(|_| Gemma2LoraDecoderLayer::new(&config, &lora_config))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            GemmaLoraLayers::Gemma1(
                (0..config.num_hidden_layers)
                    .map(|_| GemmaLoraDecoderLayer::new(&config, &lora_config))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        };

        let norm = GemmaRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            config,
            lora_config,
            embed_tokens,
            layers,
            norm,
            embedding_scale,
        })
    }

    /// Forward pass.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        self.forward_with_checkpoint(input_ids, mask, None)
    }

    /// Forward pass with optional gradient checkpointing.
    pub fn forward_with_checkpoint(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        checkpoint_config: Option<&CheckpointConfig>,
    ) -> Result<Array, LoraError> {
        // Get embeddings and scale
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;
        let scale = Array::from_f32(self.embedding_scale);
        hidden_states = hidden_states.multiply(&scale)?;

        let mask = if mask.is_none() {
            let seq_len = input_ids.dim(1);
            Some(create_causal_mask(seq_len)?)
        } else {
            mask.cloned()
        };

        let layers_per_block = checkpoint_config
            .map(|c| c.layers_per_block)
            .unwrap_or(usize::MAX);
        let checkpointing_enabled = checkpoint_config.map(|c| c.enabled).unwrap_or(false);

        match &mut self.layers {
            GemmaLoraLayers::Gemma1(layers) => {
                for (idx, layer) in layers.iter_mut().enumerate() {
                    hidden_states = layer.forward(&hidden_states, mask.as_ref())?;

                    // Checkpoint boundary marker
                    // NOTE: We do NOT call eval() here - that breaks the gradient computation graph.
                    if checkpointing_enabled && (idx + 1) % layers_per_block == 0 {
                        tracing::trace!("Checkpoint boundary at layer {}", idx + 1);
                    }
                }
            }
            GemmaLoraLayers::Gemma2(layers) => {
                for (idx, layer) in layers.iter_mut().enumerate() {
                    hidden_states = layer.forward(&hidden_states, mask.as_ref())?;

                    // Checkpoint boundary marker
                    // NOTE: We do NOT call eval() here - that breaks the gradient computation graph.
                    if checkpointing_enabled && (idx + 1) % layers_per_block == 0 {
                        tracing::trace!("Checkpoint boundary at layer {}", idx + 1);
                    }
                }
            }
        }

        Ok(self.norm.forward(&hidden_states)?)
    }

    /// Forward pass with KV cache for efficient inference.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional mutable reference to KV cache
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        // Get embeddings and scale
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;
        let scale = Array::from_f32(self.embedding_scale);
        hidden_states = hidden_states.multiply(&scale)?;

        // Don't create explicit causal mask - fused SDPA handles it internally
        // with proper dtype handling. Only pass through user-provided masks.

        // Pass through transformer layers
        match (&mut self.layers, cache) {
            (GemmaLoraLayers::Gemma1(layers), Some(cache)) => {
                for (layer_idx, layer) in layers.iter_mut().enumerate() {
                    hidden_states =
                        layer.forward_with_cache(&hidden_states, mask, Some((cache, layer_idx)))?;
                }
            }
            (GemmaLoraLayers::Gemma2(layers), Some(cache)) => {
                for (layer_idx, layer) in layers.iter_mut().enumerate() {
                    hidden_states =
                        layer.forward_with_cache(&hidden_states, mask, Some((cache, layer_idx)))?;
                }
            }
            (GemmaLoraLayers::Gemma1(layers), None) => {
                let seq_len = input_ids.dim(1);
                let mask = Some(create_causal_mask(seq_len)?);
                for layer in layers {
                    hidden_states = layer.forward(&hidden_states, mask.as_ref())?;
                }
            }
            (GemmaLoraLayers::Gemma2(layers), None) => {
                let seq_len = input_ids.dim(1);
                let mask = Some(create_causal_mask(seq_len)?);
                for layer in layers {
                    hidden_states = layer.forward(&hidden_states, mask.as_ref())?;
                }
            }
        }

        // Final norm
        Ok(self.norm.forward(&hidden_states)?)
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.layers.num_trainable_params()
    }
}

/// LoRA-enabled Gemma model with LM head.
#[derive(Debug)]
pub struct GemmaLoraForCausalLM {
    /// Base model with LoRA.
    pub model: GemmaLoraModel,
    /// Gradient checkpointing configuration.
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl GemmaLoraForCausalLM {
    /// Create a new LoRA Gemma model with LM head.
    pub fn new(config: GemmaConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let model = GemmaLoraModel::new(config, lora_config)?;

        Ok(Self {
            model,
            checkpoint_config: None,
        })
    }

    /// Enable gradient checkpointing.
    pub fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        self.checkpoint_config = Some(CheckpointConfig {
            enabled: true,
            layers_per_block,
            eval_at_boundaries: true,
        });
    }

    /// Disable gradient checkpointing.
    pub fn disable_gradient_checkpointing(&mut self) {
        self.checkpoint_config = None;
    }

    /// Forward pass producing logits.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let checkpoint_config = self.checkpoint_config.clone();
        self.forward_with_checkpoint(input_ids, mask, checkpoint_config.as_ref())
    }

    /// Forward pass with optional gradient checkpointing.
    pub fn forward_with_checkpoint(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        checkpoint_config: Option<&CheckpointConfig>,
    ) -> Result<Array, LoraError> {
        let hidden_states =
            self.model
                .forward_with_checkpoint(input_ids, mask, checkpoint_config)?;
        // Gemma always ties embeddings
        Ok(self.model.embed_tokens.as_linear(&hidden_states)?)
    }

    /// Forward pass with KV cache for efficient inference.
    ///
    /// KV caching provides O(n) complexity per token instead of O(n²),
    /// enabling fast autoregressive generation.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `mask` - Optional attention mask
    /// * `cache` - Optional mutable reference to KV cache
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let hidden_states = self.model.forward_with_cache(input_ids, mask, cache)?;
        // Gemma always ties embeddings
        Ok(self.model.embed_tokens.as_linear(&hidden_states)?)
    }

    /// Create a KV cache for this model.
    ///
    /// # Arguments
    /// * `max_seq_len` - Maximum sequence length to cache
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        let config = KVCacheConfig::new(
            self.model.config.num_hidden_layers as usize,
            max_seq_len,
            self.model.config.num_kv_heads() as usize,
            self.model.config.get_head_dim() as usize,
        );
        KVCache::new(config)
    }

    /// Get all trainable LoRA parameters.
    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        let mut params = HashMap::new();

        // Helper to add layer params
        macro_rules! add_layer_params {
            ($layer:expr, $i:expr) => {
                let prefix = format!("layers.{}", $i);
                params.insert(
                    Rc::from(format!("{}.self_attn.q_proj.lora_a", prefix)),
                    $layer.self_attn.q_proj.lora_a.clone(),
                );
                params.insert(
                    Rc::from(format!("{}.self_attn.q_proj.lora_b", prefix)),
                    $layer.self_attn.q_proj.lora_b.clone(),
                );
                params.insert(
                    Rc::from(format!("{}.self_attn.k_proj.lora_a", prefix)),
                    $layer.self_attn.k_proj.lora_a.clone(),
                );
                params.insert(
                    Rc::from(format!("{}.self_attn.k_proj.lora_b", prefix)),
                    $layer.self_attn.k_proj.lora_b.clone(),
                );
                params.insert(
                    Rc::from(format!("{}.self_attn.v_proj.lora_a", prefix)),
                    $layer.self_attn.v_proj.lora_a.clone(),
                );
                params.insert(
                    Rc::from(format!("{}.self_attn.v_proj.lora_b", prefix)),
                    $layer.self_attn.v_proj.lora_b.clone(),
                );
                params.insert(
                    Rc::from(format!("{}.self_attn.o_proj.lora_a", prefix)),
                    $layer.self_attn.o_proj.lora_a.clone(),
                );
                params.insert(
                    Rc::from(format!("{}.self_attn.o_proj.lora_b", prefix)),
                    $layer.self_attn.o_proj.lora_b.clone(),
                );
                params.insert(
                    Rc::from(format!("{}.mlp.gate_proj.lora_a", prefix)),
                    $layer.mlp.gate_proj.lora_a.clone(),
                );
                params.insert(
                    Rc::from(format!("{}.mlp.gate_proj.lora_b", prefix)),
                    $layer.mlp.gate_proj.lora_b.clone(),
                );
                params.insert(
                    Rc::from(format!("{}.mlp.up_proj.lora_a", prefix)),
                    $layer.mlp.up_proj.lora_a.clone(),
                );
                params.insert(
                    Rc::from(format!("{}.mlp.up_proj.lora_b", prefix)),
                    $layer.mlp.up_proj.lora_b.clone(),
                );
                params.insert(
                    Rc::from(format!("{}.mlp.down_proj.lora_a", prefix)),
                    $layer.mlp.down_proj.lora_a.clone(),
                );
                params.insert(
                    Rc::from(format!("{}.mlp.down_proj.lora_b", prefix)),
                    $layer.mlp.down_proj.lora_b.clone(),
                );
            };
        }

        match &self.model.layers {
            GemmaLoraLayers::Gemma1(layers) => {
                for (i, layer) in layers.iter().enumerate() {
                    add_layer_params!(layer, i);
                }
            }
            GemmaLoraLayers::Gemma2(layers) => {
                for (i, layer) in layers.iter().enumerate() {
                    add_layer_params!(layer, i);
                }
            }
        }

        params
    }

    /// Set LoRA parameters.
    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        macro_rules! set_layer_params {
            ($layer:expr, $i:expr) => {
                let prefix = format!("layers.{}", $i);
                macro_rules! set_param {
                    ($param:expr, $key:expr) => {
                        if let Some(value) = params.get(&Rc::from($key)) {
                            $param = value.clone();
                        }
                    };
                }
                set_param!(
                    $layer.self_attn.q_proj.lora_a,
                    format!("{}.self_attn.q_proj.lora_a", prefix)
                );
                set_param!(
                    $layer.self_attn.q_proj.lora_b,
                    format!("{}.self_attn.q_proj.lora_b", prefix)
                );
                set_param!(
                    $layer.self_attn.k_proj.lora_a,
                    format!("{}.self_attn.k_proj.lora_a", prefix)
                );
                set_param!(
                    $layer.self_attn.k_proj.lora_b,
                    format!("{}.self_attn.k_proj.lora_b", prefix)
                );
                set_param!(
                    $layer.self_attn.v_proj.lora_a,
                    format!("{}.self_attn.v_proj.lora_a", prefix)
                );
                set_param!(
                    $layer.self_attn.v_proj.lora_b,
                    format!("{}.self_attn.v_proj.lora_b", prefix)
                );
                set_param!(
                    $layer.self_attn.o_proj.lora_a,
                    format!("{}.self_attn.o_proj.lora_a", prefix)
                );
                set_param!(
                    $layer.self_attn.o_proj.lora_b,
                    format!("{}.self_attn.o_proj.lora_b", prefix)
                );
                set_param!(
                    $layer.mlp.gate_proj.lora_a,
                    format!("{}.mlp.gate_proj.lora_a", prefix)
                );
                set_param!(
                    $layer.mlp.gate_proj.lora_b,
                    format!("{}.mlp.gate_proj.lora_b", prefix)
                );
                set_param!(
                    $layer.mlp.up_proj.lora_a,
                    format!("{}.mlp.up_proj.lora_a", prefix)
                );
                set_param!(
                    $layer.mlp.up_proj.lora_b,
                    format!("{}.mlp.up_proj.lora_b", prefix)
                );
                set_param!(
                    $layer.mlp.down_proj.lora_a,
                    format!("{}.mlp.down_proj.lora_a", prefix)
                );
                set_param!(
                    $layer.mlp.down_proj.lora_b,
                    format!("{}.mlp.down_proj.lora_b", prefix)
                );
            };
        }

        match &mut self.model.layers {
            GemmaLoraLayers::Gemma1(layers) => {
                for (i, layer) in layers.iter_mut().enumerate() {
                    set_layer_params!(layer, i);
                }
            }
            GemmaLoraLayers::Gemma2(layers) => {
                for (i, layer) in layers.iter_mut().enumerate() {
                    set_layer_params!(layer, i);
                }
            }
        }
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    /// Get configuration.
    pub fn config(&self) -> &GemmaConfig {
        &self.model.config
    }

    /// Get LoRA configuration.
    pub fn lora_config(&self) -> &LoraConfig {
        &self.model.lora_config
    }

    /// Save LoRA weights to safetensors.
    pub fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        let params = self.lora_parameters();
        Array::save_safetensors(params, None, path)?;
        Ok(())
    }

    /// Load LoRA weights from safetensors.
    ///
    /// # Arguments
    /// * `path` - Path to either a directory containing `lora_weights.safetensors` or a direct `.safetensors` file
    pub fn load_lora_weights(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), LoraError> {
        let path = path.as_ref();
        let file_path = if path.is_dir() {
            path.join("lora_weights.safetensors")
        } else {
            path.to_path_buf()
        };
        let loaded = Array::load_safetensors(&file_path)?;

        macro_rules! load_layer_params {
            ($layer:expr, $i:expr, $loaded:expr) => {
                let prefix = format!("layers.{}", $i);
                macro_rules! load_param {
                    ($param:expr, $key:expr) => {
                        if let Some(value) = $loaded.get(&Rc::from($key) as &str) {
                            $param = value.clone();
                        }
                    };
                }
                load_param!(
                    $layer.self_attn.q_proj.lora_a,
                    format!("{}.self_attn.q_proj.lora_a", prefix)
                );
                load_param!(
                    $layer.self_attn.q_proj.lora_b,
                    format!("{}.self_attn.q_proj.lora_b", prefix)
                );
                load_param!(
                    $layer.self_attn.k_proj.lora_a,
                    format!("{}.self_attn.k_proj.lora_a", prefix)
                );
                load_param!(
                    $layer.self_attn.k_proj.lora_b,
                    format!("{}.self_attn.k_proj.lora_b", prefix)
                );
                load_param!(
                    $layer.self_attn.v_proj.lora_a,
                    format!("{}.self_attn.v_proj.lora_a", prefix)
                );
                load_param!(
                    $layer.self_attn.v_proj.lora_b,
                    format!("{}.self_attn.v_proj.lora_b", prefix)
                );
                load_param!(
                    $layer.self_attn.o_proj.lora_a,
                    format!("{}.self_attn.o_proj.lora_a", prefix)
                );
                load_param!(
                    $layer.self_attn.o_proj.lora_b,
                    format!("{}.self_attn.o_proj.lora_b", prefix)
                );
                load_param!(
                    $layer.mlp.gate_proj.lora_a,
                    format!("{}.mlp.gate_proj.lora_a", prefix)
                );
                load_param!(
                    $layer.mlp.gate_proj.lora_b,
                    format!("{}.mlp.gate_proj.lora_b", prefix)
                );
                load_param!(
                    $layer.mlp.up_proj.lora_a,
                    format!("{}.mlp.up_proj.lora_a", prefix)
                );
                load_param!(
                    $layer.mlp.up_proj.lora_b,
                    format!("{}.mlp.up_proj.lora_b", prefix)
                );
                load_param!(
                    $layer.mlp.down_proj.lora_a,
                    format!("{}.mlp.down_proj.lora_a", prefix)
                );
                load_param!(
                    $layer.mlp.down_proj.lora_b,
                    format!("{}.mlp.down_proj.lora_b", prefix)
                );
            };
        }

        match &mut self.model.layers {
            GemmaLoraLayers::Gemma1(layers) => {
                for (i, layer) in layers.iter_mut().enumerate() {
                    load_layer_params!(layer, i, loaded);
                }
            }
            GemmaLoraLayers::Gemma2(layers) => {
                for (i, layer) in layers.iter_mut().enumerate() {
                    load_layer_params!(layer, i, loaded);
                }
            }
        }

        Ok(())
    }

    /// Load base model weights.
    pub fn load_base_weights(&mut self, weights: &HashMap<String, Array>) -> Result<(), LoraError> {
        // Load embed_tokens
        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        // Helper macro for loading layer weights
        macro_rules! load_layer_base_weights {
            ($layer:expr, $prefix:expr, $weights:expr) => {
                if let Some(w) = $weights.get(&format!("{}.self_attn.q_proj.weight", $prefix)) {
                    $layer.self_attn.q_proj.weight = w.clone();
                }
                if let Some(w) = $weights.get(&format!("{}.self_attn.k_proj.weight", $prefix)) {
                    $layer.self_attn.k_proj.weight = w.clone();
                }
                if let Some(w) = $weights.get(&format!("{}.self_attn.v_proj.weight", $prefix)) {
                    $layer.self_attn.v_proj.weight = w.clone();
                }
                if let Some(w) = $weights.get(&format!("{}.self_attn.o_proj.weight", $prefix)) {
                    $layer.self_attn.o_proj.weight = w.clone();
                }
                if let Some(w) = $weights.get(&format!("{}.mlp.gate_proj.weight", $prefix)) {
                    $layer.mlp.gate_proj.weight = w.clone();
                }
                if let Some(w) = $weights.get(&format!("{}.mlp.up_proj.weight", $prefix)) {
                    $layer.mlp.up_proj.weight = w.clone();
                }
                if let Some(w) = $weights.get(&format!("{}.mlp.down_proj.weight", $prefix)) {
                    $layer.mlp.down_proj.weight = w.clone();
                }
                if let Some(w) = $weights.get(&format!("{}.input_layernorm.weight", $prefix)) {
                    $layer.input_layernorm.weight = Param::new(w.clone());
                }
                if let Some(w) =
                    $weights.get(&format!("{}.post_attention_layernorm.weight", $prefix))
                {
                    $layer.post_attention_layernorm.weight = Param::new(w.clone());
                }
            };
        }

        // Load transformer layers
        match &mut self.model.layers {
            GemmaLoraLayers::Gemma1(layers) => {
                for (i, layer) in layers.iter_mut().enumerate() {
                    let prefix = format!("model.layers.{}", i);
                    load_layer_base_weights!(layer, prefix, weights);
                }
            }
            GemmaLoraLayers::Gemma2(layers) => {
                for (i, layer) in layers.iter_mut().enumerate() {
                    let prefix = format!("model.layers.{}", i);
                    load_layer_base_weights!(layer, prefix, weights);
                    // Gemma2 extra layernorms
                    if let Some(w) =
                        weights.get(&format!("{}.pre_feedforward_layernorm.weight", prefix))
                    {
                        layer.pre_feedforward_layernorm.weight = Param::new(w.clone());
                    }
                    if let Some(w) =
                        weights.get(&format!("{}.post_feedforward_layernorm.weight", prefix))
                    {
                        layer.post_feedforward_layernorm.weight = Param::new(w.clone());
                    }
                }
            }
        }

        // Load final norm
        if let Some(w) = weights.get("model.norm.weight") {
            self.model.norm.weight = Param::new(w.clone());
        }

        Ok(())
    }

    /// Load base model weights from directory.
    pub fn load_base_weights_from_dir(
        &mut self,
        model_dir: impl AsRef<std::path::Path>,
    ) -> Result<(), LoraError> {
        let model_dir = model_dir.as_ref();

        let single_file = model_dir.join("model.safetensors");
        if single_file.exists() {
            let weights = Array::load_safetensors(&single_file)?;
            return self.load_base_weights(&weights);
        }

        let index_path = model_dir.join("model.safetensors.index.json");
        if !index_path.exists() {
            return Err(LoraError::Mlx(Exception::custom(
                "No model.safetensors or model.safetensors.index.json found".to_string(),
            )));
        }

        let index_content = std::fs::read_to_string(&index_path)
            .map_err(|e| LoraError::Mlx(Exception::custom(e.to_string())))?;

        #[derive(serde::Deserialize)]
        struct WeightIndex {
            weight_map: HashMap<String, String>,
        }

        let index: WeightIndex = serde_json::from_str(&index_content)
            .map_err(|e| LoraError::Mlx(Exception::custom(e.to_string())))?;

        let shard_files: std::collections::HashSet<&String> = index.weight_map.values().collect();

        let mut all_weights = HashMap::new();
        for shard_file in shard_files {
            let shard_path = model_dir.join(shard_file);
            let shard_weights = Array::load_safetensors(&shard_path)?;
            all_weights.extend(shard_weights);
        }

        self.load_base_weights(&all_weights)
    }

    /// Evaluate all model parameters.
    pub fn eval_all(&self) -> Result<(), LoraError> {
        self.model.embed_tokens.weight.value.as_ref().eval()?;

        macro_rules! eval_layer {
            ($layer:expr) => {
                $layer.self_attn.q_proj.weight.eval()?;
                $layer.self_attn.k_proj.weight.eval()?;
                $layer.self_attn.v_proj.weight.eval()?;
                $layer.self_attn.o_proj.weight.eval()?;
                $layer.mlp.gate_proj.weight.eval()?;
                $layer.mlp.up_proj.weight.eval()?;
                $layer.mlp.down_proj.weight.eval()?;
                $layer.self_attn.q_proj.lora_a.eval()?;
                $layer.self_attn.q_proj.lora_b.eval()?;
                $layer.self_attn.k_proj.lora_a.eval()?;
                $layer.self_attn.k_proj.lora_b.eval()?;
                $layer.self_attn.v_proj.lora_a.eval()?;
                $layer.self_attn.v_proj.lora_b.eval()?;
                $layer.self_attn.o_proj.lora_a.eval()?;
                $layer.self_attn.o_proj.lora_b.eval()?;
                $layer.mlp.gate_proj.lora_a.eval()?;
                $layer.mlp.gate_proj.lora_b.eval()?;
                $layer.mlp.up_proj.lora_a.eval()?;
                $layer.mlp.up_proj.lora_b.eval()?;
                $layer.mlp.down_proj.lora_a.eval()?;
                $layer.mlp.down_proj.lora_b.eval()?;
                $layer.input_layernorm.weight.value.as_ref().eval()?;
                $layer
                    .post_attention_layernorm
                    .weight
                    .value
                    .as_ref()
                    .eval()?;
            };
        }

        match &self.model.layers {
            GemmaLoraLayers::Gemma1(layers) => {
                for layer in layers {
                    eval_layer!(layer);
                }
            }
            GemmaLoraLayers::Gemma2(layers) => {
                for layer in layers {
                    eval_layer!(layer);
                    layer
                        .pre_feedforward_layernorm
                        .weight
                        .value
                        .as_ref()
                        .eval()?;
                    layer
                        .post_feedforward_layernorm
                        .weight
                        .value
                        .as_ref()
                        .eval()?;
                }
            }
        }

        self.model.norm.weight.value.as_ref().eval()?;
        Ok(())
    }
}

impl ModuleParameters for GemmaLoraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.model.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();

        // Helper macro to build layer params for any layer type with self_attn and mlp
        macro_rules! build_layer_params_ref {
            ($layer:expr, $i:expr, $params:expr) => {{
                let prefix: Rc<str> = Rc::from(format!("layers.{}", $i));
                let mut layer_params = HashMap::new();

                let mut attn_params = HashMap::new();
                let mut q_params = HashMap::new();
                q_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&$layer.self_attn.q_proj.lora_a),
                );
                q_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&$layer.self_attn.q_proj.lora_b),
                );
                attn_params.insert(Rc::from("q_proj"), NestedValue::Map(q_params));

                let mut k_params = HashMap::new();
                k_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&$layer.self_attn.k_proj.lora_a),
                );
                k_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&$layer.self_attn.k_proj.lora_b),
                );
                attn_params.insert(Rc::from("k_proj"), NestedValue::Map(k_params));

                let mut v_params = HashMap::new();
                v_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&$layer.self_attn.v_proj.lora_a),
                );
                v_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&$layer.self_attn.v_proj.lora_b),
                );
                attn_params.insert(Rc::from("v_proj"), NestedValue::Map(v_params));

                let mut o_params = HashMap::new();
                o_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&$layer.self_attn.o_proj.lora_a),
                );
                o_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&$layer.self_attn.o_proj.lora_b),
                );
                attn_params.insert(Rc::from("o_proj"), NestedValue::Map(o_params));

                layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

                let mut mlp_params = HashMap::new();
                let mut gate_params = HashMap::new();
                gate_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&$layer.mlp.gate_proj.lora_a),
                );
                gate_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&$layer.mlp.gate_proj.lora_b),
                );
                mlp_params.insert(Rc::from("gate_proj"), NestedValue::Map(gate_params));

                let mut up_params = HashMap::new();
                up_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&$layer.mlp.up_proj.lora_a),
                );
                up_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&$layer.mlp.up_proj.lora_b),
                );
                mlp_params.insert(Rc::from("up_proj"), NestedValue::Map(up_params));

                let mut down_params = HashMap::new();
                down_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&$layer.mlp.down_proj.lora_a),
                );
                down_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&$layer.mlp.down_proj.lora_b),
                );
                mlp_params.insert(Rc::from("down_proj"), NestedValue::Map(down_params));

                layer_params.insert(Rc::from("mlp"), NestedValue::Map(mlp_params));
                $params.insert(prefix, NestedValue::Map(layer_params));
            }};
        }

        match &self.model.layers {
            GemmaLoraLayers::Gemma1(layers) => {
                for (i, layer) in layers.iter().enumerate() {
                    build_layer_params_ref!(layer, i, params);
                }
            }
            GemmaLoraLayers::Gemma2(layers) => {
                for (i, layer) in layers.iter().enumerate() {
                    build_layer_params_ref!(layer, i, params);
                }
            }
        }

        params
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut params = ModuleParamMut::new();

        // Helper macro to build mutable layer params for any layer type with self_attn and mlp
        macro_rules! build_layer_params_mut {
            ($layer:expr, $i:expr, $params:expr) => {{
                let prefix: Rc<str> = Rc::from(format!("layers.{}", $i));
                let mut layer_params = HashMap::new();

                let mut attn_params = HashMap::new();
                let mut q_params = HashMap::new();
                q_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&mut $layer.self_attn.q_proj.lora_a),
                );
                q_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&mut $layer.self_attn.q_proj.lora_b),
                );
                attn_params.insert(Rc::from("q_proj"), NestedValue::Map(q_params));

                let mut k_params = HashMap::new();
                k_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&mut $layer.self_attn.k_proj.lora_a),
                );
                k_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&mut $layer.self_attn.k_proj.lora_b),
                );
                attn_params.insert(Rc::from("k_proj"), NestedValue::Map(k_params));

                let mut v_params = HashMap::new();
                v_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&mut $layer.self_attn.v_proj.lora_a),
                );
                v_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&mut $layer.self_attn.v_proj.lora_b),
                );
                attn_params.insert(Rc::from("v_proj"), NestedValue::Map(v_params));

                let mut o_params = HashMap::new();
                o_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&mut $layer.self_attn.o_proj.lora_a),
                );
                o_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&mut $layer.self_attn.o_proj.lora_b),
                );
                attn_params.insert(Rc::from("o_proj"), NestedValue::Map(o_params));

                layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

                let mut mlp_params = HashMap::new();
                let mut gate_params = HashMap::new();
                gate_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&mut $layer.mlp.gate_proj.lora_a),
                );
                gate_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&mut $layer.mlp.gate_proj.lora_b),
                );
                mlp_params.insert(Rc::from("gate_proj"), NestedValue::Map(gate_params));

                let mut up_params = HashMap::new();
                up_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&mut $layer.mlp.up_proj.lora_a),
                );
                up_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&mut $layer.mlp.up_proj.lora_b),
                );
                mlp_params.insert(Rc::from("up_proj"), NestedValue::Map(up_params));

                let mut down_params = HashMap::new();
                down_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&mut $layer.mlp.down_proj.lora_a),
                );
                down_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&mut $layer.mlp.down_proj.lora_b),
                );
                mlp_params.insert(Rc::from("down_proj"), NestedValue::Map(down_params));

                layer_params.insert(Rc::from("mlp"), NestedValue::Map(mlp_params));
                $params.insert(prefix, NestedValue::Map(layer_params));
            }};
        }

        match &mut self.model.layers {
            GemmaLoraLayers::Gemma1(layers) => {
                for (i, layer) in layers.iter_mut().enumerate() {
                    build_layer_params_mut!(layer, i, params);
                }
            }
            GemmaLoraLayers::Gemma2(layers) => {
                for (i, layer) in layers.iter_mut().enumerate() {
                    build_layer_params_mut!(layer, i, params);
                }
            }
        }

        params
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        self.parameters()
    }

    fn freeze_parameters(&mut self, _recursive: bool) {}
    fn unfreeze_parameters(&mut self, _recursive: bool) {}
    fn all_frozen(&self) -> Option<bool> {
        Some(false)
    }
    fn any_frozen(&self) -> Option<bool> {
        Some(false)
    }
}

impl crate::TrainableModel for GemmaLoraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        GemmaLoraForCausalLM::forward(self, input_ids, mask)
    }

    fn num_trainable_params(&self) -> usize {
        GemmaLoraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        GemmaLoraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        GemmaLoraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        GemmaLoraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        GemmaLoraForCausalLM::load_lora_weights(self, path)
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        GemmaLoraForCausalLM::enable_gradient_checkpointing(self, layers_per_block)
    }

    fn disable_gradient_checkpointing(&mut self) {
        GemmaLoraForCausalLM::disable_gradient_checkpointing(self)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        true
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        GemmaLoraForCausalLM::forward_with_cache(self, input_ids, mask, cache)
    }

    fn create_cache(&self, max_seq_len: usize) -> Option<KVCache> {
        Some(GemmaLoraForCausalLM::create_cache(self, max_seq_len))
    }

    fn supports_kv_cache(&self) -> bool {
        true
    }
}

fn create_causal_mask(seq_len: i32) -> Result<Array, Exception> {
    let mask = mlx_rs::ops::tri::<f32>(seq_len, None, None)?;
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let zero = Array::from_f32(0.0);
    mlx_rs::ops::r#where(&mask.eq(&zero)?, &neg_inf, &zero)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> GemmaConfig {
        GemmaConfig {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: Some(16),
            max_position_embeddings: 512,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            ..Default::default()
        }
    }

    fn small_lora_config() -> LoraConfig {
        LoraConfig {
            r: 8,
            alpha: 16.0,
            dropout: 0.0,
            use_rslora: false,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
            bias: pmetal_core::LoraBias::None,
            init_lora_weights: true,
            use_dora: false,
        }
    }

    #[test]
    fn test_gemma_lora_attention() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut attn = GemmaLoraAttention::new(&config, &lora_config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = attn.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_gemma_lora_model() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut model = GemmaLoraForCausalLM::new(config, lora_config).unwrap();

        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();

        assert_eq!(logits.shape(), &[1, 4, 1000]);
    }

    #[test]
    fn test_gemma_lora_param_count() {
        let config = small_config();
        let lora_config = small_lora_config();
        let model = GemmaLoraForCausalLM::new(config, lora_config).unwrap();

        assert!(model.num_trainable_params() > 0);
        let params = model.lora_parameters();
        assert!(!params.is_empty());
    }

    fn small_gemma2_config() -> GemmaConfig {
        GemmaConfig {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: Some(16),
            max_position_embeddings: 512,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            is_gemma2: true,
            attn_logit_softcapping: Some(50.0),
            sliding_window: Some(4096),
            ..Default::default()
        }
    }

    #[test]
    fn test_gemma2_lora_model() {
        let config = small_gemma2_config();
        let lora_config = small_lora_config();
        let mut model = GemmaLoraForCausalLM::new(config.clone(), lora_config).unwrap();

        // Verify Gemma2 layers were created
        match &model.model.layers {
            GemmaLoraLayers::Gemma2(layers) => {
                assert_eq!(layers.len(), 2);
            }
            _ => panic!("Expected Gemma2 layers"),
        }

        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();

        assert_eq!(logits.shape(), &[1, 4, 1000]);
    }

    #[test]
    fn test_gemma2_lora_extra_norms() {
        let config = small_gemma2_config();
        let lora_config = small_lora_config();
        let layer = Gemma2LoraDecoderLayer::new(&config, &lora_config).unwrap();

        // Verify Gemma2 has extra normalization layers
        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();

        // Check pre-feedforward norm works
        let normed = layer.pre_feedforward_layernorm.forward(&x).unwrap();
        assert_eq!(normed.shape(), &[1, 4, 64]);

        // Check post-feedforward norm works
        let normed = layer.post_feedforward_layernorm.forward(&x).unwrap();
        assert_eq!(normed.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_gemma2_lora_params() {
        let config = small_gemma2_config();
        let lora_config = small_lora_config();
        let model = GemmaLoraForCausalLM::new(config, lora_config).unwrap();

        let params = model.lora_parameters();
        assert!(!params.is_empty());

        // Verify param structure
        assert!(params.contains_key(&Rc::from("layers.0.self_attn.q_proj.lora_a")));
        assert!(params.contains_key(&Rc::from("layers.1.mlp.down_proj.lora_b")));
    }

    #[test]
    fn test_gemma2_lora_module_params() {
        let config = small_gemma2_config();
        let lora_config = small_lora_config();
        let model = GemmaLoraForCausalLM::new(config, lora_config).unwrap();

        // Test ModuleParameters implementation
        let params = model.parameters();
        assert!(!params.entries.is_empty());
    }
}
