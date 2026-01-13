//! LoRA-enabled Phi model architecture (Phi-3, Phi-3.5, Phi-4).
//!
//! Implements Phi with LoRA adapters for efficient fine-tuning.
//! Key differences from Llama:
//! - Partial RoPE (applied to subset of head dimensions)
//! - Fused gate_up_proj for SwiGLU
//! - QKV bias support (Phi-4)
//!
//! # Performance Optimizations (SOTA)
//!
//! This implementation uses several state-of-the-art optimizations:
//! - **Fast RMS Norm**: Uses `mlx_rs::fast::rms_norm()` for fused kernel execution
//! - **Compiled activations**: Uses optimized `nn::silu()` for SwiGLU

use std::collections::HashMap;
use std::rc::Rc;

use mlx_rs::{
    builder::Builder,
    error::Exception,
    fast,
    module::{Module, ModuleParamMut, ModuleParamRef, ModuleParameters, Param},
    nested::NestedValue,
    nn::{self, RopeBuilder},
    ops::indexing::IndexOp,
    Array,
};

use pmetal_core::LoraConfig;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_mlx::kernels::{fused_sdpa, rope::apply_rope, AttentionMaskType, FusedAttentionConfig};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};
use pmetal_models::architectures::phi::{PhiActivation, PhiConfig};

use crate::{LoraError, LoraLinear};

/// LoRA-enabled Phi RMS LayerNorm.
///
/// Uses `mlx_rs::fast::rms_norm()` for optimized fused kernel execution.
#[derive(Debug)]
pub struct PhiLoraRmsNorm {
    /// Weight parameter.
    pub weight: Param<Array>,
    /// Epsilon.
    pub eps: f32,
}

impl PhiLoraRmsNorm {
    /// Create a new RMS LayerNorm.
    pub fn new(hidden_size: i32, eps: f32) -> Self {
        let weight = Param::new(Array::ones::<f32>(&[hidden_size]).unwrap());
        Self { weight, eps }
    }

    /// Forward pass using optimized fast::rms_norm.
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        // Use the optimized fused RMS norm kernel
        fast::rms_norm(x, &*self.weight, self.eps)
    }
}

/// LoRA-enabled attention layer for Phi with partial RoPE.
#[derive(Debug)]
pub struct PhiLoraAttention {
    /// Query projection with LoRA.
    pub q_proj: LoraLinear,
    /// Key projection with LoRA.
    pub k_proj: LoraLinear,
    /// Value projection with LoRA.
    pub v_proj: LoraLinear,
    /// Output projection with LoRA.
    pub o_proj: LoraLinear,
    /// RoPE layer (for partial dimensions).
    pub rope: nn::Rope,
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads.
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// RoPE dimension (partial).
    pub rope_dim: i32,
    /// Attention scale.
    pub scale: f32,
}

impl PhiLoraAttention {
    /// Create a new LoRA attention layer.
    pub fn new(config: &PhiConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let head_dim = config.head_dim();
        let rope_dim = config.rope_dim();

        let rank = lora_config.r as i32;
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;

        let q_proj = LoraLinear::new(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            rank,
            alpha,
            use_rslora,
            config.qkv_bias,
        )?;
        let k_proj = LoraLinear::new(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            rank,
            alpha,
            use_rslora,
            config.qkv_bias,
        )?;
        let v_proj = LoraLinear::new(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            rank,
            alpha,
            use_rslora,
            config.qkv_bias,
        )?;
        let o_proj = LoraLinear::new(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            rank,
            alpha,
            use_rslora,
            false,
        )?;

        let rope = RopeBuilder::new(rope_dim)
            .traditional(false)
            .base(config.rope_theta)
            .scale(1.0)
            .build()
            .unwrap();

        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            n_heads: config.num_attention_heads,
            n_kv_heads: config.num_key_value_heads,
            head_dim,
            rope_dim,
            scale,
        })
    }

    /// Forward pass through attention.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let (batch, seq_len, _) = (x.dim(0), x.dim(1), x.dim(2));

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [batch, seq, n_heads, head_dim]
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;

        // Apply partial RoPE
        let (q_rope, q_pass) = self.split_rotary(&q)?;
        let (k_rope, k_pass) = self.split_rotary(&k)?;

        let q_rope = Module::forward(&mut self.rope, &q_rope)?;
        let k_rope = Module::forward(&mut self.rope, &k_rope)?;

        // Concatenate RoPE and pass-through parts
        let q = mlx_rs::ops::concatenate_axis(&[&q_rope, &q_pass], -1)?;
        let k = mlx_rs::ops::concatenate_axis(&[&k_rope, &k_pass], -1)?;

        // Transpose for attention: [batch, n_heads, seq, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Expand KV heads for GQA
        let k = if self.n_kv_heads < self.n_heads {
            let repeats = self.n_heads / self.n_kv_heads;
            expand_kv_heads(&k, repeats)?
        } else {
            k
        };
        let v = if self.n_kv_heads < self.n_heads {
            let repeats = self.n_heads / self.n_kv_heads;
            expand_kv_heads(&v, repeats)?
        } else {
            v
        };

        // Scaled dot-product attention
        let scores = q.matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?;
        let scores = scores.multiply(Array::from_f32(self.scale))?;

        let scores = if let Some(m) = mask {
            scores.add(m)?
        } else {
            scores
        };

        let weights = mlx_rs::ops::softmax_axis(&scores, -1, None)?;
        let output = weights.matmul(&v)?;

        // Transpose back and project
        let output = output.transpose_axes(&[0, 2, 1, 3])?;
        let output = output.reshape(&[batch, seq_len, self.n_heads * self.head_dim])?;

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

        // Apply partial RoPE with cache offset
        let (queries, keys, values) = if let Some((ref cache_ref, _layer_idx)) = cache {
            let offset = cache_ref.rope_offset();

            // Apply partial RoPE to query
            let (q_rope, q_pass) = self.split_rotary_transposed(&queries)?;
            let q_rope = apply_rope(&q_rope, self.rope_dim, false, self.rope.base, 1.0, offset)?;
            let queries = mlx_rs::ops::concatenate_axis(&[&q_rope, &q_pass], -1)?;

            // Apply partial RoPE to key
            let (k_rope, k_pass) = self.split_rotary_transposed(&keys)?;
            let k_rope = apply_rope(&k_rope, self.rope_dim, false, self.rope.base, 1.0, offset)?;
            let keys = mlx_rs::ops::concatenate_axis(&[&k_rope, &k_pass], -1)?;

            (queries, keys, values)
        } else {
            // No cache - use standard RoPE
            let (q_rope, q_pass) = self.split_rotary_transposed(&queries)?;
            let (k_rope, k_pass) = self.split_rotary_transposed(&keys)?;

            let q_rope = Module::forward(&mut self.rope, &q_rope)?;
            let k_rope = Module::forward(&mut self.rope, &k_rope)?;

            let queries = mlx_rs::ops::concatenate_axis(&[&q_rope, &q_pass], -1)?;
            let keys = mlx_rs::ops::concatenate_axis(&[&k_rope, &k_pass], -1)?;

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

        // Use fused attention kernel for inference (more efficient than standard attention)
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(AttentionMaskType::Causal);

        let output = fused_sdpa(&queries, &keys, &values, &attn_config, mask)
            .map_err(|e| LoraError::Mlx(e))?;

        // Reshape back: [B, heads, L, head_dim] -> [B, L, hidden]
        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, -1])?;

        // Output projection
        self.o_proj.forward(&output).map_err(LoraError::from)
    }

    /// Split tensor into RoPE and pass-through parts.
    fn split_rotary(&self, x: &Array) -> Result<(Array, Array), Exception> {
        let rope_part = x.index((.., .., .., ..self.rope_dim));
        let pass_part = x.index((.., .., .., self.rope_dim..));
        Ok((rope_part, pass_part))
    }

    /// Split tensor into RoPE and pass-through parts (for transposed layout).
    /// Input: [B, heads, seq, head_dim]
    fn split_rotary_transposed(&self, x: &Array) -> Result<(Array, Array), Exception> {
        let rope_part = x.index((.., .., .., ..self.rope_dim));
        let pass_part = x.index((.., .., .., self.rope_dim..));
        Ok((rope_part, pass_part))
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

/// LoRA-enabled MLP layer for Phi with fused gate_up.
#[derive(Debug)]
pub struct PhiLoraMLP {
    /// Fused gate_up projection with LoRA.
    pub gate_up_proj: LoraLinear,
    /// Down projection with LoRA.
    pub down_proj: LoraLinear,
    /// Activation type.
    pub activation: PhiActivation,
    /// Intermediate size.
    pub intermediate_size: i32,
}

impl PhiLoraMLP {
    /// Create a new LoRA MLP layer.
    pub fn new(config: &PhiConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let rank = lora_config.r as i32;
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;

        // For SwiGLU, gate_up_proj projects to 2x intermediate_size
        let proj_size = match config.hidden_act {
            PhiActivation::SwiGLU => config.intermediate_size * 2,
            _ => config.intermediate_size,
        };

        let gate_up_proj = LoraLinear::new(
            config.hidden_size,
            proj_size,
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
            gate_up_proj,
            down_proj,
            activation: config.hidden_act,
            intermediate_size: config.intermediate_size,
        })
    }

    /// Forward pass through MLP.
    ///
    /// Uses optimized compiled activations for better performance:
    /// - SwiGLU: Uses `nn::silu()` (compiled kernel)
    /// - GELU: Uses `nn::gelu()` (compiled kernel)
    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let hidden = self.gate_up_proj.forward(x)?;

        let activated = match self.activation {
            PhiActivation::SwiGLU => {
                // Split into gate and up projections
                let gate = hidden.index((.., .., ..self.intermediate_size));
                let up = hidden.index((.., .., self.intermediate_size..));
                // SwiGLU: silu(gate) * up - use compiled silu kernel
                let gate_activated = nn::silu(&gate)?;
                gate_activated.multiply(&up)?
            }
            PhiActivation::GeluApprox | PhiActivation::GeluExact => nn::gelu(&hidden)?,
        };

        self.down_proj.forward(&activated)
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.gate_up_proj.num_trainable_params() + self.down_proj.num_trainable_params()
    }
}

/// LoRA-enabled Phi decoder layer.
#[derive(Debug)]
pub struct PhiLoraDecoderLayer {
    /// Self-attention layer with LoRA.
    pub self_attn: PhiLoraAttention,
    /// MLP layer with LoRA.
    pub mlp: PhiLoraMLP,
    /// Input layer norm.
    pub input_layernorm: PhiLoraRmsNorm,
    /// Post-attention layer norm.
    pub post_attention_layernorm: PhiLoraRmsNorm,
}

impl PhiLoraDecoderLayer {
    /// Create a new decoder layer with LoRA.
    pub fn new(config: &PhiConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let self_attn = PhiLoraAttention::new(config, lora_config)?;
        let mlp = PhiLoraMLP::new(config, lora_config)?;

        let input_layernorm = PhiLoraRmsNorm::new(config.hidden_size, config.rms_norm_eps);
        let post_attention_layernorm = PhiLoraRmsNorm::new(config.hidden_size, config.rms_norm_eps);

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let residual = x.clone();
        let hidden = self.input_layernorm.forward(x)?;
        let hidden = self.self_attn.forward(&hidden, mask)?;
        let hidden = residual.add(&hidden)?;

        let residual = hidden.clone();
        let hidden = self.post_attention_layernorm.forward(&hidden)?;
        let hidden = self.mlp.forward(&hidden)?;
        Ok(residual.add(&hidden)?)
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

/// LoRA-enabled Phi model (without LM head).
#[derive(Debug)]
pub struct PhiLoraModel {
    /// Configuration.
    pub config: PhiConfig,
    /// LoRA configuration.
    pub lora_config: LoraConfig,
    /// Token embeddings (frozen).
    pub embed_tokens: nn::Embedding,
    /// Transformer layers with LoRA.
    pub layers: Vec<PhiLoraDecoderLayer>,
    /// Final layer norm (frozen).
    pub norm: PhiLoraRmsNorm,
}

impl PhiLoraModel {
    /// Create a new LoRA Phi model.
    pub fn new(config: PhiConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers)
            .map(|_| PhiLoraDecoderLayer::new(&config, &lora_config))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = PhiLoraRmsNorm::new(config.hidden_size, config.rms_norm_eps);

        Ok(Self {
            config,
            lora_config,
            embed_tokens,
            layers,
            norm,
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
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;

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

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, mask.as_ref())?;

            if checkpointing_enabled && (idx + 1) % layers_per_block == 0 {
                hidden_states.eval().map_err(LoraError::Mlx)?;
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
        // Get embeddings
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;

        // Don't create explicit causal mask - fused SDPA handles it internally
        // with proper dtype handling. Only pass through user-provided masks.

        // Pass through transformer layers
        match cache {
            Some(cache) => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    hidden_states =
                        layer.forward_with_cache(&hidden_states, mask, Some((cache, layer_idx)))?;
                }
            }
            None => {
                for layer in &mut self.layers {
                    hidden_states = layer.forward(&hidden_states, mask)?;
                }
            }
        }

        // Final norm
        Ok(self.norm.forward(&hidden_states)?)
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_trainable_params()).sum()
    }
}

/// LoRA-enabled Phi model with LM head.
#[derive(Debug)]
pub struct PhiLoraForCausalLM {
    /// Base model with LoRA.
    pub model: PhiLoraModel,
    /// LM head (frozen).
    pub lm_head: nn::Linear,
    /// Gradient checkpointing configuration.
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl PhiLoraForCausalLM {
    /// Create a new LoRA Phi model with LM head.
    pub fn new(config: PhiConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let lm_head = nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
            .bias(false)
            .build()
            .unwrap();
        let model = PhiLoraModel::new(config, lora_config)?;

        Ok(Self {
            model,
            lm_head,
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
        Ok(Module::forward(&mut self.lm_head, &hidden_states)?)
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
        Ok(Module::forward(&mut self.lm_head, &hidden_states)?)
    }

    /// Create a KV cache for this model.
    ///
    /// # Arguments
    /// * `max_seq_len` - Maximum sequence length to cache
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        let config = KVCacheConfig::new(
            self.model.config.num_hidden_layers as usize,
            max_seq_len,
            self.model.config.num_key_value_heads as usize,
            self.model.config.head_dim() as usize,
        );
        KVCache::new(config)
    }

    /// Get all trainable LoRA parameters.
    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        let mut params = HashMap::new();

        for (i, layer) in self.model.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);

            // Attention LoRA params
            params.insert(
                Rc::from(format!("{}.self_attn.q_proj.lora_a", prefix)),
                layer.self_attn.q_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.q_proj.lora_b", prefix)),
                layer.self_attn.q_proj.lora_b.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.k_proj.lora_a", prefix)),
                layer.self_attn.k_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.k_proj.lora_b", prefix)),
                layer.self_attn.k_proj.lora_b.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.v_proj.lora_a", prefix)),
                layer.self_attn.v_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.v_proj.lora_b", prefix)),
                layer.self_attn.v_proj.lora_b.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.o_proj.lora_a", prefix)),
                layer.self_attn.o_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.self_attn.o_proj.lora_b", prefix)),
                layer.self_attn.o_proj.lora_b.clone(),
            );

            // MLP LoRA params (fused gate_up)
            params.insert(
                Rc::from(format!("{}.mlp.gate_up_proj.lora_a", prefix)),
                layer.mlp.gate_up_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.mlp.gate_up_proj.lora_b", prefix)),
                layer.mlp.gate_up_proj.lora_b.clone(),
            );
            params.insert(
                Rc::from(format!("{}.mlp.down_proj.lora_a", prefix)),
                layer.mlp.down_proj.lora_a.clone(),
            );
            params.insert(
                Rc::from(format!("{}.mlp.down_proj.lora_b", prefix)),
                layer.mlp.down_proj.lora_b.clone(),
            );
        }

        params
    }

    /// Set LoRA parameters.
    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);

            macro_rules! set_param {
                ($param:expr, $key:expr) => {
                    if let Some(value) = params.get(&Rc::from($key)) {
                        $param = value.clone();
                    }
                };
            }

            set_param!(
                layer.self_attn.q_proj.lora_a,
                format!("{}.self_attn.q_proj.lora_a", prefix)
            );
            set_param!(
                layer.self_attn.q_proj.lora_b,
                format!("{}.self_attn.q_proj.lora_b", prefix)
            );
            set_param!(
                layer.self_attn.k_proj.lora_a,
                format!("{}.self_attn.k_proj.lora_a", prefix)
            );
            set_param!(
                layer.self_attn.k_proj.lora_b,
                format!("{}.self_attn.k_proj.lora_b", prefix)
            );
            set_param!(
                layer.self_attn.v_proj.lora_a,
                format!("{}.self_attn.v_proj.lora_a", prefix)
            );
            set_param!(
                layer.self_attn.v_proj.lora_b,
                format!("{}.self_attn.v_proj.lora_b", prefix)
            );
            set_param!(
                layer.self_attn.o_proj.lora_a,
                format!("{}.self_attn.o_proj.lora_a", prefix)
            );
            set_param!(
                layer.self_attn.o_proj.lora_b,
                format!("{}.self_attn.o_proj.lora_b", prefix)
            );

            set_param!(
                layer.mlp.gate_up_proj.lora_a,
                format!("{}.mlp.gate_up_proj.lora_a", prefix)
            );
            set_param!(
                layer.mlp.gate_up_proj.lora_b,
                format!("{}.mlp.gate_up_proj.lora_b", prefix)
            );
            set_param!(
                layer.mlp.down_proj.lora_a,
                format!("{}.mlp.down_proj.lora_a", prefix)
            );
            set_param!(
                layer.mlp.down_proj.lora_b,
                format!("{}.mlp.down_proj.lora_b", prefix)
            );
        }
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    /// Get configuration.
    pub fn config(&self) -> &PhiConfig {
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

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);

            macro_rules! load_param {
                ($param:expr, $key:expr) => {
                    if let Some(value) = loaded.get(&Rc::from($key) as &str) {
                        $param = value.clone();
                    }
                };
            }

            load_param!(
                layer.self_attn.q_proj.lora_a,
                format!("{}.self_attn.q_proj.lora_a", prefix)
            );
            load_param!(
                layer.self_attn.q_proj.lora_b,
                format!("{}.self_attn.q_proj.lora_b", prefix)
            );
            load_param!(
                layer.self_attn.k_proj.lora_a,
                format!("{}.self_attn.k_proj.lora_a", prefix)
            );
            load_param!(
                layer.self_attn.k_proj.lora_b,
                format!("{}.self_attn.k_proj.lora_b", prefix)
            );
            load_param!(
                layer.self_attn.v_proj.lora_a,
                format!("{}.self_attn.v_proj.lora_a", prefix)
            );
            load_param!(
                layer.self_attn.v_proj.lora_b,
                format!("{}.self_attn.v_proj.lora_b", prefix)
            );
            load_param!(
                layer.self_attn.o_proj.lora_a,
                format!("{}.self_attn.o_proj.lora_a", prefix)
            );
            load_param!(
                layer.self_attn.o_proj.lora_b,
                format!("{}.self_attn.o_proj.lora_b", prefix)
            );

            load_param!(
                layer.mlp.gate_up_proj.lora_a,
                format!("{}.mlp.gate_up_proj.lora_a", prefix)
            );
            load_param!(
                layer.mlp.gate_up_proj.lora_b,
                format!("{}.mlp.gate_up_proj.lora_b", prefix)
            );
            load_param!(
                layer.mlp.down_proj.lora_a,
                format!("{}.mlp.down_proj.lora_a", prefix)
            );
            load_param!(
                layer.mlp.down_proj.lora_b,
                format!("{}.mlp.down_proj.lora_b", prefix)
            );
        }

        Ok(())
    }

    /// Load base model weights.
    pub fn load_base_weights(&mut self, weights: &HashMap<String, Array>) -> Result<(), LoraError> {
        // Load embed_tokens
        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        // Load transformer layers
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", i);

            if let Some(w) = weights.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
                layer.self_attn.q_proj.weight = w.clone();
            }
            if let Some(w) = weights.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
                layer.self_attn.k_proj.weight = w.clone();
            }
            if let Some(w) = weights.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
                layer.self_attn.v_proj.weight = w.clone();
            }
            if let Some(w) = weights.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
                layer.self_attn.o_proj.weight = w.clone();
            }

            // Handle bias if present
            if let Some(b) = weights.get(&format!("{}.self_attn.q_proj.bias", prefix)) {
                layer.self_attn.q_proj.bias = Some(b.clone());
            }
            if let Some(b) = weights.get(&format!("{}.self_attn.k_proj.bias", prefix)) {
                layer.self_attn.k_proj.bias = Some(b.clone());
            }
            if let Some(b) = weights.get(&format!("{}.self_attn.v_proj.bias", prefix)) {
                layer.self_attn.v_proj.bias = Some(b.clone());
            }

            if let Some(w) = weights.get(&format!("{}.mlp.gate_up_proj.weight", prefix)) {
                layer.mlp.gate_up_proj.weight = w.clone();
            }
            if let Some(w) = weights.get(&format!("{}.mlp.down_proj.weight", prefix)) {
                layer.mlp.down_proj.weight = w.clone();
            }

            if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
                layer.input_layernorm.weight = Param::new(w.clone());
            }
            if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
                layer.post_attention_layernorm.weight = Param::new(w.clone());
            }
        }

        // Load final norm
        if let Some(w) = weights.get("model.norm.weight") {
            self.model.norm.weight = Param::new(w.clone());
        }

        // Load lm_head
        if let Some(w) = weights.get("lm_head.weight") {
            self.lm_head.weight = Param::new(w.clone());
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

        for layer in &self.model.layers {
            layer.self_attn.q_proj.weight.eval()?;
            layer.self_attn.k_proj.weight.eval()?;
            layer.self_attn.v_proj.weight.eval()?;
            layer.self_attn.o_proj.weight.eval()?;
            layer.mlp.gate_up_proj.weight.eval()?;
            layer.mlp.down_proj.weight.eval()?;

            layer.self_attn.q_proj.lora_a.eval()?;
            layer.self_attn.q_proj.lora_b.eval()?;
            layer.self_attn.k_proj.lora_a.eval()?;
            layer.self_attn.k_proj.lora_b.eval()?;
            layer.self_attn.v_proj.lora_a.eval()?;
            layer.self_attn.v_proj.lora_b.eval()?;
            layer.self_attn.o_proj.lora_a.eval()?;
            layer.self_attn.o_proj.lora_b.eval()?;
            layer.mlp.gate_up_proj.lora_a.eval()?;
            layer.mlp.gate_up_proj.lora_b.eval()?;
            layer.mlp.down_proj.lora_a.eval()?;
            layer.mlp.down_proj.lora_b.eval()?;

            layer.input_layernorm.weight.value.as_ref().eval()?;
            layer
                .post_attention_layernorm
                .weight
                .value
                .as_ref()
                .eval()?;
        }

        self.model.norm.weight.value.as_ref().eval()?;
        self.lm_head.weight.value.as_ref().eval()?;
        Ok(())
    }
}

impl ModuleParameters for PhiLoraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.model.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();

        for (i, layer) in self.model.layers.iter().enumerate() {
            let prefix: Rc<str> = Rc::from(format!("layers.{}", i));
            let mut layer_params = HashMap::new();

            let mut attn_params = HashMap::new();
            let mut q_params = HashMap::new();
            q_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.self_attn.q_proj.lora_a),
            );
            q_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.self_attn.q_proj.lora_b),
            );
            attn_params.insert(Rc::from("q_proj"), NestedValue::Map(q_params));

            let mut k_params = HashMap::new();
            k_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.self_attn.k_proj.lora_a),
            );
            k_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.self_attn.k_proj.lora_b),
            );
            attn_params.insert(Rc::from("k_proj"), NestedValue::Map(k_params));

            let mut v_params = HashMap::new();
            v_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.self_attn.v_proj.lora_a),
            );
            v_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.self_attn.v_proj.lora_b),
            );
            attn_params.insert(Rc::from("v_proj"), NestedValue::Map(v_params));

            let mut o_params = HashMap::new();
            o_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.self_attn.o_proj.lora_a),
            );
            o_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.self_attn.o_proj.lora_b),
            );
            attn_params.insert(Rc::from("o_proj"), NestedValue::Map(o_params));

            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            let mut mlp_params = HashMap::new();
            let mut gate_up_params = HashMap::new();
            gate_up_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.mlp.gate_up_proj.lora_a),
            );
            gate_up_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.mlp.gate_up_proj.lora_b),
            );
            mlp_params.insert(Rc::from("gate_up_proj"), NestedValue::Map(gate_up_params));

            let mut down_params = HashMap::new();
            down_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&layer.mlp.down_proj.lora_a),
            );
            down_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&layer.mlp.down_proj.lora_b),
            );
            mlp_params.insert(Rc::from("down_proj"), NestedValue::Map(down_params));

            layer_params.insert(Rc::from("mlp"), NestedValue::Map(mlp_params));
            params.insert(prefix, NestedValue::Map(layer_params));
        }

        params
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut params = ModuleParamMut::new();

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix: Rc<str> = Rc::from(format!("layers.{}", i));
            let mut layer_params = HashMap::new();

            let mut attn_params = HashMap::new();
            let mut q_params = HashMap::new();
            q_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.self_attn.q_proj.lora_a),
            );
            q_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.self_attn.q_proj.lora_b),
            );
            attn_params.insert(Rc::from("q_proj"), NestedValue::Map(q_params));

            let mut k_params = HashMap::new();
            k_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.self_attn.k_proj.lora_a),
            );
            k_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.self_attn.k_proj.lora_b),
            );
            attn_params.insert(Rc::from("k_proj"), NestedValue::Map(k_params));

            let mut v_params = HashMap::new();
            v_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.self_attn.v_proj.lora_a),
            );
            v_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.self_attn.v_proj.lora_b),
            );
            attn_params.insert(Rc::from("v_proj"), NestedValue::Map(v_params));

            let mut o_params = HashMap::new();
            o_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.self_attn.o_proj.lora_a),
            );
            o_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.self_attn.o_proj.lora_b),
            );
            attn_params.insert(Rc::from("o_proj"), NestedValue::Map(o_params));

            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            let mut mlp_params = HashMap::new();
            let mut gate_up_params = HashMap::new();
            gate_up_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.mlp.gate_up_proj.lora_a),
            );
            gate_up_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.mlp.gate_up_proj.lora_b),
            );
            mlp_params.insert(Rc::from("gate_up_proj"), NestedValue::Map(gate_up_params));

            let mut down_params = HashMap::new();
            down_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut layer.mlp.down_proj.lora_a),
            );
            down_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut layer.mlp.down_proj.lora_b),
            );
            mlp_params.insert(Rc::from("down_proj"), NestedValue::Map(down_params));

            layer_params.insert(Rc::from("mlp"), NestedValue::Map(mlp_params));
            params.insert(prefix, NestedValue::Map(layer_params));
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

impl crate::TrainableModel for PhiLoraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        PhiLoraForCausalLM::forward(self, input_ids, mask)
    }

    fn num_trainable_params(&self) -> usize {
        PhiLoraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        PhiLoraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        PhiLoraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        PhiLoraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        PhiLoraForCausalLM::load_lora_weights(self, path)
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        PhiLoraForCausalLM::enable_gradient_checkpointing(self, layers_per_block)
    }

    fn disable_gradient_checkpointing(&mut self) {
        PhiLoraForCausalLM::disable_gradient_checkpointing(self)
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
        PhiLoraForCausalLM::forward_with_cache(self, input_ids, mask, cache)
    }

    fn create_cache(&self, max_seq_len: usize) -> Option<KVCache> {
        Some(PhiLoraForCausalLM::create_cache(self, max_seq_len))
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

    fn small_config() -> PhiConfig {
        PhiConfig {
            model_type: "phi".to_string(),
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            max_position_embeddings: 512,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            rms_norm_eps: 1e-5,
            qkv_bias: false,
            hidden_act: PhiActivation::SwiGLU,
            sliding_window: None,
            layer_norm_type: pmetal_models::architectures::phi::LayerNormType::RmsNorm,
            original_max_position_embeddings: None,
            rope_scaling: None,
            tie_word_embeddings: true,
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
            bias: "none".to_string(),
            init_lora_weights: true,
            use_dora: false,
        }
    }

    #[test]
    fn test_phi_lora_attention() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut attn = PhiLoraAttention::new(&config, &lora_config).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 64], None, None, None).unwrap();
        let output = attn.forward(&x, None).unwrap();

        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_phi_lora_model() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut model = PhiLoraForCausalLM::new(config.clone(), lora_config).unwrap();

        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();

        assert_eq!(logits.shape(), &[1, 4, config.vocab_size]);
    }

    #[test]
    fn test_phi_lora_param_count() {
        let config = small_config();
        let lora_config = small_lora_config();
        let model = PhiLoraForCausalLM::new(config, lora_config).unwrap();

        assert!(model.num_trainable_params() > 0);
        let params = model.lora_parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_phi_partial_rope() {
        let config = small_config();
        assert_eq!(config.head_dim(), 16);
        assert_eq!(config.rope_dim(), 8); // 50% of head_dim
    }

    #[test]
    fn test_phi_kv_cache() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut model = PhiLoraForCausalLM::new(config, lora_config).unwrap();

        // Check that model supports KV cache
        use crate::TrainableModel;
        assert!(model.supports_kv_cache());

        // Create a cache
        let mut cache = model.create_cache(128);
        assert_eq!(cache.rope_offset(), 0);

        // Test forward pass with cache
        let input_ids = mlx_rs::Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let logits = model
            .forward_with_cache(&input_ids, None, Some(&mut cache))
            .unwrap();

        assert_eq!(logits.shape(), &[1, 4, 1000]);
        assert_eq!(cache.rope_offset(), 4);

        // Test incremental generation
        let next_token = mlx_rs::Array::from_slice(&[5_i32], &[1, 1]);
        let next_logits = model
            .forward_with_cache(&next_token, None, Some(&mut cache))
            .unwrap();

        assert_eq!(next_logits.shape(), &[1, 1, 1000]);
        assert_eq!(cache.rope_offset(), 5);
    }
}
