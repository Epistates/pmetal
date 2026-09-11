//! LoRA-enabled Granite model architecture.
//!
//! Implements Granite (dense decoder variant) with LoRA adapters on attention and MLP
//! projections for efficient fine-tuning. Hybrid (Mamba2) layers are left frozen when
//! present; only attention layers receive LoRA adapters.

use std::collections::HashMap;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, Exception, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, nn,
};

use pmetal_core::LoraConfig;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::{apply_rope, apply_rope_with_positions},
};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};
use pmetal_models::architectures::granite::{GraniteConfig, GraniteLayerType};
use pmetal_models::architectures::utils::create_causal_mask;

use crate::lora::LoraProjection;
use crate::lora_helpers::{
    LoraDecoderStack, collect_lora_parameters, count_trainable_params, load_lora_weights_impl,
    save_lora_weights_impl, set_lora_parameters as helpers_set_lora_parameters,
};
use crate::{LinearAdapter, LoraError, LoraLinear};

// =============================================================================
// Attention
// =============================================================================

/// LoRA-enabled attention layer for Granite.
///
/// Applies LoRA (or DoRA when `use_dora = true`) to q_proj, k_proj, v_proj, and o_proj.
#[derive(Debug)]
pub struct GraniteLoraAttention {
    /// Number of attention heads.
    pub n_heads: i32,
    /// Number of KV heads (GQA).
    pub n_kv_heads: i32,
    /// Head dimension.
    pub head_dim: i32,
    /// Attention scale factor.
    pub scale: f32,

    /// Query projection with LoRA or DoRA adapter.
    pub q_proj: LinearAdapter,
    /// Key projection with LoRA or DoRA adapter.
    pub k_proj: LinearAdapter,
    /// Value projection with LoRA or DoRA adapter.
    pub v_proj: LinearAdapter,
    /// Output projection with LoRA or DoRA adapter.
    pub o_proj: LinearAdapter,
    /// RoPE layer.
    pub rope: nn::Rope,
}

impl GraniteLoraAttention {
    pub fn new(config: &GraniteConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.resolved_head_dim();
        let hidden_size = config.hidden_size;
        let scale = config.attention_scale();

        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;
        let use_dora = lora_config.use_dora;

        let q_rank = crate::effective_rank(lora_config, "q_proj") as i32;
        let k_rank = crate::effective_rank(lora_config, "k_proj") as i32;
        let v_rank = crate::effective_rank(lora_config, "v_proj") as i32;
        let o_rank = crate::effective_rank(lora_config, "o_proj") as i32;

        let q_proj = LinearAdapter::new(
            hidden_size,
            n_heads * head_dim,
            q_rank,
            alpha,
            use_rslora,
            false,
            use_dora,
        )?;
        let k_proj = LinearAdapter::new(
            hidden_size,
            n_kv_heads * head_dim,
            k_rank,
            alpha,
            use_rslora,
            false,
            use_dora,
        )?;
        let v_proj = LinearAdapter::new(
            hidden_size,
            n_kv_heads * head_dim,
            v_rank,
            alpha,
            use_rslora,
            false,
            use_dora,
        )?;
        let o_proj = LinearAdapter::new(
            n_heads * head_dim,
            hidden_size,
            o_rank,
            alpha,
            use_rslora,
            false,
            use_dora,
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
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let keys = keys
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let values = values
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);

        let queries = pmetal_bridge::compat::Module::forward(&mut self.rope, &queries)?;
        let keys = pmetal_bridge::compat::Module::forward(&mut self.rope, &keys)?;

        let keys = if self.n_kv_heads < self.n_heads {
            expand_kv_heads(&keys, self.n_heads / self.n_kv_heads)?
        } else {
            keys
        };
        let values = if self.n_kv_heads < self.n_heads {
            expand_kv_heads(&values, self.n_heads / self.n_kv_heads)?
        } else {
            values
        };

        let scores = queries.matmul(&keys.transpose_axes(&[0, 1, 3, 2]));
        let scores = scores.multiply(&Array::from_f32(self.scale));
        let scores = if let Some(m) = mask {
            scores.add(m)
        } else {
            scores
        };

        let weights = pmetal_bridge::compat::ops::softmax_axis(&scores, -1);
        let output = weights.matmul(&values);
        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);

        self.o_proj.forward(&output).map_err(LoraError::from)
    }

    /// Forward pass with explicit position IDs for packed sequence training.
    pub fn forward_with_positions(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Result<Array, LoraError> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let queries = queries
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let keys = keys
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let values = values
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);

        let rope_dims = self.rope.dimensions;
        let rope_base = self.rope.base;
        let rope_scale = self.rope.scale;
        let rope_traditional = self.rope.traditional;
        let queries = apply_rope_with_positions(
            &queries,
            position_ids,
            rope_dims,
            rope_traditional,
            rope_base,
            rope_scale,
        )?;
        let keys = apply_rope_with_positions(
            &keys,
            position_ids,
            rope_dims,
            rope_traditional,
            rope_base,
            rope_scale,
        )?;

        let keys = if self.n_kv_heads < self.n_heads {
            expand_kv_heads(&keys, self.n_heads / self.n_kv_heads)?
        } else {
            keys
        };
        let values = if self.n_kv_heads < self.n_heads {
            expand_kv_heads(&values, self.n_heads / self.n_kv_heads)?
        } else {
            values
        };

        let scores = queries.matmul(&keys.transpose_axes(&[0, 1, 3, 2]));
        let scores = scores.multiply(&Array::from_f32(self.scale));
        let scores = if let Some(m) = mask {
            scores.add(m)
        } else {
            scores
        };

        let weights = pmetal_bridge::compat::ops::softmax_axis(&scores, -1);
        let output = weights.matmul(&values);
        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);

        self.o_proj.forward(&output).map_err(LoraError::from)
    }

    /// Forward pass with KV cache for efficient inference.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let queries = queries
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let keys = keys
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let values = values
            .reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);

        let (queries, keys, values) = if let Some((ref cache_ref, _layer_idx)) = cache {
            let offset = cache_ref.rope_offset();
            let queries = apply_rope(&queries, self.head_dim, false, self.rope.base, 1.0, offset)?;
            let keys = apply_rope(&keys, self.head_dim, false, self.rope.base, 1.0, offset)?;
            (queries, keys, values)
        } else {
            let queries = pmetal_bridge::compat::Module::forward(&mut self.rope, &queries)?;
            let keys = pmetal_bridge::compat::Module::forward(&mut self.rope, &keys)?;
            (queries, keys, values)
        };

        let (keys, values) = if let Some((cache, layer_idx)) = cache {
            cache
                .update_and_fetch(layer_idx, &keys, &values)
                .map_err(LoraError::Mlx)?
        } else {
            (keys, values)
        };

        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(AttentionMaskType::Causal);

        let output =
            fused_sdpa(&queries, &keys, &values, &attn_config, mask).map_err(LoraError::Mlx)?;

        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);

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

// =============================================================================
// Helper
// =============================================================================

fn expand_kv_heads(x: &Array, repeats: i32) -> Result<Array, Exception> {
    let shape = x.shape();
    let batch = shape[0];
    let n_kv_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    let x = x.reshape(&[batch, n_kv_heads, 1, seq_len, head_dim]);
    let x = pmetal_bridge::compat::ops::broadcast_to(
        &x,
        &[batch, n_kv_heads, repeats, seq_len, head_dim],
    );
    Ok(x.reshape(&[batch, n_kv_heads * repeats, seq_len, head_dim]))
}

// =============================================================================
// MLP
// =============================================================================

/// LoRA-enabled SwiGLU MLP for Granite.
#[derive(Debug)]
pub struct GraniteLoraMLP {
    /// Gate projection with LoRA.
    pub gate_proj: LoraLinear,
    /// Up projection with LoRA.
    pub up_proj: LoraLinear,
    /// Down projection with LoRA.
    pub down_proj: LoraLinear,
}

impl GraniteLoraMLP {
    pub fn new(config: &GraniteConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;

        let gate_rank = crate::effective_rank(lora_config, "gate_proj") as i32;
        let up_rank = crate::effective_rank(lora_config, "up_proj") as i32;
        let down_rank = crate::effective_rank(lora_config, "down_proj") as i32;

        let gate_proj = LoraLinear::new(
            config.hidden_size,
            config.intermediate_size,
            gate_rank,
            alpha,
            use_rslora,
            false,
        )?;
        let up_proj = LoraLinear::new(
            config.hidden_size,
            config.intermediate_size,
            up_rank,
            alpha,
            use_rslora,
            false,
        )?;
        let down_proj = LoraLinear::new(
            config.intermediate_size,
            config.hidden_size,
            down_rank,
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

    /// Forward pass (SwiGLU activation).
    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let gate = self.gate_proj.forward(x)?;
        let gate = nn::silu(&gate);
        let up = self.up_proj.forward(x)?;
        let hidden = gate.multiply(&up);
        self.down_proj.forward(&hidden)
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.gate_proj.num_trainable_params()
            + self.up_proj.num_trainable_params()
            + self.down_proj.num_trainable_params()
    }
}

// =============================================================================
// Decoder layer
// =============================================================================

/// LoRA-enabled Granite decoder layer.
///
/// For hybrid models, Mamba2 layers have no LoRA adapters and are forwarded frozen.
/// For dense models all layers are attention and all receive LoRA.
#[derive(Debug)]
pub struct GraniteLoraDecoderLayer {
    /// Whether this layer is an attention layer (vs Mamba2 for hybrid models).
    pub is_attention: bool,
    /// `config.residual_multiplier`, applied to both residual branches.
    ///
    /// Must match the base model's: an adapter trained against unscaled
    /// residuals is being fit to different math than inference will run.
    pub residual_multiplier: f32,
    /// Self-attention with LoRA — `Some` for attention layers, `None` for Mamba2.
    pub self_attn: Option<GraniteLoraAttention>,
    /// Frozen Mamba2 passthrough — present only for hybrid Mamba2 layers.
    pub mamba: Option<pmetal_models::architectures::granite::GraniteMamba2>,
    /// MLP with LoRA.
    pub mlp: GraniteLoraMLP,
    /// Input layer norm (frozen).
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layer norm (frozen).
    pub post_attention_layernorm: nn::RmsNorm,
}

impl GraniteLoraDecoderLayer {
    pub fn new(
        config: &GraniteConfig,
        lora_config: &LoraConfig,
        layer_idx: usize,
    ) -> Result<Self, LoraError> {
        let layer_type = config.layer_type(layer_idx);
        let is_attention = layer_type == GraniteLayerType::Attention;

        let self_attn = if is_attention {
            Some(GraniteLoraAttention::new(config, lora_config)?)
        } else {
            None
        };

        let mamba = if !is_attention {
            Some(
                pmetal_models::architectures::granite::GraniteMamba2::new(config)
                    .map_err(LoraError::Mlx)?,
            )
        } else {
            None
        };

        let mlp = GraniteLoraMLP::new(config, lora_config)?;

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .unwrap();
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .unwrap();

        Ok(Self {
            is_attention,
            residual_multiplier: config.residual_multiplier,
            self_attn,
            mamba,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let normed = pmetal_bridge::compat::Module::forward(&mut self.input_layernorm, x)?;

        let mixer_out = if self.is_attention {
            self.self_attn.as_mut().unwrap().forward(&normed, mask)?
        } else {
            self.mamba
                .as_mut()
                .unwrap()
                .forward(&normed)
                .map_err(LoraError::Mlx)?
        };

        let m = self.residual_multiplier;
        let h = x.add(&mixer_out.mul_scalar(m));

        let normed =
            pmetal_bridge::compat::Module::forward(&mut self.post_attention_layernorm, &h)?;
        let mlp_out = self.mlp.forward(&normed)?;

        Ok(h.add(&mlp_out.mul_scalar(m)))
    }

    /// Forward pass with KV cache (attention layers use cache; Mamba2 layers ignore it).
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let normed = pmetal_bridge::compat::Module::forward(&mut self.input_layernorm, x)?;

        let mixer_out = if self.is_attention {
            self.self_attn
                .as_mut()
                .unwrap()
                .forward_with_cache(&normed, mask, cache)?
        } else {
            self.mamba
                .as_mut()
                .unwrap()
                .forward(&normed)
                .map_err(LoraError::Mlx)?
        };

        let m = self.residual_multiplier;
        let h = x.add(&mixer_out.mul_scalar(m));

        let normed =
            pmetal_bridge::compat::Module::forward(&mut self.post_attention_layernorm, &h)?;
        let mlp_out = self.mlp.forward(&normed)?;

        Ok(h.add(&mlp_out.mul_scalar(m)))
    }

    /// Forward pass with explicit position IDs for packed sequence training.
    pub fn forward_with_positions(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Result<Array, LoraError> {
        let normed = pmetal_bridge::compat::Module::forward(&mut self.input_layernorm, x)?;

        let mixer_out = if self.is_attention {
            self.self_attn
                .as_mut()
                .unwrap()
                .forward_with_positions(&normed, mask, position_ids)?
        } else {
            self.mamba
                .as_mut()
                .unwrap()
                .forward(&normed)
                .map_err(LoraError::Mlx)?
        };

        let m = self.residual_multiplier;
        let h = x.add(&mixer_out.mul_scalar(m));

        let normed =
            pmetal_bridge::compat::Module::forward(&mut self.post_attention_layernorm, &h)?;
        let mlp_out = self.mlp.forward(&normed)?;

        Ok(h.add(&mlp_out.mul_scalar(m)))
    }

    /// Number of trainable parameters in this layer (LoRA params only).
    pub fn num_trainable_params(&self) -> usize {
        let attn = self
            .self_attn
            .as_ref()
            .map(|a| a.num_trainable_params())
            .unwrap_or(0);
        attn + self.mlp.num_trainable_params()
    }
}

// =============================================================================
// Model
// =============================================================================

/// LoRA-enabled Granite model (without LM head).
#[derive(Debug)]
pub struct GraniteLoraModel {
    /// Model configuration.
    pub config: GraniteConfig,
    /// LoRA configuration.
    pub lora_config: LoraConfig,
    /// Token embeddings (frozen).
    pub embed_tokens: nn::Embedding,
    /// Transformer layers with LoRA on attention layers.
    pub layers: Vec<GraniteLoraDecoderLayer>,
    /// Final RMSNorm (frozen).
    pub norm: nn::RmsNorm,
}

impl GraniteLoraModel {
    pub fn new(config: GraniteConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| GraniteLoraDecoderLayer::new(&config, &lora_config, i))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .unwrap();

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

    /// NEFTune forward: embed tokens, add uniform noise, then run transformer layers.
    pub fn forward_noised(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        noise_alpha: f32,
        checkpoint_config: Option<&CheckpointConfig>,
    ) -> Result<Array, LoraError> {
        let mut hidden_states =
            pmetal_bridge::compat::Module::forward(&mut self.embed_tokens, input_ids)?
                .mul_scalar(self.config.embedding_multiplier);

        let seq_len = input_ids.dim(1) as f32;
        let embed_dim = hidden_states.dim(2) as f32;
        let mag = noise_alpha / (seq_len * embed_dim).sqrt();

        let noise = pmetal_bridge::compat::random::uniform_range(
            -mag,
            mag,
            hidden_states.shape(),
            pmetal_bridge::compat::Dtype::Float32,
        );
        hidden_states = hidden_states.add(&noise);

        let mask = if mask.is_none() {
            let seq_len_i = input_ids.dim(1);
            Some(create_causal_mask(seq_len_i)?)
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
                tracing::trace!("NEFTune checkpoint boundary at layer {}", idx + 1);
            }
        }

        Ok(pmetal_bridge::compat::Module::forward(
            &mut self.norm,
            &hidden_states,
        )?)
    }

    /// Forward pass with optional gradient checkpointing.
    pub fn forward_with_checkpoint(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        checkpoint_config: Option<&CheckpointConfig>,
    ) -> Result<Array, LoraError> {
        let mut hidden_states =
            pmetal_bridge::compat::Module::forward(&mut self.embed_tokens, input_ids)?
                .mul_scalar(self.config.embedding_multiplier);

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
                tracing::trace!("Checkpoint boundary at layer {}", idx + 1);
            }
        }

        Ok(pmetal_bridge::compat::Module::forward(
            &mut self.norm,
            &hidden_states,
        )?)
    }

    /// Forward pass with KV cache for efficient inference.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let mut hidden_states =
            pmetal_bridge::compat::Module::forward(&mut self.embed_tokens, input_ids)?
                .mul_scalar(self.config.embedding_multiplier);

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

        Ok(pmetal_bridge::compat::Module::forward(
            &mut self.norm,
            &hidden_states,
        )?)
    }

    /// Forward pass with explicit position IDs for packed sequence training.
    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Result<Array, LoraError> {
        let mut hidden_states =
            pmetal_bridge::compat::Module::forward(&mut self.embed_tokens, input_ids)?
                .mul_scalar(self.config.embedding_multiplier);

        for layer in &mut self.layers {
            hidden_states = layer.forward_with_positions(&hidden_states, mask, position_ids)?;
        }

        Ok(pmetal_bridge::compat::Module::forward(
            &mut self.norm,
            &hidden_states,
        )?)
    }

    /// Number of trainable (LoRA) parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_trainable_params()).sum()
    }
}

impl LoraDecoderStack for GraniteLoraModel {
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn attn_projections(&self, layer: usize) -> Vec<&dyn LoraProjection> {
        let l = &self.layers[layer];
        if let Some(ref attn) = l.self_attn {
            vec![&attn.q_proj, &attn.k_proj, &attn.v_proj, &attn.o_proj]
        } else {
            vec![]
        }
    }

    fn attn_projections_mut(&mut self, layer: usize) -> Vec<&mut dyn LoraProjection> {
        let l = &mut self.layers[layer];
        if let Some(ref mut attn) = l.self_attn {
            vec![
                &mut attn.q_proj,
                &mut attn.k_proj,
                &mut attn.v_proj,
                &mut attn.o_proj,
            ]
        } else {
            vec![]
        }
    }

    fn mlp_projections(&self, layer: usize) -> Vec<&dyn LoraProjection> {
        let l = &self.layers[layer];
        vec![&l.mlp.gate_proj, &l.mlp.up_proj, &l.mlp.down_proj]
    }

    fn mlp_projections_mut(&mut self, layer: usize) -> Vec<&mut dyn LoraProjection> {
        let l = &mut self.layers[layer];
        vec![
            &mut l.mlp.gate_proj,
            &mut l.mlp.up_proj,
            &mut l.mlp.down_proj,
        ]
    }
}

// =============================================================================
// ForCausalLM
// =============================================================================

/// LoRA-enabled Granite model with LM head.
#[derive(Debug)]
pub struct GraniteLoraForCausalLM {
    /// Base model with LoRA.
    pub model: GraniteLoraModel,
    /// LM head (frozen, optional for tied weights).
    pub lm_head: Option<nn::Linear>,
    /// Gradient checkpointing configuration.
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl GraniteLoraForCausalLM {
    /// Project hidden states to logits, applying `logits_scaling`.
    ///
    /// Three forward variants (plain, checkpointed, cached) shared a copy of
    /// this block; they now share the divide too, so a Granite adapter is
    /// trained against the same logit scale inference produces.
    fn project_logits(&mut self, hidden_states: &Array) -> Result<Array, LoraError> {
        let logits = if let Some(ref mut lm_head) = self.lm_head {
            pmetal_bridge::compat::Module::forward(lm_head, hidden_states)?
        } else {
            self.model.embed_tokens.as_linear(hidden_states)
        };
        Ok(logits.div_scalar(self.model.config.logits_scaling))
    }

    /// Create a new LoRA Granite model with LM head.
    pub fn new(config: GraniteConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let tie_weights = config.tie_word_embeddings;
        let model = GraniteLoraModel::new(config.clone(), lora_config)?;

        let lm_head = if !tie_weights {
            let head = nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                .bias(false)
                .build()
                .unwrap();
            Some(head)
        } else {
            None
        };

        Ok(Self {
            model,
            lm_head,
            checkpoint_config: None,
        })
    }

    /// Enable gradient checkpointing for memory-efficient training.
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

        self.project_logits(&hidden_states)
    }

    /// Forward pass returning hidden states before lm_head, for Cut Cross-Entropy.
    pub fn forward_hidden_states(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
        let checkpoint_config = self.checkpoint_config.clone();
        self.model
            .forward_with_checkpoint(input_ids, mask, checkpoint_config.as_ref())
    }

    /// Forward pass returning hidden states with explicit position IDs, for CCE + packed training.
    pub fn forward_hidden_states_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Result<Array, LoraError> {
        self.model
            .forward_with_positions(input_ids, mask, position_ids)
    }

    /// Get the LM head weight for Cut Cross-Entropy.
    pub fn get_lm_head_weight(&self) -> Option<Array> {
        if let Some(ref lm_head) = self.lm_head {
            Some(lm_head.weight.value.clone())
        } else {
            Some(self.model.embed_tokens.weight.value.clone())
        }
    }

    /// NEFTune forward with uniform embedding noise.
    pub fn forward_noised(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        noise_alpha: f32,
    ) -> Result<Array, LoraError> {
        let checkpoint_config = self.checkpoint_config.clone();
        let hidden_states =
            self.model
                .forward_noised(input_ids, mask, noise_alpha, checkpoint_config.as_ref())?;

        self.project_logits(&hidden_states)
    }

    /// Forward pass with KV cache for efficient inference.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let hidden_states = self.model.forward_with_cache(input_ids, mask, cache)?;

        self.project_logits(&hidden_states)
    }

    /// Create a KV cache for this model.
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        let config = KVCacheConfig::new(
            self.model.config.num_hidden_layers as usize,
            max_seq_len,
            self.model.config.num_key_value_heads as usize,
            self.model.config.resolved_head_dim() as usize,
        );
        KVCache::new(config)
    }

    /// Get all trainable LoRA parameters as a flat HashMap.
    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        collect_lora_parameters(&self.model)
    }

    /// Apply gradient updates to LoRA parameters (SGD step).
    pub fn apply_gradients(
        &mut self,
        gradients: &HashMap<Rc<str>, Array>,
        learning_rate: f32,
    ) -> Result<(), LoraError> {
        let lr = Array::from_f32(learning_rate);

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);

            // Attention adapter params (only present for attention layers).
            if let Some(ref mut attn) = layer.self_attn {
                for (proj_name, proj) in [
                    ("q_proj", &mut attn.q_proj),
                    ("k_proj", &mut attn.k_proj),
                    ("v_proj", &mut attn.v_proj),
                    ("o_proj", &mut attn.o_proj),
                ] {
                    let a_key = format!("{}.self_attn.{}.lora_a", prefix, proj_name);
                    let b_key = format!("{}.self_attn.{}.lora_b", prefix, proj_name);
                    if let Some(grad) = gradients.get(&Rc::from(a_key)) {
                        *proj.lora_a_mut() = proj.lora_a().subtract(&grad.multiply(&lr));
                    }
                    if let Some(grad) = gradients.get(&Rc::from(b_key)) {
                        *proj.lora_b_mut() = proj.lora_b().subtract(&grad.multiply(&lr));
                    }
                }
            }

            // MLP LoRA params.
            for (proj_name, proj) in [
                ("gate_proj", &mut layer.mlp.gate_proj),
                ("up_proj", &mut layer.mlp.up_proj),
                ("down_proj", &mut layer.mlp.down_proj),
            ] {
                let a_key = format!("{}.mlp.{}.lora_a", prefix, proj_name);
                let b_key = format!("{}.mlp.{}.lora_b", prefix, proj_name);
                if let Some(grad) = gradients.get(&Rc::from(a_key)) {
                    *proj.lora_a_mut() = proj.lora_a().subtract(&grad.multiply(&lr));
                }
                if let Some(grad) = gradients.get(&Rc::from(b_key)) {
                    *proj.lora_b_mut() = proj.lora_b().subtract(&grad.multiply(&lr));
                }
            }
        }

        Ok(())
    }

    /// Set LoRA parameters from a HashMap (used by autodiff).
    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        helpers_set_lora_parameters(&mut self.model, params);
    }

    /// Evaluate all LoRA parameters (force computation).
    pub fn eval_lora_params(&mut self) -> Result<(), LoraError> {
        for layer in &mut self.model.layers {
            if let Some(ref mut attn) = layer.self_attn {
                attn.q_proj.lora_a_mut().eval();
                attn.q_proj.lora_b_mut().eval();
                attn.k_proj.lora_a_mut().eval();
                attn.k_proj.lora_b_mut().eval();
                attn.v_proj.lora_a_mut().eval();
                attn.v_proj.lora_b_mut().eval();
                attn.o_proj.lora_a_mut().eval();
                attn.o_proj.lora_b_mut().eval();
            }
            layer.mlp.gate_proj.lora_a_mut().eval();
            layer.mlp.gate_proj.lora_b_mut().eval();
            layer.mlp.up_proj.lora_a_mut().eval();
            layer.mlp.up_proj.lora_b_mut().eval();
            layer.mlp.down_proj.lora_a_mut().eval();
            layer.mlp.down_proj.lora_b_mut().eval();
        }
        Ok(())
    }

    /// Number of trainable (LoRA) parameters.
    pub fn num_trainable_params(&self) -> usize {
        count_trainable_params(&self.model)
    }

    /// Model configuration.
    pub fn config(&self) -> &GraniteConfig {
        &self.model.config
    }

    /// LoRA configuration.
    pub fn lora_config(&self) -> &LoraConfig {
        &self.model.lora_config
    }

    /// Merge LoRA adapter weights into base weights.
    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        for layer in &mut self.model.layers {
            if let Some(ref mut attn) = layer.self_attn {
                attn.q_proj.merge()?;
                attn.k_proj.merge()?;
                attn.v_proj.merge()?;
                attn.o_proj.merge()?;
            }
            layer.mlp.gate_proj.merge()?;
            layer.mlp.up_proj.merge()?;
            layer.mlp.down_proj.merge()?;
        }
        Ok(())
    }

    /// Unmerge is not supported — reload base weights to undo a merge.
    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        Err(LoraError::InvalidState(
            "unmerge_lora is not supported: reload base model weights to undo a merge".to_string(),
        ))
    }

    /// Save LoRA weights to a safetensors file.
    pub fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        save_lora_weights_impl(&self.model, path)
    }

    /// Load LoRA weights from a safetensors file or directory.
    pub fn load_lora_weights(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), LoraError> {
        load_lora_weights_impl(&mut self.model, path)
    }

    /// Load base model weights from a HashMap of weight tensors.
    ///
    /// Expected weight name format (HuggingFace convention):
    /// - `model.embed_tokens.weight`
    /// - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
    /// - `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
    /// - `model.layers.{i}.input_layernorm.weight`
    /// - `model.layers.{i}.post_attention_layernorm.weight`
    /// - `model.norm.weight`
    /// - `lm_head.weight` (if not tied)
    pub fn load_base_weights(
        &mut self,
        weights: &std::collections::HashMap<String, Array>,
    ) -> Result<(), LoraError> {
        use pmetal_bridge::compat::Param;

        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", i);

            if let Some(ref mut attn) = layer.self_attn {
                if let Some(w) = weights.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
                    *attn.q_proj.weight_mut() = w.clone();
                }
                if let Some(w) = weights.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
                    *attn.k_proj.weight_mut() = w.clone();
                }
                if let Some(w) = weights.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
                    *attn.v_proj.weight_mut() = w.clone();
                }
                if let Some(w) = weights.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
                    *attn.o_proj.weight_mut() = w.clone();
                }
            }

            if let Some(w) = weights.get(&format!("{}.mlp.gate_proj.weight", prefix)) {
                *layer.mlp.gate_proj.weight_mut() = w.clone();
            }
            if let Some(w) = weights.get(&format!("{}.mlp.up_proj.weight", prefix)) {
                *layer.mlp.up_proj.weight_mut() = w.clone();
            }
            if let Some(w) = weights.get(&format!("{}.mlp.down_proj.weight", prefix)) {
                *layer.mlp.down_proj.weight_mut() = w.clone();
            }

            if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
                layer.input_layernorm.weight = Param::new(w.clone());
            }
            if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
                layer.post_attention_layernorm.weight = Param::new(w.clone());
            }
        }

        if let Some(w) = weights.get("model.norm.weight") {
            self.model.norm.weight = Param::new(w.clone());
        }

        if let Some(ref mut lm_head) = self.lm_head {
            if let Some(w) = weights.get("lm_head.weight") {
                lm_head.weight = Param::new(w.clone());
            }
        }

        Ok(())
    }

    /// Load base model weights from safetensor files in a directory.
    pub fn load_base_weights_from_dir(
        &mut self,
        model_dir: impl AsRef<std::path::Path>,
    ) -> Result<(), LoraError> {
        let model_dir = model_dir.as_ref();

        let single_file = model_dir.join("model.safetensors");
        if single_file.exists() {
            let weights =
                crate::sanitize_loaded_weights(crate::load_safetensors_map(&single_file)?)?;
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
            weight_map: std::collections::HashMap<String, String>,
        }

        let index: WeightIndex = serde_json::from_str(&index_content)
            .map_err(|e| LoraError::Mlx(Exception::custom(e.to_string())))?;

        let shard_files: std::collections::HashSet<&String> = index.weight_map.values().collect();

        let mut all_weights = std::collections::HashMap::new();
        for shard_file in shard_files {
            let shard_path = model_dir.join(shard_file);
            let shard_weights =
                crate::sanitize_loaded_weights(crate::load_safetensors_map(&shard_path)?)?;
            all_weights.extend(shard_weights);
        }

        self.load_base_weights(&all_weights)
    }

    /// Evaluate all model parameters (force computation).
    pub fn eval_all(&mut self) -> Result<(), LoraError> {
        self.model.embed_tokens.weight.value.eval();

        for layer in &mut self.model.layers {
            if let Some(ref mut attn) = layer.self_attn {
                attn.q_proj.weight_mut().eval();
                attn.k_proj.weight_mut().eval();
                attn.v_proj.weight_mut().eval();
                attn.o_proj.weight_mut().eval();
                attn.q_proj.lora_a_mut().eval();
                attn.q_proj.lora_b_mut().eval();
                attn.k_proj.lora_a_mut().eval();
                attn.k_proj.lora_b_mut().eval();
                attn.v_proj.lora_a_mut().eval();
                attn.v_proj.lora_b_mut().eval();
                attn.o_proj.lora_a_mut().eval();
                attn.o_proj.lora_b_mut().eval();
            }
            layer.mlp.gate_proj.weight_mut().eval();
            layer.mlp.up_proj.weight_mut().eval();
            layer.mlp.down_proj.weight_mut().eval();
            layer.mlp.gate_proj.lora_a_mut().eval();
            layer.mlp.gate_proj.lora_b_mut().eval();
            layer.mlp.up_proj.lora_a_mut().eval();
            layer.mlp.up_proj.lora_b_mut().eval();
            layer.mlp.down_proj.lora_a_mut().eval();
            layer.mlp.down_proj.lora_b_mut().eval();

            layer.input_layernorm.weight.value.eval();
            layer.post_attention_layernorm.weight.value.eval();
        }

        self.model.norm.weight.value.eval();

        if let Some(ref mut lm_head) = self.lm_head {
            lm_head.weight.value.eval();
        }

        Ok(())
    }
}

// =============================================================================
// ModuleParameters
// =============================================================================

impl ModuleParameters for GraniteLoraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.model.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();

        for (i, layer) in self.model.layers.iter().enumerate() {
            let prefix: Rc<str> = Rc::from(format!("layers.{}", i));
            let mut layer_params = HashMap::new();

            // Attention LoRA params (only for attention layers).
            if let Some(ref attn) = layer.self_attn {
                let mut attn_params = HashMap::new();

                for (name, proj) in [
                    ("q_proj", &attn.q_proj as &dyn LoraProjection),
                    ("k_proj", &attn.k_proj),
                    ("v_proj", &attn.v_proj),
                    ("o_proj", &attn.o_proj),
                ] {
                    let mut p = HashMap::new();
                    p.insert(Rc::from("lora_a"), NestedValue::Value(proj.lora_a()));
                    p.insert(Rc::from("lora_b"), NestedValue::Value(proj.lora_b()));
                    // Include magnitude for DoRA.
                    for (extra_name, extra_val) in proj.extra_params() {
                        p.insert(Rc::from(extra_name), NestedValue::Value(extra_val));
                    }
                    attn_params.insert(Rc::from(name), NestedValue::Map(p));
                }

                layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));
            }

            // MLP LoRA params.
            let mut mlp_params = HashMap::new();
            for (name, proj) in [
                ("gate_proj", &layer.mlp.gate_proj as &dyn LoraProjection),
                ("up_proj", &layer.mlp.up_proj),
                ("down_proj", &layer.mlp.down_proj),
            ] {
                let mut p = HashMap::new();
                p.insert(Rc::from("lora_a"), NestedValue::Value(proj.lora_a()));
                p.insert(Rc::from("lora_b"), NestedValue::Value(proj.lora_b()));
                mlp_params.insert(Rc::from(name), NestedValue::Map(p));
            }
            layer_params.insert(Rc::from("mlp"), NestedValue::Map(mlp_params));

            params.insert(prefix, NestedValue::Map(layer_params));
        }

        params
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut params = ModuleParamMut::new();

        fn adapter_params_mut<'a>(
            adapter: &'a mut LinearAdapter,
        ) -> HashMap<Rc<str>, NestedValue<&'a mut Array>> {
            let mut m: HashMap<Rc<str>, NestedValue<&'a mut Array>> = HashMap::new();
            match adapter {
                LinearAdapter::Lora(l) => {
                    m.insert(Rc::from("lora_a"), NestedValue::Value(&mut l.lora_a));
                    m.insert(Rc::from("lora_b"), NestedValue::Value(&mut l.lora_b));
                }
                LinearAdapter::Dora(d) => {
                    m.insert(Rc::from("lora_a"), NestedValue::Value(&mut d.lora_a));
                    m.insert(Rc::from("lora_b"), NestedValue::Value(&mut d.lora_b));
                    m.insert(Rc::from("magnitude"), NestedValue::Value(&mut d.magnitude));
                }
            }
            m
        }

        fn lora_params_mut<'a>(
            l: &'a mut LoraLinear,
        ) -> HashMap<Rc<str>, NestedValue<&'a mut Array>> {
            let mut m: HashMap<Rc<str>, NestedValue<&'a mut Array>> = HashMap::new();
            m.insert(Rc::from("lora_a"), NestedValue::Value(&mut l.lora_a));
            m.insert(Rc::from("lora_b"), NestedValue::Value(&mut l.lora_b));
            m
        }

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix: Rc<str> = Rc::from(format!("layers.{}", i));
            let mut layer_params = HashMap::new();

            if let Some(ref mut attn) = layer.self_attn {
                let mut attn_params = HashMap::new();
                attn_params.insert(
                    Rc::from("q_proj"),
                    NestedValue::Map(adapter_params_mut(&mut attn.q_proj)),
                );
                attn_params.insert(
                    Rc::from("k_proj"),
                    NestedValue::Map(adapter_params_mut(&mut attn.k_proj)),
                );
                attn_params.insert(
                    Rc::from("v_proj"),
                    NestedValue::Map(adapter_params_mut(&mut attn.v_proj)),
                );
                attn_params.insert(
                    Rc::from("o_proj"),
                    NestedValue::Map(adapter_params_mut(&mut attn.o_proj)),
                );
                layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));
            }

            let mut mlp_params = HashMap::new();
            mlp_params.insert(
                Rc::from("gate_proj"),
                NestedValue::Map(lora_params_mut(&mut layer.mlp.gate_proj)),
            );
            mlp_params.insert(
                Rc::from("up_proj"),
                NestedValue::Map(lora_params_mut(&mut layer.mlp.up_proj)),
            );
            mlp_params.insert(
                Rc::from("down_proj"),
                NestedValue::Map(lora_params_mut(&mut layer.mlp.down_proj)),
            );
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

// Wire TrainableModel via shared macro.
crate::impl_trainable_model!(GraniteLoraForCausalLM);

// =============================================================================
// Helpers
// =============================================================================

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> GraniteConfig {
        GraniteConfig {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: Some(16),
            max_position_embeddings: 512,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
            is_hybrid: false,
            layer_types: None,
            ..Default::default()
        }
    }

    fn small_lora_config() -> LoraConfig {
        LoraConfig {
            r: 4,
            alpha: 8.0,
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
            loraplus_lr_ratio: None,
            use_dora: false,
        }
    }

    #[test]
    fn test_granite_lora_attention() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut attn = GraniteLoraAttention::new(&config, &lora_config).unwrap();

        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let output = attn.forward(&x, None).unwrap();
        assert_eq!(output.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_granite_lora_model_forward() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut model = GraniteLoraForCausalLM::new(config, lora_config).unwrap();

        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3, 4]).reshape(&[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();
        assert_eq!(logits.shape(), &[1, 4, 1000]);
    }

    #[test]
    fn test_granite_lora_param_count() {
        let config = small_config();
        let lora_config = small_lora_config();
        let model = GraniteLoraForCausalLM::new(config, lora_config).unwrap();

        assert!(model.num_trainable_params() > 0);

        let params = model.lora_parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_granite_lora_merge() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut model = GraniteLoraForCausalLM::new(config, lora_config).unwrap();

        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3, 4]).reshape(&[1, 4]);

        let output_before = model.forward(&input_ids, None).unwrap();
        output_before.eval().unwrap();

        model.merge_lora().unwrap();

        let output_after = model.forward(&input_ids, None).unwrap();
        output_after.eval().unwrap();

        let diff = output_before.subtract(&output_after).unwrap();
        let max_diff = diff.abs().unwrap().max(None).unwrap();
        max_diff.eval().unwrap();
        assert!(max_diff.item::<f32>() < 1e-4);

        assert!(model.unmerge_lora().is_err());
    }

    #[test]
    fn test_granite_lora_dora() {
        let config = small_config();
        let lora_config = LoraConfig {
            r: 4,
            alpha: 8.0,
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
            loraplus_lr_ratio: None,
            use_dora: true,
        };
        let mut model = GraniteLoraForCausalLM::new(config, lora_config).unwrap();

        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3, 4]).reshape(&[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();
        assert_eq!(logits.shape(), &[1, 4, 1000]);
    }

    #[test]
    fn test_granite_lora_rslora() {
        let config = small_config();
        let lora_config = LoraConfig {
            r: 4,
            alpha: 8.0,
            dropout: 0.0,
            use_rslora: true,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            bias: pmetal_core::LoraBias::None,
            init_lora_weights: true,
            loraplus_lr_ratio: None,
            use_dora: false,
        };
        let model = GraniteLoraForCausalLM::new(config, lora_config).unwrap();
        assert!(model.num_trainable_params() > 0);
    }
}
