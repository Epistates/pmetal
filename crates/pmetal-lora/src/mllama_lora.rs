//! LoRA-enabled MLlama (Llama 3.2 Vision) model architecture.
//!
//! Applies LoRA adapters to the **text decoder stack only**. The vision encoder
//! (`MllamaVisionModel`) and the multi-modal projector are not instantiated here
//! and are kept fully frozen in the base model during training.
//!
//! The text decoder has two attention sublayers per selected layer:
//! - Self-attention (all layers): q/k/v/o receive LoRA adapters.
//! - Cross-attention (layers listed in `cross_attention_layers`): q/k/v/o receive
//!   LoRA adapters as well — these are entirely text-side learnable projections.
//!
//! MLP gate/up/down projections also receive LoRA adapters on every layer.

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
use pmetal_models::architectures::mllama::MllamaConfig;
use pmetal_models::architectures::utils::create_causal_mask;

use crate::lora::LoraProjection;
use crate::lora_helpers::{
    LoraDecoderStack, collect_lora_parameters, count_trainable_params, load_lora_weights_impl,
    save_lora_weights_impl, set_lora_parameters as helpers_set_lora_parameters,
};
use crate::{LinearAdapter, LoraError, LoraLinear};

// ---------------------------------------------------------------------------
// Self-attention layer
// ---------------------------------------------------------------------------

/// LoRA-enabled self-attention layer for MLlama text decoder.
///
/// Applies LoRA (or DoRA) to q_proj, k_proj, v_proj, and o_proj.
#[derive(Debug)]
pub struct MllamaLoraSelfAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    pub q_proj: LinearAdapter,
    pub k_proj: LinearAdapter,
    pub v_proj: LinearAdapter,
    pub o_proj: LinearAdapter,
    pub rope: nn::Rope,
}

impl MllamaLoraSelfAttention {
    pub fn new(config: &MllamaConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let tc = &config.text_config.llama;
        let n_heads = tc.num_attention_heads;
        let n_kv_heads = tc.num_kv_heads();
        let head_dim = tc.get_head_dim();
        let scale = (head_dim as f32).sqrt().recip();

        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;
        let use_dora = lora_config.use_dora;

        let q_rank = crate::effective_rank(lora_config, "q_proj") as i32;
        let k_rank = crate::effective_rank(lora_config, "k_proj") as i32;
        let v_rank = crate::effective_rank(lora_config, "v_proj") as i32;
        let o_rank = crate::effective_rank(lora_config, "o_proj") as i32;

        let q_proj = LinearAdapter::new(
            tc.hidden_size,
            n_heads * head_dim,
            q_rank,
            alpha,
            use_rslora,
            false,
            use_dora,
        )?;
        let k_proj = LinearAdapter::new(
            tc.hidden_size,
            n_kv_heads * head_dim,
            k_rank,
            alpha,
            use_rslora,
            false,
            use_dora,
        )?;
        let v_proj = LinearAdapter::new(
            tc.hidden_size,
            n_kv_heads * head_dim,
            v_rank,
            alpha,
            use_rslora,
            false,
            use_dora,
        )?;
        let o_proj = LinearAdapter::new(
            n_heads * head_dim,
            tc.hidden_size,
            o_rank,
            alpha,
            use_rslora,
            false,
            use_dora,
        )?;

        let rope = nn::RopeBuilder::new(head_dim)
            .base(tc.rope_theta)
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

        let keys = expand_kv_heads_if_needed(&keys, self.n_heads, self.n_kv_heads)?;
        let values = expand_kv_heads_if_needed(&values, self.n_heads, self.n_kv_heads)?;

        let scores = queries
            .matmul(&keys.transpose_axes(&[0, 1, 3, 2]))
            .multiply(&Array::from_f32(self.scale));

        let scores = if let Some(m) = mask {
            scores.add(m)
        } else {
            scores
        };

        let weights = pmetal_bridge::compat::ops::softmax_axis(&scores, -1);
        let output = weights
            .matmul(&values)
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);

        self.o_proj.forward(&output).map_err(LoraError::from)
    }

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

        let keys = expand_kv_heads_if_needed(&keys, self.n_heads, self.n_kv_heads)?;
        let values = expand_kv_heads_if_needed(&values, self.n_heads, self.n_kv_heads)?;

        let scores = queries
            .matmul(&keys.transpose_axes(&[0, 1, 3, 2]))
            .multiply(&Array::from_f32(self.scale));

        let scores = if let Some(m) = mask {
            scores.add(m)
        } else {
            scores
        };

        let weights = pmetal_bridge::compat::ops::softmax_axis(&scores, -1);
        let output = weights
            .matmul(&values)
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);

        self.o_proj.forward(&output).map_err(LoraError::from)
    }

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

        let (queries, keys, values) = if let Some((ref cache_ref, _)) = cache {
            let offset = cache_ref.rope_offset();
            let q = apply_rope(&queries, self.head_dim, false, self.rope.base, 1.0, offset)?;
            let k = apply_rope(&keys, self.head_dim, false, self.rope.base, 1.0, offset)?;
            (q, k, values)
        } else {
            let q = pmetal_bridge::compat::Module::forward(&mut self.rope, &queries)?;
            let k = pmetal_bridge::compat::Module::forward(&mut self.rope, &keys)?;
            (q, k, values)
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

    pub fn num_trainable_params(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self.v_proj.num_trainable_params()
            + self.o_proj.num_trainable_params()
    }
}

// ---------------------------------------------------------------------------
// Cross-attention layer (text-side only — keys/values come from vision)
// ---------------------------------------------------------------------------

/// LoRA-enabled cross-attention layer for MLlama.
///
/// `q_proj` attends from the text hidden states; `k_proj` / `v_proj` project
/// the vision hidden states (already projected to `text_dim` by the multi-modal
/// projector).  All four projections are text-side parameters and receive LoRA.
#[derive(Debug)]
pub struct MllamaLoraCrossAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    pub q_proj: LinearAdapter,
    pub k_proj: LinearAdapter,
    pub v_proj: LinearAdapter,
    pub o_proj: LinearAdapter,
}

impl MllamaLoraCrossAttention {
    pub fn new(config: &MllamaConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let tc = &config.text_config.llama;
        let n_heads = tc.num_attention_heads;
        let n_kv_heads = tc.num_kv_heads();
        let head_dim = tc.get_head_dim();
        let text_dim = tc.hidden_size;

        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;
        let use_dora = lora_config.use_dora;

        // Re-use the same per-module rank names as self-attn — cross-attn shapes match.
        let q_rank = crate::effective_rank(lora_config, "q_proj") as i32;
        let k_rank = crate::effective_rank(lora_config, "k_proj") as i32;
        let v_rank = crate::effective_rank(lora_config, "v_proj") as i32;
        let o_rank = crate::effective_rank(lora_config, "o_proj") as i32;

        let q_proj = LinearAdapter::new(
            text_dim,
            n_heads * head_dim,
            q_rank,
            alpha,
            use_rslora,
            false,
            use_dora,
        )?;
        // k/v project from vision states which share text_dim after the projector.
        let k_proj = LinearAdapter::new(
            text_dim,
            n_kv_heads * head_dim,
            k_rank,
            alpha,
            use_rslora,
            false,
            use_dora,
        )?;
        let v_proj = LinearAdapter::new(
            text_dim,
            n_kv_heads * head_dim,
            v_rank,
            alpha,
            use_rslora,
            false,
            use_dora,
        )?;
        let o_proj = LinearAdapter::new(
            n_heads * head_dim,
            text_dim,
            o_rank,
            alpha,
            use_rslora,
            false,
            use_dora,
        )?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }

    /// Forward: x is text hidden states, cross_states are vision features.
    pub fn forward(&mut self, x: &Array, cross_states: &Array) -> Result<Array, LoraError> {
        let batch = x.shape()[0];
        let seq = x.shape()[1];
        let vis_seq = cross_states.shape()[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(cross_states)?;
        let v = self.v_proj.forward(cross_states)?;

        let q = q
            .reshape(&[batch, seq, self.n_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let k = k
            .reshape(&[batch, vis_seq, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let v = v
            .reshape(&[batch, vis_seq, self.n_kv_heads, self.head_dim])
            .transpose_axes(&[0, 2, 1, 3]);

        let k = expand_kv_heads_if_needed(&k, self.n_heads, self.n_kv_heads)?;
        let v = expand_kv_heads_if_needed(&v, self.n_heads, self.n_kv_heads)?;

        // No causal mask for cross-attention (full attendance to vision tokens).
        let scores = q
            .matmul(&k.transpose_axes(&[0, 1, 3, 2]))
            .multiply(&Array::from_f32(self.scale));

        let probs = pmetal_bridge::compat::ops::softmax_axis(&scores, -1);
        let output = probs
            .matmul(&v)
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq, -1]);

        self.o_proj.forward(&output).map_err(LoraError::from)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self.v_proj.num_trainable_params()
            + self.o_proj.num_trainable_params()
    }
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

/// LoRA-enabled MLP layer for MLlama text decoder.
#[derive(Debug)]
pub struct MllamaLoraMLP {
    pub gate_proj: LoraLinear,
    pub up_proj: LoraLinear,
    pub down_proj: LoraLinear,
}

impl MllamaLoraMLP {
    pub fn new(config: &MllamaConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let tc = &config.text_config.llama;
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;

        let gate_rank = crate::effective_rank(lora_config, "gate_proj") as i32;
        let up_rank = crate::effective_rank(lora_config, "up_proj") as i32;
        let down_rank = crate::effective_rank(lora_config, "down_proj") as i32;

        let gate_proj = LoraLinear::new(
            tc.hidden_size,
            tc.intermediate_size,
            gate_rank,
            alpha,
            use_rslora,
            false,
        )?;
        let up_proj = LoraLinear::new(
            tc.hidden_size,
            tc.intermediate_size,
            up_rank,
            alpha,
            use_rslora,
            false,
        )?;
        let down_proj = LoraLinear::new(
            tc.intermediate_size,
            tc.hidden_size,
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

    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let gate = nn::silu(&self.gate_proj.forward(x)?);
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.multiply(&up))
    }

    pub fn num_trainable_params(&self) -> usize {
        self.gate_proj.num_trainable_params()
            + self.up_proj.num_trainable_params()
            + self.down_proj.num_trainable_params()
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

/// LoRA-enabled MLlama text decoder layer.
///
/// Optionally contains a cross-attention sublayer (only on layers listed in
/// `config.text_config.cross_attention_layers`).
#[derive(Debug)]
pub struct MllamaLoraDecoderLayer {
    pub self_attn: MllamaLoraSelfAttention,
    /// Present only for layers in `cross_attention_layers`.
    pub cross_attn: Option<MllamaLoraCrossAttention>,
    pub mlp: MllamaLoraMLP,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
    /// Pre-cross-attention norm (only when `cross_attn` is `Some`).
    pub cross_attention_layernorm: Option<nn::RmsNorm>,
}

impl MllamaLoraDecoderLayer {
    pub fn new(
        config: &MllamaConfig,
        lora_config: &LoraConfig,
        layer_id: usize,
    ) -> Result<Self, LoraError> {
        let tc = &config.text_config.llama;

        let self_attn = MllamaLoraSelfAttention::new(config, lora_config)?;
        let mlp = MllamaLoraMLP::new(config, lora_config)?;

        let input_layernorm = nn::RmsNormBuilder::new(tc.hidden_size)
            .eps(tc.rms_norm_eps)
            .build()
            .unwrap();
        let post_attention_layernorm = nn::RmsNormBuilder::new(tc.hidden_size)
            .eps(tc.rms_norm_eps)
            .build()
            .unwrap();

        let has_cross = config
            .text_config
            .cross_attention_layers
            .contains(&(layer_id as i32));

        let (cross_attn, cross_attention_layernorm) = if has_cross {
            let ca = MllamaLoraCrossAttention::new(config, lora_config)?;
            let norm = nn::RmsNormBuilder::new(tc.hidden_size)
                .eps(tc.rms_norm_eps)
                .build()
                .unwrap();
            (Some(ca), Some(norm))
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
        cross_states: Option<&Array>,
    ) -> Result<Array, LoraError> {
        let normed = pmetal_bridge::compat::Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self.self_attn.forward(&normed, mask)?;
        let mut h = x.add(&attn_out);

        if let (Some(ca), Some(cs), Some(cn)) = (
            self.cross_attn.as_mut(),
            cross_states,
            self.cross_attention_layernorm.as_mut(),
        ) {
            let normed = pmetal_bridge::compat::Module::forward(cn, &h)?;
            let cross_out = ca.forward(&normed, cs)?;
            h = h.add(&cross_out);
        }

        let normed =
            pmetal_bridge::compat::Module::forward(&mut self.post_attention_layernorm, &h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok(h.add(&mlp_out))
    }

    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cross_states: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let normed = pmetal_bridge::compat::Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self.self_attn.forward_with_cache(&normed, mask, cache)?;
        let mut h = x.add(&attn_out);

        if let (Some(ca), Some(cs), Some(cn)) = (
            self.cross_attn.as_mut(),
            cross_states,
            self.cross_attention_layernorm.as_mut(),
        ) {
            let normed = pmetal_bridge::compat::Module::forward(cn, &h)?;
            let cross_out = ca.forward(&normed, cs)?;
            h = h.add(&cross_out);
        }

        let normed =
            pmetal_bridge::compat::Module::forward(&mut self.post_attention_layernorm, &h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok(h.add(&mlp_out))
    }

    pub fn forward_with_positions(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cross_states: Option<&Array>,
        position_ids: &Array,
    ) -> Result<Array, LoraError> {
        let normed = pmetal_bridge::compat::Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self
            .self_attn
            .forward_with_positions(&normed, mask, position_ids)?;
        let mut h = x.add(&attn_out);

        if let (Some(ca), Some(cs), Some(cn)) = (
            self.cross_attn.as_mut(),
            cross_states,
            self.cross_attention_layernorm.as_mut(),
        ) {
            let normed = pmetal_bridge::compat::Module::forward(cn, &h)?;
            let cross_out = ca.forward(&normed, cs)?;
            h = h.add(&cross_out);
        }

        let normed =
            pmetal_bridge::compat::Module::forward(&mut self.post_attention_layernorm, &h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok(h.add(&mlp_out))
    }

    /// Number of trainable parameters in this layer (self-attn + cross-attn + MLP).
    pub fn num_trainable_params(&self) -> usize {
        let ca = self
            .cross_attn
            .as_ref()
            .map(|c| c.num_trainable_params())
            .unwrap_or(0);
        self.self_attn.num_trainable_params() + ca + self.mlp.num_trainable_params()
    }
}

// ---------------------------------------------------------------------------
// Inner model (no LM head)
// ---------------------------------------------------------------------------

/// LoRA-enabled MLlama text model (decoder stack without LM head).
///
/// The vision encoder is **not** present here. Pass pre-computed
/// `cross_attention_states` (vision features projected to `text_dim`) when
/// calling `forward`; use `None` for text-only inputs.
#[derive(Debug)]
pub struct MllamaLoraModel {
    pub config: MllamaConfig,
    pub lora_config: LoraConfig,

    /// Token embeddings (frozen).
    pub embed_tokens: nn::Embedding,
    /// Text decoder layers with LoRA.
    pub layers: Vec<MllamaLoraDecoderLayer>,
    /// Final layer norm (frozen).
    pub norm: nn::RmsNorm,
}

impl MllamaLoraModel {
    pub fn new(config: MllamaConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let tc = &config.text_config.llama;
        let embed_tokens = nn::Embedding::new(tc.vocab_size, tc.hidden_size)?;

        let layers = (0..tc.num_hidden_layers as usize)
            .map(|id| MllamaLoraDecoderLayer::new(&config, &lora_config, id))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::RmsNormBuilder::new(tc.hidden_size)
            .eps(tc.rms_norm_eps)
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

    /// Forward through the text decoder stack.
    ///
    /// `cross_states` — optional pre-computed vision features `[batch, vis_seq, hidden]`
    /// that have already been projected to `text_dim` by the multi-modal projector.
    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cross_states: Option<&Array>,
    ) -> Result<Array, LoraError> {
        self.forward_with_checkpoint(input_ids, mask, cross_states, None)
    }

    pub fn forward_with_checkpoint(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cross_states: Option<&Array>,
        checkpoint_config: Option<&CheckpointConfig>,
    ) -> Result<Array, LoraError> {
        let mut h = pmetal_bridge::compat::Module::forward(&mut self.embed_tokens, input_ids)?;

        let mask = if mask.is_none() {
            let seq_len = input_ids.dim(1);
            Some(create_causal_mask(seq_len)?)
        } else {
            mask.cloned()
        };

        let layers_per_block = checkpoint_config
            .map(|c| c.layers_per_block)
            .unwrap_or(usize::MAX);
        let checkpointing = checkpoint_config.map(|c| c.enabled).unwrap_or(false);

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            h = layer.forward(&h, mask.as_ref(), cross_states)?;
            if checkpointing && (idx + 1) % layers_per_block == 0 {
                tracing::trace!("Checkpoint boundary at layer {}", idx + 1);
            }
        }

        Ok(pmetal_bridge::compat::Module::forward(&mut self.norm, &h)?)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cross_states: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let mut h = pmetal_bridge::compat::Module::forward(&mut self.embed_tokens, input_ids)?;

        match cache {
            Some(cache) => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    h = layer.forward_with_cache(
                        &h,
                        mask,
                        cross_states,
                        Some((cache, layer_idx)),
                    )?;
                }
            }
            None => {
                for layer in &mut self.layers {
                    h = layer.forward(&h, mask, cross_states)?;
                }
            }
        }

        Ok(pmetal_bridge::compat::Module::forward(&mut self.norm, &h)?)
    }

    pub fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cross_states: Option<&Array>,
        position_ids: &Array,
    ) -> Result<Array, LoraError> {
        let mut h = pmetal_bridge::compat::Module::forward(&mut self.embed_tokens, input_ids)?;

        for layer in &mut self.layers {
            h = layer.forward_with_positions(&h, mask, cross_states, position_ids)?;
        }

        Ok(pmetal_bridge::compat::Module::forward(&mut self.norm, &h)?)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_trainable_params()).sum()
    }
}

// ---------------------------------------------------------------------------
// LoraDecoderStack — needed by lora_helpers utilities
// ---------------------------------------------------------------------------
//
// `LoraDecoderStack` is designed for homogeneous stacks where every layer has
// the same set of projections.  MLlama layers are heterogeneous: only some have
// cross-attention.  We expose *only* the self-attention projections through the
// trait and handle cross-attention parameters manually in the `ModuleParameters`
// impl and the explicit helpers below.

impl LoraDecoderStack for MllamaLoraModel {
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn attn_projections(&self, layer: usize) -> Vec<&dyn LoraProjection> {
        let l = &self.layers[layer];
        vec![
            &l.self_attn.q_proj,
            &l.self_attn.k_proj,
            &l.self_attn.v_proj,
            &l.self_attn.o_proj,
        ]
    }

    fn attn_projections_mut(&mut self, layer: usize) -> Vec<&mut dyn LoraProjection> {
        let l = &mut self.layers[layer];
        vec![
            &mut l.self_attn.q_proj,
            &mut l.self_attn.k_proj,
            &mut l.self_attn.v_proj,
            &mut l.self_attn.o_proj,
        ]
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

// ---------------------------------------------------------------------------
// Full model with LM head
// ---------------------------------------------------------------------------

/// LoRA-enabled MLlama model with LM head.
///
/// This is the entry-point for training / inference.  The vision encoder is
/// absent; callers that have pixel values must run the frozen vision model
/// separately and pass the projected features as `cross_attention_states`.
#[derive(Debug)]
pub struct MllamaLoraForCausalLM {
    pub model: MllamaLoraModel,
    /// LM head (frozen).
    pub lm_head: Option<nn::Linear>,
    /// Gradient checkpointing configuration.
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl MllamaLoraForCausalLM {
    pub fn new(config: MllamaConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let tc = config.text_config.llama.clone();
        let tie_weights = tc.tie_word_embeddings;
        let model = MllamaLoraModel::new(config, lora_config)?;

        let lm_head = if !tie_weights {
            Some(
                nn::LinearBuilder::new(tc.hidden_size, tc.vocab_size)
                    .bias(false)
                    .build()
                    .unwrap(),
            )
        } else {
            None
        };

        Ok(Self {
            model,
            lm_head,
            checkpoint_config: None,
        })
    }

    pub fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        self.checkpoint_config = Some(CheckpointConfig {
            enabled: true,
            layers_per_block,
            eval_at_boundaries: true,
        });
    }

    pub fn disable_gradient_checkpointing(&mut self) {
        self.checkpoint_config = None;
    }

    // ------------------------------------------------------------------
    // Forward helpers
    // ------------------------------------------------------------------

    fn apply_lm_head(&mut self, hidden: Array) -> Result<Array, LoraError> {
        if let Some(ref mut head) = self.lm_head {
            Ok(pmetal_bridge::compat::Module::forward(head, &hidden)?)
        } else {
            Ok(self.model.embed_tokens.as_linear(&hidden))
        }
    }

    /// Standard forward (text-only; cross_states = None).
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        self.forward_multimodal(input_ids, mask, None)
    }

    /// Forward with optional pre-computed vision features.
    ///
    /// Pass `cross_states = Some(projected_vision)` when image features are
    /// available; `None` for text-only steps.
    pub fn forward_multimodal(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cross_states: Option<&Array>,
    ) -> Result<Array, LoraError> {
        let ckpt = self.checkpoint_config.clone();
        let hidden =
            self.model
                .forward_with_checkpoint(input_ids, mask, cross_states, ckpt.as_ref())?;
        self.apply_lm_head(hidden)
    }

    pub fn forward_hidden_states(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
        let ckpt = self.checkpoint_config.clone();
        self.model
            .forward_with_checkpoint(input_ids, mask, None, ckpt.as_ref())
    }

    pub fn forward_hidden_states_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Result<Array, LoraError> {
        self.model
            .forward_with_positions(input_ids, mask, None, position_ids)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cross_states: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let hidden = self
            .model
            .forward_with_cache(input_ids, mask, cross_states, cache)?;
        self.apply_lm_head(hidden)
    }

    // ------------------------------------------------------------------
    // Cache helpers
    // ------------------------------------------------------------------

    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        let tc = &self.model.config.text_config.llama;
        let config = KVCacheConfig::new(
            tc.num_hidden_layers as usize,
            max_seq_len,
            tc.num_kv_heads() as usize,
            tc.get_head_dim() as usize,
        );
        KVCache::new(config)
    }

    // ------------------------------------------------------------------
    // LoRA parameter utilities
    // ------------------------------------------------------------------

    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        // Start with self-attn + MLP from the shared helper.
        let mut params = collect_lora_parameters(&self.model);

        // Append cross-attention LoRA params for layers that have them.
        for (i, layer) in self.model.layers.iter().enumerate() {
            let Some(ref ca) = layer.cross_attn else {
                continue;
            };
            for (proj_name, proj) in [
                ("q_proj", &ca.q_proj),
                ("k_proj", &ca.k_proj),
                ("v_proj", &ca.v_proj),
                ("o_proj", &ca.o_proj),
            ] {
                let a_key: Rc<str> =
                    Rc::from(format!("layers.{}.cross_attn.{}.lora_a", i, proj_name));
                let b_key: Rc<str> =
                    Rc::from(format!("layers.{}.cross_attn.{}.lora_b", i, proj_name));
                params.insert(a_key, proj.lora_a().clone());
                params.insert(b_key, proj.lora_b().clone());
            }
        }

        params
    }

    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        helpers_set_lora_parameters(&mut self.model, params);

        // Restore cross-attention LoRA params.
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let Some(ref mut ca) = layer.cross_attn else {
                continue;
            };
            for (proj_name, proj) in [
                ("q_proj", &mut ca.q_proj),
                ("k_proj", &mut ca.k_proj),
                ("v_proj", &mut ca.v_proj),
                ("o_proj", &mut ca.o_proj),
            ] {
                let a_key = Rc::from(format!("layers.{}.cross_attn.{}.lora_a", i, proj_name));
                let b_key = Rc::from(format!("layers.{}.cross_attn.{}.lora_b", i, proj_name));
                if let Some(a) = params.get(&a_key) {
                    *proj.lora_a_mut() = a.clone();
                }
                if let Some(b) = params.get(&b_key) {
                    *proj.lora_b_mut() = b.clone();
                }
            }
        }
    }

    pub fn num_trainable_params(&self) -> usize {
        count_trainable_params(&self.model)
    }

    pub fn config(&self) -> &MllamaConfig {
        &self.model.config
    }

    pub fn lora_config(&self) -> &LoraConfig {
        &self.model.lora_config
    }

    // ------------------------------------------------------------------
    // Weight I/O
    // ------------------------------------------------------------------

    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        for layer in &mut self.model.layers {
            layer.self_attn.q_proj.merge()?;
            layer.self_attn.k_proj.merge()?;
            layer.self_attn.v_proj.merge()?;
            layer.self_attn.o_proj.merge()?;
            layer.mlp.gate_proj.merge()?;
            layer.mlp.up_proj.merge()?;
            layer.mlp.down_proj.merge()?;
            if let Some(ref mut ca) = layer.cross_attn {
                ca.q_proj.merge()?;
                ca.k_proj.merge()?;
                ca.v_proj.merge()?;
                ca.o_proj.merge()?;
            }
        }
        Ok(())
    }

    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        Err(LoraError::InvalidState(
            "unmerge_lora is not supported: reload base model weights to undo a merge".to_string(),
        ))
    }

    pub fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        save_lora_weights_impl(&self.model, path)
    }

    pub fn load_lora_weights(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), LoraError> {
        load_lora_weights_impl(&mut self.model, path)
    }

    /// Load base (frozen) model weights from a weight map.
    ///
    /// Loads text decoder weights only; vision encoder weights are not expected
    /// here and will be ignored if present in the map.
    ///
    /// Expected key format (HuggingFace):
    /// - `language_model.model.embed_tokens.weight`
    /// - `language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
    /// - `language_model.model.layers.{i}.cross_attn.{q,k,v,o}_proj.weight`
    /// - `language_model.model.layers.{i}.mlp.{gate,up,down}_proj.weight`
    /// - `language_model.model.layers.{i}.input_layernorm.weight`
    /// - `language_model.model.layers.{i}.post_attention_layernorm.weight`
    /// - `language_model.model.layers.{i}.cross_attention_layernorm.weight`
    /// - `language_model.model.norm.weight`
    /// - `language_model.lm_head.weight`
    pub fn load_base_weights(&mut self, weights: &HashMap<String, Array>) -> Result<(), LoraError> {
        use pmetal_bridge::compat::Param;

        let lm_prefix = "language_model.model";

        if let Some(w) = weights.get(&format!("{}.embed_tokens.weight", lm_prefix)) {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let pfx = format!("{}.layers.{}", lm_prefix, i);

            macro_rules! load {
                ($target:expr, $key:expr) => {
                    if let Some(w) = weights.get(&format!("{}.{}", pfx, $key)) {
                        *$target = w.clone();
                    }
                };
            }

            load!(
                layer.self_attn.q_proj.weight_mut(),
                "self_attn.q_proj.weight"
            );
            load!(
                layer.self_attn.k_proj.weight_mut(),
                "self_attn.k_proj.weight"
            );
            load!(
                layer.self_attn.v_proj.weight_mut(),
                "self_attn.v_proj.weight"
            );
            load!(
                layer.self_attn.o_proj.weight_mut(),
                "self_attn.o_proj.weight"
            );

            load!(layer.mlp.gate_proj.weight_mut(), "mlp.gate_proj.weight");
            load!(layer.mlp.up_proj.weight_mut(), "mlp.up_proj.weight");
            load!(layer.mlp.down_proj.weight_mut(), "mlp.down_proj.weight");

            if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", pfx)) {
                layer.input_layernorm.weight = Param::new(w.clone());
            }
            if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", pfx)) {
                layer.post_attention_layernorm.weight = Param::new(w.clone());
            }

            if let Some(ref mut ca) = layer.cross_attn {
                load!(ca.q_proj.weight_mut(), "cross_attn.q_proj.weight");
                load!(ca.k_proj.weight_mut(), "cross_attn.k_proj.weight");
                load!(ca.v_proj.weight_mut(), "cross_attn.v_proj.weight");
                load!(ca.o_proj.weight_mut(), "cross_attn.o_proj.weight");
            }
            if let Some(ref mut cn) = layer.cross_attention_layernorm {
                if let Some(w) = weights.get(&format!("{}.cross_attention_layernorm.weight", pfx)) {
                    cn.weight = Param::new(w.clone());
                }
            }
        }

        if let Some(w) = weights.get(&format!("{}.norm.weight", lm_prefix)) {
            self.model.norm.weight = Param::new(w.clone());
        }

        if let Some(ref mut lm_head) = self.lm_head {
            if let Some(w) = weights.get("language_model.lm_head.weight") {
                lm_head.weight = Param::new(w.clone());
            }
        }

        Ok(())
    }

    pub fn load_base_weights_from_dir(
        &mut self,
        model_dir: impl AsRef<std::path::Path>,
    ) -> Result<(), LoraError> {
        let model_dir = model_dir.as_ref();
        let single = model_dir.join("model.safetensors");
        if single.exists() {
            let weights = crate::sanitize_loaded_weights(crate::load_safetensors_map(&single)?)?;
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
            let shard_weights = crate::sanitize_loaded_weights(crate::load_safetensors_map(
                &model_dir.join(shard_file),
            )?)?;
            all_weights.extend(shard_weights);
        }

        self.load_base_weights(&all_weights)
    }

    pub fn get_lm_head_weight(&self) -> Option<Array> {
        if let Some(ref lm_head) = self.lm_head {
            Some(lm_head.weight.value.clone())
        } else {
            Some(self.model.embed_tokens.weight.value.clone())
        }
    }

    pub fn eval_all(&mut self) -> Result<(), LoraError> {
        self.model.embed_tokens.weight.value.eval();

        for layer in &mut self.model.layers {
            layer.self_attn.q_proj.weight_mut().eval();
            layer.self_attn.k_proj.weight_mut().eval();
            layer.self_attn.v_proj.weight_mut().eval();
            layer.self_attn.o_proj.weight_mut().eval();
            layer.self_attn.q_proj.lora_a_mut().eval();
            layer.self_attn.q_proj.lora_b_mut().eval();
            layer.self_attn.k_proj.lora_a_mut().eval();
            layer.self_attn.k_proj.lora_b_mut().eval();
            layer.self_attn.v_proj.lora_a_mut().eval();
            layer.self_attn.v_proj.lora_b_mut().eval();
            layer.self_attn.o_proj.lora_a_mut().eval();
            layer.self_attn.o_proj.lora_b_mut().eval();

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

            if let Some(ref mut ca) = layer.cross_attn {
                ca.q_proj.weight_mut().eval();
                ca.k_proj.weight_mut().eval();
                ca.v_proj.weight_mut().eval();
                ca.o_proj.weight_mut().eval();
                ca.q_proj.lora_a_mut().eval();
                ca.q_proj.lora_b_mut().eval();
                ca.k_proj.lora_a_mut().eval();
                ca.k_proj.lora_b_mut().eval();
                ca.v_proj.lora_a_mut().eval();
                ca.v_proj.lora_b_mut().eval();
                ca.o_proj.lora_a_mut().eval();
                ca.o_proj.lora_b_mut().eval();
            }
            if let Some(ref mut cn) = layer.cross_attention_layernorm {
                cn.weight.value.eval();
            }
        }

        self.model.norm.weight.value.eval();
        if let Some(ref mut h) = self.lm_head {
            h.weight.value.eval();
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ModuleParameters impl
// ---------------------------------------------------------------------------

impl ModuleParameters for MllamaLoraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.model.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();

        for (i, layer) in self.model.layers.iter().enumerate() {
            let prefix: Rc<str> = Rc::from(format!("layers.{}", i));
            let mut layer_params = HashMap::new();

            // Self-attention LoRA params.
            let mut sa_params = HashMap::new();
            for (name, proj) in [
                ("q_proj", &layer.self_attn.q_proj),
                ("k_proj", &layer.self_attn.k_proj),
                ("v_proj", &layer.self_attn.v_proj),
                ("o_proj", &layer.self_attn.o_proj),
            ] {
                let mut m = HashMap::new();
                m.insert(Rc::from("lora_a"), NestedValue::Value(proj.lora_a()));
                m.insert(Rc::from("lora_b"), NestedValue::Value(proj.lora_b()));
                sa_params.insert(Rc::from(name), NestedValue::Map(m));
            }
            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(sa_params));

            // Cross-attention LoRA params (optional).
            if let Some(ref ca) = layer.cross_attn {
                let mut ca_params = HashMap::new();
                for (name, proj) in [
                    ("q_proj", &ca.q_proj),
                    ("k_proj", &ca.k_proj),
                    ("v_proj", &ca.v_proj),
                    ("o_proj", &ca.o_proj),
                ] {
                    let mut m = HashMap::new();
                    m.insert(Rc::from("lora_a"), NestedValue::Value(proj.lora_a()));
                    m.insert(Rc::from("lora_b"), NestedValue::Value(proj.lora_b()));
                    ca_params.insert(Rc::from(name), NestedValue::Map(m));
                }
                layer_params.insert(Rc::from("cross_attn"), NestedValue::Map(ca_params));
            }

            // MLP LoRA params.
            let mut mlp_params = HashMap::new();
            for (name, proj) in [
                ("gate_proj", &layer.mlp.gate_proj as &dyn LoraProjection),
                ("up_proj", &layer.mlp.up_proj),
                ("down_proj", &layer.mlp.down_proj),
            ] {
                let mut m = HashMap::new();
                m.insert(Rc::from("lora_a"), NestedValue::Value(proj.lora_a()));
                m.insert(Rc::from("lora_b"), NestedValue::Value(proj.lora_b()));
                mlp_params.insert(Rc::from(name), NestedValue::Map(m));
            }
            layer_params.insert(Rc::from("mlp"), NestedValue::Map(mlp_params));

            params.insert(prefix, NestedValue::Map(layer_params));
        }

        params
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut params = ModuleParamMut::new();

        fn adapter_mut(a: &mut LinearAdapter) -> HashMap<Rc<str>, NestedValue<&mut Array>> {
            let mut m = HashMap::new();
            match a {
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

        fn lora_linear_mut(l: &mut LoraLinear) -> HashMap<Rc<str>, NestedValue<&mut Array>> {
            let mut m = HashMap::new();
            m.insert(Rc::from("lora_a"), NestedValue::Value(&mut l.lora_a));
            m.insert(Rc::from("lora_b"), NestedValue::Value(&mut l.lora_b));
            m
        }

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix: Rc<str> = Rc::from(format!("layers.{}", i));
            let mut layer_params = HashMap::new();

            // Self-attention.
            let mut sa_params = HashMap::new();
            sa_params.insert(
                Rc::from("q_proj"),
                NestedValue::Map(adapter_mut(&mut layer.self_attn.q_proj)),
            );
            sa_params.insert(
                Rc::from("k_proj"),
                NestedValue::Map(adapter_mut(&mut layer.self_attn.k_proj)),
            );
            sa_params.insert(
                Rc::from("v_proj"),
                NestedValue::Map(adapter_mut(&mut layer.self_attn.v_proj)),
            );
            sa_params.insert(
                Rc::from("o_proj"),
                NestedValue::Map(adapter_mut(&mut layer.self_attn.o_proj)),
            );
            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(sa_params));

            // Cross-attention (optional).
            if let Some(ref mut ca) = layer.cross_attn {
                let mut ca_params = HashMap::new();
                ca_params.insert(
                    Rc::from("q_proj"),
                    NestedValue::Map(adapter_mut(&mut ca.q_proj)),
                );
                ca_params.insert(
                    Rc::from("k_proj"),
                    NestedValue::Map(adapter_mut(&mut ca.k_proj)),
                );
                ca_params.insert(
                    Rc::from("v_proj"),
                    NestedValue::Map(adapter_mut(&mut ca.v_proj)),
                );
                ca_params.insert(
                    Rc::from("o_proj"),
                    NestedValue::Map(adapter_mut(&mut ca.o_proj)),
                );
                layer_params.insert(Rc::from("cross_attn"), NestedValue::Map(ca_params));
            }

            // MLP.
            let mut mlp_params = HashMap::new();
            mlp_params.insert(
                Rc::from("gate_proj"),
                NestedValue::Map(lora_linear_mut(&mut layer.mlp.gate_proj)),
            );
            mlp_params.insert(
                Rc::from("up_proj"),
                NestedValue::Map(lora_linear_mut(&mut layer.mlp.up_proj)),
            );
            mlp_params.insert(
                Rc::from("down_proj"),
                NestedValue::Map(lora_linear_mut(&mut layer.mlp.down_proj)),
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

// ---------------------------------------------------------------------------
// TrainableModel
// ---------------------------------------------------------------------------

impl crate::TrainableModel for MllamaLoraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        MllamaLoraForCausalLM::forward(self, input_ids, mask)
    }

    fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Result<Array, LoraError> {
        let ckpt = self.checkpoint_config.clone();
        let hidden = self
            .model
            .forward_with_positions(input_ids, mask, None, position_ids)?;
        let _ = ckpt;
        self.apply_lm_head(hidden)
    }

    fn forward_with_images(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        pixel_values: Option<&Array>,
    ) -> Result<Array, LoraError> {
        // `pixel_values` must have been projected to `cross_states` before reaching
        // this point when the caller has a full vision stack.  For adapter-only
        // training workflows the cross states are often precomputed and passed
        // through `pixel_values` directly (already projected to text_dim).
        // If the tensor shape matches the text hidden dim on the last axis we
        // treat it as pre-projected cross states; otherwise we fall back to
        // text-only forward to avoid a shape crash.
        let cross_states = pixel_values.and_then(|pv| {
            let last_dim = pv.shape().last().copied().unwrap_or(0);
            if last_dim == self.model.config.text_config.llama.hidden_size {
                Some(pv)
            } else {
                None
            }
        });
        self.forward_multimodal(input_ids, mask, cross_states)
    }

    fn is_multimodal(&self) -> bool {
        true
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        MllamaLoraForCausalLM::forward_with_cache(self, input_ids, mask, None, cache)
    }

    fn supports_kv_cache(&self) -> bool {
        true
    }

    fn create_cache(&self, max_seq_len: usize) -> Option<KVCache> {
        Some(MllamaLoraForCausalLM::create_cache(self, max_seq_len))
    }

    fn num_trainable_params(&self) -> usize {
        MllamaLoraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        MllamaLoraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        MllamaLoraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        MllamaLoraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        MllamaLoraForCausalLM::load_lora_weights(self, path)
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        MllamaLoraForCausalLM::enable_gradient_checkpointing(self, layers_per_block)
    }

    fn disable_gradient_checkpointing(&mut self) {
        MllamaLoraForCausalLM::disable_gradient_checkpointing(self)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        true
    }

    fn forward_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Option<Result<Array, LoraError>> {
        Some(self.forward_hidden_states(input_ids, mask))
    }

    fn forward_hidden_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Option<Result<Array, LoraError>> {
        Some(self.forward_hidden_states_with_positions(input_ids, mask, position_ids))
    }

    fn lm_head_weight(&self) -> Option<Array> {
        MllamaLoraForCausalLM::get_lm_head_weight(self)
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn expand_kv_heads_if_needed(x: &Array, n_heads: i32, n_kv_heads: i32) -> Result<Array, LoraError> {
    if n_kv_heads < n_heads {
        let repeats = n_heads / n_kv_heads;
        Ok(expand_kv_heads(x, repeats).map_err(LoraError::Mlx)?)
    } else {
        Ok(x.clone())
    }
}

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_models::architectures::mllama::{MllamaTextConfig, MllamaVisionConfig};

    fn small_config() -> MllamaConfig {
        use pmetal_models::architectures::llama::LlamaConfig;
        let llama = LlamaConfig {
            vocab_size: 512,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: None,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            ..Default::default()
        };
        let vision_config = MllamaVisionConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_global_layers: 1,
            attention_heads: 4,
            image_size: 28,
            patch_size: 14,
            intermediate_layers_indices: vec![1],
            vision_output_dim: 64,
            ..Default::default()
        };
        MllamaConfig {
            text_config: MllamaTextConfig {
                llama,
                // Layers 1 and 3 have cross-attention.
                cross_attention_layers: vec![1, 3],
            },
            vision_config,
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
    fn test_mllama_lora_self_attn() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut attn = MllamaLoraSelfAttention::new(&config, &lora_config).unwrap();

        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let out = attn.forward(&x, None).unwrap();
        assert_eq!(out.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_mllama_lora_cross_attn() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut ca = MllamaLoraCrossAttention::new(&config, &lora_config).unwrap();

        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let cross = pmetal_bridge::compat::random::normal(
            &[1, 8, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let out = ca.forward(&x, &cross).unwrap();
        assert_eq!(out.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_mllama_lora_text_only_forward() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut model = MllamaLoraForCausalLM::new(config, lora_config).unwrap();

        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3, 4]).reshape(&[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();

        assert_eq!(logits.shape(), &[1, 4, 512]);
    }

    #[test]
    fn test_mllama_lora_with_cross_states() {
        let config = small_config();
        let lora_config = small_lora_config();
        let mut model = MllamaLoraForCausalLM::new(config, lora_config).unwrap();

        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3, 4]).reshape(&[1, 4]);
        // Pre-projected vision features with hidden_size = 64.
        let cross_states = pmetal_bridge::compat::random::normal(
            &[1, 16, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let logits = model
            .forward_multimodal(&input_ids, None, Some(&cross_states))
            .unwrap();

        assert_eq!(logits.shape(), &[1, 4, 512]);
    }

    #[test]
    fn test_mllama_lora_param_count() {
        let config = small_config();
        let lora_config = small_lora_config();
        let model = MllamaLoraForCausalLM::new(config, lora_config).unwrap();

        assert!(model.num_trainable_params() > 0);

        let params = model.lora_parameters();
        assert!(!params.is_empty());

        // Cross-attention LoRA params should be present for layers 1 and 3.
        assert!(params.contains_key(&Rc::from("layers.1.cross_attn.q_proj.lora_a") as &Rc<str>));
        assert!(params.contains_key(&Rc::from("layers.3.cross_attn.v_proj.lora_b") as &Rc<str>));
        // Layer 0 has no cross-attn.
        assert!(!params.contains_key(&Rc::from("layers.0.cross_attn.q_proj.lora_a") as &Rc<str>));
    }

    #[test]
    fn test_mllama_lora_dora() {
        let config = small_config();
        let lora_config = LoraConfig {
            r: 4,
            alpha: 8.0,
            dropout: 0.0,
            use_rslora: false,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            bias: pmetal_core::LoraBias::None,
            init_lora_weights: true,
            loraplus_lr_ratio: None,
            use_dora: true,
        };
        let model = MllamaLoraForCausalLM::new(config, lora_config).unwrap();
        assert!(model.num_trainable_params() > 0);
    }
}
