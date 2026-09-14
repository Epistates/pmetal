//! QLoRA-enabled Phi model architecture (Phi-3, Phi-3.5, Phi-4).
//!
//! Implements Phi with QLoRA (Quantized LoRA) for memory-efficient fine-tuning.
//! Base weights are stored in 4-bit NF4 format, reducing memory by ~87.5%.
//! LoRA adapters (A, B matrices) remain in full precision for training.
//!
//! Key Phi differences preserved from phi_lora.rs:
//! - Partial RoPE (applied to a subset of head dimensions)
//! - Fused gate_up_proj for SwiGLU (unlike Llama's separate gate/up)
//! - QKV bias support (Phi-4)

use std::collections::HashMap;
use std::rc::Rc;

use pmetal_bridge::compat::fast;
use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, Param,
    nn,
};

use pmetal_core::LoraConfig;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_mlx::kernels::{AttentionMaskType, FusedAttentionConfig, fused_sdpa, rope::apply_rope};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};
use pmetal_models::architectures::phi::{PhiActivation, PhiConfig};
use pmetal_models::architectures::utils::create_causal_mask;

use crate::{LoraError, QLoraConfig, QLoraLinear};

// ─── RMSNorm (reuse from phi_lora pattern, plain Param weight) ────────────────

/// QLoRA-enabled Phi RMS LayerNorm (frozen, full-precision).
#[derive(Debug)]
pub struct PhiQloraRmsNorm {
    /// Weight parameter.
    pub weight: Param<Array>,
    /// Epsilon.
    pub eps: f32,
}

impl PhiQloraRmsNorm {
    /// Create a new RMS LayerNorm initialised to ones.
    pub fn new(hidden_size: i32, eps: f32) -> Self {
        let weight = Param::new(pmetal_bridge::compat::ops::ones(
            &[hidden_size],
            pmetal_bridge::compat::Dtype::Float32,
        ));
        Self { weight, eps }
    }

    /// Forward pass using optimised fast::rms_norm.
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        Ok(fast::rms_norm(x, &self.weight, self.eps))
    }
}

// ─── Attention ────────────────────────────────────────────────────────────────

/// QLoRA-enabled attention layer for Phi with partial RoPE.
#[derive(Debug)]
pub struct PhiQLoraAttention {
    /// Query projection with QLoRA.
    pub q_proj: QLoraLinear,
    /// Key projection with QLoRA.
    pub k_proj: QLoraLinear,
    /// Value projection with QLoRA.
    pub v_proj: QLoraLinear,
    /// Output projection with QLoRA.
    pub o_proj: QLoraLinear,
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

impl PhiQLoraAttention {
    /// Create a new QLoRA attention layer.
    pub fn new(config: &PhiConfig, qlora_config: &QLoraConfig) -> Result<Self, LoraError> {
        let head_dim = config.head_dim();
        let rope_dim = config.rope_dim();

        let mut q_config = qlora_config.clone();
        q_config.lora.r = crate::effective_rank(&qlora_config.lora, "q_proj");
        let q_proj = QLoraLinear::new(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            &q_config,
            config.qkv_bias,
        )?;

        let mut k_config = qlora_config.clone();
        k_config.lora.r = crate::effective_rank(&qlora_config.lora, "k_proj");
        let k_proj = QLoraLinear::new(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            &k_config,
            config.qkv_bias,
        )?;

        let mut v_config = qlora_config.clone();
        v_config.lora.r = crate::effective_rank(&qlora_config.lora, "v_proj");
        let v_proj = QLoraLinear::new(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            &v_config,
            config.qkv_bias,
        )?;

        let mut o_config = qlora_config.clone();
        o_config.lora.r = crate::effective_rank(&qlora_config.lora, "o_proj");
        let o_proj = QLoraLinear::new(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            &o_config,
            false,
        )?;

        let rope = nn::RopeBuilder::new(rope_dim)
            .traditional(false)
            .base(config.rope_theta)
            .scale(1.0)
            .build()
            .map_err(LoraError::Mlx)?;

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

    /// Forward pass through attention (training / no-cache path).
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let (batch, seq_len, _) = (x.dim(0), x.dim(1), x.dim(2));

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);

        // Apply partial RoPE
        let (q_rope, q_pass) = split_rotary_last(&q, self.rope_dim);
        let (k_rope, k_pass) = split_rotary_last(&k, self.rope_dim);

        let q_rope = Module::forward(&mut self.rope, &q_rope)?;
        let k_rope = Module::forward(&mut self.rope, &k_rope)?;

        let q = pmetal_bridge::compat::ops::concatenate_axis(&[&q_rope, &q_pass], -1);
        let k = pmetal_bridge::compat::ops::concatenate_axis(&[&k_rope, &k_pass], -1);

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3]);
        let k = k.transpose_axes(&[0, 2, 1, 3]);
        let v = v.transpose_axes(&[0, 2, 1, 3]);

        let k = if self.n_kv_heads < self.n_heads {
            expand_kv_heads(&k, self.n_heads / self.n_kv_heads)?
        } else {
            k
        };
        let v = if self.n_kv_heads < self.n_heads {
            expand_kv_heads(&v, self.n_heads / self.n_kv_heads)?
        } else {
            v
        };

        let scores = q.matmul(&k.transpose_axes(&[0, 1, 3, 2]));
        let scores = scores.multiply(&Array::from_f32(self.scale));
        let scores = if let Some(m) = mask {
            scores.add(m)
        } else {
            scores
        };

        let weights = pmetal_bridge::compat::ops::softmax_axis(&scores, -1);
        let output = weights.matmul(&v);

        let output = output.transpose_axes(&[0, 2, 1, 3]);
        let output = output.reshape(&[batch, seq_len, self.n_heads * self.head_dim]);

        self.o_proj.forward(&output).map_err(LoraError::from)
    }

    /// Forward pass with KV cache (inference path).
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

        let queries = queries.reshape(&[batch, seq_len, self.n_heads, self.head_dim]);
        let keys = keys.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);
        let values = values.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);

        let queries = queries.transpose_axes(&[0, 2, 1, 3]);
        let keys = keys.transpose_axes(&[0, 2, 1, 3]);
        let values = values.transpose_axes(&[0, 2, 1, 3]);

        let (queries, keys, values) = if let Some((ref cache_ref, layer_idx)) = cache {
            let offset = cache_ref.rope_offset_for(layer_idx);

            let (q_rope, q_pass) = split_rotary_last(&queries, self.rope_dim);
            let q_rope = apply_rope(&q_rope, self.rope_dim, false, self.rope.base, 1.0, offset)?;
            let queries = pmetal_bridge::compat::ops::concatenate_axis(&[&q_rope, &q_pass], -1);

            let (k_rope, k_pass) = split_rotary_last(&keys, self.rope_dim);
            let k_rope = apply_rope(&k_rope, self.rope_dim, false, self.rope.base, 1.0, offset)?;
            let keys = pmetal_bridge::compat::ops::concatenate_axis(&[&k_rope, &k_pass], -1);

            (queries, keys, values)
        } else {
            let (q_rope, q_pass) = split_rotary_last(&queries, self.rope_dim);
            let (k_rope, k_pass) = split_rotary_last(&keys, self.rope_dim);

            let q_rope = Module::forward(&mut self.rope, &q_rope)?;
            let k_rope = Module::forward(&mut self.rope, &k_rope)?;

            let queries = pmetal_bridge::compat::ops::concatenate_axis(&[&q_rope, &q_pass], -1);
            let keys = pmetal_bridge::compat::ops::concatenate_axis(&[&k_rope, &k_pass], -1);

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

    /// Number of trainable LoRA parameters in this attention block.
    pub fn num_trainable_params(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self.v_proj.num_trainable_params()
            + self.o_proj.num_trainable_params()
    }

    /// Memory usage for this attention block: (quantized, lora, total) bytes.
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (q_q, q_l, q_t) = self.q_proj.memory_usage();
        let (k_q, k_l, k_t) = self.k_proj.memory_usage();
        let (v_q, v_l, v_t) = self.v_proj.memory_usage();
        let (o_q, o_l, o_t) = self.o_proj.memory_usage();
        (
            q_q + k_q + v_q + o_q,
            q_l + k_l + v_l + o_l,
            q_t + k_t + v_t + o_t,
        )
    }
}

fn split_rotary_last(x: &Array, rope_dim: i32) -> (Array, Array) {
    let rope_part = pmetal_bridge::compat::ops::slice_last_to(x, rope_dim);
    let pass_part = pmetal_bridge::compat::ops::slice_last_from(x, rope_dim);
    (rope_part, pass_part)
}

fn expand_kv_heads(x: &Array, repeats: i32) -> Result<Array, Exception> {
    let shape = x.shape();
    let (batch, n_kv, seq, hd) = (shape[0], shape[1], shape[2], shape[3]);
    let x = x.reshape(&[batch, n_kv, 1, seq, hd]);
    let x = pmetal_bridge::compat::ops::broadcast_to(&x, &[batch, n_kv, repeats, seq, hd]);
    Ok(x.reshape(&[batch, n_kv * repeats, seq, hd]))
}

// ─── MLP ──────────────────────────────────────────────────────────────────────

/// QLoRA-enabled MLP layer for Phi with fused gate_up projection.
#[derive(Debug)]
pub struct PhiQloraMLP {
    /// Fused gate+up projection with QLoRA.
    pub gate_up_proj: QLoraLinear,
    /// Down projection with QLoRA.
    pub down_proj: QLoraLinear,
    /// Activation type.
    pub activation: PhiActivation,
    /// Intermediate size (for splitting gate_up).
    pub intermediate_size: i32,
}

impl PhiQloraMLP {
    /// Create a new QLoRA MLP layer.
    pub fn new(config: &PhiConfig, qlora_config: &QLoraConfig) -> Result<Self, LoraError> {
        // gate_up_proj is the fused gate+up; look up via "gate_proj" in target_modules
        let proj_size = match config.hidden_act {
            PhiActivation::SwiGLU => config.intermediate_size * 2,
            _ => config.intermediate_size,
        };

        let mut gate_up_config = qlora_config.clone();
        gate_up_config.lora.r = crate::effective_rank(&qlora_config.lora, "gate_proj");
        let gate_up_proj = QLoraLinear::new(config.hidden_size, proj_size, &gate_up_config, false)?;

        let mut down_config = qlora_config.clone();
        down_config.lora.r = crate::effective_rank(&qlora_config.lora, "down_proj");
        let down_proj = QLoraLinear::new(
            config.intermediate_size,
            config.hidden_size,
            &down_config,
            false,
        )?;

        Ok(Self {
            gate_up_proj,
            down_proj,
            activation: config.hidden_act,
            intermediate_size: config.intermediate_size,
        })
    }

    /// Forward pass.
    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let hidden = self.gate_up_proj.forward(x)?;

        let activated = match self.activation {
            PhiActivation::SwiGLU => {
                let gate =
                    pmetal_bridge::compat::ops::slice_last_to(&hidden, self.intermediate_size);
                let up =
                    pmetal_bridge::compat::ops::slice_last_from(&hidden, self.intermediate_size);
                let gate_activated = nn::silu(&gate);
                gate_activated.multiply(&up)
            }
            PhiActivation::GeluApprox | PhiActivation::GeluExact => {
                (self.activation.act_fn())(&hidden)
            }
        };

        self.down_proj.forward(&activated)
    }

    /// Number of trainable LoRA parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.gate_up_proj.num_trainable_params() + self.down_proj.num_trainable_params()
    }

    /// Memory usage: (quantized, lora, total) bytes.
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (g_q, g_l, g_t) = self.gate_up_proj.memory_usage();
        let (d_q, d_l, d_t) = self.down_proj.memory_usage();
        (g_q + d_q, g_l + d_l, g_t + d_t)
    }
}

// ─── Decoder Layer ────────────────────────────────────────────────────────────

/// QLoRA-enabled Phi decoder layer.
#[derive(Debug)]
pub struct PhiQloraDecoderLayer {
    /// Self-attention layer with QLoRA.
    pub self_attn: PhiQLoraAttention,
    /// MLP layer with QLoRA.
    pub mlp: PhiQloraMLP,
    /// Input layer norm (frozen).
    pub input_layernorm: PhiQloraRmsNorm,
    /// Post-attention layer norm (frozen).
    pub post_attention_layernorm: PhiQloraRmsNorm,
}

impl PhiQloraDecoderLayer {
    /// Create a new QLoRA decoder layer.
    pub fn new(config: &PhiConfig, qlora_config: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            self_attn: PhiQLoraAttention::new(config, qlora_config)?,
            mlp: PhiQloraMLP::new(config, qlora_config)?,
            input_layernorm: PhiQloraRmsNorm::new(config.hidden_size, config.rms_norm_eps),
            post_attention_layernorm: PhiQloraRmsNorm::new(config.hidden_size, config.rms_norm_eps),
        })
    }

    /// Forward pass (training / no-cache path).
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, mask)?;
        let h = x.add(&attn_out);

        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok(h.add(&mlp_out))
    }

    /// Forward pass with KV cache.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward_with_cache(&normed, mask, cache)?;
        let h = x.add(&attn_out);

        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok(h.add(&mlp_out))
    }

    /// Number of trainable LoRA parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.self_attn.num_trainable_params() + self.mlp.num_trainable_params()
    }

    /// Memory usage: (quantized, lora, total) bytes.
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (a_q, a_l, a_t) = self.self_attn.memory_usage();
        let (m_q, m_l, m_t) = self.mlp.memory_usage();
        (a_q + m_q, a_l + m_l, a_t + m_t)
    }
}

// ─── Model (no LM head) ───────────────────────────────────────────────────────

/// QLoRA-enabled Phi model backbone (without LM head).
#[derive(Debug)]
pub struct PhiQloraModel {
    /// Architecture configuration.
    pub config: PhiConfig,
    /// QLoRA configuration.
    pub qlora_config: QLoraConfig,
    /// Token embeddings (frozen, full-precision).
    pub embed_tokens: nn::Embedding,
    /// Transformer layers with QLoRA.
    pub layers: Vec<PhiQloraDecoderLayer>,
    /// Final layer norm (frozen).
    pub norm: PhiQloraRmsNorm,
}

impl PhiQloraModel {
    /// Create a new QLoRA Phi model.
    pub fn new(config: PhiConfig, qlora_config: QLoraConfig) -> Result<Self, LoraError> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers = (0..config.num_hidden_layers)
            .map(|_| PhiQloraDecoderLayer::new(&config, &qlora_config))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = PhiQloraRmsNorm::new(config.hidden_size, config.rms_norm_eps);

        Ok(Self {
            config,
            qlora_config,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Forward pass.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;

        let mask = if mask.is_none() {
            let seq_len = input_ids.dim(1);
            Some(create_causal_mask(seq_len)?)
        } else {
            mask.cloned()
        };

        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, mask.as_ref())?;
        }

        Ok(self.norm.forward(&hidden_states)?)
    }

    /// Forward pass with KV cache.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let mut hidden_states = Module::forward(&mut self.embed_tokens, input_ids)?;

        match cache {
            Some(cache) => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    hidden_states =
                        layer.forward_with_cache(&hidden_states, mask, Some((cache, layer_idx)))?;
                }
            }
            None => {
                for layer in &mut self.layers {
                    hidden_states = layer.forward(&hidden_states, None)?;
                }
            }
        }

        Ok(self.norm.forward(&hidden_states)?)
    }

    /// Number of trainable LoRA parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_trainable_params()).sum()
    }

    /// Memory usage: (quantized, lora, total) bytes.
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (mut tq, mut tl, mut tt) = (0, 0, 0);
        for layer in &self.layers {
            let (q, l, t) = layer.memory_usage();
            tq += q;
            tl += l;
            tt += t;
        }
        (tq, tl, tt)
    }
}

// ─── ForCausalLM ─────────────────────────────────────────────────────────────

/// QLoRA-enabled Phi model with LM head.
///
/// Memory-efficient fine-tuning with 4-bit NF4 quantized base weights.
/// Phi uses a fused `gate_up_proj` (not separate gate/up), preserved here.
#[derive(Debug)]
pub struct PhiQloraForCausalLM {
    /// Base model with QLoRA.
    pub model: PhiQloraModel,
    /// LM head (frozen, full-precision).
    pub lm_head: nn::Linear,
    /// Gradient checkpointing configuration.
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl PhiQloraForCausalLM {
    /// Create from a plain `LoraConfig` (wraps into `QLoraConfig`).
    pub fn new(config: PhiConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        Self::with_qlora_config(config, QLoraConfig::from_lora(lora_config))
    }

    /// Create with an explicit `QLoraConfig`.
    pub fn with_qlora_config(
        config: PhiConfig,
        qlora_config: QLoraConfig,
    ) -> Result<Self, LoraError> {
        let lm_head = nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
            .bias(false)
            .build()
            .unwrap();
        let model = PhiQloraModel::new(config, qlora_config)?;
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
        let hidden_states = self.model.forward(input_ids, mask)?;
        Ok(Module::forward(&mut self.lm_head, &hidden_states)?)
    }

    /// Forward pass returning hidden states before lm_head (for Cut Cross-Entropy).
    pub fn forward_hidden_states(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
        self.model.forward(input_ids, mask)
    }

    /// Get the LM head weight for Cut Cross-Entropy.
    pub fn get_lm_head_weight(&self) -> Option<Array> {
        Some(self.lm_head.weight.value.clone())
    }

    /// Forward with NEFTune noise (Phi has no dedicated noised path; delegates).
    pub fn forward_noised(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        _noise_alpha: f32,
    ) -> Result<Array, LoraError> {
        self.forward(input_ids, mask)
    }

    /// Forward hidden states with explicit position IDs (position IDs ignored for Phi).
    pub fn forward_hidden_states_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        _position_ids: &Array,
    ) -> Result<Array, LoraError> {
        self.forward_hidden_states(input_ids, mask)
    }

    /// Forward pass with KV cache.
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
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        let config = KVCacheConfig::new(
            self.model.config.num_hidden_layers as usize,
            max_seq_len,
            self.model.config.num_key_value_heads as usize,
            self.model.config.head_dim() as usize,
        );
        KVCache::new(config)
    }

    /// Get all trainable LoRA parameters as a flat HashMap.
    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        let mut params = HashMap::new();
        for (i, layer) in self.model.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);
            macro_rules! add {
                ($proj:expr, $section:expr, $name:expr) => {
                    params.insert(
                        Rc::from(format!("{}.{}.{}.lora_a", prefix, $section, $name)),
                        $proj.lora_a.clone(),
                    );
                    params.insert(
                        Rc::from(format!("{}.{}.{}.lora_b", prefix, $section, $name)),
                        $proj.lora_b.clone(),
                    );
                };
            }
            add!(layer.self_attn.q_proj, "self_attn", "q_proj");
            add!(layer.self_attn.k_proj, "self_attn", "k_proj");
            add!(layer.self_attn.v_proj, "self_attn", "v_proj");
            add!(layer.self_attn.o_proj, "self_attn", "o_proj");
            add!(layer.mlp.gate_up_proj, "mlp", "gate_up_proj");
            add!(layer.mlp.down_proj, "mlp", "down_proj");
        }
        params
    }

    /// Set LoRA parameters from a flat HashMap.
    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{}", i);
            macro_rules! set {
                ($proj:expr, $section:expr, $name:expr) => {
                    let key_a: Rc<str> =
                        Rc::from(format!("{}.{}.{}.lora_a", prefix, $section, $name));
                    let key_b: Rc<str> =
                        Rc::from(format!("{}.{}.{}.lora_b", prefix, $section, $name));
                    if let Some(v) = params.get(&key_a) {
                        $proj.lora_a = v.clone();
                    }
                    if let Some(v) = params.get(&key_b) {
                        $proj.lora_b = v.clone();
                    }
                };
            }
            set!(layer.self_attn.q_proj, "self_attn", "q_proj");
            set!(layer.self_attn.k_proj, "self_attn", "k_proj");
            set!(layer.self_attn.v_proj, "self_attn", "v_proj");
            set!(layer.self_attn.o_proj, "self_attn", "o_proj");
            set!(layer.mlp.gate_up_proj, "mlp", "gate_up_proj");
            set!(layer.mlp.down_proj, "mlp", "down_proj");
        }
    }

    /// Number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    /// Memory usage: (quantized_bytes, lora_bytes, total_bytes).
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        self.model.memory_usage()
    }

    /// Memory savings ratio vs. full-precision.
    ///
    /// Returns `(quantized + lora) / fp32_equivalent` — lower is better.
    pub fn memory_savings(&self) -> f32 {
        let (quantized, lora, _) = self.memory_usage();
        let full_precision = self
            .model
            .layers
            .iter()
            .map(|l| {
                l.self_attn.q_proj.num_frozen_params() * 4
                    + l.self_attn.k_proj.num_frozen_params() * 4
                    + l.self_attn.v_proj.num_frozen_params() * 4
                    + l.self_attn.o_proj.num_frozen_params() * 4
                    + l.mlp.gate_up_proj.num_frozen_params() * 4
                    + l.mlp.down_proj.num_frozen_params() * 4
            })
            .sum::<usize>()
            + lora;
        (quantized + lora) as f32 / full_precision as f32
    }

    /// Get architecture configuration.
    pub fn config(&self) -> &PhiConfig {
        &self.model.config
    }

    /// Get QLoRA configuration.
    pub fn qlora_config(&self) -> &QLoraConfig {
        &self.model.qlora_config
    }

    /// Save LoRA adapter weights to a safetensors file or directory.
    pub fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        let params = self.lora_parameters();
        crate::save_safetensors_map(path, &params)
    }

    /// Load LoRA adapter weights from a safetensors file or directory.
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
        let loaded = crate::load_safetensors_map(&file_path)?;
        let params: HashMap<Rc<str>, Array> = loaded
            .into_iter()
            .map(|(k, v)| (Rc::from(k.as_str()), v))
            .collect();
        self.set_lora_parameters(&params);
        Ok(())
    }

    /// `merge_lora` is not supported: NF4 base weights cannot be losslessly merged.
    ///
    /// To obtain a merged checkpoint, dequantize the base first.
    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        Err(LoraError::InvalidState(
            "NF4 base cannot be losslessly merged — dequantize first".to_string(),
        ))
    }

    /// `unmerge_lora` is not supported for the same reason as `merge_lora`.
    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        Err(LoraError::InvalidState(
            "NF4 base cannot be losslessly merged — dequantize first".to_string(),
        ))
    }

    /// Load and NF4-quantize base weights from a safetensors map.
    pub fn load_and_quantize_weights(
        &mut self,
        weights: &HashMap<String, Array>,
    ) -> Result<(), LoraError> {
        // Embeddings kept in full precision
        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        let qlora_config = self.model.qlora_config.clone();

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", i);

            macro_rules! quantize {
                ($proj:expr, $key:expr) => {
                    if let Some(w) = weights.get(&format!("{}.{}", prefix, $key)) {
                        $proj = QLoraLinear::from_weight(w, None, &qlora_config)?;
                    }
                };
                ($proj:expr, $key:expr, $bias_key:expr) => {
                    let bias = weights.get(&format!("{}.{}", prefix, $bias_key));
                    if let Some(w) = weights.get(&format!("{}.{}", prefix, $key)) {
                        $proj = QLoraLinear::from_weight(w, bias, &qlora_config)?;
                    }
                };
            }

            quantize!(
                layer.self_attn.q_proj,
                "self_attn.q_proj.weight",
                "self_attn.q_proj.bias"
            );
            quantize!(
                layer.self_attn.k_proj,
                "self_attn.k_proj.weight",
                "self_attn.k_proj.bias"
            );
            quantize!(
                layer.self_attn.v_proj,
                "self_attn.v_proj.weight",
                "self_attn.v_proj.bias"
            );
            quantize!(layer.self_attn.o_proj, "self_attn.o_proj.weight");
            quantize!(layer.mlp.gate_up_proj, "mlp.gate_up_proj.weight");
            quantize!(layer.mlp.down_proj, "mlp.down_proj.weight");

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
        if let Some(w) = weights.get("lm_head.weight") {
            self.lm_head.weight = Param::new(w.clone());
        }

        Ok(())
    }

    /// Load f32 base weights from a model directory, NF4-quantize, and freeze them.
    pub fn load_and_quantize_from_dir(
        &mut self,
        model_dir: impl AsRef<std::path::Path>,
    ) -> Result<(), LoraError> {
        let model_dir = model_dir.as_ref();

        let single = model_dir.join("model.safetensors");
        if single.exists() {
            let weights = crate::sanitize_loaded_weights(crate::load_safetensors_map(&single)?)?;
            return self.load_and_quantize_weights(&weights);
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
            let shard_weights =
                crate::sanitize_loaded_weights(crate::load_safetensors_map(&shard_path)?)?;
            all_weights.extend(shard_weights);
        }

        self.load_and_quantize_weights(&all_weights)
    }

    /// Reload a merged (full-precision) checkpoint, replacing quantized base weights.
    pub fn load_base_weights_from_dir(
        &mut self,
        model_dir: impl AsRef<std::path::Path>,
    ) -> Result<(), LoraError> {
        // Re-quantize from the merged checkpoint.
        self.load_and_quantize_from_dir(model_dir)
    }

    /// Evaluate all LoRA parameters.
    pub fn eval_lora_params(&mut self) {
        for layer in &mut self.model.layers {
            layer.self_attn.q_proj.lora_a.eval();
            layer.self_attn.q_proj.lora_b.eval();
            layer.self_attn.k_proj.lora_a.eval();
            layer.self_attn.k_proj.lora_b.eval();
            layer.self_attn.v_proj.lora_a.eval();
            layer.self_attn.v_proj.lora_b.eval();
            layer.self_attn.o_proj.lora_a.eval();
            layer.self_attn.o_proj.lora_b.eval();
            layer.mlp.gate_up_proj.lora_a.eval();
            layer.mlp.gate_up_proj.lora_b.eval();
            layer.mlp.down_proj.lora_a.eval();
            layer.mlp.down_proj.lora_b.eval();
        }
    }
}

// ─── ModuleParameters (manual — LoraLinear / QLoraLinear don't derive it) ────

impl ModuleParameters for PhiQloraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.model.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();

        for (i, layer) in self.model.layers.iter().enumerate() {
            let prefix: Rc<str> = Rc::from(format!("layers.{}", i));
            let mut layer_params: HashMap<Rc<str>, NestedValue<&Array>> = HashMap::new();

            let mut attn_params: HashMap<Rc<str>, NestedValue<&Array>> = HashMap::new();

            macro_rules! attn_proj {
                ($proj:expr, $name:expr) => {{
                    let mut p: HashMap<Rc<str>, NestedValue<&Array>> = HashMap::new();
                    p.insert(Rc::from("lora_a"), NestedValue::Value(&$proj.lora_a));
                    p.insert(Rc::from("lora_b"), NestedValue::Value(&$proj.lora_b));
                    attn_params.insert(Rc::from($name), NestedValue::Map(p));
                }};
            }
            attn_proj!(layer.self_attn.q_proj, "q_proj");
            attn_proj!(layer.self_attn.k_proj, "k_proj");
            attn_proj!(layer.self_attn.v_proj, "v_proj");
            attn_proj!(layer.self_attn.o_proj, "o_proj");
            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            let mut mlp_params: HashMap<Rc<str>, NestedValue<&Array>> = HashMap::new();
            macro_rules! mlp_proj {
                ($proj:expr, $name:expr) => {{
                    let mut p: HashMap<Rc<str>, NestedValue<&Array>> = HashMap::new();
                    p.insert(Rc::from("lora_a"), NestedValue::Value(&$proj.lora_a));
                    p.insert(Rc::from("lora_b"), NestedValue::Value(&$proj.lora_b));
                    mlp_params.insert(Rc::from($name), NestedValue::Map(p));
                }};
            }
            mlp_proj!(layer.mlp.gate_up_proj, "gate_up_proj");
            mlp_proj!(layer.mlp.down_proj, "down_proj");
            layer_params.insert(Rc::from("mlp"), NestedValue::Map(mlp_params));

            params.insert(prefix, NestedValue::Map(layer_params));
        }

        params
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut params = ModuleParamMut::new();

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix: Rc<str> = Rc::from(format!("layers.{}", i));
            let mut layer_params: HashMap<Rc<str>, NestedValue<&mut Array>> = HashMap::new();

            let mut attn_params: HashMap<Rc<str>, NestedValue<&mut Array>> = HashMap::new();

            macro_rules! attn_proj_mut {
                ($proj:expr, $name:expr) => {{
                    let mut p: HashMap<Rc<str>, NestedValue<&mut Array>> = HashMap::new();
                    p.insert(Rc::from("lora_a"), NestedValue::Value(&mut $proj.lora_a));
                    p.insert(Rc::from("lora_b"), NestedValue::Value(&mut $proj.lora_b));
                    attn_params.insert(Rc::from($name), NestedValue::Map(p));
                }};
            }
            attn_proj_mut!(layer.self_attn.q_proj, "q_proj");
            attn_proj_mut!(layer.self_attn.k_proj, "k_proj");
            attn_proj_mut!(layer.self_attn.v_proj, "v_proj");
            attn_proj_mut!(layer.self_attn.o_proj, "o_proj");
            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            let mut mlp_params: HashMap<Rc<str>, NestedValue<&mut Array>> = HashMap::new();
            macro_rules! mlp_proj_mut {
                ($proj:expr, $name:expr) => {{
                    let mut p: HashMap<Rc<str>, NestedValue<&mut Array>> = HashMap::new();
                    p.insert(Rc::from("lora_a"), NestedValue::Value(&mut $proj.lora_a));
                    p.insert(Rc::from("lora_b"), NestedValue::Value(&mut $proj.lora_b));
                    mlp_params.insert(Rc::from($name), NestedValue::Map(p));
                }};
            }
            mlp_proj_mut!(layer.mlp.gate_up_proj, "gate_up_proj");
            mlp_proj_mut!(layer.mlp.down_proj, "down_proj");
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

// ─── TrainableModel (manual — QLoRA files don't use impl_trainable_model!) ───

impl crate::TrainableModel for PhiQloraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        PhiQloraForCausalLM::forward(self, input_ids, mask)
    }

    fn num_trainable_params(&self) -> usize {
        PhiQloraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        PhiQloraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        PhiQloraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        PhiQloraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        PhiQloraForCausalLM::load_lora_weights(self, path)
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        PhiQloraForCausalLM::enable_gradient_checkpointing(self, layers_per_block)
    }

    fn disable_gradient_checkpointing(&mut self) {
        PhiQloraForCausalLM::disable_gradient_checkpointing(self)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        true
    }

    fn supports_kv_cache(&self) -> bool {
        true
    }

    fn create_cache(&self, max_seq_len: usize) -> Option<KVCache> {
        Some(PhiQloraForCausalLM::create_cache(self, max_seq_len))
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        PhiQloraForCausalLM::forward_with_cache(self, input_ids, mask, cache)
    }

    fn forward_noised(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        noise_alpha: f32,
    ) -> Result<Array, LoraError> {
        PhiQloraForCausalLM::forward_noised(self, input_ids, mask, noise_alpha)
    }

    fn forward_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Option<Result<Array, LoraError>> {
        Some(PhiQloraForCausalLM::forward_hidden_states(
            self, input_ids, mask,
        ))
    }

    fn forward_hidden_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Option<Result<Array, LoraError>> {
        Some(PhiQloraForCausalLM::forward_hidden_states_with_positions(
            self,
            input_ids,
            mask,
            position_ids,
        ))
    }

    fn lm_head_weight(&self) -> Option<Array> {
        PhiQloraForCausalLM::get_lm_head_weight(self)
    }
}

// ─── Causal mask helper ───────────────────────────────────────────────────────

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_models::architectures::phi::LayerNormType;

    fn small_config() -> PhiConfig {
        PhiConfig {
            model_type: "phi".to_string(),
            vocab_size: 512,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            max_position_embeddings: 128,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            rms_norm_eps: 1e-5,
            qkv_bias: false,
            hidden_act: PhiActivation::SwiGLU,
            sliding_window: None,
            layer_norm_type: LayerNormType::RmsNorm,
            original_max_position_embeddings: None,
            rope_scaling: None,
            tie_word_embeddings: true,
        }
    }

    fn small_qlora_config() -> QLoraConfig {
        QLoraConfig {
            lora: LoraConfig {
                r: 4,
                alpha: 8.0,
                dropout: 0.0,
                use_rslora: false,
                target_modules: vec![
                    "q_proj".to_string(),
                    "k_proj".to_string(),
                    "v_proj".to_string(),
                    "o_proj".to_string(),
                    "gate_proj".to_string(),
                    "down_proj".to_string(),
                ],
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_phi_qlora_construction() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let model = PhiQloraForCausalLM::with_qlora_config(config, qlora_config);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
    }

    #[test]
    fn test_phi_qlora_trainable_params_nonzero() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let model = PhiQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        assert!(
            model.num_trainable_params() > 0,
            "expected trainable params > 0"
        );

        // Base (frozen) params should be inaccessible via lora_parameters()
        let lora_params = model.lora_parameters();
        assert!(
            !lora_params.is_empty(),
            "lora_parameters should not be empty"
        );

        // Every value in lora_parameters is a LoRA adapter (not a frozen quantized weight)
        for (key, arr) in &lora_params {
            assert!(
                key.contains("lora_a") || key.contains("lora_b"),
                "unexpected key in lora_parameters: {}",
                key
            );
            let _ = arr; // shapes vary by rank / layer dims
        }
    }

    #[test]
    fn test_phi_qlora_forward() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let mut model =
            PhiQloraForCausalLM::with_qlora_config(config.clone(), qlora_config).unwrap();

        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3, 4]).reshape(&[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();
        assert_eq!(logits.shape(), &[1, 4, config.vocab_size]);
    }

    #[test]
    fn test_phi_qlora_supports_kv_cache() {
        use crate::TrainableModel;

        let config = small_config();
        let qlora_config = small_qlora_config();
        let mut model =
            PhiQloraForCausalLM::with_qlora_config(config.clone(), qlora_config).unwrap();

        assert!(model.supports_kv_cache());

        let mut cache = PhiQloraForCausalLM::create_cache(&model, 128);
        assert_eq!(cache.rope_offset_for(0), 0);

        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3, 4]).reshape(&[1, 4]);
        let logits = model
            .forward_with_cache(&input_ids, None, Some(&mut cache))
            .unwrap();
        assert_eq!(logits.shape(), &[1, 4, config.vocab_size]);
        assert_eq!(cache.rope_offset_for(0), 4);
    }

    #[test]
    fn test_phi_qlora_memory_savings() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let model = PhiQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        let savings = model.memory_savings();
        assert!(
            savings < 0.5,
            "expected significant memory savings, got {}",
            savings
        );
    }

    #[test]
    fn test_phi_qlora_merge_returns_error() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let mut model = PhiQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        assert!(model.merge_lora().is_err());
        assert!(model.unmerge_lora().is_err());
    }

    #[test]
    fn test_phi_qlora_set_parameters_roundtrip() {
        let config = small_config();
        let qlora_config = small_qlora_config();
        let mut model = PhiQloraForCausalLM::with_qlora_config(config, qlora_config).unwrap();

        let params = model.lora_parameters();
        // Overwrite with same params — should succeed without panic
        model.set_lora_parameters(&params);
        let params2 = model.lora_parameters();
        assert_eq!(params.len(), params2.len());
    }
}
