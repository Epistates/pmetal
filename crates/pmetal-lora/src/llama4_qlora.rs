//! QLoRA-enabled Llama 4 model architecture.
//!
//! Implements Llama 4 with QLoRA (Quantized LoRA) for memory-efficient fine-tuning.
//! Base weights are stored in 4-bit NF4 format; LoRA adapters remain in full
//! precision.
//!
//! ## Llama 4 specifics
//!
//! - **Attention**: QLoRA on q/k/v/o.  QK norms, temperature scalars, and MoD
//!   routers stay frozen in full precision.
//! - **MoE layers**: shared expert gets QLoRA on gate/up/down.  Routed experts
//!   stay frozen and quantized.
//! - **Dense layers**: QLoRA on gate/up/down of the `mlp` field.
//! - **iRoPE**: per-layer `uses_rope` flag is preserved from the LoRA model.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, nn,
    ops,
};
use pmetal_core::LoraConfig;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::apply_rope,
};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};
use pmetal_models::architectures::llama4::{
    Llama4Expert, Llama4ModRouter, Llama4Router, Llama4TextConfig,
};

use crate::{
    LoraError, QLoraConfig, QLoraLinear,
    llama4_lora::{
        Llama4LoraAttention, Llama4LoraDecoderLayer, Llama4LoraForCausalLM,
        Llama4LoraMLP, Llama4LoraMoE, Llama4LoraModel, Llama4LoraSharedExpert,
    },
    qlora::quantize_lora_layer,
};

// =============================================================================
// Helper: quantize a LinearAdapter base weight -> QLoraLinear
// =============================================================================

/// Quantize a `LinearAdapter` (LoRA or DoRA) into a `QLoraLinear` by extracting
/// the base weight and re-quantizing.  LoRA A/B matrices are re-initialized to
/// zero/random (as in all other QLoRA implementations — the LoRA adapters are
/// re-learned from scratch during QLoRA fine-tuning).
fn quantize_adapter(
    adapter: &crate::LinearAdapter,
    qcfg: &QLoraConfig,
) -> Result<QLoraLinear, LoraError> {
    QLoraLinear::from_weight(adapter.weight(), None, qcfg)
}

// =============================================================================
// Attention
// =============================================================================

/// QLoRA-enabled Llama 4 attention layer.
///
/// Base weights (q/k/v/o) are quantized to NF4; LoRA adapters remain in full
/// precision.  QK norms and temperature-tuning scalars are carried over from the
/// LoRA model unchanged (they are scale parameters, not projection matrices).
#[derive(Debug)]
pub struct Llama4QloraAttention {
    /// Layer index (used for iRoPE determination).
    pub layer_idx: usize,
    /// Whether this layer uses RoPE (true) or NoPE (false).
    pub uses_rope: bool,
    /// Number of query heads.
    pub n_heads: i32,
    /// Number of KV heads.
    pub n_kv_heads: i32,
    /// Per-head dimension.
    pub head_dim: i32,
    /// Attention scale = 1/sqrt(head_dim).
    pub scale: f32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// RoPE scale factor (default 1.0).
    pub rope_scale: f32,
    /// Enable temperature tuning for NoPE long-context layers.
    pub attn_temperature_tuning: bool,
    /// Floor scale for temperature computation.
    pub floor_scale: f32,
    /// Attention scale multiplier for temperature tuning.
    pub attn_scale: f32,

    /// Query projection — quantized base + LoRA.
    pub q_proj: QLoraLinear,
    /// Key projection — quantized base + LoRA.
    pub k_proj: QLoraLinear,
    /// Value projection — quantized base + LoRA.
    pub v_proj: QLoraLinear,
    /// Output projection — quantized base + LoRA.
    pub o_proj: QLoraLinear,

    /// Optional per-head Q normalisation (frozen, full precision).
    pub q_norm: Option<nn::RmsNorm>,
    /// Optional per-head K normalisation (frozen, full precision).
    pub k_norm: Option<nn::RmsNorm>,
}

impl Llama4QloraAttention {
    fn from_lora(attn: Llama4LoraAttention, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            layer_idx: attn.layer_idx,
            uses_rope: attn.uses_rope,
            n_heads: attn.n_heads,
            n_kv_heads: attn.n_kv_heads,
            head_dim: attn.head_dim,
            scale: attn.scale,
            rope_theta: attn.rope_theta,
            rope_scale: attn.rope_scale,
            attn_temperature_tuning: attn.attn_temperature_tuning,
            floor_scale: attn.floor_scale,
            attn_scale: attn.attn_scale,
            q_proj: quantize_adapter(&attn.q_proj, qcfg)?,
            k_proj: quantize_adapter(&attn.k_proj, qcfg)?,
            v_proj: quantize_adapter(&attn.v_proj, qcfg)?,
            o_proj: quantize_adapter(&attn.o_proj, qcfg)?,
            q_norm: attn.q_norm,
            k_norm: attn.k_norm,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        let mut q = self.q_proj.forward(x)?;
        let mut k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // [B, T, n*D] -> [B, T, n, D]
        q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim]);
        k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);

        // QK normalisation (before RoPE).
        if let (Some(qn), Some(kn)) = (&mut self.q_norm, &mut self.k_norm) {
            q = Module::forward(qn, &q)?;
            k = Module::forward(kn, &k)?;
        }

        // RoPE: need [B, H, T, D] layout.
        if self.uses_rope {
            let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
            let q_t = q.transpose_axes(&[0, 2, 1, 3]);
            let k_t = k.transpose_axes(&[0, 2, 1, 3]);
            let q_r = apply_rope(&q_t, self.head_dim, false, self.rope_theta, self.rope_scale, offset)
                .map_err(LoraError::Mlx)?;
            let k_r = apply_rope(&k_t, self.head_dim, false, self.rope_theta, self.rope_scale, offset)
                .map_err(LoraError::Mlx)?;
            q = q_r.transpose_axes(&[0, 2, 1, 3]);
            k = k_r.transpose_axes(&[0, 2, 1, 3]);
        }

        // [B, T, H, D] -> [B, H, T, D]
        let q = q.transpose_axes(&[0, 2, 1, 3]);
        let k = k.transpose_axes(&[0, 2, 1, 3]);
        let v = v.transpose_axes(&[0, 2, 1, 3]);

        // Temperature scaling for NoPE long-context layers.
        let q = if !self.uses_rope && self.attn_temperature_tuning {
            apply_temperature_scaling(q, seq_len, self.floor_scale, self.attn_scale)
        } else {
            q
        };

        // Update KV cache.
        let (keys, values) = if let Some((cache_ref, layer_idx)) = cache {
            cache_ref
                .update_and_fetch(layer_idx, &k, &v)
                .map_err(LoraError::Mlx)?
        } else {
            (k, v)
        };

        // Fused SDPA (handles GQA internally).
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(if mask.is_some() {
                AttentionMaskType::None
            } else {
                AttentionMaskType::Causal
            });

        let output = fused_sdpa(&q, &keys, &values, &attn_config, mask)
            .map_err(LoraError::Mlx)?;

        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);
        self.o_proj.forward(&output)
    }

    fn num_trainable_params(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self.v_proj.num_trainable_params()
            + self.o_proj.num_trainable_params()
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        let (qq, ql, qt) = self.q_proj.memory_usage();
        let (kq, kl, kt) = self.k_proj.memory_usage();
        let (vq, vl, vt) = self.v_proj.memory_usage();
        let (oq, ol, ot) = self.o_proj.memory_usage();
        (qq + kq + vq + oq, ql + kl + vl + ol, qt + kt + vt + ot)
    }
}

// =============================================================================
// Shared expert (MoE layers only)
// =============================================================================

/// QLoRA-enabled shared expert for Llama 4 MoE layers.
#[derive(Debug)]
pub struct Llama4QloraSharedExpert {
    /// Gate projection — quantized + LoRA.
    pub gate_proj: QLoraLinear,
    /// Up projection — quantized + LoRA.
    pub up_proj: QLoraLinear,
    /// Down projection — quantized + LoRA.
    pub down_proj: QLoraLinear,
}

impl Llama4QloraSharedExpert {
    fn from_lora(se: Llama4LoraSharedExpert, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            gate_proj: quantize_lora_layer(&se.gate_proj, qcfg)?,
            up_proj: quantize_lora_layer(&se.up_proj, qcfg)?,
            down_proj: quantize_lora_layer(&se.down_proj, qcfg)?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let gate = nn::silu(&self.gate_proj.forward(x)?);
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.multiply(&up))
    }

    fn num_trainable_params(&self) -> usize {
        self.gate_proj.num_trainable_params()
            + self.up_proj.num_trainable_params()
            + self.down_proj.num_trainable_params()
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        let (gq, gl, gt) = self.gate_proj.memory_usage();
        let (uq, ul, ut) = self.up_proj.memory_usage();
        let (dq, dl, dt) = self.down_proj.memory_usage();
        (gq + uq + dq, gl + ul + dl, gt + ut + dt)
    }
}

// =============================================================================
// MoE block
// =============================================================================

/// QLoRA-enabled Llama 4 MoE block.
///
/// Router and routed experts are frozen and quantized.  Only `shared_expert`
/// carries LoRA adapters.
#[derive(Debug)]
pub struct Llama4QloraMoE {
    /// Frozen router.
    pub router: Llama4Router,
    /// Frozen + quantized routed experts.
    pub experts: Vec<Llama4Expert>,
    /// QLoRA-enabled shared expert.
    pub shared_expert: Llama4QloraSharedExpert,
    /// Top-k experts selected per token.
    pub num_experts_per_tok: i32,
}

impl Llama4QloraMoE {
    fn from_lora(moe: Llama4LoraMoE, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            router: moe.router,
            experts: moe.experts,
            shared_expert: Llama4QloraSharedExpert::from_lora(moe.shared_expert, qcfg)?,
            num_experts_per_tok: moe.num_experts_per_tok,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let shape = x.shape().to_vec();
        let hidden_size = *shape.last().unwrap();
        let total_tokens: i32 = shape.iter().take(shape.len() - 1).product();
        let flat_x = x.reshape(&[total_tokens, hidden_size]);

        // Route (frozen).
        let (expert_indices, expert_weights, _router_logits) =
            self.router.forward(&flat_x).map_err(LoraError::Mlx)?;

        // Shared expert (QLoRA-adapted).
        let shared_out = self.shared_expert.forward(&flat_x)?;

        // Routed experts — sparse dispatch (frozen).
        let expert_indices = expert_indices.as_type::<i32>();
        expert_indices.eval();
        expert_weights.eval();

        let top_k = self.num_experts_per_tok as usize;
        let n_tokens = total_tokens as usize;
        let expert_ids: Vec<i32> = expert_indices
            .as_slice::<u32>()
            .iter()
            .map(|&v| v as i32)
            .collect();
        let routing_weights: Vec<f32> = expert_weights.as_slice().to_vec();

        let mut expert_assignments: Vec<Vec<(usize, f32)>> =
            vec![Vec::new(); self.experts.len()];
        for token_idx in 0..n_tokens {
            for slot in 0..top_k {
                let flat_idx = token_idx * top_k + slot;
                let eid = expert_ids[flat_idx] as usize;
                let w = routing_weights[flat_idx];
                if eid < self.experts.len() {
                    expert_assignments[eid].push((token_idx, w));
                }
            }
        }

        let input_dtype = flat_x.dtype();
        let mut combined_out =
            ops::zeros_dtype(&[total_tokens, hidden_size], input_dtype);

        for (expert_idx, assignments) in expert_assignments.iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }
            let token_indices: Vec<i32> =
                assignments.iter().map(|&(idx, _)| idx as i32).collect();
            let weights: Vec<f32> = assignments.iter().map(|&(_, w)| w).collect();

            let idx_array = Array::from_slice(&token_indices, &[token_indices.len() as i32]);
            let weight_array = Array::from_slice(&weights, &[weights.len() as i32, 1]);

            let expert_input = flat_x.take_axis(&idx_array, 0);
            let expert_out = self.experts[expert_idx]
                .forward(&expert_input)
                .map_err(LoraError::Mlx)?;
            let weighted_out = expert_out.multiply(&weight_array);

            let updates = weighted_out.reshape(&[token_indices.len() as i32, 1, hidden_size]);
            combined_out = pmetal_bridge::compat::indexing::scatter_add_single(
                &combined_out,
                &idx_array,
                &updates,
                0,
            );
        }

        let output = shared_out.add(&combined_out);
        Ok(output.reshape(&shape))
    }

    fn num_trainable_params(&self) -> usize {
        self.shared_expert.num_trainable_params()
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        self.shared_expert.memory_usage()
    }
}

// =============================================================================
// Dense MLP (non-MoE layers)
// =============================================================================

/// QLoRA-enabled dense MLP for Llama 4 non-MoE layers.
#[derive(Debug)]
pub struct Llama4QloraMLP {
    /// Gate projection — quantized + LoRA.
    pub gate_proj: QLoraLinear,
    /// Up projection — quantized + LoRA.
    pub up_proj: QLoraLinear,
    /// Down projection — quantized + LoRA.
    pub down_proj: QLoraLinear,
}

impl Llama4QloraMLP {
    fn from_lora(mlp: Llama4LoraMLP, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            gate_proj: quantize_lora_layer(&mlp.gate_proj, qcfg)?,
            up_proj: quantize_lora_layer(&mlp.up_proj, qcfg)?,
            down_proj: quantize_lora_layer(&mlp.down_proj, qcfg)?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let gate = nn::silu(&self.gate_proj.forward(x)?);
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.multiply(&up))
    }

    fn num_trainable_params(&self) -> usize {
        self.gate_proj.num_trainable_params()
            + self.up_proj.num_trainable_params()
            + self.down_proj.num_trainable_params()
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        let (gq, gl, gt) = self.gate_proj.memory_usage();
        let (uq, ul, ut) = self.up_proj.memory_usage();
        let (dq, dl, dt) = self.down_proj.memory_usage();
        (gq + uq + dq, gl + ul + dl, gt + ut + dt)
    }
}

// =============================================================================
// Decoder layer
// =============================================================================

/// QLoRA-enabled Llama 4 decoder layer.
#[derive(Debug)]
pub struct Llama4QloraDecoderLayer {
    /// Layer index.
    pub layer_idx: usize,
    /// Whether this layer's FFN is an MoE block.
    pub is_moe: bool,

    /// Attention block with QLoRA.
    pub self_attn: Llama4QloraAttention,
    /// Dense MLP with QLoRA (Some iff not MoE).
    pub mlp: Option<Llama4QloraMLP>,
    /// MoE block with QLoRA on shared expert (Some iff MoE).
    pub moe: Option<Llama4QloraMoE>,
    /// Input layernorm (frozen, full precision).
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layernorm (frozen, full precision).
    pub post_attention_layernorm: nn::RmsNorm,
    /// MoD router (frozen, full precision).
    pub mod_router: Option<Llama4ModRouter>,
}

impl Llama4QloraDecoderLayer {
    fn from_lora(layer: Llama4LoraDecoderLayer, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        let is_moe = layer.is_moe;
        let (mlp, moe) = if is_moe {
            (
                None,
                Some(Llama4QloraMoE::from_lora(layer.moe.unwrap(), qcfg)?),
            )
        } else {
            (
                Some(Llama4QloraMLP::from_lora(layer.mlp.unwrap(), qcfg)?),
                None,
            )
        };

        Ok(Self {
            layer_idx: layer.layer_idx,
            is_moe,
            self_attn: Llama4QloraAttention::from_lora(layer.self_attn, qcfg)?,
            mlp,
            moe,
            input_layernorm: layer.input_layernorm,
            post_attention_layernorm: layer.post_attention_layernorm,
            mod_router: layer.mod_router,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let normed = Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self.self_attn.forward(&normed, mask, cache)?;
        let h = x.add(&attn_out);

        let normed = Module::forward(&mut self.post_attention_layernorm, &h)?;
        let ffn_out = if self.is_moe {
            self.moe.as_mut().unwrap().forward(&normed)?
        } else {
            self.mlp.as_mut().unwrap().forward(&normed)?
        };
        Ok(h.add(&ffn_out))
    }

    fn num_trainable_params(&self) -> usize {
        let ffn_params = if self.is_moe {
            self.moe.as_ref().map_or(0, |m| m.num_trainable_params())
        } else {
            self.mlp.as_ref().map_or(0, |m| m.num_trainable_params())
        };
        self.self_attn.num_trainable_params() + ffn_params
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        let (aq, al, at) = self.self_attn.memory_usage();
        let (mq, ml, mt) = if self.is_moe {
            self.moe.as_ref().map_or((0, 0, 0), |m| m.memory_usage())
        } else {
            self.mlp.as_ref().map_or((0, 0, 0), |m| m.memory_usage())
        };
        (aq + mq, al + ml, at + mt)
    }
}

// =============================================================================
// Inner model
// =============================================================================

/// QLoRA-enabled Llama 4 text model (without LM head).
#[derive(Debug)]
pub struct Llama4QloraModel {
    /// Architecture configuration.
    pub config: Llama4TextConfig,
    /// QLoRA configuration.
    pub qlora_config: QLoraConfig,
    /// Token embeddings (frozen, full precision).
    pub embed_tokens: nn::Embedding,
    /// Decoder layers with QLoRA.
    pub layers: Vec<Llama4QloraDecoderLayer>,
    /// Final RMSNorm (frozen, full precision).
    pub norm: nn::RmsNorm,
}

impl Llama4QloraModel {
    fn from_lora(model: Llama4LoraModel, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            config: model.config,
            qlora_config: qcfg.clone(),
            embed_tokens: model.embed_tokens,
            layers: model
                .layers
                .into_iter()
                .map(|l| Llama4QloraDecoderLayer::from_lora(l, qcfg))
                .collect::<Result<Vec<_>, _>>()?,
            norm: model.norm,
        })
    }

    fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let mut h = self.embed_tokens.forward(input_ids);
        match cache {
            Some(cache_ref) => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    h = layer.forward(&h, mask, Some((cache_ref, layer_idx)))?;
                }
            }
            None => {
                for layer in &mut self.layers {
                    h = layer.forward(&h, mask, None)?;
                }
            }
        }
        Ok(Module::forward(&mut self.norm, &h)?)
    }

    fn num_trainable_params(&self) -> usize {
        self.layers.iter().map(Llama4QloraDecoderLayer::num_trainable_params).sum()
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        self.layers.iter().fold((0, 0, 0), |acc, l| {
            let (q, lo, t) = l.memory_usage();
            (acc.0 + q, acc.1 + lo, acc.2 + t)
        })
    }
}

// =============================================================================
// ForCausalLM wrapper
// =============================================================================

/// QLoRA-enabled Llama 4 model with LM head.
///
/// Typical memory usage for a 17B Scout model: ~10GB (vs 34GB for bfloat16).
#[derive(Debug)]
pub struct Llama4QloraForCausalLM {
    /// Base model with QLoRA.
    pub model: Llama4QloraModel,
    /// LM head (frozen, full precision).
    pub lm_head: nn::Linear,
    /// Gradient checkpointing configuration (interface parity only).
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl Llama4QloraForCausalLM {
    /// Create a new QLoRA Llama 4 model from a `LoraConfig`.
    pub fn new(config: Llama4TextConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        Self::with_qlora_config(config, QLoraConfig::from_lora(lora_config))
    }

    /// Create with an explicit `QLoraConfig`.
    pub fn with_qlora_config(
        config: Llama4TextConfig,
        qlora_config: QLoraConfig,
    ) -> Result<Self, LoraError> {
        let lora = Llama4LoraForCausalLM::new(config, qlora_config.lora.clone())?;
        Self::from_lora(lora, qlora_config)
    }

    fn from_lora(lora: Llama4LoraForCausalLM, qcfg: QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            model: Llama4QloraModel::from_lora(lora.model, &qcfg)?,
            lm_head: lora.lm_head,
            checkpoint_config: None,
        })
    }

    /// Enable gradient checkpointing (interface parity with LoRA model).
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
        self.forward_with_cache(input_ids, mask, None)
    }

    /// Forward with optional KV cache.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let hidden = self.model.forward(input_ids, mask, cache)?;
        Module::forward(&mut self.lm_head, &hidden).map_err(LoraError::Mlx)
    }

    /// Hidden states before LM head (for Cut Cross-Entropy).
    pub fn forward_hidden_states(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
        self.model.forward(input_ids, mask, None)
    }

    /// Hidden states with position IDs (packed sequence — delegates to standard forward).
    pub fn forward_hidden_states_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        _position_ids: &Array,
    ) -> Result<Array, LoraError> {
        self.forward_hidden_states(input_ids, mask)
    }

    /// NEFTune forward — adds uniform embedding noise.
    pub fn forward_noised(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        noise_alpha: f32,
    ) -> Result<Array, LoraError> {
        let mut h = self.model.embed_tokens.forward(input_ids);

        let seq_len = input_ids.dim(1) as f32;
        let embed_dim = h.dim(2) as f32;
        let mag = noise_alpha / (seq_len * embed_dim).sqrt();
        let noise = pmetal_bridge::compat::random::uniform_range(
            -mag,
            mag,
            h.shape(),
            pmetal_bridge::compat::Dtype::Float32,
        );
        h = h.add(&noise);

        // Run layers (no cache).
        for layer in &mut self.model.layers {
            h = layer.forward(&h, mask, None)?;
        }
        h = Module::forward(&mut self.model.norm, &h)?;
        Module::forward(&mut self.lm_head, &h).map_err(LoraError::Mlx)
    }

    /// Create a KV cache for this model.
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        KVCache::new(KVCacheConfig::new(
            self.model.config.num_hidden_layers as usize,
            max_seq_len,
            self.model.config.num_key_value_heads as usize,
            self.model.config.head_dim as usize,
        ))
    }

    /// LM head weight for Cut Cross-Entropy.
    pub fn get_lm_head_weight(&self) -> Option<Array> {
        Some(self.lm_head.weight.value.clone())
    }

    /// Total trainable parameter count (LoRA adapters only).
    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    /// Memory usage breakdown: (quantized_bytes, lora_bytes, total_bytes).
    pub fn memory_usage(&self) -> (usize, usize, usize) {
        self.model.memory_usage()
    }

    /// Memory savings ratio vs. full bfloat16.
    pub fn memory_savings(&self) -> f32 {
        let (quantized, lora, _) = self.memory_usage();
        let full_precision = self
            .model
            .layers
            .iter()
            .map(|layer| {
                let attn = layer.self_attn.q_proj.num_frozen_params() * 4
                    + layer.self_attn.k_proj.num_frozen_params() * 4
                    + layer.self_attn.v_proj.num_frozen_params() * 4
                    + layer.self_attn.o_proj.num_frozen_params() * 4;
                let ffn = if layer.is_moe {
                    let se = layer.moe.as_ref().unwrap();
                    se.shared_expert.gate_proj.num_frozen_params() * 4
                        + se.shared_expert.up_proj.num_frozen_params() * 4
                        + se.shared_expert.down_proj.num_frozen_params() * 4
                } else {
                    let mlp = layer.mlp.as_ref().unwrap();
                    mlp.gate_proj.num_frozen_params() * 4
                        + mlp.up_proj.num_frozen_params() * 4
                        + mlp.down_proj.num_frozen_params() * 4
                };
                attn + ffn
            })
            .sum::<usize>()
            + lora;
        (quantized + lora) as f32 / full_precision.max(1) as f32
    }

    // -------------------------------------------------------------------------
    // LoRA parameter management
    // -------------------------------------------------------------------------

    /// All LoRA adapter parameters as a flat HashMap.
    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        let mut params = HashMap::new();
        for (i, layer) in self.model.layers.iter().enumerate() {
            let attn_prefix = format!("layers.{i}.self_attn");
            for (name, proj) in [
                ("q_proj", &layer.self_attn.q_proj),
                ("k_proj", &layer.self_attn.k_proj),
                ("v_proj", &layer.self_attn.v_proj),
                ("o_proj", &layer.self_attn.o_proj),
            ] {
                params.insert(
                    Rc::from(format!("{attn_prefix}.{name}.lora_a")),
                    proj.lora_a.clone(),
                );
                params.insert(
                    Rc::from(format!("{attn_prefix}.{name}.lora_b")),
                    proj.lora_b.clone(),
                );
            }
            let mlp_prefix = format!("layers.{i}.mlp");
            if layer.is_moe {
                let se = layer.moe.as_ref().unwrap();
                for (name, proj) in [
                    ("gate_proj", &se.shared_expert.gate_proj),
                    ("up_proj", &se.shared_expert.up_proj),
                    ("down_proj", &se.shared_expert.down_proj),
                ] {
                    params.insert(
                        Rc::from(format!("{mlp_prefix}.{name}.lora_a")),
                        proj.lora_a.clone(),
                    );
                    params.insert(
                        Rc::from(format!("{mlp_prefix}.{name}.lora_b")),
                        proj.lora_b.clone(),
                    );
                }
            } else {
                let mlp = layer.mlp.as_ref().unwrap();
                for (name, proj) in [
                    ("gate_proj", &mlp.gate_proj),
                    ("up_proj", &mlp.up_proj),
                    ("down_proj", &mlp.down_proj),
                ] {
                    params.insert(
                        Rc::from(format!("{mlp_prefix}.{name}.lora_a")),
                        proj.lora_a.clone(),
                    );
                    params.insert(
                        Rc::from(format!("{mlp_prefix}.{name}.lora_b")),
                        proj.lora_b.clone(),
                    );
                }
            }
        }
        params
    }

    /// Inject parameter values (used by autodiff).
    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        macro_rules! set_param {
            ($dst:expr, $key:expr) => {
                if let Some(value) = params.get(&Rc::from($key) as &Rc<str>) {
                    $dst = value.clone();
                }
            };
        }
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let attn_prefix = format!("layers.{i}.self_attn");
            set_param!(layer.self_attn.q_proj.lora_a, format!("{attn_prefix}.q_proj.lora_a"));
            set_param!(layer.self_attn.q_proj.lora_b, format!("{attn_prefix}.q_proj.lora_b"));
            set_param!(layer.self_attn.k_proj.lora_a, format!("{attn_prefix}.k_proj.lora_a"));
            set_param!(layer.self_attn.k_proj.lora_b, format!("{attn_prefix}.k_proj.lora_b"));
            set_param!(layer.self_attn.v_proj.lora_a, format!("{attn_prefix}.v_proj.lora_a"));
            set_param!(layer.self_attn.v_proj.lora_b, format!("{attn_prefix}.v_proj.lora_b"));
            set_param!(layer.self_attn.o_proj.lora_a, format!("{attn_prefix}.o_proj.lora_a"));
            set_param!(layer.self_attn.o_proj.lora_b, format!("{attn_prefix}.o_proj.lora_b"));
            let mlp_prefix = format!("layers.{i}.mlp");
            if layer.is_moe {
                let se = &mut layer.moe.as_mut().unwrap().shared_expert;
                set_param!(se.gate_proj.lora_a, format!("{mlp_prefix}.gate_proj.lora_a"));
                set_param!(se.gate_proj.lora_b, format!("{mlp_prefix}.gate_proj.lora_b"));
                set_param!(se.up_proj.lora_a, format!("{mlp_prefix}.up_proj.lora_a"));
                set_param!(se.up_proj.lora_b, format!("{mlp_prefix}.up_proj.lora_b"));
                set_param!(se.down_proj.lora_a, format!("{mlp_prefix}.down_proj.lora_a"));
                set_param!(se.down_proj.lora_b, format!("{mlp_prefix}.down_proj.lora_b"));
            } else {
                let mlp = layer.mlp.as_mut().unwrap();
                set_param!(mlp.gate_proj.lora_a, format!("{mlp_prefix}.gate_proj.lora_a"));
                set_param!(mlp.gate_proj.lora_b, format!("{mlp_prefix}.gate_proj.lora_b"));
                set_param!(mlp.up_proj.lora_a, format!("{mlp_prefix}.up_proj.lora_a"));
                set_param!(mlp.up_proj.lora_b, format!("{mlp_prefix}.up_proj.lora_b"));
                set_param!(mlp.down_proj.lora_a, format!("{mlp_prefix}.down_proj.lora_a"));
                set_param!(mlp.down_proj.lora_b, format!("{mlp_prefix}.down_proj.lora_b"));
            }
        }
    }

    /// Save LoRA weights to safetensors.
    pub fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        crate::save_safetensors_map(path, &self.lora_parameters())
    }

    /// Load LoRA weights from safetensors.
    pub fn load_lora_weights(&mut self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        let path = path.as_ref();
        let file_path = if path.is_dir() {
            path.join("lora_weights.safetensors")
        } else {
            path.to_path_buf()
        };
        let loaded = crate::load_safetensors_map(&file_path)?;
        let ps: HashMap<Rc<str>, Array> = loaded
            .into_iter()
            .map(|(k, v)| (Rc::from(k.as_str()), v))
            .collect();
        self.set_lora_parameters(&ps);
        Ok(())
    }

    /// Force-evaluate all LoRA adapter parameters.
    pub fn eval_all(&mut self) -> Result<(), LoraError> {
        // Embeddings.
        self.model.embed_tokens.weight.value.eval();

        for layer in &mut self.model.layers {
            // Attention quantized base + LoRA.
            for proj in [
                &mut layer.self_attn.q_proj,
                &mut layer.self_attn.k_proj,
                &mut layer.self_attn.v_proj,
                &mut layer.self_attn.o_proj,
            ] {
                proj.lora_a.eval();
                proj.lora_b.eval();
            }

            // FFN LoRA adapters (dense or MoE shared expert).
            let (gate, up, down) = if layer.is_moe {
                let se = &mut layer.moe.as_mut().unwrap().shared_expert;
                (&mut se.gate_proj, &mut se.up_proj, &mut se.down_proj)
            } else {
                let mlp = layer.mlp.as_mut().unwrap();
                (&mut mlp.gate_proj, &mut mlp.up_proj, &mut mlp.down_proj)
            };
            gate.lora_a.eval();
            gate.lora_b.eval();
            up.lora_a.eval();
            up.lora_b.eval();
            down.lora_a.eval();
            down.lora_b.eval();

            // Layer norms.
            layer.input_layernorm.weight.value.eval();
            layer.post_attention_layernorm.weight.value.eval();
        }

        self.model.norm.weight.value.eval();
        self.lm_head.weight.value.eval();
        Ok(())
    }

    /// Merge LoRA weights into quantized base weights.
    ///
    /// Note: merging is generally inadvisable for quantized bases because the
    /// dequantize-add-requantize round-trip introduces error.  This method is
    /// provided for completeness but the result should be considered approximate.
    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        // QLoraLinear does not expose a merge path; stub out with error guidance.
        Err(LoraError::InvalidState(
            "merge_lora is not supported for QLoRA models: the dequantize-merge-requantize \
             round-trip is lossy.  Export base weights with `load_base_weights_from_dir` and \
             merge the LoRA adapters at full precision."
                .to_string(),
        ))
    }

    /// Unmerge is not supported.
    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        Err(LoraError::InvalidState(
            "unmerge_lora is not supported for QLoRA models".to_string(),
        ))
    }

    /// Load and quantize base model weights from a HashMap.
    ///
    /// Weight key format matches HuggingFace (with or without `language_model.` prefix):
    /// - `model.embed_tokens.weight`
    /// - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
    /// - `model.layers.{i}.self_attn.{q,k}_norm.weight`
    /// - `model.layers.{i}.feed_forward.shared_expert.{gate,up,down}_proj.weight`
    /// - `model.layers.{i}.feed_forward.experts.{j}.{gate,up,down}_proj.weight`
    /// - `model.layers.{i}.feed_forward.router.gate.weight`
    /// - `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
    /// - `model.layers.{i}.input_layernorm.weight`
    /// - `model.layers.{i}.post_attention_layernorm.weight`
    /// - `model.norm.weight`
    /// - `lm_head.weight`
    pub fn load_base_weights(
        &mut self,
        weights: &HashMap<String, Array>,
    ) -> Result<(), LoraError> {
        use pmetal_bridge::compat::Param;

        let try_get = |key: &str| -> Option<&Array> {
            weights
                .get(key)
                .or_else(|| weights.get(&format!("language_model.{}", key)))
        };

        if let Some(w) = try_get("model.embed_tokens.weight") {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        let qcfg = self.model.qlora_config.clone();

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let p = format!("model.layers.{}", i);

            // Attention projections — re-quantize from full-precision weight.
            macro_rules! quantize_proj {
                ($proj:expr, $key:expr) => {
                    if let Some(w) = try_get(&$key) {
                        *$proj = QLoraLinear::from_weight(w, None, &qcfg)?;
                    }
                };
            }

            quantize_proj!(
                &mut layer.self_attn.q_proj,
                format!("{}.self_attn.q_proj.weight", p)
            );
            quantize_proj!(
                &mut layer.self_attn.k_proj,
                format!("{}.self_attn.k_proj.weight", p)
            );
            quantize_proj!(
                &mut layer.self_attn.v_proj,
                format!("{}.self_attn.v_proj.weight", p)
            );
            quantize_proj!(
                &mut layer.self_attn.o_proj,
                format!("{}.self_attn.o_proj.weight", p)
            );

            // QK norms (full precision).
            if let Some(qn) = &mut layer.self_attn.q_norm {
                if let Some(w) = try_get(&format!("{}.self_attn.q_norm.weight", p)) {
                    qn.weight = Param::new(w.clone());
                }
            }
            if let Some(kn) = &mut layer.self_attn.k_norm {
                if let Some(w) = try_get(&format!("{}.self_attn.k_norm.weight", p)) {
                    kn.weight = Param::new(w.clone());
                }
            }

            // FFN — MoE or dense.
            if layer.is_moe {
                if let Some(moe) = &mut layer.moe {
                    // Router (full precision).
                    if let Some(w) = try_get(&format!("{}.feed_forward.router.gate.weight", p)) {
                        moe.router.gate.weight = Param::new(w.clone());
                    }
                    // Routed experts (frozen — not quantized in this implementation;
                    // they are full-precision nn::Linear inside Llama4Expert).
                    for (j, expert) in moe.experts.iter_mut().enumerate() {
                        for (name, proj) in [
                            ("gate_proj", &mut expert.gate_proj),
                            ("up_proj", &mut expert.up_proj),
                            ("down_proj", &mut expert.down_proj),
                        ] {
                            let key = format!(
                                "{}.feed_forward.experts.{}.{}.weight",
                                p, j, name
                            );
                            if let Some(w) = try_get(&key) {
                                proj.weight = Param::new(w.clone());
                            }
                        }
                    }
                    // Shared expert — re-quantize.
                    for (name, proj) in [
                        ("gate_proj", &mut moe.shared_expert.gate_proj),
                        ("up_proj", &mut moe.shared_expert.up_proj),
                        ("down_proj", &mut moe.shared_expert.down_proj),
                    ] {
                        let key = format!(
                            "{}.feed_forward.shared_expert.{}.weight",
                            p, name
                        );
                        if let Some(w) = try_get(&key) {
                            *proj = QLoraLinear::from_weight(w, None, &qcfg)?;
                        }
                    }
                }
            } else if let Some(mlp) = &mut layer.mlp {
                for (name, proj) in [
                    ("gate_proj", &mut mlp.gate_proj),
                    ("up_proj", &mut mlp.up_proj),
                    ("down_proj", &mut mlp.down_proj),
                ] {
                    if let Some(w) = try_get(&format!("{}.mlp.{}.weight", p, name)) {
                        *proj = QLoraLinear::from_weight(w, None, &qcfg)?;
                    }
                }
            }

            // Layer norms (full precision).
            if let Some(w) = try_get(&format!("{}.input_layernorm.weight", p)) {
                layer.input_layernorm.weight = Param::new(w.clone());
            }
            if let Some(w) = try_get(&format!("{}.post_attention_layernorm.weight", p)) {
                layer.post_attention_layernorm.weight = Param::new(w.clone());
            }
        }

        if let Some(w) = try_get("model.norm.weight") {
            self.model.norm.weight = Param::new(w.clone());
        }
        if let Some(w) = try_get("lm_head.weight") {
            self.lm_head.weight = Param::new(w.clone());
        }

        Ok(())
    }

    /// Load and quantize base weights from a directory (single-file or sharded).
    pub fn load_base_weights_from_dir(
        &mut self,
        model_dir: impl AsRef<Path>,
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

        self.load_base_weights(&all_weights)
    }

    /// Architecture configuration accessor.
    pub fn config(&self) -> &Llama4TextConfig {
        &self.model.config
    }

    /// QLoRA configuration accessor.
    pub fn qlora_config(&self) -> &QLoraConfig {
        &self.model.qlora_config
    }
}

// =============================================================================
// ModuleParameters
// =============================================================================

impl ModuleParameters for Llama4QloraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.model.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();
        for (i, layer) in self.model.layers.iter().enumerate() {
            let mut layer_params = HashMap::new();

            // Attention LoRA params.
            let mut attn_params = HashMap::new();
            for (name, lora_a, lora_b) in [
                ("q_proj", &layer.self_attn.q_proj.lora_a, &layer.self_attn.q_proj.lora_b),
                ("k_proj", &layer.self_attn.k_proj.lora_a, &layer.self_attn.k_proj.lora_b),
                ("v_proj", &layer.self_attn.v_proj.lora_a, &layer.self_attn.v_proj.lora_b),
                ("o_proj", &layer.self_attn.o_proj.lora_a, &layer.self_attn.o_proj.lora_b),
            ] {
                let mut m = HashMap::new();
                m.insert(Rc::from("lora_a"), NestedValue::Value(lora_a));
                m.insert(Rc::from("lora_b"), NestedValue::Value(lora_b));
                attn_params.insert(Rc::from(name), NestedValue::Map(m));
            }
            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            // FFN / shared expert LoRA params.
            let mut mlp_params = HashMap::new();
            let mlp_triples: [(&str, &Array, &Array); 3] = if layer.is_moe {
                let se = &layer.moe.as_ref().unwrap().shared_expert;
                [
                    ("gate_proj", &se.gate_proj.lora_a, &se.gate_proj.lora_b),
                    ("up_proj", &se.up_proj.lora_a, &se.up_proj.lora_b),
                    ("down_proj", &se.down_proj.lora_a, &se.down_proj.lora_b),
                ]
            } else {
                let mlp = layer.mlp.as_ref().unwrap();
                [
                    ("gate_proj", &mlp.gate_proj.lora_a, &mlp.gate_proj.lora_b),
                    ("up_proj", &mlp.up_proj.lora_a, &mlp.up_proj.lora_b),
                    ("down_proj", &mlp.down_proj.lora_a, &mlp.down_proj.lora_b),
                ]
            };
            for (name, lora_a, lora_b) in mlp_triples {
                let mut m = HashMap::new();
                m.insert(Rc::from("lora_a"), NestedValue::Value(lora_a));
                m.insert(Rc::from("lora_b"), NestedValue::Value(lora_b));
                mlp_params.insert(Rc::from(name), NestedValue::Map(m));
            }
            layer_params.insert(Rc::from("mlp"), NestedValue::Map(mlp_params));

            params.insert(
                Rc::from(format!("layers.{i}")),
                NestedValue::Map(layer_params),
            );
        }
        params
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut params = ModuleParamMut::new();
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let mut layer_params = HashMap::new();

            // Attention LoRA params — inline per-projection to satisfy borrow checker.
            let mut attn_params = HashMap::new();
            {
                let mut m = HashMap::new();
                m.insert(Rc::from("lora_a"), NestedValue::Value(&mut layer.self_attn.q_proj.lora_a));
                m.insert(Rc::from("lora_b"), NestedValue::Value(&mut layer.self_attn.q_proj.lora_b));
                attn_params.insert(Rc::from("q_proj"), NestedValue::Map(m));
            }
            {
                let mut m = HashMap::new();
                m.insert(Rc::from("lora_a"), NestedValue::Value(&mut layer.self_attn.k_proj.lora_a));
                m.insert(Rc::from("lora_b"), NestedValue::Value(&mut layer.self_attn.k_proj.lora_b));
                attn_params.insert(Rc::from("k_proj"), NestedValue::Map(m));
            }
            {
                let mut m = HashMap::new();
                m.insert(Rc::from("lora_a"), NestedValue::Value(&mut layer.self_attn.v_proj.lora_a));
                m.insert(Rc::from("lora_b"), NestedValue::Value(&mut layer.self_attn.v_proj.lora_b));
                attn_params.insert(Rc::from("v_proj"), NestedValue::Map(m));
            }
            {
                let mut m = HashMap::new();
                m.insert(Rc::from("lora_a"), NestedValue::Value(&mut layer.self_attn.o_proj.lora_a));
                m.insert(Rc::from("lora_b"), NestedValue::Value(&mut layer.self_attn.o_proj.lora_b));
                attn_params.insert(Rc::from("o_proj"), NestedValue::Map(m));
            }
            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            // FFN / shared expert LoRA params.
            let mut mlp_params = HashMap::new();
            if layer.is_moe {
                let se = &mut layer.moe.as_mut().unwrap().shared_expert;
                {
                    let mut m = HashMap::new();
                    m.insert(Rc::from("lora_a"), NestedValue::Value(&mut se.gate_proj.lora_a));
                    m.insert(Rc::from("lora_b"), NestedValue::Value(&mut se.gate_proj.lora_b));
                    mlp_params.insert(Rc::from("gate_proj"), NestedValue::Map(m));
                }
                {
                    let mut m = HashMap::new();
                    m.insert(Rc::from("lora_a"), NestedValue::Value(&mut se.up_proj.lora_a));
                    m.insert(Rc::from("lora_b"), NestedValue::Value(&mut se.up_proj.lora_b));
                    mlp_params.insert(Rc::from("up_proj"), NestedValue::Map(m));
                }
                {
                    let mut m = HashMap::new();
                    m.insert(Rc::from("lora_a"), NestedValue::Value(&mut se.down_proj.lora_a));
                    m.insert(Rc::from("lora_b"), NestedValue::Value(&mut se.down_proj.lora_b));
                    mlp_params.insert(Rc::from("down_proj"), NestedValue::Map(m));
                }
            } else {
                let mlp = layer.mlp.as_mut().unwrap();
                {
                    let mut m = HashMap::new();
                    m.insert(Rc::from("lora_a"), NestedValue::Value(&mut mlp.gate_proj.lora_a));
                    m.insert(Rc::from("lora_b"), NestedValue::Value(&mut mlp.gate_proj.lora_b));
                    mlp_params.insert(Rc::from("gate_proj"), NestedValue::Map(m));
                }
                {
                    let mut m = HashMap::new();
                    m.insert(Rc::from("lora_a"), NestedValue::Value(&mut mlp.up_proj.lora_a));
                    m.insert(Rc::from("lora_b"), NestedValue::Value(&mut mlp.up_proj.lora_b));
                    mlp_params.insert(Rc::from("up_proj"), NestedValue::Map(m));
                }
                {
                    let mut m = HashMap::new();
                    m.insert(Rc::from("lora_a"), NestedValue::Value(&mut mlp.down_proj.lora_a));
                    m.insert(Rc::from("lora_b"), NestedValue::Value(&mut mlp.down_proj.lora_b));
                    mlp_params.insert(Rc::from("down_proj"), NestedValue::Map(m));
                }
            }
            layer_params.insert(Rc::from("mlp"), NestedValue::Map(mlp_params));

            params.insert(
                Rc::from(format!("layers.{i}")),
                NestedValue::Map(layer_params),
            );
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

// Wire the `TrainableModel` trait via the shared macro.
crate::impl_trainable_model!(Llama4QloraForCausalLM);

// =============================================================================
// Helpers
// =============================================================================

/// Apply temperature scaling to Q for NoPE long-context layers.
fn apply_temperature_scaling(
    q: Array,
    seq_len: i32,
    floor_scale: f32,
    attn_scale: f32,
) -> Array {
    let ones = ops::ones(&[seq_len], pmetal_bridge::compat::Dtype::Float32);
    let positions =
        ops::arange_from(0, seq_len).as_dtype(pmetal_bridge::compat::Dtype::Float32.as_i32());
    let pos_plus_one = positions.add(&ones);
    let floored = ops::floor(&pos_plus_one.divide(&Array::from_f32(floor_scale)));
    let log_vals = ops::log(&floored.add(&ones));
    let scales = log_vals
        .multiply(&Array::from_f32(attn_scale))
        .add(&ones)
        .reshape(&[1, 1, seq_len, 1]);
    q.multiply(&scales)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> Llama4TextConfig {
        Llama4TextConfig {
            vocab_size: 512,
            hidden_size: 64,
            intermediate_size: 32,
            intermediate_size_mlp: 48,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            max_position_embeddings: 128,
            tie_word_embeddings: false,
            num_experts_per_tok: 1,
            num_local_experts: 2,
            interleave_moe_layer_step: 1,
            moe_layers: None,
            no_rope_layer_interval: 4,
            no_rope_layers: None,
            attention_chunk_size: 64,
            use_qk_norm: true,
            attn_temperature_tuning: true,
            floor_scale: 64,
            attn_scale: 0.1,
            router_aux_loss_coef: 0.001,
            use_mod: false,
            mod_capacity: 0.5,
            mod_layers: None,
            mod_layer_interval: 2,
        }
    }

    fn mixed_config() -> Llama4TextConfig {
        Llama4TextConfig {
            interleave_moe_layer_step: 2,
            ..small_config()
        }
    }

    fn small_qlora_config() -> QLoraConfig {
        QLoraConfig::from_lora(LoraConfig {
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
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
            bias: pmetal_core::LoraBias::None,
            init_lora_weights: true,
            loraplus_lr_ratio: None,
            use_dora: false,
        })
    }

    #[test]
    fn test_llama4_qlora_builds_moe_only() {
        let model =
            Llama4QloraForCausalLM::with_qlora_config(small_config(), small_qlora_config())
                .unwrap();
        // Both layers are MoE (interleave_moe_layer_step == 1).
        assert!(model.model.layers[0].is_moe);
        assert!(model.model.layers[1].is_moe);
        assert!(model.num_trainable_params() > 0);
    }

    #[test]
    fn test_llama4_qlora_forward() {
        let mut model =
            Llama4QloraForCausalLM::with_qlora_config(small_config(), small_qlora_config())
                .unwrap();
        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3, 4]).reshape(&[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();
        assert_eq!(logits.shape(), &[1, 4, 512]);
    }

    #[test]
    fn test_llama4_qlora_mixed_layers() {
        let mut model =
            Llama4QloraForCausalLM::with_qlora_config(mixed_config(), small_qlora_config())
                .unwrap();
        // interleave_moe_layer_step=2: layer 0 MoE, layer 1 dense.
        assert!(model.model.layers[0].is_moe);
        assert!(!model.model.layers[1].is_moe);

        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3]).reshape(&[1, 3]);
        let logits = model.forward(&input_ids, None).unwrap();
        assert_eq!(logits.shape(), &[1, 3, 512]);
    }

    #[test]
    fn test_llama4_qlora_kv_cache() {
        let mut model =
            Llama4QloraForCausalLM::with_qlora_config(small_config(), small_qlora_config())
                .unwrap();
        let mut cache = model.create_cache(128);
        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3]).reshape(&[1, 3]);
        let logits = model.forward_with_cache(&input_ids, None, Some(&mut cache)).unwrap();
        assert_eq!(logits.shape(), &[1, 3, 512]);
    }

    #[test]
    fn test_llama4_qlora_param_count() {
        let model =
            Llama4QloraForCausalLM::with_qlora_config(small_config(), small_qlora_config())
                .unwrap();
        assert!(model.num_trainable_params() > 0);
        let params = model.lora_parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_llama4_qlora_trainable_model_trait() {
        use crate::TrainableModel;

        let mut model =
            Llama4QloraForCausalLM::with_qlora_config(small_config(), small_qlora_config())
                .unwrap();

        assert!(model.supports_kv_cache());
        assert!(model.supports_gradient_checkpointing());

        let input_ids = Array::from_i32_slice(&[10_i32, 20]).reshape(&[1, 2]);
        let logits = model.forward(&input_ids, None).unwrap();
        assert_eq!(logits.shape(), &[1, 2, 512]);

        let hidden = model.forward_hidden(&input_ids, None).unwrap().unwrap();
        assert_eq!(hidden.shape(), &[1, 2, 64]);
    }

    #[test]
    fn test_llama4_qlora_memory_usage() {
        let model =
            Llama4QloraForCausalLM::with_qlora_config(small_config(), small_qlora_config())
                .unwrap();
        let (quantized, lora, total) = model.memory_usage();
        assert!(quantized > 0);
        assert!(lora > 0);
        assert_eq!(total, quantized + lora);
    }

    #[test]
    fn test_llama4_qlora_no_nan() {
        let mut model =
            Llama4QloraForCausalLM::with_qlora_config(small_config(), small_qlora_config())
                .unwrap();
        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3, 4]).reshape(&[1, 4]);
        let output = model.forward(&input_ids, None).unwrap();
        output.eval();

        let has_nan = pmetal_bridge::compat::ops::any(
            &pmetal_bridge::compat::ops::is_nan(&output),
            None,
            false,
        );
        has_nan.eval();
        assert!(
            !pmetal_bridge::compat::ops::item_bool(&has_nan),
            "Output should not have NaN values"
        );
    }

    #[test]
    fn test_llama4_qlora_checkpoint_config() {
        let mut model =
            Llama4QloraForCausalLM::with_qlora_config(small_config(), small_qlora_config())
                .unwrap();
        model.enable_gradient_checkpointing(1);
        assert!(model.checkpoint_config.is_some());
        model.disable_gradient_checkpointing();
        assert!(model.checkpoint_config.is_none());
    }

    #[test]
    fn test_llama4_qlora_merge_errors() {
        let mut model =
            Llama4QloraForCausalLM::with_qlora_config(small_config(), small_qlora_config())
                .unwrap();
        assert!(model.merge_lora().is_err());
        assert!(model.unmerge_lora().is_err());
    }
}
