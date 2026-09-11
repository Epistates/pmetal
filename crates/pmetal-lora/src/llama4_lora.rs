//! LoRA-enabled Llama 4 model architecture.
//!
//! Implements Llama 4 with LoRA adapters for efficient fine-tuning.
//!
//! ## Llama 4 specifics
//!
//! - **Attention**: LoRA on q/k/v/o projections.  QK normalisation and
//!   temperature tuning are kept frozen (they hold scale parameters, not
//!   the weight matrices that LoRA targets).
//! - **MoE layers**: LoRA on the `shared_expert` only.  Routed experts stay
//!   frozen — they are gated by sparse routing and updating them via dense
//!   gradients causes interference across experts.
//! - **Dense layers** (interleaved with MoE in Maverick, or absent in Scout
//!   all-MoE config): LoRA on gate/up/down projections of the `mlp` field.
//! - **iRoPE**: layer-specific `uses_rope` flag is preserved verbatim.
//! - **MoD**: Mixture-of-Depths router stays frozen; MoD logic is unchanged.

use std::collections::HashMap;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, Exception, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, nn, ops,
};

use pmetal_core::LoraConfig;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_mlx::kernels::{AttentionMaskType, FusedAttentionConfig, fused_sdpa, rope::apply_rope};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};
use pmetal_models::architectures::llama4::{
    Llama4Expert, Llama4MoE, Llama4ModRouter, Llama4Router, Llama4TextConfig,
};

use crate::lora::LoraProjection;
use crate::lora_helpers::training_attention_mask;
use crate::lora_helpers::{
    LoraDecoderStack, collect_lora_parameters, count_trainable_params, load_lora_weights_impl,
    save_lora_weights_impl, set_lora_parameters as helpers_set_lora_parameters,
};
use crate::{LinearAdapter, LoraError, LoraLinear};

// =============================================================================
// Attention
// =============================================================================

/// LoRA-enabled Llama 4 attention layer.
///
/// LoRA (or DoRA) adapters are placed on q/k/v/o projections.  QK norms and
/// all temperature-tuning scalars remain frozen — they hold scale/norm
/// parameters that LoRA does not target.
///
/// Supports both RoPE layers and NoPE layers (iRoPE): the `uses_rope` flag is
/// set per-layer at construction time from the config.
#[derive(Debug)]
pub struct Llama4LoraAttention {
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

    /// Query projection — LoRA/DoRA adapter.
    pub q_proj: LinearAdapter,
    /// Key projection — LoRA/DoRA adapter.
    pub k_proj: LinearAdapter,
    /// Value projection — LoRA/DoRA adapter.
    pub v_proj: LinearAdapter,
    /// Output projection — LoRA/DoRA adapter.
    pub o_proj: LinearAdapter,

    /// Optional per-head Q normalisation (frozen).
    pub q_norm: Option<nn::RmsNorm>,
    /// Optional per-head K normalisation (frozen).
    pub k_norm: Option<nn::RmsNorm>,
}

impl Llama4LoraAttention {
    /// Create a new LoRA/DoRA attention layer for Llama 4.
    pub fn new(
        config: &Llama4TextConfig,
        layer_idx: usize,
        lora_config: &LoraConfig,
    ) -> Result<Self, LoraError> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;

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

        let uses_rope = config.uses_rope(layer_idx as i32);

        // QK norms — weightless RMS norm (eps 1e-6), applied AFTER RoPE and only
        // on RoPE layers (mlx-lm: `use_qk_norm = args.use_qk_norm and use_rope`).
        // Weight stays at default ones == `mx.fast.rms_norm(x, None, eps=1e-6)`.
        let (q_norm, k_norm) = if config.use_qk_norm && uses_rope {
            (
                Some(nn::RmsNormBuilder::new(head_dim).eps(1e-6).build().unwrap()),
                Some(nn::RmsNormBuilder::new(head_dim).eps(1e-6).build().unwrap()),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            layer_idx,
            uses_rope,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            rope_theta: config.rope_theta,
            rope_scale: 1.0,
            attn_temperature_tuning: config.attn_temperature_tuning,
            floor_scale: config.floor_scale as f32,
            attn_scale: config.attn_scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
        })
    }

    /// Apply RoPE to a tensor shaped `[B, T, H, D]`.
    ///
    /// `rope_apply` (fast rope) expects `[B, H, T, D]` so we transpose around
    /// the call, matching the base model's `apply_rope` helper.
    fn apply_rope_bhd(&self, x: &Array, offset: i32) -> Result<Array, LoraError> {
        // [B, T, H, D] -> [B, H, T, D]
        let x_t = x.transpose_axes(&[0, 2, 1, 3]);
        let result = apply_rope(
            &x_t,
            self.head_dim,
            true, // Llama 4 uses traditional (interleaved) RoPE
            self.rope_theta,
            self.rope_scale,
            offset,
        )
        .map_err(LoraError::Mlx)?;
        // [B, H, T, D] -> [B, T, H, D]
        Ok(result.transpose_axes(&[0, 2, 1, 3]))
    }

    /// Forward pass (training, no cache).
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        let mut q = self.q_proj.forward(x)?;
        let mut k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // [B, T, n*D] -> [B, T, n, D]
        q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim]);
        k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);

        // Apply RoPE only for RoPE layers; NoPE layers skip positional encoding.
        if self.uses_rope {
            q = self.apply_rope_bhd(&q, 0)?;
            k = self.apply_rope_bhd(&k, 0)?;
        }

        // QK normalisation (weightless, applied AFTER RoPE on RoPE layers only).
        if let (Some(qn), Some(kn)) = (&mut self.q_norm, &mut self.k_norm) {
            q = pmetal_bridge::compat::Module::forward(qn, &q)?;
            k = pmetal_bridge::compat::Module::forward(kn, &k)?;
        }

        // [B, T, H, D] -> [B, H, T, D]
        let q = q.transpose_axes(&[0, 2, 1, 3]);
        let mut k = k.transpose_axes(&[0, 2, 1, 3]);
        let mut v = v.transpose_axes(&[0, 2, 1, 3]);

        // GQA: expand KV heads to match query heads.
        let repeat = self.n_heads / self.n_kv_heads;
        if repeat > 1 {
            k = expand_kv_heads(&k, repeat).map_err(LoraError::Mlx)?;
            v = expand_kv_heads(&v, repeat).map_err(LoraError::Mlx)?;
        }

        // Temperature scaling for NoPE long-context layers.
        let q = if !self.uses_rope && self.attn_temperature_tuning {
            apply_temperature_scaling(q, seq_len, self.floor_scale, self.attn_scale)
        } else {
            q
        };

        // Scaled dot-product attention.
        let k_t = k.transpose_axes(&[0, 1, 3, 2]);
        let mut scores = q.matmul(&k_t);
        scores = scores.multiply(&Array::from_f32(self.scale));

        if let Some(m) = mask {
            scores = scores.add(m);
        }

        let probs = pmetal_bridge::compat::ops::softmax_axis(&scores, -1);
        let output = probs.matmul(&v);

        // [B, H, T, D] -> [B, T, H*D]
        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);
        self.o_proj.forward(&output)
    }

    /// Forward pass with KV cache for efficient autoregressive inference.
    pub fn forward_with_cache(
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

        q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim]);
        k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);

        // Determine RoPE offset from cache and apply positional encoding.
        if self.uses_rope {
            let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
            // Transpose to [B, H, T, D] for rope_apply, then back.
            let q_t = q.transpose_axes(&[0, 2, 1, 3]);
            let k_t = k.transpose_axes(&[0, 2, 1, 3]);
            let q_r = apply_rope(
                &q_t,
                self.head_dim,
                true, // Llama 4 uses traditional (interleaved) RoPE
                self.rope_theta,
                self.rope_scale,
                offset,
            )
            .map_err(LoraError::Mlx)?;
            let k_r = apply_rope(
                &k_t,
                self.head_dim,
                true, // Llama 4 uses traditional (interleaved) RoPE
                self.rope_theta,
                self.rope_scale,
                offset,
            )
            .map_err(LoraError::Mlx)?;
            q = q_r.transpose_axes(&[0, 2, 1, 3]);
            k = k_r.transpose_axes(&[0, 2, 1, 3]);
        }

        // QK normalisation (weightless, applied AFTER RoPE on RoPE layers only).
        if let (Some(qn), Some(kn)) = (&mut self.q_norm, &mut self.k_norm) {
            q = pmetal_bridge::compat::Module::forward(qn, &q)?;
            k = pmetal_bridge::compat::Module::forward(kn, &k)?;
        }

        // Transpose to [B, H, T, D] before cache update (cache expects this layout).
        let q = q.transpose_axes(&[0, 2, 1, 3]);
        let k = k.transpose_axes(&[0, 2, 1, 3]);
        let v = v.transpose_axes(&[0, 2, 1, 3]);

        // Update KV cache.
        let (keys, values) = if let Some((cache, layer_idx)) = cache {
            cache
                .update_and_fetch(layer_idx, &k, &v)
                .map_err(LoraError::Mlx)?
        } else {
            (k, v)
        };

        // Fused SDPA (handles GQA internally).
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(AttentionMaskType::Causal);

        let output = fused_sdpa(&q, &keys, &values, &attn_config, mask).map_err(LoraError::Mlx)?;

        let output = output
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);
        self.o_proj.forward(&output)
    }

    /// Number of trainable LoRA parameters in this attention block.
    pub fn num_trainable_params(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self.v_proj.num_trainable_params()
            + self.o_proj.num_trainable_params()
    }
}

// =============================================================================
// Shared expert with LoRA (used inside MoE layers)
// =============================================================================

/// LoRA-enabled shared expert for Llama 4 MoE layers.
///
/// The `routed_experts` field holds the frozen routed experts verbatim from
/// the base model.  Only the shared expert receives LoRA adapters.
#[derive(Debug)]
pub struct Llama4LoraSharedExpert {
    /// Gate projection — LoRA adapter.
    pub gate_proj: LoraLinear,
    /// Up projection — LoRA adapter.
    pub up_proj: LoraLinear,
    /// Down projection — LoRA adapter.
    pub down_proj: LoraLinear,
}

impl Llama4LoraSharedExpert {
    /// Create a shared expert with LoRA on gate/up/down.
    pub fn new(config: &Llama4TextConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;

        // Shared expert uses the routed-expert intermediate size (not intermediate_size_mlp).
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;

        let gate_rank = crate::effective_rank(lora_config, "gate_proj") as i32;
        let up_rank = crate::effective_rank(lora_config, "up_proj") as i32;
        let down_rank = crate::effective_rank(lora_config, "down_proj") as i32;

        Ok(Self {
            gate_proj: LoraLinear::new(hidden, inter, gate_rank, alpha, use_rslora, false)?,
            up_proj: LoraLinear::new(hidden, inter, up_rank, alpha, use_rslora, false)?,
            down_proj: LoraLinear::new(inter, hidden, down_rank, alpha, use_rslora, false)?,
        })
    }

    /// SwiGLU forward.
    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let gate = self.gate_proj.forward(x)?;
        let gate = nn::silu(&gate);
        let up = self.up_proj.forward(x)?;
        let hidden = gate.multiply(&up);
        self.down_proj.forward(&hidden)
    }

    /// Number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.gate_proj.num_trainable_params()
            + self.up_proj.num_trainable_params()
            + self.down_proj.num_trainable_params()
    }
}

// =============================================================================
// LoRA-enabled dense MLP (used in non-MoE decoder layers)
// =============================================================================

/// LoRA-enabled dense MLP for Llama 4 non-MoE layers.
#[derive(Debug)]
pub struct Llama4LoraMLP {
    /// Gate projection — LoRA adapter.
    pub gate_proj: LoraLinear,
    /// Up projection — LoRA adapter.
    pub up_proj: LoraLinear,
    /// Down projection — LoRA adapter.
    pub down_proj: LoraLinear,
}

impl Llama4LoraMLP {
    /// Create a dense MLP layer with LoRA adapters.
    ///
    /// Uses `intermediate_size_mlp` (the dense-layer size, distinct from the
    /// MoE routed-expert `intermediate_size`).
    pub fn new(config: &Llama4TextConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;

        let hidden = config.hidden_size;
        let inter = config.intermediate_size_mlp;

        let gate_rank = crate::effective_rank(lora_config, "gate_proj") as i32;
        let up_rank = crate::effective_rank(lora_config, "up_proj") as i32;
        let down_rank = crate::effective_rank(lora_config, "down_proj") as i32;

        Ok(Self {
            gate_proj: LoraLinear::new(hidden, inter, gate_rank, alpha, use_rslora, false)?,
            up_proj: LoraLinear::new(hidden, inter, up_rank, alpha, use_rslora, false)?,
            down_proj: LoraLinear::new(inter, hidden, down_rank, alpha, use_rslora, false)?,
        })
    }

    /// SwiGLU forward.
    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let gate = self.gate_proj.forward(x)?;
        let gate = nn::silu(&gate);
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.multiply(&up))
    }

    /// Number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.gate_proj.num_trainable_params()
            + self.up_proj.num_trainable_params()
            + self.down_proj.num_trainable_params()
    }
}

// =============================================================================
// LoRA-enabled MoE block
// =============================================================================

/// LoRA-enabled Llama 4 MoE block.
///
/// The router and all routed experts are frozen; only `shared_expert`
/// carries LoRA adapters.
#[derive(Debug)]
pub struct Llama4LoraMoE {
    /// Frozen routing layer.
    pub router: Llama4Router,
    /// Frozen routed experts.
    pub experts: Vec<Llama4Expert>,
    /// LoRA-enabled shared expert (always applied to all tokens).
    pub shared_expert: Llama4LoraSharedExpert,

    /// Number of routed experts to select per token (top-k).
    pub num_experts_per_tok: i32,
}

impl Llama4LoraMoE {
    pub fn new(config: &Llama4TextConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let router = Llama4Router::new(
            config.hidden_size,
            config.num_local_experts,
            config.num_experts_per_tok,
        )
        .map_err(LoraError::Mlx)?;

        let experts = (0..config.num_local_experts)
            .map(|_| Llama4Expert::new(config.hidden_size, config.intermediate_size))
            .collect::<Result<Vec<_>, _>>()
            .map_err(LoraError::Mlx)?;

        let shared_expert = Llama4LoraSharedExpert::new(config, lora_config)?;

        Ok(Self {
            router,
            experts,
            shared_expert,
            num_experts_per_tok: config.num_experts_per_tok,
        })
    }

    /// Forward pass: route tokens to sparse routed experts + shared expert.
    ///
    /// Mirrors `Llama4MoE::forward` exactly, except `shared_expert` runs
    /// through LoRA adapters.
    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let shape = x.shape().to_vec();
        let hidden_size = *shape.last().unwrap();
        let total_tokens: i32 = shape.iter().take(shape.len() - 1).product();
        let flat_x = x.reshape(&[total_tokens, hidden_size]);

        // Route (frozen).
        let (expert_indices, expert_weights, _router_logits) =
            self.router.forward(&flat_x).map_err(LoraError::Mlx)?;

        // Shared expert contribution (LoRA-adapted).
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

        let mut expert_assignments: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.experts.len()];
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
            pmetal_bridge::compat::ops::zeros_dtype(&[total_tokens, hidden_size], input_dtype);

        for (expert_idx, assignments) in expert_assignments.iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }
            let token_indices: Vec<i32> = assignments.iter().map(|&(idx, _)| idx as i32).collect();
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

    /// Number of trainable parameters (shared expert only).
    pub fn num_trainable_params(&self) -> usize {
        self.shared_expert.num_trainable_params()
    }
}

// =============================================================================
// Decoder layer
// =============================================================================

/// LoRA-enabled Llama 4 decoder layer.
///
/// Mirrors `Llama4DecoderLayer`: either holds a LoRA-adapted `Llama4LoraMoE`
/// (MoE layers) or a `Llama4LoraMLP` (dense layers), plus the optional
/// MoD router which remains frozen.
#[derive(Debug)]
pub struct Llama4LoraDecoderLayer {
    /// Layer index.
    pub layer_idx: usize,
    /// Whether this layer's FFN block is an MoE block.
    pub is_moe: bool,
    /// MoD capacity factor (None when MoD is disabled for this layer).
    pub mod_capacity: Option<f32>,

    /// Attention block with LoRA.
    pub self_attn: Llama4LoraAttention,
    /// Dense MLP with LoRA (Some iff not MoE).
    pub mlp: Option<Llama4LoraMLP>,
    /// MoE block with LoRA on shared expert (Some iff MoE).
    pub moe: Option<Llama4LoraMoE>,
    /// Input layernorm (frozen).
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layernorm (frozen).
    pub post_attention_layernorm: nn::RmsNorm,
    /// MoD router (frozen, Some only when `mod_capacity.is_some()`).
    pub mod_router: Option<Llama4ModRouter>,
}

impl Llama4LoraDecoderLayer {
    pub fn new(
        config: &Llama4TextConfig,
        layer_idx: usize,
        lora_config: &LoraConfig,
    ) -> Result<Self, LoraError> {
        let self_attn = Llama4LoraAttention::new(config, layer_idx, lora_config)?;

        let is_moe = config.is_moe_layer(layer_idx as i32);
        let (mlp, moe) = if is_moe {
            (None, Some(Llama4LoraMoE::new(config, lora_config)?))
        } else {
            (Some(Llama4LoraMLP::new(config, lora_config)?), None)
        };

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .unwrap();
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .unwrap();

        let mod_capacity = if config.is_mod_layer(layer_idx as i32) {
            Some(config.mod_capacity)
        } else {
            None
        };
        let mod_router = if mod_capacity.is_some() {
            Some(Llama4ModRouter::new(config.hidden_size).map_err(LoraError::Mlx)?)
        } else {
            None
        };

        Ok(Self {
            layer_idx,
            is_moe,
            mod_capacity,
            self_attn,
            mlp,
            moe,
            input_layernorm,
            post_attention_layernorm,
            mod_router,
        })
    }

    /// Standard full-sequence forward.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        // Attention with residual.
        let normed = pmetal_bridge::compat::Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self.self_attn.forward(&normed, mask)?;
        let h = x.add(&attn_out);

        // FFN (MoE or dense) with residual.
        let normed =
            pmetal_bridge::compat::Module::forward(&mut self.post_attention_layernorm, &h)?;
        let ffn_out = if self.is_moe {
            self.moe.as_mut().unwrap().forward(&normed)?
        } else {
            self.mlp.as_mut().unwrap().forward(&normed)?
        };
        Ok(h.add(&ffn_out))
    }

    /// Forward with KV cache threading.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let normed = pmetal_bridge::compat::Module::forward(&mut self.input_layernorm, x)?;
        let attn_out = self.self_attn.forward_with_cache(&normed, mask, cache)?;
        let h = x.add(&attn_out);

        let normed =
            pmetal_bridge::compat::Module::forward(&mut self.post_attention_layernorm, &h)?;
        let ffn_out = if self.is_moe {
            self.moe.as_mut().unwrap().forward(&normed)?
        } else {
            self.mlp.as_mut().unwrap().forward(&normed)?
        };
        Ok(h.add(&ffn_out))
    }

    /// Number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        let ffn_params = if self.is_moe {
            self.moe.as_ref().map_or(0, |m| m.num_trainable_params())
        } else {
            self.mlp.as_ref().map_or(0, |m| m.num_trainable_params())
        };
        self.self_attn.num_trainable_params() + ffn_params
    }
}

// =============================================================================
// Inner text model
// =============================================================================

/// LoRA-enabled Llama 4 text model (without LM head).
#[derive(Debug)]
pub struct Llama4LoraModel {
    /// Architecture configuration.
    pub config: Llama4TextConfig,
    /// LoRA configuration.
    pub lora_config: LoraConfig,
    /// Token embeddings (frozen).
    pub embed_tokens: nn::Embedding,
    /// Decoder layers.
    pub layers: Vec<Llama4LoraDecoderLayer>,
    /// Final RMSNorm (frozen).
    pub norm: nn::RmsNorm,
}

impl Llama4LoraModel {
    pub fn new(config: Llama4TextConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let embed_tokens =
            nn::Embedding::new(config.vocab_size, config.hidden_size).map_err(LoraError::Mlx)?;

        let layers = (0..config.num_hidden_layers)
            .map(|i| Llama4LoraDecoderLayer::new(&config, i as usize, &lora_config))
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

    /// Forward pass (builds causal mask internally).
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        self.forward_with_checkpoint(input_ids, mask, None)
    }

    /// Forward with optional gradient checkpointing markers.
    pub fn forward_with_checkpoint(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        checkpoint_config: Option<&CheckpointConfig>,
    ) -> Result<Array, LoraError> {
        let mut hidden_states =
            pmetal_bridge::compat::Module::forward(&mut self.embed_tokens, input_ids)?;

        let mask = if mask.is_none() {
            let seq_len = input_ids.dim(1);
            Some(training_attention_mask(seq_len, None)?)
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

    /// Forward with KV cache.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let mut hidden_states =
            pmetal_bridge::compat::Module::forward(&mut self.embed_tokens, input_ids)?;

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

    /// NEFTune: embed tokens, add uniform noise, then run layers.
    pub fn forward_noised(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        noise_alpha: f32,
        checkpoint_config: Option<&CheckpointConfig>,
    ) -> Result<Array, LoraError> {
        let mut hidden_states =
            pmetal_bridge::compat::Module::forward(&mut self.embed_tokens, input_ids)?;

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
            Some(training_attention_mask(seq_len_i, None)?)
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

    /// Number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_trainable_params()).sum()
    }
}

// =============================================================================
// LoraDecoderStack for generic helper functions
// =============================================================================
//
// For MoE layers, we expose the shared expert's three projections as "mlp"
// entries (gate_proj / up_proj / down_proj) — this keeps the generic
// collect/set/save/load helpers working without special-casing.
//
// For dense layers, the mlp projections are the standard ones.
//
// In both cases attn projections are always q/k/v/o.

impl LoraDecoderStack for Llama4LoraModel {
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
        if l.is_moe {
            let se = l.moe.as_ref().unwrap();
            vec![
                &se.shared_expert.gate_proj,
                &se.shared_expert.up_proj,
                &se.shared_expert.down_proj,
            ]
        } else {
            let mlp = l.mlp.as_ref().unwrap();
            vec![&mlp.gate_proj, &mlp.up_proj, &mlp.down_proj]
        }
    }

    fn mlp_projections_mut(&mut self, layer: usize) -> Vec<&mut dyn LoraProjection> {
        let l = &mut self.layers[layer];
        if l.is_moe {
            let se = &mut l.moe.as_mut().unwrap().shared_expert;
            vec![&mut se.gate_proj, &mut se.up_proj, &mut se.down_proj]
        } else {
            let mlp = l.mlp.as_mut().unwrap();
            vec![&mut mlp.gate_proj, &mut mlp.up_proj, &mut mlp.down_proj]
        }
    }
}

// =============================================================================
// ForCausalLM wrapper
// =============================================================================

/// LoRA-enabled Llama 4 model with LM head.
#[derive(Debug)]
pub struct Llama4LoraForCausalLM {
    /// Base model with LoRA.
    pub model: Llama4LoraModel,
    /// LM head (frozen).
    pub lm_head: nn::Linear,
    /// Gradient checkpointing configuration.
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl Llama4LoraForCausalLM {
    /// Create a new LoRA Llama 4 model with LM head.
    pub fn new(config: Llama4TextConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let lm_head = nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
            .bias(false)
            .build()
            .unwrap();
        let model = Llama4LoraModel::new(config, lora_config)?;
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
        let cc = self.checkpoint_config.clone();
        self.forward_with_checkpoint(input_ids, mask, cc.as_ref())
    }

    /// Forward with explicit checkpoint config.
    pub fn forward_with_checkpoint(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        checkpoint_config: Option<&CheckpointConfig>,
    ) -> Result<Array, LoraError> {
        let hidden = self
            .model
            .forward_with_checkpoint(input_ids, mask, checkpoint_config)?;
        Ok(pmetal_bridge::compat::Module::forward(
            &mut self.lm_head,
            &hidden,
        )?)
    }

    /// Hidden states before LM head (for Cut Cross-Entropy).
    pub fn forward_hidden_states(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
        let cc = self.checkpoint_config.clone();
        self.model
            .forward_with_checkpoint(input_ids, mask, cc.as_ref())
    }

    /// Hidden states with position IDs (packed sequence training).
    ///
    /// Llama 4 attention does not yet expose a separate position-ID path, so
    /// this delegates to the standard hidden-state forward.  The signature is
    /// present so `impl_trainable_model!` can wire the trait method.
    pub fn forward_hidden_states_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        _position_ids: &Array,
    ) -> Result<Array, LoraError> {
        self.forward_hidden_states(input_ids, mask)
    }

    /// NEFTune forward.
    pub fn forward_noised(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        noise_alpha: f32,
    ) -> Result<Array, LoraError> {
        let cc = self.checkpoint_config.clone();
        let hidden = self
            .model
            .forward_noised(input_ids, mask, noise_alpha, cc.as_ref())?;
        Ok(pmetal_bridge::compat::Module::forward(
            &mut self.lm_head,
            &hidden,
        )?)
    }

    /// Forward with KV cache.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let hidden = self.model.forward_with_cache(input_ids, mask, cache)?;
        Ok(pmetal_bridge::compat::Module::forward(
            &mut self.lm_head,
            &hidden,
        )?)
    }

    /// Create a KV cache for this model.
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        let config = KVCacheConfig::new(
            self.model.config.num_hidden_layers as usize,
            max_seq_len,
            self.model.config.num_key_value_heads as usize,
            self.model.config.head_dim as usize,
        );
        KVCache::new(config)
    }

    /// LM head weight `[vocab_size, hidden_dim]` (for Cut Cross-Entropy).
    pub fn get_lm_head_weight(&self) -> Option<Array> {
        Some(self.lm_head.weight.value.clone())
    }

    // -------------------------------------------------------------------------
    // LoRA parameter management
    // -------------------------------------------------------------------------

    /// All LoRA adapter parameters as a flat HashMap.
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

            // Attention adapters.
            for (proj_name, proj) in [
                ("q_proj", &mut layer.self_attn.q_proj),
                ("k_proj", &mut layer.self_attn.k_proj),
                ("v_proj", &mut layer.self_attn.v_proj),
                ("o_proj", &mut layer.self_attn.o_proj),
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

            // MLP / shared expert adapters.
            let mlp_section = "mlp";
            let projs: [(&str, &mut LoraLinear); 3] = if layer.is_moe {
                let se = &mut layer.moe.as_mut().unwrap().shared_expert;
                [
                    ("gate_proj", &mut se.gate_proj),
                    ("up_proj", &mut se.up_proj),
                    ("down_proj", &mut se.down_proj),
                ]
            } else {
                let mlp = layer.mlp.as_mut().unwrap();
                [
                    ("gate_proj", &mut mlp.gate_proj),
                    ("up_proj", &mut mlp.up_proj),
                    ("down_proj", &mut mlp.down_proj),
                ]
            };
            for (proj_name, proj) in projs {
                let a_key = format!("{}.{}.{}.lora_a", prefix, mlp_section, proj_name);
                let b_key = format!("{}.{}.{}.lora_b", prefix, mlp_section, proj_name);
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

    /// Inject parameter values (used by autodiff).
    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        helpers_set_lora_parameters(&mut self.model, params);
    }

    /// Force-evaluate all LoRA parameters.
    pub fn eval_lora_params(&mut self) -> Result<(), LoraError> {
        for layer in &mut self.model.layers {
            layer.self_attn.q_proj.lora_a_mut().eval();
            layer.self_attn.q_proj.lora_b_mut().eval();
            layer.self_attn.k_proj.lora_a_mut().eval();
            layer.self_attn.k_proj.lora_b_mut().eval();
            layer.self_attn.v_proj.lora_a_mut().eval();
            layer.self_attn.v_proj.lora_b_mut().eval();
            layer.self_attn.o_proj.lora_a_mut().eval();
            layer.self_attn.o_proj.lora_b_mut().eval();

            let (gate, up, down) = if layer.is_moe {
                let se = &mut layer.moe.as_mut().unwrap().shared_expert;
                (&mut se.gate_proj, &mut se.up_proj, &mut se.down_proj)
            } else {
                let mlp = layer.mlp.as_mut().unwrap();
                (&mut mlp.gate_proj, &mut mlp.up_proj, &mut mlp.down_proj)
            };
            gate.lora_a_mut().eval();
            gate.lora_b_mut().eval();
            up.lora_a_mut().eval();
            up.lora_b_mut().eval();
            down.lora_a_mut().eval();
            down.lora_b_mut().eval();
        }
        Ok(())
    }

    /// Evaluate all model parameters (force GPU materialization).
    ///
    /// Evaluates both base weights and LoRA weights across the full parameter
    /// tree, ensuring everything is materialized on the device.
    pub fn eval_all(&mut self) -> Result<(), LoraError> {
        // Embeddings
        self.model.embed_tokens.weight.value.eval();

        // Layers
        for layer in &mut self.model.layers {
            // Attention base weights
            layer.self_attn.q_proj.weight_mut().eval();
            layer.self_attn.k_proj.weight_mut().eval();
            layer.self_attn.v_proj.weight_mut().eval();
            layer.self_attn.o_proj.weight_mut().eval();

            // Attention LoRA weights
            layer.self_attn.q_proj.lora_a_mut().eval();
            layer.self_attn.q_proj.lora_b_mut().eval();
            layer.self_attn.k_proj.lora_a_mut().eval();
            layer.self_attn.k_proj.lora_b_mut().eval();
            layer.self_attn.v_proj.lora_a_mut().eval();
            layer.self_attn.v_proj.lora_b_mut().eval();
            layer.self_attn.o_proj.lora_a_mut().eval();
            layer.self_attn.o_proj.lora_b_mut().eval();

            // FFN base + LoRA weights (dense or MoE shared expert)
            let (gate, up, down) = if layer.is_moe {
                let se = &mut layer.moe.as_mut().unwrap().shared_expert;
                (&mut se.gate_proj, &mut se.up_proj, &mut se.down_proj)
            } else {
                let mlp = layer.mlp.as_mut().unwrap();
                (&mut mlp.gate_proj, &mut mlp.up_proj, &mut mlp.down_proj)
            };
            gate.weight_mut().eval();
            gate.lora_a_mut().eval();
            gate.lora_b_mut().eval();
            up.weight_mut().eval();
            up.lora_a_mut().eval();
            up.lora_b_mut().eval();
            down.weight_mut().eval();
            down.lora_a_mut().eval();
            down.lora_b_mut().eval();

            // Layer norms (frozen)
            layer.input_layernorm.weight.value.eval();
            layer.post_attention_layernorm.weight.value.eval();
        }

        // Final norm
        self.model.norm.weight.value.eval();

        // LM head
        self.lm_head.weight.value.eval();

        Ok(())
    }

    /// Total number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        count_trainable_params(&self.model)
    }

    /// Merge LoRA weights into base weights.
    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        for layer in &mut self.model.layers {
            layer.self_attn.q_proj.merge()?;
            layer.self_attn.k_proj.merge()?;
            layer.self_attn.v_proj.merge()?;
            layer.self_attn.o_proj.merge()?;

            if layer.is_moe {
                let se = &mut layer.moe.as_mut().unwrap().shared_expert;
                se.gate_proj.merge()?;
                se.up_proj.merge()?;
                se.down_proj.merge()?;
            } else {
                let mlp = layer.mlp.as_mut().unwrap();
                mlp.gate_proj.merge()?;
                mlp.up_proj.merge()?;
                mlp.down_proj.merge()?;
            }
        }
        Ok(())
    }

    /// Unmerge is not supported.
    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        Err(LoraError::InvalidState(
            "unmerge_lora is not supported: reload base model weights to undo a merge".to_string(),
        ))
    }

    /// Save LoRA weights to safetensors.
    pub fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        save_lora_weights_impl(&self.model, path)
    }

    /// Load LoRA weights from safetensors.
    pub fn load_lora_weights(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), LoraError> {
        load_lora_weights_impl(&mut self.model, path)
    }

    /// Load base model weights from a HashMap.
    ///
    /// Weight key format matches HuggingFace:
    /// - `language_model.model.embed_tokens.weight`
    /// - `language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
    /// - `language_model.model.layers.{i}.self_attn.{q,k}_norm.weight`
    /// - `language_model.model.layers.{i}.feed_forward.shared_expert.{gate,up,down}_proj.weight`
    /// - `language_model.model.layers.{i}.feed_forward.experts.{j}.{gate,up,down}_proj.weight`
    /// - `language_model.model.layers.{i}.feed_forward.router.gate.weight`
    /// - `language_model.model.layers.{i}.mlp.{gate,up,down}_proj.weight`
    /// - `language_model.model.layers.{i}.input_layernorm.weight`
    /// - `language_model.model.layers.{i}.post_attention_layernorm.weight`
    /// - `language_model.model.norm.weight`
    /// - `language_model.lm_head.weight`
    pub fn load_base_weights(
        &mut self,
        weights: &std::collections::HashMap<String, Array>,
    ) -> Result<(), LoraError> {
        use pmetal_bridge::compat::Param;

        // Support both with and without the "language_model." prefix.
        let try_get = |key: &str| -> Option<&Array> {
            weights
                .get(key)
                .or_else(|| weights.get(&format!("language_model.{}", key)))
        };

        if let Some(w) = try_get("model.embed_tokens.weight") {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let p = format!("model.layers.{}", i);

            // Attention projections.
            if let Some(w) = try_get(&format!("{}.self_attn.q_proj.weight", p)) {
                *layer.self_attn.q_proj.weight_mut() = w.clone();
            }
            if let Some(w) = try_get(&format!("{}.self_attn.k_proj.weight", p)) {
                *layer.self_attn.k_proj.weight_mut() = w.clone();
            }
            if let Some(w) = try_get(&format!("{}.self_attn.v_proj.weight", p)) {
                *layer.self_attn.v_proj.weight_mut() = w.clone();
            }
            if let Some(w) = try_get(&format!("{}.self_attn.o_proj.weight", p)) {
                *layer.self_attn.o_proj.weight_mut() = w.clone();
            }

            // QK norms.
            if let Some(q_norm) = &mut layer.self_attn.q_norm {
                if let Some(w) = try_get(&format!("{}.self_attn.q_norm.weight", p)) {
                    q_norm.weight = Param::new(w.clone());
                }
            }
            if let Some(k_norm) = &mut layer.self_attn.k_norm {
                if let Some(w) = try_get(&format!("{}.self_attn.k_norm.weight", p)) {
                    k_norm.weight = Param::new(w.clone());
                }
            }

            // FFN — MoE or dense.
            if layer.is_moe {
                if let Some(moe) = &mut layer.moe {
                    // Router.
                    if let Some(w) = try_get(&format!("{}.feed_forward.router.gate.weight", p)) {
                        moe.router.gate.weight = Param::new(w.clone());
                    }
                    // Routed experts (frozen base weights).
                    for (j, expert) in moe.experts.iter_mut().enumerate() {
                        for (name, proj) in [
                            ("gate_proj", &mut expert.gate_proj),
                            ("up_proj", &mut expert.up_proj),
                            ("down_proj", &mut expert.down_proj),
                        ] {
                            let key = format!("{}.feed_forward.experts.{}.{}.weight", p, j, name);
                            if let Some(w) = try_get(&key) {
                                proj.weight = Param::new(w.clone());
                            }
                        }
                    }
                    // Shared expert base weights.
                    for (name, proj) in [
                        ("gate_proj", &mut moe.shared_expert.gate_proj),
                        ("up_proj", &mut moe.shared_expert.up_proj),
                        ("down_proj", &mut moe.shared_expert.down_proj),
                    ] {
                        let key = format!("{}.feed_forward.shared_expert.{}.weight", p, name);
                        if let Some(w) = try_get(&key) {
                            *proj.weight_mut() = w.clone();
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
                        *proj.weight_mut() = w.clone();
                    }
                }
            }

            // Layer norms.
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

    /// Load base weights from a directory (single-file or sharded safetensors).
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

    /// Accessor for the architecture config.
    pub fn config(&self) -> &Llama4TextConfig {
        &self.model.config
    }

    /// Accessor for the LoRA config.
    pub fn lora_config(&self) -> &LoraConfig {
        &self.model.lora_config
    }
}

// =============================================================================
// ModuleParameters
// =============================================================================

impl ModuleParameters for Llama4LoraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.model.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();

        for (i, layer) in self.model.layers.iter().enumerate() {
            let prefix: Rc<str> = Rc::from(format!("layers.{}", i));
            let mut layer_params = HashMap::new();

            // Attention params.
            let mut attn_params = HashMap::new();
            for (name, proj) in [
                ("q_proj", &layer.self_attn.q_proj as &dyn LoraProjection),
                ("k_proj", &layer.self_attn.k_proj),
                ("v_proj", &layer.self_attn.v_proj),
                ("o_proj", &layer.self_attn.o_proj),
            ] {
                let mut m = HashMap::new();
                m.insert(Rc::from("lora_a"), NestedValue::Value(proj.lora_a()));
                m.insert(Rc::from("lora_b"), NestedValue::Value(proj.lora_b()));
                attn_params.insert(Rc::from(name), NestedValue::Map(m));
            }
            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            // MLP / shared expert params.
            let mut mlp_params = HashMap::new();
            let (gate, up, down) = if layer.is_moe {
                let se = &layer.moe.as_ref().unwrap().shared_expert;
                (
                    &se.gate_proj as &dyn LoraProjection,
                    &se.up_proj as &dyn LoraProjection,
                    &se.down_proj as &dyn LoraProjection,
                )
            } else {
                let mlp = layer.mlp.as_ref().unwrap();
                (
                    &mlp.gate_proj as &dyn LoraProjection,
                    &mlp.up_proj as &dyn LoraProjection,
                    &mlp.down_proj as &dyn LoraProjection,
                )
            };
            for (name, proj) in [("gate_proj", gate), ("up_proj", up), ("down_proj", down)] {
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

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix: Rc<str> = Rc::from(format!("layers.{}", i));
            let mut layer_params = HashMap::new();

            // Attention params — use inline helpers to avoid E0499.
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

            let mut attn_params = HashMap::new();
            attn_params.insert(
                Rc::from("q_proj"),
                NestedValue::Map(adapter_params_mut(&mut layer.self_attn.q_proj)),
            );
            attn_params.insert(
                Rc::from("k_proj"),
                NestedValue::Map(adapter_params_mut(&mut layer.self_attn.k_proj)),
            );
            attn_params.insert(
                Rc::from("v_proj"),
                NestedValue::Map(adapter_params_mut(&mut layer.self_attn.v_proj)),
            );
            attn_params.insert(
                Rc::from("o_proj"),
                NestedValue::Map(adapter_params_mut(&mut layer.self_attn.o_proj)),
            );
            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            let mut mlp_params = HashMap::new();
            if layer.is_moe {
                let se = &mut layer.moe.as_mut().unwrap().shared_expert;
                mlp_params.insert(
                    Rc::from("gate_proj"),
                    NestedValue::Map(lora_params_mut(&mut se.gate_proj)),
                );
                mlp_params.insert(
                    Rc::from("up_proj"),
                    NestedValue::Map(lora_params_mut(&mut se.up_proj)),
                );
                mlp_params.insert(
                    Rc::from("down_proj"),
                    NestedValue::Map(lora_params_mut(&mut se.down_proj)),
                );
            } else {
                let mlp = layer.mlp.as_mut().unwrap();
                mlp_params.insert(
                    Rc::from("gate_proj"),
                    NestedValue::Map(lora_params_mut(&mut mlp.gate_proj)),
                );
                mlp_params.insert(
                    Rc::from("up_proj"),
                    NestedValue::Map(lora_params_mut(&mut mlp.up_proj)),
                );
                mlp_params.insert(
                    Rc::from("down_proj"),
                    NestedValue::Map(lora_params_mut(&mut mlp.down_proj)),
                );
            }
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

// Wire the `TrainableModel` trait via the shared macro.
crate::impl_trainable_model!(Llama4LoraForCausalLM);

// =============================================================================
// Helpers
// =============================================================================

/// Expand KV heads for grouped query attention: `[B, n_kv, T, D]` -> `[B, n_heads, T, D]`.
fn expand_kv_heads(x: &Array, repeat: i32) -> Result<Array, Exception> {
    let shape = x.shape();
    let (b, n_kv, t, d) = (shape[0], shape[1], shape[2], shape[3]);
    let expanded = ops::broadcast_to(&x.reshape(&[b, n_kv, 1, t, d]), &[b, n_kv, repeat, t, d]);
    Ok(expanded.reshape(&[b, n_kv * repeat, t, d]))
}

/// Apply temperature scaling to Q for NoPE long-context layers.
///
/// `scale_i = log(floor((i + 1) / floor_scale) + 1) * attn_scale + 1`
fn apply_temperature_scaling(q: Array, seq_len: i32, floor_scale: f32, attn_scale: f32) -> Array {
    let ones = ops::ones(&[seq_len], pmetal_bridge::compat::Dtype::Float32);
    let positions =
        ops::arange_from(0, seq_len).as_dtype(pmetal_bridge::compat::Dtype::Float32.as_i32());
    let pos_plus_one = positions.add(&ones);
    let floored = ops::floor(&pos_plus_one.divide(&Array::from_f32(floor_scale)));
    let log_vals = ops::log(&floored.add(&ones));
    let scales = log_vals.multiply(&Array::from_f32(attn_scale)).add(&ones);
    let scales = scales.reshape(&[1, 1, seq_len, 1]);
    q.multiply(&scales)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal Llama 4 Scout-like config (very small dimensions for testing).
    fn small_config() -> Llama4TextConfig {
        Llama4TextConfig {
            vocab_size: 512,
            hidden_size: 64,
            intermediate_size: 32,     // MoE expert size
            intermediate_size_mlp: 48, // Dense layer size
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            max_position_embeddings: 128,
            tie_word_embeddings: false,
            // All layers are MoE (interleave_moe_layer_step = 1).
            num_experts_per_tok: 1,
            num_local_experts: 2,
            interleave_moe_layer_step: 1,
            moe_layers: None,
            // iRoPE: NoPE every 4th layer (so both layers in this tiny model use RoPE).
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

    /// Mixed-layer config: layer 0 dense, layer 1 MoE.
    fn mixed_config() -> Llama4TextConfig {
        Llama4TextConfig {
            interleave_moe_layer_step: 2, // MoE every 2 layers: layer 0 dense, layer 1 MoE
            ..small_config()
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
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
            bias: pmetal_core::LoraBias::None,
            init_lora_weights: true,
            loraplus_lr_ratio: None,
            use_dora: false,
        }
    }

    #[test]
    fn test_llama4_lora_attention_rope() {
        let config = small_config();
        let lc = small_lora_config();
        // Layer 0 uses RoPE (0 % 4 != 0 is false → layer 0 is NoPE, layer 1 is RoPE).
        // Actually: uses_rope = layer_idx % no_rope_layer_interval != 0
        // RoPE on layers where (idx + 1) % 4 != 0; layer 1 -> (2) % 4 != 0 -> RoPE.
        let mut attn = Llama4LoraAttention::new(&config, 1, &lc).unwrap();
        assert!(attn.uses_rope);

        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let out = attn.forward(&x, None).unwrap();
        assert_eq!(out.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_llama4_lora_attention_nope() {
        let config = small_config();
        let lc = small_lora_config();
        // NoPE on layers where (idx + 1) % 4 == 0; layer 3 -> (4) % 4 == 0 -> NoPE.
        let mut attn = Llama4LoraAttention::new(&config, 3, &lc).unwrap();
        assert!(!attn.uses_rope);

        let x = pmetal_bridge::compat::random::normal(
            &[1, 4, 64],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let out = attn.forward(&x, None).unwrap();
        assert_eq!(out.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_llama4_lora_moe_layer() {
        let config = small_config();
        let lc = small_lora_config();
        let mut model = Llama4LoraForCausalLM::new(config, lc).unwrap();

        // Both layers are MoE (interleave_moe_layer_step == 1).
        assert!(model.model.layers[0].is_moe);
        assert!(model.model.layers[1].is_moe);

        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3, 4]).reshape(&[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();
        assert_eq!(logits.shape(), &[1, 4, 512]);
    }

    #[test]
    fn test_llama4_lora_mixed_layers() {
        let config = mixed_config();
        let lc = small_lora_config();
        let mut model = Llama4LoraForCausalLM::new(config, lc).unwrap();

        // interleave_moe_layer_step=2: the LAST layer of each pair is MoE
        // (idx % 2 == 1), so layer 0 is dense and layer 1 is MoE.
        assert!(!model.model.layers[0].is_moe);
        assert!(model.model.layers[1].is_moe);

        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3]).reshape(&[1, 3]);
        let logits = model.forward(&input_ids, None).unwrap();
        assert_eq!(logits.shape(), &[1, 3, 512]);
    }

    #[test]
    fn test_llama4_lora_param_count() {
        let config = small_config();
        let lc = small_lora_config();
        let model = Llama4LoraForCausalLM::new(config, lc).unwrap();

        assert!(model.num_trainable_params() > 0);
        let params = model.lora_parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_llama4_lora_kv_cache() {
        let config = small_config();
        let lc = small_lora_config();
        let mut model = Llama4LoraForCausalLM::new(config, lc).unwrap();

        let mut cache = model.create_cache(128);

        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3]).reshape(&[1, 3]);
        let logits = model
            .forward_with_cache(&input_ids, None, Some(&mut cache))
            .unwrap();
        assert_eq!(logits.shape(), &[1, 3, 512]);
    }

    #[test]
    fn test_llama4_lora_trainable_model_trait() {
        use crate::TrainableModel;

        let config = small_config();
        let lc = small_lora_config();
        let mut model = Llama4LoraForCausalLM::new(config, lc).unwrap();

        assert!(model.supports_kv_cache());
        // The per-architecture LoRA models accept the setting and log at the
        // boundaries, but never checkpoint. Real support lives on `AdaptedModel`.
        assert!(!model.supports_gradient_checkpointing());

        let input_ids = Array::from_i32_slice(&[10_i32, 20]).reshape(&[1, 2]);
        let logits = model.forward(&input_ids, None).unwrap();
        assert_eq!(logits.shape(), &[1, 2, 512]);

        let hidden = model.forward_hidden(&input_ids, None).unwrap().unwrap();
        assert_eq!(hidden.shape(), &[1, 2, 64]);
    }

    #[test]
    fn test_llama4_lora_merge() {
        let config = small_config();
        let lc = small_lora_config();
        let mut model = Llama4LoraForCausalLM::new(config, lc).unwrap();

        let input_ids = Array::from_i32_slice(&[1_i32, 2, 3]).reshape(&[1, 3]);
        let before = model.forward(&input_ids, None).unwrap();
        before.eval().unwrap();

        model.merge_lora().unwrap();

        let after = model.forward(&input_ids, None).unwrap();
        after.eval().unwrap();

        let diff = before.subtract(&after).unwrap();
        let max_diff = diff.abs().unwrap().max(None).unwrap();
        max_diff.eval().unwrap();
        assert!(max_diff.item::<f32>() < 1e-4);

        assert!(model.unmerge_lora().is_err());
    }

    #[test]
    fn test_llama4_lora_checkpoint() {
        let config = small_config();
        let lc = small_lora_config();
        let mut model = Llama4LoraForCausalLM::new(config, lc).unwrap();

        model.enable_gradient_checkpointing(1);
        assert!(model.checkpoint_config.is_some());

        let input_ids = Array::from_i32_slice(&[1_i32, 2]).reshape(&[1, 2]);
        let logits = model.forward(&input_ids, None).unwrap();
        assert_eq!(logits.shape(), &[1, 2, 512]);

        model.disable_gradient_checkpointing();
        assert!(model.checkpoint_config.is_none());
    }
}
