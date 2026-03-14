//! LoRA-enabled Qwen 3.5 (qwen3_next) hybrid architecture.
//!
//! Implements Qwen3.5 with LoRA adapters for efficient fine-tuning.
//!
//! LoRA placement strategy:
//! - **Full attention layers** (every `full_attention_interval`-th layer):
//!   LoRA on q_proj, k_proj, v_proj, o_proj.
//!   Note: q_proj outputs `n_heads * head_dim * 2` (for gated output split).
//! - **GDN linear attention layers** (all other layers):
//!   LoRA on in_proj_qkv, in_proj_z, and out_proj.
//!   conv1d, in_proj_b, in_proj_a, dt_bias, a_log, and norm remain frozen.
//! - **Dense MLP layers**: LoRA on gate_proj, up_proj, down_proj.
//! - **MoE layers**: LoRA on shared_expert's gate_proj, up_proj, down_proj only.
//!   The 512 routed experts (switch_mlp) are frozen.

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Once;
use std::sync::atomic::{AtomicUsize, Ordering};

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    module::{Module, ModuleParamMut, ModuleParamRef, ModuleParameters, Param},
    nested::NestedValue,
    nn,
    ops::{self, indexing::IndexOp},
};

use pmetal_core::LoraConfig;
use pmetal_mlx::gather_mm;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, differentiable_attention, fused_sdpa,
    gated_delta::gated_delta_update,
    get_training_context,
    rope::{RopeScaling, apply_rope},
};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig, MambaCache, MambaCacheEntry};
use pmetal_models::ModelConfig;
use pmetal_models::architectures::qwen3_next::{
    Qwen3NextConfig, Qwen3NextRMSNormGated, Qwen3NextSparseMoeBlock, sanitize_weights,
};

use crate::{LoraError, LoraLinear};

// ============================================================================
// Layer ID counter (for differentiable_attention caching)
// ============================================================================

static LAYER_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);
static GRAD_CKPT_WARN: Once = Once::new();

/// Reset the global layer ID counter for Qwen3Next LoRA models.
///
/// Must be called at model initialization so IDs start from 0 for each
/// new model instance.
pub fn reset_qwen3_next_layer_ids() {
    LAYER_ID_COUNTER.store(0, Ordering::SeqCst);
}

// ============================================================================
// Qwen3NextLoraAttention — full attention with gated output + partial RoPE
// ============================================================================

/// LoRA-enabled full attention layer for Qwen3.5.
///
/// Differences from standard Qwen3 attention:
/// - q_proj outputs `n_heads * head_dim * 2` — the extra half is a sigmoid gate
///   applied to the attention output before o_proj.
/// - Only the first `rope_dims` dimensions of each head receive RoPE.
/// - LoRA is applied to the full q_proj (gated dimension included).
#[derive(Debug)]
pub struct Qwen3NextLoraAttention {
    // Scalar configuration
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub effective_base: f32,
    pub rope_scale: f32,
    pub layer_id: usize,

    // LoRA projections
    /// q_proj: hidden -> n_heads * head_dim * 2 (with gate)
    pub q_proj: LoraLinear,
    pub k_proj: LoraLinear,
    pub v_proj: LoraLinear,
    /// o_proj: n_heads * head_dim -> hidden
    pub o_proj: LoraLinear,

    // Frozen norms (Qwen3-style per-head Q/K normalization)
    pub q_norm: nn::RmsNorm,
    pub k_norm: nn::RmsNorm,
}

impl Qwen3NextLoraAttention {
    pub fn new(config: &Qwen3NextConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let head_dim = config.get_head_dim();
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.get_num_kv_heads();
        let scale = (head_dim as f32).powf(-0.5);

        let layer_id = LAYER_ID_COUNTER.fetch_add(1, Ordering::SeqCst);

        let rope_scaling = config
            .rope_scaling
            .as_ref()
            .map(RopeScaling::from_config_map)
            .unwrap_or(RopeScaling::None);
        let rope_scale = rope_scaling.scale();
        let effective_base = rope_scaling.effective_base(config.rope_theta, head_dim);
        let rope_dims = config.rope_dims();

        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;
        let q_rank = crate::effective_rank(lora_config, "q_proj") as i32;
        let k_rank = crate::effective_rank(lora_config, "k_proj") as i32;
        let v_rank = crate::effective_rank(lora_config, "v_proj") as i32;
        let o_rank = crate::effective_rank(lora_config, "o_proj") as i32;

        // q_proj outputs n_heads * head_dim * 2 (gate is the extra head_dim per head)
        let q_proj = LoraLinear::new(
            config.hidden_size,
            n_heads * head_dim * 2,
            q_rank,
            alpha,
            use_rslora,
            false,
        )?;
        let k_proj = LoraLinear::new(
            config.hidden_size,
            n_kv_heads * head_dim,
            k_rank,
            alpha,
            use_rslora,
            false,
        )?;
        let v_proj = LoraLinear::new(
            config.hidden_size,
            n_kv_heads * head_dim,
            v_rank,
            alpha,
            use_rslora,
            false,
        )?;
        let o_proj = LoraLinear::new(
            n_heads * head_dim,
            config.hidden_size,
            o_rank,
            alpha,
            use_rslora,
            false,
        )?;

        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()
            .map_err(LoraError::Mlx)?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()
            .map_err(LoraError::Mlx)?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            rope_dims,
            effective_base,
            rope_scale,
            layer_id,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
        })
    }

    /// Training forward — uses differentiable_attention for O(n) memory when
    /// the sequence length crosses the Metal FlashAttention threshold.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];

        // Project Q (with gate in second half) and K, V
        let q_proj_out = self.q_proj.forward(x)?;
        // [B, L, n_heads, head_dim * 2] — split along last dim
        let q_gate = q_proj_out.reshape(&[b, l, self.n_heads, self.head_dim * 2])?;
        let queries = q_gate.index((.., .., .., ..self.head_dim));
        // gate: [B, L, n_heads * head_dim] — used after attention output
        let gate = q_gate.index((.., .., .., self.head_dim..)).reshape(&[
            b,
            l,
            self.n_heads * self.head_dim,
        ])?;

        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Apply Q/K per-head normalization
        let queries = Module::forward(&mut self.q_norm, &queries)?;
        let keys_reshaped = keys.reshape(&[b, l, self.n_kv_heads, self.head_dim])?;
        let keys_normed = Module::forward(&mut self.k_norm, &keys_reshaped)?;
        let values = values.reshape(&[b, l, self.n_kv_heads, self.head_dim])?;

        // Transpose to [B, heads, L, head_dim]
        let queries = queries.transpose_axes(&[0, 2, 1, 3])?;
        let keys = keys_normed.transpose_axes(&[0, 2, 1, 3])?;
        let values = values.transpose_axes(&[0, 2, 1, 3])?;

        // Partial RoPE — only apply to first `rope_dims` dimensions
        let queries = apply_rope(
            &queries,
            self.rope_dims,
            false,
            self.effective_base,
            self.rope_scale,
            0,
        )?;
        let keys = apply_rope(
            &keys,
            self.rope_dims,
            false,
            self.effective_base,
            self.rope_scale,
            0,
        )?;

        // Choose FlashAttention vs standard based on seq length and training context
        let is_training = get_training_context()
            .map(|ctx| ctx.lock().map(|c| c.is_training()).unwrap_or(false))
            .unwrap_or(false);

        const FLASH_ATTENTION_SEQ_THRESHOLD: i32 = 2048;
        let use_flash = is_training && l >= FLASH_ATTENTION_SEQ_THRESHOLD;

        let output = if use_flash {
            let fa_config = FusedAttentionConfig {
                num_heads: self.n_heads,
                num_kv_heads: self.n_kv_heads,
                head_dim: self.head_dim,
                scale: self.scale,
                mask_type: AttentionMaskType::Causal,
                logit_softcapping: None,
            };
            differentiable_attention(self.layer_id, &queries, &keys, &values, &fa_config)
                .map_err(|e| LoraError::Mlx(Exception::custom(e.to_string())))?
        } else {
            // Expand KV for GQA
            let (keys, values) = if self.n_kv_heads < self.n_heads {
                let r = self.n_heads / self.n_kv_heads;
                (expand_kv_heads(&keys, r)?, expand_kv_heads(&values, r)?)
            } else {
                (keys, values)
            };

            let scores = queries.matmul(&keys.transpose_axes(&[0, 1, 3, 2])?)?;
            let scores = scores.multiply(Array::from_f32(self.scale))?;
            let scores = if let Some(m) = mask {
                scores.add(m)?
            } else {
                scores
            };
            let weights = ops::softmax_axis(&scores, -1, None)?;
            weights.matmul(&values)?
        };

        // [B, heads, L, head_dim] -> [B, L, n_heads * head_dim]
        let output =
            output
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, l, self.n_heads * self.head_dim])?;

        // Gated output: o_proj(output * sigmoid(gate))
        let gated = output.multiply(&nn::sigmoid(&gate)?)?;
        self.o_proj.forward(&gated)
    }

    /// Cache-aware forward for efficient inference.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];

        let q_proj_out = self.q_proj.forward(x)?;
        let q_gate = q_proj_out.reshape(&[b, l, self.n_heads, self.head_dim * 2])?;
        let queries = q_gate.index((.., .., .., ..self.head_dim));
        let gate = q_gate.index((.., .., .., self.head_dim..)).reshape(&[
            b,
            l,
            self.n_heads * self.head_dim,
        ])?;

        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let queries = Module::forward(&mut self.q_norm, &queries)?;
        let keys_reshaped = keys.reshape(&[b, l, self.n_kv_heads, self.head_dim])?;
        let keys_normed = Module::forward(&mut self.k_norm, &keys_reshaped)?;
        let values = values.reshape(&[b, l, self.n_kv_heads, self.head_dim])?;

        let queries = queries.transpose_axes(&[0, 2, 1, 3])?;
        let keys = keys_normed.transpose_axes(&[0, 2, 1, 3])?;
        let values = values.transpose_axes(&[0, 2, 1, 3])?;

        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let queries = apply_rope(
            &queries,
            self.rope_dims,
            false,
            self.effective_base,
            self.rope_scale,
            offset,
        )?;
        let keys = apply_rope(
            &keys,
            self.rope_dims,
            false,
            self.effective_base,
            self.rope_scale,
            offset,
        )?;

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

        let output =
            output
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, l, self.n_heads * self.head_dim])?;

        let gated = output.multiply(&nn::sigmoid(&gate)?)?;
        self.o_proj.forward(&gated)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self.v_proj.num_trainable_params()
            + self.o_proj.num_trainable_params()
    }
}

// ============================================================================
// Qwen3NextLoraGDN — Gated Delta Net with LoRA on in_proj_qkv + in_proj_z + out_proj
// ============================================================================

/// LoRA-enabled Gated Delta Net linear attention.
///
/// LoRA is applied to:
/// - `in_proj_qkv`: hidden -> key_dim*2 + value_dim
/// - `in_proj_z`: hidden -> value_dim
/// - `out_proj`: value_dim -> hidden
///
/// Frozen components (no LoRA):
/// - `conv1d`: depthwise temporal convolution over q/k/v
/// - `in_proj_b`: projects beta scalars for the GDN kernel
/// - `in_proj_a`: projects alpha scalars for the GDN kernel
/// - `dt_bias`, `a_log`: SSM state-space parameters
/// - `norm`: per-head RMSNorm with optional silu gate
#[derive(Debug)]
pub struct Qwen3NextLoraGDN {
    // Frozen base components
    pub conv1d: nn::Conv1d,
    pub in_proj_b: nn::Linear,
    pub in_proj_a: nn::Linear,
    pub norm: Qwen3NextRMSNormGated,
    pub dt_bias: Param<Array>,
    pub a_log: Param<Array>,

    // LoRA projections
    pub in_proj_qkv: LoraLinear,
    pub in_proj_z: LoraLinear,
    pub out_proj: LoraLinear,

    // Scalar dims (derived from config)
    pub hidden_size: i32,
    pub num_v_heads: i32,
    pub num_k_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub key_dim: i32,
    pub value_dim: i32,
    pub conv_dim: i32,
    pub conv_kernel_size: i32,
}

impl Qwen3NextLoraGDN {
    pub fn new(config: &Qwen3NextConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let hidden_size = config.hidden_size;
        let num_v_heads = config.linear_num_value_heads;
        let num_k_heads = config.linear_num_key_heads;
        let head_k_dim = config.linear_key_head_dim;
        let head_v_dim = config.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_kernel_size = config.linear_conv_kernel_dim;
        let conv_dim = key_dim * 2 + value_dim;

        // Frozen conv1d: depthwise over [q, k, v] concatenated.
        let conv1d = nn::Conv1dBuilder::new(1, conv_dim, conv_kernel_size)
            .bias(false)
            .groups(conv_dim)
            .padding(0)
            .build()
            .map_err(LoraError::Mlx)?;

        // Frozen in_proj_b / in_proj_a: separate projections to per v-head scalars
        let in_proj_b = nn::LinearBuilder::new(hidden_size, num_v_heads)
            .bias(false)
            .build()
            .map_err(LoraError::Mlx)?;
        let in_proj_a = nn::LinearBuilder::new(hidden_size, num_v_heads)
            .bias(false)
            .build()
            .map_err(LoraError::Mlx)?;

        // Frozen SSM parameters
        let dt_bias = Param::new(Array::ones::<f32>(&[num_v_heads]).map_err(LoraError::Mlx)?);
        let a_log = Param::new(
            mlx_rs::random::uniform::<_, f32>(0.0, 16.0, &[num_v_heads], None)
                .map_err(LoraError::Mlx)?
                .log()
                .map_err(LoraError::Mlx)?,
        );

        // Frozen norm
        let norm =
            Qwen3NextRMSNormGated::new(head_v_dim, config.rms_norm_eps).map_err(LoraError::Mlx)?;

        // LoRA projections — 3 separate linears matching HF weight format
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;
        let in_proj_qkv_rank = crate::effective_rank(lora_config, "in_proj_qkv") as i32;
        let in_proj_z_rank = crate::effective_rank(lora_config, "in_proj_z") as i32;
        let out_proj_rank = crate::effective_rank(lora_config, "out_proj") as i32;

        let in_proj_qkv = LoraLinear::new(
            hidden_size,
            key_dim * 2 + value_dim,
            in_proj_qkv_rank,
            alpha,
            use_rslora,
            false,
        )?;
        let in_proj_z = LoraLinear::new(
            hidden_size,
            value_dim,
            in_proj_z_rank,
            alpha,
            use_rslora,
            false,
        )?;
        let out_proj = LoraLinear::new(
            value_dim,
            hidden_size,
            out_proj_rank,
            alpha,
            use_rslora,
            false,
        )?;

        Ok(Self {
            conv1d,
            in_proj_b,
            in_proj_a,
            norm,
            dt_bias,
            a_log,
            in_proj_qkv,
            in_proj_z,
            out_proj,
            hidden_size,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size,
        })
    }

    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, LoraError> {
        let shape = inputs.shape();
        let b = shape[0];
        let s = shape[1];

        // 4 separate projections matching HF weight format (qwen3_5.py:136-139)
        let qkv = self.in_proj_qkv.forward(inputs)?;
        let z =
            self.in_proj_z
                .forward(inputs)?
                .reshape(&[b, s, self.num_v_heads, self.head_v_dim])?;
        let b_val = Module::forward(&mut self.in_proj_b, inputs)?;
        let a = Module::forward(&mut self.in_proj_a, inputs)?;

        // Convolution state management (from MambaCache)
        let conv_state = if let Some(ref c) = cache {
            c.conv_state.clone()
        } else {
            None
        };
        let conv_state = conv_state.unwrap_or_else(|| {
            Array::zeros::<f32>(&[b, self.conv_kernel_size - 1, self.conv_dim]).unwrap()
        });

        // Mask QKV BEFORE conv (Python line 149-150)
        let qkv = if let Some(msk) = mask {
            let mask_expanded = msk.reshape(&[msk.dim(0), msk.dim(1), 1])?;
            ops::r#where(&mask_expanded, &qkv, &Array::from_f32(0.0))?
        } else {
            qkv
        };

        // Prepend conv state and run depthwise conv1d + silu
        let conv_input = ops::concatenate_axis(&[&conv_state, &qkv], 1)?;

        // Update conv state in cache
        if let Some(c) = cache.as_deref_mut() {
            let keep = self.conv_kernel_size - 1;
            let total_len = conv_input.dim(1);
            c.conv_state = Some(conv_input.index((.., (total_len - keep).., ..)));
        }

        let conv_out = nn::silu(&Module::forward(&mut self.conv1d, &conv_input)?)?;

        // Split conv output — simple positional split on last dim (Python line 156-163)
        let q_conv = conv_out.index((.., .., ..self.key_dim));
        let k_conv = conv_out.index((.., .., self.key_dim..self.key_dim * 2));
        let v_conv = conv_out.index((.., .., self.key_dim * 2..));

        // Trim to last S timesteps (conv prepended state adds padding)
        let out_len = q_conv.dim(1);
        let q_conv = q_conv.index((.., (out_len - s).., ..)).reshape(&[
            b,
            s,
            self.num_k_heads,
            self.head_k_dim,
        ])?;
        let k_conv = k_conv.index((.., (out_len - s).., ..)).reshape(&[
            b,
            s,
            self.num_k_heads,
            self.head_k_dim,
        ])?;
        let v_conv = v_conv.index((.., (out_len - s).., ..)).reshape(&[
            b,
            s,
            self.num_v_heads,
            self.head_v_dim,
        ])?;

        // Q/K RMS normalization with scaling (frozen, using identity weight)
        let inv_scale = (self.head_k_dim as f32).powf(-0.5);
        let q_normed =
            mlx_rs::fast::rms_norm(&q_conv, &Array::ones::<f32>(&[self.head_k_dim])?, 1e-6)?
                .multiply(&Array::from_f32(inv_scale * inv_scale))?;
        let k_normed =
            mlx_rs::fast::rms_norm(&k_conv, &Array::ones::<f32>(&[self.head_k_dim])?, 1e-6)?
                .multiply(&Array::from_f32(inv_scale))?;

        // Get SSM state from cache
        let ssm_state = cache.as_ref().and_then(|c| c.ssm_state.as_ref());

        // Run GDN recurrence (frozen kernel).
        // Training (cache=None) must use sequential path — tri_inv has no VJP.
        // Inference (cache=Some) uses fast chunk path for prefill.
        let is_training = cache.is_none();
        let (out, new_state) = gated_delta_update(
            &q_normed,
            &k_normed,
            &v_conv,
            &a,
            &b_val,
            self.a_log.as_ref(),
            self.dt_bias.as_ref(),
            ssm_state,
            mask,
            is_training,
        )?;

        // Update SSM state
        if let Some(c) = cache {
            c.ssm_state = Some(new_state);
        }

        // Gated norm then LoRA'd out_proj
        let out = self.norm.forward(&out, Some(&z))?;
        self.out_proj.forward(&out.reshape(&[b, s, -1])?)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.in_proj_qkv.num_trainable_params()
            + self.in_proj_z.num_trainable_params()
            + self.out_proj.num_trainable_params()
    }
}

// ============================================================================
// Qwen3NextLoraMLP — Dense SwiGLU MLP with LoRA
// ============================================================================

/// LoRA-enabled dense MLP layer (SwiGLU).
#[derive(Debug)]
pub struct Qwen3NextLoraMLP {
    pub gate_proj: LoraLinear,
    pub up_proj: LoraLinear,
    pub down_proj: LoraLinear,
}

impl Qwen3NextLoraMLP {
    pub fn new(dim: i32, hidden_dim: i32, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;
        let gate_rank = crate::effective_rank(lora_config, "gate_proj") as i32;
        let up_rank = crate::effective_rank(lora_config, "up_proj") as i32;
        let down_rank = crate::effective_rank(lora_config, "down_proj") as i32;

        let gate_proj = LoraLinear::new(dim, hidden_dim, gate_rank, alpha, use_rslora, false)?;
        let up_proj = LoraLinear::new(dim, hidden_dim, up_rank, alpha, use_rslora, false)?;
        let down_proj = LoraLinear::new(hidden_dim, dim, down_rank, alpha, use_rslora, false)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let gate = self.gate_proj.forward(x)?;
        let gate = nn::silu(gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = gate.multiply(&up)?;
        self.down_proj.forward(&hidden)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.gate_proj.num_trainable_params()
            + self.up_proj.num_trainable_params()
            + self.down_proj.num_trainable_params()
    }
}

// ============================================================================
// Qwen3NextLoraSharedExpert — LoRA on shared expert inside MoE
// ============================================================================

/// LoRA-enabled shared expert within the MoE block.
///
/// The 512 routed experts (switch_mlp) remain frozen. Only this shared expert
/// receives LoRA adapters.
#[derive(Debug)]
pub struct Qwen3NextLoraSharedExpert {
    pub gate_proj: LoraLinear,
    pub up_proj: LoraLinear,
    pub down_proj: LoraLinear,
}

impl Qwen3NextLoraSharedExpert {
    pub fn new(config: &Qwen3NextConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let dim = config.hidden_size;
        let hidden_dim = config.shared_expert_intermediate_size;
        Qwen3NextLoraMLP::new(dim, hidden_dim, lora_config).map(|m| Self {
            gate_proj: m.gate_proj,
            up_proj: m.up_proj,
            down_proj: m.down_proj,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let gate = self.gate_proj.forward(x)?;
        let gate = nn::silu(gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = gate.multiply(&up)?;
        self.down_proj.forward(&hidden)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.gate_proj.num_trainable_params()
            + self.up_proj.num_trainable_params()
            + self.down_proj.num_trainable_params()
    }
}

// ============================================================================
// Qwen3NextLoraSparseMoE — frozen routed experts + LoRA'd shared expert
// ============================================================================

/// MoE block with frozen switch experts and LoRA on shared expert.
#[derive(Debug)]
pub struct Qwen3NextLoraSparseMoE {
    // Frozen routing and routed experts
    pub gate: nn::Linear,
    pub switch_mlp_gate_proj: Param<Array>,
    pub switch_mlp_up_proj: Param<Array>,
    pub switch_mlp_down_proj: Param<Array>,
    pub shared_expert_gate: nn::Linear,

    // LoRA'd shared expert
    pub shared_expert: Qwen3NextLoraSharedExpert,

    // Scalar config
    pub num_experts: i32,
    pub top_k: i32,
    pub norm_topk_prob: bool,
}

impl Qwen3NextLoraSparseMoE {
    pub fn new(config: &Qwen3NextConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let dim = config.hidden_size;
        let intermediate_size = config.moe_intermediate_size;
        let num_experts = config.num_experts;

        let gate = nn::LinearBuilder::new(dim, num_experts)
            .bias(false)
            .build()
            .map_err(LoraError::Mlx)?;

        // Stacked expert weights — frozen
        let gate_proj =
            Array::zeros::<f32>(&[num_experts, intermediate_size, dim]).map_err(LoraError::Mlx)?;
        let up_proj =
            Array::zeros::<f32>(&[num_experts, intermediate_size, dim]).map_err(LoraError::Mlx)?;
        let down_proj =
            Array::zeros::<f32>(&[num_experts, dim, intermediate_size]).map_err(LoraError::Mlx)?;

        let shared_expert_gate = nn::LinearBuilder::new(dim, 1)
            .bias(false)
            .build()
            .map_err(LoraError::Mlx)?;

        let shared_expert = Qwen3NextLoraSharedExpert::new(config, lora_config)?;

        Ok(Self {
            gate,
            switch_mlp_gate_proj: Param::new(gate_proj),
            switch_mlp_up_proj: Param::new(up_proj),
            switch_mlp_down_proj: Param::new(down_proj),
            shared_expert_gate,
            shared_expert,
            num_experts,
            top_k: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let shape = x.shape();
        let batch_seq: i32 = shape[..shape.len() - 1].iter().product();
        let hidden = shape[shape.len() - 1];
        let x_flat = x.reshape(&[batch_seq, hidden])?;

        // Routing
        let gate_logits = Module::forward(&mut self.gate, &x_flat)?;
        let gates = ops::softmax_axis(
            &if gate_logits.dtype() != mlx_rs::Dtype::Float32 {
                gate_logits.as_type::<f32>()?
            } else {
                gate_logits
            },
            -1,
            None,
        )?;

        let k = self.top_k;
        let neg_gates = gates.negative()?;
        let sorted_indices = ops::argsort_axis(&neg_gates, -1)?;
        let top_indices = sorted_indices.index((.., ..k));
        let top_weights = gates.take_along_axis(&top_indices, -1)?;

        let top_weights = if self.norm_topk_prob {
            let weight_sum = top_weights.sum_axis(-1, Some(true))?;
            let safe_sum = ops::maximum(&weight_sum, &Array::from_f32(1e-8))?;
            top_weights.divide(&safe_sum)?
        } else {
            top_weights
        };

        let top_indices_i32 = top_indices.as_type::<i32>()?;

        // SwitchGLU with frozen expert weights
        let gate_out = gather_mm(
            &x_flat,
            self.switch_mlp_gate_proj.as_ref(),
            None,
            Some(&top_indices_i32),
            false,
        )?;
        let up_out = gather_mm(
            &x_flat,
            self.switch_mlp_up_proj.as_ref(),
            None,
            Some(&top_indices_i32),
            false,
        )?;

        let activated = nn::silu(&gate_out)?.multiply(&up_out)?;

        let down_out = gather_mm(
            &activated.reshape(&[batch_seq * k, -1])?,
            self.switch_mlp_down_proj.as_ref(),
            None,
            Some(&top_indices_i32.reshape(&[batch_seq * k, 1])?),
            false,
        )?
        .reshape(&[batch_seq, k, hidden])?;

        let y = down_out
            .multiply(&top_weights.reshape(&[batch_seq, k, 1])?)?
            .sum_axis(-2, false)?;

        // LoRA'd shared expert with sigmoid gate
        let shared_y = self.shared_expert.forward(&x_flat)?;
        let shared_gate = nn::sigmoid(&Module::forward(&mut self.shared_expert_gate, &x_flat)?)?;
        let shared_y = shared_gate.multiply(&shared_y)?;

        let result = y.add(&shared_y)?;
        result.reshape(shape).map_err(LoraError::Mlx)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.shared_expert.num_trainable_params()
    }
}

// ============================================================================
// Feed-forward enum (Dense or MoE)
// ============================================================================

#[derive(Debug)]
pub enum Qwen3NextLoraFeedForward {
    Dense(Qwen3NextLoraMLP),
    MoE(Qwen3NextLoraSparseMoE),
}

impl Qwen3NextLoraFeedForward {
    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        match self {
            Self::Dense(m) => m.forward(x),
            Self::MoE(m) => m.forward(x),
        }
    }

    pub fn num_trainable_params(&self) -> usize {
        match self {
            Self::Dense(m) => m.num_trainable_params(),
            Self::MoE(m) => m.num_trainable_params(),
        }
    }
}

// ============================================================================
// Qwen3NextLoraDecoderLayer — hybrid transformer block
// ============================================================================

/// Hybrid decoder layer: uses `linear_attn` (GDN) OR `self_attn` (full attention)
/// based on layer index. Option fields produce correct HF weight key names.
#[derive(Debug)]
pub struct Qwen3NextLoraDecoderLayer {
    pub is_linear: bool,
    pub is_moe: bool,
    pub linear_attn: Option<Qwen3NextLoraGDN>,
    pub self_attn: Option<Qwen3NextLoraAttention>,
    pub mlp: Qwen3NextLoraFeedForward,
    /// Input layernorm (frozen — uses (1+w) pattern, applied by sanitize_weights).
    pub input_layernorm: nn::RmsNorm,
    /// Post-attention layernorm (frozen).
    pub post_attention_layernorm: nn::RmsNorm,
}

impl Qwen3NextLoraDecoderLayer {
    pub fn new(
        config: &Qwen3NextConfig,
        lora_config: &LoraConfig,
        layer_idx: usize,
    ) -> Result<Self, LoraError> {
        let is_linear = config.is_linear_layer(layer_idx);
        let is_moe = config.use_moe_at(layer_idx);

        let linear_attn = if is_linear {
            Some(Qwen3NextLoraGDN::new(config, lora_config)?)
        } else {
            None
        };
        let self_attn = if !is_linear {
            Some(Qwen3NextLoraAttention::new(config, lora_config)?)
        } else {
            None
        };

        let mlp = if is_moe {
            Qwen3NextLoraFeedForward::MoE(Qwen3NextLoraSparseMoE::new(config, lora_config)?)
        } else {
            Qwen3NextLoraFeedForward::Dense(Qwen3NextLoraMLP::new(
                config.hidden_size,
                config.intermediate_size,
                lora_config,
            )?)
        };

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .map_err(LoraError::Mlx)?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .map_err(LoraError::Mlx)?;

        Ok(Self {
            is_linear,
            is_moe,
            linear_attn,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Training forward (no KV cache).
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        mamba_cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, LoraError> {
        let normed = Module::forward(&mut self.input_layernorm, x)?;

        let r = if self.is_linear {
            self.linear_attn
                .as_mut()
                .expect("linear_attn must be Some for linear layers")
                .forward(&normed, mask, mamba_cache)?
        } else {
            self.self_attn
                .as_mut()
                .expect("self_attn must be Some for attention layers")
                .forward(&normed, mask)?
        };
        let h = x.add(&r)?;

        let mlp_in = Module::forward(&mut self.post_attention_layernorm, &h)?;
        Ok(h.add(&self.mlp.forward(&mlp_in)?)?)
    }

    /// Cache-aware forward for inference.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        mamba_cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, LoraError> {
        let normed = Module::forward(&mut self.input_layernorm, x)?;

        let r = if self.is_linear {
            self.linear_attn
                .as_mut()
                .expect("linear_attn must be Some for linear layers")
                .forward(&normed, mask, mamba_cache)?
        } else {
            self.self_attn
                .as_mut()
                .expect("self_attn must be Some for attention layers")
                .forward_with_cache(&normed, mask, kv_cache)?
        };
        let h = x.add(&r)?;

        let mlp_in = Module::forward(&mut self.post_attention_layernorm, &h)?;
        Ok(h.add(&self.mlp.forward(&mlp_in)?)?)
    }

    pub fn num_trainable_params(&self) -> usize {
        let mixer_params = if let Some(ref gdn) = self.linear_attn {
            gdn.num_trainable_params()
        } else if let Some(ref attn) = self.self_attn {
            attn.num_trainable_params()
        } else {
            0
        };
        mixer_params + self.mlp.num_trainable_params()
    }
}

// ============================================================================
// Qwen3NextLoraModel
// ============================================================================

/// Qwen3.5 model stack (without LM head).
#[derive(Debug)]
pub struct Qwen3NextLoraModel {
    pub config: Qwen3NextConfig,
    pub lora_config: LoraConfig,
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<Qwen3NextLoraDecoderLayer>,
    pub norm: nn::RmsNorm,
}

impl Qwen3NextLoraModel {
    pub fn new(config: Qwen3NextConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        reset_qwen3_next_layer_ids();

        let embed_tokens =
            nn::Embedding::new(config.vocab_size, config.hidden_size).map_err(LoraError::Mlx)?;

        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| Qwen3NextLoraDecoderLayer::new(&config, &lora_config, i))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .map_err(LoraError::Mlx)?;

        Ok(Self {
            config,
            lora_config,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        self.forward_with_checkpoint(input_ids, mask, None)
    }

    pub fn forward_with_checkpoint(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        checkpoint_config: Option<&CheckpointConfig>,
    ) -> Result<Array, LoraError> {
        let mut hidden = Module::forward(&mut self.embed_tokens, input_ids)?;

        // Create separate masks for full attention vs GDN layers (matching base model):
        // - Full attention: uses the causal mask from the caller (4D [1,1,T,T])
        // - GDN (linear attention): uses None — GDN expects 2D [B,T] token-validity,
        //   not a 4D attention mask. Passing 4D causes reshape errors.
        let fa_mask = mask;
        let ssm_mask: Option<&Array> = None;

        let layers_per_block = checkpoint_config
            .map(|c| c.layers_per_block)
            .unwrap_or(usize::MAX);
        let checkpointing_enabled = checkpoint_config.map(|c| c.enabled).unwrap_or(false);

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let layer_mask = if layer.is_linear { ssm_mask } else { fa_mask };
            hidden = layer.forward(&hidden, layer_mask, None)?;

            if checkpointing_enabled && (idx + 1) % layers_per_block == 0 {
                GRAD_CKPT_WARN.call_once(|| {
                    tracing::info!(
                        "Qwen3Next uses eager evaluation for memory management \
                         (gradient checkpointing requires custom_vjp not yet in MLX-rs)"
                    );
                });
            }
        }

        Ok(Module::forward(&mut self.norm, &hidden)?)
    }

    /// Cache-aware forward for inference with both KV cache and Mamba cache.
    ///
    /// This mirrors the base model's `forward_with_cache`: attention layers use
    /// KV cache for efficient autoregressive decoding, GDN layers use MambaCache
    /// for recurrent state.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut kv_cache: Option<&mut KVCache>,
        mut mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, LoraError> {
        let mut hidden = Module::forward(&mut self.embed_tokens, input_ids)?;

        // Dual mask split (same as training forward and base model):
        // - Full attention layers: causal mask from caller
        // - GDN layers: None (recurrent, no attention mask needed)
        let fa_mask = mask;
        let ssm_mask: Option<&Array> = None;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let kv = if !layer.is_linear {
                kv_cache.as_deref_mut().map(|c| (c, layer_idx))
            } else {
                None
            };
            let mamba = if layer.is_linear {
                mamba_cache
                    .as_deref_mut()
                    .and_then(|c| c.get_mut(layer_idx))
            } else {
                None
            };
            let layer_mask = if layer.is_linear { ssm_mask } else { fa_mask };
            hidden = layer.forward_with_cache(&hidden, layer_mask, kv, mamba)?;
        }

        Ok(Module::forward(&mut self.norm, &hidden)?)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_trainable_params()).sum()
    }
}

// ============================================================================
// Qwen3NextLoraForCausalLM
// ============================================================================

/// Qwen3.5 causal language model with LoRA adapters.
#[derive(Debug)]
pub struct Qwen3NextLoraForCausalLM {
    pub model: Qwen3NextLoraModel,
    /// LM head — absent when `tie_word_embeddings = true`.
    pub lm_head: Option<nn::Linear>,
    /// Gradient checkpointing configuration.
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl Qwen3NextLoraForCausalLM {
    pub fn new(config: Qwen3NextConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let tie_weights = config.tie_word_embeddings;
        let model = Qwen3NextLoraModel::new(config.clone(), lora_config)?;

        let lm_head = if !tie_weights {
            Some(
                nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                    .bias(false)
                    .build()
                    .map_err(LoraError::Mlx)?,
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

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let checkpoint_config = self.checkpoint_config.clone();
        let hidden_states =
            self.model
                .forward_with_checkpoint(input_ids, mask, checkpoint_config.as_ref())?;
        self.lm_head_forward(&hidden_states)
    }

    /// Cache-aware forward for autoregressive inference.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, LoraError> {
        let h = self
            .model
            .forward_with_cache(input_ids, mask, kv_cache, mamba_cache)?;
        self.lm_head_forward(&h)
    }

    /// Create a KV cache sized for the attention layers.
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        let config = &self.model.config;
        KVCache::new(KVCacheConfig::new(
            config.num_hidden_layers as usize,
            max_seq_len,
            config.num_kv_heads() as usize,
            config.head_dim() as usize,
        ))
    }

    /// Create a Mamba cache for GDN layers.
    pub fn create_mamba_cache(&self) -> MambaCache {
        MambaCache::new(self.model.config.num_hidden_layers as usize)
    }

    fn lm_head_forward(&mut self, h: &Array) -> Result<Array, LoraError> {
        if let Some(ref mut lm_head) = self.lm_head {
            Ok(Module::forward(lm_head, h)?)
        } else {
            Ok(self.model.embed_tokens.as_linear(h)?)
        }
    }

    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    pub fn config(&self) -> &Qwen3NextConfig {
        &self.model.config
    }

    pub fn lora_config(&self) -> &LoraConfig {
        &self.model.lora_config
    }

    // -------------------------------------------------------------------------
    // LoRA parameter utilities
    // -------------------------------------------------------------------------

    /// Collect all trainable LoRA parameters as a flat HashMap.
    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        let mut params = HashMap::new();

        for (i, layer) in self.model.layers.iter().enumerate() {
            let prefix = format!("layers.{i}");

            if let Some(ref attn) = layer.self_attn {
                for (proj_name, lora) in [
                    ("q_proj", &attn.q_proj),
                    ("k_proj", &attn.k_proj),
                    ("v_proj", &attn.v_proj),
                    ("o_proj", &attn.o_proj),
                ] {
                    params.insert(
                        Rc::from(format!("{prefix}.self_attn.{proj_name}.lora_a")),
                        lora.lora_a.clone(),
                    );
                    params.insert(
                        Rc::from(format!("{prefix}.self_attn.{proj_name}.lora_b")),
                        lora.lora_b.clone(),
                    );
                }
            }

            if let Some(ref gdn) = layer.linear_attn {
                for (proj_name, lora) in [
                    ("in_proj_qkv", &gdn.in_proj_qkv),
                    ("in_proj_z", &gdn.in_proj_z),
                    ("out_proj", &gdn.out_proj),
                ] {
                    params.insert(
                        Rc::from(format!("{prefix}.linear_attn.{proj_name}.lora_a")),
                        lora.lora_a.clone(),
                    );
                    params.insert(
                        Rc::from(format!("{prefix}.linear_attn.{proj_name}.lora_b")),
                        lora.lora_b.clone(),
                    );
                }
            }

            match &layer.mlp {
                Qwen3NextLoraFeedForward::Dense(mlp) => {
                    for (proj_name, lora) in [
                        ("gate_proj", &mlp.gate_proj),
                        ("up_proj", &mlp.up_proj),
                        ("down_proj", &mlp.down_proj),
                    ] {
                        params.insert(
                            Rc::from(format!("{prefix}.mlp.{proj_name}.lora_a")),
                            lora.lora_a.clone(),
                        );
                        params.insert(
                            Rc::from(format!("{prefix}.mlp.{proj_name}.lora_b")),
                            lora.lora_b.clone(),
                        );
                    }
                }
                Qwen3NextLoraFeedForward::MoE(moe) => {
                    let se = &moe.shared_expert;
                    for (proj_name, lora) in [
                        ("gate_proj", &se.gate_proj),
                        ("up_proj", &se.up_proj),
                        ("down_proj", &se.down_proj),
                    ] {
                        params.insert(
                            Rc::from(format!("{prefix}.mlp.shared_expert.{proj_name}.lora_a")),
                            lora.lora_a.clone(),
                        );
                        params.insert(
                            Rc::from(format!("{prefix}.mlp.shared_expert.{proj_name}.lora_b")),
                            lora.lora_b.clone(),
                        );
                    }
                }
            }
        }

        params
    }

    /// Restore LoRA parameters from a HashMap.
    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        macro_rules! set_param {
            ($dst:expr, $key:expr) => {
                if let Some(v) = params.get(&Rc::from($key) as &Rc<str>) {
                    $dst = v.clone();
                }
            };
        }

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{i}");

            if let Some(ref mut attn) = layer.self_attn {
                set_param!(
                    attn.q_proj.lora_a,
                    format!("{prefix}.self_attn.q_proj.lora_a")
                );
                set_param!(
                    attn.q_proj.lora_b,
                    format!("{prefix}.self_attn.q_proj.lora_b")
                );
                set_param!(
                    attn.k_proj.lora_a,
                    format!("{prefix}.self_attn.k_proj.lora_a")
                );
                set_param!(
                    attn.k_proj.lora_b,
                    format!("{prefix}.self_attn.k_proj.lora_b")
                );
                set_param!(
                    attn.v_proj.lora_a,
                    format!("{prefix}.self_attn.v_proj.lora_a")
                );
                set_param!(
                    attn.v_proj.lora_b,
                    format!("{prefix}.self_attn.v_proj.lora_b")
                );
                set_param!(
                    attn.o_proj.lora_a,
                    format!("{prefix}.self_attn.o_proj.lora_a")
                );
                set_param!(
                    attn.o_proj.lora_b,
                    format!("{prefix}.self_attn.o_proj.lora_b")
                );
            }

            if let Some(ref mut gdn) = layer.linear_attn {
                set_param!(
                    gdn.in_proj_qkv.lora_a,
                    format!("{prefix}.linear_attn.in_proj_qkv.lora_a")
                );
                set_param!(
                    gdn.in_proj_qkv.lora_b,
                    format!("{prefix}.linear_attn.in_proj_qkv.lora_b")
                );
                set_param!(
                    gdn.in_proj_z.lora_a,
                    format!("{prefix}.linear_attn.in_proj_z.lora_a")
                );
                set_param!(
                    gdn.in_proj_z.lora_b,
                    format!("{prefix}.linear_attn.in_proj_z.lora_b")
                );
                set_param!(
                    gdn.out_proj.lora_a,
                    format!("{prefix}.linear_attn.out_proj.lora_a")
                );
                set_param!(
                    gdn.out_proj.lora_b,
                    format!("{prefix}.linear_attn.out_proj.lora_b")
                );
            }

            match &mut layer.mlp {
                Qwen3NextLoraFeedForward::Dense(mlp) => {
                    set_param!(
                        mlp.gate_proj.lora_a,
                        format!("{prefix}.mlp.gate_proj.lora_a")
                    );
                    set_param!(
                        mlp.gate_proj.lora_b,
                        format!("{prefix}.mlp.gate_proj.lora_b")
                    );
                    set_param!(mlp.up_proj.lora_a, format!("{prefix}.mlp.up_proj.lora_a"));
                    set_param!(mlp.up_proj.lora_b, format!("{prefix}.mlp.up_proj.lora_b"));
                    set_param!(
                        mlp.down_proj.lora_a,
                        format!("{prefix}.mlp.down_proj.lora_a")
                    );
                    set_param!(
                        mlp.down_proj.lora_b,
                        format!("{prefix}.mlp.down_proj.lora_b")
                    );
                }
                Qwen3NextLoraFeedForward::MoE(moe) => {
                    let se = &mut moe.shared_expert;
                    set_param!(
                        se.gate_proj.lora_a,
                        format!("{prefix}.mlp.shared_expert.gate_proj.lora_a")
                    );
                    set_param!(
                        se.gate_proj.lora_b,
                        format!("{prefix}.mlp.shared_expert.gate_proj.lora_b")
                    );
                    set_param!(
                        se.up_proj.lora_a,
                        format!("{prefix}.mlp.shared_expert.up_proj.lora_a")
                    );
                    set_param!(
                        se.up_proj.lora_b,
                        format!("{prefix}.mlp.shared_expert.up_proj.lora_b")
                    );
                    set_param!(
                        se.down_proj.lora_a,
                        format!("{prefix}.mlp.shared_expert.down_proj.lora_a")
                    );
                    set_param!(
                        se.down_proj.lora_b,
                        format!("{prefix}.mlp.shared_expert.down_proj.lora_b")
                    );
                }
            }
        }
    }

    /// Save LoRA adapters to a safetensors file.
    pub fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        let params = self.lora_parameters();
        Array::save_safetensors(params, None, path)?;
        Ok(())
    }

    /// Load LoRA adapters from a safetensors file or directory.
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

        macro_rules! load_param {
            ($dst:expr, $key:expr) => {
                if let Some(v) = loaded.get(&Rc::from($key) as &str) {
                    $dst = v.clone();
                }
            };
        }

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{i}");

            if let Some(ref mut attn) = layer.self_attn {
                load_param!(
                    attn.q_proj.lora_a,
                    format!("{prefix}.self_attn.q_proj.lora_a")
                );
                load_param!(
                    attn.q_proj.lora_b,
                    format!("{prefix}.self_attn.q_proj.lora_b")
                );
                load_param!(
                    attn.k_proj.lora_a,
                    format!("{prefix}.self_attn.k_proj.lora_a")
                );
                load_param!(
                    attn.k_proj.lora_b,
                    format!("{prefix}.self_attn.k_proj.lora_b")
                );
                load_param!(
                    attn.v_proj.lora_a,
                    format!("{prefix}.self_attn.v_proj.lora_a")
                );
                load_param!(
                    attn.v_proj.lora_b,
                    format!("{prefix}.self_attn.v_proj.lora_b")
                );
                load_param!(
                    attn.o_proj.lora_a,
                    format!("{prefix}.self_attn.o_proj.lora_a")
                );
                load_param!(
                    attn.o_proj.lora_b,
                    format!("{prefix}.self_attn.o_proj.lora_b")
                );
            }

            if let Some(ref mut gdn) = layer.linear_attn {
                load_param!(
                    gdn.in_proj_qkv.lora_a,
                    format!("{prefix}.linear_attn.in_proj_qkv.lora_a")
                );
                load_param!(
                    gdn.in_proj_qkv.lora_b,
                    format!("{prefix}.linear_attn.in_proj_qkv.lora_b")
                );
                load_param!(
                    gdn.in_proj_z.lora_a,
                    format!("{prefix}.linear_attn.in_proj_z.lora_a")
                );
                load_param!(
                    gdn.in_proj_z.lora_b,
                    format!("{prefix}.linear_attn.in_proj_z.lora_b")
                );
                load_param!(
                    gdn.out_proj.lora_a,
                    format!("{prefix}.linear_attn.out_proj.lora_a")
                );
                load_param!(
                    gdn.out_proj.lora_b,
                    format!("{prefix}.linear_attn.out_proj.lora_b")
                );
            }

            match &mut layer.mlp {
                Qwen3NextLoraFeedForward::Dense(mlp) => {
                    load_param!(
                        mlp.gate_proj.lora_a,
                        format!("{prefix}.mlp.gate_proj.lora_a")
                    );
                    load_param!(
                        mlp.gate_proj.lora_b,
                        format!("{prefix}.mlp.gate_proj.lora_b")
                    );
                    load_param!(mlp.up_proj.lora_a, format!("{prefix}.mlp.up_proj.lora_a"));
                    load_param!(mlp.up_proj.lora_b, format!("{prefix}.mlp.up_proj.lora_b"));
                    load_param!(
                        mlp.down_proj.lora_a,
                        format!("{prefix}.mlp.down_proj.lora_a")
                    );
                    load_param!(
                        mlp.down_proj.lora_b,
                        format!("{prefix}.mlp.down_proj.lora_b")
                    );
                }
                Qwen3NextLoraFeedForward::MoE(moe) => {
                    let se = &mut moe.shared_expert;
                    load_param!(
                        se.gate_proj.lora_a,
                        format!("{prefix}.mlp.shared_expert.gate_proj.lora_a")
                    );
                    load_param!(
                        se.gate_proj.lora_b,
                        format!("{prefix}.mlp.shared_expert.gate_proj.lora_b")
                    );
                    load_param!(
                        se.up_proj.lora_a,
                        format!("{prefix}.mlp.shared_expert.up_proj.lora_a")
                    );
                    load_param!(
                        se.up_proj.lora_b,
                        format!("{prefix}.mlp.shared_expert.up_proj.lora_b")
                    );
                    load_param!(
                        se.down_proj.lora_a,
                        format!("{prefix}.mlp.shared_expert.down_proj.lora_a")
                    );
                    load_param!(
                        se.down_proj.lora_b,
                        format!("{prefix}.mlp.shared_expert.down_proj.lora_b")
                    );
                }
            }
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Weight loading from SafeTensors
    // -------------------------------------------------------------------------

    /// Load base model weights from a HashMap (already loaded SafeTensors).
    ///
    /// Weight name convention mirrors the Python HuggingFace/MLX-LM layout.
    /// `sanitize_weights` must be called before invoking this method so that
    /// expert weights are stacked and norm offsets are applied.
    pub fn load_base_weights(&mut self, weights: &HashMap<String, Array>) -> Result<(), LoraError> {
        // Embedding
        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let pfx = format!("model.layers.{i}");

            // Layer norms (frozen)
            if let Some(w) = weights.get(&format!("{pfx}.input_layernorm.weight")) {
                layer.input_layernorm.weight = Param::new(w.clone());
            }
            if let Some(w) = weights.get(&format!("{pfx}.post_attention_layernorm.weight")) {
                layer.post_attention_layernorm.weight = Param::new(w.clone());
            }

            // Full attention layers
            if let Some(ref mut attn) = layer.self_attn {
                if let Some(w) = weights.get(&format!("{pfx}.self_attn.q_proj.weight")) {
                    attn.q_proj.weight = w.clone();
                }
                if let Some(w) = weights.get(&format!("{pfx}.self_attn.k_proj.weight")) {
                    attn.k_proj.weight = w.clone();
                }
                if let Some(w) = weights.get(&format!("{pfx}.self_attn.v_proj.weight")) {
                    attn.v_proj.weight = w.clone();
                }
                if let Some(w) = weights.get(&format!("{pfx}.self_attn.o_proj.weight")) {
                    attn.o_proj.weight = w.clone();
                }
                if let Some(w) = weights.get(&format!("{pfx}.self_attn.q_norm.weight")) {
                    attn.q_norm.weight = Param::new(w.clone());
                }
                if let Some(w) = weights.get(&format!("{pfx}.self_attn.k_norm.weight")) {
                    attn.k_norm.weight = Param::new(w.clone());
                }
            }

            // GDN (linear attention) layers
            if let Some(ref mut gdn) = layer.linear_attn {
                if let Some(w) = weights.get(&format!("{pfx}.linear_attn.in_proj_qkv.weight")) {
                    gdn.in_proj_qkv.weight = w.clone();
                }
                if let Some(w) = weights.get(&format!("{pfx}.linear_attn.in_proj_z.weight")) {
                    gdn.in_proj_z.weight = w.clone();
                }
                if let Some(w) = weights.get(&format!("{pfx}.linear_attn.in_proj_b.weight")) {
                    gdn.in_proj_b.weight = Param::new(w.clone());
                }
                if let Some(w) = weights.get(&format!("{pfx}.linear_attn.in_proj_a.weight")) {
                    gdn.in_proj_a.weight = Param::new(w.clone());
                }
                if let Some(w) = weights.get(&format!("{pfx}.linear_attn.out_proj.weight")) {
                    gdn.out_proj.weight = w.clone();
                }
                if let Some(w) = weights.get(&format!("{pfx}.linear_attn.conv1d.weight")) {
                    gdn.conv1d.weight = Param::new(w.clone());
                }
                if let Some(w) = weights.get(&format!("{pfx}.linear_attn.dt_bias")) {
                    gdn.dt_bias = Param::new(w.clone());
                }
                if let Some(w) = weights.get(&format!("{pfx}.linear_attn.a_log")) {
                    gdn.a_log = Param::new(w.clone());
                }
                // Gated RMSNorm weight
                if let Some(w) = weights.get(&format!("{pfx}.linear_attn.norm.weight")) {
                    gdn.norm.weight = Param::new(w.clone());
                }
            }

            // MLP weights
            match &mut layer.mlp {
                Qwen3NextLoraFeedForward::Dense(mlp) => {
                    if let Some(w) = weights.get(&format!("{pfx}.mlp.gate_proj.weight")) {
                        mlp.gate_proj.weight = w.clone();
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.mlp.up_proj.weight")) {
                        mlp.up_proj.weight = w.clone();
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.mlp.down_proj.weight")) {
                        mlp.down_proj.weight = w.clone();
                    }
                }
                Qwen3NextLoraFeedForward::MoE(moe) => {
                    // Frozen routing gate
                    if let Some(w) = weights.get(&format!("{pfx}.mlp.gate.weight")) {
                        moe.gate.weight = Param::new(w.clone());
                    }
                    // Frozen stacked expert weights (post sanitize_weights)
                    if let Some(w) = weights.get(&format!("{pfx}.mlp.switch_mlp.gate_proj.weight"))
                    {
                        moe.switch_mlp_gate_proj = Param::new(w.clone());
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.mlp.switch_mlp.up_proj.weight")) {
                        moe.switch_mlp_up_proj = Param::new(w.clone());
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.mlp.switch_mlp.down_proj.weight"))
                    {
                        moe.switch_mlp_down_proj = Param::new(w.clone());
                    }
                    // Frozen shared expert gate
                    if let Some(w) = weights.get(&format!("{pfx}.mlp.shared_expert_gate.weight")) {
                        moe.shared_expert_gate.weight = Param::new(w.clone());
                    }
                    // LoRA'd shared expert
                    let se = &mut moe.shared_expert;
                    if let Some(w) =
                        weights.get(&format!("{pfx}.mlp.shared_expert.gate_proj.weight"))
                    {
                        se.gate_proj.weight = w.clone();
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.mlp.shared_expert.up_proj.weight"))
                    {
                        se.up_proj.weight = w.clone();
                    }
                    if let Some(w) =
                        weights.get(&format!("{pfx}.mlp.shared_expert.down_proj.weight"))
                    {
                        se.down_proj.weight = w.clone();
                    }
                }
            }
        }

        // Final norm
        if let Some(w) = weights.get("model.norm.weight") {
            self.model.norm.weight = Param::new(w.clone());
        }

        // LM head (absent when tie_word_embeddings = true)
        if let Some(ref mut lm_head) = self.lm_head {
            if let Some(w) = weights.get("lm_head.weight") {
                lm_head.weight = Param::new(w.clone());
            }
        }

        Ok(())
    }

    /// Load base weights from SafeTensors files in a directory.
    ///
    /// Handles both single-file (`model.safetensors`) and sharded
    /// (`model.safetensors.index.json` + shards) layouts.
    pub fn load_base_weights_from_dir(
        &mut self,
        model_dir: impl AsRef<std::path::Path>,
    ) -> Result<(), LoraError> {
        let model_dir = model_dir.as_ref();

        let single_file = model_dir.join("model.safetensors");
        if single_file.exists() {
            let mut weights =
                crate::sanitize_loaded_weights(Array::load_safetensors(&single_file)?)?;
            sanitize_weights(&mut weights, &self.model.config).map_err(LoraError::Mlx)?;
            return self.load_base_weights(&weights);
        }

        let index_path = model_dir.join("model.safetensors.index.json");
        if !index_path.exists() {
            return Err(LoraError::Mlx(Exception::custom(
                "No model.safetensors or model.safetensors.index.json found in model directory"
                    .to_string(),
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

        let mut all_weights: HashMap<String, Array> = HashMap::new();
        for shard_file in shard_files {
            let shard_path = model_dir.join(shard_file);
            let shard_weights =
                crate::sanitize_loaded_weights(Array::load_safetensors(&shard_path)?)?;
            all_weights.extend(shard_weights);
        }

        sanitize_weights(&mut all_weights, &self.model.config).map_err(LoraError::Mlx)?;
        self.load_base_weights(&all_weights)
    }

    /// Force evaluation of all parameters (materialise lazy tensors).
    pub fn eval_all(&self) -> Result<(), LoraError> {
        self.model.embed_tokens.weight.value.as_ref().eval()?;

        for layer in &self.model.layers {
            layer.input_layernorm.weight.value.as_ref().eval()?;
            layer
                .post_attention_layernorm
                .weight
                .value
                .as_ref()
                .eval()?;

            if let Some(ref attn) = layer.self_attn {
                attn.q_proj.weight.eval()?;
                attn.k_proj.weight.eval()?;
                attn.v_proj.weight.eval()?;
                attn.o_proj.weight.eval()?;
                attn.q_norm.weight.value.as_ref().eval()?;
                attn.k_norm.weight.value.as_ref().eval()?;
                // LoRA adapters
                attn.q_proj.lora_a.eval()?;
                attn.q_proj.lora_b.eval()?;
                attn.k_proj.lora_a.eval()?;
                attn.k_proj.lora_b.eval()?;
                attn.v_proj.lora_a.eval()?;
                attn.v_proj.lora_b.eval()?;
                attn.o_proj.lora_a.eval()?;
                attn.o_proj.lora_b.eval()?;
            }
            if let Some(ref gdn) = layer.linear_attn {
                gdn.in_proj_qkv.weight.eval()?;
                gdn.in_proj_z.weight.eval()?;
                gdn.out_proj.weight.eval()?;
                // LoRA adapters
                gdn.in_proj_qkv.lora_a.eval()?;
                gdn.in_proj_qkv.lora_b.eval()?;
                gdn.in_proj_z.lora_a.eval()?;
                gdn.in_proj_z.lora_b.eval()?;
                gdn.out_proj.lora_a.eval()?;
                gdn.out_proj.lora_b.eval()?;
                // Frozen components
                gdn.in_proj_b.weight.value.as_ref().eval()?;
                gdn.in_proj_a.weight.value.as_ref().eval()?;
                gdn.conv1d.weight.value.as_ref().eval()?;
                gdn.dt_bias.value.as_ref().eval()?;
                gdn.a_log.value.as_ref().eval()?;
                gdn.norm.weight.value.as_ref().eval()?;
            }

            match &layer.mlp {
                Qwen3NextLoraFeedForward::Dense(mlp) => {
                    mlp.gate_proj.weight.eval()?;
                    mlp.up_proj.weight.eval()?;
                    mlp.down_proj.weight.eval()?;
                    mlp.gate_proj.lora_a.eval()?;
                    mlp.gate_proj.lora_b.eval()?;
                    mlp.up_proj.lora_a.eval()?;
                    mlp.up_proj.lora_b.eval()?;
                    mlp.down_proj.lora_a.eval()?;
                    mlp.down_proj.lora_b.eval()?;
                }
                Qwen3NextLoraFeedForward::MoE(moe) => {
                    moe.gate.weight.value.as_ref().eval()?;
                    moe.switch_mlp_gate_proj.value.as_ref().eval()?;
                    moe.switch_mlp_up_proj.value.as_ref().eval()?;
                    moe.switch_mlp_down_proj.value.as_ref().eval()?;
                    moe.shared_expert_gate.weight.value.as_ref().eval()?;
                    let se = &moe.shared_expert;
                    se.gate_proj.weight.eval()?;
                    se.up_proj.weight.eval()?;
                    se.down_proj.weight.eval()?;
                    se.gate_proj.lora_a.eval()?;
                    se.gate_proj.lora_b.eval()?;
                    se.up_proj.lora_a.eval()?;
                    se.up_proj.lora_b.eval()?;
                    se.down_proj.lora_a.eval()?;
                    se.down_proj.lora_b.eval()?;
                }
            }
        }

        self.model.norm.weight.value.as_ref().eval()?;

        if let Some(ref lm_head) = self.lm_head {
            lm_head.weight.value.as_ref().eval()?;
        }

        Ok(())
    }

    /// Merge LoRA weights into base weights for deployment.
    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        for layer in &mut self.model.layers {
            if let Some(ref mut attn) = layer.self_attn {
                attn.q_proj.merge()?;
                attn.k_proj.merge()?;
                attn.v_proj.merge()?;
                attn.o_proj.merge()?;
            }
            if let Some(ref mut gdn) = layer.linear_attn {
                gdn.in_proj_qkv.merge()?;
                gdn.in_proj_z.merge()?;
                gdn.out_proj.merge()?;
            }
            match &mut layer.mlp {
                Qwen3NextLoraFeedForward::Dense(mlp) => {
                    mlp.gate_proj.merge()?;
                    mlp.up_proj.merge()?;
                    mlp.down_proj.merge()?;
                }
                Qwen3NextLoraFeedForward::MoE(moe) => {
                    let se = &mut moe.shared_expert;
                    se.gate_proj.merge()?;
                    se.up_proj.merge()?;
                    se.down_proj.merge()?;
                }
            }
        }
        Ok(())
    }

    /// Unmerge is not reversible — reload base weights to undo a merge.
    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        Err(LoraError::InvalidState(
            "unmerge_lora is not supported: reload base model weights to undo a merge".to_string(),
        ))
    }
}

// ============================================================================
// ModuleParameters for Qwen3NextLoraForCausalLM
// ============================================================================

impl ModuleParameters for Qwen3NextLoraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.model.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();

        for (i, layer) in self.model.layers.iter().enumerate() {
            let layer_key: Rc<str> = Rc::from(format!("layers.{i}"));
            let mut layer_map = HashMap::new();

            // --- Mixer parameters ---
            if let Some(ref attn) = layer.self_attn {
                let mut attn_map = HashMap::new();

                let mut q_params = HashMap::new();
                q_params.insert(Rc::from("lora_a"), NestedValue::Value(&attn.q_proj.lora_a));
                q_params.insert(Rc::from("lora_b"), NestedValue::Value(&attn.q_proj.lora_b));
                attn_map.insert(Rc::from("q_proj"), NestedValue::Map(q_params));

                let mut k_params = HashMap::new();
                k_params.insert(Rc::from("lora_a"), NestedValue::Value(&attn.k_proj.lora_a));
                k_params.insert(Rc::from("lora_b"), NestedValue::Value(&attn.k_proj.lora_b));
                attn_map.insert(Rc::from("k_proj"), NestedValue::Map(k_params));

                let mut v_params = HashMap::new();
                v_params.insert(Rc::from("lora_a"), NestedValue::Value(&attn.v_proj.lora_a));
                v_params.insert(Rc::from("lora_b"), NestedValue::Value(&attn.v_proj.lora_b));
                attn_map.insert(Rc::from("v_proj"), NestedValue::Map(v_params));

                let mut o_params = HashMap::new();
                o_params.insert(Rc::from("lora_a"), NestedValue::Value(&attn.o_proj.lora_a));
                o_params.insert(Rc::from("lora_b"), NestedValue::Value(&attn.o_proj.lora_b));
                attn_map.insert(Rc::from("o_proj"), NestedValue::Map(o_params));

                layer_map.insert(Rc::from("self_attn"), NestedValue::Map(attn_map));
            }
            if let Some(ref gdn) = layer.linear_attn {
                let mut gdn_map = HashMap::new();

                let mut qkv_params = HashMap::new();
                qkv_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&gdn.in_proj_qkv.lora_a),
                );
                qkv_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&gdn.in_proj_qkv.lora_b),
                );
                gdn_map.insert(Rc::from("in_proj_qkv"), NestedValue::Map(qkv_params));

                let mut z_params = HashMap::new();
                z_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&gdn.in_proj_z.lora_a),
                );
                z_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&gdn.in_proj_z.lora_b),
                );
                gdn_map.insert(Rc::from("in_proj_z"), NestedValue::Map(z_params));

                let mut out_params = HashMap::new();
                out_params.insert(Rc::from("lora_a"), NestedValue::Value(&gdn.out_proj.lora_a));
                out_params.insert(Rc::from("lora_b"), NestedValue::Value(&gdn.out_proj.lora_b));
                gdn_map.insert(Rc::from("out_proj"), NestedValue::Map(out_params));

                layer_map.insert(Rc::from("linear_attn"), NestedValue::Map(gdn_map));
            }

            // --- MLP parameters ---
            let mut mlp_map = HashMap::new();
            match &layer.mlp {
                Qwen3NextLoraFeedForward::Dense(mlp) => {
                    let mut gate_params = HashMap::new();
                    gate_params.insert(
                        Rc::from("lora_a"),
                        NestedValue::Value(&mlp.gate_proj.lora_a),
                    );
                    gate_params.insert(
                        Rc::from("lora_b"),
                        NestedValue::Value(&mlp.gate_proj.lora_b),
                    );
                    mlp_map.insert(Rc::from("gate_proj"), NestedValue::Map(gate_params));

                    let mut up_params = HashMap::new();
                    up_params.insert(Rc::from("lora_a"), NestedValue::Value(&mlp.up_proj.lora_a));
                    up_params.insert(Rc::from("lora_b"), NestedValue::Value(&mlp.up_proj.lora_b));
                    mlp_map.insert(Rc::from("up_proj"), NestedValue::Map(up_params));

                    let mut down_params = HashMap::new();
                    down_params.insert(
                        Rc::from("lora_a"),
                        NestedValue::Value(&mlp.down_proj.lora_a),
                    );
                    down_params.insert(
                        Rc::from("lora_b"),
                        NestedValue::Value(&mlp.down_proj.lora_b),
                    );
                    mlp_map.insert(Rc::from("down_proj"), NestedValue::Map(down_params));
                }
                Qwen3NextLoraFeedForward::MoE(moe) => {
                    let se = &moe.shared_expert;
                    let mut se_map = HashMap::new();

                    let mut gate_params = HashMap::new();
                    gate_params
                        .insert(Rc::from("lora_a"), NestedValue::Value(&se.gate_proj.lora_a));
                    gate_params
                        .insert(Rc::from("lora_b"), NestedValue::Value(&se.gate_proj.lora_b));
                    se_map.insert(Rc::from("gate_proj"), NestedValue::Map(gate_params));

                    let mut up_params = HashMap::new();
                    up_params.insert(Rc::from("lora_a"), NestedValue::Value(&se.up_proj.lora_a));
                    up_params.insert(Rc::from("lora_b"), NestedValue::Value(&se.up_proj.lora_b));
                    se_map.insert(Rc::from("up_proj"), NestedValue::Map(up_params));

                    let mut down_params = HashMap::new();
                    down_params
                        .insert(Rc::from("lora_a"), NestedValue::Value(&se.down_proj.lora_a));
                    down_params
                        .insert(Rc::from("lora_b"), NestedValue::Value(&se.down_proj.lora_b));
                    se_map.insert(Rc::from("down_proj"), NestedValue::Map(down_params));

                    mlp_map.insert(Rc::from("shared_expert"), NestedValue::Map(se_map));
                }
            }
            layer_map.insert(Rc::from("mlp"), NestedValue::Map(mlp_map));

            params.insert(layer_key, NestedValue::Map(layer_map));
        }

        params
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut params = ModuleParamMut::new();

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let layer_key: Rc<str> = Rc::from(format!("layers.{i}"));
            let mut layer_map = HashMap::new();

            if let Some(ref mut attn) = layer.self_attn {
                let mut attn_map = HashMap::new();

                let mut q_params = HashMap::new();
                q_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&mut attn.q_proj.lora_a),
                );
                q_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&mut attn.q_proj.lora_b),
                );
                attn_map.insert(Rc::from("q_proj"), NestedValue::Map(q_params));

                let mut k_params = HashMap::new();
                k_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&mut attn.k_proj.lora_a),
                );
                k_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&mut attn.k_proj.lora_b),
                );
                attn_map.insert(Rc::from("k_proj"), NestedValue::Map(k_params));

                let mut v_params = HashMap::new();
                v_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&mut attn.v_proj.lora_a),
                );
                v_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&mut attn.v_proj.lora_b),
                );
                attn_map.insert(Rc::from("v_proj"), NestedValue::Map(v_params));

                let mut o_params = HashMap::new();
                o_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&mut attn.o_proj.lora_a),
                );
                o_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&mut attn.o_proj.lora_b),
                );
                attn_map.insert(Rc::from("o_proj"), NestedValue::Map(o_params));

                layer_map.insert(Rc::from("self_attn"), NestedValue::Map(attn_map));
            }
            if let Some(ref mut gdn) = layer.linear_attn {
                let mut gdn_map = HashMap::new();

                let mut qkv_params = HashMap::new();
                qkv_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&mut gdn.in_proj_qkv.lora_a),
                );
                qkv_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&mut gdn.in_proj_qkv.lora_b),
                );
                gdn_map.insert(Rc::from("in_proj_qkv"), NestedValue::Map(qkv_params));

                let mut z_params = HashMap::new();
                z_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&mut gdn.in_proj_z.lora_a),
                );
                z_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&mut gdn.in_proj_z.lora_b),
                );
                gdn_map.insert(Rc::from("in_proj_z"), NestedValue::Map(z_params));

                let mut out_params = HashMap::new();
                out_params.insert(
                    Rc::from("lora_a"),
                    NestedValue::Value(&mut gdn.out_proj.lora_a),
                );
                out_params.insert(
                    Rc::from("lora_b"),
                    NestedValue::Value(&mut gdn.out_proj.lora_b),
                );
                gdn_map.insert(Rc::from("out_proj"), NestedValue::Map(out_params));

                layer_map.insert(Rc::from("linear_attn"), NestedValue::Map(gdn_map));
            }

            let mut mlp_map = HashMap::new();
            match &mut layer.mlp {
                Qwen3NextLoraFeedForward::Dense(mlp) => {
                    let mut gate_params = HashMap::new();
                    gate_params.insert(
                        Rc::from("lora_a"),
                        NestedValue::Value(&mut mlp.gate_proj.lora_a),
                    );
                    gate_params.insert(
                        Rc::from("lora_b"),
                        NestedValue::Value(&mut mlp.gate_proj.lora_b),
                    );
                    mlp_map.insert(Rc::from("gate_proj"), NestedValue::Map(gate_params));

                    let mut up_params = HashMap::new();
                    up_params.insert(
                        Rc::from("lora_a"),
                        NestedValue::Value(&mut mlp.up_proj.lora_a),
                    );
                    up_params.insert(
                        Rc::from("lora_b"),
                        NestedValue::Value(&mut mlp.up_proj.lora_b),
                    );
                    mlp_map.insert(Rc::from("up_proj"), NestedValue::Map(up_params));

                    let mut down_params = HashMap::new();
                    down_params.insert(
                        Rc::from("lora_a"),
                        NestedValue::Value(&mut mlp.down_proj.lora_a),
                    );
                    down_params.insert(
                        Rc::from("lora_b"),
                        NestedValue::Value(&mut mlp.down_proj.lora_b),
                    );
                    mlp_map.insert(Rc::from("down_proj"), NestedValue::Map(down_params));
                }
                Qwen3NextLoraFeedForward::MoE(moe) => {
                    let se = &mut moe.shared_expert;
                    let mut se_map = HashMap::new();

                    let mut gate_params = HashMap::new();
                    gate_params.insert(
                        Rc::from("lora_a"),
                        NestedValue::Value(&mut se.gate_proj.lora_a),
                    );
                    gate_params.insert(
                        Rc::from("lora_b"),
                        NestedValue::Value(&mut se.gate_proj.lora_b),
                    );
                    se_map.insert(Rc::from("gate_proj"), NestedValue::Map(gate_params));

                    let mut up_params = HashMap::new();
                    up_params.insert(
                        Rc::from("lora_a"),
                        NestedValue::Value(&mut se.up_proj.lora_a),
                    );
                    up_params.insert(
                        Rc::from("lora_b"),
                        NestedValue::Value(&mut se.up_proj.lora_b),
                    );
                    se_map.insert(Rc::from("up_proj"), NestedValue::Map(up_params));

                    let mut down_params = HashMap::new();
                    down_params.insert(
                        Rc::from("lora_a"),
                        NestedValue::Value(&mut se.down_proj.lora_a),
                    );
                    down_params.insert(
                        Rc::from("lora_b"),
                        NestedValue::Value(&mut se.down_proj.lora_b),
                    );
                    se_map.insert(Rc::from("down_proj"), NestedValue::Map(down_params));

                    mlp_map.insert(Rc::from("shared_expert"), NestedValue::Map(se_map));
                }
            }
            layer_map.insert(Rc::from("mlp"), NestedValue::Map(mlp_map));

            params.insert(layer_key, NestedValue::Map(layer_map));
        }

        params
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        self.parameters()
    }

    fn freeze_parameters(&mut self, _recurse: bool) {}
    fn unfreeze_parameters(&mut self, _recurse: bool) {}
    fn all_frozen(&self) -> Option<bool> {
        Some(false)
    }
    fn any_frozen(&self) -> Option<bool> {
        Some(false)
    }
}

// ============================================================================
// TrainableModel for Qwen3NextLoraForCausalLM
// ============================================================================

impl crate::TrainableModel for Qwen3NextLoraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        Qwen3NextLoraForCausalLM::forward(self, input_ids, mask)
    }

    fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        _position_ids: &Array,
    ) -> Result<Array, LoraError> {
        // Hybrid models do not use explicit position IDs — GDN layers use
        // recurrent state and full attention layers use implicit rope offsets.
        Qwen3NextLoraForCausalLM::forward(self, input_ids, mask)
    }

    fn num_trainable_params(&self) -> usize {
        Qwen3NextLoraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        Qwen3NextLoraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        Qwen3NextLoraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        Qwen3NextLoraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        Qwen3NextLoraForCausalLM::load_lora_weights(self, path)
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        Qwen3NextLoraForCausalLM::enable_gradient_checkpointing(self, layers_per_block)
    }

    fn disable_gradient_checkpointing(&mut self) {
        Qwen3NextLoraForCausalLM::disable_gradient_checkpointing(self)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        true
    }

    fn supports_kv_cache(&self) -> bool {
        true
    }

    fn create_cache(&self, max_seq_len: usize) -> Option<KVCache> {
        Some(Qwen3NextLoraForCausalLM::create_cache(self, max_seq_len))
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Expand KV heads for grouped query attention (GQA).
fn expand_kv_heads(x: &Array, repeats: i32) -> Result<Array, LoraError> {
    let shape = x.shape();
    let batch = shape[0];
    let n_kv_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    let x = x.reshape(&[batch, n_kv_heads, 1, seq_len, head_dim])?;
    let x = ops::broadcast_to(&x, &[batch, n_kv_heads, repeats, seq_len, head_dim])?;
    Ok(x.reshape(&[batch, n_kv_heads * repeats, seq_len, head_dim])?)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Qwen3NextConfig {
        Qwen3NextConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: Some(16),
            vocab_size: 100,
            linear_num_value_heads: 2,
            linear_num_key_heads: 1,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            num_experts: 0,
            num_experts_per_tok: 0,
            decoder_sparse_step: 1,
            moe_intermediate_size: 16,
            shared_expert_intermediate_size: 32,
            mlp_only_layers: vec![],
            norm_topk_prob: false,
            tie_word_embeddings: true,
            ..Default::default()
        }
    }

    fn tiny_lora_config() -> LoraConfig {
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
                "in_proj_qkv".to_string(),
                "in_proj_z".to_string(),
                "out_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
            bias: pmetal_core::LoraBias::None,
            init_lora_weights: true,
            use_dora: false,
        }
    }

    #[test]
    fn test_qwen3_next_lora_construction() {
        let config = tiny_config();
        let lora_config = tiny_lora_config();
        let model = Qwen3NextLoraForCausalLM::new(config, lora_config);
        assert!(model.is_ok(), "Model construction should succeed");
        let model = model.unwrap();
        assert!(
            model.num_trainable_params() > 0,
            "Should have trainable parameters"
        );
    }

    #[test]
    fn test_lora_param_keys() {
        let config = tiny_config();
        let lora_config = tiny_lora_config();
        let model = Qwen3NextLoraForCausalLM::new(config, lora_config).unwrap();
        let params = model.lora_parameters();

        // Layers 0, 1, 2 are linear (GDN), layer 3 is full attention
        // (full_attention_interval = 4)
        assert!(
            params.contains_key(&Rc::from("layers.0.linear_attn.in_proj_qkv.lora_a")),
            "GDN layer should have linear_attn.in_proj_qkv.lora_a"
        );
        assert!(
            params.contains_key(&Rc::from("layers.3.self_attn.q_proj.lora_a")),
            "Full attention layer should have q_proj.lora_a"
        );
    }

    #[test]
    fn test_layer_type_dispatch() {
        let config = tiny_config();
        // layers 0, 1, 2 should be linear; layer 3 should be full attention
        assert!(config.is_linear_layer(0));
        assert!(config.is_linear_layer(1));
        assert!(config.is_linear_layer(2));
        assert!(!config.is_linear_layer(3));
    }

    #[test]
    fn test_qwen3_next_lora_forward() {
        let config = tiny_config();
        let lora_config = tiny_lora_config();
        let mut model = Qwen3NextLoraForCausalLM::new(config, lora_config).unwrap();

        let input_ids = Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let result = model.forward(&input_ids, None);
        assert!(
            result.is_ok(),
            "Forward pass should succeed: {:?}",
            result.err()
        );
        let logits = result.unwrap();
        assert_eq!(logits.shape(), &[1, 4, 100], "Logits shape mismatch");
    }

    #[test]
    fn test_set_lora_parameters_roundtrip() {
        let config = tiny_config();
        let lora_config = tiny_lora_config();
        let mut model = Qwen3NextLoraForCausalLM::new(config, lora_config).unwrap();

        let original_params = model.lora_parameters();
        model.set_lora_parameters(&original_params);
        let restored_params = model.lora_parameters();

        assert_eq!(
            original_params.len(),
            restored_params.len(),
            "Parameter count should be preserved"
        );
    }
}
