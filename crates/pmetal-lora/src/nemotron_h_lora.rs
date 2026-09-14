//! LoRA-enabled Nemotron-H hybrid architecture.
//!
//! Implements Nemotron-H with LoRA adapters for efficient fine-tuning.
//!
//! LoRA placement strategy:
//! - **Attention layers** (`*` blocks): LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`.
//! - **Mamba layers** (`M` blocks): ALL components stay frozen — SSM state-space
//!   parameters are not appropriate targets for LoRA.
//! - **MLP layers** (`-` blocks): LoRA on `up_proj`, `down_proj`.
//!   Note: NemotronH MLP uses relu² — no gate_proj.
//! - **MoE layers** (`E` blocks): LoRA on `shared_expert` only (`up_proj`, `down_proj`).
//!   Routed experts stay frozen.

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Once;

use pmetal_bridge::compat::{
    Array, Module, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, Param, nn, ops,
};

use pmetal_core::LoraConfig;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_mlx::kernels::{AttentionMaskType, FusedAttentionConfig, fused_sdpa, rope::apply_rope};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig, MambaCache, MambaCacheEntry};
use pmetal_models::architectures::nemotron_h::{
    Expert, MambaRMSNormGated, MoELayer, NemotronHConfig, NemotronHMixer, load_nemotron_weights,
    ssm_attention, ssm_update_single,
};

use crate::{LoraError, LoraLinear};

static GRAD_CKPT_WARN: Once = Once::new();

// ============================================================================
// NemotronHLoraAttention — standard GQA attention with LoRA
// ============================================================================

/// LoRA-enabled attention mixer for Nemotron-H `*` blocks.
///
/// No gated output, no per-head Q/K norms — standard multi-head attention
/// with RoPE and grouped-query attention (GQA).
#[derive(Debug)]
pub struct NemotronHLoraAttention {
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_theta: f32,

    pub q_proj: LoraLinear,
    pub k_proj: LoraLinear,
    pub v_proj: LoraLinear,
    pub o_proj: LoraLinear,
}

impl NemotronHLoraAttention {
    pub fn new(config: &NemotronHConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let head_dim = config.attention_head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let scale = (head_dim as f32).sqrt().recip();

        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;
        let use_dora = lora_config.use_dora;

        let q_rank = crate::effective_rank(lora_config, "q_proj") as i32;
        let k_rank = crate::effective_rank(lora_config, "k_proj") as i32;
        let v_rank = crate::effective_rank(lora_config, "v_proj") as i32;
        let o_rank = crate::effective_rank(lora_config, "o_proj") as i32;

        let q_proj = LoraLinear::new(
            config.hidden_size,
            num_heads * head_dim,
            q_rank,
            alpha,
            use_rslora,
            use_dora,
        )?;
        let k_proj = LoraLinear::new(
            config.hidden_size,
            num_kv_heads * head_dim,
            k_rank,
            alpha,
            use_rslora,
            use_dora,
        )?;
        let v_proj = LoraLinear::new(
            config.hidden_size,
            num_kv_heads * head_dim,
            v_rank,
            alpha,
            use_rslora,
            use_dora,
        )?;
        let o_proj = LoraLinear::new(
            num_heads * head_dim,
            config.hidden_size,
            o_rank,
            alpha,
            use_rslora,
            use_dora,
        )?;

        Ok(Self {
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            rope_theta: config.rope_theta,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }

    /// Training forward (no KV cache).
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        self.forward_impl(x, mask, None)
    }

    /// Cache-aware forward for inference.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        self.forward_impl(x, mask, cache)
    }

    fn forward_impl(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape(&[batch, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim]);
        let v = v.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim]);

        let q = q.transpose_axes(&[0, 2, 1, 3]);
        let k = k.transpose_axes(&[0, 2, 1, 3]);
        let v = v.transpose_axes(&[0, 2, 1, 3]);

        let offset = cache
            .as_ref()
            .map_or(0, |(c, layer)| c.rope_offset_for(*layer));
        let q = apply_rope(&q, self.head_dim, false, self.rope_theta, 1.0, offset)
            .map_err(LoraError::Mlx)?;
        let k = apply_rope(&k, self.head_dim, false, self.rope_theta, 1.0, offset)
            .map_err(LoraError::Mlx)?;

        let (k, v) = if let Some((cache, layer_idx)) = cache {
            cache
                .update_and_fetch(layer_idx, &k, &v)
                .map_err(LoraError::Mlx)?
        } else {
            (k, v)
        };

        let attn_config =
            FusedAttentionConfig::new(self.num_heads, self.num_kv_heads, self.head_dim)
                .with_scale(self.scale)
                .with_mask_type(AttentionMaskType::Causal);

        let output = fused_sdpa(&q, &k, &v, &attn_config, mask).map_err(LoraError::Mlx)?;

        let output = output.transpose_axes(&[0, 2, 1, 3]).reshape(&[
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ]);

        self.o_proj.forward(&output)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self.v_proj.num_trainable_params()
            + self.o_proj.num_trainable_params()
    }
}

// ============================================================================
// NemotronHLoraMLP — relu² MLP with LoRA on up/down
// ============================================================================

/// LoRA-enabled MLP for Nemotron-H `-` blocks.
///
/// Uses relu² activation: `down_proj(relu(up_proj(x))^2)`.
/// No `gate_proj` — this is NOT a SwiGLU block.
#[derive(Debug)]
pub struct NemotronHLoraMLP {
    pub up_proj: LoraLinear,
    pub down_proj: LoraLinear,
}

impl NemotronHLoraMLP {
    pub fn new(dim: i32, hidden_dim: i32, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;
        let use_dora = lora_config.use_dora;
        let up_rank = crate::effective_rank(lora_config, "up_proj") as i32;
        let down_rank = crate::effective_rank(lora_config, "down_proj") as i32;

        let up_proj = LoraLinear::new(dim, hidden_dim, up_rank, alpha, use_rslora, use_dora)?;
        let down_proj = LoraLinear::new(hidden_dim, dim, down_rank, alpha, use_rslora, use_dora)?;

        Ok(Self { up_proj, down_proj })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let up = self.up_proj.forward(x)?;
        // relu² activation matching the base model
        let activated = nn::relu(&up).square();
        self.down_proj.forward(&activated)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.up_proj.num_trainable_params() + self.down_proj.num_trainable_params()
    }
}

// ============================================================================
// NemotronHLoraSharedExpert — LoRA on shared expert within MoE
// ============================================================================

/// LoRA-enabled shared expert for Nemotron-H `E` blocks.
///
/// Routed experts stay frozen; only the shared expert receives LoRA adapters.
/// Shared expert also uses relu² (no gate_proj).
#[derive(Debug)]
pub struct NemotronHLoraSharedExpert {
    pub up_proj: LoraLinear,
    pub down_proj: LoraLinear,
}

impl NemotronHLoraSharedExpert {
    pub fn new(
        hidden_size: i32,
        intermediate_size: i32,
        lora_config: &LoraConfig,
    ) -> Result<Self, LoraError> {
        let inner = NemotronHLoraMLP::new(hidden_size, intermediate_size, lora_config)?;
        Ok(Self {
            up_proj: inner.up_proj,
            down_proj: inner.down_proj,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let up = self.up_proj.forward(x)?;
        let activated = nn::relu(&up).square();
        self.down_proj.forward(&activated)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.up_proj.num_trainable_params() + self.down_proj.num_trainable_params()
    }
}

// ============================================================================
// NemotronHLoraMoE — frozen routed experts + LoRA'd shared expert
// ============================================================================

/// MoE block with frozen routed experts and LoRA on the shared expert.
///
/// Routing and all per-token expert computations stay frozen. Only the shared
/// expert (always active for every token) receives LoRA adapters.
#[derive(Debug)]
pub struct NemotronHLoraMoE {
    /// Frozen router + routed experts from base model.
    pub moe_layer: MoELayer,
    /// Stacked expert weights for gather_mm (optional optimisation).
    pub stacked_moe_up: Option<Array>,
    pub stacked_moe_down: Option<Array>,
    /// LoRA'd shared expert.
    pub shared_expert: Option<NemotronHLoraSharedExpert>,
}

impl NemotronHLoraMoE {
    pub fn new(config: &NemotronHConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let num_experts = config.n_routed_experts.unwrap_or(8);
        let top_k = config.num_experts_per_tok.unwrap_or(2);
        let n_group = config.n_group.unwrap_or(1);
        let topk_group = config.topk_group.unwrap_or(1);
        let norm_topk_prob = config.norm_topk_prob.unwrap_or(false);
        let moe_intermediate_size = config
            .moe_intermediate_size
            .unwrap_or(config.intermediate_size);
        let shared_intermediate_size = config.moe_shared_expert_intermediate_size;
        let use_shared_expert = config.n_shared_experts.unwrap_or(0) > 0;
        let routed_scaling_factor = config.routed_scaling_factor.unwrap_or(1.0);

        let moe_layer = MoELayer::new(
            config.hidden_size,
            moe_intermediate_size,
            shared_intermediate_size,
            num_experts,
            top_k,
            n_group,
            topk_group,
            norm_topk_prob,
            // We manage the shared expert ourselves via LoRA; pass false to the
            // base MoELayer so it does not allocate a frozen shared expert.
            false,
            routed_scaling_factor,
            config.mlp_bias,
        )
        .map_err(LoraError::Mlx)?;

        let shared_expert = if use_shared_expert {
            let intermediate = shared_intermediate_size.unwrap_or(moe_intermediate_size);
            Some(NemotronHLoraSharedExpert::new(
                config.hidden_size,
                intermediate,
                lora_config,
            )?)
        } else {
            None
        };

        Ok(Self {
            moe_layer,
            stacked_moe_up: None,
            stacked_moe_down: None,
            shared_expert,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        // Frozen routed-expert dispatch — returns [B, L, H]
        let routed_out = if let (Some(stacked_up), Some(stacked_down)) =
            (&self.stacked_moe_up, &self.stacked_moe_down)
        {
            self.moe_layer
                .forward_stacked(x, stacked_up, stacked_down)
                .map_err(LoraError::Mlx)?
        } else {
            self.moe_layer.forward(x).map_err(LoraError::Mlx)?
        };

        // LoRA'd shared expert (always active for all tokens).
        // MoELayer.forward returns [B, L, H]; flatten x to [B*L, H] for the
        // shared expert then reshape back to match.
        let out = if let Some(ref mut shared) = self.shared_expert {
            let orig_shape = x.shape();
            let batch_seq: i32 = orig_shape[..orig_shape.len() - 1].iter().product();
            let hidden = orig_shape[orig_shape.len() - 1];
            let x_flat = x.reshape(&[batch_seq, hidden]);
            let shared_out = shared.forward(&x_flat)?;
            // shared_out is [B*L, H]; reshape to [B, L, H] and add
            let shared_out = shared_out.reshape(routed_out.shape());
            routed_out.add(&shared_out)
        } else {
            routed_out
        };

        Ok(out)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.shared_expert
            .as_ref()
            .map(|s| s.num_trainable_params())
            .unwrap_or(0)
    }
}

// ============================================================================
// NemotronHLoraMixer — per-block dispatch enum
// ============================================================================

/// Mixer dispatch for a single NemotronH block.
///
/// Exactly one variant is active per layer, determined by `block_type`.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum NemotronHLoraMixer {
    /// Mamba-2 SSM — all components frozen.
    Mamba(NemotronHMixer),
    /// Full attention — q/k/v/o have LoRA adapters.
    Attention(NemotronHLoraAttention),
    /// Dense MLP — up/down have LoRA adapters.
    Mlp(NemotronHLoraMLP),
    /// Mixture-of-experts — shared expert has LoRA adapters.
    MoE(NemotronHLoraMoE),
}

impl NemotronHLoraMixer {
    pub fn block_type(&self) -> char {
        match self {
            Self::Mamba(_) => 'M',
            Self::Attention(_) => '*',
            Self::Mlp(_) => '-',
            Self::MoE(_) => 'E',
        }
    }

    pub fn is_mamba(&self) -> bool {
        matches!(self, Self::Mamba(_))
    }

    /// Training forward (no caches).
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        match self {
            Self::Mamba(m) => m
                .forward_with_cache(x, mask, None, None, None)
                .map_err(LoraError::Mlx),
            Self::Attention(a) => a.forward(x, mask),
            Self::Mlp(m) => m.forward(x),
            Self::MoE(m) => m.forward(x),
        }
    }

    /// Cache-aware forward for inference.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        mamba_cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, LoraError> {
        match self {
            Self::Mamba(m) => m
                .forward_with_cache(x, mask, None, None, mamba_cache)
                .map_err(LoraError::Mlx),
            Self::Attention(a) => a.forward_with_cache(x, mask, kv_cache),
            Self::Mlp(m) => m.forward(x),
            Self::MoE(m) => m.forward(x),
        }
    }

    pub fn num_trainable_params(&self) -> usize {
        match self {
            Self::Mamba(_) => 0,
            Self::Attention(a) => a.num_trainable_params(),
            Self::Mlp(m) => m.num_trainable_params(),
            Self::MoE(m) => m.num_trainable_params(),
        }
    }
}

// ============================================================================
// NemotronHLoraBlock — single transformer block
// ============================================================================

/// A single NemotronH block: pre-norm + mixer + residual.
///
/// NemotronH uses a **single** pre-norm (not pre + post like Qwen3Next).
#[derive(Debug)]
pub struct NemotronHLoraBlock {
    /// Pre-norm applied before the mixer (frozen).
    pub norm: nn::RmsNorm,
    pub mixer: NemotronHLoraMixer,
}

impl NemotronHLoraBlock {
    pub fn new(
        config: &NemotronHConfig,
        lora_config: &LoraConfig,
        block_type: char,
    ) -> Result<Self, LoraError> {
        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_epsilon)
            .build()
            .map_err(LoraError::Mlx)?;

        let mixer = match block_type {
            'M' => {
                let base = NemotronHMixer::new_mamba(config).map_err(LoraError::Mlx)?;
                NemotronHLoraMixer::Mamba(base)
            }
            '*' => NemotronHLoraMixer::Attention(NemotronHLoraAttention::new(config, lora_config)?),
            '-' => NemotronHLoraMixer::Mlp(NemotronHLoraMLP::new(
                config.hidden_size,
                config.intermediate_size,
                lora_config,
            )?),
            'E' => NemotronHLoraMixer::MoE(NemotronHLoraMoE::new(config, lora_config)?),
            // Treat unknown patterns as attention (matches base model default).
            _ => NemotronHLoraMixer::Attention(NemotronHLoraAttention::new(config, lora_config)?),
        };

        Ok(Self { norm, mixer })
    }

    /// Training forward.
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let normed = Module::forward(&mut self.norm, x)?;
        let r = self.mixer.forward(&normed, mask)?;
        Ok(x.add(&r))
    }

    /// Cache-aware forward.
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<(&mut KVCache, usize)>,
        mamba_cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, LoraError> {
        let normed = Module::forward(&mut self.norm, x)?;
        let r = self
            .mixer
            .forward_with_cache(&normed, mask, kv_cache, mamba_cache)?;
        Ok(x.add(&r))
    }

    pub fn num_trainable_params(&self) -> usize {
        self.mixer.num_trainable_params()
    }
}

// ============================================================================
// NemotronHLoraModel — backbone
// ============================================================================

/// Nemotron-H backbone with LoRA adapters.
#[derive(Debug)]
pub struct NemotronHLoraModel {
    pub config: NemotronHConfig,
    pub lora_config: LoraConfig,
    pub embeddings: nn::Embedding,
    pub layers: Vec<NemotronHLoraBlock>,
    pub norm_f: nn::RmsNorm,
}

impl NemotronHLoraModel {
    pub fn new(config: NemotronHConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let embeddings =
            nn::Embedding::new(config.vocab_size, config.hidden_size).map_err(LoraError::Mlx)?;

        let layer_types = config.layer_types();
        let layers = layer_types
            .iter()
            .map(|&bt| NemotronHLoraBlock::new(&config, &lora_config, bt))
            .collect::<Result<Vec<_>, _>>()?;

        let norm_f = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_epsilon)
            .build()
            .map_err(LoraError::Mlx)?;

        Ok(Self {
            config,
            lora_config,
            embeddings,
            layers,
            norm_f,
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
        let mut hidden = Module::forward(&mut self.embeddings, input_ids)?;

        let layers_per_block = checkpoint_config
            .map(|c| c.layers_per_block)
            .unwrap_or(usize::MAX);
        let checkpointing_enabled = checkpoint_config.map(|c| c.enabled).unwrap_or(false);

        // Attention layers get the causal mask; Mamba/MLP/MoE layers see None.
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let layer_mask = if layer.mixer.is_mamba() { None } else { mask };
            hidden = layer.forward(&hidden, layer_mask)?;

            if checkpointing_enabled && (idx + 1) % layers_per_block == 0 {
                GRAD_CKPT_WARN.call_once(|| {
                    tracing::info!(
                        "NemotronH uses eager evaluation for memory management \
                         (gradient checkpointing requires custom_vjp not yet in MLX-rs)"
                    );
                });
            }
        }

        Ok(Module::forward(&mut self.norm_f, &hidden)?)
    }

    /// Cache-aware forward for inference.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        mut kv_cache: Option<&mut KVCache>,
        mut mamba_cache: Option<&mut MambaCache>,
    ) -> Result<Array, LoraError> {
        let mut hidden = Module::forward(&mut self.embeddings, input_ids)?;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let kv = if matches!(layer.mixer, NemotronHLoraMixer::Attention(_)) {
                kv_cache.as_deref_mut().map(|c| (c, layer_idx))
            } else {
                None
            };
            let mamba = if layer.mixer.is_mamba() {
                mamba_cache
                    .as_deref_mut()
                    .and_then(|c| c.get_mut(layer_idx))
            } else {
                None
            };
            let layer_mask = if layer.mixer.is_mamba() { None } else { mask };
            hidden = layer.forward_with_cache(&hidden, layer_mask, kv, mamba)?;
        }

        Ok(Module::forward(&mut self.norm_f, &hidden)?)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_trainable_params()).sum()
    }
}

// ============================================================================
// NemotronHLoraForCausalLM — top-level model
// ============================================================================

/// Nemotron-H causal language model with LoRA adapters.
#[derive(Debug)]
pub struct NemotronHLoraForCausalLM {
    pub model: NemotronHLoraModel,
    /// LM head — absent when `tie_word_embeddings = true`.
    pub lm_head: Option<nn::Linear>,
    /// Gradient checkpointing configuration.
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl NemotronHLoraForCausalLM {
    pub fn new(config: NemotronHConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let tie_weights = config.tie_word_embeddings;
        let model = NemotronHLoraModel::new(config.clone(), lora_config)?;

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
        let hidden =
            self.model
                .forward_with_checkpoint(input_ids, mask, checkpoint_config.as_ref())?;
        self.lm_head_forward(&hidden)
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

    fn lm_head_forward(&mut self, h: &Array) -> Result<Array, LoraError> {
        if let Some(ref mut lm_head) = self.lm_head {
            Ok(Module::forward(lm_head, h)?)
        } else {
            Ok(self.model.embeddings.as_linear(h))
        }
    }

    /// Forward returning hidden states before lm_head (for Cut Cross-Entropy).
    pub fn forward_hidden_states(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
        let checkpoint_config = self.checkpoint_config.clone();
        self.model
            .forward_with_checkpoint(input_ids, mask, checkpoint_config.as_ref())
    }

    /// Return LM head weight for Cut Cross-Entropy.
    pub fn get_lm_head_weight(&self) -> Option<Array> {
        if let Some(ref lm_head) = self.lm_head {
            Some(lm_head.weight.value.clone())
        } else {
            Some(self.model.embeddings.weight.value.clone())
        }
    }

    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    pub fn config(&self) -> &NemotronHConfig {
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

            match &layer.mixer {
                NemotronHLoraMixer::Mamba(_) => {
                    // No trainable parameters in frozen Mamba layers.
                }
                NemotronHLoraMixer::Attention(attn) => {
                    for (proj_name, lora) in [
                        ("q_proj", &attn.q_proj),
                        ("k_proj", &attn.k_proj),
                        ("v_proj", &attn.v_proj),
                        ("o_proj", &attn.o_proj),
                    ] {
                        params.insert(
                            Rc::from(format!("{prefix}.mixer.{proj_name}.lora_a")),
                            lora.lora_a.clone(),
                        );
                        params.insert(
                            Rc::from(format!("{prefix}.mixer.{proj_name}.lora_b")),
                            lora.lora_b.clone(),
                        );
                    }
                }
                NemotronHLoraMixer::Mlp(mlp) => {
                    for (proj_name, lora) in
                        [("up_proj", &mlp.up_proj), ("down_proj", &mlp.down_proj)]
                    {
                        params.insert(
                            Rc::from(format!("{prefix}.mixer.{proj_name}.lora_a")),
                            lora.lora_a.clone(),
                        );
                        params.insert(
                            Rc::from(format!("{prefix}.mixer.{proj_name}.lora_b")),
                            lora.lora_b.clone(),
                        );
                    }
                }
                NemotronHLoraMixer::MoE(moe) => {
                    if let Some(ref se) = moe.shared_expert {
                        for (proj_name, lora) in
                            [("up_proj", &se.up_proj), ("down_proj", &se.down_proj)]
                        {
                            params.insert(
                                Rc::from(format!(
                                    "{prefix}.mixer.shared_expert.{proj_name}.lora_a"
                                )),
                                lora.lora_a.clone(),
                            );
                            params.insert(
                                Rc::from(format!(
                                    "{prefix}.mixer.shared_expert.{proj_name}.lora_b"
                                )),
                                lora.lora_b.clone(),
                            );
                        }
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

            match &mut layer.mixer {
                NemotronHLoraMixer::Mamba(_) => {}
                NemotronHLoraMixer::Attention(attn) => {
                    set_param!(attn.q_proj.lora_a, format!("{prefix}.mixer.q_proj.lora_a"));
                    set_param!(attn.q_proj.lora_b, format!("{prefix}.mixer.q_proj.lora_b"));
                    set_param!(attn.k_proj.lora_a, format!("{prefix}.mixer.k_proj.lora_a"));
                    set_param!(attn.k_proj.lora_b, format!("{prefix}.mixer.k_proj.lora_b"));
                    set_param!(attn.v_proj.lora_a, format!("{prefix}.mixer.v_proj.lora_a"));
                    set_param!(attn.v_proj.lora_b, format!("{prefix}.mixer.v_proj.lora_b"));
                    set_param!(attn.o_proj.lora_a, format!("{prefix}.mixer.o_proj.lora_a"));
                    set_param!(attn.o_proj.lora_b, format!("{prefix}.mixer.o_proj.lora_b"));
                }
                NemotronHLoraMixer::Mlp(mlp) => {
                    set_param!(mlp.up_proj.lora_a, format!("{prefix}.mixer.up_proj.lora_a"));
                    set_param!(mlp.up_proj.lora_b, format!("{prefix}.mixer.up_proj.lora_b"));
                    set_param!(
                        mlp.down_proj.lora_a,
                        format!("{prefix}.mixer.down_proj.lora_a")
                    );
                    set_param!(
                        mlp.down_proj.lora_b,
                        format!("{prefix}.mixer.down_proj.lora_b")
                    );
                }
                NemotronHLoraMixer::MoE(moe) => {
                    if let Some(ref mut se) = moe.shared_expert {
                        set_param!(
                            se.up_proj.lora_a,
                            format!("{prefix}.mixer.shared_expert.up_proj.lora_a")
                        );
                        set_param!(
                            se.up_proj.lora_b,
                            format!("{prefix}.mixer.shared_expert.up_proj.lora_b")
                        );
                        set_param!(
                            se.down_proj.lora_a,
                            format!("{prefix}.mixer.shared_expert.down_proj.lora_a")
                        );
                        set_param!(
                            se.down_proj.lora_b,
                            format!("{prefix}.mixer.shared_expert.down_proj.lora_b")
                        );
                    }
                }
            }
        }
    }

    /// Save LoRA adapters to a safetensors file.
    pub fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        let params = self.lora_parameters();
        crate::save_safetensors_map(path, &params)
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
        let loaded = crate::load_safetensors_map(&file_path)?;
        let params: HashMap<Rc<str>, Array> = loaded
            .into_iter()
            .map(|(k, v)| (Rc::from(k.as_str()), v))
            .collect();
        self.set_lora_parameters(&params);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Weight loading from SafeTensors
    // -------------------------------------------------------------------------

    /// Load base model weights from a HashMap.
    ///
    /// Weight name convention mirrors `load_nemotron_weights` in the base arch.
    /// Prefix: `backbone.layers.{i}.mixer.{component}.weight`
    pub fn load_base_weights(&mut self, weights: &HashMap<String, Array>) -> Result<(), LoraError> {
        // Embeddings
        for key in &["backbone.embeddings.weight", "model.embed_tokens.weight"] {
            if let Some(w) = weights.get(*key) {
                self.model.embeddings.weight = Param::new(w.clone());
                break;
            }
        }

        let layer_types = self.model.config.layer_types();

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let pfx = format!("backbone.layers.{i}");

            // Per-block pre-norm
            if let Some(w) = weights.get(&format!("{pfx}.norm.weight")) {
                layer.norm.weight = Param::new(w.clone());
            }

            let block_type = layer_types.get(i).copied().unwrap_or('*');

            match (&mut layer.mixer, block_type) {
                (NemotronHLoraMixer::Mamba(m), 'M') => {
                    // Delegate to the frozen base mixer's weight fields directly.
                    if let Some(ref mut in_proj) = m.in_proj {
                        if let Some(w) = weights.get(&format!("{pfx}.mixer.in_proj.weight")) {
                            in_proj.weight = Param::new(w.clone());
                        }
                    }
                    if let Some(ref mut conv1d) = m.conv1d {
                        if let Some(w) = weights.get(&format!("{pfx}.mixer.conv1d.weight")) {
                            conv1d.weight = Param::new(w.transpose_axes(&[0, 2, 1]));
                        }
                        if let Some(b) = weights.get(&format!("{pfx}.mixer.conv1d.bias")) {
                            conv1d.bias = Param::new(Some(b.clone()));
                        }
                    }
                    if let Some(ref mut out_proj) = m.out_proj {
                        if let Some(w) = weights.get(&format!("{pfx}.mixer.out_proj.weight")) {
                            out_proj.weight = Param::new(w.clone());
                        }
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.mixer.A_log")) {
                        m.a_log = Some(w.clone());
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.mixer.D")) {
                        m.d = Some(w.clone());
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.mixer.dt_bias")) {
                        m.dt_bias = Some(w.clone());
                    }
                    if let Some(ref mut gn) = m.gated_norm {
                        if let Some(w) = weights.get(&format!("{pfx}.mixer.norm.weight")) {
                            gn.weight = w.clone();
                        }
                    }
                }
                (NemotronHLoraMixer::Attention(attn), '*') => {
                    if let Some(w) = weights.get(&format!("{pfx}.mixer.q_proj.weight")) {
                        attn.q_proj.weight = w.clone();
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.mixer.k_proj.weight")) {
                        attn.k_proj.weight = w.clone();
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.mixer.v_proj.weight")) {
                        attn.v_proj.weight = w.clone();
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.mixer.o_proj.weight")) {
                        attn.o_proj.weight = w.clone();
                    }
                }
                (NemotronHLoraMixer::Mlp(mlp), '-') => {
                    if let Some(w) = weights.get(&format!("{pfx}.mixer.up_proj.weight")) {
                        mlp.up_proj.weight = w.clone();
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.mixer.down_proj.weight")) {
                        mlp.down_proj.weight = w.clone();
                    }
                }
                (NemotronHLoraMixer::MoE(moe), 'E') => {
                    // Router gate
                    if let Some(w) = weights.get(&format!("{pfx}.mixer.gate.weight")) {
                        moe.moe_layer.router.gate.weight = Param::new(w.clone());
                    }
                    if let Some(b) =
                        weights.get(&format!("{pfx}.mixer.gate.e_score_correction_bias"))
                    {
                        moe.moe_layer.router.e_score_correction_bias = b.clone();
                    }
                    // Frozen routed experts
                    for (idx, expert) in moe.moe_layer.experts.iter_mut().enumerate() {
                        if let Some(w) =
                            weights.get(&format!("{pfx}.mixer.experts.{idx}.up_proj.weight"))
                        {
                            expert.up_proj.weight = Param::new(w.clone());
                        }
                        if let Some(w) =
                            weights.get(&format!("{pfx}.mixer.experts.{idx}.down_proj.weight"))
                        {
                            expert.down_proj.weight = Param::new(w.clone());
                        }
                    }
                    // LoRA'd shared expert
                    if let Some(ref mut se) = moe.shared_expert {
                        if let Some(w) =
                            weights.get(&format!("{pfx}.mixer.shared_experts.up_proj.weight"))
                        {
                            se.up_proj.weight = w.clone();
                        }
                        if let Some(w) =
                            weights.get(&format!("{pfx}.mixer.shared_experts.down_proj.weight"))
                        {
                            se.down_proj.weight = w.clone();
                        }
                    }
                }
                _ => {}
            }
        }

        // Final norm
        for key in &["backbone.norm_f.weight", "model.norm.weight"] {
            if let Some(w) = weights.get(*key) {
                self.model.norm_f.weight = Param::new(w.clone());
                break;
            }
        }

        // LM head
        if let Some(ref mut lm_head) = self.lm_head {
            if let Some(w) = weights.get("lm_head.weight") {
                lm_head.weight = Param::new(w.clone());
            }
        }

        Ok(())
    }

    /// Load base weights from SafeTensors files in a directory.
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
            return Err(LoraError::Mlx(pmetal_bridge::compat::Exception::custom(
                "No model.safetensors or model.safetensors.index.json found".to_string(),
            )));
        }

        let index_content = std::fs::read_to_string(&index_path)
            .map_err(|e| LoraError::Mlx(pmetal_bridge::compat::Exception::custom(e.to_string())))?;

        #[derive(serde::Deserialize)]
        struct WeightIndex {
            weight_map: HashMap<String, String>,
        }

        let index: WeightIndex = serde_json::from_str(&index_content)
            .map_err(|e| LoraError::Mlx(pmetal_bridge::compat::Exception::custom(e.to_string())))?;

        let shard_files: std::collections::HashSet<&String> = index.weight_map.values().collect();
        let mut all_weights: HashMap<String, Array> = HashMap::new();
        for shard_file in shard_files {
            let shard_path = model_dir.join(shard_file);
            let shard_weights =
                crate::sanitize_loaded_weights(crate::load_safetensors_map(&shard_path)?)?;
            all_weights.extend(shard_weights);
        }

        self.load_base_weights(&all_weights)
    }

    /// Force evaluation of all parameters.
    pub fn eval_all(&mut self) -> Result<(), LoraError> {
        self.model.embeddings.weight.value.eval();

        for layer in &mut self.model.layers {
            layer.norm.weight.value.eval();

            match &mut layer.mixer {
                NemotronHLoraMixer::Mamba(_) => {
                    // Frozen; no LoRA adapters to eval.
                }
                NemotronHLoraMixer::Attention(attn) => {
                    attn.q_proj.weight.eval();
                    attn.k_proj.weight.eval();
                    attn.v_proj.weight.eval();
                    attn.o_proj.weight.eval();
                    attn.q_proj.lora_a.eval();
                    attn.q_proj.lora_b.eval();
                    attn.k_proj.lora_a.eval();
                    attn.k_proj.lora_b.eval();
                    attn.v_proj.lora_a.eval();
                    attn.v_proj.lora_b.eval();
                    attn.o_proj.lora_a.eval();
                    attn.o_proj.lora_b.eval();
                }
                NemotronHLoraMixer::Mlp(mlp) => {
                    mlp.up_proj.weight.eval();
                    mlp.down_proj.weight.eval();
                    mlp.up_proj.lora_a.eval();
                    mlp.up_proj.lora_b.eval();
                    mlp.down_proj.lora_a.eval();
                    mlp.down_proj.lora_b.eval();
                }
                NemotronHLoraMixer::MoE(moe) => {
                    if let Some(ref mut se) = moe.shared_expert {
                        se.up_proj.weight.eval();
                        se.down_proj.weight.eval();
                        se.up_proj.lora_a.eval();
                        se.up_proj.lora_b.eval();
                        se.down_proj.lora_a.eval();
                        se.down_proj.lora_b.eval();
                    }
                }
            }
        }

        self.model.norm_f.weight.value.eval();
        if let Some(ref mut lm_head) = self.lm_head {
            lm_head.weight.value.eval();
        }

        Ok(())
    }

    /// Merge LoRA weights into base weights for deployment.
    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        for layer in &mut self.model.layers {
            match &mut layer.mixer {
                NemotronHLoraMixer::Mamba(_) => {}
                NemotronHLoraMixer::Attention(attn) => {
                    attn.q_proj.merge()?;
                    attn.k_proj.merge()?;
                    attn.v_proj.merge()?;
                    attn.o_proj.merge()?;
                }
                NemotronHLoraMixer::Mlp(mlp) => {
                    mlp.up_proj.merge()?;
                    mlp.down_proj.merge()?;
                }
                NemotronHLoraMixer::MoE(moe) => {
                    if let Some(ref mut se) = moe.shared_expert {
                        se.up_proj.merge()?;
                        se.down_proj.merge()?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Unmerge is not reversible — reload base weights to undo.
    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        Err(LoraError::InvalidState(
            "unmerge_lora is not supported: reload base model weights to undo a merge".to_string(),
        ))
    }
}

// ============================================================================
// ModuleParameters for NemotronHLoraForCausalLM
// ============================================================================

impl ModuleParameters for NemotronHLoraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.model.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();

        for (i, layer) in self.model.layers.iter().enumerate() {
            let layer_key: Rc<str> = Rc::from(format!("layers.{i}"));
            let mut mixer_map = HashMap::new();

            match &layer.mixer {
                NemotronHLoraMixer::Mamba(_) => {}
                NemotronHLoraMixer::Attention(attn) => {
                    for (proj_name, lora) in [
                        ("q_proj", &attn.q_proj),
                        ("k_proj", &attn.k_proj),
                        ("v_proj", &attn.v_proj),
                        ("o_proj", &attn.o_proj),
                    ] {
                        let mut p = HashMap::new();
                        p.insert(Rc::from("lora_a"), NestedValue::Value(&lora.lora_a));
                        p.insert(Rc::from("lora_b"), NestedValue::Value(&lora.lora_b));
                        mixer_map.insert(Rc::from(proj_name), NestedValue::Map(p));
                    }
                }
                NemotronHLoraMixer::Mlp(mlp) => {
                    for (proj_name, lora) in
                        [("up_proj", &mlp.up_proj), ("down_proj", &mlp.down_proj)]
                    {
                        let mut p = HashMap::new();
                        p.insert(Rc::from("lora_a"), NestedValue::Value(&lora.lora_a));
                        p.insert(Rc::from("lora_b"), NestedValue::Value(&lora.lora_b));
                        mixer_map.insert(Rc::from(proj_name), NestedValue::Map(p));
                    }
                }
                NemotronHLoraMixer::MoE(moe) => {
                    if let Some(ref se) = moe.shared_expert {
                        let mut se_map = HashMap::new();
                        for (proj_name, lora) in
                            [("up_proj", &se.up_proj), ("down_proj", &se.down_proj)]
                        {
                            let mut p = HashMap::new();
                            p.insert(Rc::from("lora_a"), NestedValue::Value(&lora.lora_a));
                            p.insert(Rc::from("lora_b"), NestedValue::Value(&lora.lora_b));
                            se_map.insert(Rc::from(proj_name), NestedValue::Map(p));
                        }
                        mixer_map.insert(Rc::from("shared_expert"), NestedValue::Map(se_map));
                    }
                }
            }

            if !mixer_map.is_empty() {
                let mut layer_map = HashMap::new();
                layer_map.insert(Rc::from("mixer"), NestedValue::Map(mixer_map));
                params.insert(layer_key, NestedValue::Map(layer_map));
            }
        }

        params
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut params = ModuleParamMut::new();

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let layer_key: Rc<str> = Rc::from(format!("layers.{i}"));
            let mut mixer_map = HashMap::new();

            match &mut layer.mixer {
                NemotronHLoraMixer::Mamba(_) => {}
                NemotronHLoraMixer::Attention(attn) => {
                    let mut q = HashMap::new();
                    q.insert(
                        Rc::from("lora_a"),
                        NestedValue::Value(&mut attn.q_proj.lora_a),
                    );
                    q.insert(
                        Rc::from("lora_b"),
                        NestedValue::Value(&mut attn.q_proj.lora_b),
                    );
                    mixer_map.insert(Rc::from("q_proj"), NestedValue::Map(q));

                    let mut k = HashMap::new();
                    k.insert(
                        Rc::from("lora_a"),
                        NestedValue::Value(&mut attn.k_proj.lora_a),
                    );
                    k.insert(
                        Rc::from("lora_b"),
                        NestedValue::Value(&mut attn.k_proj.lora_b),
                    );
                    mixer_map.insert(Rc::from("k_proj"), NestedValue::Map(k));

                    let mut v = HashMap::new();
                    v.insert(
                        Rc::from("lora_a"),
                        NestedValue::Value(&mut attn.v_proj.lora_a),
                    );
                    v.insert(
                        Rc::from("lora_b"),
                        NestedValue::Value(&mut attn.v_proj.lora_b),
                    );
                    mixer_map.insert(Rc::from("v_proj"), NestedValue::Map(v));

                    let mut o = HashMap::new();
                    o.insert(
                        Rc::from("lora_a"),
                        NestedValue::Value(&mut attn.o_proj.lora_a),
                    );
                    o.insert(
                        Rc::from("lora_b"),
                        NestedValue::Value(&mut attn.o_proj.lora_b),
                    );
                    mixer_map.insert(Rc::from("o_proj"), NestedValue::Map(o));
                }
                NemotronHLoraMixer::Mlp(mlp) => {
                    let mut up = HashMap::new();
                    up.insert(
                        Rc::from("lora_a"),
                        NestedValue::Value(&mut mlp.up_proj.lora_a),
                    );
                    up.insert(
                        Rc::from("lora_b"),
                        NestedValue::Value(&mut mlp.up_proj.lora_b),
                    );
                    mixer_map.insert(Rc::from("up_proj"), NestedValue::Map(up));

                    let mut down = HashMap::new();
                    down.insert(
                        Rc::from("lora_a"),
                        NestedValue::Value(&mut mlp.down_proj.lora_a),
                    );
                    down.insert(
                        Rc::from("lora_b"),
                        NestedValue::Value(&mut mlp.down_proj.lora_b),
                    );
                    mixer_map.insert(Rc::from("down_proj"), NestedValue::Map(down));
                }
                NemotronHLoraMixer::MoE(moe) => {
                    if let Some(ref mut se) = moe.shared_expert {
                        let mut se_map = HashMap::new();

                        let mut up = HashMap::new();
                        up.insert(
                            Rc::from("lora_a"),
                            NestedValue::Value(&mut se.up_proj.lora_a),
                        );
                        up.insert(
                            Rc::from("lora_b"),
                            NestedValue::Value(&mut se.up_proj.lora_b),
                        );
                        se_map.insert(Rc::from("up_proj"), NestedValue::Map(up));

                        let mut down = HashMap::new();
                        down.insert(
                            Rc::from("lora_a"),
                            NestedValue::Value(&mut se.down_proj.lora_a),
                        );
                        down.insert(
                            Rc::from("lora_b"),
                            NestedValue::Value(&mut se.down_proj.lora_b),
                        );
                        se_map.insert(Rc::from("down_proj"), NestedValue::Map(down));

                        mixer_map.insert(Rc::from("shared_expert"), NestedValue::Map(se_map));
                    }
                }
            }

            if !mixer_map.is_empty() {
                let mut layer_map = HashMap::new();
                layer_map.insert(Rc::from("mixer"), NestedValue::Map(mixer_map));
                params.insert(layer_key, NestedValue::Map(layer_map));
            }
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
// TrainableModel for NemotronHLoraForCausalLM
// ============================================================================

impl crate::TrainableModel for NemotronHLoraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        NemotronHLoraForCausalLM::forward(self, input_ids, mask)
    }

    fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        _position_ids: &Array,
    ) -> Result<Array, LoraError> {
        // Hybrid models do not use explicit position IDs — Mamba layers use
        // recurrent state, attention layers use implicit RoPE offsets.
        NemotronHLoraForCausalLM::forward(self, input_ids, mask)
    }

    fn num_trainable_params(&self) -> usize {
        NemotronHLoraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        NemotronHLoraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        NemotronHLoraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        NemotronHLoraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        NemotronHLoraForCausalLM::load_lora_weights(self, path)
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        NemotronHLoraForCausalLM::enable_gradient_checkpointing(self, layers_per_block)
    }

    fn disable_gradient_checkpointing(&mut self) {
        NemotronHLoraForCausalLM::disable_gradient_checkpointing(self)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        false
    }

    /// NemotronH is a hybrid arch — KV cache alone is insufficient.
    /// Use `forward_with_cache` with both KV + Mamba caches for inference.
    fn supports_kv_cache(&self) -> bool {
        false
    }

    fn forward_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Option<Result<Array, LoraError>> {
        Some(NemotronHLoraForCausalLM::forward_hidden_states(
            self, input_ids, mask,
        ))
    }

    fn lm_head_weight(&self) -> Option<Array> {
        NemotronHLoraForCausalLM::get_lm_head_weight(self)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> NemotronHConfig {
        NemotronHConfig {
            model_type: "nemotron_h".to_string(),
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 3,
            max_position_embeddings: 512,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            attention_bias: false,
            head_dim: Some(16),
            mamba_num_heads: 4,
            mamba_head_dim: 8,
            mamba_proj_bias: false,
            ssm_state_size: 8,
            conv_kernel: 4,
            n_groups: 1,
            time_step_limit: (0.001, 0.1),
            time_step_min: None,
            time_step_max: None,
            mlp_bias: false,
            mlp_hidden_act: "relu2".to_string(),
            layer_norm_epsilon: 1e-5,
            use_bias: false,
            use_conv_bias: true,
            tie_word_embeddings: true,
            // Pattern: Mamba, Attention, MLP
            hybrid_override_pattern: Some("M*-".to_string()),
            moe_intermediate_size: None,
            moe_shared_expert_intermediate_size: None,
            n_group: None,
            n_routed_experts: None,
            n_shared_experts: None,
            topk_group: None,
            num_experts_per_tok: None,
            norm_topk_prob: None,
            routed_scaling_factor: None,
            rope_theta: 10000.0,
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
    fn test_nemotron_h_lora_construction() {
        let config = tiny_config();
        let lora_config = tiny_lora_config();
        let model = NemotronHLoraForCausalLM::new(config, lora_config);
        assert!(model.is_ok(), "Model construction should succeed");
        let model = model.unwrap();
        assert!(
            model.num_trainable_params() > 0,
            "Should have trainable parameters"
        );
    }

    #[test]
    fn test_layer_type_dispatch() {
        let config = tiny_config();
        let lora_config = tiny_lora_config();
        let model = NemotronHLoraForCausalLM::new(config, lora_config).unwrap();

        // Pattern "M*-": layer 0 = Mamba, layer 1 = Attention, layer 2 = MLP
        assert!(
            matches!(model.model.layers[0].mixer, NemotronHLoraMixer::Mamba(_)),
            "Layer 0 should be Mamba"
        );
        assert!(
            matches!(
                model.model.layers[1].mixer,
                NemotronHLoraMixer::Attention(_)
            ),
            "Layer 1 should be Attention"
        );
        assert!(
            matches!(model.model.layers[2].mixer, NemotronHLoraMixer::Mlp(_)),
            "Layer 2 should be MLP"
        );
    }

    #[test]
    fn test_mamba_layers_have_no_lora_params() {
        let config = tiny_config();
        let lora_config = tiny_lora_config();
        let model = NemotronHLoraForCausalLM::new(config, lora_config).unwrap();

        // Layer 0 is Mamba — should not appear in lora_parameters
        let params = model.lora_parameters();
        let mamba_keys: Vec<_> = params
            .keys()
            .filter(|k| k.starts_with("layers.0."))
            .collect();
        assert!(
            mamba_keys.is_empty(),
            "Mamba layer should have no LoRA params, found: {:?}",
            mamba_keys
        );
    }

    #[test]
    fn test_attention_lora_param_keys() {
        let config = tiny_config();
        let lora_config = tiny_lora_config();
        let model = NemotronHLoraForCausalLM::new(config, lora_config).unwrap();
        let params = model.lora_parameters();

        // Layer 1 is Attention
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            for ab in &["lora_a", "lora_b"] {
                let key = format!("layers.1.mixer.{proj}.{ab}");
                assert!(
                    params.contains_key(&Rc::from(key.as_str())),
                    "Missing key: {key}"
                );
            }
        }
    }

    #[test]
    fn test_mlp_lora_param_keys() {
        let config = tiny_config();
        let lora_config = tiny_lora_config();
        let model = NemotronHLoraForCausalLM::new(config, lora_config).unwrap();
        let params = model.lora_parameters();

        // Layer 2 is MLP
        for proj in &["up_proj", "down_proj"] {
            for ab in &["lora_a", "lora_b"] {
                let key = format!("layers.2.mixer.{proj}.{ab}");
                assert!(
                    params.contains_key(&Rc::from(key.as_str())),
                    "Missing key: {key}"
                );
            }
        }
    }

    #[test]
    fn test_lora_param_roundtrip() {
        let config = tiny_config();
        let lora_config = tiny_lora_config();
        let mut model = NemotronHLoraForCausalLM::new(config, lora_config).unwrap();

        let original = model.lora_parameters();
        model.set_lora_parameters(&original);
        let restored = model.lora_parameters();

        assert_eq!(
            original.len(),
            restored.len(),
            "Parameter count should be preserved after roundtrip"
        );
    }

    #[test]
    fn test_supports_kv_cache_is_false() {
        let config = tiny_config();
        let lora_config = tiny_lora_config();
        let model = NemotronHLoraForCausalLM::new(config, lora_config).unwrap();
        assert!(
            !crate::TrainableModel::supports_kv_cache(&model),
            "Hybrid model must report supports_kv_cache = false"
        );
    }

    #[test]
    fn test_nemotron_h_lora_forward() {
        let config = tiny_config();
        let lora_config = tiny_lora_config();
        let mut model = NemotronHLoraForCausalLM::new(config, lora_config).unwrap();

        let input_ids = Array::from_slice(&[1_i32, 2, 3, 4], &[1, 4]);
        let result = model.forward(&input_ids, None);
        assert!(
            result.is_ok(),
            "Forward pass should succeed: {:?}",
            result.err()
        );
        let logits = result.unwrap();
        assert_eq!(logits.shape(), &[1, 4, 64], "Logits shape mismatch");
    }
}
