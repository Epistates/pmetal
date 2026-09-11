//! LoRA-enabled DeepSeek V2/V3 model architecture.
//!
//! Implements DeepSeek with LoRA adapters for efficient fine-tuning.
//!
//! ## LoRA placement strategy
//!
//! **MLA attention projections** (all present projections receive LoRA):
//! - `q_a_proj` / `q_b_proj` — when `q_lora_rank` is set (V3 style)
//! - `q_proj` — when `q_lora_rank` is absent (direct projection)
//! - `kv_a_proj_with_mqa` — latent KV compression (always present)
//! - `kv_b_proj` — latent KV expansion (always present)
//! - `o_proj` — output projection (always present)
//!
//! **MoE layers**: LoRA on `shared_experts` (gate/up/down) only.
//! The routed experts in `moe` are kept frozen.
//!
//! **Dense MLP layers**: LoRA on gate/up/down projections.

use std::collections::HashMap;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, Exception, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, Param, nn, ops,
};

use pmetal_core::LoraConfig;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_mlx::kernels::rope::{apply_rope, apply_rope_with_freqs};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};
use pmetal_models::architectures::deepseek::{DeepSeekConfig, DeepSeekMoE, mla_scale_and_yarn};
use pmetal_models::architectures::utils::create_causal_mask;
use pmetal_models::common::yarn::YarnRope;

use crate::lora::LoraProjection;
use crate::lora_helpers::{
    collect_lora_parameters, count_trainable_params, load_lora_weights_impl,
    save_lora_weights_impl, set_lora_parameters as helpers_set_lora_parameters,
};
use crate::{LoraError, LoraLinear, impl_trainable_model};

// ─── MLA Attention ───────────────────────────────────────────────────────────

/// Represents which Q-projection variant the layer uses.
#[derive(Debug)]
pub enum DeepSeekLoraQProj {
    /// Two-stage: q_a_proj (hidden→q_lora_rank) then q_b_proj (q_lora_rank→n_heads*q_head_dim).
    LoRa {
        q_a_proj: Box<LoraLinear>,
        q_a_layernorm: Box<nn::RmsNorm>,
        q_b_proj: Box<LoraLinear>,
    },
    /// Direct: q_proj (hidden→n_heads*q_head_dim).
    Direct { q_proj: Box<LoraLinear> },
}

impl DeepSeekLoraQProj {
    fn num_trainable_params(&self) -> usize {
        match self {
            Self::LoRa {
                q_a_proj, q_b_proj, ..
            } => q_a_proj.num_trainable_params() + q_b_proj.num_trainable_params(),
            Self::Direct { q_proj } => q_proj.num_trainable_params(),
        }
    }
}

/// LoRA-enabled MLA attention layer for DeepSeek V2/V3.
///
/// Applies LoRA to all trainable MLA projections:
/// `q_a_proj`, `q_b_proj` (or `q_proj`), `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`.
#[derive(Debug)]
pub struct DeepSeekLoraAttention {
    pub config: DeepSeekConfig,
    pub n_heads: i32,
    pub scale: f32,
    pub layer_id: usize,
    /// YARN tables, when `rope_scaling` asks for them. `None` means plain
    /// traditional RoPE at `rope_theta`.
    pub yarn_rope: Option<YarnRope>,

    /// Q projection (either two-stage LoRa or direct).
    pub q: DeepSeekLoraQProj,
    /// KV compression projection (hidden → kv_lora_rank + qk_rope_head_dim).
    pub kv_a_proj_with_mqa: LoraLinear,
    /// KV layernorm (frozen).
    pub kv_a_layernorm: nn::RmsNorm,
    /// KV expansion projection (kv_lora_rank → n_heads * (qk_nope_head_dim + v_head_dim)).
    pub kv_b_proj: LoraLinear,
    /// Output projection (n_heads * v_head_dim → hidden).
    pub o_proj: LoraLinear,
}

impl DeepSeekLoraAttention {
    pub fn new(
        config: &DeepSeekConfig,
        lora_config: &LoraConfig,
        layer_id: usize,
    ) -> Result<Self, LoraError> {
        let hidden = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let q_head_dim = config.q_head_dim();
        // YARN folds mscale^2 into the softmax scale at the same moment it
        // builds the frequency tables, so both come from one call.
        let (scale, yarn_rope) = mla_scale_and_yarn(config);

        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;

        let q = if let Some(q_lora_rank) = config.q_lora_rank {
            let q_a_rank = crate::effective_rank(lora_config, "q_a_proj") as i32;
            let q_b_rank = crate::effective_rank(lora_config, "q_b_proj") as i32;
            let q_a_proj =
                LoraLinear::new(hidden, q_lora_rank, q_a_rank, alpha, use_rslora, false)?;
            let q_a_layernorm = nn::RmsNormBuilder::new(q_lora_rank)
                .eps(1e-6)
                .build()
                .map_err(LoraError::Mlx)?;
            let q_b_proj = LoraLinear::new(
                q_lora_rank,
                n_heads * q_head_dim,
                q_b_rank,
                alpha,
                use_rslora,
                false,
            )?;
            DeepSeekLoraQProj::LoRa {
                q_a_proj: Box::new(q_a_proj),
                q_a_layernorm: Box::new(q_a_layernorm),
                q_b_proj: Box::new(q_b_proj),
            }
        } else {
            let q_rank = crate::effective_rank(lora_config, "q_proj") as i32;
            let q_proj = LoraLinear::new(
                hidden,
                n_heads * q_head_dim,
                q_rank,
                alpha,
                use_rslora,
                false,
            )?;
            DeepSeekLoraQProj::Direct {
                q_proj: Box::new(q_proj),
            }
        };

        let kv_a_rank = crate::effective_rank(lora_config, "kv_a_proj_with_mqa") as i32;
        let kv_a_proj_with_mqa = LoraLinear::new(
            hidden,
            config.kv_lora_rank + config.qk_rope_head_dim,
            kv_a_rank,
            alpha,
            use_rslora,
            false,
        )?;

        let kv_a_layernorm = nn::RmsNormBuilder::new(config.kv_lora_rank)
            .eps(1e-6)
            .build()
            .map_err(LoraError::Mlx)?;

        let kv_b_rank = crate::effective_rank(lora_config, "kv_b_proj") as i32;
        let kv_b_proj = LoraLinear::new(
            config.kv_lora_rank,
            n_heads * (config.qk_nope_head_dim + config.v_head_dim),
            kv_b_rank,
            alpha,
            use_rslora,
            false,
        )?;

        let o_rank = crate::effective_rank(lora_config, "o_proj") as i32;
        let o_proj = LoraLinear::new(
            n_heads * config.v_head_dim,
            hidden,
            o_rank,
            alpha,
            use_rslora,
            false,
        )?;

        Ok(Self {
            config: config.clone(),
            n_heads,
            scale,
            yarn_rope,
            layer_id,
            q,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
        })
    }

    fn project_q(&mut self, x: &Array) -> Result<Array, LoraError> {
        match &mut self.q {
            DeepSeekLoraQProj::LoRa {
                q_a_proj,
                q_a_layernorm,
                q_b_proj,
            } => {
                let q_a_out = q_a_proj.forward(x)?;
                let q_a_norm =
                    pmetal_bridge::compat::Module::forward(q_a_layernorm.as_mut(), &q_a_out)
                        .map_err(LoraError::Mlx)?;
                q_b_proj.forward(&q_a_norm)
            }
            DeepSeekLoraQProj::Direct { q_proj } => q_proj.forward(x),
        }
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);

        // Q path
        let q = self.project_q(x)?;
        let q_head_dim = self.config.q_head_dim();
        let q = q
            .reshape(&[batch, seq_len, self.n_heads, q_head_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let q_parts =
            pmetal_bridge::compat::ops::split_sections(&q, &[self.config.qk_nope_head_dim], -1);
        let q_nope = &q_parts[0];
        let q_pe = &q_parts[1];

        // KV path
        let compressed_kv = self.kv_a_proj_with_mqa.forward(x)?;
        let kv_parts = pmetal_bridge::compat::ops::split_sections(
            &compressed_kv,
            &[self.config.kv_lora_rank],
            -1,
        );
        let compressed_latent = &kv_parts[0];
        let k_pe_raw = &kv_parts[1];
        let k_pe = k_pe_raw
            .reshape(&[batch, seq_len, 1, self.config.qk_rope_head_dim])
            .transpose_axes(&[0, 2, 1, 3]);

        let kv_norm =
            pmetal_bridge::compat::Module::forward(&mut self.kv_a_layernorm, compressed_latent)
                .map_err(LoraError::Mlx)?;
        let kv = self.kv_b_proj.forward(&kv_norm)?;
        let kv_dim = self.config.qk_nope_head_dim + self.config.v_head_dim;
        let kv = kv
            .reshape(&[batch, seq_len, self.n_heads, kv_dim])
            .transpose_axes(&[0, 2, 1, 3]);
        let kv_split =
            pmetal_bridge::compat::ops::split_sections(&kv, &[self.config.qk_nope_head_dim], -1);
        let k_nope = &kv_split[0];
        let values = &kv_split[1];

        // DeepSeek MLA uses traditional (interleaved) RoPE. The split-half
        // form rotates the wrong element pairs and silently corrupts every
        // attention score. With rope_scaling configured, apply YARN per-dim
        // frequencies + embedding mscale; without it, plain RoPE at rope_theta.
        let rope_dim = self.config.qk_rope_head_dim;
        let (q_pe, k_pe) = if let Some(yarn) = &self.yarn_rope {
            let (q_in, k_in) = if yarn.mscale != 1.0 {
                let m = Array::from_f32(yarn.mscale);
                (q_pe.multiply(&m), k_pe.multiply(&m))
            } else {
                (q_pe.clone(), k_pe.clone())
            };
            (
                apply_rope_with_freqs(&q_in, &yarn.inv_freq, rope_dim, true, offset)
                    .map_err(LoraError::Mlx)?,
                apply_rope_with_freqs(&k_in, &yarn.inv_freq, rope_dim, true, offset)
                    .map_err(LoraError::Mlx)?,
            )
        } else {
            (
                apply_rope(q_pe, rope_dim, true, self.config.rope_theta, 1.0, offset)
                    .map_err(LoraError::Mlx)?,
                apply_rope(&k_pe, rope_dim, true, self.config.rope_theta, 1.0, offset)
                    .map_err(LoraError::Mlx)?,
            )
        };

        let k_pe_broad = pmetal_bridge::compat::ops::broadcast_to(
            &k_pe,
            &[batch, self.n_heads, seq_len, self.config.qk_rope_head_dim],
        );
        let keys = pmetal_bridge::compat::ops::concatenate_axis(&[k_nope, &k_pe_broad], -1);
        let queries = pmetal_bridge::compat::ops::concatenate_axis(&[q_nope, &q_pe], -1);

        let (keys, values) = if let Some((cache, layer_idx)) = cache {
            cache
                .update_and_fetch(layer_idx, &keys, values)
                .map_err(LoraError::Mlx)?
        } else {
            (keys, values.clone())
        };

        let mut attn_weights = queries
            .matmul(&keys.transpose_axes(&[0, 1, 3, 2]))
            .multiply(&Array::from_f32(self.scale));
        if let Some(m) = mask {
            attn_weights = attn_weights.add(m);
        }
        let attn_weights = pmetal_bridge::compat::ops::softmax_axis(&attn_weights, -1);
        let output = attn_weights
            .matmul(&values)
            .transpose_axes(&[0, 2, 1, 3])
            .reshape(&[batch, seq_len, -1]);

        self.o_proj.forward(&output)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.q.num_trainable_params()
            + self.kv_a_proj_with_mqa.num_trainable_params()
            + self.kv_b_proj.num_trainable_params()
            + self.o_proj.num_trainable_params()
    }
}

// ─── MLP / MoE ───────────────────────────────────────────────────────────────

/// LoRA-enabled dense MLP layer (SwiGLU).
#[derive(Debug)]
pub struct DeepSeekLoraMLP {
    pub gate_proj: LoraLinear,
    pub up_proj: LoraLinear,
    pub down_proj: LoraLinear,
}

impl DeepSeekLoraMLP {
    pub fn new(
        hidden: i32,
        intermediate: i32,
        lora_config: &LoraConfig,
    ) -> Result<Self, LoraError> {
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;
        let gate_rank = crate::effective_rank(lora_config, "gate_proj") as i32;
        let up_rank = crate::effective_rank(lora_config, "up_proj") as i32;
        let down_rank = crate::effective_rank(lora_config, "down_proj") as i32;
        Ok(Self {
            gate_proj: LoraLinear::new(hidden, intermediate, gate_rank, alpha, use_rslora, false)?,
            up_proj: LoraLinear::new(hidden, intermediate, up_rank, alpha, use_rslora, false)?,
            down_proj: LoraLinear::new(intermediate, hidden, down_rank, alpha, use_rslora, false)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let hidden = nn::silu(&gate).multiply(&up);
        self.down_proj.forward(&hidden)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.gate_proj.num_trainable_params()
            + self.up_proj.num_trainable_params()
            + self.down_proj.num_trainable_params()
    }
}

/// LoRA-enabled MoE layer for DeepSeek.
///
/// The routed experts are kept entirely frozen (held in the base `DeepSeekMoE`
/// with its internal shared expert disabled).  The shared expert is replaced by
/// a `DeepSeekLoraMLP` that receives LoRA adapters.
///
/// Construction strategy:
/// 1. Build a `DeepSeekMoE` with `n_shared_experts = None` in its config so
///    `forward_stacked` only performs the routing dispatch.
/// 2. Hold a separate `DeepSeekLoraMLP` for the shared expert.
/// 3. During `forward`: run the frozen routing path, then add the LoRA shared
///    expert output if present.
#[derive(Debug)]
pub struct DeepSeekLoraMoE {
    /// Frozen routed experts (shared expert field is None on this instance).
    pub frozen_moe: DeepSeekMoE,
    /// LoRA-adapted shared expert (replaces the frozen shared expert).
    pub shared_experts: Option<DeepSeekLoraMLP>,
}

impl DeepSeekLoraMoE {
    pub fn new(config: &DeepSeekConfig, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        // Build a config variant with no shared experts so the frozen MoE only
        // handles the routed dispatch path.
        let mut routing_config = config.clone();
        routing_config.n_shared_experts = None;
        let frozen_moe = DeepSeekMoE::new(&routing_config).map_err(LoraError::Mlx)?;

        let shared_experts = if let Some(n_shared) = config.n_shared_experts {
            Some(DeepSeekLoraMLP::new(
                config.hidden_size,
                config.moe_intermediate_size * n_shared,
                lora_config,
            )?)
        } else {
            None
        };

        Ok(Self {
            frozen_moe,
            shared_experts,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        // Frozen routing dispatch (shared expert disabled on this instance).
        let moe_out = self.frozen_moe.forward(x).map_err(LoraError::Mlx)?;
        if let Some(ref mut shared) = self.shared_experts {
            Ok(moe_out.add(&shared.forward(x)?))
        } else {
            Ok(moe_out)
        }
    }

    pub fn num_trainable_params(&self) -> usize {
        self.shared_experts
            .as_ref()
            .map_or(0, |s| s.num_trainable_params())
    }
}

/// Dispatch enum for dense vs MoE MLP.
#[derive(Debug)]
pub enum DeepSeekLoraMlpType {
    Dense(Box<DeepSeekLoraMLP>),
    MoE(Box<DeepSeekLoraMoE>),
}

impl DeepSeekLoraMlpType {
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

// ─── Decoder layer ───────────────────────────────────────────────────────────

/// LoRA-enabled DeepSeek decoder layer.
#[derive(Debug)]
pub struct DeepSeekLoraDecoderLayer {
    pub layer_id: usize,
    pub self_attn: DeepSeekLoraAttention,
    pub mlp: DeepSeekLoraMlpType,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
}

impl DeepSeekLoraDecoderLayer {
    pub fn new(
        config: &DeepSeekConfig,
        lora_config: &LoraConfig,
        layer_id: usize,
    ) -> Result<Self, LoraError> {
        let self_attn = DeepSeekLoraAttention::new(config, lora_config, layer_id)?;
        let mlp = if config.is_moe_layer(layer_id as i32) {
            DeepSeekLoraMlpType::MoE(Box::new(DeepSeekLoraMoE::new(config, lora_config)?))
        } else {
            DeepSeekLoraMlpType::Dense(Box::new(DeepSeekLoraMLP::new(
                config.hidden_size,
                config.intermediate_size,
                lora_config,
            )?))
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
            layer_id,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let normed = pmetal_bridge::compat::Module::forward(&mut self.input_layernorm, x)
            .map_err(LoraError::Mlx)?;
        let attn_out = self.self_attn.forward(&normed, mask, cache)?;
        let h = x.add(&attn_out);

        let normed = pmetal_bridge::compat::Module::forward(&mut self.post_attention_layernorm, &h)
            .map_err(LoraError::Mlx)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok(h.add(&mlp_out))
    }

    pub fn num_trainable_params(&self) -> usize {
        self.self_attn.num_trainable_params() + self.mlp.num_trainable_params()
    }
}

// ─── ModuleParameters for composite types ────────────────────────────────────

// DeepSeekLoraQProj
impl ModuleParameters for DeepSeekLoraQProj {
    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::LoRa {
                q_a_proj,
                q_a_layernorm,
                q_b_proj,
            } => {
                let mut m = ModuleParamRef::new();
                let mut qa = HashMap::new();
                qa.insert(Rc::from("lora_a"), NestedValue::Value(&q_a_proj.lora_a));
                qa.insert(Rc::from("lora_b"), NestedValue::Value(&q_a_proj.lora_b));
                m.insert(Rc::from("q_a_proj"), NestedValue::Map(qa));
                m.extend(q_a_layernorm.parameters());
                let mut qb = HashMap::new();
                qb.insert(Rc::from("lora_a"), NestedValue::Value(&q_b_proj.lora_a));
                qb.insert(Rc::from("lora_b"), NestedValue::Value(&q_b_proj.lora_b));
                m.insert(Rc::from("q_b_proj"), NestedValue::Map(qb));
                m
            }
            Self::Direct { q_proj } => {
                let mut m = ModuleParamRef::new();
                let mut qp = HashMap::new();
                qp.insert(Rc::from("lora_a"), NestedValue::Value(&q_proj.lora_a));
                qp.insert(Rc::from("lora_b"), NestedValue::Value(&q_proj.lora_b));
                m.insert(Rc::from("q_proj"), NestedValue::Map(qp));
                m
            }
        }
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        self.parameters()
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            Self::LoRa {
                q_a_proj,
                q_a_layernorm,
                q_b_proj,
            } => {
                let mut m = ModuleParamMut::new();
                let mut qa = HashMap::new();
                qa.insert(Rc::from("lora_a"), NestedValue::Value(&mut q_a_proj.lora_a));
                qa.insert(Rc::from("lora_b"), NestedValue::Value(&mut q_a_proj.lora_b));
                m.insert(Rc::from("q_a_proj"), NestedValue::Map(qa));
                m.extend(q_a_layernorm.parameters_mut());
                let mut qb = HashMap::new();
                qb.insert(Rc::from("lora_a"), NestedValue::Value(&mut q_b_proj.lora_a));
                qb.insert(Rc::from("lora_b"), NestedValue::Value(&mut q_b_proj.lora_b));
                m.insert(Rc::from("q_b_proj"), NestedValue::Map(qb));
                m
            }
            Self::Direct { q_proj } => {
                let mut m = ModuleParamMut::new();
                let mut qp = HashMap::new();
                qp.insert(Rc::from("lora_a"), NestedValue::Value(&mut q_proj.lora_a));
                qp.insert(Rc::from("lora_b"), NestedValue::Value(&mut q_proj.lora_b));
                m.insert(Rc::from("q_proj"), NestedValue::Map(qp));
                m
            }
        }
    }

    fn num_parameters(&self) -> usize {
        match self {
            Self::LoRa {
                q_a_proj,
                q_a_layernorm,
                q_b_proj,
            } => {
                // lora_a + lora_b for each LoraLinear, plus layernorm weight
                (q_a_proj.lora_a.size() + q_a_proj.lora_b.size())
                    + q_a_layernorm.num_parameters()
                    + (q_b_proj.lora_a.size() + q_b_proj.lora_b.size())
            }
            Self::Direct { q_proj } => q_proj.lora_a.size() + q_proj.lora_b.size(),
        }
    }
}

impl ModuleParameters for DeepSeekLoraAttention {
    fn num_parameters(&self) -> usize {
        self.q.num_parameters()
            + self.kv_a_layernorm.num_parameters()
            + self.kv_a_proj_with_mqa.lora_a.size()
            + self.kv_a_proj_with_mqa.lora_b.size()
            + self.kv_b_proj.lora_a.size()
            + self.kv_b_proj.lora_b.size()
            + self.o_proj.lora_a.size()
            + self.o_proj.lora_b.size()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut m = self.q.parameters();
        m.extend(self.kv_a_layernorm.parameters());
        // Inline LoRA arrays for kv_a_proj_with_mqa, kv_b_proj, o_proj
        let mut kva = HashMap::new();
        kva.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&self.kv_a_proj_with_mqa.lora_a),
        );
        kva.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&self.kv_a_proj_with_mqa.lora_b),
        );
        m.insert(Rc::from("kv_a_proj_with_mqa"), NestedValue::Map(kva));
        let mut kvb = HashMap::new();
        kvb.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&self.kv_b_proj.lora_a),
        );
        kvb.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&self.kv_b_proj.lora_b),
        );
        m.insert(Rc::from("kv_b_proj"), NestedValue::Map(kvb));
        let mut op = HashMap::new();
        op.insert(Rc::from("lora_a"), NestedValue::Value(&self.o_proj.lora_a));
        op.insert(Rc::from("lora_b"), NestedValue::Value(&self.o_proj.lora_b));
        m.insert(Rc::from("o_proj"), NestedValue::Map(op));
        m
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        let mut m = self.q.trainable_parameters();
        let mut kva = HashMap::new();
        kva.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&self.kv_a_proj_with_mqa.lora_a),
        );
        kva.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&self.kv_a_proj_with_mqa.lora_b),
        );
        m.insert(Rc::from("kv_a_proj_with_mqa"), NestedValue::Map(kva));
        let mut kvb = HashMap::new();
        kvb.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&self.kv_b_proj.lora_a),
        );
        kvb.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&self.kv_b_proj.lora_b),
        );
        m.insert(Rc::from("kv_b_proj"), NestedValue::Map(kvb));
        let mut op = HashMap::new();
        op.insert(Rc::from("lora_a"), NestedValue::Value(&self.o_proj.lora_a));
        op.insert(Rc::from("lora_b"), NestedValue::Value(&self.o_proj.lora_b));
        m.insert(Rc::from("o_proj"), NestedValue::Map(op));
        m
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut m = self.q.parameters_mut();
        m.extend(self.kv_a_layernorm.parameters_mut());
        let mut kva = HashMap::new();
        kva.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&mut self.kv_a_proj_with_mqa.lora_a),
        );
        kva.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&mut self.kv_a_proj_with_mqa.lora_b),
        );
        m.insert(Rc::from("kv_a_proj_with_mqa"), NestedValue::Map(kva));
        let mut kvb = HashMap::new();
        kvb.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&mut self.kv_b_proj.lora_a),
        );
        kvb.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&mut self.kv_b_proj.lora_b),
        );
        m.insert(Rc::from("kv_b_proj"), NestedValue::Map(kvb));
        let mut op = HashMap::new();
        op.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&mut self.o_proj.lora_a),
        );
        op.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&mut self.o_proj.lora_b),
        );
        m.insert(Rc::from("o_proj"), NestedValue::Map(op));
        m
    }
}

impl ModuleParameters for DeepSeekLoraMLP {
    fn num_parameters(&self) -> usize {
        self.gate_proj.lora_a.size()
            + self.gate_proj.lora_b.size()
            + self.up_proj.lora_a.size()
            + self.up_proj.lora_b.size()
            + self.down_proj.lora_a.size()
            + self.down_proj.lora_b.size()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut m = ModuleParamRef::new();
        let mut gp = HashMap::new();
        gp.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&self.gate_proj.lora_a),
        );
        gp.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&self.gate_proj.lora_b),
        );
        m.insert(Rc::from("gate_proj"), NestedValue::Map(gp));
        let mut up = HashMap::new();
        up.insert(Rc::from("lora_a"), NestedValue::Value(&self.up_proj.lora_a));
        up.insert(Rc::from("lora_b"), NestedValue::Value(&self.up_proj.lora_b));
        m.insert(Rc::from("up_proj"), NestedValue::Map(up));
        let mut dp = HashMap::new();
        dp.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&self.down_proj.lora_a),
        );
        dp.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&self.down_proj.lora_b),
        );
        m.insert(Rc::from("down_proj"), NestedValue::Map(dp));
        m
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        self.parameters()
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut m = ModuleParamMut::new();
        let mut gp = HashMap::new();
        gp.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&mut self.gate_proj.lora_a),
        );
        gp.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&mut self.gate_proj.lora_b),
        );
        m.insert(Rc::from("gate_proj"), NestedValue::Map(gp));
        let mut up = HashMap::new();
        up.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&mut self.up_proj.lora_a),
        );
        up.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&mut self.up_proj.lora_b),
        );
        m.insert(Rc::from("up_proj"), NestedValue::Map(up));
        let mut dp = HashMap::new();
        dp.insert(
            Rc::from("lora_a"),
            NestedValue::Value(&mut self.down_proj.lora_a),
        );
        dp.insert(
            Rc::from("lora_b"),
            NestedValue::Value(&mut self.down_proj.lora_b),
        );
        m.insert(Rc::from("down_proj"), NestedValue::Map(dp));
        m
    }
}

impl ModuleParameters for DeepSeekLoraMoE {
    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut m = self.frozen_moe.parameters();
        if let Some(ref s) = self.shared_experts {
            m.extend(s.parameters());
        }
        m
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        // Only shared_experts carry LoRA; frozen_moe is entirely frozen.
        if let Some(ref s) = self.shared_experts {
            s.trainable_parameters()
        } else {
            HashMap::new()
        }
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut m = self.frozen_moe.parameters_mut();
        if let Some(ref mut s) = self.shared_experts {
            m.extend(s.parameters_mut());
        }
        m
    }

    fn num_parameters(&self) -> usize {
        self.frozen_moe.num_parameters()
            + self
                .shared_experts
                .as_ref()
                .map_or(0, |s| s.num_parameters())
    }

    fn freeze_parameters(&mut self, recurse: bool) {
        self.frozen_moe.freeze_parameters(recurse);
        if let Some(ref mut s) = self.shared_experts {
            s.freeze_parameters(recurse);
        }
    }

    fn unfreeze_parameters(&mut self, recurse: bool) {
        self.frozen_moe.unfreeze_parameters(recurse);
        if let Some(ref mut s) = self.shared_experts {
            s.unfreeze_parameters(recurse);
        }
    }
}

impl ModuleParameters for DeepSeekLoraMlpType {
    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(m) => m.parameters(),
            Self::MoE(m) => m.parameters(),
        }
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(m) => m.trainable_parameters(),
            Self::MoE(m) => m.trainable_parameters(),
        }
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            Self::Dense(m) => m.parameters_mut(),
            Self::MoE(m) => m.parameters_mut(),
        }
    }

    fn num_parameters(&self) -> usize {
        match self {
            Self::Dense(m) => m.num_parameters(),
            Self::MoE(m) => m.num_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recurse: bool) {
        match self {
            Self::Dense(m) => m.freeze_parameters(recurse),
            Self::MoE(m) => m.freeze_parameters(recurse),
        }
    }

    fn unfreeze_parameters(&mut self, recurse: bool) {
        match self {
            Self::Dense(m) => m.unfreeze_parameters(recurse),
            Self::MoE(m) => m.unfreeze_parameters(recurse),
        }
    }
}

impl ModuleParameters for DeepSeekLoraDecoderLayer {
    fn num_parameters(&self) -> usize {
        self.self_attn.num_parameters()
            + self.mlp.num_parameters()
            + self.input_layernorm.num_parameters()
            + self.post_attention_layernorm.num_parameters()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut m = ModuleParamRef::new();
        m.insert(
            Rc::from("self_attn"),
            NestedValue::Map(self.self_attn.parameters()),
        );
        m.insert(Rc::from("mlp"), NestedValue::Map(self.mlp.parameters()));
        m.extend(self.input_layernorm.parameters());
        m.extend(self.post_attention_layernorm.parameters());
        m
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        let mut m = ModuleParamRef::new();
        m.insert(
            Rc::from("self_attn"),
            NestedValue::Map(self.self_attn.trainable_parameters()),
        );
        m.insert(
            Rc::from("mlp"),
            NestedValue::Map(self.mlp.trainable_parameters()),
        );
        m
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut m = ModuleParamMut::new();
        m.insert(
            Rc::from("self_attn"),
            NestedValue::Map(self.self_attn.parameters_mut()),
        );
        m.insert(Rc::from("mlp"), NestedValue::Map(self.mlp.parameters_mut()));
        m.extend(self.input_layernorm.parameters_mut());
        m.extend(self.post_attention_layernorm.parameters_mut());
        m
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// LoRA-enabled DeepSeek trunk (without LM head).
#[derive(Debug)]
pub struct DeepSeekLoraModel {
    pub config: DeepSeekConfig,
    pub lora_config: LoraConfig,
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<DeepSeekLoraDecoderLayer>,
    pub norm: nn::RmsNorm,
}

impl DeepSeekLoraModel {
    pub fn new(config: DeepSeekConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let embed_tokens =
            nn::Embedding::new(config.vocab_size, config.hidden_size).map_err(LoraError::Mlx)?;
        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| DeepSeekLoraDecoderLayer::new(&config, &lora_config, i))
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

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        checkpoint_config: Option<&CheckpointConfig>,
    ) -> Result<Array, LoraError> {
        let mut h = pmetal_bridge::compat::Module::forward(&mut self.embed_tokens, input_ids)
            .map_err(LoraError::Mlx)?;

        let mask = if mask.is_none() {
            let seq_len = input_ids.dim(1);
            Some(create_causal_mask(seq_len).map_err(LoraError::Mlx)?)
        } else {
            mask.cloned()
        };

        let layers_per_block = checkpoint_config
            .map(|c| c.layers_per_block)
            .unwrap_or(usize::MAX);
        let checkpointing = checkpoint_config.map(|c| c.enabled).unwrap_or(false);

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            h = layer.forward(&h, mask.as_ref(), None)?;
            if checkpointing && (idx + 1) % layers_per_block == 0 {
                tracing::trace!("deepseek_lora checkpoint boundary at layer {}", idx + 1);
            }
        }

        pmetal_bridge::compat::Module::forward(&mut self.norm, &h).map_err(LoraError::Mlx)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let mut h = pmetal_bridge::compat::Module::forward(&mut self.embed_tokens, input_ids)
            .map_err(LoraError::Mlx)?;

        match cache {
            Some(cache) => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    h = layer.forward(&h, mask, Some((cache, layer_idx)))?;
                }
            }
            None => {
                for layer in &mut self.layers {
                    h = layer.forward(&h, mask, None)?;
                }
            }
        }

        pmetal_bridge::compat::Module::forward(&mut self.norm, &h).map_err(LoraError::Mlx)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_trainable_params()).sum()
    }
}

impl ModuleParameters for DeepSeekLoraModel {
    fn num_parameters(&self) -> usize {
        self.embed_tokens.num_parameters()
            + self
                .layers
                .iter()
                .map(|l| l.num_parameters())
                .sum::<usize>()
            + self.norm.num_parameters()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut m = ModuleParamRef::new();
        m.extend(self.embed_tokens.parameters());
        for (i, layer) in self.layers.iter().enumerate() {
            m.insert(
                Rc::from(format!("{}", i)),
                NestedValue::Map(layer.parameters()),
            );
        }
        m.extend(self.norm.parameters());
        m
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        let mut m = ModuleParamRef::new();
        for (i, layer) in self.layers.iter().enumerate() {
            m.insert(
                Rc::from(format!("{}", i)),
                NestedValue::Map(layer.trainable_parameters()),
            );
        }
        m
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut m = ModuleParamMut::new();
        m.extend(self.embed_tokens.parameters_mut());
        for (i, layer) in self.layers.iter_mut().enumerate() {
            m.insert(
                Rc::from(format!("{}", i)),
                NestedValue::Map(layer.parameters_mut()),
            );
        }
        m.extend(self.norm.parameters_mut());
        m
    }
}

// ─── LoRA parameter management helpers ───────────────────────────────────────

/// Collect all LoRA adapter weights from the model.
///
/// Key schema (mirrors the base model weight naming for easy correspondence):
/// - `layers.{i}.self_attn.q_a_proj.lora_{a,b}` (when q_lora_rank is set)
/// - `layers.{i}.self_attn.q_b_proj.lora_{a,b}` (when q_lora_rank is set)
/// - `layers.{i}.self_attn.q_proj.lora_{a,b}`    (when q_lora_rank is absent)
/// - `layers.{i}.self_attn.kv_a_proj_with_mqa.lora_{a,b}`
/// - `layers.{i}.self_attn.kv_b_proj.lora_{a,b}`
/// - `layers.{i}.self_attn.o_proj.lora_{a,b}`
/// - `layers.{i}.mlp.gate_proj.lora_{a,b}` / up_proj / down_proj  (dense layers)
/// - `layers.{i}.mlp.shared_experts.gate_proj.lora_{a,b}` / ...   (MoE layers)
fn collect_deepseek_lora_params(model: &DeepSeekLoraModel) -> HashMap<Rc<str>, Array> {
    let mut params = HashMap::new();

    for (i, layer) in model.layers.iter().enumerate() {
        let lp = format!("layers.{}", i);

        // Attention projections
        let attn = &layer.self_attn;
        match &attn.q {
            DeepSeekLoraQProj::LoRa {
                q_a_proj, q_b_proj, ..
            } => {
                insert_lora(&mut params, &format!("{}.self_attn.q_a_proj", lp), q_a_proj);
                insert_lora(&mut params, &format!("{}.self_attn.q_b_proj", lp), q_b_proj);
            }
            DeepSeekLoraQProj::Direct { q_proj } => {
                insert_lora(&mut params, &format!("{}.self_attn.q_proj", lp), q_proj);
            }
        }
        insert_lora(
            &mut params,
            &format!("{}.self_attn.kv_a_proj_with_mqa", lp),
            &attn.kv_a_proj_with_mqa,
        );
        insert_lora(
            &mut params,
            &format!("{}.self_attn.kv_b_proj", lp),
            &attn.kv_b_proj,
        );
        insert_lora(
            &mut params,
            &format!("{}.self_attn.o_proj", lp),
            &attn.o_proj,
        );

        // MLP / MoE projections
        match &layer.mlp {
            DeepSeekLoraMlpType::Dense(m) => {
                insert_lora(&mut params, &format!("{}.mlp.gate_proj", lp), &m.gate_proj);
                insert_lora(&mut params, &format!("{}.mlp.up_proj", lp), &m.up_proj);
                insert_lora(&mut params, &format!("{}.mlp.down_proj", lp), &m.down_proj);
            }
            DeepSeekLoraMlpType::MoE(m) => {
                if let Some(ref s) = m.shared_experts {
                    let sp = format!("{}.mlp.shared_experts", lp);
                    insert_lora(&mut params, &format!("{}.gate_proj", sp), &s.gate_proj);
                    insert_lora(&mut params, &format!("{}.up_proj", sp), &s.up_proj);
                    insert_lora(&mut params, &format!("{}.down_proj", sp), &s.down_proj);
                }
            }
        }
    }

    params
}

fn insert_lora(params: &mut HashMap<Rc<str>, Array>, prefix: &str, proj: &LoraLinear) {
    params.insert(Rc::from(format!("{}.lora_a", prefix)), proj.lora_a.clone());
    params.insert(Rc::from(format!("{}.lora_b", prefix)), proj.lora_b.clone());
}

fn set_deepseek_lora_params(model: &mut DeepSeekLoraModel, params: &HashMap<Rc<str>, Array>) {
    for (i, layer) in model.layers.iter_mut().enumerate() {
        let lp = format!("layers.{}", i);

        let attn = &mut layer.self_attn;
        match &mut attn.q {
            DeepSeekLoraQProj::LoRa {
                q_a_proj, q_b_proj, ..
            } => {
                apply_lora_keys(&format!("{}.self_attn.q_a_proj", lp), params, q_a_proj);
                apply_lora_keys(&format!("{}.self_attn.q_b_proj", lp), params, q_b_proj);
            }
            DeepSeekLoraQProj::Direct { q_proj } => {
                apply_lora_keys(&format!("{}.self_attn.q_proj", lp), params, q_proj);
            }
        }
        apply_lora_keys(
            &format!("{}.self_attn.kv_a_proj_with_mqa", lp),
            params,
            &mut attn.kv_a_proj_with_mqa,
        );
        apply_lora_keys(
            &format!("{}.self_attn.kv_b_proj", lp),
            params,
            &mut attn.kv_b_proj,
        );
        apply_lora_keys(
            &format!("{}.self_attn.o_proj", lp),
            params,
            &mut attn.o_proj,
        );

        match &mut layer.mlp {
            DeepSeekLoraMlpType::Dense(m) => {
                apply_lora_keys(&format!("{}.mlp.gate_proj", lp), params, &mut m.gate_proj);
                apply_lora_keys(&format!("{}.mlp.up_proj", lp), params, &mut m.up_proj);
                apply_lora_keys(&format!("{}.mlp.down_proj", lp), params, &mut m.down_proj);
            }
            DeepSeekLoraMlpType::MoE(m) => {
                if let Some(ref mut s) = m.shared_experts {
                    let sp = format!("{}.mlp.shared_experts", lp);
                    apply_lora_keys(&format!("{}.gate_proj", sp), params, &mut s.gate_proj);
                    apply_lora_keys(&format!("{}.up_proj", sp), params, &mut s.up_proj);
                    apply_lora_keys(&format!("{}.down_proj", sp), params, &mut s.down_proj);
                }
            }
        }
    }
}

fn apply_lora_keys(prefix: &str, params: &HashMap<Rc<str>, Array>, proj: &mut LoraLinear) {
    if let Some(v) = params.get(&Rc::from(format!("{}.lora_a", prefix))) {
        proj.lora_a = v.clone();
    }
    if let Some(v) = params.get(&Rc::from(format!("{}.lora_b", prefix))) {
        proj.lora_b = v.clone();
    }
}

fn save_deepseek_lora_weights(
    model: &DeepSeekLoraModel,
    path: impl AsRef<std::path::Path>,
) -> Result<(), LoraError> {
    let params = collect_deepseek_lora_params(model);
    crate::save_safetensors_map(path, &params)
}

fn load_deepseek_lora_weights(
    model: &mut DeepSeekLoraModel,
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
    set_deepseek_lora_params(model, &params);
    Ok(())
}

// ─── ForCausalLM ─────────────────────────────────────────────────────────────

/// LoRA-enabled DeepSeek causal language model.
#[derive(Debug)]
pub struct DeepSeekLoraForCausalLM {
    pub model: DeepSeekLoraModel,
    /// LM head (frozen; present when `tie_word_embeddings` is false).
    pub lm_head: Option<nn::Linear>,
    /// Gradient checkpointing configuration.
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl DeepSeekLoraForCausalLM {
    pub fn new(config: DeepSeekConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let tie = config.tie_word_embeddings;
        let lm_head = if !tie {
            Some(
                nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                    .bias(false)
                    .build()
                    .map_err(LoraError::Mlx)?,
            )
        } else {
            None
        };
        let model = DeepSeekLoraModel::new(config, lora_config)?;
        Ok(Self {
            model,
            lm_head,
            checkpoint_config: None,
        })
    }

    fn apply_lm_head(&mut self, h: &Array) -> Array {
        if let Some(ref mut head) = self.lm_head {
            pmetal_bridge::compat::Module::forward(head, h).unwrap()
        } else {
            self.model.embed_tokens.as_linear(h)
        }
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let ckpt = self.checkpoint_config.clone();
        let h = self.model.forward(input_ids, mask, ckpt.as_ref())?;
        Ok(self.apply_lm_head(&h))
    }

    pub fn forward_hidden_states(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
        let ckpt = self.checkpoint_config.clone();
        self.model.forward(input_ids, mask, ckpt.as_ref())
    }

    /// Stub: no packed-sequence position IDs support yet (DeepSeek uses RoPE offset
    /// which is not position-ID aware). Falls back to standard forward.
    pub fn forward_hidden_states_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        _position_ids: &Array,
    ) -> Result<Array, LoraError> {
        self.forward_hidden_states(input_ids, mask)
    }

    pub fn forward_noised(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        noise_alpha: f32,
    ) -> Result<Array, LoraError> {
        let ckpt = self.checkpoint_config.clone();
        let mut h = pmetal_bridge::compat::Module::forward(&mut self.model.embed_tokens, input_ids)
            .map_err(LoraError::Mlx)?;

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

        let mask_owned = if mask.is_none() {
            let sl = input_ids.dim(1);
            Some(create_causal_mask(sl).map_err(LoraError::Mlx)?)
        } else {
            mask.cloned()
        };

        let layers_per_block = ckpt
            .as_ref()
            .map(|c| c.layers_per_block)
            .unwrap_or(usize::MAX);
        let checkpointing = ckpt.as_ref().map(|c| c.enabled).unwrap_or(false);

        for (idx, layer) in self.model.layers.iter_mut().enumerate() {
            h = layer.forward(&h, mask_owned.as_ref(), None)?;
            if checkpointing && (idx + 1) % layers_per_block == 0 {
                tracing::trace!("deepseek_lora neftune checkpoint at layer {}", idx + 1);
            }
        }

        let h = pmetal_bridge::compat::Module::forward(&mut self.model.norm, &h)
            .map_err(LoraError::Mlx)?;
        Ok(self.apply_lm_head(&h))
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let h = self.model.forward_with_cache(input_ids, mask, cache)?;
        Ok(self.apply_lm_head(&h))
    }

    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        let cfg = &self.model.config;
        KVCache::new(
            KVCacheConfig::new(
                cfg.num_hidden_layers as usize,
                max_seq_len,
                cfg.num_attention_heads as usize,
                cfg.q_head_dim() as usize,
            )
            .with_value_head_dim(cfg.v_head_dim as usize),
        )
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

    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        collect_deepseek_lora_params(&self.model)
    }

    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        set_deepseek_lora_params(&mut self.model, params);
    }

    pub fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        save_deepseek_lora_weights(&self.model, path)
    }

    pub fn load_lora_weights(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), LoraError> {
        load_deepseek_lora_weights(&mut self.model, path)
    }

    pub fn get_lm_head_weight(&self) -> Option<Array> {
        if let Some(ref head) = self.lm_head {
            Some(head.weight.value.clone())
        } else {
            Some(self.model.embed_tokens.weight.value.clone())
        }
    }

    /// Load frozen base model weights from a flat weight map (HuggingFace key schema).
    ///
    /// Populates every frozen weight in the model: embeddings, attention projections,
    /// MLP / MoE expert weights, layer norms, and the LM head. LoRA adapter matrices
    /// (`lora_a`, `lora_b`) are left untouched.
    ///
    /// # Weight key schema (HF format)
    /// - `model.embed_tokens.weight`
    /// - `model.layers.{i}.self_attn.q_a_proj.weight` / `q_b_proj.weight` (q_lora_rank set)
    /// - `model.layers.{i}.self_attn.q_proj.weight` (q_lora_rank absent)
    /// - `model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight`
    /// - `model.layers.{i}.self_attn.kv_a_layernorm.weight`
    /// - `model.layers.{i}.self_attn.kv_b_proj.weight`
    /// - `model.layers.{i}.self_attn.o_proj.weight`
    /// - `model.layers.{i}.mlp.{gate,up,down}_proj.weight` (dense layers)
    /// - `model.layers.{i}.mlp.gate.weight` (MoE gate linear)
    /// - `model.layers.{i}.mlp.gate.e_score_correction_bias` (MoE gate bias)
    /// - `model.layers.{i}.mlp.experts.{j}.gate_proj.weight` / `up_proj` / `down_proj`
    /// - `model.layers.{i}.mlp.shared_experts.{gate,up,down}_proj.weight`
    /// - `model.layers.{i}.input_layernorm.weight`
    /// - `model.layers.{i}.post_attention_layernorm.weight`
    /// - `model.norm.weight`
    /// - `lm_head.weight`
    pub fn load_base_weights(&mut self, weights: &HashMap<String, Array>) -> Result<(), LoraError> {
        // Embeddings
        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let pfx = format!("model.layers.{i}");

            // Attention projections — base weights inside LoraLinear
            match &mut layer.self_attn.q {
                DeepSeekLoraQProj::LoRa {
                    q_a_proj,
                    q_b_proj,
                    q_a_layernorm,
                } => {
                    if let Some(w) = weights.get(&format!("{pfx}.self_attn.q_a_proj.weight")) {
                        *q_a_proj.weight_mut() = w.clone();
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.self_attn.q_a_layernorm.weight")) {
                        q_a_layernorm.weight = Param::new(w.clone());
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.self_attn.q_b_proj.weight")) {
                        *q_b_proj.weight_mut() = w.clone();
                    }
                }
                DeepSeekLoraQProj::Direct { q_proj } => {
                    if let Some(w) = weights.get(&format!("{pfx}.self_attn.q_proj.weight")) {
                        *q_proj.weight_mut() = w.clone();
                    }
                }
            }

            if let Some(w) = weights.get(&format!("{pfx}.self_attn.kv_a_proj_with_mqa.weight")) {
                *layer.self_attn.kv_a_proj_with_mqa.weight_mut() = w.clone();
            }
            if let Some(w) = weights.get(&format!("{pfx}.self_attn.kv_a_layernorm.weight")) {
                layer.self_attn.kv_a_layernorm.weight = Param::new(w.clone());
            }
            if let Some(w) = weights.get(&format!("{pfx}.self_attn.kv_b_proj.weight")) {
                *layer.self_attn.kv_b_proj.weight_mut() = w.clone();
            }
            if let Some(w) = weights.get(&format!("{pfx}.self_attn.o_proj.weight")) {
                *layer.self_attn.o_proj.weight_mut() = w.clone();
            }

            // Layer norms
            if let Some(w) = weights.get(&format!("{pfx}.input_layernorm.weight")) {
                layer.input_layernorm.weight = Param::new(w.clone());
            }
            if let Some(w) = weights.get(&format!("{pfx}.post_attention_layernorm.weight")) {
                layer.post_attention_layernorm.weight = Param::new(w.clone());
            }

            // MLP / MoE
            match &mut layer.mlp {
                DeepSeekLoraMlpType::Dense(mlp) => {
                    if let Some(w) = weights.get(&format!("{pfx}.mlp.gate_proj.weight")) {
                        *mlp.gate_proj.weight_mut() = w.clone();
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.mlp.up_proj.weight")) {
                        *mlp.up_proj.weight_mut() = w.clone();
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.mlp.down_proj.weight")) {
                        *mlp.down_proj.weight_mut() = w.clone();
                    }
                }
                DeepSeekLoraMlpType::MoE(moe) => {
                    // Frozen MoE gate
                    if let Some(w) = weights.get(&format!("{pfx}.mlp.gate.weight")) {
                        moe.frozen_moe.gate.weight.weight = Param::new(w.clone());
                    }
                    if let Some(b) = weights.get(&format!("{pfx}.mlp.gate.e_score_correction_bias"))
                    {
                        moe.frozen_moe.gate.e_score_correction_bias = b.clone();
                    }

                    // Frozen routed experts (w1=gate_proj, w3=up_proj, w2=down_proj)
                    for (j, expert) in moe.frozen_moe.moe.experts.iter_mut().enumerate() {
                        if let Some(w) =
                            weights.get(&format!("{pfx}.mlp.experts.{j}.gate_proj.weight"))
                        {
                            expert.w1.weight = w.clone();
                        }
                        if let Some(w) =
                            weights.get(&format!("{pfx}.mlp.experts.{j}.up_proj.weight"))
                        {
                            expert.w3.weight = w.clone();
                        }
                        if let Some(w) =
                            weights.get(&format!("{pfx}.mlp.experts.{j}.down_proj.weight"))
                        {
                            expert.w2.weight = w.clone();
                        }
                    }

                    // LoRA-adapted shared expert base weights
                    if let Some(ref mut se) = moe.shared_experts {
                        if let Some(w) =
                            weights.get(&format!("{pfx}.mlp.shared_experts.gate_proj.weight"))
                        {
                            *se.gate_proj.weight_mut() = w.clone();
                        }
                        if let Some(w) =
                            weights.get(&format!("{pfx}.mlp.shared_experts.up_proj.weight"))
                        {
                            *se.up_proj.weight_mut() = w.clone();
                        }
                        if let Some(w) =
                            weights.get(&format!("{pfx}.mlp.shared_experts.down_proj.weight"))
                        {
                            *se.down_proj.weight_mut() = w.clone();
                        }
                    }
                }
            }
        }

        // Final norm
        if let Some(w) = weights.get("model.norm.weight") {
            self.model.norm.weight = Param::new(w.clone());
        }

        // LM head (only when weights are not tied)
        if let Some(ref mut lm_head) = self.lm_head {
            if let Some(w) = weights.get("lm_head.weight") {
                lm_head.weight = Param::new(w.clone());
            }
        }

        Ok(())
    }

    /// Load frozen base model weights from safetensors files in a directory.
    ///
    /// Handles both single-file (`model.safetensors`) and sharded models
    /// (`model.safetensors.index.json` + multiple shard files). Uses
    /// [`WeightLoader`][pmetal_models::WeightLoader] for format-agnostic loading.
    pub fn load_base_weights_from_dir(
        &mut self,
        model_dir: &std::path::Path,
    ) -> Result<(), LoraError> {
        let weights = pmetal_models::WeightLoader::load_safetensors(model_dir)
            .map_err(|e| LoraError::InvalidState(format!("failed to load base weights: {e:?}")))?;
        self.load_base_weights(&weights)
    }

    /// Evaluate (materialise) all model parameters on the device.
    ///
    /// Forces evaluation of both the frozen base weights and the LoRA adapter matrices
    /// so that subsequent forward passes do not carry deferred computation graphs.
    pub fn eval_all(&mut self) -> Result<(), LoraError> {
        self.model.embed_tokens.weight.value.eval();

        for layer in &mut self.model.layers {
            // Attention — base weights
            match &layer.self_attn.q {
                DeepSeekLoraQProj::LoRa {
                    q_a_proj,
                    q_b_proj,
                    q_a_layernorm,
                } => {
                    q_a_proj.weight().eval();
                    q_a_proj.lora_a.eval();
                    q_a_proj.lora_b.eval();
                    q_a_layernorm.weight.value.eval();
                    q_b_proj.weight().eval();
                    q_b_proj.lora_a.eval();
                    q_b_proj.lora_b.eval();
                }
                DeepSeekLoraQProj::Direct { q_proj } => {
                    q_proj.weight().eval();
                    q_proj.lora_a.eval();
                    q_proj.lora_b.eval();
                }
            }
            layer.self_attn.kv_a_proj_with_mqa.weight().eval();
            layer.self_attn.kv_a_proj_with_mqa.lora_a.eval();
            layer.self_attn.kv_a_proj_with_mqa.lora_b.eval();
            layer.self_attn.kv_a_layernorm.weight.value.eval();
            layer.self_attn.kv_b_proj.weight().eval();
            layer.self_attn.kv_b_proj.lora_a.eval();
            layer.self_attn.kv_b_proj.lora_b.eval();
            layer.self_attn.o_proj.weight().eval();
            layer.self_attn.o_proj.lora_a.eval();
            layer.self_attn.o_proj.lora_b.eval();

            // Layer norms
            layer.input_layernorm.weight.value.eval();
            layer.post_attention_layernorm.weight.value.eval();

            // MLP / MoE
            match &mut layer.mlp {
                DeepSeekLoraMlpType::Dense(mlp) => {
                    mlp.gate_proj.weight().eval();
                    mlp.gate_proj.lora_a.eval();
                    mlp.gate_proj.lora_b.eval();
                    mlp.up_proj.weight().eval();
                    mlp.up_proj.lora_a.eval();
                    mlp.up_proj.lora_b.eval();
                    mlp.down_proj.weight().eval();
                    mlp.down_proj.lora_a.eval();
                    mlp.down_proj.lora_b.eval();
                }
                DeepSeekLoraMlpType::MoE(moe) => {
                    moe.frozen_moe.gate.weight.weight.value.eval();
                    moe.frozen_moe.gate.e_score_correction_bias.eval();
                    for expert in &moe.frozen_moe.moe.experts {
                        expert.w1.weight.eval();
                        expert.w3.weight.eval();
                        expert.w2.weight.eval();
                    }
                    if let Some(ref se) = moe.shared_experts {
                        se.gate_proj.weight().eval();
                        se.gate_proj.lora_a.eval();
                        se.gate_proj.lora_b.eval();
                        se.up_proj.weight().eval();
                        se.up_proj.lora_a.eval();
                        se.up_proj.lora_b.eval();
                        se.down_proj.weight().eval();
                        se.down_proj.lora_a.eval();
                        se.down_proj.lora_b.eval();
                    }
                }
            }
        }

        self.model.norm.weight.value.eval();

        if let Some(ref lm_head) = self.lm_head {
            lm_head.weight.value.eval();
        }

        Ok(())
    }

    /// Merge LoRA weights into base weights across all adapted projections
    /// (MLA q-path, kv_a/kv_b, o_proj, and shared-expert MLP when present).
    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        for layer in &mut self.model.layers {
            match &mut layer.self_attn.q {
                DeepSeekLoraQProj::LoRa {
                    q_a_proj, q_b_proj, ..
                } => {
                    q_a_proj.merge()?;
                    q_b_proj.merge()?;
                }
                DeepSeekLoraQProj::Direct { .. } => {}
            }
            layer.self_attn.kv_a_proj_with_mqa.merge()?;
            layer.self_attn.kv_b_proj.merge()?;
            layer.self_attn.o_proj.merge()?;

            match &mut layer.mlp {
                DeepSeekLoraMlpType::Dense(mlp) => {
                    mlp.gate_proj.merge()?;
                    mlp.up_proj.merge()?;
                    mlp.down_proj.merge()?;
                }
                DeepSeekLoraMlpType::MoE(moe) => {
                    if let Some(shared) = moe.shared_experts.as_mut() {
                        shared.gate_proj.merge()?;
                        shared.up_proj.merge()?;
                        shared.down_proj.merge()?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Unmerge is not supported. Reload base weights via
    /// [`load_base_weights_from_dir`][Self::load_base_weights_from_dir] to undo a merge.
    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        Err(LoraError::InvalidState(
            "unmerge_lora is not supported: reload base model weights to undo a merge".to_string(),
        ))
    }
}

impl ModuleParameters for DeepSeekLoraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.model.num_parameters() + self.lm_head.as_ref().map_or(0, |h| h.num_parameters())
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut m = ModuleParamRef::new();
        m.insert(Rc::from("model"), NestedValue::Map(self.model.parameters()));
        if let Some(ref head) = self.lm_head {
            m.extend(head.parameters());
        }
        m
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        let mut m = ModuleParamRef::new();
        m.insert(
            Rc::from("model"),
            NestedValue::Map(self.model.trainable_parameters()),
        );
        m
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut m = ModuleParamMut::new();
        m.insert(
            Rc::from("model"),
            NestedValue::Map(self.model.parameters_mut()),
        );
        if let Some(ref mut head) = self.lm_head {
            m.extend(head.parameters_mut());
        }
        m
    }
}
impl_trainable_model!(DeepSeekLoraForCausalLM);

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_models::architectures::deepseek::DeepSeekConfig;

    fn tiny_config() -> DeepSeekConfig {
        DeepSeekConfig {
            vocab_size: 512,
            hidden_size: 16,
            intermediate_size: 32,
            moe_intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(4),
            n_shared_experts: Some(1),
            n_routed_experts: Some(4),
            num_experts_per_tok: 2,
            moe_layer_freq: 1,
            first_k_dense_replace: 0,
            kv_lora_rank: 8,
            q_lora_rank: Some(12),
            qk_nope_head_dim: 8,
            qk_rope_head_dim: 8,
            v_head_dim: 8,
            ..DeepSeekConfig::default()
        }
    }

    fn default_lora_config() -> LoraConfig {
        LoraConfig {
            r: 4,
            alpha: 8.0,
            use_rslora: false,
            use_dora: false,
            ..LoraConfig::default()
        }
    }

    #[test]
    fn test_deepseek_lora_model_constructs() {
        let config = tiny_config();
        let lora_config = default_lora_config();
        let model = DeepSeekLoraForCausalLM::new(config, lora_config).unwrap();
        assert!(model.num_trainable_params() > 0);
    }

    #[test]
    fn test_deepseek_lora_forward_produces_correct_shape() {
        let config = tiny_config();
        let lora_config = default_lora_config();
        let mut model = DeepSeekLoraForCausalLM::new(config.clone(), lora_config).unwrap();

        let input_ids = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();
        logits.eval().unwrap();

        assert_eq!(logits.dim(0), 1);
        assert_eq!(logits.dim(1), 4);
        assert_eq!(logits.dim(2), config.vocab_size);
    }

    #[test]
    fn test_deepseek_lora_forward_with_cache() {
        let config = tiny_config();
        let lora_config = default_lora_config();
        let mut model = DeepSeekLoraForCausalLM::new(config, lora_config).unwrap();
        let mut cache = model.create_cache(32);

        let tok1 = Array::from_slice(&[1i32], &[1, 1]);
        let tok2 = Array::from_slice(&[2i32], &[1, 1]);

        let logits1 = model
            .forward_with_cache(&tok1, None, Some(&mut cache))
            .unwrap();
        let logits2 = model
            .forward_with_cache(&tok2, None, Some(&mut cache))
            .unwrap();
        logits1.eval().unwrap();
        logits2.eval().unwrap();

        assert_eq!(cache.seq_len(), 2);
    }

    #[test]
    fn test_deepseek_lora_lora_parameters_roundtrip() {
        let config = tiny_config();
        let lora_config = default_lora_config();
        let mut model = DeepSeekLoraForCausalLM::new(config, lora_config).unwrap();

        let params = model.lora_parameters();
        assert!(!params.is_empty());

        // Overwrite all LoRA params with zeros then restore
        let zeroed: HashMap<Rc<str>, Array> = params
            .iter()
            .map(|(k, v)| {
                let z = pmetal_bridge::compat::ops::zeros_dtype(v.shape(), v.dtype());
                (k.clone(), z)
            })
            .collect();
        model.set_lora_parameters(&zeroed);

        let params2 = model.lora_parameters();
        for arr in params2.values() {
            arr.eval().unwrap();
        }
        assert_eq!(params2.len(), params.len());
    }

    #[test]
    fn test_deepseek_lora_direct_q_proj_variant() {
        // Test the no-q_lora_rank variant (direct q_proj)
        let mut config = tiny_config();
        config.q_lora_rank = None;
        let lora_config = default_lora_config();
        let mut model = DeepSeekLoraForCausalLM::new(config.clone(), lora_config).unwrap();

        let input_ids = Array::from_slice(&[1i32, 2], &[1, 2]);
        let logits = model.forward(&input_ids, None).unwrap();
        logits.eval().unwrap();

        assert_eq!(logits.dim(2), config.vocab_size);

        // Verify q_proj keys in params
        let params = model.lora_parameters();
        let has_q_proj = params.keys().any(|k| k.contains("q_proj.lora_a"));
        assert!(has_q_proj, "expected q_proj lora keys for direct variant");
        let has_q_a = params.keys().any(|k| k.contains("q_a_proj.lora_a"));
        assert!(!has_q_a, "q_a_proj should not appear in direct-q variant");
    }

    #[test]
    fn test_deepseek_lora_moe_shared_expert_trainable_routed_frozen() {
        let config = tiny_config(); // moe_layer_freq=1, first_k_dense_replace=0 → all layers MoE
        let lora_config = default_lora_config();
        let model = DeepSeekLoraForCausalLM::new(config, lora_config).unwrap();

        let params = model.lora_parameters();
        // Shared expert keys must be present
        let has_shared = params
            .keys()
            .any(|k| k.contains("shared_experts.gate_proj.lora_a"));
        assert!(
            has_shared,
            "expected shared_experts LoRA keys in MoE layers"
        );
        // Routed expert keys must be absent (frozen) — match ".experts." to exclude "shared_experts."
        let has_routed = params.keys().any(|k| k.contains(".experts."));
        assert!(
            !has_routed,
            "routed expert weights must not appear in LoRA params"
        );
    }
}
