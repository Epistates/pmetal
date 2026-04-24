//! QLoRA-enabled DeepSeek V2/V3 model architecture (4-bit quantized base + LoRA adapters).
//!
//! ## Strategy
//!
//! All MLA attention projections (`q_a_proj`/`q_b_proj` or `q_proj`, `kv_a_proj_with_mqa`,
//! `kv_b_proj`, `o_proj`) and dense/shared-expert MLP projections carry `QLoraLinear` layers
//! (quantized base weight + full-precision LoRA A/B).
//!
//! The routed MoE experts remain frozen and are held inside a `DeepSeekMoE` instance
//! exactly as in the non-quantized variant — they receive no LoRA adapters and are not
//! quantized (their base dtype is preserved as loaded).
//!
//! ## LoRA placement
//! - MLA attention: q_a_proj / q_b_proj (or q_proj), kv_a_proj_with_mqa, kv_b_proj, o_proj
//! - Dense MLP: gate_proj / up_proj / down_proj
//! - MoE: shared expert gate_proj / up_proj / down_proj only; routed experts frozen

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, Exception, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, Param, nn,
    ops,
};
use pmetal_core::LoraConfig;
use pmetal_mlx::gradient_checkpoint::CheckpointConfig;
use pmetal_mlx::kernels::rope::apply_rope;
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};
use pmetal_models::architectures::deepseek::{DeepSeekConfig, DeepSeekMoE};

use crate::{LoraError, QLoraConfig, QLoraLinear};

// ─── Q-projection enum ───────────────────────────────────────────────────────

/// Q-projection variant for DeepSeek QLoRA attention.
#[derive(Debug)]
pub enum DeepSeekQloraQProj {
    /// Two-stage: q_a_proj (hidden → q_lora_rank) then q_b_proj (q_lora_rank → n_heads*q_head_dim).
    LoRa {
        q_a_proj: QLoraLinear,
        q_a_layernorm: nn::RmsNorm,
        q_b_proj: QLoraLinear,
    },
    /// Direct: q_proj (hidden → n_heads*q_head_dim).
    Direct { q_proj: QLoraLinear },
}

impl DeepSeekQloraQProj {
    fn new(
        config: &DeepSeekConfig,
        qcfg: &QLoraConfig,
    ) -> Result<Self, LoraError> {
        let hidden = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let q_head_dim = config.q_head_dim();

        if let Some(q_lora_rank) = config.q_lora_rank {
            let mut qa_cfg = qcfg.clone();
            qa_cfg.lora.r = crate::effective_rank(&qcfg.lora, "q_a_proj");
            let q_a_proj = QLoraLinear::new(hidden, q_lora_rank, &qa_cfg, false)?;

            let q_a_layernorm = nn::RmsNormBuilder::new(q_lora_rank)
                .eps(1e-6)
                .build()
                .map_err(LoraError::Mlx)?;

            let mut qb_cfg = qcfg.clone();
            qb_cfg.lora.r = crate::effective_rank(&qcfg.lora, "q_b_proj");
            let q_b_proj =
                QLoraLinear::new(q_lora_rank, n_heads * q_head_dim, &qb_cfg, false)?;

            Ok(Self::LoRa { q_a_proj, q_a_layernorm, q_b_proj })
        } else {
            let mut qp_cfg = qcfg.clone();
            qp_cfg.lora.r = crate::effective_rank(&qcfg.lora, "q_proj");
            let q_proj =
                QLoraLinear::new(hidden, n_heads * q_head_dim, &qp_cfg, false)?;
            Ok(Self::Direct { q_proj })
        }
    }

    fn project(&mut self, x: &Array) -> Result<Array, LoraError> {
        match self {
            Self::LoRa { q_a_proj, q_a_layernorm, q_b_proj } => {
                let q_a_out = q_a_proj.forward(x)?;
                let q_a_norm =
                    pmetal_bridge::compat::Module::forward(q_a_layernorm, &q_a_out)
                        .map_err(LoraError::Mlx)?;
                q_b_proj.forward(&q_a_norm)
            }
            Self::Direct { q_proj } => q_proj.forward(x),
        }
    }

    fn num_trainable_params(&self) -> usize {
        match self {
            Self::LoRa { q_a_proj, q_b_proj, .. } => {
                q_a_proj.num_trainable_params() + q_b_proj.num_trainable_params()
            }
            Self::Direct { q_proj } => q_proj.num_trainable_params(),
        }
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        match self {
            Self::LoRa { q_a_proj, q_b_proj, .. } => {
                let (aq, al, at) = q_a_proj.memory_usage();
                let (bq, bl, bt) = q_b_proj.memory_usage();
                (aq + bq, al + bl, at + bt)
            }
            Self::Direct { q_proj } => q_proj.memory_usage(),
        }
    }
}

// ─── Attention ────────────────────────────────────────────────────────────────

/// QLoRA-enabled MLA attention layer for DeepSeek V2/V3.
#[derive(Debug)]
pub struct DeepSeekQloraAttention {
    pub config: DeepSeekConfig,
    pub n_heads: i32,
    pub scale: f32,
    pub layer_id: usize,
    pub q: DeepSeekQloraQProj,
    pub kv_a_proj_with_mqa: QLoraLinear,
    pub kv_a_layernorm: nn::RmsNorm,
    pub kv_b_proj: QLoraLinear,
    pub o_proj: QLoraLinear,
}

impl DeepSeekQloraAttention {
    pub fn new(
        config: &DeepSeekConfig,
        qcfg: &QLoraConfig,
        layer_id: usize,
    ) -> Result<Self, LoraError> {
        let hidden = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let q_head_dim = config.q_head_dim();
        let scale = (q_head_dim as f32).powf(-0.5);

        let q = DeepSeekQloraQProj::new(config, qcfg)?;

        let mut kva_cfg = qcfg.clone();
        kva_cfg.lora.r = crate::effective_rank(&qcfg.lora, "kv_a_proj_with_mqa");
        let kv_a_proj_with_mqa = QLoraLinear::new(
            hidden,
            config.kv_lora_rank + config.qk_rope_head_dim,
            &kva_cfg,
            false,
        )?;

        let kv_a_layernorm = nn::RmsNormBuilder::new(config.kv_lora_rank)
            .eps(1e-6)
            .build()
            .map_err(LoraError::Mlx)?;

        let mut kvb_cfg = qcfg.clone();
        kvb_cfg.lora.r = crate::effective_rank(&qcfg.lora, "kv_b_proj");
        let kv_b_proj = QLoraLinear::new(
            config.kv_lora_rank,
            n_heads * (config.qk_nope_head_dim + config.v_head_dim),
            &kvb_cfg,
            false,
        )?;

        let mut o_cfg = qcfg.clone();
        o_cfg.lora.r = crate::effective_rank(&qcfg.lora, "o_proj");
        let o_proj = QLoraLinear::new(n_heads * config.v_head_dim, hidden, &o_cfg, false)?;

        Ok(Self {
            config: config.clone(),
            n_heads,
            scale,
            layer_id,
            q,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
        })
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
        let q = self.q.project(x)?;
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

        // RoPE
        let q_pe = apply_rope(
            q_pe,
            self.config.qk_rope_head_dim,
            false,
            self.config.rope_theta,
            1.0,
            offset,
        )
        .map_err(LoraError::Mlx)?;
        let k_pe = apply_rope(
            &k_pe,
            self.config.qk_rope_head_dim,
            false,
            self.config.rope_theta,
            1.0,
            offset,
        )
        .map_err(LoraError::Mlx)?;

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

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (qq, ql, qt) = self.q.memory_usage();
        let (kaq, kal, kat) = self.kv_a_proj_with_mqa.memory_usage();
        let (kbq, kbl, kbt) = self.kv_b_proj.memory_usage();
        let (oq, ol, ot) = self.o_proj.memory_usage();
        (qq + kaq + kbq + oq, ql + kal + kbl + ol, qt + kat + kbt + ot)
    }
}

// ─── Dense MLP ───────────────────────────────────────────────────────────────

/// QLoRA-enabled dense MLP (SwiGLU) for DeepSeek dense layers.
#[derive(Debug)]
pub struct DeepSeekQloraMLP {
    pub gate_proj: QLoraLinear,
    pub up_proj: QLoraLinear,
    pub down_proj: QLoraLinear,
}

impl DeepSeekQloraMLP {
    pub fn new(hidden: i32, intermediate: i32, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        let mut gate_cfg = qcfg.clone();
        gate_cfg.lora.r = crate::effective_rank(&qcfg.lora, "gate_proj");
        let mut up_cfg = qcfg.clone();
        up_cfg.lora.r = crate::effective_rank(&qcfg.lora, "up_proj");
        let mut down_cfg = qcfg.clone();
        down_cfg.lora.r = crate::effective_rank(&qcfg.lora, "down_proj");
        Ok(Self {
            gate_proj: QLoraLinear::new(hidden, intermediate, &gate_cfg, false)?,
            up_proj: QLoraLinear::new(hidden, intermediate, &up_cfg, false)?,
            down_proj: QLoraLinear::new(intermediate, hidden, &down_cfg, false)?,
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

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (gq, gl, gt) = self.gate_proj.memory_usage();
        let (uq, ul, ut) = self.up_proj.memory_usage();
        let (dq, dl, dt) = self.down_proj.memory_usage();
        (gq + uq + dq, gl + ul + dl, gt + ut + dt)
    }
}

// ─── Shared expert ───────────────────────────────────────────────────────────

/// QLoRA-enabled shared expert for DeepSeek MoE layers.
pub type DeepSeekQloraSharedExpert = DeepSeekQloraMLP;

// ─── MoE block ───────────────────────────────────────────────────────────────

/// QLoRA-enabled MoE block for DeepSeek.
///
/// Routed experts are held frozen inside `frozen_moe`; the shared expert carries
/// QLoRA adapters.
#[derive(Debug)]
pub struct DeepSeekQloraMoE {
    /// Frozen routed experts (shared expert disabled on this instance).
    pub frozen_moe: DeepSeekMoE,
    /// QLoRA-adapted shared expert (optional).
    pub shared_experts: Option<DeepSeekQloraSharedExpert>,
}

impl DeepSeekQloraMoE {
    pub fn new(config: &DeepSeekConfig, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        // Build a config variant that disables shared experts so the frozen MoE
        // only handles the routed dispatch path.
        let mut routing_config = config.clone();
        routing_config.n_shared_experts = None;
        let frozen_moe = DeepSeekMoE::new(&routing_config).map_err(LoraError::Mlx)?;

        let shared_experts = if let Some(n_shared) = config.n_shared_experts {
            Some(DeepSeekQloraMLP::new(
                config.hidden_size,
                config.moe_intermediate_size * n_shared,
                qcfg,
            )?)
        } else {
            None
        };

        Ok(Self { frozen_moe, shared_experts })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let moe_out = self.frozen_moe.forward(x).map_err(LoraError::Mlx)?;
        if let Some(ref mut se) = self.shared_experts {
            Ok(moe_out.add(&se.forward(x)?))
        } else {
            Ok(moe_out)
        }
    }

    pub fn num_trainable_params(&self) -> usize {
        self.shared_experts
            .as_ref()
            .map_or(0, |s| s.num_trainable_params())
    }

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        self.shared_experts
            .as_ref()
            .map_or((0, 0, 0), |s| s.memory_usage())
    }
}

// ─── MLP dispatch enum ───────────────────────────────────────────────────────

#[derive(Debug)]
pub enum DeepSeekQloraMlpType {
    Dense(DeepSeekQloraMLP),
    MoE(DeepSeekQloraMoE),
}

impl DeepSeekQloraMlpType {
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

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        match self {
            Self::Dense(m) => m.memory_usage(),
            Self::MoE(m) => m.memory_usage(),
        }
    }
}

// ─── Decoder layer ───────────────────────────────────────────────────────────

/// QLoRA-enabled DeepSeek decoder layer.
#[derive(Debug)]
pub struct DeepSeekQloraDecoderLayer {
    pub layer_id: usize,
    pub self_attn: DeepSeekQloraAttention,
    pub mlp: DeepSeekQloraMlpType,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
}

impl DeepSeekQloraDecoderLayer {
    pub fn new(
        config: &DeepSeekConfig,
        qcfg: &QLoraConfig,
        layer_id: usize,
    ) -> Result<Self, LoraError> {
        let self_attn = DeepSeekQloraAttention::new(config, qcfg, layer_id)?;
        let mlp = if config.is_moe_layer(layer_id as i32) {
            DeepSeekQloraMlpType::MoE(DeepSeekQloraMoE::new(config, qcfg)?)
        } else {
            DeepSeekQloraMlpType::Dense(DeepSeekQloraMLP::new(
                config.hidden_size,
                config.intermediate_size,
                qcfg,
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
        Ok(Self { layer_id, self_attn, mlp, input_layernorm, post_attention_layernorm })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let normed =
            pmetal_bridge::compat::Module::forward(&mut self.input_layernorm, x)
                .map_err(LoraError::Mlx)?;
        let attn_out = self.self_attn.forward(&normed, mask, cache)?;
        let h = x.add(&attn_out);

        let normed =
            pmetal_bridge::compat::Module::forward(&mut self.post_attention_layernorm, &h)
                .map_err(LoraError::Mlx)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok(h.add(&mlp_out))
    }

    pub fn num_trainable_params(&self) -> usize {
        self.self_attn.num_trainable_params() + self.mlp.num_trainable_params()
    }

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        let (aq, al, at) = self.self_attn.memory_usage();
        let (mq, ml, mt) = self.mlp.memory_usage();
        (aq + mq, al + ml, at + mt)
    }
}

// ─── ModuleParameters ─────────────────────────────────────────────────────────

impl ModuleParameters for DeepSeekQloraQProj {
    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::LoRa { q_a_proj, q_a_layernorm, q_b_proj } => {
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
            Self::LoRa { q_a_proj, q_a_layernorm, q_b_proj } => {
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
            Self::LoRa { q_a_proj, q_a_layernorm, q_b_proj } => {
                (q_a_proj.lora_a.size() + q_a_proj.lora_b.size())
                    + q_a_layernorm.num_parameters()
                    + (q_b_proj.lora_a.size() + q_b_proj.lora_b.size())
            }
            Self::Direct { q_proj } => q_proj.lora_a.size() + q_proj.lora_b.size(),
        }
    }
}

impl ModuleParameters for DeepSeekQloraAttention {
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
        let mut kva = HashMap::new();
        kva.insert(Rc::from("lora_a"), NestedValue::Value(&self.kv_a_proj_with_mqa.lora_a));
        kva.insert(Rc::from("lora_b"), NestedValue::Value(&self.kv_a_proj_with_mqa.lora_b));
        m.insert(Rc::from("kv_a_proj_with_mqa"), NestedValue::Map(kva));
        let mut kvb = HashMap::new();
        kvb.insert(Rc::from("lora_a"), NestedValue::Value(&self.kv_b_proj.lora_a));
        kvb.insert(Rc::from("lora_b"), NestedValue::Value(&self.kv_b_proj.lora_b));
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
        kva.insert(Rc::from("lora_a"), NestedValue::Value(&self.kv_a_proj_with_mqa.lora_a));
        kva.insert(Rc::from("lora_b"), NestedValue::Value(&self.kv_a_proj_with_mqa.lora_b));
        m.insert(Rc::from("kv_a_proj_with_mqa"), NestedValue::Map(kva));
        let mut kvb = HashMap::new();
        kvb.insert(Rc::from("lora_a"), NestedValue::Value(&self.kv_b_proj.lora_a));
        kvb.insert(Rc::from("lora_b"), NestedValue::Value(&self.kv_b_proj.lora_b));
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
        kvb.insert(Rc::from("lora_a"), NestedValue::Value(&mut self.kv_b_proj.lora_a));
        kvb.insert(Rc::from("lora_b"), NestedValue::Value(&mut self.kv_b_proj.lora_b));
        m.insert(Rc::from("kv_b_proj"), NestedValue::Map(kvb));
        let mut op = HashMap::new();
        op.insert(Rc::from("lora_a"), NestedValue::Value(&mut self.o_proj.lora_a));
        op.insert(Rc::from("lora_b"), NestedValue::Value(&mut self.o_proj.lora_b));
        m.insert(Rc::from("o_proj"), NestedValue::Map(op));
        m
    }
}

impl ModuleParameters for DeepSeekQloraMLP {
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
        gp.insert(Rc::from("lora_a"), NestedValue::Value(&self.gate_proj.lora_a));
        gp.insert(Rc::from("lora_b"), NestedValue::Value(&self.gate_proj.lora_b));
        m.insert(Rc::from("gate_proj"), NestedValue::Map(gp));
        let mut up = HashMap::new();
        up.insert(Rc::from("lora_a"), NestedValue::Value(&self.up_proj.lora_a));
        up.insert(Rc::from("lora_b"), NestedValue::Value(&self.up_proj.lora_b));
        m.insert(Rc::from("up_proj"), NestedValue::Map(up));
        let mut dp = HashMap::new();
        dp.insert(Rc::from("lora_a"), NestedValue::Value(&self.down_proj.lora_a));
        dp.insert(Rc::from("lora_b"), NestedValue::Value(&self.down_proj.lora_b));
        m.insert(Rc::from("down_proj"), NestedValue::Map(dp));
        m
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        self.parameters()
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut m = ModuleParamMut::new();
        let mut gp = HashMap::new();
        gp.insert(Rc::from("lora_a"), NestedValue::Value(&mut self.gate_proj.lora_a));
        gp.insert(Rc::from("lora_b"), NestedValue::Value(&mut self.gate_proj.lora_b));
        m.insert(Rc::from("gate_proj"), NestedValue::Map(gp));
        let mut up = HashMap::new();
        up.insert(Rc::from("lora_a"), NestedValue::Value(&mut self.up_proj.lora_a));
        up.insert(Rc::from("lora_b"), NestedValue::Value(&mut self.up_proj.lora_b));
        m.insert(Rc::from("up_proj"), NestedValue::Map(up));
        let mut dp = HashMap::new();
        dp.insert(Rc::from("lora_a"), NestedValue::Value(&mut self.down_proj.lora_a));
        dp.insert(Rc::from("lora_b"), NestedValue::Value(&mut self.down_proj.lora_b));
        m.insert(Rc::from("down_proj"), NestedValue::Map(dp));
        m
    }
}

impl ModuleParameters for DeepSeekQloraMoE {
    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut m = self.frozen_moe.parameters();
        if let Some(ref s) = self.shared_experts {
            m.extend(s.parameters());
        }
        m
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
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
            + self.shared_experts.as_ref().map_or(0, |s| s.num_parameters())
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

impl ModuleParameters for DeepSeekQloraMlpType {
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

impl ModuleParameters for DeepSeekQloraDecoderLayer {
    fn num_parameters(&self) -> usize {
        self.self_attn.num_parameters()
            + self.mlp.num_parameters()
            + self.input_layernorm.num_parameters()
            + self.post_attention_layernorm.num_parameters()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut m = ModuleParamRef::new();
        m.insert(Rc::from("self_attn"), NestedValue::Map(self.self_attn.parameters()));
        m.insert(Rc::from("mlp"), NestedValue::Map(self.mlp.parameters()));
        m.extend(self.input_layernorm.parameters());
        m.extend(self.post_attention_layernorm.parameters());
        m
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        let mut m = ModuleParamRef::new();
        m.insert(Rc::from("self_attn"), NestedValue::Map(self.self_attn.trainable_parameters()));
        m.insert(Rc::from("mlp"), NestedValue::Map(self.mlp.trainable_parameters()));
        m
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut m = ModuleParamMut::new();
        m.insert(Rc::from("self_attn"), NestedValue::Map(self.self_attn.parameters_mut()));
        m.insert(Rc::from("mlp"), NestedValue::Map(self.mlp.parameters_mut()));
        m.extend(self.input_layernorm.parameters_mut());
        m.extend(self.post_attention_layernorm.parameters_mut());
        m
    }
}

// ─── Model trunk ─────────────────────────────────────────────────────────────

/// QLoRA-enabled DeepSeek trunk (without LM head).
#[derive(Debug)]
pub struct DeepSeekQloraModel {
    pub config: DeepSeekConfig,
    pub qlora_config: QLoraConfig,
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<DeepSeekQloraDecoderLayer>,
    pub norm: nn::RmsNorm,
}

impl DeepSeekQloraModel {
    pub fn new(config: DeepSeekConfig, qcfg: QLoraConfig) -> Result<Self, LoraError> {
        let embed_tokens =
            nn::Embedding::new(config.vocab_size, config.hidden_size).map_err(LoraError::Mlx)?;
        let layers = (0..config.num_hidden_layers as usize)
            .map(|i| DeepSeekQloraDecoderLayer::new(&config, &qcfg, i))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()
            .map_err(LoraError::Mlx)?;
        Ok(Self { config, qlora_config: qcfg, embed_tokens, layers, norm })
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
                tracing::trace!("deepseek_qlora checkpoint boundary at layer {}", idx + 1);
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

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        self.layers.iter().fold((0, 0, 0), |(tq, tl, tt), l| {
            let (q, lo, t) = l.memory_usage();
            (tq + q, tl + lo, tt + t)
        })
    }
}

impl ModuleParameters for DeepSeekQloraModel {
    fn num_parameters(&self) -> usize {
        self.embed_tokens.num_parameters()
            + self.layers.iter().map(|l| l.num_parameters()).sum::<usize>()
            + self.norm.num_parameters()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut m = ModuleParamRef::new();
        m.extend(self.embed_tokens.parameters());
        for (i, layer) in self.layers.iter().enumerate() {
            m.insert(Rc::from(format!("{}", i)), NestedValue::Map(layer.parameters()));
        }
        m.extend(self.norm.parameters());
        m
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        let mut m = ModuleParamRef::new();
        for (i, layer) in self.layers.iter().enumerate() {
            m.insert(Rc::from(format!("{}", i)), NestedValue::Map(layer.trainable_parameters()));
        }
        m
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut m = ModuleParamMut::new();
        m.extend(self.embed_tokens.parameters_mut());
        for (i, layer) in self.layers.iter_mut().enumerate() {
            m.insert(Rc::from(format!("{}", i)), NestedValue::Map(layer.parameters_mut()));
        }
        m.extend(self.norm.parameters_mut());
        m
    }
}

// ─── LoRA parameter helpers ───────────────────────────────────────────────────

fn collect_qlora_lora_params(model: &DeepSeekQloraModel) -> HashMap<Rc<str>, Array> {
    let mut params = HashMap::new();
    for (i, layer) in model.layers.iter().enumerate() {
        let lp = format!("layers.{}", i);
        let attn = &layer.self_attn;
        match &attn.q {
            DeepSeekQloraQProj::LoRa { q_a_proj, q_b_proj, .. } => {
                insert_q(&mut params, &format!("{}.self_attn.q_a_proj", lp), q_a_proj);
                insert_q(&mut params, &format!("{}.self_attn.q_b_proj", lp), q_b_proj);
            }
            DeepSeekQloraQProj::Direct { q_proj } => {
                insert_q(&mut params, &format!("{}.self_attn.q_proj", lp), q_proj);
            }
        }
        insert_q(
            &mut params,
            &format!("{}.self_attn.kv_a_proj_with_mqa", lp),
            &attn.kv_a_proj_with_mqa,
        );
        insert_q(&mut params, &format!("{}.self_attn.kv_b_proj", lp), &attn.kv_b_proj);
        insert_q(&mut params, &format!("{}.self_attn.o_proj", lp), &attn.o_proj);
        match &layer.mlp {
            DeepSeekQloraMlpType::Dense(m) => {
                insert_q(&mut params, &format!("{}.mlp.gate_proj", lp), &m.gate_proj);
                insert_q(&mut params, &format!("{}.mlp.up_proj", lp), &m.up_proj);
                insert_q(&mut params, &format!("{}.mlp.down_proj", lp), &m.down_proj);
            }
            DeepSeekQloraMlpType::MoE(m) => {
                if let Some(ref s) = m.shared_experts {
                    let sp = format!("{}.mlp.shared_experts", lp);
                    insert_q(&mut params, &format!("{}.gate_proj", sp), &s.gate_proj);
                    insert_q(&mut params, &format!("{}.up_proj", sp), &s.up_proj);
                    insert_q(&mut params, &format!("{}.down_proj", sp), &s.down_proj);
                }
            }
        }
    }
    params
}

fn insert_q(params: &mut HashMap<Rc<str>, Array>, prefix: &str, proj: &QLoraLinear) {
    params.insert(Rc::from(format!("{}.lora_a", prefix)), proj.lora_a.clone());
    params.insert(Rc::from(format!("{}.lora_b", prefix)), proj.lora_b.clone());
}

fn set_qlora_lora_params(model: &mut DeepSeekQloraModel, params: &HashMap<Rc<str>, Array>) {
    for (i, layer) in model.layers.iter_mut().enumerate() {
        let lp = format!("layers.{}", i);
        let attn = &mut layer.self_attn;
        match &mut attn.q {
            DeepSeekQloraQProj::LoRa { q_a_proj, q_b_proj, .. } => {
                apply_q_keys(&format!("{}.self_attn.q_a_proj", lp), params, q_a_proj);
                apply_q_keys(&format!("{}.self_attn.q_b_proj", lp), params, q_b_proj);
            }
            DeepSeekQloraQProj::Direct { q_proj } => {
                apply_q_keys(&format!("{}.self_attn.q_proj", lp), params, q_proj);
            }
        }
        apply_q_keys(
            &format!("{}.self_attn.kv_a_proj_with_mqa", lp),
            params,
            &mut attn.kv_a_proj_with_mqa,
        );
        apply_q_keys(&format!("{}.self_attn.kv_b_proj", lp), params, &mut attn.kv_b_proj);
        apply_q_keys(&format!("{}.self_attn.o_proj", lp), params, &mut attn.o_proj);
        match &mut layer.mlp {
            DeepSeekQloraMlpType::Dense(m) => {
                apply_q_keys(&format!("{}.mlp.gate_proj", lp), params, &mut m.gate_proj);
                apply_q_keys(&format!("{}.mlp.up_proj", lp), params, &mut m.up_proj);
                apply_q_keys(&format!("{}.mlp.down_proj", lp), params, &mut m.down_proj);
            }
            DeepSeekQloraMlpType::MoE(m) => {
                if let Some(ref mut s) = m.shared_experts {
                    let sp = format!("{}.mlp.shared_experts", lp);
                    apply_q_keys(&format!("{}.gate_proj", sp), params, &mut s.gate_proj);
                    apply_q_keys(&format!("{}.up_proj", sp), params, &mut s.up_proj);
                    apply_q_keys(&format!("{}.down_proj", sp), params, &mut s.down_proj);
                }
            }
        }
    }
}

fn apply_q_keys(prefix: &str, params: &HashMap<Rc<str>, Array>, proj: &mut QLoraLinear) {
    if let Some(v) = params.get(&Rc::from(format!("{}.lora_a", prefix))) {
        proj.lora_a = v.clone();
    }
    if let Some(v) = params.get(&Rc::from(format!("{}.lora_b", prefix))) {
        proj.lora_b = v.clone();
    }
}

fn save_qlora_lora_weights(
    model: &DeepSeekQloraModel,
    path: impl AsRef<Path>,
) -> Result<(), LoraError> {
    let params = collect_qlora_lora_params(model);
    crate::save_safetensors_map(path, &params)
}

fn load_qlora_lora_weights(
    model: &mut DeepSeekQloraModel,
    path: impl AsRef<Path>,
) -> Result<(), LoraError> {
    let path = path.as_ref();
    let file_path = if path.is_dir() {
        path.join("lora_weights.safetensors")
    } else {
        path.to_path_buf()
    };
    let loaded = crate::load_safetensors_map(&file_path)?;
    let params: HashMap<Rc<str>, Array> =
        loaded.into_iter().map(|(k, v)| (Rc::from(k.as_str()), v)).collect();
    set_qlora_lora_params(model, &params);
    Ok(())
}

fn create_causal_mask(seq_len: i32) -> Result<Array, Exception> {
    let indices: Vec<i32> = (0..seq_len).collect();
    let row = Array::from_slice(&indices, &[1, seq_len]);
    let col = Array::from_slice(&indices, &[seq_len, 1]);
    let mask_bool = row.less_equal(&col);
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let zero = Array::from_f32(0.0f32);
    Ok(pmetal_bridge::compat::ops::r#where(&mask_bool, &zero, &neg_inf)
        .reshape(&[1, 1, seq_len, seq_len]))
}

// ─── ForCausalLM ─────────────────────────────────────────────────────────────

/// QLoRA-enabled DeepSeek causal language model.
///
/// Compatible with all pmetal training loops. Base weights are stored in 4-bit NF4;
/// LoRA A/B matrices remain in full precision and are the only trainable parameters.
#[derive(Debug)]
pub struct DeepSeekQloraForCausalLM {
    pub model: DeepSeekQloraModel,
    /// LM head (frozen; present when `tie_word_embeddings` is false).
    pub lm_head: Option<nn::Linear>,
    /// Gradient checkpointing configuration.
    pub checkpoint_config: Option<CheckpointConfig>,
}

impl DeepSeekQloraForCausalLM {
    /// Build a new QLoRA model from a `LoraConfig` (uses default NF4 quantization settings).
    pub fn new(config: DeepSeekConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let qcfg = QLoraConfig::from_lora(lora_config);
        Self::with_qlora_config(config, qcfg)
    }

    /// Build with an explicit `QLoraConfig`.
    pub fn with_qlora_config(
        config: DeepSeekConfig,
        qcfg: QLoraConfig,
    ) -> Result<Self, LoraError> {
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
        let model = DeepSeekQloraModel::new(config, qcfg)?;
        Ok(Self { model, lm_head, checkpoint_config: None })
    }

    fn apply_lm_head(&mut self, h: &Array) -> Array {
        if let Some(ref mut head) = self.lm_head {
            pmetal_bridge::compat::Module::forward(head, h).unwrap()
        } else {
            self.model.embed_tokens.as_linear(h)
        }
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
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

    /// Stub: no packed-sequence position-ID support.
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
        let mut h =
            pmetal_bridge::compat::Module::forward(&mut self.model.embed_tokens, input_ids)
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

        let layers_per_block =
            ckpt.as_ref().map(|c| c.layers_per_block).unwrap_or(usize::MAX);
        let checkpointing = ckpt.as_ref().map(|c| c.enabled).unwrap_or(false);

        for (idx, layer) in self.model.layers.iter_mut().enumerate() {
            h = layer.forward(&h, mask_owned.as_ref(), None)?;
            if checkpointing && (idx + 1) % layers_per_block == 0 {
                tracing::trace!("deepseek_qlora neftune checkpoint at layer {}", idx + 1);
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
        collect_qlora_lora_params(&self.model)
    }

    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        set_qlora_lora_params(&mut self.model, params);
    }

    pub fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        save_qlora_lora_weights(&self.model, path)
    }

    pub fn load_lora_weights(&mut self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        load_qlora_lora_weights(&mut self.model, path)
    }

    pub fn get_lm_head_weight(&self) -> Option<Array> {
        if let Some(ref head) = self.lm_head {
            Some(head.weight.value.clone())
        } else {
            Some(self.model.embed_tokens.weight.value.clone())
        }
    }

    /// Load frozen base model weights from a flat weight map (HuggingFace key schema) and
    /// quantize them to NF4.
    ///
    /// Exact weight key schema ported from `DeepSeekLoraForCausalLM::load_base_weights`:
    ///
    /// - `model.embed_tokens.weight`
    /// - `model.layers.{i}.self_attn.q_a_proj.weight` / `q_b_proj.weight` (q_lora_rank set)
    /// - `model.layers.{i}.self_attn.q_proj.weight` (q_lora_rank absent)
    /// - `model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight`
    /// - `model.layers.{i}.self_attn.kv_a_layernorm.weight`
    /// - `model.layers.{i}.self_attn.kv_b_proj.weight`
    /// - `model.layers.{i}.self_attn.o_proj.weight`
    /// - `model.layers.{i}.mlp.{gate,up,down}_proj.weight` (dense layers)
    /// - `model.layers.{i}.mlp.gate.weight` + `e_score_correction_bias` (MoE gate)
    /// - `model.layers.{i}.mlp.experts.{j}.{gate,up,down}_proj.weight` (routed experts)
    /// - `model.layers.{i}.mlp.shared_experts.{gate,up,down}_proj.weight`
    /// - `model.layers.{i}.input_layernorm.weight`
    /// - `model.layers.{i}.post_attention_layernorm.weight`
    /// - `model.norm.weight`
    /// - `lm_head.weight`
    pub fn load_base_weights(
        &mut self,
        weights: &HashMap<String, Array>,
    ) -> Result<(), LoraError> {
        let qcfg = self.model.qlora_config.clone();

        // Embeddings (kept in full precision)
        if let Some(w) = weights.get("model.embed_tokens.weight") {
            self.model.embed_tokens.weight = Param::new(w.clone());
        }

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let pfx = format!("model.layers.{i}");

            // Attention projections — re-quantize from loaded full-precision weights
            match &mut layer.self_attn.q {
                DeepSeekQloraQProj::LoRa { q_a_proj, q_b_proj, q_a_layernorm } => {
                    if let Some(w) = weights.get(&format!("{pfx}.self_attn.q_a_proj.weight")) {
                        *q_a_proj = QLoraLinear::from_weight(w, None, &qcfg)?;
                    }
                    if let Some(w) =
                        weights.get(&format!("{pfx}.self_attn.q_a_layernorm.weight"))
                    {
                        q_a_layernorm.weight = Param::new(w.clone());
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.self_attn.q_b_proj.weight")) {
                        *q_b_proj = QLoraLinear::from_weight(w, None, &qcfg)?;
                    }
                }
                DeepSeekQloraQProj::Direct { q_proj } => {
                    if let Some(w) = weights.get(&format!("{pfx}.self_attn.q_proj.weight")) {
                        *q_proj = QLoraLinear::from_weight(w, None, &qcfg)?;
                    }
                }
            }

            if let Some(w) =
                weights.get(&format!("{pfx}.self_attn.kv_a_proj_with_mqa.weight"))
            {
                layer.self_attn.kv_a_proj_with_mqa =
                    QLoraLinear::from_weight(w, None, &qcfg)?;
            }
            if let Some(w) = weights.get(&format!("{pfx}.self_attn.kv_a_layernorm.weight")) {
                layer.self_attn.kv_a_layernorm.weight = Param::new(w.clone());
            }
            if let Some(w) = weights.get(&format!("{pfx}.self_attn.kv_b_proj.weight")) {
                layer.self_attn.kv_b_proj = QLoraLinear::from_weight(w, None, &qcfg)?;
            }
            if let Some(w) = weights.get(&format!("{pfx}.self_attn.o_proj.weight")) {
                layer.self_attn.o_proj = QLoraLinear::from_weight(w, None, &qcfg)?;
            }

            // Layer norms (full precision)
            if let Some(w) = weights.get(&format!("{pfx}.input_layernorm.weight")) {
                layer.input_layernorm.weight = Param::new(w.clone());
            }
            if let Some(w) =
                weights.get(&format!("{pfx}.post_attention_layernorm.weight"))
            {
                layer.post_attention_layernorm.weight = Param::new(w.clone());
            }

            // MLP / MoE
            match &mut layer.mlp {
                DeepSeekQloraMlpType::Dense(mlp) => {
                    if let Some(w) = weights.get(&format!("{pfx}.mlp.gate_proj.weight")) {
                        mlp.gate_proj = QLoraLinear::from_weight(w, None, &qcfg)?;
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.mlp.up_proj.weight")) {
                        mlp.up_proj = QLoraLinear::from_weight(w, None, &qcfg)?;
                    }
                    if let Some(w) = weights.get(&format!("{pfx}.mlp.down_proj.weight")) {
                        mlp.down_proj = QLoraLinear::from_weight(w, None, &qcfg)?;
                    }
                }
                DeepSeekQloraMlpType::MoE(moe) => {
                    // Frozen MoE gate (kept in base dtype — not quantized)
                    if let Some(w) = weights.get(&format!("{pfx}.mlp.gate.weight")) {
                        moe.frozen_moe.gate.weight.weight = Param::new(w.clone());
                    }
                    if let Some(b) =
                        weights.get(&format!("{pfx}.mlp.gate.e_score_correction_bias"))
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

                    // QLoRA-adapted shared expert base weights
                    if let Some(ref mut se) = moe.shared_experts {
                        if let Some(w) =
                            weights.get(&format!("{pfx}.mlp.shared_experts.gate_proj.weight"))
                        {
                            se.gate_proj = QLoraLinear::from_weight(w, None, &qcfg)?;
                        }
                        if let Some(w) =
                            weights.get(&format!("{pfx}.mlp.shared_experts.up_proj.weight"))
                        {
                            se.up_proj = QLoraLinear::from_weight(w, None, &qcfg)?;
                        }
                        if let Some(w) =
                            weights.get(&format!("{pfx}.mlp.shared_experts.down_proj.weight"))
                        {
                            se.down_proj = QLoraLinear::from_weight(w, None, &qcfg)?;
                        }
                    }
                }
            }
        }

        // Final norm (full precision)
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

    /// Load frozen base model weights from safetensors files in a directory and quantize.
    ///
    /// Handles single-file (`model.safetensors`) and sharded models.
    pub fn load_base_weights_from_dir(&mut self, model_dir: &Path) -> Result<(), LoraError> {
        let weights = pmetal_models::WeightLoader::load_safetensors(model_dir)
            .map_err(|e| LoraError::InvalidState(format!("failed to load base weights: {e:?}")))?;
        self.load_base_weights(&weights)
    }

    /// Evaluate (materialise) all model parameters on the device.
    pub fn eval_all(&mut self) -> Result<(), LoraError> {
        self.model.embed_tokens.weight.value.eval();

        for layer in &mut self.model.layers {
            match &layer.self_attn.q {
                DeepSeekQloraQProj::LoRa { q_a_proj, q_b_proj, q_a_layernorm } => {
                    q_a_proj.lora_a.eval();
                    q_a_proj.lora_b.eval();
                    q_a_layernorm.weight.value.eval();
                    q_b_proj.lora_a.eval();
                    q_b_proj.lora_b.eval();
                }
                DeepSeekQloraQProj::Direct { q_proj } => {
                    q_proj.lora_a.eval();
                    q_proj.lora_b.eval();
                }
            }
            layer.self_attn.kv_a_proj_with_mqa.lora_a.eval();
            layer.self_attn.kv_a_proj_with_mqa.lora_b.eval();
            layer.self_attn.kv_a_layernorm.weight.value.eval();
            layer.self_attn.kv_b_proj.lora_a.eval();
            layer.self_attn.kv_b_proj.lora_b.eval();
            layer.self_attn.o_proj.lora_a.eval();
            layer.self_attn.o_proj.lora_b.eval();

            layer.input_layernorm.weight.value.eval();
            layer.post_attention_layernorm.weight.value.eval();

            match &layer.mlp {
                DeepSeekQloraMlpType::Dense(mlp) => {
                    mlp.gate_proj.lora_a.eval();
                    mlp.gate_proj.lora_b.eval();
                    mlp.up_proj.lora_a.eval();
                    mlp.up_proj.lora_b.eval();
                    mlp.down_proj.lora_a.eval();
                    mlp.down_proj.lora_b.eval();
                }
                DeepSeekQloraMlpType::MoE(moe) => {
                    moe.frozen_moe.gate.weight.weight.value.eval();
                    moe.frozen_moe.gate.e_score_correction_bias.eval();
                    for expert in &moe.frozen_moe.moe.experts {
                        expert.w1.weight.eval();
                        expert.w3.weight.eval();
                        expert.w2.weight.eval();
                    }
                    if let Some(ref se) = moe.shared_experts {
                        se.gate_proj.lora_a.eval();
                        se.gate_proj.lora_b.eval();
                        se.up_proj.lora_a.eval();
                        se.up_proj.lora_b.eval();
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

    /// Stub: merge LoRA weights into base (no-op).
    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        Ok(())
    }

    /// Stub: unmerge LoRA weights from base (no-op).
    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        Ok(())
    }

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        self.model.memory_usage()
    }

    pub fn config(&self) -> &DeepSeekConfig {
        &self.model.config
    }

    pub fn qlora_config(&self) -> &QLoraConfig {
        &self.model.qlora_config
    }
}

// ─── ModuleParameters for ForCausalLM ────────────────────────────────────────

impl ModuleParameters for DeepSeekQloraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.model.num_parameters()
            + self.lm_head.as_ref().map_or(0, |h| h.num_parameters())
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
        m.insert(Rc::from("model"), NestedValue::Map(self.model.trainable_parameters()));
        m
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut m = ModuleParamMut::new();
        m.insert(Rc::from("model"), NestedValue::Map(self.model.parameters_mut()));
        if let Some(ref mut head) = self.lm_head {
            m.extend(head.parameters_mut());
        }
        m
    }
}

crate::impl_trainable_model!(DeepSeekQloraForCausalLM);

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainable::TrainableModel;
    use pmetal_models::architectures::deepseek::DeepSeekConfig;

    fn tiny_moe_config() -> DeepSeekConfig {
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

    fn tiny_dense_config() -> DeepSeekConfig {
        // All-dense (no MoE), Direct Q-proj path
        DeepSeekConfig {
            vocab_size: 512,
            hidden_size: 16,
            intermediate_size: 32,
            moe_intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(4),
            n_shared_experts: None,
            n_routed_experts: None,
            num_experts_per_tok: 0,
            moe_layer_freq: 0,
            first_k_dense_replace: 2,
            kv_lora_rank: 8,
            q_lora_rank: None,
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
    fn test_deepseek_qlora_moe_constructs() {
        let model =
            DeepSeekQloraForCausalLM::new(tiny_moe_config(), default_lora_config()).unwrap();
        assert!(model.num_trainable_params() > 0);
    }

    #[test]
    fn test_deepseek_qlora_dense_constructs() {
        let model =
            DeepSeekQloraForCausalLM::new(tiny_dense_config(), default_lora_config()).unwrap();
        assert!(model.num_trainable_params() > 0);
    }

    #[test]
    fn test_deepseek_qlora_forward_shape_moe() {
        let config = tiny_moe_config();
        let mut model =
            DeepSeekQloraForCausalLM::new(config.clone(), default_lora_config()).unwrap();
        let input_ids = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();
        logits.eval().unwrap();
        assert_eq!(logits.dim(0), 1);
        assert_eq!(logits.dim(1), 4);
        assert_eq!(logits.dim(2), config.vocab_size);
    }

    #[test]
    fn test_deepseek_qlora_forward_shape_dense() {
        let config = tiny_dense_config();
        let mut model =
            DeepSeekQloraForCausalLM::new(config.clone(), default_lora_config()).unwrap();
        let input_ids = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);
        let logits = model.forward(&input_ids, None).unwrap();
        logits.eval().unwrap();
        assert_eq!(logits.dim(2), config.vocab_size);
    }

    #[test]
    fn test_deepseek_qlora_forward_with_cache() {
        let config = tiny_moe_config();
        let mut model =
            DeepSeekQloraForCausalLM::new(config, default_lora_config()).unwrap();
        let mut cache = model.create_cache(32);

        let tok1 = Array::from_slice(&[1i32], &[1, 1]);
        let logits1 = model.forward_with_cache(&tok1, None, Some(&mut cache)).unwrap();
        logits1.eval().unwrap();

        let tok2 = Array::from_slice(&[2i32], &[1, 1]);
        let logits2 = model.forward_with_cache(&tok2, None, Some(&mut cache)).unwrap();
        logits2.eval().unwrap();

        assert_eq!(logits2.dim(1), 1);
    }

    #[test]
    fn test_deepseek_qlora_lora_params_roundtrip() {
        let mut model =
            DeepSeekQloraForCausalLM::new(tiny_moe_config(), default_lora_config()).unwrap();
        let params = model.lora_parameters();
        assert!(!params.is_empty());
        model.set_lora_parameters(&params);
        let params2 = model.lora_parameters();
        assert_eq!(params.len(), params2.len());
    }

    #[test]
    fn test_deepseek_qlora_supports_kv_cache() {
        let model =
            DeepSeekQloraForCausalLM::new(tiny_moe_config(), default_lora_config()).unwrap();
        assert!(TrainableModel::supports_kv_cache(&model));
        assert!(TrainableModel::create_cache(&model, 64).is_some());
    }

    #[test]
    fn test_deepseek_qlora_memory_usage() {
        let model =
            DeepSeekQloraForCausalLM::new(tiny_moe_config(), default_lora_config()).unwrap();
        let (_quantized, lora, _total) = model.memory_usage();
        assert!(lora > 0, "LoRA adapters must consume memory");
    }

    #[test]
    fn test_deepseek_qlora_trainable_model_trait() {
        use crate::TrainableModel;
        let mut model =
            DeepSeekQloraForCausalLM::new(tiny_moe_config(), default_lora_config()).unwrap();
        assert!(model.supports_kv_cache());
        assert!(model.supports_gradient_checkpointing());
        assert!(model.num_trainable_params() > 0);

        let input_ids = Array::from_slice(&[1i32, 2], &[1, 2]);
        let logits = TrainableModel::forward(&mut model, &input_ids, None).unwrap();
        logits.eval().unwrap();
    }
}
