//! QLoRA-enabled Qwen3.5 / Qwen3Next hybrid architecture.
//!
//! This mirrors the existing LoRA placement strategy:
//! - Full attention layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`
//! - GDN layers: `in_proj_qkv`, `in_proj_z`, `out_proj`
//! - Dense MLP layers: `gate_proj`, `up_proj`, `down_proj`
//! - MoE layers: shared expert only (`gate_proj`, `up_proj`, `down_proj`)

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::sync::Once;
use std::sync::atomic::{AtomicUsize, Ordering};

use pmetal_bridge::compat::ops;
use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, Param,
    nn,
};
use pmetal_core::LoraConfig;
use pmetal_mlx::gather_mm;
use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, differentiable_attention,
    gated_delta::gated_delta_update,
    get_training_context,
    rope::{RopeScaling, apply_rope},
};
use pmetal_mlx::kv_cache::MambaCacheEntry;
use pmetal_models::architectures::qwen3_next::{Qwen3NextConfig, Qwen3NextRMSNormGated};

use crate::{
    LoraError, QLoraConfig, QLoraLinear, TrainableModel,
    qlora::quantize_lora_layer,
    qwen3_next_lora::{
        Qwen3NextLoraAttention, Qwen3NextLoraDecoderLayer, Qwen3NextLoraFeedForward,
        Qwen3NextLoraForCausalLM, Qwen3NextLoraGDN, Qwen3NextLoraMLP, Qwen3NextLoraModel,
        Qwen3NextLoraSharedExpert, Qwen3NextLoraSparseMoE,
    },
};

static LAYER_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);
static GRAD_CKPT_WARN: Once = Once::new();

fn reset_qwen3_next_layer_ids() {
    LAYER_ID_COUNTER.store(0, Ordering::SeqCst);
}

#[derive(Debug)]
pub struct Qwen3NextQloraAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,
    pub effective_base: f32,
    pub rope_scale: f32,
    pub layer_id: usize,
    pub q_proj: QLoraLinear,
    pub k_proj: QLoraLinear,
    pub v_proj: QLoraLinear,
    pub o_proj: QLoraLinear,
    pub q_norm: nn::RmsNorm,
    pub k_norm: nn::RmsNorm,
}

impl Qwen3NextQloraAttention {
    fn from_lora(attn: Qwen3NextLoraAttention, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            n_heads: attn.n_heads,
            n_kv_heads: attn.n_kv_heads,
            head_dim: attn.head_dim,
            scale: attn.scale,
            rope_dims: attn.rope_dims,
            effective_base: attn.effective_base,
            rope_scale: attn.rope_scale,
            layer_id: LAYER_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            q_proj: quantize_lora_layer(&attn.q_proj, qcfg)?,
            k_proj: quantize_lora_layer(&attn.k_proj, qcfg)?,
            v_proj: quantize_lora_layer(&attn.v_proj, qcfg)?,
            o_proj: quantize_lora_layer(&attn.o_proj, qcfg)?,
            q_norm: attn.q_norm,
            k_norm: attn.k_norm,
        })
    }

    fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];

        let q_proj_out = self.q_proj.forward(x)?;
        let q_gate = q_proj_out.reshape(&[b, l, self.n_heads, self.head_dim * 2]);
        let queries = pmetal_bridge::compat::ops::slice_last_to(&q_gate, self.head_dim);
        let gate = pmetal_bridge::compat::ops::slice_last_from(&q_gate, self.head_dim).reshape(&[
            b,
            l,
            self.n_heads * self.head_dim,
        ]);

        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let queries = Module::forward(&mut self.q_norm, &queries)?;
        let keys_reshaped = keys.reshape(&[b, l, self.n_kv_heads, self.head_dim]);
        let keys_normed = Module::forward(&mut self.k_norm, &keys_reshaped)?;
        let values = values.reshape(&[b, l, self.n_kv_heads, self.head_dim]);

        let queries = queries.transpose_axes(&[0, 2, 1, 3]);
        let keys = keys_normed.transpose_axes(&[0, 2, 1, 3]);
        let values = values.transpose_axes(&[0, 2, 1, 3]);

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
                value_head_dim: None,
                scale: self.scale,
                mask_type: AttentionMaskType::Causal,
                logit_softcapping: None,
            };
            differentiable_attention(self.layer_id, &queries, &keys, &values, &fa_config)
                .map_err(|e| LoraError::Mlx(Exception::custom(e.to_string())))?
        } else {
            let (keys, values) = if self.n_kv_heads < self.n_heads {
                let r = self.n_heads / self.n_kv_heads;
                (expand_kv_heads(&keys, r)?, expand_kv_heads(&values, r)?)
            } else {
                (keys, values)
            };
            let scores = queries.matmul(&keys.transpose_axes(&[0, 1, 3, 2]));
            let scores = scores.multiply(&Array::from_f32(self.scale));
            let scores = if let Some(m) = mask {
                scores.add(m)
            } else {
                scores
            };
            let weights = ops::softmax_axis(&scores, -1);
            weights.matmul(&values)
        };

        let output =
            output
                .transpose_axes(&[0, 2, 1, 3])
                .reshape(&[b, l, self.n_heads * self.head_dim]);
        let gated = output.multiply(&nn::sigmoid(&gate));
        self.o_proj.forward(&gated)
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

#[derive(Debug)]
pub struct Qwen3NextQloraGDN {
    pub conv1d: nn::Conv1d,
    pub in_proj_b: nn::Linear,
    pub in_proj_a: nn::Linear,
    pub norm: Qwen3NextRMSNormGated,
    pub dt_bias: Param<Array>,
    pub a_log: Param<Array>,
    pub in_proj_qkv: QLoraLinear,
    pub in_proj_z: QLoraLinear,
    pub out_proj: QLoraLinear,
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

impl Qwen3NextQloraGDN {
    fn from_lora(gdn: Qwen3NextLoraGDN, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            conv1d: gdn.conv1d,
            in_proj_b: gdn.in_proj_b,
            in_proj_a: gdn.in_proj_a,
            norm: gdn.norm,
            dt_bias: gdn.dt_bias,
            a_log: gdn.a_log,
            in_proj_qkv: quantize_lora_layer(&gdn.in_proj_qkv, qcfg)?,
            in_proj_z: quantize_lora_layer(&gdn.in_proj_z, qcfg)?,
            out_proj: quantize_lora_layer(&gdn.out_proj, qcfg)?,
            hidden_size: gdn.hidden_size,
            num_v_heads: gdn.num_v_heads,
            num_k_heads: gdn.num_k_heads,
            head_k_dim: gdn.head_k_dim,
            head_v_dim: gdn.head_v_dim,
            key_dim: gdn.key_dim,
            value_dim: gdn.value_dim,
            conv_dim: gdn.conv_dim,
            conv_kernel_size: gdn.conv_kernel_size,
        })
    }

    fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        mut cache: Option<&mut MambaCacheEntry>,
    ) -> Result<Array, LoraError> {
        let shape = inputs.shape();
        let b = shape[0];
        let s = shape[1];

        let qkv = self.in_proj_qkv.forward(inputs)?;
        let z = self
            .in_proj_z
            .forward(inputs)?
            .reshape(&[b, s, self.num_v_heads, self.head_v_dim]);
        let b_val = Module::forward(&mut self.in_proj_b, inputs)?;
        let a = Module::forward(&mut self.in_proj_a, inputs)?;

        let conv_state = if let Some(ref c) = cache {
            c.conv_state.clone()
        } else {
            None
        };
        let conv_state = conv_state.unwrap_or_else(|| {
            pmetal_bridge::compat::ops::zeros(
                &[b, self.conv_kernel_size - 1, self.conv_dim],
                pmetal_bridge::compat::Dtype::Float32,
            )
        });

        let qkv = if let Some(msk) = mask {
            let mask_expanded = msk.reshape(&[msk.dim(0), msk.dim(1), 1]);
            ops::r#where(&mask_expanded, &qkv, &Array::from_f32(0.0))
        } else {
            qkv
        };

        let conv_input = ops::concatenate_axis(&[&conv_state, &qkv], 1);
        if let Some(c) = cache.as_deref_mut() {
            let keep = self.conv_kernel_size - 1;
            let total_len = conv_input.dim(1);
            c.conv_state = Some(pmetal_bridge::compat::ops::slice_axis_from(
                &conv_input,
                1,
                total_len - keep,
            ));
        }

        let conv_out = nn::silu(&Module::forward(&mut self.conv1d, &conv_input)?);
        let q_conv = pmetal_bridge::compat::ops::slice_last_to(&conv_out, self.key_dim);
        let k_conv =
            pmetal_bridge::compat::ops::slice_axis(&conv_out, -1, self.key_dim, self.key_dim * 2);
        let v_conv = pmetal_bridge::compat::ops::slice_last_from(&conv_out, self.key_dim * 2);

        let out_len = q_conv.dim(1);
        let q_conv = pmetal_bridge::compat::ops::slice_axis_from(&q_conv, 1, out_len - s)
            .reshape(&[b, s, self.num_k_heads, self.head_k_dim]);
        let k_conv = pmetal_bridge::compat::ops::slice_axis_from(&k_conv, 1, out_len - s)
            .reshape(&[b, s, self.num_k_heads, self.head_k_dim]);
        let v_conv = pmetal_bridge::compat::ops::slice_axis_from(&v_conv, 1, out_len - s)
            .reshape(&[b, s, self.num_v_heads, self.head_v_dim]);

        let inv_scale = (self.head_k_dim as f32).powf(-0.5);
        let ones_weight = pmetal_bridge::compat::ops::ones(
            &[self.head_k_dim],
            pmetal_bridge::compat::Dtype::Float32,
        );
        let q_normed = pmetal_bridge::compat::fast::rms_norm(&q_conv, &ones_weight, 1e-6)
            .multiply(&Array::from_f32(inv_scale * inv_scale));
        let k_normed = pmetal_bridge::compat::fast::rms_norm(&k_conv, &ones_weight, 1e-6)
            .multiply(&Array::from_f32(inv_scale));

        let ssm_state = cache.as_ref().and_then(|c| c.ssm_state.as_ref());
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
        if let Some(c) = cache {
            c.ssm_state = Some(new_state);
        }

        let out = self.norm.forward(&out, Some(&z))?;
        self.out_proj.forward(&out.reshape(&[b, s, -1]))
    }

    fn num_trainable_params(&self) -> usize {
        self.in_proj_qkv.num_trainable_params()
            + self.in_proj_z.num_trainable_params()
            + self.out_proj.num_trainable_params()
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        let (qq, ql, qt) = self.in_proj_qkv.memory_usage();
        let (zq, zl, zt) = self.in_proj_z.memory_usage();
        let (oq, ol, ot) = self.out_proj.memory_usage();
        (qq + zq + oq, ql + zl + ol, qt + zt + ot)
    }
}

#[derive(Debug)]
pub struct Qwen3NextQloraMLP {
    pub gate_proj: QLoraLinear,
    pub up_proj: QLoraLinear,
    pub down_proj: QLoraLinear,
}

impl Qwen3NextQloraMLP {
    fn from_lora(mlp: Qwen3NextLoraMLP, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
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

#[derive(Debug)]
pub struct Qwen3NextQloraSharedExpert {
    pub gate_proj: QLoraLinear,
    pub up_proj: QLoraLinear,
    pub down_proj: QLoraLinear,
}

impl Qwen3NextQloraSharedExpert {
    fn from_lora(se: Qwen3NextLoraSharedExpert, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
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

#[derive(Debug)]
pub struct Qwen3NextQloraSparseMoE {
    pub gate: nn::Linear,
    pub switch_mlp_gate_proj: Param<Array>,
    pub switch_mlp_up_proj: Param<Array>,
    pub switch_mlp_down_proj: Param<Array>,
    pub shared_expert_gate: nn::Linear,
    pub shared_expert: Qwen3NextQloraSharedExpert,
    pub num_experts: i32,
    pub top_k: i32,
    pub norm_topk_prob: bool,
}

impl Qwen3NextQloraSparseMoE {
    fn from_lora(moe: Qwen3NextLoraSparseMoE, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            gate: moe.gate,
            switch_mlp_gate_proj: moe.switch_mlp_gate_proj,
            switch_mlp_up_proj: moe.switch_mlp_up_proj,
            switch_mlp_down_proj: moe.switch_mlp_down_proj,
            shared_expert_gate: moe.shared_expert_gate,
            shared_expert: Qwen3NextQloraSharedExpert::from_lora(moe.shared_expert, qcfg)?,
            num_experts: moe.num_experts,
            top_k: moe.top_k,
            norm_topk_prob: moe.norm_topk_prob,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let shape = x.shape();
        let batch_seq: i32 = shape[..shape.len() - 1].iter().product();
        let hidden = shape[shape.len() - 1];
        let x_flat = x.reshape(&[batch_seq, hidden]);

        let gate_logits = Module::forward(&mut self.gate, &x_flat)?;
        let gates = ops::softmax_axis(
            &if gate_logits.dtype() != pmetal_bridge::compat::Dtype::Float32 {
                gate_logits.as_type::<f32>()
            } else {
                gate_logits
            },
            -1,
        );

        let k = self.top_k;
        let neg_gates = gates.negative();
        let sorted_indices = ops::argsort_axis(&neg_gates, -1);
        let top_indices = pmetal_bridge::compat::ops::slice_last_to(&sorted_indices, k);
        let top_weights = gates.take_along_axis(&top_indices, -1);
        let top_weights = if self.norm_topk_prob {
            let weight_sum = top_weights.sum_axis(-1, true);
            let safe_sum = ops::maximum(&weight_sum, &Array::from_f32(1e-8));
            top_weights.divide(&safe_sum)
        } else {
            top_weights
        };
        let top_indices_i32 = top_indices.as_type::<i32>();

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
        let activated = nn::silu(&gate_out).multiply(&up_out);
        let down_out = gather_mm(
            &activated.reshape(&[batch_seq * k, -1]),
            self.switch_mlp_down_proj.as_ref(),
            None,
            Some(&top_indices_i32.reshape(&[batch_seq * k, 1])),
            false,
        )?
        .reshape(&[batch_seq, k, hidden]);
        let y = down_out
            .multiply(&top_weights.reshape(&[batch_seq, k, 1]))
            .sum_axis(-2, false);

        let shared_y = self.shared_expert.forward(&x_flat)?;
        let shared_gate = nn::sigmoid(&Module::forward(&mut self.shared_expert_gate, &x_flat)?);
        let shared_y = shared_gate.multiply(&shared_y);
        Ok(y.add(&shared_y).reshape(shape))
    }

    fn num_trainable_params(&self) -> usize {
        self.shared_expert.num_trainable_params()
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        self.shared_expert.memory_usage()
    }
}

// See `Qwen3MoELoraFeedForward` for the size-delta rationale.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum Qwen3NextQloraFeedForward {
    Dense(Qwen3NextQloraMLP),
    MoE(Qwen3NextQloraSparseMoE),
}

impl Qwen3NextQloraFeedForward {
    fn from_lora(ffn: Qwen3NextLoraFeedForward, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        match ffn {
            Qwen3NextLoraFeedForward::Dense(mlp) => {
                Ok(Self::Dense(Qwen3NextQloraMLP::from_lora(mlp, qcfg)?))
            }
            Qwen3NextLoraFeedForward::MoE(moe) => {
                Ok(Self::MoE(Qwen3NextQloraSparseMoE::from_lora(moe, qcfg)?))
            }
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        match self {
            Self::Dense(m) => m.forward(x),
            Self::MoE(m) => m.forward(x),
        }
    }

    fn num_trainable_params(&self) -> usize {
        match self {
            Self::Dense(m) => m.num_trainable_params(),
            Self::MoE(m) => m.num_trainable_params(),
        }
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        match self {
            Self::Dense(m) => m.memory_usage(),
            Self::MoE(m) => m.memory_usage(),
        }
    }
}

#[derive(Debug)]
pub struct Qwen3NextQloraDecoderLayer {
    pub is_linear: bool,
    pub linear_attn: Option<Qwen3NextQloraGDN>,
    pub self_attn: Option<Qwen3NextQloraAttention>,
    pub mlp: Qwen3NextQloraFeedForward,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
}

impl Qwen3NextQloraDecoderLayer {
    fn from_lora(layer: Qwen3NextLoraDecoderLayer, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            is_linear: layer.is_linear,
            linear_attn: layer
                .linear_attn
                .map(|gdn| Qwen3NextQloraGDN::from_lora(gdn, qcfg))
                .transpose()?,
            self_attn: layer
                .self_attn
                .map(|attn| Qwen3NextQloraAttention::from_lora(attn, qcfg))
                .transpose()?,
            mlp: Qwen3NextQloraFeedForward::from_lora(layer.mlp, qcfg)?,
            input_layernorm: layer.input_layernorm,
            post_attention_layernorm: layer.post_attention_layernorm,
        })
    }

    fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let normed = Module::forward(&mut self.input_layernorm, x)?;
        let r = if self.is_linear {
            self.linear_attn
                .as_mut()
                .expect("linear_attn must be Some for linear layers")
                .forward(&normed, mask, None)?
        } else {
            self.self_attn
                .as_mut()
                .expect("self_attn must be Some for attention layers")
                .forward(&normed, mask)?
        };
        let h = x.add(&r);
        let mlp_in = Module::forward(&mut self.post_attention_layernorm, &h)?;
        Ok(h.add(&self.mlp.forward(&mlp_in)?))
    }

    fn num_trainable_params(&self) -> usize {
        let mixer = if let Some(ref gdn) = self.linear_attn {
            gdn.num_trainable_params()
        } else if let Some(ref attn) = self.self_attn {
            attn.num_trainable_params()
        } else {
            0
        };
        mixer + self.mlp.num_trainable_params()
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        let mixer = if let Some(ref gdn) = self.linear_attn {
            gdn.memory_usage()
        } else if let Some(ref attn) = self.self_attn {
            attn.memory_usage()
        } else {
            (0, 0, 0)
        };
        let mlp = self.mlp.memory_usage();
        (mixer.0 + mlp.0, mixer.1 + mlp.1, mixer.2 + mlp.2)
    }
}

#[derive(Debug)]
pub struct Qwen3NextQloraModel {
    pub config: Qwen3NextConfig,
    pub qlora_config: QLoraConfig,
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<Qwen3NextQloraDecoderLayer>,
    pub norm: nn::RmsNorm,
}

impl Qwen3NextQloraModel {
    fn from_lora(model: Qwen3NextLoraModel, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        reset_qwen3_next_layer_ids();
        Ok(Self {
            config: model.config,
            qlora_config: qcfg.clone(),
            embed_tokens: model.embed_tokens,
            layers: model
                .layers
                .into_iter()
                .map(|layer| Qwen3NextQloraDecoderLayer::from_lora(layer, qcfg))
                .collect::<Result<Vec<_>, _>>()?,
            norm: model.norm,
        })
    }

    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let mut hidden = Module::forward(&mut self.embed_tokens, input_ids)?;
        let fa_mask = mask;
        let ssm_mask: Option<&Array> = None;
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let layer_mask = if layer.is_linear { ssm_mask } else { fa_mask };
            hidden = layer.forward(&hidden, layer_mask)?;
            if (idx + 1) % usize::MAX == 0 {
                GRAD_CKPT_WARN.call_once(|| {
                    tracing::info!("Qwen3Next QLoRA uses eager evaluation for memory management");
                });
            }
        }
        Module::forward(&mut self.norm, &hidden).map_err(LoraError::Mlx)
    }

    fn num_trainable_params(&self) -> usize {
        self.layers
            .iter()
            .map(Qwen3NextQloraDecoderLayer::num_trainable_params)
            .sum()
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        self.layers.iter().fold((0, 0, 0), |acc, layer| {
            let (q, l, t) = layer.memory_usage();
            (acc.0 + q, acc.1 + l, acc.2 + t)
        })
    }
}

#[derive(Debug)]
pub struct Qwen3NextQloraForCausalLM {
    pub model: Qwen3NextQloraModel,
    pub lm_head: Option<nn::Linear>,
}

impl Qwen3NextQloraForCausalLM {
    pub fn new(config: Qwen3NextConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        Self::with_qlora_config(config, QLoraConfig::from_lora(lora_config))
    }

    pub fn with_qlora_config(
        config: Qwen3NextConfig,
        qcfg: QLoraConfig,
    ) -> Result<Self, LoraError> {
        let lora = Qwen3NextLoraForCausalLM::new(config, qcfg.lora.clone())?;
        Self::from_lora(lora, qcfg)
    }

    fn from_lora(lora: Qwen3NextLoraForCausalLM, qcfg: QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            model: Qwen3NextQloraModel::from_lora(lora.model, &qcfg)?,
            lm_head: lora.lm_head,
        })
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let hidden = self.model.forward(input_ids, mask)?;
        if let Some(ref mut lm_head) = self.lm_head {
            Module::forward(lm_head, &hidden).map_err(LoraError::Mlx)
        } else {
            Ok(self.model.embed_tokens.as_linear(&hidden))
        }
    }

    pub fn forward_hidden_states(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
        self.model.forward(input_ids, mask)
    }

    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    pub fn get_lm_head_weight(&self) -> Option<Array> {
        if let Some(ref lm_head) = self.lm_head {
            Some(lm_head.weight.value.clone())
        } else {
            Some(self.model.embed_tokens.weight.value.clone())
        }
    }

    pub fn load_and_quantize_from_dir(
        &mut self,
        model_dir: impl AsRef<Path>,
    ) -> Result<(), LoraError> {
        let mut lora = Qwen3NextLoraForCausalLM::new(
            self.model.config.clone(),
            self.model.qlora_config.lora.clone(),
        )?;
        lora.load_base_weights_from_dir(model_dir)?;
        *self = Self::from_lora(lora, self.model.qlora_config.clone())?;
        Ok(())
    }

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
                Qwen3NextQloraFeedForward::Dense(mlp) => {
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
                Qwen3NextQloraFeedForward::MoE(moe) => {
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
                Qwen3NextQloraFeedForward::Dense(mlp) => {
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
                Qwen3NextQloraFeedForward::MoE(moe) => {
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

    pub fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        crate::save_safetensors_map(path, &self.lora_parameters())
    }

    pub fn load_lora_weights(&mut self, path: impl AsRef<Path>) -> Result<(), LoraError> {
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

    pub fn memory_usage(&self) -> (usize, usize, usize) {
        self.model.memory_usage()
    }

    pub fn memory_savings(&self) -> f32 {
        let (quantized, lora, _) = self.memory_usage();
        let full_precision = self
            .model
            .layers
            .iter()
            .map(|layer| {
                let mixer = if let Some(ref attn) = layer.self_attn {
                    attn.q_proj.num_frozen_params() * 4
                        + attn.k_proj.num_frozen_params() * 4
                        + attn.v_proj.num_frozen_params() * 4
                        + attn.o_proj.num_frozen_params() * 4
                } else if let Some(ref gdn) = layer.linear_attn {
                    gdn.in_proj_qkv.num_frozen_params() * 4
                        + gdn.in_proj_z.num_frozen_params() * 4
                        + gdn.out_proj.num_frozen_params() * 4
                } else {
                    0
                };
                let mlp = match &layer.mlp {
                    Qwen3NextQloraFeedForward::Dense(m) => {
                        m.gate_proj.num_frozen_params() * 4
                            + m.up_proj.num_frozen_params() * 4
                            + m.down_proj.num_frozen_params() * 4
                    }
                    Qwen3NextQloraFeedForward::MoE(m) => {
                        m.shared_expert.gate_proj.num_frozen_params() * 4
                            + m.shared_expert.up_proj.num_frozen_params() * 4
                            + m.shared_expert.down_proj.num_frozen_params() * 4
                    }
                };
                mixer + mlp
            })
            .sum::<usize>()
            + lora;
        (quantized + lora) as f32 / full_precision as f32
    }
}

impl ModuleParameters for Qwen3NextQloraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();
        for (i, layer) in self.model.layers.iter().enumerate() {
            let layer_key: Rc<str> = Rc::from(format!("layers.{i}"));
            let mut layer_map = HashMap::new();
            if let Some(ref attn) = layer.self_attn {
                let mut attn_map = HashMap::new();
                for (name, proj) in [
                    ("q_proj", &attn.q_proj),
                    ("k_proj", &attn.k_proj),
                    ("v_proj", &attn.v_proj),
                    ("o_proj", &attn.o_proj),
                ] {
                    let mut proj_map = HashMap::new();
                    proj_map.insert(Rc::from("lora_a"), NestedValue::Value(&proj.lora_a));
                    proj_map.insert(Rc::from("lora_b"), NestedValue::Value(&proj.lora_b));
                    attn_map.insert(Rc::from(name), NestedValue::Map(proj_map));
                }
                layer_map.insert(Rc::from("self_attn"), NestedValue::Map(attn_map));
            }
            if let Some(ref gdn) = layer.linear_attn {
                let mut gdn_map = HashMap::new();
                for (name, proj) in [
                    ("in_proj_qkv", &gdn.in_proj_qkv),
                    ("in_proj_z", &gdn.in_proj_z),
                    ("out_proj", &gdn.out_proj),
                ] {
                    let mut proj_map = HashMap::new();
                    proj_map.insert(Rc::from("lora_a"), NestedValue::Value(&proj.lora_a));
                    proj_map.insert(Rc::from("lora_b"), NestedValue::Value(&proj.lora_b));
                    gdn_map.insert(Rc::from(name), NestedValue::Map(proj_map));
                }
                layer_map.insert(Rc::from("linear_attn"), NestedValue::Map(gdn_map));
            }
            let mut mlp_map = HashMap::new();
            match &layer.mlp {
                Qwen3NextQloraFeedForward::Dense(mlp) => {
                    for (name, proj) in [
                        ("gate_proj", &mlp.gate_proj),
                        ("up_proj", &mlp.up_proj),
                        ("down_proj", &mlp.down_proj),
                    ] {
                        let mut proj_map = HashMap::new();
                        proj_map.insert(Rc::from("lora_a"), NestedValue::Value(&proj.lora_a));
                        proj_map.insert(Rc::from("lora_b"), NestedValue::Value(&proj.lora_b));
                        mlp_map.insert(Rc::from(name), NestedValue::Map(proj_map));
                    }
                }
                Qwen3NextQloraFeedForward::MoE(moe) => {
                    let mut shared_map = HashMap::new();
                    for (name, proj) in [
                        ("gate_proj", &moe.shared_expert.gate_proj),
                        ("up_proj", &moe.shared_expert.up_proj),
                        ("down_proj", &moe.shared_expert.down_proj),
                    ] {
                        let mut proj_map = HashMap::new();
                        proj_map.insert(Rc::from("lora_a"), NestedValue::Value(&proj.lora_a));
                        proj_map.insert(Rc::from("lora_b"), NestedValue::Value(&proj.lora_b));
                        shared_map.insert(Rc::from(name), NestedValue::Map(proj_map));
                    }
                    mlp_map.insert(Rc::from("shared_expert"), NestedValue::Map(shared_map));
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
                for (name, proj) in [
                    ("q_proj", &mut attn.q_proj),
                    ("k_proj", &mut attn.k_proj),
                    ("v_proj", &mut attn.v_proj),
                    ("o_proj", &mut attn.o_proj),
                ] {
                    let mut proj_map = HashMap::new();
                    proj_map.insert(Rc::from("lora_a"), NestedValue::Value(&mut proj.lora_a));
                    proj_map.insert(Rc::from("lora_b"), NestedValue::Value(&mut proj.lora_b));
                    attn_map.insert(Rc::from(name), NestedValue::Map(proj_map));
                }
                layer_map.insert(Rc::from("self_attn"), NestedValue::Map(attn_map));
            }
            if let Some(ref mut gdn) = layer.linear_attn {
                let mut gdn_map = HashMap::new();
                for (name, proj) in [
                    ("in_proj_qkv", &mut gdn.in_proj_qkv),
                    ("in_proj_z", &mut gdn.in_proj_z),
                    ("out_proj", &mut gdn.out_proj),
                ] {
                    let mut proj_map = HashMap::new();
                    proj_map.insert(Rc::from("lora_a"), NestedValue::Value(&mut proj.lora_a));
                    proj_map.insert(Rc::from("lora_b"), NestedValue::Value(&mut proj.lora_b));
                    gdn_map.insert(Rc::from(name), NestedValue::Map(proj_map));
                }
                layer_map.insert(Rc::from("linear_attn"), NestedValue::Map(gdn_map));
            }
            let mut mlp_map = HashMap::new();
            match &mut layer.mlp {
                Qwen3NextQloraFeedForward::Dense(mlp) => {
                    for (name, proj) in [
                        ("gate_proj", &mut mlp.gate_proj),
                        ("up_proj", &mut mlp.up_proj),
                        ("down_proj", &mut mlp.down_proj),
                    ] {
                        let mut proj_map = HashMap::new();
                        proj_map.insert(Rc::from("lora_a"), NestedValue::Value(&mut proj.lora_a));
                        proj_map.insert(Rc::from("lora_b"), NestedValue::Value(&mut proj.lora_b));
                        mlp_map.insert(Rc::from(name), NestedValue::Map(proj_map));
                    }
                }
                Qwen3NextQloraFeedForward::MoE(moe) => {
                    let mut shared_map = HashMap::new();
                    for (name, proj) in [
                        ("gate_proj", &mut moe.shared_expert.gate_proj),
                        ("up_proj", &mut moe.shared_expert.up_proj),
                        ("down_proj", &mut moe.shared_expert.down_proj),
                    ] {
                        let mut proj_map = HashMap::new();
                        proj_map.insert(Rc::from("lora_a"), NestedValue::Value(&mut proj.lora_a));
                        proj_map.insert(Rc::from("lora_b"), NestedValue::Value(&mut proj.lora_b));
                        shared_map.insert(Rc::from(name), NestedValue::Map(proj_map));
                    }
                    mlp_map.insert(Rc::from("shared_expert"), NestedValue::Map(shared_map));
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

    fn freeze_parameters(&mut self, _recursive: bool) {}
    fn unfreeze_parameters(&mut self, _recursive: bool) {}
    fn all_frozen(&self) -> Option<bool> {
        Some(false)
    }
    fn any_frozen(&self) -> Option<bool> {
        Some(false)
    }
}

impl TrainableModel for Qwen3NextQloraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        Qwen3NextQloraForCausalLM::forward(self, input_ids, mask)
    }

    fn num_trainable_params(&self) -> usize {
        Qwen3NextQloraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        Qwen3NextQloraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        Qwen3NextQloraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        Qwen3NextQloraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        Qwen3NextQloraForCausalLM::load_lora_weights(self, path)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        false
    }

    fn forward_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Option<Result<Array, LoraError>> {
        Some(Qwen3NextQloraForCausalLM::forward_hidden_states(
            self, input_ids, mask,
        ))
    }

    fn lm_head_weight(&self) -> Option<Array> {
        self.get_lm_head_weight()
    }
}

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

    #[test]
    fn qwen3_next_qlora_builds() {
        let model =
            Qwen3NextQloraForCausalLM::with_qlora_config(tiny_config(), QLoraConfig::default())
                .unwrap();
        assert!(model.num_trainable_params() > 0);
    }
}
