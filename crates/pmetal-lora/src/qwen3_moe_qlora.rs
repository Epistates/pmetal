//! QLoRA-enabled Qwen3-MoE architecture.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, nn,
    ops,
};
use pmetal_core::LoraConfig;
use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::{RopeScaling, apply_rope},
};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};
use pmetal_models::architectures::qwen3_moe::Qwen3MoEConfig;

use crate::{
    LoraError, QLoraConfig, QLoraLinear, TrainableModel,
    qlora::quantize_lora_layer,
    qwen3_moe_lora::{
        Qwen3MoELoraAttention, Qwen3MoELoraDecoderLayer, Qwen3MoELoraFeedForward,
        Qwen3MoELoraForCausalLM, Qwen3MoELoraMoEBlock, Qwen3MoELoraModel,
    },
};

#[derive(Debug)]
pub struct Qwen3MoEQLoraAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_scale: f32,
    pub effective_base: f32,
    pub q_proj: QLoraLinear,
    pub k_proj: QLoraLinear,
    pub v_proj: QLoraLinear,
    pub o_proj: QLoraLinear,
    pub q_norm: nn::RmsNorm,
    pub k_norm: nn::RmsNorm,
}

impl Qwen3MoEQLoraAttention {
    fn from_lora(attn: Qwen3MoELoraAttention, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            n_heads: attn.n_heads,
            n_kv_heads: attn.n_kv_heads,
            head_dim: attn.head_dim,
            scale: attn.scale,
            rope_scale: attn.rope_scale,
            effective_base: attn.effective_base,
            q_proj: quantize_lora_layer(&attn.q_proj, qcfg)?,
            k_proj: quantize_lora_layer(&attn.k_proj, qcfg)?,
            v_proj: quantize_lora_layer(&attn.v_proj, qcfg)?,
            o_proj: quantize_lora_layer(&attn.o_proj, qcfg)?,
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
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let mut cache = cache;

        let mut q = self.q_proj.forward(x)?;
        let mut k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim]);
        k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);

        q = Module::forward(&mut self.q_norm, &q)?;
        k = Module::forward(&mut self.k_norm, &k)?;

        q = q.transpose_axes(&[0, 2, 1, 3]);
        k = k.transpose_axes(&[0, 2, 1, 3]);
        let v = v.transpose_axes(&[0, 2, 1, 3]);

        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let q = apply_rope(
            &q,
            self.head_dim,
            false,
            self.effective_base,
            self.rope_scale,
            offset,
        )?;
        let k = apply_rope(
            &k,
            self.head_dim,
            false,
            self.effective_base,
            self.rope_scale,
            offset,
        )?;

        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(if mask.is_some() {
                AttentionMaskType::None
            } else {
                AttentionMaskType::Causal
            });

        if mask.is_none()
            && let Some((cache_ref, layer_idx)) = cache.as_mut()
            && let Some(output) =
                (*cache_ref).try_turboquant_attention(*layer_idx, &q, &k, &v, &attn_config)?
        {
            let output = output.transpose_axes(&[0, 2, 1, 3]);
            let output = output.reshape(&[batch, seq_len, self.n_heads * self.head_dim]);
            return self.o_proj.forward(&output);
        }

        let (k, v) = if let Some((cache_ref, layer_idx)) = cache {
            cache_ref
                .update_and_fetch(layer_idx, &k, &v)
                .map_err(LoraError::Mlx)?
        } else {
            (k, v)
        };

        let output = fused_sdpa(&q, &k, &v, &attn_config, mask).map_err(LoraError::Mlx)?;
        let output = output.transpose_axes(&[0, 2, 1, 3]);
        let output = output.reshape(&[batch, seq_len, self.n_heads * self.head_dim]);
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

#[derive(Debug)]
pub struct Qwen3MoEQLoraDenseMLP {
    pub gate_proj: QLoraLinear,
    pub up_proj: QLoraLinear,
    pub down_proj: QLoraLinear,
}

impl Qwen3MoEQLoraDenseMLP {
    fn from_lora(
        mlp: crate::qwen3_moe_lora::Qwen3MoELoraDenseMLP,
        qcfg: &QLoraConfig,
    ) -> Result<Self, LoraError> {
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
pub struct Qwen3MoEQLoraMoEBlock {
    pub top_k: usize,
    pub norm_topk_prob: bool,
    pub gate: QLoraLinear,
    pub experts: Vec<pmetal_mlx::moe::Expert>,
    pub stacked_gate_proj: Option<Array>,
    pub stacked_up_proj: Option<Array>,
    pub stacked_down_proj: Option<Array>,
    pub stacked_weight_signature: Option<Vec<usize>>,
}

impl Qwen3MoEQLoraMoEBlock {
    fn from_lora(block: Qwen3MoELoraMoEBlock, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            top_k: block.top_k,
            norm_topk_prob: block.norm_topk_prob,
            gate: quantize_lora_layer(&block.gate, qcfg)?,
            experts: block.experts,
            stacked_gate_proj: block.stacked_gate_proj,
            stacked_up_proj: block.stacked_up_proj,
            stacked_down_proj: block.stacked_down_proj,
            stacked_weight_signature: block.stacked_weight_signature,
        })
    }

    fn current_stacked_weight_signature(&self) -> Vec<usize> {
        let mut signature = Vec::with_capacity(self.experts.len() * 3);
        for expert in &self.experts {
            signature.push(expert.w1.weight.as_ref().id());
            signature.push(expert.w3.weight.as_ref().id());
            signature.push(expert.w2.weight.as_ref().id());
        }
        signature
    }

    fn stack_expert_weights(&self) -> Result<(Array, Array, Array), Exception> {
        let gate_weights: Vec<Array> = self
            .experts
            .iter()
            .map(|expert| expert.w1.weight.as_ref().t())
            .collect();
        let up_weights: Vec<Array> = self
            .experts
            .iter()
            .map(|expert| expert.w3.weight.as_ref().t())
            .collect();
        let down_weights: Vec<Array> = self
            .experts
            .iter()
            .map(|expert| expert.w2.weight.as_ref().t())
            .collect();
        Ok((
            ops::stack_axis(&gate_weights, 0),
            ops::stack_axis(&up_weights, 0),
            ops::stack_axis(&down_weights, 0),
        ))
    }

    fn ensure_stacked_moe(&mut self) -> Result<(), Exception> {
        let signature = self.current_stacked_weight_signature();
        let needs_refresh = self.stacked_gate_proj.is_none()
            || self.stacked_up_proj.is_none()
            || self.stacked_down_proj.is_none()
            || self.stacked_weight_signature.as_ref() != Some(&signature);
        if needs_refresh {
            let (stacked_gate_proj, stacked_up_proj, stacked_down_proj) =
                self.stack_expert_weights()?;
            stacked_gate_proj.eval();
            stacked_up_proj.eval();
            stacked_down_proj.eval();
            self.stacked_gate_proj = Some(stacked_gate_proj);
            self.stacked_up_proj = Some(stacked_up_proj);
            self.stacked_down_proj = Some(stacked_down_proj);
            self.stacked_weight_signature = Some(signature);
        }
        Ok(())
    }

    fn batched_matmul(&self, x: &Array, w: &Array) -> Result<Array, Exception> {
        let x_expanded = x.reshape(&[x.dim(0), 1, x.dim(1)]);
        Ok(ops::matmul(&x_expanded, w).squeeze_axes(&[1]))
    }

    fn route_topk(&mut self, hidden_flat: &Array) -> Result<(i32, i32, Array, Array), LoraError> {
        let batch_seq = hidden_flat.dim(0);
        let hidden_size = hidden_flat.dim(1);
        let gate_logits = self.gate.forward(hidden_flat)?;
        let gate_logits_f32 = if gate_logits.dtype() != pmetal_bridge::compat::Dtype::Float32 {
            gate_logits.as_type::<f32>()
        } else {
            gate_logits
        };
        let routing_probs = ops::softmax_axis(&gate_logits_f32, -1);
        let (top_indices, normalized_weights) = pmetal_models::moe_routing::topk_normalize(
            &routing_probs,
            self.top_k as i32,
            self.norm_topk_prob,
        )
        .map_err(LoraError::Mlx)?;
        Ok((batch_seq, hidden_size, top_indices, normalized_weights))
    }

    fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        self.ensure_stacked_moe().map_err(LoraError::Mlx)?;
        let shape = x.shape();
        let hidden_flat = x.reshape(&[
            shape[..shape.len() - 1].iter().product(),
            shape[shape.len() - 1],
        ]);
        let (batch_seq, hidden_size, top_indices, normalized_weights) =
            self.route_topk(&hidden_flat)?;
        let top_k = self.top_k as i32;
        let mut output = ops::zeros_dtype(&[batch_seq, hidden_size], hidden_flat.dtype());

        for slot in 0..top_k {
            let slot_experts =
                ops::slice_axis(&top_indices, -1, slot, slot + 1).reshape(&[top_indices.dim(0)]);
            let slot_weights = ops::slice_axis(&normalized_weights, -1, slot, slot + 1);

            let gate_weights = self
                .stacked_gate_proj
                .as_ref()
                .unwrap()
                .take_axis(&slot_experts, 0);
            let up_weights = self
                .stacked_up_proj
                .as_ref()
                .unwrap()
                .take_axis(&slot_experts, 0);
            let down_weights = self
                .stacked_down_proj
                .as_ref()
                .unwrap()
                .take_axis(&slot_experts, 0);

            let gate_out = self.batched_matmul(&hidden_flat, &gate_weights)?;
            let up_out = self.batched_matmul(&hidden_flat, &up_weights)?;
            let activated = nn::silu(&gate_out).multiply(&up_out);
            let slot_out = self.batched_matmul(&activated, &down_weights)?;
            output = output.add(&slot_out.multiply(&slot_weights));
        }

        let mut output_shape = shape.to_vec();
        output_shape[shape.len() - 1] = hidden_size;
        Ok(output.reshape(&output_shape))
    }

    fn num_trainable_params(&self) -> usize {
        self.gate.num_trainable_params()
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        self.gate.memory_usage()
    }
}

// See `Qwen3MoELoraFeedForward` for the size-delta rationale.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum Qwen3MoEQLoraFeedForward {
    Dense(Qwen3MoEQLoraDenseMLP),
    MoE(Qwen3MoEQLoraMoEBlock),
}

impl Qwen3MoEQLoraFeedForward {
    fn from_lora(ffn: Qwen3MoELoraFeedForward, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        match ffn {
            Qwen3MoELoraFeedForward::Dense(mlp) => {
                Ok(Self::Dense(Qwen3MoEQLoraDenseMLP::from_lora(mlp, qcfg)?))
            }
            Qwen3MoELoraFeedForward::MoE(moe) => {
                Ok(Self::MoE(Qwen3MoEQLoraMoEBlock::from_lora(moe, qcfg)?))
            }
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::MoE(moe) => moe.forward(x),
        }
    }

    fn num_trainable_params(&self) -> usize {
        match self {
            Self::Dense(mlp) => mlp.num_trainable_params(),
            Self::MoE(moe) => moe.num_trainable_params(),
        }
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        match self {
            Self::Dense(mlp) => mlp.memory_usage(),
            Self::MoE(moe) => moe.memory_usage(),
        }
    }
}

#[derive(Debug)]
pub struct Qwen3MoEQLoraDecoderLayer {
    pub self_attn: Qwen3MoEQLoraAttention,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
    pub ffn: Qwen3MoEQLoraFeedForward,
}

impl Qwen3MoEQLoraDecoderLayer {
    fn from_lora(layer: Qwen3MoELoraDecoderLayer, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            self_attn: Qwen3MoEQLoraAttention::from_lora(layer.self_attn, qcfg)?,
            input_layernorm: layer.input_layernorm,
            post_attention_layernorm: layer.post_attention_layernorm,
            ffn: Qwen3MoEQLoraFeedForward::from_lora(layer.ffn, qcfg)?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let residual = x.clone();
        let hidden = Module::forward(&mut self.input_layernorm, x)?;
        let hidden = self.self_attn.forward(&hidden, mask, cache)?;
        let hidden = residual.add(&hidden);

        let residual = hidden.clone();
        let hidden = Module::forward(&mut self.post_attention_layernorm, &hidden)?;
        let hidden = self.ffn.forward(&hidden)?;
        Ok(residual.add(&hidden))
    }

    fn num_trainable_params(&self) -> usize {
        self.self_attn.num_trainable_params() + self.ffn.num_trainable_params()
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        let (aq, al, at) = self.self_attn.memory_usage();
        let (mq, ml, mt) = self.ffn.memory_usage();
        (aq + mq, al + ml, at + mt)
    }
}

#[derive(Debug)]
pub struct Qwen3MoEQLoraModel {
    pub config: Qwen3MoEConfig,
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<Qwen3MoEQLoraDecoderLayer>,
    pub norm: nn::RmsNorm,
}

impl Qwen3MoEQLoraModel {
    fn from_lora(model: Qwen3MoELoraModel, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            config: model.config,
            embed_tokens: model.embed_tokens,
            layers: model
                .layers
                .into_iter()
                .map(|layer| Qwen3MoEQLoraDecoderLayer::from_lora(layer, qcfg))
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
        self.layers
            .iter()
            .map(Qwen3MoEQLoraDecoderLayer::num_trainable_params)
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
pub struct Qwen3MoEQLoraForCausalLM {
    pub model: Qwen3MoEQLoraModel,
    pub lm_head: Option<nn::Linear>,
    pub qlora_config: QLoraConfig,
}

impl Qwen3MoEQLoraForCausalLM {
    pub fn new(config: Qwen3MoEConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        Self::with_qlora_config(config, QLoraConfig::from_lora(lora_config))
    }

    pub fn with_qlora_config(
        config: Qwen3MoEConfig,
        qlora_config: QLoraConfig,
    ) -> Result<Self, LoraError> {
        let lora = Qwen3MoELoraForCausalLM::new(config, qlora_config.lora.clone())?;
        Self::from_lora(lora, qlora_config)
    }

    fn from_lora(lora: Qwen3MoELoraForCausalLM, qcfg: QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            model: Qwen3MoEQLoraModel::from_lora(lora.model, &qcfg)?,
            lm_head: lora.lm_head,
            qlora_config: qcfg,
        })
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        self.forward_with_cache(input_ids, mask, None)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let hidden = self.model.forward(input_ids, mask, cache)?;
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
        self.model.forward(input_ids, mask, None)
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

    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        KVCache::new(KVCacheConfig::new(
            self.model.config.num_hidden_layers as usize,
            max_seq_len,
            self.model.config.num_kv_heads() as usize,
            self.model.config.head_dim as usize,
        ))
    }

    pub fn load_and_quantize_from_dir(
        &mut self,
        model_dir: impl AsRef<Path>,
    ) -> Result<(), LoraError> {
        let mut lora = Qwen3MoELoraForCausalLM::new(
            self.model.config.clone(),
            self.qlora_config.lora.clone(),
        )?;
        lora.load_base_weights_from_dir(model_dir)?;
        *self = Self::from_lora(lora, self.qlora_config.clone())?;
        Ok(())
    }

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
            match &layer.ffn {
                Qwen3MoEQLoraFeedForward::Dense(mlp) => {
                    let mlp_prefix = format!("layers.{i}.mlp");
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
                Qwen3MoEQLoraFeedForward::MoE(block) => {
                    params.insert(
                        Rc::from(format!("layers.{i}.mlp.gate.lora_a")),
                        block.gate.lora_a.clone(),
                    );
                    params.insert(
                        Rc::from(format!("layers.{i}.mlp.gate.lora_b")),
                        block.gate.lora_b.clone(),
                    );
                }
            }
        }
        params
    }

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
            set_param!(
                layer.self_attn.q_proj.lora_a,
                format!("{attn_prefix}.q_proj.lora_a")
            );
            set_param!(
                layer.self_attn.q_proj.lora_b,
                format!("{attn_prefix}.q_proj.lora_b")
            );
            set_param!(
                layer.self_attn.k_proj.lora_a,
                format!("{attn_prefix}.k_proj.lora_a")
            );
            set_param!(
                layer.self_attn.k_proj.lora_b,
                format!("{attn_prefix}.k_proj.lora_b")
            );
            set_param!(
                layer.self_attn.v_proj.lora_a,
                format!("{attn_prefix}.v_proj.lora_a")
            );
            set_param!(
                layer.self_attn.v_proj.lora_b,
                format!("{attn_prefix}.v_proj.lora_b")
            );
            set_param!(
                layer.self_attn.o_proj.lora_a,
                format!("{attn_prefix}.o_proj.lora_a")
            );
            set_param!(
                layer.self_attn.o_proj.lora_b,
                format!("{attn_prefix}.o_proj.lora_b")
            );
            match &mut layer.ffn {
                Qwen3MoEQLoraFeedForward::Dense(mlp) => {
                    let mlp_prefix = format!("layers.{i}.mlp");
                    set_param!(
                        mlp.gate_proj.lora_a,
                        format!("{mlp_prefix}.gate_proj.lora_a")
                    );
                    set_param!(
                        mlp.gate_proj.lora_b,
                        format!("{mlp_prefix}.gate_proj.lora_b")
                    );
                    set_param!(mlp.up_proj.lora_a, format!("{mlp_prefix}.up_proj.lora_a"));
                    set_param!(mlp.up_proj.lora_b, format!("{mlp_prefix}.up_proj.lora_b"));
                    set_param!(
                        mlp.down_proj.lora_a,
                        format!("{mlp_prefix}.down_proj.lora_a")
                    );
                    set_param!(
                        mlp.down_proj.lora_b,
                        format!("{mlp_prefix}.down_proj.lora_b")
                    );
                }
                Qwen3MoEQLoraFeedForward::MoE(block) => {
                    set_param!(block.gate.lora_a, format!("layers.{i}.mlp.gate.lora_a"));
                    set_param!(block.gate.lora_b, format!("layers.{i}.mlp.gate.lora_b"));
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
                let attn = layer.self_attn.q_proj.num_frozen_params() * 4
                    + layer.self_attn.k_proj.num_frozen_params() * 4
                    + layer.self_attn.v_proj.num_frozen_params() * 4
                    + layer.self_attn.o_proj.num_frozen_params() * 4;
                let ffn = match &layer.ffn {
                    Qwen3MoEQLoraFeedForward::Dense(mlp) => {
                        mlp.gate_proj.num_frozen_params() * 4
                            + mlp.up_proj.num_frozen_params() * 4
                            + mlp.down_proj.num_frozen_params() * 4
                    }
                    Qwen3MoEQLoraFeedForward::MoE(block) => block.gate.num_frozen_params() * 4,
                };
                attn + ffn
            })
            .sum::<usize>()
            + lora;
        (quantized + lora) as f32 / full_precision as f32
    }
}

impl ModuleParameters for Qwen3MoEQLoraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();
        for (i, layer) in self.model.layers.iter().enumerate() {
            let mut layer_params = HashMap::new();
            let mut attn_params = HashMap::new();
            for (name, proj) in [
                ("q_proj", &layer.self_attn.q_proj),
                ("k_proj", &layer.self_attn.k_proj),
                ("v_proj", &layer.self_attn.v_proj),
                ("o_proj", &layer.self_attn.o_proj),
            ] {
                let mut proj_params = HashMap::new();
                proj_params.insert(Rc::from("lora_a"), NestedValue::Value(&proj.lora_a));
                proj_params.insert(Rc::from("lora_b"), NestedValue::Value(&proj.lora_b));
                attn_params.insert(Rc::from(name), NestedValue::Map(proj_params));
            }
            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            let mut mlp_params = HashMap::new();
            match &layer.ffn {
                Qwen3MoEQLoraFeedForward::Dense(mlp) => {
                    for (name, proj) in [
                        ("gate_proj", &mlp.gate_proj),
                        ("up_proj", &mlp.up_proj),
                        ("down_proj", &mlp.down_proj),
                    ] {
                        let mut proj_params = HashMap::new();
                        proj_params.insert(Rc::from("lora_a"), NestedValue::Value(&proj.lora_a));
                        proj_params.insert(Rc::from("lora_b"), NestedValue::Value(&proj.lora_b));
                        mlp_params.insert(Rc::from(name), NestedValue::Map(proj_params));
                    }
                }
                Qwen3MoEQLoraFeedForward::MoE(block) => {
                    let mut gate_params = HashMap::new();
                    gate_params.insert(Rc::from("lora_a"), NestedValue::Value(&block.gate.lora_a));
                    gate_params.insert(Rc::from("lora_b"), NestedValue::Value(&block.gate.lora_b));
                    mlp_params.insert(Rc::from("gate"), NestedValue::Map(gate_params));
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

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut params = ModuleParamMut::new();
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let mut layer_params = HashMap::new();
            let mut attn_params = HashMap::new();
            for (name, proj) in [
                ("q_proj", &mut layer.self_attn.q_proj),
                ("k_proj", &mut layer.self_attn.k_proj),
                ("v_proj", &mut layer.self_attn.v_proj),
                ("o_proj", &mut layer.self_attn.o_proj),
            ] {
                let mut proj_params = HashMap::new();
                proj_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut proj.lora_a));
                proj_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut proj.lora_b));
                attn_params.insert(Rc::from(name), NestedValue::Map(proj_params));
            }
            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            let mut mlp_params = HashMap::new();
            match &mut layer.ffn {
                Qwen3MoEQLoraFeedForward::Dense(mlp) => {
                    for (name, proj) in [
                        ("gate_proj", &mut mlp.gate_proj),
                        ("up_proj", &mut mlp.up_proj),
                        ("down_proj", &mut mlp.down_proj),
                    ] {
                        let mut proj_params = HashMap::new();
                        proj_params
                            .insert(Rc::from("lora_a"), NestedValue::Value(&mut proj.lora_a));
                        proj_params
                            .insert(Rc::from("lora_b"), NestedValue::Value(&mut proj.lora_b));
                        mlp_params.insert(Rc::from(name), NestedValue::Map(proj_params));
                    }
                }
                Qwen3MoEQLoraFeedForward::MoE(block) => {
                    let mut gate_params = HashMap::new();
                    gate_params.insert(
                        Rc::from("lora_a"),
                        NestedValue::Value(&mut block.gate.lora_a),
                    );
                    gate_params.insert(
                        Rc::from("lora_b"),
                        NestedValue::Value(&mut block.gate.lora_b),
                    );
                    mlp_params.insert(Rc::from("gate"), NestedValue::Map(gate_params));
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

impl TrainableModel for Qwen3MoEQLoraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        Qwen3MoEQLoraForCausalLM::forward(self, input_ids, mask)
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        Qwen3MoEQLoraForCausalLM::forward_with_cache(self, input_ids, mask, cache)
    }

    fn num_trainable_params(&self) -> usize {
        self.num_trainable_params()
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        self.lora_parameters()
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        self.set_lora_parameters(params)
    }

    fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        self.save_lora_weights(path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        self.load_lora_weights(path)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        false
    }

    fn supports_kv_cache(&self) -> bool {
        true
    }

    fn create_cache(&self, max_seq_len: usize) -> Option<KVCache> {
        Some(self.create_cache(max_seq_len))
    }

    fn forward_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Option<Result<Array, LoraError>> {
        Some(self.forward_hidden_states(input_ids, mask))
    }

    fn lm_head_weight(&self) -> Option<Array> {
        self.get_lm_head_weight()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Qwen3MoEConfig {
        Qwen3MoEConfig {
            vocab_size: 128,
            hidden_size: 32,
            intermediate_size: 64,
            moe_intermediate_size: Some(16),
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: 8,
            num_experts: 4,
            num_experts_per_tok: 2,
            decoder_sparse_step: 1,
            ..Default::default()
        }
    }

    #[test]
    fn qwen3_moe_qlora_builds() {
        let model =
            Qwen3MoEQLoraForCausalLM::with_qlora_config(tiny_config(), QLoraConfig::default())
                .unwrap();
        assert!(model.num_trainable_params() > 0);
    }
}
