//! QLoRA-enabled Gemma 4 text model.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, Param,
    nn, ops,
};
use pmetal_core::LoraConfig;
use pmetal_models::architectures::gemma4::{Gemma4Config, Gemma4RmsNorm};

use crate::{
    LoraError, QLoraConfig, QLoraLinear, TrainableModel,
    gemma4_lora::{
        Gemma4LoraAttention, Gemma4LoraDecoderLayer, Gemma4LoraForCausalLM, Gemma4LoraModel,
        Gemma4LoraPerLayerInputBlock, Gemma4LoraPerLayerInputs,
    },
    qlora::quantize_lora_layer,
};

fn rms_norm_noscale(x: &Array, eps: f32) -> Array {
    pmetal_bridge::compat::fast::rms_norm_opt(x, None, eps)
}

fn layer_per_input(per_layer_inputs: &Array, layer_idx: usize) -> Array {
    let b = per_layer_inputs.dim(0);
    let s = per_layer_inputs.dim(1);
    let d = per_layer_inputs.dim(3);
    per_layer_inputs
        .slice(
            &[0, 0, layer_idx as i32, 0],
            &[b, s, layer_idx as i32 + 1, d],
        )
        .squeeze(2)
}

fn apply_gemma4_partial_rope(
    x: &Array,
    head_dim: i32,
    rotated_dims: i32,
    base: f32,
    offset: i32,
    partial_freqs: Option<&Array>,
) -> Result<Array, Exception> {
    if rotated_dims == 0 {
        return Ok(x.clone());
    }
    if rotated_dims == head_dim {
        return pmetal_mlx::kernels::rope::apply_rope(x, head_dim, false, base, 1.0, offset);
    }
    if let Some(freqs) = partial_freqs {
        return Ok(pmetal_bridge::compat::fast::rope_with_freqs(
            x, head_dim, false, 1.0, offset, freqs,
        ));
    }

    let shape = x.shape();
    let b = shape[0];
    let h = shape[1];
    let l = shape[2];
    let half = head_dim / 2;
    let rot_half = rotated_dims / 2;
    let left = x.slice(&[0, 0, 0, 0], &[b, h, l, half]);
    let right = x.slice(&[0, 0, 0, half], &[b, h, l, head_dim]);
    let left_rot = left.slice(&[0, 0, 0, 0], &[b, h, l, rot_half]);
    let right_rot = right.slice(&[0, 0, 0, 0], &[b, h, l, rot_half]);
    let rotated_input = ops::concatenate_axis(&[&left_rot, &right_rot], -1);
    let effective_base = base.powf(rotated_dims as f32 / head_dim as f32);
    let rotated = pmetal_mlx::kernels::rope::apply_rope(
        &rotated_input,
        rotated_dims,
        false,
        effective_base,
        1.0,
        offset,
    )?;
    let new_left_rot = rotated.slice(&[0, 0, 0, 0], &[b, h, l, rot_half]);
    let new_right_rot = rotated.slice(&[0, 0, 0, rot_half], &[b, h, l, rotated_dims]);
    let left_tail = left.slice(&[0, 0, 0, rot_half], &[b, h, l, half]);
    let right_tail = right.slice(&[0, 0, 0, rot_half], &[b, h, l, half]);
    let new_left = ops::concatenate_axis(&[&new_left_rot, &left_tail], -1);
    let new_right = ops::concatenate_axis(&[&new_right_rot, &right_tail], -1);
    Ok(ops::concatenate_axis(&[&new_left, &new_right], -1))
}

#[derive(Debug)]
pub struct Gemma4QLoraPerLayerInputs {
    pub embed_tokens: nn::Embedding,
    pub model_projection: QLoraLinear,
    pub projection_norm: Gemma4RmsNorm,
    pub embed_scale: f32,
    pub projection_scale: f32,
    pub input_scale: f32,
    pub num_layers: i32,
    pub per_layer_dim: i32,
    pub vocab_size: i32,
}

impl Gemma4QLoraPerLayerInputs {
    fn from_lora(inputs: Gemma4LoraPerLayerInputs, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            embed_tokens: inputs.embed_tokens,
            model_projection: quantize_lora_layer(&inputs.model_projection, qcfg)?,
            projection_norm: inputs.projection_norm,
            embed_scale: inputs.embed_scale,
            projection_scale: inputs.projection_scale,
            input_scale: inputs.input_scale,
            num_layers: inputs.num_layers,
            per_layer_dim: inputs.per_layer_dim,
            vocab_size: inputs.vocab_size,
        })
    }

    fn compute(&mut self, input_ids: &Array, inputs_embeds: &Array) -> Result<Array, LoraError> {
        let ge_zero = ops::greater_equal(input_ids, &Array::from_i32(0));
        let lt_vocab = ops::less(input_ids, &Array::from_i32(self.vocab_size));
        let mask = ops::logical_and(&ge_zero, &lt_vocab);
        let safe_input_ids = mask.where_cond(input_ids, &ops::zeros_like(input_ids));
        let per_layer_embeds = self
            .embed_tokens
            .forward(&safe_input_ids)
            .multiply(&Array::from_f32(self.embed_scale))
            .reshape(&[
                input_ids.dim(0),
                input_ids.dim(1),
                self.num_layers,
                self.per_layer_dim,
            ]);
        let projection = self
            .model_projection
            .forward(inputs_embeds)?
            .multiply(&Array::from_f32(self.projection_scale))
            .reshape(&[
                input_ids.dim(0),
                input_ids.dim(1),
                self.num_layers,
                self.per_layer_dim,
            ]);
        let projection = self.projection_norm.forward(&projection);
        Ok(projection
            .add(&per_layer_embeds)
            .multiply(&Array::from_f32(self.input_scale)))
    }

    fn num_trainable_params(&self) -> usize {
        self.model_projection.num_trainable_params()
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        self.model_projection.memory_usage()
    }
}

#[derive(Debug)]
pub struct Gemma4QLoraPerLayerInputBlock {
    pub gate_proj: QLoraLinear,
    pub projection: QLoraLinear,
    pub post_norm: Gemma4RmsNorm,
}

impl Gemma4QLoraPerLayerInputBlock {
    fn from_lora(
        block: Gemma4LoraPerLayerInputBlock,
        qcfg: &QLoraConfig,
    ) -> Result<Self, LoraError> {
        Ok(Self {
            gate_proj: quantize_lora_layer(&block.gate_proj, qcfg)?,
            projection: quantize_lora_layer(&block.projection, qcfg)?,
            post_norm: block.post_norm,
        })
    }

    fn forward(&mut self, hidden: &Array, layer_input: &Array) -> Result<Array, LoraError> {
        let residual = hidden.clone();
        let gate = self.gate_proj.forward(hidden)?;
        let activated = nn::gelu_tanh_approximate(&gate);
        let projected = self.projection.forward(&activated.multiply(layer_input))?;
        let projected = self.post_norm.forward(&projected);
        Ok(residual.add(&projected))
    }

    fn num_trainable_params(&self) -> usize {
        self.gate_proj.num_trainable_params() + self.projection.num_trainable_params()
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        let (gq, gl, gt) = self.gate_proj.memory_usage();
        let (pq, pl, pt) = self.projection.memory_usage();
        (gq + pq, gl + pl, gt + pt)
    }
}

#[derive(Debug)]
pub struct Gemma4QLoraMlp {
    pub gate_proj: QLoraLinear,
    pub up_proj: QLoraLinear,
    pub down_proj: QLoraLinear,
}

impl Gemma4QLoraMlp {
    fn from_lora(
        mlp: crate::gemma4_lora::Gemma4LoraMlp,
        qcfg: &QLoraConfig,
    ) -> Result<Self, LoraError> {
        Ok(Self {
            gate_proj: quantize_lora_layer(&mlp.gate_proj, qcfg)?,
            up_proj: quantize_lora_layer(&mlp.up_proj, qcfg)?,
            down_proj: quantize_lora_layer(&mlp.down_proj, qcfg)?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gelu_gate = nn::gelu_tanh_approximate(&gate);
        self.down_proj.forward(&gelu_gate.multiply(&up))
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
pub struct Gemma4QLoraAttention {
    pub q_proj: QLoraLinear,
    pub k_proj: QLoraLinear,
    pub v_proj: Option<QLoraLinear>,
    pub o_proj: QLoraLinear,
    pub q_norm: Gemma4RmsNorm,
    pub k_norm: Gemma4RmsNorm,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_base: f32,
    pub rope_partial_dims: i32,
    pub rms_norm_eps: f32,
    pub sliding_window: Option<i32>,
    pub rope_partial_freqs: Option<Array>,
}

impl Gemma4QLoraAttention {
    fn from_lora(attn: Gemma4LoraAttention, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            q_proj: quantize_lora_layer(&attn.q_proj, qcfg)?,
            k_proj: quantize_lora_layer(&attn.k_proj, qcfg)?,
            v_proj: attn
                .v_proj
                .map(|p| quantize_lora_layer(&p, qcfg))
                .transpose()?,
            o_proj: quantize_lora_layer(&attn.o_proj, qcfg)?,
            q_norm: attn.q_norm,
            k_norm: attn.k_norm,
            n_heads: attn.n_heads,
            n_kv_heads: attn.n_kv_heads,
            head_dim: attn.head_dim,
            rope_base: attn.rope_base,
            rope_partial_dims: attn.rope_partial_dims,
            rms_norm_eps: attn.rms_norm_eps,
            sliding_window: attn.sliding_window,
            rope_partial_freqs: attn.rope_partial_freqs,
        })
    }

    fn attention_mask_type(
        &self,
        query_len: i32,
        key_len: i32,
        mask: Option<&Array>,
    ) -> pmetal_mlx::kernels::AttentionMaskType {
        use pmetal_mlx::kernels::AttentionMaskType;
        if mask.is_some() {
            AttentionMaskType::None
        } else if let Some(w) = self.sliding_window {
            if query_len == 1 && key_len <= w {
                AttentionMaskType::None
            } else {
                AttentionMaskType::SlidingWindow(w)
            }
        } else if query_len == 1 {
            AttentionMaskType::None
        } else {
            AttentionMaskType::Causal
        }
    }

    fn attend(
        &mut self,
        q: &Array,
        k: &Array,
        v: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
        let query_len = q.dim(2);
        let key_len = k.dim(2);
        let attn_config = pmetal_mlx::kernels::FusedAttentionConfig::new(
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
        )
        .with_scale(1.0)
        .with_mask_type(self.attention_mask_type(query_len, key_len, mask));
        let output =
            pmetal_mlx::kernels::fused_sdpa(q, k, v, &attn_config, mask).map_err(LoraError::Mlx)?;
        let b = q.dim(0);
        let output = output.transpose_axes(&[0, 2, 1, 3]).reshape(&[
            b,
            query_len,
            self.n_heads * self.head_dim,
        ]);
        self.o_proj.forward(&output)
    }

    fn project_queries(&mut self, x: &Array, offset: i32) -> Result<Array, LoraError> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];
        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[b, l, self.n_heads, self.head_dim]);
        let q = self.q_norm.forward(&q).transpose_axes(&[0, 2, 1, 3]);
        apply_gemma4_partial_rope(
            &q,
            self.head_dim,
            self.rope_partial_dims,
            self.rope_base,
            offset,
            self.rope_partial_freqs.as_ref(),
        )
        .map_err(LoraError::Mlx)
    }

    fn project_qkv(&mut self, x: &Array, offset: i32) -> Result<(Array, Array, Array), LoraError> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];
        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[b, l, self.n_heads, self.head_dim]);
        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[b, l, self.n_kv_heads, self.head_dim]);
        let v_raw = match self.v_proj.as_mut() {
            Some(v_proj) => v_proj
                .forward(x)?
                .reshape(&[b, l, self.n_kv_heads, self.head_dim]),
            None => k.clone(),
        };
        let q = self.q_norm.forward(&q).transpose_axes(&[0, 2, 1, 3]);
        let k = self.k_norm.forward(&k).transpose_axes(&[0, 2, 1, 3]);
        let v = rms_norm_noscale(&v_raw, self.rms_norm_eps).transpose_axes(&[0, 2, 1, 3]);
        let q = apply_gemma4_partial_rope(
            &q,
            self.head_dim,
            self.rope_partial_dims,
            self.rope_base,
            offset,
            self.rope_partial_freqs.as_ref(),
        )
        .map_err(LoraError::Mlx)?;
        let k = apply_gemma4_partial_rope(
            &k,
            self.head_dim,
            self.rope_partial_dims,
            self.rope_base,
            offset,
            self.rope_partial_freqs.as_ref(),
        )
        .map_err(LoraError::Mlx)?;
        Ok((q, k, v))
    }

    fn forward_collect_kv(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        offset: i32,
    ) -> Result<(Array, Array, Array), LoraError> {
        let (q, k, v) = self.project_qkv(x, offset)?;
        let output = self.attend(&q, &k, &v, mask)?;
        Ok((output, k, v))
    }

    fn forward_with_shared_kv(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        source_keys: &Array,
        source_values: &Array,
        offset: i32,
    ) -> Result<Array, LoraError> {
        let q = self.project_queries(x, offset)?;
        self.attend(&q, source_keys, source_values, mask)
    }

    fn num_trainable_params(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self
                .v_proj
                .as_ref()
                .map(|p| p.num_trainable_params())
                .unwrap_or(0)
            + self.o_proj.num_trainable_params()
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        let (qq, ql, qt) = self.q_proj.memory_usage();
        let (kq, kl, kt) = self.k_proj.memory_usage();
        let (oq, ol, ot) = self.o_proj.memory_usage();
        let (vq, vl, vt) = self
            .v_proj
            .as_ref()
            .map(QLoraLinear::memory_usage)
            .unwrap_or((0, 0, 0));
        (qq + kq + vq + oq, ql + kl + vl + ol, qt + kt + vt + ot)
    }
}

#[derive(Debug)]
pub struct Gemma4QLoraDecoderLayer {
    pub input_layernorm: Gemma4RmsNorm,
    pub self_attn: Gemma4QLoraAttention,
    pub post_attention_layernorm: Gemma4RmsNorm,
    pub pre_feedforward_layernorm: Gemma4RmsNorm,
    pub mlp: Gemma4QLoraMlp,
    pub post_feedforward_layernorm: Gemma4RmsNorm,
    pub per_layer_input_block: Option<Gemma4QLoraPerLayerInputBlock>,
    pub layer_scalar: Param<Array>,
    pub kv_shared_source_layer: Option<usize>,
}

impl Gemma4QLoraDecoderLayer {
    fn from_lora(layer: Gemma4LoraDecoderLayer, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            input_layernorm: layer.input_layernorm,
            self_attn: Gemma4QLoraAttention::from_lora(layer.self_attn, qcfg)?,
            post_attention_layernorm: layer.post_attention_layernorm,
            pre_feedforward_layernorm: layer.pre_feedforward_layernorm,
            mlp: Gemma4QLoraMlp::from_lora(layer.mlp, qcfg)?,
            post_feedforward_layernorm: layer.post_feedforward_layernorm,
            per_layer_input_block: layer
                .per_layer_input_block
                .map(|b| Gemma4QLoraPerLayerInputBlock::from_lora(b, qcfg))
                .transpose()?,
            layer_scalar: layer.layer_scalar,
            kv_shared_source_layer: layer.kv_shared_source_layer,
        })
    }

    fn finish_forward(
        &mut self,
        residual_in: &Array,
        attn_out: &Array,
        layer_input: Option<&Array>,
    ) -> Result<Array, LoraError> {
        let h = self.post_attention_layernorm.forward(attn_out);
        let h = residual_in.add(&h);
        let residual = h.clone();
        let h = self.pre_feedforward_layernorm.forward(&h);
        let h = self.mlp.forward(&h)?;
        let h = self.post_feedforward_layernorm.forward(&h);
        let mut h = residual.add(&h);
        if let Some(layer_input) = layer_input
            && let Some(ref mut block) = self.per_layer_input_block
        {
            h = block.forward(&h, layer_input)?;
        }
        Ok(h.multiply(self.layer_scalar.as_ref()))
    }

    fn forward_collect_kv(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        offset: i32,
        layer_input: Option<&Array>,
    ) -> Result<(Array, Array, Array), LoraError> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        let (attn_out, keys, values) = self.self_attn.forward_collect_kv(&h, mask, offset)?;
        let hidden = self.finish_forward(&residual, &attn_out, layer_input)?;
        Ok((hidden, keys, values))
    }

    fn forward_with_shared_kv(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        source_keys: &Array,
        source_values: &Array,
        offset: i32,
        layer_input: Option<&Array>,
    ) -> Result<Array, LoraError> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        let attn_out =
            self.self_attn
                .forward_with_shared_kv(&h, mask, source_keys, source_values, offset)?;
        self.finish_forward(&residual, &attn_out, layer_input)
    }

    fn num_trainable_params(&self) -> usize {
        self.self_attn.num_trainable_params()
            + self.mlp.num_trainable_params()
            + self
                .per_layer_input_block
                .as_ref()
                .map(|b| b.num_trainable_params())
                .unwrap_or(0)
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        let (aq, al, at) = self.self_attn.memory_usage();
        let (mq, ml, mt) = self.mlp.memory_usage();
        let (bq, bl, bt) = self
            .per_layer_input_block
            .as_ref()
            .map(Gemma4QLoraPerLayerInputBlock::memory_usage)
            .unwrap_or((0, 0, 0));
        (aq + mq + bq, al + ml + bl, at + mt + bt)
    }
}

#[derive(Debug)]
pub struct Gemma4QLoraModel {
    pub embed_tokens: nn::Embedding,
    pub per_layer_inputs: Option<Gemma4QLoraPerLayerInputs>,
    pub layers: Vec<Gemma4QLoraDecoderLayer>,
    pub norm: Gemma4RmsNorm,
    pub config: Gemma4Config,
    pub embed_scale: f32,
}

impl Gemma4QLoraModel {
    fn from_lora(model: Gemma4LoraModel, qcfg: &QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            embed_tokens: model.embed_tokens,
            per_layer_inputs: model
                .per_layer_inputs
                .map(|inputs| Gemma4QLoraPerLayerInputs::from_lora(inputs, qcfg))
                .transpose()?,
            layers: model
                .layers
                .into_iter()
                .map(|layer| Gemma4QLoraDecoderLayer::from_lora(layer, qcfg))
                .collect::<Result<Vec<_>, _>>()?,
            norm: model.norm,
            config: model.config,
            embed_scale: model.embed_scale,
        })
    }

    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let mut h = self.embed_tokens.forward(input_ids);
        h = h.multiply(&Array::from_f32(self.embed_scale));
        let per_layer_inputs = if let Some(inputs) = self.per_layer_inputs.as_mut() {
            Some(inputs.compute(input_ids, &h)?)
        } else {
            None
        };
        let mut local_shared_kv = if self.config.num_kv_shared_layers() > 0 {
            Some((0..self.layers.len()).map(|_| None).collect::<Vec<_>>())
        } else {
            None
        };

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_input = per_layer_inputs
                .as_ref()
                .map(|inputs| layer_per_input(inputs, i));
            let layer_input_ref = layer_input.as_ref();
            if let Some(shared_source) = layer.kv_shared_source_layer {
                let (source_keys, source_values) = local_shared_kv
                    .as_ref()
                    .and_then(|entries| entries.get(shared_source))
                    .and_then(|entry| entry.as_ref())
                    .ok_or_else(|| {
                        LoraError::Mlx(Exception::custom(format!(
                            "Gemma 4 shared-KV layer {i} missing source layer {shared_source} activations"
                        )))
                    })?;
                h = layer.forward_with_shared_kv(
                    &h,
                    mask,
                    source_keys,
                    source_values,
                    0,
                    layer_input_ref,
                )?;
            } else if let Some(ref mut shared_kv) = local_shared_kv {
                let (next_h, keys, values) =
                    layer.forward_collect_kv(&h, mask, 0, layer_input_ref)?;
                shared_kv[i] = Some((keys, values));
                h = next_h;
            } else {
                let (next_h, _, _) = layer.forward_collect_kv(&h, mask, 0, layer_input_ref)?;
                h = next_h;
            }
        }
        Ok(self.norm.forward(&h))
    }

    fn num_trainable_params(&self) -> usize {
        self.layers
            .iter()
            .map(Gemma4QLoraDecoderLayer::num_trainable_params)
            .sum::<usize>()
            + self
                .per_layer_inputs
                .as_ref()
                .map(Gemma4QLoraPerLayerInputs::num_trainable_params)
                .unwrap_or(0)
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        let (mut quant, mut lora, mut total) = self
            .per_layer_inputs
            .as_ref()
            .map(Gemma4QLoraPerLayerInputs::memory_usage)
            .unwrap_or((0, 0, 0));
        for layer in &self.layers {
            let (q, l, t) = layer.memory_usage();
            quant += q;
            lora += l;
            total += t;
        }
        (quant, lora, total)
    }
}

#[derive(Debug)]
pub struct Gemma4QloraForCausalLM {
    pub model: Gemma4QLoraModel,
    pub qlora_config: QLoraConfig,
}

impl Gemma4QloraForCausalLM {
    pub fn new(config: Gemma4Config, lora_config: LoraConfig) -> Result<Self, LoraError> {
        Self::with_qlora_config(config, QLoraConfig::from_lora(lora_config))
    }

    pub fn with_qlora_config(config: Gemma4Config, qcfg: QLoraConfig) -> Result<Self, LoraError> {
        let lora = Gemma4LoraForCausalLM::new(config, qcfg.lora.clone())?;
        Self::from_lora(lora, qcfg)
    }

    fn from_lora(lora: Gemma4LoraForCausalLM, qcfg: QLoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            model: Gemma4QLoraModel::from_lora(lora.model, &qcfg)?,
            qlora_config: qcfg,
        })
    }

    fn logit_softcap(&self, logits: &Array) -> Array {
        if let Some(cap) = self.model.config.final_logit_softcapping {
            let cap_arr = Array::from_f32(cap);
            let scaled = logits.divide(&cap_arr);
            let tanh = ops::tanh(&scaled);
            tanh.multiply(&cap_arr)
        } else {
            logits.clone()
        }
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let hidden = self.model.forward(input_ids, mask)?;
        let logits = self.model.embed_tokens.as_linear(&hidden);
        Ok(self.logit_softcap(&logits))
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
        Some(self.model.embed_tokens.weight.value.clone())
    }

    pub fn load_and_quantize_from_dir(
        &mut self,
        model_dir: impl AsRef<Path>,
    ) -> Result<(), LoraError> {
        let mut lora =
            Gemma4LoraForCausalLM::new(self.model.config.clone(), self.qlora_config.lora.clone())?;
        lora.load_base_weights_from_dir(model_dir)?;
        *self = Self::from_lora(lora, self.qlora_config.clone())?;
        Ok(())
    }

    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        let mut params = HashMap::new();
        if let Some(ref inputs) = self.model.per_layer_inputs {
            params.insert(
                Rc::from("model.per_layer_inputs.model_projection.lora_a"),
                inputs.model_projection.lora_a.clone(),
            );
            params.insert(
                Rc::from("model.per_layer_inputs.model_projection.lora_b"),
                inputs.model_projection.lora_b.clone(),
            );
        }
        for (i, layer) in self.model.layers.iter().enumerate() {
            let attn_prefix = format!("layers.{i}.self_attn");
            for (name, proj) in [
                ("q_proj", &layer.self_attn.q_proj),
                ("k_proj", &layer.self_attn.k_proj),
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
            if let Some(ref v_proj) = layer.self_attn.v_proj {
                params.insert(
                    Rc::from(format!("{attn_prefix}.v_proj.lora_a")),
                    v_proj.lora_a.clone(),
                );
                params.insert(
                    Rc::from(format!("{attn_prefix}.v_proj.lora_b")),
                    v_proj.lora_b.clone(),
                );
            }
            let mlp_prefix = format!("layers.{i}.mlp");
            for (name, proj) in [
                ("gate_proj", &layer.mlp.gate_proj),
                ("up_proj", &layer.mlp.up_proj),
                ("down_proj", &layer.mlp.down_proj),
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
            if let Some(ref block) = layer.per_layer_input_block {
                let prefix = format!("layers.{i}.per_layer_input_block");
                params.insert(
                    Rc::from(format!("{prefix}.gate_proj.lora_a")),
                    block.gate_proj.lora_a.clone(),
                );
                params.insert(
                    Rc::from(format!("{prefix}.gate_proj.lora_b")),
                    block.gate_proj.lora_b.clone(),
                );
                params.insert(
                    Rc::from(format!("{prefix}.projection.lora_a")),
                    block.projection.lora_a.clone(),
                );
                params.insert(
                    Rc::from(format!("{prefix}.projection.lora_b")),
                    block.projection.lora_b.clone(),
                );
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
        if let Some(ref mut inputs) = self.model.per_layer_inputs {
            set_param!(
                inputs.model_projection.lora_a,
                "model.per_layer_inputs.model_projection.lora_a".to_string()
            );
            set_param!(
                inputs.model_projection.lora_b,
                "model.per_layer_inputs.model_projection.lora_b".to_string()
            );
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
            if let Some(ref mut v_proj) = layer.self_attn.v_proj {
                set_param!(v_proj.lora_a, format!("{attn_prefix}.v_proj.lora_a"));
                set_param!(v_proj.lora_b, format!("{attn_prefix}.v_proj.lora_b"));
            }
            set_param!(
                layer.self_attn.o_proj.lora_a,
                format!("{attn_prefix}.o_proj.lora_a")
            );
            set_param!(
                layer.self_attn.o_proj.lora_b,
                format!("{attn_prefix}.o_proj.lora_b")
            );
            let mlp_prefix = format!("layers.{i}.mlp");
            set_param!(
                layer.mlp.gate_proj.lora_a,
                format!("{mlp_prefix}.gate_proj.lora_a")
            );
            set_param!(
                layer.mlp.gate_proj.lora_b,
                format!("{mlp_prefix}.gate_proj.lora_b")
            );
            set_param!(
                layer.mlp.up_proj.lora_a,
                format!("{mlp_prefix}.up_proj.lora_a")
            );
            set_param!(
                layer.mlp.up_proj.lora_b,
                format!("{mlp_prefix}.up_proj.lora_b")
            );
            set_param!(
                layer.mlp.down_proj.lora_a,
                format!("{mlp_prefix}.down_proj.lora_a")
            );
            set_param!(
                layer.mlp.down_proj.lora_b,
                format!("{mlp_prefix}.down_proj.lora_b")
            );
            if let Some(ref mut block) = layer.per_layer_input_block {
                let prefix = format!("layers.{i}.per_layer_input_block");
                set_param!(block.gate_proj.lora_a, format!("{prefix}.gate_proj.lora_a"));
                set_param!(block.gate_proj.lora_b, format!("{prefix}.gate_proj.lora_b"));
                set_param!(
                    block.projection.lora_a,
                    format!("{prefix}.projection.lora_a")
                );
                set_param!(
                    block.projection.lora_b,
                    format!("{prefix}.projection.lora_b")
                );
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
        let mut full_precision = lora;
        if let Some(ref inputs) = self.model.per_layer_inputs {
            full_precision += inputs.model_projection.num_frozen_params() * 4;
        }
        for layer in &self.model.layers {
            full_precision += layer.self_attn.q_proj.num_frozen_params() * 4;
            full_precision += layer.self_attn.k_proj.num_frozen_params() * 4;
            full_precision += layer.self_attn.o_proj.num_frozen_params() * 4;
            if let Some(ref v_proj) = layer.self_attn.v_proj {
                full_precision += v_proj.num_frozen_params() * 4;
            }
            full_precision += layer.mlp.gate_proj.num_frozen_params() * 4;
            full_precision += layer.mlp.up_proj.num_frozen_params() * 4;
            full_precision += layer.mlp.down_proj.num_frozen_params() * 4;
            if let Some(ref block) = layer.per_layer_input_block {
                full_precision += block.gate_proj.num_frozen_params() * 4;
                full_precision += block.projection.num_frozen_params() * 4;
            }
        }
        (quantized + lora) as f32 / full_precision as f32
    }
}

impl ModuleParameters for Gemma4QloraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();
        if let Some(ref inputs) = self.model.per_layer_inputs {
            let mut input_params = HashMap::new();
            let mut proj_params = HashMap::new();
            proj_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&inputs.model_projection.lora_a),
            );
            proj_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&inputs.model_projection.lora_b),
            );
            input_params.insert(Rc::from("model_projection"), NestedValue::Map(proj_params));
            params.insert(
                Rc::from("model.per_layer_inputs"),
                NestedValue::Map(input_params),
            );
        }
        for (i, layer) in self.model.layers.iter().enumerate() {
            let mut layer_params = HashMap::new();
            let mut attn_params = HashMap::new();
            for (name, proj) in [
                ("q_proj", &layer.self_attn.q_proj),
                ("k_proj", &layer.self_attn.k_proj),
                ("o_proj", &layer.self_attn.o_proj),
            ] {
                let mut proj_params = HashMap::new();
                proj_params.insert(Rc::from("lora_a"), NestedValue::Value(&proj.lora_a));
                proj_params.insert(Rc::from("lora_b"), NestedValue::Value(&proj.lora_b));
                attn_params.insert(Rc::from(name), NestedValue::Map(proj_params));
            }
            if let Some(ref v_proj) = layer.self_attn.v_proj {
                let mut proj_params = HashMap::new();
                proj_params.insert(Rc::from("lora_a"), NestedValue::Value(&v_proj.lora_a));
                proj_params.insert(Rc::from("lora_b"), NestedValue::Value(&v_proj.lora_b));
                attn_params.insert(Rc::from("v_proj"), NestedValue::Map(proj_params));
            }
            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            let mut mlp_params = HashMap::new();
            for (name, proj) in [
                ("gate_proj", &layer.mlp.gate_proj),
                ("up_proj", &layer.mlp.up_proj),
                ("down_proj", &layer.mlp.down_proj),
            ] {
                let mut proj_params = HashMap::new();
                proj_params.insert(Rc::from("lora_a"), NestedValue::Value(&proj.lora_a));
                proj_params.insert(Rc::from("lora_b"), NestedValue::Value(&proj.lora_b));
                mlp_params.insert(Rc::from(name), NestedValue::Map(proj_params));
            }
            layer_params.insert(Rc::from("mlp"), NestedValue::Map(mlp_params));

            if let Some(ref block) = layer.per_layer_input_block {
                let mut block_params = HashMap::new();
                for (name, proj) in [
                    ("gate_proj", &block.gate_proj),
                    ("projection", &block.projection),
                ] {
                    let mut proj_params = HashMap::new();
                    proj_params.insert(Rc::from("lora_a"), NestedValue::Value(&proj.lora_a));
                    proj_params.insert(Rc::from("lora_b"), NestedValue::Value(&proj.lora_b));
                    block_params.insert(Rc::from(name), NestedValue::Map(proj_params));
                }
                layer_params.insert(
                    Rc::from("per_layer_input_block"),
                    NestedValue::Map(block_params),
                );
            }
            params.insert(
                Rc::from(format!("layers.{i}")),
                NestedValue::Map(layer_params),
            );
        }
        params
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut params = ModuleParamMut::new();
        if let Some(ref mut inputs) = self.model.per_layer_inputs {
            let mut input_params = HashMap::new();
            let mut proj_params = HashMap::new();
            proj_params.insert(
                Rc::from("lora_a"),
                NestedValue::Value(&mut inputs.model_projection.lora_a),
            );
            proj_params.insert(
                Rc::from("lora_b"),
                NestedValue::Value(&mut inputs.model_projection.lora_b),
            );
            input_params.insert(Rc::from("model_projection"), NestedValue::Map(proj_params));
            params.insert(
                Rc::from("model.per_layer_inputs"),
                NestedValue::Map(input_params),
            );
        }
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let mut layer_params = HashMap::new();
            let mut attn_params = HashMap::new();
            for (name, proj) in [
                ("q_proj", &mut layer.self_attn.q_proj),
                ("k_proj", &mut layer.self_attn.k_proj),
                ("o_proj", &mut layer.self_attn.o_proj),
            ] {
                let mut proj_params = HashMap::new();
                proj_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut proj.lora_a));
                proj_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut proj.lora_b));
                attn_params.insert(Rc::from(name), NestedValue::Map(proj_params));
            }
            if let Some(ref mut v_proj) = layer.self_attn.v_proj {
                let mut proj_params = HashMap::new();
                proj_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut v_proj.lora_a));
                proj_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut v_proj.lora_b));
                attn_params.insert(Rc::from("v_proj"), NestedValue::Map(proj_params));
            }
            layer_params.insert(Rc::from("self_attn"), NestedValue::Map(attn_params));

            let mut mlp_params = HashMap::new();
            for (name, proj) in [
                ("gate_proj", &mut layer.mlp.gate_proj),
                ("up_proj", &mut layer.mlp.up_proj),
                ("down_proj", &mut layer.mlp.down_proj),
            ] {
                let mut proj_params = HashMap::new();
                proj_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut proj.lora_a));
                proj_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut proj.lora_b));
                mlp_params.insert(Rc::from(name), NestedValue::Map(proj_params));
            }
            layer_params.insert(Rc::from("mlp"), NestedValue::Map(mlp_params));

            if let Some(ref mut block) = layer.per_layer_input_block {
                let mut block_params = HashMap::new();
                for (name, proj) in [
                    ("gate_proj", &mut block.gate_proj),
                    ("projection", &mut block.projection),
                ] {
                    let mut proj_params = HashMap::new();
                    proj_params.insert(Rc::from("lora_a"), NestedValue::Value(&mut proj.lora_a));
                    proj_params.insert(Rc::from("lora_b"), NestedValue::Value(&mut proj.lora_b));
                    block_params.insert(Rc::from(name), NestedValue::Map(proj_params));
                }
                layer_params.insert(
                    Rc::from("per_layer_input_block"),
                    NestedValue::Map(block_params),
                );
            }
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

impl TrainableModel for Gemma4QloraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        Gemma4QloraForCausalLM::forward(self, input_ids, mask)
    }

    fn num_trainable_params(&self) -> usize {
        Gemma4QloraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        Gemma4QloraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        Gemma4QloraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        Gemma4QloraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        Gemma4QloraForCausalLM::load_lora_weights(self, path)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        false
    }

    fn forward_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Option<Result<Array, LoraError>> {
        Some(Gemma4QloraForCausalLM::forward_hidden_states(
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

    fn tiny_config() -> Gemma4Config {
        Gemma4Config {
            model_type: "gemma4_text".to_string(),
            vocab_size: 128,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            global_head_dim: None,
            num_global_key_value_heads: None,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            attention_k_eq_v: false,
            tie_word_embeddings: false,
            sliding_window: 16,
            final_logit_softcapping: None,
            layer_types: vec!["sliding_attention".to_string(); 2],
            rope_parameters: None,
            _raw_rope_parameters: None,
            hidden_size_per_layer_input: Some(8),
            vocab_size_per_layer_input: Some(128),
            hidden_activation: Some("gelu_pytorch_tanh".to_string()),
            num_kv_shared_layers: Some(0),
            use_double_wide_mlp: Some(false),
            enable_moe_block: Some(false),
        }
    }

    #[test]
    fn gemma4_qlora_builds() {
        let model =
            Gemma4QloraForCausalLM::with_qlora_config(tiny_config(), QLoraConfig::default())
                .unwrap();
        assert!(model.num_trainable_params() > 0);
    }
}
