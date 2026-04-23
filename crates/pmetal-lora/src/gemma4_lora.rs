//! LoRA-enabled Gemma 4 text model.
//!
//! Adapter placement:
//! - Attention: `q_proj`, `k_proj`, optional `v_proj`, `o_proj`
//! - MLP: `gate_proj`, `up_proj`, `down_proj`
//! - Per-layer-input stack for 2B/4B models:
//!   - shared `model_projection`
//!   - per-layer `gate_proj` and `projection`

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, Param,
    nn, ops,
};
use pmetal_core::LoraConfig;
use pmetal_mlx::kernels::{AttentionMaskType, FusedAttentionConfig, fused_sdpa, rope::apply_rope};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};
use pmetal_models::architectures::gemma4::{
    Gemma4Attention as BaseGemma4Attention, Gemma4Config,
    Gemma4DecoderLayer as BaseGemma4DecoderLayer, Gemma4ForCausalLM as BaseGemma4ForCausalLM,
    Gemma4Mlp as BaseGemma4Mlp, Gemma4Model as BaseGemma4Model,
    Gemma4PerLayerInputBlock as BaseGemma4PerLayerInputBlock,
    Gemma4PerLayerInputs as BaseGemma4PerLayerInputs, Gemma4RmsNorm, load_gemma4_weights,
};

use crate::{LoraError, LoraLinear};

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
        return apply_rope(x, head_dim, false, base, 1.0, offset);
    }
    if let Some(freqs) = partial_freqs {
        return Ok(pmetal_bridge::compat::fast::rope_with_freqs(
            x, head_dim, false, 1.0, offset, freqs,
        ));
    }
    if rotated_dims % 2 != 0 || head_dim % 2 != 0 {
        return Err(Exception::custom(format!(
            "gemma4 partial rope requires even head_dim ({head_dim}) and rotated_dims ({rotated_dims})"
        )));
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
    let rotated = apply_rope(
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
pub struct Gemma4LoraPerLayerInputs {
    pub embed_tokens: nn::Embedding,
    pub model_projection: LoraLinear,
    pub projection_norm: Gemma4RmsNorm,
    pub embed_scale: f32,
    pub projection_scale: f32,
    pub input_scale: f32,
    pub num_layers: i32,
    pub per_layer_dim: i32,
    pub vocab_size: i32,
}

impl Gemma4LoraPerLayerInputs {
    pub fn from_base(
        base: BaseGemma4PerLayerInputs,
        lora_config: &LoraConfig,
    ) -> Result<Self, LoraError> {
        Ok(Self {
            embed_tokens: base.embed_tokens,
            model_projection: LoraLinear::from_linear(
                &base.model_projection,
                crate::effective_rank(lora_config, "model_projection") as i32,
                lora_config.alpha,
                lora_config.use_rslora,
            )?,
            projection_norm: base.projection_norm,
            embed_scale: base.embed_scale,
            projection_scale: base.projection_scale,
            input_scale: base.input_scale,
            num_layers: base.num_layers,
            per_layer_dim: base.per_layer_dim,
            vocab_size: base.vocab_size,
        })
    }

    pub fn compute(
        &mut self,
        input_ids: &Array,
        inputs_embeds: &Array,
    ) -> Result<Array, LoraError> {
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

    pub fn num_trainable_params(&self) -> usize {
        self.model_projection.num_trainable_params()
    }
}

#[derive(Debug)]
pub struct Gemma4LoraPerLayerInputBlock {
    pub gate_proj: LoraLinear,
    pub projection: LoraLinear,
    pub post_norm: Gemma4RmsNorm,
}

impl Gemma4LoraPerLayerInputBlock {
    pub fn from_base(
        base: BaseGemma4PerLayerInputBlock,
        lora_config: &LoraConfig,
    ) -> Result<Self, LoraError> {
        Ok(Self {
            gate_proj: LoraLinear::from_linear(
                &base.gate_proj,
                crate::effective_rank(lora_config, "gate_proj") as i32,
                lora_config.alpha,
                lora_config.use_rslora,
            )?,
            projection: LoraLinear::from_linear(
                &base.projection,
                crate::effective_rank(lora_config, "projection") as i32,
                lora_config.alpha,
                lora_config.use_rslora,
            )?,
            post_norm: base.post_norm,
        })
    }

    pub fn forward(&mut self, hidden: &Array, layer_input: &Array) -> Result<Array, LoraError> {
        let residual = hidden.clone();
        let gate = self.gate_proj.forward(hidden)?;
        let activated = nn::gelu_tanh_approximate(&gate);
        let projected = self.projection.forward(&activated.multiply(layer_input))?;
        let projected = self.post_norm.forward(&projected);
        Ok(residual.add(&projected))
    }

    pub fn num_trainable_params(&self) -> usize {
        self.gate_proj.num_trainable_params() + self.projection.num_trainable_params()
    }
}

#[derive(Debug)]
pub struct Gemma4LoraMlp {
    pub gate_proj: LoraLinear,
    pub up_proj: LoraLinear,
    pub down_proj: LoraLinear,
}

impl Gemma4LoraMlp {
    pub fn from_base(base: BaseGemma4Mlp, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            gate_proj: LoraLinear::from_linear(
                &base.gate_proj,
                crate::effective_rank(lora_config, "gate_proj") as i32,
                lora_config.alpha,
                lora_config.use_rslora,
            )?,
            up_proj: LoraLinear::from_linear(
                &base.up_proj,
                crate::effective_rank(lora_config, "up_proj") as i32,
                lora_config.alpha,
                lora_config.use_rslora,
            )?,
            down_proj: LoraLinear::from_linear(
                &base.down_proj,
                crate::effective_rank(lora_config, "down_proj") as i32,
                lora_config.alpha,
                lora_config.use_rslora,
            )?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gelu_gate = nn::gelu_tanh_approximate(&gate);
        self.down_proj.forward(&gelu_gate.multiply(&up))
    }

    pub fn num_trainable_params(&self) -> usize {
        self.gate_proj.num_trainable_params()
            + self.up_proj.num_trainable_params()
            + self.down_proj.num_trainable_params()
    }
}

#[derive(Debug)]
pub struct Gemma4LoraAttention {
    pub q_proj: LoraLinear,
    pub k_proj: LoraLinear,
    pub v_proj: Option<LoraLinear>,
    pub o_proj: LoraLinear,
    pub q_norm: Gemma4RmsNorm,
    pub k_norm: Gemma4RmsNorm,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_base: f32,
    pub rope_partial_dims: i32,
    pub is_full_attention: bool,
    pub rms_norm_eps: f32,
    pub use_k_eq_v: bool,
    pub sliding_window: Option<i32>,
    pub rope_partial_freqs: Option<Array>,
}

impl Gemma4LoraAttention {
    pub fn from_base(
        base: BaseGemma4Attention,
        lora_config: &LoraConfig,
    ) -> Result<Self, LoraError> {
        Ok(Self {
            q_proj: LoraLinear::from_linear(
                &base.q_proj,
                crate::effective_rank(lora_config, "q_proj") as i32,
                lora_config.alpha,
                lora_config.use_rslora,
            )?,
            k_proj: LoraLinear::from_linear(
                &base.k_proj,
                crate::effective_rank(lora_config, "k_proj") as i32,
                lora_config.alpha,
                lora_config.use_rslora,
            )?,
            v_proj: base
                .v_proj
                .map(|proj| {
                    LoraLinear::from_linear(
                        &proj,
                        crate::effective_rank(lora_config, "v_proj") as i32,
                        lora_config.alpha,
                        lora_config.use_rslora,
                    )
                })
                .transpose()?,
            o_proj: LoraLinear::from_linear(
                &base.o_proj,
                crate::effective_rank(lora_config, "o_proj") as i32,
                lora_config.alpha,
                lora_config.use_rslora,
            )?,
            q_norm: base.q_norm,
            k_norm: base.k_norm,
            n_heads: base.n_heads,
            n_kv_heads: base.n_kv_heads,
            head_dim: base.head_dim,
            rope_base: base.rope_base,
            rope_partial_dims: base.rope_partial_dims,
            is_full_attention: base.is_full_attention,
            rms_norm_eps: base.rms_norm_eps,
            use_k_eq_v: base.use_k_eq_v,
            sliding_window: base.sliding_window,
            rope_partial_freqs: base.rope_partial_freqs,
        })
    }

    fn attention_mask_type(
        &self,
        query_len: i32,
        key_len: i32,
        mask: Option<&Array>,
    ) -> AttentionMaskType {
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
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(1.0)
            .with_mask_type(self.attention_mask_type(query_len, key_len, mask));
        let output = fused_sdpa(q, k, v, &attn_config, mask).map_err(LoraError::Mlx)?;
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

        let partial_freqs = self.rope_partial_freqs.as_ref();
        let q = apply_gemma4_partial_rope(
            &q,
            self.head_dim,
            self.rope_partial_dims,
            self.rope_base,
            offset,
            partial_freqs,
        )
        .map_err(LoraError::Mlx)?;
        let k = apply_gemma4_partial_rope(
            &k,
            self.head_dim,
            self.rope_partial_dims,
            self.rope_base,
            offset,
            partial_freqs,
        )
        .map_err(LoraError::Mlx)?;
        Ok((q, k, v))
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        mut cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let (q, k, v) = self.project_qkv(x, offset)?;
        let (k, v) = if let Some((cache_ref, layer_idx)) = cache.as_mut() {
            (*cache_ref)
                .update_and_fetch(*layer_idx, &k, &v)
                .map_err(LoraError::Mlx)?
        } else {
            (k, v)
        };
        self.attend(&q, &k, &v, mask)
    }

    pub fn forward_collect_kv(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        offset: i32,
    ) -> Result<(Array, Array, Array), LoraError> {
        let (q, k, v) = self.project_qkv(x, offset)?;
        let output = self.attend(&q, &k, &v, mask)?;
        Ok((output, k, v))
    }

    pub fn forward_with_shared_kv(
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

    pub fn num_trainable_params(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self
                .v_proj
                .as_ref()
                .map(|p| p.num_trainable_params())
                .unwrap_or(0)
            + self.o_proj.num_trainable_params()
    }
}

#[derive(Debug)]
pub struct Gemma4LoraDecoderLayer {
    pub input_layernorm: Gemma4RmsNorm,
    pub self_attn: Gemma4LoraAttention,
    pub post_attention_layernorm: Gemma4RmsNorm,
    pub pre_feedforward_layernorm: Gemma4RmsNorm,
    pub mlp: Gemma4LoraMlp,
    pub post_feedforward_layernorm: Gemma4RmsNorm,
    pub per_layer_input_block: Option<Gemma4LoraPerLayerInputBlock>,
    pub layer_scalar: Param<Array>,
    pub kv_shared_source_layer: Option<usize>,
}

impl Gemma4LoraDecoderLayer {
    pub fn from_layer(
        layer: BaseGemma4DecoderLayer,
        lora_config: &LoraConfig,
    ) -> Result<Self, LoraError> {
        Ok(Self {
            input_layernorm: layer.input_layernorm,
            self_attn: Gemma4LoraAttention::from_base(layer.self_attn, lora_config)?,
            post_attention_layernorm: layer.post_attention_layernorm,
            pre_feedforward_layernorm: layer.pre_feedforward_layernorm,
            mlp: Gemma4LoraMlp::from_base(layer.mlp, lora_config)?,
            post_feedforward_layernorm: layer.post_feedforward_layernorm,
            per_layer_input_block: layer
                .per_layer_input_block
                .map(|block| Gemma4LoraPerLayerInputBlock::from_base(block, lora_config))
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

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
        layer_input: Option<&Array>,
    ) -> Result<Array, LoraError> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        let h = self.self_attn.forward(&h, mask, cache)?;
        self.finish_forward(&residual, &h, layer_input)
    }

    pub fn forward_collect_kv(
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

    pub fn forward_with_shared_kv(
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

    pub fn num_trainable_params(&self) -> usize {
        self.self_attn.num_trainable_params()
            + self.mlp.num_trainable_params()
            + self
                .per_layer_input_block
                .as_ref()
                .map(|b| b.num_trainable_params())
                .unwrap_or(0)
    }
}

#[derive(Debug)]
pub struct Gemma4LoraModel {
    pub embed_tokens: nn::Embedding,
    pub per_layer_inputs: Option<Gemma4LoraPerLayerInputs>,
    pub layers: Vec<Gemma4LoraDecoderLayer>,
    pub norm: Gemma4RmsNorm,
    pub config: Gemma4Config,
    pub embed_scale: f32,
}

impl Gemma4LoraModel {
    pub fn from_model(model: BaseGemma4Model, lora_config: &LoraConfig) -> Result<Self, LoraError> {
        let config = model.config.clone();
        let layers = model
            .layers
            .into_iter()
            .map(|layer| Gemma4LoraDecoderLayer::from_layer(layer, lora_config))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            embed_tokens: model.embed_tokens,
            per_layer_inputs: model
                .per_layer_inputs
                .map(|inputs| Gemma4LoraPerLayerInputs::from_base(inputs, lora_config))
                .transpose()?,
            layers,
            norm: model.norm,
            config,
            embed_scale: model.embed_scale,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let mut h = self.embed_tokens.forward(input_ids);
        h = h.multiply(&Array::from_f32(self.embed_scale));
        let per_layer_inputs = if let Some(inputs) = self.per_layer_inputs.as_mut() {
            Some(inputs.compute(input_ids, &h)?)
        } else {
            None
        };
        let mut cache = cache;
        let mut local_shared_kv = if cache.is_none() && self.config.num_kv_shared_layers() > 0 {
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
                let rope_offset = cache.as_ref().map(|c| c.rope_offset()).unwrap_or(0);
                if let Some(cache_ref) = cache.as_ref() {
                    let (source_keys, source_values) = cache_ref.get(shared_source).ok_or_else(|| {
                        LoraError::Mlx(Exception::custom(format!(
                            "Gemma 4 shared-KV layer {i} missing source layer {shared_source} cache"
                        )))
                    })?;
                    h = layer.forward_with_shared_kv(
                        &h,
                        mask,
                        &source_keys,
                        &source_values,
                        rope_offset,
                        layer_input_ref,
                    )?;
                } else {
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
                        rope_offset,
                        layer_input_ref,
                    )?;
                }
            } else if let Some(ref mut shared_kv) = local_shared_kv {
                let (next_h, keys, values) =
                    layer.forward_collect_kv(&h, mask, 0, layer_input_ref)?;
                shared_kv[i] = Some((keys, values));
                h = next_h;
            } else {
                let c = cache.as_deref_mut().map(|c| (c, i));
                h = layer.forward(&h, mask, c, layer_input_ref)?;
            }
        }
        Ok(self.norm.forward(&h))
    }

    pub fn num_trainable_params(&self) -> usize {
        self.layers
            .iter()
            .map(Gemma4LoraDecoderLayer::num_trainable_params)
            .sum::<usize>()
            + self
                .per_layer_inputs
                .as_ref()
                .map(|inputs| inputs.num_trainable_params())
                .unwrap_or(0)
    }
}

#[derive(Debug)]
pub struct Gemma4LoraForCausalLM {
    pub model: Gemma4LoraModel,
    lora_config: LoraConfig,
}

impl Gemma4LoraForCausalLM {
    pub fn new(config: Gemma4Config, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let base = BaseGemma4ForCausalLM::new(config).map_err(LoraError::Mlx)?;
        Self::from_base(base, lora_config)
    }

    pub fn from_base(
        base: BaseGemma4ForCausalLM,
        lora_config: LoraConfig,
    ) -> Result<Self, LoraError> {
        Ok(Self {
            model: Gemma4LoraModel::from_model(base.model, &lora_config)?,
            lora_config,
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
        self.forward_with_cache(input_ids, mask, None)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let hidden = self.model.forward(input_ids, mask, cache)?;
        let logits = self.model.embed_tokens.as_linear(&hidden);
        Ok(self.logit_softcap(&logits))
    }

    pub fn forward_hidden_states(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
        self.model.forward(input_ids, mask, None)
    }

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
        _noise_alpha: f32,
    ) -> Result<Array, LoraError> {
        self.forward(input_ids, mask)
    }

    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        KVCache::new(KVCacheConfig::new(
            self.model.config.num_hidden_layers as usize,
            max_seq_len,
            self.model.config.num_key_value_heads as usize,
            self.model.config.head_dim as usize,
        ))
    }

    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        if let Some(ref mut inputs) = self.model.per_layer_inputs {
            inputs.model_projection.merge()?;
        }
        for layer in &mut self.model.layers {
            layer.self_attn.q_proj.merge()?;
            layer.self_attn.k_proj.merge()?;
            if let Some(ref mut v_proj) = layer.self_attn.v_proj {
                v_proj.merge()?;
            }
            layer.self_attn.o_proj.merge()?;
            layer.mlp.gate_proj.merge()?;
            layer.mlp.up_proj.merge()?;
            layer.mlp.down_proj.merge()?;
            if let Some(ref mut block) = layer.per_layer_input_block {
                block.gate_proj.merge()?;
                block.projection.merge()?;
            }
        }
        Ok(())
    }

    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        Err(LoraError::InvalidState(
            "unmerge_lora is not supported: reload base model weights to undo a merge".to_string(),
        ))
    }

    pub fn load_base_weights_from_dir(
        &mut self,
        model_dir: impl AsRef<Path>,
    ) -> Result<(), LoraError> {
        let model_dir = model_dir.as_ref();
        let mut base =
            BaseGemma4ForCausalLM::new(self.model.config.clone()).map_err(LoraError::Mlx)?;
        let weights = pmetal_models::loader::load_weights(model_dir).map_err(|e| {
            LoraError::InvalidState(format!("failed to load Gemma4 weights: {e:?}"))
        })?;
        load_gemma4_weights(&mut base, &weights).map_err(LoraError::Mlx)?;
        *self = Self::from_base(base, self.lora_config.clone())?;
        Ok(())
    }

    pub fn eval_all(&mut self) -> Result<(), LoraError> {
        self.model.embed_tokens.weight.value.eval();
        self.model.norm.weight.value.eval();
        if let Some(ref mut inputs) = self.model.per_layer_inputs {
            inputs.embed_tokens.weight.value.eval();
            inputs.model_projection.weight.eval();
            inputs.model_projection.lora_a.eval();
            inputs.model_projection.lora_b.eval();
            inputs.projection_norm.weight.value.eval();
        }
        for layer in &mut self.model.layers {
            layer.input_layernorm.weight.value.eval();
            layer.post_attention_layernorm.weight.value.eval();
            layer.pre_feedforward_layernorm.weight.value.eval();
            layer.post_feedforward_layernorm.weight.value.eval();
            layer.layer_scalar.value.eval();

            layer.self_attn.q_proj.weight.eval();
            layer.self_attn.q_proj.lora_a.eval();
            layer.self_attn.q_proj.lora_b.eval();
            layer.self_attn.k_proj.weight.eval();
            layer.self_attn.k_proj.lora_a.eval();
            layer.self_attn.k_proj.lora_b.eval();
            if let Some(ref mut v_proj) = layer.self_attn.v_proj {
                v_proj.weight.eval();
                v_proj.lora_a.eval();
                v_proj.lora_b.eval();
            }
            layer.self_attn.o_proj.weight.eval();
            layer.self_attn.o_proj.lora_a.eval();
            layer.self_attn.o_proj.lora_b.eval();
            layer.self_attn.q_norm.weight.value.eval();
            layer.self_attn.k_norm.weight.value.eval();

            layer.mlp.gate_proj.weight.eval();
            layer.mlp.gate_proj.lora_a.eval();
            layer.mlp.gate_proj.lora_b.eval();
            layer.mlp.up_proj.weight.eval();
            layer.mlp.up_proj.lora_a.eval();
            layer.mlp.up_proj.lora_b.eval();
            layer.mlp.down_proj.weight.eval();
            layer.mlp.down_proj.lora_a.eval();
            layer.mlp.down_proj.lora_b.eval();

            if let Some(ref mut block) = layer.per_layer_input_block {
                block.gate_proj.weight.eval();
                block.gate_proj.lora_a.eval();
                block.gate_proj.lora_b.eval();
                block.projection.weight.eval();
                block.projection.lora_a.eval();
                block.projection.lora_b.eval();
                block.post_norm.weight.value.eval();
            }
        }
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
                let block_prefix = format!("layers.{i}.per_layer_input_block");
                params.insert(
                    Rc::from(format!("{block_prefix}.gate_proj.lora_a")),
                    block.gate_proj.lora_a.clone(),
                );
                params.insert(
                    Rc::from(format!("{block_prefix}.gate_proj.lora_b")),
                    block.gate_proj.lora_b.clone(),
                );
                params.insert(
                    Rc::from(format!("{block_prefix}.projection.lora_a")),
                    block.projection.lora_a.clone(),
                );
                params.insert(
                    Rc::from(format!("{block_prefix}.projection.lora_b")),
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
                let block_prefix = format!("layers.{i}.per_layer_input_block");
                set_param!(
                    block.gate_proj.lora_a,
                    format!("{block_prefix}.gate_proj.lora_a")
                );
                set_param!(
                    block.gate_proj.lora_b,
                    format!("{block_prefix}.gate_proj.lora_b")
                );
                set_param!(
                    block.projection.lora_a,
                    format!("{block_prefix}.projection.lora_a")
                );
                set_param!(
                    block.projection.lora_b,
                    format!("{block_prefix}.projection.lora_b")
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

    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    pub fn config(&self) -> &Gemma4Config {
        &self.model.config
    }

    pub fn lora_config(&self) -> &LoraConfig {
        &self.lora_config
    }

    pub fn get_lm_head_weight(&self) -> Option<Array> {
        Some(self.model.embed_tokens.weight.value.clone())
    }
}

impl ModuleParameters for Gemma4LoraForCausalLM {
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

impl crate::TrainableModel for Gemma4LoraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        Gemma4LoraForCausalLM::forward(self, input_ids, mask)
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        Gemma4LoraForCausalLM::forward_with_cache(self, input_ids, mask, cache)
    }

    fn create_cache(&self, max_seq_len: usize) -> Option<KVCache> {
        Some(Gemma4LoraForCausalLM::create_cache(self, max_seq_len))
    }

    fn supports_kv_cache(&self) -> bool {
        true
    }

    fn num_trainable_params(&self) -> usize {
        Gemma4LoraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        Gemma4LoraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        Gemma4LoraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        Gemma4LoraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        Gemma4LoraForCausalLM::load_lora_weights(self, path)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        false
    }

    fn forward_noised(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        noise_alpha: f32,
    ) -> Result<Array, LoraError> {
        Gemma4LoraForCausalLM::forward_noised(self, input_ids, mask, noise_alpha)
    }

    fn forward_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Option<Result<Array, LoraError>> {
        Some(Gemma4LoraForCausalLM::forward_hidden_states(
            self, input_ids, mask,
        ))
    }

    fn forward_hidden_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Option<Result<Array, LoraError>> {
        Some(Gemma4LoraForCausalLM::forward_hidden_states_with_positions(
            self,
            input_ids,
            mask,
            position_ids,
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
    fn gemma4_lora_tracks_per_layer_input_projection() {
        let model = Gemma4LoraForCausalLM::new(
            tiny_config(),
            LoraConfig {
                r: 4,
                alpha: 8.0,
                ..Default::default()
            },
        )
        .unwrap();

        assert!(
            model
                .lora_parameters()
                .contains_key(&Rc::from("model.per_layer_inputs.model_projection.lora_a"))
        );
    }
}
