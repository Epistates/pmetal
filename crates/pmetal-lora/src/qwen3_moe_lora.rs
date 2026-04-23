//! LoRA-enabled Qwen3-MoE architecture.
//!
//! Adapter placement:
//! - Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
//! - Dense FFN layers: `gate_proj`, `up_proj`, `down_proj`
//! - Sparse MoE layers: router `gate` only
//!
//! Routed expert weights remain frozen. This keeps adapter size bounded while
//! still letting training adjust expert selection on sparse layers.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, Param,
    nn, ops,
};
use pmetal_core::LoraConfig;
use pmetal_mlx::kernels::{
    AttentionMaskType, FusedAttentionConfig, fused_sdpa,
    rope::{RopeScaling, apply_rope},
};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};
use pmetal_models::architectures::qwen3_moe::{
    Qwen3MoE as BaseQwen3MoE, Qwen3MoEAttention as BaseQwen3MoEAttention,
    Qwen3MoEBlock as BaseQwen3MoEBlock, Qwen3MoEConfig,
    Qwen3MoEDecoderLayer as BaseQwen3MoEDecoderLayer, Qwen3MoEDenseMLP as BaseQwen3MoEDenseMLP,
    Qwen3MoEFeedForward as BaseQwen3MoEFeedForward, Qwen3MoEModel as BaseQwen3MoEModel,
};
use pmetal_models::loader::load_generic_weights;

use crate::{LoraError, LoraLinear};

#[derive(Debug)]
pub struct Qwen3MoELoraAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_scale: f32,
    pub effective_base: f32,
    pub q_proj: LoraLinear,
    pub k_proj: LoraLinear,
    pub v_proj: LoraLinear,
    pub o_proj: LoraLinear,
    pub q_norm: nn::RmsNorm,
    pub k_norm: nn::RmsNorm,
}

impl Qwen3MoELoraAttention {
    pub fn from_attention(
        config: &Qwen3MoEConfig,
        attn: BaseQwen3MoEAttention,
        lora_config: &LoraConfig,
    ) -> Result<Self, LoraError> {
        let rope_scaling = config
            .rope_scaling
            .as_ref()
            .map(RopeScaling::from_config_map)
            .unwrap_or(RopeScaling::None);
        let head_dim = config.head_dim;
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;

        Ok(Self {
            n_heads: config.num_attention_heads,
            n_kv_heads: config.num_kv_heads(),
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope_scale: rope_scaling.scale(),
            effective_base: rope_scaling.effective_base(config.rope_theta, head_dim),
            q_proj: LoraLinear::from_linear(
                &attn.q_proj,
                crate::effective_rank(lora_config, "q_proj") as i32,
                alpha,
                use_rslora,
            )?,
            k_proj: LoraLinear::from_linear(
                &attn.k_proj,
                crate::effective_rank(lora_config, "k_proj") as i32,
                alpha,
                use_rslora,
            )?,
            v_proj: LoraLinear::from_linear(
                &attn.v_proj,
                crate::effective_rank(lora_config, "v_proj") as i32,
                alpha,
                use_rslora,
            )?,
            o_proj: LoraLinear::from_linear(
                &attn.o_proj,
                crate::effective_rank(lora_config, "o_proj") as i32,
                alpha,
                use_rslora,
            )?,
            q_norm: attn.q_norm,
            k_norm: attn.k_norm,
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

    pub fn num_trainable_params(&self) -> usize {
        self.q_proj.num_trainable_params()
            + self.k_proj.num_trainable_params()
            + self.v_proj.num_trainable_params()
            + self.o_proj.num_trainable_params()
    }
}

#[derive(Debug)]
pub struct Qwen3MoELoraDenseMLP {
    pub gate_proj: LoraLinear,
    pub up_proj: LoraLinear,
    pub down_proj: LoraLinear,
}

impl Qwen3MoELoraDenseMLP {
    pub fn from_mlp(
        mlp: BaseQwen3MoEDenseMLP,
        lora_config: &LoraConfig,
    ) -> Result<Self, LoraError> {
        let alpha = lora_config.alpha;
        let use_rslora = lora_config.use_rslora;
        Ok(Self {
            gate_proj: LoraLinear::from_linear(
                &mlp.gate_proj,
                crate::effective_rank(lora_config, "gate_proj") as i32,
                alpha,
                use_rslora,
            )?,
            up_proj: LoraLinear::from_linear(
                &mlp.up_proj,
                crate::effective_rank(lora_config, "up_proj") as i32,
                alpha,
                use_rslora,
            )?,
            down_proj: LoraLinear::from_linear(
                &mlp.down_proj,
                crate::effective_rank(lora_config, "down_proj") as i32,
                alpha,
                use_rslora,
            )?,
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
}

#[derive(Debug)]
pub struct Qwen3MoELoraMoEBlock {
    pub top_k: usize,
    pub norm_topk_prob: bool,
    pub gate: LoraLinear,
    pub experts: Vec<pmetal_mlx::moe::Expert>,
    pub stacked_gate_proj: Option<Array>,
    pub stacked_up_proj: Option<Array>,
    pub stacked_down_proj: Option<Array>,
    pub stacked_weight_signature: Option<Vec<usize>>,
}

impl Qwen3MoELoraMoEBlock {
    pub fn from_block(
        block: BaseQwen3MoEBlock,
        lora_config: &LoraConfig,
    ) -> Result<Self, LoraError> {
        Ok(Self {
            top_k: block.top_k,
            norm_topk_prob: block.norm_topk_prob,
            gate: LoraLinear::from_linear(
                &block.gate,
                crate::effective_rank(lora_config, "gate") as i32,
                lora_config.alpha,
                lora_config.use_rslora,
            )?,
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

    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
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

    pub fn num_trainable_params(&self) -> usize {
        self.gate.num_trainable_params()
    }
}

// Size delta (Dense ~2 KB vs MoE ~1.1 KB) is bounded per decoder layer and not
// on a hot dispatch path — layers are constructed once and accessed by `&mut`.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum Qwen3MoELoraFeedForward {
    Dense(Qwen3MoELoraDenseMLP),
    MoE(Qwen3MoELoraMoEBlock),
}

impl Qwen3MoELoraFeedForward {
    pub fn forward(&mut self, x: &Array) -> Result<Array, LoraError> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::MoE(moe) => moe.forward(x),
        }
    }

    pub fn num_trainable_params(&self) -> usize {
        match self {
            Self::Dense(mlp) => mlp.num_trainable_params(),
            Self::MoE(moe) => moe.num_trainable_params(),
        }
    }
}

#[derive(Debug)]
pub struct Qwen3MoELoraDecoderLayer {
    pub self_attn: Qwen3MoELoraAttention,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
    pub ffn: Qwen3MoELoraFeedForward,
}

impl Qwen3MoELoraDecoderLayer {
    pub fn from_layer(
        config: &Qwen3MoEConfig,
        layer: BaseQwen3MoEDecoderLayer,
        lora_config: &LoraConfig,
    ) -> Result<Self, LoraError> {
        Ok(Self {
            self_attn: Qwen3MoELoraAttention::from_attention(config, layer.self_attn, lora_config)?,
            input_layernorm: layer.input_layernorm,
            post_attention_layernorm: layer.post_attention_layernorm,
            ffn: match layer.ffn {
                BaseQwen3MoEFeedForward::Dense(mlp) => Qwen3MoELoraFeedForward::Dense(
                    Qwen3MoELoraDenseMLP::from_mlp(mlp, lora_config)?,
                ),
                BaseQwen3MoEFeedForward::MoE(block) => Qwen3MoELoraFeedForward::MoE(
                    Qwen3MoELoraMoEBlock::from_block(block, lora_config)?,
                ),
            },
        })
    }

    pub fn forward(
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

    pub fn num_trainable_params(&self) -> usize {
        self.self_attn.num_trainable_params() + self.ffn.num_trainable_params()
    }
}

#[derive(Debug)]
pub struct Qwen3MoELoraModel {
    pub config: Qwen3MoEConfig,
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<Qwen3MoELoraDecoderLayer>,
    pub norm: nn::RmsNorm,
}

impl Qwen3MoELoraModel {
    pub fn from_model(
        model: BaseQwen3MoEModel,
        lora_config: &LoraConfig,
    ) -> Result<Self, LoraError> {
        let config = model.config.clone();
        let layers = model
            .layers
            .into_iter()
            .map(|layer| Qwen3MoELoraDecoderLayer::from_layer(&config, layer, lora_config))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            config,
            embed_tokens: model.embed_tokens,
            layers,
            norm: model.norm,
        })
    }

    pub fn forward(
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

    pub fn num_trainable_params(&self) -> usize {
        self.layers
            .iter()
            .map(Qwen3MoELoraDecoderLayer::num_trainable_params)
            .sum()
    }
}

#[derive(Debug)]
pub struct Qwen3MoELoraForCausalLM {
    pub model: Qwen3MoELoraModel,
    pub lm_head: Option<nn::Linear>,
    lora_config: LoraConfig,
}

impl Qwen3MoELoraForCausalLM {
    pub fn new(config: Qwen3MoEConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let base = BaseQwen3MoE::new(config).map_err(LoraError::Mlx)?;
        Self::from_base(base, lora_config)
    }

    pub fn from_base(base: BaseQwen3MoE, lora_config: LoraConfig) -> Result<Self, LoraError> {
        Ok(Self {
            model: Qwen3MoELoraModel::from_model(base.model, &lora_config)?,
            lm_head: base.lm_head,
            lora_config,
        })
    }

    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        let hidden = self.model.forward(input_ids, mask, None)?;
        self.lm_head_forward(&hidden)
    }

    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let hidden = self.model.forward(input_ids, mask, cache)?;
        self.lm_head_forward(&hidden)
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

    fn lm_head_forward(&mut self, hidden: &Array) -> Result<Array, LoraError> {
        if let Some(ref mut lm_head) = self.lm_head {
            Module::forward(lm_head, hidden).map_err(LoraError::Mlx)
        } else {
            Ok(self.model.embed_tokens.as_linear(hidden))
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

    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        for layer in &mut self.model.layers {
            layer.self_attn.q_proj.merge()?;
            layer.self_attn.k_proj.merge()?;
            layer.self_attn.v_proj.merge()?;
            layer.self_attn.o_proj.merge()?;
            match &mut layer.ffn {
                Qwen3MoELoraFeedForward::Dense(mlp) => {
                    mlp.gate_proj.merge()?;
                    mlp.up_proj.merge()?;
                    mlp.down_proj.merge()?;
                }
                Qwen3MoELoraFeedForward::MoE(block) => {
                    block.gate.merge()?;
                }
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
        let mut base = BaseQwen3MoE::new(self.model.config.clone()).map_err(LoraError::Mlx)?;
        load_generic_weights(&mut base, model_dir).map_err(|e| {
            LoraError::InvalidState(format!("failed to load Qwen3-MoE base weights: {e:?}"))
        })?;
        base.init_stacked_moe().map_err(LoraError::Mlx)?;
        *self = Self::from_base(base, self.lora_config.clone())?;
        Ok(())
    }

    pub fn eval_all(&mut self) -> Result<(), LoraError> {
        self.model.embed_tokens.weight.value.eval();
        self.model.norm.weight.value.eval();
        if let Some(ref mut lm_head) = self.lm_head {
            lm_head.weight.value.eval();
        }
        for layer in &mut self.model.layers {
            layer.input_layernorm.weight.value.eval();
            layer.post_attention_layernorm.weight.value.eval();
            layer.self_attn.q_proj.weight.eval();
            layer.self_attn.k_proj.weight.eval();
            layer.self_attn.v_proj.weight.eval();
            layer.self_attn.o_proj.weight.eval();
            layer.self_attn.q_proj.lora_a.eval();
            layer.self_attn.q_proj.lora_b.eval();
            layer.self_attn.k_proj.lora_a.eval();
            layer.self_attn.k_proj.lora_b.eval();
            layer.self_attn.v_proj.lora_a.eval();
            layer.self_attn.v_proj.lora_b.eval();
            layer.self_attn.o_proj.lora_a.eval();
            layer.self_attn.o_proj.lora_b.eval();
            layer.self_attn.q_norm.weight.value.eval();
            layer.self_attn.k_norm.weight.value.eval();
            match &mut layer.ffn {
                Qwen3MoELoraFeedForward::Dense(mlp) => {
                    mlp.gate_proj.weight.eval();
                    mlp.up_proj.weight.eval();
                    mlp.down_proj.weight.eval();
                    mlp.gate_proj.lora_a.eval();
                    mlp.gate_proj.lora_b.eval();
                    mlp.up_proj.lora_a.eval();
                    mlp.up_proj.lora_b.eval();
                    mlp.down_proj.lora_a.eval();
                    mlp.down_proj.lora_b.eval();
                }
                Qwen3MoELoraFeedForward::MoE(block) => {
                    block.ensure_stacked_moe().map_err(LoraError::Mlx)?;
                    block.gate.weight.eval();
                    block.gate.lora_a.eval();
                    block.gate.lora_b.eval();
                }
            }
        }
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
                Qwen3MoELoraFeedForward::Dense(mlp) => {
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
                Qwen3MoELoraFeedForward::MoE(block) => {
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
                Qwen3MoELoraFeedForward::Dense(mlp) => {
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
                Qwen3MoELoraFeedForward::MoE(block) => {
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

    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    pub fn config(&self) -> &Qwen3MoEConfig {
        &self.model.config
    }

    pub fn lora_config(&self) -> &LoraConfig {
        &self.lora_config
    }

    pub fn get_lm_head_weight(&self) -> Option<Array> {
        if let Some(ref lm_head) = self.lm_head {
            Some(lm_head.weight.value.clone())
        } else {
            Some(self.model.embed_tokens.weight.value.clone())
        }
    }
}

impl ModuleParameters for Qwen3MoELoraForCausalLM {
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
                Qwen3MoELoraFeedForward::Dense(mlp) => {
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
                Qwen3MoELoraFeedForward::MoE(block) => {
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
                Qwen3MoELoraFeedForward::Dense(mlp) => {
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
                Qwen3MoELoraFeedForward::MoE(block) => {
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

impl crate::TrainableModel for Qwen3MoELoraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        Qwen3MoELoraForCausalLM::forward(self, input_ids, mask)
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        Qwen3MoELoraForCausalLM::forward_with_cache(self, input_ids, mask, cache)
    }

    fn create_cache(&self, max_seq_len: usize) -> Option<KVCache> {
        Some(Qwen3MoELoraForCausalLM::create_cache(self, max_seq_len))
    }

    fn supports_kv_cache(&self) -> bool {
        true
    }

    fn num_trainable_params(&self) -> usize {
        Qwen3MoELoraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        Qwen3MoELoraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        Qwen3MoELoraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        Qwen3MoELoraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        Qwen3MoELoraForCausalLM::load_lora_weights(self, path)
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
        Qwen3MoELoraForCausalLM::forward_noised(self, input_ids, mask, noise_alpha)
    }

    fn forward_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Option<Result<Array, LoraError>> {
        Some(Qwen3MoELoraForCausalLM::forward_hidden_states(
            self, input_ids, mask,
        ))
    }

    fn forward_hidden_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Option<Result<Array, LoraError>> {
        Some(
            Qwen3MoELoraForCausalLM::forward_hidden_states_with_positions(
                self,
                input_ids,
                mask,
                position_ids,
            ),
        )
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
    fn qwen3_moe_lora_builds_router_params() {
        let model = Qwen3MoELoraForCausalLM::new(
            tiny_config(),
            LoraConfig {
                r: 4,
                alpha: 8.0,
                ..Default::default()
            },
        )
        .unwrap();

        assert!(model.num_trainable_params() > 0);
        assert!(
            model
                .lora_parameters()
                .keys()
                .any(|k| k.contains("layers.0.mlp.gate.lora_a"))
        );
    }

    #[test]
    fn qwen3_moe_eval_all_materializes_stacked_experts() {
        let mut model = Qwen3MoELoraForCausalLM::new(
            tiny_config(),
            LoraConfig {
                r: 4,
                alpha: 8.0,
                ..Default::default()
            },
        )
        .unwrap();

        model.eval_all().unwrap();

        let Qwen3MoELoraFeedForward::MoE(block) = &model.model.layers[0].ffn else {
            panic!("expected MoE layer");
        };
        assert!(block.stacked_gate_proj.is_some());
        assert!(block.stacked_up_proj.is_some());
        assert!(block.stacked_down_proj.is_some());
    }
}
