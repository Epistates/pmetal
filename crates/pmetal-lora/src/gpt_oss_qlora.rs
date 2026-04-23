//! QLoRA-enabled GPT-OSS wrapper.
//!
//! Like the standard GPT-OSS LoRA path, this adapts attention projections only
//! and keeps the MoE experts frozen.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, nn,
};
use pmetal_core::LoraConfig;
use pmetal_mlx::kernels::{AttentionMaskType, FusedAttentionConfig, fused_sdpa, rope::apply_rope};
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};
use pmetal_models::architectures::gpt_oss::{GptOssConfig, GptOssMoE};

use crate::{LoraError, QLoraConfig, QLoraLinear, gpt_oss_lora::GptOssLoraForCausalLM};

#[derive(Debug)]
pub struct GptOssQloraAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_theta: f32,
    pub sliding_window: i32,
    pub attention_type: pmetal_models::architectures::gpt_oss::AttentionType,
    pub q_proj: QLoraLinear,
    pub k_proj: QLoraLinear,
    pub v_proj: QLoraLinear,
    pub o_proj: QLoraLinear,
}

impl GptOssQloraAttention {
    fn from_lora(
        attn: pmetal_models::architectures::gpt_oss::GptOssLoraAttention,
        qlora_config: &QLoraConfig,
    ) -> Result<Self, LoraError> {
        fn quantize(
            layer: pmetal_models::architectures::gpt_oss::LoraLinear,
            qcfg: &QLoraConfig,
        ) -> Result<QLoraLinear, LoraError> {
            let mut q = QLoraLinear::from_weight(&layer.weight, layer.bias.as_ref(), qcfg)?;
            q.lora_a = layer.lora_a;
            q.lora_b = layer.lora_b;
            Ok(q)
        }

        Ok(Self {
            n_heads: attn.n_heads,
            n_kv_heads: attn.n_kv_heads,
            head_dim: attn.head_dim,
            scale: attn.scale,
            rope_theta: attn.rope_theta,
            sliding_window: attn.sliding_window,
            attention_type: attn.attention_type,
            q_proj: quantize(attn.q_proj, qlora_config)?,
            k_proj: quantize(attn.k_proj, qlora_config)?,
            v_proj: quantize(attn.v_proj, qlora_config)?,
            o_proj: quantize(attn.o_proj, qlora_config)?,
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

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim]);

        let q = q.transpose_axes(&[0, 2, 1, 3]);
        let k = k.transpose_axes(&[0, 2, 1, 3]);
        let v = v.transpose_axes(&[0, 2, 1, 3]);

        let offset = cache.as_ref().map(|(c, _)| c.rope_offset()).unwrap_or(0);
        let q = apply_rope(&q, self.head_dim, false, self.rope_theta, 1.0, offset)
            .map_err(LoraError::Mlx)?;
        let k = apply_rope(&k, self.head_dim, false, self.rope_theta, 1.0, offset)
            .map_err(LoraError::Mlx)?;

        let mask_type = match self.attention_type {
            pmetal_models::architectures::gpt_oss::AttentionType::SlidingAttention => {
                AttentionMaskType::SlidingWindow(self.sliding_window)
            }
            pmetal_models::architectures::gpt_oss::AttentionType::FullAttention => {
                if mask.is_some() {
                    AttentionMaskType::None
                } else {
                    AttentionMaskType::Causal
                }
            }
        };
        let attn_config = FusedAttentionConfig::new(self.n_heads, self.n_kv_heads, self.head_dim)
            .with_scale(self.scale)
            .with_mask_type(mask_type);

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
pub struct GptOssQloraDecoderLayer {
    pub self_attn: GptOssQloraAttention,
    pub mlp: GptOssMoE,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
}

impl GptOssQloraDecoderLayer {
    fn from_lora(
        layer: pmetal_models::architectures::gpt_oss::GptOssLoraDecoderLayer,
        qlora_config: &QLoraConfig,
    ) -> Result<Self, LoraError> {
        Ok(Self {
            self_attn: GptOssQloraAttention::from_lora(layer.self_attn, qlora_config)?,
            mlp: layer.mlp,
            input_layernorm: layer.input_layernorm,
            post_attention_layernorm: layer.post_attention_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&mut KVCache, usize)>,
    ) -> Result<Array, LoraError> {
        let residual = x;
        let hidden = self.input_layernorm.forward(x);
        let hidden = self.self_attn.forward(&hidden, mask, cache)?;
        let hidden = residual.add(&hidden);

        let residual = &hidden;
        let hidden = self.post_attention_layernorm.forward(&hidden);
        let hidden = self.mlp.forward(&hidden).map_err(LoraError::Mlx)?;
        Ok(residual.add(&hidden))
    }

    fn num_trainable_params(&self) -> usize {
        self.self_attn.num_trainable_params()
    }

    fn memory_usage(&self) -> (usize, usize, usize) {
        self.self_attn.memory_usage()
    }
}

#[derive(Debug)]
pub struct GptOssQloraModel {
    pub config: GptOssConfig,
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<GptOssQloraDecoderLayer>,
    pub norm: nn::RmsNorm,
}

impl GptOssQloraModel {
    fn from_lora(
        model: pmetal_models::architectures::gpt_oss::GptOssLoraModel,
        qlora_config: &QLoraConfig,
    ) -> Result<Self, LoraError> {
        Ok(Self {
            config: model.config().clone(),
            embed_tokens: model.embed_tokens,
            layers: model
                .layers
                .into_iter()
                .map(|layer| GptOssQloraDecoderLayer::from_lora(layer, qlora_config))
                .collect::<Result<Vec<_>, _>>()?,
            norm: model.norm,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        let mut hidden = self.embed_tokens.forward(input_ids);
        match cache {
            Some(cache_ref) => {
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    hidden = layer.forward(&hidden, mask, Some((cache_ref, layer_idx)))?;
                }
            }
            None => {
                for layer in &mut self.layers {
                    hidden = layer.forward(&hidden, mask, None)?;
                }
            }
        }
        Ok(self.norm.forward(&hidden))
    }

    fn num_trainable_params(&self) -> usize {
        self.layers
            .iter()
            .map(GptOssQloraDecoderLayer::num_trainable_params)
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
pub struct GptOssQloraForCausalLM {
    pub model: GptOssQloraModel,
    pub lm_head: nn::Linear,
    pub qlora_config: QLoraConfig,
}

impl GptOssQloraForCausalLM {
    pub fn new(config: GptOssConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        Self::with_qlora_config(config, QLoraConfig::from_lora(lora_config))
    }

    pub fn with_qlora_config(
        config: GptOssConfig,
        qlora_config: QLoraConfig,
    ) -> Result<Self, LoraError> {
        let lora = GptOssLoraForCausalLM::new(config, qlora_config.lora.clone())?;
        Self::from_lora(lora, qlora_config)
    }

    fn from_lora(
        lora: GptOssLoraForCausalLM,
        qlora_config: QLoraConfig,
    ) -> Result<Self, LoraError> {
        Ok(Self {
            model: GptOssQloraModel::from_lora(lora.inner.model, &qlora_config)?,
            lm_head: lora.inner.lm_head,
            qlora_config,
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
        Module::forward(&mut self.lm_head, &hidden).map_err(LoraError::Mlx)
    }

    pub fn forward_hidden_states(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
        self.model.forward(input_ids, mask, None)
    }

    pub fn get_lm_head_weight(&self) -> Option<Array> {
        Some(self.lm_head.weight.value.clone())
    }

    pub fn num_trainable_params(&self) -> usize {
        self.model.num_trainable_params()
    }

    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        let config = &self.model.config;
        KVCache::new(KVCacheConfig::new(
            config.num_hidden_layers as usize,
            max_seq_len,
            config.num_key_value_heads as usize,
            config.head_dim as usize,
        ))
    }

    pub fn load_and_quantize_from_dir(
        &mut self,
        model_dir: impl AsRef<Path>,
    ) -> Result<(), LoraError> {
        let mut lora =
            GptOssLoraForCausalLM::new(self.model.config.clone(), self.qlora_config.lora.clone())?;
        lora.load_base_weights_from_dir(model_dir)?;
        *self = Self::from_lora(lora, self.qlora_config.clone())?;
        Ok(())
    }

    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        let mut params = HashMap::new();
        for (i, layer) in self.model.layers.iter().enumerate() {
            let prefix = format!("layers.{i}.self_attn");
            for (name, proj) in [
                ("q_proj", &layer.self_attn.q_proj),
                ("k_proj", &layer.self_attn.k_proj),
                ("v_proj", &layer.self_attn.v_proj),
                ("o_proj", &layer.self_attn.o_proj),
            ] {
                params.insert(
                    Rc::from(format!("{prefix}.{name}.lora_a")),
                    proj.lora_a.clone(),
                );
                params.insert(
                    Rc::from(format!("{prefix}.{name}.lora_b")),
                    proj.lora_b.clone(),
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
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{i}.self_attn");
            set_param!(
                layer.self_attn.q_proj.lora_a,
                format!("{prefix}.q_proj.lora_a")
            );
            set_param!(
                layer.self_attn.q_proj.lora_b,
                format!("{prefix}.q_proj.lora_b")
            );
            set_param!(
                layer.self_attn.k_proj.lora_a,
                format!("{prefix}.k_proj.lora_a")
            );
            set_param!(
                layer.self_attn.k_proj.lora_b,
                format!("{prefix}.k_proj.lora_b")
            );
            set_param!(
                layer.self_attn.v_proj.lora_a,
                format!("{prefix}.v_proj.lora_a")
            );
            set_param!(
                layer.self_attn.v_proj.lora_b,
                format!("{prefix}.v_proj.lora_b")
            );
            set_param!(
                layer.self_attn.o_proj.lora_a,
                format!("{prefix}.o_proj.lora_a")
            );
            set_param!(
                layer.self_attn.o_proj.lora_b,
                format!("{prefix}.o_proj.lora_b")
            );
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
                layer.self_attn.q_proj.num_frozen_params() * 4
                    + layer.self_attn.k_proj.num_frozen_params() * 4
                    + layer.self_attn.v_proj.num_frozen_params() * 4
                    + layer.self_attn.o_proj.num_frozen_params() * 4
            })
            .sum::<usize>()
            + lora;
        (quantized + lora) as f32 / full_precision as f32
    }
}

impl ModuleParameters for GptOssQloraForCausalLM {
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

impl crate::TrainableModel for GptOssQloraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        GptOssQloraForCausalLM::forward(self, input_ids, mask)
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        GptOssQloraForCausalLM::forward_with_cache(self, input_ids, mask, cache)
    }

    fn num_trainable_params(&self) -> usize {
        GptOssQloraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        GptOssQloraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        GptOssQloraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        GptOssQloraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        GptOssQloraForCausalLM::load_lora_weights(self, path)
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

    fn tiny_config() -> GptOssConfig {
        GptOssConfig {
            hidden_size: 32,
            intermediate_size: 48,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            head_dim: 8,
            max_position_embeddings: 256,
            initial_context_length: 32,
            num_local_experts: 4,
            experts_per_token: 2,
            num_experts_per_tok: Some(2),
            vocab_size: 128,
            ..Default::default()
        }
    }

    #[test]
    fn gpt_oss_qlora_builds() {
        let model =
            GptOssQloraForCausalLM::with_qlora_config(tiny_config(), QLoraConfig::default())
                .unwrap();
        assert!(model.num_trainable_params() > 0);
    }
}
