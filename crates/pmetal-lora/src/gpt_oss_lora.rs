//! LoRA-enabled GPT-OSS model wrapper.
//!
//! GPT-OSS already has an internal attention-only LoRA implementation in
//! `pmetal-models`. This file bridges that implementation into the training
//! surface exposed by `pmetal-lora`: adapter save/load, `ModuleParameters`,
//! `TrainableModel`, and loading from a pretrained model directory.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use pmetal_bridge::compat::{
    Array, Exception, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue,
};
use pmetal_core::LoraConfig;
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};
use pmetal_models::architectures::gpt_oss::{
    GptOssConfig, GptOssForCausalLM as BaseGptOssForCausalLM,
    GptOssLoraForCausalLM as InnerGptOssLoraForCausalLM,
};
use pmetal_models::loader::load_generic_weights;

use crate::LoraError;

/// GPT-OSS causal LM with LoRA adapters.
#[derive(Debug)]
pub struct GptOssLoraForCausalLM {
    pub(crate) inner: InnerGptOssLoraForCausalLM,
    lora_config: LoraConfig,
}

impl GptOssLoraForCausalLM {
    /// Create a new randomly initialised GPT-OSS LoRA model.
    pub fn new(config: GptOssConfig, lora_config: LoraConfig) -> Result<Self, LoraError> {
        let base = BaseGptOssForCausalLM::new(config).map_err(LoraError::Mlx)?;
        Self::from_base(base, lora_config)
    }

    /// Convert a loaded base GPT-OSS model into its LoRA training wrapper.
    pub fn from_base(
        base: BaseGptOssForCausalLM,
        lora_config: LoraConfig,
    ) -> Result<Self, LoraError> {
        let inner = base.into_lora(&lora_config).map_err(LoraError::Mlx)?;
        Ok(Self { inner, lora_config })
    }

    /// Forward pass producing logits.
    pub fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        self.inner
            .forward(input_ids, mask, None)
            .map_err(LoraError::Mlx)
    }

    /// Forward pass with KV cache for autoregressive decoding.
    pub fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        self.inner
            .forward(input_ids, mask, cache)
            .map_err(LoraError::Mlx)
    }

    /// Forward pass returning hidden states before the LM head.
    pub fn forward_hidden_states(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, LoraError> {
        self.inner
            .model
            .forward(input_ids, mask, None)
            .map_err(LoraError::Mlx)
    }

    /// GPT-OSS does not use explicit packed-sequence position ids in the LoRA path.
    pub fn forward_hidden_states_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        _position_ids: &Array,
    ) -> Result<Array, LoraError> {
        self.forward_hidden_states(input_ids, mask)
    }

    /// NEFTune fallback.
    pub fn forward_noised(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        _noise_alpha: f32,
    ) -> Result<Array, LoraError> {
        self.forward(input_ids, mask)
    }

    /// Create a KV cache matching the base GPT-OSS attention layout.
    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        let config = self.config();
        KVCache::new(KVCacheConfig::new(
            config.num_hidden_layers as usize,
            max_seq_len,
            config.num_key_value_heads as usize,
            config.head_dim as usize,
        ))
    }

    /// Merge LoRA weights into the base attention projections.
    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        self.inner.merge().map_err(LoraError::Mlx)
    }

    /// Unmerge is not supported.
    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        Err(LoraError::InvalidState(
            "unmerge_lora is not supported: reload base model weights to undo a merge".to_string(),
        ))
    }

    /// Load pretrained base weights from a model directory.
    pub fn load_base_weights_from_dir(
        &mut self,
        model_dir: impl AsRef<Path>,
    ) -> Result<(), LoraError> {
        let mut base = BaseGptOssForCausalLM::new(self.config().clone()).map_err(LoraError::Mlx)?;
        load_generic_weights(&mut base, model_dir).map_err(|e| {
            LoraError::InvalidState(format!("failed to load GPT-OSS base weights: {e:?}"))
        })?;
        base.init_stacked_moe().map_err(LoraError::Mlx)?;
        *self = Self::from_base(base, self.lora_config.clone())?;
        Ok(())
    }

    /// Force evaluation of all adapter parameters.
    pub fn eval_all(&mut self) -> Result<(), LoraError> {
        for layer in &mut self.inner.model.layers {
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
            layer.input_layernorm.weight.value.eval();
            layer.post_attention_layernorm.weight.value.eval();
            layer.mlp.init_stacked_moe().map_err(LoraError::Mlx)?;
        }
        self.inner.model.embed_tokens.weight.value.eval();
        self.inner.model.norm.weight.value.eval();
        self.inner.lm_head.weight.value.eval();
        Ok(())
    }

    /// Flat LoRA adapter map.
    pub fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        let mut params = HashMap::new();
        for (i, layer) in self.inner.model.layers.iter().enumerate() {
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

    /// Restore LoRA adapter tensors from a flat map.
    pub fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        macro_rules! set_param {
            ($dst:expr, $key:expr) => {
                if let Some(value) = params.get(&Rc::from($key) as &Rc<str>) {
                    $dst = value.clone();
                }
            };
        }

        for (i, layer) in self.inner.model.layers.iter_mut().enumerate() {
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

    /// Save LoRA weights to a safetensors file.
    pub fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        crate::save_safetensors_map(path, &self.lora_parameters())
    }

    /// Load LoRA weights from a safetensors file or directory.
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

    /// Number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.inner.num_trainable_params()
    }

    /// Base GPT-OSS config.
    pub fn config(&self) -> &GptOssConfig {
        self.inner.model.config()
    }

    /// LoRA config.
    pub fn lora_config(&self) -> &LoraConfig {
        &self.lora_config
    }

    /// LM head weight for Cut Cross-Entropy.
    pub fn get_lm_head_weight(&self) -> Option<Array> {
        Some(self.inner.lm_head.weight.value.clone())
    }
}

impl ModuleParameters for GptOssLoraForCausalLM {
    fn num_parameters(&self) -> usize {
        self.num_trainable_params()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut params = ModuleParamRef::new();
        for (i, layer) in self.inner.model.layers.iter().enumerate() {
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
        for (i, layer) in self.inner.model.layers.iter_mut().enumerate() {
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

impl crate::TrainableModel for GptOssLoraForCausalLM {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        GptOssLoraForCausalLM::forward(self, input_ids, mask)
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        GptOssLoraForCausalLM::forward_with_cache(self, input_ids, mask, cache)
    }

    fn create_cache(&self, max_seq_len: usize) -> Option<KVCache> {
        Some(GptOssLoraForCausalLM::create_cache(self, max_seq_len))
    }

    fn supports_kv_cache(&self) -> bool {
        true
    }

    fn num_trainable_params(&self) -> usize {
        GptOssLoraForCausalLM::num_trainable_params(self)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        GptOssLoraForCausalLM::lora_parameters(self)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        GptOssLoraForCausalLM::set_lora_parameters(self, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        GptOssLoraForCausalLM::save_lora_weights(self, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<Path>) -> Result<(), LoraError> {
        GptOssLoraForCausalLM::load_lora_weights(self, path)
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
        GptOssLoraForCausalLM::forward_noised(self, input_ids, mask, noise_alpha)
    }

    fn forward_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Option<Result<Array, LoraError>> {
        Some(GptOssLoraForCausalLM::forward_hidden_states(
            self, input_ids, mask,
        ))
    }

    fn forward_hidden_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Option<Result<Array, LoraError>> {
        Some(GptOssLoraForCausalLM::forward_hidden_states_with_positions(
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
    fn gpt_oss_lora_constructs_and_counts_params() {
        let model = GptOssLoraForCausalLM::new(
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
                .any(|k| k.contains("layers.0.self_attn.q_proj.lora_a"))
        );
    }

    #[test]
    fn gpt_oss_eval_all_materializes_stacked_experts() {
        let mut model = GptOssLoraForCausalLM::new(
            tiny_config(),
            LoraConfig {
                r: 4,
                alpha: 8.0,
                ..Default::default()
            },
        )
        .unwrap();

        model.eval_all().unwrap();
        assert!(model.inner.model.layers[0].mlp.has_stacked_moe());
    }
}
