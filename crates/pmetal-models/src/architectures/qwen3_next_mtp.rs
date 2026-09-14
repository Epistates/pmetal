//! Qwen3Next bundled multi-token prediction model.
//!
//! Qwen3.6-style checkpoints store a small MTP predictor under `mtp.*` in the
//! same safetensors as the target model. The predictor is not a standalone
//! draft LM: it consumes the current token embedding plus a hidden state from
//! the target/draft stream, runs a full-attention decoder layer, and projects
//! through the shared LM head.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use pmetal_bridge::compat::{
    Array, Exception, Module, ModuleParameters, ModuleParametersExt, Param, nn, ops, transforms,
};
use pmetal_bridge::impl_module_params;
use pmetal_mlx::kv_cache::{KVCache, KVCacheConfig};

use super::qwen3_next::{Qwen3NextConfig, Qwen3NextDecoderLayer, Qwen3NextRoutedExpertMode};

#[derive(Debug)]
pub struct Qwen3NextMtpModel {
    pub embed_tokens: nn::Embedding,
    pub fc: nn::Linear,
    pub layers: Vec<Qwen3NextDecoderLayer>,
    pub norm: nn::RmsNorm,
    pub pre_fc_norm_embedding: nn::RmsNorm,
    pub pre_fc_norm_hidden: nn::RmsNorm,
    pub config: Qwen3NextConfig,
}
impl_module_params!(
    Qwen3NextMtpModel;
    embed_tokens,
    fc,
    layers,
    norm,
    pre_fc_norm_embedding,
    pre_fc_norm_hidden
);

impl Qwen3NextMtpModel {
    pub fn new(
        config: Qwen3NextConfig,
        routed_expert_mode: Qwen3NextRoutedExpertMode,
    ) -> Result<Self, Exception> {
        let num_mtp_layers = config.mtp_num_hidden_layers();
        if num_mtp_layers == 0 {
            return Err(Exception::custom(
                "Qwen MTP requested but config has no mtp_num_hidden_layers/num_nextn_predict_layers",
            ));
        }

        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;
        let fc = nn::LinearBuilder::new(config.hidden_size * 2, config.hidden_size)
            .bias(false)
            .build()?;
        let layers = (0..num_mtp_layers)
            .map(|idx| Qwen3NextDecoderLayer::new_mtp(&config, idx, routed_expert_mode))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let pre_fc_norm_embedding = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let pre_fc_norm_hidden = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            embed_tokens,
            fc,
            layers,
            norm,
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            config,
        })
    }

    pub fn forward_hidden(
        &mut self,
        input_ids: &Array,
        hidden_states: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        step_idx: usize,
    ) -> Result<Array, Exception> {
        let token_embeds = self.embed_tokens.forward(input_ids);
        let token_embeds = self.pre_fc_norm_embedding.forward(&token_embeds);
        let hidden_states = self.pre_fc_norm_hidden.forward(hidden_states);
        let fused = ops::concatenate_axis(&[&token_embeds, &hidden_states], -1);
        let mut h = self.fc.forward(&fused);

        let layer_idx = step_idx % self.layers.len();
        let layer = &mut self.layers[layer_idx];
        let kv = cache.map(|cache| (cache, layer_idx));
        h = layer.forward(&h, mask, kv, None, None)?;
        Ok(self.norm.forward(&h))
    }
}

#[derive(Debug)]
pub struct Qwen3NextMtpForCausalLM {
    pub model: Qwen3NextMtpModel,
    pub lm_head: Option<nn::Linear>,
    pub config: Qwen3NextConfig,
}
impl_module_params!(Qwen3NextMtpForCausalLM; model, lm_head);

impl Qwen3NextMtpForCausalLM {
    pub fn new(config: Qwen3NextConfig) -> Result<Self, Exception> {
        let model = Qwen3NextMtpModel::new(config.clone(), Qwen3NextRoutedExpertMode::Resident)?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(
                nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                    .bias(false)
                    .build()?,
            )
        };
        Ok(Self {
            model,
            lm_head,
            config,
        })
    }

    pub fn create_cache(&self, max_seq_len: usize) -> KVCache {
        KVCache::new(KVCacheConfig::new(
            self.config.mtp_num_hidden_layers(),
            max_seq_len,
            self.config.get_num_kv_heads() as usize,
            self.config.get_head_dim() as usize,
        ))
    }

    fn lm_head_forward(&mut self, hidden: &Array) -> Result<Array, Exception> {
        if let Some(ref mut lm_head) = self.lm_head {
            lm_head.forward(hidden)
        } else {
            Ok(self.model.embed_tokens.as_linear(hidden))
        }
    }

    pub fn forward_logits(
        &mut self,
        input_ids: &Array,
        hidden_states: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        step_idx: usize,
    ) -> Result<(Array, Array), Exception> {
        let hidden = self
            .model
            .forward_hidden(input_ids, hidden_states, mask, cache, step_idx)?;
        let logits = self.lm_head_forward(&hidden)?;
        Ok((hidden, logits))
    }

    pub fn quantize_fp8_weights(&mut self) -> Result<(), Exception> {
        crate::fp8_utils::quantize_model_linears(self)
    }
}

pub fn load_qwen3_next_mtp_from_dir(
    model_dir: impl AsRef<Path>,
    target_config: &Qwen3NextConfig,
) -> Result<Qwen3NextMtpForCausalLM, Exception> {
    let model_dir = model_dir.as_ref();
    let mut config = target_config.clone();
    if config.mtp_num_hidden_layers() == 0 {
        config.mtp_num_hidden_layers = Some(1);
    }

    let mut model = Qwen3NextMtpForCausalLM::new(config.clone())?;
    let mut weights = crate::loader::load_weights_filtered(model_dir, keep_qwen3_next_mtp_weight)
        .map_err(|e| Exception::custom(format!("{e:?}")))?;
    sanitize_qwen3_next_mtp_weights(&mut weights, &config)?;
    load_qwen3_next_mtp_weights(&mut model, &weights)?;
    eval_mtp_parameters(&model)?;
    Ok(model)
}

fn keep_qwen3_next_mtp_weight(key: &str) -> bool {
    let stripped = key.strip_prefix("model.language_model.").unwrap_or(key);
    stripped.starts_with("mtp.")
        || stripped.starts_with("model.mtp.")
        || stripped == "model.embed_tokens.weight"
        || stripped == "lm_head.weight"
}

fn sanitize_qwen3_next_mtp_weights(
    weights: &mut HashMap<String, Array>,
    config: &Qwen3NextConfig,
) -> Result<(), Exception> {
    let should_shift_norms = weights.keys().any(|key| {
        key.contains("mtp.") || key.contains("conv1d.weight") && weights[key].ndim() == 3
    });

    let original_keys: Vec<String> = weights.keys().cloned().collect();
    for old_key in original_keys {
        let mut new_key = old_key.clone();
        if new_key.starts_with("model.language_model.") {
            new_key = new_key.replacen("model.language_model.", "", 1);
        }
        if let Some(rest) = new_key.strip_prefix("model.mtp.") {
            new_key = format!("model.{rest}");
        } else if let Some(rest) = new_key.strip_prefix("mtp.") {
            new_key = format!("model.{rest}");
        }
        if new_key.contains(".A_log") {
            new_key = new_key.replace(".A_log", ".a_log");
        }
        if new_key != old_key
            && let Some(value) = weights.remove(&old_key)
        {
            weights.insert(new_key, value);
        }
    }

    stack_mtp_expert_weights(weights, config)?;

    if should_shift_norms {
        let norm_suffixes = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            "model.pre_fc_norm_embedding.weight",
            "model.pre_fc_norm_hidden.weight",
        ];
        let keys: Vec<String> = weights.keys().cloned().collect();
        for key in keys {
            if norm_suffixes.iter().any(|suffix| key.ends_with(suffix))
                && let Some(value) = weights.get(&key)
                && value.ndim() == 1
            {
                weights.insert(key, value.add(&Array::from_f32(1.0)));
            }
        }
    }

    if config.tie_word_embeddings {
        weights.remove("lm_head.weight");
    }
    Ok(())
}

fn stack_mtp_expert_weights(
    weights: &mut HashMap<String, Array>,
    config: &Qwen3NextConfig,
) -> Result<(), Exception> {
    let num_layers = config.mtp_num_hidden_layers();
    for layer_idx in 0..num_layers {
        let prefix = format!("model.layers.{layer_idx}.mlp");
        let fused_gate_up = format!("{prefix}.experts.gate_up_proj");
        let fused_down = format!("{prefix}.experts.down_proj");
        if let Some(gate_up) = weights.remove(&fused_gate_up) {
            let inter = config.moe_intermediate_size;
            let gate = ops::slice_axis(&gate_up, 1, 0, inter);
            let up = ops::slice_axis(&gate_up, 1, inter, inter * 2);
            weights.insert(format!("{prefix}.switch_mlp_gate_proj"), gate);
            weights.insert(format!("{prefix}.switch_mlp_up_proj"), up);
        }
        if let Some(down) = weights.remove(&fused_down) {
            weights.insert(format!("{prefix}.switch_mlp_down_proj"), down);
        }

        for name in ["gate_proj", "up_proj", "down_proj"] {
            let mut expert_weights = Vec::new();
            for expert_idx in 0..config.num_experts {
                let key = format!("{prefix}.experts.{expert_idx}.{name}.weight");
                if let Some(weight) = weights.remove(&key) {
                    expert_weights.push(weight);
                }
            }
            if !expert_weights.is_empty() {
                let stacked = ops::stack_axis(&expert_weights, 0);
                let dest = match name {
                    "gate_proj" => format!("{prefix}.switch_mlp_gate_proj"),
                    "up_proj" => format!("{prefix}.switch_mlp_up_proj"),
                    "down_proj" => format!("{prefix}.switch_mlp_down_proj"),
                    _ => unreachable!(),
                };
                weights.insert(dest, stacked);
            }
        }
    }
    Ok(())
}

fn load_qwen3_next_mtp_weights(
    model: &mut Qwen3NextMtpForCausalLM,
    weights: &HashMap<String, Array>,
) -> Result<(), Exception> {
    let mut params = model.flatten_params_mut();
    let expected_keys: HashSet<String> = params.keys().map(|key| key.to_string()).collect();
    let mut loaded = HashSet::new();

    for (key, value) in weights {
        if let Some(param) = params.get_mut(&**key) {
            **param = value.clone();
            loaded.insert(key.clone());
        }
    }

    let missing: Vec<String> = expected_keys
        .difference(&loaded)
        .take(20)
        .cloned()
        .collect();
    if !missing.is_empty() {
        return Err(Exception::custom(format!(
            "Qwen MTP weight loading is missing model parameters: {:?}",
            missing
        )));
    }
    Ok(())
}

fn eval_mtp_parameters(model: &Qwen3NextMtpForCausalLM) -> Result<(), Exception> {
    const PARAM_EVAL_BATCH_SIZE: usize = 128;
    let params = model.flatten_params();
    let arrays: Vec<Array> = params.into_values().collect();
    for chunk in arrays.chunks(PARAM_EVAL_BATCH_SIZE) {
        transforms::eval(chunk.iter())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Qwen3NextConfig {
        Qwen3NextConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 4,
            num_attention_heads: 1,
            num_key_value_heads: Some(1),
            head_dim: Some(4),
            vocab_size: 16,
            num_experts: 0,
            mtp_num_hidden_layers: Some(1),
            ..Default::default()
        }
    }

    #[test]
    fn qwen_mtp_sanitize_remaps_and_shifts_norms() {
        let config = tiny_config();
        let mut weights = HashMap::from([
            (
                "mtp.pre_fc_norm_hidden.weight".to_string(),
                Array::from_f32_slice(&[0.0, 0.5, 1.0, 1.5], &[4]),
            ),
            (
                "model.language_model.mtp.layers.0.self_attn.q_norm.weight".to_string(),
                Array::from_f32_slice(&[0.0, 0.0, 0.0, 0.0], &[4]),
            ),
        ]);

        sanitize_qwen3_next_mtp_weights(&mut weights, &config).unwrap();
        assert!(weights.contains_key("model.pre_fc_norm_hidden.weight"));
        assert!(weights.contains_key("model.layers.0.self_attn.q_norm.weight"));
        let shifted = weights
            .get("model.pre_fc_norm_hidden.weight")
            .unwrap()
            .clone();
        shifted.eval();
        let values: &[f32] = shifted.as_slice();
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[3] - 2.5).abs() < 1e-6);
    }
}
