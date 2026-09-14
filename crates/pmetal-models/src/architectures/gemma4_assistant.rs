//! Gemma 4 MTP assistant model.
//!
//! Gemma 4's MTP checkpoint is not a standalone small CausalLM. It is an
//! assistant tower that consumes the target model's token embedding plus final
//! hidden state, attends with Q-only layers over target KV states, and projects
//! its output back into the target hidden space for autoregressive drafting.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use pmetal_bridge::compat::{
    Array, Dtype, Exception, Module, ModuleParameters, ModuleParametersExt, Param, nn, ops,
    transforms,
};
use pmetal_bridge::impl_module_params;
use pmetal_mlx::kernels::rope::RopePositions;
use serde::{Deserialize, Serialize};

use super::gemma4::{Gemma4Config, Gemma4Model, LoadReport};

fn default_assistant_model_type() -> String {
    "gemma4_assistant".to_string()
}

fn default_num_centroids() -> i32 {
    2048
}

fn default_centroid_intermediate_top_k() -> i32 {
    32
}

fn default_tie_word_embeddings() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gemma4AssistantConfig {
    #[serde(default = "default_assistant_model_type")]
    pub model_type: String,
    pub text_config: Gemma4Config,
    pub backbone_hidden_size: i32,
    #[serde(default)]
    pub use_ordered_embeddings: bool,
    #[serde(default = "default_num_centroids")]
    pub num_centroids: i32,
    #[serde(default = "default_centroid_intermediate_top_k")]
    pub centroid_intermediate_top_k: i32,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}

impl Gemma4AssistantConfig {
    pub fn validate(&self) -> Result<(), Exception> {
        if self.backbone_hidden_size <= 0 {
            return Err(Exception::custom(
                "Gemma 4 assistant requires backbone_hidden_size > 0",
            ));
        }
        if self.text_config.hidden_size <= 0 {
            return Err(Exception::custom(
                "Gemma 4 assistant text_config.hidden_size must be > 0",
            ));
        }
        let shared = self.text_config.num_kv_shared_layers.unwrap_or(0);
        if shared != self.text_config.num_hidden_layers {
            return Err(Exception::custom(format!(
                "Gemma 4 assistant expects all text layers to share target KV states; got num_kv_shared_layers={shared}, num_hidden_layers={}",
                self.text_config.num_hidden_layers
            )));
        }
        if self.use_ordered_embeddings {
            if self.num_centroids <= 0 || self.centroid_intermediate_top_k <= 0 {
                return Err(Exception::custom(
                    "Gemma 4 ordered assistant embeddings require positive centroid counts",
                ));
            }
            if self.text_config.vocab_size % self.num_centroids != 0 {
                return Err(Exception::custom(format!(
                    "Gemma 4 assistant vocab_size ({}) must be divisible by num_centroids ({})",
                    self.text_config.vocab_size, self.num_centroids
                )));
            }
        }
        self.text_config.pruned_unsupported_blocks()
    }
}

#[derive(Debug, Clone)]
pub struct Gemma4AssistantGenerationConfig {
    pub num_assistant_tokens: usize,
}

impl Default for Gemma4AssistantGenerationConfig {
    fn default() -> Self {
        Self {
            num_assistant_tokens: 6,
        }
    }
}

impl Gemma4AssistantGenerationConfig {
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self, Exception> {
        let path = model_dir.as_ref().join("generation_config.json");
        if !path.exists() {
            return Ok(Self::default());
        }
        let content =
            std::fs::read_to_string(&path).map_err(|e| Exception::custom(e.to_string()))?;
        let json: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| Exception::custom(e.to_string()))?;
        let num_assistant_tokens = json
            .get("num_assistant_tokens")
            .and_then(|value| value.as_u64())
            .map(|value| value.max(1) as usize)
            .unwrap_or_else(|| Self::default().num_assistant_tokens);
        Ok(Self {
            num_assistant_tokens,
        })
    }
}

#[derive(Debug)]
pub struct Gemma4AssistantSharedKvStates {
    pub full_attention: Option<(Array, Array)>,
    pub sliding_attention: Option<(Array, Array)>,
}

#[derive(Debug)]
pub struct Gemma4AssistantOutput {
    /// Assistant hidden state in assistant text hidden size.
    pub hidden_state: Array,
    /// Projected state in target backbone hidden size.
    pub backbone_hidden_state: Array,
}

#[derive(Debug)]
pub struct Gemma4AssistantMaskedEmbedder {
    pub centroids: nn::Linear,
    pub token_ordering: Param<Array>,
    pub num_centroids: i32,
    pub vocab_size: i32,
    pub vocab_size_per_centroid: i32,
    pub top_k: i32,
}
impl_module_params!(Gemma4AssistantMaskedEmbedder; centroids, token_ordering);

impl Gemma4AssistantMaskedEmbedder {
    pub fn new(config: &Gemma4AssistantConfig) -> Result<Self, Exception> {
        let vocab_size = config.text_config.vocab_size;
        let vocab_size_per_centroid = vocab_size / config.num_centroids;
        Ok(Self {
            centroids: nn::LinearBuilder::new(config.text_config.hidden_size, config.num_centroids)
                .bias(false)
                .build()?,
            token_ordering: Param::new(ops::zeros(
                &[config.num_centroids, vocab_size_per_centroid],
                Dtype::Int32,
            )),
            num_centroids: config.num_centroids,
            vocab_size,
            vocab_size_per_centroid,
            top_k: config.centroid_intermediate_top_k,
        })
    }

    fn selected_logits(
        &self,
        hidden_state: &Array,
        lm_head_weight: &Array,
    ) -> Result<(Array, Array), Exception> {
        let hidden_dim = hidden_state
            .shape()
            .last()
            .copied()
            .ok_or_else(|| Exception::custom("Gemma 4 assistant hidden state has no shape"))?;
        let hidden = hidden_state.reshape(&[-1, hidden_dim]);
        let rows = hidden.dim(0);

        let k = self.top_k.min(self.num_centroids).max(1);
        let centroid_logits = self.centroids.forward(&hidden);
        let neg_logits = centroid_logits.multiply(&Array::from_f32(-1.0));
        let partitioned = neg_logits.argpartition(k - 1, -1);
        let top_centroids = partitioned
            .slice(&[0, 0], &[rows, k])
            .as_dtype(Dtype::Int32.as_i32());

        let ordering = self
            .token_ordering
            .as_ref()
            .reshape(&[self.num_centroids, self.vocab_size_per_centroid])
            .as_dtype(Dtype::Int32.as_i32());
        let selected_tokens = ordering
            .take_axis(&top_centroids, 0)
            .reshape(&[rows, k * self.vocab_size_per_centroid])
            .as_dtype(Dtype::Int32.as_i32());
        let active = selected_tokens.dim(1);
        let flat_tokens = selected_tokens
            .reshape(&[-1])
            .as_dtype(Dtype::Int32.as_i32());
        let selected_weights =
            lm_head_weight
                .take_axis(&flat_tokens, 0)
                .reshape(&[rows, active, hidden.dim(1)]);
        let selected_logits = hidden
            .reshape(&[rows, 1, hidden.dim(1)])
            .matmul(&selected_weights.transpose_axes(&[0, 2, 1]))
            .squeeze(1);
        Ok((selected_tokens, selected_logits))
    }

    pub fn logits(&self, hidden_state: &Array, lm_head_weight: &Array) -> Result<Array, Exception> {
        let (selected_tokens, selected_logits) =
            self.selected_logits(hidden_state, lm_head_weight)?;
        let rows = selected_logits.dim(0);
        let full = ops::zeros(&[rows, self.vocab_size], selected_logits.dtype())
            .add(&Array::from_f32(f32::NEG_INFINITY));
        Ok(ops::put_along_axis(
            &full,
            &selected_tokens,
            &selected_logits,
            -1,
        ))
    }

    pub fn greedy_token(
        &self,
        hidden_state: &Array,
        lm_head_weight: &Array,
    ) -> Result<u32, Exception> {
        let (selected_tokens, selected_logits) =
            self.selected_logits(hidden_state, lm_head_weight)?;
        let rows = selected_logits.dim(0);
        if rows != 1 {
            return Err(Exception::custom(format!(
                "Gemma 4 assistant ordered greedy decode expects one row, got {rows}"
            )));
        }
        let best = selected_logits.argmax(-1).as_dtype(Dtype::Int32.as_i32());
        let token = ops::take_along_axis(&selected_tokens, &best.reshape(&[rows, 1]), -1);
        Ok(token.item::<i32>() as u32)
    }
}

#[derive(Debug)]
pub struct Gemma4AssistantForCausalLM {
    pub model: Gemma4Model,
    pub lm_head: nn::Linear,
    pub pre_projection: nn::Linear,
    pub post_projection: nn::Linear,
    pub masked_embedding: Option<Gemma4AssistantMaskedEmbedder>,
    pub config: Gemma4AssistantConfig,
}
impl_module_params!(
    Gemma4AssistantForCausalLM;
    model,
    lm_head,
    pre_projection,
    post_projection,
    masked_embedding
);

impl Gemma4AssistantForCausalLM {
    pub fn new(mut config: Gemma4AssistantConfig) -> Result<Self, Exception> {
        if config.text_config.num_kv_shared_layers.is_none() {
            config.text_config.num_kv_shared_layers = Some(config.text_config.num_hidden_layers);
        }
        config.validate()?;
        let text_config = config.text_config.clone();
        let model = Gemma4Model::new(text_config.clone())?;
        let lm_head = nn::LinearBuilder::new(text_config.hidden_size, text_config.vocab_size)
            .bias(false)
            .build()?;
        let pre_projection =
            nn::LinearBuilder::new(config.backbone_hidden_size * 2, text_config.hidden_size)
                .bias(false)
                .build()?;
        let post_projection =
            nn::LinearBuilder::new(text_config.hidden_size, config.backbone_hidden_size)
                .bias(false)
                .build()?;
        let masked_embedding = if config.use_ordered_embeddings {
            Some(Gemma4AssistantMaskedEmbedder::new(&config)?)
        } else {
            None
        };
        Ok(Self {
            model,
            lm_head,
            pre_projection,
            post_projection,
            masked_embedding,
            config,
        })
    }

    pub fn forward_hidden(
        &mut self,
        inputs_embeds: &Array,
        shared_kv_states: &Gemma4AssistantSharedKvStates,
        position_id: i32,
    ) -> Result<Gemma4AssistantOutput, Exception> {
        let mut h = self.pre_projection.forward(inputs_embeds);
        for (idx, layer) in self.model.layers.iter_mut().enumerate() {
            let source = if self.config.text_config.is_full_attention(idx) {
                shared_kv_states.full_attention.as_ref()
            } else {
                shared_kv_states.sliding_attention.as_ref()
            }
            .ok_or_else(|| {
                let kind = if self.config.text_config.is_full_attention(idx) {
                    "full_attention"
                } else {
                    "sliding_attention"
                };
                Exception::custom(format!(
                    "Gemma 4 assistant missing target shared KV state for {kind} layer {idx}"
                ))
            })?;

            h = layer.forward_with_shared_kv(
                &h,
                None,
                &source.0,
                &source.1,
                RopePositions::Offset(position_id),
                None,
            )?;
        }
        let hidden_state = self.model.norm.forward(&h);
        let backbone_hidden_state = self.post_projection.forward(&hidden_state);
        Ok(Gemma4AssistantOutput {
            hidden_state,
            backbone_hidden_state,
        })
    }

    pub fn draft_greedy_token(
        &mut self,
        inputs_embeds: &Array,
        shared_kv_states: &Gemma4AssistantSharedKvStates,
        position_id: i32,
    ) -> Result<(u32, Array), Exception> {
        let output = self.forward_hidden(inputs_embeds, shared_kv_states, position_id)?;
        let token = if let Some(masked) = self.masked_embedding.as_ref() {
            masked.greedy_token(&output.hidden_state, self.lm_head.weight.as_ref())?
        } else {
            let logits = self.lm_head.forward(&output.hidden_state);
            logits.argmax(-1).item::<u32>()
        };
        Ok((token, output.backbone_hidden_state))
    }

    pub fn forward_logits(
        &mut self,
        inputs_embeds: &Array,
        shared_kv_states: &Gemma4AssistantSharedKvStates,
        position_id: i32,
    ) -> Result<(Array, Array), Exception> {
        let output = self.forward_hidden(inputs_embeds, shared_kv_states, position_id)?;
        let logits = if let Some(masked) = self.masked_embedding.as_ref() {
            masked.logits(&output.hidden_state, self.lm_head.weight.as_ref())?
        } else {
            let vocab_size = self.config.text_config.vocab_size;
            self.lm_head
                .forward(&output.hidden_state)
                .reshape(&[-1, vocab_size])
        };
        Ok((output.backbone_hidden_state, logits))
    }
}

pub fn load_gemma4_assistant_from_dir(
    model_dir: impl AsRef<Path>,
) -> Result<(Gemma4AssistantForCausalLM, Gemma4AssistantGenerationConfig), Exception> {
    let model_dir = model_dir.as_ref();
    let config_path = model_dir.join("config.json");
    let config_content =
        std::fs::read_to_string(&config_path).map_err(|e| Exception::custom(e.to_string()))?;
    let config: Gemma4AssistantConfig =
        json5::from_str(&config_content).map_err(|e| Exception::custom(e.to_string()))?;
    let mut model = Gemma4AssistantForCausalLM::new(config)?;
    let weights =
        crate::loader::load_weights(model_dir).map_err(|e| Exception::custom(format!("{e:?}")))?;
    let report = load_gemma4_assistant_weights(&mut model, &weights)?;
    if !report.skipped.is_empty() {
        tracing::info!(
            "Gemma 4 assistant weight load: {} loaded, {} skipped (first: {:?})",
            report.loaded,
            report.skipped.len(),
            report.skipped.first()
        );
    }
    eval_assistant_parameters(&model)?;
    let generation_config = Gemma4AssistantGenerationConfig::from_dir(model_dir)?;
    Ok((model, generation_config))
}

pub fn load_gemma4_assistant_weights(
    model: &mut Gemma4AssistantForCausalLM,
    raw_weights: &HashMap<String, Array>,
) -> Result<LoadReport, Exception> {
    let weights: HashMap<String, Array> = raw_weights
        .iter()
        .filter_map(|(key, value)| {
            let stripped = key
                .strip_prefix("model.language_model.")
                .map(|rest| format!("model.{rest}"))
                .unwrap_or_else(|| key.clone());
            if stripped.contains("embed_vision")
                || stripped.contains("vision_tower")
                || stripped.contains("audio_tower")
                || stripped.contains("multi_modal_projector")
            {
                None
            } else {
                Some((stripped, value.clone()))
            }
        })
        .collect();

    let mut report = LoadReport::default();
    let mut loaded_keys = HashSet::new();
    {
        let mut params = model.flatten_params_mut();
        for (key, value) in &weights {
            if let Some(param) = params.get_mut(key) {
                **param = value.clone();
                loaded_keys.insert(key.clone());
                report.loaded += 1;
            }
        }
    }

    if model.config.tie_word_embeddings && !loaded_keys.contains("lm_head.weight") {
        model.lm_head.weight = Param::new(model.model.embed_tokens.weight.value.clone());
        loaded_keys.insert("lm_head.weight".to_string());
        report.loaded += 1;
    }

    let mut required = vec![
        "model.embed_tokens.weight",
        "model.norm.weight",
        "pre_projection.weight",
        "post_projection.weight",
        "lm_head.weight",
    ];
    if model.config.use_ordered_embeddings {
        required.push("masked_embedding.centroids.weight");
        required.push("masked_embedding.token_ordering");
    }
    let missing: Vec<String> = required
        .into_iter()
        .filter(|key| !loaded_keys.contains(*key))
        .map(str::to_string)
        .collect();
    if !missing.is_empty() {
        return Err(Exception::custom(format!(
            "Gemma 4 assistant missing required weights: {}",
            missing.join(", ")
        )));
    }

    let param_keys: HashSet<String> = model
        .flatten_params()
        .into_keys()
        .map(|key| key.to_string())
        .collect();
    report.skipped = param_keys
        .into_iter()
        .filter(|key| !loaded_keys.contains(key))
        .collect();
    Ok(report)
}

fn eval_assistant_parameters(model: &Gemma4AssistantForCausalLM) -> Result<(), Exception> {
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

    fn tiny_text_config() -> Gemma4Config {
        Gemma4Config {
            model_type: "gemma4_text".to_string(),
            vocab_size: 8,
            hidden_size: 2,
            intermediate_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            global_head_dim: Some(2),
            num_global_key_value_heads: Some(1),
            max_position_embeddings: 16,
            rms_norm_eps: 1e-6,
            attention_k_eq_v: false,
            tie_word_embeddings: true,
            sliding_window: 8,
            final_logit_softcapping: None,
            layer_types: vec!["sliding_attention".to_string()],
            rope_parameters: None,
            _raw_rope_parameters: None,
            hidden_size_per_layer_input: None,
            vocab_size_per_layer_input: None,
            hidden_activation: None,
            num_kv_shared_layers: Some(1),
            use_double_wide_mlp: Some(false),
            enable_moe_block: Some(false),
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
        }
    }

    #[test]
    fn ordered_embedder_greedy_uses_selected_centroid_tokens() {
        let config = Gemma4AssistantConfig {
            model_type: "gemma4_assistant".to_string(),
            text_config: tiny_text_config(),
            backbone_hidden_size: 4,
            use_ordered_embeddings: true,
            num_centroids: 2,
            centroid_intermediate_top_k: 1,
            tie_word_embeddings: true,
        };
        let mut embedder = Gemma4AssistantMaskedEmbedder::new(&config).unwrap();
        embedder.centroids.weight = Param::new(Array::from_f32_slice(
            &[
                0.0, 0.0, // centroid 0
                1.0, 0.0, // centroid 1
            ],
            &[2, 2],
        ));
        embedder.token_ordering = Param::new(
            Array::from_i32_slice(&[
                0, 1, 2, 3, // centroid 0
                4, 5, 6, 7, // centroid 1
            ])
            .reshape(&[2, 4]),
        );
        let lm_head = Array::from_f32_slice(
            &[
                0.0, 0.0, // token 0
                0.0, 0.0, // token 1
                0.0, 0.0, // token 2
                0.0, 0.0, // token 3
                0.1, 0.0, // token 4
                0.2, 0.0, // token 5
                0.9, 0.0, // token 6
                0.3, 0.0, // token 7
            ],
            &[8, 2],
        );
        let hidden = Array::from_f32_slice(&[1.0, 0.0], &[1, 1, 2]);

        assert_eq!(embedder.greedy_token(&hidden, &lm_head).unwrap(), 6);
        let logits = embedder.logits(&hidden, &lm_head).unwrap();
        assert_eq!(logits.shape(), vec![1, 8]);
        logits.eval();
        let values: &[f32] = logits.as_slice();
        assert!((values[6] - 0.9).abs() < 1e-6);
        assert!(values[0].is_infinite() && values[0].is_sign_negative());
    }
}
