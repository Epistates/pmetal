#![allow(unsafe_code)]
//! ML-based reward model for RL training.
//!
//! Loads a pretrained reward model (e.g., ArmoRM, Skywork-Reward) and uses it
//! to score prompt+completion pairs. The reward model runs inference-only on GPU
//! alongside the policy model.
//!
//! # Architecture
//!
//! A reward model is a causal LM trunk with a scalar head on its last hidden
//! state, as a sequence-classification head is:
//!
//! ```text
//! input_ids → Transformer → final-norm hidden state of the last token
//!           → Linear(hidden_dim, 1) → scalar reward
//! ```
//!
//! The head is the checkpoint's `score.weight` (`[1, hidden_dim]`) and, when it
//! has one, `score.bias`. A model without that head is not a reward model and
//! is refused: the logits of a language model are not a score, and training
//! against their mean would optimize an arbitrary number.
//!
//! # Usage
//!
//! ```no_run
//! use pmetal_trainer::reward_model::{MLRewardModel, RewardModelConfig};
//! use pmetal_data::Tokenizer;
//!
//! let config = RewardModelConfig {
//!     model_path: "/path/to/reward-model".into(),
//!     max_length: 2048,
//!     ..Default::default()
//! };
//! let tokenizer = Tokenizer::from_model_dir("/path/to/reward-model").unwrap();
//! let reward_model = MLRewardModel::from_pretrained(
//!     "/path/to/reward-model",
//!     tokenizer,
//!     config,
//! ).unwrap();
//! ```

use pmetal_data::Tokenizer;
use pmetal_runtime::compat::{Array, Dtype, ops};
use pmetal_runtime::inline_array;
use std::path::Path;
use std::sync::Mutex;

use super::grpo::{GrpoError, GrpoResult, RewardFunction};

/// Configuration for ML-based reward model inference.
#[derive(Debug, Clone)]
pub struct RewardModelConfig {
    /// Path to the pretrained reward model (HF model ID or local path).
    ///
    /// Used only for display/logging — the actual path is passed to
    /// `MLRewardModel::from_pretrained`.
    pub model_path: String,
    /// Maximum input sequence length (prompt + completion).
    ///
    /// Inputs longer than this are truncated from the right.
    pub max_length: usize,
    /// Optional template string for formatting prompt+completion into a single
    /// input sequence for the reward model.
    ///
    /// Use `{prompt}` and `{completion}` as placeholders:
    /// ```text
    /// "Human: {prompt}\nAssistant: {completion}"
    /// ```
    ///
    /// When `None`, the prompt and completion are concatenated directly.
    pub chat_template: Option<String>,
}

impl Default for RewardModelConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            max_length: 2048,
            chat_template: None,
        }
    }
}

/// ML-based reward model wrapping a pretrained language model.
///
/// The model is loaded once and used exclusively for forward-pass inference
/// during training.  No gradients are computed — the reward model is frozen.
///
/// # Thread Safety
///
/// `DynamicModel::forward` requires `&mut self`.  A `Mutex` provides exclusive
/// access, making the wrapper `Send + Sync` for use inside `CombinedReward`.
pub struct MLRewardModel {
    /// The underlying dynamic model (any supported PMetal architecture).
    model: Mutex<pmetal_models::DynamicModel>,
    /// Tokenizer for the reward model (may differ from the policy tokenizer).
    tokenizer: Tokenizer,
    /// The score head's weight, `[1, hidden_dim]`.
    score_weight: Array,
    /// The score head's bias, `[1]`, if it has one.
    score_bias: Option<Array>,
    /// Configuration.
    config: RewardModelConfig,
}

/// The scalar a score head reads from the last token's hidden state:
/// `hidden[last] · weight + bias`, in f32 whatever the checkpoint's dtype.
///
/// `hidden` is `[1, seq, hidden_dim]` and `weight` `[1, hidden_dim]`.
fn score_last_token(hidden: &Array, weight: &Array, bias: Option<&Array>) -> f64 {
    let f32_dtype = Dtype::Float32.as_i32();
    let (batch, seq, dim) = (hidden.dim(0), hidden.dim(1), hidden.dim(2));
    let last = hidden
        .slice(&[0, seq - 1, 0], &[batch, seq, dim])
        .reshape(&[batch, dim])
        .as_dtype(f32_dtype);
    let mut score = ops::matmul(&last, &weight.as_dtype(f32_dtype).transpose_axes(&[1, 0]));
    if let Some(bias) = bias {
        score = score.add(&bias.as_dtype(f32_dtype));
    }
    score.eval();
    f64::from(score.item_f32())
}

/// Whether a `score.weight` of `shape` is a scalar head over a trunk of
/// `hidden_size`: `[1, hidden_size]`.
fn check_score_head(shape: &[i32], hidden_size: usize) -> GrpoResult<()> {
    match shape {
        [1, inner] if *inner as usize == hidden_size => Ok(()),
        _ => Err(GrpoError::Reward(format!(
            "the score head has shape {shape:?}; a reward model's is [1, {hidden_size}] (a scalar \
             head on the {hidden_size}-wide hidden state)"
        ))),
    }
}

impl MLRewardModel {
    /// Load a reward model from a pretrained checkpoint directory.
    ///
    /// Automatically detects the architecture from `config.json`, loads weights
    /// via `DynamicModel::load`, and loads the `score` head from the
    /// safetensors files. A checkpoint without a scalar score head over the
    /// trunk's hidden size is an error.
    ///
    /// # Arguments
    ///
    /// * `model_path` — local directory containing `config.json` + safetensors.
    /// * `tokenizer` — pre-loaded tokenizer for the reward model.  Call
    ///   `Tokenizer::from_model_dir(model_path)` to obtain one.
    /// * `config` — reward model inference configuration.
    pub fn from_pretrained(
        model_path: impl AsRef<Path>,
        tokenizer: Tokenizer,
        config: RewardModelConfig,
    ) -> GrpoResult<Self> {
        let path = model_path.as_ref();
        tracing::info!("Loading reward model from: {}", path.display());

        let model = pmetal_models::DynamicModel::load(path)
            .map_err(|e| GrpoError::Reward(format!("Failed to load reward model: {}", e)))?;

        let (score_weight, score_bias) = Self::try_load_score_head(path);
        let score_weight = score_weight.ok_or_else(|| {
            GrpoError::Reward(format!(
                "{} has no `score.weight`: it is not a reward model (a language model's logits \
                 are not a score)",
                path.display()
            ))
        })?;
        check_score_head(score_weight.shape(), model.hidden_size() as usize)?;
        if let Some(bias) = &score_bias {
            if bias.size() != 1 {
                return Err(GrpoError::Reward(format!(
                    "the score head's bias has {} entries, expected 1",
                    bias.size()
                )));
            }
        }

        tracing::info!(
            "Reward model ready: arch={:?}, scalar score head over the last token's hidden state",
            model.architecture(),
        );

        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
            score_weight,
            score_bias,
            config,
        })
    }

    /// Attempt to load an explicit score head (`score.weight` / `score.bias`)
    /// from the safetensors files in `model_dir`.
    ///
    /// Scans every `.safetensors` file in the directory and returns the first
    /// match, or `(None, None)` if there is none.
    fn try_load_score_head(model_dir: &Path) -> (Option<Array>, Option<Array>) {
        // Collect all safetensors files in the directory (non-recursive).
        let sf_paths: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        for sf_path in &sf_paths {
            let sf_path_str = match sf_path.to_str() {
                Some(s) => s,
                None => continue,
            };
            match inline_array::load_safetensors_shard(sf_path_str) {
                Some(tensors) => {
                    let tensors: std::collections::HashMap<String, Array> =
                        tensors.into_iter().collect();
                    // ArmoRM / Skywork: "score.weight"
                    // Some models wrap under "model.score.weight"
                    let weight = tensors
                        .get("score.weight")
                        .or_else(|| tensors.get("model.score.weight"))
                        .cloned();
                    let bias = tensors
                        .get("score.bias")
                        .or_else(|| tensors.get("model.score.bias"))
                        .cloned();

                    if weight.is_some() {
                        tracing::info!("Found score head weights in {}", sf_path.display());
                        return (weight, bias);
                    }
                }
                None => {
                    tracing::warn!("Failed to probe score head in {}", sf_path.display());
                }
            }
        }

        (None, None)
    }

    /// Format a single prompt+completion pair into the reward model's input
    /// string, applying the chat template if configured.
    fn format_input(&self, prompt: &str, completion: &str) -> String {
        match &self.config.chat_template {
            Some(template) => template
                .replace("{prompt}", prompt)
                .replace("{completion}", completion),
            None => format!("{}{}", prompt, completion),
        }
    }

    /// Tokenize the formatted input string and truncate to `max_length`.
    ///
    /// Returns the `[1, seq_len]` int32 ids.
    fn tokenize_input(&self, text: &str) -> GrpoResult<Array> {
        let ids = self
            .tokenizer
            .encode(text)
            .map_err(|e| GrpoError::Tokenizer(e.to_string()))?;

        // Truncate to configured max_length.
        let ids: Vec<i32> = ids
            .into_iter()
            .take(self.config.max_length)
            .map(|x| x as i32)
            .collect();

        let seq_len = ids.len() as i32;
        let input_ids = Array::from_slice(&ids, &[1, seq_len]);
        Ok(input_ids)
    }

    /// Score a single prompt+completion pair: the score head on the last
    /// token's final-norm hidden state. Inference only; no gradient is taken.
    fn score_single(
        &self,
        model: &mut pmetal_models::DynamicModel,
        prompt: &str,
        completion: &str,
    ) -> GrpoResult<f64> {
        let text = self.format_input(prompt, completion);
        let input_ids = self.tokenize_input(&text)?;
        if input_ids.dim(1) == 0 {
            return Err(GrpoError::Reward(
                "the reward model's input is empty".into(),
            ));
        }

        let hidden = model
            .forward_hidden(&input_ids, None)
            .map_err(|e| GrpoError::Reward(format!("Reward model forward pass failed: {}", e)))?;
        let score = score_last_token(&hidden, &self.score_weight, self.score_bias.as_ref());
        pmetal_runtime::check_unobserved_error()
            .map_err(|e| GrpoError::Reward(format!("reward scoring failed: {e}")))?;
        if !score.is_finite() {
            return Err(GrpoError::Reward(format!(
                "the reward model scored {score}"
            )));
        }
        Ok(score)
    }

    /// Return a reference to the reward model config.
    pub fn config(&self) -> &RewardModelConfig {
        &self.config
    }
}

// SAFETY: `DynamicModel` contains MLX `Array`s which are internally reference-
// counted pointers into MLX-managed GPU/CPU memory.  They are not `Send` in
// the general case.  However, our `Mutex<DynamicModel>` guarantees exclusive
// access — only one thread holds the lock at a time — which is the only safe
// access pattern that MLX supports for concurrent callers sharing a model.
// The `Tokenizer` wraps `tokenizers::Tokenizer` which is `Send + Sync`.
// The score head arrays are read-only after construction.
unsafe impl Send for MLRewardModel {}
unsafe impl Sync for MLRewardModel {}

impl RewardFunction for MLRewardModel {
    fn compute(
        &self,
        prompts: &[String],
        completions: &[String],
        _images: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        let mut model = self.model.lock().map_err(|e| {
            GrpoError::Reward(format!("Failed to acquire reward model lock: {}", e))
        })?;

        let mut rewards = Vec::with_capacity(completions.len());
        for (prompt, completion) in prompts.iter().zip(completions.iter()) {
            let reward = self.score_single(&mut model, prompt, completion)?;
            rewards.push(reward);
        }

        Ok(rewards)
    }

    fn name(&self) -> &str {
        "ml_reward_model"
    }
}

impl std::fmt::Debug for MLRewardModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MLRewardModel")
            .field("max_length", &self.config.max_length)
            .field("has_score_bias", &self.score_bias.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reward_model_config_default() {
        let config = RewardModelConfig::default();
        assert_eq!(config.max_length, 2048);
        assert!(config.chat_template.is_none());
    }

    /// The score is the head applied to the *last* token's hidden state (not
    /// the logits, whose mean the old fallback took), in f32: `w · h_last + b`.
    #[test]
    fn the_score_is_the_head_on_the_last_hidden_state() {
        // [1, 3 tokens, 4 dims]: only the last row counts.
        let hidden = Array::from_slice(
            &[
                9.0f32, 9.0, 9.0, 9.0, // ignored
                8.0, 8.0, 8.0, 8.0, // ignored
                1.0, -2.0, 0.5, 4.0, // last
            ],
            &[1, 3, 4],
        )
        .as_dtype(Dtype::Bfloat16.as_i32());
        let weight = Array::from_slice(&[0.5f32, 0.25, -1.0, 0.125], &[1, 4]);
        let bias = Array::from_slice(&[0.75f32], &[1]);
        // 0.5 - 0.5 - 0.5 + 0.5 = 0, plus the bias.
        let got = score_last_token(&hidden, &weight, Some(&bias));
        assert!((got - 0.75).abs() < 1e-6, "{got}");
        let got = score_last_token(&hidden, &weight, None);
        assert!(got.abs() < 1e-6, "{got}");
        // Another head reads the same hidden state differently.
        let weight = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[1, 4]);
        let got = score_last_token(&hidden, &weight, None);
        assert!((got - 3.5).abs() < 1e-6, "{got}");
    }

    /// Only a scalar head over the trunk's hidden size is a reward head.
    #[test]
    fn only_a_scalar_head_over_the_hidden_size_is_accepted() {
        assert!(check_score_head(&[1, 32], 32).is_ok());
        // A classification head, a head over another width and a vocabulary-
        // sized head (the old logits fallback) are refused.
        assert!(check_score_head(&[2, 32], 32).is_err());
        assert!(check_score_head(&[1, 64], 32).is_err());
        assert!(check_score_head(&[1000, 1], 32).is_err());
    }

    #[test]
    fn test_format_input_no_template() {
        // We can't easily instantiate MLRewardModel without a real model,
        // so test the template logic inline.
        let template: Option<String> = None;
        let prompt = "What is 2+2?";
        let completion = "4";
        let result = match &template {
            Some(t) => t
                .replace("{prompt}", prompt)
                .replace("{completion}", completion),
            None => format!("{}{}", prompt, completion),
        };
        assert_eq!(result, "What is 2+2?4");
    }

    #[test]
    fn test_format_input_with_template() {
        let template = Some("Human: {prompt}\nAssistant: {completion}".to_string());
        let prompt = "What is 2+2?";
        let completion = "4";
        let result = match &template {
            Some(t) => t
                .replace("{prompt}", prompt)
                .replace("{completion}", completion),
            None => format!("{}{}", prompt, completion),
        };
        assert_eq!(result, "Human: What is 2+2?\nAssistant: 4");
    }
}
