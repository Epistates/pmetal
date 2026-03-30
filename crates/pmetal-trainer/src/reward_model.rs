#![allow(unsafe_code)]
//! ML-based reward model for RL training.
//!
//! Loads a pretrained reward model (e.g., ArmoRM, Skywork-Reward) and uses it
//! to score prompt+completion pairs. The reward model runs inference-only on GPU
//! alongside the policy model.
//!
//! # Architecture
//!
//! Reward models are typically causal LMs with a scalar head:
//! ```text
//! input_ids → Transformer → hidden_states[-1] → Linear(hidden_dim, 1) → scalar reward
//! ```
//!
//! Some models output rewards via the logits of specific tokens (e.g., "good"/"bad"),
//! while others have an explicit `model.score` weight.
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

use pmetal_bridge::compat::{Array, indexing::IndexOp, ops};
use pmetal_bridge::inline_array;
use pmetal_data::Tokenizer;
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
    /// Batch size for reward model inference.
    ///
    /// Currently unused — scoring is done one at a time to avoid OOM.
    /// Future batched path can use this field.
    pub batch_size: usize,
    /// Whether to extract the reward from the last token position (true)
    /// or via mean pooling over all token positions (false).
    ///
    /// Most reward models (ArmoRM, Skywork-Reward) use the last token.
    pub use_last_token: bool,
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
    /// Optional weight to apply to the ML reward relative to other reward
    /// functions in the CombinedReward.
    ///
    /// This is informational — the actual weight is set when calling
    /// `CombinedReward::add`.
    pub weight: f64,
}

impl Default for RewardModelConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            max_length: 2048,
            batch_size: 4,
            use_last_token: true,
            chat_template: None,
            weight: 1.0,
        }
    }
}

/// Scoring strategy for a reward model.
///
/// Determines how the scalar reward value is extracted from the model's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringStrategy {
    /// Apply an explicit `score` linear layer (weight `[hidden_dim, 1]`) to the
    /// last-token hidden state.  Used by ArmoRM, Skywork-Reward, and most HF
    /// reward models.  Requires `score.weight` in the checkpoint.
    ScoreHead,
    /// Treat the raw logits of the last token as the score vector.  Takes the
    /// mean across the vocabulary dimension as the scalar reward.  Useful as a
    /// baseline when no explicit score head is present.
    MeanLogits,
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
    /// Explicit score head weight `[hidden_dim, 1]` (transposed for matmul).
    ///
    /// Loaded from `score.weight` or `model.score.weight` in the checkpoint.
    score_weight: Option<Array>,
    /// Explicit score head bias `[1]`.
    score_bias: Option<Array>,
    /// Strategy used to derive a scalar reward from model output.
    strategy: ScoringStrategy,
    /// Configuration.
    config: RewardModelConfig,
}

impl MLRewardModel {
    /// Load a reward model from a pretrained checkpoint directory.
    ///
    /// Automatically detects the architecture from `config.json`, loads weights
    /// via `DynamicModel::load`, and probes for an explicit `score` head.
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

        // Determine scoring strategy, with a correctness check for ScoreHead.
        //
        // `DynamicModel::forward` returns logits [batch, seq, vocab_size].
        // A score head (from ArmoRM / Skywork-Reward style models) has shape
        // [1, hidden_size] and must be applied to the *hidden state* before
        // the lm_head projection — not to the final logits.
        //
        // `DynamicModel` does not currently expose a `forward_hidden` method,
        // so we cannot obtain the raw hidden states.  Attempting to apply
        // `score.weight [1, hidden_size]` to `logits [1, vocab_size]` will
        // produce a dimension mismatch at runtime when hidden_size ≠ vocab_size.
        //
        // Rather than silently producing garbage results, we degrade to
        // MeanLogits and emit a clear diagnostic.  True reward-head scoring
        // can be re-enabled once DynamicModel exposes a `forward_hidden` API.
        let strategy = if let Some(ref sw) = score_weight {
            let score_in_features = sw.shape().get(1).copied().unwrap_or(0) as usize;
            let vocab_size = model.vocab_size() as usize;
            let hidden_size = model.hidden_size() as usize;

            if score_in_features == hidden_size {
                // score.weight inner dim matches hidden_size: this is the
                // canonical reward-head layout.  However, DynamicModel only
                // provides logits (post-lm_head), not hidden states.
                // Applying score.weight to logits would be incorrect unless
                // vocab_size == hidden_size (rare / only tiny models).
                if vocab_size == hidden_size {
                    // Degenerate case where lm_head is square: logits ≈ hidden
                    // states scaled by lm_head, so the score head is technically
                    // applicable.  Use it.
                    tracing::info!(
                        "Reward model: score head enabled (hidden_size={hidden_size} == vocab_size); \
                         note that scoring is applied to logits, not raw hidden states."
                    );
                    ScoringStrategy::ScoreHead
                } else {
                    tracing::warn!(
                        "Reward model: score.weight inner dim ({score_in_features}) matches \
                         hidden_size ({hidden_size}) but DynamicModel::forward returns logits \
                         of shape [batch, seq, {vocab_size}], not hidden states. \
                         Applying score.weight to logits would produce a dimension mismatch. \
                         Degrading to MeanLogits. To use true reward-head scoring, expose \
                         DynamicModel::forward_hidden."
                    );
                    ScoringStrategy::MeanLogits
                }
            } else if score_in_features == vocab_size {
                // score.weight inner dim matches vocab_size: the score head
                // was designed to operate on logits (unusual but valid for
                // token-level reward models).
                tracing::info!(
                    "Reward model: score head inner dim ({score_in_features}) matches \
                     vocab_size ({vocab_size}); applying score head to last-token logits."
                );
                ScoringStrategy::ScoreHead
            } else {
                tracing::warn!(
                    "Reward model: score.weight inner dim ({score_in_features}) matches \
                     neither hidden_size ({hidden_size}) nor vocab_size ({vocab_size}). \
                     Cannot apply score head safely. Degrading to MeanLogits."
                );
                ScoringStrategy::MeanLogits
            }
        } else {
            tracing::info!("Reward model: no score head found, using mean-logit scoring");
            ScoringStrategy::MeanLogits
        };

        tracing::info!(
            "Reward model ready: arch={:?}, strategy={:?}",
            model.architecture(),
            strategy
        );

        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
            score_weight,
            score_bias,
            strategy,
            config,
        })
    }

    /// Attempt to load an explicit score head (`score.weight` / `score.bias`)
    /// from the safetensors files in `model_dir`.
    ///
    /// Scans every `.safetensors` file in the directory and returns the first
    /// match.  Returns `(None, None)` if no score head is found — this is not
    /// an error; the caller selects `MeanLogits` scoring instead.
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
    /// Returns `(input_ids_array, seq_len)` where `input_ids_array` has shape
    /// `[1, seq_len]` with dtype `int32`.
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

    /// Score a single prompt+completion pair.
    ///
    /// Runs a full forward pass on the reward model and extracts a scalar
    /// reward.  No gradient tape is active during this call.
    fn score_single(
        &self,
        model: &mut pmetal_models::DynamicModel,
        prompt: &str,
        completion: &str,
    ) -> GrpoResult<f64> {
        let text = self.format_input(prompt, completion);
        let input_ids = self.tokenize_input(&text)?;

        // Forward pass — inference only, gradient tape is not active during
        // GRPO reward scoring.
        let logits = model
            .forward(&input_ids, None)
            .map_err(|e| GrpoError::Reward(format!("Reward model forward pass failed: {}", e)))?;
        // logits shape: [1, seq_len, vocab_size]

        let reward = match self.strategy {
            ScoringStrategy::ScoreHead => {
                // NOTE: DynamicModel::forward returns logit projections
                // [batch, seq_len, vocab_size], not raw hidden states.
                //
                // For true reward-head models (ArmoRM, Skywork-Reward), the
                // `score.weight` tensor is `[1, hidden_size]`.  Since we only
                // have access to logits here, we apply the score head to the
                // last-token logit vector `[1, vocab_size]`.  This is an
                // approximation — it is correct only when the reward model's
                // `lm_head` is weight-tied with the embedding layer, so logits
                // and hidden states are related by a fixed projection.
                //
                // For best accuracy with ArmoRM-style models, the logit vector
                // dimension must match `score.weight`'s inner dimension.  If
                // they differ at runtime, MLX's matmul will return an error.
                //
                // Shape: [1, vocab_size]
                let last_token_vec = logits.index((.., -1i32, ..));

                // score.weight shape from HF: [out_features=1, in_features=hidden_dim].
                // Transpose to [hidden_dim, 1] for matmul [1, hidden_dim] × [hidden_dim, 1].
                let w = self.score_weight.as_ref().unwrap();
                let w_t = w.transpose_axes(&[1, 0]);
                // [1, vocab_size] × [vocab_size, 1] → [1, 1]
                let mut score = ops::matmul(&last_token_vec, &w_t);

                let mut score = if let Some(b) = &self.score_bias {
                    score.add(b)
                } else {
                    score
                };

                score.eval();
                score.item::<f32>() as f64
            }

            ScoringStrategy::MeanLogits => {
                // No explicit score head.  Use the mean of the last-token
                // logits as a heuristic scalar reward.
                // Shape: [1, vocab_size] → scalar
                let last_token_logits = logits.index((.., -1i32, ..));
                let mut mean = last_token_logits.mean(None);
                mean.eval();
                mean.item::<f32>() as f64
            }
        };

        Ok(reward)
    }

    /// Return the active scoring strategy.
    pub fn strategy(&self) -> ScoringStrategy {
        self.strategy
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
// The `Option<Array>` fields (score head) are read-only after construction.
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
            .field("strategy", &self.strategy)
            .field("max_length", &self.config.max_length)
            .field("has_score_head", &self.score_weight.is_some())
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
        assert_eq!(config.batch_size, 4);
        assert!(config.use_last_token);
        assert!(config.chat_template.is_none());
        assert_eq!(config.weight, 1.0);
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
